"""Unit witnesses for the fits, the truth harness, the controls and the property path of
``learned-rule-strata``.

The study's calibration cells read the reported SE of each stratum and marginal learned-rule
value against a per-replicate truth that the harness refits. A harness that refit a different
rule, integrated the wrong stratum or dropped ``S`` from its grid, or a control computed from
the wrong share, curve or regression, would pass or fail a cell for the wrong reason. These
checks rebuild each piece another way, and each deliberate mutation must move it.

Every fit draws from a throwaway seed at the declared size n = 1,000, where a fit with its
harness takes about a fifth of a second. No test asserts a bias, a coverage or an SE ratio of
the study from one sample.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from cleverly.exceptions import DataError
from tests.studies import _fold_evaluated_strata_law as strata
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_strata_law as law
from tests.studies import learned_rule_strata as study
from tests.studies import learned_rule_strata_properties as properties
from tests.studies.evidence import property_failures, replication

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_995_000
N = 1_000
CAP = 3


@cache
def _measured(setting: str, learner: str = "declared") -> tuple[pd.DataFrame, law.Measured]:
    frame = law.draw(setting, N, THROWAWAY + law.SETTINGS.index(setting))
    chosen = strata.drop_w1_learner() if learner == "drop_w1" else None
    return frame, law.measured(frame, setting, THROWAWAY + 10, learner=chosen)


SETTINGS = pytest.mark.parametrize("setting", law.SETTINGS)
BOUNDED = pytest.mark.parametrize("setting", law.BOUNDED_SETTINGS)


def _weights(frame: pd.DataFrame, setting: str) -> np.ndarray:
    weighted = law.LAWS[setting].weighted
    return frame[strata.WEIGHT].to_numpy(float) if weighted else np.ones(len(frame))


class TestTheFit:
    @SETTINGS
    def test_the_fit_reports_every_stratum_value_with_its_stored_curve(self, setting: str) -> None:
        _, fitted = _measured(setting)
        assert set(fitted.result.estimates) == set(law.PARAMETERS)
        for name in law.PARAMETERS:
            curve = np.asarray(fitted.result.estimates[name].influence_curve, dtype=float)
            assert curve.shape == (N,)
            assert float(fitted.result.estimates[name].std_error) == pytest.approx(
                strata.curve_se(curve), rel=1e-12
            )

    @SETTINGS
    def test_the_marginal_value_is_the_weighted_share_mixture(self, setting: str) -> None:
        frame, fitted = _measured(setting)
        weights = _weights(frame, setting)
        codes = frame["S"].to_numpy()
        shares = np.array([weights[codes == s].sum() for s in law.STRATA]) / weights.sum()
        parts = [float(fitted.result.estimates[law.parameter(f"s{s}")].psi) for s in law.STRATA]
        marginal = float(fitted.result.estimates[law.ESTIMAND].psi)
        assert marginal == pytest.approx(float(shares @ parts), abs=1e-12)

    @SETTINGS
    def test_the_boundary_stratum_splits_and_the_others_lean_to_the_limit(
        self, setting: str
    ) -> None:
        """The rule boundary lies in the declared stratum: there the fit treats a share strictly
        between 0.1 and 0.9, while every other stratum leans to the limit rule's side."""
        _, fitted = _measured(setting)
        boundary = law.boundary_stratum(setting)
        assert boundary == (1 if setting in law.BOUNDED_SETTINGS else 2)
        limit = law.limit_rule(setting)
        for code in law.STRATA:
            share = law.treated_share(fitted.result, code)
            if code == boundary:
                assert 0.1 < share < 0.9
            else:
                middle = np.array([0.5 * (strata.EDGES[code] + strata.EDGES[code + 1])])
                assert (share > 0.5) == bool(limit(middle, 0.0)[0])


class TestTheHarness:
    @SETTINGS
    def test_the_refit_reproduces_every_fold_rule_on_every_row(self, setting: str) -> None:
        _, fitted = _measured(setting)
        truth = fitted.truth
        assert truth.rows_checked == N
        assert len(truth.fold_values) == law.N_FOLDS
        for code in law.STRATA:
            expected = np.mean([values[code] for values in truth.fold_values])
            assert truth.truths[law.parameter(f"s{code}")] == pytest.approx(expected, rel=1e-15)
        mixture = sum(
            share * truth.truths[law.parameter(f"s{code}")]
            for code, share in zip(law.STRATA, law.SHARES, strict=True)
        )
        assert truth.truths[law.ESTIMAND] == pytest.approx(mixture, rel=1e-15)
        assert all(value > 0.0 for value in truth.oracle_ses.values())

    def test_a_rule_that_differs_on_one_row_stops_the_replication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, fitted = _measured("unweighted")
        honest = base.rule_of(fitted.result)

        def flipped(result: object) -> np.ndarray:
            rule = honest.copy()
            rule[7] = ~rule[7]
            return rule

        monkeypatch.setattr(base, "rule_of", flipped)
        with pytest.raises(base.HarnessMismatch, match="1 of"):
            law.harness(fitted.result, law.learner_of("unweighted"), "unweighted")

    def test_a_stored_fold_rule_that_differs_off_its_fold_stops_the_replication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second check reads the rows outside each fold, which rule L4 does not."""
        _, fitted = _measured("unweighted")
        stored = law.fold_rules(fitted.result.nuisance)
        assert stored is not None
        _, test = next(iter(fitted.result.nuisance.folds))
        outside = np.setdiff1d(np.arange(N), np.asarray(test))[0]
        edited = stored.copy()
        edited[outside, 0] = ~edited[outside, 0]
        monkeypatch.setattr(law, "fold_rules", lambda nuisance: edited)
        with pytest.raises(base.HarnessMismatch, match="stored fold rule on 1 of"):
            law.harness(fitted.result, law.learner_of("unweighted"), "unweighted")

    @BOUNDED
    @pytest.mark.filterwarnings(
        "ignore::tests.studies.fractional_glm.QuasiBinomialSeparationWarning"
    )
    def test_the_refit_must_read_a_proportion_as_a_proportion(
        self, setting: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate mutation: RM30's refit reads the outcome as a class label, which on a
        proportion learns another rule, so the harness must refuse it."""
        _, fitted = _measured(setting)
        monkeypatch.setattr(law, "outcome_task", lambda result: "classification")
        with pytest.raises(base.HarnessMismatch, match="validation rows"):
            law.harness(fitted.result, law.learner_of(setting), setting)

    @SETTINGS
    def test_a_known_rule_integrates_to_the_arm_means(
        self, setting: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rule that treats stratum 1 only, read from the grid's ``S`` column, integrates to
        the treated arm's mean in stratum 1 and E1's ``ey[0]`` truth elsewhere. A grid without
        the stratum column, or with the wrong code, would give another number."""
        spec = law.LAWS[setting]
        twin = "weighted" if spec.weighted else "unweighted"
        control = strata.truths(twin)
        weights = law._weights(setting)
        low, high = strata.EDGES[1], strata.EDGES[2]
        treated = sum(
            w
            * quad(
                lambda x, w2=w2: float(strata.qbar0(1.0, x, w2, spec.slope, spec.intercept)),
                low,
                high,
                epsabs=1e-13,
                epsrel=1e-12,
            )[0]
            for w2, w in weights.items()
        ) / ((high - low) * sum(weights.values()))

        def stratum_one(result: Any, learner: Any, fold: int) -> Any:
            return lambda covariates: covariates[:, 2] == 1.0

        _, fitted = _measured(setting)
        monkeypatch.setattr(law, "fold_rule", stratum_one)
        monkeypatch.setattr(law, "fold_rules", lambda nuisance: None)
        with pytest.raises(base.HarnessMismatch, match="stores no fold rules"):
            law.harness(fitted.result, None, setting)
        rule = stratum_one(None, None, 0)
        moments = law.rule_moments(
            setting,
            {
                (code, w2): rule(law.grid_design(code, w2))
                for code in law.STRATA
                for w2 in base.W2_LEVELS
            },
        )
        for code in law.STRATA:
            expected = treated if code == 1 else control[f"ey[0][S={code}]"]
            assert moments.stratum_values[code] == pytest.approx(expected, abs=1e-8)
        assert treated - control["ey[0][S=1]"] != pytest.approx(0.0, abs=0.005)

    def test_the_grid_carries_the_stratum_code_of_its_interval(self) -> None:
        for code in law.STRATA:
            for w2 in base.W2_LEVELS:
                design = law.grid_design(code, w2)
                assert design.shape == (strata.POINTS, 3)
                assert (design[:, 2] == code).all() and (design[:, 1] == w2).all()
                np.testing.assert_array_equal(
                    strata.stratum_of(design[:-1, 0]), np.full(strata.POINTS - 1, code)
                )
        # The upper edge of a stratum's grid is the next stratum's first point, a measure-zero
        # point whose own code the grid does not read.
        assert strata.stratum_of(law.grid_design(0, 0.0)[-1, 0]) == 1

    def test_the_truth_moves_with_the_learner(self) -> None:
        """The targeting control's truth is the value of the rules its own learner learned: a
        rule of ``W2`` alone, so it treats every row of a ``W2`` level alike in each fold."""
        frame, fitted = _measured("unweighted", "drop_w1")
        stored = law.fold_rules(fitted.result.nuisance)
        assert stored is not None
        w2 = frame["W2"].to_numpy()
        for fold in range(law.N_FOLDS):
            for level in base.W2_LEVELS:
                assert len(set(stored[w2 == level, fold])) == 1
        _, declared = _measured("unweighted")
        assert fitted.truth.truths[law.parameter("s0")] != pytest.approx(
            declared.truth.truths[law.parameter("s0")], abs=0.02
        )


class TestTheUnnormalisedControl:
    @SETTINGS
    def test_it_is_the_curve_times_the_fold_share(self, setting: str) -> None:
        frame, fitted = _measured(setting)
        folds = strata.fold_index(fitted.result)
        weights = _weights(frame, setting)
        codes = frame["S"].to_numpy()
        for code in law.STRATA:
            label = f"s{code}"
            share = np.array(
                [
                    weights[(folds == v) & (codes == code)].sum() / weights[folds == v].sum()
                    for v in range(law.N_FOLDS)
                ]
            )[folds]
            curve = fitted.result.estimates[law.parameter(label)].influence_curve
            expected = np.sqrt(np.sum((curve * share) ** 2)) / N
            assert law.unnormalised_se(fitted.result, label) == pytest.approx(expected, rel=1e-12)
            ratio = law.unnormalised_se(fitted.result, label) / float(
                fitted.result.estimates[law.parameter(label)].std_error
            )
            assert ratio == pytest.approx(law.SHARES[code], abs=0.05)

    def test_the_marginal_label_is_refused(self) -> None:
        _, fitted = _measured("unweighted")
        with pytest.raises(ValueError, match="marginal"):
            law.unnormalised_se(fitted.result, "all")


class TestTheShareDroppedControl:
    @SETTINGS
    def test_it_is_the_marginal_curve_less_its_share_term(self, setting: str) -> None:
        frame, fitted = _measured(setting)
        weights = _weights(frame, setting)
        codes = frame["S"].to_numpy()
        psi = {s: float(fitted.result.estimates[law.parameter(f"s{s}")].psi) for s in law.STRATA}
        marginal = fitted.result.estimates[law.ESTIMAND]
        term = weights / weights.mean() * (np.array([psi[s] for s in codes]) - float(marginal.psi))
        less = np.asarray(marginal.influence_curve, dtype=float) - term
        assert law.share_dropped_se(fitted.result) == pytest.approx(
            np.sqrt(np.sum(less**2)) / N, rel=1e-9
        )

    @BOUNDED
    def test_the_share_term_is_load_bearing_on_the_bounded_outcome(self, setting: str) -> None:
        _, fitted = _measured(setting)
        ratio = law.share_dropped_se(fitted.result) / float(
            fitted.result.estimates[law.ESTIMAND].std_error
        )
        assert ratio < 0.85


class TestTheTargetingControl:
    @SETTINGS
    def test_the_untargeted_plug_in_is_the_fold_average_at_the_fits_rule(
        self, setting: str
    ) -> None:
        frame, fitted = _measured(setting)
        folds = strata.fold_index(fitted.result)
        weights = _weights(frame, setting)
        codes = frame["S"].to_numpy()
        q0, q1 = strata.initial_arms(fitted.result)
        rule = law.fold_rules(fitted.result.nuisance)
        assert rule is not None
        own = rule[np.arange(N), folds]
        values = np.where(own, q1, q0)
        parts = []
        for code in law.STRATA:
            fold_values = [
                np.average(values[mask], weights=weights[mask])
                for mask in ((folds == v) & (codes == code) for v in range(law.N_FOLDS))
            ]
            parts.append(np.mean(fold_values))
            assert law.untargeted(fitted.result, f"s{code}") == pytest.approx(parts[-1], rel=1e-12)
        shares = np.array([weights[codes == s].sum() for s in law.STRATA]) / weights.sum()
        assert law.untargeted(fitted.result, "all") == pytest.approx(shares @ parts, rel=1e-12)

    def test_the_control_learner_is_stratum_blind_and_the_fluctuation_repairs_it(self) -> None:
        _, fitted = _measured("unweighted", "drop_w1")
        plug_ins = [law.untargeted(fitted.result, f"s{s}") for s in law.STRATA]
        truths = [fitted.truth.truths[law.parameter(f"s{s}")] for s in law.STRATA]
        assert np.ptp(plug_ins) < 0.03 < 0.2 < np.ptp(truths)
        for label in study.NECESSITY_LABELS:
            parameter_label = label.removeprefix("unweighted_")
            targeted = float(fitted.result.estimates[law.parameter(parameter_label)].psi)
            assert abs(targeted - law.untargeted(fitted.result, parameter_label)) > 0.01


def _property_rows() -> pd.DataFrame:
    return _cached_property_rows().copy()


@cache
def _cached_property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(
        n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestThePropertyPath:
    """``generate_property_rows`` and ``summarize_properties`` end to end on a few draws."""

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
        assert set(rows["cell"]) == declared
        assert (rows.groupby("cell").size() == CAP).all()
        assert (rows["n"] == properties.CALIBRATION_N).all()
        assert (rows["rule_rows_checked"] == N).all()
        summary = properties.summarize_properties(rows).set_index("cell")
        assert set(summary.index) == declared

    def test_each_row_reads_its_own_truth(self) -> None:
        """The positives of one draw share no truth: each parameter reads its own target."""
        rows = _property_rows()
        for setting in law.SETTINGS:
            cells = [f"{setting}_{label}__declared" for label in law.PARAMETER_LABELS]
            group = rows.loc[rows["cell"].isin(cells)]
            by_replicate = group.groupby("replicate")["truth"].nunique()
            assert (by_replicate == len(cells)).all()
            # The truth of the boundary stratum and of the marginal value varies by replication,
            # which the record declares. A stratum the fold rules treat alike throughout has a
            # fixed truth.
            varying = group.groupby("cell")["truth"].nunique()
            boundary = law.boundary_stratum(setting)
            assert varying[f"{setting}_s{boundary}__declared"] > 1
            assert varying[f"{setting}_all__declared"] > 1
        assert study.STUDY.truth_varies_by_replicate

    def test_the_controls_share_the_positive_estimate_and_truth(self) -> None:
        rows = _property_rows()
        pairs = [(label, study.UNNORMALISED) for label in study.STRATUM_LABELS] + [
            (label, study.SHARE_DROPPED) for label in study.SHARE_DROPPED_LABELS
        ]
        for label, kind in pairs:
            positive = rows.loc[rows["cell"] == f"{label}__declared"].set_index("replicate")
            control = rows.loc[rows["cell"] == f"{label}__{kind}"].set_index("replicate")
            assert (control["role"] == "control").all()
            np.testing.assert_array_equal(control["estimate"], positive["estimate"])
            np.testing.assert_array_equal(control["truth"], positive["truth"])
            assert (control["std_error"] < positive["std_error"]).all()

    def test_the_shared_rule_reproduces_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        rule = property_failures.rule_for("interval_calibration", properties)
        assert rule is not None
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
        own = summary.set_index("cell")
        calibration = own.loc[own["property"] == "interval_calibration"]
        assert set(verdicts.index) == set(calibration.index)
        for cell in verdicts.index:
            assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell
        assert property_failures.undeclared_verdict_writers(properties) == []

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "weighted", 1), 4 * 3 + 3 + 1),
            (("interval_calibration", "binary_steep_weighted", 1), 4 * 3 + 3),
            (("targeting_necessity", "unweighted", 1), 6),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One empty stratum-fold refusal is one lost draw of all four parameters' cells: the
        positives with their controls (no share-dropped control in a binary setting), or the
        six necessity cells."""
        cached = _property_rows()

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, setting, replicate = key[0][:3]
            if (family, setting, replicate) == lost:
                raise DataError("validation fold 2 holds no row of baseline stratum S=0")
            pattern = rf"^{setting}_(?:s0|s1|s2|all)__"
            derived = {c.split("/", 1)[1] for cs in properties.DERIVED_CELLS.values() for c in cs}
            mine = cached.loc[
                (cached["property"] == family)
                & (cached["replicate"] == replicate)
                & cached["cell"].str.match(pattern)
                & ~cached["cell"].isin(derived)
            ]
            return mine.to_dict("records")

        monkeypatch.setattr(properties, "_replication", worker)
        with replication.recording(study.STUDY.implementation) as recorded:
            rows = properties.generate_property_rows(
                n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
            )
        records = property_failures.frame(recorded)
        assert records is not None and len(records) == 1
        rows = property_failures.stamp(rows, records)
        property_failures.reconcile(rows, records, derived=properties.DERIVED_CELLS)
        stamped = rows.groupby("cell")["failed_replicates"].max()
        hit = {
            cell.split("/", 1)[1]
            for cell in properties._cells_of((lost[:2],))  # type: ignore[arg-type]
        }
        assert len(hit) == cost
        assert {cell for cell, count in stamped.items() if count} == hit
        assert (rows.loc[rows["cell"].isin(hit)].groupby("cell").size() == CAP - 1).all()

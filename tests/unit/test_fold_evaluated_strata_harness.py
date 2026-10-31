"""Unit witnesses for the fits, the controls and the property path of ``fold-evaluated-strata``.

The study's calibration cells read the reported SE of each stratum and marginal parameter, its
two construction controls recompute an SE from the stored curves, and its targeting control
recomputes the untargeted fold plug-in. A control computed from the wrong share, the wrong curve
or the wrong regression would pass or fail a cell for the wrong reason. These checks rebuild each
piece another way, from the frame and the fold labels, and each deliberate mutation must move it.

Every fit draws from a throwaway seed at the declared size n = 1,000, where a fit takes about a
tenth of a second. No test asserts a bias, a coverage or an SE ratio of the study from one sample.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import DataError
from cleverly.learners.crossfit import make_folds
from tests.studies import _fold_evaluated_strata_law as law
from tests.studies import _learned_rule_law as base
from tests.studies import fold_evaluated_strata as study
from tests.studies import fold_evaluated_strata_properties as properties
from tests.studies.evidence import property_failures, replication

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_994_000
N = 1_000
CAP = 3


@cache
def _measured(setting: str, learner: str = "declared") -> tuple[pd.DataFrame, law.Measured]:
    frame = law.draw(setting, N, THROWAWAY + law.SETTINGS.index(setting))
    chosen = law.drop_w1_learner() if learner == "drop_w1" else None
    return frame, law.measured(frame, setting, THROWAWAY + 10, learner=chosen)


@cache
def _paired() -> tuple[pd.DataFrame, law.Paired]:
    frame = law.draw(law.SPARSE, N, THROWAWAY + 20)
    return frame, law.paired(frame, law.SPARSE, THROWAWAY + 10)


SETTINGS = pytest.mark.parametrize("setting", law.SETTINGS)
BOUNDED = pytest.mark.parametrize("setting", law.BOUNDED_SETTINGS)


def _frame_folds(frame: pd.DataFrame, result: Any) -> np.ndarray:
    """The fold of each row, rebuilt from the fold seed rather than read off the fit."""
    folds = make_folds(len(frame), law.N_FOLDS, random_state=THROWAWAY + 10)
    index = np.empty(len(frame), dtype=np.int64)
    for fold, (_, test) in enumerate(folds):
        index[np.asarray(test)] = fold
    np.testing.assert_array_equal(index, law.fold_index(result))
    return index


def _frame_weights(frame: pd.DataFrame, setting: str) -> np.ndarray:
    weighted = law.LAWS[setting].weighted
    return frame[law.WEIGHT].to_numpy(float) if weighted else np.ones(len(frame))


class TestTheFit:
    @SETTINGS
    def test_the_fit_reports_every_parameter_with_its_stored_curve(self, setting: str) -> None:
        """Each reported SE is ``sqrt(sum phi^2) / n`` of its stored curve, the rule the controls
        reuse, and the weighted fit reads the weight column."""
        _, fitted = _measured(setting)
        result = fitted.result
        assert set(law.PARAMETERS) <= set(result.estimates)
        for name in law.PARAMETERS:
            estimate = result.estimates[name]
            assert estimate.std_error == pytest.approx(
                law.curve_se(estimate.influence_curve), rel=1e-10
            )
        weights = np.asarray(result.data.weights, dtype=float)
        assert bool(np.ptp(weights) > 0) is law.LAWS[setting].weighted
        assert result.config.cv_evaluation and result.config.n_folds == law.N_FOLDS

    @SETTINGS
    def test_each_setting_draws_its_outcome_and_fits_its_learner(self, setting: str) -> None:
        """A binary setting draws ``Y`` in {0, 1} and fits the logistic learner without
        ``q_bounds``; a bounded one draws a proportion strictly inside (0, 1)."""
        frame, _ = _measured(setting)
        y = frame["Y"].to_numpy()
        binary = law.LAWS[setting].outcome == "binary"
        if binary:
            assert set(np.unique(y)) == {0.0, 1.0}
        else:
            assert y.min() > 0.0 and y.max() < 1.0 and len(np.unique(y)) == N
        estimator = law.estimator(5, setting)
        assert estimator.q_bounds == (None if binary else (0.0, 1.0))
        model = estimator.outcome_learner.named_steps["model"]
        assert type(model).__name__ == ("LogisticRegression" if binary else "QuasiBinomialGLM")
        # The binary twin draws the bounded setting's covariates and treatment from one seed.
        twin = law.draw("unweighted", 50, 3)
        np.testing.assert_array_equal(
            law.draw(setting, 50, 3)[["W1", "W2", "A"]], twin[["W1", "W2", "A"]]
        )

    def test_the_stratum_column_is_the_declared_cut(self) -> None:
        frame, fitted = _measured("unweighted")
        np.testing.assert_array_equal(frame["S"], law.stratum_of(frame["W1"]))
        # The fit numbers its strata in the order the data present them; the controls read S.
        np.testing.assert_array_equal(law.stratum_values(fitted.result), frame["S"])
        shares = np.bincount(frame["S"].to_numpy(), minlength=3) / N
        np.testing.assert_allclose(shares, law.SHARES, atol=0.05)

    def test_the_fits_stratum_codes_are_not_the_values(self) -> None:
        """The nonzero witness for :func:`law.stratum_values`: on the rows-only sample the fit's
        codes differ from ``S``, so a control that read the codes as values would read the wrong
        stratum's rows, and the rebuilds below would disagree with it."""
        frame, fitted = _measured("unweighted")
        codes = np.asarray(fitted.result.data.strata)
        assert np.mean(codes != frame["S"].to_numpy()) > 0.2

    @SETTINGS
    def test_the_marginal_estimate_is_the_weighted_share_mixture(self, setting: str) -> None:
        frame, fitted = _measured(setting)
        weights = _frame_weights(frame, setting)
        strata = frame["S"].to_numpy()
        shares = np.array([weights[strata == s].sum() for s in law.STRATA]) / weights.sum()
        for target, name in law.TARGETS.items():
            parts = [
                fitted.result.estimates[law.parameter(f"{target}_s{s}")].psi for s in law.STRATA
            ]
            assert fitted.result.estimates[name].psi == pytest.approx(shares @ parts, abs=1e-12)


class TestTheUnnormalisedControl:
    @SETTINGS
    def test_it_is_the_curve_times_the_fold_share_rebuilt_from_the_frame(
        self, setting: str
    ) -> None:
        frame, fitted = _measured(setting)
        folds = _frame_folds(frame, fitted.result)
        weights = _frame_weights(frame, setting)
        strata = frame["S"].to_numpy()
        for target in law.TARGETS:
            for code in law.STRATA:
                label = f"{target}_s{code}"
                share = np.array(
                    [
                        weights[(folds == v) & (strata == code)].sum() / weights[folds == v].sum()
                        for v in range(law.N_FOLDS)
                    ]
                )[folds]
                curve = fitted.result.estimates[law.parameter(label)].influence_curve
                expected = np.sqrt(np.sum((curve * share) ** 2)) / N
                assert law.unnormalised_se(fitted.result, label) == pytest.approx(
                    expected, rel=1e-12
                )

    @SETTINGS
    def test_the_fold_share_is_not_the_global_share(self, setting: str) -> None:
        """The nonzero witness: the fold shares differ from the sample share, so a control built
        with the global share would be another number."""
        _, fitted = _measured(setting)
        for code in law.STRATA:
            fold = law.fold_shares(fitted.result, code)
            weights = np.asarray(fitted.result.data.weights, dtype=float)
            strata = law.stratum_values(fitted.result)
            overall = weights[strata == code].sum() / weights.sum()
            assert np.ptp(fold) > 0.005
            assert np.max(np.abs(fold - overall)) > 0.002

    def test_the_weighted_share_reads_weight_mass_not_row_counts(self) -> None:
        """The mutation the weighted cell exists for: row-count shares differ from mass shares."""
        frame, fitted = _measured("weighted")
        folds = law.fold_index(fitted.result)
        strata = frame["S"].to_numpy()
        for code in law.STRATA:
            counts = np.array([np.mean(strata[folds == v] == code) for v in range(law.N_FOLDS)])[
                folds
            ]
            assert np.max(np.abs(law.fold_shares(fitted.result, code) - counts)) > 0.002

    @SETTINGS
    def test_it_shrinks_each_stratum_se_by_about_its_share(self, setting: str) -> None:
        _, fitted = _measured(setting)
        for target in law.TARGETS:
            for code, share in zip(law.STRATA, law.SHARES, strict=True):
                label = f"{target}_s{code}"
                ratio = law.unnormalised_se(fitted.result, label) / float(
                    fitted.result.estimates[law.parameter(label)].std_error
                )
                assert ratio == pytest.approx(share, abs=0.03)

    def test_a_marginal_label_is_refused(self) -> None:
        _, fitted = _measured("unweighted")
        with pytest.raises(ValueError, match="marginal"):
            law.unnormalised_se(fitted.result, "ate_all")


class TestTheShareDroppedControl:
    @SETTINGS
    def test_it_is_the_marginal_curve_less_its_share_term(self, setting: str) -> None:
        """Two routes: the stratum part of the mixture from the stored stratum curves, and the
        reported marginal curve less the share term recomputed from the reported estimates."""
        _, fitted = _measured(setting)
        for target, name in law.TARGETS.items():
            label = f"{target}_all"
            curve = np.asarray(fitted.result.estimates[name].influence_curve, dtype=float)
            stratum_part = curve - law.share_term(fitted.result, label)
            assert law.share_dropped_se(fitted.result, label) == pytest.approx(
                law.curve_se(stratum_part), rel=1e-10
            )

    @SETTINGS
    def test_the_share_term_is_load_bearing_for_ate_and_ey1(self, setting: str) -> None:
        """On one sample each ratio sits near its asymptotic value: in a bounded setting about
        0.85 for ``ate``, 0.66 for ``ey[1]``, and 0.95 for ``ey[0]``, which carries no control;
        in a binary setting all three sit above the band floor, so no binary label carries
        one."""
        _, fitted = _measured(setting)

        def ratio(target: str) -> float:
            label = f"{target}_all"
            reported = float(fitted.result.estimates[law.TARGETS[target]].std_error)
            return law.share_dropped_se(fitted.result, label) / reported

        for target in law.TARGETS:
            assert ratio(target) == pytest.approx(law.share_term_ratio(setting, target), abs=0.06)
        if setting in law.BOUNDED_SETTINGS:
            assert ratio("ey1") < ratio("ate") < 0.93 < ratio("ey0")
        else:
            assert min(ratio(target) for target in law.TARGETS) > 0.9
            assert not any(label.startswith(setting) for label in study.SHARE_DROPPED_LABELS)

    def test_a_stratum_label_is_refused(self) -> None:
        _, fitted = _measured("unweighted")
        with pytest.raises(ValueError, match="stratum parameter"):
            law.share_dropped_se(fitted.result, "ate_s1")


class TestTheTargetingControl:
    @SETTINGS
    def test_the_untargeted_plug_in_is_the_fold_average_of_the_initial_regression(
        self, setting: str
    ) -> None:
        frame, fitted = _measured(setting)
        folds = _frame_folds(frame, fitted.result)
        weights = _frame_weights(frame, setting)
        strata = frame["S"].to_numpy()
        q0, q1 = law.initial_arms(fitted.result)
        for target, values in {"ate": q1 - q0, "ey0": q0, "ey1": q1}.items():
            parts = []
            for code in law.STRATA:
                fold_values = [
                    np.average(values[mask], weights=weights[mask])
                    for mask in ((folds == v) & (strata == code) for v in range(law.N_FOLDS))
                ]
                parts.append(np.mean(fold_values))
                assert law.untargeted(fitted.result, f"{target}_s{code}") == pytest.approx(
                    parts[-1], rel=1e-12
                )
            shares = np.array([weights[strata == s].sum() for s in law.STRATA]) / weights.sum()
            assert law.untargeted(fitted.result, f"{target}_all") == pytest.approx(
                shares @ parts, rel=1e-12
            )

    def test_the_initial_regression_is_out_of_fold(self) -> None:
        """Fold 0's initial ``Qbar(1, W)`` is the learner fitted on the other folds' rows."""
        frame, fitted = _measured("unweighted")
        own = law.fold_index(fitted.result) == 0
        design = frame[["A", *law.COVARIATES]].to_numpy(float)
        model = law.outcome_learner().fit(design[~own], frame["Y"].to_numpy()[~own])
        treated = design[own].copy()
        treated[:, 0] = 1.0
        _, q1 = law.initial_arms(fitted.result)
        np.testing.assert_allclose(q1[own], model.predict_proba(treated)[:, 1], atol=1e-6)
        whole = law.outcome_learner().fit(design, frame["Y"].to_numpy())
        assert np.max(np.abs(q1[own] - whole.predict_proba(treated)[:, 1])) > 1e-3

    def test_the_fluctuation_moves_every_stratum_estimate(self) -> None:
        """With ``drop_w1_learner()`` the targeted and untargeted stratum estimates differ."""
        _, fitted = _measured("unweighted", "drop_w1")
        for label in study.NECESSITY_LABELS:
            parameter_label = label.removeprefix("unweighted_")
            targeted = float(fitted.result.estimates[law.parameter(parameter_label)].psi)
            assert abs(targeted - law.untargeted(fitted.result, parameter_label)) > 0.01

    def test_the_control_learner_ignores_the_stratum(self) -> None:
        """``drop_w1_learner()`` keeps ``A`` and ``W2`` only, so its stratum plug-ins nearly
        agree, while the truths spread across strata."""
        _, fitted = _measured("unweighted", "drop_w1")
        truth = law.truths("unweighted")
        for target in law.TARGETS:
            plug_ins = [law.untargeted(fitted.result, f"{target}_s{s}") for s in law.STRATA]
            truths = [truth[law.parameter(f"{target}_s{s}")] for s in law.STRATA]
            assert np.ptp(plug_ins) < 0.03 < 0.1 < np.ptp(truths)


class TestTheEmptyCellRefusal:
    def test_the_frame_check_is_the_fits_refusal(self) -> None:
        """A sample whose stratum 0 holds no treated row in one fold refuses with a DataError, and
        the pilot's frame check flags it."""
        frame = law.draw("unweighted", N, THROWAWAY + 7)
        folds = make_folds(N, law.N_FOLDS, random_state=THROWAWAY + 10)
        (_, test), *_ = folds
        test = np.asarray(test)
        emptied = frame.copy()
        rows = test[emptied["S"].to_numpy()[test] == 0]
        emptied.loc[rows, "A"] = 0.0
        assert not law.empty_cell(frame, "unweighted", folds)
        assert law.empty_cell(emptied, "unweighted", folds)
        with pytest.raises(DataError, match="inside baseline stratum"):
            law.fit(emptied, "unweighted", THROWAWAY + 10)


class TestTheSparseEventDiagnostic:
    def test_the_stacked_fit_is_the_paired_fit_on_the_same_folds(self) -> None:
        """Both fits read the same sample and the same folds; only the evaluation differs, so
        the estimates are close but not equal."""
        _, fitted = _paired()
        declared, stacked = fitted.declared.result, fitted.stacked.result
        assert declared.config.cv_evaluation and not stacked.config.cv_evaluation
        np.testing.assert_array_equal(law.fold_index(declared), law.fold_index(stacked))
        gaps = [
            abs(float(declared.estimates[name].psi) - float(stacked.estimates[name].psi))
            for name in law.PARAMETERS
        ]
        assert 0.0 < max(gaps) < 0.05

    def test_a_mismatched_fold_draw_stops_the_draw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame, _ = _paired()
        calls: list[int] = []
        real = law.fold_index

        def shifted(result: Any) -> np.ndarray:
            calls.append(1)
            index = real(result)
            return np.roll(index, len(calls) - 1)

        monkeypatch.setattr(law, "fold_index", shifted)
        with pytest.raises(RuntimeError, match="other folds"):
            law.paired(frame, law.SPARSE, THROWAWAY + 10)

    def test_the_event_count_is_the_rarer_value_of_each_stratum_arm(self) -> None:
        """Rebuilt from the frame; ``ey[1]`` in stratum 0 has few events, the nonzero witness
        that the cell reads the sparse regime."""
        frame, _ = _paired()
        s, a, y = (frame[c].to_numpy() for c in ("S", "A", "Y"))

        def rarer(code: int, arm: float) -> int:
            cell = y[(s == code) & (a == arm)]
            return int(min(cell.sum(), cell.size - cell.sum()))

        assert law.stratum_arm_events(frame, "ey1_s0") == rarer(0, 1.0) < 15
        assert law.stratum_arm_events(frame, "ey0_s2") == rarer(2, 0.0)
        assert law.stratum_arm_events(frame, "ate_s1") == min(rarer(1, 0.0), rarer(1, 1.0))
        assert law.stratum_arm_events(frame, "ate_all") == min(
            rarer(code, arm) for code in law.STRATA for arm in (0.0, 1.0)
        )
        with pytest.raises(ValueError, match="binary outcome"):
            law.stratum_arm_events(law.draw("unweighted", 50, 3), "ate_s0")

    def test_the_sparse_law_is_the_steeper_blip(self) -> None:
        truth = law.truths(law.SPARSE)
        assert truth["ey[1][S=0]"] == pytest.approx(0.0710, abs=5e-4)
        assert truth["ey[0][S=0]"] == law.truths("unweighted")["ey[0][S=0]"]


@cache
def _property_rows() -> pd.DataFrame:
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
        summary = properties.summarize_properties(rows).set_index("cell")
        assert set(summary.index) == declared

    def test_each_row_reads_its_own_parameter_and_truth(self) -> None:
        rows = _property_rows()
        for setting in law.SETTINGS:
            truth = law.truths(setting)
            for label in law.PARAMETER_LABELS:
                cell = f"{setting}_{label}__declared"
                group = rows.loc[rows["cell"] == cell]
                assert (group["truth"] == truth[law.parameter(label)]).all(), cell

    def test_the_controls_share_the_positive_estimate(self) -> None:
        """Each construction control is the positive's estimate with another SE."""
        rows = _property_rows()
        pairs = [(label, study.UNNORMALISED) for label in study.STRATUM_LABELS] + [
            (label, study.SHARE_DROPPED) for label in study.SHARE_DROPPED_LABELS
        ]
        for label, kind in pairs:
            positive = rows.loc[rows["cell"] == f"{label}__declared"].set_index("replicate")
            control = rows.loc[rows["cell"] == f"{label}__{kind}"].set_index("replicate")
            assert (control["role"] == "control").all()
            np.testing.assert_array_equal(control["estimate"], positive["estimate"])
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

    def test_the_sparse_cells_report_both_fits_and_state_no_verdict(self) -> None:
        rows = _property_rows()
        sparse = rows.loc[rows["property"] == study.SPARSE_FAMILY]
        assert set(sparse["role"]) == {"diagnostic"}
        assert sparse[list(properties.SPARSE_COLUMNS)].notna().all().all()
        others = rows.loc[rows["property"] != study.SPARSE_FAMILY]
        assert others[list(properties.SPARSE_COLUMNS)].isna().all().all()
        assert (sparse["stacked_estimate"] != sparse["estimate"]).any()
        summary = properties.summarize_properties(rows)
        reported = summary.loc[summary["property"] == study.SPARSE_FAMILY]
        assert len(reported) == 12 and set(reported["role"]) == {"diagnostic"}
        assert reported["passed"].all() and reported["property_passed"].all()
        # A bootstrap of three draws can resample one value three times, whose spread is zero,
        # so an upper ratio endpoint may be undefined here; every other column is a number.
        columns = [
            *(c for c in properties.SPARSE_SUMMARY if not c.endswith("_ci_upper")),
            "se_ratio_ci_lower",
        ]
        assert reported[columns].notna().all().all()
        for _, row in reported.iterrows():
            group = sparse.loc[sparse["cell"] == row["cell"]]
            assert row["events_mean"] == pytest.approx(group["stratum_arm_events"].mean())
            covered = np.abs(group["stacked_estimate"] - group["truth"]) <= (
                law.CRITICAL * group["stacked_std_error"]
            )
            assert row["stacked_coverage"] == pytest.approx(covered.mean())
        # The diagnostic writes verdict columns, so the module declares its rules.
        assert property_failures.undeclared_verdict_writers(properties) == []
        assert property_failures.rule_for(study.SPARSE_FAMILY, properties) is None

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "weighted", 1), 12 * 3 + 9 + 2),
            (("interval_calibration", "binary_weighted", 1), 12 * 3 + 9),
            ((study.SPARSE_FAMILY, law.SPARSE, 1), 12),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One empty stratum-fold refusal is one lost draw of all 12 parameters' cells: the
        positives with their controls (no share-dropped control in a binary setting), or the
        12 diagnostic cells."""
        cached = _property_rows()

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, setting, replicate = key[0][:3]
            if (family, setting, replicate) == lost:
                raise DataError("validation fold 2 holds no row of baseline stratum S=0")
            prefix = "unweighted_" if family == "targeting_necessity" else f"{setting}_"
            derived = {c.split("/", 1)[1] for cs in properties.DERIVED_CELLS.values() for c in cs}
            mine = cached.loc[
                (cached["property"] == family)
                & (cached["replicate"] == replicate)
                & cached["cell"].str.startswith(prefix)
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

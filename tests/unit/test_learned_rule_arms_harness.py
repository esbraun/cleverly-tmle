"""Unit witnesses for the law, the fits, the truth harness and the controls of ``learned-rule-arms``.

The study's verdicts read the error ``estimate - truth`` of each row, so a wrong truth, a wrong
arm in the rule, a learner other than the declared one, or a control that does not compute what
it names would fail or pass a cell for the wrong reason. These checks recompute each piece
another way. Every fit draws from a throwaway seed, and no test asserts a bias, a coverage or an
SE ratio.
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from cleverly._support import learner_family
from cleverly.exceptions import DataError
from cleverly.interventions.learned import fold_rules, realized_arms
from tests.studies import _learned_rule_arms_law as law
from tests.studies import learned_rule_arms as study
from tests.studies import learned_rule_arms_properties as properties
from tests.studies.evidence import property_failures, replication

base = law.base

pytestmark = pytest.mark.xdist_group("learned_rule_arms_harness")

#: Throwaway seeds, far from every declared and pilot seed.
THROWAWAY = 9_981_000
N = 600
#: The property draws of each cell the end-to-end checks fit.
CAP = 3


@cache
def _frame(n: int = N) -> pd.DataFrame:
    return law.draw(n, THROWAWAY)


@cache
def _measured(terms: str = "declared") -> law.Measured:
    return law.measured(_frame(), law.outcome_learner(terms), THROWAWAY + 7)


@cache
def _cached_property_rows() -> pd.DataFrame:
    return properties.generate_property_rows(
        n_jobs=1, record=base.smoke_record(study.STUDY), cap=CAP
    )


def _property_rows() -> pd.DataFrame:
    return _cached_property_rows().copy()


# ------------------------------------------------------------------------------- the law


class TestTheLaw:
    def test_the_mechanism_sums_to_one_with_every_arm_at_least_0_15(self) -> None:
        for w2 in law.W2_LEVELS:
            g = law.g0(law.GRID, w2)
            np.testing.assert_allclose(g.sum(axis=0), 1.0, atol=1e-14)
            assert g.min() >= 0.15
        assert min(law.g0(law.GRID, w2).min() for w2 in law.W2_LEVELS) == pytest.approx(
            0.166170, abs=1e-6
        )

    def test_the_draw_follows_the_mechanism_and_the_outcome_law(self) -> None:
        """A 200,000-row draw against ``g0`` and ``Qbar0`` by arm (each within 4 MC SEs)."""
        frame = law.draw(200_000, THROWAWAY + 1)
        w1, w2, a = (frame[c].to_numpy() for c in ("W1", "W2", "A"))
        assert set(np.unique(a)) == {0.0, 1.0, 2.0}
        g = law.g0(w1, w2)
        for arm in law.ARMS:
            for moment in (np.ones_like(w1), w1):
                hit = (a == arm) * moment
                expected = g[arm] * moment
                assert abs(hit.mean() - expected.mean()) < 4 * hit.std() / np.sqrt(hit.size)
            y = frame["Y"].to_numpy()[a == arm]
            q = law.qbar0(arm, w1[a == arm], w2[a == arm])
            assert abs(y.mean() - q.mean()) < 4 * y.std() / np.sqrt(y.size)
        # A deliberate mutation: the arm drawn from g0 shifted by one code misses the W1 moment.
        shifted = (a + 1) % 3
        assert abs(np.mean((shifted == 1) * w1) - np.mean(g[1] * w1)) > 0.05

    def test_the_blips_are_the_declared_ones(self) -> None:
        w1 = np.array([-1.0, 0.0, 0.5])
        np.testing.assert_allclose(law.blip(1, w1, 1.0), -1.0 + 6.0 * w1)
        np.testing.assert_allclose(
            law.blip(2, w1, 1.0), -1.75 - 6.0 * w1 - 1.0 - 4.0 * (w1**2 - 1 / 3)
        )
        np.testing.assert_array_equal(law.blip(0, w1, 1.0), 0.0)


# ------------------------------------------------------------------------- the learners


class TestTheLearners:
    @pytest.mark.parametrize("terms", list(law.TERMS))
    def test_every_learner_reads_parametric_and_clones(self, terms: str) -> None:
        """Plan Q4: ``learner_family`` reads the marked class, not a Pipeline, as parametric."""
        learner = law.outcome_learner(terms)
        assert learner_family(learner) == "parametric"
        assert clone(learner).get_params() == {"terms": terms}
        assert learner_family(law.treatment_learner()) == "parametric"

    def test_the_treatment_learner_is_the_unpenalized_multinomial_mle(self) -> None:
        learner = law.treatment_learner()
        assert np.inf == learner.C and learner.max_iter == 5000
        frame = _frame()
        model = clone(learner).fit(frame[["W1", "W2"]].to_numpy(), frame["A"].to_numpy())
        assert model.coef_.shape == (3, 2)

    def test_the_treatment_learner_fits_without_a_warning(self) -> None:
        """``C=np.inf`` is the spelling scikit-learn 1.8 and later keep for the unpenalized fit;
        the deprecated ``penalty=None`` raises a ``FutureWarning`` there (design review
        finding 4)."""
        frame = _frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            law.treatment_learner().fit(frame[["W1", "W2"]].to_numpy(), frame["A"].to_numpy())

    def test_c_infinity_is_the_penalty_none_fit(self) -> None:
        """The condition the deviation from ``penalty=None`` rests on: the library maps both
        spellings to one solver path, so the coefficients agree. Skipped where ``penalty=None``
        warns, as it does from scikit-learn 1.8 until its removal."""
        frame = _frame()
        x, a = frame[["W1", "W2"]].to_numpy(), frame["A"].to_numpy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reference = LogisticRegression(penalty=None, max_iter=5000).fit(x, a)
        if caught:
            pytest.skip(f"penalty=None warns here: {caught[0].category.__name__}")
        model = law.treatment_learner().fit(x, a)
        np.testing.assert_allclose(model.coef_, reference.coef_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.intercept_, reference.intercept_, rtol=0, atol=1e-12)

    def test_the_declared_limit_is_the_true_regression(self) -> None:
        for w2 in law.W2_LEVELS:
            truth = np.vstack([law.qbar0(arm, law.GRID, w2) for arm in law.ARMS])
            np.testing.assert_allclose(
                law.limit_predictions("declared", law.GRID, w2), truth, atol=1e-10
            )
            # The deliberate mutation: the learner without the quadratic does not reach it.
            gap = np.abs(law.limit_predictions("missing_quadratic", law.GRID, w2) - truth)
            assert gap.max() > 0.1

    def test_the_features_are_the_declared_products(self) -> None:
        design = np.array([[0.0, 1.0, 0.5, 1.0], [1.0, 0.0, -0.5, 0.0]])
        np.testing.assert_allclose(
            law.features(design, law.TERMS["declared"]),
            [[0, 1, 0.5, 1, 0, 0, 0.5, 1, 0.25], [1, 0, -0.5, 0, -0.5, 0, 0, 0, 0]],
        )
        assert law.features(design, law.TERMS["drop_w1"]).shape == (2, 5)
        assert "W1" not in "".join(law.TERMS["drop_w1"])


# ------------------------------------------------------------------------------ the fits


class TestTheFits:
    def test_the_estimator_is_the_declared_one(self) -> None:
        estimator = law.estimator(law.outcome_learner(), 5)
        assert estimator.learned_rule is not None and estimator.learned_rule.loss == "plug_in"
        assert estimator.cv_evaluation and estimator.targeting_scheme == "pooled"
        assert estimator.n_folds == 10 and estimator.g_bounds == "auto"
        assert estimator.repeats == 1 and estimator.random_state == 5
        assert not estimator.interventions

    def test_the_fit_keys_three_arms_fold_evaluated(self) -> None:
        result = _measured().result
        key = result.config.support_key
        assert key.treatment == ("arms3",) and key.splitting == ("fold_evaluated",)
        assert tuple(result.estimates) == (law.ESTIMAND,)
        stored = fold_rules(result.nuisance)
        assert stored is not None and stored.dtype == np.int8 and stored.shape == (N, 10)
        assert set(np.unique(stored)) == {0, 1, 2}

    def test_every_row_and_fold_rule_is_checked(self) -> None:
        for terms in law.TERMS:
            assert _measured(terms).truth.rows_checked == N

    def test_the_arm_design_is_the_packages(self) -> None:
        data = _measured().result.data
        covariates = np.asarray(data.covariates, dtype=float)
        for arm in law.ARMS:
            np.testing.assert_array_equal(
                law.arm_design(arm, covariates), data.counterfactual_design(arm)
            )


# ----------------------------------------------------------------------------- the truth


class TestTheTruth:
    def test_the_truth_is_the_fold_average_of_the_fold_values(self) -> None:
        truth = _measured().truth
        assert len(truth.fold_values) == 10
        assert truth.truth == pytest.approx(np.mean(truth.fold_values), rel=1e-15)
        assert len(set(np.round(truth.fold_values, 12))) > 1

    def test_the_rule_value_and_variance_match_monte_carlo(self) -> None:
        """``Psi_d`` and ``Var D*(d)`` of one refit fold rule against 400,000 draws."""
        result = _measured().result
        data = result.data
        train, _ = next(iter(result.nuisance.folds))
        model = law.fit_on_rows(
            law.outcome_learner(),
            np.asarray(data.treatment_design(), dtype=float),
            np.asarray(data.outcome, dtype=float),
            np.asarray(data.weights, dtype=float),
            np.asarray(train),
            "classification",
            None,
        )
        value, variance = law.rule_value(law.grid_rule(model))
        frame = law.draw(400_000, THROWAWAY + 40)
        w1, w2, a, y = (frame[c].to_numpy() for c in ("W1", "W2", "A", "Y"))
        rule = law.argmax_rule(model, np.column_stack([w1, w2])).astype(int)
        q = law.qbar0_rows(rule, w1, w2)
        g = np.take_along_axis(law.g0(w1, w2), rule[None, :], axis=0)[0]
        curve = (a == rule) * (y - q) / g + q
        assert curve.mean() == pytest.approx(value, abs=4 * curve.std() / np.sqrt(curve.size))
        assert np.var(curve) == pytest.approx(variance, rel=0.02)

    def test_the_optimal_rule_is_the_declared_learners_limit_rule(self) -> None:
        optimal = law.optimal_rule()
        limit = law.limit_rule("declared")
        for w2 in law.W2_LEVELS:
            np.testing.assert_array_equal(optimal[w2], limit[w2])
        missing = law.limit_rule("missing_quadratic")
        assert any(not np.array_equal(optimal[w2], missing[w2]) for w2 in law.W2_LEVELS)

    def test_a_rule_that_differs_on_one_validation_row_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = _measured().result
        honest = realized_arms(result.nuisance.regimes)

        def shifted(regimes: Any) -> np.ndarray:
            rule = honest.copy()
            rule[11] = (rule[11] + 1) % 3
            return rule

        monkeypatch.setattr(law, "realized_arms", shifted)
        with pytest.raises(law.HarnessMismatch, match="validation rows"):
            law.harness(result, law.outcome_learner())

    def test_a_stored_fold_rule_that_differs_off_its_fold_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation: a stored rule of fold 0 changed on a row outside fold 0."""
        result = _measured().result
        honest = fold_rules(result.nuisance)
        assert honest is not None
        _, test = next(iter(result.nuisance.folds))
        row = int(np.setdiff1d(np.arange(N), np.asarray(test))[0])

        def edited(nuisance: Any) -> np.ndarray:
            rules = honest.copy()
            rules[row, 0] = (rules[row, 0] + 1) % 3
            return rules

        monkeypatch.setattr(law, "fold_rules", edited)
        with pytest.raises(law.HarnessMismatch, match="stored fold rule"):
            law.harness(result, law.outcome_learner())

    def test_the_harness_refits_the_learner_it_is_given(self) -> None:
        """Mutation: the declared fit's rules refit with the learner without the quadratic."""
        with pytest.raises(law.HarnessMismatch):
            law.harness(_measured().result, law.outcome_learner("missing_quadratic"))

    def test_a_tie_goes_to_the_lowest_arm_code(self) -> None:
        class Flat:
            classes_ = np.array([0, 1])

            def predict_proba(self, x: np.ndarray) -> np.ndarray:
                p = np.where(x[:, 0] == 1.0, 0.4, 0.6)  # arm 1 below; arms 0 and 2 tie
                return np.column_stack([1 - p, p])

        rule = law.argmax_rule(Flat(), np.zeros((4, 2)))
        np.testing.assert_array_equal(rule, 0)


# --------------------------------------------------------------------------- the controls


class TestTheControls:
    def test_the_untargeted_value_is_the_fold_average_of_the_plug_ins(self) -> None:
        result = _measured().result
        nuisance = result.nuisance
        arms = [
            np.asarray(nuisance.scaler.unscale_levels(nuisance.outcome.arms[level]))
            for level in nuisance.outcome.levels
        ]
        rule = realized_arms(nuisance.regimes)
        plug_in = np.array([arms[int(arm)][i] for i, arm in enumerate(rule)])
        longhand = np.mean([plug_in[np.asarray(test)].mean() for _, test in nuisance.folds])
        assert law.untargeted_estimate(result) == pytest.approx(longhand, rel=1e-14)
        assert law.untargeted_estimate(result) != pytest.approx(
            float(result.estimates[law.ESTIMAND].psi), abs=1e-6
        )

    def test_the_margin_columns_are_computed_longhand(self) -> None:
        result = _measured("missing_quadratic").result
        nuisance = result.nuisance
        columns = law.margin_columns(result, "missing_quadratic")
        rule = realized_arms(nuisance.regimes)
        covariates = np.asarray(result.data.covariates, dtype=float)
        limit = np.array(
            [
                int(np.argmax(law.limit_predictions("missing_quadratic", np.array([w1]), w2)))
                for w1, w2 in covariates
            ]
        )
        assert columns["limit_disagreement"] == pytest.approx(np.mean(rule != limit))
        predictions = np.sort(
            np.vstack([nuisance.outcome.arms[level] for level in nuisance.outcome.levels]), axis=0
        )
        gap = predictions[-1] - predictions[-2]
        assert columns["near_tie_share"] == pytest.approx(np.mean(gap < law.NEAR_TIE))
        assert columns["min_arm_share"] == pytest.approx(min(np.bincount(rule, minlength=3)) / N)
        # The record's per-fold shares agree with the realized rule.
        (split,) = result.extra["learned_rule"].splits
        for shares, (_, test) in zip(split.arm_shares, nuisance.folds, strict=True):
            np.testing.assert_allclose(
                shares, np.bincount(rule[np.asarray(test)], minlength=3) / len(test)
            )

    def test_the_drop_w1_limit_rule_is_arm_1_everywhere(self) -> None:
        for w2, rule in law.limit_rule("drop_w1").items():
            assert set(np.unique(rule)) == {1}, w2


# ------------------------------------------------------------------------ the primary rows


class TestThePrimaryRows:
    def test_one_row_with_its_own_truth(self) -> None:
        record = base.smoke_record(study.STUDY)
        key = (study.PRIMARY_LAW, 0, N)
        frame = study._sample(record, key)
        rows = pd.DataFrame(study._primary(record, key, frame))
        assert list(rows["estimand"]) == [law.ESTIMAND]
        covered = (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
        np.testing.assert_array_equal(rows["covered"], covered.astype(int))
        assert (rows["rule_rows_checked"] == N).all()
        assert set(rows.columns) >= set(law.HARNESS_COLUMNS)

    def test_the_harness_artifact_keys_each_replication(self) -> None:
        rows = pd.DataFrame(
            {
                "scenario": [study.PRIMARY_LAW] * 2,
                "replicate": [1, 0],
                "truth": [0.7, 0.71],
                "oracle_se": [0.01, 0.01],
                "rule_rows_checked": [N, N],
                "solver_warnings": [0, 0],
            }
        )
        artifact = study.extra_artifacts(rows)[study.HARNESS_ARTIFACT]
        assert list(artifact["replicate"]) == [0, 1]


# --------------------------------------------------------------------- the property path


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestThePropertyPath:
    """``generate_property_rows`` and ``summarize_properties`` end to end on a few draws."""

    def test_every_declared_cell_is_summarized(self) -> None:
        rows = _property_rows()
        declared = {cell for cells in study.PROPERTY_CELLS.values() for cell in cells}
        assert set(rows["cell"]) == declared
        assert (rows.groupby("cell").size() == CAP).all()
        assert (rows["n"] == properties.CALIBRATION_N).all()
        assert (rows["rule_rows_checked"] == rows["n"]).all()
        summary = properties.summarize_properties(rows)
        assert set(summary["cell"]) == declared

    def test_each_family_fits_its_own_learner_on_its_own_draws(self) -> None:
        rows = _property_rows()
        truths = {
            family: set(rows.loc[rows["property"] == family, "truth"])
            for family in ("interval_calibration", study.COVERAGE_FAMILY, study.NECESSITY_FAMILY)
        }
        assert not truths["interval_calibration"] & truths[study.COVERAGE_FAMILY]
        assert not truths["interval_calibration"] & truths[study.NECESSITY_FAMILY]

    def test_the_controls_and_diagnostics_rescale_the_positive_of_the_same_fit(self) -> None:
        rows = _property_rows()
        for label in study.MARGIN_LABELS:
            positive = rows.loc[rows["cell"] == f"{label}__declared"].set_index("replicate")
            margin = rows.loc[rows["cell"] == f"{label}__{study.MARGIN}"].set_index("replicate")
            assert set(margin["role"]) == {"diagnostic"}
            for column in ("estimate", "truth", "std_error"):
                np.testing.assert_array_equal(margin[column], positive[column])
            assert margin[list(properties.MARGIN_COLUMNS)].notna().all().all()
            assert positive[list(properties.MARGIN_COLUMNS)].isna().all().all()
        positive = rows.loc[rows["cell"] == f"{study.COVERAGE_LABEL}__declared"]
        control = rows.loc[rows["cell"] == f"{study.COVERAGE_LABEL}__{study.COVERAGE_CONTROL}"]
        positive, control = positive.set_index("replicate"), control.set_index("replicate")
        np.testing.assert_allclose(
            control["std_error"], study.COVERAGE_SHRINK * positive["std_error"], rtol=1e-15
        )
        np.testing.assert_array_equal(control["estimate"], positive["estimate"])

    def test_the_necessity_control_is_the_untargeted_fit_of_the_same_draw(self) -> None:
        rows = _property_rows()
        rows = rows.loc[rows["property"] == study.NECESSITY_FAMILY]
        targeted, untargeted = (
            rows.loc[rows["cell"] == cell].set_index("replicate") for cell in study.NECESSITY_CELLS
        )
        assert (set(targeted["role"]), set(untargeted["role"])) == ({"positive"}, {"control"})
        np.testing.assert_array_equal(untargeted["truth"], targeted["truth"])
        np.testing.assert_array_equal(untargeted["std_error"], targeted["std_error"])
        assert (untargeted["estimate"] != targeted["estimate"]).all()

    def test_the_coverage_verdict_reads_the_declared_floor(self) -> None:
        """Mutations on a hand-made summary: each cell passes on one side of the floor and fails
        on the other, and a positive without its bias leg fails."""
        floor = study.STUDY.margins.calibration_coverage[0]
        summary = pd.DataFrame(
            {
                "property": [study.COVERAGE_FAMILY] * 4,
                "cell": [f"c{i}" for i in range(4)],
                "role": ["positive", "positive", "control", "control"],
                "bias_equivalent": [True, True, False, False],
                "coverage_ci_lower": [floor, floor - 1e-9, 0.0, 0.0],
                "coverage_ci_upper": [1.0, 1.0, floor - 1e-9, floor],
                "passed": [False] * 4,
            }
        )
        properties.coverage_verdicts(summary, floor=floor)
        assert list(summary["passed"]) == [True, False, True, False]
        summary.loc[0, "bias_equivalent"] = False
        properties.coverage_verdicts(summary, floor=floor)
        assert not summary.loc[0, "passed"]

    def test_the_margin_report_publishes_the_means(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows).set_index("cell")
        for label in study.MARGIN_LABELS:
            cell = f"{label}__{study.MARGIN}"
            group = rows.loc[rows["cell"] == cell]
            for column in properties.MARGIN_COLUMNS:
                assert summary.loc[cell, f"mean_{column}"] == pytest.approx(group[column].mean())
            assert summary.loc[cell, "lost_arm_share"] == pytest.approx(
                (group["min_arm_share"] == 0).mean()
            )
            assert summary.loc[cell, "passed"] and summary.loc[cell, "property_passed"]
            assert summary.loc[cell, "role"] == "diagnostic"

    def test_the_declared_rules_reproduce_the_studys_verdicts(self) -> None:
        rows = _property_rows()
        summary = properties.summarize_properties(rows)
        own = summary.set_index("cell")
        ctx = property_failures.module_context(study.STUDY, rows, summary, {}, properties)
        for family in ("interval_calibration", study.COVERAGE_FAMILY, study.NECESSITY_FAMILY):
            rule = property_failures.rule_for(family, properties)
            assert rule is not None, family
            verdicts = property_failures.rule_verdicts(rule, summary, ctx).set_index("cell")
            assert set(verdicts.index) == set(own.index[own["property"] == family])
            for cell in verdicts.index:
                assert bool(verdicts.loc[cell, "passed"]) == bool(own.loc[cell, "passed"]), cell
        assert property_failures.rule_for(study.MARGIN_FAMILY, properties) is None

    @pytest.mark.parametrize(
        ("lost", "cost"),
        [
            (("interval_calibration", "three_arms", 1), 4),
            (("conservative_coverage", "missing_quadratic", 1), 3),
            (("targeting_necessity", "three_arms_drop_w1", 1), 2),
        ],
    )
    def test_a_lost_fit_costs_every_cell_it_writes(
        self, lost: tuple[str, str, int], cost: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lost calibration draw costs its positive, its two derived controls and its margin
        row; a coverage draw its positive, its control and its margin row."""
        cached = _property_rows()
        derived = {c.split("/", 1)[1] for cs in properties.DERIVED_CELLS.values() for c in cs}

        def worker(key: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
            family, label, replicate = key[0][:3]
            if (family, label, replicate) == lost:
                raise DataError("a draw with one row in an arm")
            cells = {
                cell.split("/", 1)[1]
                for cell in properties._cells_of(key)
                if cell.split("/", 1)[1] not in derived
            }
            mine = cached.loc[(cached["replicate"] == replicate) & cached["cell"].isin(cells)]
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

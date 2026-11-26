"""The step-down max-T adjustment of ``variable_importance`` and its support label (X125).

The step-down is checked three ways. An identity test compares it with a brute-force loop on a
fixed draw matrix. Exact-law and limit tests compare it with closed forms: the bivariate normal
probability, the Sidak step-down at the identity correlation, and the smallest p-value at
perfect correlation. Deliberate mutations (no running maximum, no floor, a single step, the
identity correlation) each fail a check the shipped procedure passes, on a hand-built witness
where every component changes the answer.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression

from cleverly import variable_importance, variable_importance_plan
from cleverly._summary import vocabulary
from cleverly._support import Support
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.inference import make_estimate
from cleverly.inference.cluster import influence_covariance
from cleverly.variable_importance import (
    BH_REASON,
    CORRELATION_TOLERANCE,
    MEASURED_N,
    MEASURED_ROWS,
    PENDING_REASON,
    AdjustmentKey,
    VariableImportanceResult,
    _bh_adjust,
    _max_t_statistics,
    _null_draws,
    _row_correlation,
    _step_down,
    variable_importance_support,
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners

# ------------------------------------------------------------------ the procedure


def _brute_force(statistics: np.ndarray, pvalues: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """An explicit loop over the steps: the reference the vectorised step-down must match."""
    m = statistics.size
    order = sorted(range(m), key=lambda j: -statistics[j])
    adjusted = np.zeros(m)
    running = 0.0
    for k, row in enumerate(order):
        remaining = order[k:]
        hits = 0
        for b in range(draws.shape[0]):
            if max(draws[b, j] for j in remaining) >= statistics[row]:
                hits += 1
        running = max(running, hits / draws.shape[0])
        adjusted[row] = min(1.0, max(running, pvalues[row]))
    return adjusted


#: A witness on which every component of the step-down changes the answer. Rows are passed
#: out of rank order (statistics 1, 3 and 2). With these draws the top row's raw share is 0, so
#: the floor at its own p-value acts; the third-ranked row's raw share (0.25) is below the
#: second's (0.5), so the running maximum acts; and the second-ranked row's share over all
#: rows (0.75) exceeds its share over the rows ranked 2 and 3 (0.5), so the step-down acts.
WITNESS_T = np.array([1.0, 3.0, 2.0])
WITNESS_P = 2.0 * stats.norm.sf(WITNESS_T)
WITNESS_DRAWS = np.array(
    [
        [0.2, 2.5, 0.5],
        [1.5, 0.1, 2.2],
        [0.4, 0.3, 2.1],
        [0.1, 0.2, 0.3],
    ]
)
WITNESS_ADJUSTED = np.array([0.5, WITNESS_P[1], 0.5])


def _no_running_maximum(t: np.ndarray, p: np.ndarray, draws: np.ndarray) -> np.ndarray:
    order = np.argsort(-t, kind="stable")
    tail = np.maximum.accumulate(draws[:, order][:, ::-1], axis=1)[:, ::-1]
    adjusted = np.empty(t.size)
    adjusted[order] = np.mean(tail >= t[order], axis=0)
    return np.minimum(np.maximum(adjusted, p), 1.0)


def _no_floor(t: np.ndarray, p: np.ndarray, draws: np.ndarray) -> np.ndarray:
    return _step_down(t, np.zeros_like(p), draws)


def _single_step(t: np.ndarray, p: np.ndarray, draws: np.ndarray) -> np.ndarray:
    exceed = np.mean(draws.max(axis=1)[:, None] >= t[None, :], axis=0)
    return np.minimum(np.maximum(exceed, p), 1.0)


class TestTheStepDown:
    def test_the_witness_reads_its_hand_computed_values(self) -> None:
        np.testing.assert_allclose(
            _step_down(WITNESS_T, WITNESS_P, WITNESS_DRAWS), WITNESS_ADJUSTED, rtol=1e-12
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_equals_the_brute_force_loop(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        m = 7
        raw = rng.normal(size=(m, m))
        covariance = raw @ raw.T + np.eye(m)
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        pvalues = rng.uniform(1e-6, 0.4, size=m)
        draws = _null_draws(correlation, 2_000, seed)
        statistics = _max_t_statistics(pvalues)
        np.testing.assert_allclose(
            _step_down(statistics, pvalues, draws),
            _brute_force(statistics, pvalues, draws),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        "mutant", [_no_running_maximum, _no_floor, _single_step], ids=lambda f: f.__name__
    )
    def test_each_mutation_fails_the_witness(self, mutant: Any) -> None:
        mutated = mutant(WITNESS_T, WITNESS_P, WITNESS_DRAWS)
        assert not np.allclose(mutated, WITNESS_ADJUSTED, rtol=1e-6), mutant.__name__
        assert not np.allclose(
            mutated, _brute_force(WITNESS_T, WITNESS_P, WITNESS_DRAWS), rtol=1e-6
        )

    def test_the_draws_are_deterministic_for_a_seed(self) -> None:
        correlation = np.array([[1.0, 0.4], [0.4, 1.0]])
        np.testing.assert_array_equal(
            _null_draws(correlation, 1_000, 7), _null_draws(correlation, 1_000, 7)
        )
        assert not np.array_equal(
            _null_draws(correlation, 1_000, 7), _null_draws(correlation, 1_000, 8)
        )


class TestExactLawsAndLimits:
    """Closed forms the Monte Carlo step-down must reach within its error."""

    DRAWS = 400_000

    def test_two_rows_match_the_bivariate_normal_probability(self) -> None:
        rho = -0.6
        correlation = np.array([[1.0, rho], [rho, 1.0]])
        pvalues = np.array([0.01, 0.2])
        t = _max_t_statistics(pvalues)
        adjusted = _step_down(t, pvalues, _null_draws(correlation, self.DRAWS, 3))
        law = stats.multivariate_normal(mean=[0.0, 0.0], cov=correlation)
        inside = law.cdf([t[0], t[0]], lower_limit=[-t[0], -t[0]])
        first = 1.0 - inside
        error = 4.0 * np.sqrt(first * (1 - first) / self.DRAWS)
        assert adjusted[0] == pytest.approx(first, abs=error)
        # The last step is the marginal test, so it reads the row's own p-value.
        assert adjusted[1] == pytest.approx(max(first, pvalues[1]), abs=4e-3)
        # A nonzero witness: the correlation lowers the first step below Sidak's.
        sidak = 1.0 - (1.0 - pvalues[0]) ** 2
        assert first < sidak - 1e-3

    def test_the_identity_correlation_is_the_sidak_step_down(self) -> None:
        pvalues = np.array([0.004, 0.02, 0.011, 0.3, 0.05])
        m = pvalues.size
        adjusted = _step_down(
            _max_t_statistics(pvalues), pvalues, _null_draws(np.eye(m), self.DRAWS, 4)
        )
        order = np.argsort(pvalues)
        sidak = np.empty(m)
        running = 0.0
        for k, row in enumerate(order):
            running = max(running, 1.0 - (1.0 - pvalues[row]) ** (m - k))
            sidak[row] = running
        np.testing.assert_allclose(adjusted, sidak, atol=4.0 * np.sqrt(0.25 / self.DRAWS))

    def test_perfect_correlation_reads_the_rows_own_p_values(self) -> None:
        m = 4
        correlation = np.full((m, m), 1.0 - 1e-6) + 1e-6 * np.eye(m)
        pvalues = np.array([0.003, 0.04, 0.2, 0.6])
        adjusted = _step_down(
            _max_t_statistics(pvalues), pvalues, _null_draws(correlation, self.DRAWS, 5)
        )
        np.testing.assert_allclose(adjusted, pvalues, atol=4.0 * np.sqrt(0.25 / self.DRAWS))


class TestTheStatistic:
    def test_a_tiny_p_value_keeps_a_finite_ordered_statistic(self) -> None:
        pvalues = np.array([1e-17, 1e-30, 1e-300, 0.5])
        t = _max_t_statistics(pvalues)
        assert np.all(np.isfinite(t))
        assert t[2] > t[1] > t[0] > t[3]
        # The witness that the edge exists: the lower-tail form rounds them to one infinity.
        assert np.all(np.isinf(stats.norm.ppf(1.0 - pvalues[:3] / 2.0)))

    def test_tiny_p_values_keep_their_rank_and_floor(self) -> None:
        pvalues = np.array([1e-17, 1e-30, 0.2])
        correlation = np.eye(3)
        adjusted = _step_down(
            _max_t_statistics(pvalues), pvalues, _null_draws(correlation, 2_000, 0)
        )
        np.testing.assert_array_equal(adjusted[:2], pvalues[:2])

    def test_a_p_value_outside_the_unit_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite values in"):
            _max_t_statistics([0.2, 1.5])


class TestAgainstBonferroniAndBH:
    """Nonzero witnesses: max-T differs from Bonferroni and BH where theory says it does."""

    def test_correlated_rows_beat_bonferroni_and_holm(self) -> None:
        m, rho = 6, 0.9
        correlation = np.full((m, m), rho) + (1 - rho) * np.eye(m)
        pvalues = np.array([0.012, 0.015, 0.02, 0.03, 0.04, 0.5])
        adjusted = _step_down(
            _max_t_statistics(pvalues), pvalues, _null_draws(correlation, 100_000, 6)
        )
        bonferroni = np.minimum(m * pvalues, 1.0)
        assert adjusted[0] < 0.5 * bonferroni[0]
        assert adjusted[0] < 0.05 < bonferroni[0]

    def test_many_small_p_values_favour_bh(self) -> None:
        pvalues = np.array([0.01, 0.012, 0.015, 0.02, 0.03])
        m = pvalues.size
        adjusted = _step_down(
            _max_t_statistics(pvalues), pvalues, _null_draws(np.eye(m), 100_000, 7)
        )
        bh = _bh_adjust(pvalues)
        assert np.all(bh <= 0.03 + 1e-12)
        assert adjusted.min() > 0.045
        assert np.all(adjusted > bh)


# ------------------------------------------------------------------ the run


def _frame(n: int = 600, seed: int = 0, clusters: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    x1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * w)))
    x2 = rng.binomial(1, 0.5, n)
    z = rng.integers(0, 3, n)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x1 + 0.3 * z + 0.6 * w))))
    frame = pd.DataFrame({"Y": y, "W": w, "X1": x1, "X2": x2, "Z": z})
    if clusters is not None:
        frame["cluster"] = np.arange(n) % clusters
    return frame


def _correlated_frame() -> pd.DataFrame:
    """``X3`` copies ``X1`` except on about 8% of rows, and ``X1`` has a moderate effect."""
    rng = np.random.default_rng(1)
    n = 600
    w = rng.normal(size=n)
    x1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * w)))
    x2 = rng.binomial(1, 0.5, n)
    x3 = np.where(rng.uniform(size=n) < 0.08, 1 - x1, x1)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.3 + 0.5 * x1 + 0.6 * w))))
    return pd.DataFrame({"Y": y, "W": w, "X1": x1, "X2": x2, "X3": x3})


def _estimator(**overrides: Any) -> TMLE:
    settings: dict[str, Any] = {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return TMLE(**settings)


def _run(frame: Any = None, **options: Any) -> VariableImportanceResult:
    settings: dict[str, Any] = {
        "outcome": "Y",
        "candidates": ("X1", "X2", "Z"),
        "covariates": ("W",),
        "estimator": _estimator(),
    }
    settings.update(options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return variable_importance(_frame() if frame is None else frame, **settings)


@pytest.fixture(scope="module")
def ranked() -> VariableImportanceResult:
    return _run()


class TestTheRun:
    def test_it_records_the_adjustment(self, ranked: VariableImportanceResult) -> None:
        assert (ranked.method, ranked.error_rate) == ("step-down max-T", "fwer")
        assert (ranked.multiplier_draws, ranked.multiplier_seed) == (10_000, 0)
        m = len(ranked)
        assert ranked.correlation is not None and ranked.correlation.shape == (m, m)
        assert not ranked.correlation.flags.writeable

    def test_the_adjusted_p_values_follow_the_procedure(
        self, ranked: VariableImportanceResult
    ) -> None:
        # The draws are made in the order the rows were fitted: candidate by candidate, each
        # candidate's parameters in its fit's order.
        fitted = [
            (candidate, name)
            for candidate in ranked.facts.candidates
            for name in ranked.fits[candidate].estimates
            if name.startswith("ate")
        ]
        by_row = {(entry.candidate, entry.parameter): entry for entry in ranked}
        estimates = [by_row[row].estimate for row in fitted]
        pvalues = np.array([estimate.pvalue for estimate in estimates])
        curves = np.column_stack([estimate.influence_curve for estimate in estimates])
        covariance = influence_covariance(curves)
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        entry_order = [fitted.index((e.candidate, e.parameter)) for e in ranked]
        np.testing.assert_allclose(
            ranked.correlation,
            correlation[np.ix_(entry_order, entry_order)],
            rtol=1e-10,
            atol=1e-12,
        )
        expected = _brute_force(
            _max_t_statistics(pvalues), pvalues, _null_draws(correlation, 10_000, 0)
        )
        np.testing.assert_allclose(
            [by_row[row].adjusted_pvalue for row in fitted], expected, rtol=1e-12
        )
        assert [e.adjusted_pvalue for e in ranked] == sorted(e.adjusted_pvalue for e in ranked)

    def test_the_three_level_rows_are_correlated(self, ranked: VariableImportanceResult) -> None:
        """A nonzero witness that the correlation enters: the two Z contrasts share an arm."""
        names = [(entry.candidate, entry.parameter) for entry in ranked]
        z = [index for index, (candidate, _) in enumerate(names) if candidate == "Z"]
        assert ranked.correlation is not None
        assert ranked.correlation[z[0], z[1]] > 0.2

    def test_the_identity_correlation_mutation_changes_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two candidates that nearly coincide (statistic correlation about 0.84) and are not
        adjusted for each other: the shipped first step sits well below Sidak's, and the
        identity-correlation mutation reproduces Sidak's step-down."""
        frame = _correlated_frame()
        options: dict[str, Any] = {
            "candidates": ("X1", "X3", "X2"),
            "adjust_for_other_candidates": False,
        }
        shipped = _run(frame, **options)
        assert shipped.correlation is not None and shipped.correlation[0, 1] > 0.75
        module = importlib.import_module("cleverly.variable_importance")
        monkeypatch.setattr(
            module, "_row_correlation", lambda estimates, labels, cluster: np.eye(len(estimates))
        )
        mutated = _run(frame, **options)
        pvalues = np.array([entry.estimate.pvalue for entry in mutated])
        m = pvalues.size
        sidak = [max(1.0 - (1.0 - pvalues[j]) ** (m - j) for j in range(k + 1)) for k in range(m)]
        np.testing.assert_allclose([e.adjusted_pvalue for e in mutated], sidak, atol=0.01)
        assert mutated[0].adjusted_pvalue - shipped[0].adjusted_pvalue > 0.02

    def test_keep_fits_false_gives_the_same_adjustment(
        self, ranked: VariableImportanceResult
    ) -> None:
        dropped = _run(keep_fits=False)
        assert [e.adjusted_pvalue for e in dropped] == [e.adjusted_pvalue for e in ranked]
        np.testing.assert_array_equal(dropped.correlation, ranked.correlation)

    def test_the_seed_and_draws_are_read(self, ranked: VariableImportanceResult) -> None:
        other = _run(multiplier_seed=11, multiplier_draws=20_000)
        assert (other.multiplier_draws, other.multiplier_seed) == (20_000, 11)
        assert [e.adjusted_pvalue for e in other] != [e.adjusted_pvalue for e in ranked]

    def test_bh_records_no_correlation(self) -> None:
        bh = _run(adjustment="bh")
        assert (bh.method, bh.error_rate) == ("Benjamini-Hochberg", "fdr")
        assert (bh.correlation, bh.multiplier_draws, bh.multiplier_seed) == (None, None, None)
        pvalues = [entry.estimate.pvalue for entry in bh]
        np.testing.assert_allclose([e.adjusted_pvalue for e in bh], _bh_adjust(pvalues))

    def test_fixed_weight_curves_carry_their_weights(self) -> None:
        """The joint covariance reads each weighted curve unweighted, so each curve must
        already carry its weights: its centred variance is the estimate's variance."""
        frame = _frame()
        frame["wt"] = np.random.default_rng(3).uniform(0.5, 2.0, len(frame))
        result = _run(frame, weights="wt")
        for entry in result:
            curve = entry.estimate.influence_curve[:, None]
            assert influence_covariance(curve)[0, 0] == pytest.approx(
                entry.estimate.variance, rel=1e-2
            )
        assert result.adjustment_support.label == "experimental"
        assert "weights=" in result.adjustment_support.reason


class TestTheSummary:
    def test_max_t_names_the_family_wise_error_rate(self, ranked: VariableImportanceResult) -> None:
        summary = ranked.summary()
        assert dict(summary.facts)[vocabulary.ADJUSTMENT] == (
            "step-down max-T, 4 tests (family-wise error rate)"
        )
        text = summary.note("multiple_testing").text
        assert text.startswith(vocabulary.NOTE_TEXTS["multiple_testing"].format(count="4"))
        assert ranked.adjustment_support.reason in text

    def test_bh_names_its_experimental_claim(self) -> None:
        summary = _run(adjustment="bh").summary()
        assert dict(summary.facts)[vocabulary.ADJUSTMENT] == (
            "Benjamini-Hochberg, 4 tests (false discovery rate)"
        )
        text = summary.note("multiple_testing_bh").text
        assert "experimental" in text and "family-wise" in text


class TestTheResultChecksItsRecord:
    def test_a_bh_result_with_a_correlation_is_refused(
        self, ranked: VariableImportanceResult
    ) -> None:
        with pytest.raises(ValueError, match="records no correlation"):
            replace(ranked, method="Benjamini-Hochberg", error_rate="fdr")

    def test_a_max_t_result_without_its_draws_is_refused(
        self, ranked: VariableImportanceResult
    ) -> None:
        with pytest.raises(ValueError, match="draws and seed"):
            replace(ranked, multiplier_draws=None)
        with pytest.raises(ValueError, match="must be"):
            replace(ranked, error_rate="fdr")


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("call", [variable_importance, variable_importance_plan])
class TestTheRefusals:
    """Each refusal the configuration decides is raised by the run and the plan alike."""

    def test_an_unknown_adjustment(self, call: Any) -> None:
        with pytest.raises(ValueError, match="adjustment must be one of"):
            call(_frame(), outcome="Y", candidates=["X1"], covariates=["W"], adjustment="holm")

    def test_too_few_draws(self, call: Any) -> None:
        with pytest.raises(ValueError, match="at least 1,000"):
            call(_frame(), outcome="Y", candidates=["X1"], covariates=["W"], multiplier_draws=999)

    def test_a_non_integer_seed(self, call: Any) -> None:
        with pytest.raises(TypeError, match="multiplier_seed must be an integer"):
            call(_frame(), outcome="Y", candidates=["X1"], covariates=["W"], multiplier_seed=True)

    def test_estimated_weights(self, call: Any) -> None:
        frame = _frame()
        frame["wt"] = 1.0 + frame["W"] ** 2
        with pytest.raises(CapabilityError) as raised:
            call(
                frame,
                outcome="Y",
                candidates=["X1", "X2"],
                covariates=["W"],
                estimator=_estimator(**never_fit_learners()),
                weights="wt",
                weights_estimated=True,
            )
        assert raised.value.refusal.reason is RefusalReason.INFERENCE_GAP
        assert "weight estimation" in str(raised.value)
        assert "adjustment='bh'" in str(raised.value)
        assert NeverFit.calls == 0

    def test_fewer_than_forty_clusters(self, call: Any) -> None:
        with pytest.raises(CapabilityError, match="39 clusters, fewer than 40") as raised:
            call(
                _frame(clusters=39),
                outcome="Y",
                candidates=["X1", "X2"],
                covariates=["W"],
                estimator=_estimator(**never_fit_learners()),
                id="cluster",
            )
        assert raised.value.refusal.reason is RefusalReason.INFERENCE_GAP
        assert NeverFit.calls == 0

    def test_bh_takes_what_max_t_refuses(self, call: Any) -> None:
        """The control: the configuration refusals belong to the step-down max-T alone."""
        result = call(
            _frame(clusters=39),
            outcome="Y",
            candidates=["X1", "X2"],
            covariates=["W"],
            estimator=_estimator(),
            id="cluster",
            adjustment="bh",
        )
        assert result is not None


class TestTheRefusalsAfterTheFits:
    def test_forty_clusters_run(self) -> None:
        result = _run(_frame(clusters=40), id="cluster")
        assert result.adjustment_support.label == "experimental"
        assert "clustered data" in result.adjustment_support.reason

    def test_a_t_reference_row_is_refused(self) -> None:
        """A clustered fold-evaluated fit reads a t reference at every cluster count."""
        with pytest.raises(CapabilityError, match="Student t reference") as raised:
            _run(
                _frame(clusters=60),
                id="cluster",
                estimator=_estimator(cross_fit=True, cv_evaluation=True, n_folds=3),
            )
        assert raised.value.refusal.reason is RefusalReason.INFERENCE_GAP

    def test_two_identical_candidates_are_one_test(self) -> None:
        frame = _frame()
        frame["X1_copy"] = frame["X1"]
        with pytest.raises(CapabilityError, match="linearly dependent") as raised:
            _run(frame, candidates=("X1", "X1_copy", "X2"), adjust_for_other_candidates=False)
        message = str(raised.value)
        assert "X1: ate" in message and "X1_copy: ate" in message and "X2: ate" not in message

    def test_a_constant_curve_is_refused(self) -> None:
        curve = np.random.default_rng(0).normal(size=50)
        moving = make_estimate("ate", 0.1, curve, n=50, rule_widened=False)
        flat = make_estimate("ate", 0.0, np.zeros(50), n=50, rule_widened=False)
        with pytest.raises(CapabilityError, match=r"\['B'\] do not"):
            _row_correlation([moving, flat], ["A", "B"], None)

    def test_a_near_singular_correlation_is_refused_below_the_tolerance(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=400)
        close = make_estimate("ate", 0.1, base, n=400, rule_widened=False)
        tied = make_estimate(
            "ate", 0.1, base + 1e-6 * rng.normal(size=400), n=400, rule_widened=False
        )
        apart = make_estimate(
            "ate", 0.1, base + 0.5 * rng.normal(size=400), n=400, rule_widened=False
        )
        with pytest.raises(CapabilityError, match="linearly dependent"):
            _row_correlation([close, tied], ["A", "B"], None)
        correlation = _row_correlation([close, apart], ["A", "C"], None)
        assert np.linalg.eigvalsh(correlation)[0] > CORRELATION_TOLERANCE


# ------------------------------------------------------------------ the support label

MEASURED_KEY = AdjustmentKey(
    engine="TMLE",
    estimand="ate",
    adjust_for_other_candidates=True,
    clustered=False,
    weighted=False,
    missing_outcome=False,
    n=MEASURED_N,
    cross_fitted=False,
    n_folds=10,
    logistic_learners=True,
    binary_outcome=True,
)
BINARY_ROW = Support("supported", "Covered.", ("canonical-tmle",), row="T01.binary")
THREE_LEVEL_ROW = Support("supported", "Covered.", ("canonical-multi-arm-tmle",), row="T08")
CROSS_FITTED_ROW = Support("supported", "Covered.", ("default-bands",), row="T02.binary")


class TestTheSupportLabel:
    @pytest.fixture
    def landed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The study's run landed: the pending reason is cleared."""
        module = importlib.import_module("cleverly.variable_importance")

        monkeypatch.setattr(module, "PENDING_REASON", None)

    def test_the_covered_key_reads_experimental_until_the_run(self) -> None:
        support = variable_importance_support(
            adjustment="max_t", key=MEASURED_KEY, entries=[BINARY_ROW, THREE_LEVEL_ROW]
        )
        assert PENDING_REASON is not None
        assert (support.label, support.reason) == ("experimental", PENDING_REASON)

    @pytest.mark.usefixtures("landed")
    def test_the_covered_key_reads_supported_once_the_run_lands(self) -> None:
        in_sample = variable_importance_support(
            adjustment="max_t", key=MEASURED_KEY, entries=[BINARY_ROW, THREE_LEVEL_ROW]
        )
        assert (in_sample.label, in_sample.evidence) == ("supported", ("variable-importance",))
        cross = replace(MEASURED_KEY, cross_fitted=True, n_folds=5)
        assert (
            variable_importance_support(
                adjustment="max_t", key=cross, entries=[CROSS_FITTED_ROW] * 12
            ).label
            == "supported"
        )

    @pytest.mark.usefixtures("landed")
    @pytest.mark.parametrize(
        ("change", "phrase"),
        [
            ({"engine": "DRTMLE"}, "TMLE only"),
            ({"estimand": "rr"}, "ATE only"),
            ({"binary_outcome": False}, "binary outcome"),
            ({"adjust_for_other_candidates": False}, "adjusts each candidate for the others"),
            ({"clustered": True}, "clustered data"),
            ({"weighted": True}, "weights="),
            ({"missing_outcome": True}, "missing outcomes"),
            ({"logistic_learners": False}, "LogisticRegression"),
            ({"cross_fitted": True, "n_folds": 10}, "5 folds"),
            ({"n": MEASURED_N - 1}, f"n = {MEASURED_N:,}"),
        ],
    )
    def test_each_departure_reads_experimental_with_its_reason(
        self, change: dict[str, Any], phrase: str
    ) -> None:
        entries = [CROSS_FITTED_ROW if change.get("cross_fitted") else BINARY_ROW]
        support = variable_importance_support(
            adjustment="max_t", key=replace(MEASURED_KEY, **change), entries=entries
        )
        assert support.label == "experimental" and phrase in support.reason
        assert PENDING_REASON is None or PENDING_REASON not in support.reason

    @pytest.mark.usefixtures("landed")
    def test_the_row_count_and_the_entries_bound_the_key(self) -> None:
        many = variable_importance_support(
            adjustment="max_t", key=MEASURED_KEY, entries=[BINARY_ROW] * (MEASURED_ROWS + 1)
        )
        assert many.label == "experimental" and f"at most {MEASURED_ROWS}" in many.reason
        weak = variable_importance_support(
            adjustment="max_t",
            key=MEASURED_KEY,
            entries=[BINARY_ROW, Support("experimental", "No row.")],
        )
        assert weak.label == "experimental" and "own estimate is experimental" in weak.reason
        cross = replace(MEASURED_KEY, cross_fitted=True, n_folds=5)
        three = variable_importance_support(
            adjustment="max_t", key=cross, entries=[CROSS_FITTED_ROW, THREE_LEVEL_ROW]
        )
        assert three.label == "experimental" and "cross-fitted binary" in three.reason

    @pytest.mark.usefixtures("landed")
    def test_bh_is_always_experimental(self) -> None:
        support = variable_importance_support(
            adjustment="bh", key=MEASURED_KEY, entries=[BINARY_ROW]
        )
        assert (support.label, support.reason) == ("experimental", BH_REASON)

    def test_a_run_reads_its_own_key(self, ranked: VariableImportanceResult) -> None:
        key = ranked.adjustment_key
        assert (key.engine, key.estimand, key.n, key.cross_fitted) == ("TMLE", "ate", 600, False)
        assert key.logistic_learners and key.binary_outcome
        assert {entry.estimate.support.row for entry in ranked} == {"T01.binary", "T08"}
        support = ranked.adjustment_support
        assert support.label == "experimental" and f"n = {MEASURED_N:,}" in support.reason

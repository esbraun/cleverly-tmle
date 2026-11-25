"""The delete-one-cluster jackknife variance (X54 part (c)).

Each witness is nonzero where the component it pins could vanish: the ``(J - 1) / J`` factor
against a linear statistic, the refits against independent fits, the nuisance-estimation term
against a stacked-estimating-equation sandwich that the efficient curve misses, the fold rule
against a split that ``J >= 2V`` would admit, and the withheld refit against a cluster whose
removal leaves an arm constant. Each has a deliberate-mutation control that fails when the
component is wrong.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataWarning, RefusalReason
from cleverly.inference import cluster_jackknife
from cleverly.inference.cluster_jackknife import (
    JACKKNIFE_FOLD_REFUSAL,
    jackknife_covariance,
    jackknife_fold_refusal,
    jackknife_learner,
)
from cleverly.inference.multiplier import JACKKNIFE_BAND_REFUSAL, simultaneous_bands
from cleverly.interventions import Incremental
from cleverly.learners import GLM, SuperLearner
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import IPSI, LTMLE, SDR
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

pytestmark = pytest.mark.xdist_group("cluster_jackknife")

tmle_module = importlib.import_module("cleverly.estimators.tmle")
estimator_module = importlib.import_module("cleverly.longitudinal.estimator")

#: Rows per cluster of the point frames.
SIZE = 30
REGIMENS = {"never": 0, "always": 1}


@pytest.fixture(autouse=True)
def _point_machinery(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lift the measured point-TMLE refusals so the witnesses reach the jackknife machinery.

    ``few-cluster-floor-tmle`` refuses the option for in-sample ``TMLE`` and defers it for
    stacked ``TMLE`` (:data:`~cleverly.inference.cluster_jackknife.JACKKNIFE_FAMILY_REFUSALS`).
    The construction itself is shared with ``LTMLE`` and stays pinned here. A test class with
    ``shipped_refusals = True`` reads the shipped map.
    """
    if not getattr(request.cls, "shipped_refusals", False):
        monkeypatch.setattr(cluster_jackknife, "JACKKNIFE_FAMILY_REFUSALS", {})


class RaisingLearner(BaseEstimator, ClassifierMixin):
    """A learner the R5 rule admits, whose ``fit`` raises: a refusal that runs it fails."""

    cleverly_learner_family: ClassVar[str] = "parametric"

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RaisingLearner:
        raise AssertionError("a learner ran before the refusal")

    def predict_proba(self, X: Any) -> np.ndarray:  # pragma: no cover - never fitted
        raise AssertionError("a learner ran before the refusal")


def glm_learners(**overrides: Any) -> dict[str, Any]:
    """Unpenalized logistic nuisances, fitted on every row, with no band."""
    return {
        "outcome_learner": LogisticRegression(C=1e10, max_iter=2000),
        "treatment_learner": LogisticRegression(C=1e10, max_iter=2000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "estimands": ("ate", "ey1", "ey0"),
        **overrides,
    }


def point_frame(clusters: int, *, seed: int = 3) -> pd.DataFrame:
    """``clusters`` clusters of 30 rows with a cluster effect on a binary outcome."""
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(clusters), SIZE)
    effect = rng.normal(scale=0.5, size=clusters)[cluster]
    w = rng.normal(size=cluster.size)
    a = rng.binomial(1, expit(0.6 * w))
    y = rng.binomial(1, expit(-0.2 + 0.8 * a + 0.7 * w + effect))
    return pd.DataFrame({"y": y, "a": a, "w": w, "g": cluster})


def fit_point(frame: pd.DataFrame, **settings: Any) -> Any:
    """A clustered binary TMLE fit, with warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(**settings).fit(frame, outcome="y", treatment="a", id="g").single()


def longitudinal_learners(**overrides: Any) -> dict[str, Any]:
    """GLM learners for every node the RM26 panel fits, with no band."""
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "random_state": 0,
        "simultaneous": False,
        **overrides,
    }


def panel(clusters: int) -> pd.DataFrame:
    """The RM26 panel, 600 rows over ``clusters`` clusters of consecutive rows."""
    frame = multivalue_panel(n=600, seed=43)
    return frame.assign(cluster=np.arange(600) * clusters // 600)


def fit_longitudinal(frame: pd.DataFrame, **settings: Any) -> Any:
    """An end-of-study LTMLE fit of the panel, with warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(REGIMENS, reference="never", **settings).fit(frame, **COLUMNS, id="cluster")


# --------------------------------------------------------------------------- the formula


class TestTheFormula:
    def test_a_hand_array_equals_the_formula(self) -> None:
        loo = np.array([[1.0, 2.0], [1.5, 1.0], [0.5, 3.5], [2.0, 2.5], [1.25, 0.25]])
        J = loo.shape[0]
        centred = loo - loo.mean(axis=0)
        by_hand = sum(np.outer(row, row) for row in centred) * (J - 1) / J
        np.testing.assert_allclose(jackknife_covariance(loo), by_hand, rtol=0, atol=1e-14)

    def test_a_mean_of_cluster_totals_gives_the_cluster_variance_over_j(self) -> None:
        # The classical identity for a linear statistic: a nonzero witness of (J - 1) / J.
        totals = np.array([0.3, 1.9, 2.4, 0.7, 5.1, 3.3, 0.2])
        J = totals.size
        loo = np.array([np.delete(totals, g).mean() for g in range(J)])
        assert jackknife_covariance(loo)[0, 0] == pytest.approx(totals.var(ddof=1) / J, rel=1e-13)

    def test_mutation_a_one_over_j_factor_fails(self) -> None:
        totals = np.array([0.3, 1.9, 2.4, 0.7, 5.1, 3.3, 0.2])
        J = totals.size
        loo = np.array([np.delete(totals, g).mean() for g in range(J)])
        centred = loo - loo.mean()
        mutated = float(centred @ centred) / J
        assert mutated != pytest.approx(totals.var(ddof=1) / J, rel=1e-3)


# --------------------------------------------------------------------------- the refits


def fixed_nuisance_loo(result: Any, keep: np.ndarray) -> float:
    """The ATE on ``keep`` re-targeted against the full fit's nuisances: no refit."""
    nuisance = result.repeats[0].nuisance
    q1, q0 = nuisance.outcome.arms[1.0][keep], nuisance.outcome.arms[0.0][keep]
    arms = list(nuisance.propensity.arms)
    g1 = nuisance.propensity.values[keep, arms.index(1.0)]
    a = result.data.treatment[keep]
    y = result.data.outcome[keep]
    h1, h0 = 1.0 / g1, 1.0 / (1.0 - g1)
    logit = np.log

    def lin(q: np.ndarray) -> np.ndarray:
        return logit(q / (1.0 - q))

    def score(eps: np.ndarray) -> np.ndarray:
        fitted = expit(lin(np.where(a == 1, q1, q0)) + eps[0] * a * h1 + eps[1] * (1 - a) * h0)
        return np.array([np.sum(a * h1 * (y - fitted)), np.sum((1 - a) * h0 * (y - fitted))])

    eps = root(score, np.zeros(2), tol=1e-13).x
    return float(np.mean(expit(lin(q1) + eps[0] * h1) - expit(lin(q0) + eps[1] * h0)))


class TestTheRefits:
    def test_in_sample_leave_one_out_values_are_independent_fits(self) -> None:
        frame = point_frame(8)
        result = fit_point(frame, **glm_learners(cluster_variance="jackknife"))
        record = result.estimates["ate"].jackknife
        column = record.covariance.names.index("ate")
        stored = record.covariance.leave_one_out[:, column]
        independent = np.array(
            [
                fit_point(frame[frame.g != code], **glm_learners()).estimates["ate"].psi
                for code in record.covariance.clusters
            ]
        )
        np.testing.assert_allclose(stored, independent, rtol=0, atol=1e-10)
        # Mutation: re-targeting against the full fit's nuisances, with no refit, moves the
        # leave-one-out values well past the tolerance.
        cluster = result.data.cluster
        fixed = np.array(
            [fixed_nuisance_loo(result, np.flatnonzero(cluster != code)) for code in range(8)]
        )
        assert np.max(np.abs(fixed - independent)) > 1e-4

    def test_cross_fitted_refits_keep_the_split_less_the_cluster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[np.ndarray, Folds]] = []
        original = tmle_module.refit_folds

        def spy(folds: Folds, keep: np.ndarray) -> Folds:
            out = original(folds, keep)
            seen.append((np.asarray(keep), out))
            return out

        monkeypatch.setattr(tmle_module, "refit_folds", spy)
        result = fit_point(
            point_frame(10), **glm_learners(cross_fit=True, n_folds=5, cluster_variance="jackknife")
        )
        full = np.asarray(result.repeats[0].folds.assignment)
        assert len(seen) == 10
        for keep, folds in seen:
            np.testing.assert_array_equal(folds.assignment, full[keep])
            assert folds.n_folds == 5
            assert np.bincount(folds.assignment, minlength=5).min() > 0
        # Mutation: a redrawn split of the kept rows differs from the frozen labels.
        from cleverly.learners.crossfit import random_partition

        keep = seen[0][0]
        redrawn = random_partition(
            keep.size, 5, cluster=result.data.cluster[keep], seed=result.repeats[0].seed + 1
        )
        assert not np.array_equal(redrawn.assignment, full[keep])

    def test_longitudinal_leave_one_out_values_are_independent_fits(self) -> None:
        frame = panel(10)
        result = fit_longitudinal(
            frame, **longitudinal_learners(n_folds=5, cluster_variance="jackknife")
        )
        name = "ate_regimen[always vs never]"
        record = result[name].jackknife
        stored = record.covariance.leave_one_out[:, record.covariance.names.index(name)]
        folds = np.asarray(result.repeats[0].folds.assignment)
        independent = []
        for code in record.covariance.clusters:
            keep = np.flatnonzero(result.data.cluster != code)
            refit = LTMLE(REGIMENS, reference="never", **longitudinal_learners(n_folds=5)).refit(
                result.data.subset(keep),
                frozen_folds=(Folds(folds[keep], 5),),
                frozen_seeds=(result.repeats[0].seed,),
            )
            independent.append(refit[name].psi)
        np.testing.assert_allclose(stored, independent, rtol=0, atol=1e-10)
        assert result[name].variance_source == "jackknife"


# --------------------------------------------------------------------------- the nuisance term


def stacked_sandwich(frame: pd.DataFrame) -> float:
    """The stacked-estimating-equation sandwich over clusters of the in-sample TMLE of the ATE.

    The stack holds the logistic propensity score, the main-terms logistic outcome score, the
    two-dimensional fluctuation score and the plug-in equation. The derivative is by central
    differences and the meat sums each cluster's rows.
    """
    w, a, y = (frame[name].to_numpy(dtype=float) for name in ("w", "a", "y"))
    cluster = frame["g"].to_numpy()
    design_g = np.column_stack([np.ones_like(w), w])
    design_q = np.column_stack([np.ones_like(w), a, w])

    def rows(theta: np.ndarray) -> np.ndarray:
        c, b, eps, psi = theta[:2], theta[2:5], theta[5:7], theta[7]
        g1 = expit(design_g @ c)
        h1, h0 = 1.0 / g1, 1.0 / (1.0 - g1)
        clever = np.column_stack([a * h1, (1 - a) * h0])

        def lin(arm: float | np.ndarray) -> np.ndarray:
            return b[0] + b[1] * arm + b[2] * w

        fitted = expit(lin(a) + clever @ eps)
        q1 = expit(lin(1.0) + eps[0] * h1)
        q0 = expit(lin(0.0) + eps[1] * h0)
        return np.column_stack(
            [
                design_g * (a - g1)[:, None],
                design_q * (y - expit(design_q @ b))[:, None],
                clever * (y - fitted)[:, None],
                (q1 - q0 - psi)[:, None],
            ]
        )

    def total(theta: np.ndarray) -> np.ndarray:
        return rows(theta).sum(axis=0)

    theta = root(total, np.zeros(8), method="hybr", tol=1e-13).x
    per_cluster = np.vstack(
        [rows(theta)[cluster == code].sum(axis=0) for code in np.unique(cluster)]
    )
    step = 1e-6
    basis = np.eye(8)
    bread = np.column_stack(
        [(total(theta + step * e) - total(theta - step * e)) / (2 * step) for e in basis]
    )
    inverse = np.linalg.inv(bread)
    return float((inverse @ per_cluster.T @ per_cluster @ inverse.T)[-1, -1])


def confounded_frame() -> pd.DataFrame:
    """100 clusters of 10 rows: strong confounding and an outcome main terms cannot fit."""
    clusters, size = 100, 10
    rng = np.random.default_rng(1)
    cluster = np.repeat(np.arange(clusters), size)
    effect = rng.normal(scale=0.3, size=clusters)[cluster]
    w = rng.normal(size=clusters * size) + 0.5 * rng.normal(size=clusters)[cluster]
    a = rng.binomial(1, expit(1.5 * w))
    y = rng.binomial(1, expit(-1.5 + 1.0 * a - 2.0 * a * w + 2.5 * w**2 + effect))
    return pd.DataFrame({"y": y, "a": a, "w": w, "g": cluster})


class TestTheNuisanceTerm:
    def test_the_jackknife_reads_the_estimated_mechanism_term(self) -> None:
        frame = confounded_frame()
        result = fit_point(
            frame,
            **glm_learners(
                estimands=("ate",), g_bounds=(1e-6, 1 - 1e-6), cluster_variance="jackknife"
            ),
        )
        estimate = result.estimates["ate"]
        sandwich = stacked_sandwich(frame)
        J = 100
        # The fixture can see the term: the efficient curve's variance misses it.
        assert abs(estimate.jackknife.influence_variance / sandwich - 1.0) > 0.15
        assert abs(estimate.variance / sandwich - 1.0) <= 5 / J
        # Mutation: the efficient-curve-only variance in its place fails the first assertion.
        assert abs(estimate.jackknife.influence_variance / sandwich - 1.0) > 5 / J


# --------------------------------------------------------------------------- R5 refusals


def assert_r5(error: pytest.ExceptionInfo[CapabilityError], what: str) -> None:
    assert error.value.refusal.reason == RefusalReason.INFERENCE_GAP
    assert "cluster_variance='jackknife'" in str(error.value)
    assert what in str(error.value)


class TestR5Refusals:
    @pytest.mark.parametrize(
        ("learner", "named"),
        [
            (None, "the outcome learner the default library"),
            (
                HistGradientBoostingClassifier(),
                "the outcome learner HistGradientBoostingClassifier",
            ),
            (
                SuperLearner([LogisticRegression(), HistGradientBoostingClassifier()]),
                "the outcome learner SuperLearner",
            ),
        ],
    )
    def test_a_data_adaptive_learner_refuses_before_any_learner(
        self, learner: Any, named: str
    ) -> None:
        settings = glm_learners(
            outcome_learner=learner,
            treatment_learner=RaisingLearner(),
            cluster_variance="jackknife",
        )
        with pytest.raises(CapabilityError) as error:
            fit_point(point_frame(10), **settings)
        assert_r5(error, named)
        assert "Fang et al." in str(error.value)

    def test_glm_and_a_super_learner_of_glms_are_admitted(self) -> None:
        assert jackknife_learner(GLM())
        assert jackknife_learner(SuperLearner([LogisticRegression(), GLM()]))
        assert not jackknife_learner(None)
        assert not jackknife_learner(SuperLearner([GLM(), HistGradientBoostingClassifier()]))

    def test_drtmle_refuses_at_construction(self) -> None:
        with pytest.raises(CapabilityError) as error:
            DRTMLE(cluster_variance="jackknife")
        assert_r5(error, "the DR-TMLE reduced regressions")

    def test_a_fold_evaluated_report_refuses_at_construction(self) -> None:
        with pytest.raises(CapabilityError) as error:
            TMLE(cross_fit=True, cv_evaluation=True, cluster_variance="jackknife")
        assert_r5(error, "a fold-evaluated estimate")

    def test_an_incremental_intervention_refuses_at_construction(self) -> None:
        with pytest.raises(CapabilityError) as error:
            TMLE(incremental=[Incremental(2.0)], cluster_variance="jackknife")
        assert_r5(error, "an incremental propensity-score intervention")

    @pytest.mark.parametrize("cls", [SDR, IPSI])
    def test_sdr_and_ipsi_refuse_the_keyword(self, cls: type) -> None:
        # Only LTMLE takes cluster_variance: SDR and IPSI reject it by name, before any learner.
        with pytest.raises(TypeError, match="unexpected keyword argument 'cluster_variance'"):
            cls(REGIMENS, cluster_variance="jackknife")

    def test_a_second_variance_refuses(self) -> None:
        with pytest.raises(CapabilityError, match="Choose one variance"):
            LTMLE(REGIMENS, n_folds=1, variance="policy_weighted", cluster_variance="jackknife")
        with pytest.raises(CapabilityError, match="Choose one variance"):
            TMLE(n_bootstrap=20, cluster_variance="jackknife")

    def test_a_fit_without_clusters_refuses_before_any_learner(self) -> None:
        settings = glm_learners(
            outcome_learner=RaisingLearner(),
            treatment_learner=RaisingLearner(),
            cluster_variance="jackknife",
        )
        with warnings.catch_warnings(), pytest.raises(CapabilityError, match="declare no clusters"):
            warnings.simplefilter("ignore")
            TMLE(**settings).fit(point_frame(10), outcome="y", treatment="a")

    def test_an_unknown_value_is_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="cluster_variance must be one of"):
            TMLE(cluster_variance="bootstrap")


# --------------------------------------------------------------------------- the fold rule


def stratified_split_frame() -> tuple[Any, Folds]:
    """12 clusters, and a split of 5 folds whose fold 0 holds one cluster although J >= 2V."""
    frame = point_frame(12)
    data = fit_point(frame, **glm_learners()).data
    fold_of_cluster = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
    return data, Folds(fold_of_cluster[data.cluster], 5)


class TestTheFoldRule:
    def test_the_rule_reads_the_split_not_j_against_2v(self) -> None:
        data, folds = stratified_split_frame()
        assert 12 >= 2 * 5
        estimator = TMLE(
            **glm_learners(
                cross_fit=True,
                n_folds=5,
                outcome_learner=RaisingLearner(),
                treatment_learner=RaisingLearner(),
                cluster_variance="jackknife",
            )
        )
        with pytest.raises(CapabilityError) as error:
            estimator.refit(data, frozen_folds=(folds,), frozen_seeds=(0,))
        assert_r5(error, "fold 0 holds 1 positive-mass cluster(s)")

    def test_mutation_a_trigger_on_j_below_2v_lets_the_split_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data, folds = stratified_split_frame()

        def by_count(cluster: Any, weights: Any, assignment: Any, n_folds: int) -> Any:
            if np.unique(cluster).size < 2 * n_folds:
                return jackknife_fold_refusal(cluster, weights, assignment, n_folds)
            return None

        monkeypatch.setattr(tmle_module, "jackknife_fold_refusal", by_count)
        estimator = TMLE(**glm_learners(cross_fit=True, n_folds=5, cluster_variance="jackknife"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.refit(data, frozen_folds=(folds,), frozen_seeds=(0,))
        # The refit without the lone cluster of fold 0 loses that fold.
        assert 0 in result.estimates["ate"].jackknife.failed_refits

    def test_stacked_tmle_refuses_five_clusters_and_runs_ten(self) -> None:
        settings = glm_learners(
            cross_fit=True,
            n_folds=5,
            outcome_learner=RaisingLearner(),
            treatment_learner=RaisingLearner(),
            cluster_variance="jackknife",
        )
        with pytest.raises(CapabilityError) as error:
            fit_point(point_frame(5), **settings)
        assert_r5(error, "With 5 clusters in 5 folds")
        result = fit_point(
            point_frame(10), **glm_learners(cross_fit=True, n_folds=5, cluster_variance="jackknife")
        )
        assert result.estimates["ate"].jackknife.failed_refits == ()
        assert np.isfinite(result.estimates["ate"].std_error)

    def test_ltmle_refuses_nine_clusters_and_runs_ten(self) -> None:
        with pytest.raises(CapabilityError) as error:
            fit_longitudinal(
                panel(9),
                **longitudinal_learners(
                    outcome_learner=RaisingLearner(), cluster_variance="jackknife"
                ),
            )
        # n_folds=10 is reduced to the 9 clusters, one per fold.
        assert_r5(error, "With 9 clusters in 9 folds")
        result = fit_longitudinal(
            panel(10), **longitudinal_learners(n_folds=5, cluster_variance="jackknife")
        )
        assert all(estimate.jackknife.failed_refits == () for estimate in result.values())

    def test_mutation_dropping_the_empty_fold_lets_five_clusters_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def compacting(folds: Folds, keep: np.ndarray) -> Folds:
            labels = np.asarray(folds.assignment)[keep]
            _, compact = np.unique(labels, return_inverse=True)
            return Folds(compact.astype(np.int64), int(compact.max()) + 1)

        monkeypatch.setattr(tmle_module, "jackknife_fold_refusal", lambda *args: None)
        monkeypatch.setattr(tmle_module, "refit_folds", compacting)
        result = fit_point(
            point_frame(5), **glm_learners(cross_fit=True, n_folds=5, cluster_variance="jackknife")
        )
        # Each refit ran with four folds: the five-cluster fit passes, so the test above fails.
        assert result.estimates["ate"].jackknife.failed_refits == ()


def test_the_fold_refusal_text_names_the_counts() -> None:
    text = JACKKNIFE_FOLD_REFUSAL.format(clusters=9, folds=9, fold=0, held=1)
    assert "With 9 clusters in 9 folds, fold 0 holds 1" in text


# --------------------------------------------------------------------------- a failed refit


def constant_arm_frame() -> pd.DataFrame:
    """Four clusters; only cluster 0 has treated rows with outcome 1."""
    frame = point_frame(4, seed=5)
    treated = frame.a == 1
    frame.loc[treated, "y"] = 0
    frame.loc[treated & (frame.g == 0), "y"] = 1
    return frame


class TestAWithheldRefit:
    def test_a_constant_arm_refit_withholds_and_names_the_cluster(self) -> None:
        frame = constant_arm_frame()
        with pytest.warns(DataWarning, match=r"cluster\(s\) \[0\]"):
            result = (
                TMLE(**glm_learners(cluster_variance="jackknife"))
                .fit(frame, outcome="y", treatment="a", id="g")
                .single()
            )
        estimate = result.estimates["ey1"]
        assert estimate.inference == "jackknife_refit_plugin"
        assert estimate.jackknife.failed_refits == (0,)
        assert estimate.jackknife.failures == ((0, "withheld its interval"),)
        assert not estimate.supplies_inference
        # The control arm reads no constant cell, so its estimate keeps its interval.
        assert result.estimates["ey0"].jackknife.failed_refits == ()
        assert result.estimates["ey0"].supplies_inference

    def test_a_refit_that_raises_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = tmle_module.leave_one_cluster_out

        def failing(cluster: Any, code: int) -> np.ndarray:
            # An empty row set makes the subset raise inside the refit for cluster 3 only.
            return np.array([], dtype=np.int64) if code == 3 else original(cluster, code)

        monkeypatch.setattr(tmle_module, "leave_one_cluster_out", failing)
        with pytest.warns(DataWarning, match=r"cluster\(s\) \[3\]"):
            result = (
                TMLE(**glm_learners(cluster_variance="jackknife"))
                .fit(point_frame(10), outcome="y", treatment="a", id="g")
                .single()
            )
        estimate = result.estimates["ate"]
        assert estimate.inference == "jackknife_refit_plugin"
        ((code, what),) = estimate.jackknife.failures
        assert code == 3
        assert what.startswith("DataError: subset has 0 rows")


# --------------------------------------------------------------------------- derived estimates


class TestDerivedEstimates:
    def test_the_ate_variance_is_the_quadratic_form_of_the_arm_vector(self) -> None:
        result = fit_point(point_frame(12), **glm_learners(cluster_variance="jackknife"))
        record = result.estimates["ate"].jackknife.covariance
        columns = [record.names.index(name) for name in ("ey1", "ey0")]
        arms = jackknife_covariance(record.leave_one_out[:, columns])
        lam = np.array([1.0, -1.0])
        assert result.estimates["ate"].variance == pytest.approx(lam @ arms @ lam, abs=1e-12)
        derived = result.contrast(lambda x: x[0] - x[1], ["ey1", "ey0"])
        assert derived.variance_source == "jackknife"
        assert derived.variance == pytest.approx(lam @ arms @ lam, abs=1e-12)

    def test_the_covariance_is_the_jackknife_covariance(self) -> None:
        result = fit_point(point_frame(12), **glm_learners(cluster_variance="jackknife"))
        record = result.estimates["ey1"].jackknife.covariance
        columns = [record.names.index(name) for name in ("ey1", "ey0")]
        np.testing.assert_allclose(
            result.covariance(["ey1", "ey0"]),
            jackknife_covariance(record.leave_one_out[:, columns]),
            rtol=1e-12,
        )

    def test_the_frame_reports_both_standard_errors(self) -> None:
        result = fit_point(point_frame(12), **glm_learners(cluster_variance="jackknife"))
        row = result.estimates["ate"].to_frame().iloc[0]
        assert row["variance_source"] == "jackknife"
        assert row["jackknife_std_err"] == pytest.approx(row["std_err"])
        assert row["influence_curve_std_err"] != pytest.approx(row["std_err"], rel=1e-6)
        assert row["jackknife_failed_refits"] == 0

    def test_a_round_trip_keeps_the_record(self) -> None:
        result = fit_point(point_frame(10), **glm_learners(cluster_variance="jackknife"))
        loaded = loads(dumps(result))
        assert loaded.estimates["ate"].std_error == result.estimates["ate"].std_error
        assert loaded.estimates["ate"].jackknife.covariance.names == ("ate", "ey1", "ey0")


# --------------------------------------------------------------------------- bands and support


class TestBandsAndSupport:
    def test_no_band_is_built_and_the_summary_says_why(self) -> None:
        result = fit_point(
            point_frame(40), **glm_learners(simultaneous=True, cluster_variance="jackknife")
        )
        assert all(estimate.reference_df is None for estimate in result.estimates.values())
        assert result.simultaneous is None
        text = " ".join(str(result.summary(detail="full")).split())
        assert "report the delete-one-cluster jackknife variance" in text
        assert "Variance: the delete-one-cluster jackknife, from 40 refits" in text
        # Mutation control: the same fit under the default variance builds its band.
        default = fit_point(point_frame(40), **glm_learners(simultaneous=True))
        assert default.simultaneous is not None

    def test_simultaneous_bands_refuses_jackknife_estimates(self) -> None:
        # At 40 clusters the normal reference applies, so the t refusal does not come first.
        result = fit_point(point_frame(40), **glm_learners(cluster_variance="jackknife"))
        with pytest.raises(CapabilityError) as error:
            simultaneous_bands(result.estimates, cluster=result.data.cluster)
        assert JACKKNIFE_BAND_REFUSAL.condition in str(error.value)

    def test_the_fit_reads_experimental_and_names_the_option(self) -> None:
        result = fit_point(point_frame(12), **glm_learners(cluster_variance="jackknife"))
        assert ("cluster_variance", "jackknife") in result.config.support_key.options
        assert result.estimates["ate"].support.label == "experimental"
        default = fit_point(point_frame(12), **glm_learners())
        assert "cluster_variance" not in dict(default.config.support_key.options)

    def test_mixing_jackknife_and_curve_estimates_refuses(self) -> None:
        jack = fit_point(point_frame(12), **glm_learners(cluster_variance="jackknife"))
        plain = fit_point(point_frame(12), **glm_learners())
        mixed = {"ey1": jack.estimates["ey1"], "ey0": plain.estimates["ey0"]}
        with pytest.raises(CapabilityError, match="mix a cluster jackknife variance"):
            cluster_jackknife_contrast(mixed)


class TestTheMeasuredRefusal:
    """X54 part (c): the registered measurement refuses the point-TMLE jackknife."""

    shipped_refusals: ClassVar[bool] = True

    def test_in_sample_tmle_refuses_with_the_measured_reason_before_any_learner(self) -> None:
        settings = glm_learners(
            outcome_learner=RaisingLearner(),
            treatment_learner=RaisingLearner(),
            cluster_variance="jackknife",
        )
        with pytest.raises(CapabilityError) as error:
            fit_point(point_frame(12), **settings)
        text = str(error.value)
        assert error.value.refusal.reason == RefusalReason.INFERENCE_GAP
        assert "few-cluster-floor-tmle" in text and "-0.07" in text and "-0.03" in text
        assert error.value.refusal.roadmap == "F28"

    def test_stacked_tmle_waits_for_the_redeclaration(self) -> None:
        with pytest.raises(CapabilityError) as error:
            TMLE(cross_fit=True, n_folds=5, cluster_variance="jackknife")
        assert "redeclared" in str(error.value)
        assert error.value.refusal.roadmap == "X54"

    def test_ltmle_keeps_the_option(self) -> None:
        assert cluster_jackknife.jackknife_family_refusal("ltmle_in_sample") is None
        assert cluster_jackknife.jackknife_family_refusal("ltmle_cross_fitted") is None
        assert cluster_jackknife.jackknife_family_refusal(None) is None

    def test_mutation_an_empty_map_lets_the_in_sample_fit_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cluster_jackknife, "JACKKNIFE_FAMILY_REFUSALS", {})
        result = fit_point(point_frame(12), **glm_learners(cluster_variance="jackknife"))
        assert result.estimates["ate"].jackknife is not None


def cluster_jackknife_contrast(estimates: Any) -> Any:
    from cleverly.inference.results import smooth_contrast

    return smooth_contrast(estimates, lambda x: x[0] - x[1], ["ey1", "ey0"], n=12 * SIZE)


def test_the_module_lists_every_public_name() -> None:
    assert set(cluster_jackknife.__all__) <= set(dir(cluster_jackknife))


def _tmle_jackknife_paragraph(families: Any) -> str:
    """The ``cluster_variance`` paragraph of ``TMLE.__doc__``, checked against ``families``.

    Returns the paragraph after asserting that it promises no study and names exactly the
    roadmap ids of the family refusals, so a family that the refusal map lifts or adds fails
    the check until the paragraph is rewritten.
    """
    import inspect
    import re

    doc = inspect.cleandoc(TMLE.__doc__ or "")
    start = doc.index("\ncluster_variance:")
    end = doc.index("\nrandom_state", start)
    paragraph = " ".join(doc[start:end].split())
    assert "experimental until" not in paragraph
    named = set(re.findall(r"roadmap ([A-Z]+\d+)", paragraph))
    assert named == {roadmap for _, roadmap in families.values()}, named
    return paragraph


class TestTheDocstringStatesTheShippedRefusals:
    """M2: the docstring said the option was experimental, while both TMLE families refuse."""

    shipped_refusals = True

    def test_the_paragraph_names_each_family_refusal(self) -> None:
        paragraph = _tmle_jackknife_paragraph(cluster_jackknife.JACKKNIFE_FAMILY_REFUSALS)
        assert "refuses" in paragraph

    def test_a_lifted_family_fails_the_check(self) -> None:
        """Mutation control: when a family stops refusing, the paragraph must change."""
        lifted = {
            name: value
            for name, value in cluster_jackknife.JACKKNIFE_FAMILY_REFUSALS.items()
            if name != "tmle_stacked"
        }
        with pytest.raises(AssertionError):
            _tmle_jackknife_paragraph(lifted)

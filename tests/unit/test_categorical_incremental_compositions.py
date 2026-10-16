r"""Does the categorical incremental fit compose with weights, clusters, folds and resampling?

The exact-law tests (C1 to C13) fit with oracle nuisances, in sample and unweighted.  This file
runs the same three-level law through the four compositions that the binary fit supports, each
against an identity that holds without a simulation.

* C14w: integer observation weights are row duplication.  The saturated learners honour
  weights, so the weighted and the duplicated fits hold the same cell frequencies.  The
  main-terms fit repeats the identity through the mechanism fluctuation, whose score carries the
  weights.
* C14c: ``id=`` changes the variance and nothing else.  Singleton clusters give the iid fit,
  and clusters of four give the cluster-robust formula written longhand.
* C14f: cross-fitting.  Five copies of the exact sample, each copy one cluster, make the five
  folds the five copies, so every held-out fold is predicted from four exact copies of the law
  and the cross-fitted fit is exact.  The main-terms 5-fold fit must solve both score equations
  and move the mechanism.
* C14r: ``retarget`` on the cached nuisances reproduces the estimates, and
  :meth:`~cleverly.interventions.IPSISet.subset` slices every array.
* C14b: a bootstrap replicate is the fit on the resampled rows, so the tilt is re-derived from
  the refitted mechanism and not sliced from the original one.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.estimators import TMLE
from cleverly.inference.bootstrap import _bootstrap_design, _bootstrap_draw
from cleverly.interventions import IPSISet
from tests import discrete_law_categorical_incremental as law
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities
from tests.unit import _categorical_incremental_support as sup

NAMES = law.MEANS + law.CONTRASTS
#: A weight or a cluster must move the estimate by more than this, or the identity is vacuous.
WITNESS = 1e-3


def _fit(frame: pd.DataFrame, learners: dict[str, Any], **settings: Any) -> Any:
    """A fit of the law's tilts; ``weights`` and ``id`` are roles, the rest are settings."""
    roles = {key: settings.pop(key) for key in ("weights", "id") if key in settings}
    config: dict[str, Any] = {
        **learners,
        "cross_fit": False,
        "incremental": sup.declared(),
        "estimands": "all",
        "simultaneous": False,
        "random_state": 0,
    }
    config.update(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(**config)
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], **roles)
            .single()
        )


def saturated(frame: pd.DataFrame, **settings: Any) -> Any:
    """Cell-frequency learners: the oracle on a sample that realises the law."""
    learners = {"outcome_learner": CellMeans(), "treatment_learner": CellProbabilities()}
    return _fit(frame, learners, **settings)


def weak(frame: pd.DataFrame, **settings: Any) -> Any:
    """Main-terms learners: the targeting step must move both nuisances."""
    learners = {
        "outcome_learner": sup.MainTermsOutcome(width=3),
        "treatment_learner": sup.MainTermsTreatment(),
        "tol": sup.TOL,
        "max_iter": sup.MAX_ITER,
    }
    return _fit(frame, learners, **settings)


def _frame() -> pd.DataFrame:
    return sup.frame()


def _weights(n: int) -> np.ndarray:
    """Integer weights one to three, so that duplication is possible and not uniform."""
    return np.random.default_rng(1).integers(1, 4, size=n).astype(float)


def _duplicated(frame: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    return frame.loc[frame.index.repeat(weights.astype(int))].reset_index(drop=True)


def _max_gap(left: Any, right: Any) -> float:
    return max(abs(left.estimates[name].psi - right.estimates[name].psi) for name in NAMES)


# ---------------------------------------------------------------------------------- C14w


class TestC14wWeights:
    @pytest.fixture(scope="class")
    def weights(self) -> np.ndarray:
        return _weights(len(_frame()))

    def test_integer_weights_equal_row_duplication_with_saturated_learners(
        self, weights: np.ndarray
    ) -> None:
        frame = _frame()
        weighted = saturated(frame.assign(wt=weights), weights="wt")
        duplicated = saturated(_duplicated(frame, weights))
        assert _max_gap(weighted, duplicated) < 1e-12
        # The weights are not uniform in effect: the unweighted estimate is a different number.
        assert _max_gap(weighted, saturated(frame)) > WITNESS

    def test_integer_weights_equal_row_duplication_through_the_mechanism_fluctuation(
        self, weights: np.ndarray
    ) -> None:
        frame = _frame()
        weighted = weak(frame.assign(wt=weights), weights="wt")
        duplicated = weak(_duplicated(frame, weights))
        unweighted = weak(frame)
        # The solvers stop at a score of 1e-10, so the two fits agree to a few orders above it.
        assert _max_gap(weighted, duplicated) < 1e-8
        assert _max_gap(weighted, unweighted) > WITNESS
        g_weighted, _ = sup.targeted(weighted, False)
        g_unweighted, _ = sup.targeted(unweighted, False)
        # The first row of each duplicated block is the row the weighted fit holds.
        first = np.cumsum(np.r_[0, weights.astype(int)[:-1]])
        g_duplicated, _ = sup.targeted(duplicated, False)
        np.testing.assert_allclose(g_weighted, g_duplicated[first], atol=1e-7, rtol=0)
        assert np.max(np.abs(g_weighted - g_unweighted)) > WITNESS


# ---------------------------------------------------------------------------------- C14c


class TestC14cClusters:
    def test_singleton_clusters_are_the_iid_fit(self) -> None:
        frame = _frame()
        iid = saturated(frame)
        singleton = saturated(frame.assign(cid=np.arange(len(frame))), id="cid")
        for name in NAMES:
            left, right = singleton.estimates[name], iid.estimates[name]
            assert left.psi == pytest.approx(right.psi, abs=1e-12)
            assert left.std_error == pytest.approx(right.std_error, abs=1e-12)
            np.testing.assert_allclose(
                left.influence_curve, right.influence_curve, atol=1e-12, rtol=0
            )

    def test_the_cluster_standard_error_is_the_longhand_formula(self) -> None:
        frame = _frame()
        ids = np.arange(len(frame)) // 4
        clustered = saturated(frame.assign(cid=ids), id="cid")
        iid = saturated(frame)
        n, n_clusters = len(frame), int(ids.max()) + 1
        for name in law.MEANS:
            estimate = clustered.estimates[name]
            sums = pd.Series(np.asarray(estimate.influence_curve)).groupby(ids).sum().to_numpy()
            longhand = np.sqrt(np.sum(sums**2) * n_clusters / (n_clusters - 1)) / n
            assert estimate.std_error == pytest.approx(longhand, abs=1e-12)
            assert estimate.psi == pytest.approx(iid.estimates[name].psi, abs=1e-12)
            assert estimate.std_error > 1.5 * iid.estimates[name].std_error


# ---------------------------------------------------------------------------------- C14f


class TestC14fCrossFitting:
    def test_five_exact_copies_as_five_folds_give_the_exact_fit(self) -> None:
        copies = pd.concat([_frame().assign(cid=copy) for copy in range(5)], ignore_index=True)
        fit = saturated(copies, id="cid", cross_fit=True, n_folds=5)
        for name in NAMES:
            assert fit.estimates[name].psi == pytest.approx(law.truth_of(name), abs=1e-12), name

    def test_a_weak_five_fold_fit_solves_both_equations_and_moves_the_mechanism(self) -> None:
        fit = weak(_frame(), cross_fit=True, n_folds=5)
        fluctuation = fit.fluctuations["ipsi"]
        assert fluctuation.mechanism.failure is None
        assert fluctuation.mechanism.relative_score < sup.SCORE
        assert fluctuation.relative_score_norm < sup.SCORE
        g_star, _ = sup.targeted(fit, False)
        np.testing.assert_allclose(g_star.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        initial = np.asarray(fit.nuisance.propensity.values, dtype=float)
        assert np.max(np.abs(g_star - initial)) > WITNESS, "a zero step hides a missing fluctuation"


# ---------------------------------------------------------------------------------- C14r


class TestC14rRetargetAndSubset:
    @pytest.mark.parametrize("learners", [saturated, weak], ids=["saturated", "weak"])
    def test_retarget_on_the_cached_nuisances_reproduces_the_estimates(self, learners: Any) -> None:
        fit = learners(_frame())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimates, _ = fit.estimator.retarget(
                fit.data, fit.nuisance, estimands=("ey_ipsi", "ate_ipsi")
            )
        assert set(estimates) == set(NAMES)
        for name in NAMES:
            assert estimates[name].psi == pytest.approx(fit.estimates[name].psi, abs=1e-12)

    def test_subset_slices_every_array_and_keeps_the_normaliser(self) -> None:
        fit = weak(_frame())
        tilts = fit.nuisance.incremental
        rows = np.flatnonzero(np.arange(tilts.propensity.shape[0]) % 3 == 0)
        mask = np.zeros(tilts.propensity.shape[0], dtype=bool)
        mask[rows] = True
        for index in (rows, mask):
            sliced = tilts.subset(index)
            assert isinstance(sliced, IPSISet)
            for field in ("values", "weights", "normaliser", "propensity"):
                np.testing.assert_array_equal(
                    getattr(sliced, field), getattr(tilts, field)[rows], err_msg=field
                )
            for position, factors in enumerate(sliced.factors):
                np.testing.assert_allclose(
                    sliced.normaliser[:, position],
                    sliced.propensity @ np.asarray(factors),
                    atol=1e-12,
                    rtol=0,
                )


# ---------------------------------------------------------------------------------- C14b


class TestC14bBootstrap:
    def test_a_replicate_is_the_fit_on_the_resampled_rows(self) -> None:
        frame = _frame()
        fit = weak(frame, n_bootstrap=4)
        plain = weak(frame)
        name = "ey_ipsi[T_a]"
        assert fit.estimates[name].psi == pytest.approx(plain.estimates[name].psi, abs=1e-12)
        summary = fit.estimates[name].bootstrap
        assert summary.n_failed == 0
        design = _bootstrap_design(fit.data, n_replicates=4, resampling="auto", random_state=0)
        for draw, value in zip(design.draws, summary.draws, strict=True):
            index, _ = _bootstrap_draw(
                fit.data.n, design.cluster, np.random.default_rng(draw.sequence)
            )
            resampled = weak(frame.iloc[index].reset_index(drop=True))
            assert value == pytest.approx(resampled.estimates[name].psi, abs=1e-8)
        assert np.ptp(summary.draws) > 1e-3, "identical draws would hide a sliced tilt"

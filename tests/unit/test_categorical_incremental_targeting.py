r"""The softmax mechanism fluctuation does real work, and the work is right.

On an oracle fit the targeting step has nothing to do: the sample realises the law, both score
equations are zero at :math:`\epsilon = 0`, and a mutation of the fluctuation leaves every
exact-law assertion unchanged.  The witnesses here use deliberately **non-saturated**
learners, a main-terms multinomial logistic :math:`\hat g` and a linear :math:`\hat{\bar Q}`,
so the targeting step moves both nuisances.

* C1t: both score equations are solved at the returned pair; the mechanism coefficient
  :math:`\epsilon_g` is not zero; rows of :math:`g^*` stay on the simplex; and, because the
  sample *is* the law and :math:`P_n\varphi^* = 0`, the bias of the estimate equals the exact
  remainder :math:`E_0[(m^* - m_0)(D^* - D_0)/D^*]` at the targeted pair;
* C4 with weak learners: all factors equal is :math:`\bar Y` only when **both** equations are
  truly solved;
* C8: the softmax solver against a brute-force grid, the logistic solver at two levels, and
  its failure vocabulary.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from cleverly.fluctuation.mechanism import (
    multinomial_mechanism_score,
    solve_mechanism,
    solve_multinomial_mechanism,
)
from cleverly.interventions import Incremental
from tests import discrete_law_categorical_incremental as law
from tests import discrete_law_multi as base
from tests.unit import _categorical_incremental_support as sup


def _mechanism_movement(fit: Any, rows: np.ndarray | None = None) -> float:
    """The largest change the targeting step made to any entry of the mechanism.

    ``mechanism.epsilon`` is the coefficient of the last alternation round, which is zero
    once the loop has settled, so the movement of :math:`g` itself is what shows the
    fluctuation worked.
    """
    g_star, _ = sup.targeted(fit, False)
    moved = np.abs(g_star - np.asarray(fit.nuisance.propensity.values, dtype=float))
    return float(np.max(moved if rows is None else moved[rows]))


@pytest.fixture(scope="module")
def unstratified() -> Any:
    return sup.weak_fit()


@pytest.fixture(scope="module")
def stratified() -> Any:
    return sup.weak_fit(strata=True)


@pytest.fixture(scope="module")
def missing() -> Any:
    return sup.weak_fit(missing=True)


class TestC1tTheMechanismFluctuationMoves:
    @pytest.mark.parametrize("name", ["T_a", "T_b"])
    @pytest.mark.parametrize("fixture", ["unstratified", "missing"])
    def test_both_equations_are_solved_and_the_bias_is_the_exact_remainder(
        self, request: pytest.FixtureRequest, fixture: str, name: str
    ) -> None:
        fit = request.getfixturevalue(fixture)
        is_missing = fixture == "missing"
        fluctuation = fit.fluctuations["ipsi"]
        assert fluctuation.relative_score_norm < sup.SCORE, "the outcome equation is open"
        assert fluctuation.mechanism.relative_score < sup.SCORE, "the mechanism equation is open"
        # The mechanism fluctuation did work: this is the term no oracle fit can move.
        assert _mechanism_movement(fit) > 1e-3
        assert fluctuation.mechanism.trace[0][2] > 1e-6, (
            "the first round started with an open score"
        )
        bias = fit.estimates[f"ey_ipsi[{name}]"].psi - law.truth_of(
            f"ey_ipsi[{name}]", missing=is_missing
        )
        closed = sup.remainder_at(fit, law.TILTS[name], is_missing, np.ones(fit.data.n, dtype=bool))
        assert bias == pytest.approx(closed, abs=1e-7)
        assert abs(closed) > 1e-3, "the misspecification must leave a visible bias"

    @pytest.mark.parametrize("fixture", ["unstratified", "stratified", "missing"])
    def test_the_targeted_mechanism_stays_on_the_simplex(
        self, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        fit = request.getfixturevalue(fixture)
        g_star, _ = sup.targeted(fit, fixture == "missing")
        np.testing.assert_allclose(g_star.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        assert g_star.min() > 0.0

    def test_the_stratified_equations_are_solved_block_by_block(self, stratified: Any) -> None:
        fluctuation = stratified.fluctuations["ipsi"]
        assert fluctuation.relative_score_norm < sup.SCORE
        assert fluctuation.mechanism.relative_score < sup.SCORE
        assert np.asarray(fluctuation.mechanism.epsilon).shape == (2 * len(law.TILTS),)
        inside = np.asarray(sup.STRATA_OF_W)[np.rint(stratified.data.covariates[:, 0]).astype(int)]
        for stratum in (0, 1):
            assert _mechanism_movement(stratified, inside == stratum) > 1e-3

    @pytest.mark.parametrize("stratum", [0, 1])
    @pytest.mark.parametrize("name", ["T_a", "T_b"])
    def test_each_stratum_bias_is_its_exact_remainder(
        self, stratified: Any, stratum: int, name: str
    ) -> None:
        inside = np.asarray(sup.STRATA_OF_W)[np.rint(stratified.data.covariates[:, 0]).astype(int)]
        rows = inside == stratum
        bias = stratified[f"ey_ipsi[{name}][V={stratum}]"].psi - law.truth_stratum(
            law.TILTS[name], stratum
        )
        closed = sup.remainder_at(stratified, law.TILTS[name], False, rows)
        assert bias == pytest.approx(closed, abs=1e-7)
        # Stratum 0 holds the single cell W = 0, where any model of W is saturated and the
        # stratum's own score equations leave nothing to be wrong; stratum 1 holds two.
        if stratum == 1:
            assert abs(closed) > 1e-3
        else:
            assert abs(closed) < 1e-9


class TestC4TheIdentityNeedsBothEquationsSolved:
    @pytest.mark.parametrize("scale", [1.0, 5.0])
    def test_equal_factors_give_the_sample_mean_with_weak_learners(self, scale: float) -> None:
        tilt = Incremental(dict.fromkeys(base.LABELS, scale), name="all equal")
        fit = sup.weak_fit(tilts=[tilt])
        y = np.asarray(fit.data.outcome, dtype=float)
        estimate = fit.estimates["ey_ipsi[all equal]"]
        assert estimate.psi == pytest.approx(float(y.mean()), abs=1e-9)
        np.testing.assert_allclose(estimate.influence_curve, y - y.mean(), atol=1e-8, rtol=0)

    def test_a_tilt_that_is_not_equal_is_not_the_sample_mean(self) -> None:
        """The control: an unequal tilt of the same weak fit differs from ``mean(Y)``."""
        fit = sup.weak_fit(tilts=[Incremental(law.T_A, name="tilted")])
        y = np.asarray(fit.data.outcome, dtype=float)
        assert abs(fit.estimates["ey_ipsi[tilted]"].psi - float(y.mean())) > 0.01


# ------------------------------------------------------------------------------------ C8


def _problem(seed: int, n: int = 400, k: int = 4, r: int = 2) -> tuple[Any, ...]:
    rng = np.random.default_rng(seed)
    g = rng.dirichlet(np.ones(k) * 2.0, size=n)
    h = rng.normal(size=(n, k, r))
    h = h - np.einsum("ij,ijr->ir", g, h)[:, None, :]
    a = np.array([rng.choice(k, p=row) for row in g]).astype(float)
    return a, g, h, rng.uniform(0.5, 1.5, size=n)


def _loglik(a: Any, g: Any, h: Any, w: Any, epsilon: Any) -> float:
    eta = np.log(g) + h @ epsilon
    eta = eta - eta.max(axis=1, keepdims=True)
    p = np.exp(eta)
    p = p / p.sum(axis=1, keepdims=True)
    return float(np.sum(w * np.log(p[np.arange(a.size), a.astype(int)])))


class TestC8TheSoftmaxSolver:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_newton_epsilon_is_the_grid_maximiser(self, seed: int) -> None:
        a, g, h, w = _problem(seed)
        fit = solve_multinomial_mechanism(a, g, h, w, tol=1e-12)
        assert fit.converged and fit.failure is None
        epsilon = np.asarray(fit.epsilon)
        best = _loglik(a, g, h, w, epsilon)
        spacing = 0.01
        grid = np.arange(-0.6, 0.6 + spacing / 2, spacing)
        values = {
            point: _loglik(a, g, h, w, np.array(point)) for point in itertools.product(grid, grid)
        }
        top = max(values, key=lambda point: values[point])
        assert values[top] <= best + 1e-9, "a grid point beats the Newton solution"
        assert np.max(np.abs(np.array(top) - epsilon)) <= spacing
        assert best > _loglik(a, g, h, w, np.zeros(2)), "the covariate must move the likelihood"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_score_is_zero_and_the_rows_stay_on_the_simplex(self, seed: int) -> None:
        a, g, h, w = _problem(seed)
        fit = solve_multinomial_mechanism(a, g, h, w, tol=1e-12)
        score, scale = multinomial_mechanism_score(a, fit.propensity, h, w)
        assert np.max(np.abs(score) / scale) < 1e-10
        np.testing.assert_allclose(fit.propensity.sum(axis=1), 1.0, atol=1e-12, rtol=0)
        assert np.max(np.abs(np.asarray(fit.epsilon))) > 1e-3
        assert fit.score_initial is not None and np.max(np.abs(fit.score_initial)) > 1e-3

    @pytest.mark.parametrize("seed", [0, 1])
    def test_two_levels_equal_the_logistic_solver(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = 300
        g1 = rng.uniform(0.15, 0.85, n)
        a = (rng.uniform(size=n) < g1).astype(float)
        blip = rng.normal(size=(n, 2))
        w = rng.uniform(0.5, 1.5, n)
        softmax = solve_multinomial_mechanism(
            a,
            np.column_stack([1.0 - g1, g1]),
            np.stack([np.zeros((n, 2)), blip], axis=1),
            w,
            tol=1e-12,
        )
        logistic = solve_mechanism(a, g1, blip, w, tol=1e-12)
        np.testing.assert_allclose(softmax.epsilon, logistic.epsilon, atol=1e-10, rtol=0)
        np.testing.assert_allclose(softmax.propensity[:, 1], logistic.propensity, atol=1e-10)

    def test_a_zero_covariate_leaves_the_mechanism_untouched(self) -> None:
        a, g, h, w = _problem(3)
        fit = solve_multinomial_mechanism(a, g, np.zeros_like(h), w)
        assert fit.converged
        np.testing.assert_array_equal(fit.propensity, g)
        assert not np.any(fit.epsilon)

    def test_a_level_that_never_occurs_drives_epsilon_off(self) -> None:
        """Separation: nobody received level 0, and its coefficient keeps one sign.

        The solver reports what the logistic one does: the score falls below the tolerance
        only as ``epsilon`` runs to minus infinity.  The stratum guard exists so that a fit
        never gets here with a stratum that lacks a level; see
        ``test_an_absent_level_in_a_stratum_refuses_before_any_learner``.
        """
        a, g, h, w = _problem(4, k=3, r=1)
        a = np.where(a == 0.0, 1.0, a)
        h = np.zeros_like(h)
        h[:, 0, 0] = 1.0
        h[:, 1:, 0] = -0.5
        fit = solve_multinomial_mechanism(a, g, h, w, max_iter=50)
        assert float(np.min(fit.epsilon)) < -10.0
        np.testing.assert_allclose(fit.propensity.sum(axis=1), 1.0, atol=1e-12, rtol=0)

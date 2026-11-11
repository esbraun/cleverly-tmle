r"""A linked working model's projection is the unique minimiser of the link's cumulant loss.

CDR-005 of the external review: the projection under a logit link used to be defined by least
squares on the mean scale, :math:`\sum_a h\,(\bar Q_a - m_a)^2`, which is not convex in
:math:`\beta`.  On the reviewer's eight-arm law below, Newton from zero converged to a
stationary point whose loss (0.05336) exceeds that of the explicit competitor
:math:`(-3.8, 2.1)` (0.04414), so the reported coefficient was not the declared argmin.

The projection now minimises :math:`\sum_a h\,\{B(\eta) - \bar Q_a\eta\}`, with :math:`B` the
link's cumulant, which is convex.  These tests pin three things on that law: the returned
coefficients minimise the cumulant loss against a grid of starts; they solve the
quasi-likelihood estimating equation; and they differ from the old least-squares stationary
point, which is the mutation control showing that the loss is live on this law.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from cleverly.msm import solve_projection

#: The reviewer's eight arms: dose ``x`` and exact counterfactual risk ``q``.
DOSE = np.array([-3.0, -1.4, 0.5, 0.6, 1.0, 1.1, 1.7, 2.5])
RISK = np.array([0.45, 0.37, 0.09, 0.15, 0.07, 0.16, 0.44, 0.80])
PHI = np.column_stack([np.ones(8), DOSE])

#: The reviewer's least-squares stationary point and the competitor that beat it.
LEAST_SQUARES_ROOT = np.array([-0.83001974, 0.11355275])
COMPETITOR = np.array([-3.8, 2.1])

#: A grid of starting points for an independent optimiser.
STARTS = [np.array([a, b]) for a in (-6.0, -3.0, 0.0, 3.0) for b in (-3.0, -1.0, 1.0, 3.0)]


def expit(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def cumulant_loss(link: str, beta: np.ndarray) -> float:
    """The mean of ``B(eta) - q eta`` over the arms, written out here."""
    eta = PHI @ beta
    if link == "logit":
        return float(np.mean(np.logaddexp(0.0, eta) - RISK * eta))
    return float(np.mean(np.exp(eta) - RISK * eta))


def least_squares_loss(beta: np.ndarray) -> float:
    return float(np.mean((RISK - expit(PHI @ beta)) ** 2))


def solve(link: str):
    return solve_projection(PHI[None], np.ones((1, 8)), RISK[None], np.ones(1), link=link)


class TestTheReviewersLaw:
    def test_least_squares_has_a_non_global_stationary_point(self) -> None:
        """The premise of the finding, so the test below is about a live defect."""
        assert least_squares_loss(COMPETITOR) < least_squares_loss(LEAST_SQUARES_ROOT) - 5e-3
        eta = PHI @ LEAST_SQUARES_ROOT
        m = expit(eta)
        gradient = PHI.T @ ((RISK - m) * m * (1.0 - m))
        assert np.max(np.abs(gradient)) < 1e-6

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_the_solution_minimises_the_cumulant_loss(self, link: str) -> None:
        fit = solve(link)
        assert fit.converged
        reached = min(
            minimize(lambda b: cumulant_loss(link, b), start, method="BFGS").fun for start in STARTS
        )
        assert cumulant_loss(link, fit.beta) <= reached + 1e-12
        assert cumulant_loss(link, fit.beta) <= cumulant_loss(link, COMPETITOR)

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_the_solution_solves_the_quasi_likelihood_equation(self, link: str) -> None:
        fit = solve(link)
        eta = PHI @ fit.beta
        m = expit(eta) if link == "logit" else np.exp(eta)
        np.testing.assert_allclose(PHI.T @ (RISK - m), 0.0, atol=1e-11)

    def test_the_solution_is_not_the_least_squares_root(self) -> None:
        """The mutation control: the loss is live, so the old target would be caught."""
        fit = solve("logit")
        assert np.max(np.abs(fit.beta - LEAST_SQUARES_ROOT)) > 1e-2

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_every_start_reaches_the_same_minimiser(self, link: str) -> None:
        """Strict convexity: an independent optimiser lands on one point from every start."""
        fit = solve(link)
        for start in STARTS:
            found = minimize(lambda b: cumulant_loss(link, b), start, method="BFGS", tol=1e-12)
            np.testing.assert_allclose(found.x, fit.beta, atol=1e-5)

r"""The weighted reduced regressions of a conditional contrast's projection (P1 PR 3, c1).

The DR-TMLE projection fits one block of reduced regressions per basis term. The known weight
:math:`\varphi_j(V)` enters the targets of :math:`Q_r` and :math:`g_{r,2}`, and
:math:`g_{r,1}` stays one unweighted regression per arm:

.. math::

    Q_{r,j}(a, w)   &= E[\varphi_j(V)\{Y - \hat{\bar Q}(a, W)\} \mid A = a, \hat g_a(W) = \hat g_a(w)] \\
    g_{r,2,j}(a|w)  &= E[\varphi_j(V)\{1_a - \hat g_a(W)\}/\hat g_a(W)
                        \mid \hat{\bar Q}(a, W) = \hat{\bar Q}(a, w)]

**Why two variant laws.** A weight that is constant on the level sets of the design cancels
from the regression, so a fit that multiplies by :math:`\varphi_j` *after* regressing returns
the same array. The base law of :mod:`tests.discrete_law` separates every cell in both
:math:`g_0` and :math:`\bar Q_0`, so it cannot see that mutant. Law L-g ties the propensity of
``w = 0`` and ``w = 1``, and law L-Q ties :math:`\bar Q_0(1, \cdot)` across the same two cells,
each with every cell count integral at ``N = 1000``.

On L-g with the arm-1 outcome error ``+0.1`` at ``w = 1, 2`` the slope's :math:`Q_r` is ``-0.0375`` on
the tied rows. The weight-after mutant gives 0 on the ``w = 0`` rows. The drop-weight mutant
gives ``-0.0375`` there too (``w^2 = w`` on ``{0, 1}``) and departs only on ``w = 2``. On L-Q
with arm 1's mechanism ``G + (0.05, -0.05, 0)`` the slope's :math:`g_{r,2}` is ``0.0341`` on
both tied rows, while the weight-after mutant gives 0 and ``-0.0354`` and the drop-weight
mutant ``-0.0354`` on both (P1 PR 3 plan, 4.1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.data import CausalData
from cleverly.estimators._nuisance import NuisanceEstimates, Propensity
from cleverly.estimators.reduced import ReducedSet, fit_reduced
from cleverly.fluctuation.iterative import InitialFit
from cleverly.learners.crossfit import Folds
from cleverly.utils.bounds import OutcomeScaler
from tests import discrete_law as law
from tests.discrete_law_longitudinal import CellMeans

ARMS = (0.0, 1.0)
EXACT = 1e-12
INERT = (1e-6, 1.0 - 1e-6)

#: Law L-g: the propensity ties ``w = 0`` and ``w = 1``.
G_TIED = np.array([0.40, 0.40, 0.25])
#: Its arm-specific wrong outcome regression on arm 1 only: ``Q + 0.1`` at ``w = 1`` and
#: ``w = 2``. The plan's ``Q + 0.1 w`` would put ``Qbar(1, 2)`` at 1.0, where no logistic
#: fluctuation moves it; the tied cells, and so the 0.0375 figure, are unchanged.
Q_WRONG_ARM1 = law.Q + np.array([[0.0, 0.0], [0.0, 0.1], [0.0, 0.1]])

#: Law L-Q: ``Qbar_0(1, .) = (0.70, 0.70, 0.80)`` ties ``w = 0`` and ``w = 1`` on arm 1.
Q_TIED = law.Q.copy()
Q_TIED[1, 1] = 0.70
#: Its wrong mechanism: arm 1's ``g_n = G + (0.05, -0.05, 0)``.
G_WRONG = law.G + np.array([0.05, -0.05, 0.0])


def basis_of(w: np.ndarray) -> np.ndarray:
    """``phi = (1, w)``, the basis of ``Linear(["W"])``."""
    return np.column_stack([np.ones(w.size), w.astype(float)])


def variant(g: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, Any]:
    """The cell counts and the sample of the variant law; the counts are integral."""
    counts = law.cell_counts(g=g, q=q)
    return counts, law.frame(counts)


def fit_at(frame: Any, g_hat: np.ndarray, q_hat: np.ndarray, basis: Any = None) -> ReducedSet:
    """The reduced set relative to the given nuisances, in sample with saturated learners."""
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])
    w = frame["W"].to_numpy().astype(int)
    a = frame["A"].to_numpy(dtype=float)
    at_one, at_zero = q_hat[w, 1], q_hat[w, 0]
    nuisance = NuisanceEstimates(
        propensity=Propensity(np.column_stack([1.0 - g_hat[w], g_hat[w]]), ARMS),
        outcome=InitialFit(
            observed=np.where(a == 1.0, at_one, at_zero), arms={1.0: at_one, 0.0: at_zero}
        ),
        scaler=OutcomeScaler(0.0, 1.0),
        folds=Folds.single(len(frame)),
    )
    reduced, _, _, _ = fit_reduced(
        data,
        nuisance,
        regression_learner=CellMeans(),
        classification_learner=CellMeans(),
        mechanism_residual_learner=CellMeans(),
        g_bounds=INERT,
        basis=None if basis is None else basis_of(w),
    )
    return reduced


def level_set_mean(values: np.ndarray, mass: np.ndarray, design: np.ndarray) -> np.ndarray:
    """``E[values | design]`` per cell, under the cell masses ``mass``."""
    out = np.empty_like(values, dtype=float)
    for level in np.unique(design):
        cells = design == level
        out[cells] = np.sum(mass[cells] * values[cells]) / np.sum(mass[cells])
    return out


def longhand_qr(counts: np.ndarray, g_hat: np.ndarray, q_hat: np.ndarray, arm: int) -> np.ndarray:
    """``Q_{r,j}(a, w)`` per cell ``(3, p)``, from the counts alone."""
    probs = counts / counts.sum()
    took = probs[:, arm, :].sum(axis=1)
    q0 = probs[:, arm, 1] / took
    residual = q0 - q_hat[:, arm]
    mechanism = g_hat if arm == 1 else 1.0 - g_hat
    phi = basis_of(np.arange(3))
    return np.column_stack(
        [level_set_mean(phi[:, j] * residual, took, mechanism) for j in range(2)]
    )


def longhand_gr(counts: np.ndarray, g_hat: np.ndarray, q_hat: np.ndarray, arm: int) -> tuple:
    """``(g_{r,1}, g_{r,2,j})`` per cell, from the counts alone."""
    probs = counts / counts.sum()
    p_w = probs.sum(axis=(1, 2))
    g0 = probs[:, arm, :].sum(axis=1) / p_w
    mechanism = g_hat if arm == 1 else 1.0 - g_hat
    ratio = (g0 - mechanism) / mechanism
    phi = basis_of(np.arange(3))
    design = q_hat[:, arm]
    gr1 = level_set_mean(g0, p_w, design)
    gr2 = np.column_stack([level_set_mean(phi[:, j] * ratio, p_w, design) for j in range(2)])
    return gr1, gr2


@pytest.fixture(scope="module")
def law_g() -> tuple[np.ndarray, Any, ReducedSet]:
    counts, frame = variant(G_TIED, law.Q)
    return counts, frame, fit_at(frame, G_TIED, Q_WRONG_ARM1, basis=True)


@pytest.fixture(scope="module")
def law_q() -> tuple[np.ndarray, Any, ReducedSet]:
    counts, frame = variant(law.G, Q_TIED)
    return counts, frame, fit_at(frame, G_WRONG, Q_TIED, basis=True)


class TestTheShapes:
    def test_the_weighted_families_carry_a_term_axis(self, law_g: tuple) -> None:
        _, frame, reduced = law_g
        n = len(frame)
        assert reduced.qr.shape == (n, 2, 2)
        assert reduced.gr2.shape == (n, 2, 2)
        assert reduced.gr1.shape == (n, 2)
        assert reduced.n_terms == 2

    def test_the_arm_mean_set_has_no_term_axis(self) -> None:
        _, frame = variant(G_TIED, law.Q)
        assert fit_at(frame, G_TIED, Q_WRONG_ARM1).n_terms is None

    def test_the_constant_term_is_the_arm_mean_set(self, law_q: tuple) -> None:
        _, frame, reduced = law_q
        plain = fit_at(frame, G_WRONG, Q_TIED)
        np.testing.assert_array_equal(reduced.qr[:, :, 0], plain.qr)
        np.testing.assert_array_equal(reduced.gr2[:, :, 0], plain.gr2)
        np.testing.assert_array_equal(reduced.gr1, plain.gr1)


class TestLawLg:
    """``Q_{r,j}`` on the tied propensity: the weight sits inside the regression."""

    @pytest.mark.parametrize("arm", [0, 1])
    def test_qr_is_the_weighted_level_set_mean(self, law_g: tuple, arm: int) -> None:
        counts, frame, reduced = law_g
        w = frame["W"].to_numpy().astype(int)
        expected = longhand_qr(counts, G_TIED, Q_WRONG_ARM1, arm)[w]
        np.testing.assert_allclose(reduced.qr[:, arm, :], expected, rtol=0.0, atol=EXACT)

    def test_the_slope_figures_and_both_mutants(self, law_g: tuple) -> None:
        _, frame, reduced = law_g
        w = frame["W"].to_numpy().astype(int)
        slope = reduced.qr[:, 1, 1]
        cell = {k: float(slope[w == k][0]) for k in range(3)}
        assert cell[0] == pytest.approx(-0.0375, abs=EXACT)
        assert cell[1] == pytest.approx(-0.0375, abs=EXACT)
        assert cell[2] == pytest.approx(-0.2, abs=EXACT)
        intercept = reduced.qr[:, 1, 0]
        # The weight-after mutant phi_j E[D | g] departs on the w = 0 rows.
        weight_after = w * intercept
        assert float(np.max(np.abs(slope - weight_after)[w == 0])) == pytest.approx(0.0375)
        # The drop-weight mutant E[D | g] departs on the w = 2 rows only.
        np.testing.assert_allclose(slope[w <= 1], intercept[w <= 1], atol=EXACT)
        assert float(np.max(np.abs(slope - intercept)[w == 2])) == pytest.approx(0.1)

    def test_the_right_arm_regression_vanishes(self, law_g: tuple) -> None:
        _, _, reduced = law_g
        np.testing.assert_allclose(reduced.qr[:, 0, :], 0.0, atol=EXACT)


class TestLawLQ:
    """``g_{r,2,j}`` on the tied outcome regression; ``g_{r,1}`` stays unweighted."""

    @pytest.mark.parametrize("arm", [0, 1])
    def test_both_reduced_mechanisms_are_the_longhand_ones(self, law_q: tuple, arm: int) -> None:
        counts, frame, reduced = law_q
        w = frame["W"].to_numpy().astype(int)
        gr1, gr2 = longhand_gr(counts, G_WRONG, Q_TIED, arm)
        np.testing.assert_allclose(reduced.gr1[:, arm], gr1[w], rtol=0.0, atol=EXACT)
        np.testing.assert_allclose(reduced.gr2[:, arm, :], gr2[w], rtol=0.0, atol=EXACT)

    def test_the_tied_witness_and_both_mutants(self, law_q: tuple) -> None:
        _, frame, reduced = law_q
        w = frame["W"].to_numpy().astype(int)
        slope = reduced.gr2[:, 1, 1]
        tied = w <= 1
        # E[w r_1 | tie] = 0.375 x 0.0909 on both tied rows.
        np.testing.assert_allclose(slope[tied], 0.375 * 0.05 / 0.55, atol=EXACT)
        assert float(slope[tied][0]) == pytest.approx(0.0341, abs=5e-5)
        intercept = reduced.gr2[:, 1, 0]
        assert float(intercept[tied][0]) == pytest.approx(-0.0354, abs=5e-5)
        # The weight-after mutant w E[r_1 | tie] is 0 and -0.0354 on the tied rows.
        weight_after = w * intercept
        for k in (0, 1):
            assert abs(float(slope[w == k][0] - weight_after[w == k][0])) > 0.03
        # The drop-weight mutant is -0.0354 on both.
        assert float(np.min(np.abs(slope - intercept)[tied])) > 0.06

    def test_gr1_reads_no_weight(self, law_q: tuple) -> None:
        _, frame, reduced = law_q
        w = frame["W"].to_numpy().astype(int)
        # P(A = 1 | tie) = 0.625 x 0.40 + 0.375 x 0.60, the unweighted pooled share.
        np.testing.assert_allclose(reduced.gr1[w <= 1, 1], 0.475, atol=EXACT)


class TestTheGuards:
    def test_the_bivariate_reduction_takes_no_basis(self) -> None:
        _, frame = variant(G_TIED, law.Q)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])
        reduced = fit_at(frame, G_TIED, Q_WRONG_ARM1)
        del reduced
        with pytest.raises(ValueError, match="univariate"):
            fit_reduced(
                data,
                NuisanceEstimates(
                    propensity=Propensity(np.full((len(frame), 2), 0.5), ARMS),
                    outcome=InitialFit(
                        observed=np.full(len(frame), 0.5),
                        arms={1.0: np.full(len(frame), 0.5), 0.0: np.full(len(frame), 0.5)},
                    ),
                    scaler=OutcomeScaler(0.0, 1.0),
                    folds=Folds.single(len(frame)),
                ),
                regression_learner=CellMeans(),
                classification_learner=CellMeans(),
                mechanism_residual_learner=CellMeans(),
                g_bounds=INERT,
                reduction="bivariate",
                basis=basis_of(frame["W"].to_numpy()),
            )

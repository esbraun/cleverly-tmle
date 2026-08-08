r"""The doubly-robust curve's decomposition, against a perturbation of the law.

Every other ``test_influence_gateaux*`` module compares what the library reports against
the Gateaux derivative of a longhand functional, and **none of them can be reused for
this estimator**.  At correct nuisances :math:`Q_r` and :math:`g_{r,2}` vanish row by row,
so the corrected curve *equals* the plain one array for array and every one of those
modules passes against a flipped sign, against a sum where a difference belongs, and
against swapped reduced mechanisms.  That is lesson 2 of
the validation programme, and it is why the object table that tracked evidence for each
theorem object carried ``TODO`` against :math:`D^{*,\#}` while thirteen sibling modules
pinned everything else.

This module is that row.  It runs on nuisances that are **wrong on purpose**, and it is
the only place here where the comparison is against a *derivative* rather than against a
longhand copy of the same formula.

**The construction, and why it has teeth.**  Theorem 1 assumes one nuisance is
consistent, so the check belongs in the union model rather than at the both-wrong
nuisances :mod:`tests.unit.test_remainder_drtmle` measures the expansion at.  There are
two such cells and each isolates one correction, because the other's reduced regression is
identically zero:

.. table::

    ============  ==================  =======  ================  =====================
    cell          misspecified        guard    live correction   vanishing
    ============  ==================  =======  ================  =====================
    ``g_right``   :math:`\bar Q`      ``Q``    :math:`D^*_g`     :math:`g_{r,2} = 0`
    ``q_right``   :math:`g`           ``g``    :math:`D^*_Q`     :math:`Q_r = 0`
    ============  ==================  =======  ================  =====================

In each, with the reductions **saturated**, the corrected curve collapses to the
*efficient* influence function at the true nuisances -- which is exactly what "an interval
valid when only one nuisance is consistent" means, and which
:func:`tests.discrete_law.eif` supplies by complex-step differentiation of the
identification formula, with no library code anywhere in the derivation.  The algebra, for
``q_right``, is one line and is the whole reason this comparison is not a tautology:

.. math::

    \frac{1}{g_1} - \frac{g_{r,2}}{g_{r,1}}
      = \frac{1}{g_1} - \frac{g_0 - g_1}{g_1 g_0}
      = \frac{1}{g_0},

and for ``g_right`` the :math:`\bar Q^*` in :math:`D^*` and the one inside
:math:`Q_r = \bar Q_0 - \bar Q^*` cancel, leaving :math:`1_a/g_0\,(Y - \bar Q_0)`.  A
flipped sign leaves twice the correction behind -- ``0.55`` to ``2.8`` here against a
``1e-12`` window.

**Targeting is load-bearing in one cell and a no-op in the other**, which is worth stating
because it decides what each cell is evidence of.  Both collapses hold at *any* state --
the correction absorbs whatever :math:`\bar Q^*` is -- *except* for the centring, and what
equation (8) buys is :math:`E_P[\bar Q^*(a, \cdot)] = \Psi_0`.  In ``g_right`` the initial
:math:`\bar Q` is wrong, so without the fluctuation the corrected curve is the efficient
one shifted by a **constant**: measured at ``-0.110`` at arm 0 and ``+0.125`` at arm 1,
flat to ``6e-16`` across the support.  In ``q_right`` the initial :math:`\bar Q` is already
:math:`\bar Q_0`, so equation (8) is solved at :math:`\epsilon = 0` and the shift is
exactly zero -- that cell says nothing about the targeting step and everything about the
reduced mechanisms.  Between them the comparison pins the plug-in and the curve
*together*, which is the pairing an uncentred curve is a failure of.

**Four things this module cannot see**, each recorded rather than left for someone to
assume otherwise, and the first two measured by running the mutation and watching it
*pass*.  They are one degeneracy wearing four hats: a cell is blind to every mutation of a
term it sets to zero.

* **Partial guards** -- whether a fit subtracts a correction whose equation it never solved. In
  each cell the *unguarded* correction is identically zero, so subtracting it changes
  nothing.  ``tests/unit/test_influence_drtmle.py`` covers it, at nuisances where both
  are wrong;
* **equation (9)'s covariate sign.**  Crossing the arm signs in
  :func:`~cleverly.fluctuation.reduced.reduced_mechanism_covariate` -- "the whole content
  of this function", by its own docstring -- leaves all of this green, because the only
  cell where :math:`D^*_g` is alive is the one whose mechanism is already right, so the
  tilt is zero whatever direction it is offered.  ``tests/unit/test_reduced_submodel.py``
  ``::TestTheMechanismCovariate`` covers it, and was watched to fail against exactly that
  mutation;
* **the pooling weight of a reduced regression.**  The reductions here are *saturated*, so
  every conditioning cell is a singleton and the weight cancels.
  ``tests/unit/test_reduced_regressions.py`` and ``test_remainder_drtmle.py``'s ``TIED_G``
  / ``TIED_Q`` are where a genuine pooling is exercised;
* **The cross-fitting construction.** Same cause: at a singleton cell the pooled
  construction and a nested one return the same arrays, so nothing here can separate them.
  fixture preconditions ensure that this module's agreement is not later
  read as evidence about fold reuse -- which is the shape of the mistake the R-parity piece
  was retired for.

What it *does* see, watched to fail in the library itself: the corrections' sign under
either guard, and :math:`g_{r,1}` and :math:`g_{r,2}` exchanged in
:func:`~cleverly.estimators.reduced.fit_reduced`.

Two tiers.  The first is this module's own longhand against the derivative; the second is
what a real ``DRTMLE`` fit reports, and it also closes a gap of its own -- before it, no
test in this repository fitted the estimator against a deliberately misspecified law at
all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly import DRTMLE
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.discrete_law_longitudinal import CellMeans

#: The estimands ``DRTMLE`` reports.  ``att`` and ``atc`` are refused by name, which is
#: why this module's list is shorter than :mod:`tests.unit.test_influence_gateaux`'s.
ESTIMANDS = ("ey1", "ey0", "ate")

ARMS = (0, 1)

#: Copied verbatim from :mod:`tests.unit.test_remainder_drtmle`, as every sibling module
#: copies it: a mechanism wrong at every covariate cell, comfortably interior.
WRONG_G = np.array([0.55, 0.35, 0.45])

#: **Not** that module's ``WRONG_Q``, and the departure is measured rather than stylistic.
#: ``law.Q + [[.10, -.15], [-.20, .10], [.05, .20]]`` reaches ``0.0`` at ``(w=1, a=0)`` and
#: ``1.0`` at ``(w=2, a=1)``, and a logistic fluctuation cannot move a regression pinned at
#: either -- ``logit`` is infinite there.  No module before this one noticed, because none
#: of them targets: :mod:`tests.unit.test_remainder_drtmle` and
#: :mod:`tests.unit.test_theorem_drtmle` both evaluate the expansion at the *initial*
#: nuisances.  The estimator itself is unaffected, since it clips before taking a logit;
#: what breaks is the longhand oracle below, which must not borrow the estimator's clip.
#:
#: Chosen so that neither arm's offset is orthogonal to ``P_W``: the plug-in at these
#: values misses the truth by ``0.11`` and ``-0.125``, which is what makes the targeting
#: step visible in the comparison rather than a no-op.
WRONG_Q = law.Q + np.array([[0.15, -0.25], [0.25, 0.20], [-0.20, -0.30]])

#: A truncation that never binds, so nothing here is measuring the bound.  Item 20's
#: clipping discrepancy is ``tests/unit/test_drtmle_fit.py``'s subject, not this module's.
INERT = (1e-6, 1.0 - 1e-6)

#: Newton steps for equation (8) in the longhand tier, run to a fixed count.  Forty is far
#: past convergence for a scalar logistic score; :class:`TestThePremisesHold` asserts the
#: solve landed rather than trusting the number.
NEWTON_STEPS = 40


class Misspecified:
    """A ``DGP``-shaped object returning declared constants rather than a law's own.

    :class:`tests.discrete_law.DiscreteLaw` supplies the conditionals of *some* law, which
    is what an oracle for a weighted fit needs; a misspecified nuisance is not a law's
    conditionals at all, so it gets its own shim.  Only ``propensity`` and
    ``outcome_mean`` are read, by :class:`~tests.conftest.OracleTreatment` and
    :class:`~tests.conftest.OracleOutcome`.
    """

    def __init__(self, g: np.ndarray, q: np.ndarray) -> None:
        self.g = np.asarray(g, dtype=float)
        self.q = np.asarray(q, dtype=float)

    @staticmethod
    def _index(covariates: Any) -> np.ndarray:
        return np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)

    def propensity(self, covariates: Any) -> np.ndarray:
        return self.g[self._index(covariates)]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: float | None) -> np.ndarray:
        return self.q[self._index(covariates), int(arm)]


#: The two off-diagonal cells, as ``(mechanism, outcome regression, guard)``.  The correct
#: nuisance is the law's own -- read off the *counts* through
#: :data:`tests.discrete_law.G_EXACT` and :data:`~tests.discrete_law.Q_EXACT`, so it is
#: exact in the sample down to the last bit rather than merely equal mathematically.
CELLS: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {
    "g_right": (law.G_EXACT, WRONG_Q, ("Q",)),
    "q_right": (WRONG_G, law.Q_EXACT, ("g",)),
}

#: The same two cells at the *truth*, where both corrections vanish row by row and every
#: assertion below passes without saying anything.  :class:`TestTheControlsBite` runs the
#: whole comparison here, which is the statement that the misspecification is what gives
#: this module its teeth -- lesson 2, written as a test rather than as a docstring.
DEGENERATE: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {
    "at the truth, Q guard": (law.G_EXACT, law.Q_EXACT, ("Q",)),
    "at the truth, g guard": (law.G_EXACT, law.Q_EXACT, ("g",)),
}


# ----------------------------------------------------------------- the longhand oracle


def _realised() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(W, A, Y)`` of the realised sample, as arrays."""
    frame = law.frame()
    return (
        frame["W"].to_numpy().astype(int),
        frame["A"].to_numpy(dtype=float),
        frame["Y"].to_numpy(dtype=float),
    )


def _saturated_reductions(g_hat: np.ndarray, q_hat: np.ndarray, arm: int) -> tuple[np.ndarray, ...]:
    r"""``(Qr, gr1, gr2)`` per covariate cell, at the saturated partition.

    .. math::

        Q_r(a, w)      &= E\bigl[Y - \hat{\bar Q}(A, W) \mid A = a, \hat g(a|W)\bigr] \\
        g_{r,1}(a|w)   &= P\bigl[A = a \mid \hat{\bar Q}(a, W)\bigr] \\
        g_{r,2}(a|w)   &= E\Bigl[\frac{1_a - \hat g(a|W)}{\hat g(a|W)}
                                 \,\Big|\, \hat{\bar Q}(a, W)\Bigr]

    Both designs take three distinct values here -- :class:`TestThePremisesHold` asserts
    it -- so each conditioning sigma-algebra is all of :math:`\sigma(W)` and the pooling
    collapses to the per-cell value.  **That is a requirement rather than a convenience.**
    A design that *tied* two cells would pool them at the base law and stop pooling them
    under any perturbation that broke the tie, so the functional the derivative is taken
    of would not be differentiable there; and the collapse to the efficient curve is
    exactly what a non-saturated reduction gives up, which is the difference Theorem 1's
    remainder carries.  :mod:`tests.unit.test_remainder_drtmle` is where a *pooling*
    reduction is exercised, with ``TIED_G`` and ``TIED_Q``.
    """
    truth = law.G_EXACT if arm == 1 else 1.0 - law.G_EXACT
    mechanism = g_hat if arm == 1 else 1.0 - g_hat
    qr = law.Q_EXACT[:, arm] - q_hat[:, arm]
    return qr, truth, (truth - mechanism) / mechanism


def _targeted(q_hat: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    r"""Equation (8)'s fluctuation of :math:`\hat{\bar Q}`, per arm, written out here.

    :math:`\bar Q^*(a, w) = \operatorname{expit}\{\operatorname{logit}\hat{\bar Q}(a, w)
    + \epsilon_a/\hat g(a|w)\}` with :math:`\epsilon_a` solving

    .. math::

        \sum_w P(W = w, A = a)\,\frac{1}{\hat g(a|w)}
               \bigl\{\bar Q_0(a, w) - \bar Q^*(a, w)\bigr\} = 0,

    which is the population form of the score the ``mean`` submodel zeroes: the covariate
    carries the indicator, so the sum runs over the rows that took ``a``, and the outcome
    residual averages to :math:`\bar Q_0(a, w)` within each cell because the sample
    realises the law.

    By Newton at a **fixed step count with no convergence test**, as
    :mod:`tests.discrete_law`'s linked working model is solved and for the same reason: a
    tolerance is a comparison, and an oracle that branched on one would stop being one
    expression.  ``expit`` is spelled out rather than imported, which is what keeps this
    module's arithmetic its own.
    """
    out = np.array(q_hat, dtype=float)
    mass = law.PROBS.sum(axis=2)  # P(W = w, A = a)
    for arm in ARMS:
        covariate = 1.0 / (g_hat if arm == 1 else 1.0 - g_hat)
        eta = np.log(q_hat[:, arm] / (1.0 - q_hat[:, arm]))
        epsilon = 0.0
        for _ in range(NEWTON_STEPS):
            star = 1.0 / (1.0 + np.exp(-(eta + epsilon * covariate)))
            score = float(np.sum(mass[:, arm] * covariate * (law.Q_EXACT[:, arm] - star)))
            slope = float(np.sum(mass[:, arm] * covariate**2 * star * (1.0 - star)))
            epsilon += score / slope
        out[:, arm] = 1.0 / (1.0 + np.exp(-(eta + epsilon * covariate)))
    return out


def _plain_curve(q_star: np.ndarray, g_hat: np.ndarray, arm: int) -> np.ndarray:
    r""":math:`D^*(\bar Q^*, \hat g)` for one arm, written out here rather than imported.

    :math:`1_a/\hat g(a|W)\{Y - \bar Q^*(a, W)\} + \bar Q^*(a, W) - \hat\Psi_a`, with
    :math:`\hat\Psi_a` the plug-in of the same array -- so the curve is centred at what
    the fit would report and not at the truth, which is the pairing the targeting step is
    what makes hold.
    """
    covariate, treatment, outcome = _realised()
    mechanism = (g_hat if int(arm) == 1 else 1.0 - g_hat)[covariate]
    prediction = q_star[covariate, int(arm)]
    indicator = (treatment == float(arm)).astype(float)
    return indicator / mechanism * (outcome - prediction) + prediction - float(np.mean(prediction))


def _corrections(q_star: np.ndarray, g_hat: np.ndarray, arm: int) -> tuple[np.ndarray, np.ndarray]:
    r"""``(D*_g, D*_Q)`` at every row of the realised sample, from the longhand reductions.

    .. math::

        D^*_g = \frac{Q_r(a, W)}{\hat g(a|W)}\{1_a - \hat g(a|W)\},
        \qquad
        D^*_Q = 1_a\,\frac{g_{r,2}(a|W)}{g_{r,1}(a|W)}\{Y - \bar Q^*(a, W)\}
    """
    covariate, treatment, outcome = _realised()
    qr, gr1, gr2 = _saturated_reductions(g_hat, q_star, int(arm))
    mechanism = (g_hat if int(arm) == 1 else 1.0 - g_hat)[covariate]
    indicator = (treatment == float(arm)).astype(float)
    return (
        qr[covariate] / mechanism * (indicator - mechanism),
        indicator * (gr2[covariate] / gr1[covariate]) * (outcome - q_star[covariate, int(arm)]),
    )


def longhand_curve(cell: str, name: str, *, sign: float = -1.0, swap: bool = False) -> np.ndarray:
    r""":math:`D^* - D^*_Q - D^*_g` for ``name``, built from this module's arithmetic alone.

    ``sign`` is the negative control for the combination -- ``-1`` is what the appendices
    derive and ``drtmle`` computes, ``+1`` the transcription error no exact-law check can
    see -- and ``swap`` exchanges :math:`g_{r,1}` and :math:`g_{r,2}`, which is the other
    mutation the reduced mechanisms are vulnerable to.

    Both corrections are subtracted whatever the cell's guard is, because in each cell the
    unguarded one is **identically zero** -- see the module docstring: this instrument
    cannot see the partial-guard defect, and saying so costs nothing while pretending otherwise would cost
    the next reader an afternoon.
    """
    lookup = CELLS if cell in CELLS else DEGENERATE
    g_hat, q_hat, _ = lookup[cell]
    q_star = _targeted(q_hat, g_hat)

    def one(arm: int) -> np.ndarray:
        d_g, d_q = _corrections(q_star, g_hat, arm)
        if swap:
            _, gr1, gr2 = _saturated_reductions(g_hat, q_star, arm)
            covariate, treatment, outcome = _realised()
            indicator = (treatment == float(arm)).astype(float)
            d_q = indicator * (gr1[covariate] / gr2[covariate]) * (outcome - q_star[covariate, arm])
        return _plain_curve(q_star, g_hat, arm) + sign * (d_g + d_q)

    if name == "ate":
        return one(1) - one(0)
    return one(1 if name == "ey1" else 0)


# ------------------------------------------------------------------------- the fixtures


def _fit(cell: str) -> Any:
    """A real ``DRTMLE`` on the discrete law, one nuisance oracle and one wrong on purpose.

    ``cross_fit=False`` for :mod:`tests.unit.test_influence_gateaux`'s reason -- neither
    learner learns from the data, so out-of-fold prediction would add fold bookkeeping to
    a fit whose answer is already determined.  :class:`~tests.discrete_law_longitudinal
    .CellMeans` is the saturated learner for the reductions, which on a sample that
    realises the law exactly *is* the conditional expectation
    :func:`_saturated_reductions` writes out.
    """
    g_hat, q_hat, guard = CELLS[cell]
    dgp = Misspecified(g_hat, q_hat)
    estimator = DRTMLE(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        reduced_outcome_learner=CellMeans(),
        reduced_treatment_learner=CellMeans(),
        guard=guard,
        estimands=ESTIMANDS,
        g_bounds=INERT,
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def fits() -> dict[str, Any]:
    """One fit per cell, shared: each class below reads a different part of the same two."""
    return {cell: _fit(cell) for cell in CELLS}


# --------------------------------------------------------------------------- the claims


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("name", ESTIMANDS)
class TestTheCorrectedCurveIsTheGateauxDerivative:
    """The module's central claim, in both tiers.

    The right-hand side is :func:`tests.discrete_law.eif`, the complex-step derivative of
    the identification formula -- an *independent* re-derivation, which
    ``tests/unit/test_oracle_independence.py`` keeps independent.
    """

    def test_the_longhand_curve_is_the_derivative(self, cell: str, name: str) -> None:
        """Tier one: this module's own arithmetic against the derivative.

        ``rtol=0`` as in every sibling module: a relative tolerance would loosen the
        window by whatever the curve's scale happens to be, on the module's central claim.
        """
        reported = longhand_curve(cell, name)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)

    def test_the_library_reports_the_derivative(
        self, fits: dict[str, Any], cell: str, name: str
    ) -> None:
        """Tier two: what a real fit reports, from end to end.

        Before this, **no test in the repository fitted ``DRTMLE`` against a deliberately
        misspecified law at all** -- ``tests/unit/test_drtmle_fit.py`` fits a simulated
        process where :math:`Q_r` is non-zero by learner error rather than by
        construction, and every module that chooses its own wrong nuisances stops short
        of a fit.
        """
        reported = fits[cell].influence_curves[name][law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-12, rtol=0)


class TestThePremisesHold:
    """If one of these failed, the comparison above would be wrong rather than the library."""

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_derivative_has_mean_zero(self, name: str) -> None:
        """An influence function integrates to zero; a numerical derivative that did not
        would say the functional, not the estimator, is what to look at."""
        assert float(np.dot(law.PROBS.reshape(-1), law.eif(name))) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("cell", sorted(CELLS))
    def test_the_reductions_are_saturated(self, cell: str) -> None:
        """Both designs take a distinct value in every covariate cell.

        Which is what makes each conditioning sigma-algebra all of :math:`\\sigma(W)`, and
        what keeps the functional differentiable -- a tie would be broken by the
        perturbation.  Asserted rather than assumed, because it is a property of the
        constants at the top of this module and nothing else enforces it.
        """
        g_hat, q_hat, _ = CELLS[cell]
        assert len(np.unique(np.round(g_hat, 12))) == 3
        for arm in ARMS:
            assert len(np.unique(np.round(q_hat[:, arm], 12))) == 3

    @pytest.mark.parametrize("cell", sorted(CELLS))
    def test_the_misspecified_nuisance_is_materially_wrong(self, cell: str) -> None:
        """One nuisance is off by a tenth or more, and the other is exact in the sample."""
        g_hat, q_hat, _ = CELLS[cell]
        gaps = (
            float(np.max(np.abs(g_hat - law.G_EXACT))),
            float(np.max(np.abs(q_hat - law.Q_EXACT))),
        )
        assert min(gaps) == 0.0
        assert max(gaps) > 0.1

    @pytest.mark.parametrize("cell", sorted(CELLS))
    def test_the_live_correction_is_not_a_zero_array(self, cell: str) -> None:
        """Nothing above is comparing zeros -- the trap this whole variant lives in.

        In each cell exactly one correction is alive and the other is identically zero,
        which is the module docstring's table asserted rather than described.
        """
        g_hat, q_hat, guard = CELLS[cell]
        q_star = _targeted(q_hat, g_hat)
        live, dead = [], []
        for arm in ARMS:
            d_g, d_q = _corrections(q_star, g_hat, arm)
            alive, gone = (d_g, d_q) if guard == ("Q",) else (d_q, d_g)
            live.append(float(np.max(np.abs(alive))))
            dead.append(float(np.max(np.abs(gone))))
        assert min(live) > 0.1
        assert max(dead) == pytest.approx(0.0, abs=1e-14)

    @pytest.mark.parametrize("cell", sorted(CELLS))
    def test_the_targeting_step_solved_its_equation(self, cell: str) -> None:
        """The longhand plug-in is the truth, which is what equation (8) buys."""
        g_hat, q_hat, _ = CELLS[cell]
        covariate = _realised()[0]
        q_star = _targeted(q_hat, g_hat)
        for name, arm in (("ey1", 1), ("ey0", 0)):
            assert float(np.mean(q_star[covariate, arm])) == pytest.approx(
                law.TRUTH[name], abs=1e-12
            )

    def test_the_untargeted_plug_in_misses_in_the_cell_that_targets(self) -> None:
        """The mirror of the last one, so ``_targeted`` cannot be a no-op unnoticed.

        ``g_right`` only, and the next test is why: with the outcome regression already
        correct there is nothing for equation (8) to move, so a cell that asserted this
        everywhere would be asserting the fluctuation does something it must not.
        """
        covariate = _realised()[0]
        for name, arm in (("ey1", 1), ("ey0", 0)):
            assert abs(float(np.mean(WRONG_Q[covariate, arm])) - law.TRUTH[name]) > 0.1

    def test_and_in_the_other_cell_the_fluctuation_is_a_no_op(self) -> None:
        """``q_right`` solves equation (8) at ``epsilon = 0``, so it tests no targeting.

        Stated rather than left to be inferred, because it decides what the cell is
        evidence *of*: ``q_right`` says the reduced mechanisms compose into ``1/g_0`` and
        says nothing about the fluctuation, while ``g_right`` says both.  Asserting the
        equality is the honest form -- the alternative, quietly parametrising the previous
        test over one cell, would leave a reader to guess which.
        """
        g_hat, q_hat, _ = CELLS["q_right"]
        np.testing.assert_allclose(_targeted(q_hat, g_hat), q_hat, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("cell", sorted(CELLS))
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_fit_reports_the_truth(self, fits: dict[str, Any], cell: str, name: str) -> None:
        """Consistency under one correct nuisance, which the curve comparison presumes."""
        assert fits[cell].estimates[name].psi == pytest.approx(law.TRUTH[name], abs=1e-12)

    @pytest.mark.parametrize("cell", sorted(CELLS))
    def test_the_fit_solved_every_equation_it_reports(
        self, fits: dict[str, Any], cell: str
    ) -> None:
        """A fit whose scores were open would make the tier-two comparison meaningless."""
        assert fits[cell].score_verdict.passed


class TestTheControlsBite:
    """What has to go red, and the one thing that has to stay green."""

    @pytest.mark.parametrize("cell", sorted(CELLS))
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_adding_the_corrections_is_caught(self, cell: str, name: str) -> None:
        """The combination is a **difference**; a sum leaves twice the correction behind.

        This is the mutation :mod:`tests.unit.test_remainder_drtmle` records as invisible
        to every exact-law check, matching the sign discrepancy's shape: what moves is the curve and
        hence the variance, never :math:`\\hat\\Psi`.
        """
        wrong = longhand_curve(cell, name, sign=+1.0)[law.first_row_of()]
        assert float(np.max(np.abs(wrong - law.eif(name)))) > 1e-2

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_swapping_the_reduced_mechanisms_is_caught(self, name: str) -> None:
        """:math:`g_{r,1}` and :math:`g_{r,2}` are a probability and a signed residual
        mean, and the R source names them the other way round from the paper -- which is
        the single easiest thing here to transcribe backwards.

        ``q_right`` **only**, and the next test is why rather than an omission.
        """
        wrong = longhand_curve("q_right", name, swap=True)[law.first_row_of()]
        assert float(np.max(np.abs(wrong - law.eif(name)))) > 1e-2

    def test_and_the_other_cell_cannot_see_that_swap_at_all(self) -> None:
        """In ``g_right`` there is nothing to swap: :math:`g_{r,2}` is identically zero.

        So the mutation does not produce a wrong number there, it produces a division by
        zero -- which is a *stronger* statement than "the test passes anyway" and is worth
        one assertion rather than a silently narrower parametrisation.  It is the same
        degeneracy the module docstring records for partial guards, in a second place: a cell
        is blind to every mutation of a term it sets to zero.
        """
        g_hat, q_hat, _ = CELLS["g_right"]
        q_star = _targeted(q_hat, g_hat)
        for arm in ARMS:
            _, _, gr2 = _saturated_reductions(g_hat, q_star, arm)
            np.testing.assert_allclose(gr2, 0.0, atol=1e-15, rtol=0)

    @pytest.mark.parametrize("cell", sorted(DEGENERATE))
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_at_the_truth_the_whole_comparison_is_vacuous(self, cell: str, name: str) -> None:
        """**The control that must not fail**, and the reason this module exists.

        At correct nuisances both corrections are zero row by row, so the corrected curve
        is the plain one and the comparison holds *whatever* the combination's sign is.
        An instrument for this variant that is only ever run here says nothing at all --
        which is what every ``test_influence_gateaux*`` module before this one is, on this
        estimator. The fixture must make the correction nonzero for its sign to be testable.
        """
        for sign in (-1.0, +1.0):
            curve = longhand_curve(cell, name, sign=sign)[law.first_row_of()]
            np.testing.assert_allclose(curve, law.eif(name), atol=1e-12, rtol=0)

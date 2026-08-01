r"""Does solving the extra equations remove the *first-order* part of the remainder?

:mod:`tests.unit.test_remainder` establishes that TMLE's remainder is a **product** of the
two nuisance errors,

.. math::

    R_2 = \int \frac{\hat g - g_0}{\hat g}\,(\bar Q_1 - \bar Q_{0,1})\, dP_0 + \dots ,

which is why one consistent nuisance suffices for *consistency*.  It is also why one
consistent nuisance does **not** suffice for *inference*: with the other factor not
shrinking, :math:`R_2` is first order in the good one's error, :math:`\sqrt n R_2` does not
vanish, and the estimator stops being asymptotically linear.  Doubly robust inference
(van der Laan 2014; Benkeser, Carone, van der Laan & Gilbert 2017) repairs that by
estimating the first-order part of :math:`R_2` with *reduced-dimension* regressions and
solving their score equations too.

Written in this package's notation, with :math:`1_a = 1\{A = a\}`, the reduced regressions
are, relative to a given :math:`\hat{\bar Q}` and :math:`\hat g`,

.. math::

    Q_r(a, w)   &= E[\, Y - \hat{\bar Q}(a, W) \mid A = a,\ \hat g(a|W) = \hat g(a|w) ] \\
    g_{r,1}(a|w) &= P(\, A = a \mid \hat{\bar Q}(a, W) = \hat{\bar Q}(a, w) ) \\
    g_{r,2}(a|w) &= E[\, \{1_a - \hat g(a|W)\}/\hat g(a|W)
                        \mid \hat{\bar Q}(a, W) = \hat{\bar Q}(a, w) ]

and the two extra score equations, in the software paper's numbering, are

.. math::

    P_n\Bigl[\ \frac{Q_r(a, W)}{g^*(a|W)}\,\{1_a - g^*(a|W)\}\ \Bigr] &= 0 \tag{9} \\
    P_n\Bigl[\ 1_a\,\frac{g_{r,2}(a|W)}{g_{r,1}(a|W)}\,\{Y - \bar Q^*(a, W)\}\ \Bigr] &= 0 .

Because both are solved, each of their :math:`P_0` means is minus a centred empirical
process term, and the reported influence curve becomes
:math:`D = D^* - D^*_Q - D^*_g`.  So the object this module evaluates is

.. math::

    R_2^{dr} = R_2 - P_0 D^*_g - P_0 D^*_Q ,

the remainder that is left once the solved equations have been accounted for.

**Why this module and not a Gateaux one.**  Under a law the sample realises exactly with a
saturated learner -- the setting of every ``test_influence_gateaux*`` module -- both
nuisances are exact, so :math:`Q_r` and :math:`g_{r,2}` have identically zero targets and
vanish *row by row*.  Both extra fluctuation coefficients are then zero and the reported
curve equals :math:`D^*` array for array, so those modules supply a degeneracy check and
would pass against a wrong sign, a sum where a difference belongs, or a wrong
:math:`g_{r,1}` -- which is a probability, does not vanish, and sits in a denominator whose
numerator does.  The remainder idiom evaluates the expansion at nuisances that are **wrong
on purpose**, where all of that is visible and every term is an exact finite sum.

**One guard removes the whole first-order remainder; two over-correct.**  That is the
finding this module exists to state, and it is not a defect.  Each extra equation subtracts a
*projection* of :math:`R_2` -- equation (9) onto :math:`\sigma(\hat g)`, the other onto
:math:`\sigma(\hat{\bar Q})`.  On a three-cell law with distinct nuisance values both
:math:`\sigma`-algebras are all of :math:`\sigma(W)`, so **either** projection recovers the
whole of :math:`R_2` and subtracting both leaves :math:`-R_2`.  Asymptotically that is
harmless, because at most one of the two errors fails to vanish and so at most one projection
is non-negligible; here it is exact arithmetic, and asserting that "the remainder vanishes
when both guards are on" would be asserting something false.  The tie constants below are
what make the single-guard claim non-trivial: with :math:`\hat g` tied across two cells,
:math:`\sigma(\hat g)` is genuinely coarser than :math:`\sigma(W)` and the g-guard no longer
removes everything.

Two things this deliberately does *not* rest on.  Nothing here runs a targeting step: the
remainder is a property of the estimating equations, evaluated at nuisances the library did
not fit.  And :math:`D^*` comes from the library while :math:`R_2^{dr}`'s closed form and the
two extra terms are written out longhand here, so a shared derivation cannot make the
comparison vacuous.

Scope: the ``mean`` submodel at two arms, which is what the variant is derived for.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from tests import discrete_law as law
from tests.conftest import binary_means

#: Copied verbatim from :mod:`tests.unit.test_remainder`, as every sibling remainder module
#: copies them, so the modules disagree about nothing except which expansion they take.
WRONG_G = np.array([0.55, 0.35, 0.45])
WRONG_Q = law.Q + np.array([[0.10, -0.15], [-0.20, 0.10], [0.05, 0.20]])

#: A wrong mechanism whose *values* tie across two covariate cells, and a wrong outcome
#: regression that does the same.  Load-bearing, and the one thing about this module that is
#: not shared with its siblings: a reduced regression conditions on the *value* of the other
#: nuisance, so with three distinct values it conditions on ``W`` itself and is a relabelling
#: rather than a pooling.  A tie is what makes it a genuine reduction, and it is what
#: separates an implementation that conditions on the estimated nuisance from one that
#: quietly conditions on the covariate.
#: A shared *offset* would preserve the difference and tie nothing, which was this constant's
#: first form and is why the mirror test below failed before the offsets were made to meet.
TIED_G = np.array([0.55, 0.35, 0.55])
TIED_Q = law.Q + np.array([[-0.10, -0.10], [0.10, 0.10], [0.05, 0.15]])

#: ``"Q"`` guards against a misspecified outcome regression and adds equation (9), which
#: fluctuates **g**; ``"g"`` guards against a misspecified mechanism and adds the other
#: equation, which fluctuates ``Qbar``.  The keyword and the nuisance it moves are crossed,
#: which is ``drtmle``'s own vocabulary and the commonest way to transcribe these backwards.
BOTH = ("Q", "g")

ESTIMANDS = ("ey1", "ey0", "ate")


def _cells(values: np.ndarray) -> np.ndarray:
    """Which covariate cells share a value -- the partition a reduced regression pools over."""
    _, inverse = np.unique(np.round(values, 12), return_inverse=True)
    return inverse


def _reduced(
    g_hat: np.ndarray, q_hat: np.ndarray, arm: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``(Qr, gr1, gr2)`` per covariate cell, written out longhand at the true law.

    Each is a conditional expectation over the cells that *share a value* of the nuisance
    being conditioned on, weighted by the law.  ``Qr`` conditions on ``A = arm`` as well, so
    its weights carry the mechanism; the two reduced mechanisms do not, so theirs do not.
    """
    mechanism = g_hat if arm == 1 else 1.0 - g_hat
    truth = law.G if arm == 1 else 1.0 - law.G
    residual = law.Q[:, arm] - q_hat[:, arm]

    qr = np.zeros(3)
    for cell in set(_cells(mechanism)):
        rows = _cells(mechanism) == cell
        weight = law.P_W[rows] * truth[rows]
        qr[rows] = np.sum(weight * residual[rows]) / np.sum(weight)

    gr1, gr2 = np.zeros(3), np.zeros(3)
    for cell in set(_cells(q_hat[:, arm])):
        rows = _cells(q_hat[:, arm]) == cell
        mass = np.sum(law.P_W[rows])
        gr1[rows] = np.sum(law.P_W[rows] * truth[rows]) / mass
        gr2[rows] = np.sum(law.P_W[rows] * (truth[rows] - mechanism[rows]) / mechanism[rows]) / mass
    return qr, gr1, gr2


#: A direction to push the reduced regressions away from their true values in, so that the
#: remainder can be measured against a *reduced* error rather than a primary one.  Not
#: constant, since a constant shift of ``Qr`` is removed by the mechanism's own centring and
#: would make the perturbation invisible.
QR_DIRECTION = np.array([0.30, -0.20, 0.10])
GR2_DIRECTION = np.array([0.20, 0.10, -0.30])


def _extra_curves(
    g_hat: np.ndarray, q_hat: np.ndarray, arm: int, *, reduced_bias: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""``(D*_g, D*_Q)`` at every row of the realised sample, built by hand.

    .. math::

        D^*_g = \frac{Q_r(a, W)}{\hat g(a|W)}\,\{1_a - \hat g(a|W)\},
        \qquad
        D^*_Q = 1_a\,\frac{g_{r,2}(a|W)}{g_{r,1}(a|W)}\,\{Y - \hat{\bar Q}(a, W)\}
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)

    qr, gr1, gr2 = _reduced(g_hat, q_hat, arm)
    qr = qr + reduced_bias * QR_DIRECTION
    gr2 = gr2 + reduced_bias * GR2_DIRECTION
    mechanism = (g_hat if arm == 1 else 1.0 - g_hat)[covariate]
    indicator = (treatment == float(arm)).astype(float)

    d_g = qr[covariate] / mechanism * (indicator - mechanism)
    d_q = indicator * (gr2[covariate] / gr1[covariate]) * (outcome - q_hat[covariate, arm])
    return d_g, d_q


def _expansion(
    g_hat: np.ndarray,
    q_hat: np.ndarray,
    *,
    guard: tuple[str, ...] = BOTH,
    sign: float = -1.0,
    reduced_bias: float = 0.0,
) -> dict[str, float]:
    r"""``R_2^{dr}`` for ``ey1``, ``ey0`` and ``ate`` at the given nuisance guesses.

    ``D^*`` is the library's, evaluated through :func:`counterfactual_means` at nuisances it
    did not fit and with no targeting step; the two extra terms are built here.  Because the
    sample realises the law exactly, the sample mean of a curve *is* its :math:`P_0` mean.

    ``sign`` is the negative control for the combination: ``-1`` is what ``drtmle`` reports,
    and ``+1`` is the plausible transcription error that no Gateaux check can see.
    """
    frame = law.frame()
    covariate = frame["W"].to_numpy().astype(int)
    treatment = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)

    at_one, at_zero = q_hat[covariate, 1], q_hat[covariate, 0]
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at_one, at_zero),
        arms={1.0: at_one, 0.0: at_zero},
    )
    submodel = submodel_for("mean", treatment, g_hat[covariate])
    psi_one, ic_one, psi_zero, ic_zero = binary_means(outcome, initial, submodel, np.ones(law.N))

    remainders = {}
    for name, arm, psi, curve in (("ey1", 1, psi_one, ic_one), ("ey0", 0, psi_zero, ic_zero)):
        d_g, d_q = _extra_curves(g_hat, q_hat, arm, reduced_bias=reduced_bias)
        extra = np.zeros(law.N)
        if "Q" in guard:  # equation (9) -- the one that fluctuates g
            extra = extra + sign * d_g
        if "g" in guard:
            extra = extra + sign * d_q
        remainders[name] = psi - law.TRUTH[name] + float(np.mean(np.asarray(curve) + extra))
    remainders["ate"] = remainders["ey1"] - remainders["ey0"]
    return remainders


def _product_form(
    g_hat: np.ndarray, q_hat: np.ndarray, *, guard: tuple[str, ...] = BOTH
) -> dict[str, float]:
    r"""``R_2^{dr}`` as theory says it must be, as an exact finite sum over the cells.

    The TMLE product form minus, for each guard that is on, the projection that guard's
    equation removes: equation (9) subtracts :math:`\int u\,E[v \mid \hat g]`, the other
    subtracts :math:`\int v\,g_0\,E[u \mid \hat{\bar Q}]/E[g_0 \mid \hat{\bar Q}]`.  Where
    the conditioning :math:`\sigma`-algebra is all of :math:`\sigma(W)` a projection is the
    identity, which is the whole of why two guards over-correct here.
    """
    remainders = {}
    for name, arm in (("ey1", 1), ("ey0", 0)):
        mechanism = g_hat if arm == 1 else 1.0 - g_hat
        truth = law.G if arm == 1 else 1.0 - law.G
        u = (mechanism - truth) / mechanism
        v = q_hat[:, arm] - law.Q[:, arm]

        qr, gr1, gr2 = _reduced(g_hat, q_hat, arm)
        value = float(np.sum(law.P_W * u * v))
        if "Q" in guard:
            value -= float(np.sum(law.P_W * qr * (truth - mechanism) / mechanism))
        if "g" in guard:
            value -= float(np.sum(law.P_W * (gr2 / gr1) * truth * (-v)))
        remainders[name] = value
    remainders["ate"] = remainders["ey1"] - remainders["ey0"]
    return remainders


class TestTheClosedFormIsTheOneTheLibraryProduces:
    """The expansion built from the library's ``D*`` matches the longhand sum, term for term."""

    @pytest.mark.parametrize("guard", [(), ("Q",), ("g",), BOTH])
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_matches_the_closed_form(self, guard: tuple[str, ...], name: str) -> None:
        actual = _expansion(WRONG_G, WRONG_Q, guard=guard)[name]
        expected = _product_form(WRONG_G, WRONG_Q, guard=guard)[name]
        assert actual == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_an_empty_guard_is_a_plain_tmle(self, name: str) -> None:
        """``guard=()`` must reproduce the product form ``test_remainder`` already pins."""
        g_error = WRONG_G - law.G
        one = float(np.sum(law.P_W * g_error / WRONG_G * (WRONG_Q[:, 1] - law.Q[:, 1])))
        zero = float(-np.sum(law.P_W * g_error / (1.0 - WRONG_G) * (WRONG_Q[:, 0] - law.Q[:, 0])))
        plain = {"ey1": one, "ey0": zero, "ate": one - zero}
        assert _expansion(WRONG_G, WRONG_Q, guard=())[name] == pytest.approx(plain[name], abs=1e-12)
        assert abs(plain[name]) > 1e-3, "the misspecification is too mild to test anything"


class TestOneGuardRemovesTheFirstOrderRemainder:
    """The mechanism the theorem rests on, stated as exact arithmetic.

    Each extra equation subtracts the projection of the remainder onto the
    :math:`\\sigma`-algebra its reduced regression conditions on.  Where that is all of
    :math:`\\sigma(W)` -- which is the case on this law whenever the conditioning nuisance
    takes three distinct values -- the projection is the identity and the guard removes the
    remainder outright.  That is what "doubly robust inference" buys, in the one setting
    where it can be checked to machine precision rather than simulated.
    """

    @pytest.mark.parametrize("guard", [("Q",), ("g",)])
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_a_single_guard_leaves_nothing(self, guard: tuple[str, ...], name: str) -> None:
        assert _expansion(WRONG_G, WRONG_Q, guard=guard)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_and_the_unguarded_remainder_is_not_already_zero(self, name: str) -> None:
        """The negative control: without a guard there is something to remove."""
        assert abs(_expansion(WRONG_G, WRONG_Q, guard=())[name]) > 1e-3

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_a_coarser_reduction_no_longer_removes_everything(self, name: str) -> None:
        """With ``g-hat`` tied across two cells the g-guard's projection is a real pooling.

        This is the case that distinguishes a reduced regression conditioning on the
        *estimated mechanism* from one conditioning on the covariate: the two agree
        everywhere else on this law.
        """
        assert abs(_expansion(TIED_G, WRONG_Q, guard=("Q",))[name]) > 1e-3

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_other_guard_still_does_on_the_same_nuisances(self, name: str) -> None:
        """``Qbar-hat`` is untied there, so the Q-projection is still saturated.

        Which is what says the previous test measures the tie rather than the tied law.
        """
        assert _expansion(TIED_G, WRONG_Q, guard=("g",))[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_tie_is_load_bearing(self, name: str) -> None:
        """Untied, the g-guard removes everything -- so the tie is what the test measures."""
        assert _expansion(WRONG_G, WRONG_Q, guard=("Q",))[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_same_holds_the_other_way_round(self, name: str) -> None:
        """A tie in ``Qbar-hat`` coarsens the *other* guard, and only the other one.

        The mirror of the two tests above, and what says the pattern is about which nuisance
        a reduction conditions on rather than about one of the two constants.
        """
        assert abs(_expansion(WRONG_G, TIED_Q, guard=("g",))[name]) > 1e-3
        assert _expansion(WRONG_G, TIED_Q, guard=("Q",))[name] == pytest.approx(0.0, abs=1e-12)


class TestTwoGuardsOverCorrectOnASaturatedLaw:
    """Both equations at once leave ``-R_2``, and that is arithmetic rather than a defect.

    Each guard subtracts a projection, and on a law where both conditioning
    :math:`\\sigma`-algebras are all of :math:`\\sigma(W)` each projection is the whole
    remainder, so the pair subtracts it twice.  Asymptotically at most one of the two nuisance
    errors fails to vanish, so at most one projection is non-negligible and the pair costs
    nothing -- which is why ``drtmle`` solves both by default.  Pinned here so that a later
    reader meeting the exact-law number does not "fix" a combination that is correct.
    """

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_both_guards_leave_minus_the_plain_remainder(self, name: str) -> None:
        plain = _expansion(WRONG_G, WRONG_Q, guard=())[name]
        both = _expansion(WRONG_G, WRONG_Q, guard=BOTH)[name]
        assert both == pytest.approx(-plain, abs=1e-12)
        assert abs(plain) > 1e-3, "the misspecification is too mild to test anything"


class TestTheCombinationIsADifferenceNotASum:
    """The sign no Gateaux module can see.

    At consistent nuisances every extra term vanishes row by row, so ``D* - D*_Q - D*_g`` and
    ``D* + D*_Q + D*_g`` are the same array and every exact-law influence-curve check passes
    against either.  Here they are not.
    """

    @pytest.mark.parametrize("guard", [("Q",), ("g",), BOTH])
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_adding_the_extra_terms_is_not_the_same_as_subtracting(
        self, guard: tuple[str, ...], name: str
    ) -> None:
        minus = _expansion(WRONG_G, WRONG_Q, guard=guard, sign=-1.0)[name]
        plus = _expansion(WRONG_G, WRONG_Q, guard=guard, sign=+1.0)[name]
        assert abs(minus - plus) > 1e-3


class TestTheReducedMechanismsAreLoadBearing:
    """``gr1`` is the term an exact-law check cannot see, because it does not vanish there.

    ``Qr`` and ``gr2`` are identically zero at the truth; ``gr1`` is a probability that is
    not, and it sits in a denominator whose numerator is.  So an implementation that dropped
    it, or that swapped it with ``gr2`` -- the R source names them the other way round from
    the paper -- would pass every degeneracy check there is.
    """

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_dropping_the_denominator_breaks_the_identity(self, name: str) -> None:
        frame = law.frame()
        covariate = frame["W"].to_numpy().astype(int)
        treatment = frame["A"].to_numpy(dtype=float)
        outcome = frame["Y"].to_numpy(dtype=float)

        # The Q-guard's curve with gr1 replaced by one, everything else unchanged.
        broken = {}
        for name_, arm in (("ey1", 1), ("ey0", 0)):
            _, _, gr2 = _reduced(WRONG_G, WRONG_Q, arm)
            indicator = (treatment == float(arm)).astype(float)
            d_q = indicator * gr2[covariate] * (outcome - WRONG_Q[covariate, arm])
            at_one, at_zero = WRONG_Q[covariate, 1], WRONG_Q[covariate, 0]
            initial = InitialFit(
                observed=np.where(treatment == 1.0, at_one, at_zero),
                arms={1.0: at_one, 0.0: at_zero},
            )
            submodel = submodel_for("mean", treatment, WRONG_G[covariate])
            psi1, ic1, psi0, ic0 = binary_means(outcome, initial, submodel, np.ones(law.N))
            psi, curve = (psi1, ic1) if arm == 1 else (psi0, ic0)
            broken[name_] = psi - law.TRUTH[name_] + float(np.mean(np.asarray(curve) - d_q))
        broken["ate"] = broken["ey1"] - broken["ey0"]
        assert abs(broken[name]) > 1e-3

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_true_reduced_mechanism_is_not_degenerate(self, name: str) -> None:
        """``gr1`` is a probability bounded away from zero and one, so the ratio is defined.

        Asserted rather than assumed: if it were degenerate the tests above would be
        comparing zeros.
        """
        del name
        for arm in (0, 1):
            _, gr1, gr2 = _reduced(WRONG_G, WRONG_Q, arm)
            assert np.all(gr1 > 0.2) and np.all(gr1 < 0.8)
            assert np.max(np.abs(gr2)) > 1e-2


class TestTheReducedOutcomeRegressionConditionsOnTheArm:
    """``Qr`` pools by the mechanism's weight, and only a tie can see that it does.

    :math:`Q_r` is :math:`E[\\,Y - \\hat{\\bar Q}(a, W) \\mid A = a, \\hat g\\,]`, so where two
    covariate cells share a value of :math:`\\hat g` they are averaged in proportion to
    :math:`P(W)\\,g_0(a|W)` -- the mass of rows *with that arm* -- and not to :math:`P(W)`
    alone.  Where every cell has its own value each group is a singleton and the weight
    cancels, which is every case above: dropping the mechanism from the weight was applied
    here and **passed all 95 tests** before this class was written.  It is pinned
    structurally, against arithmetic written out rather than against the helper.
    """

    #: The pooled group ``TIED_G`` creates: cells 0 and 2 share the value 0.55.
    POOLED = (0, 2)

    def test_the_pooled_value_is_weighted_by_the_mechanism(self) -> None:
        qr, _, _ = _reduced(TIED_G, WRONG_Q, 1)
        residual = law.Q[:, 1] - WRONG_Q[:, 1]
        rows = np.array(self.POOLED)
        weight = law.P_W[rows] * law.G[rows]
        expected = float(np.sum(weight * residual[rows]) / np.sum(weight))
        assert qr[0] == pytest.approx(expected, abs=1e-15)
        assert qr[2] == pytest.approx(expected, abs=1e-15)

    def test_and_that_is_not_the_unweighted_average(self) -> None:
        """The negative control: the two weightings have to disagree, or nothing is pinned."""
        residual = law.Q[:, 1] - WRONG_Q[:, 1]
        rows = np.array(self.POOLED)
        weighted = float(
            np.sum(law.P_W[rows] * law.G[rows] * residual[rows])
            / np.sum(law.P_W[rows] * law.G[rows])
        )
        unweighted = float(np.sum(law.P_W[rows] * residual[rows]) / np.sum(law.P_W[rows]))
        assert abs(weighted - unweighted) > 1e-3

    def test_the_reduced_mechanisms_are_not_weighted_by_it(self) -> None:
        """``gr1`` and ``gr2`` condition on ``Qbar`` alone, so they pool by ``P(W)``.

        The mirror of the above, and the reason the two cannot share a helper: a weight that
        belongs in one is a bug in the other.
        """
        _, gr1, _ = _reduced(WRONG_G, TIED_Q, 1)
        rows = np.array([0, 1])  # TIED_Q ties cells 0 and 1
        expected = float(np.sum(law.P_W[rows] * law.G[rows]) / np.sum(law.P_W[rows]))
        assert gr1[0] == pytest.approx(expected, abs=1e-15)


class TestTheRemainderIsAProductOfAReducedErrorAndAPrimaryOne:
    """What is left is second order in the *reduced* regressions' error, not in the nuisances'.

    This is the theorem's actual mechanism, and it is why the reduced regressions may be
    estimated at univariate rates however badly the primary nuisances do: what survives a
    guard is a **product** of that guard's reduced error with a primary nuisance error, so
    each factor only has to shrink at :math:`n^{-1/4}`.

    Measuring it needs the reduced regressions perturbed away from their true values, which
    is what :data:`QR_DIRECTION` is for.  Scaling the primary errors alone cannot show it: on
    this law they are removed *exactly* by a saturated reduction, so the remainder is zero at
    every scale and there is no rate to measure -- which is a fact about an exact law rather
    than about the estimator, and the reason the fitted-reduced-regression case belongs to the
    stage that has learners.
    """

    #: Successive halvings, starting at 0.4 rather than 1.  The claim is about the limit, so
    #: the starting point is free -- and it has to be chosen, because the obvious path hits an
    #: accidental tie: ``law.G + t (WRONG_G - law.G)`` makes cells 0 and 1 equal at exactly
    #: ``t = 0.5`` (``0.40 + 0.15t == 0.60 - 0.25t``) and cells 1 and 2 equal at ``t = 7/9``.
    #: At such a ``t`` the reduction is a genuine pooling rather than a relabelling and the
    #: guard stops removing everything, so a path through one measures two things at once.
    #: :meth:`_assert_the_reduction_stays_saturated` fails loudly rather than letting that
    #: pass as a rate.
    SCALES = (0.4, 0.2, 0.1, 0.05, 0.025)

    @staticmethod
    def _worst(values: dict[str, float]) -> float:
        return max(abs(value) for value in values.values())

    @classmethod
    def _assert_the_reduction_stays_saturated(cls, g_hat: np.ndarray, q_hat: np.ndarray) -> None:
        for values, what in (
            (g_hat, "mechanism"),
            (q_hat[:, 1], "Qbar(1, .)"),
            (q_hat[:, 0], "Qbar(0, .)"),
        ):
            assert len(set(np.round(values, 12).tolist())) == 3, (
                f"two covariate cells share a {what} value on this path, so the reduction is a "
                "pooling rather than a relabelling and the rate below would be measuring the "
                "coarsening instead; move SCALES off the tie"
            )

    @classmethod
    def _rates(cls, *, shrink_nuisances: bool) -> list[float]:
        """Ratios of successive remainders as the reduced error -- and maybe both -- halve."""
        ratios: list[float] = []
        previous = None
        for scale in cls.SCALES:
            step = scale if shrink_nuisances else 1.0
            g_hat = law.G + step * (WRONG_G - law.G)
            q_hat = law.Q + step * (WRONG_Q - law.Q)
            cls._assert_the_reduction_stays_saturated(g_hat, q_hat)
            current = cls._worst(_expansion(g_hat, q_hat, guard=("Q",), reduced_bias=scale))
            assert current > 1e-14, "the perturbation is too small to measure a rate"
            if previous is not None:
                ratios.append(previous / current)
            previous = current
        return ratios

    def test_halving_both_errors_quarters_the_remainder(self) -> None:
        rates = self._rates(shrink_nuisances=True)
        assert rates[-1] == pytest.approx(4.0, abs=0.25), rates

    def test_it_is_not_second_order_in_the_reduced_error_alone(self) -> None:
        """Halving only the reduced error halves it -- it is a product, not a square.

        The negative control that says the previous test measures a product rather than a
        reduced regression that happens to enter quadratically.
        """
        rates = self._rates(shrink_nuisances=False)
        assert rates[-1] == pytest.approx(2.0, abs=0.25), rates

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_an_exact_reduced_regression_leaves_nothing_at_any_scale(self, name: str) -> None:
        """And with no reduced error there is no remainder at all, at every scale.

        Which is what makes the perturbation above the thing being measured.
        """
        for scale in self.SCALES:
            g_hat = law.G + scale * (WRONG_G - law.G)
            q_hat = law.Q + scale * (WRONG_Q - law.Q)
            self._assert_the_reduction_stays_saturated(g_hat, q_hat)
            assert _expansion(g_hat, q_hat, guard=("Q",))[name] == pytest.approx(0.0, abs=1e-14)


class TestItVanishesWhenEitherNuisanceIsRight:
    """Double robustness for *consistency* survives the extra terms, under every guard.

    The extra terms are built out of residuals that are zero when the nuisance they are taken
    against is right, so a guard cannot introduce a bias where there was none.
    """

    @pytest.mark.parametrize("guard", [(), ("Q",), ("g",), BOTH])
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_the_propensity_is_right(self, guard: tuple[str, ...], name: str) -> None:
        assert _expansion(law.G, WRONG_Q, guard=guard)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("guard", [(), ("Q",), ("g",), BOTH])
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_the_outcome_regression_is_right(
        self, guard: tuple[str, ...], name: str
    ) -> None:
        assert _expansion(WRONG_G, law.Q, guard=guard)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("guard", [(), ("Q",), ("g",), BOTH])
    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_vanishes_when_both_are_right(self, guard: tuple[str, ...], name: str) -> None:
        assert _expansion(law.G, law.Q, guard=guard)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_reduced_regressions_are_zero_at_the_truth(self, name: str) -> None:
        """Why the above holds, and why no exact-law Gateaux check can see this variant.

        ``Qr`` and ``gr2`` are identically zero when the nuisance they are residuals of is
        right -- row by row, not merely on average -- so the reported curve *equals* ``D*``
        array for array at the truth.
        """
        del name
        for arm in (0, 1):
            qr, _, gr2 = _reduced(law.G, law.Q, arm)
            np.testing.assert_allclose(qr, 0.0, atol=1e-15, rtol=0)
            np.testing.assert_allclose(gr2, 0.0, atol=1e-15, rtol=0)

r"""The doubly-robust influence curve: :math:`D = D^* - D^*_Q - D^*_g`.

The variance is the whole of what this variant buys, so the curve is the one part of it
that can be wrong in a way nothing else here would notice.  Two facts shape how it has to
be checked.

**The point estimate cannot see it.**  All three empirical means are driven to zero by the
targeting, so the combination moves no reported :math:`\hat\Psi` however its signs go.  A
sum where a difference belongs is invisible to every estimate-based assertion in the
package.

**The exact law cannot see it either.**  With a saturated learner :math:`Q_r` and
:math:`g_{r,2}` are identically zero *row by row*, so both extra terms vanish at every
observation and the reported curve is :math:`D^*` to machine precision.  Every
``test_influence_gateaux*`` module would therefore pass against a wrong sign, a wrong
:math:`g_{r,1}`, or an omitted term.

So the instruments here are the same ones :mod:`tests.unit.test_remainder_drtmle` uses --
nuisances that are **wrong on purpose** on the finite-support law, where every term is an
exact finite sum -- and the sign is carried as an explicit negative control.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.targeting import build_submodel
from cleverly.estimators.tmle import correction_parts
from cleverly.fluctuation.mechanism import mechanism_score
from cleverly.fluctuation.reduced import reduced_mechanism_covariate
from cleverly.inference.influence import (
    counterfactual_mean_parts,
    counterfactual_means,
    reduced_correction_parts,
    reduced_corrections,
)
from tests import discrete_law as law
from tests.unit.test_reduced_regressions import ARMS, INERT_BOUNDS, causal_data, nuisances
from tests.unit.test_reduction_alternation import alternate
from tests.unit.test_remainder_drtmle import BOTH, WRONG_G, WRONG_Q, _extra_curves


def solved(g_hat: np.ndarray = WRONG_G, q_hat: np.ndarray = WRONG_Q, *, max_outer: int = 50):
    """``(fluctuation, corrections)`` from a converged alternation at wrong nuisances."""
    fluctuation = alternate(g_hat, q_hat, max_outer=max_outer)
    data = causal_data()
    corrections = reduced_corrections(
        data.outcome,
        fluctuation.targeted,
        data.treatment,
        fluctuation.reduction.reduced,
        fluctuation.mechanism.propensity,
        bounds=INERT_BOUNDS,
        # Off the record the alternation left rather than written in, so this helper
        # follows whatever guard its caller ran under -- which is what the fit does.
        guard=fluctuation.reduction.guard,
    )
    return fluctuation, corrections


class TestTheTermsAreTheOnesTheSourceComputes:
    r""":math:`D^*_g = Q_r/g^*\,(1_a - g^*)` and :math:`D^*_Q = 1_a\,g_{r,2}/g_{r,1}\,r`."""

    def test_each_term_matches_the_longhand(self) -> None:
        """Against ``test_remainder_drtmle``'s own arithmetic, evaluated at the exiting pair.

        That module writes the two curves out per cell from the law; this one reads them
        off the arrays the alternation produced. The comparison is only meaningful because
        the reductions there are the *same* saturated conditional expectations the refit
        produces, which is what the exact law is for.
        """
        fluctuation, corrections = solved()
        reduced = fluctuation.reduction.reduced
        data = causal_data()
        mechanism = fluctuation.mechanism.propensity

        for j, arm in enumerate(ARMS):
            arm_mechanism = mechanism if arm == 1.0 else 1.0 - mechanism
            indicator = (np.asarray(data.treatment, dtype=float) == arm).astype(float)
            d_g = reduced.qr[:, j] / arm_mechanism * (indicator - arm_mechanism)
            d_q = (
                indicator
                * (reduced.gr2[:, j] / reduced.gr1[:, j])
                * (data.outcome - fluctuation.targeted.observed)
            )
            np.testing.assert_allclose(corrections[arm], d_g + d_q, rtol=0, atol=1e-14)

    def test_both_terms_are_materially_non_zero(self) -> None:
        """Or every comparison above would be a comparison of zeros."""
        _, corrections = solved()
        for arm in ARMS:
            assert np.max(np.abs(corrections[arm])) > 1e-2

    def test_the_terms_are_what_the_solved_equations_zero(self) -> None:
        """Their empirical means *are* equations (9) and (10), so the targeting zeroes them.

        Which is also why the subtraction cannot move ``psi``: a curve differing by a
        mean-zero term has the same mean.
        """
        _, corrections = solved()
        for arm in ARMS:
            assert abs(float(np.mean(corrections[arm]))) < 1e-8


class TestOnlyTheGuardedEquationsCorrectionIsInTheCurve:
    """The partial-guard correction invariant at the arrays, with no fit anywhere in it.

    ``guard=`` is crossed: ``"Q"`` adds equation (9), which fluctuates ``g`` and whose
    correction is ``D*_g``; ``"g"`` adds equation (10) and ``D*_Q``.  A fit that solves one
    of them must subtract one term; an earlier implementation subtracted both, so the
    unsolved equation's mean, which nothing had driven anywhere, went into the reported
    curve.

    ``tests/unit/test_remainder_drtmle.py`` states the same rule twice, on the curve side
    (``_expansion``) and on the theory side (``_product_form``), and its
    ``TestOneGuardRemovesTheFirstOrderRemainder`` is what says the rule is the right one.
    What is checked here is that the *library* obeys it, which is the claim nothing made.
    """

    def _terms(self):
        """``d_g`` and ``d_q`` per arm from the longhand module, not from the library."""
        return {arm: _extra_curves(WRONG_G, WRONG_Q, int(arm)) for arm in ARMS}

    def _corrections(self, guard: tuple[str, ...]):
        data = causal_data()
        nuisance = nuisances(WRONG_G, WRONG_Q)
        cell = law.frame()["W"].to_numpy().astype(int)
        return reduced_corrections(
            data.outcome,
            nuisance.outcome,
            data.treatment,
            _reduced_set_from_longhand(),
            WRONG_G[cell],
            bounds=INERT_BOUNDS,
            guard=guard,
        )

    def test_the_fixture_is_not_degenerate(self) -> None:
        """First, because every assertion below is vacuous at the truth.

        At correct nuisances ``Q_r`` and ``g_{r,2}`` vanish row by row, so ``d_g`` and
        ``d_q`` are both zero and *every* guard gives the same array -- lesson 2, in the
        place it is easiest to walk into.  ``WRONG_G``/``WRONG_Q`` are what make the three
        guards distinguishable, and this asserts they do.
        """
        terms = self._terms()
        for arm in ARMS:
            d_g, d_q = terms[arm]
            assert np.max(np.abs(d_g)) > 1e-2
            assert np.max(np.abs(d_q)) > 1e-2
        both, only_q, only_g = (self._corrections(g) for g in (BOTH, ("Q",), ("g",)))
        for arm in ARMS:
            assert not np.allclose(both[arm], only_q[arm])
            assert not np.allclose(both[arm], only_g[arm])
            assert not np.allclose(only_q[arm], only_g[arm])

    def test_the_q_guard_subtracts_the_mechanism_correction_alone(self) -> None:
        corrections = self._corrections(("Q",))
        for arm in ARMS:
            d_g, _ = self._terms()[arm]
            np.testing.assert_allclose(corrections[arm], d_g, rtol=0, atol=1e-14)

    def test_the_g_guard_subtracts_the_outcome_correction_alone(self) -> None:
        corrections = self._corrections(("g",))
        for arm in ARMS:
            _, d_q = self._terms()[arm]
            np.testing.assert_allclose(corrections[arm], d_q, rtol=0, atol=1e-14)

    def test_both_guards_subtract_both_and_are_unchanged(self) -> None:
        """The regression side: the ordinary fit must not have moved by a bit."""
        corrections = self._corrections(BOTH)
        for arm in ARMS:
            d_g, d_q = self._terms()[arm]
            np.testing.assert_allclose(corrections[arm], d_g + d_q, rtol=0, atol=1e-14)

    def test_an_empty_guard_is_refused_rather_than_answered_with_zeros(self) -> None:
        """Such a fit fits no reductions at all and must not reach here.

        Zeros would make ``guard=()`` the plain estimator recovered by a branch, which is
        what ``DRTMLE._nuisances``' short circuit exists to avoid.
        """
        with pytest.raises(ValueError, match="at least one guard"):
            self._corrections(())


class TestTheGuardReachesTheCorrectionsThroughTheProductionPath:
    """The wiring claim: the alternation's own record is what selects, on a real solve.

    :func:`~cleverly.estimators.tmle.correction_parts` is the function both the reported
    curve and :func:`~cleverly.validation.drtmle.correction_check` go through, so passing
    it a fluctuation a partial guard actually produced is what says the two cannot select
    differently.  Item 23 was exactly this call not reading ``reduction.guard`` while the
    check one line later did.
    """

    @pytest.mark.parametrize("guard", [("Q",), ("g",), BOTH])
    def test_the_parts_carry_the_guard_the_alternation_ran_under(
        self, guard: tuple[str, ...]
    ) -> None:
        fluctuation = alternate(WRONG_G, WRONG_Q, guard=guard)
        data = causal_data()
        nuisance = nuisances(WRONG_G, WRONG_Q)
        parts = correction_parts(data, nuisance, fluctuation, fluctuation.targeted, data.outcome)
        assert parts.guard == guard
        for arm in ARMS:
            expected = np.zeros_like(parts.d_g[arm])
            if "Q" in guard:
                expected = expected + parts.d_g[arm]
            if "g" in guard:
                expected = expected + parts.d_q[arm]
            np.testing.assert_allclose(parts.total()[arm], expected, rtol=0, atol=1e-15)

    @pytest.mark.parametrize(
        ("guard", "tilts_mechanism"), [(("Q",), True), (("g",), False), (BOTH, True)]
    )
    def test_solving_an_equation_and_storing_its_score_are_the_same_event(
        self, guard: tuple[str, ...], tilts_mechanism: bool
    ) -> None:
        """The equivalence the whole report rests on, stated once.

        For each equation, three things coincide: the guard names it, the alternation
        stored a score for it, and the curve subtracts its correction.  ``correction_check``
        reads the first and reports the second as ``stored``; if they could come apart, a
        row would be judged against a score belonging to some other state.
        """
        fluctuation = alternate(WRONG_G, WRONG_Q, guard=guard)
        assert (fluctuation.mechanism is not None) is tilts_mechanism
        assert (fluctuation.mechanism is not None) is ("Q" in guard)
        assert (np.asarray(fluctuation.reduction.score).size > 0) is ("g" in guard)


class TestTheCurveIsMeanZeroEvenWhenTheLoopStopsEarly:
    r"""The property the estimator rests on, checked where it is not free.

    A TMLE is asymptotically linear with the curve it reports *because* that curve's
    empirical mean is zero.  For this variant the two extra terms are built from the
    reduced-dimension regressions, and the alternation solves equation (9) at the previous
    round's refit and equation (10) at the current round's *first* refit, then refits once
    more before the record is built -- ``drtmle``'s ordering, kept.  So neither extra
    equation is solved at the arrays the curve is built from, and without
    ``_close_at_frozen_reductions`` the curve's mean is zero only insofar as the loop
    converged.

    **Every other mean-zero assertion in the package runs on a converged fit**, where the
    gap is ``1e-10`` whether or not it is closed -- so none of them can see this.  Stopping
    the refitting rounds after one is what makes the difference visible: measured on a
    800-row fit, ``3.7e-3`` against a standard error of ``0.105`` before the closing pass
    and ``5.8e-7`` after, and ``score_check`` went from FAIL to PASS.
    """

    @pytest.mark.parametrize("rounds", [1, 2, 5])
    def test_it_holds_after_a_single_refitting_round(self, rounds: int) -> None:
        fluctuation, corrections = solved(max_outer=rounds)

        assert fluctuation.reduction.n_outer <= rounds, "the cap must actually bite"
        assert fluctuation.reduction.closing > 0, "the closing pass must have run"
        for arm in ARMS:
            assert abs(float(np.mean(corrections[arm]))) < 1e-8

    def test_the_reported_scores_are_the_curve_s_mean(self) -> None:
        """Not merely both small -- the same number, which is what makes the check honest.

        The curve's mean per arm *is* equation (8)'s score minus equations (10)'s and (9)'s,
        so a diagnostic reporting one and a curve built from the other would agree only by
        accident. This is the invariant the earlier defect broke.
        """
        fluctuation, corrections = solved(max_outer=1)
        reduction = fluctuation.reduction

        for j, arm in enumerate(ARMS):
            combined = (
                float(fluctuation.score[j])
                - float(reduction.score[j])
                - float(fluctuation.mechanism.score[j])
            )
            np.testing.assert_allclose(
                float(np.mean(corrections[arm])),
                float(reduction.score[j]) + float(fluctuation.mechanism.score[j]),
                rtol=0,
                atol=1e-14,
            )
            assert abs(combined) < 1e-8


class TestTheCombinationIsADifferenceNotASum:
    """The one thing settled by reading ``drtmle`` rather than the paper, and invisible elsewhere."""

    def test_the_reported_curve_subtracts_them(self) -> None:
        fluctuation, corrections = solved()
        data = causal_data()
        submodel_means = counterfactual_means(
            data.outcome,
            fluctuation.targeted,
            _mean_submodel(fluctuation),
            data.weights,
            data.observed,
        )
        with_guard = counterfactual_means(
            data.outcome,
            fluctuation.targeted,
            _mean_submodel(fluctuation),
            data.weights,
            data.observed,
            corrections,
        )

        for arm in ARMS:
            plain = np.asarray(submodel_means[arm].influence_curve)
            reported = np.asarray(with_guard[arm].influence_curve)
            np.testing.assert_allclose(reported, plain - corrections[arm], rtol=0, atol=1e-14)
            # The negative control: a sum is the plausible transcription error, and it is
            # a different array by a wide margin.
            assert np.max(np.abs(reported - (plain + corrections[arm]))) > 1e-2

    def test_it_does_not_move_the_estimate(self) -> None:
        fluctuation, corrections = solved()
        data = causal_data()
        submodel = _mean_submodel(fluctuation)
        without = counterfactual_means(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed
        )
        with_guard = counterfactual_means(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed, corrections
        )

        for arm in ARMS:
            assert without[arm].psi == with_guard[arm].psi

    def test_the_variance_it_reports_is_a_different_number(self) -> None:
        fluctuation, corrections = solved()
        data = causal_data()
        submodel = _mean_submodel(fluctuation)
        without = counterfactual_means(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed
        )
        with_guard = counterfactual_means(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed, corrections
        )

        moved = [
            abs(
                float(np.var(with_guard[arm].influence_curve))
                / float(np.var(without[arm].influence_curve))
                - 1.0
            )
            for arm in ARMS
        ]
        assert max(moved) > 1e-3, "the variance is the whole of what the extra terms are for"


class TestAtTheTruthItIsThePlainCurve:
    """Row by row, which is what makes every exact-law instrument blind to this."""

    def test_the_corrections_vanish_at_every_observation(self) -> None:
        _, corrections = solved(law.G, law.Q)
        for arm in ARMS:
            np.testing.assert_allclose(corrections[arm], 0.0, atol=1e-14)

    def test_and_the_reported_curve_equals_the_plain_one(self) -> None:
        """To machine precision rather than bit for bit, and the difference is the point.

        The reductions are zero to floating point and not *exactly* zero -- a saturated
        regression of a residual that sums to zero still returns something of order 1e-17 --
        so subtracting them moves the last bit. Anything asserting bit-for-bit equality here
        would be asserting an arithmetic accident.
        """
        fluctuation, corrections = solved(law.G, law.Q)
        data = causal_data()
        submodel = _mean_submodel(fluctuation)
        without = counterfactual_means(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed
        )
        with_guard = counterfactual_means(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed, corrections
        )
        for arm in ARMS:
            np.testing.assert_allclose(
                with_guard[arm].influence_curve,
                without[arm].influence_curve,
                rtol=0,
                atol=1e-13,
            )


class TestTheDecompositionKeepsUp:
    """``counterfactual_mean_parts`` must not disagree with the curve it decomposes.

    It is a diagnostic and nothing reads it in the estimation path, which is exactly why a
    silent drift here would survive: no existing test compares it on a fit that has a third
    term.
    """

    def test_the_parts_sum_to_the_curve(self) -> None:
        fluctuation, corrections = solved()
        data = causal_data()
        submodel = _mean_submodel(fluctuation)
        means = counterfactual_means(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed, corrections
        )
        parts = counterfactual_mean_parts(
            data.outcome, fluctuation.targeted, submodel, data.weights, data.observed, corrections
        )

        for arm in ARMS:
            np.testing.assert_allclose(
                parts[arm].total, means[arm].influence_curve, rtol=0, atol=1e-12
            )
            assert set(parts[arm].shares()) == {"residual", "plugin", "guard"}
            assert parts[arm].shares()["guard"] > 0.0

    def test_a_plain_fit_reports_two_parts_as_it_always_did(self) -> None:
        fluctuation, _ = solved()
        data = causal_data()
        parts = counterfactual_mean_parts(
            data.outcome, fluctuation.targeted, _mean_submodel(fluctuation), data.weights
        )
        assert set(parts[1.0].shares()) == {"residual", "plugin"}
        assert parts[1.0].guard is None


#: A hand-built state where the mechanism truncation **binds**, which is the whole point of
#: it: every identity below is satisfied for free on a draw where nothing clips, and that
#: degeneracy is what hid the defect for two revisions.  Rows 0 and 4 fall outside
#: :data:`TIGHT_BOUNDS` and no other does, so the arithmetic is checkable by hand.
RAW_G1 = np.array([0.02, 0.50, 0.90, 0.50, 0.01, 0.70])
TIGHT_BOUNDS = (0.05, 0.95)
#: Non-constant and mean one, this package's convention.  A weight of all ones would make
#: the weighted and unweighted statements of every identity below the same statement.
WEIGHTS = np.array([0.4, 1.6, 0.5, 1.5, 0.6, 1.4])
TREATMENT = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def hand_built(qr_scale: tuple[float, float] = (1.0, 1.0)):
    """``(reduced, targeted, outcome)`` with every array chosen rather than fitted.

    Nothing here is a plausible fit and nothing needs to be: the identities this fixture is
    for are algebraic, so what it has to be is *exact*, non-degenerate at both arms, and
    clipped on rows a reader can count.  ``qr_scale`` scales each arm's :math:`Q_r` column,
    which is what lets a caller aim the per-arm clipping bias.
    """
    from cleverly.estimators.reduced import ReducedSet
    from cleverly.fluctuation.iterative import InitialFit

    qr = np.column_stack(
        [
            qr_scale[0] * np.array([0.30, -0.20, 0.10, 0.40, -0.50, 0.25]),
            qr_scale[1] * np.array([-0.15, 0.35, 0.20, -0.30, 0.45, 0.10]),
        ]
    )
    gr1 = np.column_stack(
        [np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.45]), np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.55])]
    )
    gr2 = np.column_stack(
        [
            np.array([0.12, -0.08, 0.05, 0.20, -0.11, 0.09]),
            np.array([-0.07, 0.14, 0.03, -0.18, 0.22, 0.06]),
        ]
    )
    reduced = ReducedSet(qr, gr1, gr2, ARMS, TIGHT_BOUNDS)
    targeted = InitialFit(
        np.array([0.55, 0.42, 0.61, 0.38, 0.47, 0.52]),
        {
            0.0: np.array([0.50, 0.40, 0.60, 0.38, 0.47, 0.52]),
            1.0: np.array([0.55, 0.42, 0.61, 0.45, 0.51, 0.58]),
        },
    )
    outcome = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    return reduced, targeted, outcome


def parts_at(bounds=TIGHT_BOUNDS, qr_scale=(1.0, 1.0)):
    """The corrections and the clipping bias at :func:`hand_built`, under ``bounds``."""
    reduced, targeted, outcome = hand_built(qr_scale)
    return (
        reduced_correction_parts(
            outcome,
            targeted,
            TREATMENT,
            reduced,
            RAW_G1,
            bounds=bounds,
            guard=BOTH,
        ),
        reduced,
        targeted,
        outcome,
    )


def stored_mechanism_score(reduced, bounds):
    """Equation (9)'s score exactly as the alternation records it: raw residual, bounded covariate.

    :func:`~cleverly.fluctuation.mechanism.mechanism_score` is the function
    :func:`~cleverly.estimators.targeting.solve_with_reduction` calls, so this is the
    recorded number rather than a restatement of it.
    """
    return mechanism_score(
        TREATMENT,
        RAW_G1,
        reduced_mechanism_covariate(reduced, RAW_G1, bounds=bounds),
        WEIGHTS,
    )[0]


class TestTheCorrectionsSplitIntoTheTermsTheEquationsSolve:
    """The correction identity's arithmetic, with no fit anywhere in it.

    The reported curve subtracts one array per arm; the alternation records one score per
    arm per equation.  Whether those are the same statement is algebra, and algebra is
    cheaper and more exact to check here than on a fitted draw -- what the fitted draws in
    ``tests/unit/test_drtmle_fit.py`` add is that the wiring reaches this.
    """

    def test_each_half_is_the_term_its_equation_solves(self) -> None:
        """Against longhand, not against ``total()``.

        ``reduced_corrections`` now *calls* :meth:`CorrectionParts.total`, so comparing the
        two would compare one expression with itself and would survive the split being
        turned into a difference -- watched, and it did.  The longhand below is what makes
        the comparison say something, and the last assertion is then the one that pins the
        sum.
        """
        parts, reduced, targeted, outcome = parts_at()
        bounded = np.clip(RAW_G1, *TIGHT_BOUNDS)
        whole = reduced_corrections(
            outcome,
            targeted,
            TREATMENT,
            reduced,
            RAW_G1,
            bounds=TIGHT_BOUNDS,
            guard=BOTH,
        )

        for j, arm in enumerate(ARMS):
            mechanism = bounded if arm == 1.0 else 1.0 - bounded
            indicator = np.equal(TREATMENT, arm).astype(float)
            np.testing.assert_allclose(
                parts.d_g[arm],
                reduced.qr[:, j] / mechanism * (indicator - mechanism),
                rtol=0,
                atol=1e-15,
            )
            np.testing.assert_allclose(
                parts.d_q[arm],
                indicator * (reduced.gr2[:, j] / reduced.gr1[:, j]) * (outcome - targeted.observed),
                rtol=0,
                atol=1e-15,
            )
            assert np.array_equal(whole[arm], parts.d_g[arm] + parts.d_q[arm])

    def test_the_fixture_clips_and_says_which_rows(self) -> None:
        """The precondition every identity below is only informative under."""
        parts, _, _, _ = parts_at()
        assert list(np.flatnonzero(parts.clipped)) == [0, 4]

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_clipping_bias_is_exactly_what_the_two_expressions_differ_by(
        self, arm: float
    ) -> None:
        r"""The correction identity, per arm and with the weights carried.

        :math:`P_n[w D^*_g] - S_g^{stored} = B_{clip}`, and *exactly*: both sides are the
        same rows of the same arrays, differing only in whether the residual reads the raw
        tilted mechanism or the truncated one.  The orientation is
        :attr:`~cleverly.inference.influence.CorrectionParts.clip_bias`'s, where
        :math:`B_{clip}` carries
        :math:`g - g^b`; the residual the check reports is its negation, and asserting the
        two agree up to that sign is the point rather than an inconvenience.
        """
        parts, reduced, _, _ = parts_at()
        j = ARMS.index(arm)
        stored = float(stored_mechanism_score(reduced, TIGHT_BOUNDS)[j])
        reported = float(np.mean(WEIGHTS * parts.d_g[arm]))
        clip_bias = float(np.mean(WEIGHTS * parts.clip_bias[arm]))

        assert abs(stored - reported) > 1e-3, "the fixture must make the two disagree"
        np.testing.assert_allclose(reported - stored, clip_bias, rtol=0, atol=1e-16)

    @pytest.mark.parametrize("arm", ARMS)
    def test_and_it_is_zero_where_the_bound_does_not_bind(self, arm: float) -> None:
        """Which is why a fixture that never clips proves nothing, and this is the control."""
        parts, reduced, _, _ = parts_at(bounds=INERT_BOUNDS)
        j = ARMS.index(arm)

        assert not parts.clipped.any()
        np.testing.assert_allclose(parts.clip_bias[arm], 0.0, rtol=0, atol=1e-16)
        np.testing.assert_allclose(
            float(stored_mechanism_score(reduced, INERT_BOUNDS)[j]),
            float(np.mean(WEIGHTS * parts.d_g[arm])),
            rtol=0,
            atol=1e-16,
        )

    def test_the_weights_are_load_bearing(self) -> None:
        """The identity is between two *weighted* means, and an unweighted one is a different number."""
        parts, reduced, _, _ = parts_at()
        stored = stored_mechanism_score(reduced, TIGHT_BOUNDS)
        for j, arm in enumerate(ARMS):
            unweighted = float(np.mean(parts.d_g[arm]))
            assert abs(float(stored[j]) - unweighted) > 1e-3
            assert abs(unweighted - float(np.mean(WEIGHTS * parts.d_g[arm]))) > 1e-3

    def test_a_per_arm_defect_can_cancel_in_the_contrast(self) -> None:
        r"""Why the check is per arm and taken **before** the contrast is built.

        ``IC_ate = IC_ey1 - IC_ey0`` rowwise, so equal per-arm clipping biases cancel in it
        exactly.  The scale below is chosen to make them equal -- solved for rather than
        guessed, so the cancellation is arithmetic and not a near miss -- and an ATE-only
        check would then report a fit whose every arm is wrong as clean.
        """
        unit, _, _, _ = parts_at()
        bias = {arm: float(np.mean(WEIGHTS * unit.clip_bias[arm])) for arm in ARMS}
        parts, _, _, _ = parts_at(qr_scale=(bias[1.0] / bias[0.0], 1.0))

        per_arm = {arm: float(np.mean(WEIGHTS * parts.clip_bias[arm])) for arm in ARMS}
        contrast = per_arm[1.0] - per_arm[0.0]

        assert min(abs(value) for value in per_arm.values()) > 1e-3
        assert abs(contrast) < 1e-15, "the cancellation is what an ATE-only check would see"


def _mean_submodel(fluctuation):
    """Equation (8)'s covariate at the exiting mechanism, which the curve's residual reads."""
    g1 = np.asarray(fluctuation.mechanism.propensity, dtype=float)
    nuisance = replace(
        nuisances(WRONG_G, WRONG_Q),
        propensity=Propensity(np.column_stack([1.0 - g1, g1]), ARMS),
    )
    return build_submodel(causal_data(), nuisance, "mean", bounds=INERT_BOUNDS, nuisance_bound=1e-8)


def test_the_longhand_module_and_this_one_agree_about_the_terms() -> None:
    """A cross-check against ``test_remainder_drtmle._extra_curves``, at *its* nuisances.

    That helper evaluates the two terms at the untargeted pair, which is where the
    remainder expansion needs them; this module evaluates them at the targeted one. Both
    read the same formulas, so agreeing here means the two modules cannot drift apart on
    what ``D*_g`` and ``D*_Q`` *are* while disagreeing on where they are evaluated.
    """
    data = causal_data()
    nuisance = nuisances(WRONG_G, WRONG_Q)
    cell = law.frame()["W"].to_numpy().astype(int)
    reduced_at_initial = _reduced_set_from_longhand()

    corrections = reduced_corrections(
        data.outcome,
        nuisance.outcome,
        data.treatment,
        reduced_at_initial,
        WRONG_G[cell],
        bounds=INERT_BOUNDS,
        guard=BOTH,
    )
    for arm in ARMS:
        d_g, d_q = _extra_curves(WRONG_G, WRONG_Q, int(arm))
        np.testing.assert_allclose(corrections[arm], d_g + d_q, rtol=0, atol=1e-14)


def _reduced_set_from_longhand():
    from tests.unit.test_reduced_submodel import reduced_at

    return reduced_at(WRONG_G, WRONG_Q)


@pytest.mark.parametrize("arm", ARMS)
def test_the_mechanism_is_read_at_the_arm_it_belongs_to(arm: float) -> None:
    """The lower arm divides by ``1 - g*``, not by ``g*``.

    Two arms and a mechanism near a half make that a small error rather than an obvious
    one, which is why it is asserted rather than left to the comparisons above.
    """
    data = causal_data()
    nuisance = nuisances(WRONG_G, WRONG_Q)
    cell = law.frame()["W"].to_numpy().astype(int)
    reduced = _reduced_set_from_longhand()
    corrections = reduced_corrections(
        data.outcome,
        nuisance.outcome,
        data.treatment,
        reduced,
        WRONG_G[cell],
        bounds=INERT_BOUNDS,
        guard=BOTH,
    )
    j = ARMS.index(arm)
    expected_mechanism = WRONG_G[cell] if arm == 1.0 else 1.0 - WRONG_G[cell]
    indicator = (np.asarray(data.treatment, dtype=float) == arm).astype(float)
    d_g = reduced.qr[:, j] / expected_mechanism * (indicator - expected_mechanism)
    d_q = (
        indicator
        * (reduced.gr2[:, j] / reduced.gr1[:, j])
        * (data.outcome - nuisance.outcome.observed)
    )
    np.testing.assert_allclose(corrections[arm], d_g + d_q, rtol=0, atol=1e-14)


class TestTheMechanismResidualAndItsDenominatorAreOneEvent:
    """``D*_g`` is mean zero at the solved mechanism only because ``1_a`` and ``g`` are the
    indicator and the probability of the **same** event.

    A reverted joint-mechanism draft left an ``observed=`` mask on ``1_a`` alone, while the
    mechanism handed in went back to being the plain treatment propensity -- so the residual
    was ``1(A=a, Delta=1) - g_a`` against a ``g_a`` that is not the probability of that
    event. No production path passed the mask, so nothing on a fit could have shown it, and
    that is exactly why it is pinned here rather than end to end.
    """

    def test_the_parameter_that_carried_the_mask_is_gone(self) -> None:
        """Structural, because the numeric defect is unreachable and cannot be witnessed.

        A fit with missing outcomes and a non-empty guard builds the five reductions and
        goes to ``missing_outcome_correction_parts``; with an empty guard it fits no
        reductions and forms no corrections. So the mask was only ever the all-ones
        default, and there is no fixture that would turn red. What can be asserted is that
        the parameter is not there to be passed again.
        """
        for function in (reduced_correction_parts, reduced_corrections):
            assert "observed" not in inspect.signature(function).parameters, function.__name__

    def test_the_residual_is_the_arm_indicator_and_nothing_else(self) -> None:
        """The identity longhand, at a ``Qr`` that is nowhere near zero.

        Without the nonzero check this would be ``0 == 0`` at the truth, where ``Qr``
        vanishes row by row and any mask at all would agree.
        """
        parts, reduced, _, _ = parts_at(qr_scale=(3.0, 2.0))
        raw = np.asarray(RAW_G1, dtype=float)
        lower, upper = TIGHT_BOUNDS
        g1 = np.clip(raw, lower, upper)

        for column, arm in enumerate(reduced.arms):
            mechanism = g1 if arm == reduced.arms[1] else 1.0 - g1
            indicator = (np.asarray(TREATMENT, dtype=float) == float(arm)).astype(float)
            qr = np.asarray(reduced.qr, dtype=float)[:, column]
            expected = qr / mechanism * (indicator - mechanism)

            np.testing.assert_array_equal(parts.d_g[arm], expected)
            assert np.max(np.abs(expected)) > 1e-3, "the witness must not be zero at the truth"

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

from dataclasses import replace

import numpy as np
import pytest

from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.targeting import build_submodel
from cleverly.inference.influence import (
    counterfactual_mean_parts,
    counterfactual_means,
    reduced_corrections,
)
from tests import discrete_law as law
from tests.unit.test_reduced_regressions import ARMS, INERT_BOUNDS, causal_data, nuisances
from tests.unit.test_reduction_alternation import alternate
from tests.unit.test_remainder_drtmle import WRONG_G, WRONG_Q, _extra_curves


def solved(g_hat: np.ndarray = WRONG_G, q_hat: np.ndarray = WRONG_Q):
    """``(fluctuation, corrections)`` from a converged alternation at wrong nuisances."""
    fluctuation = alternate(g_hat, q_hat)
    data = causal_data()
    corrections = reduced_corrections(
        data.outcome,
        fluctuation.targeted,
        data.treatment,
        fluctuation.reduction.reduced,
        fluctuation.mechanism.propensity,
        bounds=INERT_BOUNDS,
        observed=data.observed,
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
            np.testing.assert_allclose(corrections[arm], d_g + d_q, atol=1e-14)

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
            np.testing.assert_allclose(reported, plain - corrections[arm], atol=1e-14)
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
        observed=data.observed,
    )
    for arm in ARMS:
        d_g, d_q = _extra_curves(WRONG_G, WRONG_Q, int(arm))
        np.testing.assert_allclose(corrections[arm], d_g + d_q, atol=1e-14)


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
        observed=data.observed,
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
    np.testing.assert_allclose(corrections[arm], d_g + d_q, atol=1e-14)

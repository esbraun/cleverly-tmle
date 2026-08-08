r"""Equation (9) solved at the bounded mechanism the reported curve reads.

:func:`~cleverly.fluctuation.mechanism.solve_mechanism` drives
:math:`P_n[H_g(A - g^*)]` to zero at the raw :math:`\operatorname{expit}` tilt.
:func:`~cleverly.inference.influence.reduced_correction_parts` subtracts
:math:`Q_r/\bar g^*(1_a - \bar g^*)` at the **truncated** one.  Those are two expressions,
identical on every row the bound leaves alone and different on every row it clips, and the
gap between them was the centring defect: one clipped row of 600 left a reported curve uncentred at
``5.8e-04`` while the solver recorded ``1e-09``.
:func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism` solves the second.

Nothing here runs a fit.  The claim is about a solver and a pair of bounds, so it is checked
against arithmetic on arrays -- the identity that closes the defect end to end is
``tests/unit/test_drtmle_fit.py``'s, at the alternation's fixed point, because within a
single solve the covariate's denominator is still at the *pre-tilt* mechanism (which is
limitation 5 and is not what this closes).

**Three properties of the fixture are load-bearing and each is asserted rather than
assumed.**  The nuisances are wrong on purpose, because :math:`Q_r` vanishes row by row at
the truth and a zero covariate makes every solver agree.  The bound must *bind*, because
where it does not there is nothing here to test -- that is the whole content of
:class:`TestTheFastPathIsTheOldSolver`.  And a binding bound must still admit a **root**:
tighten it and there is no :math:`\epsilon` at all, which is
:class:`TestNoRootIsNotReportedAsASolution` rather than a fixture to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.mechanism import (
    mechanism_score,
    solve_bounded_mechanism,
    solve_mechanism,
)
from cleverly.fluctuation.reduced import reduced_mechanism_covariate
from tests import discrete_law as law
from tests.unit.test_reduced_submodel import INERT, realised, reduced_at
from tests.unit.test_remainder_drtmle import WRONG_G, WRONG_Q

#: Binds on a fifth of the rows and still admits a root.  Both halves are measured below --
#: at ``(0.28, 0.72)`` the equation has no solution at all, and the gap between the two is
#: narrow, which is why this constant is not "some tight bound".
BINDING = (0.25, 0.75)

#: No root: every row pins, so the derivative is zero everywhere and no epsilon moves the
#: score.  The state :class:`TestNoRootIsNotReportedAsASolution` exists for.
PINNING = (0.45, 0.55)


def setup(bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(treatment, mechanism, covariate, weights)`` at nuisances that are wrong on purpose."""
    cell, treatment, _ = realised()
    covariate = reduced_mechanism_covariate(
        reduced_at(WRONG_G, WRONG_Q), WRONG_G[cell], bounds=bounds
    )
    return treatment, WRONG_G[cell], covariate, np.ones(law.N)


class TestTheFastPathIsTheOldSolver:
    """Where the bound does not bind, this **is** :func:`solve_mechanism`, array for array.

    Not an optimisation and not a tolerance: the clip is the identity on every row, so the
    unconstrained root *is* the bounded root, and the object is returned untouched.  That is
    what makes every module fitting at inert bounds a regression surface by construction --
    ``tests/unit/test_influence_gateaux_drtmle.py`` asserts ``1e-12`` with ``rtol=0`` on a
    real fit, and it is on this branch.
    """

    def test_it_returns_the_unconstrained_solve_unchanged(self) -> None:
        treatment, mechanism, covariate, weights = setup(INERT)
        plain = solve_mechanism(treatment, mechanism, covariate, weights)
        bounded = solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=INERT)

        np.testing.assert_array_equal(bounded.propensity, plain.propensity)
        np.testing.assert_array_equal(bounded.epsilon, plain.epsilon)
        np.testing.assert_array_equal(bounded.score, plain.score)
        assert bounded.loglik == plain.loglik
        assert bounded.hessian_condition == plain.hessian_condition
        assert bounded.failure is plain.failure

    def test_and_the_fixture_would_have_noticed_a_difference(self) -> None:
        """The control: this bound is inert *for this tilt*, which is a fact about the draw."""
        treatment, mechanism, covariate, weights = setup(INERT)
        plain = solve_mechanism(treatment, mechanism, covariate, weights)

        assert np.min(plain.propensity) > INERT[0] and np.max(plain.propensity) < INERT[1]
        assert np.max(np.abs(plain.epsilon)) > 1.0, "an untilted mechanism would test nothing"

    def test_a_plain_solve_that_failed_is_returned_with_its_failure(self) -> None:
        """The case that makes the fast path load-bearing rather than a shortcut.

        Where the bound never binds and the *plain* solve stopped short -- a singular
        Hessian, a cap, a line search that gave up -- the bounded branch would happily go
        looking for the root with a different solver and find it.  That would be a better
        answer to a different question: this function must not improve a fit whose bound
        does not bind, or the DRTMLE path stops being the surface it is here to stay.
        ``max_iter=1`` is the cheapest way to manufacture such a solve.
        """
        treatment, mechanism, covariate, weights = setup(INERT)
        plain = solve_mechanism(treatment, mechanism, covariate, weights, max_iter=1)
        bounded = solve_bounded_mechanism(
            treatment, mechanism, covariate, weights, bounds=INERT, max_iter=1
        )

        assert plain.failure == "max_iter_reached" and not plain.converged
        assert bounded.failure == plain.failure and bounded.converged is plain.converged
        np.testing.assert_array_equal(bounded.propensity, plain.propensity)
        np.testing.assert_array_equal(bounded.score, plain.score)


class TestTheScoreIsTakenAtTheTruncatedResidual:
    """The equation this solves, stated as arithmetic rather than as a docstring."""

    def test_the_returned_score_is_the_truncated_one(self) -> None:
        treatment, mechanism, covariate, weights = setup(BINDING)
        bounded = solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=BINDING)

        for column in range(covariate.shape[1]):
            longhand = np.mean(weights * covariate[:, column] * (treatment - bounded.propensity))
            np.testing.assert_allclose(bounded.score[column], longhand, atol=1e-15, rtol=0)

    def test_and_it_is_zero(self) -> None:
        treatment, mechanism, covariate, weights = setup(BINDING)
        bounded = solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=BINDING)

        assert bounded.converged and bounded.failure is None
        assert np.max(np.abs(bounded.score)) < 1e-15

    def test_clipping_after_the_unconstrained_solve_does_not_solve_it(self) -> None:
        """Candidate A, and the test that tells it from what landed.

        R's ``fluctuateG`` clips its fitted values and returns *that*, which makes one array
        out of two and is internally consistent -- and does not solve the clipped state's
        score equation, because a hard clip is a projection applied *after* the
        optimisation.  Measured here at ``2.0e-04`` against ``2.4e-17``: twelve orders, on
        the same covariate and the same bound.
        """
        treatment, mechanism, covariate, weights = setup(BINDING)
        plain = solve_mechanism(treatment, mechanism, covariate, weights)
        after_the_clip, _ = mechanism_score(
            treatment, np.clip(plain.propensity, *BINDING), covariate, weights
        )
        bounded = solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=BINDING)

        assert np.max(np.abs(after_the_clip)) > 1e-5
        assert np.max(np.abs(after_the_clip)) > 1e10 * np.max(np.abs(bounded.score))

    def test_the_bound_binds_here_and_the_epsilon_moved_because_of_it(self) -> None:
        """Both preconditions, so neither assertion above can pass for the wrong reason."""
        treatment, mechanism, covariate, weights = setup(BINDING)
        plain = solve_mechanism(treatment, mechanism, covariate, weights)
        bounded = solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=BINDING)

        pinned = np.mean(bounded.propensity <= BINDING[0]) + np.mean(
            bounded.propensity >= BINDING[1]
        )
        assert pinned > 0.1, "the bound must bind or there is nothing here to solve"
        assert not np.allclose(bounded.epsilon, plain.epsilon)


class TestTheReturnedMechanismIsWhatTheCurveReads:
    """Inside the bounds on every row, which is what makes the identity structural."""

    def test_every_row_is_inside_the_bounds(self) -> None:
        treatment, mechanism, covariate, weights = setup(BINDING)
        bounded = solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=BINDING)

        assert np.min(bounded.propensity) >= BINDING[0]
        assert np.max(bounded.propensity) <= BINDING[1]

    def test_a_vanishing_covariate_still_comes_back_truncated(self) -> None:
        r"""At the truth :math:`Q_r \equiv 0`, so nothing is tilted -- and the bound still applies.

        The array the curve divides by is the truncated one whether or not a tilt moved it,
        and this is the path where no root finder runs at all: a zero covariate has no
        derivative to hand one.
        """
        cell, treatment, _ = realised()
        weights = np.ones(law.N)
        covariate = reduced_mechanism_covariate(
            reduced_at(law.G, law.Q), law.G[cell], bounds=(0.3, 0.7)
        )
        bounded = solve_bounded_mechanism(
            treatment, law.G[cell], covariate, weights, bounds=(0.3, 0.7)
        )

        assert np.max(np.abs(covariate)) < 1e-14, "the fixture's whole point"
        np.testing.assert_array_equal(bounded.epsilon, np.zeros(2))
        assert np.min(bounded.propensity) == 0.3
        assert bounded.converged and bounded.failure is None


class TestNoRootIsNotReportedAsASolution:
    """The acceptance criterion the validation plan states in those words.

    A bounded equation need not have a solution: pin every row and the clip is flat
    everywhere, so no :math:`\\epsilon` moves the score at all.  Returning the last iterate
    with ``converged=True`` is the one outcome this must not have, because the reported
    interval would then rest on an equation nothing solved.
    """

    def test_it_says_the_rows_are_pinned(self) -> None:
        treatment, mechanism, covariate, weights = setup(PINNING)
        bounded = solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=PINNING)

        assert not bounded.converged
        assert bounded.failure == "bounds_pinned"
        assert np.max(np.abs(bounded.score)) > 1e-3

    def test_a_bound_that_binds_without_a_root_is_not_the_same_state(self) -> None:
        """``(0.28, 0.72)`` leaves a *small* score and still has no root, which is the trap.

        A threshold on the score alone would call this converged at a looser tolerance and
        report an interval; what says otherwise is that the equation has no solution, and
        the endpoint -- half the rows pinned -- is what names it.
        """
        treatment, mechanism, covariate, weights = setup((0.28, 0.72))
        bounded = solve_bounded_mechanism(
            treatment, mechanism, covariate, weights, bounds=(0.28, 0.72)
        )

        assert not bounded.converged
        assert bounded.failure == "bounds_pinned"
        assert 1e-6 < np.max(np.abs(bounded.score)) < 1e-4

    @pytest.mark.parametrize("bounds", [(0.0, 0.9), (0.1, 1.0), (0.6, 0.4), (-0.1, 0.9)])
    def test_a_bound_that_is_not_a_probability_interval_is_refused(self, bounds) -> None:
        treatment, mechanism, covariate, weights = setup(INERT)

        with pytest.raises(ValueError, match="0 < lo < hi < 1"):
            solve_bounded_mechanism(treatment, mechanism, covariate, weights, bounds=bounds)

r"""The three-equation alternation, before any estimator reaches it.

:func:`~cleverly.estimators.targeting.solve_with_reduction` solves equation (8) -- the
ordinary outcome score -- beside equation (9), which fluctuates the mechanism, and equation
(10), which fluctuates the outcome regression a second time.  None of the three can be
solved once and left, and the reduced-dimension regressions the last two read are refitted
against the current pair on every round, which is what ``drtmle`` does and what the
starred :math:`Q_r^*`, :math:`g_r^*` in the source's statement of the equations mean.

The instrument here is the exact law with a **saturated** learner for the reductions, so a
refit is the exact conditional expectation given the design value and nothing is
approximate.  Two consequences shape every test below.

* At **deliberately wrong** nuisances all three equations are non-trivial, so this is where
  convergence, the joint likelihood and the refit can be checked at all.
* At the **truth** the reductions are identically zero row by row, both extra coefficients
  are zero, and the alternation must reproduce a plain ``TMLE`` array for array.  That is
  the degeneracy every ``test_influence_gateaux*`` module would be reduced to, pinned here
  deliberately rather than relied on silently.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from cleverly.estimators._nuisance import NuisanceEstimates, Propensity
from cleverly.estimators.reduced import ReducedSet, fit_reduced
from cleverly.estimators.targeting import (
    _UNSOLVED,
    ReductionSpec,
    TargetingSpec,
    build_submodel,
    needs_reduction,
    solve_submodel,
    solve_with_reduction,
)
from cleverly.fluctuation._score import score_columns
from cleverly.fluctuation.iterative import Fluctuation
from cleverly.fluctuation.reduced import reduced_mechanism_covariate, reduced_outcome_submodel
from tests import discrete_law as law
from tests.discrete_law_longitudinal import CellMeans
from tests.unit.test_reduced_regressions import (
    ARMS,
    INERT_BOUNDS,
    causal_data,
    fitted,
    nuisances,
)
from tests.unit.test_remainder_drtmle import BOTH, WRONG_G, WRONG_Q

SPEC = TargetingSpec()

#: The lower bound on the missingness and intermediate mechanisms. Neither exists on this
#: law, so it never bites; spelled out rather than imported so the call reads completely.
NUISANCE_BOUND = 1e-8


def refit_closure(data, counter: list[NuisanceEstimates] | None = None):
    """A saturated refit of the three reduced regressions, recording what it was handed."""

    def refit(current: NuisanceEstimates) -> tuple[ReducedSet, tuple[ReducedSet, ...]]:
        if counter is not None:
            counter.append(current)
        reduced, _, at_companion = fit_reduced(
            data,
            current,
            regression_learner=CellMeans(),
            classification_learner=CellMeans(),
            g_bounds=INERT_BOUNDS,
            companion=current.companion,
        )
        return reduced, at_companion

    return refit


def alternate(
    g_hat: np.ndarray,
    q_hat: np.ndarray,
    *,
    guard: tuple[str, ...] = BOTH,
    counter: list[NuisanceEstimates] | None = None,
    max_outer: int = 50,
) -> Fluctuation:
    """Run the alternation at the given nuisance guesses and return its fluctuation.

    ``max_outer`` is exposed so a test can stop the *refitting* rounds early and read what
    the closing pass leaves behind; the closing pass runs whatever this is set to.
    """
    data = causal_data()
    nuisance = replace(nuisances(g_hat, q_hat), reduced=fitted(g_hat, q_hat))
    _, fluctuation = solve_with_reduction(
        data,
        nuisance,
        "mean",
        SPEC,
        reduction=ReductionSpec(refit=refit_closure(data, counter), guard=guard),
        bounds=INERT_BOUNDS,
        nuisance_bound=NUISANCE_BOUND,
        scaled=nuisance.scaler.scale(data.outcome),
        weights=data.weights,
        observed=data.observed,
        max_outer=max_outer,
    )
    return fluctuation


def plain(g_hat: np.ndarray, q_hat: np.ndarray) -> Fluctuation:
    """The ordinary single-equation targeting step, for the comparisons that need one."""
    data = causal_data()
    nuisance = nuisances(g_hat, q_hat)
    submodel = build_submodel(
        data, nuisance, "mean", bounds=INERT_BOUNDS, nuisance_bound=NUISANCE_BOUND
    )
    return solve_submodel(
        nuisance.scaler.scale(data.outcome),
        nuisance.outcome,
        submodel,
        data.weights,
        data.observed,
        SPEC,
    )


class TestAllThreeEquationsAreSolved:
    """The point of the loop, and the thing a stale-score test would let through."""

    def test_every_score_reaches_the_tolerance(self) -> None:
        fluctuation = alternate(WRONG_G, WRONG_Q)
        reduction = fluctuation.reduction
        mechanism = fluctuation.mechanism

        assert fluctuation.relative_score_norm <= SPEC.tol
        assert reduction.relative_score <= SPEC.tol
        assert mechanism.relative_score <= SPEC.tol
        assert reduction.converged and reduction.failure is None

    def test_the_scores_it_reports_are_the_ones_at_the_pair_it_exits_at(self) -> None:
        r"""Recomputed rather than remembered, which is the whole of the stale-score trap.

        Equation (10) is solved and then equation (8) moves :math:`\bar Q^*` underneath it;
        equation (9)'s covariate reads a :math:`Q_r` refitted after both.  A loop reporting
        the scores where they were *solved* would exit having solved one equation and left
        two open, and every number it reported would look converged.
        """
        fluctuation = alternate(WRONG_G, WRONG_Q)
        reduction = fluctuation.reduction
        data = causal_data()

        # The outcome is binary here, so the scaler is the identity and this is the
        # scaled outcome the fluctuation was solved against.
        scaled = data.outcome
        submodel = reduced_outcome_submodel(data.treatment, reduction.reduced, bounds=INERT_BOUNDS)
        recomputed = score_columns(
            scaled,
            fluctuation.targeted.observed,
            submodel.observed,
            data.weights,
            np.asarray(data.observed, dtype=bool),
        )
        np.testing.assert_allclose(reduction.score, recomputed, rtol=0, atol=1e-14)

        indicator = (np.asarray(data.treatment, dtype=float) == 1.0).astype(float)
        covariate = reduced_mechanism_covariate(
            reduction.reduced, fluctuation.mechanism.propensity, bounds=INERT_BOUNDS
        )
        mechanism_score = np.mean(
            data.weights[:, None]
            * covariate
            * (indicator - fluctuation.mechanism.propensity)[:, None],
            axis=0,
        )
        np.testing.assert_allclose(fluctuation.mechanism.score, mechanism_score, rtol=0, atol=1e-14)

    def test_it_terminated_on_its_own(self) -> None:
        """Not on the outer cap, which would make the reported scores a truncation."""
        reduction = alternate(WRONG_G, WRONG_Q).reduction
        assert 1 <= reduction.n_outer < 50


class TestHowTheLoopExited:
    """Which of the three exits fired -- a fact nothing else on the record carries.

    ``rounds`` cannot separate a stall from a cap without ``max_outer``, which is a function
    default rather than a field.  And ``failure`` answers a different question: it is set
    from the *closing pass's* scores against a statistical threshold, without ever reading
    the round count.  The two therefore come apart in both directions, which is why one
    cannot stand in for the other and why the claim "such a fit runs to the outer cap" had
    nothing to be checked against before this was recorded.

    **A stall gets no test.**  It needs a draw on which neither the joint likelihood climbs
    nor the worst score improves by a factor of ``_STALL_FACTOR``, and no pair on this law
    produces one -- the exact-law refit is saturated, so the loop either descends cleanly or
    runs out of rounds.  A test that manufactured a stall by other means would pin the
    manufacture rather than the behaviour, so the exit is left checked only for membership,
    where an estimator fit meets it.
    """

    def test_a_pair_that_settles_exits_on_the_tolerance(self) -> None:
        assert alternate(WRONG_G, WRONG_Q).reduction.exit_reason == "tolerance"

    def test_a_loop_that_runs_out_of_rounds_says_cap_where_failure_says_nothing(self) -> None:
        """The two disagreeing on one fit is the point, not an artefact of this one.

        Capped at a single refitting round the loop exits with rounds left unrun, and the
        closing pass then drives all three scores well under ``_UNSOLVED`` -- so ``failure``
        is ``None`` and ``converged`` is ``True`` on a fit that never terminated on its own.
        A reader inferring the cap from either of those would infer the opposite.
        """
        reduction = alternate(WRONG_G, WRONG_Q, max_outer=1).reduction

        assert reduction.exit_reason == "cap"
        assert reduction.n_outer == 1
        assert reduction.failure is None

    @pytest.mark.parametrize("max_outer", [1, 2, 50])
    def test_without_the_closing_pass_the_scores_would_fail(self, max_outer: int) -> None:
        """The pass is load-bearing on **every** fit here, not only on a capped one.

        "Remove the closing pass and the scores fail" is checkable off the record rather than
        by patching the function out, which is the better instrument because patching would
        have to rebuild the record the pass returns and could only be wrong in its own way.
        :attr:`ReductionFluctuation.trace` carries one row per **refitting** round and then
        one final row for the pass, so the loop's own last row *is* the state a fit without
        the pass would report -- and each row is ``(round, outcome, reduced, mechanism,
        joint)``, so the worst of the three scores is ``max(row[1:4])``.

        Measured across the cap: the loop's last row is ``4.4e-02`` at one round, ``3.7e-04``
        at two and ``2.2e-06`` at its own natural exit -- above :data:`_UNSOLVED` in all
        three -- against ``1.1e-11``, ``2.1e-10`` and ``5.4e-11`` after the pass.  So even the
        loop that terminated on its own tolerance leaves equations the reported curve would
        not satisfy, which is the whole reason the alternation does not simply stop.

        The reason is structural rather than a matter of tuning: the loop solves equation (9)
        at the *previous* round's reductions and equation (10) at the current round's first
        refit, then refits once more before the record is built -- so neither extra equation
        is ever solved at the arrays the curve is finally built from.
        """
        reduction = alternate(WRONG_G, WRONG_Q, max_outer=max_outer).reduction
        *loop, closing = reduction.trace

        assert len(loop) == reduction.rounds, "the trace's shape is not one row per round"
        assert max(loop[-1][1:4]) > _UNSOLVED, (
            "the loop's own exit already satisfies the equations, so this fixture cannot "
            "see whether the closing pass does anything"
        )
        assert max(closing[1:4]) <= _UNSOLVED
        assert reduction.failure is None

    def test_the_closing_pass_reports_its_own_cap_separately(self) -> None:
        """Equation (9)'s stage stops on ``max_steps`` or on the tolerance, and says which.

        Its covariate reads the mechanism it tilts, so each solve leaves a residual at the
        post-tilt covariate that iterating shrinks without removing.  Whether twenty steps
        were enough is therefore a property of the fit rather than a constant, and the flag
        is asserted here against the score it is a statement about -- at two refitting rounds
        the stage lands at ``2.1e-10`` and is capped, and at the loop's own exit it reaches
        ``4.2e-12`` and is not.
        """
        capped = alternate(WRONG_G, WRONG_Q, max_outer=2)
        settled = alternate(WRONG_G, WRONG_Q)

        assert capped.reduction.closing_capped
        assert capped.mechanism.relative_score > SPEC.tol
        assert not settled.reduction.closing_capped
        assert settled.mechanism.relative_score <= SPEC.tol


class TestTheJointLikelihoodNeverDecreases:
    r"""Why this terminates rather than merely settling.

    Equation (9) is a weighted logistic MLE of :math:`A \mid W`; equations (8) and (10) are
    the outcome quasi-likelihood.  Those are separate factors of the likelihood of
    :math:`(A, Y) \mid W`, so each step maximises its own factor with the others held fixed
    and the joint value cannot go down -- the argument
    :mod:`cleverly.fluctuation.mechanism` writes out for the two-equation loop, carried
    over.  **Refitting the reductions between rounds does not break it**: a refit changes
    the direction of the next submodel, not the value at the point it passes through.

    This is also the instrument that pins *continuing* rather than restarting from
    :math:`\bar Q^0`, which is where this loop parts company with
    :func:`~cleverly.estimators.targeting.solve_with_projection`.  Both outcome steps were
    mutated to restart and the trace fell -- ``-1459.70`` to ``-1478.36`` between the first
    two rounds -- while the point estimates stayed put to eight figures.  So the estimate is
    not the thing that shows it and no assertion about ``psi`` would have caught it; the
    likelihood is, and the mechanism equation went unsolved besides.
    """

    def test_the_trace_is_monotone(self) -> None:
        reduction = alternate(WRONG_G, WRONG_Q).reduction
        joint = [row[4] for row in reduction.trace]

        assert len(joint) >= 2, "one round cannot show a monotone sequence"
        assert all(later >= earlier - 1e-9 for earlier, later in pairwise(joint))


class TestTheReductionsAreRefittedInsideTheLoop:
    """Seam 5, decided in favour of the source: the equations are stated at starred values."""

    def test_it_refits_against_the_targeted_pair(self) -> None:
        seen: list[NuisanceEstimates] = []
        data = causal_data()
        initial = replace(nuisances(WRONG_G, WRONG_Q), reduced=fitted(WRONG_G, WRONG_Q))
        _, fluctuation = solve_with_reduction(
            data,
            initial,
            "mean",
            SPEC,
            reduction=ReductionSpec(refit=refit_closure(data, seen), guard=BOTH),
            bounds=INERT_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            scaled=initial.scaler.scale(data.outcome),
            weights=data.weights,
            observed=data.observed,
        )

        assert len(seen) >= 2, "both guards refit, so a round makes more than one call"
        # And the closing pass refits nothing -- that is the whole of what makes it a
        # *closing* pass rather than another round. Counted rather than assumed: a refit in
        # there would move the arrays out from under the equations it exists to solve at.
        assert len(seen) == 2 * fluctuation.reduction.n_outer
        for handed in seen:
            assert not np.allclose(handed.outcome.arms[1.0], initial.outcome.arms[1.0])
            assert not np.allclose(handed.propensity.arm(1.0), initial.propensity.arm(1.0))
        # And the split travels with them, which is what `fit_reduced` takes a whole
        # NuisanceEstimates for rather than two arrays.
        assert all(handed.folds is initial.folds for handed in seen)

        assert not np.allclose(fluctuation.reduction.reduced.qr, initial.reduced.qr)

    def test_the_reported_reductions_are_the_refitted_ones(self) -> None:
        """``result.nuisance.reduced`` keeps the initial fit, exactly as the mechanism does."""
        data = causal_data()
        nuisance = replace(nuisances(WRONG_G, WRONG_Q), reduced=fitted(WRONG_G, WRONG_Q))
        _, fluctuation = solve_with_reduction(
            data,
            nuisance,
            "mean",
            SPEC,
            reduction=ReductionSpec(refit=refit_closure(data), guard=BOTH),
            bounds=INERT_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            scaled=nuisance.scaler.scale(data.outcome),
            weights=data.weights,
            observed=data.observed,
        )

        assert fluctuation.reduction.reduced is not nuisance.reduced
        np.testing.assert_array_equal(nuisance.reduced.qr, fitted(WRONG_G, WRONG_Q).qr)
        np.testing.assert_array_equal(
            nuisance.propensity.arm(1.0), nuisances(WRONG_G, WRONG_Q).propensity.arm(1.0)
        )


class TestTheGuardSelectsTheEquations:
    """``drtmle``'s vocabulary, crossed: ``"Q"`` moves ``g`` and ``"g"`` moves ``Qbar``."""

    def test_guarding_against_a_wrong_outcome_regression_fluctuates_the_mechanism(self) -> None:
        fluctuation = alternate(WRONG_G, WRONG_Q, guard=("Q",))
        initial = nuisances(WRONG_G, WRONG_Q).propensity.arm(1.0)

        assert fluctuation.mechanism is not None
        assert not np.allclose(fluctuation.mechanism.propensity, initial)
        assert fluctuation.reduction.epsilon.size == 0, "equation (10) was not asked for"

    def test_guarding_against_a_wrong_mechanism_fluctuates_the_outcome_regression(self) -> None:
        fluctuation = alternate(WRONG_G, WRONG_Q, guard=("g",))
        reduction = fluctuation.reduction

        assert fluctuation.mechanism is None
        assert reduction.epsilon.size == 2
        # The regression moved, and the equation that moved it started somewhere. Neither
        # is a statement about `epsilon`, which is the *last round's* step and is near zero
        # at any converged fixed point -- see `ReductionFluctuation`.
        assert not np.allclose(
            fluctuation.targeted.arms[1.0], plain(WRONG_G, WRONG_Q).targeted.arms[1.0]
        )
        assert np.max(np.abs(reduction.score_initial)) > 1e-4
        assert np.max(np.abs(reduction.score)) < 1e-10
        assert reduction.trace[0][2] > reduction.trace[-1][2]

    def test_both_guards_solve_both(self) -> None:
        fluctuation = alternate(WRONG_G, WRONG_Q, guard=BOTH)

        assert fluctuation.mechanism is not None
        assert fluctuation.reduction.guard == BOTH
        assert fluctuation.reduction.names == ("h_dr0", "h_dr1")


class TestAtTheTruthItIsAPlainTMLE:
    r""":math:`Q_r \equiv g_{r,2} \equiv 0`, so both extra equations are already solved.

    Which is why the exact-law instruments this package leans on elsewhere cannot see what
    this estimator buys: they can only check that it has not broken the ordinary path.
    """

    def test_both_extra_coefficients_are_zero(self) -> None:
        fluctuation = alternate(law.G, law.Q)

        np.testing.assert_allclose(fluctuation.mechanism.epsilon, 0.0, atol=1e-14)
        np.testing.assert_allclose(fluctuation.reduction.epsilon, 0.0, atol=1e-14)
        np.testing.assert_array_equal(
            fluctuation.mechanism.propensity, nuisances(law.G, law.Q).propensity.arm(1.0)
        )

    def test_the_targeted_regression_is_the_plain_one(self) -> None:
        fluctuation = alternate(law.G, law.Q)
        once = plain(law.G, law.Q)

        for arm in ARMS:
            np.testing.assert_allclose(
                fluctuation.targeted.arms[arm], once.targeted.arms[arm], rtol=0, atol=1e-15
            )


class TestTheRefusals:
    def test_it_refuses_nuisances_with_no_reductions(self) -> None:
        data = causal_data()
        with pytest.raises(ValueError, match="reduced-dimension regressions"):
            solve_with_reduction(
                data,
                nuisances(WRONG_G, WRONG_Q),
                "mean",
                SPEC,
                reduction=ReductionSpec(refit=refit_closure(data)),
                bounds=INERT_BOUNDS,
                nuisance_bound=NUISANCE_BOUND,
                scaled=data.outcome,
                weights=data.weights,
                observed=data.observed,
            )

    def test_an_empty_guard_is_a_plain_tmle_and_must_not_arrive_here(self) -> None:
        """A fit wanting one carries no reductions at all, which is bit for bit the old path."""
        with pytest.raises(ValueError, match="empty guard"):
            alternate(WRONG_G, WRONG_Q, guard=())

    def test_inconsistent_data_and_nuisance_arms_are_refused(self) -> None:
        data = causal_data()
        three = replace(
            nuisances(WRONG_G, WRONG_Q),
            propensity=Propensity(np.full((law.N, 3), 1 / 3), (0.0, 1.0, 2.0)),
            reduced=ReducedSet(
                qr=np.zeros((law.N, 3)),
                gr1=np.full((law.N, 3), 0.3),
                gr2=np.zeros((law.N, 3)),
                arms=(0.0, 1.0, 2.0),
                g_bounds=INERT_BOUNDS,
            ),
        )
        with pytest.raises(ValueError, match="the two must describe the same ones"):
            solve_with_reduction(
                data,
                three,
                "mean",
                SPEC,
                reduction=ReductionSpec(refit=refit_closure(data)),
                bounds=INERT_BOUNDS,
                nuisance_bound=NUISANCE_BOUND,
                scaled=data.outcome,
                weights=data.weights,
                observed=data.observed,
            )


class TestTheDispatchPredicate:
    """Keyed on the nuisances, because the group is still ``"mean"``."""

    def test_it_reads_the_nuisances_rather_than_the_group_name(self) -> None:
        plain_nuisance = nuisances(WRONG_G, WRONG_Q)
        carrying = replace(plain_nuisance, reduced=fitted(WRONG_G, WRONG_Q))

        assert not needs_reduction(plain_nuisance, "mean")
        assert needs_reduction(carrying, "mean")
        # Every other group has no reduced-regression derivation behind it, and a fit that
        # reported one would be reporting a parameter nothing has derived.
        assert not needs_reduction(carrying, "att")
        assert not needs_reduction(carrying, "msm")

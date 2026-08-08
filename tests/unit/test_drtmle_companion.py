r"""Is the evaluation companion the fit's own state, at rows the fit never saw?

``evaluation=`` exists to make the corrected remainder

.. code-block:: text

    R_remaining = psi-hat - psi_0 - (P_n - P_0) D-hat_DR

computable, which needs :math:`P_0\hat D` -- and :math:`P_n\hat D` is **refused** as a
substitute for it, since that is the quantity targeting drove to zero and it answers a
different question.  :math:`P_0\hat D` needs the curve as a *function* of :math:`(W, A, Y)`,
and an array of out-of-fold predictions defines one nowhere.  ``DRTMLE(evaluation=...)``
supplies it, and this module is what says the thing it supplies is the fit's own state
rather than something that resembles it.  ``docs/drtmle.md``'s *The remainder terms, and the
rate conditions* is what the quantity is for.

**Two claims, and they fail against different mutations.**

*The companion changes nothing.*  It contributes to no fit, no fold and no score, so a fit
that declares one is bit for bit a fit that does not.  A companion row leaking into any
``fit_mask`` breaks this and would leave every other assertion here passing.

*The companion at the fitting rows is the fit.*  Hand the fitting frame back in as the
evaluation sample and fold ``k``'s slab, read at the rows fold ``k`` holds out, must equal
the production array -- for the initial mechanism and regression, for the three reduced
regressions, and for the targeted :math:`\bar Q^*` and :math:`g^*` the alternation leaves.
That is the anchor the whole instrument rests on, and it is sharp: it fails if a slab is
read one fold out, if a round's tilt is dropped, if the companion travels along the
production covariate instead of its own, or if a refit is taken at a stale design.

It is an identity rather than a comparison against a second implementation, which is the
whole reason the companion is *carried through the solvers* instead of replayed afterwards
from ``(initial, epsilon)``.  :class:`TestTheIdentityIsNotVacuous` is what says the identity
has content -- the slabs genuinely disagree away from their own fold, so an implementation
that returned one array ``K`` times could not pass it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly import DRTMLE
from cleverly.datasets import make_linear_ate, make_weak_overlap

#: Small, cross-fitted, and ``glm`` throughout: every claim here is an exact identity, so
#: the sample size only has to be enough for three folds to be fittable.
N = 240
FOLDS = 3
SETTINGS: dict[str, Any] = {
    "outcome_learner": "glm",
    "treatment_learner": "glm",
    "reduced_outcome_learner": "glm",
    "reduced_treatment_learner": "glm",
    "n_folds": FOLDS,
    "learner_folds": 2,
    "random_state": 0,
    "simultaneous": False,
    "estimands": ("ate", "ey1", "ey0"),
}

#: The window the identity below is asserted in, with ``rtol=0`` beside it everywhere.
#:
#: **Not** bit-for-bit, and the reason is worth stating rather than absorbing into a loose
#: tolerance.  The production array comes from ``model.predict(matrix[test])`` and the
#: companion slab from ``model.predict(matrix)`` -- the *same* fitted model at the *same*
#: design rows, but in a batch of a different height, and a BLAS matrix product blocks
#: differently at different heights.  What that produces is a last-bit disagreement,
#: measured here at ``1.1e-16`` on the mechanism and ``5.6e-17`` on the targeted regression
#: against arrays of order one.  Every mutation of this construction is orders above it:
#: reading one fold out moves a value by ``1e-2``.
EXACT = 1e-12


def _fit(frame: Any, *, evaluation: Any = None, **overrides: Any) -> Any:
    estimator = DRTMLE(**{**SETTINGS, **overrides}, evaluation=evaluation)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def _reduction(result: Any) -> Any:
    return result.repeats[0].fluctuations["mean"].reduction


def _same(actual: Any, desired: Any) -> None:
    """Floating-point equality, ``rtol=0`` -- see :data:`EXACT`."""
    np.testing.assert_allclose(actual, desired, rtol=0.0, atol=EXACT)


def _held_out(result: Any, fold: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(result.nuisance.folds.assignment) == fold)


@pytest.fixture(scope="module")
def frame() -> Any:
    return make_linear_ate(N, seed=11)[0]


@pytest.fixture(scope="module")
def plain(frame: Any) -> Any:
    """The ordinary fit, with no companion declared."""
    return _fit(frame)


@pytest.fixture(scope="module")
def paired(frame: Any) -> Any:
    """The same fit, handed its **own** rows as the evaluation sample.

    Which is what makes the identity below checkable at all: on any other draw there is no
    production array for a companion slab to be compared against.
    """
    return _fit(frame, evaluation=frame)


class TestTheCompanionChangesNothing:
    """A declared companion is inert, in every array and every diagnostic."""

    def test_the_point_estimate_and_interval_are_bit_for_bit(self, plain: Any, paired: Any) -> None:
        for name in SETTINGS["estimands"]:
            assert paired.estimates[name].psi == plain.estimates[name].psi
            assert paired.estimates[name].std_error == plain.estimates[name].std_error

    def test_the_whole_influence_curve_is_bit_for_bit(self, plain: Any, paired: Any) -> None:
        for name in SETTINGS["estimands"]:
            np.testing.assert_array_equal(
                paired.estimates[name].influence_curve, plain.estimates[name].influence_curve
            )

    def test_the_targeted_state_and_every_epsilon_are_bit_for_bit(
        self, plain: Any, paired: Any
    ) -> None:
        one, two = plain.repeats[0].fluctuations["mean"], paired.repeats[0].fluctuations["mean"]
        np.testing.assert_array_equal(two.epsilon, one.epsilon)
        np.testing.assert_array_equal(two.targeted.observed, one.targeted.observed)
        np.testing.assert_array_equal(two.mechanism.propensity, one.mechanism.propensity)
        np.testing.assert_array_equal(two.reduction.epsilon, one.reduction.epsilon)
        np.testing.assert_array_equal(two.reduction.reduced.qr, one.reduction.reduced.qr)

    def test_the_loop_took_the_same_route(self, plain: Any, paired: Any) -> None:
        one, two = _reduction(plain), _reduction(paired)
        assert two.rounds == one.rounds
        assert two.exit_reason == one.exit_reason
        assert two.closing == one.closing

    def test_a_fit_without_one_carries_no_companion(self, plain: Any) -> None:
        assert plain.nuisance.companion is None
        assert _reduction(plain).evaluation is None


class TestTheCompanionAtTheFittingRowsIsTheFit:
    """Fold ``k``'s slab, read where fold ``k`` holds out, is the production array.

    Asserted at :data:`EXACT` with ``rtol=0``, which is floating-point equality for these
    arrays rather than a statistical window -- see the constant for why the last bit is not
    reachable.
    """

    def test_the_initial_mechanism_matches_fold_by_fold(self, paired: Any) -> None:
        companion = paired.nuisance.companion
        production = paired.nuisance.propensity.values
        for fold in range(companion.n_folds):
            rows = _held_out(paired, fold)
            _same(companion.propensity[fold].values[rows], production[rows])

    def test_the_initial_regression_matches_fold_by_fold(self, paired: Any) -> None:
        companion = paired.nuisance.companion
        production = paired.nuisance.outcome
        for fold in range(companion.n_folds):
            rows = _held_out(paired, fold)
            _same(companion.outcome[fold].observed[rows], production.observed[rows])
            for arm, values in production.arms.items():
                _same(companion.outcome[fold].arms[arm][rows], values[rows])

    def test_the_initial_reduced_regressions_match_fold_by_fold(self, paired: Any) -> None:
        companion = paired.nuisance.companion
        production = paired.nuisance.reduced
        assert len(companion.reduced) == companion.n_folds
        for fold in range(companion.n_folds):
            rows = _held_out(paired, fold)
            for name in ("qr", "gr1", "gr2"):
                _same(getattr(companion.reduced[fold], name)[rows], getattr(production, name)[rows])

    def test_the_targeted_regression_matches_fold_by_fold(self, paired: Any) -> None:
        """The claim the whole lockstep exists for: the *moved* arrays agree too."""
        record = _reduction(paired)
        targeted = paired.repeats[0].fluctuations["mean"].targeted
        for fold in range(record.evaluation.n_folds):
            rows = _held_out(paired, fold)
            moved = record.evaluation.outcome[fold]
            _same(moved.observed[rows], targeted.observed[rows])
            for arm, values in targeted.arms.items():
                _same(moved.arms[arm][rows], values[rows])

    def test_the_targeted_mechanism_matches_fold_by_fold(self, paired: Any) -> None:
        record = _reduction(paired)
        targeted = paired.repeats[0].fluctuations["mean"].mechanism.propensity
        for fold in range(record.evaluation.n_folds):
            rows = _held_out(paired, fold)
            _same(record.evaluation.propensity[fold].arm(1.0)[rows], targeted[rows])

    def test_the_final_reduced_regressions_match_fold_by_fold(self, paired: Any) -> None:
        record = _reduction(paired)
        for fold in range(record.evaluation.n_folds):
            rows = _held_out(paired, fold)
            for name in ("qr", "gr1", "gr2"):
                _same(
                    getattr(record.evaluation.reduced[fold], name)[rows],
                    getattr(record.reduced, name)[rows],
                )


class TestTheIdentityIsNotVacuous:
    """The slabs differ from one another, so agreeing on one's own fold has content.

    Without this, an implementation that predicted every row with one model -- or that
    returned the production array ``K`` times -- would satisfy every equality above.
    """

    def test_two_folds_disagree_away_from_their_own_rows(self, paired: Any) -> None:
        companion = paired.nuisance.companion
        assert companion.n_folds >= 2
        first, second = companion.propensity[0].arm(1.0), companion.propensity[1].arm(1.0)
        assert not np.allclose(first, second)

    def test_the_targeted_slabs_differ_too(self, paired: Any) -> None:
        evaluation = _reduction(paired).evaluation
        first = evaluation.outcome[0].arms[1.0]
        second = evaluation.outcome[1].arms[1.0]
        assert not np.allclose(first, second)


class TestTheFoldWeightsAreTheEstimators:
    """The averaging convention is ``n_k / n`` -- the **held-out counts** -- not equal weights.

    The two coincide only on a balanced split, so an implementation that returned
    ``1/K`` would pass every other test in this module and be silently wrong wherever
    ``n`` is not a multiple of ``K`` or the split is stratified.  This is the one place
    the convention is asserted; :mod:`tests.unit.test_drtmle_crossfit` says so and does
    not repeat it.
    """

    def test_the_weights_are_the_held_out_counts(self, paired: Any) -> None:
        companion = paired.nuisance.companion
        counts = np.bincount(np.asarray(paired.nuisance.folds.assignment), minlength=FOLDS)
        np.testing.assert_allclose(companion.fold_weights, counts / counts.sum())
        assert companion.fold_weights.sum() == pytest.approx(1.0)

    def test_the_counts_are_of_the_fitting_sample_and_not_the_companion(self, frame: Any) -> None:
        """A companion of a different size leaves the weights where they were.

        The weights say how much of :math:`P_0\\hat D` each fold's function accounts for,
        which is a property of the *split* -- reading them off the companion instead would
        make them uniform whatever the split did.
        """
        smaller = make_linear_ate(60, seed=12)[0]
        result = _fit(frame, evaluation=smaller)
        companion = result.nuisance.companion
        assert companion.n == 60
        assert sum(companion.fold_sizes) == N


class TestABoundActiveFitKeepsTheIdentity:
    """Where the truncation binds, an endpoint plus an ``epsilon`` is not the state.

    :attr:`~cleverly.fluctuation.iterative.Fluctuation.carried` exists because the outcome
    solve applies its tilt once per Newton step and shrinks after each, and
    :func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism` clips -- so a companion
    rebuilt from ``(initial, epsilon)`` recovers the endpoint only on a fit where nothing
    touched a bound.  ``weak_overlap`` at a forced ``g_bounds`` is the fit where something
    does, and it is the fixture on which bounded targeting was characterized.
    """

    @pytest.fixture(scope="class")
    def pinched(self) -> Any:
        frame = make_weak_overlap(N, seed=5)[0]
        return _fit(frame, evaluation=frame, g_bounds=(0.15, 0.85))

    def test_the_bound_binds_at_all(self, pinched: Any) -> None:
        """The non-failing control: without a clipped row this fixture proves nothing."""
        targeted = pinched.repeats[0].fluctuations["mean"].mechanism.propensity
        assert np.isclose(targeted.min(), 0.15) or np.isclose(targeted.max(), 0.85)

    def test_the_identity_still_holds_fold_by_fold(self, pinched: Any) -> None:
        record = _reduction(pinched)
        targeted = pinched.repeats[0].fluctuations["mean"]
        for fold in range(record.evaluation.n_folds):
            rows = _held_out(pinched, fold)
            _same(record.evaluation.outcome[fold].observed[rows], targeted.targeted.observed[rows])
            _same(
                record.evaluation.propensity[fold].arm(1.0)[rows],
                targeted.mechanism.propensity[rows],
            )


class TestWhatACompanionIsRefusedFor:
    """Four combinations, each of which would come back describing a fit nobody ran."""

    def test_repeats_are_refused_by_name(self, frame: Any) -> None:
        with pytest.raises(ValueError, match="repeats"):
            DRTMLE(**SETTINGS, repeats=2, evaluation=frame)

    def test_the_one_step_walk_is_refused_on_cost(self, frame: Any) -> None:
        with pytest.raises(NotImplementedError, match="one_step"):
            DRTMLE(**SETTINGS, targeting="one_step", evaluation=frame)

    def test_the_weighted_submodel_is_refused_by_name(self, frame: Any) -> None:
        with pytest.raises(NotImplementedError, match="target_weights"):
            DRTMLE(**SETTINGS, target_weights=True, evaluation=frame)

    def test_a_companion_on_different_covariates_is_refused(self, frame: Any) -> None:
        renamed = frame.rename(columns={"W1": "Z1"})
        with pytest.raises(Exception, match=r"W1|Z1|covariate"):
            _fit(frame, evaluation=renamed)

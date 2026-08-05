r"""The population reference for the reduced regressions, and the guards around it.

``benchmarks/drtmle_reference.py`` is ``docs/roadmap.md``'s **E2** instrument: the three
reduced regressions at their population limits on a *continuous* law, injected through
:attr:`~cleverly.estimators.targeting.ReductionSpec.refit`.  It is a **numerical reference
and not an oracle**, and almost everything here exists because that distinction has teeth.

**The motivating measurement, taken before these tests were written.**  On one Tier-1
``q-drift`` draw at ``n = 600``, a deliberately coarse eight-bin reference reached
``psi = +1.5108`` against the spline reference's ``+1.5129`` and a truth of ``+1.5000`` --
*closer to the truth than the good reference*, while being a far worse estimate of the
reduced functions and taking the alternation to its outer cap rather than to tolerance.  So
"the final estimate moved sensibly" is not evidence that a reference is any good, and a gate
that reads the estimate rather than the function would have passed the bad one.  That is the
whole argument for :class:`TestACoarseReferenceIsRejectedOnTheFunction`.

What is **not** here is a fidelity gate built from a refinement difference.  The movement
between two knot counts is the statistic ``docs/roadmap.md`` withdrew for the quadrature
ladder and then for the branches; E2 inheriting it would rebuild the mistake it exists to
repair.  The knot ladder is a stability column and the gate is elsewhere.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from benchmarks import drtmle_injection as injection
from benchmarks import drtmle_remainder as remainder
from benchmarks.drtmle_reference import (
    POINTS_PER_PARAMETER,
    EqualCountBins,
    ReferenceReductionDRTMLE,
    SaturatedCells,
    SplineProjection,
    held_out_risk,
    reference_reductions,
)

from cleverly import DRTMLE

#: Small enough to keep the fast tier fast and large enough to clear the reference's own
#: points-per-parameter budget: ``spline(16)`` has 19 parameters and so needs 1,216 rows,
#: while ``qr``'s ``| A = a`` mask keeps roughly half of the ``2 * points`` block.
POINTS = 2_048
N = 400
CELL = "q-drift"


@pytest.fixture(scope="module")
def law() -> Any:
    return injection.base_law()


@pytest.fixture(scope="module")
def stack(law: Any) -> Any:
    """Two Sobol blocks: one the reference is fitted on, one it must not be.

    Separate blocks rather than one, because the reference's error propagates into the fit
    *deterministically* -- sharing a scramble with the block :math:`P_0\\hat D` is integrated
    on would make the two the same random variable with a covariance nobody can sign.
    """
    return remainder.stacked_companion(law, points=POINTS, scrambles=(7, 8))


@pytest.fixture(scope="module")
def fitted(law: Any, stack: Any) -> dict[str, Any]:
    """One fit per reduction arm, shared across every test that reads a fit."""
    frame, _ = law.sample(N, seed=101)
    keywords = dict(
        injection.settings(CELL, N),
        reduced_outcome_learner="glm",
        reduced_treatment_learner="glm",
        random_state=5,
        evaluation=stack.frame,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {
            "glm": DRTMLE(**keywords).fit(frame, outcome="Y", treatment="A").single(),
            "spline": ReferenceReductionDRTMLE(
                dgp=law,
                reference=SplineProjection(16),
                window=stack.blocks[0].window,
                row_weights=stack.weights,
                **keywords,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single(),
        }
    return out


def _weights(size: int) -> np.ndarray:
    return np.full(size, 0.5)


class TestTheSmoothersAreWhatTheyClaim:
    """Each candidate against a target whose truth is known, with no estimator involved."""

    def test_the_spline_recovers_a_smooth_function(self) -> None:
        index = np.linspace(0.05, 0.95, 4_000)
        target = np.sin(3.0 * index) + 0.5 * index**2
        fitted = SplineProjection(16).fit(index, target, _weights(index.size))

        assert float(np.max(np.abs(fitted(index) - target))) < 1e-4

    def test_the_saturated_smoother_is_exact_on_a_discrete_index(self) -> None:
        """Which is why the exact-law control needs it rather than the spline.

        When the index takes finitely many values, conditioning on it is conditioning on a
        discrete variable and the conditional expectation is a finite sum -- *exactly*, not to
        a tolerance.  A spline is not exact there, so reusing the reference as its own control
        would compare an approximation against itself.
        """
        index = np.repeat([0.2, 0.5, 0.9], 400)
        target = np.repeat([1.0, -2.0, 3.5], 400) + np.tile([0.1, -0.1], 600)
        fitted = SaturatedCells().fit(index, target, _weights(index.size))

        expected = np.repeat([1.0, -2.0, 3.5], 400)
        np.testing.assert_allclose(fitted(index), expected, rtol=0, atol=1e-12)

    def test_the_coarse_control_is_genuinely_coarse(self) -> None:
        """The negative control has to *fail*, or the gate that rejects it proves nothing."""
        index = np.linspace(0.05, 0.95, 4_000)
        target = np.sin(3.0 * index) + 0.5 * index**2
        coarse = EqualCountBins(8).fit(index, target, _weights(index.size))
        fine = SplineProjection(16).fit(index, target, _weights(index.size))

        assert float(np.max(np.abs(coarse(index) - target))) > 1e-2
        assert float(np.max(np.abs(coarse(index) - target))) > 100.0 * float(
            np.max(np.abs(fine(index) - target))
        )


class TestTheReferenceIsAWeightedProjection:
    """Two structural properties the fidelity decomposition rests on."""

    def test_it_is_linear_in_the_target(self) -> None:
        r"""``ref(aT1 + bT2) == a ref(T1) + b ref(T2)``, exactly.

        The held-out-risk gate's cross term vanishes because the reference *is* a weighted
        :math:`L_2` projection.  A clip, a link or a robust loss entering later would keep
        every array in range and quietly break that, so linearity is pinned rather than
        assumed.
        """
        index = np.linspace(0.05, 0.95, 4_000)
        weights = _weights(index.size)
        first, second = np.sin(4.0 * index), index**3
        spec = SplineProjection(16)

        combined = spec.fit(index, 0.3 * first - 1.7 * second, weights)(index)
        apart = 0.3 * spec.fit(index, first, weights)(index) - 1.7 * spec.fit(
            index, second, weights
        )(index)
        np.testing.assert_allclose(combined, apart, rtol=1e-9, atol=1e-12)

    def test_it_does_not_depend_on_row_order(self) -> None:
        index = np.linspace(0.05, 0.95, 4_000)
        target = np.sin(3.0 * index)
        weights = np.linspace(0.2, 0.8, index.size)
        order = np.random.default_rng(0).permutation(index.size)
        spec = SplineProjection(16)

        straight = spec.fit(index, target, weights)(index)
        shuffled = spec.fit(index[order], target[order], weights[order])(index)
        np.testing.assert_allclose(straight, shuffled, rtol=1e-9, atol=1e-12)

    def test_a_basis_thinner_than_its_budget_is_refused(self) -> None:
        """Refused rather than warned about, and refused rather than silently widened.

        A near-interpolating reference reports its own variance as the target's structure,
        which is the same coupling ``validation-plan.md`` records for a bin count raised
        without the rows behind it.
        """
        spec = SplineProjection(32)
        rows = POINTS_PER_PARAMETER * spec.n_parameters - 1
        index = np.linspace(0.05, 0.95, rows)

        with pytest.raises(ValueError, match="rows per parameter"):
            spec.fit(index, np.sin(index), _weights(rows))


class TestTheWeightsMustBeTheLaws:
    """The guard that makes "fitted on the reference block" structural rather than hoped."""

    def test_a_draw_block_is_refused(self, law: Any, stack: Any) -> None:
        r"""An i.i.d. block carries ones where a quasi-random block carries :math:`g_0(a|W)`.

        Everything the provider does with the arm rests on the weights being the law's own
        conditional probabilities: ``gr1``'s target is the bare indicator and its limit is
        :math:`E[g_0(a|W) \mid \cdot]` **only** because the weighted average over a Sobol
        point's two rows turns one into the other.  Point the window at a draw block and
        every array stays in range while the limit answers for the wrong measure -- so this
        is checked rather than documented.
        """
        mixed = remainder.stacked_companion(
            law, points=POINTS, scrambles=(7,), draw_rows=1_000, draw_seeds=(3,)
        )
        draw_block = next(block for block in mixed.blocks if block.rule == "draw")
        sobol_block = next(block for block in mixed.blocks if block.rule == "sobol")

        # The guard fires on the weights alone, before anything is fitted, so no
        # `NuisanceEstimates` stand-in is needed -- only the law's own `g_0(1 | W)` at the
        # companion rows, which is what a correct block's weights are.
        from benchmarks.drtmle_reference import _check_the_weights_are_the_laws

        columns = [name for name in mixed.frame.columns if name not in ("A", "Y")]
        covariates = np.column_stack(
            [np.asarray(mixed.frame[name], dtype=float) for name in columns]
        )
        truth = np.asarray(law.propensity(covariates), dtype=float)
        indicator = (np.asarray(mixed.frame["A"], dtype=float) == 1.0).astype(float)

        # The Sobol block passes, which is what makes the refusal below a discrimination
        # rather than a guard that rejects everything.
        _check_the_weights_are_the_laws(mixed.weights, truth, indicator, sobol_block.window, 1.0)
        with pytest.raises(ValueError, match="not the law's own"):
            _check_the_weights_are_the_laws(mixed.weights, truth, indicator, draw_block.window, 1.0)


class TestTheEstimatorDoesNotCollideWithItsBaseClass:
    """``TMLE.reference`` is the reference **arm**, and this class must not shadow it."""

    def test_the_reference_attribute_is_still_an_arm(self, law: Any, stack: Any) -> None:
        estimator = ReferenceReductionDRTMLE(
            dgp=law,
            reference=SplineProjection(16),
            window=stack.blocks[0].window,
            row_weights=stack.weights,
            **injection.settings(CELL, N),
        )
        # Whatever `TMLE.__init__` put there -- a level or `None` -- and never a projection.
        assert not isinstance(estimator.reference, SplineProjection)


class TestTheReferenceIsRecomputedAtTheCurrentPair:
    """The mutation: a provider closing over the *initial* pair answers a different question.

    It would still pass a great deal, which is why this is the mutation this module watches.
    Equations (9) and (10) are stated at **starred** reductions, so ``refit`` is handed the
    targeted nuisances every round precisely so the reductions move with them.
    """

    def test_the_provider_moves_when_its_argument_moves(
        self, law: Any, stack: Any, fitted: dict[str, Any]
    ) -> None:
        result = fitted["spline"]
        nuisance = result.nuisance
        window, weights = stack.blocks[0].window, stack.weights
        bounds = nuisance.reduced.g_bounds

        def build(state: Any) -> Any:
            return reference_reductions(
                state,
                dgp=law,
                reference=SplineProjection(16),
                window=window,
                row_weights=weights,
                g_bounds=bounds,
            )[0]

        from dataclasses import replace

        def shift(fit: Any, amount: float) -> Any:
            return fit.__class__(
                arms={arm: values + amount for arm, values in fit.arms.items()},
                observed=fit.observed + amount,
            )

        first = build(nuisance)

        # **The companion side is the one that matters**, and it is the half a weaker version
        # of this test misses. The reference is *fitted* on the companion's copies, so a
        # provider that rebuilt only its production predictions -- or one that closed over the
        # initial companion -- would still respond to the production designs below while
        # answering with a function belonging to an earlier round.
        at_companion = replace(
            nuisance,
            companion=replace(
                nuisance.companion,
                outcome=tuple(shift(each, 0.05) for each in nuisance.companion.outcome),
            ),
        )
        assert float(np.max(np.abs(build(at_companion).gr1 - first.gr1))) > 1e-6, (
            "gr1 is fitted on the companion's Qbar-hat and must move when that moves"
        )

        # And the production side, which is what the fitted function is evaluated at.
        at_production = replace(nuisance, outcome=shift(nuisance.outcome, 0.05))
        assert float(np.max(np.abs(build(at_production).gr1 - first.gr1))) > 1e-6, (
            "gr1 is evaluated at the production Qbar-hat and must move when that moves"
        )

    def test_the_alternation_actually_ran(self, fitted: dict[str, Any]) -> None:
        """Anti-vacuity for the test above: with `epsilon` at zero it would be a tautology."""
        reduction = fitted["spline"].repeats[0].fluctuations["mean"].reduction
        assert reduction.rounds > 1
        assert float(np.max(np.abs(np.asarray(reduction.epsilon, dtype=float)))) > 1e-8


@pytest.fixture(scope="module")
def learned(law: Any, stack: Any) -> Any:
    """A fit whose per-fold companion nuisances actually **differ**, unlike Tier 1's.

    This fixture exists because of a measurement rather than a preference.  Tier 1 *injects*
    both nuisances as analytic functions of ``W``, so every outer fold's companion copy is
    the same array to the bit -- the sum of fold ``k``'s companion mechanism over the
    reference block came back ``1402.0213`` for all five folds.  Fold routing is therefore
    **invisible** on Tier 1, in the same way ``CLAUDE.md`` records a saturated learner making
    a column that is a function of the others invisible.  A routing test taken there would
    have passed against any routing whatever.

    So the routing test takes a fit with *fitted* primaries at the same law and size.
    """
    frame, _ = law.sample(N, seed=101)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            DRTMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                reduced_outcome_learner="glm",
                reduced_treatment_learner="glm",
                n_folds=5,
                learner_folds=3,
                simultaneous=False,
                estimands=("ate", "ey1", "ey0"),
                random_state=5,
                evaluation=stack.frame,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )


class TestTheProviderRoutesFoldsAndArms:
    """Where a value lands, checked structurally rather than through the estimate."""

    def test_every_production_row_is_read_from_its_own_folds_reference(
        self, law: Any, stack: Any, learned: Any
    ) -> None:
        """Fold ``k``'s reference predicts at fold ``k``'s rows and at no others.

        Pinned with a reference whose value identifies the rows it was fitted on, so the
        production column says which fold's model produced it -- the structural idiom
        ``test_sequential_design.py`` uses, and the right one here because two correct-looking
        routings differ only in which rows they touch.
        """
        nuisance = learned.nuisance
        assignment = np.asarray(nuisance.folds.assignment, dtype=int)
        companion = nuisance.companion
        window = stack.blocks[0].window

        class IndexStamp:
            """Returns a constant that identifies the rows it was fitted on.

            The stamp is read off the *data* rather than off a call counter, so the test does
            not have to assume the provider's loop order -- which is exactly the thing a
            routing test must not take on trust.
            """

            label = "index-stamp"

            def fit(self, index: np.ndarray, target: np.ndarray, weights: np.ndarray) -> Any:
                mine = float(np.sum(np.asarray(index, dtype=float)))
                return lambda values: np.full(np.asarray(values).size, mine)

        produced, _ = reference_reductions(
            nuisance,
            dgp=law,
            reference=IndexStamp(),
            window=window,
            row_weights=stack.weights,
            g_bounds=nuisance.reduced.g_bounds,
        )

        # `qr` conditions on the mechanism, so fold k's stamp is the sum of fold k's
        # companion mechanism over the reference block's rows at *this column's* arm.
        # Column 0 is `arms[0]`, which is `0.0` and not `1.0` -- reading it as the treated
        # arm is the slip this comment exists to stop, and it fails as a mismatch rather
        # than as an error. Recomputed longhand rather than read back out of the provider.
        arm = produced.arms[0]
        treatment = np.asarray(companion.data.treatment, dtype=float)
        block = np.zeros(treatment.size, dtype=bool)
        block[window.start : window.stop] = True
        taken = block & (treatment == float(arm))
        expected = {
            fold: float(np.sum(np.asarray(companion.propensity[fold].arm(arm))[taken]))
            for fold in range(companion.n_folds)
        }
        assert len(set(expected.values())) == companion.n_folds, (
            "the folds' companion mechanisms must differ, or this test cannot tell them apart"
        )

        for fold, stamp in expected.items():
            mine = assignment == fold
            assert mine.any()
            np.testing.assert_allclose(produced.qr[mine, 0], stamp, rtol=1e-9)

    def test_the_arms_get_their_own_columns(self, learned: Any) -> None:
        produced = learned.nuisance.reduced
        assert produced.qr.shape[1] == len(produced.arms)
        assert not np.allclose(produced.qr[:, 0], produced.qr[:, 1])


class TestACoarseReferenceIsRejectedOnTheFunction:
    """The gate must read the *function*, because the estimate does not separate the two.

    This is the class the module docstring's measurement motivates: an eight-bin reference
    landed a final ``psi`` as close to the truth as a good one.  So the rejection has to come
    from the reduced functions themselves.
    """

    def test_the_coarse_reference_is_worse_on_the_function_it_estimates(self) -> None:
        index = np.linspace(0.05, 0.95, 8_000)
        weights = np.full(index.size, 0.5)
        # The shape a reduced regression actually has here: smooth and monotone in the
        # conditioning nuisance, which is what both candidates are asked to recover.
        target = 1.0 / (1.0 + np.exp(-6.0 * (index - 0.5)))

        coarse = EqualCountBins(8).fit(index, target, weights)(index)
        fine = SplineProjection(16).fit(index, target, weights)(index)
        loss = lambda values: float(np.average((values - target) ** 2, weights=weights))  # noqa: E731

        assert loss(coarse) > 25.0 * loss(fine)


class TestTheHeldOutRiskIsTwoSided:
    """The gate has to reject a reference that is too **fine** as well as one too coarse.

    This is the class that says the gate is not a refinement difference in disguise -- and the
    reason is narrower than it first looks, which is worth writing down because the wide
    version is false.  A knot ladder's movement *does* rise with over-fitting on this law;
    what it cannot do is **orient**.  It is a magnitude, so it says two rungs disagree and not
    which is nearer, it is undefined for the rung a ladder starts on, and measured here it
    overstates the true weighted error by an order of magnitude at every rung -- the same
    failure ``delta`` was withdrawn for.  A held-out risk ranks the candidates, because the
    rows it is scored on are rows the candidate did not see.
    """

    @staticmethod
    def _split() -> tuple[np.ndarray, ...]:
        """Two independent blocks from one law: one to fit on, one to score on."""
        rng = np.random.default_rng(11)
        fit_index = rng.uniform(0.02, 0.98, 6_000)
        score_index = rng.uniform(0.02, 0.98, 6_000)
        mean = lambda u: 1.0 / (1.0 + np.exp(-6.0 * (u - 0.5)))  # noqa: E731
        fit_target = mean(fit_index) + rng.normal(0.0, 0.35, fit_index.size)
        score_target = mean(score_index) + rng.normal(0.0, 0.35, score_index.size)
        return fit_index, fit_target, score_index, score_target

    def test_a_coarse_reference_scores_worse(self) -> None:
        fit_index, fit_target, score_index, score_target = self._split()
        weights = np.full(fit_index.size, 0.5)

        coarse = EqualCountBins(4).fit(fit_index, fit_target, weights)
        good = SplineProjection(8).fit(fit_index, fit_target, weights)

        assert held_out_risk(coarse, score_index, score_target, weights) > held_out_risk(
            good, score_index, score_target, weights
        )

    def test_an_over_fitted_reference_also_scores_worse(self) -> None:
        """The half a ladder cannot see, and the reason this gate exists.

        A near-interpolating basis has a *smaller* movement against the next rung down and a
        *larger* held-out risk.  So refinement and fidelity point in opposite directions here,
        and only one of them is a gate.
        """
        fit_index, fit_target, score_index, score_target = self._split()
        weights = np.full(fit_index.size, 0.5)

        good = SplineProjection(8).fit(fit_index, fit_target, weights)
        # Deliberately far past the points-per-parameter budget, which is why it is built
        # by hand here rather than through the guarded constructor path.
        greedy = SplineProjection(90).fit(fit_index, fit_target, weights)

        assert held_out_risk(greedy, score_index, score_target, weights) > held_out_risk(
            good, score_index, score_target, weights
        )

    def test_the_risk_difference_tracks_the_squared_error_difference(self) -> None:
        """The identity the gate rests on, checked rather than argued.

        The cross term vanishes because the reference is a weighted `L2` projection, so a
        *difference* of held-out risks estimates a difference of squared errors against the
        truth -- which is knowable here because this law's conditional mean is written down.
        """
        fit_index, fit_target, score_index, _ = self._split()
        weights = np.full(fit_index.size, 0.5)
        mean = lambda u: 1.0 / (1.0 + np.exp(-6.0 * (u - 0.5)))  # noqa: E731

        coarse = EqualCountBins(4).fit(fit_index, fit_target, weights)
        good = SplineProjection(8).fit(fit_index, fit_target, weights)
        noise = np.random.default_rng(3).normal(0.0, 0.35, score_index.size)
        score_target = mean(score_index) + noise

        risk_gap = held_out_risk(coarse, score_index, score_target, weights) - held_out_risk(
            good, score_index, score_target, weights
        )
        truth_gap = float(
            np.mean((coarse(score_index) - mean(score_index)) ** 2)
            - np.mean((good(score_index) - mean(score_index)) ** 2)
        )
        assert risk_gap == pytest.approx(truth_gap, abs=0.01)

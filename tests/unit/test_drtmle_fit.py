r"""``DRTMLE`` as an estimator: what it reports, what it refuses, and what it must not move.

The statistical claim -- that the interval stays valid when one nuisance is inconsistent --
is a coverage statement and belongs in the nightly tier.  What is checkable here is
everything around it: that all three score equations are solved and reported, that the
*point* estimate is a plain TMLE's, that the reported nuisances are the ones that were
fitted, and that every combination the derivation does not cover is refused by name rather
than approximated.

One fit, shared: each class below reads a different part of the same result.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cleverly import CTMLE, DRTMLE, TMLE
from cleverly.data import CausalData
from cleverly.datasets import nonlinear_dgp
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.serialize import load
from cleverly.estimators.targeting import _solved, build_submodel
from cleverly.estimators.tmle import DEFAULT_NUISANCE_BOUND
from cleverly.inference.influence import counterfactual_means, reduced_corrections
from tests.conftest import FAST_KWARGS

#: The mean group, which is what the reduced-dimension regressions are derived for. A
#: default binary report also asks for ``att`` and ``atc``, which are refused -- spelled out
#: here for the same reason ``tests/e2e/test_ctmle.py`` spells them out.
ESTIMANDS = ("ate", "ey1", "ey0")

SETTINGS = {**FAST_KWARGS, "estimands": ESTIMANDS}


def frame():
    """600 rows, which is the *cheaper* end here rather than the more generous one.

    The alternation refits three reduced regressions per arm on every round, so this
    module's cost is round count times folds and barely depends on ``n``. Fewer rows makes
    the nuisances noisier, the coupling looser and the loop *longer*: 400 rows measured 28s
    against 600's 25s, and 3 folds instead of 5 measured 38s. Do not "optimise" either down.

    The closing pass adds a bounded number of further solves and refits nothing, so it costs
    arithmetic rather than folds and does not enter that arithmetic.
    """
    sample, _ = nonlinear_dgp().sample(600, seed=3)
    return sample


@pytest.fixture(scope="module")
def fit():
    return DRTMLE(**SETTINGS).fit(frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def ordinary():
    return TMLE(**SETTINGS).fit(frame(), outcome="Y", treatment="A").single()


class TestWhatItReports:
    def test_it_reports_the_mean_group_under_its_own_names(self, fit) -> None:
        """A different estimator behind the same parameters, exactly as ``CTMLE`` is."""
        assert set(fit.estimates) == set(ESTIMANDS)

    def test_all_three_score_equations_are_reported_and_solved(self, fit) -> None:
        check = fit.validation.score_check()
        names = {row.name for row in check.rows}

        assert {"mean", "mean (mechanism)", "mean (reduced)"} <= names
        assert check.passed

    def test_the_verdict_is_reachable_without_knowing_the_subsystem(self, fit) -> None:
        """``res.score_verdict`` is the same object, derived rather than stored."""
        assert fit.score_verdict.rows == fit.validation.score_check().rows
        assert fit.score_verdict.passed

    def test_a_passing_fit_adds_no_line_to_the_summary(self, fit) -> None:
        """The verdict is silent on the common path; only a failure interrupts a reader."""
        assert "score check" not in fit.summary()

    def test_the_alternation_terminated_on_its_own(self, fit) -> None:
        """On *this* process, and the qualification is the point.

        A converged exit is not something this loop reliably does, and
        :class:`TestTheExtraEquationsAreIllConditionedWhereTheMechanismIsRight` is the
        case where it does not. Here it does, so the round count is worth pinning as a
        floor on how well it can behave; ``score_check`` is what decides whether an exit
        at the cap matters.
        """
        reduction = fit.repeats[0].fluctuations["mean"].reduction
        assert 1 <= reduction.n_outer < 50
        assert reduction.exit_reason != "cap"
        assert reduction.failure is None
        assert reduction.ill_conditioned == 0

    def test_the_joint_likelihood_never_decreases(self, fit) -> None:
        """Coordinate ascent on one likelihood is why this terminates rather than settles."""
        joint = [row[4] for row in fit.repeats[0].fluctuations["mean"].reduction.trace]
        assert all(later >= earlier - 1e-9 for earlier, later in pairwise(joint))

    def test_it_records_what_it_fitted(self, fit) -> None:
        report = fit.extra["drtmle"]
        assert report.guard == ("Q", "g")
        assert report.reduction == "univariate"
        assert set(report.diagnostics) == {"qr", "gr1", "gr2"}


class TestThePointEstimateIsAPlainTMLEs:
    r"""The extra equations move the *variance*, and they move it by design.

    All three empirical means are driven to zero, so no extra term can move
    :math:`\hat\Psi`; what the reductions buy is an influence curve entitled to be believed
    when only one nuisance is consistent.  Saying so in a test is the cheapest way to stop a
    reader taking the interval's change for an improvement in the estimate.
    """

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_estimates_agree_closely(self, fit, ordinary, name: str) -> None:
        """Within a fifth of a standard error, and **not** an equality.

        The extra fluctuations do move ``Qbar*``, and the plug-in is its mean, so the two
        estimators do not agree exactly and it would be wrong to pin them as though they
        did: measured at 0.08 standard errors for ``ate`` here. What cannot differ is the
        *estimand*. The threshold is set where a bug that moved the answer by a standard
        error would fail and this arithmetic does not.
        """
        difference = abs(fit.estimates[name].psi - ordinary.estimates[name].psi)
        assert difference < 0.2 * ordinary.estimates[name].std_error

    def test_the_standard_errors_do_not_all_agree(self, fit, ordinary) -> None:
        moved = [
            abs(fit.estimates[name].std_error / ordinary.estimates[name].std_error - 1.0)
            for name in ESTIMANDS
        ]
        assert max(moved) > 1e-3, "the extra terms are what the variance is for"


class TestTheReportedNuisancesAreTheFittedOnes:
    """The targeted mechanism and the refitted reductions live on the fluctuation."""

    def test_the_mechanism_on_the_result_is_the_initial_one(self, fit) -> None:
        targeted = fit.repeats[0].fluctuations["mean"].mechanism.propensity
        initial = fit.nuisance.propensity.arm(1.0)

        assert not np.allclose(targeted, initial), "targeting must have moved something"
        assert fit.nuisance.reduced is not None

    def test_the_reductions_on_the_result_are_the_initial_ones(self, fit) -> None:
        """``result.nuisance.reduced`` is the fit's, not the alternation's last refit."""
        final = fit.repeats[0].fluctuations["mean"].reduction.reduced
        assert not np.allclose(final.qr, fit.nuisance.reduced.qr)


class TestAnEmptyGuardIsAPlainTMLE:
    r"""Bit for bit, and by construction rather than by a loop that exits early.

    A ``guard=()`` fit fits no reduced regressions at all, so ``needs_reduction`` is false
    and the targeting goes down exactly the path it went down before this class existed.
    That is the canary for an alternation that leaks into the ordinary estimator.
    """

    def test_it_reproduces_the_ordinary_estimator(self, ordinary) -> None:
        bare = DRTMLE(guard=(), **SETTINGS).fit(frame(), outcome="Y", treatment="A").single()

        for name in ESTIMANDS:
            assert bare.estimates[name].psi == ordinary.estimates[name].psi
            assert bare.estimates[name].std_error == ordinary.estimates[name].std_error
            np.testing.assert_array_equal(
                bare.estimates[name].influence_curve, ordinary.estimates[name].influence_curve
            )
        fluctuation = bare.repeats[0].fluctuations["mean"]
        np.testing.assert_array_equal(
            fluctuation.epsilon, ordinary.repeats[0].fluctuations["mean"].epsilon
        )
        assert fluctuation.reduction is None and fluctuation.mechanism is None
        assert bare.nuisance.reduced is None


class TestTheCurveReadsWhatTheAlternationLeft:
    r"""The targeted mechanism and the refitted reductions, not the fit's own arrays.

    Both live on the fluctuation and both moved; a curve built from ``result.nuisance``
    would be the curve of a fit nobody ran, and would still have mean zero and still look
    entirely reasonable.
    """

    def test_it_divides_by_the_targeted_mechanism(self, fit) -> None:
        data = CausalData.from_frame(frame(), outcome="Y", treatment="A", covariates=None)
        fluctuation = fit.repeats[0].fluctuations["mean"]
        reduction = fluctuation.reduction
        scaled = fit.nuisance.scaler.scale(data.outcome)

        at_targeted = reduced_corrections(
            scaled,
            fluctuation.targeted,
            data.treatment,
            reduction.reduced,
            fluctuation.mechanism.propensity,
            bounds=reduction.bounds,
            observed=data.observed,
        )
        at_initial = reduced_corrections(
            scaled,
            fluctuation.targeted,
            data.treatment,
            reduction.reduced,
            fit.nuisance.propensity.arm(1.0),
            bounds=reduction.bounds,
            observed=data.observed,
        )
        assert np.max(np.abs(at_targeted[1.0] - at_initial[1.0])) > 1e-6

        # Both sides on the [0, 1] scale the curve is built on; the reported one has been
        # mapped back to the outcome's own by `TargetContext.finish`, which is a scaling by
        # `range` and nothing else.
        plain_curve = _plain_curve(fit, data, fluctuation)
        reported = np.asarray(fit.estimates["ey1"].influence_curve) / fit.nuisance.scaler.range
        np.testing.assert_allclose(
            reported, plain_curve - data.weights * at_targeted[1.0], atol=1e-12
        )

    def test_the_reported_curve_still_has_mean_zero(self, fit) -> None:
        """Three solved equations rather than one, so this is a check on all of them."""
        for name in ESTIMANDS:
            assert abs(float(np.mean(fit.estimates[name].influence_curve))) < 1e-7


class TestItSurvivesARoundTrip:
    def test_the_estimates_and_curves_come_back(self, fit, tmp_path) -> None:
        """The curve is what is stored, so a reloaded fit reports the doubly-robust one."""
        back = load(fit.save(tmp_path / "fit.npz"))

        for name in ESTIMANDS:
            assert back.estimates[name].psi == fit.estimates[name].psi
            np.testing.assert_array_equal(
                back.estimates[name].influence_curve, fit.estimates[name].influence_curve
            )
        assert back.nuisance.reduced is not None

    def test_the_score_check_is_the_same_check_after_a_round_trip(self, fit, tmp_path) -> None:
        """A reloaded fit answers the same question, not a narrower one.

        ``score_check`` reads ``Fluctuation.mechanism`` and ``.reduction``, so a file that
        dropped them reported **one** fluctuation row where this fit solves three -- and a
        verdict computed from one equation can pass where the verdict computed from three
        failed.  That was the state until format version 10, and the round-trip test above
        could not see it: what it round-tripped was the estimates, and the estimates were
        always fine.
        """
        back = load(fit.save(tmp_path / "verdict.npz"))

        live, after = fit.validation.score_check(), back.validation.score_check()
        assert [row.name for row in after.rows] == [row.name for row in live.rows]
        assert [row.score for row in after.rows] == [row.score for row in live.rows]
        assert after.passed == live.passed

        reduction = back.repeats[0].fluctuations["mean"].reduction
        assert reduction is not None
        np.testing.assert_array_equal(
            reduction.reduced.qr, fit.repeats[0].fluctuations["mean"].reduction.reduced.qr
        )


class TestTheAlternationCanBeIllConditioned:
    r"""Equation (10) is not always solvable to machine precision, and the reason is structural.

    Its covariate is ``gr2 / gr1``, and ``gr2 = E[(1_a - g-hat)/g-hat | Qbar]`` **vanishes
    exactly where the mechanism is right**.  So the better ``g-hat`` is, the closer that
    covariate is to zero and the worse conditioned its Newton solve: observed at
    ``mean|h| = 1e-3`` with ``|epsilon|`` reaching 280 and a singular Hessian in a third of
    the rounds, on a fit whose fold split was drawn unseeded.

    **How often that happens was measured rather than assumed, and the first measurement was
    wrong.**  Six seeded fits at ``n = 800`` on this one process reported no ill-conditioned
    solve and a worst score of ``1e-9``, which read as a minority behaviour of particular
    draws.  A 96-fit sweep -- four processes by two sizes by twelve seeds, tabulated in
    ``docs/roadmap.md`` under *How the alternation exits* -- says otherwise: the solve is
    ill-conditioned on 5 of 12 ``linear`` draws at ``n = 600`` and 9 of 12 at ``n = 1,200``,
    and highest exactly where the mechanism is easiest to get right, which is what the
    paragraph above predicts and what sweeping only hard processes would have hidden.  A fit
    that hits it reports ``failure = "max_iter_reached"`` and ``score_check`` says NO, and
    that is the diagnostic working rather than something to accommodate.

    What is asserted below is therefore the invariant that holds either way, not either
    outcome: pinning ``ill_conditioned > 0`` would be pinning a seed.

    **The closing pass changes what an exit at the cap costs, and not whether it happens.**
    Equations (8) and (10) are re-solved jointly at the reductions the curve reads, so the
    reported curve is mean-zero even on a draw the alternation could not settle -- which is
    why ``failure`` and the score check are no longer two ways of asking the same question,
    and why they are asserted apart below.
    """

    @pytest.fixture(scope="class")
    def hard(self):
        from cleverly.datasets import make_nonlinear_ate

        sample, _ = make_nonlinear_ate(n=600, seed=0)
        return (
            DRTMLE(**{**SETTINGS, "estimands": ("ate",)})
            .fit(sample, outcome="Y", treatment="A")
            .single()
        )

    def test_the_conditioning_is_reported_either_way(self, hard) -> None:
        """Whatever the loop did, it is on the record rather than inferred.

        ``failure`` is no longer implied by the round count: the closing pass can settle the
        equations a capped alternation left open, so a fit can report ``n_outer == 50`` and
        no failure. That is the whole point of the pass, and asserting the old coupling
        would forbid it.

        Which exit fired is checked for *membership* rather than for a value: this fit is
        the one whose behaviour depends on the draw, so pinning ``"tolerance"`` here would
        pin a seed. What is pinned is the one direction that holds by construction -- a cap
        exit means the rounds ran out, where the converse does not follow, since the last
        round of a full fifty may still break on the tolerance.
        """
        reduction = hard.repeats[0].fluctuations["mean"].reduction
        assert reduction.ill_conditioned >= 0
        assert 1 <= reduction.n_outer <= 50
        assert reduction.closing > 0, "the closing pass runs on every fit that has reductions"
        assert reduction.exit_reason in {"tolerance", "stall", "cap"}
        if reduction.exit_reason == "cap":
            assert reduction.n_outer == 50
        if reduction.n_outer < 50:
            assert reduction.failure is None

    def test_and_the_score_check_passes_regardless(self, hard) -> None:
        """Because the question is whether the score matters, not whether it is tiny."""
        check = hard.validation.score_check()
        assert check.passed, check.summary()
        worst = max(abs(row.score) for row in check.rows)
        assert worst < 1e-3 * hard.estimates["ate"].std_error


class TestAnEquationStopsOnEitherRuler:
    r"""``_solved`` accepts a relative score *or* a negligible absolute one, and the second
    branch is the whole of what the exit criterion change was.

    **This is a unit test of the predicate rather than an assertion about a fit, and that is
    deliberate.**  The change it pins moves the loop's exit from ``stall`` at 30 rounds to
    ``tolerance`` at 3 on the fits measured, but it does not move the fit: the closing pass
    re-solves all three equations afterwards, so ``psi``, the curve and ``score_check`` come
    out the same either way -- ``ate`` moved by ``4.1e-5``, which is ``2.4e-4`` of a standard
    error.  That is why the whole 61-test suite passed identically before and after, and why
    a test asserting something about the *result* cannot pin this no matter how it is
    written.  What can be pinned is the predicate, and removing its absolute branch turns the
    second case below red immediately.

    Asserting ``exit_reason == "tolerance"`` on a fitted result would be the other candidate,
    and it is rejected for the reason the class above rejects it: which exit fires is a
    property of the draw, and six fits are not enough to make it a property of the estimator.
    The sweep in ``docs/roadmap.md`` had 2 of 96 reach the tolerance under the *old* rule and
    the new rule has not been swept, so pinning it here would pin a seed.

    The magnitudes are the measured ones.  On a 400-row ``linear`` fit the round the loop
    gave up at had equation (10) at ``2.3e-8`` relative -- six orders above ``spec.tol`` --
    while its absolute score was near ``1.1e-10``, against a negligible bar of ``1e-3/400``.
    That gap is the item-7 defect in two numbers.
    """

    TOL = 1e-10
    NEGLIGIBLE = 1e-3 / 400.0

    def test_a_small_relative_score_is_solved_as_it_always_was(self) -> None:
        """Equation (8)'s path, unchanged: ``1/g`` is bounded below, so the ratio decides."""
        assert _solved(relative=1e-17, absolute=1.0, tol=self.TOL, negligible=self.NEGLIGIBLE)

    def test_a_negligible_absolute_score_is_solved_though_the_ratio_is_not(self) -> None:
        """Equations (9) and (10)'s path, and the branch the change added.

        Both measured magnitudes, from the round a 400-row fit stalled on. Delete the
        absolute branch of ``_solved`` and this is the assertion that fails.
        """
        assert _solved(relative=2.3e-8, absolute=1.1e-10, tol=self.TOL, negligible=self.NEGLIGIBLE)

    def test_a_score_that_is_large_on_both_rulers_is_not_solved(self) -> None:
        """The change loosens which ruler is used, not what counts as solved on either.

        ``1e-3`` absolute is roughly the worst score the weak-overlap fits report, and they
        are the ones the diagnostic must go on failing -- see item 11.
        """
        assert not _solved(relative=2.3e-8, absolute=1e-3, tol=self.TOL, negligible=self.NEGLIGIBLE)

    def test_the_bar_tightens_with_the_sample_size(self) -> None:
        """``_NEGLIGIBLE / n``, so a score that is negligible at 400 rows need not be at 40,000.

        The bar stands in for ``score_check``'s ``tolerance * se / sqrt(n)`` with
        ``se = O(n**-0.5)``; item 12 records that this is an assumption rather than a
        measurement. What the substitution must not lose is the direction, which is what
        this pins.
        """
        absolute = 1.1e-10
        assert _solved(relative=1.0, absolute=absolute, tol=self.TOL, negligible=1e-3 / 400)
        assert not _solved(
            relative=1.0, absolute=absolute, tol=self.TOL, negligible=1e-3 / 40_000_000
        )


class TestTheRefusals:
    """Each names what the derivation would need, rather than reporting a plain number."""

    def test_a_multi_valued_treatment(self) -> None:
        """``estimands=("ate",)`` because ``ey1``/``ey0`` are refused a step earlier.

        The registry rejects a binary-only *parameter name* at three arms before any
        estimator sees the data, which is a different refusal from this one -- and reporting
        it here would leave the doubly-robust refusal untested.
        """
        sample = frame().copy()
        sample.loc[sample.index[:100], "A"] = 2
        with pytest.raises(NotImplementedError, match="binary treatment"):
            DRTMLE(**{**SETTINGS, "estimands": ("ate",)}).fit(sample, outcome="Y", treatment="A")

    def test_the_conditional_effects(self) -> None:
        with pytest.raises(NotImplementedError, match="ATT and ATC"):
            DRTMLE(**{**SETTINGS, "estimands": ("ate", "att")}).fit(
                frame(), outcome="Y", treatment="A"
            )

    def test_the_bivariate_reduction(self) -> None:
        with pytest.raises(NotImplementedError, match="bivariate"):
            DRTMLE(reduction="bivariate", **SETTINGS)

    def test_an_unknown_guard(self) -> None:
        with pytest.raises(ValueError, match="guard entries"):
            DRTMLE(guard=("Qbar",), **SETTINGS)

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("incremental", [1.5]),
            ("shifts", [0.5]),
        ],
    )
    def test_the_other_parameter_axes(self, keyword: str, value) -> None:
        with pytest.raises(NotImplementedError, match=f"{keyword}="):
            DRTMLE(**{**SETTINGS, keyword: value})

    def test_fold_wise_targeting(self) -> None:
        with pytest.raises(NotImplementedError, match="pooled only"):
            DRTMLE(targeting_scheme="fold", **SETTINGS)

    def test_a_missing_outcome(self) -> None:
        sample = frame().copy()
        sample["D"] = 1
        sample.loc[sample.index[:50], "D"] = 0
        sample.loc[sample.index[:50], "Y"] = np.nan
        with pytest.raises(NotImplementedError, match="delta="):
            DRTMLE(**SETTINGS).fit(sample, outcome="Y", treatment="A", delta="D")

    def test_combining_it_with_ctmle(self) -> None:
        class Both(DRTMLE, CTMLE):
            pass

        with pytest.raises(NotImplementedError, match="CTMLE are not combined"):
            Both(**SETTINGS).fit(frame(), outcome="Y", treatment="A")

    def test_a_plain_tmle_will_not_retarget_a_doubly_robust_fit(self, fit) -> None:
        """It has no learners to refit the reductions with, and says so rather than guessing.

        The alternation refits inside itself, so re-solving against the cached arrays would
        answer a different question from the one the fit answered -- quietly.
        """
        data = CausalData.from_frame(frame(), outcome="Y", treatment="A", covariates=None)
        with pytest.raises(NotImplementedError, match="no learners"):
            TMLE(**SETTINGS).retarget(data, fit.nuisance, estimands=ESTIMANDS)


def _plain_curve(fit, data, fluctuation):
    """``ey1``'s influence curve without the two extra terms, at the same targeted pair."""
    from dataclasses import replace

    g1 = np.asarray(fluctuation.mechanism.propensity, dtype=float)
    nuisance = replace(
        fit.nuisance, propensity=Propensity(np.column_stack([1.0 - g1, g1]), fit.nuisance.arms)
    )
    submodel = build_submodel(
        data,
        nuisance,
        "mean",
        bounds=fluctuation.reduction.bounds,
        nuisance_bound=DEFAULT_NUISANCE_BOUND,
    )
    means = counterfactual_means(
        fit.nuisance.scaler.scale(data.outcome),
        fluctuation.targeted,
        submodel,
        data.weights,
        data.observed,
    )
    return np.asarray(means[1.0].influence_curve)

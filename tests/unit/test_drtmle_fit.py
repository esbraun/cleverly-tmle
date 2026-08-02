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

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from cleverly import CTMLE, DRTMLE, TMLE
from cleverly.data import CausalData
from cleverly.datasets import nonlinear_dgp
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.serialize import load
from cleverly.estimators.targeting import _solved, build_submodel
from cleverly.estimators.tmle import DEFAULT_NUISANCE_BOUND, correction_parts
from cleverly.inference.influence import counterfactual_means, reduced_corrections
from cleverly.validation.drtmle import correction_check
from cleverly.validation.score import DEFAULT_TOLERANCE
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


@pytest.fixture(scope="module")
def repeated():
    """Two draws, which is the whole marginal cost of item 18 and is enough for it.

    ``repeats=`` costs one full fit per draw and a fit here is ~25s, so this fixture is the
    most expensive thing in the module after ``fit``. Two rather than three deliberately:
    the averaging *rule* -- mean of the estimates, elementwise mean of the curves, variance
    recomputed from the average -- is already pinned on a plain TMLE in
    ``tests/unit/test_repeated_crossfit.py``, and a third draw would re-pay for the same
    claim. What is new here is that the doubly-robust *construction* survives being
    averaged over draws, and two independent sets of three equations is what that needs.
    """
    return DRTMLE(**SETTINGS, repeats=2).fit(frame(), outcome="Y", treatment="A").single()


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

    def test_the_verdict_does_not_call_the_corrected_curve_efficient(self, fit, ordinary) -> None:
        """Validity is not efficiency, and the sign-off is where the package said it was.

        The pair is the test. A doubly-robust fit solves three equations, two of them the
        corrections, and what they leave is the estimator's influence function at the
        nuisance limits rather than the canonical gradient -- so signing it off as "the
        estimated efficient score equation" asserted exactly what
        ``reduced_corrections`` exists to deny. A plain fit's verdict is unchanged, word
        for word, because there the phrase is right; ``README.md``'s transcript quotes it.
        """
        corrected = fit.validation.score_check()
        plain = ordinary.validation.score_check()

        assert corrected.corrected and not plain.corrected
        assert "efficient" not in corrected.summary().lower().split("validity is not")[0]
        assert "Validity is not efficiency" in corrected.summary()
        assert "solved the estimated efficient score equation" in plain.summary()

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
        # And in the *report* as well as in the arrays: `corrected` is read off the
        # reduction records, so this fit gets a plain fit's verdict word for word.
        check = bare.validation.score_check()
        assert not check.corrected
        assert "solved the estimated efficient score equation" in check.summary()


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
            guard=reduction.guard,
            observed=data.observed,
        )
        at_initial = reduced_corrections(
            scaled,
            fluctuation.targeted,
            data.treatment,
            reduction.reduced,
            fit.nuisance.propensity.arm(1.0),
            bounds=reduction.bounds,
            guard=reduction.guard,
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


class TestTheCorrectionsAreTheOnesTheFitSolvedFor:
    r"""Piece B1a on a fit that has nothing wrong with it, which is half of what it is for.

    ``tests/unit/test_influence_drtmle.py`` checks the algebra on hand-built arrays; what is
    checked here is that a real fit's returned state reaches it -- the refitted reductions,
    the targeted mechanism, the fit's own truncation and its weights, none of which the
    array-level fixture exercises.

    This fixture's draw clips **nothing**, so every identity below holds under every
    convention piece B1b might select.  That makes it a weak test of the identity and the
    right test of everything around it: that the rows exist, one per arm and per equation,
    before any contrast; that a clean fit is not accused of anything; and that
    :math:`B_{clip}` is exactly zero where the bound never binds, which is what makes it a
    diagnosis rather than a fudge factor when it is not.
    """

    def test_there_is_a_row_per_arm_and_per_equation(self, fit) -> None:
        """Per arm and **before** the contrast -- an ATE-only check cannot see a cancelling pair."""
        check = fit.validation.correction_check()

        assert {(row.arm, row.equation) for row in check.rows} == {
            (arm, equation) for arm in (0.0, 1.0) for equation in ("D*_g", "D*_Q")
        }
        assert all(row.solved for row in check.rows), "both guards are on"

    def test_every_identity_holds_at_roundoff(self, fit) -> None:
        check = fit.validation.correction_check()

        assert check.passed
        for row in check.rows:
            assert abs(row.residual) < 1e-15, row.name

    def test_the_bound_never_binds_here_and_the_diagnostic_says_so(self, fit) -> None:
        """The control for the fixture that does clip: zero rows, zero bias, exactly."""
        check = fit.validation.correction_check()

        assert check.clipped == 0
        for row in check.rows:
            if row.equation == "D*_g":
                assert row.clip_bias == 0.0

    def test_it_adds_no_failing_row_to_a_clean_fit(self, fit) -> None:
        """The new rows are on every doubly-robust report, so a passing fit must stay passing."""
        check = fit.validation.score_check()

        assert check.passed
        assert not check.identity_failures
        assert {row.kind for row in check.rows} == {
            "fluctuation",
            "correction",
            "identity",
            "influence curve",
        }

    def test_a_plain_fit_gets_no_such_rows(self, ordinary) -> None:
        """No estimand outside this variant reports a correction, so none gains a row."""
        assert ordinary.validation.correction_check().rows == ()
        assert {row.kind for row in ordinary.validation.score_check().rows} == {
            "fluctuation",
            "influence curve",
        }

    def test_the_means_it_reports_are_weighted(self, fit) -> None:
        """Every score here is weighted, so a check taking plain means would be a different check.

        No fit in this module carries weights -- they would be one more full alternation to
        pay for -- so the claim is made by *swapping the weights on a fitted result* and
        watching every reported mean move.  That is enough: what is in doubt is whether
        this function reads ``data.weights`` at all, and an unweighted fixture answers it
        the same way a weighted one would while costing nothing.
        """
        rows = np.arange(fit.data.n)
        tilted = replace(fit.data, weights=1.0 + 0.5 * np.cos(rows))
        moved = correction_check(replace(fit, data=tilted), tolerance=DEFAULT_TOLERANCE)
        plain = {
            (row.arm, row.equation): row.reported for row in fit.validation.correction_check().rows
        }

        for row in moved.rows:
            assert abs(row.reported - plain[(row.arm, row.equation)]) > 1e-12, row.name

    def test_the_check_survives_a_round_trip(self, fit, tmp_path) -> None:
        """Derived from the records rather than stored, so a reloaded fit answers for itself.

        The same reason ``score_verdict`` is derived: a flag written at fit time is one
        nothing could check afterwards, and this check's whole subject is a disagreement
        between what a fit *recorded* and what it *reports*.
        """
        back = load(fit.save(tmp_path / "fit.npz"))
        before = fit.validation.correction_check()
        after = back.validation.correction_check()

        assert after.rows == before.rows
        assert after.passed


@pytest.fixture(scope="module")
def single_guard():
    """``guard=("g",)`` on this module's own draw -- the first partial-guard fit anywhere here.

    **1.8s measured**, against ~25s for the default-guard ``fit``, and the difference is
    structural rather than luck: ``reduction.refit`` is called only inside the ``"Q"``
    branches of the alternation, so this fit runs no reduced refits at all. That is why the
    cheap direction is the one taken end to end and ``guard=("Q",)`` is left to
    ``tests/unit/test_influence_drtmle.py``, where a solve on the exact law costs
    milliseconds.

    ``frame()`` rather than a fresh draw for three reasons: no second sample to fit, the
    same draw as ``fit`` so the two are comparable, and -- decisively -- this draw is
    already known to clip **zero** rows, which is the precondition that makes what follows
    item 23 and not item 20 wearing a different hat.
    """
    return DRTMLE(guard=("g",), **SETTINGS).fit(frame(), outcome="Y", treatment="A").single()


class TestASingleGuardSubtractsOnlyTheCorrectionItSolvedFor:
    """``docs/roadmap.md`` item 23, end to end, which is where it was never checked.

    ``guard=`` is crossed, so a fit guarding ``"g"`` solves equation (10) and subtracts
    ``D*_Q``, and never poses equation (9) at all.  It used to subtract ``D*_g`` anyway --
    a term whose mean nothing had driven anywhere.  Nothing here saw it because no test in
    this repository fitted a partial guard end to end; B1a's instrument found it on its
    first run against one.
    """

    def test_the_preconditions_this_reads_as_item_23_under(self, single_guard) -> None:
        """Asserted rather than assumed, because each one is how it could be a different bug.

        Zero clipped rows is what separates this from item 20, whose whole mechanism is the
        truncation.  No mechanism fluctuation is what says equation (9) was never posed --
        so ``D*_g``'s mean here is not a solver's residual but an arbitrary number.
        """
        check = single_guard.validation.correction_check()
        fluctuation = single_guard.repeats[0].fluctuations["mean"]

        assert check.clipped == 0
        assert fluctuation.mechanism is None

    def test_the_unsolved_correction_is_large_enough_to_matter(self, single_guard) -> None:
        """The negative control: without it every assertion below could hold vacuously.

        Measured at ``1.2e-03`` and ``3.1e-04`` on the outcome scale against a ``5.4e-06``
        bar -- 225 and 58 times over, on the *good*-overlap draw this module fits
        everything else on.  Before item 23 closed, these went into the reported curve.
        """
        check = single_guard.validation.correction_check()
        unsolved = [row for row in check.rows if row.equation == "D*_g"]

        assert len(unsolved) == 2
        for row in unsolved:
            assert abs(row.reported) > 20 * check.threshold, row.name

    def test_the_reported_curve_subtracts_the_solved_term_and_only_it(self, single_guard) -> None:
        """The claim itself, against the arrays rather than against a summary of them."""
        data = CausalData.from_frame(frame(), outcome="Y", treatment="A", covariates=None)
        fluctuation = single_guard.repeats[0].fluctuations["mean"]
        parts = correction_parts(
            data,
            single_guard.nuisance,
            fluctuation,
            fluctuation.targeted,
            single_guard.nuisance.scaler.scale(data.outcome),
        )
        plain = _plain_curve(single_guard, data, fluctuation)
        reported = (
            np.asarray(single_guard.estimates["ey1"].influence_curve)
            / single_guard.nuisance.scaler.range
        )

        np.testing.assert_allclose(reported, plain - parts.d_q[1.0], rtol=0, atol=1e-12)
        # And it is materially not what it was: the sum is a different curve, which is the
        # whole of what this item moved.
        assert np.max(np.abs(reported - (plain - parts.d_q[1.0] - parts.d_g[1.0]))) > 1e-6

    def test_every_estimand_s_curve_is_centred(self, single_guard) -> None:
        """The consequence a reader cares about, and the one that used to fail."""
        for name, estimate in single_guard.estimates.items():
            assert abs(float(np.mean(estimate.influence_curve))) < 1e-8, name

    def test_the_report_says_which_equation_it_left_out(self, single_guard) -> None:
        """Reported rather than dropped: a partial-guard report must not be quietly smaller.

        The ``D*_g`` rows are the only thing that says what the guard did not buy, so they
        stay -- as ``diagnostic`` rows, held to no threshold, which is what stops a correct
        fit failing for a term nothing subtracts.
        """
        check = single_guard.validation.score_check()
        kinds = {row.name: row.kind for row in check.rows}

        assert check.passed
        assert kinds["mean (D*_g)[0]"] == kinds["mean (D*_g)[1]"] == "diagnostic"
        assert kinds["mean (D*_Q)[0]"] == kinds["mean (D*_Q)[1]"] == "correction"
        # Equation (9) was never posed, so there is no fluctuation row for it either.
        assert "mean (mechanism)" not in kinds
        assert not check.identity_failures
        assert not check.corrections.correction_failures()

    def test_the_verdict_names_the_curve_this_fit_actually_reports(self, single_guard) -> None:
        """Derived from the rows, so it cannot go on claiming a term the curve dropped."""
        summary = single_guard.validation.score_check().summary()

        assert "D = D* - D*_Q," in summary
        assert "D*_g" not in summary.split("Validity is not efficiency")[1]

    def test_it_survives_a_round_trip(self, single_guard, tmp_path) -> None:
        """``guard`` is already on the serialised record, so no format bump was needed.

        Compared field by field rather than by row equality, which the default-guard
        sibling can use and this one cannot: an unsolved row's ``stored`` is ``nan``, and
        ``nan != nan`` would make this pass for the wrong reason on any two objects.
        """
        back = load(single_guard.save(tmp_path / "single.npz"))
        after, before = (fit.validation.correction_check().rows for fit in (back, single_guard))

        assert len(after) == len(before) == 4
        for new, old in zip(after, before, strict=True):
            assert (new.arm, new.equation, new.solved) == (old.arm, old.equation, old.solved)
            assert new.reported == old.reported
            assert np.isnan(new.stored) == np.isnan(old.stored)
        assert {row.equation for row in after if not row.solved} == {"D*_g"}
        np.testing.assert_array_equal(
            back.estimates["ey1"].influence_curve,
            single_guard.estimates["ey1"].influence_curve,
        )


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
    ``docs/drtmle-investigation-log.md`` under *How the alternation exits* -- says otherwise:
    the solve is ill-conditioned on 5 of 12 ``linear`` draws at ``n = 600`` and 9 of 12 at ``n = 1,200``,
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
    The sweep in ``docs/drtmle-investigation-log.md`` had 2 of 96 reach the tolerance under
    the *old* rule and
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

    # The same fallback `correction_parts` takes: without the `"Q"` guard no mechanism was
    # tilted, and the initial one is what the fit's other equation was solved beside. A
    # bare `.mechanism.propensity` here is an `AttributeError` on a single-guard fit.
    g1 = np.asarray(
        fluctuation.mechanism.propensity
        if fluctuation.mechanism is not None
        else fit.nuisance.propensity.arm(1.0),
        dtype=float,
    )
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


class TestEachDrawSolvesItsOwnEquations:
    r"""``repeats=`` on a doubly-robust fit, which averages more than a plain one does.

    Averaging influence curves over split draws is ordinary for a cross-fitted estimator.
    What is not ordinary is that both of this variant's additions are split-dependent: each
    draw fits its *own* reduced regressions against its own folds and runs its own
    alternation.  ``_fit_reduced`` is deliberately unseeded so that a refit matches its fit,
    which leaves the primary split as the only thing ``repeats=`` varies -- the right
    design, and the reason this is a check rather than a bug report.

    What the check found is in :class:`TestTheReportedCurveIsNotAlwaysCentred` below, and it
    is not about ``repeats=``.

    Note what the roadmap originally proposed as this row's mutation and why it is not used:
    "drop a repeat and watch the averaged curve decentre" cannot fail.  A centred curve
    carries its own :math:`-\psi_r`, so the mean of *any* subset of centred curves is
    centred.  What a dropped draw moves is ``psi`` and the row count of the score check,
    and that is what the tests below bite on.
    """

    def test_every_draw_gets_its_own_three_rows(self, repeated) -> None:
        """Six fluctuation rows, not three -- each draw solved its own set, and solved it."""
        check = repeated.validation.score_check()
        rows = {row.name: row for row in check.rows}

        for draw in (0, 1):
            for stem in ("mean", "mean (mechanism)", "mean (reduced)"):
                row = rows[f"{stem}[draw {draw}]"]
                assert row.passed, row.name
        assert sum(1 for row in check.rows if row.kind == "fluctuation") == 6

    def test_the_reductions_follow_the_draw(self, repeated) -> None:
        """The claim item 18 rests on: ``repeats=`` varies the reductions, not only the folds.

        A draw's reduced regressions are fitted against *that* draw's folds, so two draws
        hold two different ``Qr``. If they did not, the average would be over fits that
        differed in the primary nuisances alone and the extra equations would be along for
        the ride rather than being redrawn with everything else.
        """
        first, second = (repeated.repeats[i].fluctuations["mean"].reduction for i in (0, 1))

        assert first is not None and second is not None
        assert not np.array_equal(first.reduced.qr, second.reduced.qr)
        assert not np.array_equal(
            repeated.repeats[0].nuisance.folds.assignment,
            repeated.repeats[1].nuisance.folds.assignment,
        )

    def test_the_report_is_the_mean_of_the_draws(self, repeated) -> None:
        """And the mean is load-bearing: the two draws do not agree to begin with."""
        for name in ESTIMANDS:
            per_draw = [repeat.psi[name] for repeat in repeated.repeats]
            assert per_draw[0] != per_draw[1]
            assert repeated.estimates[name].psi == pytest.approx(float(np.mean(per_draw)))

    def test_no_draw_is_silently_dropped(self, repeated) -> None:
        """``average_estimates`` warns and drops a name missing from some draws.

        That path is pinned on hand-built estimates in
        ``tests/unit/test_repeated_crossfit.py``; what is checked here is that this
        estimator never reaches it, which a stalled alternation returning a short report
        would.
        """
        assert repeated.n_repeats == 2
        for repeat in repeated.repeats:
            assert set(repeat.psi) >= set(ESTIMANDS)


class TestTheReportedCurveIsNotAlwaysCentred:
    r"""What checking ``repeats=`` found, which is a defect in the *fit* and not in ``repeats=``.

    On a quarter of splits the curve the interval is built from has a mean five or six
    orders of magnitude above the bar, while all three fluctuation rows report their scores
    solved to ``1e-11`` or better.  Measured over 24 draws -- twelve ``repeats=2`` fits on
    this module's frame -- **six** leave :math:`P_n[D^*_Q + D^*_g]` above ``1e-8``, at
    magnitudes from ``2e-05`` to ``7e-04``, every one of them exiting on ``"tolerance"``
    with no failure recorded and no ill-conditioned round.

    So the recorded score for equation (9) and the mean of the :math:`D^*_g` the curve
    actually subtracts **disagree**: on this module's ``repeated`` fixture the first is
    ``3.7e-11`` and the second ``-2.3e-04``.  That is the class of defect
    :class:`TestTheCurveReadsWhatTheAlternationLeft` exists for and which no test here
    previously covered, because every test above reads one fit on one split.

    **The cause is located and it is not two states.**  Recomputing the recorded score from
    the returned ``fluctuation.mechanism.propensity`` and ``fluctuation.reduction.reduced``
    reproduces it bit for bit; what differs is that the fluctuation solves
    :math:`P_n[H_g (A - g^*)] = 0` at the *raw* tilted mechanism while
    :func:`~cleverly.inference.influence.reduced_corrections` truncates :math:`g^*` inside
    the residual too.  The two therefore agree on every row the bound leaves alone: on this
    fixture draw 0 clips **0** of 600 rows and is centred at ``1e-11``, draw 1 clips **5**
    and is off by ``2.3e-04``, and the reported curve's mean is ``-range`` times the score
    at the truncated residual to three figures.  Six seeded single fits give the converse --
    the five with no clipped row pass, the one with a single clipped row fails at
    ``5.8e-04``.

    **Rewritten by piece B1a, and it is no longer a class about a symptom.**  It used to pin
    the defect's numbers -- an uncentred curve beside three solved fluctuation rows --
    because that was all the package could see.  What it asserts now is the *identity*: that
    the score the solver stored and the mean of the term the curve subtracts are one
    evaluation of one expression at the returned state, per arm, weighted, on one outcome
    scale, and on a draw where the bound **binds**.  That last condition is why this fixture
    is kept rather than replaced with a better-behaved seed: draw 1 clips five rows of 600
    and draw 0 clips none, so both halves of the claim live in one fit.

    The identity still **fails** here, and that is the point.  B1a is the instrument and not
    the remedy: which mechanism equation (9) ought to be solved against is a derivation with
    more than two candidates -- the theorem's own algorithm truncates nothing at all -- and
    it is piece B1b, waiting on the published Theorem 1.  Every assertion below is valid
    under each of those conventions; what changes when one is adopted is that these rows
    pass.

    ``repeats=`` is **not** the cause and refusing it would be misdiagnosing this: a draw of
    a repeated fit is an ordinary fit, and the affected draws include first draws.  What
    ``repeats=`` did was give the module more than one split to look at.

    ``score_check`` caught this before B1a, on the *influence-curve* rows, which are
    computed from the curve rather than from a record of what the solver reported -- item
    16, arriving on the first case nobody constructed.  What it could not do is say which
    arm, which equation, or that the cause was an expression rather than a solver, and a
    reader following its advice would have gone looking for a convergence problem that was
    not there.
    """

    def test_the_identity_fails_on_the_draw_that_clips_and_holds_on_the_one_that_does_not(
        self, repeated
    ) -> None:
        """Both halves in one test, because either alone is misleading.

        A draw where nothing clips satisfies the identity under every convention, so a
        fixture chosen for passing would prove nothing; a draw where it fails is only
        evidence if a sibling draw of the same fit passes.
        """
        rows = repeated.validation.correction_check().rows
        by_draw = {draw: [row for row in rows if row.draw == draw] for draw in (0, 1)}

        assert all(row.clipped == 0 for row in by_draw[0])
        assert all(row.clipped == 5 for row in by_draw[1]), "the bound must still bind here"
        for row in by_draw[0]:
            assert abs(row.residual) < 1e-15, row.name
        offenders = [row for row in by_draw[1] if abs(row.residual) > 1e-8]
        assert {row.equation for row in offenders} == {"D*_g"}
        assert {row.arm for row in offenders} == {0.0, 1.0}

    def test_the_clipping_bias_accounts_for_the_whole_of_it(self, repeated) -> None:
        """`B_clip` is the diagnosis, so it has to reproduce the discrepancy and not merely rhyme.

        Sign per ``docs/drtmle-validation-plan.md``: the residual is the stored score minus
        the reported mean, and :math:`B_{clip}` carries :math:`g - g^b`, so the two are
        negatives of one another.
        """
        rows = [
            row
            for row in repeated.validation.correction_check().rows
            if row.draw == 1 and row.equation == "D*_g"
        ]
        assert len(rows) == 2
        for row in rows:
            assert abs(row.clip_bias) > 1e-5, "or the fixture stopped exercising this"
            np.testing.assert_allclose(row.residual, -row.clip_bias, rtol=0, atol=1e-15)

    def test_equation_ten_is_the_control_and_holds_on_every_draw(self, repeated) -> None:
        """Nothing truncates on that side, so an instrument that fired there would be broken."""
        for row in repeated.validation.correction_check().rows:
            if row.equation == "D*_Q":
                assert abs(row.residual) < 1e-15, row.name

    def test_the_score_check_catches_it_where_the_fluctuation_rows_do_not(self, repeated) -> None:
        """And now names it: the fluctuation rows still pass, and the identity rows do not."""
        check = repeated.validation.score_check()

        assert not check.passed
        assert all(row.passed for row in check.rows if row.kind == "fluctuation")
        assert {row.name for row in check.identity_failures} == {
            "mean (D*_g)[0] identity[draw 1]",
            "mean (D*_g)[1] identity[draw 1]",
        }
        # The estimand rows are the consequence, and they were the only witness before B1a.
        assert set(ESTIMANDS) <= {row.name for row in check.failures}

    def test_and_the_summary_says_a_defect_rather_than_a_convergence_failure(
        self, repeated
    ) -> None:
        """Item 16's machinery, carrying B1a's distinction.

        "The score equation was not solved" is the wrong advice here -- it sends a reader
        to ``one_step`` and a smaller step size for a fit whose solver did its job. The
        wording has to name the state identity, and the two failures must not be reported
        as one.
        """
        summary = repeated.summary()
        assert "score check: FAIL" in summary
        assert "state identity" in summary
        assert "iterating longer will not fix" in summary
        assert "do not describe this estimate" in summary

    def test_the_curve_a_defect_produces_is_the_arms_own(self, repeated) -> None:
        r"""Which arm is wrong, by how much, and on one outcome scale.

        The reported curve's mean is minus the mean of the corrections it subtracts,
        averaged over the draws -- so this ties each estimand's miss to a *named arm's*
        correction rather than to a fit-level number, and it is what fails if the rows are
        reported on the fitting scale instead of the outcome's.
        """
        rows = repeated.validation.correction_check().rows
        per_arm = {
            arm: sum(row.reported for row in rows if row.arm == arm) / repeated.n_repeats
            for arm in (0.0, 1.0)
        }

        assert repeated.estimates["ey1"].score == pytest.approx(-per_arm[1.0], abs=1e-9)
        assert repeated.estimates["ey0"].score == pytest.approx(-per_arm[0.0], abs=1e-9)
        assert repeated.estimates["ate"].score == pytest.approx(
            -(per_arm[1.0] - per_arm[0.0]), abs=1e-9
        )

    def test_the_averaged_curve_inherits_it_rather_than_causing_it(self, repeated) -> None:
        """The average is exact, so the miss comes from a draw and not from averaging.

        Stated as an identity rather than as a tolerance: whatever the draws' curves are,
        the reported one is their elementwise mean, so its mean is the mean of theirs. A
        centring failure therefore has to come from a draw -- which is why this is not a
        ``repeats=`` defect.
        """
        offenders = [
            abs(float(np.mean(fit_curve)))
            for fit_curve in (repeated.estimates[name].influence_curve for name in ESTIMANDS)
        ]
        assert max(offenders) > 1e-5, "the fixture is the split this was measured on"
        # And no fluctuation row is anywhere near it, which is the whole disagreement.
        worst = max(
            abs(row.score)
            for row in repeated.validation.score_check().rows
            if row.kind == "fluctuation"
        )
        assert worst < 1e-8

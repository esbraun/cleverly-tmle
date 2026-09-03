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

import math
import warnings
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from cleverly import AssessmentStatus, CapabilityError
from cleverly.data import CausalData
from cleverly.datasets import nonlinear_dgp
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators._nuisance import Propensity
from cleverly.estimators.serialize import load
from cleverly.estimators.targeting import (
    DEFAULT_MAX_OUTER,
    _negligible_bar,
    _solved,
    build_submodel,
)
from cleverly.estimators.tmle import (
    DEFAULT_NUISANCE_BOUND,
    correction_parts,
    reported_mechanism,
)
from cleverly.inference.influence import counterfactual_means, reduced_corrections
from cleverly.validation.drtmle import MARGIN_ACTIVE, CorrectionRow, correction_check
from cleverly.validation.score import DEFAULT_TOLERANCE
from tests.conftest import FAST_KWARGS

#: The mean group, which is what the reduced-dimension regressions are derived for. A
#: default binary report also asks for ``att`` and ``atc``, which are refused -- spelled out
#: here for the same reason ``tests/e2e/test_ctmle.py`` spells them out.
ESTIMANDS = ("ate", "ey1", "ey0")

SETTINGS = {**FAST_KWARGS, "estimands": ESTIMANDS}


def frame():
    """600 rows, which is the *cheaper* end here rather than the more generous one.

    The alternation fits three reduced families per arm on every round, so this
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
def bivariate_fit():
    """The canonical two-column reduction on the same cross-fitted draw as ``fit``."""
    return (
        DRTMLE(reduction="bivariate", **SETTINGS).fit(frame(), outcome="Y", treatment="A").single()
    )


@pytest.fixture(scope="module")
def repeated():
    """Two draws, enough to verify that repeated cross-fitting changes the reductions.

    ``repeats=`` costs one full fit per draw and a fit here is ~25s, so this fixture is the
    most expensive thing in the module after ``fit``. Two rather than three deliberately:
    the median *rule* -- median points with within-plus-between variance -- is already
    pinned on a plain TMLE in
    ``tests/unit/test_repeated_crossfit.py``, and a third draw would re-pay for the same
    claim. What is new here is that the doubly-robust *construction* survives combination
    over draws, and two independent sets of three equations is what that needs.
    """
    return DRTMLE(**SETTINGS, repeats=2).fit(frame(), outcome="Y", treatment="A").single()


class TestWhatItReports:
    def test_it_reports_the_mean_group_under_its_own_names(self, fit) -> None:
        """A different estimator behind the same parameters, exactly as ``CTMLE`` is."""
        assert set(fit.estimates) == set(ESTIMANDS)

    def test_all_three_score_equations_are_reported_and_solved(self, fit) -> None:
        check = fit.diagnostics.score_equations()
        names = {row.name for row in check.rows}

        assert {"mean", "mean (mechanism)", "mean (reduced)"} <= names
        assert check.passed

    def test_the_verdict_is_reachable_without_knowing_the_subsystem(self, fit) -> None:
        """``res.score_verdict`` is the same object, derived rather than stored."""
        assert fit.score_verdict.rows == fit.diagnostics.score_equations().rows
        assert fit.score_verdict.passed

    def test_a_fit_that_solved_its_equations_reports_that_it_did(self, fit) -> None:
        """``failure`` and ``converged`` read the same pair of rulers the loop's exit does.

        Both used to be read on the *relative* score alone, and
        :data:`~cleverly.estimators.targeting._NEGLIGIBLE` had already written down why that
        is the wrong instrument for equations (9) and (10): their covariates are small or
        move under their own solve, so neither can reach a bar stated as a fraction of its
        own magnitude.  The consequences were a failure recorded on a fit whose scores pass,
        and ``converged`` false on a clean tolerance exit.

        ``closing_capped`` is **not** asserted here, and that is the point of saying so.  It
        records how the closing stage *ended* rather than whether the score it left matters,
        it is routinely true on a fit that solved everything, and
        ``tests/unit/test_reduction_alternation.py`` pins the pair of fits it separates.
        Redefining it to mean "the score matters" was tried and reverted: that question is
        ``failure``'s, and the two are asserted apart because they are two facts.

        Pinned against the *score check* rather than against constants, so the assertion is
        "the record agrees with the fit" rather than a second copy of the rule.
        """
        reduction = fit.repeats[0].fluctuations["mean"].reduction

        assert fit.score_verdict.passed
        assert reduction.failure is None
        assert reduction.converged

    def test_the_record_carries_the_alternation_cap_that_applied(self, fit) -> None:
        """``max_outer`` is on the record because ``rounds`` alone cannot say "ran to the cap".

        This project published a manifest claiming ``max_iter: 100`` for an alternation that
        ran at 50: ``max_iter`` caps the Newton steps inside one fluctuation and never was
        the loop's.  Two keywords now, and the one that applied is recorded.
        """
        reduction = fit.repeats[0].fluctuations["mean"].reduction

        assert reduction.max_outer == DEFAULT_MAX_OUTER
        assert reduction.rounds <= reduction.max_outer
        assert (reduction.exit_reason == "cap") == (reduction.rounds == reduction.max_outer)

    def test_a_shorter_cap_is_reachable_and_is_recorded_as_the_one_that_ran(self) -> None:
        """The keyword reaches the solver, which is what no caller could do before.

        Two rounds rather than the default, so the loop is forced to the cap: a test that
        only checked the attribute was stored would pass on an estimator that still ignored
        it.
        """
        capped = DRTMLE(**SETTINGS, max_outer=2).fit(frame(), outcome="Y", treatment="A").single()
        reduction = capped.repeats[0].fluctuations["mean"].reduction

        assert reduction.max_outer == 2
        assert reduction.rounds == 2
        assert reduction.exit_reason == "cap"

    def test_an_alternation_cap_below_one_round_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="max_outer must be at least one round"):
            DRTMLE(**SETTINGS, max_outer=0)

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
        corrected = fit.diagnostics.score_equations()
        plain = ordinary.diagnostics.score_equations()

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
        """Half of a pair -- the other half is in ``tests/unit/test_drtmle_missing.py``.

        ``reduction`` used to be the *constructor argument* rather than the construction,
        so it read ``"univariate"`` on a fit that ran the five missing-outcome
        regressions.  Reading it off the fitted set makes the two halves disagree unless
        the label is right, and the ``diagnostics`` keys are what stop the label from
        being a word nothing depends on: the two constructions do not fit the same
        regressions, so a wrong label and the right keys cannot both hold.
        """
        report = fit.extra["drtmle"]
        assert report.guard == ("Q", "g")
        assert report.reduction == "univariate"
        assert set(report.diagnostics) == {"qr", "gr1", "gr2"}
        assert report.missingness_bound is None, "no reduction was formed at that bound"

    def test_validation_warns_when_solved_equation_ten_was_ill_conditioned(
        self, fit, tmp_path
    ) -> None:
        draw = fit.repeats[0]
        fluctuation = draw.fluctuations["mean"]
        reduction = replace(fluctuation.reduction, ill_conditioned=3, rounds=5)
        warned = replace(
            fit,
            repeats=(
                replace(
                    draw,
                    fluctuations={
                        **draw.fluctuations,
                        "mean": replace(fluctuation, reduction=reduction),
                    },
                ),
            ),
            assessment_cache={},
        )

        assert warned.diagnostics.score_equations().passed
        item = warned.validate()["score_equations"]
        assert item.status is AssessmentStatus.WARNING
        assert "3 of 5 refitting round(s) (60.0%)" in item.detail
        restored = load(warned.save(tmp_path / "conditioning-warning.npz"))
        assert restored.validate()["score_equations"].status is AssessmentStatus.WARNING

    def test_validation_passes_cleanly_without_ill_conditioned_rounds(self, fit) -> None:
        item = replace(fit, assessment_cache={}).validate()["score_equations"]
        assert item.status is AssessmentStatus.PASSED


class TestTheBivariateFit:
    """The pinned R construction runs through the production cross-fitted alternation."""

    def test_it_solves_and_reports_the_bivariate_correction(self, bivariate_fit) -> None:
        reduction = bivariate_fit.repeats[0].fluctuations["mean"].reduction

        assert reduction.reduced.reduction == "bivariate"
        assert np.isnan(reduction.reduced.gr2).all()
        assert set(bivariate_fit.extra["drtmle"].diagnostics) == {"qr", "gr1"}
        assert bivariate_fit.diagnostics.score_equations().passed
        assert bivariate_fit.diagnostics.corrections().passed

    def test_it_round_trips_without_becoming_univariate(self, bivariate_fit, tmp_path) -> None:
        back = load(bivariate_fit.save(tmp_path / "bivariate.joblib"))
        before = bivariate_fit.repeats[0].fluctuations["mean"].reduction.reduced
        after = back.repeats[0].fluctuations["mean"].reduction.reduced

        assert after.reduction == "bivariate"
        np.testing.assert_allclose(after.qr, before.qr, rtol=0, atol=0)
        np.testing.assert_allclose(after.gr1, before.gr1, rtol=0, atol=0)
        assert np.isnan(after.gr2).all()


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
        capability = bare.diagnostics.capability("corrections")
        assert capability.status is AssessmentStatus.NOT_APPLICABLE
        assert "subtracts no correction term" in capability.reason
        with pytest.raises(CapabilityError, match="subtracts no correction term"):
            bare.diagnostics.corrections()
        # And in the *report* as well as in the arrays: `corrected` is read off the
        # reduction records, so this fit gets a plain fit's verdict word for word.
        check = bare.diagnostics.score_equations()
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
        )
        at_initial = reduced_corrections(
            scaled,
            fluctuation.targeted,
            data.treatment,
            reduction.reduced,
            fit.nuisance.propensity.arm(1.0),
            bounds=reduction.bounds,
            guard=reduction.guard,
        )
        assert np.max(np.abs(at_targeted[1.0] - at_initial[1.0])) > 1e-6

        # Both sides on the [0, 1] scale the curve is built on; the reported one has been
        # mapped back to the outcome's own by `TargetContext.finish`, which is a scaling by
        # `range` and nothing else.
        plain_curve = _plain_curve(fit, data, fluctuation)
        reported = np.asarray(fit.estimates["ey1"].influence_curve) / fit.nuisance.scaler.range
        np.testing.assert_allclose(
            reported, plain_curve - data.weights * at_targeted[1.0], rtol=0, atol=1e-12
        )

    def test_the_reported_curve_still_has_mean_zero(self, fit) -> None:
        """Three solved equations rather than one, so this is a check on all of them."""
        for name in ESTIMANDS:
            assert abs(float(np.mean(fit.estimates[name].influence_curve))) < 1e-7


class TestTheCorrectionsAreTheOnesTheFitSolvedFor:
    r"""The correction identity on a fit that has nothing wrong with it.

    ``tests/unit/test_influence_drtmle.py`` checks the algebra on hand-built arrays; what is
    checked here is that a real fit's returned state reaches it -- the refitted reductions,
    the targeted mechanism, the fit's own truncation and its weights, none of which the
    array-level fixture exercises.

    This fixture's draw clips **nothing**, so every identity below holds under every
    bounded convention might select. That makes it a weak test of the identity and the
    right test of everything around it: that the rows exist, one per arm and per equation,
    before any contrast; that a clean fit is not accused of anything; and that
    :math:`B_{clip}` is exactly zero where the bound never binds, which is what makes it a
    diagnosis rather than a fudge factor when it is not.
    """

    def test_there_is_a_row_per_arm_and_per_equation(self, fit) -> None:
        """Per arm and **before** the contrast -- an ATE-only check cannot see a cancelling pair."""
        # Complete the argument-free contract on the method-specific route, which is
        # intentionally unavailable on the ordinary-TMLE fixtures in that contract suite.
        capability = fit.diagnostics.capability("corrections")
        assert capability.available and not capability.requires_arguments
        fit.diagnostics._bind_arguments("corrections", {}, partial=False)
        check = fit.diagnostics.corrections()

        assert {(row.arm, row.equation) for row in check.rows} == {
            (arm, equation) for arm in (0.0, 1.0) for equation in ("D*_g", "D*_Q")
        }
        assert all(row.solved for row in check.rows), "both guards are on"

    def test_every_identity_holds_at_roundoff(self, fit) -> None:
        check = fit.diagnostics.corrections()

        assert check.passed
        for row in check.rows:
            assert abs(row.residual) < 1e-15, row.name

    def test_the_bound_never_binds_here_and_the_diagnostic_says_so(self, fit) -> None:
        """The control for the fixture that does clip: zero rows, zero bias, exactly."""
        check = fit.diagnostics.corrections()

        assert check.clipped == 0
        for row in check.rows:
            if row.equation == "D*_g":
                assert row.clip_bias == 0.0

    def test_it_adds_no_failing_row_to_a_clean_fit(self, fit) -> None:
        """The new rows are on every doubly-robust report, so a passing fit must stay passing."""
        check = fit.diagnostics.score_equations()

        assert check.passed
        assert not check.identity_failures
        assert {row.kind for row in check.rows} == {
            "fluctuation",
            "correction",
            "identity",
            "influence curve",
        }

    def test_an_identity_residual_is_a_defect_and_not_an_unsolved_equation(self, fit) -> None:
        """The two failures are different failures, and neither may be reported as the other.

        An **identity residual** says the loop posed one expression and the curve carries
        another: a software defect, which no amount of further iteration repairs.  A
        **correction score** above the inferential bar says the fit did not solve equations it
        posed correctly: an ordinary convergence problem, whose remedy is a smaller step or
        more rounds.  Conflating them sends a reader to the wrong remedy in both directions.

        Built from rows rather than from a fit, because a fit that exhibits either is a fit
        this package does not produce -- and the classification is what is under test, not the
        arithmetic that feeds it.  ``residual = stored - reported``, so a row is made to fail
        one test or the other by moving exactly one of the two numbers.
        """
        real = fit.diagnostics.corrections()

        def row(**overrides):
            base = {
                "draw": 0,
                "arm": 1.0,
                "label": "1",
                "equation": "D*_g",
                "stored": 0.0,
                "reported": 0.0,
                "solved": True,
                "clipped": 0,
            }
            return CorrectionRow(**{**base, **overrides})

        def check(*rows):
            return replace(real, rows=rows)

        # `stored` and `reported` disagree, and the term itself is negligible: a defect.
        broken = check(row(stored=1.0, reported=0.0))
        assert len(broken.identity_failures()) == 1
        assert broken.correction_failures() == ()
        assert not broken.passed

        # The two expressions agree exactly, and what they agree on is far too large: an
        # unsolved equation, and emphatically not a defect.
        unsolved = check(row(stored=9.0, reported=9.0))
        assert unsolved.identity_failures() == ()
        assert len(unsolved.correction_failures()) == 1
        assert not unsolved.passed

        # And the row a single guard leaves unsolved is judged against neither, however
        # large: nothing subtracts it, so it cannot make an interval wrong.
        informational = check(row(stored=float("nan"), reported=9.0, solved=False))
        assert informational.identity_failures() == ()
        assert informational.correction_failures() == ()
        assert informational.passed

    def test_a_plain_fit_gets_no_such_rows(self, ordinary) -> None:
        """No estimand outside this variant reports a correction, so none gains a row."""
        with pytest.raises(CapabilityError, match="does not use the correction system"):
            ordinary.diagnostics.corrections()
        assert {row.kind for row in ordinary.diagnostics.score_equations().rows} == {
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
            (row.arm, row.equation): row.reported for row in fit.diagnostics.corrections().rows
        }

        for row in moved.rows:
            assert abs(row.reported - plain[(row.arm, row.equation)]) > 1e-12, row.name

    def test_the_check_survives_a_round_trip(self, fit, tmp_path) -> None:
        """Derived from the records rather than stored, so a reloaded fit answers for itself.

        The same reason ``score_verdict`` is derived: a flag written at fit time is one
        nothing could check afterwards, and this check's whole subject is a disagreement
        between what a fit *recorded* and what it *reports*.
        """
        back = load(fit.save(tmp_path / "fit.joblib"))
        before = fit.diagnostics.corrections()
        after = back.diagnostics.corrections()

        assert after.rows == before.rows
        assert after.passed


@pytest.fixture(scope="module")
def single_guard():
    """``guard=("g",)`` on this module's own draw -- the first partial-guard fit anywhere here.

    This is the cheaper partial direction: it refits only ``gr1`` and ``gr2`` before solving
    equation (10), while ``guard=("Q",)`` is left to
    ``tests/unit/test_influence_drtmle.py``, where a solve on the exact law costs milliseconds.

    ``frame()`` rather than a fresh draw for three reasons: no second sample to fit, the
    same draw as ``fit`` so the two are comparable, and -- decisively -- this draw is
    already known to clip **zero** rows, which is the precondition that makes what follows
    the partial-guard defect rather than bounded-mechanism centring in another guise.
    """
    return DRTMLE(guard=("g",), **SETTINGS).fit(frame(), outcome="Y", treatment="A").single()


class TestASingleGuardSubtractsOnlyTheCorrectionItSolvedFor:
    """The partial-guard correction invariant, fitted end to end.

    ``guard=`` is crossed, so a fit guarding ``"g"`` solves equation (10) and subtracts
    ``D*_Q``, and never poses equation (9) at all.  It used to subtract ``D*_g`` anyway --
    a term whose mean nothing had driven anywhere.  Nothing here saw it because no test in
    this repository fitted a partial guard end to end; the correction check found it on its
    first run against one.
    """

    def test_the_preconditions_this_reads_as_item_23_under(self, single_guard) -> None:
        """Asserted rather than assumed, because each one is how it could be a different bug.

        Zero clipped rows separates this from the bounded-mechanism defect, whose mechanism is the
        truncation.  No mechanism fluctuation is what says equation (9) was never posed --
        so ``D*_g``'s mean here is not a solver's residual but an arbitrary number.
        """
        check = single_guard.diagnostics.corrections()
        fluctuation = single_guard.repeats[0].fluctuations["mean"]

        assert check.clipped == 0
        assert fluctuation.mechanism is None

    def test_the_reduced_mechanisms_are_refitted_at_the_targeted_regression(
        self, single_guard
    ) -> None:
        """The nonzero witness: equation (10) must not keep its initial regressions."""
        initial = single_guard.nuisance.reduced
        reduced = single_guard.repeats[0].fluctuations["mean"].reduction.reduced

        assert np.max(np.abs(reduced.gr1 - initial.gr1)) > 1e-2
        assert np.max(np.abs(reduced.gr2 - initial.gr2)) > 1e-2

    def test_the_unsolved_correction_is_large_enough_to_matter(self, single_guard) -> None:
        """The negative control: without it every assertion below could hold vacuously.

        Measured at ``1.2e-03`` and ``3.1e-04`` on the outcome scale against a ``5.4e-06``
        bar -- 225 and 58 times over, on the *good*-overlap draw this module fits
        everything else on. An earlier implementation put these into the reported curve.
        """
        check = single_guard.diagnostics.corrections()
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
        check = single_guard.diagnostics.score_equations()
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
        summary = single_guard.diagnostics.score_equations().summary()

        assert "D = D* - D*_Q," in summary
        assert "D*_g" not in summary.split("Validity is not efficiency")[1]

    def test_it_survives_a_round_trip(self, single_guard, tmp_path) -> None:
        """``guard`` is already on the serialised record, so no format bump was needed.

        Compared field by field rather than by row equality, which the default-guard
        sibling can use and this one cannot: an unsolved row's ``stored`` is ``nan``, and
        ``nan != nan`` would make this pass for the wrong reason on any two objects.
        """
        back = load(single_guard.save(tmp_path / "single.joblib"))
        after, before = (fit.diagnostics.corrections().rows for fit in (back, single_guard))

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
        back = load(fit.save(tmp_path / "fit.joblib"))

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
        back = load(fit.save(tmp_path / "verdict.joblib"))

        live, after = fit.diagnostics.score_equations(), back.diagnostics.score_equations()
        assert [row.name for row in after.rows] == [row.name for row in live.rows]
        assert [row.score for row in after.rows] == [row.score for row in live.rows]
        assert after.passed == live.passed

        reduction = back.repeats[0].fluctuations["mean"].reduction
        assert reduction is not None
        np.testing.assert_array_equal(
            reduction.reduced.qr, fit.repeats[0].fluctuations["mean"].reduction.reduced.qr
        )


class TestTheAlternationCanBeIllConditioned:
    r"""Equation (10) is not always solvable to its inner tolerance, for a structural reason.

    Its covariate is ``gr2 / gr1``, and ``gr2 = E[(1_a - g-hat)/g-hat | Qbar]`` **vanishes
    exactly where the mechanism is right**.  So the better ``g-hat`` is, the closer that
    covariate is to zero and the more numerically delicate its Newton solve: observed at
    ``mean|h| = 1e-3`` with ``|epsilon|`` reaching 280 and a singular Hessian in a third of
    the rounds, on a fit whose fold split was drawn unseeded.

    **How often that happens was measured rather than assumed, and the first measurement was
    wrong.**  Six seeded fits at ``n = 800`` on this one process reported no ill-conditioned
    solve and a worst score of ``1e-9``, which read as a minority behaviour of particular
    draws.  A 96-fit sweep -- four processes by two sizes by twelve seeds, tabulated in
    tabulated at the ``drtmle-validation-archive-2026-08`` tag -- says otherwise:
    the solve is numerically difficult on 5 of 12 ``linear`` draws at ``n = 600`` and 9 of 12 at ``n = 1,200``,
    and highest exactly where the mechanism is easiest to get right, which is what the
    paragraph above predicts and what sweeping only hard processes would have hidden.
    The historical ``ill_conditioned`` field counts either a singular solve or one that
    stopped at working precision just above its inner tolerance. The final score check can
    still pass, and ``validate()`` now warns so the numerical event is not hidden by that pass.

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
        check = hard.diagnostics.score_equations()
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
    The archived 96-fit sweep had 2 of 96 reach the tolerance under
    the *old* rule and
    the new rule has not been swept, so pinning it here would pin a seed.

    The magnitudes are the measured ones.  On a 400-row ``linear`` fit the round the loop
    gave up at had equation (10) at ``2.3e-8`` relative -- six orders above ``spec.tol`` --
    while its absolute score was near ``1.1e-10``, against a negligible bar of ``1e-3/400``.
    That gap is the item-7 defect in two numbers.

    The last case is item **12**'s and is about the bar rather than the predicate: what
    ``_negligible_bar`` *is*, now that it is stated as a numerical criterion in its own right
    rather than as a proxy for the one ``score_check`` applies to the reported fit.
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
        are the ones the diagnostic must continue failing.
        """
        assert not _solved(relative=2.3e-8, absolute=1e-3, tol=self.TOL, negligible=self.NEGLIGIBLE)

    def test_the_bar_tightens_with_the_sample_size(self) -> None:
        """The bar falls with ``n``, so a score negligible at 400 rows need not be at 40,000."""
        absolute = 1.1e-10
        assert _solved(
            relative=1.0, absolute=absolute, tol=self.TOL, negligible=_negligible_bar(400)
        )
        assert not _solved(
            relative=1.0, absolute=absolute, tol=self.TOL, negligible=_negligible_bar(40_000_000)
        )

    def test_the_bar_renders_an_o_and_not_an_O(self) -> None:
        r"""``bar(n) * sqrt(n) -> 0``, which is what makes it a rendering of :math:`o_p(n^{-1/2})`.

        This is the reported-score half and the property the bar's *own* justification now
        rests on.  It used to rest on ``score_check``'s ``tolerance * se / sqrt(n)`` with
        ``se = O(n**-0.5)`` substituted in -- an assumption, and a circular one, since the
        loop runs before the estimate exists.  Asymptotic linearity asks for
        :math:`P_n D = o_p(n^{-1/2})`, and the finite-sample rendering of an :math:`o` is a
        deterministic :math:`c_n/\sqrt n` with :math:`c_n \to 0`.  Nothing here reads a
        standard error, and that is the point: whether the fit is entitled to a Wald interval
        is ``score_check``'s question, at the realised ``se``, and is not this one.

        The mutation is a bar of ``1e-3/sqrt(n)`` -- a legitimate-looking sequence rendering
        an :math:`O` instead.  ``bar(n) * sqrt(n)`` is then flat at ``1e-3``, so both
        assertions below fail: the sequence stops decreasing and never crosses.  It was run.

        The second assertion straddles rather than taking a ratio, because a ratio here is a
        fact about the two sizes chosen: :math:`c_n` falls as :math:`n^{-1/2}`, so over the
        five decades below it drops by ``632`` and any threshold written as a multiple of the
        first entry is a restatement of that arithmetic rather than of the limit.  Crossing a
        fixed level is the property, and it is the one an :math:`O` cannot have.
        """
        sizes = (100, 400, 1_600, 40_000, 40_000_000)
        scaled = [_negligible_bar(n) * math.sqrt(n) for n in sizes]

        assert all(later < earlier for earlier, later in pairwise(scaled))
        assert scaled[0] > 1e-6 > scaled[-1]


class TestTheRefusals:
    """Unsupported compositions name what their derivation would need."""

    def test_a_multi_valued_treatment_is_fitted_armwise(self) -> None:
        sample = frame().copy()
        sample.loc[sample.index[:100], "A"] = 2
        fit = (
            DRTMLE(**{**SETTINGS, "estimands": ("ate",)})
            .fit(sample, outcome="Y", treatment="A")
            .single()
        )
        assert fit.nuisance.propensity.n_arms == 3
        assert len(fit.estimates) == 2

    def test_the_conditional_effects(self) -> None:
        with pytest.raises(NotImplementedError, match="ATT and ATC"):
            DRTMLE(**{**SETTINGS, "estimands": ("ate", "att")}).fit(
                frame(), outcome="Y", treatment="A"
            )

    def test_the_bivariate_reduction(self) -> None:
        estimator = DRTMLE(reduction="bivariate", **SETTINGS)

        assert estimator.reduction == "bivariate"

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
        with pytest.raises(NotImplementedError, match="canonical cvFolds mapping only"):
            DRTMLE(targeting_scheme="fold", **SETTINGS)

    def test_fold_wise_evaluation(self) -> None:
        with pytest.raises(NotImplementedError, match="corrected parameter and influence curve"):
            DRTMLE(cv_evaluation=True, **SETTINGS)

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

    def test_a_reduced_learner_that_raises_fails_the_fit_rather_than_the_round(self) -> None:
        """Fail closed: an exception from a reduction's learner leaves no result at all.

        The alternation refits the three reductions inside itself, once per round, so an
        exception has somewhere plausible to be caught -- and catching it would produce the
        worst outcome available: a fit that returns a ``psi``, an ``se`` and a green score
        check, built from whatever the previous round's arrays happened to be.  Nothing in
        the report would say a round had been skipped.

        The only handler on this path is
        :class:`~cleverly.learners.super_learner.SuperLearner`'s, which drops a **candidate**
        from its library on purpose; a learner handed in directly has no library to be
        dropped from, and this is what says so.
        """

        class Exploding:
            """Raises where a reduction is fitted, and nowhere earlier."""

            def fit(self, x, y, sample_weight=None):
                raise RuntimeError("the reduced learner exploded")

            def predict(self, x):  # pragma: no cover - the fit never returns
                raise AssertionError("a model that never fitted was asked to predict")

            def get_params(self, deep=True):
                return {}

            def set_params(self, **params):
                return self

        estimator = DRTMLE(**SETTINGS, reduced_outcome_learner=Exploding())
        with pytest.raises(RuntimeError, match="the reduced learner exploded"):
            estimator.fit(frame(), outcome="Y", treatment="A")

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

    # The same mechanism `correction_parts` reads, through the same function, so this
    # helper cannot drift from it: without the `"Q"` guard no mechanism was tilted, and the
    # initial one is what the fit's other equation was solved beside. A bare
    # `.mechanism.propensity` here is an `AttributeError` on a single-guard fit.
    #
    # The complement re-wrap below is what confines this helper to a complete-data binary
    # fit, which is every caller: a missing-outcome fit's mechanism comes back `(n, 2)` and
    # off the simplex, and rebuilding it as `[1 - g1, g1]` would be the very error
    # `Propensity(simplex=)` exists to name.
    g1 = np.asarray(reported_mechanism(fit.nuisance, fluctuation, fit.nuisance.arms), dtype=float)
    assert g1.ndim == 1, "this helper is for complete-data binary fits"
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
    r"""``repeats=`` on a doubly-robust fit, which combines more than a plain one does.

    Combining reports over split draws is ordinary for a cross-fitted estimator.
    What is not ordinary is that both of this variant's additions are split-dependent: each
    draw fits its *own* reduced regressions against its own folds and runs its own
    alternation.  ``_fit_reduced`` is deliberately unseeded so that a refit matches its fit,
    which leaves the primary split as the only thing ``repeats=`` varies -- the right
    design, and the reason this is a check rather than a bug report.

    What the check found is in :class:`TestTheReportedCurveIsNotAlwaysCentred` below, and it
    is not about ``repeats=``.

    Note the obvious mutation for this row, and why it is not used:
    "drop a repeat and watch the combined curve decentre" cannot fail. A centred curve
    carries its own :math:`-\psi_r`, so the central draw or pair remains
    centred.  What a dropped draw moves is ``psi`` and the row count of the score check,
    and that is what the tests below bite on.
    """

    def test_every_draw_gets_its_own_three_rows(self, repeated) -> None:
        """Six fluctuation rows, not three -- each draw solved its own set, and solved it."""
        check = repeated.diagnostics.score_equations()
        rows = {row.name: row for row in check.rows}

        for draw in (0, 1):
            for stem in ("mean", "mean (mechanism)", "mean (reduced)"):
                row = rows[f"{stem}[draw {draw}]"]
                assert row.passed, row.name
        assert sum(1 for row in check.rows if row.kind == "fluctuation") == 6

    def test_the_reductions_follow_the_draw(self, repeated) -> None:
        """``repeats=`` varies the reductions, not only the primary folds.

        A draw's reduced regressions are fitted against *that* draw's folds, so two draws
        hold two different ``Qr``. If they did not, the combination would use fits that
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

    def test_the_report_is_the_median_of_the_draws(self, repeated) -> None:
        """The DR construction reaches the aggregation, on draws that disagree.

        Two draws only, so the median and the mean coincide here and this cannot tell them
        apart.  What it checks is that each DR draw produces its own point and that the
        report combines them rather than returning draw 0.  The median rule itself is
        pinned at three draws in ``tests/unit/test_repeated_crossfit.py``.
        """
        for name in ESTIMANDS:
            per_draw = [repeat.psi[name] for repeat in repeated.repeats]
            assert per_draw[0] != per_draw[1]
            assert repeated.estimates[name].psi == pytest.approx(float(np.median(per_draw)))

    def test_no_draw_is_silently_dropped(self, repeated) -> None:
        """``median_estimates`` warns and drops a name missing from some draws.

        That path is pinned on hand-built estimates in
        ``tests/unit/test_repeated_crossfit.py``; what is checked here is that this
        estimator never reaches it, which a stalled alternation returning a short report
        would.
        """
        assert repeated.n_repeats == 2
        for repeat in repeated.repeats:
            assert set(repeat.psi) >= set(ESTIMANDS)


class TestTheReportedCurveIsCentredWhereTheBoundBinds:
    r"""The draw that exposed bounded-mechanism centring, kept as a regression fixture.

    **What this fixture was.**  On a quarter of splits the curve the interval is built from
    had a mean five or six orders of magnitude above the bar, while all three fluctuation
    rows reported their scores solved to ``1e-11`` or better.  Measured over 24 draws --
    twelve ``repeats=2`` fits on this module's frame -- **six** left
    :math:`P_n[D^*_Q + D^*_g]` above ``1e-8``, at magnitudes from ``2e-05`` to ``7e-04``,
    every one exiting on ``"tolerance"`` with no failure recorded and no ill-conditioned
    round.  On this ``repeated`` fixture the recorded score for equation (9) was ``3.7e-11``
    and the mean of the :math:`D^*_g` the curve subtracted was ``-2.3e-04``.

    **The cause was never two states.**  Recomputing the recorded score from the returned
    ``fluctuation.mechanism.propensity`` and ``fluctuation.reduction.reduced`` reproduced it
    bit for bit; what differed is that the fluctuation solved
    :math:`P_n[H_g (A - g^*)] = 0` at the *raw* tilted mechanism while
    :func:`~cleverly.inference.influence.reduced_corrections` truncates :math:`g^*` inside
    the residual too -- so the two agreed on every row the bound left alone and parted
    company on every row it clipped.  Draw 0 clipped **0** of 600 and was centred at
    ``1e-11``; draw 1 clipped **5** and was off by ``2.3e-04``.

    **What this class asserts now.** Bounded targeting puts
    :func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism` at the ``DRTMLE`` call
    sites: the score is solved at the truncated tilt, which is the expression the curve
    carries, and the alternation carries that truncated array forward.  So the identity
    **holds** on the draw it used to fail on, and the assertions below are the correction checks
    wrote, unchanged in tolerance -- per arm, before the contrast, weighted, on one outcome
    scale -- with their verdicts the other way up.  Nothing here was loosened to make them
    pass, which is the only reason they are worth reading.

    **This fixture is still the right one and the witness had to change.**  The condition
    The identity must be checked where the bound *binds*, and
    ``CorrectionRow.clipped`` -- the count at the exit -- is now **zero on every fit by
    construction**, since a converged tilt sits inside the bounds.  Selecting a fixture on
    it would select nothing at all, making the check degenerate.
    :attr:`~cleverly.validation.CorrectionRow.margin` is what says draw 1 is still the hard
    one, and this class asserts it below.

    **And it is `margin` rather than `initial_clipped`, which is worth a line because this
    docstring used to say the other thing.**  Item 25's witness now puts the initial
    mechanism's clipped count on every fit, so the claim became checkable -- and it is
    **zero on both draws here**: at ``g_bounds="auto"`` this frame's cross-fitted mechanism
    never leaves the bounds, and what left them is the *tilt*.  So the two draws are
    separated by the exit margin (``0.089`` against ``1.3e-06``, five orders) and not by a
    clipped count, and the fixture's precondition is a property of the targeting step's
    feasible set rather than of the draw's overlap.  ``TestTheContractSaysWhichEstimator``
    is where a fit with the initial truncation active is fitted on purpose.

    ``repeats=`` was **not** the cause and refusing it would have been misdiagnosing this: a
    draw of a repeated fit is an ordinary fit, and the affected draws included first draws.
    What ``repeats=`` did was give the module more than one split to look at.

    ``score_check`` caught this before the dedicated correction check, on the *influence-curve* rows, which are
    computed from the curve rather than from a record of what the solver reported -- item
    16, arriving on the first case nobody constructed.  What it could not do is say which
    arm, which equation, or that the cause was an expression rather than a solver, and a
    reader following its advice would have gone looking for a convergence problem that was
    not there.
    """

    def test_the_identity_holds_on_the_draw_that_clips_as_well_as_the_one_that_does_not(
        self, repeated
    ) -> None:
        """Both halves in one test, because either alone is misleading.

        A draw where the bound never bit satisfies the identity under every convention, so a
        fixture chosen for that would prove nothing.  What makes draw 1 evidence is that its
        *initial* mechanism leaves the bounds -- the precondition the centring defect needed and the one
        thing here the targeting convention cannot have moved.
        """
        rows = repeated.diagnostics.corrections().rows
        by_draw = {draw: [row for row in rows if row.draw == draw] for draw in (0, 1)}

        assert all(row.margin > 1e-2 for row in by_draw[0])
        assert all(row.margin < 1e-4 for row in by_draw[1]), "the bound must still bind"
        for draw in (0, 1):
            for row in by_draw[draw]:
                assert abs(row.residual) < 1e-15, row.name

    def test_the_targeted_mechanism_no_longer_leaves_the_bounds_at_all(self, repeated) -> None:
        """Which is *why* the identity holds, and why ``B_clip`` is now zero rather than small.

        The alternation carries the truncated tilt forward, so at a fixed point there is no
        raw array left for the clipping bias to measure a distance to.  Asserting it is
        exactly zero rather than negligible is the point: a small non-zero here would mean a
        row still sitting outside the bounds at the exit, which was the failing state.
        """
        check = repeated.diagnostics.corrections()

        assert check.clipped == 0
        assert check.margin < 1e-4, "on a fixture where the bound had something to do"
        for row in check.rows:
            if row.equation == "D*_g":
                assert row.clip_bias == 0.0

    def test_equation_ten_is_the_control_and_holds_on_every_draw(self, repeated) -> None:
        """Nothing truncates on that side, so an instrument that fired there would be broken."""
        for row in repeated.diagnostics.corrections().rows:
            if row.equation == "D*_Q":
                assert abs(row.residual) < 1e-15, row.name

    def test_the_score_check_passes_on_every_kind_of_row(self, repeated) -> None:
        """The rows that used to fail, named, so a regression names itself rather than a count.

        The two identity rows below are the ones the centring defect broke, and the estimand rows were
        the only witness before the dedicated correction check existed. Asserting the *names* rather than
        ``check.passed`` alone is what makes this fail loudly on the draw it was written for
        rather than quietly somewhere else.
        """
        check = repeated.diagnostics.score_equations()

        assert check.passed
        assert not check.identity_failures
        assert {"mean (D*_g)[0] identity[draw 1]", "mean (D*_g)[1] identity[draw 1]"} <= {
            row.name for row in check.rows
        }
        assert all(row.passed for row in check.rows)

    def test_and_the_summary_says_nothing_because_there_is_nothing_to_say(self, repeated) -> None:
        """Item 16's machinery, from the other side.

        A passing fit prints no extra line, which is why
        every transcript in the README and the guide is unchanged.  The three phrases below
        describe the identity defect, and none of them belongs on this fit now.
        """
        summary = repeated.summary()
        assert "score check: FAIL" not in summary
        assert "state identity" not in summary
        assert "iterating longer will not fix" not in summary
        assert "do not describe this estimate" not in summary

    def test_the_curve_is_still_the_arms_own(self, repeated) -> None:
        r"""Which arm contributes what, and on one outcome scale.

        The reported curve's mean is minus the mean of the central draws' corrections.
        It was written when both sides were large enough to see;
        it is an *identity* rather than a statement about a defect, so it holds now that
        both sides are ``1e-10`` -- and it is still what fails if the rows are reported on
        the fitting scale instead of the outcome's.
        """
        rows = repeated.diagnostics.corrections().rows
        per_arm = {
            arm: sum(row.reported for row in rows if row.arm == arm) / repeated.n_repeats
            for arm in (0.0, 1.0)
        }

        assert repeated.estimates["ey1"].score == pytest.approx(-per_arm[1.0], abs=1e-9)
        assert repeated.estimates["ey0"].score == pytest.approx(-per_arm[0.0], abs=1e-9)
        assert repeated.estimates["ate"].score == pytest.approx(
            -(per_arm[1.0] - per_arm[0.0]), abs=1e-9
        )

    def test_the_combined_curve_is_centred_and_agrees_with_the_fluctuation_rows(
        self, repeated
    ) -> None:
        """The two numbers that used to disagree by five orders, now on the same side.

        The curve's mean was ``2e-04`` on this fixture while every fluctuation row reported
        ``1e-11``, and the gap between them was the defect -- the reported curve and the
        solver's record describing different expressions.  Both are read here rather than
        only the first, because a curve that is centred while the rows are not, or the
        reverse, is the state this piece closed and neither number alone would say so.
        """
        curves = [
            abs(float(np.mean(repeated.estimates[name].influence_curve))) for name in ESTIMANDS
        ]
        worst = max(
            abs(row.score)
            for row in repeated.diagnostics.score_equations().rows
            if row.kind == "fluctuation"
        )

        assert max(curves) < 1e-8
        assert worst < 1e-8


@pytest.fixture(scope="module")
def pinched():
    """One fit whose three mechanism-side truncations are all active, on purpose.

    A *lever* rather than a hard process: ``g_bounds=(0.3, 0.7)`` on this module's own
    process clips the initial mechanism, pins the tilt to the boundary and floors
    :math:`g_{r,1}`, which activates all three relevant truncations at 400 rows and ~5s.
    ``weak_overlap_dgp`` reaches the same state at ``g_bounds="auto"`` and costs several
    times as much for a claim about a *witness* rather than about a process -- and the sweep
    already measures it there (``clip share`` ``0.231`` to ``0.338``).

    Cheaper than the module's other fixtures deliberately: 400 rows and a tight bound make
    the alternation *shorter* here rather than longer, because a pinched mechanism leaves
    equation (9) nothing to move.
    """
    with warnings.catch_warnings():
        # A 0.3-0.7 bound on this process warns about extrapolation on a tenth of the rows,
        # which is the fixture's precondition rather than a surprise.
        warnings.simplefilter("ignore")
        return (
            DRTMLE(**SETTINGS, g_bounds=(0.3, 0.7))
            .fit(nonlinear_dgp().sample(400, seed=3)[0], outcome="Y", treatment="A")
            .single()
        )


class TestTheContractSaysWhichEstimator:
    """Item 25: which estimator a fit's numbers are evidence about, on the face of the fit.

    Truncation is not in Theorem 1's algorithm -- there is one mechanism, produced by an
    unconstrained ``expit`` fluctuation, and boundedness is an assumption about :math:`g_0`
    rather than an operation on :math:`\\hat g`.  So the theorem-backed guarantee is claimed
    for a fit whose truncations are **inactive**, where ``solve_bounded_mechanism`` returns
    the unconstrained solve bit for bit and the estimator *is* the theorem's; a fit where one
    binds is empirically supported and outside the theorem.

    Until this class there was no way to ask a fit which of the two it was.  ``margin``
    covered the targeted mechanism at the exit and **nothing** covered the initial mechanism
    or :math:`g_{r,1}` -- and the third is the one with no assumption in the theorem at all,
    since it is a regression of an arm indicator on :math:`\\hat{\\bar Q}` and
    :math:`g_0 > \\delta` says nothing about it.

    **The pair of fits is the test.**  A label that only ever read ``"theorem"`` would be
    a degenerate check for a third time -- a column that could not disagree -- so the class fits
    one of each and asserts they disagree.
    """

    def test_an_ordinary_fit_is_the_theorems_estimator(self, fit) -> None:
        """No truncation active, so Theorem 1 applies as written and the label says so."""
        check = fit.diagnostics.corrections()

        assert check.contract == "theorem"
        assert check.truncations_active == ()
        assert check.initial_clip_share == 0.0
        assert check.margin > MARGIN_ACTIVE
        assert check.gr1_margin > MARGIN_ACTIVE

    def test_a_pinched_fit_is_outside_it_and_names_all_three(self, pinched) -> None:
        """The other side of the label, with each truncation named rather than counted.

        The three are different objects with different standing -- two are operations on a
        mechanism the theorem assumes bounded and the third has no assumption behind it --
        so a reader has to be able to see which bit.
        """
        check = pinched.diagnostics.corrections()

        assert check.contract == "bound-active"
        assert check.truncations_active == (
            "g-hat at the initial fit",
            "g* at the exit",
            "g_r1",
        )
        assert check.initial_clip_share > 0.05
        assert check.gr1_margin < 0.0, "gr1 is stored untruncated, so this column is signed"

    def test_bound_active_is_a_scope_label_and_not_a_failure(self, pinched) -> None:
        """The mistake this class exists not to make, asserted rather than promised.

        Every identity holds and every score is negligible on exactly these fits -- that is
        what bounded targeting measured on ``weak_overlap`` at a forced bound -- so folding the label into
        a verdict would report a sound fit as broken, and a reader would route around the
        regime the variant is most needed in.
        """
        check = pinched.diagnostics.corrections()

        assert check.passed
        assert check.identity_failures() == ()
        assert check.correction_failures() == ()
        assert pinched.diagnostics.score_equations().passed
        assert "score check: FAIL" not in pinched.summary()

    def test_the_label_and_its_three_numbers_are_on_the_face_of_the_check(self, pinched) -> None:
        """A user must not have to recompute ``clip share`` the way the sweep does.

        The summary names the contract and the
        frame carries the two new columns per draw, so "which side of the line is this fit
        on" is a read rather than an analysis.
        """
        check = pinched.diagnostics.corrections()
        summary = check.summary()
        frame = check.to_frame()

        assert "contract: bound-active" in summary
        assert "outside Theorem 1" in summary
        assert {"initial_clipped", "gr1_margin"} <= set(frame.columns)

    def test_a_plain_tmle_has_no_contract_to_report(self, ordinary) -> None:
        """No corrections, no mechanism tilt, nothing for the label to be about."""
        with pytest.raises(CapabilityError, match="does not use the correction system"):
            ordinary.diagnostics.corrections()

    def test_a_complete_data_fit_has_no_observation_witness_to_report(self, pinched) -> None:
        """``nan``, not zero, and the difference is the whole point of the sentinel.

        The missing-outcome construction divides by a second mechanism and reports two
        further truncation columns.  A complete-data fit has no such mechanism, so a zero
        share would read as "nothing clipped" where the truth is "nothing to clip" -- and
        :attr:`truncations_active` would then name a truncation that does not exist on
        this fit.  The exact tuple above is what says it does not.
        """
        check = pinched.diagnostics.corrections()

        assert np.isnan(check.observation_margin)
        assert np.isnan(check.observation_clip_share)
        assert not check.has_observation_mechanism


@pytest.fixture(scope="module")
def benkeser(fit):
    """The same draw and the same nuisances, reached by the working paper's own order.

    ``fit``'s settings and ``fit``'s frame, changing exactly one thing, which is what makes
    the comparison below about the *route*: the initial nuisances are a deterministic
    function of the data and ``random_state``, so both fits enter their alternation from
    one pair and every later difference is the order's.  It is declared a dependent of
    ``fit`` rather than merely built the same way so that the two cannot drift apart
    silently if the shared settings above are edited.
    """
    del fit  # the dependency is the point; the object is refit here under the other order
    return (
        DRTMLE(**SETTINGS, update_order="benkeser")
        .fit(frame(), outcome="Y", treatment="A")
        .single()
    )


class TestBothUpdateOrdersReachTheTheoremsExit:
    r"""Item 22's numerical half, at one draw: two routes, one stated fixed point.

    The 2016 working paper states a six-step recursion (pp. 10-11;
    ``docs/technical-reference/dr-tmle/targeting.md``) and R ``drtmle`` takes a different route to the same fixed point. Reading the paper
    settled the *theoretical* half -- its step 7 states termination as the three empirical
    means being approximately zero, so the order is one way of reaching a fixed point rather
    than something Theorem 1 assumes about the collection returned -- and left the numerical
    half: whether the two routes land in the same place on real data.

    **What is checkable here is one draw, and the distribution is
    the update-order comparison.** A single fit cannot say the two orders agree
    *generally*; what it can say is that the second route exists, exits where the theorem
    asks, and does not move this fit's estimate -- and that is the precondition for the
    sweep being worth dispatching at all.  The numbers below were measured before they were
    asserted, with the tolerances set an order clear of what was seen rather than at it.

    Two rules from the concordance are obeyed rather than restated.  The comparison is of
    **scores and estimates, never fluctuation coefficients**: the submodels a round passes
    through differ between the orders, so an ``epsilon`` from one is not an ``epsilon`` from
    the other.  And it is taken at the **same nuisances**, which is what the shared frame and
    ``random_state`` buy.
    """

    def test_the_benkeser_order_exits_where_step_seven_says_it_should(self, benkeser) -> None:
        """The three empirical means, at the state this fit returned.

        This is the paper's own termination condition read off the returned collection, and
        it is the whole of what Theorem 1 asks about the route.  ``correction_check`` is what
        makes it a statement about the *reported* state rather than about what a solver
        recorded, which is the distinction items 20 and 23 were both found in.
        """
        check = benkeser.diagnostics.corrections()

        assert check.passed, check.summary()
        assert benkeser.diagnostics.score_equations().passed
        for row in check.rows:
            assert abs(row.residual) < 1e-15, row.name

    def test_the_two_routes_agree_on_the_estimate(self, fit, benkeser) -> None:
        """The update-order comparison, in the units it has to be read in.

        A difference between two fixed points is only meaningful beside the standard error
        of the thing being estimated, so the bar is a share of ``se`` rather than an absolute
        one.  Measured at ``9e-03`` of a standard error on ``ate`` here and ``7e-04`` on a
        400-row draw, and asserted at ``0.05`` -- headroom of half an order, not two, because
        what would be worth knowing is a disagreement of a *fraction* of ``se``.
        """
        for name in ESTIMANDS:
            reference = fit.estimates[name]
            difference = abs(benkeser.estimates[name].psi - reference.psi)
            assert difference < 0.05 * reference.std_error, name

    def test_but_not_exactly_on_the_reported_variance_and_that_is_the_finding(
        self, fit, benkeser
    ) -> None:
        r"""The routes agree on :math:`\hat\Psi` and disagree slightly on :math:`\sigma^2_n`.

        **Measured rather than expected**: the ``ate`` standard error is ``0.13231`` under
        this package's order and ``0.12929`` under the paper's, a ratio of ``0.977``, while
        the point estimates agree to ``9e-03`` of one.  On a 400-row draw the same ratio was
        ``1.0006``.  Both fits solve all three equations at their returned state -- ``1e-09``
        and ``6e-10`` -- so neither is unconverged, and this is not a tolerance to tighten.

        **Why it is not a contradiction of the test above.**  Step 7 constrains the three
        *empirical means*, and the reported variance is the second moment of a curve built
        from ``reduction.reduced``, which step 7 says nothing about.  The two routes refit the
        reductions at different vintages of the outcome regression by construction -- the
        paper's :math:`g_{r,1}` and :math:`g_{r,2}` come from the once-updated regression and
        its :math:`Q_r` from the twice-updated one -- and they exit holding visibly different
        ones: ``sd(g_{r,2})`` of ``0.024`` against ``0.031``, and ``sd(g_{r,2}/g_{r,1})`` of
        ``0.058`` against ``0.042``.  Different reductions, same three means, different
        corrections subtracted, and so a different :math:`\sigma^2_n`.

        So the bar here is deliberately wide and deliberately *not* a pass mark: whether a
        couple of per cent is what this gap always is, or whether it opens under weak overlap,
        requires a distribution over draws. The update-order comparison's paper-order arm reports
        exactly this ratio. What one draw can pin
        is that the gap is in the variance rather than in the estimate, which is the thing a
        reader would otherwise assume the other way round.
        """
        ratio = benkeser.estimates["ate"].std_error / fit.estimates["ate"].std_error

        assert 0.9 < ratio < 1.1, "a route difference of more than a tenth is a different claim"

    def test_and_they_are_genuinely_two_routes_rather_than_one(self, fit, benkeser) -> None:
        """The control, without which the agreement above proves nothing.

        Two fits that ran the *same* code would agree exactly, so an agreement test alone
        passes most loudly when the branch it is about has been deleted.  What separates them
        is the trace: the paper's order solves equation (8) first and refits the reductions at
        two different vintages of the outcome regression, so it reaches the fixed point by a
        different number of rounds through different intermediate states.

        Asserting *inequality* rather than a particular round count, because how many rounds
        either route takes is a property of the draw and pinning it would make this a test of
        the seed.
        """
        ours = fit.repeats[0].fluctuations["mean"].reduction
        theirs = benkeser.repeats[0].fluctuations["mean"].reduction

        assert theirs.rounds != ours.rounds or theirs.trace[0][1:] != ours.trace[0][1:]

    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            ("drtmle", ["eq9", "eq10", "eq8"]),
            ("benkeser", ["eq8", "eq10", "eq9"]),
        ],
    )
    def test_the_round_solves_the_equations_in_the_declared_order(
        self, monkeypatch, order, expected
    ) -> None:
        """The structural pin, and the one thing here that adjudicates *which* route ran.

        Every other test in this class reads a fitted result, and a result is a poor witness
        for an order: the two routes reach nearly the same place, so a paper-order fit that
        had quietly run this package's order would pass all of them but the trace control --
        and that one is a comparison of two numbers that could coincide.  This reads the
        sequence of *solves* instead, which is what the order is.

        Two hooks on the targeting module's namespace and nothing in the library moved,
        matching the bounded-targeting prototype configuration.
        The first round is all that is asserted: a round is the unit the order is defined
        over, and later rounds repeat it.
        """
        from cleverly.estimators import targeting

        seen: list[str] = []
        mechanism, submodel = targeting.solve_bounded_mechanism, targeting.solve_submodel

        def record_mechanism(*args, **kwargs):
            seen.append("eq9")
            return mechanism(*args, **kwargs)

        def record_submodel(scaled, initial, model, *args, **kwargs):
            # Equation (10)'s columns are named `h_dr<arm>` by `reduced_outcome_submodel`
            # and equation (8)'s are the `mean` group's own, so the submodel says which
            # equation this solve is without the hook counting calls or knowing the order
            # it is checking.
            seen.append("eq10" if any(name.startswith("h_dr") for name in model.names) else "eq8")
            return submodel(scaled, initial, model, *args, **kwargs)

        monkeypatch.setattr(targeting, "solve_bounded_mechanism", record_mechanism)
        monkeypatch.setattr(targeting, "solve_submodel", record_submodel)
        small, _ = nonlinear_dgp().sample(200, seed=11)
        DRTMLE(**SETTINGS, update_order=order).fit(small, outcome="Y", treatment="A")

        # The priming equation-(8) solve happens before the loop under both orders, so the
        # round starts at the second entry.
        assert seen[0] == "eq8"
        assert seen[1:4] == expected

    @pytest.mark.parametrize("order", ["drtmle", "benkeser"])
    def test_every_round_reads_equation_eight_at_the_state_it_exits_at(
        self, monkeypatch, order
    ) -> None:
        """One expectation for both orders, which is what deleting the branch bought.

        Equation (8)'s score has to describe the pair the round *exits* at, as the other two
        already do.  Under this package's order it is solved last and the restatement is a
        bit-for-bit no-op; under the paper's it is solved first and steps 4 and 6 move both
        the regression it fluctuated and the mechanism it divides by.  One unconditional call
        covers both, so there is no longer a branch that could be right for one order and
        wrong for the other -- which is the state the call used to be in, and it was
        invisible: deleting the restatement then left **68 of this module's 69 tests
        passing**, because :func:`_close_at_frozen_reductions` re-solves all three equations
        and makes the reported fit identical either way.

        What that leaves here is a call-site pin with a single expectation. The claim it
        rests on -- that recomputing a fluctuation's score at its returned state reproduces
        the recorded one exactly -- is ``tests/unit/test_fluctuation_score.py``'s, which is
        where the mutation for *that* lives.
        """
        from cleverly.estimators import targeting

        calls: list[int] = []
        original = targeting._restated_outcome_score

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(targeting, "_restated_outcome_score", counted)
        small, _ = nonlinear_dgp().sample(200, seed=11)
        fit = DRTMLE(**SETTINGS, update_order=order).fit(small, outcome="Y", treatment="A").single()

        assert len(calls) == fit.repeats[0].fluctuations["mean"].reduction.rounds

    def test_an_unknown_order_is_refused_by_name(self) -> None:
        """Both names in the message, since the wrong one is the interesting case."""
        for former in ("cleverly", "paper"):
            with pytest.raises(ValueError, match="update_order must be one of"):
                DRTMLE(update_order=former)

"""An incremental fit end to end, against a process whose truth is known.

Three claims, in the order ``tests/e2e/test_oracle_shift.py`` sets out, because a failure
in each means something different:

1. both score equations are solved -- if not, the alternation is broken;
2. the estimate matches an independently written one-step estimator -- if not, the
   plug-in or one of the two clever covariates is wrong;
3. the estimate lands within sampling error of the population truth -- if not, the
   estimand is misdefined.

Everything above this in the suite is exact but *internal*: the Gateaux module proves the
influence curve is the efficient one on a law the sample realises exactly, and the
remainder module proves the expansion. Neither would notice a fit that solved the right
equations against the wrong population. This one would.

Two further claims are specific to this axis. That a *misspecified mechanism* biases it
and a misspecified outcome regression does not -- the one-sided robustness, measured here
rather than taken from the derivation. And that on a process with real positivity trouble
the tilt's clever covariate stays bounded by ``max(delta, 1/delta)`` while the arm-indexed
one runs away, checked as a deterministic comparison rather than a coverage study.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.datasets import make_missing_outcome, make_nonlinear_ate, make_weak_overlap
from cleverly.datasets.synthetic import missing_outcome_dgp, nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.interventions import Incremental
from tests.conftest import FAST_KWARGS

DELTAS = (1.0, 2.0, 0.5)
TILTS = [Incremental(delta) for delta in DELTAS]
N = 3000

#: The outcome learner stays ``"glm"`` and the *treatment* learner does not, which is not
#: a budget compromise but the estimand's guarantee written as a configuration.  An
#: incremental intervention is robust to a wrong ``Qbar`` and not to a wrong ``g``, and
#: this process's propensity is nonlinear -- so a glm mechanism is misspecified in exactly
#: the way that has no fallback.  ``TestTheMechanismIsTheHalfThatMustBeRight`` measures
#: what that costs; the tests here need it right so they can measure something else.
#: One flexible nuisance rather than two keeps the fit at ~2s.
LEARNERS = {**FAST_KWARGS, "n_folds": 5, "treatment_learner": "fast"}


@pytest.fixture(scope="module")
def fit():
    frame, _ = make_nonlinear_ate(n=N, seed=11)
    return TMLE(**LEARNERS, incremental=TILTS).fit(frame, outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def truth():
    return nonlinear_dgp().incremental_truth(DELTAS)


class TestTheScoreEquationsAreSolved:
    def test_the_check_passes_and_covers_both_halves(self, fit) -> None:
        check = fit.validation.score_check()
        assert check.passed, check.summary()
        kinds = {row.name for row in check.rows if row.kind == "fluctuation"}
        assert kinds == {"ipsi", "ipsi (mechanism)"}

    def test_the_mechanism_equation_is_solved_to_working_precision(self, fit) -> None:
        mechanism = fit.repeats[0].fluctuations["ipsi"].mechanism
        assert mechanism.failure is None
        assert mechanism.relative_score < 1e-8

    def test_the_alternation_terminated_on_its_own(self, fit) -> None:
        """Not on the outer cap, which would mean the reported scores were a truncation."""
        mechanism = fit.repeats[0].fluctuations["ipsi"].mechanism
        assert 1 <= len(mechanism.trace) < 50


class TestAgainstAnIndependentEstimator:
    """A one-step estimator written out here, sharing no code with the fluctuation path.

    It reads the *nuisance predictions* off the fit and writes out only what lies between
    them: the tilt, the two clever covariates and the three terms of the curve.  Being a
    one-step rather than a plug-in, it should agree with the TMLE to within the difference
    between two asymptotically equivalent estimators -- which at this ``n`` is small, and
    far smaller than either one's standard error.
    """

    @staticmethod
    def one_step(fit, delta: float) -> float:
        nuisance = fit.nuisance
        scaler = nuisance.scaler
        g = nuisance.propensity.arm(1.0)
        q1 = scaler.unscale_level(nuisance.outcome.arms[1.0])
        q0 = scaler.unscale_level(nuisance.outcome.arms[0.0])
        observed = scaler.unscale_level(nuisance.outcome.observed)
        a = np.asarray(fit.data.treatment, dtype=float)
        y = np.asarray(fit.data.outcome, dtype=float)

        d = delta * g + 1.0 - g
        plug_in = (delta * g * q1 + (1.0 - g) * q0) / d
        residual = ((delta * a + 1.0 - a) / d) * (y - observed)
        mechanism = delta * (q1 - q0) / d**2 * (a - g)
        return float(np.mean(plug_in + residual + mechanism))

    @pytest.mark.parametrize("delta", DELTAS)
    def test_the_two_agree(self, fit, delta: float) -> None:
        name = "natural course" if delta == 1.0 else f"odds x{delta:g}"
        reference = self.one_step(fit, delta)
        estimate = fit.estimates[f"ey_ipsi[{name}]"]
        assert estimate.psi == pytest.approx(reference, abs=5e-3)
        # ... and the gap is well inside the noise, which is what "equivalent" means here.
        assert abs(estimate.psi - reference) < 0.5 * estimate.std_error


class TestTheTruthIsRecovered:
    @pytest.mark.parametrize(
        "name",
        [
            "ey_ipsi[natural course]",
            "ey_ipsi[odds x2]",
            "ey_ipsi[odds x0.5]",
            "ate_ipsi[odds x2 vs natural course]",
            "ate_ipsi[odds x0.5 vs natural course]",
        ],
    )
    def test_within_sampling_error(self, fit, truth, name: str) -> None:
        estimate = fit.estimates[name]
        deviation = abs(estimate.psi - truth[name])
        assert deviation < 4.0 * estimate.std_error, (
            f"{name}: {estimate.psi:.5g} vs truth {truth[name]:.5g}, "
            f"{deviation / estimate.std_error:.2f} standard errors away"
        )

    def test_tilting_up_and_down_move_opposite_ways(self, fit, truth) -> None:
        up = truth["ate_ipsi[odds x2 vs natural course]"]
        down = truth["ate_ipsi[odds x0.5 vs natural course]"]
        assert up > 0 > down, "the process must have an effect for this to test anything"
        assert fit.estimates["ate_ipsi[odds x2 vs natural course]"].psi > 0
        assert fit.estimates["ate_ipsi[odds x0.5 vs natural course]"].psi < 0


class TestTheDegenerateTilt:
    """``delta = 1`` is an identity, not an estimate, and survives every path."""

    def test_it_is_the_sample_mean(self, fit) -> None:
        assert fit.estimates["ey_ipsi[natural course]"].psi == pytest.approx(
            float(np.mean(fit.data.outcome)), abs=1e-8
        )

    def test_it_survives_canonical_common_targeting(self) -> None:
        """The canonical common update preserves an identity needing no truth."""
        frame, _ = make_nonlinear_ate(n=1200, seed=3)
        fit = (
            TMLE(**LEARNERS, incremental=TILTS, cv_evaluation=True)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert fit.estimates["ey_ipsi[natural course]"].psi == pytest.approx(
            float(np.mean(fit.data.outcome)), abs=1e-6
        )

    def test_fold_specific_mechanism_alternation_is_refused(self) -> None:
        with pytest.raises(ValueError, match="both equations re-solved inside every fold"):
            TMLE(**LEARNERS, incremental=TILTS, targeting_scheme="fold")

    def test_it_survives_repeated_cross_fitting(self) -> None:
        """Averaging over draws must not break an identity each draw satisfies."""
        frame, _ = make_nonlinear_ate(n=1200, seed=4)
        fit = (
            TMLE(**LEARNERS, incremental=TILTS, repeats=2)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert fit.estimates["ey_ipsi[natural course]"].psi == pytest.approx(
            float(np.mean(fit.data.outcome)), abs=1e-6
        )


class TestTheMechanismIsTheHalfThatMustBeRight:
    """The estimand's one-sided robustness, measured rather than asserted from theory.

    ``make_nonlinear_ate``'s propensity is nonlinear, so a ``glm`` mechanism is genuinely
    misspecified on it.  The outcome regression is left at ``glm`` throughout -- that half
    *is* protected, and holding it fixed is what isolates the mechanism as the cause.

    The shape of the damage is worth recording too: the per-tilt *means* barely move,
    because they are dominated by a level that a wrong ``g`` mixes only slightly; it is
    the *contrasts*, where the level cancels, that carry the bias. A reader who checked
    only ``ey_ipsi`` would conclude the fit was fine.
    """

    @pytest.fixture(scope="class")
    def misspecified(self):
        frame, _ = make_nonlinear_ate(n=N, seed=11)
        return (
            TMLE(**{**FAST_KWARGS, "n_folds": 5}, incremental=TILTS)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    NAME = "ate_ipsi[odds x0.5 vs natural course]"

    def test_a_misspecified_mechanism_biases_the_contrast(self, misspecified, truth) -> None:
        estimate = misspecified.estimates[self.NAME]
        assert abs(estimate.psi - truth[self.NAME]) > 3.0 * estimate.std_error

    def test_and_a_flexible_one_repairs_it_with_the_outcome_learner_unchanged(
        self, fit, truth
    ) -> None:
        estimate = fit.estimates[self.NAME]
        assert abs(estimate.psi - truth[self.NAME]) < 2.0 * estimate.std_error

    def test_the_means_hide_what_the_contrasts_show(self, misspecified, truth) -> None:
        name = "ey_ipsi[odds x0.5]"
        estimate = misspecified.estimates[name]
        assert abs(estimate.psi - truth[name]) < 3.0 * estimate.std_error


class TestWhatTheTiltBuysUnderWeakOverlap:
    """The claim the axis exists for, as a deterministic comparison of leverage.

    Not a coverage study: what is asserted is a property of the clever covariates on one
    realised sample, which is exact and costs one fit.  Whether the resulting interval
    covers is a repeated-sampling question and belongs in the nightly tier.

    ``make_weak_overlap``'s propensity is linear in ``W``, so ``glm`` estimates the
    mechanism consistently here -- which matters, because this test is about leverage and
    not about misspecification, and the two would otherwise be confounded.
    """

    def test_the_tilt_covariate_is_bounded_where_the_arm_covariate_is_not(self) -> None:
        frame, _ = make_weak_overlap(n=1500, seed=7, strength=4.0)
        fit = (
            TMLE(**{**FAST_KWARGS, "n_folds": 5}, incremental=[Incremental(2.0)])
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        g = fit.nuisance.propensity.arm(1.0)
        assert g.min() < 0.02, "the process must actually have overlap trouble"

        tilt = fit.sensitivity.incremental_support()["odds x2"]
        assert tilt.max_ratio <= 2.0 + 1e-12
        assert tilt.ess_ratio > 0.5

        # What an arm-indexed fit would divide by on the same mechanism, for contrast.
        assert (1.0 / g).max() > 50.0
        assert (1.0 / g).max() > 10.0 * tilt.max_ratio


class TestAMissingOutcomeIsCarriedByTheMissingnessModel:
    """The ``delta=`` path end to end, on the process it was refused for.

    ``incremental=`` used to refuse ``delta=``; ``tests/unit/test_influence_gateaux_ipsi_mar.py``
    shows the composition is the efficient influence function and
    ``tests/unit/test_remainder_ipsi_mar.py`` shows what the guarantee becomes.  This is
    the population check neither of those can make, and it is aimed squarely at the branch
    of that guarantee which is *new*: with ``g`` right, a consistent ``pi`` rescues an
    inconsistent ``Qbar``.  Without missingness there is nothing to rescue, since ``Qbar``
    plays no part in the incremental guarantee at all.

    ``strength=2`` is what makes that a real claim.  The outcome mean picks up curvature
    and an ``A``-by-``W1`` interaction that a main-effects GLM cannot reach, while the
    propensity and the missingness mechanism both stay linear -- so every learner here is
    ``"glm"`` and exactly one of the three is misspecified, which is the configuration the
    remainder says must still work.  A complete-case fit on the same data is the control:
    the observed rows carry a shifted ``W1``, so the same wrong outcome model extrapolated
    over them lands somewhere else.
    """

    @pytest.fixture(scope="class")
    def missing(self):
        return make_missing_outcome(n=N, seed=13, strength=2.0)[0]

    @pytest.fixture(scope="class")
    def missing_truth(self):
        return missing_outcome_dgp(2.0).incremental_truth(DELTAS)

    @pytest.fixture(scope="class")
    def missing_fit(self, missing):
        return (
            TMLE(**FAST_KWARGS, incremental=TILTS)
            .fit(missing, outcome="Y", treatment="A", delta="Delta")
            .single()
        )

    @pytest.fixture(scope="class")
    def complete_case(self, missing):
        """The same estimator on the recorded rows only, with no missingness model."""
        rows = missing[missing["Delta"] == 1].drop(columns=["Delta"])
        return TMLE(**FAST_KWARGS, incremental=TILTS).fit(rows, outcome="Y", treatment="A").single()

    def test_the_process_is_hard_enough_to_be_testing_something(self, missing) -> None:
        observed = float((missing["Delta"] == 1).mean())
        assert 0.5 < observed < 0.9, "roughly a third of the outcomes must be missing"

    @pytest.mark.parametrize(
        "name",
        [
            "ey_ipsi[natural course]",
            "ey_ipsi[odds x2]",
            "ey_ipsi[odds x0.5]",
            "ate_ipsi[odds x2 vs natural course]",
            "ate_ipsi[odds x0.5 vs natural course]",
        ],
    )
    def test_within_sampling_error_of_the_truth(self, missing_fit, missing_truth, name) -> None:
        estimate = missing_fit.estimates[name]
        deviation = abs(estimate.psi - missing_truth[name])
        assert deviation < 4.0 * estimate.std_error, (
            f"{name}: {estimate.psi:.5g} vs truth {missing_truth[name]:.5g}, "
            f"{deviation / estimate.std_error:.2f} standard errors away"
        )

    def test_a_complete_case_fit_is_visibly_worse(
        self, missing_fit, complete_case, missing_truth
    ) -> None:
        """The control. If this passed for both, the missingness model would be idle."""
        name = "ey_ipsi[odds x2]"
        corrected = abs(missing_fit.estimates[name].psi - missing_truth[name])
        dropped = abs(complete_case.estimates[name].psi - missing_truth[name])
        assert dropped > 2.0 * corrected, (
            f"complete-case is {dropped:.4g} from the truth against {corrected:.4g} "
            "corrected -- the mechanism has to be doing work here"
        )

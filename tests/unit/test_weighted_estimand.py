r"""Does a weighted fit estimate the parameter the package says it estimates?

:mod:`cleverly.data.weighting` makes a specific claim, and it is the kind of claim that
is easy to state and easy to get wrong: with observation weights :math:`w`, the estimand
is the causal parameter of the *tilted* law :math:`dP_w = w\,dP/E[w]`, and its efficient
influence function is

.. math::

    D^*_{\Psi_w}(o) = \frac{w(o)}{E[w]}\, D^*_{P_w}(o),

which is what the library reports.  "Multiply the influence curve by the weights" is
also what an implementation would do if it had never asked the question, so agreeing with
the library's own arithmetic proves nothing.  This module checks the claim from the
definition instead, on the finite-support law of :mod:`tests.discrete_law`:
:math:`\Psi(P_w)` is written out longhand from the identification formula, differentiated
along a contamination of :math:`P` by complex step, and compared to what the fit reports.
Nothing in the derivation touches a clever covariate.

Two weight functions are used, and the second is the interesting one:

* ``w = 1 + 3W/5`` -- a function of baseline covariates.  The survey case: the tilt moves
  the covariate distribution and leaves every conditional alone.
* ``w = 1 + A/2 + 4Y/5`` -- a function of the treatment *and the outcome*.  The tilt now
  changes :math:`g` and :math:`\bar Q` themselves, so an implementation that reweighted
  the final average while leaving the nuisances at :math:`P_0` would estimate something
  else entirely.  This is the case that pins the statement down.

Because the weights depend on the observed row only, the oracle nuisances handed to the
estimator are those of the tilted law -- which is exactly what a weighted learner
converges to, and what makes ``epsilon_hat`` zero so that the reported influence curve is
the EIF itself rather than an estimate of it.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.data import CausalData
from cleverly.exceptions import DataError, WeightingWarning
from tests import discrete_law as law
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment, fast_tmle

ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc", "rr", "or")

#: ``(label, weight function of (w, a, y))``.  See the module docstring.
WEIGHT_FUNCTIONS = {
    "baseline": lambda w, a, y: 1.0 + 0.6 * w,
    "treatment_and_outcome": lambda w, a, y: 1.0 + 0.5 * a + 0.8 * y,
}


def _fit(label: str) -> tuple[object, np.ndarray]:
    """A weighted, oracle-nuisance fit on the discrete law, plus its cell weights."""
    cells = law.cell_weights(WEIGHT_FUNCTIONS[label])
    tilted = law.DiscreteLaw(law.tilt(law.PROBS, cells))
    frame = law.frame().assign(w=law.row_weights(cells))
    estimator = TMLE(
        outcome_learner=OracleOutcome(tilted),
        treatment_learner=OracleTreatment(tilted),
        cross_fit=False,
        estimands="all",
        simultaneous=False,
        random_state=0,
    )
    return estimator.fit(
        frame, outcome="Y", treatment="A", covariates=["W"], weights="w"
    ).single(), cells


@pytest.fixture(scope="module", params=sorted(WEIGHT_FUNCTIONS))
def weighted_fit(request):
    return _fit(request.param)


class TestTheDerivationItself:
    """Properties the numerical derivative must have before it can referee anything."""

    @pytest.mark.parametrize("label", sorted(WEIGHT_FUNCTIONS))
    def test_the_weighted_eif_has_mean_zero_under_the_sampling_law(self, label: str) -> None:
        # Mean zero under P, not under P_w: the rows are drawn from P, and an influence
        # function is centred at the law that generates them. This is the property that
        # separates the right weighted EIF from the plausible wrong ones.
        cells = law.cell_weights(WEIGHT_FUNCTIONS[label])
        for name in ESTIMANDS:
            centred = float((law.PROBS.reshape(-1) * law.weighted_eif(name, cells)).sum())
            assert centred == pytest.approx(0.0, abs=1e-11)

    def test_constant_weights_reproduce_the_unweighted_derivation(self) -> None:
        ones = np.ones(len(law.SUPPORT))
        for name in ESTIMANDS:
            np.testing.assert_allclose(
                law.weighted_eif(name, ones), law.eif(name), rtol=0, atol=1e-12
            )

    def test_the_tilt_moves_the_estimand(self) -> None:
        # If Psi(P_w) equalled Psi(P) the comparisons below would hold for a fit that
        # ignored the weights, so the weighting has to be doing something first.
        for label in WEIGHT_FUNCTIONS:
            cells = law.cell_weights(WEIGHT_FUNCTIONS[label])
            tilted = float(law.weighted_functional(law.PROBS, "ate", cells))
            assert abs(tilted - law.TRUTH["ate"]) > 1e-3


class TestTheWeightedFitIsTheWeightedParameter:
    def test_targeting_has_nothing_left_to_do(self, weighted_fit) -> None:
        result, _ = weighted_fit
        for fluctuation in result.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_is_the_tilted_functional(self, weighted_fit, name: str) -> None:
        result, cells = weighted_fit
        estimate = result.estimates[name]
        psi = estimate.log_psi if estimate.scale == "ratio" else estimate.psi
        assert psi == pytest.approx(
            float(law.weighted_functional(law.PROBS, name, cells)), abs=1e-12
        )

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_influence_curve_is_the_weighted_eif(self, weighted_fit, name: str) -> None:
        result, cells = weighted_fit
        reported = np.asarray(result.estimates[name].influence_curve)[law.first_row_of()]
        np.testing.assert_allclose(reported, law.weighted_eif(name, cells), atol=1e-11, rtol=0)

    def test_dropping_the_normalisation_term_would_be_caught(self, weighted_fit) -> None:
        """The negative control: the check distinguishes the Hajek form from the naive one.

        The estimator is a ratio ``sum(w f) / sum(w)``, so its influence curve carries the
        centring ``w (f - psi)`` that linearises the random denominator.  Reporting
        ``w f - psi`` instead -- forgetting that the weights themselves are estimated by
        their sample mean -- is the classic error, and it has to fail here by far more
        than the tolerance above.
        """
        result, cells = weighted_fit
        truth = law.weighted_eif("ey1", cells)
        rows = law.first_row_of()
        weights = np.asarray(result.data.weights)[rows]
        psi = result.psi("ey1")
        naive = truth + (weights - 1.0) * psi
        assert np.max(np.abs(naive - truth)) > 1e-2


class TestConventions:
    def test_the_fit_is_invariant_to_the_scale_of_the_weights(self, weighted_fit) -> None:
        result, cells = weighted_fit
        tilted = law.DiscreteLaw(law.tilt(law.PROBS, cells))
        rescaled = (
            TMLE(
                outcome_learner=OracleOutcome(tilted),
                treatment_learner=OracleTreatment(tilted),
                cross_fit=False,
                estimands="all",
                simultaneous=False,
                random_state=0,
            )
            .fit(
                law.frame().assign(w=17.5 * law.row_weights(cells)),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                weights="w",
            )
            .single()
        )
        for name in ESTIMANDS:
            assert rescaled.psi(name) == pytest.approx(result.psi(name), rel=0, abs=1e-14)
            assert rescaled[name].std_error == pytest.approx(
                result[name].std_error, rel=0, abs=1e-14
            )

    def test_zero_weights_cost_nothing(self) -> None:
        """A zero-weight row leaves the estimate alone and still counts towards ``n``.

        Both halves of that are the documented convention, and the second one looks
        alarming until the arithmetic is done: the variance divides by the larger ``n``,
        but the normalisation has already scaled the surviving influence-curve values up
        by the same factor, and the two cancel.  So zero-weighting rows and deleting them
        give the same estimate *and* the same standard error, up to the degrees-of-freedom
        correction.  That is the right answer -- a zero weight excludes a row from the
        target population, it does not pretend the row was never sampled -- and it is
        worth pinning down, because an implementation that dropped the rows before
        counting ``n`` would agree here and disagree wherever the weights are not 0/1.
        """
        frame = law.frame()
        keep = np.asarray(frame["W"] != 2)
        # Oracle nuisances, so the comparison is between the two *estimators* rather than
        # between two regularised glm fits on slightly different design matrices. They are
        # correct for both fits at once: zero-weighting a stratum of W leaves every
        # conditional given W alone.
        dgp = law.DiscreteLaw()
        kwargs = {
            "outcome_learner": OracleOutcome(dgp),
            "treatment_learner": OracleTreatment(dgp),
            "cross_fit": False,
            "random_state": 0,
        }
        weighted = (
            TMLE(**kwargs)
            .fit(
                frame.assign(w=keep.astype(float)),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                weights="w",
            )
            .single()
        )
        dropped = (
            TMLE(**kwargs)
            .fit(
                frame.loc[keep].reset_index(drop=True), outcome="Y", treatment="A", covariates=["W"]
            )
            .single()
        )
        assert weighted.psi("ate") == pytest.approx(dropped.psi("ate"), rel=1e-9)
        assert weighted.n == law.N
        assert dropped.n == int(keep.sum())
        assert weighted["ate"].std_error == pytest.approx(dropped["ate"].std_error, rel=1e-3)


class TestFrequencyWeightsAreRefused:
    """Counts are a different experiment, and the package says so rather than guessing."""

    def test_weights_type_frequency_is_an_error(self) -> None:
        frame = law.frame().assign(w=1.0)
        with pytest.raises(DataError, match="frequency"):
            TMLE(outcome_learner="glm", treatment_learner="glm").fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                weights="w",
                weights_type="frequency",
            ).single()

    def test_count_looking_weights_warn(self) -> None:
        rng = np.random.default_rng(0)
        frame = law.frame().assign(w=rng.integers(1, 5, size=law.N).astype(float))
        with pytest.warns(WeightingWarning, match="counts"):
            fast_tmle(cross_fit=False).fit(
                frame, outcome="Y", treatment="A", covariates=["W"], weights="w"
            ).single()

    def test_repeating_rows_is_not_the_same_as_weighting_them(self) -> None:
        """Why the refusal is not pedantry.

        Three copies of every row and a weight of three on every row give the same point
        estimate -- the tilt is trivial either way -- but the repeated frame has three
        times the sample size, so its standard error is smaller by ``sqrt(3)``.  Reading
        counts as probability weights therefore overstates the standard error by exactly
        that factor, which is the number quoted in the error message.
        """
        frame = law.frame()
        dgp = law.DiscreteLaw()
        kwargs = {
            "outcome_learner": OracleOutcome(dgp),
            "treatment_learner": OracleTreatment(dgp),
            "cross_fit": False,
            "random_state": 0,
        }
        weighted = (
            TMLE(**kwargs)
            .fit(frame.assign(w=3.0), outcome="Y", treatment="A", covariates=["W"], weights="w")
            .single()
        )
        repeated = (
            TMLE(**kwargs)
            .fit(
                frame.loc[frame.index.repeat(3)].reset_index(drop=True),
                outcome="Y",
                treatment="A",
                covariates=["W"],
            )
            .single()
        )
        assert weighted.psi("ate") == pytest.approx(repeated.psi("ate"), rel=1e-9)
        assert weighted["ate"].std_error / repeated["ate"].std_error == pytest.approx(
            np.sqrt(3.0), rel=1e-3
        )


class TestTheReport:
    def test_the_effective_sample_size_matches_kish(self) -> None:
        cells = law.cell_weights(WEIGHT_FUNCTIONS["baseline"])
        frame = law.frame().assign(w=law.row_weights(cells))
        result = (
            fast_tmle(cross_fit=False)
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], weights="w")
            .single()
        )
        w = np.asarray(frame["w"], dtype=float)
        expected = float(w.sum() ** 2 / np.square(w).sum())
        report = result.data.weight_report()
        assert report.effective_n == pytest.approx(expected)
        assert report.design_effect == pytest.approx(law.N / expected)
        assert report.scale == pytest.approx(float(w.mean()))
        assert "weight-tilted population" in report.summary()
        assert "weight-tilted population" in result.summary()

    def test_estimated_weights_are_declared_in_the_report(self) -> None:
        frame = law.frame().assign(w=1.0 + 0.5 * law.frame()["W"])
        result = (
            fast_tmle(cross_fit=False)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                weights="w",
                weights_estimated=True,
            )
            .single()
        )
        summary = result.data.weight_report().summary()
        assert "estimated" in summary
        # The report has to name n_bootstrap as *not* the fix, or a reader told to
        # "use a bootstrap" will reach for the one this package ships.
        assert "n_bootstrap" in summary
        assert "estimated" in result.summary()

    def test_a_bootstrap_on_estimated_weights_says_it_does_not_help(self) -> None:
        """The trap: the package's own bootstrap conditions on the fitted weights too.

        Every replicate inherits the weight column and renormalises it, so bootstrapping
        adds nothing to the estimated-weight problem.  Silence here would be read as
        endorsement, since the surrounding documentation is what sends the user looking
        for a bootstrap in the first place.
        """
        frame = law.frame().assign(w=1.0 + 0.5 * law.frame()["W"])
        estimator = TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            cross_fit=False,
            estimands=("ate",),
            n_bootstrap=2,
            random_state=0,
        )
        with pytest.warns(WeightingWarning, match="n_bootstrap"):
            estimator.fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                weights="w",
                weights_estimated=True,
            ).single()

    def test_no_such_warning_for_weights_that_were_not_estimated(self) -> None:
        frame = law.frame().assign(w=1.0 + 0.5 * law.frame()["W"])
        estimator = TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            cross_fit=False,
            estimands=("ate",),
            n_bootstrap=2,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightingWarning)
            estimator.fit(frame, outcome="Y", treatment="A", covariates=["W"], weights="w").single()


class TestSampleSizeDependentSettings:
    """``n`` is not the sample size a weighted fit is working from.

    The variance takes care of itself -- normalisation scales the surviving influence
    curve values up by exactly the factor the larger ``n`` divides out, which is why
    zero-weighting rows and deleting them agree.  Everything the estimator *tunes* from
    the sample size does not take care of itself, and ``g_bounds="auto"`` is the one that
    matters: ``5 / (sqrt(n) log n)`` resolved at the row count leaves the clever covariate
    freer than the information in the sample supports, by a factor approaching three at a
    design effect of four.
    """

    def _bound(self, weights: np.ndarray | float) -> float:
        frame = law.frame().assign(w=weights)
        result = (
            fast_tmle(cross_fit=False)
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], weights="w")
            .single()
        )
        return float(result.config.g_bounds[0])

    def test_constant_weights_leave_the_auto_bound_alone(self) -> None:
        # Kish equals n exactly for constant weights, so an unweighted fit and a
        # uniformly weighted one must truncate identically.
        unweighted = (
            fast_tmle(cross_fit=False)
            .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
            .single()
        )
        assert self._bound(4.0) == pytest.approx(float(unweighted.config.g_bounds[0]))
        assert unweighted.config.auto_bounds_n is None

    def test_the_auto_bound_tightens_with_the_design_effect(self) -> None:
        cells = law.cell_weights(WEIGHT_FUNCTIONS["baseline"])
        weights = law.row_weights(cells)
        effective = float(weights.sum() ** 2 / np.square(weights).sum())
        expected = 5.0 / (np.sqrt(effective) * np.log(effective))
        assert self._bound(weights) == pytest.approx(expected)
        assert expected > 5.0 / (np.sqrt(law.N) * np.log(law.N))

    def test_zero_weighting_and_deleting_agree_on_the_bound_too(self) -> None:
        """Where the row count would have made two analyses of one subpopulation differ.

        Zero-weighting a stratum leaves a Kish effective size of exactly the number of
        retained rows, so the auto rule now hands both analyses the same truncation.
        Resolved at ``n`` they would have differed, silently, for no statistical reason.
        """
        frame = law.frame()
        keep = np.asarray(frame["W"] != 2)
        dropped = (
            fast_tmle(cross_fit=False)
            .fit(
                frame.loc[keep].reset_index(drop=True), outcome="Y", treatment="A", covariates=["W"]
            )
            .single()
        )
        assert self._bound(keep.astype(float)) == pytest.approx(
            float(dropped.config.g_bounds[0]), rel=1e-12
        )

    def test_the_summary_says_which_sample_size_the_bound_came_from(self) -> None:
        cells = law.cell_weights(WEIGHT_FUNCTIONS["baseline"])
        frame = law.frame().assign(w=law.row_weights(cells))
        result = (
            fast_tmle(cross_fit=False)
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], weights="w")
            .single()
        )
        assert "resolved at the effective n" in result.summary()
        assert result.config.auto_bounds_n == pytest.approx(result.data.effective_n)

    def test_an_explicit_bound_is_never_second_guessed(self) -> None:
        cells = law.cell_weights(WEIGHT_FUNCTIONS["baseline"])
        frame = law.frame().assign(w=law.row_weights(cells))
        result = (
            TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                cross_fit=False,
                g_bounds=0.01,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], weights="w")
            .single()
        )
        assert result.config.g_bounds == (0.01, 0.99)
        assert result.config.auto_bounds_n is None

    def test_concentrated_weights_warn_at_construction(self) -> None:
        # A quarter of the rows carrying almost all the mass: design effect above 4, so
        # the estimate and every sample-size-dependent setting rest on a small part of
        # the sample and the user is told without having to ask for a report.
        # Non-integer, so the count heuristic stays quiet and this asserts one thing.
        heavy = np.where(np.arange(law.N) < law.N // 40, 50.5, 1.0)
        with pytest.warns(WeightingWarning, match="concentrated"):
            CausalData.from_frame(
                law.frame().assign(w=heavy),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                weights="w",
            )

    def test_ordinary_weights_do_not_warn(self) -> None:
        cells = law.cell_weights(WEIGHT_FUNCTIONS["baseline"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightingWarning)
            CausalData.from_frame(
                law.frame().assign(w=law.row_weights(cells)),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                weights="w",
            )

    def test_constant_weights_report_nothing_to_report(self) -> None:
        cells = np.ones(len(law.SUPPORT))
        frame = law.frame().assign(w=law.row_weights(cells))
        result = (
            fast_tmle(cross_fit=False)
            .fit(frame, outcome="Y", treatment="A", covariates=["W"], weights="w")
            .single()
        )
        assert not result.data.is_weighted
        assert "unweighted" in result.data.weight_report().summary()


class TestWeightsAndMissingOutcomesTogether:
    r"""The two corrections compose -- checked, not assumed.

    Weighting and missingness both reweight, and it is not obvious from the code that they
    reweight *compatibly*: the observation weights multiply the influence curve row-wise,
    while :math:`1/\pi` sits inside the clever covariate, and the missingness model is
    itself fitted under the weights.  Composing them wrongly would be easy and invisible
    -- the score equation would still be solved, and the estimate would still look
    reasonable.

    The claim is that nothing special happens: the estimand is still
    :math:`\Psi(P_w)`, the identification formula is still applied at the tilted law, and
    the influence curve is still :math:`(w/E[w])\,D^*_{P_w}`.  What makes that testable is
    that the tilt moves the missingness mechanism too -- a weighted missingness learner
    converges to :math:`P_w(\Delta = 1 \mid A, W)`, not to :math:`P_0`'s -- so the oracle
    handed to the estimator is the tilted law's, and the derivative is taken of
    :math:`\Psi(P_w)` written out longhand over the observed-data support.
    """

    #: Weights of ``(w, a, k)``.  ``k`` is the observed-data outcome cell, so a weight can
    #: depend on ``Y`` only where ``Y`` exists -- which is the only kind of outcome-
    #: dependent weight a real design could supply.
    WEIGHT_FUNCTIONS: ClassVar[dict[str, Any]] = {
        "baseline": lambda w, a, k: 1.0 + 0.6 * w,
        "arm_and_outcome": lambda w, a, k: 1.0 + 0.5 * a + 0.8 * (k == mar.OBSERVED_ONE),
    }

    @staticmethod
    def _fit(label: str):
        cells = mar.cell_weights(TestWeightsAndMissingOutcomesTogether.WEIGHT_FUNCTIONS[label])
        tilted = mar.DiscreteLaw(mar.tilt(mar.PROBS, cells))
        frame = mar.frame().assign(w=mar.row_weights(cells))
        estimator = TMLE(
            outcome_learner=OracleOutcome(tilted),
            treatment_learner=OracleTreatment(tilted),
            missingness_learner=OracleMissingness(tilted),
            cross_fit=False,
            estimands="all",
            simultaneous=False,
            random_state=0,
        )
        fitted = estimator.fit(
            frame, outcome="Y", treatment="A", covariates=["W"], delta="Delta", weights="w"
        ).single()
        return fitted, cells

    @pytest.fixture(scope="class", params=sorted(WEIGHT_FUNCTIONS))
    def fit(self, request):
        return self._fit(request.param)

    def test_an_outcome_dependent_tilt_moves_the_missingness_mechanism(self) -> None:
        # The premise that makes this more than a restatement of the unweighted case. If
        # P_w(Delta = 1 | A, W) equalled P_0's, an implementation that fitted the
        # missingness model unweighted would pass every assertion below.  A weight that
        # depends on whether -- and on what -- the outcome was recorded is what moves it.
        cells = mar.cell_weights(self.WEIGHT_FUNCTIONS["arm_and_outcome"])
        tilted = mar.DiscreteLaw(mar.tilt(mar.PROBS, cells))
        assert np.max(np.abs(tilted.pi - mar.PI)) > 0.05
        assert np.max(np.abs(tilted.g - mar.G)) > 1e-3
        assert np.max(np.abs(tilted.q - mar.Q)) > 1e-3

    def test_a_baseline_tilt_moves_only_the_covariate_distribution(self) -> None:
        # The other half of the pair, and the reason both are worth running: a weight
        # that is a function of W alone leaves every conditional -- including the
        # missingness mechanism -- exactly where it was, and reweights only the marginal
        # the plug-in averages against.
        cells = mar.cell_weights(self.WEIGHT_FUNCTIONS["baseline"])
        tilted = mar.DiscreteLaw(mar.tilt(mar.PROBS, cells))
        np.testing.assert_allclose(tilted.pi, mar.PI, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tilted.g, mar.G, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tilted.q, mar.Q, rtol=0, atol=1e-12)

    def test_targeting_has_nothing_left_to_do(self, fit) -> None:
        result, _ = fit
        for fluctuation in result.fluctuations.values():
            assert np.max(np.abs(fluctuation.epsilon)) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_point_estimate_is_the_tilted_functional(self, fit, name: str) -> None:
        result, cells = fit
        estimate = result.estimates[name]
        psi = estimate.log_psi if estimate.scale == "ratio" else estimate.psi
        assert psi == pytest.approx(
            float(mar.weighted_functional(mar.PROBS, name, cells)), abs=1e-12
        )

    @pytest.mark.parametrize("name", ESTIMANDS)
    def test_the_influence_curve_is_the_weighted_eif(self, fit, name: str) -> None:
        result, cells = fit
        reported = np.asarray(result.estimates[name].influence_curve)[mar.first_row_of()]
        np.testing.assert_allclose(reported, mar.weighted_eif(name, cells), atol=1e-11, rtol=0)

    def test_ignoring_the_weights_would_be_caught(self, fit) -> None:
        # The negative control for the composition: the weighted and unweighted estimands
        # have to be far enough apart that agreement above is evidence of something.
        _, cells = fit
        assert (
            abs(float(mar.weighted_functional(mar.PROBS, "ate", cells)) - mar.TRUTH["ate"]) > 1e-3
        )

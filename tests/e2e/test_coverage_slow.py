r"""The statistical validation tier: does the estimator's *inference* work?

Everything in the fast tier checks that the estimator computes what it claims on a
given sample.  This tier checks the claim that actually matters to a user reading a
p-value: that the reported uncertainty is honest.

Three properties, each of which fails in a different, diagnosable way:

**Coverage.**  A nominal 95% interval must contain the truth about 95% of the time.
Under-coverage means every inference drawn from the estimator is overconfident, and it
is invisible on any single dataset.

**Root-n consistency.**  :math:`\sqrt{n} \times \mathrm{bias}` must stay bounded as
:math:`n` grows.  Bias that shrinks more slowly than :math:`n^{-1/2}` eventually
dominates the standard error, and coverage degrades as the sample gets *larger* -- the
opposite of the usual intuition, and the reason this is checked as a rate rather than a
level.

**Type I error.**  Under a null process the rejection rate must sit near the nominal
level.  This is the direct check that the whole machinery is calibrated.

These runs take minutes rather than seconds, so they are marked ``slow`` and run
nightly rather than on every push.  The thresholds are set from the Monte Carlo standard
error of each quantity, so a pass is evidence rather than a formality.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import DGP, binary_outcome_dgp, linear_dgp, nonlinear_dgp
from cleverly.utils.bounds import expit
from cleverly.validation import CoverageStudy

pytestmark = pytest.mark.slow

#: Replications per study.  At 400 the Monte Carlo standard error of a coverage estimate
#: near 0.95 is about 1.1 percentage points, enough to resolve a real shortfall.
REPLICATES = 400


def _study(
    dgp: DGP,
    *,
    n: int,
    reps: int = REPLICATES,
    estimands: tuple[str, ...] = ("ate",),
    library: str = "glm",
    fit_kwargs: dict[str, object] | None = None,
    seed: int = 2024,
) -> object:
    return CoverageStudy(
        dgp=dgp,
        estimator=lambda: TMLE(
            outcome_learner=library,
            treatment_learner=library,
            n_folds=5,
            learner_folds=3,
            estimands=estimands,
            simultaneous=False,
            random_state=0,
        ),
        n=n,
        n_replicates=reps,
        estimands=estimands,
        fit_kwargs=fit_kwargs or {"outcome": "Y", "treatment": "A"},
        seed=seed,
        n_jobs=2,
    ).run()


class TestCoverage:
    @pytest.mark.parametrize("n", [500, 2000])
    def test_confidence_intervals_cover_at_the_nominal_rate(self, n: int) -> None:
        summary = _study(linear_dgp(), n=n)["ate"]
        # Both nuisance models are correctly specified here, so nothing but the
        # inference machinery can be responsible for a shortfall.
        assert 0.93 <= summary.coverage <= 0.97, summary
        assert abs(summary.coverage - 0.95) < 3.0 * summary.coverage_se

    def test_the_reported_standard_error_matches_the_actual_variability(self) -> None:
        summary = _study(linear_dgp(), n=1000)["ate"]
        # The ratio of the mean reported standard error to the observed spread of the
        # estimates. Below one means the reported uncertainty is optimistic, which is
        # how under-coverage almost always arises.
        assert 0.93 <= summary.se_ratio <= 1.07, summary

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc", "ey1", "ey0"])
    def test_every_estimand_covers(self, estimand: str) -> None:
        summary = _study(linear_dgp(), n=1000, estimands=("ate", "att", "atc", "ey1", "ey0"))[
            estimand
        ]
        assert 0.92 <= summary.coverage <= 0.98, summary

    def test_ratio_estimands_cover_on_the_log_scale(self) -> None:
        study = _study(
            binary_outcome_dgp(),
            n=2000,
            estimands=("rr", "or"),
        )
        for name in ("rr", "or"):
            assert 0.92 <= study[name].coverage <= 0.98, study[name]

    def test_flexible_learners_still_cover(self) -> None:
        # The case that motivates cross-fitting: machine-learning nuisance estimators,
        # where in-sample predictions would destroy the coverage guarantee. Fewer
        # replications than the parametric studies because each fit costs ~10x as much;
        # 120 still resolves the +/- 0.05 window asserted here.
        summary = _study(nonlinear_dgp(), n=1000, reps=120, library="fast")["ate"]
        assert 0.90 <= summary.coverage <= 0.99, summary


class TestConsistency:
    def test_root_n_bias_stays_bounded(self) -> None:
        sizes = (500, 2000, 8000)
        scaled = []
        for n in sizes:
            summary = _study(linear_dgp(), n=n, reps=200)["ate"]
            scaled.append(abs(summary.root_n_bias))
        # sqrt(n) * bias must not grow with n. Allowing a factor of 2.5 leaves room for
        # Monte Carlo noise while still catching a bias that vanishes too slowly.
        assert scaled[-1] < 2.5 * min(scaled), dict(zip(sizes, scaled, strict=True))

    def test_the_standard_error_halves_when_n_quadruples(self) -> None:
        small = _study(linear_dgp(), n=500, reps=200)["ate"]
        large = _study(linear_dgp(), n=2000, reps=200)["ate"]
        assert large.mean_std_error / small.mean_std_error == pytest.approx(0.5, abs=0.05)

    def test_the_estimate_converges_on_the_truth(self) -> None:
        small = _study(nonlinear_dgp(), n=400, reps=100, library="fast")["ate"]
        large = _study(nonlinear_dgp(), n=1600, reps=100, library="fast")["ate"]
        assert abs(large.bias) < abs(small.bias)


def _null_dgp() -> DGP:
    """A process with confounding but genuinely no treatment effect."""

    def propensity(w: np.ndarray) -> np.ndarray:
        return expit(0.5 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del a, z  # the treatment does not enter: the sharp null holds
        return 1.0 + 0.9 * w[:, 0] + 0.6 * w[:, 1] - 0.4 * w[:, 2]

    return DGP(
        name="null_effect",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


class TestTypeIError:
    def test_the_rejection_rate_is_near_the_nominal_level(self) -> None:
        dgp = _null_dgp()
        assert dgp.truth()["ate"] == pytest.approx(0.0, abs=1e-9)
        summary = _study(dgp, n=1000)["ate"]
        # Under the null, "reject at 5%" should happen 5% of the time. The binomial
        # standard error at 400 replications is about 1.1 points.
        assert abs(summary.rejection_rate - 0.05) < 3.0 * np.sqrt(
            0.05 * 0.95 / summary.n_replicates
        ), summary
        assert 0.93 <= summary.coverage <= 0.97

    def test_the_null_estimate_is_unbiased(self) -> None:
        summary = _study(_null_dgp(), n=1000)["ate"]
        assert abs(summary.bias) < 3.0 * summary.bias_se


class TestCvTmleUnderWeakOverlap:
    def test_cross_fitting_protects_coverage_with_flexible_learners(self) -> None:
        """In-sample nuisance fits are what cross-fitting exists to prevent.

        Without cross-fitting, a flexible learner's residuals are too small, the
        influence-curve variance is understated, and coverage falls.  This compares the
        two directly rather than asserting the fix works in the abstract.
        """
        common = {
            "outcome_learner": "fast",
            "treatment_learner": "fast",
            "n_folds": 5,
            "learner_folds": 3,
            "estimands": ("ate",),
            "simultaneous": False,
            "random_state": 0,
        }
        results = {}
        for label, cross_fit in (("cross-fitted", True), ("in-sample", False)):
            results[label] = CoverageStudy(
                dgp=nonlinear_dgp(),
                estimator=lambda cf=cross_fit: TMLE(cross_fit=cf, **common),
                n=600,
                n_replicates=120,
                estimands=("ate",),
                seed=99,
                n_jobs=2,
            ).run()["ate"]

        assert results["cross-fitted"].coverage > results["in-sample"].coverage
        assert results["cross-fitted"].coverage >= 0.90


class TestClusteredInference:
    def test_cluster_robust_intervals_cover_when_plain_ones_do_not(self) -> None:
        from cleverly.datasets import clustered_dgp

        dgp = clustered_dgp(cluster_size=10)
        columns = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
        ignoring = _study(dgp, n=1000, reps=200, fit_kwargs=columns)["ate"]
        clustered = _study(dgp, n=1000, reps=200, fit_kwargs={**columns, "id": "cluster"})["ate"]
        # The DGP shares an unobserved latent within clusters. Ignoring it understates
        # the variance and coverage collapses; accounting for it restores calibration.
        assert ignoring.coverage < 0.90
        assert clustered.coverage > ignoring.coverage + 0.03
        assert clustered.se_ratio > ignoring.se_ratio

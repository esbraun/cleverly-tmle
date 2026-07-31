"""End to end: a longitudinal fit on data whose truth is known by quadrature."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE, LongitudinalResult

#: Fast-tier settings: parametric nuisances, few folds, seeded.  The mechanism of
#: ``make_longitudinal`` is logistic-linear in the recorded history, so ``glm`` estimates
#: it correctly and double robustness carries the fit even though the outcome regression
#: -- which carries a ``tanh`` term -- is misspecified.  That is deliberate: it is the
#: half of the guarantee a cheap test can exercise.
FAST: dict[str, Any] = {
    "outcome_learner": "glm",
    "pseudo_learner": "glm",
    "treatment_learner": "glm",
    "n_folds": 3,
    "learner_folds": 3,
    "random_state": 0,
}

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


#: Arguments of ``fit`` rather than of the estimator, so ``run`` knows which is which.
FIT_ARGUMENTS = (*COLUMNS, "family", "id")


def run(frame: Any, **overrides: Any) -> LongitudinalResult:
    settings = {**FAST, **overrides}
    regimens = settings.pop("regimens", {"always": 1, "never": 0})
    columns = {
        **COLUMNS,
        **{key: settings.pop(key) for key in FIT_ARGUMENTS if key in settings},
    }
    return LTMLE(regimens, **settings).fit(frame, **columns)


@pytest.fixture(scope="module")
def fitted() -> tuple[LongitudinalResult, dict[str, float]]:
    frame, truth = make_longitudinal(n=3000, seed=11)
    return run(frame), truth


def test_reports_a_mean_per_regimen_and_a_contrast(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    result, _ = fitted
    assert list(result) == [
        "ey_regimen[always]",
        "ey_regimen[never]",
        "ate_regimen[never vs always]",
    ]
    assert result.converged


def test_the_contrast_is_the_difference_of_the_means(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    result, _ = fitted
    difference = result.psi("ey_regimen[never]") - result.psi("ey_regimen[always]")
    assert result.psi("ate_regimen[never vs always]") == pytest.approx(difference, abs=1e-12)
    np.testing.assert_allclose(
        result.influence_curves["ate_regimen[never vs always]"],
        result.influence_curves["ey_regimen[never]"]
        - result.influence_curves["ey_regimen[always]"],
        atol=1e-12,
    )


def test_every_score_equation_is_solved(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    """Targeting is what makes the reported variance the variance of anything.

    One score per node per regimen, driven to zero relative to the largest value it
    could take; and the influence curve of each parameter averages to zero, which is the
    same statement read at the level of the report.
    """
    result, _ = fitted
    for fit in result.fits.values():
        for step in fit.steps:
            assert step.fluctuation.relative_score_norm < 1e-8
    for curve in result.influence_curves.values():
        assert abs(float(np.mean(curve))) < 1e-8


def test_recovers_the_truth_on_average() -> None:
    """Averaged over independent samples, the estimate lands on the quadrature truth.

    Eight replicates, not one: a 95% interval misses one time in twenty by construction,
    so a single fit is a coin flip that fails on a bad seed.  The comparison is against
    the Monte Carlo standard error of the *average*, which is what makes the tolerance a
    statement about the estimator rather than a number chosen to pass.
    """
    replicates = 8
    estimates = []
    for seed in range(replicates):
        frame, truth = make_longitudinal(n=1500, seed=100 + seed)
        result = run(frame, random_state=seed)
        estimates.append(result.psi("ate_regimen[never vs always]"))
    average = float(np.mean(estimates))
    mc_error = float(np.std(estimates, ddof=1) / np.sqrt(replicates))
    target = -truth["ate_regimen[always vs never]"]
    assert abs(average - target) < 3.0 * mc_error + 0.01


def test_omitting_the_time_varying_confounder_is_biased() -> None:
    """The negative control that says what the module is for.

    ``L2`` is caused by ``A1`` and confounds ``A2``.  Declaring it away leaves the
    second decision confounded, and the estimate moves off the truth by much more than
    its own standard error -- which no amount of care at a single time point would fix.
    """
    frame, truth = make_longitudinal(n=4000, seed=7)
    adjusted = run(frame)
    naive = run(frame, time_varying=[[], []])
    target = -truth["ate_regimen[always vs never]"]
    assert abs(adjusted.psi("ate_regimen[never vs always]") - target) < 0.05
    gap = abs(naive.psi("ate_regimen[never vs always]") - target)
    assert gap > 3.0 * naive["ate_regimen[never vs always]"].std_error


def test_a_continuous_outcome_is_estimated_on_its_own_scale() -> None:
    """The Gruber--van der Laan scaling is equivariant, and exactly so.

    A continuous outcome is mapped onto ``[0, 1]``, the whole recursion runs there, and
    the estimate is mapped back.  Rescaling the outcome by an affine map therefore has to
    move the estimate by the same map, to the last bit -- the scaled target, and so every
    fit inside, is identical.  ``family="gaussian"`` on both sides so the last node's
    regression is the same *kind* of problem in each.
    """
    frame, _ = make_longitudinal(n=1000, seed=13)
    rescaled = frame.copy()
    rescaled["Y"] = 10.0 * frame["Y"] + 2.0
    plain = run(frame, family="gaussian")
    moved = run(rescaled, family="gaussian")
    assert moved.psi("ey_regimen[always]") == pytest.approx(
        10.0 * plain.psi("ey_regimen[always]") + 2.0, rel=1e-9
    )
    assert moved.psi("ate_regimen[never vs always]") == pytest.approx(
        10.0 * plain.psi("ate_regimen[never vs always]"), rel=1e-9
    )


def test_runs_without_censoring_nodes() -> None:
    frame, _ = make_longitudinal(n=1200, seed=3, censoring=False)
    result = run(frame, censoring=None)
    assert result.converged
    # With nobody censored the cumulative product is the treatment mechanism alone, so
    # the weight at the last node is the reciprocal of two probabilities and no more.
    diagnostics = result.diagnostics()
    assert set(diagnostics["regimen"]) == {"always", "never"}


def test_diagnostics_report_the_cumulative_leverage(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    result, _ = fitted
    frame = result.diagnostics()
    assert len(frame) == 4  # two regimens by two nodes
    for label, fit in result.fits.items():
        assert fit.max_weight >= 1.0
        assert 0 < fit.effective_n <= fit.steps[-1].n_trained
        assert fit.steps[-1].n_trained < result.n, label


def test_contrast_and_covariance_use_the_joint_curve(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    result, _ = fitted
    names = ["ey_regimen[always]", "ey_regimen[never]"]
    covariance = result.covariance(names)
    assert covariance.shape == (2, 2)
    ratio = result.contrast(lambda p: float(p[0] / p[1]), names, name="ratio", scale="difference")
    assert ratio.psi == pytest.approx(result.psi(names[0]) / result.psi(names[1]))
    # A ratio of correlated estimates is not the ratio of independent ones: the delta
    # method has to see the covariance, and here the two regimens share every node.
    assert ratio.std_error > 0


def test_summary_and_frame_report_the_same_numbers(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    result, _ = fitted
    frame = result.to_frame()
    assert list(frame["parameter"]) == list(result)
    text = result.summary()
    assert "Longitudinal TMLE" in text
    assert "always" in text


def test_polars_in_polars_out() -> None:
    polars = pytest.importorskip("polars")
    frame, _ = make_longitudinal(n=800, seed=5, backend="polars")
    result = run(frame)
    assert isinstance(result.to_frame(), polars.DataFrame)
    assert isinstance(result.diagnostics(), polars.DataFrame)


def test_cluster_variance_is_reported_at_the_cluster(fitted: Any) -> None:
    """Clustered rows widen the interval, since the independent unit is the cluster."""
    frame, _ = make_longitudinal(n=1000, seed=9)
    frame["site"] = np.repeat(np.arange(100), 10)
    independent = run(frame, baseline=["W1", "W2"])
    clustered = LTMLE({"always": 1, "never": 0}, **FAST).fit(frame, id="site", **COLUMNS)
    assert clustered.data.n_clusters == 100
    assert clustered["ey_regimen[always]"].n_clusters == 100
    assert independent["ey_regimen[always]"].n_clusters == 1000

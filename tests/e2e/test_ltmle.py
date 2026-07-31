"""End to end: a longitudinal fit on data whose truth is known by quadrature."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("family", ["binomial", "gaussian"])
def test_the_contrast_is_the_difference_of_the_means(family: str) -> None:
    """And on a continuous outcome too, where it is not a tautology.

    On a binary outcome the scaler is the identity, so this reduces to ``a - b == a - b``
    and would pass whatever ``unscale_difference`` did.  A continuous outcome puts a real
    affine map between the recursion's ``[0, 1]`` scale and the report, and the identity
    then says that map is linear on differences -- which is the claim worth making.
    """
    frame, _ = make_longitudinal(n=1000, seed=31)
    if family == "gaussian":
        frame = frame.copy()
        frame["Y"] = 7.0 * frame["Y"] - 3.0
    result = run(frame, family=family)
    difference = result.psi("ey_regimen[never]") - result.psi("ey_regimen[always]")
    assert result.psi("ate_regimen[never vs always]") == pytest.approx(difference, abs=1e-12)
    np.testing.assert_allclose(
        result.influence_curves["ate_regimen[never vs always]"],
        result.influence_curves["ey_regimen[never]"]
        - result.influence_curves["ey_regimen[always]"],
        atol=1e-12,
        rtol=0,
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


def three_node_frame(n: int, seed: int) -> tuple[Any, float]:
    """Three treatment nodes, and the truth for "always" by direct simulation.

    Under the intervention the mechanism drops out, so the counterfactual mean is an
    ordinary expectation of the outcome regression over the intervened covariate process
    -- simulated here at a size that puts the Monte Carlo error near 1e-3.  Local to this
    test rather than added to ``cleverly.datasets``: it exists to give the recursion a
    middle node, not to be a process anyone would analyse.
    """
    from scipy.special import expit

    def draw(size: int, rng: Any, plan: tuple[float, float, float] | None) -> Any:
        w = rng.standard_normal(size)
        a1 = plan[0] if plan else rng.binomial(1, expit(0.4 * w)).astype(float)
        l2 = 0.5 * w + 0.8 * a1 + rng.standard_normal(size)
        a2 = plan[1] if plan else rng.binomial(1, expit(0.5 * l2 + 0.3 * a1)).astype(float)
        l3 = 0.4 * l2 + 0.7 * a2 + rng.standard_normal(size)
        a3 = plan[2] if plan else rng.binomial(1, expit(0.5 * l3 + 0.3 * a2)).astype(float)
        index = -0.3 + 0.4 * a1 + 0.5 * a2 + 0.6 * a3 + 0.3 * l3 + 0.2 * w
        return w, a1, l2, a2, l3, a3, expit(index), rng

    rng = np.random.default_rng(seed)
    w, a1, l2, a2, l3, a3, probability, rng = draw(n, rng, None)
    frame = pd.DataFrame(
        {
            "W": w,
            "A1": a1,
            "L2": l2,
            "A2": a2,
            "L3": l3,
            "A3": a3,
            "Y": rng.binomial(1, probability).astype(float),
        }
    )
    *_, truth_probability, _ = draw(400_000, np.random.default_rng(seed + 1), (1.0, 1.0, 1.0))
    return frame, float(np.mean(truth_probability))


def test_a_three_node_recursion_recovers_the_truth() -> None:
    """The middle node exists only from ``T = 3`` on, and it is the interesting one.

    At two nodes every node is either the last -- regressing the outcome itself -- or the
    first -- whose prediction nothing consumes.  The node in the middle is the only one
    that is both a pseudo-outcome regression *and* has a targeted prediction handed to it
    from behind and handed on in front, which is where "the recursion carries the targeted
    prediction forward, not the initial one" can actually go wrong.  Every earlier fit and
    every oracle law in this repository has two nodes.
    """
    frame, truth = three_node_frame(n=4000, seed=41)
    result = LTMLE(
        {"always": 1, "never": 0},
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=3,
        learner_folds=3,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2", "A3"],
        baseline=["W"],
        time_varying=[[], ["L2"], ["L3"]],
    )
    assert result.converged
    fit = result.fits["always"]
    assert [step.time for step in fit.steps] == [1, 2, 3]
    # Every one of the three score equations, not just the two a T=2 fit has.
    for step in fit.steps:
        assert step.fluctuation.relative_score_norm < 1e-8
    assert len(result.diagnostics()) == 6  # two regimens by three nodes
    estimate = result["ey_regimen[always]"]
    assert abs(estimate.psi - truth) < 3.0 * estimate.std_error


def test_the_three_node_recursion_carries_the_targeted_prediction_forward() -> None:
    """The middle node's outcome is the *targeted* later prediction, not the initial one.

    Read off the fit rather than argued: the pseudo-outcome each node was regressed on is
    reconstructed from the next node's ``targeted`` array, and the two must agree on the
    rows the regression saw.  Reading ``initial`` there instead would leave the residual
    of one node to accumulate into the next, which is exactly the bug this rules out.
    """
    frame, _ = three_node_frame(n=1500, seed=42)
    result = LTMLE(
        {"always": 1},
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=3,
        learner_folds=3,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2", "A3"],
        baseline=["W"],
        time_varying=[[], ["L2"], ["L3"]],
    )
    steps = {step.time: step for step in result.fits["always"].steps}
    for time in (1, 2):
        later = steps[time + 1]
        # Targeting moved the later node, so "targeted" and "initial" are distinguishable
        # and the check below has something to distinguish.
        assert np.max(np.abs(later.targeted - later.initial)) > 1e-6
        # Each node's influence-curve term is h_t (Q*_{t+1} - Q*_t): built from the
        # targeted prediction of the node behind it.
        trained = steps[time].trained_on
        assert trained.any()
        assert np.all(np.isfinite(later.targeted[trained]))


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
    """With nobody censored the cumulative product is the treatment mechanism alone.

    The claim the old assertion only stated in a comment: with ``censoring=None`` the
    denominator has ``T`` factors rather than ``2T``, so the clever covariate at the last
    node is exactly the reciprocal of the two treatment probabilities.  Recomputed here
    from the mechanism the fit stored, which is the only way to see that the censoring
    branch really did drop out rather than contributing a silent one.
    """
    frame, _ = make_longitudinal(n=1200, seed=3, censoring=False)
    result = run(frame, censoring=None)
    assert result.converged
    fit = result.fits["always"]
    lower, upper = result.config.g_bounds
    g1 = np.clip(result.mechanism.treatment[0]["always"], lower, upper)
    g2 = np.clip(result.mechanism.treatment[1]["always"], lower, upper)
    np.testing.assert_allclose(fit.cumulative[:, -1], g1 * g2, atol=1e-12, rtol=0)
    followed = fit.steps[-1].trained_on
    np.testing.assert_allclose(
        fit.steps[-1].clever[followed], 1.0 / (g1 * g2)[followed], atol=1e-10, rtol=0
    )


def test_diagnostics_report_the_cumulative_leverage(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    """The reported leverage is the leverage of the right node, over the right rows.

    ``max_weight >= 1`` and ``effective_n <= n_trained`` are arithmetic and Cauchy--
    Schwarz respectively -- neither can fail, so neither says anything.  What can fail is
    reading the weight off the wrong node or the wrong mask, which is what this pins:
    every number in the diagnostics frame is recomputed here from the step it claims to
    describe.
    """
    result, _ = fitted
    frame = result.diagnostics()
    assert len(frame) == 4  # two regimens by two nodes
    rows = {(row["regimen"], row["time"]): row for _, row in frame.iterrows()}
    for label, fit in result.fits.items():
        for step in fit.steps:
            row = rows[(label, step.time)]
            weights = step.clever[step.trained_on]
            assert row["n_followed"] == int(step.trained_on.sum())
            assert row["max_weight"] == pytest.approx(float(np.max(weights)), abs=0)
            kish = float(np.sum(weights) ** 2 / np.sum(weights**2))
            assert row["effective_n"] == pytest.approx(kish, abs=0)
        # The summary quotes the *final* node, where the product is longest and the
        # leverage is therefore largest -- not the first, and not an average.
        assert fit.max_weight == pytest.approx(float(np.max(fit.steps[-1].clever)), abs=0)
        assert fit.max_weight >= max(float(np.max(step.clever)) for step in fit.steps[:-1])
        assert rows[(label, result.data.n_times)]["effective_n"] == pytest.approx(
            fit.effective_n, abs=0
        )


def test_each_mechanism_factor_is_truncated_before_the_product() -> None:
    """Truncating the factors and truncating the product are different operations.

    They coincide at one time point, which is why this needs saying at all.  Over ``T``
    nodes a bound of ``b`` on each factor caps the cumulative product's *reciprocal* at
    ``b**-2T``; bounding the product afterwards would cap it at ``b**-1``.  With a bound
    this tight the two differ by orders of magnitude, so a fit that truncated the product
    would fail here rather than merely reporting a smaller weight.
    """
    frame, _ = make_longitudinal(n=1200, seed=17)
    tight = run(frame, g_bounds=(0.3, 0.7))
    fit = tight.fits["always"]
    # Four factors -- two treatment, two censoring -- each at least 0.3.
    assert fit.max_weight <= 0.3**-4 + 1e-9
    assert fit.max_weight > 0.3**-1
    lower, upper = tight.config.g_bounds
    expected = np.ones(tight.n)
    for time in range(tight.data.n_times):
        expected = expected * np.clip(tight.mechanism.treatment[time]["always"], lower, upper)
        expected = expected * np.clip(tight.mechanism.censoring[time]["always"], lower, upper)
    np.testing.assert_allclose(fit.cumulative[:, -1], expected, atol=1e-12, rtol=0)
    # Note the weight is *larger* here than under the default bound, not smaller: the
    # upper bound clips a near-one probability down too, which shrinks the product.  That
    # is the same asymmetry `Propensity.bounded` keeps at one time point, and it is why
    # "tighter g_bounds" is not a reliable way to reduce leverage in either estimator.
    assert fit.max_weight > run(frame).fits["always"].max_weight


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
    assert list(frame["estimand"]) == list(result)
    text = result.summary()
    assert "Longitudinal TMLE" in text
    assert "always" in text


def test_the_tidy_frame_uses_the_same_column_names_as_a_point_treatment_fit(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    """One library, one set of column names.

    Both result types build the row from ``ParameterEstimate.to_dict``, so a caller who
    writes ``frame["psi"]`` does not have to know which estimator produced the frame.
    """
    result, _ = fitted
    frame = result.to_frame()
    assert set(frame.columns) == set(result["ey_regimen[always]"].to_dict())
    assert {"estimand", "psi", "std_err", "ci_lower", "ci_upper", "p_value"} <= set(frame.columns)
    row = dict(zip(frame.columns, frame.to_numpy()[0], strict=True))
    assert row["psi"] == pytest.approx(result.psi("ey_regimen[always]"))


def test_the_confidence_label_follows_alpha_sig() -> None:
    """The header names the level that was actually computed.

    A hardcoded "95% CI" over a 90% interval is the kind of error that survives review
    precisely because the number beside it is right.
    """
    frame, _ = make_longitudinal(n=800, seed=21)
    result = LTMLE({"always": 1, "never": 0}, alpha_sig=0.10, **FAST).fit(frame, **COLUMNS)
    assert "90% CI" in result.summary()
    assert "confidence level: 90%" in result.summary()
    wide = result["ey_regimen[always]"].ci
    tight = (
        LTMLE({"always": 1, "never": 0}, alpha_sig=0.01, **FAST)
        .fit(frame, **COLUMNS)["ey_regimen[always]"]
        .ci
    )
    assert tight[0] < wide[0] and wide[1] < tight[1]


def test_the_two_alphas_cannot_be_swapped_silently() -> None:
    """``alpha`` and ``alpha_sig`` mean here what they mean on ``TMLE``.

    The pair used to be spelled the other way round, so ``LTMLE(alpha=0.9995)`` -- the
    value a reader of the ``TMLE`` docstring would type for the probability shrink --
    silently produced a 0.05 %-level interval.
    """
    with pytest.raises(ValueError, match="alpha_sig= is the significance level"):
        LTMLE({"always": 1}, alpha=0.05)
    with pytest.raises(ValueError, match="alpha= is the probability shrink"):
        LTMLE({"always": 1}, alpha_sig=0.9995)


def test_simultaneous_bands_are_wider_than_the_pointwise_intervals(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    """A fit reporting several correlated regimens gets joint bands, as a TMLE fit does."""
    result, _ = fitted
    bands = result.simultaneous
    assert bands is not None
    assert bands.critical_value > bands.pointwise_critical_value
    assert set(bands.bands) == set(result)
    for name, (low, high) in bands.bands.items():
        point_low, point_high = result[name].ci
        assert low <= point_low and point_high <= high
    assert "simultaneous" in result.summary()


def test_a_single_regimen_gets_no_bands() -> None:
    """A band over one estimand is its pointwise interval, so none is reported."""
    frame, _ = make_longitudinal(n=800, seed=22)
    result = LTMLE({"always": 1}, **FAST).fit(frame, **COLUMNS)
    assert result.simultaneous is None


def test_polars_in_polars_out() -> None:
    polars = pytest.importorskip("polars")
    frame, _ = make_longitudinal(n=800, seed=5, backend="polars")
    result = run(frame)
    assert isinstance(result.to_frame(), polars.DataFrame)
    assert isinstance(result.diagnostics(), polars.DataFrame)


def test_the_two_backends_produce_identical_numbers() -> None:
    """Not merely close: identical.

    Everything user-facing goes through narwhals and the estimator never branches on the
    backend, so the same data read from a polars frame and a pandas frame has to produce
    the same bits -- ``to_numpy`` on either gives one float64 array and there is nothing
    downstream that could see the difference.  A type check alone would pass a fit that
    silently took a different code path for polars.
    """
    pytest.importorskip("polars")
    pandas_frame, _ = make_longitudinal(n=800, seed=5, backend="pandas")
    polars_frame, _ = make_longitudinal(n=800, seed=5, backend="polars")
    left = run(pandas_frame)
    right = run(polars_frame)
    assert list(left) == list(right)
    for name in left:
        assert left.psi(name) == right.psi(name)
        assert left[name].std_error == right[name].std_error
        np.testing.assert_array_equal(left.influence_curves[name], right.influence_curves[name])
    assert left.summary().splitlines()[:5] == right.summary().splitlines()[:5]


def test_cluster_variance_is_reported_at_the_cluster() -> None:
    """Clustered rows widen the interval, since the independent unit is the cluster.

    The data has an unobserved effect shared within each cluster that moves both
    treatment decisions and the outcome, so the influence curves really are correlated
    within a cluster and ignoring ``id=`` really does understate the standard error.
    Over i.i.d. rows carrying an ``id`` column the two variances agree by construction,
    which is why counting the clusters is not a test of anything.
    """
    frame, _ = make_longitudinal(n=2000, seed=9, cluster_size=20)
    independent = run(frame)
    clustered = LTMLE({"always": 1, "never": 0}, **FAST).fit(frame, id="id", **COLUMNS)

    assert clustered.data.n_clusters == 100
    assert clustered["ey_regimen[always]"].n_clusters == 100
    assert independent["ey_regimen[always]"].n_clusters == 2000
    # A strictly wider interval for every parameter reported.  The point estimates move
    # a little too, which is not the variance leaking into the estimand: ``id=`` also
    # keeps a cluster whole inside a fold, so the nuisance fits are not the same fits.
    for name in clustered:
        assert clustered.psi(name) == pytest.approx(independent.psi(name), abs=0.02)
        assert clustered[name].std_error > independent[name].std_error
    assert "cluster-robust variance" in clustered.summary()


class TestItRefusesByName:
    """The refusals the README's table promises, which used to be bare ``TypeError``s.

    Each of these was reachable only as "unexpected keyword argument", which names no
    reason and reads as though the argument were misspelled rather than out of scope.
    """

    @pytest.mark.parametrize(
        "keyword,expected",
        [
            ("weights", "weighted efficient influence function"),
            ("intermediate", "different identification"),
            ("msm", "own weight function"),
            ("interventions", "Declare it in regimens="),
            ("shifts", "continuous dose"),
            ("incremental", "tilted mechanisms"),
            ("event", "node at every time point"),
            ("competing", "cumulative incidences"),
            ("n_bootstrap", "whole backward recursion"),
            ("cross_fit", "n_folds=1"),
        ],
    )
    def test_a_point_treatment_keyword_says_what_it_would_need(
        self, keyword: str, expected: str
    ) -> None:
        with pytest.raises(TypeError, match=expected):
            LTMLE({"always": 1}, **{keyword: 1})

    def test_a_genuine_typo_still_says_so(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument 'oucome_learner'"):
            LTMLE({"always": 1}, oucome_learner="glm")

    def test_delta_points_at_the_censoring_column(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=1)
        with pytest.raises(TypeError, match="final censoring column"):
            LTMLE({"always": 1}, **FAST).fit(frame, delta="observed", **COLUMNS)

    def test_column_names_cannot_be_combined_with_a_container(self) -> None:
        from cleverly.longitudinal import LongitudinalData

        frame, _ = make_longitudinal(n=200, seed=1)
        data = LongitudinalData.from_frame(frame, **COLUMNS)
        with pytest.raises(ValueError, match="cannot be combined with a LongitudinalData"):
            LTMLE({"always": 1}, **FAST).fit(data, outcome="Y")
        with pytest.raises(ValueError, match="'family'"):
            LTMLE({"always": 1}, **FAST).fit(data, family="gaussian")
        # A bare container is the supported call and still works.
        assert LTMLE({"always": 1}, **FAST).fit(data).converged

    def test_q_bounds_on_a_binary_outcome(self) -> None:
        """Silently ignored before: the identity scaler never looked at it."""
        frame, _ = make_longitudinal(n=200, seed=1)
        with pytest.raises(ValueError, match="q_bounds does not apply to a binary outcome"):
            LTMLE({"always": 1}, q_bounds=(0.0, 1.0), **FAST).fit(frame, **COLUMNS)

    def test_an_unknown_reference_names_the_declared_regimens(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=1)
        with pytest.raises(KeyError, match="not one of the declared regimens"):
            LTMLE({"always": 1}, reference="nope", **FAST).fit(frame, **COLUMNS)

    def test_an_unknown_parameter_names_what_is_reported(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        with pytest.raises(KeyError, match="was not estimated"):
            result["ey_regimen[nope]"]
        with pytest.raises(KeyError, match="unknown parameter"):
            result.covariance(["ey_regimen[nope]"])
        with pytest.raises(ValueError, match="no parameters selected"):
            result.covariance([])

    def test_a_dataframe_fit_needs_the_column_names(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=1)
        with pytest.raises(TypeError, match="needs outcome=, treatment= and baseline="):
            LTMLE({"always": 1}, **FAST).fit(frame)


class TestTheSuitesItCannotServe:
    """``AttributeError`` before, on three methods the README advertises unqualified."""

    def test_sensitivity_says_why_retargeting_is_not_enough(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        with pytest.raises(NotImplementedError, match="pseudo-outcome of every earlier node"):
            _ = result.sensitivity

    def test_validation_points_at_the_per_node_diagnostics(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        with pytest.raises(NotImplementedError, match=re.escape("result.diagnostics")):
            _ = result.validation

    def test_save_says_what_the_format_has_no_place_for(
        self, fitted: tuple[LongitudinalResult, dict[str, float]], tmp_path: Any
    ) -> None:
        result, _ = fitted
        with pytest.raises(NotImplementedError, match="node ordering"):
            result.save(tmp_path / "fit.json")

    def test_the_bootstrap_blames_the_missing_method_not_positivity(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        """It used to report "the fit is too unstable to bootstrap".

        Every replicate died on a missing ``subset`` inside the loop's blanket
        ``except Exception``, so a structural gap came out as a statistical diagnosis
        recommending ``res.sensitivity.positivity()`` -- itself unavailable here.
        """
        from cleverly.inference.bootstrap import run_bootstrap

        result, _ = fitted
        with pytest.raises(TypeError, match="needs a subset\\(\\) on the data container"):
            run_bootstrap(result.data, lambda data: {}, n_replicates=5)  # type: ignore[arg-type]

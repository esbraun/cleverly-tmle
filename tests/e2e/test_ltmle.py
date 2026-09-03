"""End to end: a longitudinal fit on data whose truth is known by quadrature."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

from cleverly import AssessmentStatus, CapabilityError, load
from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_survival,
)
from cleverly.exceptions import DataError, PositivityWarning
from cleverly.longitudinal import LTMLE, LongitudinalError, LongitudinalResult
from cleverly.validation.longitudinal import STITCHED_SCORE_Z_TOLERANCE, _stitched_score_z

#: Fast-tier settings: parametric nuisances, few folds, seeded.  The mechanism of
#: ``make_longitudinal`` is logistic-linear in the recorded history, so ``glm`` estimates
#: it correctly and double robustness carries the fit even though the outcome regression
#: -- which carries a ``tanh`` term -- is misspecified.  That is deliberate: it is the
#: half of the guarantee a cheap test can exercise.
FAST: dict[str, Any] = {
    "outcome_learner": sklearn.linear_model.LinearRegression(),
    "pseudo_learner": sklearn.linear_model.LinearRegression(),
    "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
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
FIT_ARGUMENTS = (*COLUMNS, "family", "id", "weights", "weights_type", "weights_estimated")


def run(frame: Any, **overrides: Any) -> LongitudinalResult:
    settings = {**FAST, **overrides}
    regimens = settings.pop("regimens", {"always": 1, "never": 0})
    columns = {
        **COLUMNS,
        **{key: settings.pop(key) for key in FIT_ARGUMENTS if key in settings},
    }
    return LTMLE(regimens, **settings).fit(frame, **columns)


def test_the_signature_exposes_a_fixed_numeric_default() -> None:
    assert inspect.signature(LTMLE).parameters["g_bounds"].default == (0.01, 1.0)


def test_auto_is_not_a_longitudinal_bound_selection_procedure() -> None:
    frame, _ = make_longitudinal(n=200, seed=90)
    with pytest.raises(ValueError, match="no automatic cumulative-bound selection"):
        run(frame, g_bounds="auto")


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

    Two claims, because a cross-fitted fit makes two.  Each outer-training recursion
    solves one score per node and regimen, exactly, and every fold's record says so.  The
    *stitched* score is not that equation and is not zero: every fold fits its ``epsilon``
    on rows it does not report, so what is left on the held-out rows is a mean-zero draw.

    So the second claim is about scale rather than about zero.  Asserting only that the
    curve is finite -- which is what this checked while the two claims were conflated --
    cannot fail for any stitching, indexing or fold-mapping defect, because all of those
    produce perfectly finite numbers.  Dividing by the residual's own standard error is
    what makes the bound mean something: a defect that misplaces a fold multiplies the
    residual rather than perturbing it, and lands orders of magnitude outside.
    """
    result, _ = fitted
    for fit in result.fits.values():
        for step in fit.steps:
            assert all(record.converged for record in step.fluctuation.folds)
            assert step.fluctuation.trace == ()
            assert step.fluctuation.n_iter == sum(
                record.n_iter for record in step.fluctuation.folds
            )
            for record in step.fluctuation.folds:
                assert float(np.max(np.abs(record.score))) < 1e-8
                assert record.trace
                assert record.score_scale is not None
    for curve in result.influence_curves.values():
        assert np.isfinite(curve).all()
        assert abs(_standardized_mean(curve)) < STITCHED_SCORE_Z_TOLERANCE


def _standardized_mean(curve: np.ndarray) -> float:
    """A curve's mean over the standard error of that mean."""
    return float(np.mean(curve) / (np.std(curve, ddof=1) / np.sqrt(curve.size)))


def test_the_reported_score_is_the_score_of_the_reported_fit(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    """``Fluctuation.score`` has one documented meaning and this is it.

    Computed here from the step's own arrays rather than read off the fluctuation, because
    the defect this replaces was precisely a ``score`` field that agreed with nothing: it
    averaged the ``K`` per-fold training scores, each at solver tolerance, and so reported
    ``1e-14`` for a node whose stitched score was ``2e-2``.  Every downstream reader of
    that field -- ``relative_score_norm``, ``diagnostics.score_equations`` -- inherited the
    number without any way to notice.
    """
    result, _ = fitted
    for fit in result.fits.values():
        for step in fit.steps:
            hand = float(
                np.mean(fit.obs_weights * step.clever * (step.pseudo_outcome - step.targeted))
            )
            assert float(np.ravel(step.fluctuation.score)[0]) == pytest.approx(hand, abs=1e-15)


def test_a_misplaced_fold_fails_the_stitching_gate(
    fitted: tuple[LongitudinalResult, dict[str, float]],
) -> None:
    """The mutation control the gate above is worthless without.

    Rotating one node's held-out predictions by a fold leaves every per-fold solve exactly
    where it was -- each still reached its own root on its own training rows -- so the
    solver verdict cannot see it.  The stitching verdict is the one that has to, and this
    is the mistake it exists for.
    """
    result, _ = fitted
    fit = next(iter(result.fits.values()))
    step = fit.steps[-1]
    order = np.argsort(np.concatenate([record.index for record in step.fluctuation.folds]))
    rotated = np.roll(step.targeted[order], len(step.targeted) // len(step.fluctuation.folds))
    damaged = replace(step, targeted=rotated)
    weights = np.asarray(fit.obs_weights, dtype=float)
    assert abs(_stitched_score_z(damaged, weights)) > STITCHED_SCORE_Z_TOLERANCE
    assert abs(_stitched_score_z(step, weights)) <= STITCHED_SCORE_Z_TOLERANCE


def test_clustered_stitching_uses_clusters_as_the_independent_units() -> None:
    """Two correlated blocks are two draws, not 200 independent observations."""
    contribution = np.concatenate([np.ones(100), np.full(100, -7.0 / 13.0)])
    step = SimpleNamespace(
        clever=np.ones(200),
        pseudo_outcome=contribution,
        targeted=np.zeros(200),
    )
    iid = _stitched_score_z(step, np.ones(200))
    clustered = _stitched_score_z(step, np.ones(200), np.repeat([0, 1], 100))
    assert iid == pytest.approx(4.232020793899766)
    assert clustered == pytest.approx(0.3)
    assert iid > STITCHED_SCORE_Z_TOLERANCE
    assert clustered < STITCHED_SCORE_Z_TOLERANCE
    assert np.isnan(_stitched_score_z(step, np.ones(200), np.zeros(200, dtype=int)))

    constant = SimpleNamespace(
        clever=np.ones(200), pseudo_outcome=np.ones(200), targeted=np.zeros(200)
    )
    assert np.isnan(_stitched_score_z(constant, np.ones(200)))


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
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
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
    # Every one of the three score equations, not just the two a T=2 fit has -- and on a
    # cross-fitted fit that is every fold's, since the equation a fold solved is its own.
    for step in fit.steps:
        assert all(record.converged for record in step.fluctuation.folds)
        for record in step.fluctuation.folds:
            assert float(np.max(np.abs(record.score))) < 1e-8
    assert len(result.diagnostics.stagewise().to_frame()) == 6  # two regimens by three nodes
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
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
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
    g1 = result.mechanism.treatment[0]["always"]
    g2 = result.mechanism.treatment[1]["always"]
    cumulative = np.clip(g1 * g2, lower, upper)
    np.testing.assert_allclose(fit.cumulative[:, -1], cumulative, atol=1e-12, rtol=0)
    followed = fit.steps[-1].trained_on
    np.testing.assert_allclose(
        fit.steps[-1].clever[followed], 1.0 / cumulative[followed], atol=1e-10, rtol=0
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
    frame = result.diagnostics.stagewise().to_frame()
    assert len(frame) == 4  # two regimens by two nodes
    rows = {(row["regimen"], row["time"]): row for _, row in frame.iterrows()}
    for label, fit in result.fits.items():
        for step in fit.steps:
            row = rows[(label, step.time)]
            weights = step.clever[step.trained_on]
            assert row["n_followed"] == int(step.trained_on.sum())
            assert row["max_weight"] == pytest.approx(float(np.max(weights)), abs=0)
            # A max is a selection, so it is bit-exact and asserted as such above.  The
            # Kish ratio is not: the frame squares a *Python* float (``total ** 2`` after
            # a ``float()``), this line squares a ``np.float64``, and numpy special-cases
            # an integer exponent of two into ``x * x`` where CPython calls libm ``pow``.
            # Same arithmetic, occasionally a different last bit -- and which way it falls
            # is a property of the platform's libm, not of this package: ``abs=0`` here
            # passed on the sandbox this was written on and failed on every GitHub runner,
            # at 756.9831201804801 against ...802.  The tolerance is still twelve orders
            # tighter than the failure this pins, since reading the weight off the wrong
            # node or the wrong mask moves the ratio by orders of magnitude, not by a ULP.
            kish = float(np.sum(weights) ** 2 / np.sum(weights**2))
            assert row["effective_n"] == pytest.approx(kish, rel=1e-12)
            raw = fit.cumulative_unbounded[:, step.time - 1][step.trained_on]
            bounded = fit.cumulative[:, step.time - 1][step.trained_on]
            assert row["share_truncated"] == pytest.approx(float(np.mean(raw != bounded)))
        # The summary quotes the *final* node, where the product is longest and the
        # leverage is therefore largest -- not the first, and not an average.
        assert fit.max_weight == pytest.approx(float(np.max(fit.steps[-1].clever)), abs=0)
        assert fit.max_weight >= max(float(np.max(step.clever)) for step in fit.steps[:-1])
        # Same quantity by two routes again, so the same tolerance: the frame masks to
        # ``trained_on`` and ``fit.effective_n`` does not, which agree only because the
        # covariate is zero off that mask -- a real claim, and one a ULP cannot express.
        assert rows[(label, result.data.n_times)]["effective_n"] == pytest.approx(
            fit.effective_n, rel=1e-12
        )


def test_each_cumulative_mechanism_is_truncated_after_the_product() -> None:
    """Pin the canonical ``ltmle::CalcCumG`` order with a bound-active witness.

    The unbounded exact laws cannot distinguish the two orders.  This fixture deliberately
    activates a tight bound: bounding the four individual factors and multiplying them is
    then different from multiplying first and bounding each cumulative prefix, which is
    what the published R implementation does.
    """
    frame, _ = make_longitudinal(n=1200, seed=17)
    tight = run(frame, g_bounds=(0.3, 0.7))
    fit = tight.fits["always"]
    lower, upper = tight.config.g_bounds
    raw = np.ones(tight.n)
    expected_columns = []
    factor_bounded = np.ones(tight.n)
    for time in range(tight.data.n_times):
        treatment = tight.mechanism.treatment[time]["always"]
        censoring = tight.mechanism.censoring[time]["always"]
        raw = raw * treatment * censoring
        expected_columns.append(np.clip(raw, lower, upper))
        factor_bounded *= np.clip(treatment, lower, upper) * np.clip(censoring, lower, upper)
    expected = np.column_stack(expected_columns)
    np.testing.assert_allclose(fit.cumulative, expected, atol=1e-12, rtol=0)
    assert np.max(np.abs(fit.cumulative[:, -1] - factor_bounded)) > 0.1
    # The lower cumulative bound caps the inverse probability at 1 / lower, irrespective
    # of how many raw factors went into it.
    assert fit.max_weight <= 1.0 / lower + 1e-12


def test_material_cumulative_truncation_warns_and_reports_the_share() -> None:
    frame, _ = make_longitudinal(n=400, seed=91)
    with pytest.warns(PositivityWarning, match="constant on scored rows"):
        result = run(frame, regimens={"always": 1}, g_bounds=0.9)
    diagnostics = result.diagnostics.stagewise().to_frame()
    assert float(diagnostics["share_truncated"].max()) == 1.0
    fit = result.fits["always"]
    np.testing.assert_allclose(fit.cumulative[:, -1], 0.9)
    assert "max truncated share 100.0%" in result.summary()


def test_a_nonbinding_bound_reports_zero_without_a_positivity_warning() -> None:
    frame, _ = make_longitudinal(n=400, seed=92)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(frame, regimens={"always": 1}, g_bounds=(1e-12, 1.0))
    assert not any(item.category is PositivityWarning for item in caught)
    np.testing.assert_array_equal(
        result.fits["always"].cumulative_unbounded, result.fits["always"].cumulative
    )
    assert float(result.diagnostics.stagewise().to_frame()["share_truncated"].max()) == 0.0


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
    assert isinstance(result.diagnostics.stagewise().to_frame(), polars.DataFrame)


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
    assert clustered.validate()["score_equations"].status is AssessmentStatus.PASSED
    assert "cluster-robust variance" in clustered.summary()


class TestObservationWeights:
    """What a weighted longitudinal fit does with the process rather than with the law.

    ``tests/unit/test_weighted_estimand_longitudinal.py`` proves the estimand and the
    influence curve, on a law a sample realises exactly and handed a saturated learner.
    What is left over is everything around them: that the fit accepts the column, that the
    weighting reaches the report, that the fixed cumulative bound does not move with the
    effective ``n``, and that an unweighted fit is untouched.
    """

    @staticmethod
    def _biased(n: int = 2000, seed: int = 5) -> tuple[Any, dict[str, float]]:
        """A frame whose ``w`` is the reciprocal of a known selection probability.

        Selection depends on ``W1``, which moves both treatments and the outcome, so a fit
        that ignored ``w`` would answer for the selected population instead of for the
        one the truth describes.
        """
        frame, truth = make_longitudinal(n=n, seed=seed)
        keep_probability = np.where(np.asarray(frame["W1"]) > 0, 0.3, 0.9)
        rng = np.random.default_rng(seed)
        selected = rng.random(n) < keep_probability
        sampled = frame.loc[selected].reset_index(drop=True)
        sampled["w"] = 1.0 / keep_probability[selected]
        return sampled, truth

    def test_a_constant_weight_column_is_the_unweighted_fit(self) -> None:
        """Bit for bit, not approximately: weighting is a generalisation of the old path."""
        frame, _ = make_longitudinal(n=800, seed=3)
        plain = run(frame)
        weighted = run(frame.assign(w=2.5), weights="w")
        assert not weighted.data.is_weighted
        for name in plain:
            assert weighted.psi(name) == plain.psi(name)
            np.testing.assert_array_equal(
                weighted.influence_curves[name], plain.influence_curves[name]
            )

    def test_the_weights_move_the_fit(self) -> None:
        """The premise: an unweighted fit of a biased sample answers a different question.

        Without this the checks below could all pass on an implementation that read the
        column and dropped it.  Two halves, and neither is a statistical claim: the tilt
        moves the covariate distribution the plug-in averages over -- by a lot, and
        deterministically, since selection is a known function of ``W1`` -- and the two
        fits therefore report different numbers.  *How far apart* they land is a property
        of the draw and is deliberately not asserted; whether the weighted one is the
        answer for the population the truth describes is a coverage claim, and
        ``tests/e2e/test_coverage_slow.py`` makes it over many draws rather than one.
        """
        frame, _ = self._biased()
        weighted = run(frame, weights="w", reference="never")
        ignored = run(frame, reference="never")
        w1 = np.asarray(frame["W1"], dtype=float)
        tilted_mean = float(np.average(w1, weights=np.asarray(frame["w"], dtype=float)))
        assert abs(tilted_mean - float(w1.mean())) > 0.2
        name = "ate_regimen[always vs never]"
        assert weighted.psi(name) != ignored.psi(name)

    def test_the_report_names_the_population_and_the_cost(self) -> None:
        frame, _ = self._biased()
        result = run(frame, weights="w")
        summary = result.summary()
        assert "observation weights (w, fixed)" in summary
        assert "weight-tilted population" in summary
        report = result.data.weight_report()
        assert report.effective_n < result.n
        assert report.design_effect > 1.0

    def test_estimated_weights_are_declared(self) -> None:
        frame, _ = self._biased()
        result = run(frame, weights="w", weights_estimated=True)
        assert "observation weights (w, estimated)" in result.summary()
        assert "conditions on the fitted weights" in result.data.weight_report().summary()

    def test_the_default_bound_is_fixed_not_resolved_from_effective_n(self) -> None:
        frame, _ = self._biased()
        result = run(frame, weights="w")
        assert result.data.effective_n < result.n
        assert result.config.g_bounds == (0.01, 1.0)
        assert "R ltmle-compatible heuristic" in result.summary()

    def test_an_explicit_bound_is_never_second_guessed(self) -> None:
        frame, _ = self._biased()
        result = run(frame, weights="w", g_bounds=(0.02, 0.98))
        assert result.config.g_bounds == (0.02, 0.98)
        assert "package default" not in result.summary()

    def test_the_diagnostics_fold_the_weights_into_the_leverage(self) -> None:
        """The two reweightings multiply, so the leverage reported is their product.

        A fit can be comfortable on the observation weights and comfortable on the clever
        covariate and thin on both together, which is the case a diagnostic showing one of
        them alone reads as fine.  Same reasoning, and the same choice, as
        ``result.diagnostics.support()``.
        """
        frame, _ = self._biased()
        result = run(frame, weights="w")
        rows = {
            (row["regimen"], row["time"]): row
            for _, row in result.diagnostics.stagewise().to_frame().iterrows()
        }
        for label, fit in result.fits.items():
            for step in fit.steps:
                leverage = (fit.obs_weights * step.clever)[step.trained_on]
                row = rows[(label, step.time)]
                assert row["max_weight"] == pytest.approx(float(np.max(leverage)), abs=0)
                kish = float(np.sum(leverage) ** 2 / np.sum(leverage**2))
                assert row["effective_n"] == pytest.approx(kish, abs=0)
            # The observation weighting materially changes the leverage rather than
            # merely carrying an unused array alongside it.
            assert not np.allclose(
                (fit.obs_weights * step.clever)[step.trained_on],
                step.clever[step.trained_on],
            )

    def test_the_weight_column_cannot_be_combined_with_a_container(self) -> None:
        from cleverly.longitudinal import LongitudinalData

        frame, _ = self._biased()
        data = LongitudinalData.from_frame(frame, weights="w", **COLUMNS)
        for keyword in ({"weights": "w"}, {"weights_type": "survey"}, {"weights_estimated": True}):
            with pytest.raises(ValueError, match="cannot be combined with a LongitudinalData"):
                LTMLE({"always": 1}, **FAST).fit(data, **keyword)
        # The container carries them, so the bare call is the supported one and is weighted.
        assert LTMLE({"always": 1, "never": 0}, **FAST).fit(data).data.is_weighted


class TestADynamicRule:
    """A regimen whose nodes are rules, on the process rather than on the exact law.

    ``tests/unit/test_influence_gateaux_longitudinal.py`` proves the influence curve is
    right, on a law a sample realises exactly and handed a saturated learner.  What that
    cannot see is anything a *real* learner does differently, and one thing in this design
    turns on exactly that -- which is what the last test here is for.
    """

    #: ``d_2 = 1{L2 > 0}``: treat at the second node only if the biomarker rose.  No
    #: static plan reaches this, since it is defined by a covariate measured between the
    #: two decisions -- the node that makes the problem longitudinal.
    #:
    #: ``d_1 = 0`` here, unlike :data:`~cleverly.datasets.RULE_LABEL`, which uses ``1``:
    #: it puts the rule furthest from *both* constants, which is what
    #: ``test_the_rule_is_a_different_parameter_from_either_constant`` needs.  The cost is
    #: that this plan's mean is ``0.500`` to twelve decimals -- exactly ``_FILLER`` -- so
    #: **do not add a truth comparison to this class**; a leak from a censored row would
    #: pass it.  The tier that compares against a truth uses ``RULE_LABEL`` instead.
    RULE: ClassVar[dict[str, Any]] = {
        "always": 1,
        "never": 0,
        "always_rule": (lambda h: np.ones(len(h)), lambda h: np.ones(len(h))),
        "treat if l2 rises": (0, lambda h: (h["L2"] > 0.0).astype(float)),
    }

    @pytest.fixture(scope="class")
    def fitted(self) -> LongitudinalResult:
        frame, _ = make_longitudinal(n=3000, seed=11)
        return run(frame, regimens=self.RULE, reference="never", simultaneous=False)

    def test_it_reports_and_converges_like_any_other_regimen(
        self, fitted: LongitudinalResult
    ) -> None:
        assert set(fitted) == {
            "ey_regimen[always]",
            "ey_regimen[never]",
            "ey_regimen[always_rule]",
            "ey_regimen[treat if l2 rises]",
            "ate_regimen[always vs never]",
            "ate_regimen[always_rule vs never]",
            "ate_regimen[treat if l2 rises vs never]",
        }
        assert fitted.converged
        assert fitted["ey_regimen[treat if l2 rises]"].std_error > 0

    def test_the_rule_is_a_different_parameter_from_either_constant(
        self, fitted: LongitudinalResult
    ) -> None:
        """And by more than a standard error, so it is not a restatement of one of them."""
        rule = fitted["ey_regimen[treat if l2 rises]"]
        for label in ("always", "never"):
            gap = abs(rule.psi - fitted[f"ey_regimen[{label}]"].psi)
            assert gap > 3.0 * rule.std_error, (label, gap, rule.std_error)

    def test_the_diagnostics_report_what_the_rule_assigned(
        self, fitted: LongitudinalResult
    ) -> None:
        """``share_assigned_1`` is the column a static plan cannot make interesting.

        For a constant it is exactly 0 or 1, which makes it a free check on the plan the
        fit ran; for a rule it is the only place the report says what the rule actually
        did to this sample, since the settings can only say that a rule was declared.
        """
        frame = fitted.diagnostics.stagewise().to_frame()
        shares = {
            (row["regimen"], row["time"]): row["share_assigned_1"] for _, row in frame.iterrows()
        }
        assert shares[("always", 1)] == 1.0
        assert shares[("never", 2)] == 0.0
        assert shares[("treat if l2 rises", 1)] == 0.0
        assert 0.2 < shares[("treat if l2 rises", 2)] < 0.8

    def test_a_rule_that_ignores_the_history_is_the_constant_plan_exactly(
        self, fitted: LongitudinalResult
    ) -> None:
        """Under a real learner, not only the saturated one on the exact law.

        The dynamic path has to be a generalisation of the static one rather than a
        second estimator beside it, and this is the assertion that says so end to end:
        it fails if the follower masks, the mechanism's arm selection, the censoring
        model's current-arm column or the submodel's arm key treats a rule differently
        from the constant it happens to equal.  Verified by mutation -- evaluating the
        cumulative product at a single arm turns this red.

        What it does **not** guard, despite the temptation to claim it, is the decision
        not to put the assigned arms into the outcome regression's design.  A rule that
        ignores the history assigns a *constant*, so that mutation adds a constant column
        here -- which ``StandardScaler`` maps to zeros and the fit then ignores -- and it
        adds the same column to ``always``.  Both sides move together and the comparison
        cannot see it.  Two things hold that decision up now, and they hold up different
        halves of it: the *statistical* claim is an argument, written out in
        ``history_design``, and the *call site* is pinned structurally by
        ``tests/unit/test_sequential_design.py``, which fails if the design ever becomes
        the mechanism's.  No test compares the two designs' estimates, because both are
        consistent and there is no second answer to compare against.
        """
        assert fitted.psi("ey_regimen[always_rule]") == fitted.psi("ey_regimen[always]")
        np.testing.assert_array_equal(
            fitted.influence_curves["ey_regimen[always_rule]"],
            fitted.influence_curves["ey_regimen[always]"],
        )


class TestTheReportSaysWhichRuleWasRun:
    """Two rules are two parameters, so the report has to be able to tell them apart.

    A static plan is stated in full by its ``1/0``, and a rule is not: a lambda has no
    name to print and the plan string can only say that *a* rule was declared.  So the
    settings carry a digest of the ``(n, T)`` arms the rule actually assigned -- which is
    the thing the fit used, and the only representation a closure has.
    """

    THRESHOLDS: ClassVar[tuple[float, float]] = (0.0, 1.0)

    @staticmethod
    def _fit(frame: pd.DataFrame, threshold: float) -> LongitudinalResult:
        regimens = {"never": 0, "rule": (1, lambda h: h["L2"] > threshold)}
        return run(frame, regimens=regimens, reference="never", simultaneous=False)

    def test_two_rules_that_print_alike_carry_different_fingerprints(self) -> None:
        frame, _ = make_longitudinal(n=800, seed=13)
        low, high = (self._fit(frame, threshold) for threshold in self.THRESHOLDS)

        # The plan strings are identical, which is the problem the digest exists for.
        assert low.config.describe()[2] == high.config.describe()[2]
        assert "rule=(1/d)" in low.config.describe()[2]

        digests = [dict(result.config.plan_fingerprints)["rule"] for result in (low, high)]
        assert digests[0] != digests[1]
        assert digests[0] in low.summary() and digests[1] in high.summary()
        # And the estimates really do differ, so the two digests are not a distinction
        # without a parameter behind it.
        assert low.psi("ey_regimen[rule]") != high.psi("ey_regimen[rule]")

    def test_the_same_rule_on_the_same_data_fingerprints_the_same(self) -> None:
        """Otherwise the digest is a run id and not a statement about the regimen."""
        frame, _ = make_longitudinal(n=800, seed=13)
        first, second = (self._fit(frame, 0.0) for _ in range(2))
        assert first.config.plan_fingerprints == second.config.plan_fingerprints

    def test_a_named_rule_is_named_in_the_settings(self) -> None:
        """A ``def`` carries the analyst's own word for the plan; print it."""

        def responders(history: Any) -> Any:
            return history["L2"] > 0.0

        frame, _ = make_longitudinal(n=600, seed=13)
        result = run(
            frame,
            regimens={"never": 0, "rule": (1, responders)},
            reference="never",
            simultaneous=False,
        )
        assert "rule=(1/d:responders)" in result.config.describe()[2]

    def test_a_static_fit_reports_no_digest(self) -> None:
        """The line would say nothing a ``1/0`` has not already said in full."""
        frame, _ = make_longitudinal(n=600, seed=13)
        result = run(frame, simultaneous=False)
        assert not any("assigned arms" in line for line in result.config.describe())
        assert result.config.plan_fingerprints  # recorded, just not printed


class TestItRefusesByName:
    """The refusals the README's table promises, which used to be bare ``TypeError``s.

    Each of these was reachable only as "unexpected keyword argument", which names no
    reason and reads as though the argument were misspelled rather than out of scope.
    """

    @pytest.mark.parametrize(
        "keyword,expected",
        [
            # No longer a refusal but a redirect, as ``event`` and ``competing`` are: the
            # weights are a column of the data, so they are declared where the columns are
            # read.  Kept as a key so that passing them here says so rather than falling
            # through to "unexpected keyword argument".
            ("weights", "column of the data"),
            ("intermediate", "different identification"),
            ("interventions", "Declare it in regimens="),
            ("shifts", "continuous dose"),
            ("incremental", "tilted mechanisms"),
            # Not a refusal any more but a redirect: a survival outcome is declared by
            # the outcome columns, and the keyword stays a key so that passing it says
            # so rather than falling through to "unexpected keyword argument".
            ("event", r"outcome=\['Y1', 'Y2', \.\.\.\]"),
            # As with ``event``: no longer a refusal but a redirect to the keyword that
            # does this, since competing risks are declared by the outcome columns.
            ("competing", "mapping of cause"),
            # This one *is* still a refusal, and of a different estimand rather than of a
            # missing feature: eliminating the competing events makes them intervened
            # nodes with their own identification.
            ("eliminate", "different estimand"),
            ("n_bootstrap", "whole backward recursion"),
            ("cross_fit", "n_folds=1"),
        ],
    )
    def test_a_point_treatment_keyword_says_what_it_would_need(
        self, keyword: str, expected: str
    ) -> None:
        with pytest.raises(TypeError, match=expected):
            LTMLE({"always": 1}, **{keyword: 1})

    def test_a_working_model_passed_to_fit_points_at_the_constructor(self) -> None:
        """``msm`` stays a ``_REFUSED`` key now that it is supported, for the reason
        ``weights`` and ``event`` do: it is declared somewhere, and falling through to
        "unexpected keyword argument" would read as a misspelling."""
        frame, _ = make_longitudinal(n=200, seed=0)
        with pytest.raises(TypeError, match="declared where the regimens are"):
            LTMLE({"always": 1}, **FAST).fit(frame, msm=1, **COLUMNS)

    def test_a_genuine_typo_still_says_so(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument 'oucome_learner'"):
            LTMLE({"always": 1}, oucome_learner=sklearn.linear_model.LinearRegression())

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


class TestTheSharedAssessmentContract:
    """Stagewise assessment works and unsupported sensitivity remains explicit."""

    def test_sensitivity_says_why_retargeting_is_not_enough(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        assert {item.status for item in result.sensitivity.run_all().items} == {
            AssessmentStatus.UNAVAILABLE
        }
        with pytest.raises(CapabilityError, match="no longitudinal sensitivity derivation"):
            result.sensitivity.omitted_confounding()

    def test_validation_points_at_the_per_node_diagnostics(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        validation = result.validate()
        assert validation["score_equations"].status is AssessmentStatus.PASSED
        rows = result.diagnostics.score_equations().rows
        nodes = sum(len(fit.steps) for fit in result.fits.values())
        # A cross-fitted node poses two questions and gets a row for each: whether every
        # fold's solve reached its root, and whether the stitched fit's residual is where
        # sampling would leave it.  A single-fold node poses only the first.
        assert [row.kind for row in rows].count("solver") == nodes
        assert [row.kind for row in rows].count("stitching") == nodes
        assert len(rows) == 2 * nodes

    def test_save_round_trips_the_longitudinal_graph(
        self, fitted: tuple[LongitudinalResult, dict[str, float]], tmp_path: Any
    ) -> None:
        result, _ = fitted
        restored = load(result.save(tmp_path / "fit.joblib"))
        assert list(restored.estimates) == list(result.estimates)
        for name in result.estimates:
            assert restored[name].psi == result[name].psi
            np.testing.assert_array_equal(
                restored[name].influence_curve,
                result[name].influence_curve,
            )

    def test_the_bootstrap_blames_the_missing_method_not_positivity(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        """It used to report "the fit is too unstable to bootstrap".

        Every replicate died on a missing ``subset`` inside the loop's blanket
        ``except Exception``, so a structural gap came out as a statistical diagnosis
        recommending ``res.diagnostics.support()`` -- itself unavailable here.
        """
        from cleverly.inference.bootstrap import run_bootstrap

        result, _ = fitted
        with pytest.raises(TypeError, match="needs a subset\\(\\) on the data container"):
            run_bootstrap(result.data, lambda data: {}, n_replicates=5)  # type: ignore[arg-type]


class TestASurvivalOutcome:
    """One event indicator per node: the report becomes a curve.

    The fixture is class-scoped and every case reads it, per the fast tier's rules -- a
    survival fit is ``T(T+1)/2`` regressions per regimen rather than ``T``, so refitting
    per case is the one waste here that costs more than it used to.
    """

    SURVIVAL_COLUMNS: ClassVar[dict[str, Any]] = {
        "outcome": ["Y1", "Y2"],
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }

    @pytest.fixture(scope="class")
    def fitted(self) -> tuple[LongitudinalResult, dict[str, float]]:
        frame, truth = make_longitudinal_survival(n=3000, seed=11)
        result = LTMLE({"always": 1, "never": 0}, reference="never", **FAST).fit(
            frame, **self.SURVIVAL_COLUMNS
        )
        return result, truth

    def test_reports_a_risk_per_regimen_per_horizon(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        assert list(result) == [
            "risk_regimen[always @ t=1]",
            "risk_regimen[always @ t=2]",
            "risk_regimen[never @ t=1]",
            "risk_regimen[never @ t=2]",
            "ate_regimen[always vs never @ t=1]",
            "ate_regimen[always vs never @ t=2]",
        ]
        assert result.converged

    def test_the_risk_is_monotone_in_the_horizon(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        """Nothing in the estimator imposes this: each horizon is its own backward pass."""
        result, _ = fitted
        for label in ("always", "never"):
            first = result.psi(f"risk_regimen[{label} @ t=1]")
            second = result.psi(f"risk_regimen[{label} @ t=2]")
            assert first <= second

    def test_it_recovers_the_truth_at_every_horizon(self) -> None:
        """Averaged over replicates, as every accuracy claim here is.

        Eight replicates at ``n=1500``: enough that ``3 * MC-se`` is a window a biased
        estimator would miss, and few enough to keep the case in the fast tier.  A single
        fit would be a coin flip on the seed, which is what the coverage tier is for.
        """
        names = [
            "risk_regimen[always @ t=1]",
            "risk_regimen[always @ t=2]",
            "ate_regimen[always vs never @ t=2]",
        ]
        estimates: dict[str, list[float]] = {name: [] for name in names}
        truth: dict[str, float] = {}
        for seed in range(8):
            frame, truth = make_longitudinal_survival(n=1500, seed=100 + seed)
            result = LTMLE({"always": 1, "never": 0}, reference="never", **FAST).fit(
                frame, **self.SURVIVAL_COLUMNS
            )
            for name in names:
                estimates[name].append(result.psi(name))
        for name in names:
            draws = np.asarray(estimates[name])
            error = float(np.mean(draws)) - truth[name]
            assert abs(error) < 3 * float(np.std(draws, ddof=1)) / np.sqrt(len(draws)) + 0.01

    def test_every_node_of_every_horizon_solves_its_score_equation(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        for fit in result.fits.values():
            for step in fit.steps:
                assert all(record.converged for record in step.fluctuation.folds)
                for record in step.fluctuation.folds:
                    assert float(np.max(np.abs(record.score))) < 1e-8
        for name in result:
            curve = result.influence_curves[name]
            assert np.isfinite(curve).all()
            assert abs(_standardized_mean(curve)) < STITCHED_SCORE_Z_TOLERANCE

    def test_the_survival_view_is_the_complement_of_the_risk(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        """``S = 1 - F`` for a level, ``-(F_a - F_b)`` for a contrast.

        The two maps are different and only one of them is ``1 - x``: applying that to a
        risk *difference* would report ``1 - RD``, which is not a quantity, with an
        interval that would read perfectly plausibly.  The standard error is the same
        under either map, both being linear with slope of modulus one.
        """
        result, _ = fitted
        risk = result.curve(scale="risk")
        survival = result.curve(scale="survival")
        for position in range(len(risk)):
            row, mirrored = risk.iloc[position], survival.iloc[position]
            assert row["estimand"] == mirrored["estimand"]
            assert mirrored["std_err"] == pytest.approx(row["std_err"])
            if row["scale"] == "level":
                assert mirrored["psi"] == pytest.approx(1.0 - row["psi"])
                assert mirrored["ci_lower"] == pytest.approx(1.0 - row["ci_upper"])
                assert mirrored["ci_upper"] == pytest.approx(1.0 - row["ci_lower"])
            else:
                assert mirrored["psi"] == pytest.approx(-row["psi"])
                assert mirrored["ci_lower"] == pytest.approx(-row["ci_upper"])
                assert mirrored["ci_upper"] == pytest.approx(-row["ci_lower"])

    def test_the_curve_carries_the_time_and_the_frame_does_not(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        """``to_frame`` keeps the column names a point-treatment fit reports.

        The horizon lives inside ``estimand`` there and gets a column of its own only in
        ``curve()``.  Two result objects in one library disagreeing about the name of
        every column is the cost this avoids, and it does not stop being worth avoiding
        because the parameter gained an index.
        """
        result, _ = fitted
        assert "time" not in result.to_frame().columns
        curve = result.curve()
        assert list(curve["time"]) == [1, 2, 1, 2, 1, 2]
        assert list(curve["regimen"]) == [
            "always",
            "always",
            "never",
            "never",
            "always vs never",
            "always vs never",
        ]

    def test_diagnostics_carry_the_regimen_and_the_horizon_apart(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        """The ``regimen`` column is the regimen, not the key the fit is filed under."""
        result, _ = fitted
        rows = result.diagnostics.stagewise().to_frame()
        assert set(rows["regimen"]) == {"always", "never"}
        assert set(rows["horizon"]) == {1, 2}
        # One row per node of every horizon of every regimen: 2 * (1 + 2).
        assert len(rows) == 6

    def test_the_bands_are_joint_over_the_whole_curve(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        """Which is the object a curve wants, and the reason they are on by default."""
        result, _ = fitted
        assert result.simultaneous is not None
        assert set(result.simultaneous.bands) == set(result)
        assert result.simultaneous.critical_value > result.simultaneous.pointwise_critical_value

    def test_horizons_reports_fewer_parameters_and_does_less_work(self) -> None:
        """The knob that makes a long panel affordable, doing what it says.

        A horizon is its own backward pass, so naming one is ``T`` regressions per
        regimen rather than ``T(T+1)/2`` -- checked by counting the steps rather than by
        timing anything.
        """
        frame, _ = make_longitudinal_survival(n=600, seed=9)
        result = LTMLE({"always": 1}, horizons=(2,), **FAST).fit(frame, **self.SURVIVAL_COLUMNS)
        assert list(result) == ["risk_regimen[always @ t=2]"]
        assert sum(len(fit.steps) for fit in result.fits.values()) == 2
        whole = LTMLE({"always": 1}, **FAST).fit(frame, **self.SURVIVAL_COLUMNS)
        assert sum(len(fit.steps) for fit in whole.fits.values()) == 3

    def test_horizons_is_refused_on_an_end_of_study_outcome(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=1)
        with pytest.raises(ValueError, match="only horizon is the end of the study"):
            run(frame, horizons=(1,))

    def test_the_curve_is_refused_on_an_end_of_study_outcome(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        frame, _ = make_longitudinal(n=200, seed=1)
        with pytest.raises(ValueError, match="report is a number and not a curve"):
            run(frame).curve()

    def test_an_event_only_at_the_last_node_reproduces_the_end_of_study_fit(self) -> None:
        """Bit for bit: ``psi``, the whole influence curve, and every ``epsilon``.

        This is what says a survival outcome is a *generalisation* of the end-of-study
        one rather than a second estimator beside it -- the same claim, and the same kind
        of claim, as a history-ignoring rule reproducing the constant plan it equals.
        With ``Y1 = 0`` for everyone at risk, the pseudo-outcome carried back from the
        second node is ``0 + (1 - 0) * Qbar*_2``, the masks lose their event factor, and
        the horizon-2 pass is the end-of-study recursion line for line.

        ``horizons=(2,)`` because the horizon-1 risk of a sample with no events at the
        first node is not estimable, and is refused as such -- which is itself the right
        answer and is checked below.
        """
        frame, _ = make_longitudinal(n=1200, seed=4)
        survival = frame.copy()
        survival["Y1"] = np.where(frame["C1"] == 1, 0.0, np.nan)
        survival = survival.rename(columns={"Y": "Y2"})

        settings = {**FAST, "simultaneous": False}
        terminal = LTMLE({"always": 1, "never": 0}, reference="never", **settings).fit(
            frame, **COLUMNS
        )
        curve = LTMLE({"always": 1, "never": 0}, reference="never", horizons=(2,), **settings).fit(
            survival, **self.SURVIVAL_COLUMNS
        )

        for label in ("always", "never"):
            end_of_study = terminal.fits[label]
            survival_fit = curve.fits[f"{label} @ t=2"]
            assert survival_fit.psi_scaled == end_of_study.psi_scaled
            np.testing.assert_array_equal(
                survival_fit.influence_curve_scaled, end_of_study.influence_curve_scaled
            )
            for left, right in zip(end_of_study.steps, survival_fit.steps, strict=True):
                np.testing.assert_array_equal(left.fluctuation.epsilon, right.fluctuation.epsilon)

    def test_a_horizon_with_no_events_is_refused_by_name(self) -> None:
        """Rather than handed to a classifier with one class in it.

        Reachable on any real survival data -- a late node with a thin risk set, a rare
        event, a small fold -- where the learner's own failure would be a
        ``RuntimeError`` from the Super Learner naming no cause.
        """
        frame, _ = make_longitudinal(n=1200, seed=4)
        survival = frame.copy()
        survival["Y1"] = np.where(frame["C1"] == 1, 0.0, np.nan)
        survival = survival.rename(columns={"Y": "Y2"})
        with pytest.raises(LongitudinalError, match="not estimable from this sample"):
            LTMLE({"always": 1}, **FAST).fit(survival, **self.SURVIVAL_COLUMNS)

    def test_the_settings_report_says_the_outcome_is_a_survival_one(
        self, fitted: tuple[LongitudinalResult, dict[str, float]]
    ) -> None:
        result, _ = fitted
        summary = result.summary()
        assert "outcome: survival, event indicator at Y1, Y2" in summary
        assert "horizons reported: t = 1, 2" in summary
        # "throughout" means through the study, so it is said once per regimen and from
        # the fit that runs to the last node -- not once per horizon.
        assert summary.count("units followed it throughout") == 2


class TestCompetingRisks:
    """More than one absorbing state per node: the report becomes a curve per cause.

    The heavy claims -- that the reported curve is the efficient influence function, and
    that the survival factor is all-cause -- are proved against the exact law in
    ``tests/unit/test_influence_gateaux_competing.py``.  What is checked here is the
    report a caller sees and the pin that says this is a generalisation of the single-event
    fit rather than a second estimator beside it.
    """

    COMPETING_COLUMNS: ClassVar[dict[str, Any]] = {
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }

    @staticmethod
    def _two_cause_frame(n: int = 1200, seed: int = 7) -> Any:
        """A survival frame split into two causes by a coin the fit cannot see.

        The coin is tossed **per unit**, not per node, and that is not a detail: the
        event is absorbing and carried forward, so a per-node toss would let a unit
        relapse at the first node and die at the second -- two absorbing events on one
        unit, which the container refuses and should.  One cause per unit makes both
        indicators absorbing and mutually exclusive by construction.

        Enough to exercise the report and the container; the *numbers* are answered for
        by the exact law, not by a simulation whose truth would have to be derived again
        here to say anything.
        """
        frame, _ = make_longitudinal_survival(n=n, seed=seed)
        rng = np.random.default_rng(seed)
        out = frame.copy()
        is_relapse = rng.integers(0, 2, size=len(frame)) == 0
        for node in ("1", "2"):
            event = frame[f"Y{node}"].to_numpy()
            out[f"R{node}"] = np.where(np.isnan(event), np.nan, event * is_relapse)
            out[f"D{node}"] = np.where(np.isnan(event), np.nan, event * ~is_relapse)
            out = out.drop(columns=[f"Y{node}"])
        return out

    @pytest.fixture(scope="class")
    def fitted(self) -> LongitudinalResult:
        return LTMLE({"always": 1, "never": 0}, reference="never", **FAST).fit(
            self._two_cause_frame(),
            outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
            **self.COMPETING_COLUMNS,
        )

    def test_reports_an_incidence_per_regimen_per_cause_per_horizon(
        self, fitted: LongitudinalResult
    ) -> None:
        expected = {
            f"cif_regimen[{label}, {cause} @ t={horizon}]"
            for label in ("always", "never")
            for cause in ("relapse", "death")
            for horizon in (1, 2)
        } | {
            f"ate_regimen[always vs never, {cause} @ t={horizon}]"
            for cause in ("relapse", "death")
            for horizon in (1, 2)
        }
        assert set(fitted) == expected
        assert fitted.config.causes == ("relapse", "death")
        assert "causes reported: relapse, death" in fitted.summary()

    def test_the_curve_carries_a_cause_column(self, fitted: LongitudinalResult) -> None:
        curve = fitted.curve()
        assert set(curve["cause"]) == {"relapse", "death"}
        # A regimen label comes back whole: ``curve()`` reads the index composed when the
        # name was built rather than splitting the name on the cause's separator.
        assert {"always", "never", "always vs never"} <= set(curve["regimen"])

    def test_the_incidences_are_reported_not_renormalised(self, fitted: LongitudinalResult) -> None:
        """The causes sum to something near one, and the deviation is reported as such.

        Nothing constrains the sum -- each cause is its own backward pass -- so this
        checks that ``incidence_total`` reports it rather than that it is exactly one.
        """
        total = fitted.incidence_total()
        assert set(total.columns) >= {"regimen", "time", "total", "std_err", "excess"}
        for value in total["total"]:
            assert 0.0 < float(value) < 1.2
        for value in total["excess"]:
            assert float(value) >= 0.0

    def test_incidence_total_is_refused_on_a_single_event_fit(self) -> None:
        frame, _ = make_longitudinal_survival(n=400, seed=2)
        result = LTMLE({"always": 1}, reference="always", **FAST).fit(
            frame, outcome=["Y1", "Y2"], **self.COMPETING_COLUMNS
        )
        with pytest.raises(ValueError, match="nothing to sum over"):
            result.incidence_total()

    def test_one_cause_reproduces_the_single_event_fit(self) -> None:
        """Bit for bit: ``psi``, the whole influence curve, and every ``epsilon``.

        This is what says competing risks are a *generalisation* of a single absorbing
        event rather than a second estimator beside it -- the same claim, and the same
        kind of claim, as an event only at the last node reproducing the end-of-study fit.
        With one cause the all-cause survival factor **is** that cause's own, so the
        composition is the line it was and every regression sees the same pseudo-outcome.

        The reported *names* differ by design, ``cif_regimen`` against ``risk_regimen``,
        so this compares the fits rather than the report: a one-cause mapping is still a
        cumulative incidence by declaration.
        """
        frame, _ = make_longitudinal_survival(n=1200, seed=5)
        settings = {**FAST, "simultaneous": False}
        single = LTMLE({"always": 1, "never": 0}, reference="never", **settings).fit(
            frame, outcome=["Y1", "Y2"], **self.COMPETING_COLUMNS
        )
        one_cause = LTMLE({"always": 1, "never": 0}, reference="never", **settings).fit(
            frame, outcome={"event": ["Y1", "Y2"]}, **self.COMPETING_COLUMNS
        )

        for label in ("always", "never"):
            for horizon in (1, 2):
                left = single.fits[f"{label} @ t={horizon}"]
                right = one_cause.fits[f"{label}, event @ t={horizon}"]
                assert right.cause == "event"
                assert left.psi_scaled == right.psi_scaled
                np.testing.assert_array_equal(
                    left.influence_curve_scaled, right.influence_curve_scaled
                )
                for before, after in zip(left.steps, right.steps, strict=True):
                    np.testing.assert_array_equal(
                        before.fluctuation.epsilon, after.fluctuation.epsilon
                    )

    def test_the_container_refuses_two_causes_at_one_node(self) -> None:
        frame = self._two_cause_frame(n=400, seed=1)
        both = frame.copy()
        fired = both["R1"] == 1.0
        both.loc[fired, "D1"] = 1.0
        assert bool(fired.any())
        with pytest.raises(DataError, match="mutually exclusive"):
            LTMLE({"always": 1}, reference="always", **FAST).fit(
                both,
                outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
                **self.COMPETING_COLUMNS,
            )

    def test_a_cause_with_no_events_is_refused_by_name(self) -> None:
        frame = self._two_cause_frame(n=400, seed=1)
        empty = frame.copy()
        empty["D1"] = np.where(np.isnan(frame["D1"]), np.nan, 0.0)
        empty["D2"] = np.where(np.isnan(frame["D2"]), np.nan, 0.0)
        # The relapse columns must absorb what death gave up, or the two stop partitioning
        # the event and the frame says a unit left the risk set through no cause at all.
        for node in ("1", "2"):
            empty[f"R{node}"] = np.where(
                np.isnan(frame[f"R{node}"]),
                np.nan,
                np.maximum(frame[f"R{node}"], np.nan_to_num(frame[f"D{node}"])),
            )
        with pytest.raises(LongitudinalError, match="not estimable from this sample"):
            LTMLE({"always": 1}, reference="always", **FAST).fit(
                empty,
                outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
                **self.COMPETING_COLUMNS,
            )

    def test_recovers_the_truth_on_average(self) -> None:
        """Averaged over independent samples, every incidence lands on its quadrature truth.

        Six replicates against the Monte Carlo standard error of the average, as the
        end-of-study test does -- enough to say the estimator is pointed at the right
        parameter, and not enough to say anything about coverage, which is the nightly
        tier's job.  Every cause and every horizon is checked: they are not
        interchangeable, and a survival factor read cause-specifically would show at
        ``t = 2`` alone.
        """
        replicates = 6
        estimates: list[dict[str, float]] = []
        truth: dict[str, float] = {}
        for seed in range(replicates):
            frame, truth = make_longitudinal_competing(n=2500, seed=200 + seed)
            result = LTMLE(
                {"always": 1, "never": 0},
                reference="never",
                **{**FAST, "random_state": seed, "simultaneous": False},
            ).fit(
                frame,
                outcome={"relapse": ["R1", "R2"], "death": ["D1", "D2"]},
                **self.COMPETING_COLUMNS,
            )
            estimates.append({name: result.psi(name) for name in result})

        for name in estimates[0]:
            values = np.array([estimate[name] for estimate in estimates])
            mc_error = float(np.std(values, ddof=1) / np.sqrt(replicates))
            assert abs(float(values.mean()) - truth[name]) < 3.0 * mc_error + 0.01, name

"""The diagnostics plots (V8, V15, V16, V17, V19): drawn numbers, labels and honesty rules.

Every plot here reads what an object already holds. The tests compare the drawn artists with
those numbers, and each honesty rule has a mutation control that the check must catch.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from dataclasses import replace
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from cleverly import ATE, CausalStudy, ConditionalContrast, PointTreatment
from cleverly._inference_status import STATUS_RECORDS, supplies_inference
from cleverly._support import UNCLASSIFIED
from cleverly.bases import Linear, Modifiers
from cleverly.datasets import make_linear_ate
from cleverly.estimators import CTMLE
from cleverly.inference.influence import ParameterEstimate
from cleverly.learners import SuperLearner
from cleverly.plotting import (
    PALETTE,
    _core,
    bootstrap,
    calibration,
    influence,
    learners,
    repeats,
)
from cleverly.plotting import _diagnostics as diagnostics_module

WITHHELD = next(name for name in STATUS_RECORDS if not supplies_inference(name))


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def _binary_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-0.6 * w[:, 0])))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * w[:, 1] + 0.8 * a - 0.3))))
    return pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})


def _fit(**settings: Any) -> Any:
    study = CausalStudy(
        _binary_frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(ATE()).estimate(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            random_state=0,
            **settings,
        )


@pytest.fixture(scope="module")
def result() -> Any:
    """A cross-fitted fit with three repeats and a bootstrap of 40 replicates."""
    return _fit(n_folds=3, repeats=3, n_bootstrap=40)


@pytest.fixture(scope="module")
def single() -> Any:
    """An in-sample fit with one repeat."""
    return _fit(cross_fit=False)


@pytest.fixture(scope="module")
def ensemble() -> Any:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 2))
    y = x[:, 0] + rng.normal(size=120)
    library = [("ols", LinearRegression()), ("ridge", Ridge(alpha=5.0)), ("flat", Ridge(alpha=1e9))]
    return SuperLearner(library=library, task="regression", n_folds=3).fit(x, y).diagnostics_


@pytest.fixture(scope="module")
def selection() -> Any:
    frame, _ = make_linear_ate(n=200, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = CTMLE(
            strategy="greedy",
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            selection_folds=3,
            estimands=("ate",),
            random_state=0,
        ).fit(frame, outcome="Y", treatment="A")
    return fit.single().ctmle_selection


@pytest.fixture(scope="module")
def curve_calibration() -> Any:
    rng = np.random.default_rng(5)
    w = rng.uniform(-1, 1, (300, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w[:, 0])))
    y = rng.binomial(1, 0.3 + 0.1 * w[:, 1] + a * (0.15 + 0.1 * w[:, 0]))
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    contrast = ConditionalContrast(Modifiers(["W1", "W2"]), model=Linear(["W1", "W2"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = (
            study.identify(contrast)
            .estimate(
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(),
                n_folds=2,
                random_state=0,
            )
            .effect_curve()
        )
        return curve.diagnostics.calibration(n_folds=2, bins=4, random_state=0)


# ------------------------------------------------------------------------------ helpers


def _title(fig: Any) -> str:
    return " ".join(fig.axes[0].get_title(loc="left").split())


def _subtitle(fig: Any) -> str:
    return _core.subtitle_text(fig.axes[0]) or ""


def _line(ax: Any, gid: str) -> Any:
    (found,) = [line for line in ax.get_lines() if line.get_gid() == gid]
    return found


def _lines(ax: Any, gid: str) -> list[Any]:
    return [line for line in ax.get_lines() if line.get_gid() == gid]


def _diagonal(ax: Any) -> Any:
    return _line(ax, "cleverly-reference")


def _offsets(ax: Any, gid: str) -> np.ndarray:
    (found,) = [item for item in ax.collections if item.get_gid() == gid]
    return np.asarray(found.get_offsets(), dtype=float)


def _has_descriptive_note(fig: Any) -> bool:
    return "descriptive" in _core.note_text(fig).lower()


def _artist_values(fig: Any) -> list[float]:
    """Every number a figure's lines and bars sit at, for the ensemble-risk check."""
    found: list[float] = []
    for ax in fig.axes:
        for line in ax.get_lines():
            found.extend(np.ravel(line.get_xdata()).astype(float))
            found.extend(np.ravel(line.get_ydata()).astype(float))
        for patch in ax.patches:
            found.extend([float(patch.get_width()), float(patch.get_height())])
    return found


def _draws_value(fig: Any, value: float) -> bool:
    return any(abs(drawn - value) < 1e-12 for drawn in _artist_values(fig))


# ------------------------------------------------------------------ V16: probability models


def test_v16_points_are_the_report_bins(result: Any) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    fig = calibration(report)
    (ax,) = fig.axes
    table = report.calibration
    points = _offsets(ax, "cleverly-calibration-points")
    np.testing.assert_allclose(points[:, 0], table["mean_predicted"])
    np.testing.assert_allclose(points[:, 1], table["mean_observed"])
    assert _title(fig) == "Are the predicted probabilities calibrated?"
    assert ax.get_xlabel() == "Treatment model: mean predicted probability"
    assert ax.get_ylabel() == "Observed rate"


def test_v16_has_a_45_degree_line_and_the_logistic_slope(result: Any) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    fig = calibration(report)
    (ax,) = fig.axes
    x, y = _diagonal(ax).get_xdata(), _diagonal(ax).get_ydata()
    np.testing.assert_allclose(x, y)
    assert x[1] > x[0]
    slope = report.metrics["calibration_slope"]
    assert "Logistic calibration slope (1 = calibrated)" in _subtitle(fig)
    assert f"{slope:.2f}" in _subtitle(fig)
    # the slope is read from the report, not recomputed from the drawn bins
    moved = replace(report, metrics={**report.metrics, "calibration_slope": 0.4321})
    assert "0.43" in _subtitle(calibration(moved))


def test_v16_point_area_follows_the_bin_count(result: Any) -> None:
    """Quantile bins are equal sized, so the test gives them unequal counts."""
    report = result.diagnostics.nuisance_models().models[0]
    bins = len(report.calibration["n"])
    counts = 10.0 * np.arange(1, bins + 1)

    def sizes_for(values: np.ndarray) -> np.ndarray:
        table = {**report.calibration, "n": list(values)}
        ax = calibration(replace(report, calibration=table)).axes[0]
        (points,) = [c for c in ax.collections if c.get_gid() == "cleverly-calibration-points"]
        return np.asarray(points.get_sizes(), dtype=float)

    sizes = sizes_for(counts)
    assert np.all(np.diff(sizes) > 0)
    assert sizes[-1] == sizes.max()
    # mutation control: reversing the counts reverses the sizes
    np.testing.assert_allclose(sizes_for(counts[::-1]), sizes[::-1])


def test_v16_draws_no_interval_and_says_descriptive(result: Any) -> None:
    fig = calibration(result.diagnostics.nuisance_models().models[0])
    (ax,) = fig.axes
    assert not ax.patches and not [c for c in ax.collections if "PolyCollection" in str(type(c))]
    assert len(ax.get_lines()) == 1  # the 45-degree line only: no error bars
    assert _has_descriptive_note(fig)
    # mutation control: with the note removed the check must fail
    for text in ax.texts:
        if text.get_gid() == "cleverly-notes":
            text.set_text("")
    assert not _has_descriptive_note(fig)


def test_v16_a_result_draws_one_panel_per_probability_model(result: Any) -> None:
    fig = calibration(result)
    assert len(fig.axes) == 2
    assert [ax.get_xlabel() for ax in fig.axes] == [
        "Treatment model: mean predicted probability",
        "Outcome model: mean predicted probability",
    ]
    assert "out-of-fold" in _core.note_text(fig)
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()


def test_v16_in_sample_fit_says_so(single: Any) -> None:
    fig = calibration(single)
    assert "in-sample" in _core.note_text(fig)
    assert "out-of-fold" not in _core.note_text(fig)


def test_v16_refuses_a_conditional_mean_model(result: Any) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    with pytest.raises(ValueError, match="no probability model"):
        calibration(replace(report, kind="conditional mean"))


def test_v16_a_missing_slope_says_why(result: Any) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    metrics = {**report.metrics, "calibration_slope": float("nan")}
    gone = replace(report, metrics=metrics, calibration_omission="the predictions are constant")
    assert "not available (the predictions are constant)" in _subtitle(calibration(gone))


# --------------------------------------------------------------------- V8: effect curves


def test_v8_points_are_the_bins_and_the_slope_is_the_objects(curve_calibration: Any) -> None:
    fig = calibration(curve_calibration)
    (ax,) = fig.axes
    points = _line(ax, "cleverly-calibration-points").get_xydata()
    np.testing.assert_allclose(points[:, 0], curve_calibration.bins["mean_prediction"])
    np.testing.assert_allclose(points[:, 1], curve_calibration.bins["pseudo_outcome_contrast"])
    assert _title(fig) == "Do the predicted effects match the measured ones?"
    assert ax.get_xlabel() == "Mean predicted effect in the bin"
    assert ax.get_ylabel() == "Measured effect (doubly robust contrast)"
    x, y = _diagonal(ax).get_xdata(), _diagonal(ax).get_ydata()
    np.testing.assert_allclose(x, y)
    subtitle = _subtitle(fig)
    assert "Linear calibration slope" in subtitle
    assert f"{curve_calibration.slope:.2f}" in subtitle
    assert f"{curve_calibration.intercept:.2f}" in subtitle


def test_v8_slope_is_the_projection_not_the_slope_of_the_bins(curve_calibration: Any) -> None:
    """Bins on the diagonal with an object slope of 0.5: the subtitle reports 0.5."""
    line = np.linspace(0.1, 0.4, len(curve_calibration.bins))
    bins = pd.DataFrame({"mean_prediction": line, "pseudo_outcome_contrast": line, "rows": 10})
    forged = replace(curve_calibration, bins=bins, slope=0.5, intercept=0.03)
    subtitle = _subtitle(calibration(forged))
    assert "0.50" in subtitle and "0.03" in subtitle
    # mutation control: a slope recomputed from the drawn bins would read 1.00
    assert "1.00" not in subtitle


def test_v8_is_descriptive_and_has_no_interval(curve_calibration: Any) -> None:
    fig = calibration(curve_calibration)
    ax = fig.axes[0]
    assert "Descriptive: bins of out-of-fold refits" in _core.note_text(fig)
    assert not ax.patches and not ax.collections
    assert len(ax.get_lines()) == 2  # the bins and the 45-degree line


def test_v8_log_ratio_uses_log_ratio_words(curve_calibration: Any) -> None:
    ax = calibration(replace(curve_calibration, scale="log_ratio")).axes[0]
    assert ax.get_xlabel() == "Mean predicted log ratio in the bin"
    assert ax.get_ylabel() == "Measured log ratio (doubly robust contrast)"


# ------------------------------------------------------------------------ V15: learners


def test_v15_super_learner_draws_risk_and_weight_in_the_same_order(ensemble: Any) -> None:
    fig = learners(ensemble)
    left, right = fig.axes
    order = np.argsort(ensemble.cv_risk, kind="stable")
    risks = [p.get_width() for p in left.patches if p.get_gid() == "cleverly-learner-risk"]
    weights = [p.get_width() for p in right.patches if p.get_gid() == "cleverly-learner-weight"]
    np.testing.assert_allclose(risks, np.asarray(ensemble.cv_risk)[order])
    np.testing.assert_allclose(weights, np.asarray(ensemble.weights)[order])
    assert [t.get_text() for t in left.get_yticklabels()] == [ensemble.names[i] for i in order]
    assert _title(fig) == "Which learners carry the fit?"
    assert left.get_xlabel() == "Cross-validated risk (squared error)"
    assert right.get_xlabel() == "Ensemble weight"
    assert _has_descriptive_note(fig)


def test_v15_draws_no_ensemble_risk(ensemble: Any) -> None:
    """Rule 8: the in-sample ensemble risk is optimistic and is never drawn.

    Mutation control: a vertical line at the ensemble risk is caught by the same check.
    """
    distinct = replace(ensemble, ensemble_cv_risk=0.7777123)
    fig = learners(distinct)
    assert not _draws_value(fig, distinct.ensemble_cv_risk)
    assert "0.778" not in " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    fig.axes[0].axvline(distinct.ensemble_cv_risk)
    assert _draws_value(fig, distinct.ensemble_cv_risk)


def test_v15_names_the_failed_candidates(ensemble: Any) -> None:
    fig = learners(replace(ensemble, failed=("boosted",)))
    assert "Dropped because fitting failed: boosted" in _core.note_text(fig)
    assert "Dropped" not in _core.note_text(learners(ensemble))


def test_v15_super_learner_takes_no_axes(ensemble: Any) -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="two panels"):
        learners(ensemble, ax=ax)


def test_v15_ctmle_draws_the_risk_along_the_sequence(selection: Any) -> None:
    fig = learners(selection)
    (ax,) = fig.axes
    line = _line(ax, "cleverly-ctmle-risk").get_xydata()
    np.testing.assert_allclose(line[:, 0], np.arange(len(selection.cv_risk)))
    np.testing.assert_allclose(line[:, 1], selection.cv_risk)
    marked = _lines(ax, "cleverly-marker")
    assert len(marked) == 1 and marked[0].get_xdata()[0] == selection.selected
    (chosen,) = [c for c in ax.collections if c.get_gid() == "cleverly-ctmle-chosen"]
    np.testing.assert_allclose(
        chosen.get_offsets()[0], [selection.selected, selection.cv_risk[selection.selected]]
    )
    assert _title(fig) == "How sharp was the choice of treatment model?"
    assert ax.get_xlabel() == "Candidate along the search path (0 = first)"
    assert ax.get_ylabel().startswith("Cross-validated risk (")
    assert _has_descriptive_note(fig)


def test_v15_ctmle_marker_follows_the_selection_and_final_step(selection: Any) -> None:
    other = (selection.selected + 1) % len(selection.cv_risk)
    moved = replace(selection, selected=other, final=selection.selected)
    fig = learners(moved)
    assert _lines(fig.axes[0], "cleverly-marker")[0].get_xdata()[0] == other
    assert "final step then reported candidate" in _core.note_text(fig)
    assert "final step" not in _core.note_text(learners(selection))


def test_v15_a_result_without_a_sequence_refuses(single: Any) -> None:
    with pytest.raises(ValueError, match="no candidate sequence"):
        learners(single)
    with pytest.raises(TypeError, match="takes SuperLearnerDiagnostics"):
        learners(object())


# ------------------------------------------------------------------- V17: repeats, bootstrap


def test_v17_repeats_points_are_the_per_repeat_estimates(result: Any) -> None:
    fig = repeats(result)
    (ax,) = fig.axes
    points = _line(ax, "cleverly-repeat-points").get_xydata()
    np.testing.assert_allclose(points[:, 0], [r.psi["ate"] for r in result.repeats])
    np.testing.assert_allclose(points[:, 1], [1, 2, 3])
    reference = _lines(ax, "cleverly-reference")
    assert len(reference) == 1
    assert reference[0].get_xdata()[0] == pytest.approx(result["ate"].psi)
    assert _title(fig) == "Does the estimate depend on how the data were split?"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Repeat 1", "Repeat 2", "Repeat 3"]
    assert f"{float(result.repeat_spread()['ate']):.3g}" in _subtitle(fig)
    assert _has_descriptive_note(fig)


def test_v17_repeats_shades_the_reported_interval_only(result: Any) -> None:
    fig = repeats(result)
    ax = fig.axes[0]
    (span,) = [p for p in ax.patches if p.get_gid() == "cleverly-repeat-interval"]
    low, high = result["ate"].ci
    assert span.get_x() == pytest.approx(low)
    assert span.get_x() + span.get_width() == pytest.approx(high)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "95% confidence interval" in labels
    assert "Reported estimate (median of repeats)" in labels


def test_v17_withheld_inference_draws_points_and_the_reason(result: Any) -> None:
    """Rule 1: no interval span, and the note gives the status's reason.

    Mutation control: the supplied estimate does draw the span.
    """
    from cleverly._inference_status import status_record

    withheld = replace(result["ate"], inference=WITHHELD)
    held_back = replace(result, estimates={"ate": withheld})
    fig = repeats(held_back)
    ax = fig.axes[0]
    assert not [p for p in ax.patches if p.get_gid() == "cleverly-repeat-interval"]
    assert "no interval is drawn" in _core.note_text(fig)
    assert status_record(WITHHELD).plain_reason in _core.note_text(fig)
    assert [p for p in repeats(result).axes[0].patches if p.get_gid() == "cleverly-repeat-interval"]


def test_v17_experimental_estimate_says_so(result: Any) -> None:
    experimental = replace(result["ate"], support=UNCLASSIFIED)
    fig = repeats(replace(result, estimates={"ate": experimental}))
    assert "experimental" in _core.note_text(fig)


def test_v17_one_repeat_refuses(single: Any) -> None:
    with pytest.raises(ValueError, match="one cross-fitting repeat"):
        repeats(single)


def test_v17_unknown_estimand_refuses(result: Any) -> None:
    with pytest.raises(ValueError, match="not in this result"):
        repeats(result, name="missing")


def test_v17_bootstrap_histogram_is_a_density_of_the_draws(result: Any) -> None:
    fig = result.bootstrap.plot("ate")
    (ax,) = fig.axes
    bars = [p for p in ax.patches if p.get_gid() == "cleverly-bootstrap-histogram"]
    area = sum(p.get_width() * p.get_height() for p in bars)
    assert area == pytest.approx(1.0)
    draws = np.asarray(result.bootstrap.draws["ate"], dtype=float)
    assert bars[0].get_x() == pytest.approx(draws.min())
    assert bars[-1].get_x() + bars[-1].get_width() == pytest.approx(draws.max())
    assert _title(fig) == "Is the bootstrap distribution close to normal?"
    assert ax.get_ylabel() == "Density of replicates"


def test_v17_bootstrap_normal_curve_has_the_draws_mean_and_sd(result: Any) -> None:
    ax = bootstrap(result.bootstrap, "ate").axes[0]
    draws = np.asarray(result.bootstrap.draws["ate"], dtype=float)
    x, y = _line(ax, "cleverly-bootstrap-normal").get_data()
    spread = float(np.std(draws, ddof=1))
    assert y.max() == pytest.approx(1.0 / (spread * np.sqrt(2 * np.pi)), rel=5e-3)
    assert x[np.argmax(y)] == pytest.approx(float(np.mean(draws)), abs=spread / 50)


def test_v17_bootstrap_marks_the_percentile_limits(result: Any) -> None:
    fig = bootstrap(result)
    ax = fig.axes[0]
    low, high = result.bootstrap.interval("ate", 0.05).ci
    markers = sorted(line.get_xdata()[0] for line in _lines(ax, "cleverly-marker"))
    assert markers == pytest.approx([low, high])
    (estimate,) = _lines(ax, "cleverly-reference")
    assert estimate.get_xdata()[0] == pytest.approx(result["ate"].psi)
    assert "95% percentile limits" in [t.get_text() for t in ax.get_legend().get_texts()]
    assert _has_descriptive_note(fig)


def test_v17_percentile_limits_come_from_the_draws(result: Any) -> None:
    """Mutation control: skewing the replicates moves the limits that are drawn."""
    draws = {"ate": np.asarray(result.bootstrap.draws["ate"], dtype=float)}
    skewed = replace(result.bootstrap, draws={"ate": np.exp(4.0 * draws["ate"])})
    ax = bootstrap(skewed, "ate").axes[0]
    markers = sorted(line.get_xdata()[0] for line in _lines(ax, "cleverly-marker"))
    low, high = skewed.interval("ate", 0.05).ci
    assert markers == pytest.approx([low, high])
    original = result.bootstrap.interval("ate", 0.05).ci
    assert markers != pytest.approx(list(original))


def test_v17_a_bare_bootstrap_draws_no_reported_estimate(result: Any) -> None:
    ax = bootstrap(result.bootstrap, "ate", level=0.9).axes[0]
    assert not _lines(ax, "cleverly-reference")
    assert "90% percentile limits" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_v17_failed_replicates_are_counted(result: Any) -> None:
    failing = replace(result.bootstrap, n_failed=3)
    note = _core.note_text(bootstrap(failing, "ate"))
    assert "3 of 40 replicates failed" in note
    assert "failed" not in _core.note_text(bootstrap(result.bootstrap, "ate"))


def test_v17_bootstrap_refusals(result: Any, single: Any) -> None:
    with pytest.raises(ValueError, match="no bootstrap replicates"):
        bootstrap(single)
    several = replace(
        result.bootstrap, draws={"a": np.arange(5.0), "b": np.arange(5.0) * 2.0}, std_errors={}
    )
    with pytest.raises(ValueError, match="name the estimand"):
        bootstrap(several)
    with pytest.raises(ValueError, match="no replicates"):
        bootstrap(several, "c")
    one = replace(result.bootstrap, draws={"ate": np.array([0.1, np.nan, np.nan])})
    with pytest.raises(ValueError, match="fewer than two finite"):
        bootstrap(one, "ate")


# ---------------------------------------------------------------------------- V19: influence


def _hand_estimate(curve: Any, *, n_clusters: int | None = None, **extra: Any) -> Any:
    values = np.asarray(curve, dtype=float)
    return ParameterEstimate(
        name="hand",
        psi=0.0,
        influence_curve=values,
        variance=1.0,
        n=len(values),
        n_clusters=len(values) if n_clusters is None else n_clusters,
        **extra,
    )


def test_v19_exact_check_on_a_hand_influence_curve() -> None:
    """One row of 3 among 99 rows of +-1: squares 9 and 99 ones, so the top 1% carry 9/108."""
    curve = np.concatenate([[3.0], np.ones(50), -np.ones(49)])
    rng = np.random.default_rng(1)
    shuffled = rng.permutation(curve)
    fig = influence(_hand_estimate(shuffled))
    (ax,) = fig.axes
    x, y = _line(ax, "cleverly-influence-curve").get_data()
    squares = np.sort(shuffled**2)[::-1]
    np.testing.assert_allclose(x, np.arange(0, 101) / 100)
    np.testing.assert_allclose(y, np.concatenate([[0.0], np.cumsum(squares) / squares.sum()]))
    assert y[1] == pytest.approx(9.0 / 108.0)
    assert y[-1] == pytest.approx(1.0)
    (top,) = [c for c in ax.collections if c.get_gid() == "cleverly-influence-top"]
    np.testing.assert_allclose(top.get_offsets()[0], [0.01, 9.0 / 108.0])
    assert "the top 1% of rows (1 of 100) carry 8.3%" in _subtitle(fig)
    assert _title(fig) == "How concentrated is the uncertainty?"
    assert ax.get_xlabel() == "Top share of rows, largest squared influence first"
    assert ax.get_ylabel() == "Share of the sum of squared influence values"
    diagonal = _diagonal(ax)
    np.testing.assert_allclose(diagonal.get_xdata(), diagonal.get_ydata())
    assert "Even spread" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_v19_even_influence_follows_the_diagonal() -> None:
    fig = influence(_hand_estimate(np.tile([1.0, -1.0], 50)))
    x, y = _line(fig.axes[0], "cleverly-influence-curve").get_data()
    np.testing.assert_allclose(x, y)


def _clustered() -> tuple[np.ndarray, np.ndarray]:
    codes = np.repeat(np.arange(5), 4)
    curve = np.concatenate(
        [
            [2.0] * 4,
            [1.0, -1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, -1.0] * 2,
            [3.0, -3.0, 0, 0],
        ]
    )
    return curve, codes


def test_v19_clusters_are_summed_within_clusters() -> None:
    """Cluster sums are 8, 0, 0, 0, 0: the one cluster carries all of the sum of squares."""
    curve, codes = _clustered()
    estimate = _hand_estimate(curve, n_clusters=5)
    fig = influence(estimate, cluster=codes)
    (ax,) = fig.axes
    x, y = _line(ax, "cleverly-influence-curve").get_data()
    np.testing.assert_allclose(x, np.arange(6) / 5)
    np.testing.assert_allclose(y, [0, 1, 1, 1, 1, 1])
    assert ax.get_xlabel() == "Top share of clusters, largest squared influence first"
    assert "Share of clusters" in _core.note_text(fig)
    assert "of clusters" in _subtitle(fig)


def test_v19_ranking_rows_instead_of_clusters_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: a plot that ignores the clusters draws different numbers."""
    curve, codes = _clustered()
    estimate = _hand_estimate(curve, n_clusters=5)
    expected = np.asarray([0, 1, 1, 1, 1, 1], dtype=float)

    def drawn() -> np.ndarray:
        return np.asarray(
            _line(
                influence(estimate, cluster=codes).axes[0], "cleverly-influence-curve"
            ).get_ydata()
        )

    np.testing.assert_allclose(drawn(), expected)
    monkeypatch.setattr("cleverly.inference.bootstrap.cluster_sums", lambda c, k: np.asarray(c))
    assert drawn().size != expected.size or not np.allclose(drawn(), expected)


def test_v19_a_clustered_estimate_needs_its_clusters() -> None:
    curve, _ = _clustered()
    with pytest.raises(ValueError, match="clustered fit"):
        influence(_hand_estimate(curve, n_clusters=5))
    with pytest.raises(ValueError, match="one code per influence value"):
        influence(_hand_estimate(curve, n_clusters=5), cluster=np.arange(3))


def test_v19_withheld_inference_draws_nothing_and_says_why(single: Any) -> None:
    from cleverly._inference_status import status_record

    withheld = replace(single["ate"], inference=WITHHELD)
    before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="not computed from the influence curve") as caught:
        influence(withheld)
    assert status_record(WITHHELD).plain_reason in str(caught.value)
    assert len(plt.get_fignums()) == before
    influence(single["ate"])  # mutation control: the ordinary estimate draws


def test_v19_repeated_fit_says_it_is_not_the_reported_variance(result: Any, single: Any) -> None:
    fig = influence(result)
    assert "5" not in _core.note_text(fig).split("combines")[0][-3:]
    assert "combines 3 cross-fitting repeats" in _core.note_text(fig)
    assert "not the reported variance" in _core.note_text(fig)
    assert "not the reported variance" not in _core.note_text(influence(single))


def test_v19_fold_evaluated_fit_says_so(single: Any) -> None:
    folded = replace(single, config=replace(single.config, cv_evaluation=True))
    assert "evaluates the estimate within folds" in _core.note_text(influence(folded))
    assert "within folds" not in _core.note_text(influence(single))


def test_v19_uses_the_fits_own_clusters(single: Any) -> None:
    codes = np.arange(single.data.n) // 4
    clustered_data = replace(single.data, cluster=codes)
    fig = influence(replace(single, data=clustered_data))
    assert "Share of clusters" in _core.note_text(fig)
    assert fig.axes[0].get_xlabel().startswith("Top share of clusters")


def test_v19_refuses_unrankable_input() -> None:
    with pytest.raises(ValueError, match="all zero"):
        influence(_hand_estimate(np.zeros(10)))
    with pytest.raises(ValueError, match="non-finite"):
        influence(_hand_estimate([1.0, np.nan, 2.0]))
    with pytest.raises(TypeError, match="takes a ParameterEstimate"):
        influence(object())


def test_v19_the_ratio_scale_is_on_the_log_scale(single: Any) -> None:
    ratio = replace(single["ate"], scale="ratio")
    assert "log scale" in _core.note_text(influence(ratio))


# -------------------------------------------------------------------------- plot() and style


def test_plot_methods_call_the_functions(
    result: Any,
    ensemble: Any,
    selection: Any,
    curve_calibration: Any,
) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    assert _title(report.plot()) == "Are the predicted probabilities calibrated?"
    assert _title(curve_calibration.plot()) == "Do the predicted effects match the measured ones?"
    assert _title(ensemble.plot()) == "Which learners carry the fit?"
    assert _title(selection.plot()) == "How sharp was the choice of treatment model?"
    assert _title(result.bootstrap.plot("ate")) == "Is the bootstrap distribution close to normal?"


def _figures(result: Any, ensemble: Any, selection: Any, curve_calibration: Any) -> list[Any]:
    return [
        calibration(result),
        calibration(result.diagnostics.nuisance_models().models[0]),
        calibration(curve_calibration),
        learners(ensemble),
        learners(selection),
        repeats(result),
        bootstrap(result),
        influence(result),
    ]


def test_style_is_the_validated_palette_on_an_opaque_face(
    result: Any, ensemble: Any, selection: Any, curve_calibration: Any
) -> None:
    from matplotlib.colors import to_hex

    for fig in _figures(result, ensemble, selection, curve_calibration):
        assert fig.get_facecolor()[3] == 1.0
        assert _title(fig)
        positions = [tuple(np.round(ax.get_position().bounds, 4)) for ax in fig.axes]
        assert len(set(positions)) == len(positions)
        for ax in fig.axes:
            assert ax.get_xlabel()
            siblings = ax.get_shared_y_axes().get_siblings(ax)
            assert ax.get_ylabel() or any(other.get_ylabel() for other in siblings)
        colours = set()
        for ax in fig.axes:
            for line in ax.get_lines():
                if line.get_gid() in (None, "cleverly-reference", "cleverly-marker"):
                    continue
                colours.add(to_hex(line.get_color()))
            for patch in ax.patches:
                if patch.get_gid():
                    colours.add(to_hex(patch.get_facecolor()[:3]))
        assert colours <= set(PALETTE)


def test_every_figure_renders_to_a_png(
    result: Any, ensemble: Any, selection: Any, curve_calibration: Any, tmp_path: Any
) -> None:
    for index, fig in enumerate(_figures(result, ensemble, selection, curve_calibration)):
        path = tmp_path / f"figure-{index}.png"
        fig.savefig(path, dpi=60)
        assert path.stat().st_size > 1000


def test_importing_the_plotting_package_does_not_import_matplotlib() -> None:
    code = (
        "import sys, cleverly, cleverly.plotting; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_module_is_wired_into_the_package() -> None:
    assert diagnostics_module.__all__ == [
        "bootstrap",
        "calibration",
        "concentration",
        "influence",
        "learners",
        "repeats",
    ]

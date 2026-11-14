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
from matplotlib.colors import to_hex
from matplotlib.text import Text
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from cleverly import ATE, CausalStudy, ConditionalContrast, PointTreatment
from cleverly._inference_status import STATUS_RECORDS, status_record, supplies_inference
from cleverly._summary.vocabulary import BOOTSTRAP_DIAGNOSTIC
from cleverly._support import UNCLASSIFIED, Support
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
from tests.unit.test_plotting_core import assert_no_note_text

pytestmark = pytest.mark.xdist_group("plotting_diagnostics")

WITHHELD = next(name for name in STATUS_RECORDS if not supplies_inference(name))
SUPPORTED = Support("supported", "A study covers it.", ("study-a",), row="R1")


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


def _fit(frame: pd.DataFrame | None = None, **settings: Any) -> Any:
    study = CausalStudy(
        _binary_frame() if frame is None else frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
    )
    settings.setdefault("outcome_learner", LogisticRegression(max_iter=1000))
    settings.setdefault("treatment_learner", LogisticRegression(max_iter=1000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(ATE()).estimate(random_state=0, **settings)


@pytest.fixture(scope="module")
def result() -> Any:
    """A cross-fitted fit with three repeats and a bootstrap of 40 replicates."""
    return _fit(n_folds=3, repeats=3, n_bootstrap=40)


@pytest.fixture(scope="module")
def single() -> Any:
    """An in-sample fit with one repeat."""
    return _fit(cross_fit=False)


@pytest.fixture(scope="module")
def library_fit() -> Any:
    """An in-sample fit whose treatment model is a two-learner Super Learner."""
    ensemble = SuperLearner(
        library=[
            ("logistic", LogisticRegression(max_iter=1000)),
            ("flat", LogisticRegression(C=1e-4, max_iter=1000)),
        ],
        task="classification",
        n_folds=3,
    )
    return _fit(cross_fit=False, treatment_learner=ensemble)


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
    return _core.title_text(fig)


def _subtitle(fig: Any) -> str:
    return _core.subtitle_text(fig.axes[0]) or ""


def _line(ax: Any, gid: str) -> Any:
    (found,) = [line for line in ax.get_lines() if line.get_gid() == gid]
    return found


def _lines(ax: Any, gid: str) -> list[Any]:
    return [line for line in ax.get_lines() if line.get_gid() == gid]


def _diagonal(ax: Any) -> Any:
    return _line(ax, _core.GID_REFERENCE)


def _offsets(ax: Any, gid: str) -> np.ndarray:
    (found,) = [item for item in ax.collections if item.get_gid() == gid]
    return np.asarray(found.get_offsets(), dtype=float)


def _has_descriptive_note(fig: Any) -> bool:
    return "descriptive" in _core.note_text(fig).lower()


def _markers(ax: Any) -> list[float]:
    return sorted(float(line.get_xdata()[0]) for line in _lines(ax, _core.GID_MARKER))


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


def _with_bootstrap(fit: Any, **changes: Any) -> Any:
    """The fit with its estimate's bootstrap summary changed."""
    estimate = fit["ate"]
    moved = replace(estimate, bootstrap=replace(estimate.bootstrap, **changes))
    return replace(fit, estimates={"ate": moved})


# ------------------------------------------------------------------ V16: probability models


def test_v16_points_are_the_report_bins(result: Any) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    fig = calibration(report)
    (ax,) = fig.axes
    table = report.calibration
    points = _offsets(ax, "cleverly-calibration-points")
    np.testing.assert_allclose(points[:, 0], table["mean_predicted"])
    np.testing.assert_allclose(points[:, 1], table["mean_observed"])
    assert _title(fig) == "Calibration of predicted probabilities"
    assert ax.get_title(loc="left") == "Treatment model"
    assert ax.get_xlabel() == "Mean predicted probability"
    assert ax.get_ylabel() == "Observed rate"


def test_v16_has_a_45_degree_line_and_the_logistic_slope(result: Any) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    assert report.evaluation == "out_of_fold"
    fig = calibration(report)
    (ax,) = fig.axes
    x, y = _diagonal(ax).get_xdata(), _diagonal(ax).get_ydata()
    np.testing.assert_allclose(x, y)
    assert x[1] > x[0]
    slope = report.metrics["calibration_slope"]
    assert "Out-of-fold calibration slope" in _subtitle(fig)
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
    # mutation control: with the notes removed the check must fail
    fig._cleverly_notes = []  # type: ignore[attr-defined]
    assert not _has_descriptive_note(fig)


def test_v16_a_result_draws_one_panel_per_probability_model(result: Any) -> None:
    fig = calibration(result)
    assert len(fig.axes) == 2
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["Treatment model", "Outcome model"]
    assert "out-of-fold" in _core.note_text(fig)
    assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
    # the fit combined three draws, and the reports describe the first
    assert "combined 3 cross-fitting draws; the bins describe draw 1" in _core.note_text(fig)


def test_v16_in_sample_fit_shows_no_slope(single: Any) -> None:
    """An in-sample slope is 1 by construction: the subtitle shows none, the note says why.

    Mutation control: the same report marked out of fold shows its slope.
    """
    fig = calibration(single)
    report = single.diagnostics.nuisance_models().models[0]
    slope = f"{report.metrics['calibration_slope']:.2f}"
    assert "in-sample" in _core.note_text(fig).lower()
    assert "out-of-fold" not in _core.note_text(fig)
    assert "1 by construction" in _core.note_text(fig)
    assert slope not in _subtitle(fig) and "no calibration slope is shown" in _subtitle(fig)
    assert report.evaluation == "in_sample"
    bare = calibration(report)
    assert slope not in _subtitle(bare) and "1 by construction" in _core.note_text(bare)
    moved = calibration(replace(report, evaluation="out_of_fold"))
    assert slope in _subtitle(moved)
    assert "out-of-fold" in _core.note_text(moved)


def test_v16_a_report_without_its_basis_says_so(result: Any) -> None:
    """Rule: a bare report whose basis is unknown never claims out-of-fold predictions."""
    report = replace(result.diagnostics.nuisance_models().models[0], evaluation=None)
    fig = calibration(report)
    assert "out-of-fold" not in _core.note_text(fig)
    assert "does not record" in _core.note_text(fig)
    assert "not recorded" in _core.note_text(fig)


def test_v16_a_working_treatment_model_shows_no_slope(result: Any) -> None:
    """A collaborative working mechanism is not the treatment law: no slope, and a note.

    Mutation control: the estimated treatment law shows its slope.
    """
    found = result.diagnostics.nuisance_models()
    slope = f"{found.models[0].metrics['calibration_slope']:.2f}"
    working = calibration(replace(found, treatment_role="collaborative_working_model"))
    assert working.axes[0].get_title(loc="left") == "Collaborative working treatment model"
    assert "Treatment model" not in _subtitle(working).split(":", 1)[1]
    assert slope not in _subtitle(working).split("Outcome model")[0]
    assert "not an estimate of the treatment law" in _core.note_text(working)
    declared = calibration(replace(found, treatment_role="declared_treatment_law"))
    assert declared.axes[0].get_title(loc="left") == "Declared treatment probabilities"
    assert "the design declares" in _core.note_text(declared)
    estimated = calibration(replace(found, treatment_role="estimated_treatment_law"))
    assert f"Treatment model {slope}" in _subtitle(estimated)
    assert "treatment law" not in _core.note_text(estimated)


def test_v16_names_every_kind_of_probability_model() -> None:
    words = diagnostics_module._model_words
    assert words("propensity[high]") == "Treatment model, arm high (one vs rest)"
    assert words("missingness") == "Missingness model (outcome observed)"
    assert words("intermediate") == "Intermediate model"
    assert words("outcome") == "Outcome model"
    assert words("propensity[high]", "collaborative_working_model").startswith("Collaborative")


def test_v16_says_weighted_only_when_the_data_are_weighted(result: Any) -> None:
    assert "weighted" not in _core.note_text(calibration(result))
    report = result.diagnostics.nuisance_models().models[0]
    assert "weighted when the data carry weights" in _core.note_text(calibration(report))
    weights = np.linspace(0.5, 1.5, result.data.n)
    weighted = replace(result, data=replace(result.data, weights=weights))
    assert "weighted by the data's weights" in _core.note_text(calibration(weighted))


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
    points = _offsets(ax, "cleverly-calibration-points")
    np.testing.assert_allclose(points[:, 0], curve_calibration.bins["mean_prediction"])
    np.testing.assert_allclose(points[:, 1], curve_calibration.bins["pseudo_outcome_contrast"])
    assert _title(fig) == "Predicted and measured effects"
    assert ax.get_xlabel() == "Mean predicted effect in the bin"
    assert ax.get_ylabel() == "Measured effect (doubly robust contrast)"
    x, y = _diagonal(ax).get_xdata(), _diagonal(ax).get_ydata()
    np.testing.assert_allclose(x, y)
    subtitle = _subtitle(fig)
    assert "Linear calibration slope" in subtitle
    assert f"{curve_calibration.slope:.2f}" in subtitle
    assert f"{curve_calibration.intercept:.2f}" in subtitle


def test_v8_bins_are_unjoined_points_sized_by_their_rows(curve_calibration: Any) -> None:
    """A line through the bins would read as a fitted curve; the bins are points."""
    rows = np.arange(1, len(curve_calibration.bins) + 1) * 10
    bins = curve_calibration.bins.assign(rows=rows)
    ax = calibration(replace(curve_calibration, bins=bins)).axes[0]
    (points,) = [c for c in ax.collections if c.get_gid() == "cleverly-calibration-points"]
    assert np.all(np.diff(points.get_sizes()) > 0)
    assert len(ax.get_lines()) == 1  # the 45-degree line only


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
    assert not ax.patches
    assert [c.get_gid() for c in ax.collections] == ["cleverly-calibration-points"]


def test_v8_log_ratio_names_the_one_step_loss(curve_calibration: Any) -> None:
    """On the log-ratio scale the slope minimises the one-step loss, not a projection.

    Mutation control: the difference scale keeps the projection words and not the loss.
    """
    fig = calibration(replace(curve_calibration, scale="log_ratio"))
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Mean predicted log ratio in the bin"
    assert ax.get_ylabel() == "Measured log ratio (doubly robust contrast)"
    assert "one-step log-ratio loss" in _subtitle(fig)
    assert "Linear" not in _subtitle(fig)
    assert "one-step log-ratio loss" in _core.note_text(fig)
    assert "projection" not in _core.note_text(fig)
    difference = calibration(curve_calibration)
    assert "least-squares projection" in _core.note_text(difference)
    assert "log-ratio" not in _core.note_text(difference) + _subtitle(difference)


def test_v8_drops_a_bin_without_a_log_ratio(curve_calibration: Any) -> None:
    bins = curve_calibration.bins.copy()
    bins.loc[1, "pseudo_outcome_contrast"] = np.nan
    fig = calibration(replace(curve_calibration, scale="log_ratio", bins=bins))
    points = _offsets(fig.axes[0], "cleverly-calibration-points")
    assert len(points) == len(bins) - 1
    assert "1 bin has no log ratio" in _core.note_text(fig)
    assert "no log ratio" not in _core.note_text(calibration(curve_calibration))
    bins["pseudo_outcome_contrast"] = np.nan
    with pytest.raises(ValueError, match="no bin of this calibration can be drawn"):
        calibration(replace(curve_calibration, scale="log_ratio", bins=bins))


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
    assert _title(fig) == "Learner risks and weights"
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
    texts = [t.get_text() for t in fig.texts] + [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "0.778" not in " ".join(texts)
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


def test_v15_draws_the_learners_of_an_ordinary_fit(library_fit: Any) -> None:
    """The fit's own reports carry the risks and weights; the plot draws exactly those.

    Mutation control: a fit without a Super Learner refuses rather than drawing nothing.
    """
    report = library_fit.diagnostics.nuisance_models()["propensity"]
    assert report.learner_risks and report.learner_loss == "neg_log_likelihood"
    for source in (library_fit, library_fit.diagnostics.nuisance_models(), report):
        fig = learners(source)
        left, right = fig.axes
        names = list(report.learner_risks)
        order = np.argsort([report.learner_risks[name] for name in names], kind="stable")
        risks = [p.get_width() for p in left.patches if p.get_gid() == "cleverly-learner-risk"]
        weights = [p.get_width() for p in right.patches if p.get_gid() == "cleverly-learner-weight"]
        np.testing.assert_allclose(risks, [report.learner_risks[names[i]] for i in order])
        np.testing.assert_allclose(weights, [report.learner_weights[names[i]] for i in order])
        assert left.get_ylabel() == "Treatment model"
        assert left.get_xlabel() == "Cross-validated risk (negative log-likelihood)"
        assert "mean over the model's Super Learner fits" in _core.note_text(fig)
    with pytest.raises(ValueError, match="fitted by a Super Learner"):
        learners(library_fit.diagnostics.nuisance_models()["outcome"])


def test_v15_ctmle_draws_the_risk_along_the_sequence(selection: Any) -> None:
    fig = learners(selection)
    (ax,) = fig.axes
    line = _line(ax, "cleverly-ctmle-risk").get_xydata()
    np.testing.assert_allclose(line[:, 0], np.arange(len(selection.cv_risk)))
    np.testing.assert_allclose(line[:, 1], selection.cv_risk)
    assert _markers(ax) == [selection.selected]
    (chosen,) = [c for c in ax.collections if c.get_gid() == "cleverly-ctmle-chosen"]
    np.testing.assert_allclose(
        chosen.get_offsets()[0], [selection.selected, selection.cv_risk[selection.selected]]
    )
    assert to_hex(chosen.get_facecolor()[0]) == PALETTE[0]  # filled, not the hollow no-interval
    assert _title(fig) == "Treatment model selection"
    assert ax.get_xlabel() == "Candidate along the search path (0 = first)"
    assert _has_descriptive_note(fig)


def test_v15_ctmle_names_a_penalized_criterion(selection: Any) -> None:
    assert selection.penalized
    penalized = learners(selection).axes[0].get_ylabel()
    assert " ".join(penalized.split()) == "Cross-validated criterion (squared error + penalty)"
    plain = learners(replace(selection, penalized=False)).axes[0].get_ylabel()
    assert plain == "Cross-validated risk (squared error)"


def test_v15_ctmle_says_when_the_search_stopped_early(selection: Any) -> None:
    last = len(selection.cv_risk) - 1
    stopped = learners(replace(selection, patience=2, stopped_at=last))
    assert f"stopped early at candidate {last} (patience=2)" in _core.note_text(stopped)
    assert "stopped early" not in _core.note_text(learners(selection))


def test_v15_ctmle_learner_strategy_names_the_chosen_learner(selection: Any) -> None:
    labels = tuple(f"learner {index}" for index in range(len(selection.cv_risk)))
    fig = learners(replace(selection, strategy="learners", labels=labels))
    assert f"({labels[selection.selected]})" in _subtitle(fig)
    assert "covariate" not in _subtitle(fig)
    assert "covariate" in _subtitle(learners(selection))


def test_v15_ctmle_marker_follows_the_selection_and_final_step(selection: Any) -> None:
    other = (selection.selected + 1) % len(selection.cv_risk)
    moved = replace(selection, selected=other, final=selection.selected)
    fig = learners(moved)
    assert _markers(fig.axes[0]) == [other]
    assert "final step then reported candidate" in _core.note_text(fig)
    assert "final step" not in _core.note_text(learners(selection))


def test_v15_refuses_what_it_cannot_draw(single: Any) -> None:
    with pytest.raises(ValueError, match="fitted by a Super Learner"):
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
    reference = _lines(ax, _core.GID_REFERENCE)
    assert len(reference) == 1
    assert reference[0].get_xdata()[0] == pytest.approx(result["ate"].psi)
    assert _title(fig) == "Estimate across repeated splits"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Repeat 1", "Repeat 2", "Repeat 3"]
    assert f"{float(result.repeat_spread()['ate']):.3g}" in _subtitle(fig)
    assert _subtitle(fig).startswith("Average effect:")
    assert _has_descriptive_note(fig)


def test_v17_repeats_shades_the_reported_interval_only(result: Any) -> None:
    fig = repeats(result)
    ax = fig.axes[0]
    (span,) = [p for p in ax.patches if p.get_gid() == "cleverly-repeat-interval"]
    low, high = result["ate"].ci
    assert span.get_x() == pytest.approx(low)
    assert span.get_x() + span.get_width() == pytest.approx(high)
    labels = _core.key_labels(fig)
    assert "95% confidence interval" in labels
    assert "Reported estimate (median of repeats)" in labels


def test_v17_withheld_inference_draws_points_and_the_reason(result: Any) -> None:
    """Rule 1: no interval span, the note gives the status's reason, and the withheld
    plug-in spread is not called a standard error.

    Mutation control: the supplied estimate draws the span and says standard error.
    """
    withheld = replace(result["ate"], inference=WITHHELD)
    held_back = replace(result, estimates={"ate": withheld})
    fig = repeats(held_back)
    ax = fig.axes[0]
    assert not [p for p in ax.patches if p.get_gid() == "cleverly-repeat-interval"]
    assert "no interval is drawn" in _core.note_text(fig)
    assert status_record(WITHHELD).plain_reason in _core.note_text(fig)
    assert "standard error" not in _subtitle(fig)
    assert "plug-in SE (diagnostic)" in _subtitle(fig)
    shown = repeats(result)
    assert [p for p in shown.axes[0].patches if p.get_gid() == "cleverly-repeat-interval"]
    assert "standard error" in _subtitle(shown)


def test_v17_experimental_estimate_says_so(result: Any) -> None:
    experimental = replace(result["ate"], support=UNCLASSIFIED)
    fig = repeats(replace(result, estimates={"ate": experimental}))
    assert "experimental" in _core.note_text(fig)


def test_v17_repeats_names_the_outcome_in_words(result: Any) -> None:
    column = result.data.outcome_name
    plain = repeats(result).axes[0].get_xlabel()
    assert plain == f"Difference in mean {column}"
    worded = repeats(result, outcome_label="transition score", units="points").axes[0]
    assert worded.get_xlabel() == "Difference in transition score (points)"
    assert column not in worded.get_xlabel()


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
    assert _title(fig) == "Bootstrap replicates"
    assert ax.get_ylabel() == "Density of replicates"


def test_v17_bootstrap_normal_curve_has_the_draws_mean_and_sd(result: Any) -> None:
    ax = bootstrap(result.bootstrap, name="ate").axes[0]
    draws = np.asarray(result.bootstrap.draws["ate"], dtype=float)
    x, y = _line(ax, "cleverly-bootstrap-normal").get_data()
    spread = float(np.std(draws, ddof=1))
    assert y.max() == pytest.approx(1.0 / (spread * np.sqrt(2 * np.pi)), rel=5e-3)
    assert x[np.argmax(y)] == pytest.approx(float(np.mean(draws)), abs=spread / 50)


def test_v17_bootstrap_of_a_fit_reads_its_own_interval(result: Any) -> None:
    """The limits are the fit's ``estimate.bootstrap.ci``, not a percentile recomputed here.

    Mutation control: moved limits in the summary move the drawn markers.
    """
    fig = bootstrap(result)
    ax = fig.axes[0]
    summary = result["ate"].bootstrap
    assert _markers(ax) == pytest.approx(sorted(summary.ci))
    (estimate,) = _lines(ax, _core.GID_REFERENCE)
    assert estimate.get_xdata()[0] == pytest.approx(result["ate"].psi)
    assert "95% percentile interval" in _core.key_labels(fig)
    assert f"bootstrap SE {summary.std_error:.3g}" in _subtitle(fig)
    assert _has_descriptive_note(fig)
    moved = _with_bootstrap(result, ci=(0.01, 0.42), interval="bca")
    moved_fig = bootstrap(moved)
    assert _markers(moved_fig.axes[0]) == pytest.approx([0.01, 0.42])
    assert "95% BCa interval" in _core.key_labels(moved_fig)


def test_v17_bootstrap_carries_its_experimental_label(result: Any) -> None:
    """Rule 3: the fixture's bootstrap interval is experimental, and the plot says so.

    Mutation control: a supported bootstrap drops the line.
    """
    assert result["ate"].bootstrap.support.label == "experimental"
    note = _core.note_text(bootstrap(result))
    assert "ate: the bootstrap interval is experimental" in note
    supported = _core.note_text(bootstrap(_with_bootstrap(result, support=SUPPORTED)))
    assert "experimental" not in supported


def test_v17_a_diagnostic_bootstrap_is_named_a_range(result: Any) -> None:
    """A declared diagnostic bootstrap prints an SD and a range, as the summary does.

    Mutation control: the inferential bootstrap prints an SE and an interval.
    """
    fig = bootstrap(_with_bootstrap(result, diagnostic=True))
    assert "bootstrap SD" in _subtitle(fig) and "bootstrap SE" not in _subtitle(fig)
    assert "percentile range (diagnostic)" in _core.key_labels(fig)
    assert "interval" not in " ".join(_core.key_labels(fig))
    expected = BOOTSTRAP_DIAGNOSTIC["declared"].format(names="ate", clusters=10)
    assert " ".join(expected.split()) in _core.note_text(fig)
    withheld = replace(result["ate"], inference=WITHHELD)
    held = bootstrap(replace(result, estimates={"ate": withheld}))
    assert "fit withholds that interval" in _core.note_text(held)
    assert "bootstrap SD" in _subtitle(held)
    inferential = bootstrap(result)
    assert "bootstrap SE" in _subtitle(inferential)
    assert "diagnostic" not in _core.note_text(inferential)


def _ratio_fit(result: Any, draws: np.ndarray) -> Any:
    estimate = result["ate"]
    summary = replace(
        estimate.bootstrap,
        draws=draws,
        ci=tuple(np.quantile(draws, [0.025, 0.975])),
        std_error=float(np.std(draws, ddof=1)),
    )
    ratio = replace(estimate, name="rr", scale="ratio", psi=1.5, log_psi=float(np.log(1.5)))
    return replace(
        result,
        estimates={"rr": replace(ratio, bootstrap=summary)},
        bootstrap=replace(result.bootstrap, draws={"rr": draws}),
    )


def test_v17_bootstrap_of_a_ratio_is_drawn_on_the_log_scale(result: Any) -> None:
    """A ratio's Wald interval is built on the log scale, so the normal curve is too.

    A log-normal sample matches its normal curve there. Mutation control: on the natural
    scale its curve would sit at the mean of the draws, not of their logs.
    """
    draws = np.exp(np.random.default_rng(3).normal(np.log(1.5), 0.4, size=400))
    fig = bootstrap(_ratio_fit(result, draws))
    ax = fig.axes[0]
    x, y = _line(ax, "cleverly-bootstrap-normal").get_data()
    logs = np.log(draws)
    assert x[np.argmax(y)] == pytest.approx(float(np.mean(logs)), abs=0.02)
    assert y.max() == pytest.approx(1.0 / (np.std(logs, ddof=1) * np.sqrt(2 * np.pi)), rel=5e-3)
    assert abs(x[np.argmax(y)] - float(np.mean(draws))) > 0.05
    assert _markers(ax) == pytest.approx(sorted(np.log(np.quantile(draws, [0.025, 0.975]))))
    assert ax.get_xlabel() == "Ratio (log scale)"
    ticks = [float(label.get_text()) for label in ax.get_xticklabels() if label.get_text()]
    assert any(tick > 1.0 for tick in ticks)  # the ticks read as ratios
    assert "Drawn on the log scale" in _core.note_text(fig)
    difference = replace(result, bootstrap=replace(result.bootstrap, draws={"ate": draws}))
    plain = bootstrap(difference.bootstrap, name="ate").axes[0]
    px, py = _line(plain, "cleverly-bootstrap-normal").get_data()
    assert px[np.argmax(py)] == pytest.approx(float(np.mean(draws)), abs=0.02)


def _binary_mean_fit(result: Any, draws: np.ndarray) -> Any:
    estimate = result["ate"]
    summary = replace(
        estimate.bootstrap,
        draws=draws,
        ci=tuple(np.quantile(draws, [0.025, 0.975])),
        std_error=float(np.std(draws, ddof=1)),
    )
    mean = replace(estimate, name="ey1", scale="level", psi=0.2, bounds=(0.0, 1.0))
    return replace(
        result,
        estimates={"ey1": replace(mean, bootstrap=summary)},
        bootstrap=replace(result.bootstrap, draws={"ey1": draws}),
    )


def test_v17_bootstrap_of_a_binary_mean_is_drawn_on_the_logit_scale(result: Any) -> None:
    """The mean of a binary outcome forms its interval on the logit scale, so the curve is drawn there.

    A logit-normal sample matches its normal curve on that scale. Mutation control: without
    the bounds the estimate keeps the identity scale and the curve sits at the mean of the
    draws instead.
    """
    from scipy.special import expit, logit

    draws = expit(np.random.default_rng(4).normal(logit(0.2), 0.5, size=400))
    fitted = _binary_mean_fit(result, draws)
    assert fitted["ey1"].interval_scale == "range_logit"
    fig = bootstrap(fitted)
    ax = fig.axes[0]
    x, y = _line(ax, "cleverly-bootstrap-normal").get_data()
    logits = logit(draws)
    assert x[np.argmax(y)] == pytest.approx(float(np.mean(logits)), abs=0.03)
    assert _markers(ax) == pytest.approx(sorted(logit(np.quantile(draws, [0.025, 0.975]))))
    assert "Drawn on the logit scale" in _core.note_text(fig)
    unbounded = replace(fitted, estimates={"ey1": replace(fitted["ey1"], bounds=None)})
    assert unbounded["ey1"].interval_scale != "range_logit"
    plain = bootstrap(unbounded).axes[0]
    px, py = _line(plain, "cleverly-bootstrap-normal").get_data()
    assert px[np.argmax(py)] == pytest.approx(float(np.mean(draws)), abs=0.01)
    assert "logit" not in _core.note_text(plain.figure)


def test_v17_a_bare_bootstrap_draws_its_percentile_limits(result: Any) -> None:
    """Mutation control: skewing the replicates moves the limits that are drawn."""
    ax = bootstrap(result.bootstrap, name="ate", level=0.9).axes[0]
    assert not _lines(ax, _core.GID_REFERENCE)
    assert "90% percentile limits" in _core.key_labels(ax.figure)
    draws = np.asarray(result.bootstrap.draws["ate"], dtype=float)
    skewed = replace(result.bootstrap, draws={"ate": np.exp(4.0 * draws)})
    markers = _markers(bootstrap(skewed, name="ate").axes[0])
    assert markers == pytest.approx(list(skewed.interval("ate", 0.05).ci))
    assert markers != pytest.approx(list(result.bootstrap.interval("ate", 0.05).ci))


def test_v17_failed_replicates_are_counted(result: Any) -> None:
    failing = replace(result.bootstrap, n_failed=3)
    note = _core.note_text(bootstrap(failing, name="ate"))
    assert "3 of 40 replicates failed" in note
    assert "failed" not in _core.note_text(bootstrap(result.bootstrap, name="ate"))


def test_v17_bootstrap_names_the_resampling(result: Any) -> None:
    assert "Rows resampled independently." in _core.note_text(bootstrap(result))
    clustered = replace(result.bootstrap, resampling="cluster")
    assert "Whole clusters resampled." in _core.note_text(bootstrap(clustered, name="ate"))


def test_v17_bootstrap_refusals(result: Any, single: Any) -> None:
    with pytest.raises(ValueError, match="no bootstrap replicates"):
        bootstrap(single)
    several = replace(
        result.bootstrap, draws={"a": np.arange(5.0), "b": np.arange(5.0) * 2.0}, std_errors={}
    )
    # the first estimand is drawn when none is named, as repeats() and influence() do
    assert "a:" in _subtitle(bootstrap(several))
    with pytest.raises(ValueError, match="no replicates"):
        bootstrap(several, name="c")
    one = replace(result.bootstrap, draws={"ate": np.array([0.1, np.nan, np.nan])})
    with pytest.raises(ValueError, match="fewer than two finite"):
        bootstrap(one, name="ate")
    constant = replace(result.bootstrap, draws={"ate": np.full(20, 0.3)})
    with pytest.raises(ValueError, match="every replicate"):
        bootstrap(constant, name="ate")


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
    """Rows 3 and -3 among 98 rows of +-1 (mean zero): squares 9, 9 and 98 ones, so the
    top 1% carry 9/116."""
    curve = np.concatenate([[3.0, -3.0], np.ones(49), -np.ones(49)])
    shuffled = np.random.default_rng(1).permutation(curve)
    fig = influence(_hand_estimate(shuffled))
    (ax,) = fig.axes
    x, y = _line(ax, "cleverly-influence-curve").get_data()
    squares = np.sort(shuffled**2)[::-1]
    np.testing.assert_allclose(x, np.arange(0, 101) / 100)
    np.testing.assert_allclose(y, np.concatenate([[0.0], np.cumsum(squares) / squares.sum()]))
    assert y[1] == pytest.approx(9.0 / 116.0)
    assert y[-1] == pytest.approx(1.0)
    (top,) = [c for c in ax.collections if c.get_gid() == "cleverly-influence-top"]
    np.testing.assert_allclose(top.get_offsets()[0], [0.01, 9.0 / 116.0])
    assert "the top 1% of rows (1 of 100) carry 7.8%" in _subtitle(fig)
    assert _title(fig) == "Concentration of the uncertainty"
    assert ax.get_xlabel() == "Top share of rows, largest squared influence first"
    assert ax.get_ylabel() == "Share of the sum of squared influence values"
    diagonal = _diagonal(ax)
    np.testing.assert_allclose(diagonal.get_xdata(), diagonal.get_ydata())
    assert "Even spread" in _core.key_labels(fig)
    # Kish count (sum of squares)^2 / sum of fourth powers = 116^2 / (2 * 81 + 98) = 51.8
    assert "Kish effective count: 52 of 100 rows" in _core.note_text(fig)


def test_v19_values_are_centred_as_the_variance_centres_them() -> None:
    """A shift by a constant does not change the variance, so it must not change the curve.

    Mutation control: uncentred squares of the shifted curve would draw a different curve.
    """
    curve = np.random.default_rng(2).normal(size=60)
    base = _line(influence(_hand_estimate(curve)).axes[0], "cleverly-influence-curve")
    shifted = _line(influence(_hand_estimate(curve + 5.0)).axes[0], "cleverly-influence-curve")
    np.testing.assert_allclose(shifted.get_ydata(), base.get_ydata())
    raw = np.sort((curve + 5.0) ** 2)[::-1]
    uncentred = np.concatenate([[0.0], np.cumsum(raw) / raw.sum()])
    assert not np.allclose(uncentred, base.get_ydata())


def test_v19_names_the_share_actually_taken() -> None:
    """Below 100 units the top unit is not 1%: the headline says how much it is.

    Mutation control: at a count of 100 the plain "top 1%" wording is exact and is kept.
    """
    forty = influence(_hand_estimate(np.random.default_rng(0).normal(size=40)))
    assert "the largest row (2.5% of 40) carries" in _subtitle(forty)
    assert "top 1%" not in _subtitle(forty)
    label = next(t for t in forty.axes[0].texts if t.get_gid() == "cleverly-influence-top-label")
    assert label.get_text().startswith("Largest row:")
    many = influence(_hand_estimate(np.random.default_rng(0).normal(size=250)))
    assert "the top 3 of 250 rows (1.2%) carry" in _subtitle(many)
    hundred = influence(_hand_estimate(np.random.default_rng(0).normal(size=100)))
    assert "the top 1% of rows (1 of 100) carry" in _subtitle(hundred)


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
    """Cluster sums are 8, 0, 0, 0, 0 with mean 1.6: centred squares 40.96 and four 2.56."""
    curve, codes = _clustered()
    estimate = _hand_estimate(curve, n_clusters=5)
    fig = influence(estimate, cluster=codes)
    (ax,) = fig.axes
    x, y = _line(ax, "cleverly-influence-curve").get_data()
    np.testing.assert_allclose(x, np.arange(6) / 5)
    total = 6.4**2 + 4 * 1.6**2
    np.testing.assert_allclose(y, np.cumsum([0, 6.4**2, *[1.6**2] * 4]) / total)
    assert ax.get_xlabel() == "Top share of clusters, largest squared influence first"
    assert "Share of clusters" in _core.note_text(fig)
    assert "the largest cluster (20% of 5) carries" in _subtitle(fig)


def test_v19_ranking_rows_instead_of_clusters_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: a plot that ignores the clusters draws different numbers."""
    curve, codes = _clustered()
    estimate = _hand_estimate(curve, n_clusters=5)
    expected = np.cumsum([0, 6.4**2, *[1.6**2] * 4]) / (6.4**2 + 4 * 1.6**2)

    def drawn() -> np.ndarray:
        fig = influence(estimate, cluster=codes)
        return np.asarray(_line(fig.axes[0], "cleverly-influence-curve").get_ydata())

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
    withheld = replace(single["ate"], inference=WITHHELD)
    before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="interval is withheld") as caught:
        influence(withheld)
    assert status_record(WITHHELD).plain_reason in str(caught.value)
    assert len(plt.get_fignums()) == before
    influence(single["ate"])  # mutation control: the ordinary estimate draws


def test_v19_an_experimental_estimate_says_so(single: Any) -> None:
    """Rule 3. Mutation control: a supported estimate carries no such line."""
    experimental = replace(single["ate"], support=UNCLASSIFIED)
    assert "ate: experimental" in _core.note_text(influence(experimental))
    supported = replace(single["ate"], support=SUPPORTED)
    assert "experimental" not in _core.note_text(influence(supported))


def test_v19_repeated_fit_names_the_curve_and_says_it_is_not_the_variance(
    result: Any, single: Any
) -> None:
    note = _core.note_text(influence(result))
    assert "combines 3 cross-fitting repeats: the curve is the median draw's curve" in note
    assert "not the reported variance" in note
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
    with pytest.raises(ValueError, match="all equal"):
        influence(_hand_estimate(np.zeros(10)))
    with pytest.raises(ValueError, match="non-finite"):
        influence(_hand_estimate([1.0, np.nan, 2.0]))
    with pytest.raises(TypeError, match="takes a ParameterEstimate"):
        influence(object())


def test_v19_the_ratio_scale_is_on_the_log_scale(single: Any) -> None:
    ratio = replace(single["ate"], scale="ratio", psi=1.15, log_psi=float(np.log(1.15)))
    assert "log scale" in _core.note_text(influence(ratio))


# -------------------------------------------------------------------------- plot() and style


def test_plot_methods_call_the_functions(
    result: Any,
    ensemble: Any,
    selection: Any,
    curve_calibration: Any,
) -> None:
    report = result.diagnostics.nuisance_models().models[0]
    assert _title(report.plot()) == "Calibration of predicted probabilities"
    assert _title(curve_calibration.plot()) == "Predicted and measured effects"
    assert _title(ensemble.plot()) == "Learner risks and weights"
    assert _title(selection.plot()) == "Treatment model selection"
    assert _title(result.bootstrap.plot("ate")) == "Bootstrap replicates"


def test_a_callers_axes_takes_the_header_and_notes(result: Any) -> None:
    """On the caller's axes the text goes on that axes, and the figure's layout is left alone."""
    fig, ax = plt.subplots()
    returned = repeats(result, ax=ax)
    assert returned is fig and not fig.texts
    assert ax.get_title(loc="left") == "Estimate across repeated splits"
    assert _core.subtitle_text(ax) and _has_descriptive_note(fig)


def _figures(
    result: Any, ensemble: Any, selection: Any, curve_calibration: Any, library_fit: Any
) -> list[Any]:
    return [
        calibration(result),
        calibration(result.diagnostics.nuisance_models().models[0]),
        calibration(curve_calibration),
        calibration(replace(curve_calibration, scale="log_ratio")),
        learners(ensemble),
        learners(selection),
        learners(library_fit),
        repeats(result),
        bootstrap(result),
        influence(result),
    ]


def _is_grey(color: str) -> bool:
    import colorsys

    red, green, blue = matplotlib.colors.to_rgb(color)
    return colorsys.rgb_to_hsv(red, green, blue)[1] < 0.2


def _colors(artist: Any) -> Any:
    if hasattr(artist, "get_edgecolor") and hasattr(artist, "get_facecolor"):
        faces = np.atleast_2d(artist.get_facecolor())
        edges = np.atleast_2d(artist.get_edgecolor())
        return np.vstack([faces, edges]) if faces.size and edges.size else faces
    return np.atleast_2d(matplotlib.colors.to_rgba(artist.get_color()))


def _style_checks(fig: Any) -> None:
    """Opaque white face, a title, labelled axes, and only palette colours.

    Every coloured (not grey) line, collection and patch, whatever its gid, must wear a
    palette colour, and an axes needs a y label or row labels in its ticks. No figure draws
    a note.
    """
    assert_no_note_text(fig)
    assert to_hex(fig.get_facecolor(), keep_alpha=True) == "#ffffffff"
    assert _title(fig)
    positions = [tuple(np.round(ax.get_position().bounds, 6)) for ax in fig.axes]
    assert len(positions) == len(set(positions)), "two axes overlay each other"
    fig.canvas.draw()
    for ax in fig.axes:
        assert ax.get_xlabel()
        siblings = ax.get_shared_y_axes().get_siblings(ax)
        ticks = [text.get_text() for text in ax.get_yticklabels()]
        assert ax.get_ylabel() or any(o.get_ylabel() for o in siblings) or any(ticks)
        used = set()
        for artist in [*ax.lines, *ax.collections, *ax.patches]:
            for color in np.atleast_2d(_colors(artist)):
                if color.size == 4 and color[3] == 0.0:
                    continue
                hexed = to_hex(color)
                if _is_grey(hexed):
                    continue
                assert hexed in PALETTE, f"{hexed} is not a palette colour"
                used.add(hexed)
        assert len(used) <= _core.MAX_SERIES


def test_the_style_checks_reject_an_off_palette_scatter() -> None:
    """Control: a scatter (a collection, with no gid) in an off-palette colour must fail."""
    fig, ax = plt.subplots()
    ax.set_title("A title", loc="left")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter([0, 1], [0, 1], color="#ff00ff")
    with pytest.raises(AssertionError, match="not a palette colour"):
        _style_checks(fig)


def test_every_figure_passes_the_style_checks(
    result: Any, ensemble: Any, selection: Any, curve_calibration: Any, library_fit: Any
) -> None:
    for fig in _figures(result, ensemble, selection, curve_calibration, library_fit):
        _style_checks(fig)


def _texts(fig: Any) -> list[Any]:
    """Every drawn text of a figure: figure texts and keys, titles, labels and visible ticks."""
    found: list[Any] = [*fig.texts, *fig.legends]
    for ax in fig.axes:
        found.extend(child for child in ax.get_children() if isinstance(child, Text))
        found.extend([ax.xaxis.label, ax.yaxis.label])
        if ax.get_legend() is not None:
            found.append(ax.get_legend())
        for axis, (low, high) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = min(low, high), max(low, high)
            for tick, label in zip(axis.get_majorticklocs(), axis.get_ticklabels(), strict=False):
                if low - 1e-9 <= tick <= high + 1e-9:
                    found.append(label)
    return [
        item
        for item in found
        if item.get_visible() and (not isinstance(item, Text) or item.get_text().strip())
    ]


def _inside(fig: Any) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    for item in _texts(fig):
        extent = item.get_window_extent(renderer)
        name = item.get_text()[:40] if isinstance(item, Text) else "a key"
        assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, f"{name!r} leaves the figure"
        assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, f"{name!r} leaves the figure"


def test_the_inside_check_rejects_text_off_the_figure() -> None:
    """Control: a note that runs past the right edge must fail."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel("x")
    fig.text(0.5, 0.02, "a note far too long to fit in the remaining width of this figure")
    with pytest.raises(AssertionError, match="leaves the figure"):
        _inside(fig)


def test_every_text_stays_inside_the_figure(
    result: Any, ensemble: Any, selection: Any, curve_calibration: Any, library_fit: Any
) -> None:
    for fig in _figures(result, ensemble, selection, curve_calibration, library_fit):
        _inside(fig)
    withheld = replace(result["ate"], inference=WITHHELD)
    _inside(repeats(replace(result, estimates={"ate": withheld})))


def test_every_figure_renders_to_a_png(
    result: Any,
    ensemble: Any,
    selection: Any,
    curve_calibration: Any,
    library_fit: Any,
    tmp_path: Any,
) -> None:
    for index, fig in enumerate(
        _figures(result, ensemble, selection, curve_calibration, library_fit)
    ):
        path = tmp_path / f"figure-{index}.png"
        fig.savefig(path, dpi=60)
        assert path.stat().st_size > 1000


def _cropped_text(fig: Any) -> list[str]:
    """The figure-level title, subtitle and key a tight bounding box would cut."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    gids = {_core.GID_TITLE, _core.GID_SUBTITLE}
    artists = [text for text in fig.texts if text.get_gid() in gids] + list(fig.legends)
    assert artists, "the figure has no figure-level header"
    cut = []
    for artist in artists:
        box = artist.get_window_extent(renderer)
        inside = (
            box.x0 >= tight.x0 - 0.5
            and box.x1 <= tight.x1 + 0.5
            and box.y0 >= tight.y0 - 0.5
            and box.y1 <= tight.y1 + 0.5
        )
        if not inside:
            label = artist.get_text() if isinstance(artist, Text) else "key"
            cut.append(label)
    return cut


def test_a_tight_save_keeps_the_figure_level_header(
    result: Any,
    ensemble: Any,
    selection: Any,
    curve_calibration: Any,
    library_fit: Any,
) -> None:
    # Jupyter's inline display and savefig(bbox_inches="tight") crop to the tight box.
    for fig in _figures(result, ensemble, selection, curve_calibration, library_fit):
        assert _cropped_text(fig) == [], _core.title_text(fig)


def test_a_header_left_out_of_the_layout_is_cropped(result: Any) -> None:
    # Mutation control: the check above must see a header the tight box drops.
    fig = repeats(result)
    title = next(text for text in fig.texts if text.get_gid() == _core.GID_TITLE)
    title.set_in_layout(False)
    assert _cropped_text(fig) == [title.get_text()]


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
        "influence",
        "learners",
        "repeats",
    ]

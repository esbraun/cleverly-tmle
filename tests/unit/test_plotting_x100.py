"""Overlap, learners and calibration of fits over time read the fit's stored numbers (X100 PR 1).

Each figure is read back from its artists and compared with the array the fit holds. Each
honesty rule has a mutation control: a deliberately wrong reader, patched in, must make the
same check fail, and a witness asserts first that the fixture can tell the two apart.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.plotting as plotting
from cleverly import (
    AdaptiveTMLEMethod,
    CausalStudy,
    CollaborativeTMLEMethod,
    ModelSpec,
    RegimeMean,
    Runtime,
    TimeToEvent,
    TrialATE,
    TrialWithExternalData,
)
from cleverly._refusal import RefusalReason
from cleverly.datasets import make_longitudinal, make_point_survival
from cleverly.estimators import atmle
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental
from cleverly.learners import GLM, SuperLearner
from cleverly.longitudinal import IPSI, LTMLE, StructuralNestedMean
from cleverly.longitudinal.structural import _extreme_share
from cleverly.plotting import _diagnostics_time, _overlap_time
from tests.unit.test_plotting_core import assert_no_note_text, assert_short_title

pytestmark = pytest.mark.xdist_group("plotting_x100")

#: Histogram bins of the overlap plot over [0, 1].
EDGES = np.linspace(0.0, 1.0, 41)
#: The columns of the two-node longitudinal fixtures.
COLUMNS = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _logistic(**settings: Any) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, **settings)


def _quiet(build: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def held() -> Any:
    """A time-to-event fit of a baseline treatment with a lower limit some rows fall below."""
    frame, _ = make_point_survival(n=600, seed=4, n_times=4)
    design = TimeToEvent(
        time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4)
    )
    return _quiet(
        lambda: CausalStudy(frame, design=design).estimate(
            RegimeMean({"arm1": 1, "arm0": 0}, horizons=(1, 2, 3, 4)),
            outcome_learner=LinearRegression(),
            pseudo_learner=LinearRegression(),
            treatment_learner=_logistic(),
            cross_fit=False,
            g_bounds=(0.35, 1.0),
        )
    )


def _confounded_survival(seed: int = 1, n: int = 600) -> pd.DataFrame:
    """A baseline treatment and an event that a strong confounder drives."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, expit(1.6 * w1))
    times = np.full(n, 4.0)
    event = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for t in range(1, 5):
        if t > 1:
            drop = alive & (rng.uniform(size=n) < expit(-2.5 + 0.8 * w2))
            times[drop] = t - 1
            alive &= ~drop
        happened = alive & (rng.uniform(size=n) < expit(-1.8 + 1.6 * w1 - 0.7 * a))
        times[happened] = t
        event[happened] = 1
        alive &= ~happened
    return pd.DataFrame({"time": times, "event": event, "A": a, "W1": w1, "W2": w2})


@pytest.fixture(scope="module")
def collaborative() -> Any:
    """A survival CTMLE whose weak outcome model makes the search add terms, per arm."""
    design = TimeToEvent(
        time="time", event="event", treatment="A", baseline=("W1", "W2"), grid=(1, 2, 3, 4)
    )
    method = CollaborativeTMLEMethod(
        models=ModelSpec(outcome_learner=_logistic(C=1e-4)), runtime=Runtime(random_state=0)
    )
    return _quiet(
        lambda: CausalStudy(_confounded_survival(), design=design).estimate(
            RegimeMean({"arm1": 1, "arm0": 0}, horizons=(2, 4)), method=method, cross_fit=False
        )
    )


@pytest.fixture(scope="module")
def stacked() -> Any:
    """An LTMLE fit with a two-member Super Learner for the treatment model."""
    frame, _ = make_longitudinal(n=400, seed=0)
    library = SuperLearner(
        library=[("logit", _logistic()), ("ridge_logit", _logistic(C=0.05))],
        task="classification",
        n_folds=3,
    )
    return _quiet(
        lambda: LTMLE(
            {"always": 1, "never": 0},
            n_folds=2,
            random_state=0,
            outcome_learner=LinearRegression(),
            treatment_learner=library,
            censoring_learner=_logistic(),
        ).fit(frame, **COLUMNS)
    )


def _external_frame(seed: int = 2, n: int = 500) -> pd.DataFrame:
    """Trial membership that a covariate drives, so some external rows have a tiny pi(W)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    trial = rng.binomial(1, expit(-0.5 + 2.5 * w))
    treated = np.where(trial == 1, rng.binomial(1, 0.5, size=n), rng.binomial(1, expit(w), size=n))
    y = w + 0.5 * treated + rng.normal(size=n)
    return pd.DataFrame({"score": y, "program": treated, "risk": w, "in_trial": trial})


def _adaptive_fit(frame: pd.DataFrame) -> Any:
    design = TrialWithExternalData(
        outcome="score", treatment="program", adjustment=("risk",), source="in_trial"
    )
    models = ModelSpec(outcome_learner=LinearRegression(), treatment_learner=_logistic())
    return _quiet(
        lambda: (
            CausalStudy(frame, design=design)
            .identify(TrialATE(population="trial"))
            .estimate(method=AdaptiveTMLEMethod(models=models, runtime=Runtime(random_state=1)))
        )
    )


@pytest.fixture(scope="module")
def adaptive() -> Any:
    return _adaptive_fit(_external_frame())


@pytest.fixture(scope="module")
def structural() -> Any:
    frame, _ = make_longitudinal(n=300, seed=0, censoring=False)
    return _quiet(
        lambda: StructuralNestedMean(
            {"always": 1, "never": 0},
            outcome_learner=GLM(family="binomial"),
            treatment_learner=GLM(family="binomial"),
            n_folds=2,
            random_state=0,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
        )
    )


@pytest.fixture
def drawn() -> Iterator[list[Figure]]:
    """Figures a test draws; each is checked for the style rules and closed."""
    figures: list[Figure] = []
    yield figures
    for fig in figures:
        plt.close(fig)


def _draw(drawn: list[Figure], plot: Callable[..., Figure], *args: Any, **kwargs: Any) -> Figure:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot(*args, **kwargs)
    drawn.append(fig)
    assert_no_note_text(fig)
    assert_short_title(fig)
    return fig


# --------------------------------------------------------------------------- readers


def _bars(ax: Any, side: str) -> np.ndarray:
    """The signed bar heights of one side of an overlap panel, one per bin."""
    found = [patch for patch in ax.patches if patch.get_gid() == f"cleverly-overlap-{side}"]
    return np.asarray([patch.get_height() for patch in found], dtype=float)


def _shares(values: Any, mask: Any, total: float) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values)[mask], bins=EDGES)
    return counts / total


def _check_mirror(ax: Any, values: Any, inside: Any, rows: Any) -> None:
    """The panel's bars are ``values`` over ``rows``: ``inside`` above, the rest below."""
    rows = np.asarray(rows, dtype=bool)
    total = float(rows.sum())
    np.testing.assert_allclose(_bars(ax, "above"), _shares(values, rows & inside, total))
    below = _bars(ax, "below")
    if below.size:
        np.testing.assert_allclose(-below, _shares(values, rows & ~inside, total))


def _check_one_side(ax: Any, values: Any, rows: Any) -> None:
    rows = np.asarray(rows, dtype=bool)
    np.testing.assert_allclose(_bars(ax, "above"), _shares(values, rows, float(rows.sum())))
    assert not _bars(ax, "below").size


def _markers(ax: Any) -> list[float]:
    return [
        float(np.asarray(line.get_xdata(), dtype=float)[0])
        for line in ax.get_lines()
        if line.get_gid() == "cleverly-marker"
    ]


def _tail_texts(ax: Any) -> list[str]:
    return [text.get_text() for text in ax.texts if text.get_gid() == "cleverly-tail-share"]


def _panels(fig: Figure) -> list[Any]:
    """The panels with overlap bars, in reading order."""
    return [ax for ax in fig.axes if _bars(ax, "above").size]


# --------------------------------------------------------------------------- B1


def test_a_baseline_treatment_draws_the_unbounded_node_one_probability(
    held: Any, drawn: list[Figure]
) -> None:
    raw = held.mechanism.treatment_observed[0][:, 1]
    bounded = held.repeats[0].fits["arm1 @ t=1"].cumulative[:, 0]
    # Witness: rows below the lower limit, where the bounded product differs from g.
    low = float(held.config.g_bounds[0])
    assert np.sum(raw < low) > 10
    assert not np.allclose(np.histogram(raw, EDGES)[0], np.histogram(bounded, EDGES)[0])
    fig = _draw(drawn, plotting.overlap, held)
    (panel,) = _panels(fig)
    codes = held.data.treatment[:, 0]
    rows = np.isfinite(codes)
    _check_mirror(panel, raw, codes == 1.0, rows)
    # No limit sits on the treatment probability: g_bounds bound the cumulative product.
    assert _markers(panel) == []
    assert any("cumulative" not in note and "g_bounds" in note for note in plotting.notes(fig))


def test_reading_the_bounded_cumulative_fails_the_baseline_check(
    held: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a reader of ``cumulative[:, 0]`` in place of g fails the same check."""
    bounded = held.repeats[0].fits["arm1 @ t=1"].cumulative[:, 0]

    def wrong(result: Any) -> tuple[Any, tuple[object, ...]]:
        return np.column_stack([1.0 - bounded, bounded]), tuple(result.data.treatment_levels[0])

    monkeypatch.setattr(_overlap_time, "baseline_probability", wrong)
    fig = _draw(drawn, plotting.overlap, held)
    codes = held.data.treatment[:, 0]
    raw = held.mechanism.treatment_observed[0][:, 1]
    with pytest.raises(AssertionError):
        _check_mirror(_panels(fig)[0], raw, codes == 1.0, np.isfinite(codes))


def test_the_censoring_panel_draws_the_cumulative_product_with_its_limit(
    held: Any, drawn: list[Figure]
) -> None:
    fig = _draw(drawn, plotting.overlap, held, censoring=True, horizon=3)
    panels = _panels(fig)
    assert len(panels) == 3
    codes = held.data.treatment[:, 0]
    low = float(held.config.g_bounds[0])
    for panel, label, code in zip(panels[1:], ("arm1", "arm0"), (1.0, 0.0), strict=True):
        product = held.repeats[0].fits[f"{label} @ t=3"].cumulative_unbounded[:, 2]
        followers = codes == code
        _check_one_side(panel, product, followers)
        assert _markers(panel) == [pytest.approx(low)]
        share = float(np.mean(product[followers] < low))
        assert _tail_texts(panel) == [f"{100 * share:.1f}% below"]
    assert any("horizon 3" in note for note in plotting.notes(fig))


def test_a_collaborative_fit_draws_each_arms_selected_model(
    collaborative: Any, drawn: list[Figure]
) -> None:
    mechanism = collaborative.mechanism
    arm1, arm0 = mechanism.treatment[0]["arm1"], mechanism.treatment[0]["arm0"]
    # Witness: each arm's model is a different selection, so the two are not complements,
    # and the row-mixed observed matrix differs from either arm's own model.
    last = {arm: collaborative.selection[(arm, 4)].selected_terms[0] for arm in ("arm1", "arm0")}
    assert last["arm1"] != last["arm0"]
    assert not np.allclose(arm1 + arm0, 1.0)
    fig = _draw(drawn, plotting.overlap, collaborative)
    panels = _panels(fig)
    assert len(panels) == 2
    codes = collaborative.data.treatment[:, 0]
    rows = np.isfinite(codes)
    for panel, values, code in zip(panels, (arm1, arm0), (1.0, 0.0), strict=True):
        _check_mirror(panel, values, codes == code, rows)
    assert "horizon 4" in fig.axes[0].get_title(loc="left")
    assert any("Other horizons' selections are not drawn" in note for note in plotting.notes(fig))


def test_reading_the_row_mixed_matrix_fails_the_collaborative_check(
    collaborative: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: ``treatment_observed[0]`` mixes the arms' models row by row."""
    observed = collaborative.mechanism.treatment_observed[0]

    def wrong(result: Any, label: str) -> Any:
        return observed[:, 1] if label == "arm1" else observed[:, 0]

    monkeypatch.setattr(_overlap_time, "selected_probability", wrong)
    fig = _draw(drawn, plotting.overlap, collaborative)
    codes = collaborative.data.treatment[:, 0]
    mechanism = collaborative.mechanism
    with pytest.raises(AssertionError):
        for panel, label, code in zip(_panels(fig), ("arm1", "arm0"), (1.0, 0.0), strict=True):
            _check_mirror(panel, mechanism.treatment[0][label], codes == code, np.isfinite(codes))


# --------------------------------------------------------------------------- B4


def _deviators(result: Any, label: str, time: int) -> tuple[Any, Any, Any]:
    fit = result.repeats[0].fits[label]
    data = result.data
    rows = data.at_risk(fit.assignment, time)
    took = data.treatment[:, time - 1] == fit.assignment[:, time - 1]
    return fit, rows, took


def test_each_time_point_mirrors_the_units_at_risk_by_whether_they_took_it(
    stacked: Any, drawn: list[Figure]
) -> None:
    fit, rows, took = _deviators(stacked, "always", 2)
    g = stacked.mechanism.treatment[1]["always"]
    # Witness: some units at risk deviate at node 2 with a small g, and the regression's
    # training rows leave them out.
    assert np.sum(rows & ~took & (g < 0.3)) >= 1
    assert np.sum(rows & ~took) > 10
    assert not np.array_equal(rows, fit.steps[1].trained_on)
    fig = _draw(drawn, plotting.overlap, stacked)
    panels = _panels(fig)
    assert len(panels) == stacked.data.n_times
    for time, panel in enumerate(panels, start=1):
        _, at_risk, took_t = _deviators(stacked, "always", time)
        _check_mirror(panel, stacked.mechanism.treatment[time - 1]["always"], took_t, at_risk)
        assert _markers(panel) == []
    assert any("pass regimen=" in note for note in plotting.notes(fig))


def test_reading_the_training_rows_fails_the_per_node_check(
    stacked: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: the rows that followed through the node hide the deviators."""
    fit = stacked.repeats[0].fits["always"]

    def wrong(data: Any, assignment: Any, time: int) -> Any:
        return fit.steps[time - 1].trained_on

    monkeypatch.setattr(_overlap_time, "node_population", wrong)
    fig = _draw(drawn, plotting.overlap, stacked, nodes=[2])
    _, rows, took = _deviators(stacked, "always", 2)
    with pytest.raises(AssertionError):
        _check_mirror(_panels(fig)[0], stacked.mechanism.treatment[1]["always"], took, rows)


def test_the_retention_panels_draw_each_nodes_probability_over_its_followers(
    stacked: Any, drawn: list[Figure]
) -> None:
    fig = _draw(drawn, plotting.overlap, stacked, regimen="never", censoring=True)
    panels = _panels(fig)
    assert len(panels) == 2 * stacked.data.n_times
    for time, panel in enumerate(panels[stacked.data.n_times :], start=1):
        _, rows, took = _deviators(stacked, "never", time)
        _check_one_side(panel, stacked.mechanism.censoring[time - 1]["never"], rows & took)
        assert _markers(panel) == []
    assert "Regimen never" in " ".join(text.get_text() for text in fig.texts)


def test_overlap_nodes_selects_panels_and_checks_them(stacked: Any, drawn: list[Figure]) -> None:
    fig = _draw(drawn, plotting.overlap, stacked, nodes=[2])
    assert [ax.get_title(loc="left") for ax in _panels(fig)] == ["Time point 2"]
    with pytest.raises(DataError, match="time points 1 to 2"):
        plotting.overlap(stacked, nodes=[3])
    with pytest.raises(DataError, match="no regimen"):
        plotting.overlap(stacked, regimen="sometimes")
    with pytest.raises(ValueError, match="does not apply"):
        plotting.overlap(stacked, horizon=2)


def test_an_incremental_policy_refuses_per_node_overlap() -> None:
    frame, _ = make_longitudinal(n=300, seed=0)
    result = _quiet(
        lambda: IPSI(
            [Incremental(2.0)],
            outcome_learner=LinearRegression(),
            pseudo_learner=LinearRegression(),
            treatment_learner=_logistic(),
            censoring_learner=_logistic(),
            n_folds=1,
        ).fit(frame, **COLUMNS)
    )
    with pytest.raises(CapabilityError) as caught:
        plotting.overlap(result)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert caught.value.refusal.roadmap is None
    assert "odds of treatment" in str(caught.value)


# --------------------------------------------------------------------------- B3


def test_adaptive_storage_leaves_the_estimates_bit_identical(
    adaptive: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored raw g feeds nothing: a fit that drops it reports the same numbers."""
    original = atmle._nuisances_on

    def without(*args: Any, **kwargs: Any) -> Any:
        values, raw_pibar, raw_g = original(*args, **kwargs)
        return values, raw_pibar, np.full_like(raw_g, np.nan)

    monkeypatch.setattr(atmle, "_nuisances_on", without)
    dropped = _adaptive_fit(_external_frame())
    assert np.isnan(dropped.fit.treatment_probability).all()
    assert dropped.fit.estimate == adaptive.fit.estimate
    assert dropped.fit.std_error == adaptive.fit.std_error
    np.testing.assert_array_equal(dropped.fit.influence_curve, adaptive.fit.influence_curve)
    # The stored arrays reproduce the summaries the fit already reported.
    fit = adaptive.fit
    external = np.asarray(adaptive.data.source) == 0.0
    low, high = fit.probability_bounds
    assert (low, high) == tuple(adaptive.g_bounds)
    assert fit.diagnostics.external_source_min == float(fit.source_probability[external].min())
    trial_g = np.clip(fit.treatment_probability[~external], low, high)
    assert fit.diagnostics.trial_row_treatment_range == (
        float(trial_g.min()),
        float(trial_g.max()),
    )


def test_adaptive_overlap_draws_the_stored_raw_probabilities(
    adaptive: Any, drawn: list[Figure]
) -> None:
    fit = adaptive.fit
    trial = np.asarray(adaptive.data.source) == 1.0
    treated = np.asarray(adaptive.data.treatment) == 1.0
    low, high = fit.probability_bounds
    # Witness: external rows below both the floor and the clip pair.
    assert np.sum(fit.source_probability[~trial] < atmle.SOURCE_FLOOR) > 0
    assert np.sum(fit.source_probability < low) > np.sum(fit.source_probability < 0.01)
    fig = _draw(drawn, plotting.overlap, adaptive)
    top, bottom = _panels(fig)
    _check_mirror(top, fit.treatment_probability, treated, trial)
    _check_mirror(bottom, fit.source_probability, trial, np.ones(trial.size, dtype=bool))
    assert sorted(_markers(bottom)) == pytest.approx(sorted([atmle.SOURCE_FLOOR, low, high]))
    share = fit.diagnostics.external_source_below_floor
    assert f"{100 * share:.1f}% below" in _tail_texts(bottom)


def test_reading_the_clipped_membership_fails_the_adaptive_check(
    adaptive: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: the clipped membership the targeting starts from differs."""
    fit = adaptive.fit
    low, high = fit.probability_bounds

    def wrong(stored: Any) -> tuple[Any, Any]:
        return stored.treatment_probability, np.clip(stored.source_probability, low, high)

    monkeypatch.setattr(_overlap_time, "adaptive_probabilities", wrong)
    fig = _draw(drawn, plotting.overlap, adaptive)
    trial = np.asarray(adaptive.data.source) == 1.0
    with pytest.raises(AssertionError):
        _check_mirror(_panels(fig)[1], fit.source_probability, trial, np.ones(trial.size, bool))


# --------------------------------------------------------------------------- B5


def test_structural_storage_reproduces_the_positivity_summary(structural: Any) -> None:
    diagnostics = structural.diagnostics
    weights = structural.data.weights
    margin = structural.config.positivity_margin
    assert len(diagnostics.class_probabilities) == len(structural.config.nodes)
    for share, own in zip(diagnostics.extreme_share, diagnostics.class_probabilities, strict=True):
        assert _extreme_share(own, weights, margin) == share


def test_structural_overlap_draws_class_probabilities_with_the_margin(
    structural: Any, drawn: list[Figure]
) -> None:
    fig = _draw(drawn, plotting.overlap, structural)
    panels = _panels(fig)
    margin = structural.config.positivity_margin
    assert len(panels) == len(structural.config.nodes)
    for time, panel in enumerate(panels, start=1):
        codes = structural.data.node_codes(time)
        values = structural.diagnostics.class_probabilities[time - 1][:, 0]
        _check_mirror(panel, values, codes == 1.0, np.isfinite(codes))
        assert sorted(_markers(panel)) == pytest.approx([margin, 1.0 - margin])
        shares = [float(text.split("%")[0].lstrip("<")) for text in _tail_texts(panel)]
        assert len(shares) == 2
    assert any("extreme_share" in note for note in plotting.notes(fig))


def test_a_structural_dose_node_refuses_overlap() -> None:
    rng = np.random.default_rng(0)
    n = 400
    w = rng.normal(size=n)
    dose = 0.5 * w + rng.normal(size=n)
    a2 = rng.binomial(1, expit(0.4 * w))
    y = 0.5 * dose + a2 + w + rng.normal(size=n)
    frame = pd.DataFrame({"W": w, "D1": dose, "A2": a2, "Y": y, "S1": w, "S2": w})
    result = _quiet(
        lambda: StructuralNestedMean(
            None,
            outcome_learner=GLM(family="gaussian"),
            treatment_learner=GLM(family="binomial"),
            regression_learner=GLM(family="gaussian"),
            random_state=0,
            simultaneous=False,
        ).fit(
            frame,
            outcome="Y",
            treatment=["D1", "A2"],
            baseline=["W"],
            continuous_treatment=["D1"],
        )
    )
    assert np.isnan(result.diagnostics.extreme_share[0])
    with pytest.raises(CapabilityError, match="holds a dose") as caught:
        plotting.overlap(result)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    fig = plotting.overlap(result, nodes=[2])
    assert len(_panels(fig)) == 1
    plt.close(fig)


# --------------------------------------------------------------------------- C1


def _chosen_x(ax: Any) -> float:
    (points,) = [item for item in ax.collections if item.get_gid() == "cleverly-ctmle-chosen"]
    return float(points.get_offsets()[0][0])


def _check_marked(fig: Figure, selections: dict[tuple[str, int], Any]) -> None:
    panels = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert len(panels) == len(selections)
    for ax in panels:
        arm, horizon = re.fullmatch(r"(.+), horizon (\d+)", ax.get_title(loc="left")).groups()
        selection = selections[(arm, int(horizon))]
        assert _chosen_x(ax) == selection.selected
        (line,) = [item for item in ax.get_lines() if item.get_gid() == "cleverly-ctmle-risk"]
        np.testing.assert_array_equal(line.get_ydata(), selection.cv_risk)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == [selection._label(index) for index in range(len(selection.moves))]


def test_the_survival_selection_path_marks_the_selected_position(
    collaborative: Any, drawn: list[Figure]
) -> None:
    selections = collaborative.selection
    # Witness: the search moved past M0, and the training risk would choose otherwise.
    assert any(selection.selected > 0 for selection in selections.values())
    assert any(
        int(np.argmin(selection.train_risk)) != selection.selected
        for selection in selections.values()
    )
    fig = _draw(drawn, plotting.learners, collaborative)
    _check_marked(fig, selections)


def test_marking_the_training_risk_minimum_fails_the_selection_check(
    collaborative: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a reader that marks the training-risk minimum fails."""

    def wrong(selection: Any) -> int:
        return int(np.argmin(selection.train_risk))

    monkeypatch.setattr(_diagnostics_time, "chosen_position", wrong)
    fig = _draw(drawn, plotting.learners, collaborative)
    with pytest.raises(AssertionError):
        _check_marked(fig, collaborative.selection)


# --------------------------------------------------------------------------- C2 and C5


def _learner_bars(ax: Any) -> dict[tuple[float, str], float]:
    return {
        (round(patch.get_x() + patch.get_width() / 2.0, 6), str(patch.get_url())): float(
            patch.get_height()
        )
        for patch in ax.patches
        if patch.get_gid() == "cleverly-learner-weight"
    }


def test_node_learners_draw_each_nodes_super_learner_weights(
    stacked: Any, drawn: list[Figure]
) -> None:
    rows = [
        row
        for row in stacked.diagnostics.nuisance_models().rows
        if row.role == "treatment" and row.model.learner_weights
    ]
    # Witness: the weights differ by node, so a reader that repeats one node fails.
    assert len(rows) == 2
    assert dict(rows[0].model.learner_weights) != dict(rows[1].model.learner_weights)
    fig = _draw(drawn, plotting.learners, stacked)
    (top,) = [ax for ax in fig.axes if ax.get_title(loc="left") == "Treatment model"]
    bars = _learner_bars(top)
    for row in rows:
        heights = {name: bars[(float(row.time), name)] for name in row.model.learner_weights}
        assert heights == pytest.approx(dict(row.model.learner_weights))
        assert sum(heights.values()) == pytest.approx(1.0)
    risks = [ax for ax in fig.axes if ax.get_ylabel().startswith("Cross-validated risk")]
    lines = {line.get_url(): line for line in risks[0].get_lines()}
    for name in rows[0].model.learner_risks:
        expected = [row.model.learner_risks[name] for row in rows]
        np.testing.assert_allclose(lines[name].get_ydata(), expected)
    assert any("mean over the model's Super Learner fits" in note for note in plotting.notes(fig))


def test_node_learners_select_the_outcome_regressions_and_refuse_unknown_ones(
    stacked: Any, drawn: list[Figure]
) -> None:
    report = stacked.diagnostics.nuisance_models()
    chosen = _diagnostics_time.choose_rows(report, None, None, None)
    assert chosen.regimen == "always"
    assert {row.regimen for row in chosen.rows} == {None, "always"}
    never = _diagnostics_time.choose_rows(report, "never", None, None)
    assert {row.regimen for row in never.rows} == {None, "never"}
    with pytest.raises(DataError, match="no regimen"):
        plotting.learners(stacked, regimen="sometimes")
    _draw(drawn, plotting.learners, report, regimen="never")


def test_node_calibration_draws_one_panel_per_probability_model_and_time_point(
    stacked: Any, drawn: list[Figure]
) -> None:
    chosen = _diagnostics_time.choose_rows(stacked.diagnostics.nuisance_models(), None, None, None)
    expected = [
        row
        for row in chosen.rows
        if row.model.kind == "probability" and len(row.model.calibration["n"])
    ]
    fig = _draw(drawn, plotting.calibration, stacked)
    panels = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert len(panels) == len(expected)
    by_title = {ax.get_title(loc="left"): ax for ax in panels}
    for row in expected:
        title = f"{_diagnostics_time.role_words(row.role)}, time point {row.time}"
        (points,) = [
            item
            for item in by_title[title].collections
            if item.get_gid() == "cleverly-calibration-points"
        ]
        np.testing.assert_allclose(
            points.get_offsets(),
            np.column_stack(
                [row.model.calibration["mean_predicted"], row.model.calibration["mean_observed"]]
            ),
        )


def test_adaptive_learners_and_calibration_refuse_with_a_true_reason(adaptive: Any) -> None:
    for plot in (plotting.learners, plotting.calibration):
        with pytest.raises(CapabilityError, match="learner name only") as caught:
            plot(adaptive)
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert caught.value.refusal.roadmap is None

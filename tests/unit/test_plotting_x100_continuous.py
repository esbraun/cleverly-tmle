"""Overlap, weights and learners of a continuous-time survival fit read its record (X100 PR 2).

Each figure is read back from its artists and compared with the arrays
:class:`~cleverly.survival.ContinuousSurvivalRecord` holds. Each reader has a mutation
control: a deliberately wrong reader, patched in, must make the same check fail, and a
witness asserts first that the fixture can tell the two apart.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.plotting as plotting
from cleverly.datasets import make_longitudinal
from cleverly.exceptions import CapabilityError, DataError
from cleverly.longitudinal import LTMLE
from cleverly.plotting import _continuous
from cleverly.survival.cox import CoxHazard
from cleverly.survival.engine import ContinuousSurvivalTMLE
from tests.unit.test_plotting_x100 import (
    _check_mirror,
    _check_one_side,
    _draw,
    _markers,
    _panels,
    _quiet,
    _tail_texts,
)

pytestmark = pytest.mark.xdist_group("plotting_x100_continuous")


def _frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    """Two competing causes; a confounder drives treatment, and treatment drives dropout."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    treated = rng.binomial(1, 1 / (1 + np.exp(-1.5 * x)))
    relapse = np.exp(-1.5 + 0.5 * x - treated)
    death = np.exp(-2.0 - 0.5 * x - 0.5 * treated)
    event_time = rng.exponential(1 / (relapse + death))
    cause = np.where(rng.uniform(size=n) < relapse / (relapse + death), 1, 2)
    dropout = rng.exponential(1 / (0.15 * np.exp(0.8 * treated - 0.5 * x)))
    return pd.DataFrame(
        {
            "time": np.minimum(event_time, dropout),
            "event": np.where(event_time <= dropout, cause, 0),
            "treated": treated,
            "x": x,
        }
    )


def _fit(**settings: Any) -> Any:
    estimator = ContinuousSurvivalTMLE(
        horizons=(1.0, 2.0),
        arms={"treated": 1, "control": 0},
        reference="control",
        causes={1: "relapse", 2: "death"},
        random_state=0,
        **settings,
    )
    return _quiet(
        lambda: estimator.fit(
            _frame(), time="time", event="event", treatment="treated", baseline=["x"]
        )
    )


@pytest.fixture(scope="module")
def floored() -> Any:
    """A cross-fitted fit whose floor of 0.15 some rows' products fall below."""
    return _fit(g_bounds=0.15)


@pytest.fixture(scope="module")
def supplied() -> Any:
    """A cross-fitted fit with a supplied treatment learner, so no treatment selection."""
    return _fit(cross_fit=True, n_folds=2, treatment_learner=LogisticRegression())


@pytest.fixture(scope="module")
def full_sample() -> Any:
    """A fit on every row with a one-member hazard library, which is not cross-validated."""
    return _fit(cross_fit=False, hazard_library=(CoxHazard("main"),))


@pytest.fixture
def drawn() -> Iterator[list[Figure]]:
    figures: list[Figure] = []
    yield figures
    for fig in figures:
        plt.close(fig)


def _arrays(result: Any) -> tuple[Any, Any, Any, float]:
    record = result.record
    p1 = np.asarray(record.p1, dtype=float)
    pi = np.stack([1.0 - p1, p1])
    return pi, np.asarray(record.retention_before), np.asarray(record.arm), record.weight_floor


def _texts(fig: Figure) -> list[str]:
    return [text.get_text() for text in fig.findobj(matplotlib.text.Text)]


def _percent(share: float) -> str:
    return f"{100 * share:.1f}%"


# --------------------------------------------------------------------------- B2


def test_overlap_draws_the_raw_treated_probability_by_received_arm(
    floored: Any, drawn: list[Figure]
) -> None:
    pi, _, arm, floor = _arrays(floored)
    clipped = np.clip(pi[1], floor, 1.0 - floor)
    # Witness: rows beyond the floor, where a clipped probability would bin differently.
    assert np.sum((pi[1] < floor) | (pi[1] > 1.0 - floor)) > 20
    fig = _draw(drawn, plotting.overlap, floored)
    (panel,) = _panels(fig)
    _check_mirror(panel, pi[1], arm == 1, np.ones(arm.size, dtype=bool))
    with pytest.raises(AssertionError):
        _check_mirror(panel, clipped, arm == 1, np.ones(arm.size, dtype=bool))
    # The floor bounds the product with the probability of staying uncensored, not pi.
    assert _markers(panel) == []
    assert panel.get_xlabel() == "Predicted probability of treated"
    assert any("not the treatment probability alone" in note for note in plotting.notes(fig))
    assert any(text.startswith("Out-of-fold predictions") for text in _texts(fig))


def test_reading_the_clipped_probability_fails_the_overlap_check(
    floored: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a reader that clips pi at the floor fails the same check."""
    pi, _, arm, floor = _arrays(floored)
    monkeypatch.setattr(
        _continuous,
        "treatment_probabilities",
        lambda record: np.clip(pi, floor, 1.0 - floor),
    )
    fig = _draw(drawn, plotting.overlap, floored)
    with pytest.raises(AssertionError):
        _check_mirror(_panels(fig)[0], pi[1], arm == 1, np.ones(arm.size, dtype=bool))


@pytest.mark.parametrize("horizon", [1.0, None])
def test_the_censoring_panels_draw_each_arms_product_before_the_target(
    floored: Any, drawn: list[Figure], horizon: float | None
) -> None:
    pi, retention, arm, floor = _arrays(floored)
    k = 0 if horizon == 1.0 else 1
    # Witnesses: the two arms' retention differ on the same rows, the two target columns
    # differ, and some products fall below the floor.
    assert np.abs(retention[0] - retention[1]).max() > 0.1
    assert np.abs(retention[..., 0] - retention[..., 1]).max() > 0.1
    fig = _draw(drawn, plotting.overlap, floored, censoring=True, horizon=horizon)
    panels = _panels(fig)
    assert len(panels) == 3
    for panel, code in zip(panels[1:], (1, 0), strict=True):
        rows = arm == code
        product = pi[code] * retention[code, :, k]
        _check_one_side(panel, product, rows)
        assert _markers(panel) == [pytest.approx(floor)]
        share = float(np.mean(product[rows] < floor))
        assert share > 0.0
        assert _tail_texts(panel) == [f"{_percent(share)} below"]
    assert any(f"target time {[1, 2][k]}" in note for note in plotting.notes(fig))


@pytest.mark.parametrize(
    ("mutation", "why"),
    [
        (lambda retention: retention[::-1], "the other arm's retention"),
        (lambda retention: retention[..., ::-1], "another target time's column"),
    ],
)
def test_reading_the_wrong_retention_fails_the_censoring_check(
    floored: Any,
    drawn: list[Figure],
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    why: str,
) -> None:
    """Mutation control: reading the other arm's axis, or the column of another time."""
    pi, retention, arm, _ = _arrays(floored)
    monkeypatch.setattr(_continuous, "retention_before", lambda record: mutation(retention))
    fig = _draw(drawn, plotting.overlap, floored, censoring=True, horizon=1.0)
    with pytest.raises(AssertionError):
        for panel, code in zip(_panels(fig)[1:], (1, 0), strict=True):
            _check_one_side(panel, pi[code] * retention[code, :, 0], arm == code)


def test_reading_the_floored_product_fails_the_censoring_check(
    floored: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a product raised to the floor leaves no share below it."""
    pi, retention, arm, floor = _arrays(floored)
    raised = np.maximum(pi[:, :, None] * retention, floor) / pi[:, :, None]
    monkeypatch.setattr(_continuous, "retention_before", lambda record: raised)
    fig = _draw(drawn, plotting.overlap, floored, censoring=True)
    with pytest.raises(AssertionError):
        for panel, code in zip(_panels(fig)[1:], (1, 0), strict=True):
            product = pi[code] * retention[code, :, 1]
            share = float(np.mean(product[arm == code] < floor))
            assert _tail_texts(panel) == [f"{_percent(share)} below"]


def test_continuous_overlap_checks_its_keywords(floored: Any) -> None:
    with pytest.raises(ValueError, match="pass censoring=True"):
        plotting.overlap(floored, horizon=1.0)
    with pytest.raises(DataError, match=r"no target time 1\.5"):
        plotting.overlap(floored, censoring=True, horizon=1.5)
    for keyword in ({"regimen": "treated"}, {"nodes": [1]}, {"cause": "death"}):
        with pytest.raises(ValueError, match="does not apply"):
            plotting.overlap(floored, **keyword)


# --------------------------------------------------------------------------- B6


def _peaks(ax: Any) -> list[np.ndarray]:
    """Each series' largest weight per target time, from its crosses."""
    return [
        np.asarray(collection.get_offsets())[:, 1]
        for collection in ax.collections
        if collection.get_gid() == "cleverly-weight-max"
    ]


def _check_weights(ax: Any, result: Any, *, counterfactual: bool = False) -> None:
    pi, retention, arm, floor = _arrays(result)
    peaks = _peaks(ax)
    assert len(peaks) == 2
    shares = []
    for found, code in zip(peaks, (0, 1), strict=True):
        rows = arm != code if counterfactual else arm == code
        product = pi[code, rows, None] * retention[code, rows, :]
        np.testing.assert_allclose(found, (1.0 / product).max(axis=0))
        shares.append([_drawn_percent(float(np.mean(product[:, k] < floor))) for k in range(2)])
    assert _tail_texts(ax) == [text for series in shares for text in series]


def _drawn_percent(share: float) -> str:
    from cleverly.plotting._weights import _percent_text

    return _percent_text(share)


def test_weights_draw_each_arms_unbounded_weight_with_the_floor(
    floored: Any, drawn: list[Figure]
) -> None:
    pi, retention, arm, floor = _arrays(floored)
    product = pi[:, :, None] * retention
    received = product[arm, np.arange(arm.size)]
    # Witness: some received weights exceed the largest weight the floor allows.
    assert (1.0 / received).max() > 2.0 / floor
    fig = _draw(drawn, plotting.weights, floored)
    (top,) = fig.axes
    _check_weights(top, floored)
    limits = {
        round(float(line.get_ydata()[0]), 9)
        for line in top.get_lines()
        if line.get_gid() == "cleverly-marker"
    }
    assert limits == {round(1.0 / floor, 9)}
    assert [tick.get_text() for tick in top.get_xticklabels()] == ["1", "2"]
    notes = plotting.notes(fig)
    assert any("given that arm, followed to that time or not" in note for note in notes)
    assert any("1 / (pi(a | L) S_c(t- | a, L))" in note for note in notes)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda pi, retention, floor: retention[::-1],
        lambda pi, retention, floor: retention[..., ::-1],
        lambda pi, retention, floor: np.maximum(pi[:, :, None] * retention, floor) / pi[:, :, None],
    ],
    ids=["other-arm", "other-time", "floored"],
)
def test_reading_the_wrong_weights_fails_the_weights_check(
    floored: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch, mutation: Any
) -> None:
    """Mutation control: the other arm's axis, another time's column, or the floored product."""
    pi, retention, _, floor = _arrays(floored)
    wrong = mutation(pi, retention, floor)
    monkeypatch.setattr(_continuous, "retention_before", lambda record: wrong)
    fig = _draw(drawn, plotting.weights, floored)
    with pytest.raises(AssertionError):
        _check_weights(fig.axes[0], floored)


def test_counterfactual_weights_draw_each_arm_over_the_other_arms_units(
    floored: Any, drawn: list[Figure]
) -> None:
    fig = _draw(drawn, plotting.weights, floored, counterfactual=True)
    _check_weights(fig.axes[0], floored, counterfactual=True)
    with pytest.raises(AssertionError):
        _check_weights(fig.axes[0], floored)
    assert any("the arm it did not receive" in text for text in _texts(fig))
    assert any("received the other arm" in note for note in plotting.notes(fig))


def test_continuous_weights_choose_arms_and_refuse_other_keywords(
    floored: Any, drawn: list[Figure]
) -> None:
    fig = _draw(drawn, plotting.weights, floored, regimens=["treated"])
    assert len(_peaks(fig.axes[0])) == 1
    with pytest.raises(DataError, match="does not hold"):
        plotting.weights(floored, regimens=["placebo"])
    for keyword in ({"horizon": 1}, {"cause": "death"}, {"repeat": 2}):
        with pytest.raises(ValueError, match="does not apply"):
            plotting.weights(floored, **keyword)


def test_a_longitudinal_fit_refuses_counterfactual_weights() -> None:
    frame, _ = make_longitudinal(n=200, seed=0)
    result = _quiet(
        lambda: LTMLE(
            {"always": 1, "never": 0},
            n_folds=1,
            random_state=0,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            censoring_learner=LogisticRegression(max_iter=1000),
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    )
    with pytest.raises(ValueError, match="counterfactual="):
        plotting.weights(result, counterfactual=True)


# --------------------------------------------------------------------------- C3


def _selection_panels(fig: Figure) -> list[Any]:
    return [ax for ax in fig.axes if ax.get_yticklabels()]


def _check_selections(fig: Figure, rows: list[tuple[str, Any]]) -> None:
    """Each panel draws one selection's losses and marks the candidate it chose."""
    panels = _selection_panels(fig)
    assert [ax.get_title(loc="left") for ax in panels] == [title for title, _ in rows]
    for ax, (_, selection) in zip(panels, rows, strict=True):
        losses = np.asarray(selection.losses, dtype=float)
        chosen = int(selection.chosen)
        dots = {
            collection.get_gid(): np.asarray(collection.get_offsets())
            for collection in ax.collections
        }
        marked = dots["cleverly-learner-chosen"]
        np.testing.assert_allclose(marked, [[losses[chosen], chosen]])
        others = [index for index in range(losses.size) if index != chosen]
        np.testing.assert_allclose(dots["cleverly-learner-loss"][:, 0], losses[others])
        ticks = [tick.get_text() for tick in ax.get_yticklabels()]
        assert ticks[chosen] == f"{selection.names[chosen]} (chosen)"


def _record_rows(result: Any) -> list[tuple[str, Any]]:
    record = result.record
    rows = [
        (f"Hazard of cause {cause}", hazard)
        for cause, hazard in zip(result.causes, record.hazards, strict=True)
    ]
    rows.append(("Censoring hazard", record.censoring))
    if record.propensity is not None:
        rows.append(("Treatment model", record.propensity))
    return rows


def test_learners_draw_each_selection_with_the_chosen_candidate_marked(
    floored: Any, drawn: list[Figure]
) -> None:
    fig = _draw(drawn, plotting.learners, floored)
    _check_selections(fig, _record_rows(floored))
    texts = _texts(fig)
    assert any("First fold's selection" in text for text in texts)
    assert not any("weight" in text.lower() for text in texts)
    assert any("first fold's selections" in note for note in plotting.notes(fig))
    # The fit's nuisance report draws the same figure.
    report = _draw(drawn, plotting.learners, floored.diagnostics.nuisance_models())
    _check_selections(report, _record_rows(floored))


def test_a_supplied_treatment_learner_has_no_panel(supplied: Any, drawn: list[Figure]) -> None:
    assert supplied.record.propensity is None
    fig = _draw(drawn, plotting.learners, supplied)
    rows = _record_rows(supplied)
    assert len(rows) == 3
    _check_selections(fig, rows)
    assert any("supplied treatment_learner" in note for note in plotting.notes(fig))


def test_a_full_sample_fit_draws_its_selection_and_an_unscored_library(
    full_sample: Any, drawn: list[Figure]
) -> None:
    assert not full_sample.record.cross_fitted
    fig = _draw(drawn, plotting.learners, full_sample)
    assert not any("First fold" in text for text in _texts(fig))
    panels = _selection_panels(fig)
    assert len(panels) == 4
    for ax in panels[:2]:
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "CoxHazard('main') (chosen, not cross-validated)"
        ]
        assert not ax.collections or all(
            not len(collection.get_offsets()) for collection in ax.collections
        )
    for ax, (_, selection) in zip(panels[2:], _record_rows(full_sample)[2:], strict=True):
        chosen = int(selection.chosen)
        (marked,) = [c for c in ax.collections if c.get_gid() == "cleverly-learner-chosen"]
        assert np.asarray(marked.get_offsets())[0, 1] == chosen
    assert any("not cross-validated" in note for note in plotting.notes(fig))


def _rechosen(result: Any) -> Any:
    """The fit with its first hazard's chosen candidate moved off the lowest loss."""
    record = result.record
    first = record.hazards[0]
    other = int(np.argmax(first.losses))
    hazards = (dataclasses.replace(first, chosen=other), *record.hazards[1:])
    return dataclasses.replace(result, record=dataclasses.replace(record, hazards=hazards))


def test_marking_the_lowest_loss_fails_the_selection_check(
    floored: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: the figure marks ``chosen``, not the lowest loss."""
    moved = _rechosen(floored)
    first = moved.record.hazards[0]
    # Witness: the recorded choice is not the lowest loss.
    assert first.chosen != int(np.argmin(first.losses))
    _check_selections(_draw(drawn, plotting.learners, moved), _record_rows(moved))
    monkeypatch.setattr(
        _continuous, "chosen_index", lambda row: int(np.nanargmin(np.asarray(row.losses)))
    )
    fig = _draw(drawn, plotting.learners, moved)
    with pytest.raises(AssertionError):
        _check_selections(fig, _record_rows(moved))


def test_continuous_learners_refuse_selectors_and_calibration_still_refuses(
    floored: Any,
) -> None:
    with pytest.raises(ValueError, match="longitudinal"):
        plotting.learners(floored, regimen="treated")
    _, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="ax= is not accepted"):
            plotting.learners(floored, ax=ax)
    finally:
        plt.close(ax.figure)
    with pytest.raises(CapabilityError, match="keeps no nuisance report"):
        plotting.calibration(floored)


def test_overlap_refuses_censoring_panels_on_a_fit_with_no_censored_row() -> None:
    frame = _frame()
    # Every censored row becomes an event of cause 1 at its time.
    frame["event"] = np.where(frame["event"] == 0, 1, frame["event"])
    estimator = ContinuousSurvivalTMLE(
        horizons=(1.0, 2.0),
        arms={"treated": 1, "control": 0},
        reference="control",
        causes={1: "relapse", 2: "death"},
        cross_fit=False,
        random_state=0,
    )
    result = _quiet(
        lambda: estimator.fit(
            frame, time="time", event="event", treatment="treated", baseline=["x"]
        )
    )
    # Witness: the stored retention is 1 everywhere, so the product panels would repeat pi.
    assert np.all(result.record.retention_before == 1.0)
    with pytest.raises(ValueError, match="no censored row"):
        plotting.overlap(result, censoring=True)
    plt.close(plotting.overlap(result))


def _duplicated(result: Any) -> Any:
    """The fit with its first hazard library listing one name twice, the second chosen."""
    record = result.record
    first = record.hazards[0]
    name = first.names[0]
    twice = dataclasses.replace(
        first, names=(name, name, *first.names[2:]), losses=first.losses, chosen=1
    )
    hazards = (twice, *record.hazards[1:])
    return dataclasses.replace(result, record=dataclasses.replace(record, hazards=hazards))


def test_learners_mark_the_recorded_position_when_a_library_repeats_a_name(
    floored: Any, drawn: list[Figure]
) -> None:
    twice = _duplicated(floored)
    fig = _draw(drawn, plotting.learners, twice)
    first = _selection_panels(fig)[0]
    (marked,) = [c for c in first.collections if c.get_gid() == "cleverly-learner-chosen"]
    assert np.asarray(marked.get_offsets())[0, 1] == 1
    # The report keeps the chosen name only, which now names two candidates.
    with pytest.raises(ValueError, match="lists more than once"):
        plotting.learners(twice.diagnostics.nuisance_models())

"""X92 PR-5: the targeting-curve plot (V7).

The tests read artists, never pixels. Each honesty rule has a mutation control, a deliberately
wrong plot that the same check must reject: the random-targeting reference at the wrong
convention, and the gain view drawing the interval of the gain instead of the TOC interval
scaled by ``q``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import targeting_curves as tc
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.plotting import _core, _targeting, targeting
from tests.unit.test_plotting_core import assert_no_note_text

pytestmark = pytest.mark.xdist_group("plotting_core")


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _fit(n: int) -> tuple[Any, Any, np.ndarray]:
    rng = np.random.default_rng(3)
    w = rng.uniform(-1, 1, (n, 2))
    a = rng.binomial(1, expit(w[:, 0]))
    y = w[:, 1] + a * (1 + w[:, 0]) + 0.3 * rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = (
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=LogisticRegression(),
                outcome_by_arm=True,
                estimands=("ate",),
                reference=0.0,
                q_bounds=(-5.0, 7.0),
                random_state=0,
            )
            .fit(data)
            .single()
        )
    return fit, data, w


@pytest.fixture(scope="module")
def curve() -> Any:
    fit, data, w = _fit(1000)
    scores = {"S": w[:, 0], "R": w[:, 0] + w[:, 1]}
    return tc.targeting_fit(fit, data, scores, pair=(1.0, 0.0), n_half_samples=200)


@pytest.fixture(scope="module")
def small() -> Any:
    """Three hundred rows: q = 0.1 holds 30 top rows, so its interval is withheld."""
    fit, data, w = _fit(300)
    return tc.targeting_fit(fit, data, {"S": w[:, 0]}, pair=(1.0, 0.0), n_half_samples=50)


def _by_gid(ax: Any, gid: str) -> list[Any]:
    return [a for a in ax.get_children() if a.get_gid() == gid]


def _reference_is_random_targeting(ax: Any, view: str, ate: float) -> bool:
    lines = [line for line in ax.get_lines() if line.get_gid() == "cleverly-reference"]
    if len(lines) != 1:
        return False
    x = np.asarray(lines[0].get_xdata(), dtype=float)
    y = np.asarray(lines[0].get_ydata(), dtype=float)
    expected = np.zeros_like(y) if view == "toc" else x * ate
    if view == "toc":
        # An axhline spans axes coordinates in x; its y data is the level.
        return bool(np.allclose(y, 0.0))
    return bool(np.allclose(y, expected) and x.min() == 0.0 and x.max() == 1.0)


def _segments(ax: Any, gid: str) -> list[np.ndarray]:
    out = []
    for collection in _by_gid(ax, gid):
        out.extend(np.asarray(path) for path in collection.get_segments())
    return out


def _band_marks(ax: Any) -> list[Any]:
    """Every artist that draws the band, a line or a collection."""
    return _by_gid(ax, _core.GID_BAND)


def _gain_view_is_scaled_toc(ax: Any, curve: Any) -> bool:
    toc = {row["q"]: row for row in curve.rows("toc")}
    gain = {row["q"]: row["estimate"] for row in curve.rows("gain")}
    drawn = _segments(ax, "cleverly-interval")
    if not drawn:
        return False
    for segment in drawn:
        q = float(segment[0, 0])
        row = toc[min(toc, key=lambda point: abs(point - q))]
        low = gain[row["q"]] - q * (row["estimate"] - row["lower"])
        high = gain[row["q"]] + q * (row["upper"] - row["estimate"])
        if not np.allclose(sorted(segment[:, 1]), [low, high], rtol=1e-12):
            return False
    return True


@pytest.mark.parametrize("view", ["toc", "gain"])
def test_the_reference_is_random_targeting(curve: Any, view: str) -> None:
    ax = targeting(curve, view=view).axes[0]
    assert _reference_is_random_targeting(ax, view, curve.ate)


def test_a_reference_at_the_wrong_convention_fails(
    curve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: random targeting drawn at the ATE level, not along q * ATE."""

    def level(ax: Any, ate: float) -> None:
        _core.reference_line(ax, ate, label="Treating at random")

    monkeypatch.setattr(_targeting, "_random_line", level)
    ax = targeting(curve, view="gain").axes[0]
    assert not _reference_is_random_targeting(ax, "gain", curve.ate)


def test_a_toc_reference_off_zero_fails(curve: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: the TOC view's random-targeting line drawn at the ATE, not at 0."""
    original = _core.reference_line

    def at_ate(ax: Any, value: float, **kwargs: Any) -> Any:
        return original(ax, curve.ate, **kwargs)

    monkeypatch.setattr(_core, "reference_line", at_ate)
    ax = targeting(curve, view="toc").axes[0]
    assert not _reference_is_random_targeting(ax, "toc", curve.ate)


def test_band_marks_sit_at_grid_points_only(curve: Any) -> None:
    ax = targeting(curve, view="toc").axes[0]
    rows = {row["q"]: row for row in curve.rows("toc")}
    marks = [
        (float(x), float(y))
        for line in ax.get_lines()
        if line.get_gid() == _core.GID_BAND
        for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
    ]
    assert marks
    for x, y in marks:
        q = min(rows, key=lambda point: abs(point - x))
        assert abs(x - q) < 1e-9 and q < 1.0
        assert round(y, 12) in {
            round(rows[q]["band_lower"], 12),
            round(rows[q]["band_upper"], 12),
        }


def test_the_gain_view_draws_the_scaled_toc_interval(curve: Any) -> None:
    ax = targeting(curve, view="gain").axes[0]
    assert _gain_view_is_scaled_toc(ax, curve)
    assert "gap to the random line" in _core.note_text(ax.figure)


def test_drawing_the_gain_interval_instead_fails(
    curve: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: the gain's own interval, which also carries the ATE's uncertainty."""

    def own(target: Any, name: str, rows: Any) -> tuple[list[float], ...]:
        gains = target.rows("gain", name)
        return (
            [row["estimate"] for row in gains],
            [row["lower"] for row in gains],
            [row["upper"] for row in gains],
        )

    monkeypatch.setattr(_targeting, "_gain_interval", own)
    ax = targeting(curve, view="gain").axes[0]
    assert not _gain_view_is_scaled_toc(ax, curve)


def test_the_difference_view_has_no_band(curve: Any) -> None:
    ax = targeting(curve, view="difference").axes[0]
    assert not _band_marks(ax)
    assert _segments(ax, "cleverly-interval")
    rows = curve.rows("toc", "S - R")
    points = [line for line in ax.get_lines() if line.get_gid() == "cleverly-point"]
    assert len(points) == len(rows)


def test_withheld_rows_draw_hollow_points_and_say_why(small: Any) -> None:
    ax = targeting(small, view="toc").axes[0]
    points = [line for line in ax.get_lines() if line.get_gid() == "cleverly-point"]
    first = points[0]
    assert first.get_markerfacecolor() == _core.FACE
    assert math.isnan(small.rows("toc")[0]["std_error"])
    intervals = _segments(ax, "cleverly-interval")
    assert all(abs(float(segment[0, 0]) - 0.1) > 1e-9 for segment in intervals)
    text = _core.note_text(ax.figure)
    assert "Hollow points have no interval: fewer than 50 rows" in text
    assert "Experimental:" in text
    assert not _band_marks(ax)
    assert_no_note_text(ax.figure)


def test_the_views_and_scores_it_refuses(small: Any) -> None:
    with pytest.raises(ValueError, match="view"):
        targeting(small, view="qini")
    with pytest.raises(ValueError, match="compare"):
        targeting(small, view="difference")
    with pytest.raises(ValueError, match="score"):
        targeting(small, score="nope")


def test_plot_draws_on_the_callers_axes(curve: Any) -> None:
    fig, ax = plt.subplots()
    assert curve.plot(view="toc", ax=ax) is fig
    assert _core.subtitle_text(ax).startswith("AUTOC ")


def test_the_labels_name_the_outcome_and_the_score(curve: Any) -> None:
    words = {curve.score: "the predicted effect"}
    gain = targeting(curve, view="gain", outcome_label="Order share", labels=words).axes[0]
    assert gain.get_ylabel() == "Change in order share from treating the top fraction"
    toc = targeting(curve, outcome_label="Order share", labels=words).axes[0]
    assert toc.get_ylabel() == "Effect on order share in the top fraction minus the average"
    title = toc.get_title(loc="left").replace("\n", " ")
    assert "treating by the predicted effect first" in title
    assert f"by {curve.score} first" not in title

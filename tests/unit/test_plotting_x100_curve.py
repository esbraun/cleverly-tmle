"""The curve of a continuous-time survival fit draws its target times only (X100 part A).

Each figure is read back from its artists and compared with ``result.curve()``,
``result.rmst()`` and ``result.rmtl()``. Each honesty rule has a mutation control: a
deliberately wrong reader or drawer, patched in, must make the same check fail, and a witness
asserts first that the fixture can tell the two apart.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import cleverly.plotting as plotting
from cleverly.exceptions import CapabilityError, DataError
from cleverly.plotting import _continuous_curve, _core
from cleverly.survival.cox import CoxHazard
from cleverly.survival.engine import ContinuousSurvivalTMLE
from tests.unit.test_plotting_x100 import _draw, _quiet
from tests.unit.test_plotting_x100_continuous import _frame

pytestmark = pytest.mark.xdist_group("plotting_x100_curve")

HORIZONS = (0.5, 1.0, 2.0)


def _fit(frame: pd.DataFrame, **settings: Any) -> Any:
    estimator = ContinuousSurvivalTMLE(
        horizons=HORIZONS,
        arms={"treated": 1, "control": 0},
        reference="control",
        cross_fit=False,
        hazard_library=(CoxHazard("main"),),
        censoring_library=(CoxHazard("main"),),
        random_state=0,
        **settings,
    )
    return _quiet(
        lambda: estimator.fit(
            frame, time="time", event="event", treatment="treated", baseline=["x"]
        )
    )


@pytest.fixture(scope="module")
def competing() -> Any:
    """Two causes, a multiplier band, and restricted means at two horizons."""
    return _fit(
        _frame(),
        causes={1: "relapse", 2: "death"},
        simultaneous=True,
        restricted_horizons=(1.0, 2.0),
    )


@pytest.fixture(scope="module")
def one_event() -> Any:
    """One event type with a multiplier band and no restricted horizon."""
    frame = _frame()
    frame["event"] = (frame["event"] > 0).astype(int)
    return _fit(frame, simultaneous=True)


@pytest.fixture
def drawn() -> Iterator[list[Figure]]:
    figures: list[Figure] = []
    yield figures
    for fig in figures:
        plt.close(fig)


# --------------------------------------------------------------------------- readers


def _series(ax: Any) -> list[dict[str, Any]]:
    """Each series on a panel, in drawing order: points, interval bars and band brackets."""
    found: dict[str, dict[str, Any]] = {}
    for line in ax.get_lines():
        gid = line.get_gid()
        if gid not in (_core.GID_POINT, _core.GID_BAND):
            continue
        entry = found.setdefault(matplotlib.colors.to_hex(line.get_color()), {})
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        if gid == _core.GID_POINT:
            entry.setdefault("x", []).extend(xs)
            entry.setdefault("psi", []).extend(ys)
        else:
            half = xs.size // 2
            entry["band"] = (xs[:half], ys[:half], ys[half:])
    for collection in ax.collections:
        if collection.get_gid() != _core.GID_INTERVAL:
            continue
        color = matplotlib.colors.to_hex(collection.get_colors()[0])
        segments = collection.get_segments()
        found[color]["interval"] = (
            np.asarray([s[0][0] for s in segments]),
            np.asarray([s[0][1] for s in segments]),
            np.asarray([s[1][1] for s in segments]),
        )
    return list(found.values())


def _expected(result: Any, view: str, cause: str | None, arm: str) -> pd.DataFrame:
    frame = result.curve(scale=view)
    rows = frame[frame["arm"] == arm]
    if cause is not None:
        rows = rows[rows["cause"] == cause]
    return rows.sort_values("time")


def _check_panel(ax: Any, result: Any, view: str, cause: str | None, arms: list[str]) -> None:
    """Each series is one arm's (or the difference's) rows of ``result.curve(view)``."""
    series = _series(ax)
    assert len(series) == len(arms)
    for found, arm in zip(series, arms, strict=True):
        rows = _expected(result, view, cause, arm)
        np.testing.assert_allclose(found["x"], rows["time"])
        np.testing.assert_allclose(found["psi"], rows["psi"])
        xs, low, high = found["interval"]
        np.testing.assert_allclose(xs, rows["time"])
        np.testing.assert_allclose(low, rows["ci_lower"])
        np.testing.assert_allclose(high, rows["ci_upper"])
        bx, blow, bhigh = found["band"]
        np.testing.assert_allclose(bx, rows["time"])
        np.testing.assert_allclose(blow, rows["band_lower"])
        np.testing.assert_allclose(bhigh, rows["band_upper"])


def _check_target_times_only(fig: Figure) -> None:
    """Every drawn x is a target time, and no line joins two of them."""
    for ax in fig.axes:
        for line in ax.get_lines():
            if line.get_gid() == _core.GID_REFERENCE:
                continue
            xs = set(np.asarray(line.get_xdata(), dtype=float).tolist())
            assert xs <= set(HORIZONS), f"{line.get_gid()} draws at {sorted(xs)}"
            if len(xs) > 1:
                assert line.get_linestyle() in ("None", "none", ""), (
                    f"{line.get_gid()} joins target times"
                )
        for collection in ax.collections:
            for segment in getattr(collection, "get_segments", list)():
                assert {float(point[0]) for point in segment} <= set(HORIZONS)
                assert len({float(point[0]) for point in segment}) == 1


# --------------------------------------------------------------------------- part A


def test_each_cause_draws_the_fits_risks_intervals_and_band(
    competing: Any, drawn: list[Figure]
) -> None:
    fig = _draw(drawn, plotting.curve, competing)
    assert len(fig.axes) == 4
    for j, cause in enumerate(competing.causes):
        assert fig.axes[j].get_title(loc="left") == f"Cause {cause}"
        _check_panel(fig.axes[j], competing, "risk", cause, ["control", "treated"])
        _check_panel(fig.axes[2 + j], competing, "risk", cause, ["treated vs control"])
    _check_target_times_only(fig)
    notes = plotting.notes(fig)
    assert any("nothing is drawn between them" in note for note in notes)
    assert any(note.startswith("Support by time: 0.5 experimental") for note in notes)
    assert any("multiplier band" in note for note in notes)


def test_a_connecting_line_fails_the_target_time_check(
    competing: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a drawer that joins the target times fails the honesty check."""
    original = _continuous_curve._draw_line

    def joined(ax: Any, points: Any, **kwargs: Any) -> None:
        original(ax, points, **kwargs)
        ax.plot([p.x for p in points], [p.psi for p in points], color=kwargs["color"])

    monkeypatch.setattr(_continuous_curve, "_draw_line", joined)
    fig = _draw(drawn, plotting.curve, competing)
    with pytest.raises(AssertionError, match="joins target times"):
        _check_target_times_only(fig)


def test_swapping_the_band_and_the_interval_fails_the_panel_check(
    competing: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: band limits drawn as the interval, and the interval as the band."""
    frame = competing.curve()
    # Witness: the band is wider than the pointwise interval.
    assert (frame["band_upper"] - frame["band_lower"] > frame["ci_upper"] - frame["ci_lower"]).all()

    def swapped(result: Any, view: str) -> Any:
        out = result.curve(scale=view).copy()
        out[["ci_lower", "ci_upper", "band_lower", "band_upper"]] = out[
            ["band_lower", "band_upper", "ci_lower", "ci_upper"]
        ].to_numpy()
        return out

    monkeypatch.setattr(_continuous_curve, "curve_frame", swapped)
    fig = _draw(drawn, plotting.curve, competing)
    with pytest.raises(AssertionError):
        _check_panel(fig.axes[0], competing, "risk", "relapse", ["control", "treated"])


def test_reading_the_other_arm_fails_the_panel_check(
    competing: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: each arm's label on the other arm's rows."""
    frame = competing.curve()
    control = frame[(frame["arm"] == "control") & (frame["cause"] == "relapse")]["psi"].to_numpy()
    treated = frame[(frame["arm"] == "treated") & (frame["cause"] == "relapse")]["psi"].to_numpy()
    assert np.abs(control - treated).max() > 0.05

    def other_arm(result: Any, view: str) -> Any:
        out = result.curve(scale=view).copy()
        out["arm"] = out["arm"].replace({"control": "treated", "treated": "control"})
        return out

    monkeypatch.setattr(_continuous_curve, "curve_frame", other_arm)
    fig = _draw(drawn, plotting.curve, competing)
    with pytest.raises(AssertionError):
        _check_panel(fig.axes[0], competing, "risk", "relapse", ["control", "treated"])


def test_the_survival_scale_draws_the_fits_survival_rows(
    one_event: Any, drawn: list[Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    fig = _draw(drawn, plotting.curve, one_event, scale="survival")
    assert len(fig.axes) == 2
    _check_panel(fig.axes[0], one_event, "survival", None, ["control", "treated"])
    _check_panel(fig.axes[1], one_event, "survival", None, ["treated vs control"])
    _check_target_times_only(fig)
    assert fig.axes[0].get_title(loc="left") == "Survival at the target times"
    assert any("mirrored" in note for note in plotting.notes(fig))
    # Mutation control: the risk scale's rows in place of the survival scale's.
    risk, survival = one_event.curve(), one_event.curve(scale="survival")
    assert np.abs(risk["psi"] - survival["psi"]).max() > 0.1
    monkeypatch.setattr(
        _continuous_curve, "curve_frame", lambda result, view: result.curve(scale="risk")
    )
    wrong = _draw(drawn, plotting.curve, one_event, scale="survival")
    with pytest.raises(AssertionError):
        _check_panel(wrong.axes[0], one_event, "survival", None, ["control", "treated"])


def test_the_fits_refusals_hold(competing: Any, one_event: Any) -> None:
    with pytest.raises(CapabilityError, match="not event-free survival"):
        plotting.curve(competing, scale="survival")
    with pytest.raises(CapabilityError, match="restricted_horizons"):
        plotting.curve(one_event, measure="rmst")
    with pytest.raises(ValueError, match="no bootstrap band"):
        plotting.curve(one_event, band="bootstrap")
    with pytest.raises(DataError, match="cause 'stroke'"):
        plotting.curve(competing, cause="stroke")
    with pytest.raises(ValueError, match="cause= does not apply"):
        plotting.curve(competing, measure="rmst", cause="death")


def test_one_cause_and_no_band(competing: Any, drawn: list[Figure]) -> None:
    fig = _draw(drawn, plotting.curve, competing, cause="death", band="none")
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert not any(line.get_gid() == _core.GID_BAND for line in ax.get_lines())
    assert any("band='none'" in note for note in plotting.notes(fig))
    assert competing.plot(cause="death").axes[0].get_title(loc="left") == (
        "Risk at the target times"
    )
    plt.close("all")


@pytest.mark.parametrize("measure", ["rmst", "rmtl"])
def test_restricted_means_draw_the_fits_estimates_at_the_restricted_horizons(
    competing: Any, drawn: list[Figure], measure: str
) -> None:
    fig = _draw(drawn, plotting.curve, competing, measure=measure)
    causes: list[str | None] = [None] if measure == "rmst" else list(competing.causes)
    assert len(fig.axes) == 2 * len(causes)
    taus = competing.restricted_horizons
    for j, cause in enumerate(causes):

        def read(arm: str, tau: float, versus: str | None = None, cause: Any = cause) -> Any:
            if measure == "rmst":
                return competing.rmst(arm, tau, versus=versus)
            return competing.rmtl(arm, tau, cause, versus=versus)

        expected = [[read(arm, tau) for tau in taus] for arm in competing.arms]
        expected.append([read("treated", tau, "control") for tau in taus])
        series = _series(fig.axes[j]) + _series(fig.axes[len(causes) + j])
        assert len(series) == 3
        for found, estimates in zip(series, expected, strict=True):
            np.testing.assert_allclose(found["x"], taus)
            np.testing.assert_allclose(found["psi"], [e.psi for e in estimates])
            np.testing.assert_allclose(found["interval"][1], [e.ci[0] for e in estimates])
            np.testing.assert_allclose(found["interval"][2], [e.ci[1] for e in estimates])
            assert "band" not in found
    assert any("No simultaneous band covers a restricted mean" in n for n in plotting.notes(fig))


def test_truth_is_drawn_at_the_target_times(one_event: Any, drawn: list[Figure]) -> None:
    name = "risk_regimen[treated @ t=1]"
    fig = _draw(drawn, plotting.curve, one_event, truth={name: 0.2})
    crosses = [
        line for ax in fig.axes for line in ax.get_lines() if line.get_gid() == _core.GID_TRUTH
    ]
    assert [float(np.asarray(c.get_xdata())[0]) for c in crosses] == [1.0]
    assert [float(np.asarray(c.get_ydata())[0]) for c in crosses] == [0.2]

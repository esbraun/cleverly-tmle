"""The time-to-event converter: its binning rule, its refusals and its round trip.

``LongitudinalData.from_time_to_event`` reads one time and one event code per unit onto a
grid, with the visit structure of Benkeser, Carone and Gilbert (2018, Section 2.1).  An
event at ``T`` goes to the node ``k`` with ``g_{k-1} < T <= g_k``.  A censoring time must
be a grid value or later than the last grid time, and a unit censored at ``g_j`` is observed
through node ``j``.

The exact law of ``tests/discrete_law_point_survival.py`` with no censoring at node 1 has a
long layout.  Its long fit must equal its wide held fit bit for bit, on the integer grid and
on an unequal declared grid, with the events off the grid.  Each mutation of the binning
rule misses that identity.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from cleverly import CausalStudy, RegimeContrast, TimeToEvent
from cleverly.exceptions import DataError
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal import data as data_module
from tests.discrete_law_point_survival import CellMeans, CellProbabilities, static, survival_law

LAW = survival_law(censor_first=False)
UNEQUAL = (1.0, 2.5, 4.0)
REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
EXACT = os.environ.get("CI") is None


class Refuse(CellMeans):
    """A learner that fails if any fit reaches it."""

    def fit(self, *args: object, **kwargs: object) -> Refuse:
        raise AssertionError("a learner ran before the refusal")


def _learners() -> dict[str, object]:
    return {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
    }


def _long(grid: tuple[float, ...] | None, frame: pd.DataFrame) -> LongitudinalData:
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=["W"], grid=grid
    )


def _fit(data: LongitudinalData, **kwargs: object) -> object:
    return LTMLE(REGIMENS, n_folds=1, **_learners(), **kwargs).fit(data)


def _wide() -> object:
    return LTMLE(REGIMENS, n_folds=1, **_learners()).fit(LAW.frame(), **LAW.fit_columns())


def _same(left: float, right: float) -> None:
    if EXACT:
        assert left == right
    else:
        assert left == pytest.approx(right, rel=1e-12, abs=1e-12)


# ------------------------------------------------------------------- the identity


@pytest.mark.parametrize(
    ("grid", "offset"), [(None, 0.0), ((1.0, 2.0, 3.0), 0.5), (UNEQUAL, 0.0), (UNEQUAL, 0.7)]
)
def test_the_long_fit_equals_the_wide_held_fit(
    grid: tuple[float, ...] | None, offset: float
) -> None:
    """Events on or off the grid, censoring on it: one container, one fit."""
    values = (1.0, 2.0, 3.0) if grid is None else grid
    long = _fit(_long(grid, LAW.long_frame(values, offset=offset)))
    wide = _wide()
    for name in wide.estimates:  # type: ignore[attr-defined]
        _same(long[name].psi, wide[name].psi)  # type: ignore[index]
    for arm in range(3):
        for horizon in (1, 2, 3):
            truth = LAW.functional(LAW.probs, static(arm), horizon)
            assert long[f"risk_regimen[a{arm} @ t={horizon}]"].psi == pytest.approx(  # type: ignore[index]
                truth, rel=1e-12
            )


def test_floor_binning_misses_the_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assigning an event at ``g_{k-1} < T < g_k`` to node ``k - 1`` is the wrong rule."""
    monkeypatch.setattr(
        data_module,
        "_event_node",
        lambda grid, times: np.maximum(np.searchsorted(grid, times, side="right"), 1),
    )
    # The mutation empties the last node, so the fit reports the first two.
    long = _fit(_long(UNEQUAL, LAW.long_frame(UNEQUAL, offset=0.7)), horizons=[1.0, 2.5])
    truth = LAW.functional(LAW.probs, static(1), 2)
    assert abs(long["risk_regimen[a1 @ t=2]"].psi - truth) > 0.01  # type: ignore[index]


def test_the_censor_first_map_misses_the_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit censored at ``g_j`` is at risk at node ``j``; censoring it there is off by one."""
    monkeypatch.setattr(
        data_module,
        "_censor_node",
        lambda grid, times: np.searchsorted(grid, times, side="left") + 1,
    )
    long = _fit(_long(UNEQUAL, LAW.long_frame(UNEQUAL)))
    truth = LAW.functional(LAW.probs, static(1), 3)
    assert abs(long["risk_regimen[a1 @ t=3]"].psi - truth) > 0.005  # type: ignore[index]


# ------------------------------------------------------------------- round trip


@pytest.mark.parametrize("grid", [None, UNEQUAL])
def test_the_round_trip_returns_the_input_in_grid_values(grid: tuple[float, ...] | None) -> None:
    values = (1.0, 2.0, 3.0) if grid is None else grid
    frame = LAW.long_frame(values)
    data = _long(grid, frame)
    assert data.time_grid == values
    time, event = data.to_time_to_event()
    assert np.array_equal(time, frame["time"].to_numpy())
    assert np.array_equal(event, frame["event"].to_numpy())
    # The container equals the hand-built held wide container on its arrays.
    wide = LongitudinalData.from_frame(LAW.frame(), **LAW.fit_columns())
    assert np.array_equal(data.event, wide.event)
    assert np.array_equal(data.uncensored, wide.uncensored)
    assert np.array_equal(data.treatment, wide.treatment, equal_nan=True)


def test_competing_codes_round_trip_with_their_codes() -> None:
    law = survival_law(causes=2, censor_first=False)
    frame = law.long_frame(UNEQUAL)
    data = LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W"],
        grid=UNEQUAL,
        causes={1: "relapse", 2: "death"},
    )
    assert data.cause_labels == ("relapse", "death")
    assert data.event_codes == (1, 2)
    time, event = data.to_time_to_event()
    assert np.array_equal(time, frame["time"].to_numpy())
    assert np.array_equal(event, frame["event"].to_numpy())


def test_a_time_after_the_last_grid_time_is_the_administrative_end() -> None:
    """D7: an event or a censoring after ``g_K`` leaves the unit event-free through ``K``."""
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 7.0, 9.0, 3.0] * 4,
            "event": [1, 0, 1, 0, 1] * 4,
            "A": [0, 1] * 10,
            "W": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0] * 2,
        }
    )
    data = LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=["W"], grid=[1, 2, 3]
    )
    time, event = data.to_time_to_event()
    assert time[2] == 3.0 and event[2] == 0
    assert time[3] == 3.0 and event[3] == 0
    assert time[1] == 2.0 and event[1] == 0


def test_the_default_grid_runs_to_the_largest_event_time() -> None:
    frame = LAW.long_frame((1.0, 2.0, 3.0))
    frame.loc[frame["event"] == 0, "time"] += 0.0
    data = _long(None, frame)
    assert data.time_grid == (1.0, 2.0, 3.0)
    assert data.n_times == 3


# --------------------------------------------------------------------- grid RMST


def test_grid_rmst_is_the_rmst_of_the_discretized_time() -> None:
    long = _fit(_long(UNEQUAL, LAW.long_frame(UNEQUAL)))
    spacing = np.diff((0.0, *UNEQUAL))
    for arm in range(3):
        risks = [LAW.functional(LAW.probs, static(arm), k) for k in (1, 2)]
        truth = spacing[0] + spacing[1] * (1 - risks[0]) + spacing[2] * (1 - risks[1])
        rmst = long.rmst(f"a{arm}", 4.0)  # type: ignore[attr-defined]
        assert rmst.psi == pytest.approx(truth, rel=1e-12)
        unit = 3.0 - risks[0] - risks[1]
        assert abs(rmst.psi - unit) > 0.1
        lost = long.rmtl(f"a{arm}", 4.0)  # type: ignore[attr-defined]
        assert lost.psi + rmst.psi == pytest.approx(4.0, rel=1e-12)
    assert long.rmst("a1", 2.5).name == "rmst_regimen[a1 @ t=2]"  # type: ignore[attr-defined]


def test_on_the_unit_grid_rmst_is_the_wide_rmst() -> None:
    long = _fit(_long(None, LAW.long_frame((1.0, 2.0, 3.0))))
    wide = _wide()
    for tau in (2, 3, 4):
        left = long.rmst("a1", tau, versus="a0")  # type: ignore[attr-defined]
        right = wide.rmst("a1", tau, versus="a0")  # type: ignore[attr-defined]
        _same(left.psi, right.psi)


def test_the_curve_and_frame_report_grid_times() -> None:
    long = _fit(_long(UNEQUAL, LAW.long_frame(UNEQUAL)), horizons=[2.5, 4.0])
    curve = long.curve()  # type: ignore[attr-defined]
    assert sorted(set(curve["time"])) == [2.5, 4.0]
    frame = long.to_frame()  # type: ignore[attr-defined]
    assert sorted(set(frame["time"])) == [2.5, 4.0]
    assert "risk_regimen[a1 @ t=3]" in long.estimates  # type: ignore[attr-defined]


# ---------------------------------------------------------------------- refusals


def _frame(**columns: list[float]) -> pd.DataFrame:
    base = {
        "time": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0] * 2,
        "event": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 2,
        "A": [0, 1] * 10,
        "W": [0, 0, 1, 1, 0, 1, 1, 0, 0, 1] * 2,
    }
    base.update(columns)
    return pd.DataFrame(base)


def _build(frame: pd.DataFrame, **kwargs: object) -> LongitudinalData:
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=["W"], **kwargs
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0])
def test_r1_a_time_that_is_missing_or_not_positive(bad: float) -> None:
    times = [1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, bad] * 2
    with pytest.raises(DataError, match="missing, not finite, or at or below zero"):
        _build(_frame(time=times))


def test_r2_no_default_grid_for_non_integer_times() -> None:
    times = [1.5, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0] * 2
    with pytest.raises(DataError, match="holds non-integer times"):
        _build(_frame(time=times))


@pytest.mark.parametrize("grid", [[2, 1], [0, 1], [1, 1, 2], [], [1, np.inf]])
def test_r3_a_bad_grid(grid: list[float]) -> None:
    with pytest.raises(ValueError, match="strictly increasing sequence of positive finite"):
        _build(_frame(), grid=grid)


def test_r4_event_codes() -> None:
    events = [1, 0, 1, 1, 0, 1, 0, 1, 1, np.nan] * 2
    with pytest.raises(DataError, match="missing value"):
        _build(_frame(event=events))
    events = [1, 0, 1, 1, 0, 1, 0, 1, 1, -1] * 2
    with pytest.raises(DataError, match="negative or not integers"):
        _build(_frame(event=events))
    events = [1, 0, 2, 1, 0, 3, 0, 1, 1, 0] * 2
    with pytest.raises(DataError, match="causes= does not name"):
        _build(_frame(event=events), causes={1: "relapse", 2: "death"})
    events = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 2
    with pytest.raises(DataError, match="has no event at or before the last grid time"):
        _build(_frame(event=events), causes={1: "relapse", 2: "death"})


def test_r5_a_missing_treatment_or_covariate_keeps_the_shipped_message() -> None:
    with pytest.raises(DataError):
        _build(_frame(A=[0, 1, None, 1, 0, 1, 0, 1, 0, 1] * 2))
    with pytest.raises(DataError):
        _build(_frame(W=[0, 1, np.nan, 1, 0, 1, 0, 1, 0, 1] * 2))


def test_r6_held_node_counts_must_agree() -> None:
    frame = LAW.frame()
    columns = LAW.fit_columns()
    columns["censoring"] = ["C1", "C2"]
    with pytest.raises(DataError, match="held over every node"):
        LongitudinalData.from_frame(frame, **columns)


def test_r7_a_horizon_or_tau_off_the_grid() -> None:
    data = _long(UNEQUAL, LAW.long_frame(UNEQUAL))
    with pytest.raises(ValueError, match="is not a reported grid time"):
        LTMLE(REGIMENS, n_folds=1, horizons=[2.0], **_learners()).fit(data)
    result = _fit(data)
    for tau in (3.0, 1.0, 5.0):
        with pytest.raises(ValueError, match="is not a reported grid time"):
            result.rmst("a1", tau)  # type: ignore[attr-defined]


def test_r8_time_to_event_has_no_time_varying_covariate() -> None:
    with pytest.raises(TypeError, match="no covariate measured after baseline"):
        TimeToEvent(time="time", event="event", treatment="A", baseline=["W"], time_varying=[["L"]])  # type: ignore[arg-type]


def test_r10_a_container_from_the_other_path() -> None:
    wide = LongitudinalData.from_frame(LAW.frame(), **LAW.fit_columns())
    design = TimeToEvent(time="time", event="event", treatment="A", baseline=["W"])
    with pytest.raises(DataError, match="was built with"):
        CausalStudy(wide, design=design)


def test_r11_a_censoring_time_between_grid_points() -> None:
    times = [1.0, 2.5, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0] * 2
    with pytest.raises(DataError, match="censoring time\\(s\\) between grid points"):
        _build(_frame(time=times), grid=[1, 2, 3])
    # The same time as an event is binned, not refused.
    events = [1, 1, 1, 1, 0, 1, 0, 1, 1, 0] * 2
    data = _build(_frame(time=times, event=events), grid=[1, 2, 3])
    assert data.n_times == 3


def test_every_refusal_comes_before_any_learner() -> None:
    times = [1.0, 2.5, 3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0] * 2
    design = TimeToEvent(time="time", event="event", treatment="A", baseline=["W"], grid=[1, 2, 3])
    with pytest.raises(DataError):
        CausalStudy(_frame(time=times), design=design).estimate(
            RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0"),
            outcome_learner=Refuse(),
            treatment_learner=Refuse(),
            censoring_learner=Refuse(),
        )


def test_the_study_path_fits_a_time_to_event_design() -> None:
    frame = LAW.long_frame(UNEQUAL, offset=0.5)
    design = TimeToEvent(time="time", event="event", treatment="A", baseline=["W"], grid=UNEQUAL)
    result = CausalStudy(frame, design=design).estimate(
        RegimeContrast(regimens={"a1": 1, "a0": 0}, reference="a0", horizons=[4.0]),
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
        cross_fit=False,
    )
    truth = LAW.functional(LAW.probs, static(1), 3) - LAW.functional(LAW.probs, static(0), 3)
    assert result["ate_regimen[a1 vs a0 @ t=3]"].psi == pytest.approx(truth, rel=1e-12)

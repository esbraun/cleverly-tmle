"""The jump bound of the continuous-time engine.

Where a unit's cause hazard jumps at one arm and time sum to at least ``1 - JUMP_BOUND``, the
engine scales them to sum to ``1 - JUMP_BOUND``, each cause keeping its share
(:func:`cleverly.survival._chunked.bound_jumps`).  The witnesses here are user hazard learners
that put chosen jumps at one unit, arm and time, with both causes nonzero there, so a bound that
dropped the shares, acted below its threshold, or reached the checks but not the walk would fail
a test.  The block walk's tests (``test_continuous_survival_blocks.py``) cover the bounded walk
against the dense one and the bounded pairs' own floor.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from cleverly.exceptions import PositivityWarning
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard, _chunked, engine
from cleverly.survival.result import jump_bound_reason, level_name

DELTA = _chunked.JUMP_BOUND
N = 300


def _sample(n: int = N, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X3 = rng.normal(size=n)
    Z = rng.binomial(1, 1 / (1 + np.exp(-0.5 * X3)))
    h1 = np.exp(-1.5 + 0.3 * X3 - 0.5 * Z)
    h2 = np.exp(-1.8 - 0.3 * X3)
    T = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    C = rng.exponential(1 / 0.15, n)
    return pd.DataFrame(
        {"time": np.minimum(T, C), "event": np.where(T <= C, cause, 0), "Z": Z, "X3": X3}
    )


class Spiked(BaseEstimator):  # type: ignore[misc]
    """A Cox model whose jump at ``at`` is set to ``jumps[cause]`` for one row at one arm.

    The learner is fitted once per cause on that cause's event indicator, so it tells the causes
    apart by their event counts, ``counts``; a fit on any other count (a selection fold) is
    the plain Cox model.  The row is the one whose covariate equals ``x``.
    """

    def __init__(
        self,
        x: float = 0.0,
        arm: int = 1,
        at: float = 0.0,
        counts: tuple[int, ...] = (),
        jumps: tuple[float, ...] = (),
    ) -> None:
        self.x = x
        self.arm = arm
        self.at = at
        self.counts = counts
        self.jumps = jumps

    def fit(self, X: Any, time: Any, event: Any) -> Spiked:
        self.model_ = CoxHazard("main").fit(X, time, event)
        events = int(np.sum(event))
        self.cause_ = self.counts.index(events) if events in self.counts else None
        return self

    def cumulative_hazard(self, X: Any, times: Any) -> np.ndarray:
        X, times = np.asarray(X, dtype=float), np.asarray(times, dtype=float)
        out = np.array(self.model_.cumulative_hazard(X, times), dtype=float)
        rows = (X[:, 1] == self.x) & (X[:, 0] == self.arm)
        if self.cause_ is None or not rows.any() or self.at not in times:
            return out
        m = int(np.flatnonzero(times == self.at)[0])
        before = out[rows, m - 1] if m else 0.0
        out[np.ix_(rows, times >= self.at)] += (self.jumps[self.cause_] + before - out[rows, m])[
            :, None
        ]
        return out


def _cell(frame: pd.DataFrame, *, own_arm: bool) -> tuple[int, int, float]:
    """A unit, the arm to spike it at and a grid time before the target at which it is at risk.

    ``own_arm`` picks a treated unit at its received arm; otherwise a control unit read at the
    treated arm, its counterfactual arm.
    """
    T, Z = frame["time"].to_numpy(), frame["Z"].to_numpy()
    grid = np.unique(T[T <= TARGET])
    at = float(grid[grid.size // 3])
    unit = int(np.flatnonzero(((1 if own_arm else 0) == Z) & (at < T))[0])
    return unit, 1, at


TARGET = 2.0


def _estimator(frame: pd.DataFrame, unit: int, arm: int, at: float, jumps: Any) -> Any:
    event = frame["event"].to_numpy()
    counts = (int(np.sum(event == 1)), int(np.sum(event == 2)))
    assert counts[0] != counts[1]
    spiked = Spiked(float(frame["X3"].iloc[unit]), arm, at, counts, tuple(jumps))
    return ContinuousSurvivalTMLE(
        [1.0, TARGET],
        cross_fit=False,
        hazard_library=[spiked],
        censoring_library=[CoxHazard("main")],
        random_state=0,
    )


@contextmanager
def _captured() -> Iterator[dict[str, Any]]:
    """The fit's initial fit, as the walk reads it."""
    seen: dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as patch:
        target = engine.ContinuousSurvivalTMLE._target

        def capture(self: Any, initial: Any, floor: float, *, settings: Any) -> Any:
            seen["initial"] = initial
            return target(self, initial, floor, settings=settings)

        patch.setattr(engine.ContinuousSurvivalTMLE, "_target", capture)
        yield seen


def _fit(frame: pd.DataFrame, unit: int, arm: int, at: float, jumps: Any) -> tuple[Any, Any, list]:
    with _captured() as seen, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _estimator(frame, unit, arm, at, jumps).fit(
            frame, time="time", event="event", treatment="Z", baseline=["X3"]
        )
    return result, seen["initial"], [w for w in caught if w.category is PositivityWarning]


def _plug_in(dlam: np.ndarray, initial: Any) -> dict[tuple[int, int, int], float]:
    """Each arm's, cause's and target's mean cumulative incidence, by hand from ``dlam``."""
    survival = np.cumprod(1.0 - dlam.sum(axis=1), axis=-1)
    before = np.concatenate([np.ones((*survival.shape[:-1], 1)), survival[..., :-1]], axis=-1)
    incidence = np.cumsum(before[:, None] * dlam, axis=-1)
    return {
        (a, j, k): float(incidence[a, j, :, m].mean())
        for a in (0, 1)
        for j in range(dlam.shape[1])
        for k, m in enumerate(initial.target_index)
    }


def test_a_two_cause_jump_at_a_counterfactual_arm_is_scaled_keeping_each_share() -> None:
    """The review's witness: causes jump by 0.9 and 0.6 at one control unit's treated arm."""
    frame = _sample()
    unit, arm, at = _cell(frame, own_arm=False)
    result, initial, caught = _fit(frame, unit, arm, at, (0.9, 0.6))
    m = int(np.searchsorted(initial.grid, at))
    raw = initial.hazards.raw_jumps(0, initial.n)
    bounded = initial.hazards.jumps(0, initial.n)
    np.testing.assert_allclose(raw[arm, :, unit, m], (0.9, 0.6), rtol=0, atol=1e-15)
    cell = bounded[arm, :, unit, m]
    # The cell sums to 1 - delta and each cause keeps its share, 0.6 and 0.4.
    assert abs(cell.sum() - (1.0 - DELTA)) <= 1e-15
    np.testing.assert_allclose(cell / cell.sum(), (0.6, 0.4), rtol=0, atol=1e-15)
    # Only that cell moved.
    changed = raw != bounded
    assert changed[arm, :, unit, m].all() and changed.sum() == 2
    record = result.record
    assert record.jump_bound_share == 1 / (2 * N)
    assert record.jump_bound_at_risk_share == 0.0
    assert initial.jump_bounded is not None and initial.jump_bounded.sum() == 1
    # One warning, naming the arm, both causes and the pair, and the label says why.
    (warning,) = caught
    text = str(warning.message)
    assert text.startswith(
        "The initial hazard fit put a combined jump of one or more at 1 (unit, arm) pair: 1 at "
        "arm '1' (causes '1' and '2')."
    )
    assert "positivity signal" in text and "overlap" in text and "while at risk" not in text
    for estimate in result.values():
        assert estimate.support.label == "experimental"
        assert jump_bound_reason(1) in estimate.support.reason
    assert result.to_dict()["record"]["jump_bound_share"] == 1 / (2 * N)
    # The walk read the bounded jumps: its plug-in at the initial fit is the hand plug-in from
    # the raw jumps scaled by hand.
    by_hand = raw.copy()
    by_hand[arm, :, unit, m] *= (1.0 - DELTA) / raw[arm, :, unit, m].sum()
    hand = _plug_in(by_hand, initial)
    initial_psi = result.initial_estimates
    for (a, j, k), value in hand.items():
        name = level_name(result.arms[a], result.causes[j], result.horizons[k])
        assert abs(initial_psi[name] - value) <= 1e-14, name
    # The bounded pair's survival after the cell is delta times the later factors.
    survival = np.cumprod(1.0 - bounded.sum(axis=1), axis=-1)
    assert survival[arm, unit, m] <= DELTA * (1 + 1e-6)


def test_a_jump_at_the_received_arm_while_at_risk_is_counted_apart() -> None:
    """Under a Cox model this needs tied events; the user learner puts it there directly."""
    frame = _sample()
    unit, arm, at = _cell(frame, own_arm=True)
    result, initial, caught = _fit(frame, unit, arm, at, (0.7, 0.5))
    record = result.record
    assert record.jump_bound_share == record.jump_bound_at_risk_share == 1 / (2 * N)
    (warning,) = caught
    assert "1 of the pairs had such a jump at the received arm while at risk" in str(
        warning.message
    )
    m = int(np.searchsorted(initial.grid, at))
    cell = initial.hazards.jumps(0, initial.n)[arm, :, unit, m]
    np.testing.assert_allclose(cell / cell.sum(), (0.7 / 1.2, 0.5 / 1.2), rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("jumps", "acts"),
    [((0.6, 0.4 - 2 * DELTA), False), ((0.6, 0.4 - DELTA / 2), True), ((0.6, 0.4), True)],
)
def test_the_bound_acts_from_one_less_delta_and_leaves_lower_cells_untouched(
    jumps: tuple[float, float], acts: bool
) -> None:
    """The threshold: a cell just below ``1 - delta`` is not written to, bit for bit."""
    frame = _sample()
    unit, arm, at = _cell(frame, own_arm=False)
    result, initial, caught = _fit(frame, unit, arm, at, jumps)
    raw = initial.hazards.raw_jumps(0, initial.n)
    bounded = initial.hazards.jumps(0, initial.n)
    assert np.array_equal(raw, bounded) is not acts
    assert (result.record.jump_bound_share > 0) is acts
    assert (initial.jump_bounded is not None) is acts
    assert len(caught) == int(acts)
    reasons = [estimate.support.reason for estimate in result.values()]
    assert all((jump_bound_reason(1) in reason) is acts for reason in reasons)
    if acts:
        m = int(np.searchsorted(initial.grid, at))
        assert abs(bounded[arm, :, unit, m].sum() - (1.0 - DELTA)) <= 1e-15

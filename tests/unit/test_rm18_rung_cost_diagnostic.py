"""The cost of the deferred multi-arm rung design: committed inputs, delta method and CSV."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests.diagnostics.rm18_rung_cost.cost import (
    ARMS,
    COLUMNS,
    CURRENT,
    HERE,
    PROPERTIES,
    SEEDS,
    SIZES,
    TARGET,
    delta_half_width,
    surrogate_half_width,
)

COMMITTED = HERE / "cost.csv"


@pytest.fixture(scope="module")
def committed() -> pd.DataFrame:
    return pd.read_csv(COMMITTED)


@pytest.fixture(scope="module")
def properties() -> pd.DataFrame:
    frame = pd.read_csv(PROPERTIES)
    return frame.loc[frame["property"] == "double_robust_contraction"].set_index("cell")


def _value(frame: pd.DataFrame, arm: str, quantity: str, column: str = "value") -> float:
    chosen = frame.loc[(frame["arm"] == arm) & (frame["quantity"] == quantity), column]
    assert len(chosen) == 1, (arm, quantity)
    return float(chosen.iloc[0])


def test_the_table_has_its_columns(committed: pd.DataFrame) -> None:
    assert list(committed.columns) == list(COLUMNS)


def test_the_delta_method_cost_follows_from_the_committed_rows(
    committed: pd.DataFrame, properties: pd.DataFrame
) -> None:
    """Recompute every delta-method figure longhand from ``properties.csv``."""
    spread = np.mean(
        [
            properties.loc[f"both_wrong_n{n}", "empirical_se"] * math.sqrt(n)
            for n in (2_000, 4_000, 8_000)
        ]
    )
    assert _value(committed, "both_wrong", "control spread c") == pytest.approx(spread, rel=1e-12)
    z = norm.ppf(0.995)
    expected = {}
    for arm in ARMS:
        bias = properties.loc[f"{arm}_n2000", "bias"]
        assert _value(committed, arm, "first-rung bias") == pytest.approx(bias, rel=1e-12)
        exact = (z * spread / (math.log(4) * bias * 2_000)) ** 2 * (2_000 + 8_000)
        assert _value(committed, arm, "delta replications, exact") == pytest.approx(exact, rel=1e-9)
        expected[arm] = math.ceil(exact)
        assert _value(committed, arm, "delta replications", "replications") == expected[arm]
        # The rounded budget reaches the target, and one fewer replication does not.
        assert delta_half_width(bias, spread, SIZES, expected[arm]) <= TARGET
        assert delta_half_width(bias, spread, SIZES, expected[arm] - 1) > TARGET
    assert expected == {"outcome_correct": 9_240, "treatment_correct": 34_267}


def test_the_current_delta_half_width_is_recorded(committed: pd.DataFrame) -> None:
    for arm in ARMS:
        bias = _value(committed, arm, "first-rung bias")
        spread = _value(committed, "both_wrong", "control spread c")
        current = committed.loc[committed["section"] == "current"]
        recorded = _value(current, arm, "delta half-width")
        assert recorded == pytest.approx(delta_half_width(bias, spread, SIZES, CURRENT), rel=1e-9)


def test_the_recorded_surrogate_budget_is_the_first_to_reach_the_target(
    committed: pd.DataFrame,
) -> None:
    """Two surrogate evaluations per arm, at the recorded budget and one grid step below it."""
    spread = _value(committed, "both_wrong", "control spread c")
    for arm in ARMS:
        bias = _value(committed, arm, "first-rung bias")
        rows = committed.loc[
            (committed["arm"] == arm)
            & (committed["quantity"] == "surrogate replications")
            & (committed["seed"] == SEEDS[0])
        ]
        (outer,) = rows["replications"].astype(int)
        at = surrogate_half_width(bias, spread, SIZES, (outer, CURRENT, outer), SEEDS[0])
        below = surrogate_half_width(
            bias, spread, SIZES, (outer - 1_000, CURRENT, outer - 1_000), SEEDS[0]
        )
        assert at <= TARGET < below, (arm, at, below)


def test_the_binary_check_is_recorded(committed: pd.DataFrame) -> None:
    rows = committed.loc[
        (committed["section"] == "binary check") & (committed["quantity"] == "surrogate half-width")
    ]
    recorded = dict(zip(rows["replications"].astype(int), rows["value"], strict=True))
    assert set(recorded) == {800, 2_000, 2_400}
    fresh = surrogate_half_width(0.0036, 1.05, (1_500, 3_000, 6_000), (2_400, 800, 2_400), SEEDS[0])
    assert recorded[2_400] == pytest.approx(fresh, rel=1e-12)
    # The binary rule records projections of about 2.0, 1.05 and 0.88; this surrogate agrees
    # within five per cent.
    for outer, projected in ((800, 2.022), (2_000, 1.049), (2_400, 0.876)):
        assert recorded[outer] == pytest.approx(projected, rel=0.05)

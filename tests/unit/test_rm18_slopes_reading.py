"""The Design SL reading of the two multi-arm DR-TMLE contraction slopes.

"Design SL" in ``docs/roadmap.md`` declares the reading table and the control rule before the
run. This file applies them to the committed ``canonical-multi-arm-drtmle`` artifacts. The slope
intervals come from ``properties.csv``. The declaration does not name the budget of the
n = 2,000 bias interval. RM18 resolved it after the run, first at 600 and after review at the
slope's own inputs, all 73,000 rows of the rung. The 99% Student interval over the first 600
rows is supplementary.
The current snapshots describe the RM33 targeted seed repair under the unchanged reading rule.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

STUDY = Path(__file__).resolve().parents[1] / "canonical" / "multi_arm_drtmle"
FAMILY = "double_robust_contraction"
POSITIVES = ("outcome_correct", "treatment_correct")
OUTER_BUDGET = 73_000
VERDICT_BUDGET = 600


def reading(slope: tuple[float, float], bias: tuple[float, float]) -> str:
    """The declared reading of one positive arm; the first row that holds names it."""
    bias_covers_zero = bias[0] <= 0.0 <= bias[1]
    if slope[0] > 0.0:
        return "grows"
    if slope[1] < 0.0:
        return "contracts at the noise floor" if bias_covers_zero else "contracts"
    return "no resolved bias" if bias_covers_zero else "does not contract"


def control_reading(slope: tuple[float, float]) -> str:
    """The registered control rule, with the declared reading of its failure."""
    if slope[1] < 0.0:
        return "instrument cannot separate a transient from contraction at this budget"
    return "passes"


def student(errors: np.ndarray) -> tuple[float, float, float]:
    mean = float(errors.mean())
    half = stats.t.ppf(0.995, len(errors) - 1) * errors.std(ddof=1) / math.sqrt(len(errors))
    return mean, mean - half, mean + half


@pytest.fixture(scope="module")
def properties() -> pd.DataFrame:
    frame = pd.read_csv(STUDY / "properties.csv", float_precision="round_trip")
    return frame.loc[frame["property"] == FAMILY].set_index("cell")


@pytest.fixture(scope="module")
def rungs() -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(
        STUDY / "property-replicates.csv.gz",
        float_precision="round_trip",
        usecols=["property", "cell", "replicate", "truth", "estimate"],
    )
    ladder = frame.loc[frame["property"] == FAMILY]
    return {str(cell): group.sort_values("replicate") for cell, group in ladder.groupby("cell")}


def _slope(properties: pd.DataFrame, scenario: str) -> tuple[float, float]:
    row = properties.loc[f"rate_{scenario}"]
    return float(row["slope_ci_lower"]), float(row["slope_ci_upper"])


def _bias(rungs: dict[str, pd.DataFrame], cell: str, budget: int) -> tuple[float, float, float]:
    rows = rungs[cell].head(budget)
    return student(rows["estimate"].to_numpy(float) - rows["truth"].to_numpy(float))


def test_the_outer_rungs_ran_the_declared_budget(rungs: dict[str, pd.DataFrame]) -> None:
    for scenario in (*POSITIVES, "both_wrong"):
        for n in (2_000, 8_000):
            assert list(rungs[f"{scenario}_n{n}"]["replicate"]) == list(range(OUTER_BUDGET))
        assert list(rungs[f"{scenario}_n4000"]["replicate"]) == list(range(VERDICT_BUDGET))


def test_the_longhand_interval_is_the_published_one_at_the_verdict_budget(
    properties: pd.DataFrame, rungs: dict[str, pd.DataFrame]
) -> None:
    for cell in rungs:
        _, low, high = _bias(rungs, cell, VERDICT_BUDGET)
        assert low == pytest.approx(properties.loc[cell, "bias_ci_lower"], rel=1e-9, abs=1e-15)
        assert high == pytest.approx(properties.loc[cell, "bias_ci_upper"], rel=1e-9, abs=1e-15)


def test_each_positive_arm_reads_the_current_observed_label(
    properties: pd.DataFrame, rungs: dict[str, pd.DataFrame]
) -> None:
    """The reading uses the n = 2,000 interval over the slope's own inputs, all 73,000 rows."""
    published = {
        "outcome_correct": (
            -0.859924,
            (-1.071846, -0.666566),
            (0.001750, 0.001497, 0.002004),
            "contracts",
        ),
        "treatment_correct": (
            -0.874310,
            (-1.023958, -0.740786),
            (0.002538, 0.002284, 0.002792),
            "contracts",
        ),
    }
    for scenario in POSITIVES:
        slope = _slope(properties, scenario)
        point, low, high = _bias(rungs, f"{scenario}_n2000", OUTER_BUDGET)
        assert properties.loc[f"rate_{scenario}", "slope"] == pytest.approx(
            published[scenario][0], abs=5e-7
        )
        assert slope == pytest.approx(published[scenario][1], abs=5e-7)
        assert (point, low, high) == pytest.approx(published[scenario][2], abs=5e-7)
        assert reading(slope, (low, high)) == published[scenario][3]
        assert bool(properties.loc[f"rate_{scenario}", "passed"])
        assert bool(properties.loc[f"rate_{scenario}", "passed"]) == (slope[1] < 0.0)


def test_the_current_full_outer_rung_bias_intervals(rungs: dict[str, pd.DataFrame]) -> None:
    """All six outer-rung intervals; only the positive n = 2,000 intervals set the reading."""
    published = {
        "outcome_correct_n2000": (0.001750, 0.001497, 0.002004),
        "outcome_correct_n8000": (0.000531, 0.000405, 0.000658),
        "treatment_correct_n2000": (0.002538, 0.002284, 0.002792),
        "treatment_correct_n8000": (0.000755, 0.000629, 0.000881),
        "both_wrong_n2000": (0.049149, 0.048898, 0.049400),
        "both_wrong_n8000": (0.049276, 0.049151, 0.049401),
    }
    for cell, expected in published.items():
        point, low, high = _bias(rungs, cell, OUTER_BUDGET)
        assert (point, low, high) == pytest.approx(expected, abs=5e-7)
        assert low > 0.0


def test_the_supplementary_verdict_budget_interval_covers_zero(
    properties: pd.DataFrame, rungs: dict[str, pd.DataFrame]
) -> None:
    """Rows 0 to 599, published before the declaration, would read the noise floor."""
    published = {
        "outcome_correct": (0.001127, -0.001651, 0.003905),
        "treatment_correct": (0.000585, -0.002274, 0.003444),
    }
    for scenario in POSITIVES:
        point, low, high = _bias(rungs, f"{scenario}_n2000", VERDICT_BUDGET)
        assert (point, low, high) == pytest.approx(published[scenario], abs=5e-7)
        assert reading(_slope(properties, scenario), (low, high)) == (
            "contracts at the noise floor"
        )


def test_the_control_passes_and_meets_its_prediction(properties: pd.DataFrame) -> None:
    slope = _slope(properties, "both_wrong")
    assert properties.loc["rate_both_wrong", "slope"] == pytest.approx(0.001866, abs=5e-7)
    assert slope == pytest.approx((-0.002220, 0.006014), abs=5e-7)
    assert control_reading(slope) == "passes"
    assert bool(properties.loc["rate_both_wrong", "passed"])
    assert bool(properties.loc["rate_both_wrong", "passed"]) == (slope[1] >= 0.0)
    # The declared prediction: the half-width shrinks by about sqrt(73,000 / 600) from 0.0438,
    # to about 0.0040, within 15%.
    predicted = 0.0438 / math.sqrt(OUTER_BUDGET / VERDICT_BUDGET)
    half_width = (slope[1] - slope[0]) / 2
    assert half_width == pytest.approx(0.004117, abs=5e-7)
    assert abs(half_width / predicted - 1) <= 0.15


@pytest.mark.parametrize(
    ("slope", "bias", "expected"),
    [
        ((0.1, 0.4), (0.001, 0.002), "grows"),
        ((0.1, 0.4), (-0.001, 0.002), "grows"),
        ((-1.0, -0.5), (0.001, 0.002), "contracts"),
        ((-1.0, -0.5), (-0.002, -0.001), "contracts"),
        ((-1.0, -0.5), (-0.001, 0.002), "contracts at the noise floor"),
        ((-1.0, 0.5), (0.001, 0.002), "does not contract"),
        ((-1.0, 0.5), (-0.001, 0.002), "no resolved bias"),
    ],
)
def test_the_reading_table(
    slope: tuple[float, float], bias: tuple[float, float], expected: str
) -> None:
    assert reading(slope, bias) == expected


def test_the_control_rule_reads_a_transient() -> None:
    assert control_reading((-0.01, -0.001)) == (
        "instrument cannot separate a transient from contraction at this budget"
    )
    assert control_reading((-0.01, 0.001)) == "passes"

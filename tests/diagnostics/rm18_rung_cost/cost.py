"""The cost of the deferred multi-arm DR-TMLE rung design, computed from committed rows alone.

RM18 of ``docs/roadmap.md`` sanctions a rung design for the two multi-arm contraction slopes
and defers its run.  "What the multi-arm rung design would cost" in that section states the
cost.  This module is the computation behind it.  It applies the sizing rule that
``CONTRACTION_REPLICATES`` in ``tests/studies/drtmle_properties.py`` declares for the binary
ladder, and it reads ``tests/canonical/multi_arm_drtmle/properties.csv``.  It fits nothing.

The rule reads two kinds of quantity and no others.

* The first-rung bias of each positive arm, carried down the ladder as ``1/n``.
* The ``both_wrong`` control's spread ``c``, the mean of ``empirical_se * sqrt(n)`` over its three
  rungs, carried as ``c / sqrt(n)``.

It does not read the slope, the interval or the verdict of either positive rate cell.

Two methods give the smallest ``R``, the replications at each outer rung, that brings the 99%
half-width of the fitted slope to one.  The middle rung keeps 600, because its centred weight
is zero.

* The delta method, in closed form:
  ``R = (z * c / (log 4 * b1 * n1))**2 * (n1 + n3)``, rounded up.
* A surrogate simulation of the kind the binary rule quoted.  Each rung's bias is drawn from a
  normal law with the rule's mean and spread, the slope of ``log |bias|`` on ``log n`` is
  fitted, and the half-width is half the 99% range of that slope over :data:`DRAWS` draws.  The
  grid search steps by :data:`GRID_STEP` and uses fixed seeds, so a rerun gives the same table.

The binary rows check the surrogate against the projections ``CONTRACTION_REPLICATES`` records.

``--output`` is required, so a bare run cannot overwrite the committed ``cost.csv``::

    python -m tests.diagnostics.rm18_rung_cost.cost --output <scratch>/cost.csv
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.studies.evidence.manifest import write_csv
from tests.studies.evidence.registry import ROOT

HERE = Path(__file__).resolve().parent
PROPERTIES = ROOT / "tests" / "canonical" / "multi_arm_drtmle" / "properties.csv"

CONFIDENCE_LEVEL = 0.99
Z = float(norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))
#: The half-width the binary rule declares as its target.
TARGET = 1.0

SIZES = (2_000, 4_000, 8_000)
#: The replications each multi-arm rung runs now, and the middle rung keeps.
CURRENT = 600
ARMS = ("outcome_correct", "treatment_correct")
CONTROL = "both_wrong"

#: The binary rule's inputs and budgets, as ``CONTRACTION_REPLICATES`` records them.
BINARY_BIAS = 0.0036
BINARY_SPREAD = 1.05
BINARY_SIZES = (1_500, 3_000, 6_000)
BINARY_MIDDLE = 800
BINARY_BUDGETS = (800, 2_000, 2_400)

DRAWS = 40_000
GRID_STEP = 1_000
GRID_LIMIT = 200_000
SEEDS = (20_260_921, 1, 2)

COLUMNS = ("section", "arm", "quantity", "seed", "replications", "value")


def inputs(properties: pd.DataFrame) -> dict[str, float]:
    """The first-rung bias of each arm and the control spread ``c``, from committed rows."""
    rows = properties.loc[properties["property"] == "double_robust_contraction"].set_index("cell")
    out = {arm: float(rows.loc[f"{arm}_n{SIZES[0]}", "bias"]) for arm in ARMS}
    spreads = [
        float(rows.loc[f"{CONTROL}_n{size}", "empirical_se"]) * math.sqrt(size) for size in SIZES
    ]
    out["c"] = float(np.mean(spreads))
    return out


def delta_half_width(bias: float, spread: float, sizes: Sequence[int], outer: int) -> float:
    """The delta-method 99% half-width of the slope at ``outer`` replications per outer rung."""
    first, last = sizes[0], sizes[-1]
    return (
        Z
        * spread
        * math.sqrt(first + last)
        / (math.log(last / first) * abs(bias) * first * math.sqrt(outer))
    )


def delta_replications(bias: float, spread: float, sizes: Sequence[int]) -> float:
    """The ``R`` at which :func:`delta_half_width` equals :data:`TARGET`, before rounding."""
    first, last = sizes[0], sizes[-1]
    return (Z * spread / (math.log(last / first) * abs(bias) * first * TARGET)) ** 2 * (
        first + last
    )


def surrogate_half_width(
    bias: float, spread: float, sizes: Sequence[int], budgets: Sequence[int], seed: int
) -> float:
    """Half the 99% range of the fitted slope over :data:`DRAWS` surrogate ladders."""
    n = np.asarray(sizes, dtype=float)
    mean = abs(bias) * n[0] / n
    sd = spread / np.sqrt(n) / np.sqrt(np.asarray(budgets, dtype=float))
    rng = np.random.default_rng(seed)
    draws = mean + rng.standard_normal((DRAWS, len(sizes))) * sd
    centred = np.log(n) - np.log(n).mean()
    weights = centred / np.sum(centred**2)
    slopes = np.log(np.abs(draws)) @ weights
    low, high = np.quantile(slopes, [(1 - CONFIDENCE_LEVEL) / 2, (1 + CONFIDENCE_LEVEL) / 2])
    return float((high - low) / 2)


def surrogate_replications(bias: float, spread: float, seed: int) -> int:
    """The smallest ``R`` on the grid whose surrogate half-width reaches :data:`TARGET`."""
    for outer in range(GRID_STEP, GRID_LIMIT + 1, GRID_STEP):
        budgets = (outer, CURRENT, outer)
        if surrogate_half_width(bias, spread, SIZES, budgets, seed) <= TARGET:
            return outer
    raise ValueError(f"no R up to {GRID_LIMIT} reaches a half-width of {TARGET}")


def table(properties: pd.DataFrame) -> pd.DataFrame:
    """Every input, check and cost as one row."""
    values = inputs(properties)
    spread = values["c"]
    rows: list[tuple[str, str, str, int | None, int | None, float]] = [
        ("input", CONTROL, "control spread c", None, None, spread)
    ]
    for arm in ARMS:
        bias = values[arm]
        rows.append(("input", arm, "first-rung bias", None, None, bias))
        rows.append(
            (
                "current",
                arm,
                "delta half-width",
                None,
                CURRENT,
                delta_half_width(bias, spread, SIZES, CURRENT),
            )
        )
        rows.append(
            (
                "current",
                arm,
                "surrogate half-width",
                SEEDS[0],
                CURRENT,
                surrogate_half_width(bias, spread, SIZES, (CURRENT,) * 3, SEEDS[0]),
            )
        )
        exact = delta_replications(bias, spread, SIZES)
        rows.append(("cost", arm, "delta replications, exact", None, None, exact))
        rows.append(("cost", arm, "delta replications", None, math.ceil(exact), float("nan")))
        for seed in SEEDS:
            outer = surrogate_replications(bias, spread, seed)
            rows.append(("cost", arm, "surrogate replications", seed, outer, float("nan")))
    for outer in BINARY_BUDGETS:
        rows.append(
            (
                "binary check",
                "outcome_correct",
                "surrogate half-width",
                SEEDS[0],
                outer,
                surrogate_half_width(
                    BINARY_BIAS,
                    BINARY_SPREAD,
                    BINARY_SIZES,
                    (outer, BINARY_MIDDLE, outer),
                    SEEDS[0],
                ),
            )
        )
    rows.append(
        (
            "binary check",
            "outcome_correct",
            "delta half-width",
            None,
            BINARY_BUDGETS[-1],
            delta_half_width(BINARY_BIAS, BINARY_SPREAD, BINARY_SIZES, BINARY_BUDGETS[-1]),
        )
    )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["seed"] = frame["seed"].astype("Int64")
    frame["replications"] = frame["replications"].astype("Int64")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"where to write the table; the committed record is {HERE / 'cost.csv'}",
    )
    arguments = parser.parse_args()
    frame = table(pd.read_csv(PROPERTIES))
    write_csv(frame, arguments.output)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(frame.to_string(index=False))
    print(f"wrote {len(frame)} rows to {arguments.output}")


if __name__ == "__main__":
    main()

"""The pre-read of ``option-outcome-by-arm`` (Fix-8 plan section 5.0), and its summary.

Before the declaration every gated calibration cell is read on :data:`CALIBRATION_DRAWS` pilot
draws at the declared n and learners, with its pooled fit beside it, and the double-robustness
pair on :data:`DRAWS` draws.  The draws come from :data:`PILOT_SEED`, a stream no study draws
from, and each draw is fitted by :func:`tests.studies._option_by_arm_law.fit` exactly as the
property phase fits it.  The cells that a run fits on one draw (the two binary K fits and their
pooled fits; each other K fit and its pooled fit) are read on one pilot draw here too.

The committed summary :data:`SUMMARY` holds one row per pilot cell: the draws, the coverage of
the Wald interval, the mean estimate, its bias against the exact truth, the SD of the estimates,
the standardized bias, the mean reported SE and the SE ratio.  The two binary K rows also carry
the correlation of their estimates across draws, which the arm-pooled sizing rule reads.  The
design test reads the summary (``tests/unit/test_option_outcome_by_arm_design.py``).

Run ``python -m tests.studies.option_outcome_by_arm_pilot run --block B --start S --draws D
--out FILE`` once per block of draws, and ``python -m tests.studies.option_outcome_by_arm_pilot
summarize FILE...`` to write the summary.  The per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import _option_by_arm_law as law

#: The pilot draws of the double-robustness pair (plan 5.0: at least 2,000).
DRAWS = 2_000
#: The pilot draws of each calibration block, ``repeated_sizing.CALIBRATION_PILOT_DRAWS``.
CALIBRATION_DRAWS = 8_000
#: The pilot's own seed stream.
PILOT_SEED = 20261114
SUMMARY = Path(__file__).with_name("option_outcome_by_arm_pilot.csv")

#: Each pilot block: the law, the n and the settings fitted on one draw (each is its cell).
BLOCKS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "binary": (
        "LB",
        2_000,
        ("binary_in_sample", "binary_stacked", "pooled_binary_in_sample", "pooled_binary_stacked"),
    ),
    "continuous": ("LB-c", 2_000, ("continuous_in_sample", "pooled_continuous_in_sample")),
    "bounded": ("LB-b", 2_000, ("bounded_stacked", "pooled_bounded_stacked")),
    "robustness": ("LB", 2_000, ("q_intercept", "both_intercept")),
}
CALIBRATION_BLOCKS = ("binary", "continuous", "bounded")


def pilot_seed(block: str, draw: int) -> int:
    """The seed of one pilot draw of one block."""
    key = sum(ord(character) * 31**index for index, character in enumerate(block)) % 2**31
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(key, draw))
    return int(sequence.generate_state(1)[0])


def records(block: str, draw: int) -> list[dict[str, Any]]:
    """Every pilot row of one draw of one block."""
    name, n, settings = BLOCKS[block]
    frame = law.draw(name, n, pilot_seed(block, draw))
    truth = law.truth(name)["ate"]
    out = []
    for setting in settings:
        estimate = law.fit(frame, setting)["ate"]
        low, high = estimate.ci
        out.append(
            {
                "block": block,
                "cell": setting,
                "n": n,
                "draw": draw,
                "truth": truth,
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "covered": int(low <= truth <= high),
            }
        )
    return out


def run(block: str, start: int, draws: int, out: Path) -> None:
    """Fit draws ``start`` to ``start + draws - 1`` of one block and write their rows."""
    rows = [row for draw in range(start, start + draws) for row in records(block, draw)]
    pd.DataFrame(rows).to_parquet(out, index=False)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One summary row per pilot cell."""
    out = []
    for (block, cell), group in rows.groupby(["block", "cell"], sort=True):
        sd = float(group["estimate"].std(ddof=1))
        bias = float(group["estimate"].mean() - group["truth"].iloc[0])
        out.append(
            {
                "block": block,
                "cell": cell,
                "n": int(group["n"].iloc[0]),
                "draws": len(group),
                "first_draw": int(group["draw"].min()),
                "last_draw": int(group["draw"].max()),
                "truth": float(group["truth"].iloc[0]),
                "mean_estimate": float(group["estimate"].mean()),
                "bias": bias,
                "sd": sd,
                "bias_over_sd": bias / sd,
                "mean_se": float(group["std_error"].mean()),
                "se_ratio": float(group["std_error"].mean()) / sd,
                "coverage": float(group["covered"].mean()),
                "estimate_correlation": np.nan,
            }
        )
    summary = pd.DataFrame(out)
    binary = rows.loc[rows["block"] == "binary"].pivot(
        index="draw", columns="cell", values="estimate"
    )
    correlation = float(binary["binary_in_sample"].corr(binary["binary_stacked"]))
    pair = summary["cell"].isin(("binary_in_sample", "binary_stacked"))
    summary.loc[pair, "estimate_correlation"] = correlation
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--block", choices=sorted(BLOCKS), required=True)
    runner.add_argument("--start", type=int, required=True)
    runner.add_argument("--draws", type=int, required=True)
    runner.add_argument("--out", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("files", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        run(arguments.block, arguments.start, arguments.draws, arguments.out)
        return
    rows = pd.concat([pd.read_parquet(path) for path in arguments.files], ignore_index=True)
    rows = rows.drop_duplicates(["block", "cell", "draw"])
    summarize(rows).to_csv(SUMMARY, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()

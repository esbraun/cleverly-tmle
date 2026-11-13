"""The pre-read of ``option-markov-order`` (Fix-8 plan section 5.0), and its summary.

Before the declaration every gated calibration cell is read on :data:`CALIBRATION_DRAWS` pilot
draws at the declared n and learners, with the full-history fit beside it on the same draws, and
the necessity control and the double-robustness positive on :data:`DRAWS` draws each.  The draws
come from :data:`PILOT_SEED`, a stream no study draws from, and each draw is fitted by
:func:`tests.studies._option_markov_law.fit` exactly as the property phase fits it.

The committed summary :data:`SUMMARY` holds one row per pilot cell (a setting and an estimand):
the draws and their first and last index, the coverage of the Wald interval, the mean estimate,
its bias against the exact truth, the SD of the estimates, the standardized bias, the mean
reported SE and the SE ratio.  The calibration rows also carry the correlation of their estimates
with each other calibration cell, which the arm-pooled sizing rule reads.  The design test reads
the summary (``tests/unit/test_option_markov_order_design.py``).

Run ``python -m tests.studies.option_markov_order_pilot run --block B --start S --draws D --out
FILE`` once per block of draws, and ``python -m tests.studies.option_markov_order_pilot
summarize FILE...`` to write the summary.  The per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import _option_markov_law as law

#: The pilot draws of the necessity and double-robustness blocks (plan 5.0: at least 2,000).
DRAWS = 2_000
#: The pilot draws of the calibration block, ``repeated_sizing.CALIBRATION_PILOT_DRAWS``.
CALIBRATION_DRAWS = 8_000
#: The pilot's own seed stream.
PILOT_SEED = 20261115
SUMMARY = Path(__file__).with_name("option_markov_order_pilot.csv")

#: Each pilot block: the law, the n and the settings fitted on one draw.
BLOCKS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "calibration": ("LF1", 2_000, ("lf1_order1", "lf1_order2", "lf1_full")),
    "necessity": ("LF1", 2_000, ("lf1_order0",)),
    "robustness": ("LF2", 2_000, ("lf2_order0",)),
}
#: The gated calibration cells, as ``setting:estimand``.
CALIBRATION_CELLS = (
    f"lf1_order1:{law.EY_ALWAYS}",
    f"lf1_order1:{law.ATE}",
    f"lf1_order2:{law.ATE}",
)


def pilot_seed(block: str, draw: int) -> int:
    """The seed of one pilot draw of one block."""
    key = sum(ord(character) * 31**index for index, character in enumerate(block)) % 2**31
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(key, draw))
    return int(sequence.generate_state(1)[0])


def records(block: str, draw: int) -> list[dict[str, Any]]:
    """Every pilot row of one draw of one block."""
    name, n, settings = BLOCKS[block]
    frame = law.draw(name, n, pilot_seed(block, draw))
    truth = law.truth()
    out = []
    for setting in settings:
        result = law.fit(frame, setting)
        for estimand in (law.EY_ALWAYS, law.ATE):
            estimate = result.estimates[estimand]
            low, high = estimate.ci
            out.append(
                {
                    "block": block,
                    "cell": f"{setting}:{estimand}",
                    "n": n,
                    "draw": draw,
                    "truth": truth[estimand],
                    "estimate": float(estimate.psi),
                    "std_error": float(estimate.std_error),
                    "covered": int(low <= truth[estimand] <= high),
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
            }
        )
    summary = pd.DataFrame(out)
    wide = rows.loc[rows["block"] == "calibration"].pivot(
        index="draw", columns="cell", values="estimate"
    )
    for other in CALIBRATION_CELLS:
        column = f"correlation[{other}]"
        summary[column] = np.nan
        for cell in CALIBRATION_CELLS:
            summary.loc[summary["cell"] == cell, column] = float(wide[cell].corr(wide[other]))
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

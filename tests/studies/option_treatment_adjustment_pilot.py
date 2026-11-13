"""The pre-read of ``option-treatment-adjustment`` (Fix-8 plan section 5.0), and its summary.

Before the declaration every gated calibration cell is read on :data:`CALIBRATION_DRAWS` pilot
draws at the declared n and learners, with the bias cells and the instrument cells beside them.  The draws
come from :data:`PILOT_SEED`, a stream no study draws from, and each draw is fitted by
:func:`tests.studies._option_adjustment_laws.fit` exactly as the property phase fits it.  Cells
that a run fits on one draw (the two K fits; the constructed-error fit and its full-mechanism
reference at one n) are read on one pilot draw here too.

The committed summary :data:`SUMMARY` holds one row per pilot cell: the draws, the coverage of
the Wald interval, the mean estimate, its bias against the exact truth, the SD of the estimates,
the standardized bias, the mean reported SE and the SE ratio.  The two K rows also carry the
correlation of their estimates across draws, which the arm-pooled sizing rule reads.  The
design test reads the summary (``tests/unit/test_option_treatment_adjustment_design.py``).

Run ``python -m tests.studies.option_treatment_adjustment_pilot run --block B --start S --draws D
--out FILE`` once per block of draws, and ``python -m
tests.studies.option_treatment_adjustment_pilot summarize FILE...`` to write the summary.  The
per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.studies import _option_adjustment_laws as laws

#: The pilot draws of each bias and parametric cell (plan 5.0: at least 2,000).
DRAWS = 2_000
#: The pilot draws of the two gated calibration cells.  At 2,000 draws both read an SE ratio
#: near 1.03, 1.7 Monte Carlo errors above 1, where the arm-pooled sizing rule cannot price the
#: pilot's own noise; the pool of 8,000 draws is ``repeated_sizing.CALIBRATION_PILOT_DRAWS``.
CALIBRATION_DRAWS = 8_000
#: The pilot draws of each ladder size; the ladder holds no calibration cell.
LADDER_DRAWS = 1_000
#: The pilot's own seed stream.
PILOT_SEED = 20261113
CRITICAL = float(norm.ppf(0.975))
SUMMARY = Path(__file__).with_name("option_treatment_adjustment_pilot.csv")

#: Each pilot block: the law, the n and the ``cell -> setting`` fits made on one draw.
BLOCKS: dict[str, tuple[str, int, dict[str, str]]] = {
    "calibration": (
        "LDA",
        2_000,
        {"in_sample": "lda_in_sample", "stacked": "lda_stacked"},
    ),
    "robustness": (
        "LDA",
        2_000,
        {"treatment_correct": "lda_q_intercept", "both_wrong": "lda_both_intercept"},
    ),
    "parametric": ("LDB", 2_000, {"parametric_n2000": "ldb_parametric"}),
    **{
        f"ladder_{n}": ("LDB", n, {f"slow_n{n}": "ldb_slow", f"reference_n{n}": "ldb_slow_full"})
        for n in (1_000, 4_000, 16_000)
    },
}


def pilot_seed(block: str, draw: int) -> int:
    """The seed of one pilot draw of one block."""
    key = sum(ord(character) * 31**index for index, character in enumerate(block)) % 2**31
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(key, draw))
    return int(sequence.generate_state(1)[0])


def records(block: str, draw: int) -> list[dict[str, Any]]:
    """Every pilot row of one draw of one block."""
    law, n, fits = BLOCKS[block]
    frame = laws.draw(law, n, pilot_seed(block, draw))
    truth = laws.truth(law)["ate"]
    out = []
    for cell, setting in fits.items():
        estimate = laws.fit(frame, setting)["ate"]
        low, high = estimate.ci
        out.append(
            {
                "block": block,
                "cell": cell,
                "setting": setting,
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
                "setting": group["setting"].iloc[0],
                "n": int(group["n"].iloc[0]),
                "draws": len(group),
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
    calibration = rows.loc[rows["block"] == "calibration"].pivot(
        index="draw", columns="cell", values="estimate"
    )
    correlation = float(calibration["in_sample"].corr(calibration["stacked"]))
    summary.loc[summary["block"] == "calibration", "estimate_correlation"] = correlation
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

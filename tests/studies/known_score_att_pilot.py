"""The pre-read of ``known-score-att`` (X35 plan section 5, rev 2.1), and its summary.

Before the declaration every cell is read on pilot draws at the declared n, learners and
settings: the six calibration labels on :data:`CALIBRATION_DRAWS` Law S draws (each draw fits
the three calibration settings, as a run does), the root-n rungs at n = 500 and 8,000 on
:data:`RATE_DRAWS` draws, and the zero-gain pairs, the accuracy cells and the control on
:data:`DRAWS` draws.  The draws come from :data:`PILOT_SEED`, a stream no study draws from,
and each draw is fitted by :func:`tests.studies.known_score_att_law.fit` exactly as the
property phase fits it.

The committed summary :data:`SUMMARY` holds one row per pilot cell (block, setting and
estimand): the draws, the coverage of the Wald interval, the mean estimate, its bias against
the exact truth, the SD of the estimates, the standardized bias, the mean reported SE and the
SE ratio.  The known-score rows of the ``calibration`` and ``zero_gain`` blocks also carry the
paired SD ratio over the arm-mean fit of the same draws and the correlation of the two
estimates, which size the paired gate.  The design test reads the summary
(``tests/unit/test_known_score_att_design.py``).

Run ``python -m tests.studies.known_score_att_pilot run --block B --start S --draws D --out
FILE`` once per block of draws, and ``python -m tests.studies.known_score_att_pilot summarize
FILE...`` to write the summary.  The per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import known_score_att_law as law

#: The pilot draws of each calibration label, ``repeated_sizing.CALIBRATION_PILOT_DRAWS``.  The
#: first 400 draws read the ATT labels' SE ratios at 0.975 to 0.977, where the plug-in pass
#: probability at the at-target size falls below the guide's 0.90, so the pilot was extended
#: to the shared 8,000 draws before the declaration (draws 0 to 7,999 of one stream).
CALIBRATION_DRAWS = 8_000
#: The pilot draws of each root-n rung off n = 2,000.
RATE_DRAWS = 2_000
#: The pilot draws of the zero-gain pairs, the accuracy cells and the control.
DRAWS = 2_000
#: The pilot's own seed stream.
PILOT_SEED = 20263599
SUMMARY = Path(__file__).with_name("known_score_att_pilot.csv")

#: Each pilot block: the law's setting family, the n, the settings fitted on one draw and the
#: estimands each reports.
BLOCKS: dict[str, tuple[int, tuple[str, ...], tuple[str, ...]]] = {
    "calibration": (
        2_000,
        ("s_known_score", "s_arm_mean", "s_known_score_stacked"),
        ("att", "atc"),
    ),
    "zero_gain": (2_000, ("s0_known_score", "s0_arm_mean"), ("att", "atc")),
    "accuracy_s": (2_000, ("s_wrong_known_score",), ("att", "atc")),
    "accuracy_b2": (2_000, ("b2_wrong_known_score",), ("att", "atc")),
    "accuracy_b3": (
        2_000,
        ("b3_wrong_known_score",),
        ("att[1.0 vs 0.0]", "att[2.0 vs 0.0]", "atc[1.0 vs 0.0]", "atc[2.0 vs 0.0]"),
    ),
    "control": (2_000, ("s_declared_constant",), ("att",)),
    "rate_500": (500, ("s_known_score",), ("att",)),
    "rate_8000": (8_000, ("s_known_score",), ("att",)),
}
#: The declared draws of each block.
BLOCK_DRAWS = {
    "calibration": CALIBRATION_DRAWS,
    "rate_500": RATE_DRAWS,
    "rate_8000": RATE_DRAWS,
    **dict.fromkeys(("zero_gain", "accuracy_s", "accuracy_b2", "accuracy_b3", "control"), DRAWS),
}
#: The blocks whose known-score rows carry the paired ratio over the arm-mean fit.
PAIRED = {
    "calibration": ("s_known_score", "s_arm_mean"),
    "zero_gain": ("s0_known_score", "s0_arm_mean"),
}


def pilot_seed(block: str, draw: int) -> int:
    """The seed of one pilot draw of one block."""
    key = sum(ord(character) * 31**index for index, character in enumerate(block)) % 2**31
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(key, draw))
    return int(sequence.generate_state(1)[0])


def records(block: str, draw: int) -> list[dict[str, Any]]:
    """Every pilot row of one draw of one block."""
    n, settings, estimands = BLOCKS[block]
    frame = law.sample(law.SETTINGS[settings[0]], n, pilot_seed(block, draw))
    out = []
    for setting in settings:
        # The rungs fit the ATT alone, as the property phase's rate cells do.
        overrides = {"estimands": ("att",)} if block.startswith("rate_") else {}
        result = law.fit(frame, setting, **overrides)
        truth = law.setting_truth(setting)
        for name in estimands:
            estimate = result[name]
            low, high = estimate.ci
            out.append(
                {
                    "block": block,
                    "setting": setting,
                    "estimand": name,
                    "n": n,
                    "draw": draw,
                    "truth": truth[name],
                    "estimate": float(estimate.psi),
                    "std_error": float(estimate.std_error),
                    "covered": int(low <= truth[name] <= high),
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
    for (block, setting, estimand), group in rows.groupby(
        ["block", "setting", "estimand"], sort=True
    ):
        sd = float(group["estimate"].std(ddof=1))
        bias = float(group["estimate"].mean() - group["truth"].iloc[0])
        record = {
            "block": block,
            "setting": setting,
            "estimand": estimand,
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
            "paired_sd_ratio": np.nan,
            "paired_correlation": np.nan,
        }
        if block in PAIRED and setting == PAIRED[block][0]:
            wide = rows.loc[(rows["block"] == block) & (rows["estimand"] == estimand)].pivot(
                index="draw", columns="setting", values="estimate"
            )
            known, arm_mean = wide[PAIRED[block][0]], wide[PAIRED[block][1]]
            record["paired_sd_ratio"] = float(known.std(ddof=1) / arm_mean.std(ddof=1))
            record["paired_correlation"] = float(known.corr(arm_mean))
        out.append(record)
    return pd.DataFrame(out)


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
    rows = rows.drop_duplicates(["block", "setting", "estimand", "draw"])
    summarize(rows).to_csv(SUMMARY, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()

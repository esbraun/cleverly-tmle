"""The pre-read of ``option-ratio-trim`` (Fix-8 plan section 5.0), and its summary.

Before the declaration every gated cell is read on :data:`DRAWS` pilot draws of law ``LE`` at
the declared n and learners, with every reported fit beside it on the same draws.  The draws come
from :data:`PILOT_SEED`, a stream no study draws from, and each draw is fitted by
:func:`tests.studies._option_trim_law.fit` exactly as the property phase fits it.

The committed summary :data:`SUMMARY` holds one row per setting: the draws and their first and
last index, the mean estimate, its bias against the exact truth, the SD of the estimates, the
standardized bias, the mean plug-in SE (the diagnostic a trimmed fit keeps), the SE ratio, the
coverage of the plug-in interval and the mean trimmed share.  The design test reads the summary
(``tests/unit/test_option_ratio_trim_design.py``).

Run ``python -m tests.studies.option_ratio_trim_pilot run --start S --draws D --out FILE`` once
per block of draws, and ``python -m tests.studies.option_ratio_trim_pilot summarize FILE...``
to write the summary.  The per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import _option_trim_law as law

#: The pilot draws (plan 5.0: at least 2,000 for a bias cell).
DRAWS = 4_000
N = 2_000
#: The pilot's own seed stream.
PILOT_SEED = 20261116
SUMMARY = Path(__file__).with_name("option_ratio_trim_pilot.csv")


def pilot_seed(draw: int) -> int:
    """The seed of one pilot draw."""
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(draw,))
    return int(sequence.generate_state(1)[0])


def records(draw: int) -> list[dict[str, Any]]:
    """Every pilot row of one draw: one per setting."""
    frame = law.draw(N, pilot_seed(draw))
    out = []
    for setting in law.SETTINGS:
        result = law.fit(frame, setting)
        estimate = result.estimates[law.ESTIMAND]
        low, high = estimate.plugin_interval
        trim = result.fits[law.REGIMEN].ratio_trim
        out.append(
            {
                "setting": setting,
                "n": N,
                "draw": draw,
                "truth": law.TRUTH,
                "estimate": float(estimate.psi),
                "std_error": float(estimate.plugin_std_error),
                "covered": int(low <= law.TRUTH <= high),
                "trimmed_share": np.nan if trim is None else float(trim.trimmed_share),
            }
        )
    return out


def run(start: int, draws: int, out: Path) -> None:
    """Fit draws ``start`` to ``start + draws - 1`` and write their rows."""
    rows = [row for draw in range(start, start + draws) for row in records(draw)]
    pd.DataFrame(rows).to_parquet(out, index=False)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """One summary row per setting."""
    out = []
    for setting, group in rows.groupby("setting", sort=True):
        sd = float(group["estimate"].std(ddof=1))
        bias = float(group["estimate"].mean() - group["truth"].iloc[0])
        out.append(
            {
                "setting": setting,
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
                "trimmed_share": float(group["trimmed_share"].mean()),
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--start", type=int, required=True)
    runner.add_argument("--draws", type=int, required=True)
    runner.add_argument("--out", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("files", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        run(arguments.start, arguments.draws, arguments.out)
        return
    rows = pd.concat([pd.read_parquet(path) for path in arguments.files], ignore_index=True)
    rows = rows.drop_duplicates(["setting", "draw"])
    summarize(rows).to_csv(SUMMARY, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()

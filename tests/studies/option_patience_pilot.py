"""The pilot of ``option-patience`` (Fix-8 plan sections 5.0 and 5.5), and its summary.

Before the declaration the gated streams are read on :data:`DRAWS` pilot draws each, from draw
0 of :data:`PILOT_SEED`, a stream no study draws from, at the declared n and settings
(:func:`tests.studies._option_patience_laws.fit`):

* ``lc1``: the greedy path with patience 1 and 2 and the whole greedy path.  The stop shares
  choose plan 5.5's ``p*`` (``LC1_PATIENCE``), and the bias at ``p*`` reads the
  ``patience_bias`` positive before it runs.
* ``lc3``: the four declared LC3 fits.  The ordered fit with patience 2 gives the stop share
  ``p`` (the share of fits whose selection excludes ``W1``) and the conditional bias ``b`` on
  those fits, which size the ``ordered_late_confounder`` control by the mixture formula; the
  whole ordered path reads the ``ordered_whole_path`` positive; the greedy pair reads the
  paired ``greedy_late_confounder`` positive and the greedy whole-path bias.

The study has no calibration cell (selector paths withhold the interval, F18), so the 2,000
draws are the plan 5.0 pre-read size of a bias cell's pilot.  The committed summary
:data:`SUMMARY` holds one row per pilot fit; the design test reads it
(``tests/unit/test_option_patience_design.py``).

Run ``python -m tests.studies.option_patience_pilot run --stream S --start A --draws D --out
FILE`` once per block of draws, and ``python -m tests.studies.option_patience_pilot summarize
FILE...`` to write the summary.  The per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import _option_patience_laws as law

#: The pilot draws of each stream (plan 5.0: at least 2,000).
DRAWS = 2_000
#: The pilot's own seed stream.
PILOT_SEED = 20261115
SUMMARY = Path(__file__).with_name("option_patience_pilot.csv")

#: The fits of each pilot stream, fitted on one draw.
STREAM_FITS: dict[str, tuple[str, ...]] = {
    "lc1": ("lc1_greedy_patience_1", "lc1_greedy_patience_2", "lc1_greedy_whole"),
    "lc3": law.STREAM_FITS["lc3"],
}
#: Each fit's whole-path partner on the same draw.
WHOLE = {
    "lc1_greedy_patience_1": "lc1_greedy_whole",
    "lc1_greedy_patience_2": "lc1_greedy_whole",
    "lc1_greedy_whole": "lc1_greedy_whole",
    "lc3_ordered_patience": "lc3_ordered_whole",
    "lc3_ordered_whole": "lc3_ordered_whole",
    "lc3_greedy_patience": "lc3_greedy_whole",
    "lc3_greedy_whole": "lc3_greedy_whole",
}


def pilot_seed(stream: str, draw: int) -> int:
    """The seed of one pilot draw of one stream."""
    key = sum(ord(character) * 31**index for index, character in enumerate(stream)) % 2**31
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(key, draw))
    return int(sequence.generate_state(1)[0])


def records(stream: str, draw: int) -> list[dict[str, Any]]:
    """Every pilot row of one draw of one stream."""
    frame = law.draw(stream, law.N, pilot_seed(stream, draw))
    truth = law.truth(stream)["ate"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {name: law.fit(frame, name) for name in STREAM_FITS[stream]}
    out = []
    for name, result in fits.items():
        whole = fits[WHOLE[name]]
        out.append(
            {
                "stream": stream,
                "cell": name,
                "draw": draw,
                "truth": truth,
                "estimate": law.ate(result),
                "whole_estimate": law.ate(whole),
                "stopped": int(law.stopped(result, whole)),
                "excludes_confounder": int(stream == "lc3" and law.excludes_confounder(result)),
                "selected": ",".join(result.extra["ctmle"].selected_covariates),
            }
        )
    return out


def run(stream: str, start: int, draws: int, out: Path) -> None:
    """Fit draws ``start`` to ``start + draws - 1`` of a stream and write their rows."""
    rows = [row for draw in range(start, start + draws) for row in records(stream, draw)]
    pd.DataFrame(rows).to_csv(out, index=False)


def summarize(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One summary row per pilot fit."""
    rows = pd.concat(frames, ignore_index=True).drop_duplicates(["cell", "draw"])
    out = []
    for cell, group in rows.groupby("cell", sort=True):
        error = group["estimate"] - group["truth"]
        whole = rows.loc[rows["cell"] == WHOLE[str(cell)]].set_index("draw").loc[group["draw"]]
        whole_sd = float((whole["estimate"]).std(ddof=1))
        difference = group["estimate"].to_numpy() - group["whole_estimate"].to_numpy()
        excluded = group["excludes_confounder"].to_numpy(dtype=bool)
        conditional = error.to_numpy()[excluded]
        out.append(
            {
                "cell": cell,
                "draws": len(group),
                "first_draw": int(group["draw"].min()),
                "last_draw": int(group["draw"].max()),
                "truth": float(group["truth"].iloc[0]),
                "bias": float(error.mean()),
                "sd": float(group["estimate"].std(ddof=1)),
                "bias_over_sd": float(error.mean() / group["estimate"].std(ddof=1)),
                "whole_sd": whole_sd,
                "stop_share": float(group["stopped"].mean()),
                "exclude_share": float(excluded.mean()),
                "conditional_bias": float(conditional.mean()) if conditional.size else np.nan,
                "conditional_bias_sd": (
                    float(conditional.std(ddof=1)) if conditional.size > 1 else np.nan
                ),
                "paired_mean_over_whole_sd": float(difference.mean() / whole_sd),
                "paired_sd_over_whole_sd": float(difference.std(ddof=1) / whole_sd),
                "identical_share": float(np.mean(difference == 0.0)),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--stream", choices=sorted(STREAM_FITS), required=True)
    runner.add_argument("--start", type=int, required=True)
    runner.add_argument("--draws", type=int, required=True)
    runner.add_argument("--out", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("files", type=Path, nargs="+")
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments.stream, arguments.start, arguments.draws, arguments.out)
    else:
        summarize([pd.read_csv(path) for path in arguments.files]).to_csv(
            SUMMARY, index=False, float_format="%.6g"
        )


if __name__ == "__main__":
    main()

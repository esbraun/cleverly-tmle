"""The sizing probe of ``refuter-calibration-band``: the null and the defect at the declared size.

:mod:`tests.unit.test_refuter_calibration_band_design` reads the probe (``X82-J-plan.md``
section 5, AGENT-BRIEF "Power-cell sizing"). The probe runs each cell's own trial path
(:func:`tests.studies.refuter_calibration_band.fit_cell` and the cell's placebo) on
:data:`TRIALS` samples of ``n = 2,000`` rows, on seeds outside every declared stream, so it
cannot change a verdict.

==========================  =====================================================================
quantity                    what one trial records
==========================  =====================================================================
``fresh``                   the band-rank placebo, unmutated: 20 refits
``stale``                   the same with the study's :func:`~tests.studies.refuter_calibration_band.stale_rows`
                            at the declared share: 20 refits
==========================  =====================================================================

Each row records the mean and the SD of the 20 placebo estimates, their ratio ``shift``, the
``t`` statistic and whether the rule rejects. The mean ``shift`` of ``fresh`` measures the
placebo's bias in placebo SD at the declared ``n``: a classifier that cannot follow the band on
the placebo law, with an outcome model misspecified on that law, moves it (plan section 4,
review M4). The mean ``shift`` of ``stale`` is the defect's size in placebo SD. One trial's ratio
carries a Monte Carlo SD of about ``1 / sqrt(20) = 0.22``.

    python -m tests.diagnostics.x82_sj_sizing.run [--jobs 4]

writes ``reading.csv`` beside this file.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import refuter_calibration_band as study

HERE = Path(__file__).resolve().parent
#: Probe trials per cell.
TRIALS = {"PJ1": 100, "LJ1": 60}
SEED_BASE = 831_000_000


def _record(cell: str, quantity: str, trial: int, test: Any) -> dict[str, Any]:
    values = np.asarray(test.values, dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    reading = study._reading(test)
    return {
        "cell": cell,
        "quantity": quantity,
        "trial": trial,
        "n": study.N,
        "refits": int(values.size),
        "stale_share": study.STALE_SHARE[cell],
        "mean": mean,
        "sd": sd,
        "shift": mean / sd,
        "statistic": reading.statistic,
        "rejected": int(reading.rejected),
    }


def probe(key: tuple[str, int]) -> list[dict[str, Any]]:
    """Both quantities of one cell on one probe trial."""
    cell, trial = key
    spec = study.CELLS[cell]
    seed = SEED_BASE + 100_000 * list(study.CELLS).index(cell) + 10 * trial
    frame = study.draw_law(spec.law, study.N, seed)
    result = study.fit_cell(cell, frame, seed + 1)
    rows = [_record(cell, "fresh", trial, study._placebo(result, spec, seed + 2))]
    with study.stale_rows(study.STALE_SHARE[cell]):
        rows.append(_record(cell, "stale", trial, study._placebo(result, spec, seed + 3)))
    return rows


def main() -> None:
    """Run every cell and trial, and write ``reading.csv``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    keys = [(cell, trial) for cell, count in TRIALS.items() for trial in range(count)]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = [row for records in pool.map(probe, keys) for row in records]
    print(f"{time.perf_counter() - start:.0f} s on {args.jobs} workers", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "reading.csv", index=False, float_format="%.6f", lineterminator="\n")
    print(table.groupby(["cell", "quantity"])[["shift", "rejected"]].mean().to_string())


if __name__ == "__main__":
    main()

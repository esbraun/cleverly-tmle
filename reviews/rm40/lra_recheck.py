# A probe, not a registered study. RM40b step 0b, the law-M GLM-pi re-check.
#
# The audit's `piglm` row in `lra_summary.log` (n = 4,000, R = 150) shows an `ey[0]` bias of
# -0.257 SD (3.1 Monte Carlo SE) that its term proxy (-0.018) does not explain. This script
# re-runs `piglm` and `piknown` of `lra_mar_sim.py` at R = 1,000 on fresh draws: replicates 150
# to 1,149, which the audit never ran. Learners, law, seeds formula and analysis are those of
# `lra_mar_sim.py`; only the replicate range and the process pool are new.
#
# | item | value |
# | --- | --- |
# | command | `python reviews/rm40/lra_recheck.py run --procs 4`, then `python reviews/rm40/lra_recheck.py analyze` |
# | seeds | `lra_mar_sim.py`'s formula: `7_000_003 * (index(s) + 1) + 7919 * n + rep`, rep = 150 .. 1,149 |
# | output | `_lra_recheck/<setting>.csv` beside this script (not committed); the analysis is `lra_recheck.log` |
# | threads | one BLAS thread per process |
#
# Decision rule, fixed before the run (RM40 plan section 5.11):
#
# | outcome | criterion | consequence |
# | --- | --- | --- |
# | A | for every arm of `piglm`, the 99% interval of bias/SD, `b +/- 2.576 * MCSE`, contains 0 | the bias is within Monte Carlo error. The RM40b page states the projection argument for a correctly specified parametric `pi` with no measured excess |
# | B | otherwise | stop and report a defect suspect; RM40b removes the parametric-`pi` projection sentence that RM40a wrote |
#
# `piknown` is the H2 control (the term is identically 0) and is read the same way, for the
# record. The SE ratio and coverage of each arm are reported beside the rule and decide nothing.
"""Re-check the law-M arm-0 bias of the RM40 audit at R = 1,000 on fresh draws."""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lra_mar_sim as sim  # noqa: E402

N = 4_000
FIRST = 150
REPLICATES = 1_000
SETTINGS = ("piglm", "piknown")
OUT = HERE / "_lra_recheck"


def _one(payload: tuple[str, int]) -> list[dict[str, Any]]:
    setting, rep = payload
    return sim.one(setting, N, rep)


def run(procs: int) -> None:
    OUT.mkdir(exist_ok=True)
    with Pool(procs, initializer=sim.init) as pool:
        for setting in SETTINGS:
            payloads = [(setting, rep) for rep in range(FIRST, FIRST + REPLICATES)]
            rows = [row for chunk in pool.imap(_one, payloads, chunksize=8) for row in chunk]
            pd.DataFrame(rows).to_csv(OUT / f"{setting}.csv", index=False)
            print(f"{setting}: {len(payloads)} fits written", flush=True)


def analyze() -> None:
    lines = []
    outcome_a = True
    for setting in SETTINGS:
        frame = pd.read_csv(OUT / f"{setting}.csv")
        errors = int((frame["estimand"] == "error").sum())
        frame = frame.loc[frame["estimand"] != "error"]
        for name, group in frame.groupby("estimand", sort=True):
            r = len(group)
            sd = float(group["psi"].std(ddof=1))
            b = float((group["psi"] - group["truth"]).mean()) / sd
            mcse = 1.0 / np.sqrt(r)
            low, high = b - 2.576 * mcse, b + 2.576 * mcse
            contains = low <= 0.0 <= high
            if setting == "piglm" and name != "ate" and not contains:
                outcome_a = False
            lines.append(
                f"{setting:8s} {name:6s} R={r:5d} errors={errors} bias/SD={b:+.3f} "
                f"99%=[{low:+.3f}, {high:+.3f}] contains0={contains} "
                f"term/SD={float(group['term'].mean()) / sd:+.3f} "
                f"SEratio={float(group['se'].mean()) / sd:.3f} "
                f"coverage={float(group['covered'].mean()):.3f} "
                f"secs/fit={float(group['secs'].mean()):.2f}"
            )
    lines.append(f"outcome: {'A' if outcome_a else 'B'}")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 2)
    else:
        analyze()

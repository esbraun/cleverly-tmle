"""Sweep the TWINS point-treatment pipeline over pair-sample seeds.

Usage: python sweep.py <booster> <output.csv> <seed> [<seed> ...]
       python sweep.py <booster> <output.csv> --range <first> <last>

Each seed runs in its own single-threaded worker process, ten at a time.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import pandas as pd


def _one(args):
    booster, seed = args
    import common

    return common.run(seed, booster=booster)


def main() -> None:
    booster, output = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    if rest[0] == "--range":
        seeds = list(range(int(rest[1]), int(rest[2]) + 1))
    else:
        seeds = [int(seed) for seed in rest]
    with ProcessPoolExecutor(max_workers=10) as pool:
        records = list(pool.map(_one, [(booster, seed) for seed in seeds]))
    frame = pd.DataFrame(records).sort_values("seed")
    frame.to_csv(output, index=False, lineterminator="\n")
    columns = [
        "seed", "gap_over_se", "max_score_after", "nu2", "cf_g_min", "propensity_cal_slope",
        "outcome_cal_slope", "rv", "bench_cf_y", "bench_cf_d", "sensitivity_ran", "error",
    ]
    print(frame[[c for c in columns if c in frame]].to_string(index=False))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

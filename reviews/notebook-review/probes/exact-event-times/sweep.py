"""Seed sweep of the fits that docs/examples/exact-event-times.ipynb shows.

Each seed ``s`` draws ``make_exact_readmission(n=1000, seed=s)`` and runs the page's two fits
with ``random_state=0``: the two-event fit with target times 1.5 and 3 months and a restricted
cut-off at 3, and the one-event fit with deaths counted as leaving follow-up.  It also records the
simple difference in the share with a recorded readmission by month 3.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 100 100 3``.  Output: ``sweep.csv`` (one row per seed), which ``summarize.py``
reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cleverly import CausalStudy, RegimeContrast, TimeToEvent
from cleverly.datasets import EXACT_ARMS, make_exact_readmission

HERE = Path(__file__).resolve().parent
HORIZONS = (1.5, 3.0)
TWO = TimeToEvent(
    time="time",
    event="event",
    treatment="offered",
    baseline=("risk_score",),
    grid="continuous",
    causes={1: "readmission", 2: "death"},
)
ONE = TimeToEvent(
    time="time",
    event="event",
    treatment="offered",
    baseline=("risk_score",),
    grid="continuous",
    causes={1: "readmission"},
)
#: The estimates of the two-event fit the page reads.
TWO_ROWS = (
    "ate_regimen[offered vs not offered, readmission @ t=3]",
    "ate_regimen[offered vs not offered, death @ t=3]",
    "rr_regimen[offered vs not offered, readmission @ t=3]",
    "rmst_regimen[offered vs not offered @ t=3]",
    "rmst_regimen[offered @ t=3]",
)
#: The estimates of the one-event fit the page reads.
ONE_ROWS = ("risk_regimen[offered @ t=3]", "risk_regimen[not offered @ t=3]")


def one_seed(seed: int) -> dict[str, float]:
    warnings.filterwarnings("ignore")
    frame, _ = make_exact_readmission(n=1_000, seed=seed)
    contrast = RegimeContrast(
        EXACT_ARMS, reference="not offered", horizons=HORIZONS, restricted_horizons=(3.0,)
    )
    two = CausalStudy(frame, design=TWO).estimate(contrast, random_state=0).to_frame()
    deaths_as_dropouts = frame.assign(event=np.where(frame["event"].eq(1), 1, 0))
    one = (
        CausalStudy(deaths_as_dropouts, design=ONE)
        .estimate(
            RegimeContrast(EXACT_ARMS, reference="not offered", horizons=HORIZONS), random_state=0
        )
        .to_frame()
    )
    row: dict[str, float] = {"seed": seed}
    for prefix, table, names in (("two", two, TWO_ROWS), ("one", one, ONE_ROWS)):
        table = table.set_index("estimand")
        for name in names:
            for column in ("psi", "std_err", "ci_lower", "ci_upper"):
                row[f"{prefix}:{name}:{column}"] = float(table.loc[name, column])
    recorded = frame["event"].eq(1) & frame["time"].le(3.0)
    shares = recorded.groupby(frame["offered"]).mean()
    row["simple_readmission_difference"] = float(shares[1] - shares[0])
    return row


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    rows = Parallel(n_jobs=workers)(delayed(one_seed)(s) for s in range(first, first + count))
    pd.DataFrame(rows).to_csv(HERE / "sweep.csv", index=False)


if __name__ == "__main__":
    main()

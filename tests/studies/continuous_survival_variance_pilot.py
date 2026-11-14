"""The pilot of the ``continuous-survival-variance`` study, on its own seed stream.

It draws no declared sample: draw ``r`` of cell ``c`` takes ``stream_seed(STUDY, "pilot", c, r)``,
a stream no declared cell reads.  Each draw is one fit of the cell's declared configuration
(:func:`tests.studies.continuous_survival_variance.engine`), and the driver writes one row per fit
and target (all eighteen, first quartile included) to ``continuous_survival_variance_pilot.csv.gz``
beside this module, or one row with the error class when the fit raised.

The committed rows give three readings of X6 plan rev 5, section 5.1, which the regeneration
script checks before a declared run:

* **V7's coefficient** (:func:`v7_halving`): if the pilot's influence-curve interval covers any
  cause-1 target at the median or upper quartile below 0.93, the coefficient halves once, to 0.15,
  before the declaration;
* **V7's non-null witness** (:func:`v7_witness`): the untargeted plug-in of each arm's cause-1 risk
  at the median has a standardized bias whose largest magnitude must exceed 0.2 SD, or V7 is
  re-declared as reported rather than gated;
* **the first-quartile plug-in check** (:func:`plug_in_checks`): each first-quartile row's
  calibration pass probability at the declared 3,600 replicates, taking the pilot's SE ratio of
  the reported (switched) interval as true.  ``guide/studies.md`` extends the pilot of a cell with
  a row below 0.90 (:func:`extension_cells`) and diagnoses a confirmed departure before the
  declaration; the cell's count can rise to the cap of 7,200.

Run it as ``python -m tests.studies.continuous_survival_variance_pilot --jobs N [--cells c ...]``.
It writes the file after every block of :data:`BLOCK` draws, so ``--resume`` continues each cell
from its first draw the file does not hold, on the same streams; ``--draws`` above 500 extends.
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_law as base
from tests.studies import continuous_survival_variance as study
from tests.studies import continuous_survival_variance_law as laws
from tests.studies import repeated_sizing as sizing
from tests.studies.continuous_survival_variance_readings import shipped_interval
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("continuous_survival_variance_pilot.csv.gz")
DRAWS = study.PILOT_REPLICATES
BLOCK = 100
#: The V7 halving rule's coverage threshold (plan 5.1).
HALVING_COVERAGE = 0.93
#: The V7 witness's standardized-bias threshold (plan 5.1).
WITNESS_BIAS = 0.2
#: The plug-in pass probability below which ``guide/studies.md`` extends a cell's pilot.
PLUG_IN_PASS = 0.90


def pilot_seed(cell: str, replicate: int) -> int:
    """The seed of pilot draw ``replicate`` of ``cell``."""
    return stream_seed(study.STUDY, "pilot", cell, replicate)


def _one(key: tuple[str, int]) -> list[dict[str, Any]]:
    cell, replicate = key
    name, n, _ = study.CELLS[cell]
    law = laws.LAWS[name]
    truth = laws.truth(name)
    frame = law.sample(n, pilot_seed(cell, replicate))
    head = {"label": cell, "replicate": replicate, "n": n}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = registered.fit_with(frame, study.engine(cell))
            seconds = time.perf_counter() - start
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{**head, "refusal": type(error).__name__}]
    initial = result.initial_estimates
    rows = []
    for estimand in truth:
        estimate = result[estimand]
        record = estimate.switched
        rows.append(
            {
                **head,
                "estimand": estimand,
                "early": estimand in laws.first_quartile(name),
                "truth": float(truth[estimand]),
                "estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ic_std_error": float(np.sqrt(record.influence_variance)),
                "pv_std_error": float(np.sqrt(max(record.predictable_variance, 0.0))),
                "variance_rel_sd": float(record.relative_sd),
                "initial_estimate": float(initial[estimand]),
                "refusal": "",
                "seconds": seconds,
            }
        )
    return rows


def run(
    n_jobs: int, cells: Sequence[str], *, draws: int = DRAWS, resume: bool = False
) -> pd.DataFrame:
    """Run each cell's draws up to ``draws``, writing the file after every block.

    With ``resume`` each cell starts at its first draw the file does not hold, and other cells'
    rows are kept; without it, the file is replaced.
    """
    frame = pd.read_csv(OUTPUT, low_memory=False) if resume and OUTPUT.exists() else pd.DataFrame()
    for cell in cells:
        held = frame.loc[frame["label"] == cell] if not frame.empty else frame
        first = int(held["replicate"].max()) + 1 if not held.empty else 0
        if not held.empty and set(held["replicate"].astype(int)) != set(range(first)):
            raise SystemExit(f"{OUTPUT.name} does not hold draws 0 to {first - 1} of {cell}")
        for start in range(first, draws, BLOCK):
            keys = [(cell, r) for r in range(start, min(start + BLOCK, draws))]
            outcomes = map_parallel(_one, keys, n_jobs=n_jobs)
            block = pd.DataFrame([row for rows in outcomes for row in rows])
            frame = pd.concat([frame, block], ignore_index=True)
            frame.to_csv(OUTPUT, index=False)
            print(f"{cell}: {min(start + BLOCK, draws)} of {draws} draws written", flush=True)
    return frame


# -------------------------------------------------------------------------- readings


def read() -> pd.DataFrame:
    """The committed pilot rows, or an empty frame before the pilot is committed."""
    return pd.read_csv(OUTPUT, low_memory=False) if OUTPUT.exists() else pd.DataFrame()


def kept_rows(rows: pd.DataFrame, cell: str) -> pd.DataFrame:
    """The rows of ``cell``'s fits that did not raise."""
    chosen = rows.loc[rows["label"] == cell]
    return chosen.loc[chosen["refusal"].fillna("").eq("")]


def ic_coverage(rows: pd.DataFrame, cell: str) -> pd.Series:
    """Each target's coverage by the influence-curve interval of the shipped scale."""
    kept = kept_rows(rows, cell)
    low, high = shipped_interval(
        kept["estimate"].to_numpy(),
        kept["ic_std_error"].to_numpy(),
        kept["estimand"].str.startswith("cif_regimen").to_numpy(),
    ).T
    covered = (low <= kept["truth"].to_numpy()) & (kept["truth"].to_numpy() <= high)
    return pd.Series(covered, index=kept.index).groupby(kept["estimand"]).mean()


def v7_halving(rows: pd.DataFrame) -> bool:
    """Whether the IC covers any cause-1 target of V7 at the median or upper quartile below 0.93."""
    coverage = ic_coverage(rows, "v7_quadratic")
    gated = [name for name in laws.gated("quadratic") if f", {base.CAUSES[1]} @" in name]
    return bool((coverage[gated] < HALVING_COVERAGE).any())


def v7_witness(rows: pd.DataFrame) -> float:
    """The largest |standardized bias| of the plug-in of an arm's cause-1 risk at V7's median."""
    kept = kept_rows(rows, "v7_quadratic")
    median = laws.LAWS["quadratic"].times[1]
    names = [base.level_name(arm, base.CAUSES[1], median) for arm in base.ARMS]
    out = 0.0
    for name in names:
        chosen = kept.loc[kept["estimand"] == name]
        error = chosen["initial_estimate"] - chosen["truth"]
        out = max(out, abs(float(error.mean() / error.std(ddof=1))))
    return out


def plug_in_checks(rows: pd.DataFrame, replicates: int = study.PRIMARY_REPLICATES) -> pd.DataFrame:
    """Each first-quartile row's SE ratio and plug-in calibration pass probability."""
    records = []
    for cell in study.CELLS:
        kept = kept_rows(rows, cell)
        for name, group in kept.loc[kept["early"].astype(bool)].groupby("estimand"):
            ratio = float(group["std_error"].mean() / group["estimate"].std(ddof=1))
            passed = float(sizing.calibration_pass(study.MARGINS, replicates, np.array([ratio]))[0])
            records.append(
                {
                    "cell": cell,
                    "estimand": name,
                    "draws": int(group["replicate"].nunique()),
                    "se_ratio": ratio,
                    "plug_in_pass": passed,
                }
            )
    return pd.DataFrame(records)


def extension_cells(rows: pd.DataFrame) -> tuple[str, ...]:
    """The cells holding a first-quartile row whose plug-in pass is below 0.90."""
    checks = plug_in_checks(rows)
    low = checks.loc[checks["plug_in_pass"] < PLUG_IN_PASS, "cell"]
    return tuple(cell for cell in study.CELLS if cell in set(low))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--cells", nargs="*", default=list(study.CELLS))
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.cells, draws=arguments.draws, resume=arguments.resume)

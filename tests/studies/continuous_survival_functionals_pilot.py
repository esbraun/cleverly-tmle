"""The sizing pilot of the ``continuous-survival-functionals`` study, on its own seed stream.

It draws no declared sample: draw ``r`` takes ``stream_seed(STUDY, "pilot", LABEL, r)``, a
stream no declared cell reads.  Each draw is one fit of the primary configuration (the default
cross-fitted fit with ``restricted_horizons=(2.32,)`` at n = 1,000), and the driver writes one row
per fit and gated target to ``continuous_survival_functionals_pilot.csv.gz`` beside this module,
or one row with the error class when the fit raised.

The committed rows size both instruments (``tests/unit/test_continuous_survival_functionals_design.py``):

* the primary count, by :func:`~tests.studies.repeated_sizing.primary_replicates` on the 25
  gated targets' covered counts and standardized biases (:func:`primary_count`);
* the calibration cell, whose fit and estimand (the RMST difference) are the primary's, by the
  at-target rule of ``guide/studies.md``, with the plug-in check at the pilot's own SE ratio
  (:func:`calibration_check`).  The first 800 draws read a plug-in pass probability of 0.54 at
  3,200 replicates (SE ratio 1.035), so the guide's rule extended the pilot to 3,200 draws on
  the same stream.

Run it as ``python -m tests.studies.continuous_survival_functionals_pilot --jobs N``.  It writes
the file after every block of :data:`BLOCK` draws, so ``--resume`` continues an interrupted run
from the first draw the file does not hold, on the same streams.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_functionals as study
from tests.studies import continuous_survival_functionals_law as functionals
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.schema import truth_on_inference_scale
from tests.studies.evidence.seeds import stream_seed

OUTPUT = Path(__file__).with_name("continuous_survival_functionals_pilot.csv.gz")
LABEL = study.GATED_CELL
DRAWS = study.PILOT_REPLICATES
#: Draws per written block, so an interrupted run loses at most one block.
BLOCK = 100


def pilot_seed(replicate: int) -> int:
    """The seed of pilot draw ``replicate``."""
    return stream_seed(study.STUDY, "pilot", LABEL, replicate)


def _one(replicate: int) -> list[dict[str, Any]]:
    frame = registered.sample(study.PRIMARY_N, pilot_seed(replicate))
    base = {"label": LABEL, "replicate": replicate, "n": study.PRIMARY_N}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = study.fit(frame)
            seconds = time.perf_counter() - start
    except Exception as error:  # a raise is a pilot finding, recorded with its class
        return [{**base, "refusal": type(error).__name__}]
    extra = {"seconds": seconds, **registered.readings(result)}
    rows = study.result_rows(result, functionals.TRUTH, LABEL, replicate)
    return [
        {
            **base,
            "estimand": row["estimand"],
            "truth": row["truth"],
            "estimate": row["estimate"],
            "inference_estimate": row["inference_estimate"],
            "inference_scale": row["inference_scale"],
            "std_error": row["std_error"],
            "covered": row["covered"],
            "initial_estimate": row["initial_estimate"],
            "refusal": "",
            **extra,
        }
        for row in rows
    ]


def run(n_jobs: int, *, draws: int = DRAWS, resume: bool = False) -> pd.DataFrame:
    """Run draws up to ``draws``, writing the file after every block.

    With ``resume`` the run starts at the first draw the file does not hold; without it, the
    file is replaced.
    """
    kept = pd.read_csv(OUTPUT, low_memory=False) if resume and OUTPUT.exists() else pd.DataFrame()
    first = int(kept["replicate"].max()) + 1 if not kept.empty else 0
    if not kept.empty:
        expected = set(range(first))
        if set(kept["replicate"].astype(int)) != expected:
            raise SystemExit(f"{OUTPUT.name} does not hold draws 0 to {first - 1}; rerun fresh")
    frame = kept
    for start in range(first, draws, BLOCK):
        outcomes = map_parallel(_one, list(range(start, min(start + BLOCK, draws))), n_jobs=n_jobs)
        block = pd.DataFrame([row for rows in outcomes for row in rows])
        frame = pd.concat([frame, block], ignore_index=True)
        frame.to_csv(OUTPUT, index=False)
        print(f"{min(start + BLOCK, draws)} of {draws} draws written to {OUTPUT}", flush=True)
    return frame


# ---------------------------------------------------------------------------- sizing


def read() -> pd.DataFrame:
    """The committed pilot rows, or an empty frame before the pilot is committed."""
    return pd.read_csv(OUTPUT, low_memory=False) if OUTPUT.exists() else pd.DataFrame()


def kept_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows of the fits that did not raise."""
    return rows.loc[rows["refusal"].fillna("").eq("")]


def primary_count(rows: pd.DataFrame) -> int:
    """:func:`~tests.studies.repeated_sizing.primary_replicates` on the gated targets.

    Clauses (a) to (c) on the covered counts and clause (d) on the standardized biases, read
    on each row's inference scale (the log for a ratio), as the record's bias gate reads them.
    """
    names = list(functionals.GATED)
    kept = kept_rows(rows)
    hits = kept.pivot(index="replicate", columns="estimand", values="covered")[names]
    covered = {f"{LABEL}/{name}": int(hits[name].sum()) for name in names}
    truth = kept.apply(
        lambda row: truth_on_inference_scale(
            row["estimand"], float(row["truth"]), str(row["inference_scale"])
        ),
        axis=1,
    )
    error = kept.assign(error=kept["inference_estimate"] - truth).pivot(
        index="replicate", columns="estimand", values="error"
    )[names]
    standardized = error / error.std(ddof=1)
    group = sizing.PilotGroup(
        standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
    )
    share = sizing.PRIMARY_CLAUSE_SHARE
    return sizing.primary_replicates(
        study.MARGINS,
        covered,
        len(hits),
        implementations=1,
        coverage_pass=share,
        bias_groups=[group],
        bias_pass=share,
    )


def at_target_count() -> int:
    """The smallest multiple of 400 whose pass probability at SE ratio 1 is at least 0.95."""
    replicates = sizing.BUDGET_STEP
    while (
        float(sizing.calibration_pass(study.MARGINS, replicates, np.array([1.0]), 0.95)[0]) < 0.95
    ):
        replicates += sizing.BUDGET_STEP
    return replicates


def calibration_ratio(rows: pd.DataFrame) -> float:
    """The pilot SE ratio of the RMST difference: mean reported SE over the empirical SD."""
    rmst = kept_rows(rows)
    rmst = rmst.loc[rmst["estimand"] == functionals.rmst_difference_name()]
    return float(rmst["std_error"].mean() / rmst["estimate"].std(ddof=1))


def calibration_check(rows: pd.DataFrame, replicates: int) -> tuple[float, float]:
    """The calibration cell's pass probability at ``replicates``: at target, and plug-in.

    The plug-in reading takes the pilot's SE ratio as true, with the Wald coverage it implies.
    ``guide/studies.md`` asks for at least 0.90 there, or a longer pilot.
    """
    at_target = float(sizing.calibration_pass(study.MARGINS, replicates, np.array([1.0]), 0.95)[0])
    ratio = calibration_ratio(rows)
    plug_in = float(sizing.calibration_pass(study.MARGINS, replicates, np.array([ratio]))[0])
    return at_target, plug_in


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run(arguments.jobs, draws=arguments.draws, resume=arguments.resume)

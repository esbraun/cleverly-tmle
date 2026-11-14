"""The pilot of the ``continuous-survival-variance`` study, on its own seed stream.

It draws no declared sample: draw ``r`` of cell ``c`` takes ``stream_seed(STUDY, "pilot", c, r)``,
a stream no declared cell reads.  Each draw is one fit of the cell's declared configuration
(:func:`tests.studies.continuous_survival_variance.engine`), and the driver writes one row per fit
and target (all eighteen, first quartile included) to ``continuous_survival_variance_pilot.csv.gz``
beside this module, or one row with the error class when the fit raised.  Every row records the
V7 coefficient of the law it was drawn from (``quadratic``; 0 for the other cells), so no reading
mixes draws from two laws.

The committed rows give these readings of X6 plan rev 5, section 5.1, which the regeneration
script checks before a declared run (``regenerate.pilot_refusals``):

* **the count** (:func:`primary_count`): the larger of the shared primary rule on the 60 gated
  rows and the at-target count of the 30 first-quartile calibration rows (3,600: family pass
  0.977 at SE ratio 1 and coverage 0.95), read on each cell's first 1,000 draws;
* **V7's coefficient** (:func:`v7_halving`, on the draws at 0.3 only): if the pilot's
  influence-curve interval covers any cause-1 target at the median or upper quartile below 0.93,
  the coefficient halves once, to 0.15, and V7's pilot is rerun at 0.15.  The point coverage of
  1,000 draws falls below 0.93 with probability 0.002 per row at a true 0.95 and 0.084 at a true
  0.94; over the six correlated rows at most 0.014 at 0.95 (union bound).  A false fire errs toward
  a milder V7, which makes the witness more likely to fail;
* **V7's non-null witness** (:func:`v7_witness`, on the draws at the declared coefficient): the
  untargeted plug-in of each arm's cause-1 risk at the median has a standardized bias whose
  largest magnitude must exceed 0.2 SD, or V7 is reported rather than gated
  (:data:`~tests.studies.continuous_survival_variance.V7_GATED`);
* **the failure share** (:func:`failure_shares`): a cell that loses more than the failure bound
  of 0.01 is diagnosed before the declaration;
* **the first-quartile plug-in check** (:func:`plug_in_checks`): each first-quartile row's
  calibration pass probability at the declared count, taking the pilot's SE ratio of the reported
  (switched) interval as true.  A cell with a row below 0.90 on its first 1,000 draws
  (:func:`extension_cells`) is extended to 2,000 draws and re-read; a row still below 0.90 is a
  confirmed departure, which needs a committed diagnosis
  (``tests/canonical/continuous_survival_variance/pilot-diagnosis.md`` naming the row) before
  the declaration (``guide/studies.md``).

Run it as ``python -m tests.studies.continuous_survival_variance_pilot --jobs N [--cells c ...]``.
It writes the file after every block of :data:`BLOCK` draws, so ``--resume`` continues each cell
from its first draw the file does not hold, on the same streams; ``--draws 2000 --cells c
--resume`` extends one cell.
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
#: V7's coefficient before the halving rule (plan 5.1).
V7_INITIAL = 0.3
#: The committed diagnosis a confirmed first-quartile departure needs before the declaration.
DIAGNOSIS = study.STUDY.artifacts / "pilot-diagnosis.md"


def pilot_seed(cell: str, replicate: int) -> int:
    """The seed of pilot draw ``replicate`` of ``cell``."""
    return stream_seed(study.STUDY, "pilot", cell, replicate)


def _one(key: tuple[str, int]) -> list[dict[str, Any]]:
    cell, replicate = key
    name, n, _ = study.ALL_CELLS[cell]
    law = laws.LAWS[name]
    truth = laws.truth(name)
    frame = law.sample(n, pilot_seed(cell, replicate))
    head = {"label": cell, "replicate": replicate, "n": n, "quadratic": law.quadratic}
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

    With ``resume`` each cell starts at its first draw the file does not hold at the cell's
    current law, and every other row is kept; without it, the file is replaced.
    """
    frame = pd.read_csv(OUTPUT, low_memory=False) if resume and OUTPUT.exists() else pd.DataFrame()
    for cell in cells:
        coefficient = study.law_of(cell).quadratic
        held = at_law(frame, cell, coefficient) if not frame.empty else frame
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


def at_law(rows: pd.DataFrame, cell: str, coefficient: float | None = None) -> pd.DataFrame:
    """``cell``'s rows drawn at V7 coefficient ``coefficient`` (the declared one by default)."""
    if coefficient is None:
        coefficient = study.law_of(cell).quadratic
    chosen = rows.loc[rows["label"] == cell]
    return chosen.loc[np.isclose(chosen["quadratic"].astype(float), coefficient)]


def kept_rows(rows: pd.DataFrame, cell: str, coefficient: float | None = None) -> pd.DataFrame:
    """The rows of ``cell``'s fits at the coefficient that did not raise."""
    chosen = at_law(rows, cell, coefficient)
    return chosen.loc[chosen["refusal"].fillna("").eq("")]


def draws(rows: pd.DataFrame, cell: str) -> int:
    """The draws ``cell`` holds at its declared law, raised or not."""
    return int(at_law(rows, cell)["replicate"].nunique())


def _covered(kept: pd.DataFrame, column: str) -> np.ndarray:
    low, high = shipped_interval(
        kept["estimate"].to_numpy(),
        kept[column].to_numpy(),
        kept["estimand"].str.startswith("cif_regimen").to_numpy(),
    ).T
    truth = kept["truth"].to_numpy()
    return (low <= truth) & (truth <= high)


def ic_coverage(rows: pd.DataFrame, cell: str, coefficient: float | None = None) -> pd.Series:
    """Each target's coverage by the influence-curve interval of the shipped scale."""
    kept = kept_rows(rows, cell, coefficient)
    covered = _covered(kept, "ic_std_error")
    return pd.Series(covered, index=kept.index).groupby(kept["estimand"]).mean()


def v7_halving(rows: pd.DataFrame) -> bool | None:
    """Whether the IC covers a cause-1 target of V7 at the median or upper quartile below 0.93.

    Read on the draws at the initial 0.3 only; ``None`` when the pilot holds none.
    """
    if kept_rows(rows, "v7_quadratic", V7_INITIAL).empty:
        return None
    coverage = ic_coverage(rows, "v7_quadratic", V7_INITIAL)
    gated = [name for name in laws.gated("quadratic") if f", {base.CAUSES[1]} @" in name]
    return bool((coverage[gated] < HALVING_COVERAGE).any())


def v7_witness(rows: pd.DataFrame) -> float | None:
    """The largest |standardized bias| of the plug-in of an arm's cause-1 risk at V7's median.

    Read on the draws at the declared coefficient, with that law's truths; ``None`` when the
    pilot holds none.
    """
    kept = kept_rows(rows, "v7_quadratic")
    if kept.empty:
        return None
    median = laws.LAWS["quadratic"].times[1]
    names = [base.level_name(arm, base.CAUSES[1], median) for arm in base.ARMS]
    out = 0.0
    for name in names:
        chosen = kept.loc[kept["estimand"] == name]
        error = chosen["initial_estimate"] - chosen["truth"]
        out = max(out, abs(float(error.mean() / error.std(ddof=1))))
    return out


def failure_shares(rows: pd.DataFrame) -> dict[str, float]:
    """Each record cell's share of pilot fits that raised, at its declared law."""
    out = {}
    for cell in study.CELLS:
        chosen = at_law(rows, cell)
        raised = chosen.loc[chosen["refusal"].fillna("").ne(""), "replicate"].nunique()
        out[cell] = raised / max(int(chosen["replicate"].nunique()), 1)
    return out


def primary_count(rows: pd.DataFrame) -> int:
    """The declared count per cell from the pilot's first 1,000 draws of each record cell.

    The larger of :func:`~tests.studies.repeated_sizing.primary_replicates` (clauses (a) to (c)
    on the gated rows' covered counts by the shipped interval, clause (d) on one bias group per
    cell, both at :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`) and
    :func:`at_target_count`.
    """
    covered: dict[str, int] = {}
    groups = []
    for cell in study.CELLS:
        kept = kept_rows(rows, cell)
        kept = kept.loc[kept["replicate"] < study.PILOT_REPLICATES]
        names = list(study.ESTIMANDS[cell])
        gated = kept.loc[kept["estimand"].isin(names)]
        hits = pd.Series(_covered(gated, "std_error"), index=gated.index)
        counts = hits.groupby(gated["estimand"]).sum()
        covered.update({f"{cell}/{name}": int(counts[name]) for name in names})
        error = gated.assign(error=gated["estimate"] - gated["truth"]).pivot(
            index="replicate", columns="estimand", values="error"
        )[names]
        standardized = error / error.std(ddof=1)
        groups.append(
            sizing.PilotGroup(
                standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(error)
            )
        )
    share = sizing.PRIMARY_CLAUSE_SHARE
    rule = sizing.primary_replicates(
        study.MARGINS,
        covered,
        study.PILOT_REPLICATES,
        implementations=1,
        coverage_pass=share,
        bias_groups=groups,
        bias_pass=share,
    )
    return max(rule, at_target_count())


def at_target_count() -> int:
    """The smallest multiple of 400 at which the first-quartile rows pass together with 0.95.

    Each row at SE ratio 1 and coverage 0.95, under the calibration rule; the family is the six
    first-quartile rows of every record cell.
    """
    rows = 6 * len(study.CELLS)
    replicates = sizing.BUDGET_STEP
    while (
        float(sizing.calibration_pass(study.MARGINS, replicates, np.array([1.0]), 0.95)[0]) ** rows
        < sizing.FAMILY_WISE_PASS
    ):
        replicates += sizing.BUDGET_STEP
    return replicates


def plug_in_checks(rows: pd.DataFrame, replicates: int, *, first: bool = False) -> pd.DataFrame:
    """Each first-quartile row's SE ratio and plug-in calibration pass probability.

    ``first`` reads each cell's first 1,000 draws only (the extension trigger); otherwise every
    draw at the declared law (the re-read after an extension).
    """
    records = []
    for cell in study.CELLS:
        kept = kept_rows(rows, cell)
        if first:
            kept = kept.loc[kept["replicate"] < study.PILOT_REPLICATES]
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


def extension_cells(rows: pd.DataFrame, replicates: int) -> tuple[str, ...]:
    """The cells holding a first-quartile row whose plug-in pass on 1,000 draws is below 0.90."""
    checks = plug_in_checks(rows, replicates, first=True)
    low = set(checks.loc[checks["plug_in_pass"] < PLUG_IN_PASS, "cell"])
    return tuple(cell for cell in study.CELLS if cell in low)


def confirmed_departures(rows: pd.DataFrame, replicates: int) -> list[str]:
    """``cell/estimand`` of each extended cell's row still below 0.90 on every draw."""
    checks = plug_in_checks(rows, replicates)
    flagged = set(extension_cells(rows, replicates))
    low = checks.loc[checks["cell"].isin(flagged) & (checks["plug_in_pass"] < PLUG_IN_PASS)]
    return [f"{cell}/{name}" for cell, name in zip(low["cell"], low["estimand"], strict=True)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--cells", nargs="*", default=list(study.ALL_CELLS))
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    run(arguments.jobs, arguments.cells, draws=arguments.draws, resume=arguments.resume)

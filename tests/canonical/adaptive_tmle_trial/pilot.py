"""The sizing pilot of ``adaptive-tmle-trial`` (and, with ``--study selective``, its sibling).

The pilot fits every scenario of the record at its declared ``n`` on seeds that no declared run
reads (:data:`PILOT_SEEDS`), writes nothing inside the repository, and prints the readings the
design test pins: the covered count of each scenario's gated interval, the SE ratio, the
standardized bias, every control row (the efficiency contrast among them) and the seconds per
replicate. It then prints the replication count that ``repeated_sizing.primary_replicates``
gives for those counts (clause (c) prices each cell at the one-sided 80% lower bound of its
pilot coverage), which is the declared ``PRIMARY_REPLICATES``. A refused replicate is missing
from the rows and so counts as not covering.

```bash
python -m tests.canonical.adaptive_tmle_trial.pilot --replicates 800 --jobs 4 > pilot.json
```
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

from tests.studies import adaptive_tmle_selective, adaptive_tmle_trial
from tests.studies import repeated_sizing as sizing

#: The record seeds of the pilots: neither study's declared seeds, nor the T3 pilot's
#: (20290399, 20290397, 20290393).
PILOT_SEEDS = {"trial": (20290401, 20290402), "selective": (20290403, 20290404)}


def pilot(study: str, replicates: int, jobs: int) -> dict[str, object]:
    """Fit the pilot of ``study`` and return its readings."""
    module = adaptive_tmle_trial if study == "trial" else adaptive_tmle_selective
    seed, resampling = PILOT_SEEDS[study]
    record = replace(module.STUDY, seed=seed, resampling_seed=resampling)
    begin = time.perf_counter()
    phase = adaptive_tmle_trial.draw_and_fit(
        replicates=replicates, n=record.n, n_jobs=jobs, record=record, cells=module.CELLS
    )
    rows = phase.rows
    cells: dict[str, dict[str, float]] = {}
    for name, group in rows.groupby("scenario", sort=True):
        error = group["estimate"] - group["truth"]
        # A replicate the method refused is lost from ``rows`` and counts as not covering.
        cells[str(name)] = {
            "fitted": len(group),
            "covered": int(group["covered"].sum()),
            "se_ratio": float(group["std_error"].mean() / error.std(ddof=1)),
            "standardized_bias": float(error.mean() / error.std(ddof=1)),
            "seconds": float(group["cell_seconds"].mean()),
        }
    controls = module.controls_table(rows, record)
    lost = replicates * len(record.scenarios) - len(rows)
    covered = {name: int(cell["covered"]) for name, cell in cells.items()}
    try:
        rule: int | str = sizing.primary_replicates(
            record.margins, covered, replicates, implementations=1
        )
    except ValueError as error:  # no budget up to the rule's cap meets it
        rule = str(error)
    return {
        "study": record.slug,
        "pilot_replicates": replicates,
        "lost_replicates": lost,
        "cells": cells,
        "controls": controls.to_dict(orient="records"),
        "primary_replicates": rule,
        "core_seconds_per_replicate_set": float(sum(c["seconds"] for c in cells.values())),
        "wall_seconds": time.perf_counter() - begin,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=sorted(PILOT_SEEDS), default="trial")
    parser.add_argument("--replicates", type=int, default=800)
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args()
    print(json.dumps(pilot(arguments.study, arguments.replicates, arguments.jobs), indent=1))

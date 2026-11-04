"""Run the exact fluctuation emulation on every paired ``lmtp_tmle`` plan, and count divergences.

Evidence for the X59 declaration (pre-queue review item 7); no run reads it.  The node-K scan
(``fluctuation_scan.py``) refits one intercept-only ``glm`` at node K, a proxy.  This script
runs ``fluctuation_emulation.R``, lmtp 9996b04's whole TMLE loop transcribed for one fold, on
each replicate's full comparator input (``reference_sample``) and each paired plan, and reads
the epsilon of every node.  A plan-replicate diverges when the emulation fails, a node's
epsilon is non-finite, or one exceeds 1e3: the adapter guard's condition
(``lmtp_9996b04_crossfit_adapter.R``).  Needs ``Rscript`` with ``data.table``.

    python -m tests.canonical.lmtp_9996b04.fluctuation_emulation_scan OUT --jobs 4

Writes ``OUT/emulation.csv`` (one row per plan and replicate, resumable) and
``OUT/emulation-summary.csv`` (per cell: replicates, divergences, largest epsilon).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import re
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from tests.canonical.lmtp_9996b04 import fluctuation_scan

HERE = Path(__file__).resolve().parent
EMULATION = HERE / "fluctuation_emulation.R"
#: The adapter guard's bound on a node's epsilon.
BOUND = 1e3
NODE = re.compile(r"^node (\d+): .* eps (\S+), y class")
TARGETED = re.compile(r"^targeted\s+(\d+)\s+(\d+)")
FIELDS = (
    "study",
    "scenario",
    "plan",
    "replicate",
    "returncode",
    "nodes",
    "max_abs_epsilon",
    "epsilons",
    "targeted_cells",
    "targeted_outside",
    "diverged",
)


def _one(job: tuple[str, str, list[str], int, str]) -> list[dict[str, object]]:
    """Every paired plan of one replicate: the emulation's epsilon at each node."""
    study, scenario, plans, replicate, scratch = job
    from tests.studies import long_horizon_reference as reference_

    warnings.simplefilter("ignore")
    reference_.plugins = lambda frame, scenario: {}  # the plug-in is not read here
    module = importlib.import_module(fluctuation_scan.STUDIES[study])
    frame = module.reference_sample(scenario, module.STUDY.n_of(scenario), replicate).frame
    path = Path(scratch) / f"{study}__{scenario}__{replicate:04d}.csv"
    frame.to_csv(path, index=False)
    rows: list[dict[str, object]] = []
    try:
        for plan in plans:
            run = subprocess.run(
                ["Rscript", str(EMULATION), str(path), plan],
                capture_output=True,
                text=True,
                check=False,
            )
            epsilons: dict[int, float] = {}
            cells = outside = None
            for line in run.stdout.splitlines():
                if found := NODE.match(line):
                    epsilons[int(found.group(1))] = float(found.group(2))
                if found := TARGETED.match(line.strip()):
                    cells, outside = int(found.group(1)), int(found.group(2))
            largest = max((abs(value) for value in epsilons.values()), default=math.nan)
            finite = all(math.isfinite(value) for value in epsilons.values())
            rows.append(
                {
                    "study": study,
                    "scenario": scenario,
                    "plan": plan,
                    "replicate": replicate,
                    "returncode": run.returncode,
                    "nodes": len(epsilons),
                    "max_abs_epsilon": largest,
                    "epsilons": ";".join(f"{k}:{epsilons[k]:.6g}" for k in sorted(epsilons)),
                    "targeted_cells": cells,
                    "targeted_outside": outside,
                    "diverged": bool(
                        run.returncode or not epsilons or not finite or largest > BOUND
                    ),
                }
            )
    finally:
        path.unlink(missing_ok=True)
    return rows


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per study, scenario and plan: replicates read, divergences, the largest epsilon."""
    return (
        table.groupby(["study", "scenario", "plan"], sort=False)
        .agg(
            replicates=("replicate", "nunique"),
            diverged=("diverged", "sum"),
            max_abs_epsilon=("max_abs_epsilon", "max"),
            targeted_outside=("targeted_outside", "max"),
        )
        .reset_index()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=__name__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--replicates", type=int, default=100, help="per S2 scenario")
    parser.add_argument(
        "--parity-replicates", type=int, default=500, help="per lmtp-parity scenario"
    )
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args(argv)
    scratch = arguments.out / "inputs"
    scratch.mkdir(parents=True, exist_ok=True)
    counts = {"lmtp-parity": arguments.parity_replicates, "categorical": arguments.replicates}
    jobs = [
        (study, scenario, plans, replicate, str(scratch))
        for study, count in counts.items()
        for scenario, plans in fluctuation_scan._cells(study, False).items()
        for replicate in range(count)
    ]
    result = arguments.out / "emulation.csv"
    done: set[tuple[str, int]] = set()
    if result.exists():
        with result.open(newline="") as stream:
            done = {(row["scenario"], int(row["replicate"])) for row in csv.DictReader(stream)}
    jobs = [job for job in jobs if (job[1], job[3]) not in done]
    fresh = not result.exists()
    with (
        result.open("a", newline="") as stream,
        ProcessPoolExecutor(arguments.jobs) as pool,
    ):
        writer = csv.DictWriter(stream, fieldnames=list(FIELDS), lineterminator="\n")
        if fresh:
            writer.writeheader()
        for count, rows in enumerate(pool.map(_one, jobs), start=1):
            writer.writerows(rows)
            stream.flush()
            if count % 50 == 0:
                print(f"{count}/{len(jobs)} replicates", flush=True)
    table = summary(pd.read_csv(result))
    table.to_csv(arguments.out / "emulation-summary.csv", index=False, lineterminator="\n")
    print(table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

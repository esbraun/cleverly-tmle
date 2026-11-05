"""The pilot of the ``ep-crr`` study (P1 plan 5.3): spreads for the sizing rule.

A pilot, not a result.  It estimates the per-replicate mean and SD of each log ISE ratio, the
nuisance L2 errors, the share of replicates whose oracle Hessian is positive definite, and
the per-replicate cost.  :func:`tests.studies.ep_crr.design_numbers` reads the committed
summary, ``ep_crr_pilot.csv`` beside this module.  The margins are the plan's.

Every draw comes from ``stream_seed(STUDY, "crr_ise_ratio_pilot", label, r)``, disjoint from
the run's ``property_sample`` stream and its primary ``SEED``::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.ep_crr_pilot --raw <scratch dir> --jobs 4 [--summarize]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from tests.studies import ep_cate_pilot
from tests.studies import ep_crr as study
from tests.studies import ep_crr_properties as properties
from tests.studies.evidence.seeds import stream_seed

STREAM = "crr_ise_ratio_pilot"

#: Pilot replicates per gated group.
REPLICATES = 24

PILOT_CSV = Path(__file__).with_name("ep_crr_pilot.csv")


def one(task: tuple[str, int]) -> dict[str, Any]:
    label, replicate = task
    group = study.GROUP_BY_LABEL[label]
    frame, _ = study.draw_from_seed(
        label, group.n, stream_seed(study.STUDY, STREAM, label, replicate)
    )
    start = time.process_time()
    law = study.LAW_BY_LABEL[group.law]
    fitted = study.fit_replicate(law, group.n, frame, replicate)
    values = {
        cell.split("__", 1)[1]: value
        for cell, value in properties.oracle_values(group, fitted).items()
    }
    for learner, errors in fitted.items():
        values[f"{learner}_oracle_pd"] = errors["oracle_pd"]
        values["product" if learner == "ensemble" else f"{learner}_product"] = (
            errors["mu_l2"] * errors["pi_l2"]
        )
        values["mu_l2" if learner == "ensemble" else f"{learner}_mu_l2"] = errors["mu_l2"]
        values["pi_l2" if learner == "ensemble" else f"{learner}_pi_l2"] = errors["pi_l2"]
    values["seconds"] = time.process_time() - start
    return {"group": label, "replicate": replicate, "values": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--summarize", action="store_true")
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    if arguments.summarize:
        summary = ep_cate_pilot.summarize(arguments.raw)
        summary.to_csv(PILOT_CSV, index=False, float_format="%.6g")
        return
    tasks = []
    for group in study.GROUPS:
        if group.kind != "gated":
            continue
        path = arguments.raw / f"{group.label}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["replicate"] for line in path.read_text().splitlines()}
        tasks += [(group.label, r) for r in range(arguments.replicates) if r not in done]
    tasks.sort(key=lambda task: -study.GROUP_BY_LABEL[task[0]].n)
    with Pool(arguments.jobs, maxtasksperchild=4) as pool:
        for record in pool.imap_unordered(one, tasks):
            with (arguments.raw / f"{record['group']}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(record["group"], record["replicate"], round(record["values"]["seconds"], 1))


if __name__ == "__main__":
    main()


__all__ = ["PILOT_CSV", "REPLICATES", "STREAM"]

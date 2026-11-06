"""The pilot of the ``ep-assessment`` study (P1 plan 5.4): spreads for the pass probabilities.

A pilot, not a result.  Per group it records the mean and SD of each reading of
:func:`tests.studies.ep_assessment.fit_replicate`, the signed risk difference (the reported
EP-minus-constant DR risk difference times the sign of its exact value), and the CPU seconds
of the calibration phase and of the whole replicate.  :func:`design_numbers` reads the
committed ``ep_assessment_pilot.csv``; the thresholds are the plan's.

Every draw comes from ``stream_seed(STUDY, "assessment_pilot", label, r)``, disjoint from the
run's ``property_sample`` stream and its primary ``SEED``::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.ep_assessment_pilot --raw <scratch dir> --jobs 4 [--summarize]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from tests.studies import ep_assessment as study
from tests.studies import ep_cate_pilot
from tests.studies.evidence.seeds import stream_seed

STREAM = "assessment_pilot"

#: Pilot replicates per group; every one is below the stability cut, so it reads stability.
REPLICATES = 10

PILOT_CSV = Path(__file__).with_name("ep_assessment_pilot.csv")


def one(task: tuple[str, int]) -> dict[str, Any]:
    label, replicate = task
    group = study.GROUP_BY_LABEL[label]
    frame, _ = study.draw_from_seed(
        label, study.N, stream_seed(study.STUDY, STREAM, label, replicate)
    )
    start = time.process_time()
    values = study.fit_replicate(group, frame, replicate)
    values["seconds"] = time.process_time() - start
    values["signed_risk_difference"] = values["risk_difference"] * float(
        np.sign(values["exact_difference"])
    )
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
        path = arguments.raw / f"{group.label}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["replicate"] for line in path.read_text().splitlines()}
        tasks += [(group.label, r) for r in range(arguments.replicates) if r not in done]
    with Pool(arguments.jobs, maxtasksperchild=2) as pool:
        for record in pool.imap_unordered(one, tasks):
            with (arguments.raw / f"{record['group']}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(record["group"], record["replicate"], round(record["values"]["seconds"], 1))


if __name__ == "__main__":
    main()


__all__ = ["PILOT_CSV", "REPLICATES", "STREAM"]

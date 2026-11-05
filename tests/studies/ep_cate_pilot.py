"""The pilot of the ``ep-cate`` study (P1 plan 5.2, finding N2): spreads for the sizing rule.

A pilot, not a result.  It estimates the per-replicate mean and SD of each gated log ISE
ratio, the nuisance L2 errors at every ``n`` (the plan's 5.0 fallback rule reads the slope of
their product), the per-replicate cost, and for ``ep_fixed_obs_4k`` the paired EP-minus-TMLE
shift in units of the TMLE's spread.  :func:`tests.studies.ep_cate.design_numbers` reads the
committed summary, ``ep_cate_pilot.csv`` beside this module.  The margins are the plan's and
do not move with it.

Every draw comes from the named stream ``stream_seed(STUDY, "ise_ratio_pilot", label, r)``,
disjoint from the run's ``property_sample`` streams and from its primary ``SEED``.  The
summary is written by::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    NUMEXPR_NUM_THREADS=1 python -m tests.studies.ep_cate_pilot \\
        --raw <scratch dir> --jobs 4 [--groups ...] [--replicates R] [--summarize]

Each replicate's record is one JSON line in ``<scratch dir>/<group>.jsonl``; ``--summarize``
reads every record there and writes the CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pandas as pd

from tests.studies import conditional_projection as projection
from tests.studies import ep_cate as study
from tests.studies import ep_cate_properties as properties
from tests.studies.evidence.seeds import stream_seed

STREAM = "ise_ratio_pilot"

#: Pilot replicates per group: d = 3, d = 20 and the fixed-size cell.
REPLICATES = {3: 20, 20: 8, "fixed": 200}

PILOT_CSV = Path(__file__).with_name("ep_cate_pilot.csv")

COLUMNS = ("group", "statistic", "replicates", "mean", "sd")


def pilot_groups() -> list[study.Group]:
    return [group for group in study.GROUPS if group.kind in ("gated", "fixed")]


def replicates_of(group: study.Group) -> int:
    if group.kind == "fixed":
        return int(REPLICATES["fixed"])
    return int(REPLICATES[study.LAW_BY_LABEL[group.law].dimension])


def one(task: tuple[str, int]) -> dict[str, Any]:
    label, replicate = task
    group = study.GROUP_BY_LABEL[label]
    seed = stream_seed(study.STUDY, STREAM, label, replicate)
    frame, _ = study.draw_from_seed(label, group.n, seed)
    start = time.process_time()
    if group.kind == "fixed":
        fitted = study.fit_fixed(frame, replicate)
        values = {
            f"ep_{projection.TERM_LABELS[term]}": float(fitted["ep"][term])
            for term in projection.TERMS
        }
        values.update(
            {
                f"tmle_{projection.TERM_LABELS[term]}": float(fitted["tmle"][term].psi)
                for term in projection.TERMS
            }
        )
    else:
        law = study.LAW_BY_LABEL[group.law]
        fitted = study.fit_replicate(law, group.n, frame, replicate)
        values = {
            cell.split("__", 1)[1]: value
            for cell, value in properties.oracle_values(group, fitted).items()
        }
        for learner in ("ensemble", "misspecified"):
            if learner in fitted:
                errors = fitted[learner]
                prefix = "" if learner == "ensemble" else "misspecified_"
                values[f"{prefix}mu_l2"] = errors["mu_l2"]
                values[f"{prefix}pi_l2"] = errors["pi_l2"]
                values[f"{prefix}product"] = errors["mu_l2"] * errors["pi_l2"]
                values[f"{prefix}ep_mc_relative"] = errors["ep_mc_se"] / errors["ep"]
    values["seconds"] = time.process_time() - start
    return {"group": label, "replicate": replicate, "values": values}


def summarize(raw: Path) -> pd.DataFrame:
    records = []
    for path in sorted(raw.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            records.append(json.loads(line))
    out = []
    by_group: dict[str, list[dict[str, float]]] = {}
    for record in records:
        by_group.setdefault(record["group"], []).append(record["values"])
    for label, values in sorted(by_group.items()):
        frame = pd.DataFrame(values)
        if label == study.FIXED_LABEL:
            for term in projection.TERMS:
                name = projection.TERM_LABELS[term]
                spread = float(frame[f"tmle_{name}"].std(ddof=1))
                shift = (frame[f"ep_{name}"] - frame[f"tmle_{name}"]) / spread
                out.append(
                    (
                        label,
                        f"shift_{name}",
                        len(frame),
                        float(shift.mean()),
                        float(shift.std(ddof=1)),
                    )
                )
            out.append(
                (
                    label,
                    "seconds",
                    len(frame),
                    float(frame["seconds"].mean()),
                    float(frame["seconds"].std(ddof=1)),
                )
            )
            continue
        for column in frame.columns:
            column_values = frame[column].dropna()
            out.append(
                (
                    label,
                    column,
                    len(column_values),
                    float(column_values.mean()),
                    float(column_values.std(ddof=1)),
                )
            )
    return pd.DataFrame(out, columns=list(COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--summarize", action="store_true")
    arguments = parser.parse_args()
    arguments.raw.mkdir(parents=True, exist_ok=True)
    if arguments.summarize:
        summarize(arguments.raw).to_csv(PILOT_CSV, index=False, float_format="%.6g")
        return
    groups = [g for g in pilot_groups() if not arguments.groups or g.label in arguments.groups]
    tasks = []
    for group in groups:
        done = set()
        path = arguments.raw / f"{group.label}.jsonl"
        if path.exists():
            done = {json.loads(line)["replicate"] for line in path.read_text().splitlines()}
        count = arguments.replicates or replicates_of(group)
        tasks += [(group.label, r) for r in range(count) if r not in done]
    # Largest first, so the pool's tail is short.
    tasks.sort(
        key=lambda t: (
            -study.GROUP_BY_LABEL[t[0]].n
            * (
                study.LAW_BY_LABEL.get(study.GROUP_BY_LABEL[t[0]].law, study.LAWS[0]).dimension
                if study.GROUP_BY_LABEL[t[0]].kind != "fixed"
                else 0.01
            )
        )
    )
    with Pool(arguments.jobs, maxtasksperchild=4) as pool:
        for record in pool.imap_unordered(one, tasks):
            with (arguments.raw / f"{record['group']}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                record["group"],
                record["replicate"],
                round(record["values"]["seconds"], 1),
                flush=True,
            )


if __name__ == "__main__":
    main()


__all__ = ["PILOT_CSV", "REPLICATES", "STREAM", "summarize"]

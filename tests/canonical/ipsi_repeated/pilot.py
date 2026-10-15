"""Resumable pilot driver for the repeated-outcome IPSI study.

The pilots P4, P5 and P6 fix the declared sizes (``docs/development/method-benchmarking.md``).
Each pilot splits its task list into chunks.  A chunk writes one part file under ``--output``.
A relaunch skips every part that exists, so an interrupted run resumes where it stopped.

``run`` fits the chunks.  ``assemble`` reads the parts, prints the tables to paste into
``tests/studies/ipsi_repeated_properties.py``, and prints the sizes the declared rules give.

======  =====================================  ========================================
pilot   tasks                                  fit
======  =====================================  ========================================
P4      scenario, replicate (2 x 2,000)        primary fit at ``PRIMARY_N``
P5      arm, replicate (3 x 8,000)             calibration fit at ``CALIBRATION_N``
P6      family, replicate (3 x 2,000)          control fit at the family's own ``n``
======  =====================================  ========================================

Examples
--------
``python -m tests.canonical.ipsi_repeated.pilot run P4 --output .tmp/p4 --jobs 8``
``python -m tests.canonical.ipsi_repeated.pilot assemble P4 --output .tmp/p4``
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import ipsi_repeated_properties as props
from tests.studies.canonical_ipsi_repeated import PRIMARY_N, SCENARIOS

PLAN = "plan.json"
CALIBRATION_ARMS = ("in_sample", "crossfit", "continuous")
FAMILIES = ("single_robustness", "repeated_outcome_necessity", "outcome_history_necessity")
DEFAULT_REPLICATES = {"P4": 2_000, "P5": 8_000, "P6": 2_000}
DEFAULT_CHUNK = {"P4": 50, "P5": 100, "P6": 50}


def task_list(pilot: str, replicates: int, n: int | None) -> list[tuple[Any, ...]]:
    """The ordered task payloads of a pilot.  ``n=None`` takes each cell's declared size."""
    if pilot == "P4":
        size = PRIMARY_N if n is None else n
        return [(s, r, size) for s in SCENARIOS for r in range(replicates)]
    if pilot == "P5":
        size = props.CALIBRATION_N if n is None else n
        return [(a, r, size) for a in CALIBRATION_ARMS for r in range(replicates)]
    if pilot == "P6":
        return [
            (f, r, props.family_n(f) if n is None else n)
            for f in FAMILIES
            for r in range(replicates)
        ]
    raise ValueError(f"unknown pilot {pilot!r}")


def _task_function(pilot: str) -> Callable[[tuple[Any, ...]], Any]:
    return {
        "P4": props.primary_pilot_task,
        "P5": props.calibration_pilot_task,
        "P6": props.property_pilot_task,
    }[pilot]


def _call(function: Callable[[tuple[Any, ...]], Any], payload: tuple[Any, ...]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(payload)


def _part_path(output: Path, start: int) -> Path:
    return output / f"part-{start:08d}.json"


def _write_atomic(path: Path, content: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content))
    os.replace(temporary, path)


def _check_plan(output: Path, plan: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / PLAN
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded != plan:
            raise SystemExit(
                f"{path} records {recorded}, this run asks for {plan}; use a new --output"
            )
    else:
        _write_atomic(path, plan)


def chunk_starts(total: int, chunk: int) -> list[int]:
    """The first task index of every chunk."""
    return list(range(0, total, chunk))


def run(
    pilot: str,
    output: Path,
    *,
    jobs: int,
    chunk: int,
    replicates: int,
    n: int | None,
    start: int = 0,
    stop: int | None = None,
    descending: bool = False,
) -> int:
    """Fit the missing chunks in ``[start, stop)`` and return how many it wrote."""
    tasks = task_list(pilot, replicates, n)
    plan = {"pilot": pilot, "replicates": replicates, "n": n, "chunk": chunk, "tasks": len(tasks)}
    _check_plan(output, plan)
    stop = len(tasks) if stop is None else min(stop, len(tasks))
    starts = [s for s in chunk_starts(len(tasks), chunk) if start <= s < stop]
    if descending:
        starts.reverse()
    function = _task_function(pilot)
    written = 0
    began = time.perf_counter()
    for position, first in enumerate(starts, start=1):
        path = _part_path(output, first)
        label = f"[{pilot}] part {position}/{len(starts)} tasks {first}-{min(first + chunk, len(tasks)) - 1}"
        if path.exists():
            print(f"{label} skipped (exists)", flush=True)
            continue
        began_chunk = time.perf_counter()
        block = tasks[first : first + chunk]
        results = map_parallel(
            functools.partial(_call, function), [(p,) for p in block], n_jobs=jobs
        )
        _write_atomic(path, results)
        written += 1
        spent = time.perf_counter() - began_chunk
        elapsed = time.perf_counter() - began
        remaining = (len(starts) - position) * elapsed / position
        print(
            f"{label} done in {spent:.1f}s, {spent / len(block):.2f}s per task, "
            f"elapsed {elapsed / 60:.1f} min, eta {remaining / 60:.1f} min",
            flush=True,
        )
    return written


def load_parts(pilot: str, output: Path) -> list[Any]:
    """Every task result in task order.  Refuses a missing part."""
    plan = json.loads((output / PLAN).read_text())
    if plan["pilot"] != pilot:
        raise SystemExit(f"{output} holds pilot {plan['pilot']}, not {pilot}")
    starts = chunk_starts(plan["tasks"], plan["chunk"])
    missing = [s for s in starts if not _part_path(output, s).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} parts missing, first at task {missing[0]}; rerun `run`")
    results: list[Any] = []
    for first in starts:
        results.extend(json.loads(_part_path(output, first).read_text()))
    return results


def assemble(pilot: str, output: Path) -> dict[str, Any]:
    """Summarize a finished pilot and apply the declared rules."""
    results = load_parts(pilot, output)
    if pilot == "P4":
        rows = pd.DataFrame([row for result in results for row in result])
        summary = props.summarize_primary_pilot(rows)
        covered, biases, replicates = summary
        print(f"PRIMARY_PILOT_REPLICATES = {replicates}")
        print(f"PRIMARY_PILOT_COVERED = {covered!r}")
        print(f"PRIMARY_PILOT_BIASES = {_rounded(biases)!r}")
        return props.apply_pilot_rules(primary=summary)
    if pilot == "P5":
        rows = pd.DataFrame([row for result in results for row in result["rows"]])
        masses = [result["mass"] for result in results if result["mass"]]
        summary_p5 = props.summarize_calibration_pilot(rows, masses)
        print(f"CALIBRATION_PILOT_REPLICATES = {summary_p5['replicates']}")
        print(f"CALIBRATION_PILOT_RATIOS = {_rounded(summary_p5['ratios'])!r}")
        print(f"CALIBRATION_PILOT_BIASES = {_rounded(summary_p5['biases'])!r}")
        print(f"CALIBRATION_PILOT_JOINT_COVERAGE = {summary_p5['joint_coverage']:.4f}")
        print(f"C6_FALLBACK_MASS = {_rounded(summary_p5['fallback_mass'])!r}")
        return props.apply_pilot_rules(calibration=summary_p5)
    rows = pd.DataFrame([row for result in results for row in result])
    summary_p6 = props.summarize_property_pilot(rows)
    print(f"PROPERTY_PILOT_REPLICATES = {summary_p6[1]}")
    print(f"PROPERTY_PILOT_BIASES = {_rounded(summary_p6[0])!r}")
    return props.apply_pilot_rules(properties=summary_p6)


def _rounded(table: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in table.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    run_parser = sub.add_parser("run", help="fit the missing chunks")
    assemble_parser = sub.add_parser("assemble", help="summarize a finished pilot")
    for item in (run_parser, assemble_parser):
        item.add_argument("pilot", choices=sorted(DEFAULT_REPLICATES))
        item.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    run_parser.add_argument("--chunk", type=int, default=None, help="tasks per part file")
    run_parser.add_argument("--replicates", type=int, default=None, help="replicates per cell")
    run_parser.add_argument("--n", type=int, default=None, help="sample size; default is declared")
    run_parser.add_argument("--from", dest="start", type=int, default=0, help="first task index")
    run_parser.add_argument("--to", dest="stop", type=int, default=None, help="end task index")
    run_parser.add_argument(
        "--down-to",
        dest="down_to",
        type=int,
        default=None,
        help="run the chunks from the end down to this task index, last chunk first",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "assemble":
        try:
            verdicts = assemble(args.pilot, args.output)
        except ValueError as error:
            print(f"the declared rule refuses these tables: {error}")
            return 1
        for key, value in verdicts.items():
            print(f"{key}: {value}")
        return 0
    start, descending = args.start, False
    if args.down_to is not None:
        start, descending = args.down_to, True
    run(
        args.pilot,
        args.output,
        jobs=args.jobs,
        chunk=args.chunk or DEFAULT_CHUNK[args.pilot],
        replicates=args.replicates or DEFAULT_REPLICATES[args.pilot],
        n=args.n,
        start=start,
        stop=args.stop,
        descending=descending,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

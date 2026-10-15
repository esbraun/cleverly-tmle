"""Resumable pilot driver for the repeated-outcome SDR study.

The pilots P1, P2 and P3 fix the declared sizes (``docs/development/method-benchmarking.md``).
Each pilot has units: the two scenarios of P1, the four arms of P2 and the six families of
P3.  The driver splits the tasks of the chosen units into chunks.  A chunk writes one part file
under ``--output``.  A relaunch skips every part that exists, so an interrupted run resumes
where it stopped.

``run`` fits the chunks.  ``assemble`` reads one or more output directories, prints the
statistics of every cell and the tables to paste into
``tests/studies/sdr_repeated_properties.py``, and prints the sizes the declared rules give.

======  =====================================  ========================================
pilot   units                                  fit
======  =====================================  ========================================
P1      scenario (2 x 2,000)                   primary fit at ``PRIMARY_N``
P2      arm (4 x 8,000)                        calibration fit at the arm's declared n
P3      family (6 x 2,000)                     control fit at the family's declared n
======  =====================================  ========================================

A directory records the resolved sample size of each unit, the git HEAD and the hashes of the
study modules in ``plan.json``.  ``assemble`` reads a unit only from a directory whose recorded
size equals the size the study declares now.  A directory at an older size is reported and
skipped.  When several directories supply a unit at the declared size, the last one wins.

Examples
--------
``python -m tests.canonical.sdr_repeated.pilot run P1 --output .tmp/p1 --jobs 8``
``python -m tests.canonical.sdr_repeated.pilot run P3 --output .tmp/p3-n2 --units repeated_outcome_necessity --n-for repeated_outcome_necessity=16000``
``python -m tests.canonical.sdr_repeated.pilot assemble P3 --output .tmp/p3 --output .tmp/p3-n2``
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import sdr_repeated_properties as props
from tests.studies.canonical_sdr_repeated import SCENARIOS

ROOT = Path(__file__).resolve().parents[3]
PLAN = "plan.json"
DEFAULT_REPLICATES = {"P1": 2_000, "P2": 8_000, "P3": 2_000}
DEFAULT_CHUNK = {"P1": 50, "P2": 100, "P3": 50}
#: The files whose bytes decide a pilot row.  The plan records their hashes.
STUDY_FILES = (
    "tests/studies/canonical_sdr_repeated.py",
    "tests/studies/sdr_repeated_properties.py",
    "tests/studies/ltmle_repeated_law.py",
    "tests/studies/ltmle_repeated_properties.py",
    "tests/studies/canonical_ltmle_repeated.py",
    "tests/studies/canonical_ltmle.py",
    "tests/studies/repeated_sizing.py",
    "tests/studies/fractional_glm.py",
    "tests/discrete_law_longitudinal.py",
    "tests/discrete_law_longitudinal_multivalue.py",
    "src/cleverly/datasets/longitudinal.py",
)
#: The estimator source trees that every fit runs.  Each ``*.py`` file in them is hashed.
STUDY_TREES = (
    "src/cleverly/longitudinal",
    "src/cleverly/data",
    "src/cleverly/inference",
    "src/cleverly/interventions",
    "src/cleverly/learners",
    "src/cleverly/utils",
)


def study_files() -> list[str]:
    """Every file whose hash the plan records: :data:`STUDY_FILES` and the estimator trees."""
    names = set(STUDY_FILES)
    for tree in STUDY_TREES:
        names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / tree).rglob("*.py"))
    return sorted(names)


def module_hashes() -> dict[str, str]:
    """The SHA-256 of each study file, by repository path."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in study_files()}


def drifted_files(recorded: Mapping[str, str]) -> list[str]:
    """The files whose bytes differ from ``recorded``, added or removed files included."""
    current = module_hashes()
    return sorted(name for name in {*recorded, *current} if recorded.get(name) != current.get(name))


def warn_on_drift(plan: Mapping[str, Any], output: Path) -> list[str]:
    """Print a loud warning when a directory was written by other code than the current tree.

    Returns the files that differ.  The assembly goes on, because a comment edit is harmless
    and the reader judges the list.
    """
    drift = drifted_files(plan.get("modules", {}))
    recorded_head = str(plan.get("head", "unknown"))
    moved = recorded_head != git_head()
    if drift or moved:
        print(
            f"WARNING: {output} was written by HEAD {recorded_head}; the tree is at {git_head()}.",
            flush=True,
        )
    if drift:
        print(
            f"WARNING: {len(drift)} study file(s) differ from the hashes this directory "
            "recorded, so its rows may not describe the current code:",
            flush=True,
        )
        for name in drift:
            print(f"  differs: {name}", flush=True)
    return drift


def git_head() -> str:
    """The commit hash of the working tree, or ``unknown`` outside a repository."""
    try:
        done = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip()


def resolve_n(
    pilot: str, units: Sequence[str], n: int | None, n_for: Mapping[str, int]
) -> dict[str, int]:
    """The sample size of each unit: ``--n-for`` first, then ``--n``, then the declared size."""
    unknown = set(n_for) - set(props.PILOT_UNITS[pilot])
    if unknown:
        raise SystemExit(f"unknown units for {pilot}: {sorted(unknown)}")
    return {
        unit: n_for.get(unit, n if n is not None else props.pilot_declared_n(pilot, unit))
        for unit in units
    }


def task_list(pilot: str, replicates: int, resolved: Mapping[str, int]) -> list[tuple[Any, ...]]:
    """The ordered task payloads, unit by unit: ``(unit, replicate, n)``."""
    return [(unit, r, n) for unit, n in resolved.items() for r in range(replicates)]


def _task_function(pilot: str) -> Callable[[tuple[Any, ...]], Any]:
    return {
        "P1": props.primary_pilot_task,
        "P2": props.calibration_pilot_task,
        "P3": props.property_pilot_task,
    }[pilot]


def _call(function: Callable[[tuple[Any, ...]], Any], payload: tuple[Any, ...]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(payload)


def _part_path(output: Path, start: int) -> Path:
    return output / f"part-{start:08d}.json"


def _write_once(path: Path, content: Any) -> bool:
    """Write ``content`` to ``path`` through a per-process temporary file.

    The temporary name holds the process id, so two launches never share it.  The file is
    linked into place, which fails when ``path`` exists, so a part that exists is never
    replaced.  Returns whether this call wrote the file.
    """
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(content))
    try:
        os.link(temporary, path)
    except FileExistsError:
        return False
    except OSError:  # a file system with no hard links
        if path.exists():
            return False
        os.replace(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _replace_plan(path: Path, content: Any) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(content, indent=1))
    os.replace(temporary, path)


def _check_plan(output: Path, plan: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / PLAN
    if path.exists():
        recorded = json.loads(path.read_text())
        wanted = {key: value for key, value in plan.items() if key != "head"}
        found = {key: value for key, value in recorded.items() if key != "head"}
        if found != wanted:
            raise SystemExit(
                f"{path} records {found}, this run asks for {wanted}; use a new --output"
            )
    else:
        _write_once(path, plan)


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
    units: Sequence[str] | None = None,
    n: int | None = None,
    n_for: Mapping[str, int] | None = None,
    start: int = 0,
    stop: int | None = None,
    descending: bool = False,
) -> int:
    """Fit the missing chunks in ``[start, stop)`` and return how many it wrote."""
    chosen = tuple(props.PILOT_UNITS[pilot] if units is None else units)
    bad = set(chosen) - set(props.PILOT_UNITS[pilot])
    if bad:
        raise SystemExit(f"unknown units for {pilot}: {sorted(bad)}")
    resolved = resolve_n(pilot, chosen, n, n_for or {})
    tasks = task_list(pilot, replicates, resolved)
    plan = {
        "pilot": pilot,
        "replicates": replicates,
        "chunk": chunk,
        "units": list(chosen),
        "resolved_n": resolved,
        "tasks": len(tasks),
        "modules": module_hashes(),
        "head": git_head(),
    }
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
        last = min(first + chunk, len(tasks)) - 1
        label = f"[{pilot}] part {position}/{len(starts)} tasks {first}-{last}"
        if path.exists():
            print(f"{label} skipped (exists)", flush=True)
            continue
        began_chunk = time.perf_counter()
        block = tasks[first : first + chunk]
        results = map_parallel(
            functools.partial(_call, function), [(p,) for p in block], n_jobs=jobs
        )
        if not _write_once(path, results):
            print(f"{label} skipped (another launch wrote it)", flush=True)
            continue
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


def load_parts(pilot: str, output: Path) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """The plan and the task results of a directory, by unit.  Refuses a missing part."""
    plan = json.loads((output / PLAN).read_text())
    if plan["pilot"] != pilot:
        raise SystemExit(f"{output} holds pilot {plan['pilot']}, not {pilot}")
    if "resolved_n" not in plan:
        raise SystemExit(f"{output} has no resolved sample sizes; run the pilot again")
    starts = chunk_starts(plan["tasks"], plan["chunk"])
    missing = [s for s in starts if not _part_path(output, s).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} parts missing, first at task {missing[0]}; rerun `run`")
    results: list[Any] = []
    for first in starts:
        results.extend(json.loads(_part_path(output, first).read_text()))
    each = plan["replicates"]
    by_unit = {
        unit: results[index * each : (index + 1) * each] for index, unit in enumerate(plan["units"])
    }
    return plan, by_unit


def _flatten(pilot: str, results: Sequence[Any]) -> pd.DataFrame:
    return pd.DataFrame([row for result in results for row in result])


def collect(pilot: str, outputs: Sequence[Path]) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """The results of each unit at its declared size, and the size each was drawn at.

    A directory at a size other than the declared one is skipped for that unit and reported.
    A unit with no directory at the declared size, or with rows whose ``n`` is not the declared
    size, stops the assembly.
    """
    chosen: dict[str, list[Any]] = {}
    sizes: dict[str, int] = {}
    for output in outputs:
        plan, by_unit = load_parts(pilot, output)
        warn_on_drift(plan, output)
        for unit, results in by_unit.items():
            declared = props.pilot_declared_n(pilot, unit)
            recorded = int(plan["resolved_n"][unit])
            if recorded != declared:
                print(
                    f"skipped {unit} in {output}: drawn at n = {recorded}, "
                    f"the study declares n = {declared}",
                    flush=True,
                )
                continue
            found = props.pilot_row_sizes(pilot, _flatten(pilot, results)).get(unit, {declared})
            if found != {declared}:
                raise SystemExit(
                    f"{output}: the rows of {unit} hold n = {sorted(found)}, "
                    f"the study declares n = {declared}"
                )
            chosen[unit] = results
            sizes[unit] = declared
    missing = [unit for unit in props.PILOT_UNITS[pilot] if unit not in chosen]
    if missing:
        raise SystemExit(
            f"no directory supplies {missing} at the declared size; "
            f"run them (`run {pilot} --units ...`) and list their directory too"
        )
    return chosen, sizes


def _show(name: str, table: pd.DataFrame) -> None:
    print(f"\n{name}: draws, covered, SE ratio, standardized bias, spread of the estimate")
    print(table.round(4).to_string())


def _pin(name: str, value: Any) -> None:
    print(f"{name} = {value!r}")


def _rounded(table: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in table.items()}


def assemble(pilot: str, outputs: Sequence[Path]) -> dict[str, Any]:
    """Summarize finished pilot directories and apply the declared rules."""
    chosen, sizes = collect(pilot, outputs)
    ordered = [chosen[unit] for unit in props.PILOT_UNITS[pilot]]
    results = [item for unit_results in ordered for item in unit_results]
    rows = _flatten(pilot, results)
    if pilot == "P1":
        table = props.primary_cell_table(rows)
        _show("P1", table)
        summary = props.summarize_primary_pilot(rows)
        _pin("PRIMARY_PILOT_N", sizes[SCENARIOS[0]])
        _pin("PRIMARY_PILOT_REPLICATES", summary[2])
        _pin("PRIMARY_PILOT_COVERED", summary[0])
        _pin("PRIMARY_PILOT_RATIOS", _rounded(table["ratio"]))
        _pin("PRIMARY_PILOT_BIASES", _rounded(summary[1]))
        return props.apply_pilot_rules(primary=summary)
    if pilot == "P2":
        summary_p2 = props.summarize_calibration_pilot(rows)
        _show("P2", props.calibration_cell_table(rows))
        _pin("CALIBRATION_PILOT_N", sizes)
        _pin("CALIBRATION_PILOT_REPLICATES", summary_p2["replicates"])
        _pin("CALIBRATION_PILOT_COVERED", summary_p2["covered"])
        _pin("CALIBRATION_PILOT_RATIOS", _rounded(summary_p2["ratios"]))
        _pin("CALIBRATION_PILOT_BIASES", _rounded(summary_p2["biases"]))
        _pin(
            "CALIBRATION_PILOT_SPREADS",
            {k: float(f"{v:.6g}") for k, v in summary_p2["spreads"].items()},
        )
        _pin("CALIBRATION_PILOT_JOINT_COVERAGE", round(summary_p2["joint_coverage"], 4))
        return props.apply_pilot_rules(calibration=summary_p2)
    table = props.property_cell_table(rows)
    _show("P3", table)
    summary_p3 = props.summarize_property_pilot(rows)
    _pin("PROPERTY_PILOT_N", sizes)
    _pin("PROPERTY_PILOT_REPLICATES", summary_p3[1])
    _pin("PROPERTY_PILOT_COVERED", {k: int(v) for k, v in table["covered"].items()})
    _pin("PROPERTY_PILOT_RATIOS", _rounded(table["ratio"]))
    _pin("PROPERTY_PILOT_BIASES", _rounded(summary_p3[0]))
    _pin("PROPERTY_PILOT_SPREADS", {k: float(f"{v:.6g}") for k, v in table["spread"].items()})
    return props.apply_pilot_rules(properties=summary_p3)


def _unit_size(text: str) -> tuple[str, int]:
    unit, _, size = text.partition("=")
    if not size.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r} is not UNIT=N")
    return unit, int(size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    run_parser = sub.add_parser("run", help="fit the missing chunks")
    assemble_parser = sub.add_parser("assemble", help="summarize finished pilot directories")
    for item in (run_parser, assemble_parser):
        item.add_argument("pilot", choices=sorted(DEFAULT_REPLICATES))
    run_parser.add_argument("--output", type=Path, required=True)
    assemble_parser.add_argument(
        "--output",
        type=Path,
        action="append",
        required=True,
        help="a pilot directory; repeat it to combine directories, the last one wins per unit",
    )
    run_parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    run_parser.add_argument("--chunk", type=int, default=None, help="tasks per part file")
    run_parser.add_argument("--replicates", type=int, default=None, help="replicates per unit")
    run_parser.add_argument(
        "--units", nargs="+", default=None, help="run these units only (default: all)"
    )
    run_parser.add_argument("--n", type=int, default=None, help="sample size of every unit")
    run_parser.add_argument(
        "--n-for",
        dest="n_for",
        type=_unit_size,
        action="append",
        default=[],
        metavar="UNIT=N",
        help="sample size of one unit; wins over --n",
    )
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
        units=args.units,
        n=args.n,
        n_for=dict(args.n_for),
        start=start,
        stop=args.stop,
        descending=descending,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

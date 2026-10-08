"""Step 0 of RM39a1: the seed probe and the determinism gate (plan section 3.8).

usage::

    python -m tests.canonical.harness_fixture.determinism inputs --work DIR [--studies a,b]
    python -m tests.canonical.harness_fixture.determinism off-harness --work DIR
    python -m tests.canonical.harness_fixture.determinism probe --work DIR [--studies a,b]
    python -m tests.canonical.harness_fixture.determinism full --work DIR --runners x,y

RM39 makes the harness call ``set.seed`` before each fit.  That is result-neutral for a runner
whose fits draw no random number, and for such a runner the committed artifacts stay valid.
This module shows which runners those are, one container at a time, each under 4 GB:

``inputs``
    Draws 3 replicates of each paired study with its own ``draw_and_fit`` (n17) and writes
    ``samples.csv.gz`` and ``truth.csv`` under ``DIR/<slug>/``.  No reference runs.
``off-harness`` (step 0.1)
    Runs the three runners that stay off the harness, as they are on ``origin/main`` at
    ``572501b8``, at 1 and at 2 cores.  Unseeded R seeds itself from the clock and the PID, so
    different bytes mean the runner draws random numbers: stop and report.
``probe`` (step 0.2)
    Runs every runner with the new harness on the calibration groups (replicate 0, which holds
    the first group of every scenario) with ``CLEVERLY_HARNESS_RNG_PROBE=1``.  Each fit prints
    ``rng-probe: <runner> <id> untouched|consumed``.
``full`` (step 0.3)
    Runs a runner on 3 replicates: its code at ``572501b8`` with the old harness at 1 and at 2
    cores, and its current code with the new harness under two different seeds files.  All four
    outputs must be byte-identical; a difference means seeding moves that study's numbers.

Every row goes to ``determinism.csv`` beside this file.  The nested-fork flag is a grep of the
runner and the files it sources (not the harness) for ``mclapply``, ``mcparallel``, ``future``
and a ``parallel =`` argument.  A draw inside a nested fork does not move the worker's
``.Random.seed``, so the probe cannot see it, and a flagged runner goes to the full check.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import re
import runpy
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from tests.canonical import runner_source

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
RESULTS = HERE / "determinism.csv"
BASE = "572501b8"
COLUMNS = (
    "runner",
    "study",
    "step",
    "groups",
    "probe",
    "nested_fork",
    "sha_old_1core",
    "sha_old_2core",
    "sha_new_seeds_a",
    "sha_new_seeds_b",
    "verdict",
    "date",
    "image",
)
IN_PLACE = ("ctmle3_oat/run_ctmle3_oat.R", "ctmle_selector/run_ctmle.R", "drtmle/run_drtmle.R")
_NESTED = re.compile(r"\bmclapply\b|\bmcparallel\b|\bfuture\b|\bparallel\s*=")
#: A study runner takes the driver's three paths; a fixture generator does not.
_RUNNER = re.compile(r"study_arguments\(|length\(args\) != 3")
#: A ``regenerate.py`` that calls one of the drivers ``_recording`` patches.
_DRIVER = re.compile(r"from tests\.canonical\.regenerate import|declared_run|learned_rule_run")


@dataclass(frozen=True)
class Target:
    """One runner of one paired study."""

    slug: str
    study: ModuleType
    reference: Any
    here: Path
    runner: str

    @property
    def root(self) -> Path:
        return self.reference.runner_root or self.here

    @property
    def path(self) -> Path:
        return self.root / self.runner

    @property
    def label(self) -> str:
        return self.path.relative_to(CANONICAL).as_posix()


@contextlib.contextmanager
def _recording() -> Iterator[list[tuple[Any, Any, Path, Any]]]:
    """Patch the three driver entry points so a study's ``regenerate.py`` only reports them."""
    from tests.canonical import declared_run, learned_rule_run, regenerate

    calls: list[tuple[Any, Any, Path, Any]] = []

    def record(study: Any, properties: Any, *, here: Path, reference: Any = None) -> None:
        calls.append((study, properties, here, reference))

    saved = (regenerate.main, declared_run.run, learned_rule_run.run)
    regenerate.main = record  # type: ignore[assignment]
    declared_run.run = record  # type: ignore[assignment]
    learned_rule_run.run = record  # type: ignore[assignment]
    try:
        yield calls
    finally:
        regenerate.main, declared_run.run, learned_rule_run.run = saved  # type: ignore[assignment]


def targets() -> list[Target]:
    """Every runner of every paired study, read from each study's ``regenerate.py``."""
    found: list[Target] = []
    for script in sorted(CANONICAL.glob("*/regenerate.py")):
        # A script that hands its study to none of the patched drivers regenerates something
        # itself when run as __main__ (ctmle_logistic_plugin writes its fixture), so it is
        # never executed here.
        if not _DRIVER.search(script.read_text(encoding="utf-8")):
            continue
        with _recording() as calls:
            arguments = list(sys.argv)
            sys.argv = [str(script)]
            try:
                runpy.run_path(str(script), run_name="__main__")
            finally:
                sys.argv = arguments
        for study, _, here, reference in calls:
            if reference is None or reference.interpreter != "Rscript":
                continue
            runners = [reference.runner] + [
                name
                for name in reference.extra_files
                if name.endswith(".R")
                and name != "study_harness.R"
                and _RUNNER.search(((reference.runner_root or here) / name).read_text("utf-8"))
            ]
            for runner in runners:
                found.append(Target(study.STUDY.slug, study, reference, here, runner))
    return found


_BLOCK = re.compile(
    r"# ---- RM39 shared resilience functions: begin ----.*?"
    r"# ---- RM39 shared resilience functions: end ----",
    re.S,
)
#: The runner's own fan-out over its groups, which forks once per worker, not inside a fit.
_DISPATCH = re.compile(r"mclapply\(\s*(?:seq_along\(groups\)|groups)\s*,")


def nested_fork(target: Target) -> bool:
    """Whether a fit itself may fork: the runner and what it sources, without the harness, the
    shared block, comments, the group dispatch and a sequential ``future`` plan."""
    harness = (CANONICAL / "study_harness.R").read_text(encoding="utf-8")
    text = _BLOCK.sub("", runner_source(target.path).replace(harness, ""))
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    text = _DISPATCH.sub("", text).replace("future::plan(future::sequential)", "")
    return bool(_NESTED.search(text))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_show(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE}:{path}"], cwd=ROOT, capture_output=True, check=True
    ).stdout


def write_inputs(work: Path, target: Target, *, jobs: int) -> None:
    folder = work / target.slug
    if (folder / "samples.csv.gz").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    drawn = target.study.draw_and_fit(replicates=3, n=target.study.PRIMARY_N, n_jobs=jobs)
    samples, truths, _ = drawn
    archive = {"method": "gzip", "mtime": 0}
    samples.to_csv(folder / "samples.csv.gz", index=False, lineterminator="\n", compression=archive)
    truths.to_csv(folder / "truth.csv", index=False, lineterminator="\n")


def cut(work: Path, target: Target, name: str, replicates: int) -> Path:
    """A copy of the study's samples with the first ``replicates`` replicates."""
    folder = work / target.slug
    out = folder / f"{name}.csv.gz"
    if not out.exists():
        frame = pd.read_csv(folder / "samples.csv.gz")
        frame.loc[frame["replicate"] < replicates].to_csv(
            out, index=False, lineterminator="\n", compression={"method": "gzip", "mtime": 0}
        )
    return out


def cut_truth(work: Path, target: Target, replicates: int) -> str:
    """The truth table cut like the samples, so a runner's count check sees one draw."""
    folder = work / target.slug
    name = f"truth_{replicates}.csv"
    frame = pd.read_csv(folder / "truth.csv")
    if "replicate" in frame:
        frame = frame.loc[frame["replicate"] < replicates]
    frame.to_csv(folder / name, index=False, lineterminator="\n")
    return name


def run(
    work: Path,
    target: Target,
    samples: Path,
    output: str,
    *,
    old: bool,
    cores: int,
    seeds: Path | None = None,
    env: dict[str, str] | None = None,
    truths: str = "truth.csv",
) -> tuple[int, str, Path]:
    """One container: the runner and harness at ``572501b8`` when ``old``, else current."""
    folder = work / target.slug
    reference = target.reference
    variables = {"CLEVERLY_R_CORES": str(cores), "CLEVERLY_REFERENCE_CORES": str(cores)}
    if seeds is not None:
        variables["CLEVERLY_R_SEEDS"] = f"/work/{seeds.name}"
    variables.update(env or {})
    mounts = ["-v", f"{folder}:/work"]
    old_dir = work / "_old"
    old_dir.mkdir(exist_ok=True)

    def frozen(path: Path) -> Path:
        name = path.relative_to(ROOT).as_posix()
        copy = old_dir / name.replace("/", "__")
        if not copy.exists():
            copy.write_bytes(_git_show(name))
        return copy

    command_tail: list[str] = []
    entrypoint: list[str] = []
    if reference.mount_runner:
        mounts += ["-v", f"{target.root}:/fixture:ro"]
        if old:
            mounts += ["-v", f"{frozen(target.path)}:/fixture/{target.runner}:ro"]
            harness = CANONICAL / "study_harness.R"
            if target.root == CANONICAL:
                mounts += ["-v", f"{frozen(harness)}:/fixture/study_harness.R:ro"]
        entrypoint = ["--entrypoint", reference.interpreter]
        command_tail.append(f"/fixture/{target.runner}")
    else:
        source = frozen(target.path) if old else target.path
        mounts += ["-v", f"{source}:/opt/{Path(target.runner).name}:ro"]
    out = folder / output
    out.unlink(missing_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "-m",
        "4g",
        "--cpus",
        str(max(cores, 1)),
        *(part for item in variables.items() for part in ("-e", "=".join(item))),
        *mounts,
        *entrypoint,
        reference.image,
        *command_tail,
        f"/work/{samples.name}",
        f"/work/{truths}",
        f"/work/{output}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    log = completed.stdout + completed.stderr
    (folder / f"{output}.log").write_text(log, encoding="utf-8", newline="\n")
    return completed.returncode, log, out


def seeds_file(work: Path, target: Target, name: str, *, shift: int = 0) -> Path:
    from tests.canonical.regenerate import write_seeds

    record = dataclasses.replace(target.study.STUDY, replicates=3)
    if shift:
        record = dataclasses.replace(record, seed=record.seed + shift)
    path = work / target.slug / f"{name}.csv"
    write_seeds(path, record, Path(target.runner).stem)
    return path


def image_id(image: str) -> str:
    return subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
    ).stdout.strip()


def record(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    if RESULTS.exists():
        frame = pd.concat([pd.read_csv(RESULTS, keep_default_na=False), frame], ignore_index=True)
    frame.to_csv(RESULTS, index=False, lineterminator="\n")


def _build(target: Target) -> None:
    context = target.reference.build_context or target.here
    subprocess.run(["docker", "build", "-t", target.reference.image, str(context)], check=True)


def off_harness(work: Path, chosen: list[Target]) -> list[dict[str, Any]]:
    rows = []
    for target in chosen:
        if target.label not in IN_PLACE:
            continue
        _build(target)
        samples = cut(work, target, "full3", 3)
        hashes = []
        for cores in (1, 2):
            code, _, out = run(work, target, samples, f"off{cores}.csv", old=True, cores=cores)
            hashes.append(_sha(out) if code == 0 and out.exists() else f"exit {code}")
        rows.append(
            {
                "runner": target.label,
                "study": target.slug,
                "step": "0.1",
                "groups": "replicates 0-2",
                "probe": "",
                "nested_fork": nested_fork(target),
                "sha_old_1core": hashes[0],
                "sha_old_2core": hashes[1],
                "verdict": "deterministic" if hashes[0] == hashes[1] else "DIFFERS: stop",
                "date": datetime.date.today().isoformat(),
                "image": image_id(target.reference.image),
            }
        )
        print(rows[-1], flush=True)
        record(rows[-1:])  # each row as it finishes, so a stopped step keeps its rows
    return rows


def probe(work: Path, chosen: list[Target]) -> list[dict[str, Any]]:
    rows = []
    for target in chosen:
        _build(target)
        samples = cut(work, target, "probe", 1)
        truths = cut_truth(work, target, 1)
        seeds = seeds_file(work, target, f"seeds_{Path(target.runner).stem}")
        code, log, _ = run(
            work,
            target,
            samples,
            f"probe_{Path(target.runner).stem}.csv",
            old=False,
            cores=1,
            seeds=seeds,
            env={"CLEVERLY_HARNESS_RNG_PROBE": "1"},
            truths=truths,
        )
        lines = re.findall(r"^rng-probe: \S+ (\S+) (untouched|consumed)$", log, re.M)
        if code != 0:
            result = f"exit {code}"
        elif not lines:
            result = "no harness fit"
        elif all(state == "untouched" for _, state in lines):
            result = "untouched"
        else:
            result = "consumed: " + " ".join(i for i, state in lines if state == "consumed")
        rows.append(
            {
                "runner": target.label,
                "study": target.slug,
                "step": "0.2",
                "groups": " ".join(i for i, _ in lines) or "replicate 0",
                "probe": result,
                "nested_fork": nested_fork(target),
                "verdict": "full check"
                if (result != "untouched" and result != "no harness fit") or nested_fork(target)
                else "deterministic by probe",
                "date": datetime.date.today().isoformat(),
                "image": image_id(target.reference.image),
            }
        )
        print(rows[-1], flush=True)
        record(rows[-1:])  # each row as it finishes, so a stopped step keeps its rows
    return rows


def full(work: Path, chosen: list[Target]) -> list[dict[str, Any]]:
    rows = []
    for target in chosen:
        _build(target)
        samples = cut(work, target, "full3", 3)
        stem = Path(target.runner).stem
        hashes = []
        for cores in (1, 2):
            code, _, out = run(
                work, target, samples, f"old{cores}_{stem}.csv", old=True, cores=cores
            )
            hashes.append(_sha(out) if code == 0 and out.exists() else f"exit {code}")
        for name, shift in (("a", 0), ("b", 1)):
            seeds = seeds_file(work, target, f"seeds_{name}_{stem}", shift=shift)
            code, _, out = run(
                work, target, samples, f"new_{name}_{stem}.csv", old=False, cores=2, seeds=seeds
            )
            hashes.append(_sha(out) if code == 0 and out.exists() else f"exit {code}")
        rows.append(
            {
                "runner": target.label,
                "study": target.slug,
                "step": "0.3",
                "groups": "replicates 0-2",
                "probe": "",
                "nested_fork": nested_fork(target),
                "sha_old_1core": hashes[0],
                "sha_old_2core": hashes[1],
                "sha_new_seeds_a": hashes[2],
                "sha_new_seeds_b": hashes[3],
                "verdict": "identical" if len(set(hashes)) == 1 else "DIFFERS: stop",
                "date": datetime.date.today().isoformat(),
                "image": image_id(target.reference.image),
            }
        )
        print(rows[-1], flush=True)
        record(rows[-1:])  # each row as it finishes, so a stopped step keeps its rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("inputs", "off-harness", "probe", "full", "list"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--studies", default="", help="comma-separated study slugs")
    parser.add_argument("--runners", default="", help="comma-separated runner labels")
    parser.add_argument("--jobs", type=int, default=2)
    arguments = parser.parse_args()
    chosen = targets()
    if arguments.studies:
        wanted = set(arguments.studies.split(","))
        chosen = [target for target in chosen if target.slug in wanted]
    if arguments.runners:
        wanted = set(arguments.runners.split(","))
        chosen = [target for target in chosen if target.label in wanted]
    arguments.work.mkdir(parents=True, exist_ok=True)
    if arguments.step == "list":
        for target in chosen:
            print(f"{target.slug}\t{target.label}\tnested_fork={nested_fork(target)}")
        return
    if arguments.step == "inputs":
        for target in chosen:
            write_inputs(arguments.work, target, jobs=arguments.jobs)
            print(f"inputs for {target.slug}", flush=True)
        return
    step = {"off-harness": off_harness, "probe": probe, "full": full}[arguments.step]
    step(arguments.work, chosen)


if __name__ == "__main__":
    main()

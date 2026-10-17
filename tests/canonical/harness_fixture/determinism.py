"""Step 0 of RM39a1: the seed probe and the determinism gate (plan section 3.8).

usage::

    python -m tests.canonical.harness_fixture.determinism inputs --work DIR [--studies a,b]
    python -m tests.canonical.harness_fixture.determinism off-harness --work DIR
    python -m tests.canonical.harness_fixture.determinism probe --work DIR [--studies a,b]
    python -m tests.canonical.harness_fixture.determinism full --work DIR --runners x,y
    python -m tests.canonical.harness_fixture.determinism stage --work DIR [--base f1840adc]
    python -m tests.canonical.harness_fixture.determinism stage-probe --work DIR

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

RM39 part a2 adds two steps:

``stage`` (plan 7.2, rev 2 M2)
    Runs a runner on 3 replicates twice under one seeds file: with every ``.R`` file of its
    ``Reference`` (the runner, each ``.R`` extra file, everything they source and the harness)
    as it was at ``--base`` (default ``f1840adc``), and as it is now.  The two outputs must be
    byte-identical, and the head run must write no failures file.  One row per edited ``.R``
    file, with step ``a2``, the file's repository path in ``source`` and the sha256 of its
    head bytes in ``source_sha256``; the verdict is ``identical`` or ``DIFFERS: stop``.
``stage-probe`` (plan 7.3, rev 2 m14)
    Runs the head runner three times on 3 replicates under an ``R_PROFILE_USER`` that traces
    one function so that it raises for the first group only (``CLEVERLY_GROUP_ID``, which the
    harness sets in each forked child): the comparator's entry function, which must be
    recorded; ``stats::glm.fit``, which is recorded or absorbed; and a truth file without
    replicate 0, whose join refusal must stay fatal.  Each outcome is ``recorded``, ``fatal``,
    ``absorbed`` or ``not reached`` (from the printed hit count).  Rows have step ``a2-probe``.

RM39 part b runs ``stage`` on the zEpid runner too, the one paired runner that is Python:
``stage --runners zepid_cvtmle/run_zepid_cvtmle.py --base 227cf50b --step-name b1``.  Its
source is the runner file alone, and its rows have the step that ``--step-name`` gives.

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
#: The commit RM39 part a2 started from: RM39a1 merged (#277).
STAGE_BASE = "f1840adc"
#: The main commit a2 merged before it landed.  A runner that main added after ``STAGE_BASE``
#: (X17's ``run_drtmle_oat_per_arm.R``) is frozen here instead, so its base run is the
#: published file rather than the head.  Every other ``.R`` file is the same at both commits.
STAGE_ADDED_BASE = "287a1fb5"
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
    # RM39a2: the edited source a step-a2 row witnesses, and the stage probe's outcome.
    "source",
    "source_sha256",
    "sha_base",
    "sha_head",
    "trace",
    "outcome",
    "hits",
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


def targets(interpreters: tuple[str, ...] = ("Rscript",)) -> list[Target]:
    """Every runner of every paired study, read from each study's ``regenerate.py``.

    Only the runners of ``interpreters`` are returned; every step but ``stage`` reads R.
    """
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
            if reference is None or reference.interpreter not in interpreters:
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


def _git_show_at(commit: str, path: str) -> bytes | None:
    """``path`` as it was at ``commit``, or ``None`` when it did not exist there."""
    shown = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
    return shown.stdout if shown.returncode == 0 else None


def _git_show(path: str) -> bytes:
    """``path`` as it was at ``572501b8``.

    A runner that a study added after that commit has no old version, and RM39 does not edit
    it, so its current bytes are its old bytes; only the harness it sources is older.
    """
    shown = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT, capture_output=True)
    if shown.returncode == 0:
        return shown.stdout
    return (ROOT / path).read_bytes()


def write_inputs(work: Path, target: Target, *, jobs: int) -> None:
    folder = work / target.slug
    if (folder / "samples.csv.gz").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    drawn = target.study.draw_and_fit(replicates=3, n=target.study.PRIMARY_N, n_jobs=jobs)
    from tests.studies.evidence.replication import load_samples

    samples, truths = load_samples(drawn.samples), drawn.truths
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
    overrides: dict[str, Path] | None = None,
) -> tuple[int, str, Path]:
    """One container: the runner and harness at ``572501b8`` when ``old``, else current.

    ``overrides`` maps a container path to a host file mounted over it read-only, after the
    other mounts (the ``stage`` step's frozen sources and the probe's R profile).
    """
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
    if overrides:
        # An override replaces a mount at the same container path; docker refuses a duplicate.
        pairs = [mounts[index + 1] for index in range(0, len(mounts), 2)]
        kept = [
            pair
            for pair in pairs
            if pair.split(":")[-2 if pair.endswith(":ro") else -1] not in overrides
        ]
        mounts = [part for pair in kept for part in ("-v", pair)]
        for inside, host in overrides.items():
            mounts += ["-v", f"{host}:{inside}:ro"]
    out = folder / output
    out.unlink(missing_ok=True)
    out.with_name(f"{out.stem}.failures.csv").unlink(missing_ok=True)
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
        held = pd.read_csv(RESULTS, keep_default_na=False, dtype=str)
        # A file written before a column existed reads that column as empty.
        held = held.reindex(columns=list(COLUMNS), fill_value="")
        frame = pd.concat([held, frame], ignore_index=True)
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


_SOURCED = re.compile(r"""source\(["']/fixture/(?P<path>[^"']+)["']\)""")


def stage_sources(target: Target) -> list[Path]:
    """Every ``.R`` file the target's run reads: the runner, what it sources, transitively,
    and the harness when the runner root holds it.

    A file the image ships but no run sources (a probe of another target, or a manual smoke
    in ``extra_files``) is not read, so this run does not witness it.
    """
    if target.reference.interpreter != "Rscript":
        # A Python runner imports nothing from the repository; its own file is its source.
        return [target.path]
    found: list[Path] = []
    pending = [target.path]
    if target.root == CANONICAL:
        pending.append(CANONICAL / "study_harness.R")
    while pending:
        path = pending.pop(0)
        if path in found or not path.exists():
            continue
        found.append(path)
        for match in _SOURCED.finditer(path.read_text(encoding="utf-8")):
            pending.append(target.root / match.group("path"))
    return found


def is_primary(target: Target) -> bool:
    """Whether the target is its study's ``Reference.runner`` rather than a probe."""
    return target.runner == target.reference.runner


def _frozen(work: Path, commit: str, path: Path) -> Path | None:
    """``path`` at ``commit`` as a file under ``work``, or ``None`` when it did not exist."""
    name = path.relative_to(ROOT).as_posix()
    copy = work / f"_frozen_{commit}" / name.replace("/", "__")
    if not copy.exists():
        held = _git_show_at(commit, name)
        if held is None:
            return None
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(held)
    return copy


def _container_path(target: Target, path: Path) -> str | None:
    """Where ``path`` sits in the container: under ``/fixture`` for a mounted runner, at
    ``/opt/<name>`` for the baked one."""
    if not target.reference.mount_runner:
        return f"/opt/{path.name}" if path == target.path else None
    try:
        return f"/fixture/{path.relative_to(target.root).as_posix()}"
    except ValueError:
        return None


def stage(work: Path, chosen: list[Target], base: str, step: str = "a2") -> list[dict[str, Any]]:
    """Plan 7.2: the head sources against the ``base`` sources, on the same inputs and seeds.

    ``step`` labels the rows, so a later edit's witness rows are told apart from RM39a2's.
    """
    rows = []
    for target in chosen:
        _build(target)
        samples = cut(work, target, "full3", 3)
        stem = Path(target.runner).stem
        seeds = seeds_file(work, target, f"seeds_stage_{stem}")
        edited: list[Path] = []
        overrides: dict[str, Path] = {}
        for path in stage_sources(target):
            frozen = _frozen(work, base, path) or _frozen(work, STAGE_ADDED_BASE, path)
            if frozen is None:
                raise RuntimeError(f"{path} exists at neither {base} nor {STAGE_ADDED_BASE}")
            if frozen.read_bytes() == path.read_bytes():
                continue
            edited.append(path)
            inside = _container_path(target, path)
            if inside is not None:
                overrides[inside] = frozen
        if not edited:
            continue
        hashes = []
        failures_written = False
        for name, frozen_sources in (("base", overrides), ("head", {})):
            code, _, out = run(
                work,
                target,
                samples,
                f"stage_{name}_{stem}.csv",
                old=False,
                cores=2,
                seeds=seeds,
                overrides=frozen_sources,
            )
            hashes.append(_sha(out) if code == 0 and out.exists() else f"exit {code}")
            if name == "head":
                failures_written = out.with_name(f"{out.stem}.failures.csv").exists()
        identical = hashes[0] == hashes[1] and not hashes[0].startswith("exit")
        verdict = "identical" if identical and not failures_written else "DIFFERS: stop"
        if failures_written:
            verdict = "DIFFERS: stop (the head run recorded a failure)"
        for path in edited:
            rows.append(
                {
                    "runner": target.label,
                    "study": target.slug,
                    "step": step,
                    "groups": "replicates 0-2",
                    "probe": "",
                    "nested_fork": nested_fork(target),
                    "verdict": verdict,
                    "date": datetime.date.today().isoformat(),
                    "image": image_id(target.reference.image),
                    "source": path.relative_to(ROOT).as_posix(),
                    "source_sha256": _sha(path),
                    "sha_base": hashes[0],
                    "sha_head": hashes[1],
                }
            )
            print(rows[-1], flush=True)
        record(rows[-len(edited) :])  # each runner as it finishes, so a stopped step keeps its rows
    return rows


#: The comparator entry function each runner's fit reaches, found by the first pattern its
#: source (with everything it sources) matches.  Order matters: an lmtp runner also sources the
#: ltmle regimen adapter for its arms.
ENTRY_POINTS = (
    (re.compile(r"cf_tmle"), "lmtp::cf_tmle"),
    (re.compile(r"drtmle::drtmle\("), "drtmle::drtmle"),
    (re.compile(r"ctmleDiscrete\("), "ctmle::ctmleDiscrete"),
    (re.compile(r"(?<![\w.])tmle3\(|fit_tmle3\("), "tmle3::fit_tmle3"),
    (re.compile(r"tmle::tmle\("), "tmle::tmle"),
    (re.compile(r"(?<![\w.])ltmle\(|do\.call\(ltmle"), "ltmle::ltmle"),
    (re.compile(r"(?<![\w.])ipsi\("), "npcausal::ipsi"),
    (re.compile(r"do\.call\(survtmle"), "survtmle::survtmle"),
)

#: The R profile of the stage probe.  ``.First`` runs after the default packages attach; it
#: traces ``CLEVERLY_PROBE_TRACE`` so that it raises in the probed groups only.  ``glm.fit`` is
#: traced before any comparator namespace loads, so a package that imports it imports the trace.
PROBE_PROFILE = """\
.First <- function() {
  groups <- strsplit(Sys.getenv("CLEVERLY_PROBE_GROUP"), ",", fixed = TRUE)[[1]]
  target <- Sys.getenv("CLEVERLY_PROBE_TRACE")
  parts <- strsplit(target, "::", fixed = TRUE)[[1]]
  tracer <- bquote({
    if (Sys.getenv("CLEVERLY_GROUP_ID") %in% .(groups)) {
      cat(sprintf("stage-probe hit: %s %s\\n", .(target), Sys.getenv("CLEVERLY_GROUP_ID")))
      stop(sprintf("cleverly stage probe: %s", .(target)))
    }
  })
  suppressMessages(trace(
    parts[[2]], where = asNamespace(parts[[1]]), tracer = tracer, print = FALSE
  ))
  invisible(NULL)
}
"""

#: The first group of each id scheme: a fitter's and an in-place runner's, and a stream's.
PROBE_GROUPS = "g000000001,r000000000"


def entry_point(target: Target) -> str | None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in stage_sources(target))
    harness = (CANONICAL / "study_harness.R").read_text(encoding="utf-8")
    source = _BLOCK.sub("", source.replace(harness, ""))
    for pattern, name in ENTRY_POINTS:
        if pattern.search(source):
            return name
    return None


def _outcome(code: int, log: str, out: Path) -> tuple[str, str, int]:
    """``recorded``, ``fatal``, ``absorbed`` or ``not reached``; the record's class and stage;
    the printed hit count."""
    hits = len(re.findall(r"^stage-probe hit: ", log, re.M))
    failures = out.with_name(f"{out.stem}.failures.csv")
    if code != 0:
        index = log.rfind("Error")
        detail = " ".join(log[index:].split())[:300] if index >= 0 else f"exit {code}"
        return "fatal", detail, hits
    if failures.exists():
        held = pd.read_csv(failures, dtype=str, keep_default_na=False)
        probed = held.loc[held["group_id"].isin(PROBE_GROUPS.split(","))]
        if not probed.empty:
            first = probed.iloc[0]
            return "recorded", f"class={first['class']} stage={first['stage']}", hits
        return "recorded elsewhere", f"{len(held)} records, none for the probed group", hits
    return ("absorbed" if hits else "not reached"), "", hits


def stage_probe(work: Path, chosen: list[Target]) -> list[dict[str, Any]]:
    """Plan 7.3 and rev 2 m14: two traces and the truth cut, per runner, at the head."""
    profile = work / "stage_probe_profile.R"
    profile.write_text(PROBE_PROFILE, encoding="utf-8", newline="\n")
    rows = []
    for target in chosen:
        if not is_primary(target):
            continue  # a probe takes no stage, so every error in it stays fatal (M3)
        _build(target)
        samples = cut(work, target, "full3", 3)
        stem = Path(target.runner).stem
        seeds = seeds_file(work, target, f"seeds_stage_{stem}")
        entry = entry_point(target)
        cases: list[tuple[str, dict[str, str], str]] = []
        for trace in (entry, "stats::glm.fit"):
            if trace is None:
                continue
            cases.append((trace, {"CLEVERLY_PROBE_TRACE": trace}, "truth.csv"))
        truth = pd.read_csv(work / target.slug / "truth.csv")
        cut_name = ""
        if "replicate" in truth:
            cut_name = f"truth_without_0_{stem}.csv"
            truth.loc[truth["replicate"] != 0].to_csv(
                work / target.slug / cut_name, index=False, lineterminator="\n"
            )
            cases.append(("truth cut", {}, cut_name))
        for trace, extra, truths in cases:
            env = {"CLEVERLY_PROBE_GROUP": PROBE_GROUPS, **extra}
            overrides: dict[str, Path] = {}
            if extra:
                env["R_PROFILE_USER"] = "/probe/profile.R"
                overrides["/probe/profile.R"] = profile
            label = re.sub(r"\W+", "_", trace)
            code, log, out = run(
                work,
                target,
                samples,
                f"stage_probe_{label}_{stem}.csv",
                old=False,
                cores=2,
                seeds=seeds,
                env=env,
                truths=truths,
                overrides=overrides,
            )
            outcome, detail, hits = _outcome(code, log, out)
            if trace == "truth cut":
                joined = bool(re.search(r"truth", detail, re.I))
                verdict = "fatal" if outcome == "fatal" and joined else "DIFFERS: stop"
            elif trace == entry:
                verdict = "recorded" if outcome == "recorded" else "DIFFERS: stop"
            else:
                verdict = (
                    outcome
                    if outcome in ("recorded", "absorbed", "not reached")
                    else ("DIFFERS: stop")
                )
            rows.append(
                {
                    "runner": target.label,
                    "study": target.slug,
                    "step": "a2-probe",
                    "groups": PROBE_GROUPS,
                    "probe": detail,
                    "nested_fork": nested_fork(target),
                    "verdict": verdict,
                    "date": datetime.date.today().isoformat(),
                    "image": image_id(target.reference.image),
                    "trace": trace,
                    "outcome": outcome,
                    "hits": hits,
                }
            )
            print(rows[-1], flush=True)
            record(rows[-1:])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step",
        choices=("inputs", "off-harness", "probe", "full", "stage", "stage-probe", "list"),
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--studies", default="", help="comma-separated study slugs")
    parser.add_argument("--runners", default="", help="comma-separated runner labels")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--base", default=STAGE_BASE, help="the commit the stage step freezes")
    parser.add_argument("--step-name", default="a2", help="the step the stage rows carry")
    arguments = parser.parse_args()
    chosen = targets(("Rscript", "python") if arguments.step == "stage" else ("Rscript",))
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
    if arguments.step == "stage":
        stage(arguments.work, chosen, arguments.base, arguments.step_name)
        return
    if arguments.step == "stage-probe":
        stage_probe(arguments.work, chosen)
        return
    step = {"off-harness": off_harness, "probe": probe, "full": full}[arguments.step]
    step(arguments.work, chosen)


if __name__ == "__main__":
    main()

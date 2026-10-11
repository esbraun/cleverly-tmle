"""The regeneration driver every registered study shares.

A study's own ``regenerate.py`` says which study it is and how to reach its reference
implementation, and nothing else.  The sequence -- draw, fit, run the reference on the same
rows, validate the replicate contract, write six artefacts and a manifest, then refuse the run
if any gate failed -- is identical for every one of them, and it was written out seven times
before this file existed.  The costly consequence was not the duplication itself: it was that a
fix landed in the copy the author happened to be in.  The survival study's copy grew a
zero-row property fallback for smoke runs and an empty-frame guard on the failure query, and
the end-of-study copy it was cloned from still lacked both.

Three shapes have to fit through one driver, so they are declared rather than branched on:

* **A mounted runner.**  The ``ltmle`` studies bind their reference sources at ``/fixture`` and
  pass the script as an argument, so the image carries only the packages.  Related studies can
  share a Docker context while retaining separate runners and artifact directories.
* **A baked runner.**  The ``tmle3`` and ``ctmle`` studies ``COPY`` the script into the image
  and name it in the ``ENTRYPOINT``, so the container takes only the three data paths.
* **No reference at all.**  ``cvtmle_fold`` compares against nothing, because no maintained
  package ships its construction.  It writes a valid *empty* equivalence artefact rather than a
  surrogate comparison, and its ``draw_and_fit`` returns rows directly instead of the
  ``(samples, truths, rows)`` triple a paired study has to hand to R.

RM39 made a run resumable.  Every run keeps its scratch in a persistent directory keyed by a
*resume key*: the declaration, the code, the reference sources, the inputs and the Python
versions.  An R reference checkpoints each group in the Docker volume ``cleverly-cache``, and
a rerun with the same key reuses the Python phase, the calibration and every finished group.
A container killed whole is run again once with half the workers.  Every error is still fatal.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import dataclasses
import enum
import gzip
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from tests.parallel import available_cores
from tests.studies.evidence.comparison import empty_equivalence, equivalence
from tests.studies.evidence.manifest import write_csv, write_manifest
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.properties import REPLICATE_COLUMNS as PROPERTY_COLUMNS
from tests.studies.evidence.registry import ROOT, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, validate_replicates
from tests.studies.evidence.seeds import reference_seed

#: Every published artefact, in the order the manifest hashes them.
ARTIFACT_NAMES = (
    "replicates.csv.gz",
    "summary.csv",
    "equivalence.csv",
    "performance-tests.csv",
    "property-replicates.csv.gz",
    "properties.csv",
)

#: Columns the reference writes as text and the schema requires as numbers.
_TEXT_COLUMNS = frozenset({"implementation", "scenario", "estimand", "inference_scale"})

#: The R harness every mounted runner sources.  Its bytes are part of every R resume key.
HARNESS = ROOT / "tests" / "canonical" / "study_harness.R"

#: The Docker volume that holds the checkpoint leaves of every R reference run.
VOLUME = "cleverly-cache"

#: What the scratch directory reuses across runs, and so what the resume key must cover.
_PYTHON_CACHE = ("samples.csv.gz", "truth.csv", "python-rows.csv.gz")

#: Exit codes of an R reference container.  3 is the harness's "a rerun resumes" stop; 137 is
#: a container killed whole, by the kernel or by Docker.
EXIT_RESUMABLE = 3
EXIT_KILLED = (137, -9)

#: A runner that fits through one of these reaches calibration.  A probe that sources the
#: harness only for its argument check fits inline, so it gets no calibration run.
_CALIBRATES = re.compile(r"\bstudy_(?:fitter|stream|calibrate)\(")

#: Lines a declared run records in its ``run.log``.  :func:`main` fills it, and
#: ``declared_run.publish_run`` hands each line to ``run_log`` as a note.
RUN_NOTES: list[str] = []


@dataclass(frozen=True)
class Reference:
    """How to run one study's canonical comparator, and what to hash as its provenance.

    ``mount_runner`` picks between the two container conventions.  ``False`` is the baked form:
    the image's ``ENTRYPOINT`` already names the script, so the container is handed only the
    three data paths.  ``True`` binds ``runner_root`` read-only at ``/fixture`` and passes the
    script as the first argument.  The root defaults to the study directory.

    ``extra_files`` are further reference sources whose bytes belong in the manifest. Examples
    include a sourced adapter, a requirement lock, or a second diagnostic runner.
    """

    image: str
    runner: str
    mount_runner: bool = False
    extra_files: tuple[str, ...] = ()
    #: Optional shared Docker context.  The cross-fitted longitudinal studies use one
    #: digest-pinned ``lmtp`` image while keeping separate runners and artefact directories.
    build_context: Path | None = None
    #: Optional root mounted at ``/fixture``.  ``runner`` and ``extra_files`` are relative
    #: to this root when supplied; existing studies continue to resolve them from ``here``.
    runner_root: Path | None = None
    #: Program that executes a mounted runner. Existing R studies retain the historical default.
    interpreter: str = "Rscript"
    #: Group ids (``g000000007``) that calibration fits besides the first group of each
    #: scenario, because the runner knows they are the heaviest.
    calibration_groups: tuple[str, ...] = ()
    #: A floor on the measured per-worker memory, in MB.
    worker_memory_mb: float | None = None
    #: ``(factor, share, reason)`` in place of the harness defaults 1.25 and 0.85.
    memory_override: tuple[float, float, str] | None = None

    def files(self, here: Path) -> list[Path]:
        context = self.build_context or here
        root = self.runner_root or here
        return [
            context / "Dockerfile",
            root / self.runner,
            *(root / name for name in self.extra_files),
        ]

    def run(
        self,
        here: Path,
        samples: Path,
        truths: Path,
        output: Path,
        *,
        cores: int,
        runner: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Run the comparator on ``samples`` and write its rows to ``output``.

        Inside :func:`main` an R reference runs resumably (:func:`_run_resumable`).  A direct
        call, or a reference with another interpreter, runs the container once.
        """
        context = self.build_context or here
        root = self.runner_root or here
        selected_runner = self.runner if runner is None else runner
        if runner is not None and runner not in self.extra_files:
            raise ValueError(f"alternate reference runner {runner!r} is not in extra_files")
        if runner is not None and not self.mount_runner:
            raise ValueError("an alternate reference runner requires mount_runner=True")
        subprocess.run(["docker", "build", "-t", self.image, str(context)], check=True)
        active = _ACTIVE.get()
        if active is not None and self.interpreter == "Rscript":
            _run_resumable(
                self, active, here, samples, truths, output, cores, selected_runner, env or {}
            )
            return
        subprocess.run(
            _docker_command(self, root, samples, truths, output, selected_runner, cores, env),
            check=True,
        )


def _docker_command(
    reference: Reference,
    root: Path,
    samples: Path,
    truths: Path,
    output: Path,
    runner: str,
    cores: int,
    env: Mapping[str, str] | None,
    *,
    extra: tuple[str, ...] = (),
) -> list[str]:
    mounts = ["-v", f"{samples.parent.resolve()}:/work"]
    arguments: list[str] = []
    if reference.mount_runner:
        mounts += ["-v", f"{root.resolve()}:/fixture:ro"]
        arguments.append(f"/fixture/{runner}")
    entrypoint = ["--entrypoint", reference.interpreter] if reference.mount_runner else []
    variables = {
        "CLEVERLY_REFERENCE_CORES": str(cores),
        "CLEVERLY_R_CORES": str(cores),
        **(env or {}),
    }
    flags = [part for name, value in variables.items() for part in ("-e", f"{name}={value}")]
    return [
        "docker",
        "run",
        "--rm",
        *extra,
        *flags,
        *mounts,
        *entrypoint,
        reference.image,
        *arguments,
        f"/work/{samples.name}",
        f"/work/{truths.name}",
        f"/work/{output.name}",
    ]


# ---- the resume key -------------------------------------------------------------------------


def canonical(value: Any) -> Any:
    """``value`` as plain JSON data, so equal declarations hash equal in every process.

    A ``Path`` inside the repository becomes its posix path relative to the root, so two
    checkouts of one commit agree.  Any type not listed raises, rather than hashing a ``repr``
    that can carry a memory address.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, enum.Enum):
        return canonical(value.value)
    if isinstance(value, Path):
        resolved = value.resolve()
        if resolved.is_relative_to(ROOT.resolve()):
            return resolved.relative_to(ROOT.resolve()).as_posix()
        return resolved.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: canonical(getattr(value, item.name)) for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, set | frozenset):
        return sorted((canonical(item) for item in value), key=lambda item: json.dumps(item))
    if isinstance(value, tuple | list):
        return [canonical(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"no canonical form for {type(value).__name__}: {value!r}")


def digest(value: Any) -> str:
    """The sha256 of ``value``'s canonical JSON."""
    text = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def declaration_hash(record: StudyRecord) -> str:
    """The declared study, without where its artefacts go.

    ``artifacts`` is the run's ``--output``.  A declared run resumed into a new empty output
    is the same run, so the output path is not part of the key.
    """
    declared = canonical(record)
    declared.pop("artifacts", None)
    return digest(declared)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha(path: Path) -> str:
    """The sha256 of a file's content, decompressed when it is a gzip archive.

    A gzip header carries a timestamp, so the archive bytes of two equal tables can differ.
    """
    data = path.read_bytes()
    return hashlib.sha256(gzip.decompress(data) if path.name.endswith(".gz") else data).hexdigest()


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, check=True, timeout=120
    ).stdout


def git_state() -> dict[str, Any]:
    """HEAD and the tree's state.  A dirty tree adds the digest of its diff and new files."""
    head = _git_bytes("rev-parse", "HEAD").decode().strip()
    status = _git_bytes("status", "--porcelain", "--untracked-files=all").decode()
    state: dict[str, Any] = {"head": head, "clean": status == ""}
    if status:
        untracked = sorted(line[3:] for line in status.splitlines() if line.startswith("?? "))
        state["dirty"] = digest(
            {
                "diff": hashlib.sha256(_git_bytes("diff", "HEAD", "--binary")).hexdigest(),
                "untracked": {
                    name: _file_sha(ROOT / name) for name in untracked if (ROOT / name).is_file()
                },
            }
        )
    return state


def python_versions() -> dict[str, str]:
    """The Python side's versions, as ``generated_with.subject`` records them."""
    import platform

    import numpy
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_key(record: StudyRecord) -> dict[str, Any]:
    """The resume-key fields known before the Python phase draws anything."""
    return {
        "declaration": declaration_hash(record),
        "git": git_state(),
        "n": record.n,
        "replicates": record.replicates,
        "versions": python_versions(),
    }


def _repository_name(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT.resolve()):
        return resolved.relative_to(ROOT.resolve()).as_posix()
    return resolved.as_posix()


def leaf_key(
    reference: Reference, key: Mapping[str, Any], here: Path, samples: Path, truths: Path
) -> dict[str, Any]:
    """Every resume-key field of one reference run, except the image ID.

    The seeds file is not listed: it is a function of the declaration and the runner name,
    which the key holds.  The group-id list is not listed: it is a function of the runner and
    the samples, which the key holds too.
    """
    files = [*reference.files(here)]
    if reference.interpreter == "Rscript":
        files.append(HARNESS)
    return {
        **key,
        "reference": {_repository_name(path): _file_sha(path) for path in files},
        "samples": content_sha(samples),
        "truth": content_sha(truths),
    }


def runs_root() -> Path:
    """Where each run's scratch directory lives, outside the repository."""
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".cache"
    return base / "cleverly" / "runs"


@dataclass
class RunContext:
    """One driver run: its key, its scratch and what it touched in the volume."""

    record: StudyRecord
    key: dict[str, Any]
    scratch: Path
    default_scratch: bool
    fresh: bool = False
    keep: bool = False
    completed: bool = False
    #: ``(image, /cache/<slug>/<key12>)`` of each R run, for cleanup.
    volume_keys: list[tuple[str, str]] = field(default_factory=list)
    cleared: set[str] = field(default_factory=set)


_ACTIVE: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "cleverly_regenerate_run", default=None
)


# ---- the host lock --------------------------------------------------------------------------


def process_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process on this host.

    ``os.kill(pid, 0)`` is the POSIX check, but on Windows signal 0 is ``CTRL_C_EVENT``, so
    there the process is opened and its exit code read instead.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel.GetLastError() == 5  # ERROR_ACCESS_DENIED: it exists
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def host_lock(scratch: Path) -> Iterator[None]:
    """Hold ``scratch/_lock`` for one driver run; a second driver on the same key refuses.

    A lock whose holder is a dead process on this host is reported and removed.
    """
    lock = scratch / "_lock"
    host = socket.gethostname()
    for _ in range(2):
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = dict(
                line.split("=", 1)
                for line in lock.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            pid = int(held.get("pid", "0") or 0)
            if held.get("host") == host and not process_alive(pid):
                print(f"removed the lock of a dead driver (pid {pid}) on {lock}", flush=True)
                lock.unlink(missing_ok=True)
                continue
            raise RuntimeError(
                f"refused: another driver holds {lock}: pid {pid} on {held.get('host')} "
                f"since {held.get('start')}"
            ) from None
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"pid={os.getpid()}\nhost={host}\nstart={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            )
        break
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


# ---- the resumable R run --------------------------------------------------------------------


def _docker_text(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True)


def image_id(image: str) -> str | None:
    """The local image's ID, or ``None`` when Docker has no such image."""
    completed = _docker_text("image", "inspect", "--format", "{{.Id}}", image)
    return completed.stdout.strip() if completed.returncode == 0 else None


def clear_stopped_container(name: str) -> None:
    """Remove a stopped container left by a whole-container kill; refuse a running one.

    The name is the container lock: Docker refuses a second live container of one name.
    """
    completed = _docker_text("inspect", "--format", "{{.State.Running}}", name)
    if completed.returncode != 0:
        return
    if completed.stdout.strip() == "true":
        raise RuntimeError(f"refused: container {name} is running this key already")
    _docker_text("rm", "--force", name)
    print(f"removed the stopped container {name}", flush=True)


def read_state(path: Path) -> dict[str, str]:
    """The ``key=value`` state file the harness writes, or an empty mapping."""
    if not path.exists():
        return {}
    return dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line
    )


def write_seeds(path: Path, record: StudyRecord, runner_stem: str) -> None:
    """One seed per ``(scenario, replicate)`` and per ``("*", replicate)``, all below 2^31."""
    rows = [
        {
            "scenario": scenario,
            "replicate": replicate,
            "seed": reference_seed(record, runner_stem, scenario, replicate),
        }
        for replicate in range(record.replicates)
        for scenario in (*record.scenarios, "*")
    ]
    write_csv(pd.DataFrame(rows, columns=["scenario", "replicate", "seed"]), path)


def _calibrate(
    reference: Reference,
    run: Any,
    work: Path,
    leaf: str,
    key12: str,
) -> dict[str, str]:
    """Run calibration once per key, and return the memory variables for the real run."""
    calibration = work / f"{leaf}.calibration.json"
    stamp = work / f"{leaf}.calibration.key"
    if not (calibration.exists() and stamp.exists() and stamp.read_text().strip() == key12):
        calibration.unlink(missing_ok=True)
        code = run({"CLEVERLY_HARNESS_CALIBRATE": "1"})
        if code != 0 or not calibration.exists():
            raise RuntimeError(
                f"the calibration run of {leaf} exited {code} and wrote no calibration file"
            )
        stamp.write_text(key12 + "\n", encoding="utf-8", newline="\n")
    else:
        print(f"reusing the calibration of {leaf}", flush=True)
    measured = json.loads(calibration.read_text(encoding="utf-8"))
    heaviest = max(measured["groups"], key=lambda group: group["per_worker"])
    declared = reference.worker_memory_mb or 0.0
    per_worker = max(float(heaviest["per_worker"]), float(declared))
    variables = {
        "CLEVERLY_R_WORKER_MB": f"{per_worker:.0f}",
        "CLEVERLY_R_WORKER_BASIS": "declared" if declared > heaviest["per_worker"] else "measured",
        "CLEVERLY_R_MEASURED": (
            f"group {heaviest['id']}: peak_growth {heaviest['peak_growth']:.0f} MB, "
            f"private_after_gc {heaviest['private_after_gc']:.0f} MB, "
            f"per_worker {heaviest['per_worker']:.0f} MB"
        ),
        "CLEVERLY_R_DECLARED_MB": "none" if not declared else f"{declared:.0f}",
    }
    if reference.memory_override is not None:
        factor, share, reason = reference.memory_override
        variables["CLEVERLY_R_MEMORY_FACTOR"] = repr(float(factor))
        variables["CLEVERLY_R_MEMORY_SHARE"] = repr(float(share))
        variables["CLEVERLY_R_MEMORY_REASON"] = reason
        print(f"memory override for {leaf}: factor {factor}, share {share}: {reason}", flush=True)
    return variables


def _run_resumable(
    reference: Reference,
    context: RunContext,
    here: Path,
    samples: Path,
    truths: Path,
    output: Path,
    cores: int,
    runner: str,
    env: Mapping[str, str],
) -> None:
    """One R reference run with a checkpoint leaf, a calibration and the exit table."""
    work = samples.parent.resolve()
    stem = Path(runner).stem
    leaf = f"{stem}__{output.stem}"
    fields = leaf_key(reference, context.key, here, samples, truths)
    key12 = digest(fields)[:12]
    image = image_id(reference.image)
    meta = work / f"{leaf}.meta.json"
    meta.write_text(
        json.dumps({**canonical(fields), "image": image, "leaf": leaf}, sort_keys=True, indent=1)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_seeds(work / f"{leaf}.seeds.csv", context.record, stem)
    slug = context.record.slug
    key_directory = f"/cache/{slug}/{key12}"
    if context.fresh and key_directory not in context.cleared:
        clear_volume(reference.image, key_directory)
        context.cleared.add(key_directory)
    context.volume_keys.append((reference.image, key_directory))
    checkpoint = f"{key_directory}/{leaf}"
    state = work / f"{leaf}.state"
    name = f"cleverly-{slug}-{key12}-{leaf}".lower()
    base = {
        "CLEVERLY_CHECKPOINT": checkpoint,
        "CLEVERLY_R_META": f"/work/{meta.name}",
        "CLEVERLY_R_SEEDS": f"/work/{leaf}.seeds.csv",
        "CLEVERLY_R_STATE": f"/work/{state.name}",
        "CLEVERLY_R_CALIBRATION": f"/work/{leaf}.calibration.json",
        "CLEVERLY_R_CALIBRATION_GROUPS": ",".join(reference.calibration_groups),
        **env,
    }
    root = reference.runner_root or here

    def run(extra: Mapping[str, str]) -> int:
        clear_stopped_container(name)
        state.unlink(missing_ok=True)
        command = _docker_command(
            reference,
            root,
            samples,
            truths,
            output,
            runner,
            cores,
            {**base, **extra},
            extra=("--init", "--name", name, "-v", f"{VOLUME}:/cache"),
        )
        return subprocess.run(command).returncode

    source = (root / runner).read_text(encoding="utf-8")
    memory = _calibrate(reference, run, work, leaf, key12) if _CALIBRATES.search(source) else {}
    cap: int | None = None
    for attempt in (1, 2):
        variables = dict(memory)
        if cap is not None:
            variables["CLEVERLY_R_WORKER_CAP"] = str(cap)
        code = run(variables)
        held = read_state(state)
        if "reused" in held:
            RUN_NOTES.append(f"resumed from {checkpoint}, reused {held['reused']} groups")
        memory_line = f"MemAvailable {held.get('mem_available_mb', 'unknown')} MB"
        if code == 0:
            stamp = output.with_name(f"{output.name}.key")
            stamp.write_text(f"{key12}\n{image}\n", encoding="utf-8", newline="\n")
            return
        if code == EXIT_RESUMABLE:
            raise RuntimeError(
                f"the reference stopped: {held.get('stop', 'see its output')}. "
                f"A rerun resumes from {checkpoint}"
            )
        if code in EXIT_KILLED and attempt == 1:
            workers = int(held.get("workers", cores) or cores)
            cap = max(1, workers // 2)
            print(
                f"container killed: {memory_line}, last workers {workers}; "
                f"running it again with CLEVERLY_R_WORKER_CAP={cap}",
                flush=True,
            )
            continue
        if code in EXIT_KILLED:
            raise RuntimeError(
                f"container killed twice, rerun to resume; {memory_line}; "
                f"checkpoints are kept in {checkpoint}"
            )
        raise RuntimeError(
            f"the reference container exited {code}: {held.get('stop', 'see its output')}. "
            f"Checkpoints are kept in {checkpoint}"
        )


def clear_volume(image: str, key_directory: str) -> None:
    """Delete one key directory in the volume, never the volume, with an image already here."""
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{VOLUME}:/cache",
            "--entrypoint",
            "rm",
            image,
            "-rf",
            key_directory,
        ],
        check=True,
    )


def cleanup(context: RunContext) -> None:
    """Delete what a finished run keeps, or print where a declared run keeps it."""
    keys = sorted(set(context.volume_keys))
    if context.keep:
        for _, key_directory in keys:
            print(f"kept the volume key {VOLUME}:{key_directory} for audit", flush=True)
        print(f"kept the host scratch {context.scratch} for audit", flush=True)
        return
    for image, key_directory in keys:
        clear_volume(image, key_directory)
    if context.default_scratch:
        shutil.rmtree(context.scratch, ignore_errors=True)


# ---- the driver -----------------------------------------------------------------------------


@dataclass
class _Phase:
    """The Python side's output, whichever shape the study's ``draw_and_fit`` returns."""

    rows: pd.DataFrame
    samples: pd.DataFrame | None = None
    truths: pd.DataFrame | None = None
    cached: bool = False
    paths: dict[str, Path] = field(default_factory=dict)
    #: The reference-result file ``_reference_rows`` actually read or wrote.  A study's
    #: ``reference_artifacts`` hook reads the same rows, and coupling the two by a literal
    #: file name broke the moment the driver renamed the file.
    reference_results: Path | None = None


def _arguments(study: ModuleType, here: Path, reference: Reference | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Regenerate {study.STUDY.name} artefacts.")
    parser.add_argument("--replicates", type=int, default=study.PRIMARY_REPLICATES)
    parser.add_argument("--n", type=int, default=study.PRIMARY_N)
    parser.add_argument("--skip-properties", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="write disposable primary diagnostics without properties or a manifest",
    )
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--jobs", type=int, default=available_cores())
    parser.add_argument(
        "--cache",
        type=Path,
        help="keep the run's scratch in this directory instead of the per-key default",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard this key's cached phases and checkpoints before the run",
    )
    # A declared run keeps its scratch and checkpoints until its PR merges, for audit.
    parser.add_argument("--keep-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--refresh-python",
        action="store_true",
        help="refit the Python phase while retaining a compatible cached reference result",
    )
    if reference is not None:
        parser.add_argument(
            "--skip-reference",
            "--skip-r",
            dest="skip_reference",
            action="store_true",
            help="reuse the committed reference rows",
        )
        parser.add_argument(
            "--reference-jobs",
            "--r-jobs",
            dest="reference_jobs",
            type=int,
            help="reference-process concurrency (defaults to --jobs)",
        )
    arguments = parser.parse_args()
    if arguments.replicates < 2 or arguments.n < 50:
        parser.error("replicates must be >= 2 and n must be >= 50")
    if arguments.replicates != study.PRIMARY_REPLICATES and not arguments.skip_properties:
        # A run whose count differs from the declared primary budget, in either direction,
        # is not a publishing run.  Each property module fixes its own budget, so the property
        # phase would run the declared study in full and show its verdicts before the
        # declaration is committed.  RM22's smoke run did exactly that.  Such a run writes
        # schema-bearing placeholders instead.
        print(
            f"--replicates {arguments.replicates} is not the declared "
            f"{study.PRIMARY_REPLICATES}, so the property study is skipped",
            flush=True,
        )
        arguments.skip_properties = True
    if getattr(arguments, "reference_jobs", None) is not None and arguments.reference_jobs < 1:
        parser.error("--reference-jobs must be >= 1")
    return arguments


def _claim_scratch(scratch: Path, key: Mapping[str, Any], *, fresh: bool) -> None:
    """Record the resume key in ``scratch``, or refuse a scratch written under another key."""
    stamp = scratch / "_key.json"
    text = json.dumps(canonical(key), sort_keys=True, indent=1) + "\n"
    cached = [
        name for name in (*_PYTHON_CACHE, "reference-results.csv") if (scratch / name).exists()
    ]
    if fresh:
        for name in cached:
            (scratch / name).unlink()
        for pattern in ("*.calibration.json", "*.calibration.key", "*.csv.key"):
            for path in scratch.glob(pattern):
                path.unlink()
    elif cached and (not stamp.exists() or stamp.read_text(encoding="utf-8") != text):
        raise RuntimeError(
            f"refused: {scratch} holds a run under another resume key; pass --fresh to "
            f"discard it, or choose another --cache"
        )
    stamp.write_text(text, encoding="utf-8", newline="\n")


def _python_phase(study: ModuleType, arguments: argparse.Namespace, scratch: Path) -> _Phase:
    """Draw the samples and fit the subject, reusing the scratch's phase when it holds one.

    :func:`main` keys the scratch by the resume key, so a phase found there was drawn under
    the same declaration, code and versions.
    """
    paths = {
        name: scratch / name
        for name in (
            "samples.csv.gz",
            "truth.csv",
            "python-rows.csv.gz",
            "reference-results.csv",
        )
    }
    if not arguments.refresh_python and all(paths[name].exists() for name in _PYTHON_CACHE):
        rows = pd.read_csv(paths["python-rows.csv.gz"])
        print(f"reusing the cached Python phase in {scratch}", flush=True)
        RUN_NOTES.append(f"resumed from {scratch}, reused the Python phase")
        return _Phase(rows=rows, cached=True, paths=paths)

    drawn = study.draw_and_fit(
        replicates=arguments.replicates, n=arguments.n, n_jobs=arguments.jobs
    )
    if not isinstance(drawn, tuple):
        # A study with no comparator has no reason to keep the realized rows around: nothing
        # else ever reads them, so it returns the estimate table straight out.
        return _Phase(rows=drawn, paths=paths)
    samples, truths, rows = drawn
    archive = {"method": "gzip", "mtime": 0}
    write_csv(samples, paths["samples.csv.gz"], compression=archive)
    write_csv(truths, paths["truth.csv"])
    write_csv(rows, paths["python-rows.csv.gz"], compression=archive)
    return _Phase(rows=rows, samples=samples, truths=truths, paths=paths)


def _reference_key(reference: Reference, here: Path, phase: _Phase) -> str | None:
    """What a cached reference result must have been written under, or ``None``.

    The key is the reference run's resume key and its image ID: equal sources, inputs,
    declaration, code and versions.  Outside :func:`main` there is no key.
    """
    active = _ACTIVE.get()
    samples = phase.paths.get("samples.csv.gz")
    truths = phase.paths.get("truth.csv")
    if active is None or samples is None or truths is None or not samples.exists():
        return None
    fields = leaf_key(reference, active.key, here, samples, truths)
    return f"{digest(fields)[:12]}\n{image_id(reference.image)}\n"


def _reference_rows(
    study: ModuleType,
    reference: Reference,
    arguments: argparse.Namespace,
    here: Path,
    phase: _Phase,
) -> pd.DataFrame:
    # ``--skip-reference`` is only registered when the study declares a reference, so the one
    # ``getattr`` here stands in for a flag an unpaired study never carries.
    if getattr(arguments, "skip_reference", False):
        committed = pd.read_csv(here / "replicates.csv.gz")
        rows = committed.loc[
            (committed["implementation"] == study.STUDY.reference)
            & (committed["replicate"] < arguments.replicates)
            & (committed["n"] == arguments.n)
        ]
        if rows.empty:
            raise RuntimeError("no compatible committed reference rows for --skip-reference")
        return rows
    cached_reference = phase.paths["reference-results.csv"]
    phase.reference_results = cached_reference
    stamp = cached_reference.with_name(f"{cached_reference.name}.key")
    expected = _reference_key(reference, here, phase)
    if (
        expected is not None
        and cached_reference.exists()
        and stamp.exists()
        and stamp.read_text(encoding="utf-8") == expected
    ):
        print(f"reusing the cached reference phase in {cached_reference.parent}", flush=True)
        RUN_NOTES.append(f"resumed from {cached_reference.parent}, reused the reference phase")
        return pd.read_csv(cached_reference)
    stamp.unlink(missing_ok=True)
    reference.run(
        here,
        phase.paths["samples.csv.gz"],
        phase.paths["truth.csv"],
        cached_reference,
        cores=(getattr(arguments, "reference_jobs", None) or arguments.jobs),
    )
    if expected is not None and reference.interpreter != "Rscript":
        # An R run writes its own stamp when it completes; any other reference gets it here.
        stamp.write_text(expected, encoding="utf-8", newline="\n")
    return pd.read_csv(cached_reference)


def _property_artifacts(
    properties: ModuleType,
    arguments: argparse.Namespace,
    here: Path,
    paths: dict[str, Path],
) -> pd.DataFrame | None:
    """The property study, or the committed rows, or a schema-bearing placeholder.

    Returns the summary when it was computed, and ``None`` when the artefacts were reused or
    stubbed -- so the failure gate below can tell "every cell passed" from "no cell ran".
    """
    if not arguments.skip_properties:
        rows = properties.generate_property_rows(n_jobs=arguments.jobs)
        write_csv(
            rows, paths["property-replicates.csv.gz"], compression={"method": "gzip", "mtime": 0}
        )
        summary = properties.summarize_properties(rows)
        write_csv(summary, paths["properties.csv"])
        return summary

    names = ("property-replicates.csv.gz", "properties.csv")
    if all((here / name).exists() for name in names):
        for name in names:
            if paths[name].resolve() != (here / name).resolve():
                paths[name].write_bytes((here / name).read_bytes())
        return None
    if paths["properties.csv"].parent.resolve() == here.resolve():
        raise RuntimeError("skipping the property study needs committed property artefacts")
    # A disposable smoke run has nothing to reuse.  Empty, schema-bearing files keep its
    # manifest complete without pretending the statistical study ran; publication to the
    # committed directory still refuses this state above.
    write_csv(
        pd.DataFrame(columns=list(PROPERTY_COLUMNS)),
        paths["property-replicates.csv.gz"],
        compression={"method": "gzip", "mtime": 0},
    )
    write_csv(pd.DataFrame(columns=["property", "cell", "passed"]), paths["properties.csv"])
    return None


def main(
    study: ModuleType,
    properties: ModuleType,
    *,
    here: Path,
    reference: Reference | None = None,
) -> None:
    """Regenerate one study's committed artefacts, and refuse the run if a gate failed.

    The run holds a host lock on its scratch directory.  Once its artefacts are written, a
    run deletes its scratch and its checkpoints, unless it is a declared run, which keeps them
    for audit.
    """
    arguments = _arguments(study, here, reference)
    RUN_NOTES.clear()
    declared = dataclasses.replace(study.STUDY, replicates=arguments.replicates, n=arguments.n)
    key = run_key(declared)
    cache = getattr(arguments, "cache", None)
    scratch = Path(cache) if cache else runs_root() / declared.slug / digest(key)[:12]
    scratch.mkdir(parents=True, exist_ok=True)
    context = RunContext(
        record=declared,
        key=key,
        scratch=scratch,
        default_scratch=not cache,
        fresh=getattr(arguments, "fresh", False),
        keep=getattr(arguments, "keep_cache", False),
    )
    print(f"run scratch: {scratch}", flush=True)
    try:
        with host_lock(scratch):
            _claim_scratch(scratch, key, fresh=context.fresh)
            token = _ACTIVE.set(context)
            try:
                _regenerate(study, properties, arguments, here, reference, context)
            finally:
                _ACTIVE.reset(token)
    finally:
        if context.completed:
            cleanup(context)


def _regenerate(
    study: ModuleType,
    properties: ModuleType,
    arguments: argparse.Namespace,
    here: Path,
    reference: Reference | None,
    context: RunContext,
) -> None:
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)
    record = dataclasses.replace(
        study.STUDY, replicates=arguments.replicates, n=arguments.n, artifacts=out
    )
    skip_reference = getattr(arguments, "skip_reference", False)
    if skip_reference and record.extra_artifacts:
        raise RuntimeError(
            "--skip-reference cannot rebuild study-specific extra artifacts: the standard "
            "replicate file intentionally omits their source columns"
        )
    # The refusal above covers every hook owner today, because each one also declares an
    # extra artefact.  Nothing enforces that coincidence, and ``tests/`` is not type checked,
    # so the missing file would otherwise reach the hook as ``None`` against a ``Path``
    # annotation and fail inside ``pd.read_csv`` on a message about the file rather than
    # about the flag that removed it.  Refuse the pair here, before any fit runs.
    if skip_reference and hasattr(study, "reference_artifacts"):
        raise RuntimeError(
            f"--skip-reference runs no reference process, so {record.slug} has no "
            "reference-result file for its reference_artifacts hook to read"
        )
    print(f"regenerating {record.name}: {record.replicates} x n={record.n}", flush=True)

    scratch = context.scratch
    phase = _python_phase(study, arguments, scratch)
    rows = phase.rows
    if reference is not None:
        rows = pd.concat(
            [phase.rows, _reference_rows(study, reference, arguments, here, phase)],
            ignore_index=True,
        )
    reference_extra_frames: dict[str, pd.DataFrame] = {}
    if reference is not None and hasattr(study, "reference_artifacts"):
        reference_extra_frames = study.reference_artifacts(
            reference=reference,
            here=here,
            samples=phase.paths["samples.csv.gz"],
            truths_path=phase.paths["truth.csv"],
            reference_results=phase.reference_results,
            output=scratch,
            cores=(getattr(arguments, "reference_jobs", None) or arguments.jobs),
        )

    extra_frames = dict(reference_extra_frames)
    row_extra_frames = study.extra_artifacts(rows) if hasattr(study, "extra_artifacts") else {}
    overlap = set(extra_frames) & set(row_extra_frames)
    if overlap:
        raise RuntimeError(
            f"duplicate extra artifacts from reference and row hooks: {sorted(overlap)}"
        )
    extra_frames.update(row_extra_frames)
    if set(extra_frames) != set(record.extra_artifacts):
        raise RuntimeError(
            f"{record.slug} produced extra artifacts {sorted(extra_frames)}, "
            f"expected {sorted(record.extra_artifacts)}"
        )
    for column in REPLICATE_COLUMNS:
        if column not in _TEXT_COLUMNS:
            rows[column] = pd.to_numeric(rows[column], errors="raise")
    rows = rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["scenario", "replicate", "estimand", "implementation"], ignore_index=True
    )
    validate_replicates(rows, record=record)

    artifact_names = (*ARTIFACT_NAMES, *record.extra_artifacts)
    paths = {name: out / name for name in artifact_names}
    write_csv(rows, paths["replicates.csv.gz"], compression={"method": "gzip", "mtime": 0})
    summaries = summarize(rows, truth_varies=record.truth_varies_by_replicate)
    write_csv(summaries, paths["summary.csv"])
    performance = independent_performance_tests(rows, record=record, n_jobs=arguments.jobs)
    write_csv(performance, paths["performance-tests.csv"])
    paired = (
        empty_equivalence()
        if reference is None
        else equivalence(rows, summaries, performance, record=record, n_jobs=arguments.jobs)
    )
    write_csv(paired, paths["equivalence.csv"])
    for name, frame in extra_frames.items():
        compression = {"method": "gzip", "mtime": 0} if name.endswith(".csv.gz") else None
        write_csv(frame, paths[name], compression=compression)

    if arguments.primary_only:
        print(performance.to_string(index=False))
        print(paired.to_string(index=False))
        print(
            "primary-only probe: no property artefacts or publishable manifest were written",
            flush=True,
        )
        context.completed = True
        return

    property_summary = _property_artifacts(properties, arguments, here, paths)

    write_manifest(
        out / "manifest.json",
        record,
        [paths[name] for name in artifact_names],
        reference_files=() if reference is None else reference.files(here),
        reference_metadata=getattr(study, "REFERENCE_METADATA", None),
        configuration=study.CONFIGURATION,
    )
    context.completed = True

    independent_failures = performance.loc[~performance["passed"]]
    if reference is not None and record.accepted_reference_failure:
        independent_failures = independent_failures.loc[
            independent_failures["implementation"] != record.reference
        ]
    failures = {
        "independent performance": independent_failures,
        # Both columns, not just ``passed``: a family whose claim spans its cells records that
        # verdict in ``property_passed`` alone, so gating on the per-row column would let a
        # failed joint claim through while every row read green.
        "statistical property": None
        if property_summary is None
        else property_summary.loc[
            ~property_summary["passed"] | ~property_summary["property_passed"]
        ],
    }
    if reference is not None:
        failures["paired similarity and non-inferiority"] = paired.loc[~paired["passed"]]
        # The subject's own verdict is gated unconditionally above.  This one is about the
        # *comparator*, and a study may declare in advance that its comparator fails its own
        # truth gates while remaining a usable similarity and non-inferiority reference --
        # see ``StudyRecord.accepted_reference_failure``.  Without the declaration the run
        # still refuses, so an unannounced reference regression cannot pass silently.
        if not record.accepted_reference_failure:
            failures["reference validity"] = paired.loc[~paired["reference_valid"]]
        elif paired["reference_valid"].all():
            raise RuntimeError(
                f"{record.slug} declares an accepted reference failure "
                f"({record.accepted_reference_failure!r}) but every reference row is valid; "
                f"remove the declaration rather than carrying a stale exception"
            )
    if hasattr(study, "scientific_failures"):
        failures.update(study.scientific_failures(extra_frames))
    reported = {
        name: frame for name, frame in failures.items() if frame is not None and not frame.empty
    }
    if reported and record.publication_policy == "gated" and not arguments.allow_failures:
        raise RuntimeError(
            "\n\n".join(
                f"{name} gates failed:\n{frame.to_string(index=False)}"
                for name, frame in reported.items()
            )
        )
    if reported and record.publication_policy == "reporting":
        print(
            "reporting policy: published scientific failures:\n"
            + "\n\n".join(
                f"{name}:\n{frame.to_string(index=False)}" for name, frame in reported.items()
            ),
            flush=True,
        )
    print(f"wrote {len(artifact_names)} artefacts and a manifest to {out}", flush=True)

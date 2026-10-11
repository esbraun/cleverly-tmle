"""The RM39 harness smoke: every R behavior of the resilience harness, run in Docker.

usage: python -m tests.canonical.harness_fixture.smoke [--work DIR]

CI installs no R, so no fast test can run the harness.  This smoke runs the fixture runners in
the ``tmle3`` image, one container at a time, and writes what it observed:

* ``smoke/results.json``: one entry per case, with the facts the case asserts;
* ``smoke/<table>.csv``: each distinct published table the cases wrote;
* ``fixture-manifest.json``: the sha256 of each input file and each output, the image ID and
  the date.

``tests/unit/test_harness_fixture.py`` gates that evidence in the fast tier.  It fails when an
input file no longer has its recorded sha256, so an edit to the harness, either fixture runner,
this file or the ``tmle3`` Dockerfile fails until this smoke runs again.  The image ID is
recorded, not gated.

The smoke is machine-heavy: its memory cases hold a parent heap of 1.5 GB and run under a 6 GB
container limit.  Run it only when no study runs on the machine.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
OUTPUTS = HERE / "smoke"
MANIFEST = HERE / "fixture-manifest.json"
IMAGE = "cleverly-tmle3-reference:ed72f8a"
CONTEXT = CANONICAL / "tmle3"
#: The harness every recorded manifest hashes, as a git blob of the commit RM39 started from.
OLD_HARNESS = "572501b8:tests/canonical/study_harness.R"

#: The files whose bytes the evidence describes.  The fast tier gates each one.
INPUTS = (
    CANONICAL / "study_harness.R",
    HERE / "fixture_runner.R",
    HERE / "fixture_random_runner.R",
    HERE / "smoke.py",
    CONTEXT / "Dockerfile",
)

REPLICATES = 12
SCENARIOS = ("a", "b")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_inputs(work: Path) -> None:
    """Twelve replicates of two scenarios, three rows each, and one seed per group."""
    rows = [
        {"scenario": scenario, "replicate": replicate, "y": replicate * 10 + row + offset}
        for replicate in range(REPLICATES)
        for scenario, offset in zip(SCENARIOS, (0.0, 0.5), strict=True)
        for row in range(3)
    ]
    pd.DataFrame(rows).to_csv(
        work / "samples.csv.gz",
        index=False,
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame({"scenario": ["a"], "replicate": [0], "truth": [0.0]}).to_csv(
        work / "truth.csv", index=False, lineterminator="\n"
    )
    seeds = [
        {"scenario": scenario, "replicate": replicate, "seed": 1000 + 7 * replicate + index}
        for replicate in range(REPLICATES)
        for index, scenario in enumerate((*SCENARIOS, "*"))
    ]
    pd.DataFrame(seeds).to_csv(work / "seeds.csv", index=False, lineterminator="\n")
    old = subprocess.run(
        ["git", "show", OLD_HARNESS], cwd=ROOT, capture_output=True, check=True
    ).stdout
    (work / "old_harness.R").write_bytes(old)


@dataclass
class Outcome:
    """What one container left behind."""

    code: int
    log: str
    table: Path | None
    fits: pd.DataFrame
    checkpoints: list[str]
    seconds: float
    calibration: dict[str, Any] = field(default_factory=dict)

    @property
    def forked(self) -> bool:
        """Every fit ran in a child of the R process that loaded the harness."""
        return bool(len(self.fits)) and bool((self.fits["pid"] != self.fits["parent"]).all())

    def passes(self) -> list[str]:
        return re.findall(r"^pass: .*$", self.log, re.M)


class Smoke:
    def __init__(self, work: Path) -> None:
        self.work = work

    def run(
        self,
        name: str,
        *,
        runner: str = "fixture_runner.R",
        env: Mapping[str, str] | None = None,
        old_harness: bool = False,
        memory: str = "2g",
        cpus: int = 2,
        keep: bool = False,
    ) -> Outcome:
        """One container.  ``keep`` resumes from the case's checkpoints and log."""
        checkpoint = self.work / f"ckpt_{name}"
        output = self.work / f"out_{name}.csv"
        fits = self.work / f"fits_{name}.csv"
        if not keep:
            for path in (output, fits):
                path.unlink(missing_ok=True)
            if checkpoint.exists():
                for child in checkpoint.iterdir():
                    child.unlink()
        variables = {
            "CLEVERLY_R_CORES": "2",
            "CLEVERLY_CHECKPOINT": f"/work/{checkpoint.name}",
            "CLEVERLY_R_SEEDS": "/work/seeds.csv",
            "CLEVERLY_FIXTURE_LOG": f"/work/{fits.name}",
            **(env or {}),
        }
        mounts = ["-v", f"{self.work}:/work", "-v", f"{CANONICAL}:/fixture:ro"]
        if old_harness:
            mounts += ["-v", f"{self.work / 'old_harness.R'}:/fixture/study_harness.R:ro"]
        command = [
            "docker",
            "run",
            "--rm",
            "--init",
            "-m",
            memory,
            "--cpus",
            str(cpus),
            *(part for item in variables.items() for part in ("-e", "=".join(item))),
            *mounts,
            "--entrypoint",
            "Rscript",
            IMAGE,
            f"/fixture/harness_fixture/{runner}",
            "/work/samples.csv.gz",
            "/work/truth.csv",
            f"/work/{output.name}",
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        log = completed.stdout + completed.stderr
        print(f"{name}: exit {completed.returncode} in {seconds:.1f}s", flush=True)
        columns = ["id", "pid", "parent", "time"]
        table = (
            pd.read_csv(fits, header=None, names=columns)
            if fits.exists()
            else pd.DataFrame(columns=columns)
        )
        calibration: dict[str, Any] = {}
        target = variables.get("CLEVERLY_R_CALIBRATION")
        if target and (self.work / Path(target).name).exists():
            calibration = json.loads((self.work / Path(target).name).read_text(encoding="utf-8"))
        return Outcome(
            code=completed.returncode,
            log=log,
            table=output if output.exists() else None,
            fits=table,
            checkpoints=sorted(path.name for path in checkpoint.iterdir())
            if checkpoint.exists()
            else [],
            seconds=seconds,
            calibration=calibration,
        )

    def rscript(self, expression: str, *, memory: str | None, env: Mapping[str, str]) -> str:
        """``Rscript -e`` with the harness sourced, for the plan arithmetic."""
        command = [
            "docker",
            "run",
            "--rm",
            *(["-m", memory] if memory else []),
            *(part for item in env.items() for part in ("-e", "=".join(item))),
            "-v",
            f"{CANONICAL}:/fixture:ro",
            "--entrypoint",
            "Rscript",
            IMAGE,
            "-e",
            f'source("/fixture/study_harness.R"); {expression}',
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        return completed.stdout + completed.stderr


def table_sha(outcome: Outcome) -> str | None:
    return None if outcome.table is None else sha256(outcome.table)


def driver_case(work: Path, parent_kill: str) -> None:
    """The whole-container kill, through the driver's own ``Reference.run`` (Q-h).

    Run in a child process, so the smoke can capture the container output the driver passes
    through.
    """
    import dataclasses

    from tests.canonical import regenerate
    from tests.studies import canonical_tmle

    record = dataclasses.replace(
        canonical_tmle.STUDY, slug="harness-fixture-smoke", replicates=REPLICATES, n=6
    )
    context = regenerate.RunContext(
        record=record,
        key={"smoke": f"{time.time_ns()}-{parent_kill}"},
        scratch=work,
        default_scratch=False,
    )
    reference = regenerate.Reference(
        image=IMAGE,
        runner="harness_fixture/fixture_runner.R",
        mount_runner=True,
        build_context=CONTEXT,
        runner_root=CANONICAL,
    )
    token = regenerate._ACTIVE.set(context)
    output = work / f"driver_{parent_kill}.csv"
    try:
        reference.run(
            CANONICAL,
            work / "samples.csv.gz",
            work / "truth.csv",
            output,
            cores=4,
            env={"CLEVERLY_FIXTURE_PARENT_KILL": parent_kill},
        )
        print("DRIVER-COMPLETE", flush=True)
    except RuntimeError as error:
        print(f"DRIVER-STOPPED: {error}", flush=True)
    finally:
        regenerate._ACTIVE.reset(token)
        regenerate.cleanup(context)


def run_cases(smoke: Smoke) -> tuple[dict[str, Any], dict[str, Path]]:
    """Every case of plan section 3.10 that needs R.  Returns the facts and the tables."""
    facts: dict[str, Any] = {}
    tables: dict[str, Path] = {}

    plain = smoke.run("plain")
    assert plain.code == 0 and plain.table is not None, plain.log
    tables["plain"] = plain.table
    facts["plain"] = {"exit": plain.code, "table": table_sha(plain), "forked": plain.forked}
    facts["run_log"] = {
        label: label in plain.log
        for label in (
            "MemTotal",
            "MemAvailable",
            "cgroup limit",
            "calibration:",
            "declared per-worker MB:",
            "parent private MB:",
            "factor",
            "share",
            "granted",
            "worker cap",
            "workers",
            "chunk",
            "binding",
        )
    }

    def case(name: str, **options: Any) -> Outcome:
        outcome = smoke.run(name, **options)
        facts[name] = {
            "exit": outcome.code,
            "table": table_sha(outcome),
            "forked": outcome.forked,
            "passes": outcome.passes(),
        }
        return outcome

    case("forks_one_worker", env={"CLEVERLY_R_CORES": "1"})
    case("forks_several_workers", env={"CLEVERLY_R_CORES": "4"}, cpus=4)
    case(
        "forks_one_chunk",
        env={
            "CLEVERLY_FIXTURE_MODE": "preschedule",
            "CLEVERLY_FIXTURE_KILL": "g000000003",
            "CLEVERLY_R_CHUNK": "100",
        },
    )
    case("forks_onecore_mclapply", env={"CLEVERLY_FIXTURE_MODE": "onecore"})
    kill = case("kill", env={"CLEVERLY_FIXTURE_KILL": "g000000003"})
    facts["kill"]["solo_second_attempt_forked"] = bool(
        (kill.fits.loc[kill.fits["id"] == "g000000003", "pid"] != kill.fits["parent"].iloc[0]).all()
    )
    facts["kill"]["attempts"] = int((kill.fits["id"] == "g000000003").sum())
    case(
        "preschedule_kill",
        env={
            "CLEVERLY_FIXTURE_MODE": "preschedule",
            "CLEVERLY_FIXTURE_KILL": "g000000003",
            "CLEVERLY_FIXTURE_SLEEP": "0.3",
        },
    )
    loop = case(
        "killed_in_loop",
        env={
            "CLEVERLY_FIXTURE_MODE": "preschedule",
            "CLEVERLY_FIXTURE_KILL": "g000000003,g000000007",
            "CLEVERLY_FIXTURE_SLEEP": "0.3",
        },
    )
    facts["killed_in_loop"]["attempts_g7"] = int((loop.fits["id"] == "g000000007").sum())
    twice = case(
        "killed_twice",
        env={"CLEVERLY_FIXTURE_KILL": "g000000003", "CLEVERLY_FIXTURE_KILL_ATTEMPTS": "2"},
    )
    facts["killed_twice"]["names_id"] = "incomplete: g000000003 were killed twice" in twice.log
    facts["killed_twice"]["memory_line"] = "MemAvailable" in twice.log.split("incomplete:")[-1]
    case(
        "concurrency_halves",
        env={"CLEVERLY_R_CORES": "4", "CLEVERLY_FIXTURE_KILL": "g000000002"},
        cpus=4,
    )
    progress = case("no_progress", env={"CLEVERLY_FIXTURE_VANISH": "g000000004"})
    facts["no_progress"]["message"] = "no progress: g000000004" in progress.log
    no_retry = case(
        "no_retry",
        env={"CLEVERLY_FIXTURE_KILL": "g000000003", "CLEVERLY_R_NO_RETRY": "1"},
    )
    facts["no_retry"]["message"] = "CLEVERLY_R_NO_RETRY=1 makes a kill fatal" in no_retry.log
    case(
        "stream_retry",
        env={
            "CLEVERLY_FIXTURE_MODE": "stream",
            "CLEVERLY_R_CORES": "4",
            "CLEVERLY_FIXTURE_KILL": "r000000002",
        },
        cpus=4,
    )

    error = case(
        "error_fatal",
        env={"CLEVERLY_FIXTURE_ERROR": "g000000005", "CLEVERLY_FIXTURE_SLEEP": "0.2"},
    )
    facts["error_fatal"]["message"] = "fixture error in g000000005" in error.log
    facts["error_fatal"]["checkpoints_kept"] = sum(
        name.endswith(".rds") for name in error.checkpoints
    )
    facts["error_fatal"]["fatal_written"] = "_fatal" in error.checkpoints
    resumed = smoke.run("error_fatal", keep=True)
    facts["error_fatal_resume"] = {
        "exit": resumed.code,
        "table": table_sha(resumed),
        "cleared_fatal": "cleared a fatal error left by an earlier run" in resumed.log,
    }

    random_full = smoke.run("random_full", runner="fixture_random_runner.R")
    assert random_full.table is not None, random_full.log
    tables["random"] = random_full.table
    facts["random_full"] = {"exit": random_full.code, "table": table_sha(random_full)}
    for name, mode, stop in (("resume_fitter", "fitter", "5"), ("resume_stream", "stream", "8")):
        stopped = smoke.run(
            name,
            runner="fixture_random_runner.R",
            env={"CLEVERLY_FIXTURE_MODE": mode, "CLEVERLY_HARNESS_STOP_AFTER": stop},
        )
        started_before = sum(name.endswith(".started") for name in stopped.checkpoints)
        again = smoke.run(
            name,
            runner="fixture_random_runner.R",
            env={"CLEVERLY_FIXTURE_MODE": mode},
            keep=True,
        )
        cleared = re.search(r"cleared (\d+) stale start markers", again.log)
        facts[name] = {
            "stopped_exit": stopped.code,
            "stale_markers": started_before,
            "cleared_markers": int(cleared.group(1)) if cleared else -1,
            "kills_counted": sum(name.endswith(".kills") for name in again.checkpoints),
            "exit": again.code,
            "table": table_sha(again),
            "reused": bool(re.search(r"resumed: reused [1-9]", again.log)),
        }
    other_cores = smoke.run(
        "random_one_core", runner="fixture_random_runner.R", env={"CLEVERLY_R_CORES": "1"}
    )
    facts["random_one_core"] = {"exit": other_cores.code, "table": table_sha(other_cores)}

    torn_first = smoke.run("torn")
    target = smoke.work / "ckpt_torn" / "g000000004.rds"
    target.write_bytes(target.read_bytes()[:10])
    torn = smoke.run("torn", keep=True)
    facts["torn"] = {
        "first_exit": torn_first.code,
        "exit": torn.code,
        "table": table_sha(torn),
        "deleted": "deleted 1 unreadable checkpoints" in torn.log,
    }

    old_plain = smoke.run("old_plain", old_harness=True, env={"CLEVERLY_FIXTURE_LOG": ""})
    old_stream = smoke.run(
        "old_stream",
        old_harness=True,
        env={"CLEVERLY_FIXTURE_LOG": "", "CLEVERLY_FIXTURE_MODE": "stream"},
    )
    new_stream = smoke.run("new_stream", env={"CLEVERLY_FIXTURE_MODE": "stream"})
    facts["old_new"] = {
        "old_fitter": table_sha(old_plain),
        "old_stream": table_sha(old_stream),
        "new_stream": table_sha(new_stream),
    }

    facts.update(memory_cases(smoke))
    facts.update(driver_cases(smoke))
    return facts, tables


def memory_cases(smoke: Smoke) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    cow: list[dict[str, Any]] = []
    for heap in (0, 500, 1500):
        calibrated = smoke.run(
            f"cow_{heap}",
            env={
                "CLEVERLY_HARNESS_CALIBRATE": "1",
                "CLEVERLY_R_CALIBRATION": f"/work/cal_cow_{heap}.json",
                "CLEVERLY_FIXTURE_HEAP_MB": str(heap),
            },
            memory="6g",
        )
        group = max(calibrated.calibration["groups"], key=lambda item: item["per_worker"])
        plan = smoke.rscript(
            f"invisible(study_worker_plan(16, {group['per_worker']}))", memory="6g", env={}
        )
        workers = int(re.search(r"workers (\d+), binding", plan).group(1))  # type: ignore[union-attr]
        cow.append(
            {
                "heap_mb": heap,
                "exit": calibrated.code,
                "private_after_gc": group["private_after_gc"],
                "peak_growth": group["peak_growth"],
                "per_worker": group["per_worker"],
                "workers_at_6g": workers,
            }
        )
    facts["memory_cow"] = cow

    transient = smoke.run(
        "transient",
        env={
            "CLEVERLY_HARNESS_CALIBRATE": "1",
            "CLEVERLY_R_CALIBRATION": "/work/cal_transient.json",
            "CLEVERLY_FIXTURE_SHAPE": "scenario",
            "CLEVERLY_FIXTURE_TRANSIENT_MB": "b:400",
        },
        memory="4g",
    )
    groups = transient.calibration["groups"]
    facts["memory_transient"] = {"exit": transient.code, "groups": groups}

    limited = smoke.rscript(
        "invisible(study_worker_plan(8, 500, 1.25, 0.85, Inf))", memory="2g", env={}
    )
    unlimited = smoke.rscript(
        "invisible(study_worker_plan(8, 100, 1.25, 0.85, Inf)); cat(study_memory_line(), '\\n')",
        memory=None,
        env={},
    )
    capped = smoke.rscript("invisible(study_worker_plan(8, 1, 1.25, 0.85, 3))", memory=None, env={})
    override = smoke.rscript(
        "invisible(study_plan(4))",
        memory=None,
        env={
            "CLEVERLY_R_WORKER_MB": "100",
            "CLEVERLY_R_MEMORY_FACTOR": "2",
            "CLEVERLY_R_MEMORY_SHARE": "0.5",
            "CLEVERLY_R_MEMORY_REASON": "smoke override",
        },
    )

    def parsed(text: str) -> dict[str, Any]:
        match = re.search(
            r"granted (\d+), worker cap (\S+), per-worker (\S+) MB, factor (\S+), share (\S+), "
            r"available (\d+) MB, workers (\d+), binding (.+)",
            text,
        )
        assert match is not None, text
        return {
            "granted": int(match.group(1)),
            "cap": match.group(2),
            "per_worker": match.group(3),
            "factor": float(match.group(4)),
            "share": float(match.group(5)),
            "available": int(match.group(6)),
            "workers": int(match.group(7)),
            "reason": match.group(8).strip(),
        }

    facts["memory_plan"] = {
        "limited": parsed(limited),
        "unlimited": parsed(unlimited),
        "unlimited_no_cgroup_limit": "cgroup limit none" in unlimited,
        "capped": parsed(capped),
        "override": parsed(override),
        "override_reason_logged": "memory override: smoke override" in override,
    }

    cost: dict[str, Any] = {}
    for heap in (0, 1500):
        outcome = smoke.run(
            f"fork_cost_{heap}",
            env={"CLEVERLY_R_CORES": "1", "CLEVERLY_FIXTURE_HEAP_MB": str(heap)},
            memory="6g",
        )
        times = outcome.fits["time"].sort_values().to_numpy()
        cost[str(heap)] = {
            "exit": outcome.code,
            "forked": outcome.forked,
            "seconds_per_fit": float((times[-1] - times[0]) / max(1, len(times) - 1)),
        }
    facts["fork_cost"] = cost
    return facts


def driver_cases(smoke: Smoke) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for parent_kill in ("1", "2"):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.canonical.harness_fixture.smoke",
                "driver-case",
                "--work",
                str(smoke.work),
                "--parent-kill",
                parent_kill,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"},
        )
        log = completed.stdout + completed.stderr
        output = smoke.work / f"driver_{parent_kill}.csv"
        facts[f"container_kill_{parent_kill}"] = {
            "complete": "DRIVER-COMPLETE" in log,
            "killed_logged": "container killed:" in log,
            "cap_passed": "CLEVERLY_R_WORKER_CAP=" in log,
            "stopped_twice": "container killed twice, rerun to resume" in log,
            "table": sha256(output) if output.exists() else None,
        }
    return facts


def publish(facts: dict[str, Any], tables: dict[str, Path]) -> None:
    """Write the outputs with LF endings, then the manifest over inputs and outputs."""
    OUTPUTS.mkdir(exist_ok=True)
    outputs: dict[str, str] = {}
    for name, path in tables.items():
        target = OUTPUTS / f"{name}.csv"
        target.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        outputs[target.relative_to(HERE).as_posix()] = sha256(target)
    results = OUTPUTS / "results.json"
    results.write_text(
        json.dumps(facts, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    outputs[results.relative_to(HERE).as_posix()] = sha256(results)
    image = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "inputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in INPUTS},
        "outputs": outputs,
        "image": image,
        "date": datetime.date.today().isoformat(),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", default="all", choices=("all", "driver-case"))
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--parent-kill", default="1")
    arguments = parser.parse_args()
    if arguments.command == "driver-case":
        driver_case(arguments.work, arguments.parent_kill)
        return
    subprocess.run(["docker", "build", "-t", IMAGE, str(CONTEXT)], check=True)
    work = arguments.work or Path(tempfile.mkdtemp(prefix="cleverly-harness-smoke-"))
    work.mkdir(parents=True, exist_ok=True)
    write_inputs(work)
    facts, tables = run_cases(Smoke(work))
    publish(facts, tables)
    print(f"wrote {MANIFEST} and {OUTPUTS}; scratch {work}", flush=True)


if __name__ == "__main__":
    main()

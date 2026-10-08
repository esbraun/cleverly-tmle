"""RM39a1: the Python half of a resumable reference run, with Docker replaced by a fake.

R behavior is smoke tier (``tests/canonical/harness_fixture/smoke.py``), because CI installs no
R.  What the driver decides is tested here: the resume key, the per-leaf input files, the host
lock, the exit table and its automatic re-invocation, the calibration hand-off, the scope by
interpreter, the cleanup rule and the seeds.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.canonical import regenerate
from tests.studies import canonical_tmle
from tests.studies.evidence import seeds
from tests.studies.evidence.registry import ROOT, registered

RECORD = dataclasses.replace(canonical_tmle.STUDY, replicates=3, n=50)
KEY: dict[str, Any] = {
    "declaration": "declared",
    "git": {"head": "abc", "clean": True},
    "n": 50,
    "replicates": 3,
    "versions": {"python": "3.13"},
}


# ---- the resume key -------------------------------------------------------------------------


def test_every_registered_record_canonicalizes() -> None:
    for record in registered():
        assert regenerate.declaration_hash(record)


def test_the_declaration_hash_is_stable_across_processes() -> None:
    script = (
        "from tests.canonical import regenerate; from tests.studies import canonical_tmle; "
        "print(regenerate.declaration_hash(canonical_tmle.STUDY))"
    )
    hashes = {
        subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert hashes == {regenerate.declaration_hash(canonical_tmle.STUDY)}


def test_two_outputs_give_one_key(tmp_path: Path) -> None:
    """A declared run resumed into a new empty ``--output`` keeps its key (M11)."""
    first = dataclasses.replace(RECORD, artifacts=tmp_path / "one")
    second = dataclasses.replace(RECORD, artifacts=tmp_path / "two")
    assert regenerate.declaration_hash(first) == regenerate.declaration_hash(second)
    assert regenerate.declaration_hash(first) != regenerate.declaration_hash(
        dataclasses.replace(first, seed=first.seed + 1)
    )


def test_canonical_refuses_a_type_it_does_not_list() -> None:
    with pytest.raises(TypeError, match="no canonical form"):
        regenerate.canonical(object())
    assert regenerate.canonical(frozenset({"b", "a"})) == ["a", "b"]
    assert regenerate.canonical(0.1) == "0.1"
    assert regenerate.canonical(ROOT / "tests") == "tests"


def _reference_tree(tmp_path: Path) -> tuple[regenerate.Reference, Path, Path, Path]:
    here = tmp_path / "study"
    here.mkdir()
    (here / "Dockerfile").write_text("FROM r\n", encoding="utf-8")
    (here / "run_fixture.R").write_text("study_fitter(groups, fit_one)\n", encoding="utf-8")
    (here / "probe.R").write_text("write.csv(out)\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    samples = work / "samples.csv.gz"
    pd.DataFrame({"replicate": [0, 1, 2], "y": [1.0, 2.0, 3.0]}).to_csv(
        samples, index=False, compression={"method": "gzip", "mtime": 0}
    )
    truths = work / "truth.csv"
    truths.write_text("replicate,truth\n0,0\n", encoding="utf-8")
    reference = regenerate.Reference(
        "image", "run_fixture.R", mount_runner=True, extra_files=("probe.R",)
    )
    return reference, here, samples, truths


@pytest.mark.parametrize(
    "change",
    ["declaration", "git", "n", "replicates", "versions", "runner", "harness", "samples", "truth"],
)
def test_each_key_field_changed_alone_changes_the_leaf_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    reference, here, samples, truths = _reference_tree(tmp_path)
    harness = tmp_path / "study_harness.R"
    harness.write_text("study_fork <- function(expr) NULL\n", encoding="utf-8")
    monkeypatch.setattr(regenerate, "HARNESS", harness)
    before = regenerate.digest(regenerate.leaf_key(reference, KEY, here, samples, truths))
    key = dict(KEY)
    if change in KEY:
        key[change] = "changed"
    elif change == "runner":
        (here / "run_fixture.R").write_text("changed\n", encoding="utf-8")
    elif change == "harness":
        harness.write_text("changed\n", encoding="utf-8")
    elif change == "samples":
        pd.DataFrame({"replicate": [0], "y": [9.0]}).to_csv(
            samples, index=False, compression="gzip"
        )
    else:
        truths.write_text("replicate,truth\n0,1\n", encoding="utf-8")
    after = regenerate.digest(regenerate.leaf_key(reference, key, here, samples, truths))
    assert after != before


def test_the_samples_key_reads_content_not_archive_bytes(tmp_path: Path) -> None:
    """A gzip header carries a time, so two archives of one table must key equal."""
    frame = pd.DataFrame({"y": [1, 2]})
    first, second = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    frame.to_csv(first, index=False, compression={"method": "gzip", "mtime": 1})
    frame.to_csv(second, index=False, compression={"method": "gzip", "mtime": 2})
    assert first.read_bytes() != second.read_bytes()
    assert regenerate.content_sha(first) == regenerate.content_sha(second)


# ---- the scratch and its key ----------------------------------------------------------------


def test_a_scratch_written_under_another_key_refuses(tmp_path: Path) -> None:
    regenerate._claim_scratch(tmp_path, KEY, fresh=False)
    (tmp_path / "python-rows.csv.gz").write_bytes(b"rows")
    regenerate._claim_scratch(tmp_path, KEY, fresh=False)
    with pytest.raises(RuntimeError, match="another resume key"):
        regenerate._claim_scratch(tmp_path, {**KEY, "n": 51}, fresh=False)
    regenerate._claim_scratch(tmp_path, {**KEY, "n": 51}, fresh=True)
    assert not (tmp_path / "python-rows.csv.gz").exists()


def test_a_cached_reference_result_is_reused_only_under_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths = _reference_tree(tmp_path)
    reference = dataclasses.replace(reference, interpreter="python")
    monkeypatch.setattr(regenerate, "image_id", lambda image: "sha256:image")
    cached = samples.parent / "reference-results.csv"
    pd.DataFrame({"replicate": [0], "n": [50]}).to_csv(cached, index=False)
    phase = regenerate._Phase(
        rows=pd.DataFrame(),
        paths={"samples.csv.gz": samples, "truth.csv": truths, "reference-results.csv": cached},
    )
    arguments = dataclasses.make_dataclass("A", ["skip_reference", "jobs"])(False, 1)
    runs: list[Any] = []
    monkeypatch.setattr(regenerate.subprocess, "run", lambda *a, **k: runs.append(a))
    context = regenerate.RunContext(record=RECORD, key=KEY, scratch=tmp_path, default_scratch=False)
    token = regenerate._ACTIVE.set(context)
    try:
        regenerate._reference_rows(None, reference, arguments, here, phase)  # type: ignore[arg-type]
        assert runs, "a result with no key stamp was reused"
        runs.clear()
        regenerate._reference_rows(None, reference, arguments, here, phase)  # type: ignore[arg-type]
        assert not runs, "a result under an equal key was rerun"
        context.key = {**KEY, "n": 51}
        regenerate._reference_rows(None, reference, arguments, here, phase)  # type: ignore[arg-type]
        assert runs, "a result under another key was reused"
    finally:
        regenerate._ACTIVE.reset(token)


# ---- the host lock --------------------------------------------------------------------------


def test_a_second_driver_on_one_key_refuses(tmp_path: Path) -> None:
    with (
        regenerate.host_lock(tmp_path),
        pytest.raises(RuntimeError, match="another driver holds"),
        regenerate.host_lock(tmp_path),
    ):
        pass
    assert not (tmp_path / "_lock").exists()


def test_a_dead_holders_lock_is_removed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    assert not regenerate.process_alive(finished.pid)
    assert regenerate.process_alive(os.getpid())
    (tmp_path / "_lock").write_text(
        f"pid={finished.pid}\nhost={socket.gethostname()}\nstart=then\n", encoding="utf-8"
    )
    with regenerate.host_lock(tmp_path):
        assert f"pid={os.getpid()}" in (tmp_path / "_lock").read_text(encoding="utf-8")
    assert "removed the lock of a dead driver" in capsys.readouterr().out


def test_a_lock_held_on_another_host_refuses(tmp_path: Path) -> None:
    (tmp_path / "_lock").write_text("pid=1\nhost=elsewhere\nstart=then\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="elsewhere"), regenerate.host_lock(tmp_path):
        pass


# ---- the resumable run, with a fake Docker --------------------------------------------------


class FakeDocker:
    """``subprocess.run`` for ``docker build`` and ``docker run``, with scripted exit codes.

    A calibration run writes ``calibration.json``.  A real run writes the state file the
    harness writes and, on exit 0, the output.
    """

    def __init__(self, codes: list[int], *, per_worker: float = 300.0) -> None:
        self.codes = list(codes)
        self.per_worker = per_worker
        self.runs: list[list[str]] = []

    @staticmethod
    def env(command: list[str]) -> dict[str, str]:
        return dict(
            command[index + 1].split("=", 1) for index, part in enumerate(command) if part == "-e"
        )

    def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["docker", "build"]:
            return subprocess.CompletedProcess(command, 0)
        self.runs.append(command)
        mount = next(
            command[index + 1]
            for index, part in enumerate(command)
            if part == "-v" and command[index + 1].endswith(":/work")
        )
        work = Path(mount.removesuffix(":/work"))
        env = self.env(command)
        if env.get("CLEVERLY_HARNESS_CALIBRATE") == "1":
            (work / Path(env["CLEVERLY_R_CALIBRATION"]).name).write_text(
                json.dumps(
                    {
                        "groups": [
                            {
                                "id": "g000000001",
                                "rss_start": 100.0,
                                "hwm_end": 150.0,
                                "peak_growth": 50.0,
                                "private_after_gc": self.per_worker - 50.0,
                                "per_worker": self.per_worker,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0)
        code = self.codes.pop(0)
        state = work / Path(env["CLEVERLY_R_STATE"]).name
        lines = ["workers=6", "reused=2", "mem_available_mb=900"]
        if code == 3:
            lines.append("stop=incomplete: g000000002 were killed twice")
        state.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if code == 0:
            (work / Path(command[-1]).name).write_text("replicate\n0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, code)


@pytest.fixture
def resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[regenerate.Reference, Path, Path, Path, regenerate.RunContext]]:
    reference, here, samples, truths = _reference_tree(tmp_path)
    monkeypatch.setattr(regenerate, "image_id", lambda image: "sha256:image")
    monkeypatch.setattr(
        regenerate, "_docker_text", lambda *a: subprocess.CompletedProcess(a, 1, "", "")
    )
    context = regenerate.RunContext(
        record=RECORD, key=KEY, scratch=tmp_path / "work", default_scratch=True
    )
    token = regenerate._ACTIVE.set(context)
    yield reference, here, samples, truths, context
    regenerate._ACTIVE.reset(token)


def _run(
    reference: regenerate.Reference, here: Path, samples: Path, truths: Path, **kw: Any
) -> None:
    output = kw.pop("output", samples.parent / "reference-results.csv")
    reference.run(here, samples, truths, output, cores=8, **kw)


def test_exit_0_completes_after_one_calibration(
    resumable: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths, context = resumable
    docker = FakeDocker([0, 0])
    monkeypatch.setattr(regenerate.subprocess, "run", docker)
    _run(reference, here, samples, truths)
    assert len(docker.runs) == 2
    calibration, real = (FakeDocker.env(command) for command in docker.runs)
    assert calibration["CLEVERLY_HARNESS_CALIBRATE"] == "1"
    assert real["CLEVERLY_R_WORKER_MB"] == "300"
    assert real["CLEVERLY_R_WORKER_BASIS"] == "measured"
    assert "per_worker 300 MB" in real["CLEVERLY_R_MEASURED"]
    assert "--init" in docker.runs[1]
    assert f"{regenerate.VOLUME}:/cache" in docker.runs[1]
    assert real["CLEVERLY_CHECKPOINT"].startswith(f"/cache/{RECORD.slug}/")
    assert any("reused 2 groups" in line for line in regenerate.RUN_NOTES)
    # A rerun under the same key reuses the calibration.
    _run(reference, here, samples, truths)
    assert len(docker.runs) == 3
    assert context.volume_keys


def test_a_declared_worker_floor_binds_above_the_measured_cost(
    resumable: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths, _ = resumable
    reference = dataclasses.replace(
        reference, worker_memory_mb=900.0, memory_override=(1.5, 0.7, "heavy tails")
    )
    docker = FakeDocker([0])
    monkeypatch.setattr(regenerate.subprocess, "run", docker)
    _run(reference, here, samples, truths)
    real = FakeDocker.env(docker.runs[-1])
    assert real["CLEVERLY_R_WORKER_MB"] == "900"
    assert real["CLEVERLY_R_WORKER_BASIS"] == "declared"
    assert real["CLEVERLY_R_MEMORY_FACTOR"] == "1.5"
    assert real["CLEVERLY_R_MEMORY_SHARE"] == "0.7"
    assert real["CLEVERLY_R_MEMORY_REASON"] == "heavy tails"


def test_exit_3_stops_with_the_harness_message(
    resumable: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths, _ = resumable
    monkeypatch.setattr(regenerate.subprocess, "run", FakeDocker([3]))
    with pytest.raises(RuntimeError, match=r"killed twice.*A rerun resumes"):
        _run(reference, here, samples, truths)


def test_one_container_kill_reinvokes_with_half_the_workers(
    resumable: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    reference, here, samples, truths, _ = resumable
    docker = FakeDocker([137, 0])
    monkeypatch.setattr(regenerate.subprocess, "run", docker)
    _run(reference, here, samples, truths)
    first, second = (FakeDocker.env(command) for command in docker.runs[1:])
    assert "CLEVERLY_R_WORKER_CAP" not in first
    assert second["CLEVERLY_R_WORKER_CAP"] == "3"
    assert "container killed: MemAvailable 900 MB, last workers 6" in capsys.readouterr().out


def test_a_second_container_kill_stops(resumable: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    reference, here, samples, truths, _ = resumable
    monkeypatch.setattr(regenerate.subprocess, "run", FakeDocker([137, 137]))
    with pytest.raises(RuntimeError, match="container killed twice, rerun to resume"):
        _run(reference, here, samples, truths)


def test_any_other_exit_is_fatal_and_keeps_the_checkpoints(
    resumable: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths, _ = resumable
    monkeypatch.setattr(regenerate.subprocess, "run", FakeDocker([1]))
    with pytest.raises(RuntimeError, match=r"exited 1.*Checkpoints are kept"):
        _run(reference, here, samples, truths)


def test_two_runners_in_one_scratch_get_distinct_leaves(
    resumable: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths, _ = resumable
    docker = FakeDocker([0, 0])
    monkeypatch.setattr(regenerate.subprocess, "run", docker)
    _run(reference, here, samples, truths)
    _run(reference, here, samples, truths, runner="probe.R", output=samples.parent / "probe.csv")
    leaves = [FakeDocker.env(command) for command in docker.runs if "--init" in command]
    real = [env for env in leaves if "CLEVERLY_HARNESS_CALIBRATE" not in env]
    assert len(real) == 2
    for name in ("CLEVERLY_CHECKPOINT", "CLEVERLY_R_META", "CLEVERLY_R_SEEDS", "CLEVERLY_R_STATE"):
        assert real[0][name] != real[1][name]
    work = samples.parent
    assert (work / "run_fixture__reference-results.meta.json").exists()
    assert (work / "probe__probe.seeds.csv").exists()
    # The probe never reaches study_fitter or study_stream, so it gets no calibration run.
    assert sum("CLEVERLY_HARNESS_CALIBRATE" in env for env in leaves) == 1


def test_a_python_reference_gets_no_calibration_seeds_or_meta(
    resumable: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M13: the zEpid runner keeps the one-container path until RM39b."""
    reference, here, samples, truths, _ = resumable
    reference = dataclasses.replace(reference, interpreter="python")
    commands: list[list[str]] = []
    monkeypatch.setattr(regenerate.subprocess, "run", lambda command, **_: commands.append(command))
    _run(reference, here, samples, truths)
    run = commands[-1]
    assert "--init" not in run and f"{regenerate.VOLUME}:/cache" not in run
    assert not any(part.startswith("CLEVERLY_R_SEEDS") for part in run)
    assert not list(samples.parent.glob("*.meta.json"))
    assert not list(samples.parent.glob("*.seeds.csv"))
    assert not list(samples.parent.glob("*.calibration.json"))


def test_a_direct_call_outside_the_driver_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths = _reference_tree(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(regenerate.subprocess, "run", lambda command, **_: commands.append(command))
    _run(reference, here, samples, truths, env={"CLEVERLY_HARNESS_RNG_PROBE": "1"})
    assert len(commands) == 2
    assert "CLEVERLY_HARNESS_RNG_PROBE=1" in commands[-1]
    assert "--init" not in commands[-1]


def test_the_meta_file_names_the_image_and_leaves_it_out_of_the_key(
    resumable: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, here, samples, truths, _ = resumable
    monkeypatch.setattr(regenerate.subprocess, "run", FakeDocker([0]))
    _run(reference, here, samples, truths)
    meta = json.loads(
        (samples.parent / "run_fixture__reference-results.meta.json").read_text(encoding="utf-8")
    )
    assert meta["image"] == "sha256:image"
    fields = {name: value for name, value in meta.items() if name not in {"image", "leaf"}}
    key12 = regenerate.digest(regenerate.leaf_key(reference, KEY, here, samples, truths))[:12]
    assert regenerate.digest(fields)[:12] == key12


# ---- cleanup --------------------------------------------------------------------------------


@pytest.mark.parametrize("keep", [False, True])
def test_cleanup_deletes_a_finished_run_and_keeps_a_declared_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    keep: bool,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cleared: list[tuple[str, str]] = []
    monkeypatch.setattr(regenerate, "clear_volume", lambda *args: cleared.append(args))
    context = regenerate.RunContext(
        record=RECORD, key=KEY, scratch=scratch, default_scratch=True, keep=keep
    )
    context.volume_keys += [("image", "/cache/slug/abc"), ("image", "/cache/slug/abc")]
    regenerate.cleanup(context)
    out = capsys.readouterr().out
    if keep:
        assert not cleared and scratch.exists()
        assert f"{regenerate.VOLUME}:/cache/slug/abc" in out and str(scratch) in out
    else:
        assert cleared == [("image", "/cache/slug/abc")] and not scratch.exists()


def test_cleanup_never_deletes_a_cache_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regenerate, "clear_volume", lambda *args: None)
    context = regenerate.RunContext(record=RECORD, key=KEY, scratch=tmp_path, default_scratch=False)
    regenerate.cleanup(context)
    assert tmp_path.exists()


def test_clearing_the_volume_deletes_one_key_directory_never_the_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(regenerate.subprocess, "run", lambda command, **_: commands.append(command))
    regenerate.clear_volume("image", "/cache/slug/abc")
    (command,) = commands
    assert command[-2:] == ["-rf", "/cache/slug/abc"]
    assert "volume" not in command


# ---- seeds ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("drawn", "expected"),
    [(0, 0), (2**31 - 1, 2**31 - 1), (2**31, 0), (2**32 - 1, 2**31 - 1)],
)
def test_a_reference_seed_is_a_valid_r_integer(
    monkeypatch: pytest.MonkeyPatch, drawn: int, expected: int
) -> None:
    monkeypatch.setattr(seeds, "stream_seed", lambda *labels: drawn)
    assert seeds.reference_seed(RECORD, "run", "binary", 0) == expected


def test_the_seeds_file_has_a_row_per_scenario_and_replicate(tmp_path: Path) -> None:
    path = tmp_path / "seeds.csv"
    regenerate.write_seeds(path, RECORD, "run_tmle3")
    frame = pd.read_csv(path, keep_default_na=False)
    scenarios = (*RECORD.scenarios, "*")
    assert len(frame) == RECORD.replicates * len(scenarios)
    assert set(frame["scenario"]) == set(scenarios)
    assert frame["seed"].between(0, 2**31 - 1).all()
    assert frame["seed"].nunique() == len(frame)
    other = tmp_path / "other.csv"
    regenerate.write_seeds(other, RECORD, "probe")
    assert not set(pd.read_csv(other)["seed"]) & set(frame["seed"])

"""The declared run form of a registered regeneration, without a study run.

Two studies enter through it: the gated RM30 learned-rule study, through
``tests/canonical/learned_rule_run.py``, and ``canonical-multi-arm-drtmle`` (RM18 Design SL),
whose ``regenerate.py`` calls :func:`tests.canonical.declared_run.run` with its reference.  A
stand-in driver writes placeholder artifacts and a real manifest, so no test here fits a model
or runs a container.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.canonical import declared_run, learned_rule_run, regenerate
from tests.canonical.multi_arm_drtmle import regenerate as multi_arm_regenerate
from tests.canonical.regenerate import ARTIFACT_NAMES, Reference
from tests.diagnostics.rm18_shared import THREAD_VARIABLES
from tests.studies import canonical_multi_arm_drtmle as multi_arm
from tests.studies import learned_rule_cvtmle as gated
from tests.studies import learned_rule_cvtmle_properties as gated_properties
from tests.studies import multi_arm_drtmle_properties as multi_arm_properties
from tests.studies.evidence import manifest
from tests.studies.evidence.registry import ROOT


def _learned_rule(here: Path) -> None:
    learned_rule_run.run(gated, gated_properties, here=here)


def _multi_arm(here: Path) -> None:
    declared_run.run(
        multi_arm, multi_arm_properties, here=here, reference=multi_arm_regenerate.REFERENCE
    )


#: ``(study module, entry point, the reference the driver must receive)``.
STUDIES: dict[str, tuple[Any, Callable[[Path], None], Reference | None]] = {
    "learned-rule-cvtmle": (gated, _learned_rule, None),
    "canonical-multi-arm-drtmle": (multi_arm, _multi_arm, multi_arm_regenerate.REFERENCE),
}


@pytest.fixture(params=sorted(STUDIES))
def study(request: pytest.FixtureRequest) -> tuple[Any, Callable[[Path], None], Any]:
    return STUDIES[request.param]


def _git_status() -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _writing_driver(*, fail: str | None = None, calls: list[Any] | None = None) -> Any:
    """A stand-in for the shared driver that writes every artifact and a real manifest.

    ``fail="before"`` raises before the manifest exists, as a failed fit does.
    ``fail="after"`` raises after it, as a failed gated verdict does.  ``calls`` collects the
    command line and the reference of each call.
    """

    def driver(
        study: Any, properties: Any, *, here: Path, reference: Reference | None = None
    ) -> None:
        if calls is not None:
            calls.append((list(sys.argv), reference))
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if fail == "before":
            raise RuntimeError("a fit failed")
        record = study.STUDY
        paths = [output / name for name in (*ARTIFACT_NAMES, *record.extra_artifacts)]
        for path in paths:
            path.write_text("artifact\n", encoding="utf-8", newline="\n")
        manifest.write_manifest(output / "manifest.json", record, paths)
        if fail == "after":
            raise RuntimeError("independent performance gates failed")

    return driver


def _guard(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    refused: Sequence[str] = (),
    *,
    threads: bool = False,
) -> None:
    """``argv`` as the command line, and a guard that refuses ``refused`` and nothing else.

    ``threads=True`` keeps the real thread refusals in the guard.
    """
    monkeypatch.setattr(sys, "argv", ["regenerate", *argv])
    monkeypatch.setattr(declared_run, "refusals", lambda smoke: list(refused))
    if not threads:
        monkeypatch.setattr(declared_run, "thread_refusals", list)
    monkeypatch.setattr(learned_rule_run, "runtime_refusals", list)


def _no_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(declared_run, "regenerate", pytest.fail)
    monkeypatch.setattr(learned_rule_run, "regenerate", pytest.fail)


class TestTheGuard:
    def test_a_declared_run_refuses_what_rule_r6_refuses(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, enter, _ = study
        _guard(monkeypatch, ["--output", str(tmp_path / "run")], ["the tree has changes"])
        _no_driver(monkeypatch)
        with pytest.raises(SystemExit, match="the tree has changes"):
            enter(tmp_path / "here")

    def test_a_declared_run_refuses_a_thread_variable_other_than_1(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, enter, _ = study
        _guard(monkeypatch, ["--output", str(tmp_path / "run")], threads=True)
        for name in THREAD_VARIABLES:
            monkeypatch.setenv(name, "1")
        monkeypatch.setenv("MKL_NUM_THREADS", "8")
        _no_driver(monkeypatch)
        with pytest.raises(SystemExit, match="MKL_NUM_THREADS is 8, not 1"):
            enter(tmp_path / "here")

    def test_the_thread_refusals_name_each_variable_that_is_not_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in THREAD_VARIABLES:
            monkeypatch.setenv(name, "1")
        assert declared_run.thread_refusals() == []
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.delenv("NUMEXPR_NUM_THREADS")
        assert declared_run.thread_refusals() == [
            "OMP_NUM_THREADS is 4, not 1 (rule R6)",
            "NUMEXPR_NUM_THREADS is unset, not 1 (rule R6)",
        ]

    @pytest.mark.parametrize(
        "extra",
        [
            ["--n", "500"],
            ["--primary-only"],
            ["--skip-properties"],
            ["--skip-reference"],
            ["--allow-failures"],
            ["--cache", "somewhere"],
            ["--refresh-python"],
        ],
    )
    def test_a_declared_run_refuses_every_flag_but_output_and_jobs(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: list[str]
    ) -> None:
        _, enter, _ = study
        _guard(monkeypatch, ["--output", str(tmp_path / "run"), "--jobs", "2", *extra])
        _no_driver(monkeypatch)
        with pytest.raises(SystemExit, match="--output, --jobs and --fresh only") as refusal:
            enter(tmp_path / "here")
        # RM30 cites its own declared rule; the multi-arm study declares no such rule.
        assert ("(rule L11)" in str(refusal.value)) is (enter is _learned_rule)

    def test_a_declared_run_refuses_the_declared_count_named_explicitly(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        module, enter, _ = study
        count = str(module.PRIMARY_REPLICATES)
        _guard(monkeypatch, ["--replicates", count, "--output", str(tmp_path)])
        _no_driver(monkeypatch)
        with pytest.raises(SystemExit, match="passes no --replicates"):
            enter(tmp_path / "here")

    @pytest.mark.parametrize("inside", [False, True])
    def test_a_declared_run_writes_to_scratch_outside_the_repository(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inside: bool
    ) -> None:
        module, enter, _ = study
        argv = ["--output", str(module.STUDY.artifacts)] if inside else []
        _guard(monkeypatch, argv)
        _no_driver(monkeypatch)
        with pytest.raises(SystemExit, match="outside"):
            enter(tmp_path / "here")

    def test_a_declared_run_refuses_a_scratch_output_that_holds_files(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, enter, _ = study
        (tmp_path / "stale.csv").write_text("x\n", encoding="utf-8")
        _guard(monkeypatch, ["--output", str(tmp_path)])
        _no_driver(monkeypatch)
        with pytest.raises(SystemExit, match="not empty"):
            enter(tmp_path / "here")

    def test_a_smoke_run_refuses_an_output_inside_the_repository(
        self, study: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, enter, _ = study
        inside = module.STUDY.artifacts
        monkeypatch.setattr(
            sys, "argv", ["regenerate", "--replicates", "4", "--output", str(inside)]
        )
        _no_driver(monkeypatch)
        with pytest.raises(SystemExit, match="outside the repository"):
            enter(inside)


class TestTheRecord:
    def test_the_manifest_records_the_tree_as_the_run_found_it(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The witness: the manifest's ``git status`` sees nothing the run wrote.

        On a clean pushed commit the manifest therefore records ``cleverly_worktree_clean:
        true``.  A run into the study directory would write the artifacts before the manifest
        reads the status, and the manifest would record ``false``.
        """
        module, enter, reference = study
        before = _git_status()
        seen: list[str] = []
        real = manifest._git

        def watched(*arguments: str) -> str:
            answer = real(*arguments)
            if arguments[:1] == ("status",):
                seen.append(answer)
            return answer

        monkeypatch.setattr(manifest, "_git", watched)
        here = tmp_path / "here"
        output = tmp_path / "run"
        _guard(monkeypatch, ["--output", str(output), "--jobs", "3"])
        calls: list[Any] = []
        monkeypatch.setattr(declared_run, "regenerate", _writing_driver(calls=calls))
        enter(here)

        assert calls == [
            (["regenerate", "--jobs", "3", "--keep-cache", "--output", str(output)], reference)
        ]
        assert seen, "the manifest read no git status"
        assert all(status == before for status in seen)
        recorded = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
        clean = recorded["generated_with"]["subject"]["cleverly_worktree_clean"]
        assert clean is (before == "")
        assert _git_status() == before
        expected = {
            *ARTIFACT_NAMES,
            *module.STUDY.extra_artifacts,
            *declared_run.RECORD_FILES,
        }
        assert {path.name for path in here.iterdir()} == expected
        for name in expected:
            assert (here / name).read_bytes() == (output / name).read_bytes()
        log = (here / "run.log").read_text(encoding="utf-8")
        assert "declared run" in log
        assert "exit code: 0" in log
        # The log records the command line as invoked, not the one rewritten for the driver.
        invoked = shlex.join([sys.executable, "regenerate", "--output", str(output), "--jobs", "3"])
        assert f"command: {invoked}\n" in log

    def test_fresh_reaches_the_driver_and_the_cache_is_kept_for_audit(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A declared run keeps its cache (``--keep-cache``), and ``--fresh`` discards it first."""
        _, enter, _ = study
        output = tmp_path / "run"
        _guard(monkeypatch, ["--output", str(output), "--jobs", "2", "--fresh"])
        calls: list[Any] = []
        monkeypatch.setattr(declared_run, "regenerate", _writing_driver(calls=calls))
        enter(tmp_path / "here")
        (argv, _), *_ = calls
        assert "--keep-cache" in argv and "--fresh" in argv

    def test_the_run_log_records_what_a_resumed_run_reused(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, enter, _ = study
        here = tmp_path / "here"
        _guard(monkeypatch, ["--output", str(tmp_path / "run")])
        driver = _writing_driver()

        def resumed(*args: Any, **kwargs: Any) -> None:
            regenerate.RUN_NOTES.append("resumed from /cache/x/y/leaf, reused 7 groups")
            driver(*args, **kwargs)

        monkeypatch.setattr(declared_run, "regenerate", resumed)
        enter(here)
        log = (here / "run.log").read_text(encoding="utf-8")
        assert "note: resumed from /cache/x/y/leaf, reused 7 groups" in log

    def test_the_study_directory_is_inside_the_repository(self, study: Any) -> None:
        """The control: writing into the study directory first is what ``git status`` saw."""
        module, _, _ = study
        assert module.STUDY.artifacts.resolve().is_relative_to(ROOT.resolve())

    def test_the_multi_arm_record_declares_the_diagnostics_the_copy_carries(self) -> None:
        assert multi_arm.STUDY.extra_artifacts == ("fit-diagnostics.csv",)
        assert multi_arm.STUDY.artifacts.resolve() == multi_arm_regenerate.HERE

    def test_a_failed_gated_verdict_still_publishes_the_run(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, enter, _ = study
        here = tmp_path / "here"
        _guard(monkeypatch, ["--output", str(tmp_path / "run")])
        monkeypatch.setattr(declared_run, "regenerate", _writing_driver(fail="after"))
        with pytest.raises(RuntimeError, match="gates failed"):
            enter(here)
        assert (here / "manifest.json").exists()
        assert "exit code: 1" in (here / "run.log").read_text(encoding="utf-8")

    def test_a_run_that_stops_before_its_manifest_publishes_nothing(
        self, study: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, enter, _ = study
        here = tmp_path / "here"
        _guard(monkeypatch, ["--output", str(tmp_path / "run")])
        monkeypatch.setattr(declared_run, "regenerate", _writing_driver(fail="before"))
        with pytest.raises(RuntimeError, match="a fit failed"):
            enter(here)
        assert not here.exists()

    def test_a_multi_arm_smoke_run_reaches_the_driver_with_its_reference(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output = tmp_path / "smoke"
        monkeypatch.setattr(
            sys,
            "argv",
            ["regenerate", "--replicates", "4", "--skip-properties", "--output", str(output)],
        )
        calls: list[Any] = []
        monkeypatch.setattr(declared_run, "regenerate", _writing_driver(calls=calls))
        _multi_arm(tmp_path / "here")
        assert calls == [
            (
                [
                    "regenerate",
                    "--replicates",
                    "4",
                    "--output",
                    str(output.resolve()),
                    "--skip-properties",
                ],
                multi_arm_regenerate.REFERENCE,
            )
        ]
        assert not (tmp_path / "here").exists()

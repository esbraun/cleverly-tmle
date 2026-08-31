"""Contracts for the shared canonical-study regeneration driver."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tests.canonical import regenerate
from tests.studies import canonical_tmle, fold_evaluated_cvtmle
from tests.studies.evidence.schema import REPLICATE_COLUMNS


def _rows(implementation: str) -> pd.DataFrame:
    row: dict[str, Any] = dict.fromkeys(REPLICATE_COLUMNS, 0.0)
    row.update(
        {
            "implementation": implementation,
            "scenario": "scenario",
            "estimand": "ate",
            "inference_scale": "identity",
        }
    )
    return pd.DataFrame([row])


def _arguments(tmp_path: Any, *, primary_only: bool) -> argparse.Namespace:
    return argparse.Namespace(
        replicates=2,
        n=50,
        skip_properties=False,
        primary_only=primary_only,
        allow_failures=False,
        output=tmp_path,
        jobs=1,
        reference_jobs=1,
        skip_reference=False,
        cache=None,
    )


def _stub_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, record: Any) -> None:
    monkeypatch.setattr(
        regenerate, "_arguments", lambda *args: _arguments(tmp_path, primary_only=True)
    )
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(regenerate, "summarize", lambda rows: pd.DataFrame({"summary": [1.0]}))
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame({"passed": [False]}),
    )
    monkeypatch.setattr(regenerate, "empty_equivalence", lambda: pd.DataFrame({"passed": []}))
    monkeypatch.setattr(
        regenerate,
        "equivalence",
        lambda *args, **kwargs: pd.DataFrame({"passed": [False], "reference_valid": [False]}),
    )
    monkeypatch.setattr(
        regenerate,
        "_property_artifacts",
        lambda *args, **kwargs: pytest.fail("primary-only generated or reused property artefacts"),
    )
    monkeypatch.setattr(
        regenerate,
        "write_manifest",
        lambda *args, **kwargs: pytest.fail("primary-only wrote a publishable manifest"),
    )


@pytest.mark.parametrize("paired", [False, True])
def test_primary_only_stops_before_properties_manifest_and_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, paired: bool
) -> None:
    record = canonical_tmle.STUDY if paired else fold_evaluated_cvtmle.STUDY
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    _stub_primary(monkeypatch, tmp_path, record)
    reference = regenerate.Reference("image", "runner") if paired else None
    if paired:
        monkeypatch.setattr(
            regenerate,
            "_reference_rows",
            lambda *args, **kwargs: _rows(str(record.reference)),
        )

    regenerate.main(study, SimpleNamespace(), here=tmp_path, reference=reference)

    assert {path.name for path in tmp_path.iterdir()} == {
        "replicates.csv.gz",
        "summary.csv",
        "equivalence.csv",
        "performance-tests.csv",
    }


def test_complete_regeneration_gates_a_failed_joint_property_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    record = fold_evaluated_cvtmle.STUDY
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    monkeypatch.setattr(
        regenerate, "_arguments", lambda *args: _arguments(tmp_path, primary_only=False)
    )
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(regenerate, "summarize", lambda rows: pd.DataFrame({"summary": [1.0]}))
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame({"passed": [True]}),
    )
    monkeypatch.setattr(regenerate, "empty_equivalence", lambda: pd.DataFrame({"passed": []}))
    monkeypatch.setattr(regenerate, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate,
        "_property_artifacts",
        lambda *args, **kwargs: pd.DataFrame({"passed": [True], "property_passed": [False]}),
    )

    with pytest.raises(RuntimeError, match="statistical property gates failed"):
        regenerate.main(study, SimpleNamespace(), here=tmp_path)


def test_skip_reference_refuses_to_synthesize_reference_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    record = dataclasses.replace(
        canonical_tmle.STUDY,
        artifacts=tmp_path,
        extra_artifacts=("fit-diagnostics.csv",),
    )
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    arguments = _arguments(tmp_path, primary_only=False)
    arguments.skip_reference = True
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)

    with pytest.raises(RuntimeError, match="cannot rebuild study-specific extra artifacts"):
        regenerate.main(
            study,
            SimpleNamespace(),
            here=tmp_path,
            reference=regenerate.Reference("image", "runner"),
        )


def test_skip_reference_refuses_a_study_whose_hook_reads_the_reference_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The refusal above covers a hook owner only because each one declares an artefact.

    Nothing enforces that coincidence, so the pair is refused on its own terms. The record
    here declares no extra artefact, which is what makes the first refusal miss it. Without
    this guard the hook receives ``None`` for a path it annotates as ``Path``, and the run
    fails inside ``pandas`` on a message about a missing file.
    """
    record = dataclasses.replace(canonical_tmle.STUDY, artifacts=tmp_path)
    assert not record.extra_artifacts, "this control needs a record the first refusal misses"
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
        reference_artifacts=lambda **options: {},
    )
    arguments = _arguments(tmp_path, primary_only=False)
    arguments.skip_reference = True
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)

    with pytest.raises(RuntimeError, match="reference_artifacts hook"):
        regenerate.main(
            study,
            SimpleNamespace(),
            here=tmp_path,
            reference=regenerate.Reference("image", "runner"),
        )


def test_standard_regeneration_runs_the_declared_reference_artifact_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    record = dataclasses.replace(
        canonical_tmle.STUDY,
        artifacts=tmp_path,
        extra_artifacts=("probe.csv.gz",),
    )
    calls: list[dict[str, Any]] = []

    def reference_artifacts(**options: Any) -> dict[str, pd.DataFrame]:
        calls.append(options)
        return {"probe.csv.gz": pd.DataFrame({"reproduced": [True]})}

    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
        reference_artifacts=reference_artifacts,
    )
    _stub_primary(monkeypatch, tmp_path, record)
    sample_path = tmp_path / "samples.csv.gz"
    truth_path = tmp_path / "truth.csv"
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(
            rows=_rows(record.implementation),
            paths={"samples.csv.gz": sample_path, "truth.csv": truth_path},
        ),
    )
    monkeypatch.setattr(
        regenerate, "_reference_rows", lambda *args, **kwargs: _rows(str(record.reference))
    )

    reference = regenerate.Reference("image", "runner")
    regenerate.main(study, SimpleNamespace(), here=tmp_path, reference=reference)

    assert len(calls) == 1
    assert calls[0]["reference"] is reference
    assert calls[0]["samples"] == sample_path
    assert calls[0]["truths_path"] == truth_path
    artifact = tmp_path / "probe.csv.gz"
    assert pd.read_csv(artifact)["reproduced"].tolist() == [True]
    assert artifact.read_bytes()[4:8] == b"\x00\x00\x00\x00"


def test_reporting_policy_publishes_failed_scientific_verdicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    record = dataclasses.replace(
        fold_evaluated_cvtmle.STUDY,
        artifacts=tmp_path,
        publication_policy="reporting",
    )
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    monkeypatch.setattr(
        regenerate, "_arguments", lambda *args: _arguments(tmp_path, primary_only=False)
    )
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(regenerate, "summarize", lambda rows: pd.DataFrame({"summary": [1.0]}))
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame({"passed": [False]}),
    )
    monkeypatch.setattr(regenerate, "empty_equivalence", lambda: pd.DataFrame({"passed": []}))
    monkeypatch.setattr(regenerate, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate,
        "_property_artifacts",
        lambda *args, **kwargs: pd.DataFrame({"passed": [False], "property_passed": [False]}),
    )

    regenerate.main(study, SimpleNamespace(), here=tmp_path)


def _stub_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, record: Any, paired: pd.DataFrame
) -> Any:
    """A complete run of ``record`` whose every gate passes except what ``paired`` says."""
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    monkeypatch.setattr(
        regenerate, "_arguments", lambda *args: _arguments(tmp_path, primary_only=False)
    )
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(
        regenerate, "_reference_rows", lambda *args, **kwargs: _rows(str(record.reference))
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(regenerate, "summarize", lambda rows: pd.DataFrame({"summary": [1.0]}))
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "implementation": [record.implementation, record.reference],
                "passed": [True, True],
            }
        ),
    )
    monkeypatch.setattr(regenerate, "equivalence", lambda *args, **kwargs: paired)
    monkeypatch.setattr(regenerate, "write_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate,
        "_property_artifacts",
        lambda *args, **kwargs: pd.DataFrame({"passed": [True], "property_passed": [True]}),
    )
    return study


def test_an_undeclared_reference_regression_refuses_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The default, and the reason the declaration below has to be written down."""
    record = dataclasses.replace(canonical_tmle.STUDY, artifacts=tmp_path)
    paired = pd.DataFrame({"passed": [True], "reference_valid": [False]})
    study = _stub_complete(monkeypatch, tmp_path, record, paired)

    with pytest.raises(RuntimeError, match="reference validity"):
        regenerate.main(
            study, SimpleNamespace(), here=tmp_path, reference=regenerate.Reference("i", "r")
        )


def test_a_declared_reference_failure_publishes_and_still_gates_the_subject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The exception is about the comparator only.

    Two runs of the same declared study, differing only in the subject's own paired verdict.
    The first publishes with an invalid reference, which is the whole point of the
    declaration.  The second must still refuse, because a comparator that fails its own truth
    gates is not a licence for the subject to fail its similarity and non-inferiority claim.
    """
    record = dataclasses.replace(
        canonical_tmle.STUDY,
        artifacts=tmp_path,
        accepted_reference_failure="the comparator's intervals under-cover on this law",
    )

    published = pd.DataFrame({"passed": [True], "reference_valid": [False]})
    study = _stub_complete(monkeypatch, tmp_path, record, published)
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "implementation": [record.implementation, record.reference],
                "passed": [True, False],
            }
        ),
    )
    regenerate.main(
        study, SimpleNamespace(), here=tmp_path, reference=regenerate.Reference("i", "r")
    )

    refused = pd.DataFrame({"passed": [False], "reference_valid": [False]})
    study = _stub_complete(monkeypatch, tmp_path, record, refused)
    with pytest.raises(RuntimeError, match="paired similarity and non-inferiority"):
        regenerate.main(
            study, SimpleNamespace(), here=tmp_path, reference=regenerate.Reference("i", "r")
        )


def test_a_stale_accepted_reference_failure_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A recorded reason for a failure that no longer happens is a claim about nothing."""
    record = dataclasses.replace(
        canonical_tmle.STUDY,
        artifacts=tmp_path,
        accepted_reference_failure="the comparator's intervals under-cover on this law",
    )
    paired = pd.DataFrame({"passed": [True], "reference_valid": [True]})
    study = _stub_complete(monkeypatch, tmp_path, record, paired)

    with pytest.raises(RuntimeError, match="stale exception"):
        regenerate.main(
            study, SimpleNamespace(), here=tmp_path, reference=regenerate.Reference("i", "r")
        )


def test_cached_reference_phase_is_reused_only_when_compatible(tmp_path: Any) -> None:
    cached = pd.DataFrame(
        {
            "implementation": ["reference", "reference"],
            "replicate": [0, 1],
            "n": [50, 50],
        }
    )
    path = tmp_path / "reference-results.csv"
    cached.to_csv(path, index=False)
    phase = regenerate._Phase(
        rows=_rows("subject"),
        cached=True,
        paths={"reference-results.csv": path},
    )
    reference = SimpleNamespace(
        run=lambda *args, **kwargs: pytest.fail("the compatible cached reference was rerun")
    )
    arguments = argparse.Namespace(replicates=2, n=50, skip_reference=False)

    rows = regenerate._reference_rows(
        SimpleNamespace(STUDY=SimpleNamespace(reference="reference")),
        reference,
        arguments,
        tmp_path,
        phase,
    )

    pd.testing.assert_frame_equal(rows, cached)


def test_incompatible_cached_reference_phase_is_refused(tmp_path: Any) -> None:
    path = tmp_path / "reference-results.csv"
    pd.DataFrame({"implementation": ["reference"], "replicate": [0], "n": [50]}).to_csv(
        path, index=False
    )
    phase = regenerate._Phase(
        rows=_rows("subject"),
        cached=True,
        paths={"reference-results.csv": path},
    )

    with pytest.raises(RuntimeError, match="incompatible replications"):
        regenerate._reference_rows(
            SimpleNamespace(STUDY=SimpleNamespace(reference="reference")),
            regenerate.Reference("image", "runner"),
            argparse.Namespace(replicates=2, n=50, skip_reference=False),
            tmp_path,
            phase,
        )


def test_an_alternate_reference_runner_must_be_mounted_and_hashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        regenerate.subprocess,
        "run",
        lambda command, **options: commands.append(command),
    )
    reference = regenerate.Reference(
        image="image",
        runner="runner.R",
        mount_runner=True,
        extra_files=("probe.R",),
    )
    reference.run(
        tmp_path,
        tmp_path / "samples.csv.gz",
        tmp_path / "truth.csv",
        tmp_path / "probe.csv",
        cores=2,
        runner="probe.R",
    )
    assert "/fixture/probe.R" in commands[-1]
    assert "CLEVERLY_REFERENCE_CORES=2" in commands[-1]
    assert "CLEVERLY_R_CORES=2" in commands[-1]

    with pytest.raises(ValueError, match="not in extra_files"):
        reference.run(
            tmp_path,
            tmp_path / "samples.csv.gz",
            tmp_path / "truth.csv",
            tmp_path / "probe.csv",
            cores=1,
            runner="unregistered.R",
        )
    with pytest.raises(ValueError, match="requires mount_runner"):
        regenerate.Reference("image", "runner.R", extra_files=("probe.R",)).run(
            tmp_path,
            tmp_path / "samples.csv.gz",
            tmp_path / "truth.csv",
            tmp_path / "probe.csv",
            cores=1,
            runner="probe.R",
        )


#: The exact keyword set ``regenerate.main`` passes to a study's ``reference_artifacts`` hook.
DRIVER_KEYWORDS: dict[str, Any] = {
    "reference": None,
    "here": Path("."),
    "samples": Path("samples.csv.gz"),
    "truths_path": Path("truth.csv"),
    "reference_results": Path("reference-results.csv"),
    "output": Path("."),
    "cores": 1,
}


def test_the_reference_artifact_hook_receives_the_path_the_driver_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The hook reads the reference rows, so it must be told where the driver put them.

    The driver renamed that file once already, and the hook that read it by literal name
    raised ``FileNotFoundError`` on every fresh regeneration until somebody ran one.
    """
    record = dataclasses.replace(
        canonical_tmle.STUDY, artifacts=tmp_path, extra_artifacts=("probe.csv.gz",)
    )
    calls: list[dict[str, Any]] = []

    def reference_artifacts(**options: Any) -> dict[str, pd.DataFrame]:
        # ``read_csv`` here is the assertion: the driver's scratch directory is gone by the
        # time the test body runs, so the hook has to prove the path resolved while it did.
        rows = pd.read_csv(options["reference_results"])
        calls.append(options)
        return {"probe.csv.gz": rows}

    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
        reference_artifacts=reference_artifacts,
    )
    _stub_primary(monkeypatch, tmp_path, record)
    written = _rows(str(record.reference))
    written["replicate"] = [0]
    written["n"] = [50]

    def python_phase(study: Any, arguments: Any, scratch: Path) -> regenerate._Phase:
        # The driver declares the scratch file names; the study never spells one out.
        paths = {
            name: scratch / name for name in ("samples.csv.gz", "truth.csv", "python-rows.csv.gz")
        }
        paths["reference-results.csv"] = scratch / "reference-results.csv"
        return regenerate._Phase(rows=_rows(record.implementation), paths=paths)

    monkeypatch.setattr(regenerate, "_python_phase", python_phase)

    class FakeReference:
        def run(self, here: Any, samples: Any, truths: Any, output: Path, **options: Any) -> None:
            written.to_csv(output, index=False)

    reference = FakeReference()
    regenerate.main(study, SimpleNamespace(), here=tmp_path, reference=reference)

    assert len(calls) == 1
    declared = calls[0]["reference_results"]
    assert declared is not None
    assert declared.name == "reference-results.csv"
    probe = pd.read_csv(tmp_path / "probe.csv.gz")
    assert probe["implementation"].tolist() == [str(record.reference)]


def test_every_registered_reference_artifact_hook_accepts_the_driver_call() -> None:
    """Catch the class of defect, not the one instance of it.

    A hook is called by keyword from one place. Binding each registered hook against that
    exact keyword set fails when the driver and a study disagree about the call.
    """
    from tests.studies.evidence.registry import registered

    hooks = {
        record.runner_module: hook
        for record in registered()
        if (hook := getattr(record.runner(), "reference_artifacts", None)) is not None
    }
    assert hooks, "no registered study declares a reference_artifacts hook"
    for module, hook in hooks.items():
        signature = inspect.signature(hook)
        try:
            signature.bind(**DRIVER_KEYWORDS)
        except TypeError as error:  # pragma: no cover - the assertion names the module
            pytest.fail(f"{module}.reference_artifacts rejects the driver call: {error}")

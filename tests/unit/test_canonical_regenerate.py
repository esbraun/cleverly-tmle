"""Contracts for the shared canonical-study regeneration driver."""

from __future__ import annotations

import argparse
import dataclasses
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
        r_jobs=1,
        skip_r=False,
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


def test_skip_r_refuses_to_synthesize_reference_diagnostics(
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
    arguments.skip_r = True
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)

    with pytest.raises(RuntimeError, match="cannot rebuild study-specific extra artifacts"):
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
    path = tmp_path / "r-results.csv"
    cached.to_csv(path, index=False)
    phase = regenerate._Phase(
        rows=_rows("subject"),
        cached=True,
        paths={"r-results.csv": path},
    )
    reference = SimpleNamespace(
        run=lambda *args, **kwargs: pytest.fail("the compatible cached reference was rerun")
    )
    arguments = argparse.Namespace(replicates=2, n=50, skip_r=False)

    rows = regenerate._reference_rows(
        SimpleNamespace(STUDY=SimpleNamespace(reference="reference")),
        reference,
        arguments,
        tmp_path,
        phase,
    )

    pd.testing.assert_frame_equal(rows, cached)


def test_incompatible_cached_reference_phase_is_refused(tmp_path: Any) -> None:
    path = tmp_path / "r-results.csv"
    pd.DataFrame({"implementation": ["reference"], "replicate": [0], "n": [50]}).to_csv(
        path, index=False
    )
    phase = regenerate._Phase(
        rows=_rows("subject"),
        cached=True,
        paths={"r-results.csv": path},
    )

    with pytest.raises(RuntimeError, match="incompatible replications"):
        regenerate._reference_rows(
            SimpleNamespace(STUDY=SimpleNamespace(reference="reference")),
            regenerate.Reference("image", "runner"),
            argparse.Namespace(replicates=2, n=50, skip_r=False),
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

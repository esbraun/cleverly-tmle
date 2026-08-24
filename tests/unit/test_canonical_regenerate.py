"""Contracts for the shared canonical-study regeneration driver."""

from __future__ import annotations

import argparse
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

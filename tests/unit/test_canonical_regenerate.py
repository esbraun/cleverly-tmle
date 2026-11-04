"""Contracts for the shared canonical-study regeneration driver."""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.canonical import regenerate
from tests.studies import canonical_tmle, fold_evaluated_cvtmle
from tests.studies.evidence.manifest import read_rows, write_rows
from tests.studies.evidence.schema import REPLICATE_COLUMNS


@pytest.fixture(autouse=True)
def _scratch_in_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Every driver run here keeps its scratch in a temporary directory, not the user's cache."""
    runs = tmp_path_factory.mktemp("runs")
    monkeypatch.setattr(regenerate, "runs_root", lambda: runs)


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
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
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


@pytest.mark.parametrize(
    ("argv", "skipped"),
    [
        ([], False),
        (["--replicates", str(fold_evaluated_cvtmle.STUDY.replicates)], False),
        (["--replicates", "2"], True),
        (["--replicates", str(fold_evaluated_cvtmle.STUDY.replicates + 1)], True),
        (["--replicates", "2", "--skip-properties"], True),
    ],
)
def test_a_run_at_another_budget_skips_the_property_study(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, argv: list[str], skipped: bool
) -> None:
    """A run whose count differs from the declared count computes no property verdict.

    Each property module fixes its own budget, so a run with ``--replicates 2`` would
    otherwise run the full property study.  RM22's 200-replication smoke run did, and it
    produced the published property verdicts before the study was committed.  A count above
    the declared one is not a publishing run either, so it skips the study too.
    """
    record = fold_evaluated_cvtmle.STUDY
    study = SimpleNamespace(STUDY=record, PRIMARY_REPLICATES=record.replicates, PRIMARY_N=record.n)
    monkeypatch.setattr("sys.argv", ["regenerate", *argv])
    arguments = regenerate._arguments(study, tmp_path, None)  # type: ignore[arg-type]
    assert arguments.skip_properties is skipped


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

    regenerate.main(study, SimpleNamespace(FAILURE_BOUND=0.0), here=tmp_path, reference=reference)

    assert {path.name for path in tmp_path.iterdir()} == {
        "replicates.parquet",
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
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
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
        lambda *args, **kwargs: (pd.DataFrame({"passed": [True], "property_passed": [False]}), ()),
    )

    with pytest.raises(RuntimeError, match="statistical property gates failed"):
        regenerate.main(study, SimpleNamespace(FAILURE_BOUND=0.0), here=tmp_path)


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
            SimpleNamespace(FAILURE_BOUND=0.0),
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
            SimpleNamespace(FAILURE_BOUND=0.0),
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
    regenerate.main(study, SimpleNamespace(FAILURE_BOUND=0.0), here=tmp_path, reference=reference)

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
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
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
        lambda *args, **kwargs: (pd.DataFrame({"passed": [False], "property_passed": [False]}), ()),
    )

    regenerate.main(study, SimpleNamespace(FAILURE_BOUND=0.0), here=tmp_path)


def _toy_properties(bad: set[int], *, bound: float = 0.0, replicates: int = 20) -> SimpleNamespace:
    """A property module of one ``power`` cell; ``bad`` fits are singular."""
    from tests.studies.evidence import property_verdicts, replication
    from tests.studies.evidence.properties import REPLICATE_COLUMNS as PROPERTY_COLUMNS

    record = dataclasses.replace(
        fold_evaluated_cvtmle.STUDY,
        margins=dataclasses.replace(fold_evaluated_cvtmle.STUDY.margins, bootstrap_replicates=50),
    )

    def fit(key: tuple[str, int], drawn: int) -> dict[str, Any]:
        if drawn in bad:
            np.linalg.inv(np.zeros((2, 2)))
        return {
            "property": "power",
            "cell": key[0],
            "role": "positive",
            "replicate": key[1],
            "n": 100,
            "requested_replicates": replicates,
            "failed_replicates": 0,
            "truth": 0.0,
            "estimate": 0.5,
            "std_error": 0.1,
            "covered": 1,
            "rejected": 1,
        }

    def generate_property_rows(*, n_jobs: int) -> pd.DataFrame:
        values = replication.fit_properties(
            [("c", replicate) for replicate in range(replicates)],
            fit,
            cells_of=lambda key: ("power/c",),
            cell_of=lambda key: key[0],
            replicate_of=lambda key: key[1],
            n_jobs=n_jobs,
            draw=lambda key: key[1],
            checkpoint=None,
        )
        return pd.DataFrame(values).loc[:, list(PROPERTY_COLUMNS)]

    def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
        summary, rates = property_verdicts.apply_shared_verdicts(rows, record, rate_labels=())
        return property_verdicts.finish(summary, rates)

    return SimpleNamespace(
        FAILURE_BOUND=bound,
        PROPERTY_WITNESS=replication.Witness("drawn index", float),
        generate_property_rows=generate_property_rows,
        summarize_properties=summarize_properties,
    )


def test_a_cell_left_below_two_survivors_reads_bound_exceeded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L6: a cell that kept one replicate is red by its bound, and the phase completes.

    The summary's spread needs two survivors, so the driver does not summarize that family.
    The mutation control: one replicate more survives, and the cell is summarized.
    """
    from tests.studies.evidence import property_failures

    record = dataclasses.replace(
        fold_evaluated_cvtmle.STUDY, artifacts=tmp_path, publication_policy="reporting"
    )
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    out = tmp_path / "out"
    arguments = _arguments(out, primary_only=False)
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame({"passed": [True]}),
    )
    monkeypatch.setattr(regenerate, "empty_equivalence", lambda: pd.DataFrame({"passed": []}))
    hashed: list[list[str]] = []
    monkeypatch.setattr(
        regenerate,
        "write_manifest",
        lambda path, record, artifacts, **kwargs: hashed.append(
            [Path(artifact).name for artifact in artifacts]
        ),
    )

    regenerate.main(study, _toy_properties(set(range(1, 20))), here=tmp_path)

    summary = pd.read_csv(out / "properties.csv")
    assert list(summary["cell"]) == ["c"]
    assert not summary["passed"].any() and not summary["property_passed"].any()
    assert int(summary["failed_replicates"].iloc[0]) == 19
    accounting = pd.read_csv(out / property_failures.PROPERTY_ACCOUNTING)
    assert list(accounting["reason"]) == ["bound exceeded"]
    assert list(accounting["complete_case"]) == [19.0]

    regenerate.main(study, _toy_properties(set(range(2, 20))), here=tmp_path)
    summary = pd.read_csv(out / "properties.csv")
    assert int(summary["replicates"].iloc[0]) == 2
    assert "coverage" in summary.columns


def test_a_study_with_no_property_cell_passes_the_property_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A gated study that declares no property cell completes and writes the verdict header.

    The adaptive-iptw-pairing declared run stopped here on a summary with no columns.
    """
    from tests.studies import adaptive_iptw_pairing_properties

    record = dataclasses.replace(
        fold_evaluated_cvtmle.STUDY, artifacts=tmp_path, publication_policy="gated"
    )
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    out = tmp_path / "out"
    arguments = _arguments(out, primary_only=False)
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame({"passed": [True]}),
    )
    monkeypatch.setattr(regenerate, "empty_equivalence", lambda: pd.DataFrame({"passed": []}))
    monkeypatch.setattr(regenerate, "write_manifest", lambda *args, **kwargs: None)

    regenerate.main(study, adaptive_iptw_pairing_properties, here=tmp_path)

    summary = pd.read_csv(out / "properties.csv")
    assert summary.empty
    assert {"passed", "property_passed"} <= set(summary.columns)


def test_a_property_failure_is_recorded_accounted_and_hashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RM39b2: one failed property fit costs its cell a replicate and turns the cell red.

    The run completes, writes both property failure files, and the manifest hashes them.  A
    later ``--skip-properties`` run keeps the committed files and still hashes them.
    """
    from tests.studies.evidence import property_failures

    record = dataclasses.replace(
        fold_evaluated_cvtmle.STUDY, artifacts=tmp_path, publication_policy="reporting"
    )
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    out = tmp_path / "out"
    arguments = _arguments(out, primary_only=False)
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame({"passed": [True]}),
    )
    monkeypatch.setattr(regenerate, "empty_equivalence", lambda: pd.DataFrame({"passed": []}))
    hashed: list[list[str]] = []
    monkeypatch.setattr(
        regenerate,
        "write_manifest",
        lambda path, record, artifacts, **kwargs: hashed.append(
            [Path(artifact).name for artifact in artifacts]
        ),
    )

    regenerate.main(study, _toy_properties({7}), here=tmp_path)

    names = (
        property_failures.PROPERTY_FAILURES,
        property_failures.PROPERTY_ACCOUNTING,
        property_failures.PROPERTY_WITNESSES,
    )
    assert all((out / name).exists() for name in names)
    assert set(names) <= set(hashed[-1])
    records = property_failures.read(out / property_failures.PROPERTY_FAILURES)
    assert records is not None and list(records["group_id"]) == ["properties/c/r000000007"]
    rows = read_rows(out / "property-replicates.parquet")
    assert 7 not in set(rows["replicate"]) and set(rows["failed_replicates"]) == {1}
    accounting = pd.read_csv(out / property_failures.PROPERTY_ACCOUNTING)
    assert set(accounting["reason"]) == {"bound exceeded"}
    witnesses = property_failures.read_witnesses(out / property_failures.PROPERTY_WITNESSES)
    assert witnesses is not None and 7 in set(witnesses["replicate"])
    assert set(witnesses["statistic"]) == {"drawn index"}
    assert list(witnesses["value"]) == [float(r) for r in witnesses["replicate"]]
    summary = pd.read_csv(out / "properties.csv")
    assert not summary["passed"].any()

    # The committed files survive a run that skips the property phase.
    committed = tmp_path / "committed"
    committed.mkdir()
    for name in ("property-replicates.parquet", "properties.csv", *names):
        (committed / name).write_bytes((out / name).read_bytes())
    arguments.skip_properties = True
    regenerate.main(study, _toy_properties(set()), here=committed)
    assert all((out / name).read_bytes() == (committed / name).read_bytes() for name in names)
    assert set(names) <= set(hashed[-1])

    # A clean run removes the stale files and hashes neither.
    arguments.skip_properties = False
    regenerate.main(study, _toy_properties(set()), here=tmp_path)
    assert not any((out / name).exists() for name in names)
    assert not set(names) & set(hashed[-1])


def _property_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[SimpleNamespace, argparse.Namespace, list[dict[str, Any]]]:
    """A complete reporting run whose manifest calls are captured, for p29 and p36."""
    record = dataclasses.replace(
        fold_evaluated_cvtmle.STUDY, artifacts=tmp_path, publication_policy="reporting"
    )
    study = SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=record.replicates,
        PRIMARY_N=record.n,
        CONFIGURATION={},
    )
    arguments = _arguments(tmp_path / "out", primary_only=False)
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame({"passed": [True]}),
    )
    monkeypatch.setattr(regenerate, "empty_equivalence", lambda: pd.DataFrame({"passed": []}))
    manifests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        regenerate,
        "write_manifest",
        lambda path, record, artifacts, **kwargs: manifests.append(kwargs["configuration"]),
    )
    return study, arguments, manifests


def test_p29_a_failure_inside_the_bound_is_judged_by_its_legs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P29: at bound 0.01, one failure of 100 is inside the allowance, never ``bound exceeded``.

    The deliberate-mutation control is the same run at bound 0, which reads ``bound exceeded``.
    """
    from tests.studies.evidence import property_failures

    study, arguments, manifests = _property_driver(monkeypatch, tmp_path)
    out = arguments.output
    regenerate.main(study, _toy_properties({7}, bound=0.01, replicates=100), here=tmp_path)
    accounting = pd.read_csv(out / property_failures.PROPERTY_ACCOUNTING)
    assert set(accounting["reason"]) <= {"within bound", "worst case fails"}
    assert set(accounting["bound"]) == {0.01}
    assert set(accounting["allowed"]) == {1}
    assert manifests[-1]["property_failure_bound"] == 0.01

    regenerate.main(study, _toy_properties({7}, bound=0.0, replicates=100), here=tmp_path)
    accounting = pd.read_csv(out / property_failures.PROPERTY_ACCOUNTING)
    assert set(accounting["reason"]) == {"bound exceeded"}
    assert manifests[-1]["property_failure_bound"] == 0.0


def test_p36_skip_properties_carries_the_committed_bound_or_its_absence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """P36: a ``--skip-properties`` run copies the committed key, never the module's bound."""
    study, arguments, manifests = _property_driver(monkeypatch, tmp_path)
    regenerate.main(study, _toy_properties(set()), here=tmp_path)
    assert manifests[-1]["property_failure_bound"] == 0.0
    committed = tmp_path / "committed"
    committed.mkdir()
    for name in ("property-replicates.parquet", "properties.csv"):
        (committed / name).write_bytes((arguments.output / name).read_bytes())
    arguments.skip_properties = True

    regenerate.main(study, _toy_properties(set(), bound=0.01), here=committed)
    assert "property_failure_bound" not in manifests[-1]

    manifest = {"configuration": {"property_failure_bound": 0.0}}
    (committed / "manifest.json").write_text(json.dumps(manifest))
    regenerate.main(study, _toy_properties(set(), bound=0.01), here=committed)
    assert manifests[-1]["property_failure_bound"] == 0.0

    (committed / "manifest.json").write_text(json.dumps({"configuration": {}}))
    regenerate.main(study, _toy_properties(set(), bound=0.01), here=committed)
    assert "property_failure_bound" not in manifests[-1]


def test_a_property_bound_raised_after_recorded_failures_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """H1: a committed property run lost replicates, so its bound may not rise for a re-run."""
    from tests.studies.evidence import property_failures

    study, arguments, manifests = _property_driver(monkeypatch, tmp_path)
    regenerate.main(study, _toy_properties({7}, bound=0.01, replicates=100), here=tmp_path)
    committed = tmp_path / "committed"
    committed.mkdir()
    for name in (
        "property-replicates.parquet",
        "properties.csv",
        property_failures.PROPERTY_FAILURES,
    ):
        (committed / name).write_bytes((arguments.output / name).read_bytes())
    (committed / "manifest.json").write_text(json.dumps({"configuration": manifests[-1]}))
    with pytest.raises(RuntimeError, match="never raised"):
        regenerate.main(study, _toy_properties({7}, bound=0.02, replicates=100), here=committed)
    # The same bound, a lower one, and a run that reuses the committed files are allowed.
    for bound in (0.01, 0.0):
        regenerate.main(study, _toy_properties({7}, bound=bound, replicates=100), here=committed)
    arguments.skip_properties = True
    regenerate.main(study, _toy_properties({7}, bound=0.02, replicates=100), here=committed)
    # Mutation control: without the key the committed run applied bound 0, so 0.01 is a raise.
    (committed / "manifest.json").write_text(json.dumps({"configuration": {}}))
    arguments.skip_properties = False
    with pytest.raises(RuntimeError, match="never raised"):
        regenerate.main(study, _toy_properties({7}, bound=0.01, replicates=100), here=committed)
    assert property_failures.raised_after_failures(tmp_path / "nothing", 0.05) is False


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
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
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
        lambda *args, **kwargs: (pd.DataFrame({"passed": [True], "property_passed": [True]}), ()),
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
            study,
            SimpleNamespace(FAILURE_BOUND=0.0),
            here=tmp_path,
            reference=regenerate.Reference("i", "r"),
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
        study,
        SimpleNamespace(FAILURE_BOUND=0.0),
        here=tmp_path,
        reference=regenerate.Reference("i", "r"),
    )

    refused = pd.DataFrame({"passed": [False], "reference_valid": [False]})
    study = _stub_complete(monkeypatch, tmp_path, record, refused)
    with pytest.raises(RuntimeError, match="paired similarity and non-inferiority"):
        regenerate.main(
            study,
            SimpleNamespace(FAILURE_BOUND=0.0),
            here=tmp_path,
            reference=regenerate.Reference("i", "r"),
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
            study,
            SimpleNamespace(FAILURE_BOUND=0.0),
            here=tmp_path,
            reference=regenerate.Reference("i", "r"),
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
    # One BLAS and OpenMP thread per worker, and a CPU limit equal to the granted cores.
    assert "OPENBLAS_NUM_THREADS=1" in commands[-1]
    assert "OMP_NUM_THREADS=1" in commands[-1]
    assert commands[-1][commands[-1].index("--cpus") + 1] == "2"

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
    regenerate.main(study, SimpleNamespace(FAILURE_BOUND=0.0), here=tmp_path, reference=reference)

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


class _Stop(Exception):
    """Ends a driver run at the point a test inspects."""


def _failure_record(implementation: str, replicates: str) -> dict[str, Any]:
    from tests.studies.evidence import failures

    record: dict[str, Any] = dict.fromkeys(failures.FAILURE_COLUMNS, "")
    record.update(
        implementation=implementation,
        group_id="g000000001",
        scenarios="scenario",
        replicates=replicates,
        attempt=1,
        seed=1,
    )
    return record


def test_skip_reference_reads_the_committed_failures_before_any_removal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """``--skip-reference`` writes into the study directory it reads the failure file from.

    A removal of stale optional files before the reference phase deleted that file first, so
    the run validated the committed rows without their failure records.
    """
    from tests.studies.evidence import failures

    # The committed run applied no bound, so this run declares none either: a raised bound
    # after recorded failures is refused (test_a_raised_bound_after_recorded_failures_...).
    record = dataclasses.replace(canonical_tmle.STUDY, artifacts=tmp_path, failure_bound={})
    reference = str(record.reference)
    committed = _rows(reference).assign(replicate=0, n=50)
    write_rows(committed, tmp_path / "replicates.parquet")
    pd.DataFrame([_failure_record(reference, "1"), _failure_record(reference, "7")]).to_csv(
        tmp_path / failures.FAILURES, index=False
    )
    study = SimpleNamespace(
        STUDY=record, PRIMARY_REPLICATES=record.replicates, PRIMARY_N=record.n, CONFIGURATION={}
    )
    arguments = _arguments(tmp_path, primary_only=False)
    arguments.skip_reference = True
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)
    monkeypatch.setattr(
        regenerate,
        "_python_phase",
        lambda *args: regenerate._Phase(rows=_rows(record.implementation)),
    )
    seen: list[Any] = []

    def validate(rows: pd.DataFrame, **options: Any) -> None:
        seen.append(options["failures"])
        raise _Stop

    monkeypatch.setattr(regenerate, "validate_replicates", validate)
    with pytest.raises(_Stop):
        regenerate.main(
            study,
            SimpleNamespace(FAILURE_BOUND=0.0),
            here=tmp_path,
            reference=regenerate.Reference("i", "r"),
        )

    (recorded,) = seen
    assert recorded is not None
    assert recorded["replicates"].tolist() == ["1"], "only replicates below --replicates count"
    assert (tmp_path / failures.FAILURES).exists()


@pytest.mark.parametrize("recorded", [False, True])
def test_a_run_removes_the_stale_failure_files_only_when_it_writes_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, recorded: bool
) -> None:
    from tests.studies.evidence import failures

    # No declared bound, so the raised-bound check does not read the stale failure file.
    record = dataclasses.replace(fold_evaluated_cvtmle.STUDY, artifacts=tmp_path, failure_bound={})
    for name in regenerate.OPTIONAL_ARTIFACT_NAMES:
        (tmp_path / name).write_text("stale\n", encoding="utf-8")
    study = SimpleNamespace(
        STUDY=record, PRIMARY_REPLICATES=record.replicates, PRIMARY_N=record.n, CONFIGURATION={}
    )
    monkeypatch.setattr(
        regenerate, "_arguments", lambda *args: _arguments(tmp_path, primary_only=False)
    )
    phase = regenerate._Phase(rows=_rows(record.implementation))
    if recorded:
        phase.reference_failures = pd.DataFrame([_failure_record("x", "0")])
    monkeypatch.setattr(regenerate, "_python_phase", lambda *args: phase)
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)

    def write_csv(*args: Any, **kwargs: Any) -> None:
        raise _Stop

    monkeypatch.setattr(regenerate, "write_rows", write_csv)
    monkeypatch.setattr(regenerate, "write_csv", write_csv)
    with pytest.raises(_Stop):
        regenerate.main(study, SimpleNamespace(FAILURE_BOUND=0.0), here=tmp_path)

    for name in (failures.FAILURES, failures.ACCOUNTING):
        assert (tmp_path / name).exists() is recorded


def test_failures_below_keeps_the_records_under_the_count() -> None:
    frame = pd.DataFrame([_failure_record("r", "0"), _failure_record("r", "3")])
    kept = regenerate._failures_below(frame, 3)
    assert kept is not None
    assert kept["replicates"].tolist() == ["0"]
    assert regenerate._failures_below(frame.iloc[1:], 3) is None


def test_failures_below_refuses_a_multi_replicate_record() -> None:
    frame = pd.DataFrame([_failure_record("r", "0;1")])
    with pytest.raises(RuntimeError, match="more than one replicate"):
        regenerate._failures_below(frame, 3)


def test_reference_failures_labels_the_harness_records(tmp_path: Any) -> None:
    """The harness writes no implementation column; the driver adds the reference's name."""
    from tests.studies.evidence import failures

    results = tmp_path / "reference-results.csv"
    study = SimpleNamespace(STUDY=canonical_tmle.STUDY)
    assert regenerate._reference_failures(study, results) is None  # type: ignore[arg-type]

    harness = [c for c in failures.FAILURE_COLUMNS if c != "implementation"]
    path = regenerate.failures_path(results)
    pd.DataFrame(columns=harness).to_csv(path, index=False)
    assert regenerate._reference_failures(study, results) is None  # type: ignore[arg-type]

    rows = [_failure_record("", "4"), {**_failure_record("", "0"), "group_id": "g000000000"}]
    pd.DataFrame(rows).loc[:, harness].to_csv(path, index=False)
    frame = regenerate._reference_failures(study, results)  # type: ignore[arg-type]
    assert frame is not None
    assert tuple(frame.columns) == failures.FAILURE_COLUMNS
    assert set(frame["implementation"]) == {str(canonical_tmle.STUDY.reference)}
    assert frame["group_id"].tolist() == ["g000000000", "g000000001"]
    assert frame["replicates"].tolist() == ["0", "4"]


def test_a_raised_bound_after_recorded_failures_is_refused_before_any_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """RM39b2: a re-run meets the same failures, so a raised bound would buy the pass."""
    from tests.studies.evidence import failures

    record = dataclasses.replace(canonical_tmle.STUDY, artifacts=tmp_path)
    pd.DataFrame([_failure_record(str(record.reference), "1")]).to_csv(
        tmp_path / failures.FAILURES, index=False
    )
    study = SimpleNamespace(
        STUDY=record, PRIMARY_REPLICATES=record.replicates, PRIMARY_N=record.n, CONFIGURATION={}
    )
    monkeypatch.setattr(
        regenerate, "_arguments", lambda *args: _arguments(tmp_path, primary_only=False)
    )
    fitted: list[bool] = []
    monkeypatch.setattr(regenerate, "_python_phase", lambda *args: fitted.append(True))
    with pytest.raises(RuntimeError, match="never raised"):
        regenerate.main(
            study,
            SimpleNamespace(FAILURE_BOUND=0.0),
            here=tmp_path,
            reference=regenerate.Reference("i", "r"),
        )
    assert fitted == []

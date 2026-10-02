"""Nonzero mutation witnesses for sparse fits, raw carry-forward and attribution."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cleverly.validation import ReplicationFailure, ReplicationRecord
from tests.canonical import repair_property_seeds as runner
from tests.studies.evidence import seed_repair as repair
from tests.studies.evidence.properties import REPLICATE_COLUMNS


def row(index: int = 0) -> dict[str, object]:
    return dict(
        zip(
            REPLICATE_COLUMNS,
            ("p", "c", "positive", index, 10, 2, 0, 0.125, 0.2, 0.1, 1, 0),
            strict=True,
        )
    )


def table(*rows: dict[str, object]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REPLICATE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


POSITION = {
    "batch": "properties",
    "property": "p",
    "cell": "c",
    "replicate": 1,
    "root": 10,
    "old_seed": 42,
    "new_seed": 43,
}


def fixture() -> tuple[dict[str, object], bytes]:
    runtime = dict(
        zip(repair.RUNTIME_FIELDS, ("3.11.13", "2.4.6", "1.17.1", "3.0.5", "1.9.0"), strict=True)
    )
    baseline = {
        "schema_version": 2,
        "slug": "test",
        "study": "historical",
        "configuration": {"margins": {"margin": 0.25}},
        "generated_with": {
            "subject": {**runtime, "cleverly_commit": "a" * 40, "cleverly_worktree_clean": False},
            "reference": {"version": "old"},
        },
        "study_module_sha256": {"historical.py": "1" * 64},
        "reference_sha256": {"reference.R": "2" * 64},
        "sha256": {
            "replicates.csv.gz": "3" * 64,
            "properties.csv": "4" * 64,
            "property-replicates.csv.gz": "5" * 64,
        },
    }
    raw = repair.json_bytes(baseline)
    plan = {
        "baseline_commit": repair.BASELINE_COMMIT,
        "source_sha256": {"runner.py": "6" * 64},
        "changes": {"test": [POSITION]},
        "studies": {
            "test": {
                "manifest_sha256": repair.digest(raw),
                "runtime": runtime,
                "artifact_sha256": baseline["sha256"],
            }
        },
    }
    plan_raw = repair.json_bytes(plan)
    operation = {
        "subject": {**runtime, "cleverly_commit": "b" * 40, "cleverly_worktree_clean": True},
        "source_sha256": plan["source_sha256"],
        "distribution_versions": [["numpy", "2.4.6"]],
    }
    operation["distribution_versions_sha256"] = repair.digest(
        repair.json_bytes(operation["distribution_versions"])
    )
    operation["producer_sha256"] = repair.digest(repair.json_bytes(operation))
    fields = repair.replay_check(row(1), row(1))
    replay = [
        {"position": POSITION, "fields": fields, "structural_fields_and_decisions_exact": True}
    ]
    composite = repair.composite_manifest(
        raw,
        plan_raw=plan_raw,
        slug="test",
        operation=operation,
        artifact_hashes={
            **baseline["sha256"],
            "properties.csv": "7" * 64,
            "property-replicates.csv.gz": "8" * 64,
        },
        property_raw_sha256="9" * 64,
        carried_property_rows_sha256="a" * 64,
        replay=replay,
    )
    composite["composite"]["replacement"]["maximum_replay_deltas"] = dict.fromkeys(
        repair.FIT_FIELDS, 0.0
    )
    return json.loads(repair.json_bytes(composite)), plan_raw


def test_raw_splice_preserves_untouched_decimal_and_order() -> None:
    baseline = table(row(), row(1)).replace(b"0.2", b"0.2000000000000000001", 1)
    replacement = {**row(1), "estimate": 0.3, "covered": 0}
    result = repair.splice_rows(baseline, [replacement], [POSITION])
    repair.verify_splice(baseline, result, [POSITION])
    assert result.splitlines()[1] == baseline.splitlines()[1]
    assert b"0.2000000000000000001" in result


@pytest.mark.parametrize(
    "mutation", ("extra", "missing", "duplicate", "metadata", "nonfinite", "failed")
)
def test_invalid_replacement_payloads_are_refused(mutation: str) -> None:
    replacements = [{**row(1), "estimate": 0.3}]
    if mutation == "extra":
        replacements.append(row())
    elif mutation == "missing":
        replacements = []
    elif mutation == "duplicate":
        replacements *= 2
    elif mutation == "metadata":
        replacements[0]["n"] = 11
    elif mutation == "nonfinite":
        replacements[0]["std_error"] = float("nan")
    else:
        replacements[0]["failed_replicates"] = 1
    with pytest.raises(ValueError):
        repair.splice_rows(table(row(), row(1)), replacements, [POSITION])


def test_altered_carried_row_is_refused() -> None:
    baseline = table(row(), row(1))
    composite = table({**row(), "estimate": 0.25}, {**row(1), "estimate": 0.3})
    with pytest.raises(ValueError, match="untouched"):
        repair.verify_splice(baseline, composite, [POSITION])


def test_unchanged_planned_row_needs_review() -> None:
    baseline = table(row(), row(1))
    with pytest.raises(ValueError, match="needs review"):
        repair.verify_splice(baseline, baseline, [POSITION])


def test_duplicate_baseline_key_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        repair.parse_lines(table(row(), row()))


@pytest.mark.parametrize("field", repair.FIT_FIELDS)
def test_replay_rule_rejects_material_numeric_drift(field: str) -> None:
    changed = {**row(), field: float(row()[field]) + 2e-9}
    with pytest.raises(ValueError, match="exceeds"):
        repair.replay_check(row(), changed)


@pytest.mark.parametrize("field", ("covered", "rejected", "role", "n", "requested_replicates"))
def test_replay_decisions_and_metadata_have_no_numeric_slack(field: str) -> None:
    changed = {**row(), field: "control" if field == "role" else float(row()[field]) + 1}
    with pytest.raises(ValueError, match="exact"):
        repair.replay_check(row(), changed)


def test_replay_rule_records_roundoff_without_claiming_exact_match() -> None:
    changed = {**row(), "estimate": 0.2 + 1e-12}
    fields = repair.replay_check(row(), changed)
    assert fields["estimate"]["exact"] is False
    assert fields["estimate"]["absolute_difference"] > 0
    assert fields["truth"]["exact"] is True


def test_replay_nonfinite_is_not_accepted_as_roundoff() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        repair.replay_check(row(), {**row(), "estimate": float("nan")})


@pytest.mark.parametrize("field", ("replicate", "covered", "rejected", "n", "requested_replicates"))
def test_fractional_structural_replay_values_are_refused(field: str) -> None:
    with pytest.raises(ValueError, match="integral"):
        repair.replay_check(row(), {**row(), field: float(row()[field]) + 0.5})


def test_reporting_red_evidence_is_retained_and_gated_red_is_not_accepted() -> None:
    assert runner.publication_gate("reporting", False)
    assert not runner.publication_gate("gated", False)
    assert runner.publication_gate("gated", True)


@pytest.mark.parametrize("mutation", ("digest", "producer", "omission"))
def test_authoritative_git_source_check_rejects_false_attribution(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    blobs = {"runner.py": b"exact runner\n", "library.py": b"exact library\n"}
    expected = {name: repair.digest(raw) for name, raw in blobs.items()}
    metadata = {"subject": {"cleverly_commit": "b" * 40}, "source_sha256": dict(expected)}
    monkeypatch.setattr(
        runner,
        "git_blob",
        lambda commit, path: blobs[path] if commit == "b" * 40 else b"other producer\n",
    )
    runner.verify_producer_sources(metadata, expected)
    if mutation == "digest":
        # Even coordinated metadata/plan strings cannot substitute for exact Git bytes.
        metadata["source_sha256"]["runner.py"] = expected["runner.py"] = "0" * 64
    elif mutation == "producer":
        metadata["subject"]["cleverly_commit"] = "c" * 40
    else:
        del metadata["source_sha256"]["library.py"]
    with pytest.raises(ValueError):
        runner.verify_producer_sources(metadata, expected)


def test_missing_completed_run_log_is_not_accepted(tmp_path: object) -> None:
    manifest, raw = fixture()
    with pytest.raises(FileNotFoundError):
        runner.verify_run_log(tmp_path, raw, manifest)  # type: ignore[arg-type]


def test_composite_preserves_historical_dirty_provenance() -> None:
    manifest, raw = fixture()
    repair.validate_composite_manifest(manifest, raw)
    assert manifest["generated_with"]["subject"]["cleverly_worktree_clean"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "primary",
        "baseline",
        "historical",
        "missing_source",
        "wrong_source",
        "wrong_producer",
        "runtime",
        "missing_position",
        "wrong_seed",
        "replay",
        "max_delta",
        "margins",
        "dirty",
        "inventory",
    ),
)
def test_composite_attribution_mutations_are_refused(mutation: str) -> None:
    manifest, raw = fixture()
    metadata = manifest["composite"]["replacement"]
    if mutation == "primary":
        manifest["sha256"]["replicates.csv.gz"] = "0" * 64
    elif mutation == "baseline":
        manifest["composite"]["baseline"]["manifest_sha256"] = "0" * 64
    elif mutation == "historical":
        manifest["generated_with"]["subject"]["cleverly_worktree_clean"] = True
    elif mutation == "missing_source":
        metadata["source_sha256"] = {}
    elif mutation == "wrong_source":
        metadata["source_sha256"]["runner.py"] = "0" * 64
    elif mutation == "wrong_producer":
        metadata["subject"]["cleverly_commit"] = "c" * 40
    elif mutation == "runtime":
        metadata["subject"]["scipy"] = "1.18.0"
    elif mutation == "missing_position":
        metadata["positions"] = []
    elif mutation == "wrong_seed":
        metadata["positions"][0]["new_seed"] += 1
    elif mutation == "replay":
        metadata["replay"][0]["fields"]["estimate"]["replayed"] += 1e-4
    elif mutation == "max_delta":
        metadata["maximum_replay_deltas"]["estimate"] = 1e-5
    elif mutation == "margins":
        manifest["configuration"]["margins"]["margin"] = 0.5
    elif mutation == "dirty":
        metadata["subject"]["cleverly_worktree_clean"] = False
    else:
        metadata["distribution_versions"] = []
    with pytest.raises((ValueError, KeyError)):
        repair.validate_composite_manifest(manifest, raw)


def test_partial_executor_preserves_original_id_and_one_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    class FakeStudy:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["n_replicates"] == 2000

        def run_replication(self, index: int, seed: int) -> tuple[ReplicationRecord, ...]:
            seen.append((index, seed))
            return (
                ReplicationRecord(
                    replicate=index,
                    seed=seed,
                    estimand="ate",
                    truth=0.125,
                    estimate=0.2,
                    std_error=0.1,
                    covered=True,
                    rejected=False,
                    inference_estimate=0.2,
                    alpha=0.05,
                ),
            )

    monkeypatch.setattr(runner, "CoverageStudy", FakeStudy)
    cell = SimpleNamespace(
        dgp=None,
        n=10,
        replicates=2000,
        estimand="ate",
        fit_kwargs={},
        seed=10,
        property="p",
        cell="c",
        role="positive",
    )
    batch = SimpleNamespace(estimator=lambda cell: lambda: None)
    position = {**POSITION, "replicate": 1552}
    result = runner.fit_position(batch, cell, position, old=False)
    assert seen == [(1552, 43)]
    assert result["replicate"] == 1552
    assert result["requested_replicates"] == 2000


def test_failed_sparse_fit_is_not_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStudy:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run_replication(self, index: int, seed: int) -> ReplicationFailure:
            return ReplicationFailure(
                replicate=index, seed=seed, error_type="FitError", message="nonzero witness"
            )

    monkeypatch.setattr(runner, "CoverageStudy", FakeStudy)
    cell = SimpleNamespace(
        dgp=None,
        n=10,
        replicates=2000,
        estimand="ate",
        fit_kwargs={},
        seed=10,
        property="p",
        cell="c",
        role="positive",
    )
    with pytest.raises(RuntimeError, match="FitError"):
        runner.fit_position(
            SimpleNamespace(estimator=lambda cell: lambda: None), cell, POSITION, old=False
        )


def test_every_declared_batch_factory_builds_fresh_learners() -> None:
    batches = repair.all_batches()
    assert len(batches) == 11 and sum(map(len, batches.values())) == 12
    for items in batches.values():
        for batch in items:
            for cell in batch.cells:
                factory = batch.estimator(cell)
                first, second = factory(), factory()
                assert first is not second
                assert first.outcome_learner is not second.outcome_learner
                assert first.treatment_learner is not second.treatment_learner


def test_plan_replacement_mutations_fail_before_any_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    batches = {"test": ()}
    # Invalid plans are rejected before seed discovery, factory creation, or sampling.
    monkeypatch.setattr(
        repair, "changed_seeds", lambda _: pytest.fail("invalid plan reached outcomes")
    )
    with pytest.raises(ValueError, match="schema/baseline"):
        repair.validate_plan({"schema_version": 1, "baseline_commit": "wrong"}, batches)


def test_preflight_binds_the_valid_plan_to_the_producer_git_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest, raw = fixture()
    plan = json.loads(raw)
    policy = tmp_path / "src/cleverly/validation/_seeds.py"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"frozen allocation policy")
    plan["source_sha256"] = {}
    plan["allocation_sha256"] = repair.digest(policy.read_bytes())
    plan["studies"]["test"]["margins"] = {"margin": 0.25}
    raw = repair.json_bytes(plan)
    committed = [raw]
    subject = manifest["composite"]["replacement"]["subject"]
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(runner, "refusals", lambda _: [])
    monkeypatch.setattr(runner, "thread_refusals", lambda: [])
    monkeypatch.setattr(runner, "validate_plan", lambda *_: None)
    monkeypatch.setattr(runner, "all_batches", lambda: {})
    monkeypatch.setattr(runner, "provenance", lambda: subject)
    monkeypatch.setattr(runner, "git_value", lambda *_: subject["cleverly_commit"])
    monkeypatch.setattr(runner, "git_blob", lambda *_: committed[0])
    monkeypatch.setattr(
        runner,
        "record_for",
        lambda _: SimpleNamespace(margins=SimpleNamespace(as_json=lambda: {"margin": 0.25})),
    )
    monkeypatch.setattr(runner.importlib.metadata, "distributions", lambda: [])
    assert runner.preflight(plan, "test", raw)["subject"] == subject
    committed[0] = raw + b"\n"
    with pytest.raises(ValueError, match="exact committed plan blob"):
        runner.preflight(plan, "test", raw)


def test_wrong_import_origin_is_refused_before_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cleverly
    from tests.diagnostics import rm18_shared

    for variable in runner.THREAD_VARIABLES:
        monkeypatch.setenv(variable, "1")
    monkeypatch.setattr(cleverly, "__file__", str(tmp_path / "other/cleverly/__init__.py"))
    monkeypatch.setattr(rm18_shared, "_git", lambda *args: "" if args[0] == "status" else "a" * 40)
    monkeypatch.setattr(runner, "validate_plan", lambda *_: pytest.fail("guard reached plan"))
    with pytest.raises(ValueError, match="cleverly is imported from"):
        runner.preflight({}, "test", b"unused")


@pytest.mark.parametrize("variable", runner.THREAD_VARIABLES)
def test_each_native_thread_guard_precedes_sampling(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.setattr(runner, "refusals", lambda _: [])
    for name in runner.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv(variable, "2")
    monkeypatch.setattr(runner, "validate_plan", lambda *_: pytest.fail("guard reached plan"))
    with pytest.raises(ValueError, match=variable):
        runner.preflight({}, "test", b"unused")


@pytest.mark.parametrize("mutation", ["hash", "count", "exit", "thread", "producer", "label"])
def test_completed_log_valid_control_and_nonzero_mutations(tmp_path: Path, mutation: str) -> None:
    manifest, raw = fixture()
    manifest["composite"]["analysis"].update(verdicts_passed=False, publication_gate_passed=True)
    manifest_raw = repair.json_bytes(manifest)
    (tmp_path / "manifest.json").write_bytes(manifest_raw)
    log = {
        "source_commit": manifest["composite"]["replacement"]["subject"]["cleverly_commit"],
        "plan_sha256": repair.digest(raw),
        "artifact_sha256": {**manifest["sha256"], "manifest.json": repair.digest(manifest_raw)},
        "replacement_positions": 1,
        "old_seed_replays": 1,
        "scientific_verdicts_passed": False,
        "publication_gate_passed": True,
        "exit_code": 0,
        "threads": dict.fromkeys(runner.THREAD_VARIABLES, "1"),
        "started": "predeclared-start",
        "finished": "completed-finish",
    }
    (tmp_path / "run.log").write_bytes(repair.json_bytes(log))
    runner.verify_run_log(tmp_path, raw, manifest)
    if mutation == "hash":
        log["artifact_sha256"]["manifest.json"] = "f" * 64
    elif mutation == "count":
        log["old_seed_replays"] = 0
    elif mutation == "exit":
        log["exit_code"] = 1
    elif mutation == "thread":
        log["threads"]["VECLIB_MAXIMUM_THREADS"] = "2"
    elif mutation == "producer":
        log["source_commit"] = "f" * 40
    else:
        log["scientific_verdicts_passed"] = True
    (tmp_path / "run.log").write_bytes(repair.json_bytes(log))
    with pytest.raises(ValueError, match="completed run log"):
        runner.verify_run_log(tmp_path, raw, manifest)

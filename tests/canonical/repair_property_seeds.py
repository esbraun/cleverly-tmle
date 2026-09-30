"""Execute or verify a committed seed-only composite property regeneration.

The primary/reference results are byte copies. Only predeclared changed sample seeds
are fitted, and all property analyses read the complete carried-plus-replaced table.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from cleverly.validation import CoverageStudy, ReplicationFailure
from tests.canonical.declared_run import outside, thread_refusals
from tests.diagnostics.rm18_shared import THREAD_VARIABLES, refusals
from tests.studies.evidence.manifest import provenance, write_csv
from tests.studies.evidence.registry import ROOT, StudyRecord, registered
from tests.studies.evidence.seed_repair import (
    ALLOCATION_COMMIT,
    BASELINE_COMMIT,
    EXPECTED_COUNTS,
    FIT_FIELDS,
    PLAN_PATH,
    REPLAY_TOLERANCE,
    RUNTIME_FIELDS,
    all_batches,
    batch_declaration,
    carried_rows_sha256,
    changed_seeds,
    composite_manifest,
    digest,
    json_bytes,
    parse_lines,
    replay_check,
    row_key,
    source_files,
    splice_rows,
    validate_complete_rows,
    validate_composite_manifest,
    validate_plan,
    verify_splice,
)


def git_blob(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"])


def git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()


def record_for(slug: str) -> StudyRecord:
    return next(record for record in registered() if record.slug == slug)


def build_plan(audit_path: Path) -> dict[str, Any]:
    """Pure declarations and Git metadata, before any fitted outcome is examined."""
    batches = all_batches()
    audit_raw = audit_path.read_bytes()
    if digest(audit_raw) != "5fa4be8cc91781f2125f144b89b9557ad66e3b585ad590e036498f6666ea1e75":
        raise ValueError("not the original complete actual-call audit")
    audit = json.loads(audit_raw)
    original = {}
    for item in audit["batches"]:
        original.setdefault(item["study"], []).append(item["cells"])
    current = {
        slug: [
            [
                [cell.property, cell.cell, cell.n, cell.replicates, cell.seed, cell.role]
                for cell in batch.cells
            ]
            for batch in items
        ]
        for slug, items in batches.items()
    }
    if original != current:
        raise ValueError("current actual calls differ from the original frozen audit")
    selected = [record_for(slug) for slug in EXPECTED_COUNTS]
    studies = {}
    for record in selected:
        relative = record.artifacts.relative_to(ROOT).as_posix()
        raw = git_blob(BASELINE_COMMIT, f"{relative}/manifest.json")
        baseline = json.loads(raw)
        studies[record.slug] = {
            "artifact_directory": relative,
            "manifest_sha256": digest(raw),
            "artifact_sha256": baseline["sha256"],
            "baseline_generated_with": baseline["generated_with"],
            "runtime": {
                field: baseline["generated_with"]["subject"][field] for field in RUNTIME_FIELDS
            },
            "margins": baseline["configuration"]["margins"],
        }
        if baseline["configuration"]["margins"] != record.margins.as_json():
            raise ValueError(f"{record.slug}: declared verdict margins changed")
    plan = {
        "schema_version": 1,
        "baseline_commit": BASELINE_COMMIT,
        "allocation_commit": ALLOCATION_COMMIT,
        "allocation_sha256": digest(
            git_blob(ALLOCATION_COMMIT, "src/cleverly/validation/_seeds.py")
        ),
        "replay_tolerance": REPLAY_TOLERANCE,
        "replay_rule": "abs(replay-old) <= 1e-9 * max(1, abs(old)); all structural fields/decisions exact",
        "original_audit_sha256": digest(audit_raw),
        "original_actual_calls": [
            {key: item[key] for key in ("study", "run_cells_route", "cells")}
            for item in audit["batches"]
        ],
        "batches": {
            slug: [batch_declaration(batch) for batch in items] for slug, items in batches.items()
        },
        "changes": {slug: changed_seeds(batches[slug]) for slug in EXPECTED_COUNTS},
        "studies": studies,
        "source_sha256": {
            name: digest((ROOT / name).read_bytes()) for name in source_files(selected)
        },
    }
    plan = json.loads(json_bytes(plan))
    validate_plan(plan, batches)
    return plan


def preflight(plan: dict[str, Any], slug: str, plan_raw: bytes) -> dict[str, Any]:
    refused = refusals(False) + thread_refusals()
    if refused:
        raise ValueError("declared execution refused: " + "; ".join(refused))
    validate_plan(plan, all_batches())
    subject = provenance()
    commit = subject["cleverly_commit"]
    if plan_raw != git_blob(commit, PLAN_PATH.relative_to(ROOT).as_posix()):
        raise ValueError("scientific operation plan is not the exact committed plan blob")
    if subject["cleverly_worktree_clean"] is not True or commit != git_value(
        "rev-parse", "@{upstream}"
    ):
        raise ValueError("scientific execution requires clean committed/pushed source")
    expected = plan["studies"][slug]
    if any(subject[field] != expected["runtime"][field] for field in RUNTIME_FIELDS):
        raise ValueError(
            f"{slug}: interpreter/dependencies differ from historical recorded runtime"
        )
    if record_for(slug).margins.as_json() != expected["margins"]:
        raise ValueError("verdict margins changed from the predeclared baseline")
    for name, expected_hash in plan["source_sha256"].items():
        if (
            digest((ROOT / name).read_bytes()) != expected_hash
            or digest(git_blob(commit, name)) != expected_hash
        ):
            raise ValueError(f"executed source differs from precommitted plan: {name}")
    if (
        digest((ROOT / "src/cleverly/validation/_seeds.py").read_bytes())
        != plan["allocation_sha256"]
    ):
        raise ValueError("allocation policy differs from frozen bb4772a9")
    distributions = sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
    )
    operation = {
        "subject": subject,
        "source_sha256": plan["source_sha256"],
        "distribution_versions": distributions,
        "distribution_versions_sha256": digest(json_bytes(distributions)),
    }
    operation["producer_sha256"] = digest(json_bytes(operation))
    return operation


def fit_position(batch: Any, cell: Any, position: dict[str, Any], *, old: bool) -> dict[str, Any]:
    """The same replication executor as a full study, with the declared original index."""
    seed = position["old_seed" if old else "new_seed"]
    study = CoverageStudy(
        dgp=cell.dgp,
        estimator=batch.estimator(cell),
        n=cell.n,
        n_replicates=cell.replicates,
        estimands=(cell.estimand,),
        fit_kwargs=dict(cell.fit_kwargs),
        seed=cell.seed,
        n_jobs=1,
    )
    outcome = study.run_replication(position["replicate"], seed)
    if isinstance(outcome, ReplicationFailure):
        raise RuntimeError(f"{position}: {outcome.error_type}: {outcome.message}")
    if len(outcome) != 1 or outcome[0].estimand != cell.estimand or outcome[0].seed != seed:
        raise ValueError("replication executor returned missing/extra/wrong-seed records")
    row = outcome[0]
    return {
        "property": cell.property,
        "cell": cell.cell,
        "role": cell.role,
        "replicate": row.replicate,
        "n": cell.n,
        "requested_replicates": cell.replicates,
        "failed_replicates": 0,
        "truth": row.truth,
        "estimate": row.estimate,
        "std_error": row.std_error,
        "covered": int(row.covered),
        "rejected": int(row.rejected),
    }


def replacement_job(
    batch: Any, cell: Any, position: dict[str, Any], baseline_row: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Workers receive only one baseline row, never the entire composite table."""
    replayed = fit_position(batch, cell, position, old=True)
    fields = replay_check(baseline_row, replayed)
    replaced = fit_position(batch, cell, position, old=False)
    return replaced, {
        "position": position,
        "fields": fields,
        "structural_fields_and_decisions_exact": True,
    }


def execute(plan_raw: bytes, slug: str, output: Path, *, jobs: int) -> bool:
    plan = json.loads(plan_raw)
    outside(output, "targeted seed repair")
    operation = preflight(plan, slug, plan_raw)
    record = record_for(slug)
    relative = plan["studies"][slug]["artifact_directory"]
    baseline_raw = git_blob(BASELINE_COMMIT, f"{relative}/manifest.json")
    if digest(baseline_raw) != plan["studies"][slug]["manifest_sha256"]:
        raise ValueError("baseline manifest changed")
    baseline = json.loads(baseline_raw)
    artifacts = {
        name: git_blob(BASELINE_COMMIT, f"{relative}/{name}") for name in baseline["sha256"]
    }
    if any(digest(raw) != baseline["sha256"][name] for name, raw in artifacts.items()):
        raise ValueError("baseline artifact bytes do not match their recorded hashes")
    property_raw = gzip.decompress(artifacts["property-replicates.csv.gz"])
    _, old_rows = parse_lines(property_raw)
    validate_complete_rows(old_rows)
    batches = {batch.name: batch for batch in all_batches()[slug]}
    cells = {
        (batch.name, cell.property, cell.cell): cell
        for batch in batches.values()
        for cell in batch.cells
    }
    positions = plan["changes"][slug]

    payloads = [
        (
            batches[position["batch"]],
            cells[(position["batch"], position["property"], position["cell"])],
            position,
            old_rows[row_key(position)][1],
        )
        for position in positions
    ]
    outcomes = map_parallel(replacement_job, payloads, n_jobs=jobs)
    composite_raw = splice_rows(property_raw, [row for row, _ in outcomes], positions)
    verify_splice(property_raw, composite_raw, positions)
    rows = pd.read_csv(io.BytesIO(composite_raw), float_precision="round_trip")
    # Complete table, original bootstrap streams, full declared budgets and verdict rules.
    summary = record.properties().summarize_properties(rows)
    artifacts["property-replicates.csv.gz"] = gzip.compress(composite_raw, mtime=0)
    for name, raw in artifacts.items():
        if name != "properties.csv":
            (output / name).write_bytes(raw)
    write_csv(summary, output / "properties.csv")
    artifacts["properties.csv"] = (output / "properties.csv").read_bytes()
    manifest = composite_manifest(
        baseline_raw,
        plan_raw=plan_raw,
        slug=slug,
        operation=operation,
        artifact_hashes={name: digest(raw) for name, raw in artifacts.items()},
        property_raw_sha256=digest(composite_raw),
        carried_property_rows_sha256=carried_rows_sha256(property_raw, positions),
        replay=[evidence for _, evidence in outcomes],
    )
    all_verdicts = bool(summary["passed"].all() and summary["property_passed"].all())
    publication_passed = publication_gate(record.publication_policy, all_verdicts)
    manifest["composite"]["analysis"].update(
        {"verdicts_passed": all_verdicts, "publication_gate_passed": publication_passed}
    )
    manifest["composite"]["replacement"]["maximum_replay_deltas"] = {
        field: max(evidence["fields"][field]["absolute_difference"] for _, evidence in outcomes)
        for field in FIT_FIELDS
    }
    validate_composite_manifest(manifest, plan_raw)
    # A run must end under the same source as its replacement fits and complete analysis.
    if preflight(plan, slug, plan_raw)["subject"] != operation["subject"]:
        raise ValueError("source/runtime changed during scientific execution")
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    verify_output(output, plan_raw, slug, require_log=False, recompute=False)
    return publication_passed


def publication_gate(policy: str, all_verdicts: bool) -> bool:
    """Reporting studies retain complete negative findings; gated studies enforce verdicts."""
    if policy not in ("gated", "reporting"):
        raise ValueError("unknown publication policy")
    return policy == "reporting" or all_verdicts


def verify_producer_sources(metadata: dict[str, Any], expected: dict[str, str]) -> None:
    if metadata["source_sha256"] != expected:
        raise ValueError("producer source key/digest completeness differs from trusted plan")
    commit = metadata["subject"]["cleverly_commit"]
    for path, expected_hash in expected.items():
        if digest(git_blob(commit, path)) != expected_hash:
            raise ValueError(f"producer source digest differs from exact recorded Git blob: {path}")


def verify_output(
    output: Path,
    plan_raw: bytes,
    slug: str,
    *,
    committed: str | None = None,
    require_log: bool = True,
    recompute: bool = True,
) -> None:
    """Independent baseline/raw-row/source attribution checks; no fitted outcomes are rerun."""
    plan = json.loads(plan_raw)
    validate_plan(plan, all_batches())
    manifest_raw = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    validate_composite_manifest(manifest, plan_raw)
    if manifest["slug"] != slug:
        raise ValueError("output study does not match requested operation")
    relative = plan["studies"][slug]["artifact_directory"]
    original_manifest = git_blob(BASELINE_COMMIT, f"{relative}/manifest.json")
    if original_manifest != manifest["composite"]["baseline"]["manifest_utf8"].encode():
        raise ValueError("embedded baseline manifest is not the immutable original Git blob")
    baseline = json.loads(original_manifest)
    for name, expected_hash in manifest["sha256"].items():
        raw = (output / name).read_bytes()
        if digest(raw) != expected_hash:
            raise ValueError(f"final artifact hash differs: {name}")
        if name not in ("properties.csv", "property-replicates.csv.gz") and raw != git_blob(
            BASELINE_COMMIT, f"{relative}/{name}"
        ):
            raise ValueError(f"carried primary/reference bytes changed: {name}")
    before = git_blob(BASELINE_COMMIT, f"{relative}/property-replicates.csv.gz")
    if digest(before) != baseline["sha256"]["property-replicates.csv.gz"]:
        raise ValueError("original property artifact does not match its recorded hash")
    after_raw = gzip.decompress((output / "property-replicates.csv.gz").read_bytes())
    verify_splice(gzip.decompress(before), after_raw, plan["changes"][slug])
    if digest(after_raw) != manifest["composite"]["composite_property_csv_sha256"]:
        raise ValueError("composite property CSV digest differs")
    if (
        carried_rows_sha256(after_raw, plan["changes"][slug])
        != manifest["composite"]["carried_property_rows_sha256"]
    ):
        raise ValueError("carried property line receipt differs")
    _, original_rows = parse_lines(gzip.decompress(before))
    for evidence in manifest["composite"]["replacement"]["replay"]:
        old = original_rows[row_key(evidence["position"])][1]
        if any(float(old[field]) != evidence["fields"][field]["baseline"] for field in FIT_FIELDS):
            raise ValueError(
                "replay evidence baseline values differ from the immutable original CSV"
            )
    for name in ("replacement", "analysis"):
        metadata = manifest["composite"][name]
        verify_producer_sources(metadata, plan["source_sha256"])
        commit = metadata["subject"]["cleverly_commit"]
        if git_blob(commit, PLAN_PATH.relative_to(ROOT).as_posix()) != plan_raw:
            raise ValueError("recorded producer did not execute the exact committed plan")
    record = record_for(slug)
    analysis = manifest["composite"]["analysis"]
    if recompute:
        summary = record.properties().summarize_properties(
            pd.read_csv(io.BytesIO(after_raw), float_precision="round_trip")
        )
        published = pd.read_csv(output / "properties.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(
            published,
            summary,
            check_dtype=False,
            check_exact=False,
            rtol=REPLAY_TOLERANCE,
            atol=REPLAY_TOLERANCE,
        )
        all_verdicts = bool(summary["passed"].all() and summary["property_passed"].all())
        if analysis["verdicts_passed"] != all_verdicts or analysis[
            "publication_gate_passed"
        ] != publication_gate(record.publication_policy, all_verdicts):
            raise ValueError(
                "recorded scientific/publication verdict differs from full recomputation"
            )
    if require_log:
        verify_run_log(output, plan_raw, manifest)
    if committed is not None:
        commit = git_value("rev-parse", "--verify", "--end-of-options", f"{committed}^{{commit}}")
        for name in (*manifest["sha256"], "manifest.json", "run.log"):
            if git_blob(commit, f"{relative}/{name}") != (output / name).read_bytes():
                raise ValueError(
                    f"committed composite artifact differs from verified output: {name}"
                )


def verify_run_log(output: Path, plan_raw: bytes, manifest: dict[str, Any]) -> None:
    log = json.loads((output / "run.log").read_bytes())
    hashes = {
        **manifest["sha256"],
        "manifest.json": digest((output / "manifest.json").read_bytes()),
    }
    count = len(manifest["composite"]["replacement"]["positions"])
    passed = manifest["composite"]["analysis"]["publication_gate_passed"]
    if (
        log.get("source_commit")
        != manifest["composite"]["replacement"]["subject"]["cleverly_commit"]
        or log.get("plan_sha256") != digest(plan_raw)
        or log.get("artifact_sha256") != hashes
        or log.get("replacement_positions") != count
        or log.get("old_seed_replays") != count
        or log.get("exit_code") != (0 if passed else 1)
        or log.get("publication_gate_passed") != passed
        or log.get("scientific_verdicts_passed")
        != manifest["composite"]["analysis"]["verdicts_passed"]
        or log.get("threads") != dict.fromkeys(THREAD_VARIABLES, "1")
        or not log.get("started")
        or not log.get("finished")
    ):
        raise ValueError("completed run log attribution/counts/hashes/exit stamp differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--study", choices=tuple(EXPECTED_COUNTS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--committed-commit")
    args = parser.parse_args()
    if args.write_plan:
        if args.audit_file is None or args.plan.exists():
            parser.error("--write-plan requires original --audit-file and a new plan path")
        args.plan.write_bytes(json_bytes(build_plan(args.audit_file)))
        return 0
    if args.study is None or args.output is None or args.jobs < 1:
        parser.error("--study, --output and positive --jobs are required")
    if args.committed_commit and not args.verify_only:
        parser.error("--committed-commit is a verification option")
    raw = args.plan.read_bytes()
    if args.verify_only:
        verify_output(args.output, raw, args.study, committed=args.committed_commit)
        print(f"verified {args.study}: carried bytes, replacement keys and composite provenance")
        return 0
    if args.plan.resolve() != PLAN_PATH.resolve():
        parser.error("scientific execution requires the canonical committed plan path")
    args.output = outside(args.output, "targeted seed repair")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("scientific execution requires a new empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.UTC).isoformat()
    wall_start = time.monotonic()
    try:
        passed = execute(raw, args.study, args.output, jobs=args.jobs)
    except Exception as error:
        (args.output / "failure.json").write_bytes(
            json_bytes({"error_type": type(error).__name__, "error": str(error)})
        )
        raise
    manifest = json.loads((args.output / "manifest.json").read_bytes())
    log = {
        "operation": "targeted_seed_repair",
        "command": sys.argv,
        "started": started,
        "finished": datetime.datetime.now(datetime.UTC).isoformat(),
        "wall_seconds": time.monotonic() - wall_start,
        "source_commit": manifest["composite"]["replacement"]["subject"]["cleverly_commit"],
        "plan_sha256": digest(raw),
        "replacement_positions": len(manifest["composite"]["replacement"]["positions"]),
        "old_seed_replays": len(manifest["composite"]["replacement"]["replay"]),
        "scientific_verdicts_passed": manifest["composite"]["analysis"]["verdicts_passed"],
        "publication_gate_passed": passed,
        "exit_code": 0 if passed else 1,
        "artifact_sha256": {
            **manifest["sha256"],
            "manifest.json": digest((args.output / "manifest.json").read_bytes()),
        },
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }
    (args.output / "run.log").write_bytes(json_bytes(log))
    verify_run_log(args.output, raw, manifest)
    print(
        json.dumps(
            {
                "study": args.study,
                "scientific_verdicts_passed": manifest["composite"]["analysis"]["verdicts_passed"],
                "publication_gate_passed": passed,
                "output": str(args.output),
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

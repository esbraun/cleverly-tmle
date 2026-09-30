"""Audited sparse seed replacement and byte-preserving composite evidence.

This operates on a precommitted plan. It never fits a carried primary row, changes a
study declaration, or treats a historical dirty commit as a reconstruction of its tree.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from cleverly.validation._seeds import sample_seed_streams
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyBatch
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

BASELINE_COMMIT = "1a6de6ae9141ce65936a47efc3147ff36672dd6a"
ALLOCATION_COMMIT = "bb4772a9161fe271c1038c111636df856429548c"
PLAN_PATH = ROOT / "tests/canonical/seed-repair-plan.json"
# R4, declared before outcomes; the rule in tests/diagnostics/rm18_shared.py.
REPLAY_TOLERANCE = 1e-9
MODE = "targeted_seed_repair"
RUNTIME_FIELDS = ("python", "numpy", "scipy", "pandas", "scikit_learn")
EXPECTED_COUNTS = {
    "canonical-tmle": 1,
    "canonical-cvtmle": 1,
    "fold-evaluated-cvtmle": 1,
    "fold-targeted-cvtmle": 1,
    "repeated-crossfit-tmle": 1,
    "canonical-multi-arm-drtmle": 269,
}
STRUCTURAL_FIELDS = (
    "property",
    "cell",
    "role",
    "replicate",
    "n",
    "requested_replicates",
    "failed_replicates",
    "covered",
    "rejected",
)
FIT_FIELDS = ("truth", "estimate", "std_error")
Key = tuple[str, str, int]


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def cell_declaration(cell: Any) -> dict[str, Any]:
    return {
        "property": cell.property,
        "cell": cell.cell,
        "n": cell.n,
        "replicates": cell.replicates,
        "seed": cell.seed,
        "role": cell.role,
        "estimand": cell.estimand,
        "law": cell.dgp.name,
        "fit_kwargs": cell.fit_kwargs,
    }


def batch_declaration(batch: PropertyBatch) -> dict[str, Any]:
    return {"name": batch.name, "cells": [cell_declaration(cell) for cell in batch.cells]}


def all_batches() -> dict[str, tuple[PropertyBatch, ...]]:
    """Inventory declared actual calls without invoking their generators or preflights."""
    out = {}
    for record in registered():
        declaration = getattr(record.properties(), "sampling_batches", None)
        if declaration is not None:
            out[record.slug] = declaration()
    return out


def changed_seeds(batches: Iterable[PropertyBatch]) -> list[dict[str, Any]]:
    changes = []
    seen: set[Key] = set()
    for batch in batches:
        budgets: dict[int, int] = {}
        for cell in batch.cells:
            budgets[cell.seed] = max(budgets.get(cell.seed, 0), cell.replicates)
        streams = sample_seed_streams(budgets)
        old = {
            root: np.random.SeedSequence(root).generate_state(count)
            for root, count in budgets.items()
        }
        for cell in batch.cells:
            for index in range(cell.replicates):
                before, after = int(old[cell.seed][index]), streams[cell.seed][index]
                if before == after:
                    continue
                key = (cell.property, cell.cell, index)
                if key in seen:
                    raise ValueError(f"duplicate replacement key: {key}")
                seen.add(key)
                changes.append(
                    {
                        "batch": batch.name,
                        "property": cell.property,
                        "cell": cell.cell,
                        "replicate": index,
                        "root": cell.seed,
                        "old_seed": before,
                        "new_seed": after,
                    }
                )
    return changes


def source_files(records: Iterable[StudyRecord]) -> tuple[str, ...]:
    """Package and complete analysis framework; no claim that a library file is irrelevant."""
    paths = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src/cleverly").rglob("*.py")}
    paths.update(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "tests/studies/evidence").glob("*.py")
    )
    paths.update(module for record in records for module in record.modules)
    paths.update(
        {
            "tests/parallel.py",
            "tests/canonical/repair_property_seeds.py",
            "tests/canonical/declared_run.py",
            "tests/canonical/regenerate.py",
            "tests/diagnostics/rm18_shared.py",
        }
    )
    return tuple(sorted(paths))


def validate_plan(
    plan: Mapping[str, Any], batches: Mapping[str, tuple[PropertyBatch, ...]]
) -> None:
    if plan.get("schema_version") != 1 or plan.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("wrong seed-repair plan schema/baseline")
    if plan.get("allocation_commit") != ALLOCATION_COMMIT:
        raise ValueError("wrong frozen allocation source")
    if plan.get("replay_tolerance") != REPLAY_TOLERANCE:
        raise ValueError("replay rule changed after declaration")
    declared = {
        slug: [batch_declaration(batch) for batch in items] for slug, items in batches.items()
    }
    # JSON normalizes tuples in declared fit arguments without dropping any information.
    if json.loads(json_bytes(declared)) != plan.get("batches"):
        raise ValueError("actual sampling batches/declarations differ from the committed plan")
    if len(batches) != 11 or sum(map(len, batches.values())) != 12:
        raise ValueError("the full 11-study/12-call audit is incomplete")
    observed = {slug: changed_seeds(items) for slug, items in batches.items()}
    observed = {slug: values for slug, values in observed.items() if values}
    if observed != plan.get("changes"):
        raise ValueError("replacement keys/roots/old-new seeds differ from the fixed planner")
    if {slug: len(values) for slug, values in observed.items()} != EXPECTED_COUNTS:
        raise ValueError("replacement coverage differs from 269 SL plus five calibration rows")
    if set(plan.get("studies", {})) != set(EXPECTED_COUNTS):
        raise ValueError("baseline provenance does not cover every affected study")


def row_key(row: Mapping[str, Any]) -> Key:
    value = float(row["replicate"])
    if not math.isfinite(value) or not value.is_integer() or value < 0:
        raise ValueError("property replicate key is not a nonnegative integer")
    return str(row["property"]), str(row["cell"]), int(value)


def parse_lines(raw: bytes) -> tuple[bytes, dict[Key, tuple[bytes, dict[str, str]]]]:
    """Keep each original line, so a CSV read/write cannot perturb a carried decimal."""
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("property table must be LF-only and end with a newline")
    lines = raw.splitlines(keepends=True)
    header = lines[0]
    if tuple(next(csv.reader([header.decode()]))) != REPLICATE_COLUMNS:
        raise ValueError("property replication schema differs")
    rows = {}
    for line in lines[1:]:
        values = next(csv.reader([line.decode()]))
        if len(values) != len(REPLICATE_COLUMNS):
            raise ValueError("malformed/multiline property row")
        row = dict(zip(REPLICATE_COLUMNS, values, strict=True))
        key = row_key(row)
        if key in rows:
            raise ValueError(f"duplicate property key {key}")
        rows[key] = (line, row)
    return header, rows


def validate_complete_rows(rows: Mapping[Key, tuple[bytes, Mapping[str, str]]]) -> None:
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for key, (_, row) in rows.items():
        groups.setdefault(key[:2], []).append(row)
        for field in FIT_FIELDS:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"nonfinite stored {field} at {key}")
        if float(row["std_error"]) < 0:
            raise ValueError(f"negative standard error at {key}")
        for field in ("covered", "rejected"):
            if row[field] not in ("0", "1"):
                raise ValueError(f"nonbinary {field} at {key}")
    for key, group in groups.items():
        counts = {int(row["requested_replicates"]) for row in group}
        if len(counts) != 1 or {int(row["failed_replicates"]) for row in group} != {0}:
            raise ValueError(f"failed/inconsistent baseline replication budget: {key}")
        count = counts.pop()
        if count != len(group) or {int(row["replicate"]) for row in group} != set(range(count)):
            raise ValueError(f"missing replication within full budget: {key}")


def replay_check(old: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    """R4 over every numeric fitted field; keys/decisions/budgets have no slack."""
    if set(replay) != set(REPLICATE_COLUMNS) or set(old) != set(REPLICATE_COLUMNS):
        raise ValueError("missing/extra replay fields")
    result = {}
    for field in STRUCTURAL_FIELDS:
        before, after = str(old[field]), str(replay[field])
        if field not in ("property", "cell", "role"):
            numeric_before, numeric_after = float(before), float(after)
            if (
                not math.isfinite(numeric_before)
                or not math.isfinite(numeric_after)
                or not numeric_before.is_integer()
                or not numeric_after.is_integer()
            ):
                raise ValueError(f"replay changed exact integral {field}")
            before, after = str(int(numeric_before)), str(int(numeric_after))
        if before != after:
            raise ValueError(f"replay changed exact {field}: {before} -> {after}")
    for field in FIT_FIELDS:
        before, after = float(old[field]), float(replay[field])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError(f"nonfinite replay {field}")
        difference = abs(after - before)
        bound = REPLAY_TOLERANCE * max(1.0, abs(before))
        result[field] = {
            "baseline": before,
            "replayed": after,
            "absolute_difference": difference,
            "bound": bound,
            "exact": before == after,
        }
        if difference > bound:
            raise ValueError(f"replay {field} difference {difference} exceeds predeclared {bound}")
    return result


def splice_rows(
    baseline_raw: bytes,
    replacements: Iterable[Mapping[str, Any]],
    planned: Iterable[Mapping[str, Any]],
) -> bytes:
    header, baseline = parse_lines(baseline_raw)
    validate_complete_rows(baseline)
    expected = {row_key(row) for row in planned}
    replacement_map = {}
    for row in replacements:
        if set(row) != set(REPLICATE_COLUMNS):
            raise ValueError("replacement columns differ from property schema")
        key = row_key(row)
        if key in replacement_map:
            raise ValueError(f"duplicate replacement {key}")
        replacement_map[key] = row
    if set(replacement_map) != expected or not expected <= set(baseline):
        raise ValueError("missing/extra replacement keys")
    lines = [header]
    for key, (line, old) in baseline.items():
        row = replacement_map.get(key)
        if row is None:
            lines.append(line)
            continue
        for field in STRUCTURAL_FIELDS[:-2]:
            if field in ("property", "cell", "role"):
                matches = str(row[field]) == old[field]
            else:
                matches = float(row[field]) == float(old[field])
            if not matches:
                raise ValueError(f"replacement changed declared {field} at {key}")
        for field in FIT_FIELDS:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"nonfinite replacement {field} at {key}")
        if (
            float(row["std_error"]) < 0
            or row["covered"] not in (0, 1)
            or row["rejected"] not in (0, 1)
        ):
            raise ValueError(f"invalid replacement inference fields at {key}")
        stream = io.StringIO(newline="")
        csv.DictWriter(stream, fieldnames=REPLICATE_COLUMNS, lineterminator="\n").writerow(row)
        lines.append(stream.getvalue().encode())
    return b"".join(lines)


def verify_splice(
    baseline_raw: bytes, composite_raw: bytes, planned: Iterable[Mapping[str, Any]]
) -> None:
    before_header, before = parse_lines(baseline_raw)
    after_header, after = parse_lines(composite_raw)
    validate_complete_rows(after)
    expected = {row_key(row) for row in planned}
    if before_header != after_header or set(before) != set(after) or not expected <= set(after):
        raise ValueError("composite row keys/header differ from baseline")
    for key in before:
        if key not in expected and before[key][0] != after[key][0]:
            raise ValueError(f"untouched property row changed: {key}")
        if key in expected:
            if before[key][0] == after[key][0]:
                raise ValueError(f"planned seed position is unchanged and needs review: {key}")
            for field in STRUCTURAL_FIELDS[:-2]:
                if before[key][1][field] != after[key][1][field]:
                    raise ValueError(f"replacement metadata changed: {key}/{field}")


def carried_rows_sha256(raw: bytes, planned: Iterable[Mapping[str, Any]]) -> str:
    """Receipt for the exact header and every line attributed to the historical run."""
    header, rows = parse_lines(raw)
    changed = {row_key(position) for position in planned}
    return digest(header + b"".join(line for key, (line, _) in rows.items() if key not in changed))


def composite_manifest(
    baseline_raw: bytes,
    *,
    plan_raw: bytes,
    slug: str,
    operation: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    property_raw_sha256: str,
    carried_property_rows_sha256: str,
    replay: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = json.loads(baseline_raw)
    plan = json.loads(plan_raw)
    manifest = copy.deepcopy(baseline)
    manifest["sha256"] = dict(artifact_hashes)
    manifest["generation_mode"] = MODE
    manifest["composite"] = {
        "baseline": {
            "commit": plan["baseline_commit"],
            "manifest_utf8": baseline_raw.decode(),
            "manifest_sha256": digest(baseline_raw),
        },
        "operation_plan": {
            "path": PLAN_PATH.relative_to(ROOT).as_posix(),
            "sha256": digest(plan_raw),
        },
        "replacement": {**operation, "positions": plan["changes"][slug], "replay": replay},
        "analysis": dict(operation),
        "artifact_attribution": {
            name: (
                "composite_replicates"
                if name == "property-replicates.csv.gz"
                else "analysis"
                if name == "properties.csv"
                else "baseline"
            )
            for name in baseline["sha256"]
        },
        "composite_property_csv_sha256": property_raw_sha256,
        "carried_property_rows_sha256": carried_property_rows_sha256,
    }
    return manifest


def validate_composite_manifest(manifest: Mapping[str, Any], plan_raw: bytes) -> None:
    """Fast-suite provenance gate; validates recorded attribution without restamping it."""
    if manifest.get("generation_mode") != MODE:
        raise ValueError("unrecognized composite generation mode")
    plan = json.loads(plan_raw)
    slug = manifest["slug"]
    section = manifest["composite"]
    baseline_info = section["baseline"]
    baseline_raw = baseline_info["manifest_utf8"].encode()
    baseline = json.loads(baseline_raw)
    expected_baseline = plan["studies"][slug]
    if baseline_info["commit"] != plan["baseline_commit"] or baseline_info[
        "manifest_sha256"
    ] != digest(baseline_raw):
        raise ValueError("baseline source/hash attribution differs")
    if digest(baseline_raw) != expected_baseline["manifest_sha256"]:
        raise ValueError("original manifest differs from immutable baseline plan")
    if baseline["sha256"] != expected_baseline["artifact_sha256"]:
        raise ValueError("baseline artifact hash declarations differ from immutable plan")
    for field in (
        "generated_with",
        "study_module_sha256",
        "reference_sha256",
        "configuration",
        "study",
        "slug",
        "schema_version",
    ):
        if manifest[field] != baseline[field]:
            raise ValueError(f"historical producer/declaration field restamped: {field}")
    if section["operation_plan"] != {
        "path": PLAN_PATH.relative_to(ROOT).as_posix(),
        "sha256": digest(plan_raw),
    }:
        raise ValueError("operation plan attribution changed")
    if section["replacement"]["positions"] != plan["changes"][slug]:
        raise ValueError("replacement key/seed provenance differs from predeclared plan")
    attribution = section["artifact_attribution"]
    expected_attribution = {
        name: (
            "composite_replicates"
            if name == "property-replicates.csv.gz"
            else "analysis"
            if name == "properties.csv"
            else "baseline"
        )
        for name in baseline["sha256"]
    }
    if attribution != expected_attribution or set(manifest["sha256"]) != set(baseline["sha256"]):
        raise ValueError("artifact attribution/coverage differs")
    for name, owner in attribution.items():
        if owner == "baseline" and manifest["sha256"][name] != baseline["sha256"][name]:
            raise ValueError(f"baseline primary/reference artifact changed: {name}")
    for name in ("replacement", "analysis"):
        metadata = section[name]
        producer = {
            key: metadata[key]
            for key in (
                "subject",
                "source_sha256",
                "distribution_versions",
                "distribution_versions_sha256",
            )
        }
        if metadata.get("producer_sha256") != digest(json_bytes(producer)):
            raise ValueError(f"{name} producer identity/digest attribution changed")
        if producer["distribution_versions_sha256"] != digest(
            json_bytes(producer["distribution_versions"])
        ):
            raise ValueError(f"{name} runtime inventory digest changed")
        subject = metadata["subject"]
        if (
            subject.get("cleverly_worktree_clean") is not True
            or len(subject.get("cleverly_commit", "")) != 40
        ):
            raise ValueError(f"missing clean committed {name} provenance")
        if any(
            subject.get(field) != expected_baseline["runtime"][field] for field in RUNTIME_FIELDS
        ):
            raise ValueError(f"{name} runtime differs from recorded baseline")
        if metadata.get("source_sha256") != plan["source_sha256"]:
            raise ValueError(f"{name} source digest attribution differs from precommitted plan")
    evidence = section["replacement"]["replay"]
    if (
        len(evidence) != len(plan["changes"][slug])
        or [item["position"] for item in evidence] != plan["changes"][slug]
    ):
        raise ValueError("missing/extra old-seed replay evidence")
    for item in evidence:
        if item.get("structural_fields_and_decisions_exact") is not True:
            raise ValueError("replay did not preserve structural fields/decisions exactly")
        checks = item["fields"]
        if set(checks) != set(FIT_FIELDS):
            raise ValueError("replay evidence omits numeric fitted fields")
        for check in checks.values():
            before, after = check["baseline"], check["replayed"]
            bound = REPLAY_TOLERANCE * max(1.0, abs(before))
            difference = abs(after - before)
            if not math.isfinite(before) or not math.isfinite(after) or difference > bound:
                raise ValueError("old-seed replay failed the declared gate")
            if check != {
                "baseline": before,
                "replayed": after,
                "absolute_difference": difference,
                "bound": bound,
                "exact": before == after,
            }:
                raise ValueError("replay diagnostic values were altered")
    expected_maxima = {
        field: max(item["fields"][field]["absolute_difference"] for item in evidence)
        for field in FIT_FIELDS
    }
    if section["replacement"].get("maximum_replay_deltas") != expected_maxima:
        raise ValueError("replay maximum differences were altered/omitted")

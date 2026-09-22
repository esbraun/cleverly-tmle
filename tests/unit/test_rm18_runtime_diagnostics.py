"""Mutation controls for the one-off RM18 runtime diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics.rm18_runtime.compare import (
    ARMS,
    CODE,
    HERE,
    STUDIES,
    Arms,
    _reference_failures,
    _same_rows,
    _verdict,
    fixed_condition_failures,
    reading,
)
from tests.diagnostics.rm18_runtime.row_drift import PROPERTY_KEYS, drift
from tests.studies.evidence.manifest import write_csv, write_lines

STUDY = next(study for study in STUDIES if study.directory == "lmtp_ltmle")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_lines(path, json.dumps(payload, indent=2) + "\n")


@pytest.fixture
def valid_arms(tmp_path: Path) -> Arms:
    """Four compact arms with the real identity and configuration declarations."""
    primary = pd.DataFrame(
        [
            {
                "implementation": STUDY.implementation,
                "scenario": "law",
                "replicate": 0,
                "n": 10,
                "estimand": "ey_regimen[never]",
                "estimate": 0.4,
                "std_error": 0.1,
                "covered": True,
            },
            {
                "implementation": STUDY.reference_implementation,
                "scenario": "law",
                "replicate": 0,
                "n": 10,
                "estimand": "ey_regimen[never]",
                "estimate": 0.41,
                "std_error": 0.11,
                "covered": True,
            },
        ]
    )
    properties = pd.DataFrame(
        [
            {
                "property": "crossfit_overfitting",
                "cell": "cross_fitted_ltmle",
                "role": "positive",
                "replicate": 0,
                "estimate": 0.4,
                "std_error": 0.1,
                "covered": True,
            }
        ]
    )
    summaries = {
        "properties.csv": pd.DataFrame(
            [
                {
                    "property": "crossfit_overfitting",
                    "cell": "cross_fitted_ltmle",
                    "role": "positive",
                    "passed": True,
                    "property_passed": False,
                }
            ]
        ),
        "equivalence.csv": pd.DataFrame(
            [
                {
                    "scenario": "law",
                    "estimand": "ey_regimen[never]",
                    "passed": True,
                    "comparison_conclusion": "equivalent",
                }
            ]
        ),
        "performance-tests.csv": pd.DataFrame(
            [
                {
                    "implementation": STUDY.implementation,
                    "scenario": "law",
                    "estimand": "ey_regimen[never]",
                    "passed": True,
                }
            ]
        ),
    }
    for arm in ARMS:
        output = tmp_path / STUDY.directory / arm / "output"
        output.mkdir(parents=True)
        write_csv(primary, output / "replicates.csv.gz", compression="gzip")
        write_csv(properties, output / "property-replicates.csv.gz", compression="gzip")
        for name, frame in summaries.items():
            write_csv(frame, output / name)

        source = HERE / "arms" / f"{STUDY.directory}-{arm}-manifest.json"
        manifest = _manifest(source)
        manifest["sha256"] = {
            name: _digest(output / name)
            for name in (
                "replicates.csv.gz",
                "property-replicates.csv.gz",
                *summaries,
            )
        }
        _write_manifest(output / "manifest.json", manifest)
    return Arms(tmp_path, STUDY)


def test_the_compact_arms_meet_every_fixed_condition(valid_arms: Arms) -> None:
    assert fixed_condition_failures(valid_arms) == []


def test_an_output_mutation_invalidates_its_manifest(valid_arms: Arms) -> None:
    path = valid_arms.output("F-R13", "properties.csv")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    failures = fixed_condition_failures(valid_arms)

    assert any("F-R13: consumed artifact properties.csv hashes to" in item for item in failures)


def test_a_missing_consumed_artifact_invalidates_without_being_read(valid_arms: Arms) -> None:
    valid_arms.output("P-R11", "replicates.csv.gz").unlink()

    failures = fixed_condition_failures(valid_arms)

    assert "P-R11: consumed artifact replicates.csv.gz is missing" in failures


def test_a_fixed_configuration_mutation_invalidates_the_harness(valid_arms: Arms) -> None:
    path = valid_arms.output("P-R11", "manifest.json")
    manifest = _manifest(path)
    manifest["configuration"]["margins"]["coverage_floor"] = 0.123
    _write_manifest(path, manifest)

    failures = fixed_condition_failures(valid_arms)

    assert any("fixed configuration differs across arms" in item for item in failures)


def test_a_reference_row_mutation_invalidates_the_harness(valid_arms: Arms) -> None:
    arm = "F-R13"
    path = valid_arms.output(arm, "replicates.csv.gz")
    rows = pd.read_csv(path)
    reference = rows["implementation"] == STUDY.reference_implementation
    rows.loc[reference, "estimate"] += 0.01
    write_csv(rows, path, compression="gzip")
    manifest_path = valid_arms.output(arm, "manifest.json")
    manifest = _manifest(manifest_path)
    manifest["sha256"][path.name] = _digest(path)
    _write_manifest(manifest_path, manifest)

    failures = fixed_condition_failures(valid_arms)

    assert f"{arm}: lmtp comparator rows differ from F-R11" in failures


def test_duplicate_comparator_keys_do_not_count_as_identical_rows() -> None:
    rows = pd.DataFrame([{"key": 1, "value": 2.0}])

    assert not _same_rows(pd.concat([rows, rows], ignore_index=True), rows, ("key",))


def test_a_weighted_reference_inference_mutation_is_reported(tmp_path: Path) -> None:
    study = next(study for study in STUDIES if study.directory == "weighted_lmtp_ltmle")
    primary = pd.DataFrame(
        [
            {
                "implementation": study.reference_implementation,
                "scenario": "law",
                "replicate": 0,
                "n": 10,
                "estimand": "ey_regimen[never]",
                "estimate": 0.4,
                "std_error": 0.1,
            }
        ]
    )
    inference = primary.assign(inference_method="native")
    for arm in ARMS:
        output = tmp_path / study.directory / arm / "output"
        output.mkdir(parents=True)
        write_csv(primary, output / "replicates.csv.gz", compression="gzip")
        candidate = inference.assign(std_error=0.2) if arm == "P-R11" else inference
        write_csv(candidate, output / "reference-inference.csv.gz", compression="gzip")

    failures = _reference_failures(Arms(tmp_path, study))

    assert "P-R11: reference-inference rows differ from F-R11" in failures


def test_a_commit_with_the_right_prefix_and_extra_text_is_not_the_pinned_commit(
    valid_arms: Arms,
) -> None:
    arm = "P-R11"
    path = valid_arms.output(arm, "manifest.json")
    manifest = _manifest(path)
    manifest["generated_with"]["subject"]["cleverly_commit"] = CODE["P"] + "-other"
    _write_manifest(path, manifest)

    failures = fixed_condition_failures(valid_arms)

    assert any(f"{arm}: cleverly_commit" in item for item in failures)


def test_a_malformed_manifest_is_a_fixed_condition_failure(valid_arms: Arms) -> None:
    path = valid_arms.output("P-R13", "manifest.json")
    manifest = _manifest(path)
    manifest["generated_with"] = "not an object"
    _write_manifest(path, manifest)

    failures = fixed_condition_failures(valid_arms)

    assert "P-R13: generated_with is not an object" in failures
    assert "P-R13: generated_with.subject is not an object" in failures


def test_the_cell_verdict_does_not_inherit_the_family_verdict() -> None:
    row = pd.Series({"passed": True, "property_passed": False})
    assert _verdict(row)


@pytest.mark.parametrize("value", ["False", 0, 1, None])
def test_a_non_boolean_cell_verdict_is_refused(value: object) -> None:
    with pytest.raises(TypeError, match="passed must be boolean"):
        _verdict(pd.Series({"passed": value}))


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({("code", "R11"): True, ("code", "R13"): True}, "code"),
        ({("runtime", "F"): True, ("runtime", "P"): True}, "runtime"),
        (
            {
                ("code", "R11"): True,
                ("code", "R13"): True,
                ("runtime", "F"): True,
                ("runtime", "P"): True,
            },
            "both",
        ),
        ({}, "neither"),
    ],
)
def test_each_declared_verdict_reading(changes: dict[tuple[str, str], bool], expected: str) -> None:
    complete = {
        ("code", "R11"): False,
        ("code", "R13"): False,
        ("runtime", "F"): False,
        ("runtime", "P"): False,
        **changes,
    }
    assert reading(complete) == expected


def test_an_incomplete_axis_change_is_refused() -> None:
    changes = {
        ("code", "R11"): True,
        ("code", "R13"): False,
        ("runtime", "F"): False,
        ("runtime", "P"): False,
    }
    with pytest.raises(AssertionError, match="verdict pattern"):
        reading(changes)


def test_a_reading_with_a_missing_comparison_is_refused() -> None:
    with pytest.raises(AssertionError, match="needs exactly"):
        reading({("code", "R11"): True})


def test_drift_reports_a_missing_value_and_an_extra_key() -> None:
    older = pd.DataFrame(
        [
            {
                "property": "p",
                "cell": "c",
                "role": "positive",
                "replicate": 0,
                "estimate": 1.0,
                "std_error": 0.1,
                "covered": True,
            }
        ]
    )
    newer = pd.concat(
        [
            older.assign(estimate=np.nan),
            older.assign(replicate=1),
        ],
        ignore_index=True,
    )

    result = drift(older, newer, PROPERTY_KEYS, tolerance=1e-9)

    assert result["rows_only_newer"] == 1
    assert result["rows_outside_tolerance"] == 1
    assert result["rows_differing"] == 1

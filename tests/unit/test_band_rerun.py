"""The band-rerun plan, its replay rule, its composite manifest and its blocked sidecars.

``tests/studies/evidence/band_rerun.py`` refits only the band keys of a study after the default
band moved (band-finite-sample plan sections 4.4, 8 and 9).  These tests hold the committed plan
to the tree without fitting, and hold the replay rule, the splice and the composite gate to
synthetic rows, each with a mutation that the gate must refuse.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.studies.evidence import band_rerun
from tests.studies.evidence.manifest import csv_text
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.registry import ROOT

BAND = "simultaneous_coverage/a__simultaneous_band"
POINT = "point/a"


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return json.loads(band_rerun.PLAN_PATH.read_bytes())


def test_the_committed_plan_holds_against_the_tree(plan: dict[str, Any]) -> None:
    assert band_rerun.validate_plan(plan) == []


def test_the_plan_excludes_the_studies_the_default_does_not_reach(plan: dict[str, Any]) -> None:
    """The curve band keeps its options, and the SDR band cells name theirs (plan 4.3, 4.4)."""
    assert set(plan["excluded"]) == {"longitudinal-sdr"}
    assert "targeting-curves" not in plan["studies"]
    assert any(
        band_rerun.EXCLUSION_WITNESS["longitudinal-sdr"] in pin
        for pin in plan["excluded"]["longitudinal-sdr"]["pins"]
    )
    for entry in plan["studies"].values():
        assert entry["band_fits"] > 0 and entry["moved_cells"], entry["artifact_directory"]


def test_every_blocked_sidecar_matches_its_record(plan: dict[str, Any]) -> None:
    for record in sorted((ROOT / "tests/canonical").glob(f"*/{band_rerun.SIDECAR}/record.json")):
        assert band_rerun.validate_sidecar(record.parent.parent, plan) == [], record


def test_no_support_or_grid_source_reads_a_sidecar() -> None:
    """A blocked run is diagnostic: its verdicts reach no support row and no grid row (4.4)."""
    sources = [*(ROOT / "src/cleverly").rglob("*.py"), *(ROOT / "tests/studies").rglob("*.py")]
    readers = [
        path.relative_to(ROOT).as_posix()
        for path in sources
        if path.name != "band_rerun.py"
        and any(
            text in path.read_text(encoding="utf-8")
            for text in ("band_rerun", *band_rerun.SIDECAR_FILES)
        )
    ]
    assert readers == []


# ---------------------------------------------------------------------------- synthetic rows


def row(cell: str, replicate: int, estimate: float, **fields: Any) -> dict[str, Any]:
    prop, name = cell.split("/", 1)
    out = {
        "property": prop,
        "cell": name,
        "role": "positive",
        "replicate": replicate,
        "n": 500,
        "requested_replicates": 2,
        "failed_replicates": 0,
        "truth": 0.25,
        "estimate": estimate,
        "std_error": 1.0,
        "covered": 1,
        "rejected": 0,
    }
    out.update(fields)
    return {column: out.get(column, 0) for column in REPLICATE_COLUMNS}


def committed() -> pd.DataFrame:
    return pd.DataFrame(
        [row(POINT, r, 0.3 + r) for r in range(2)] + [row(BAND, r, 1.5) for r in range(2)]
    )


def call() -> band_rerun.Call:
    return band_rerun.Call(
        label="properties",
        keys=[0, 1],
        selected=[0, 1],
        options={"cells_of": lambda key: [POINT, BAND], "replicate_of": lambda key: key},
    )


def refitted(point_shift: float = 0.0) -> list[list[dict[str, Any]]]:
    return [
        [row(POINT, r, 0.3 + r + point_shift), row(BAND, r, 2.5, covered=0, std_error=1.5)]
        for r in range(2)
    ]


def test_the_replay_rule_admits_rounding_and_refuses_a_real_change() -> None:
    before = row(POINT, 0, 0.3)
    assert band_rerun.compare(before, row(POINT, 0, 0.3 + 1e-12))["estimate"] < 1e-9
    with pytest.raises(band_rerun.ReplayMiss, match="estimate moved"):
        band_rerun.compare(before, row(POINT, 0, 0.3 + 1e-6))
    with pytest.raises(band_rerun.ReplayMiss, match="covered"):
        band_rerun.compare(before, row(POINT, 0, 0.3, covered=0))


def test_a_refit_replaces_only_band_rows() -> None:
    rows = committed()
    refit = band_rerun.replay(
        [call()], {"properties": refitted()}, band_rerun.committed_index(rows), [BAND]
    )
    assert (refit.compared, refit.exact, len(refit.band_rows)) == (2, 2, 2)
    composite = band_rerun.splice(rows, refit.band_rows)
    assert band_rerun.carried_digest(composite, [BAND]) == band_rerun.carried_digest(rows, [BAND])
    mask = band_rerun.band_mask(composite, [BAND])
    assert composite.loc[mask, "covered"].tolist() == [0, 0]
    assert composite.loc[~mask].equals(rows.loc[~mask, list(REPLICATE_COLUMNS)])


def test_a_moved_non_band_row_is_a_replay_miss() -> None:
    """Mutation control: a refit whose point estimate moves takes a complete rerun."""
    rows = committed()
    with pytest.raises(band_rerun.ReplayMiss):
        band_rerun.replay(
            [call()], {"properties": refitted(1e-4)}, band_rerun.committed_index(rows), [BAND]
        )


def test_the_splice_refuses_a_new_band_row() -> None:
    with pytest.raises(ValueError, match="new or repeated"):
        band_rerun.splice(committed(), [row(BAND, 7, 2.5)])


# ---------------------------------------------------------------------------- the composite


def operation() -> dict[str, Any]:
    subject = {"cleverly_commit": "a" * 40, "cleverly_worktree_clean": True, "numpy": "9.0"}
    out: dict[str, Any] = {
        "subject": subject,
        "source_sha256": {"src/cleverly/__init__.py": "0" * 64},
        "distribution_versions": [["numpy", "9.0"]],
        "distribution_versions_sha256": "1" * 64,
    }
    out["producer_sha256"] = band_rerun.digest(band_rerun.json_bytes(out))
    return out


def composite_fixture() -> tuple[dict[str, Any], bytes]:
    rows = committed()
    baseline = {
        "slug": "synthetic",
        "study": "synthetic",
        "schema_version": 1,
        "generated_with": {"subject": {"numpy": "8.0"}},
        "configuration": {},
        "sha256": {
            "property-replicates.parquet": "2" * 64,
            "properties.csv": "3" * 64,
            "replicates.parquet": "4" * 64,
        },
    }
    baseline_raw = band_rerun.json_bytes(baseline)
    selection = [{"label": "properties", "band_keys": 2, "cells": {BAND: 2}}]
    entry = {
        "manifest_sha256": band_rerun.digest(baseline_raw),
        "selection": selection,
        "band_fits": 2,
        "moved_cells": [BAND],
        "baseline_subject": {"numpy": "8.0"},
        "carried_rows_sha256": band_rerun.carried_digest(rows, [BAND]),
        "band_cells": [{"cell": "a__simultaneous_band", "replicates": 2}],
    }
    plan_raw = band_rerun.json_bytes(
        {"baseline_commit": band_rerun.BASELINE_COMMIT, "studies": {"synthetic": entry}}
    )
    refit = band_rerun.replay(
        [call()], {"properties": refitted()}, band_rerun.committed_index(rows), [BAND]
    )
    composite = band_rerun.splice(rows, refit.band_rows)
    hashes = dict(baseline["sha256"], **{"property-replicates.parquet": "5" * 64})
    manifest = band_rerun.composite_manifest(
        baseline_raw,
        plan_raw=plan_raw,
        slug="synthetic",
        operation=operation(),
        artifact_hashes=hashes,
        composite_raw=csv_text(composite),
        refit=refit,
        readings=[{"cell": "a__simultaneous_band"}],
        command=["python", "-m", "tests.studies.evidence.band_rerun"],
    )
    return manifest, plan_raw


def test_a_composite_manifest_passes_its_gate() -> None:
    manifest, plan_raw = composite_fixture()
    band_rerun.validate_composite_manifest(manifest, plan_raw)
    assert manifest["composite"]["replacement"]["declared_stack"] == {
        "numpy": {"baseline": "8.0", "ran": "9.0"}
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "baseline_artifact",
        "restamped_producer",
        "carried_receipt",
        "replay_rule",
        "selection",
        "dirty_tree",
    ],
)
def test_the_composite_gate_refuses_each_mutation(mutation: str) -> None:
    manifest, plan_raw = composite_fixture()
    altered = copy.deepcopy(manifest)
    section = altered["composite"]
    if mutation == "baseline_artifact":
        altered["sha256"]["replicates.parquet"] = "6" * 64
    elif mutation == "restamped_producer":
        altered["generated_with"] = {"subject": {"numpy": "9.0"}}
    elif mutation == "carried_receipt":
        section["carried_property_rows_sha256"] = "7" * 64
    elif mutation == "replay_rule":
        section["replacement"]["replay"]["tolerance"] = 1e-6
    elif mutation == "selection":
        section["replacement"]["band_fits"] = 3
    else:
        section["analysis"]["subject"]["cleverly_worktree_clean"] = False
    with pytest.raises(ValueError):
        band_rerun.validate_composite_manifest(altered, plan_raw)


def test_a_sidecar_with_an_altered_file_is_refused(tmp_path: Path, plan: dict[str, Any]) -> None:
    """Mutation control on the blocked sidecar: its files are held to their recorded hashes."""
    slug, entry = next(iter(plan["studies"].items()))
    directory = ROOT / entry["artifact_directory"]
    side = tmp_path / band_rerun.SIDECAR
    side.mkdir()
    for name in band_rerun.SIDECAR_FILES:
        (side / name).write_bytes(b"rows\n")
    record = {
        "mode": band_rerun.MODE,
        "blocked": True,
        "slug": slug,
        "baseline_commit": band_rerun.BASELINE_COMMIT,
        "plan_sha256": "8" * 64,
        "sha256": {name: band_rerun.digest(b"rows\n") for name in band_rerun.SIDECAR_FILES},
    }
    (side / "record.json").write_text(json.dumps(record))
    (tmp_path / "manifest.json").write_bytes((directory / "manifest.json").read_bytes())
    findings = band_rerun.validate_sidecar(tmp_path, plan)
    assert findings == [f"{slug}: the sidecar sits in another study's directory"]
    (side / band_rerun.SIDECAR_FILES[0]).write_bytes(b"other rows\n")
    findings = band_rerun.validate_sidecar(tmp_path, plan)
    assert f"{slug}: {band_rerun.SIDECAR_FILES[0]} differs from its recorded hash" in findings

"""The published canonical-TMLE evidence artifacts are complete and reproducible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tests.studies.canonical_properties import summarize_properties
from tests.studies.canonical_tmle import (
    ARTIFACT_COLUMNS,
    BINARY_ESTIMANDS,
    COMMON_ESTIMANDS,
    PRIMARY_N,
    PRIMARY_REPLICATES,
    R_BASE_IMAGE,
    equivalence,
    independent_performance_tests,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "tests" / "canonical" / "tmle3"


def _rows() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "replicates.csv.gz")


def test_manifest_hashes_every_published_result() -> None:
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "replicates.csv.gz",
        "summary.csv",
        "equivalence.csv",
        "performance-tests.csv",
        "property-replicates.csv.gz",
        "properties.csv",
    }
    assert set(manifest["sha256"]) == expected
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() == digest
    assert manifest["configuration"]["replicates"] == PRIMARY_REPLICATES
    assert manifest["configuration"]["n"] == PRIMARY_N
    assert manifest["configuration"]["cross_fit"] is False
    assert manifest["configuration"]["simultaneous_intervals"] is False
    assert manifest["configuration"]["statistical_testing"]["confidence_level"] == 0.99
    assert manifest["generated_with"]["r_base_image"] == R_BASE_IMAGE
    assert set(manifest["reference_sha256"]) == {"Dockerfile", "run_tmle3.R"}
    for name, digest in manifest["reference_sha256"].items():
        assert hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() == digest


def test_every_shared_estimand_has_both_implementations_and_every_replication() -> None:
    rows = _rows()
    assert tuple(rows.columns) == ARTIFACT_COLUMNS
    expected = {"continuous": set(COMMON_ESTIMANDS), "binary": set(BINARY_ESTIMANDS)}
    for scenario, estimands in expected.items():
        subset = rows.loc[rows["scenario"] == scenario]
        assert set(subset["estimand"]) == estimands
        for implementation in ("cleverly", "tmle3"):
            by_estimand = subset.loc[subset["implementation"] == implementation].groupby("estimand")
            assert set(by_estimand.size().index) == estimands
            assert (by_estimand.size() == PRIMARY_REPLICATES).all()
            assert (by_estimand["n"].first() == PRIMARY_N).all()


def test_published_summaries_are_recomputed_from_the_replication_rows() -> None:
    actual = pd.read_csv(ARTIFACTS / "summary.csv")
    recomputed = summarize(_rows())
    pd.testing.assert_frame_equal(actual, recomputed, check_exact=False, rtol=1e-12, atol=1e-12)


def test_every_cross_implementation_performance_gate_passes() -> None:
    rows = _rows()
    summary = summarize(rows)
    actual = pd.read_csv(ARTIFACTS / "equivalence.csv")
    recomputed = equivalence(rows, summary)
    pd.testing.assert_frame_equal(actual, recomputed, check_exact=False, rtol=1e-12, atol=1e-12)
    assert (actual["confidence_level"] == 0.99).all()
    assert actual["paired_similarity_99"].all(), actual.loc[~actual["paired_similarity_99"]]
    assert actual["passed"].all(), actual.loc[~actual["passed"]]


def test_cleverly_is_noninferior_to_the_r_reference_at_99_percent() -> None:
    actual = pd.read_csv(ARTIFACTS / "equivalence.csv")
    assert actual["cleverly_not_inferior_99"].all(), actual.loc[~actual["cleverly_not_inferior_99"]]


def test_each_implementation_independently_performs_at_99_percent() -> None:
    rows = _rows()
    actual = pd.read_csv(ARTIFACTS / "performance-tests.csv")
    recomputed = independent_performance_tests(rows)
    pd.testing.assert_frame_equal(actual, recomputed, check_exact=False, rtol=1e-12, atol=1e-12)
    assert set(actual["implementation"]) == {"cleverly", "tmle3"}
    assert (actual["confidence_level"] == 0.99).all()
    assert actual["bias_equivalent"].all(), actual.loc[~actual["bias_equivalent"]]
    assert actual["coverage_calibrated"].all(), actual.loc[~actual["coverage_calibrated"]]
    assert actual["se_calibrated"].all(), actual.loc[~actual["se_calibrated"]]
    assert actual["passed"].all(), actual.loc[~actual["passed"]]


@pytest.mark.parametrize("implementation", ["cleverly", "tmle3"])
def test_independent_harness_rejects_each_method_without_implicating_the_other(
    implementation: str,
) -> None:
    baseline = _rows().query("scenario == 'continuous' and estimand == 'ate'").copy()
    selected = baseline["implementation"] == implementation
    other = "tmle3" if implementation == "cleverly" else "cleverly"

    bias_mutation = baseline.copy()
    bias_mutation.loc[selected, "inference_estimate"] += float(
        baseline.loc[selected, "inference_estimate"].std(ddof=1)
    )
    bias_result = independent_performance_tests(bias_mutation).set_index("implementation")
    assert not bool(bias_result.loc[implementation, "bias_equivalent"])
    assert bool(bias_result.loc[other, "passed"])

    coverage_mutation = baseline.copy()
    coverage_mutation.loc[selected, "covered"] = 0
    coverage_result = independent_performance_tests(coverage_mutation).set_index("implementation")
    assert not bool(coverage_result.loc[implementation, "coverage_calibrated"])
    assert bool(coverage_result.loc[other, "passed"])

    se_mutation = baseline.copy()
    se_mutation.loc[selected, "std_error"] *= 2.0
    se_result = independent_performance_tests(se_mutation).set_index("implementation")
    assert not bool(se_result.loc[implementation, "se_calibrated"])
    assert bool(se_result.loc[other, "passed"])


def test_paired_harness_rejects_a_material_cleverly_regression() -> None:
    mutated = _rows().query("scenario == 'continuous' and estimand == 'ate'").copy()
    selected = mutated["implementation"] == "cleverly"
    mutated.loc[selected, "estimate"] += 0.05
    mutated.loc[selected, "covered"] = 0
    result = equivalence(mutated, summarize(mutated)).iloc[0]
    assert not bool(result["paired_similarity_99"])
    assert not bool(result["cleverly_not_inferior_99"])
    assert not bool(result["passed"])


def test_the_nonzero_binary_fixture_is_bounded_by_its_reported_uncertainty() -> None:
    fixture = _rows().query("scenario == 'binary' and replicate == 0")
    wide = fixture.pivot(index="estimand", columns="implementation")
    pooled_se = 0.5 * (wide["std_error"]["cleverly"] + wide["std_error"]["tmle3"])
    difference = (wide["estimate"]["cleverly"] - wide["estimate"]["tmle3"]).abs()
    assert (difference <= 0.05 * pooled_se).all(), difference / pooled_se

    # tmle3 exposes the pre-targeting plug-in. At least one target must move enough that
    # agreement cannot be explained by a zero fluctuation.
    moved = (wide["estimate"]["tmle3"] - wide["initial_estimate"]["tmle3"]).abs()
    assert moved.max() > 1e-3


def test_paper_property_verdicts_are_recomputed_from_the_replication_rows() -> None:
    rows = pd.read_csv(ARTIFACTS / "property-replicates.csv.gz")
    actual = pd.read_csv(ARTIFACTS / "properties.csv")
    recomputed = summarize_properties(rows)
    pd.testing.assert_frame_equal(actual, recomputed, check_exact=False, rtol=1e-12, atol=1e-12)
    assert actual["passed"].all(), actual.loc[~actual["passed"]]


def test_native_interval_scales_are_recorded_instead_of_forced_to_match() -> None:
    rows = _rows()
    paf = rows.query("estimand == 'paf'")
    assert set(paf.query("implementation == 'cleverly'")["inference_scale"]) == {"identity"}
    assert set(paf.query("implementation == 'tmle3'")["inference_scale"]) == {
        "negative_log_complement"
    }
    for name in ("rr", "or"):
        assert set(rows.loc[rows["estimand"] == name, "inference_scale"]) == {"log"}


@pytest.mark.parametrize(
    "name", ["summary.csv", "equivalence.csv", "performance-tests.csv", "properties.csv"]
)
def test_reader_facing_csvs_are_nonempty(name: str) -> None:
    assert not pd.read_csv(ARTIFACTS / name).empty

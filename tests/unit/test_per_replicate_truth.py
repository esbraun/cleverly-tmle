"""The per-replicate truth of the evidence framework (RM30, "The per-replicate truth").

A record that sets ``truth_varies_by_replicate`` reads every statistic on the error
``estimate - truth`` of each row.  The witness here is a table whose truth varies much more
than its error, so the SD of the estimate and the SD of the error differ by a factor of about
five.  Each statistic must come out on the error, and a mutation that reads the SD of the
estimate instead moves every number the tests pin.

Every record that leaves the field unset keeps its arithmetic byte for byte.  The fast tier's
recomputation of each registered study from its committed rows is the witness of that half.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studies.evidence.manifest import write_manifest
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.properties import (
    paired_displacement,
    rate,
    ratio_intervals,
    spread_values,
    summarize_cells,
)
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, necessity_verdicts
from tests.studies.evidence.registry import Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, validate_replicates

REPLICATES = 400
N = 2_000
TRUTH_SD = 0.05
ERROR_SD = 0.01


def _record(**changes: object) -> StudyRecord:
    values: dict[str, object] = {
        "name": "per-replicate truth",
        "slug": "per-replicate-truth",
        "artifacts": Path("."),
        "document": "test.md",
        "anchor": "test",
        "scenarios": {"law": ("ey_learned_rule",)},
        "replicates": REPLICATES,
        "n": N,
        "seed": 17,
        "margins": Margins(bootstrap_replicates=500),
        "truth_varies_by_replicate": True,
    }
    values.update(changes)
    return StudyRecord(**values)  # type: ignore[arg-type]


def _draws(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truths that vary five times more than the errors, and SEs equal to the error SD."""
    rng = np.random.default_rng(seed)
    truths = 0.5 + TRUTH_SD * rng.standard_normal(REPLICATES)
    errors = ERROR_SD * rng.standard_normal(REPLICATES)
    std_errors = np.full(REPLICATES, ERROR_SD)
    return truths, truths + errors, std_errors


def _primary_rows(record: StudyRecord) -> pd.DataFrame:
    truths, estimates, std_errors = _draws()
    low, high = estimates - 1.959964 * std_errors, estimates + 1.959964 * std_errors
    frame = pd.DataFrame(
        {
            "implementation": record.implementation,
            "scenario": "law",
            "replicate": np.arange(REPLICATES),
            "n": N,
            "estimand": "ey_learned_rule",
            "truth": truths,
            "estimate": estimates,
            "inference_estimate": estimates,
            "std_error": std_errors,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": ((low <= truths) & (truths <= high)).astype(int),
            "initial_estimate": np.nan,
        }
    )
    return frame.loc[:, list(REPLICATE_COLUMNS)]


def _property_rows(cell: str = "law__correctly_specified", *, n: int = N) -> pd.DataFrame:
    truths, estimates, std_errors = _draws()
    low, high = estimates - 1.959964 * std_errors, estimates + 1.959964 * std_errors
    return pd.DataFrame(
        {
            "property": "interval_calibration",
            "cell": cell,
            "role": "positive",
            "replicate": np.arange(REPLICATES),
            "n": n,
            "requested_replicates": REPLICATES,
            "failed_replicates": 0,
            "truth": truths,
            "estimate": estimates,
            "std_error": std_errors,
            "covered": ((low <= truths) & (truths <= high)).astype(int),
            "rejected": 0,
        }
    )


def _error_ratio() -> tuple[float, float]:
    """``mean SE / SD(error)`` and ``mean SE / SD(estimate)`` of the witness table."""
    truths, estimates, std_errors = _draws()
    return (
        float(std_errors.mean() / np.std(estimates - truths, ddof=1)),
        float(std_errors.mean() / np.std(estimates, ddof=1)),
    )


def test_the_witness_separates_the_two_spreads() -> None:
    """The fixture itself: a statistic that read the estimate would be about five times off."""
    on_error, on_estimate = _error_ratio()
    assert 0.9 < on_error < 1.1
    assert on_estimate < 0.3


class TestTheRecord:
    def test_the_field_defaults_to_one_truth(self) -> None:
        assert not _record(truth_varies_by_replicate=False).truth_varies_by_replicate
        assert not dataclasses.fields(StudyRecord)[-1].default

    def test_a_reference_is_refused_beside_it(self) -> None:
        with pytest.raises(ValueError, match="cannot declare a reference"):
            _record(reference="R")

    def test_only_a_flagged_manifest_names_it(self, tmp_path: Path) -> None:
        for flagged in (True, False):
            path = tmp_path / f"{flagged}.json"
            write_manifest(path, _record(truth_varies_by_replicate=flagged), [])
            configuration = json.loads(path.read_text(encoding="utf-8"))["configuration"]
            assert ("truth_varies_by_replicate" in configuration) is flagged


class TestTheSchema:
    def test_a_varying_truth_passes_only_the_flagged_record(self) -> None:
        record = _record()
        rows = _primary_rows(record)
        validate_replicates(rows, record=record)
        with pytest.raises(ValueError, match="the truth column is joined wrong"):
            validate_replicates(rows, record=_record(truth_varies_by_replicate=False))

    def test_coverage_is_still_checked_against_each_rows_own_truth(self) -> None:
        record = _record()
        rows = _primary_rows(record)
        rows.loc[0, "covered"] = 1 - rows.loc[0, "covered"]
        with pytest.raises(ValueError, match="disagrees with their own interval"):
            validate_replicates(rows, record=record)

    def test_a_scale_other_than_the_level_is_refused(self) -> None:
        record = _record()
        rows = _primary_rows(record)
        rows["inference_scale"] = "log"
        with pytest.raises(ValueError, match="level scale only"):
            validate_replicates(rows, record=record)

    def test_a_non_finite_truth_is_refused(self) -> None:
        record = _record()
        rows = _primary_rows(record)
        rows.loc[3, "truth"] = np.nan
        with pytest.raises(ValueError, match="non-finite truths"):
            validate_replicates(rows, record=record)


class TestThePrimaryStatistics:
    def test_the_summary_reads_the_error(self) -> None:
        record = _record()
        rows = _primary_rows(record)
        truths, estimates, _ = _draws()
        errors = estimates - truths
        row = summarize(rows, truth_varies=True).iloc[0]
        on_error, on_estimate = _error_ratio()
        assert row["se_ratio"] == pytest.approx(on_error, rel=1e-12)
        assert row["se_ratio"] != pytest.approx(on_estimate, rel=0.5)
        assert row["empirical_se"] == pytest.approx(np.std(errors, ddof=1), rel=1e-12)
        assert row["bias"] == pytest.approx(errors.mean(), rel=1e-9, abs=1e-15)
        assert row["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)), rel=1e-12)
        assert row["mean_estimate"] == pytest.approx(estimates.mean(), rel=1e-12)
        assert row["truth"] == pytest.approx(truths.mean(), rel=1e-12)
        assert row["truth_min"] == truths.min()
        assert row["truth_max"] == truths.max()

    def test_only_the_flagged_summary_carries_the_truth_range(self) -> None:
        record = _record()
        rows = _primary_rows(record)
        rows["truth"] = 0.5
        rows["covered"] = ((rows["ci_lower"] <= 0.5) & (rows["ci_upper"] >= 0.5)).astype(int)
        assert "truth_min" not in summarize(rows).columns
        flagged = summarize(rows, truth_varies=True)
        assert list(flagged.columns[5:8]) == ["truth", "truth_min", "truth_max"]

    def test_the_performance_verdicts_read_the_error(self) -> None:
        record = _record()
        rows = _primary_rows(record)
        truths, estimates, _ = _draws()
        verdict = independent_performance_tests(rows, record=record, n_jobs=1).iloc[0]
        on_error, _ = _error_ratio()
        assert verdict["se_ratio"] == pytest.approx(on_error, rel=1e-12)
        assert verdict["se_ratio_ci_lower"] < on_error < verdict["se_ratio_ci_upper"]
        assert verdict["se_calibrated"]
        assert verdict["bias"] == pytest.approx((estimates - truths).mean(), rel=1e-12)
        spread = np.std(estimates - truths, ddof=1)
        assert verdict["bias_margin"] == pytest.approx(0.25 * spread, rel=1e-12)

    def test_the_unflagged_verdict_on_the_same_rows_reads_the_estimate(self) -> None:
        """The deliberate mutation the witness exists for: read the estimate's spread."""
        record = _record()
        rows = _primary_rows(record)
        rows["truth"] = float(rows["truth"].mean())
        rows["covered"] = (
            (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
        ).astype(int)
        mutated = dataclasses.replace(record, truth_varies_by_replicate=False)
        verdict = independent_performance_tests(rows, record=mutated, n_jobs=1).iloc[0]
        _, on_estimate = _error_ratio()
        assert verdict["se_ratio"] == pytest.approx(on_estimate, rel=1e-12)
        assert not verdict["se_calibrated"]


class TestThePropertyStatistics:
    def test_the_cell_summary_reads_the_error(self) -> None:
        truths, estimates, _ = _draws()
        row = summarize_cells(
            _property_rows(), margin=0.25, confidence_level=0.99, alpha=0.05, truth_varies=True
        ).iloc[0]
        on_error, _ = _error_ratio()
        assert row["se_ratio"] == pytest.approx(on_error, rel=1e-12)
        assert row["empirical_se"] == pytest.approx(np.std(estimates - truths, ddof=1), rel=1e-12)
        assert row["truth"] == pytest.approx(truths.mean(), rel=1e-12)
        assert (row["truth_min"], row["truth_max"]) == (truths.min(), truths.max())
        assert row["mean_estimate"] == pytest.approx(estimates.mean(), rel=1e-12)
        assert row["bias_equivalent"]

    def test_the_ratio_bootstrap_resamples_the_error_and_its_se_together(self) -> None:
        on_error, on_estimate = _error_ratio()
        flagged = ratio_intervals(
            _property_rows(),
            replicates=500,
            confidence_level=0.99,
            seed=1,
            bound=ERROR_SD * np.sqrt(N),
            truth_varies=True,
        )
        assert flagged["se_ratio"].contains(on_error)
        assert flagged["efficiency_empirical"].contains(1.0 / on_error)
        mutated = ratio_intervals(_property_rows(), replicates=500, confidence_level=0.99, seed=1)
        assert mutated["se_ratio"].contains(on_estimate)
        assert not mutated["se_ratio"].contains(on_error)

    def test_the_shared_calibration_verdict_reads_the_error(self) -> None:
        summary, _ = apply_shared_verdicts(_property_rows(), _record(), rate_labels=())
        row = summary.iloc[0]
        on_error, on_estimate = _error_ratio()
        assert row["se_ratio"] == pytest.approx(on_error, rel=1e-12)
        assert row["se_ratio_ci_lower"] < on_error < row["se_ratio_ci_upper"]
        mutated = dataclasses.replace(_record(), truth_varies_by_replicate=False)
        rows = _property_rows()
        rows["truth"] = float(rows["truth"].mean())
        row = apply_shared_verdicts(rows, mutated, rate_labels=())[0].iloc[0]
        assert row["se_ratio"] == pytest.approx(on_estimate, rel=1e-12)
        assert row["se_ratio_ci_upper"] < on_error
        assert not bool(row["passed"])

    def test_the_displacement_subtracts_each_arms_own_truth(self) -> None:
        positive = _property_rows("law__positive")
        control = _property_rows("law__control")
        control["role"] = "control"
        # The control's truth sits one error SD above the positive arm's, and its estimate
        # moves with it: each arm is unbiased for its own truth, so the error form reads no
        # displacement where the estimate form reads one error SD.
        control["truth"] += ERROR_SD
        control["estimate"] += ERROR_SD
        rows = pd.concat([positive, control], ignore_index=True)
        rows["property"] = "fold_locality"
        on_error = paired_displacement(
            rows, "fold_locality", "law__positive", "law__control", truth_varies=True
        )
        on_estimate = paired_displacement(rows, "fold_locality", "law__positive", "law__control")
        assert on_error == pytest.approx(0.0, abs=1e-9)
        assert on_estimate == pytest.approx(ERROR_SD / np.std(_draws()[1], ddof=1), rel=1e-9)

    def test_the_necessity_rule_passes_the_flag_to_the_displacement(self) -> None:
        positive = _property_rows("law__positive")
        control = _property_rows("law__control")
        control["role"] = "control"
        control["estimate"] += 10 * ERROR_SD
        control["truth"] += 10 * ERROR_SD
        rows = pd.concat([positive, control], ignore_index=True)
        rows["property"] = "fold_locality"
        summary = summarize_cells(
            rows, margin=0.25, confidence_level=0.99, alpha=0.05, truth_varies=True
        )
        summary["passed"] = False
        summary["property_passed"] = None
        summary["displacement"] = np.nan
        necessity_verdicts(
            summary,
            rows,
            family="fold_locality",
            labels=("law",),
            arms=("positive", "control"),
            column="displacement",
            threshold=0.5,
            truth_varies=True,
        )
        assert summary["displacement"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert not bool(summary["property_passed"].iloc[0])

    def test_the_rate_reads_the_spread_of_the_error(self) -> None:
        frames = []
        for size in (500, 2_000, 8_000):
            frame = _property_rows(f"n_{size}", n=size)
            truths, _, _ = _draws(seed=size)
            errors = (
                ERROR_SD
                * np.sqrt(N / size)
                * np.random.default_rng(size + 1).standard_normal(REPLICATES)
            )
            frame["truth"] = truths
            frame["estimate"] = truths + errors
            frame["property"] = "root_n_and_efficiency"
            frames.append(frame)
        rows = pd.concat(frames, ignore_index=True)
        on_error = rate(
            rows,
            property_name="root_n_and_efficiency",
            bootstrap_replicates=200,
            confidence_level=0.99,
            seed=1,
            truth_varies=True,
        )
        on_estimate = rate(
            rows,
            property_name="root_n_and_efficiency",
            bootstrap_replicates=200,
            confidence_level=0.99,
            seed=1,
        )
        assert on_error.slope == pytest.approx(-0.5, abs=0.05)
        assert on_estimate.slope > -0.1

    def test_the_spread_values_of_one_truth_are_the_estimates_themselves(self) -> None:
        rows = _property_rows()
        assert spread_values(rows, truth_varies=False) is not None
        np.testing.assert_array_equal(
            spread_values(rows, truth_varies=False), rows["estimate"].to_numpy(dtype=float)
        )
        np.testing.assert_array_equal(
            spread_values(rows, truth_varies=True),
            rows["estimate"].to_numpy(dtype=float) - rows["truth"].to_numpy(dtype=float),
        )

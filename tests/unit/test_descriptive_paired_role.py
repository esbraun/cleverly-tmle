"""A descriptive paired comparison (``StudyRecord.paired_role="descriptive"``).

A study declares the role when its two implementations do not compute the same selected
estimate by construction.  The paired endpoints are still published, but nothing reads them as
a verdict: ``equivalence.csv`` has no ``passed`` column, the failure accounting adds no paired
row, and the red-cell ledger lists none of its rows.  Each check below has a gated twin that
must behave the other way, so a check that passes on both roles would fail here.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studies.evidence import claims, failures, red_cells
from tests.studies.evidence.comparison import (
    DESCRIPTIVE_COLUMNS,
    DESCRIPTIVE_CONCLUSION,
    EQUIVALENCE_COLUMNS,
    equivalence,
)
from tests.studies.evidence.document import render
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.registry import Margins, StudyRecord

SUBJECT = "cleverly"
REFERENCE = "R ref"
TRUTH = 1.0
SPREAD = 0.1


def _record(tmp_path: Path, role: str) -> StudyRecord:
    return StudyRecord(
        name="descriptive test",
        slug="descriptive-test",
        artifacts=tmp_path,
        document="test.md",
        anchor="test",
        scenarios={"law": ("ate",)},
        replicates=60,
        n=500,
        seed=11,
        reference=REFERENCE,
        comparison="matching implementation",
        margins=Margins(bootstrap_replicates=200),
        paired_role=role,  # type: ignore[arg-type]
    )


def _rows(record: StudyRecord, *, reference_shift: float) -> pd.DataFrame:
    """Two implementations of one estimator, the reference moved by ``reference_shift`` SDs."""
    rng = np.random.default_rng(5)
    shared = rng.normal(size=record.replicates)
    out = []
    for implementation, shift in ((SUBJECT, 0.0), (REFERENCE, reference_shift)):
        noise = shared + 0.05 * rng.normal(size=record.replicates)
        for replicate in range(record.replicates):
            estimate = TRUTH + SPREAD * (shift + float(noise[replicate]))
            low, high = estimate - 2.576 * SPREAD, estimate + 2.576 * SPREAD
            out.append(
                {
                    "implementation": implementation,
                    "scenario": "law",
                    "replicate": replicate,
                    "n": record.n,
                    "estimand": "ate",
                    "truth": TRUTH,
                    "estimate": estimate,
                    "inference_estimate": estimate,
                    "std_error": SPREAD,
                    "ci_lower": low,
                    "ci_upper": high,
                    "inference_scale": "identity",
                    "covered": bool(low <= TRUTH <= high),
                    "initial_estimate": estimate,
                }
            )
    return pd.DataFrame(out)


def _tables(record: StudyRecord) -> dict[str, pd.DataFrame]:
    """The committed verdict tables of a run whose paired row fails its similarity leg."""
    rows = _rows(record, reference_shift=1.0)
    performance = independent_performance_tests(rows, record=record, n_jobs=1)
    paired = equivalence(rows, summarize(rows), performance, record=record, n_jobs=1)
    properties = pd.DataFrame(columns=["property", "cell", "role", "passed", "property_passed"])
    for name, frame in (
        ("performance-tests.csv", performance),
        ("equivalence.csv", paired),
        ("properties.csv", properties),
    ):
        frame.to_csv(record.artifact(name), index=False)
    return {"performance": performance, "equivalence": paired, "properties": properties}


def test_a_descriptive_row_has_no_verdict_column(tmp_path: Path) -> None:
    record = _record(tmp_path, "descriptive")
    paired = _tables(record)["equivalence"]
    assert list(paired.columns) == list(DESCRIPTIVE_COLUMNS)
    assert "passed" not in paired.columns
    assert set(paired["comparison_conclusion"]) == {DESCRIPTIVE_CONCLUSION}
    # The endpoints are published: the similarity leg fails on this shift.
    assert not bool(paired["paired_similarity"].iloc[0])


def test_the_gated_twin_keeps_its_verdict(tmp_path: Path) -> None:
    record = _record(tmp_path, "gated")
    paired = _tables(record)["equivalence"]
    assert list(paired.columns) == list(EQUIVALENCE_COLUMNS)
    assert not bool(paired["passed"].iloc[0])


def test_the_ledger_reads_no_descriptive_row(tmp_path: Path) -> None:
    record = _record(tmp_path, "descriptive")
    _tables(record)
    tables = red_cells.frames(record)
    assert tables["equivalence"].empty
    assert [row for row in red_cells.red_rows(record, tables) if row.key.kind == "paired"] == []
    assert red_cells.verdict_count(tables) == len(tables["performance"])


def test_the_ledger_reads_the_gated_twin(tmp_path: Path) -> None:
    record = _record(tmp_path, "gated")
    _tables(record)
    tables = red_cells.frames(record)
    paired = [row for row in red_cells.red_rows(record, tables) if row.key.kind == "paired"]
    assert [row.key.key for row in paired] == ["law/ate"]


def test_no_paired_accounting_for_a_descriptive_row(tmp_path: Path) -> None:
    lost = pd.DataFrame(
        [
            {
                "implementation": REFERENCE,
                "group_id": "r000000001",
                "scenarios": "law",
                "replicates": "1",
                "attempt": 1,
                "class": "fitting_failure",
                "stage": "fit",
                "call": "ctmleGlmnet(...)",
                "message": "singular",
                "seed": 1,
            }
        ],
        columns=list(failures.FAILURE_COLUMNS),
    )
    descriptive = _record(tmp_path, "descriptive")
    rows = _rows(descriptive, reference_shift=0.0)
    rows = rows.loc[~((rows["implementation"] == REFERENCE) & (rows["replicate"] == 1))]
    assert failures.paired_accounting(rows, lost, record=descriptive, n_jobs=1).empty
    gated = dataclasses.replace(descriptive, paired_role="gated")
    assert not failures.paired_accounting(rows, lost, record=gated, n_jobs=1).empty


def test_a_descriptive_record_needs_a_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a reference"):
        dataclasses.replace(_record(tmp_path, "descriptive"), reference=None, comparison=None)


def test_a_descriptive_record_has_no_pass_count(tmp_path: Path) -> None:
    descriptive = _record(tmp_path, "descriptive")
    assert "paired_tests_passed" not in claims.quantities(descriptive)
    assert "paired_tests_passed" in claims.quantities(_record(tmp_path, "gated"))


# --------------------------------------------------------------- zero compared replicates


def _summary(compared: int) -> dict[str, pd.DataFrame]:
    statistic = math.nan if compared == 0 else 0.0123
    return {
        claims.SELECTION_SUMMARY: pd.DataFrame(
            [
                {
                    "scenario": "binary_lasso",
                    "replicates": 800,
                    "adapter_replicates": 800 - compared,
                    "compared_replicates": compared,
                    "confidence_level": 0.99,
                    "mean_difference": statistic,
                    "sd_difference": statistic,
                    "ci_lower": statistic,
                    "ci_upper": statistic,
                    "max_abs_difference": statistic,
                    "rmse_cleverly": statistic,
                    "rmse_reference": statistic,
                }
            ]
        )
    }


@pytest.mark.parametrize("column", sorted(claims.COMPARED_STATISTICS))
def test_zero_compared_replicates_print_no_number(tmp_path: Path, column: str) -> None:
    record = _record(tmp_path, "descriptive")
    name = f"{claims.SELECTION_SUMMARY}[binary_lasso]:{column}"
    assert claims.citation(record, name, _summary(0), render) == claims.NO_COMPARISON
    # The control: one compared replicate prints its number.
    assert claims.citation(record, name, _summary(1), render) == "0.0123"


def test_the_counts_print_at_zero(tmp_path: Path) -> None:
    record = _record(tmp_path, "descriptive")
    data = _summary(0)
    for column, expected in (("compared_replicates", "0"), ("adapter_replicates", "800")):
        name = f"{claims.SELECTION_SUMMARY}[binary_lasso]:{column}"
        assert claims.citation(record, name, data, render) == expected

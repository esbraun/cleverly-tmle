"""The Fix-6 owner moves, tied to the committed results they rest on.

Each owner and support-row change of Fix-6 reads a declared rule over committed artefacts: the
``RM18-one-sided-bias`` acceptance over ``drtmle-foldwise-n1500``, row ``D12`` over
``multi-arm-drtmle-foldwise``, the ``X8-drtmle-one-sided-bias`` acceptance over the stratified
ladder, and the S1 reading of the default's ``ey[1][V=2]`` row.  If a regeneration moves any of
those readings, the owner text and the dispositions are stale, and these tests say which.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.studies import drtmle_foldwise_n1500 as unit_b
from tests.studies import multi_arm_drtmle_foldwise as unit_m
from tests.studies import stratified_drtmle_foldwise as unit_s
from tests.studies import stratified_drtmle_foldwise_ladder as ladder
from tests.studies.evidence.red_cells import OWNERS, RedKey
from tests.studies.evidence.support_dispositions import (
    DISPOSITIONS,
    EXPERIMENTAL,
    FLAGGED_EXPERIMENTAL,
    LIMIT_OWNERS,
)


def _csv(module: object, name: str) -> pd.DataFrame:
    return pd.read_csv(module.STUDY.artifact(name))  # type: ignore[attr-defined]


def test_the_rm18_acceptance_holds_on_the_committed_rows() -> None:
    """Both paired cells pass, with the record's primary rows passing (D2, R-a)."""
    cells = _csv(unit_b, "properties.csv")
    assert set(cells["cell"]) == set(unit_b.SCENARIOS)
    assert cells["passed"].all() and cells["property_passed"].all()
    assert _csv(unit_b, "performance-tests.csv")["passed"].all()
    assert "RM18-one-sided-bias" in LIMIT_OWNERS
    assert "RM18-one-sided-bias" not in FLAGGED_EXPERIMENTAL


def test_the_multi_arm_record_is_green_so_d12_reads_supported() -> None:
    assert _csv(unit_m, "performance-tests.csv")["passed"].all()
    cells = _csv(unit_m, "properties.csv")
    assert cells["passed"].all() and cells["property_passed"].all()
    assert not [d for d in DISPOSITIONS.values() if d.row == "D12"]


def test_the_x8_acceptance_holds_on_the_committed_ladder() -> None:
    assert ladder.x8_acceptance(_csv(unit_s, "properties.csv")) == ladder.ACCEPTED
    assert "X8-drtmle-one-sided-bias" in LIMIT_OWNERS
    assert "X8-drtmle-one-sided-bias" not in FLAGGED_EXPERIMENTAL


def test_the_small_stratum_row_reads_defect_suspect_and_d13_marks_it_experimental() -> None:
    """S1: the red row fails the finite-sample rule, so it is no stated limit."""
    performance = _csv(unit_s, "performance-tests.csv")
    (row,) = performance.loc[performance["estimand"] == "ey[1][V=2]"].to_dict("records")
    assert unit_s.small_stratum_reading(row) == unit_s.SMALL_STRATUM_DEFECT
    others = performance.loc[performance["estimand"] != "ey[1][V=2]"]
    assert others["passed"].all()
    key = RedKey(
        "stratified-drtmle-foldwise",
        "truth",
        f"{unit_s.STUDY.implementation}/stratified_both_correct/ey[1][V=2]",
    )
    assert OWNERS[key] == unit_s.SMALL_STRATUM_OWNER
    disposition = DISPOSITIONS[key]
    assert (disposition.row, disposition.kinds) == ("D13", ("wald",))
    assert disposition.disposition == EXPERIMENTAL
    assert disposition.limit == ""


@pytest.mark.parametrize(
    ("row", "finite"),
    [
        pytest.param(
            {
                "passed": False,
                "bias_equivalent": True,
                "coverage_ci_lower": 0.89,
                "coverage_ci_upper": 0.92,
                "se_ratio_ci_lower": 0.95,
                "se_ratio_ci_upper": 1.03,
            },
            True,
            id="se-ratio-contains-one",
        ),
        pytest.param(
            {
                "passed": False,
                "bias_equivalent": True,
                "coverage_ci_lower": 0.89,
                "coverage_ci_upper": 0.92,
                "se_ratio_ci_lower": 0.86,
                "se_ratio_ci_upper": 0.91,
            },
            False,
            id="se-ratio-excludes-one",
        ),
    ],
)
def test_control_the_reading_separates_the_two_branches(row: dict, finite: bool) -> None:
    """A reading that always said defect suspect would pass the committed test above alone."""
    expected = unit_s.SMALL_STRATUM_FINITE_SAMPLE if finite else unit_s.SMALL_STRATUM_DEFECT
    assert unit_s.small_stratum_reading(row) == expected

"""Named regeneration gate for canonical TMLE's repeated-sampling properties."""

from __future__ import annotations

import pytest

from tests.studies.canonical_properties import generate_property_rows, summarize_properties

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def study():  # type: ignore[no-untyped-def]
    return summarize_properties(generate_property_rows())


def test_double_robustness_has_three_positive_cells_and_a_negative_control(study) -> None:  # type: ignore[no-untyped-def]
    rows = study.query("property == 'double_robustness'")
    assert set(rows["cell"]) == {
        "both_correct",
        "outcome_correct",
        "treatment_correct",
        "both_wrong",
    }
    assert rows["passed"].all(), rows.loc[~rows["passed"]]


def test_root_n_rate_efficiency_and_interval_calibration(study) -> None:  # type: ignore[no-untyped-def]
    rows = study.query("property == 'root_n_and_efficiency'")
    assert set(rows["cell"]) == {"n_500", "n_2000"}
    assert rows["passed"].all(), rows.loc[~rows["passed"]]


def test_type_i_error_is_calibrated_under_the_sharp_null(study) -> None:  # type: ignore[no-untyped-def]
    rows = study.query("property == 'type_i_error'")
    assert len(rows) == 1
    assert rows["passed"].all(), rows

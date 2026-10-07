"""The readings of the red cells of ``longitudinal-mtp``, owner ``mtp-longitudinal-limits``.

Each test rebuilds a reading's numbers from the committed rows, so the reading cannot drift from
the evidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.studies import canonical_longitudinal_mtp as study
from tests.studies.evidence.property_verdicts import OVERFIT_SE_CONTROL_CEILING

pytestmark = pytest.mark.xdist_group("longitudinal_mtp_design")

#: The cells the owner holds.
OWNED = {
    ("interval_calibration", "categorical_mtp__correctly_specified"),
    ("crossfit_overfitting", "in_sample_control"),
}
#: A record, not a check: the categorical efficiency ratio with saturated learners, at
#: n = 2,000 and n = 8,000, 300 draws each (``_x12_s2_diag.py`` beside the plan).
CATEGORICAL_EFFICIENCY = {2_000: 1.066, 8_000: 1.024}


def _summary() -> pd.DataFrame:
    return pd.read_csv(study.STUDY.artifact("properties.csv")).set_index(["property", "cell"])


def test_the_owned_cells_are_the_red_ones() -> None:
    summary = _summary()
    red = {key for key, passed in summary["passed"].items() if not passed}
    assert red == OWNED


def test_the_categorical_cell_reads_as_finite_sample() -> None:
    """Coverage passes, the efficiency ratios sit just above the band, and they shrink with n."""
    cell = _summary().loc[("interval_calibration", "categorical_mtp__correctly_specified")]
    assert float(cell["coverage_ci_lower"]) >= 0.92 and float(cell["coverage_ci_upper"]) <= 0.98
    assert 1.05 < float(cell["efficiency_reported_ratio"]) < 1.15
    assert CATEGORICAL_EFFICIENCY[8_000] < CATEGORICAL_EFFICIENCY[2_000] < 1.1


def test_the_in_sample_control_sits_at_its_ceiling() -> None:
    """The control is anti-conservative, as it must be, but its interval reaches the ceiling."""
    summary = _summary()
    control = summary.loc[("crossfit_overfitting", "in_sample_control")]
    positive = summary.loc[("crossfit_overfitting", "cross_fitted_mtp_ltmle")]
    assert float(control["se_ratio"]) < OVERFIT_SE_CONTROL_CEILING
    assert float(control["se_ratio_ci_upper"]) > OVERFIT_SE_CONTROL_CEILING
    assert float(positive["se_ratio"]) - float(control["se_ratio"]) > 0.3
    assert bool(positive["passed"])

"""The reading of the red cell of ``longitudinal-mtp``, owner ``mtp-longitudinal-limits``.

Each test rebuilds a reading's numbers from committed rows, so the reading cannot drift from the
evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics.longitudinal_mtp_categorical_efficiency import run as diagnostic
from tests.studies import canonical_longitudinal_mtp as study
from tests.studies import longitudinal_mtp_properties as properties
from tests.studies.evidence.property_verdicts import (
    OVERFIT_COVERAGE_GAIN,
    OVERFIT_SE_CONTROL_CEILING,
)

pytestmark = pytest.mark.xdist_group("longitudinal_mtp_design")

#: The cells the owner holds.
OWNED = {("interval_calibration", "categorical_mtp__correctly_specified")}
CATEGORICAL = ("interval_calibration", "categorical_mtp__correctly_specified")


def _summary() -> pd.DataFrame:
    return pd.read_csv(study.STUDY.artifact("properties.csv")).set_index(["property", "cell"])


def _reported_efficiency(arm: str, n: int) -> float:
    rows = pd.read_csv(diagnostic.HERE / "rows.csv.gz", float_precision="round_trip")
    chosen = rows.loc[(rows["arm"] == arm) & (rows["n"] == n)]
    assert len(chosen) == diagnostic.REPLICATES
    bound = properties.EFFICIENCY_SD["categorical_mtp"]
    return float(chosen["std_error"].mean() * np.sqrt(n) / bound)


def test_the_owned_cells_are_the_red_ones() -> None:
    summary = _summary()
    red = {key for key, passed in summary["property_passed"].items() if not passed}
    assert red == OWNED


def test_the_categorical_cell_fails_on_efficiency_alone() -> None:
    """Coverage and the SE ratio pass; the efficiency ratios sit just above the band."""
    cell = _summary().loc[CATEGORICAL]
    assert float(cell["coverage_ci_lower"]) >= 0.92 and float(cell["coverage_ci_upper"]) <= 0.98
    assert 0.93 < float(cell["se_ratio_ci_lower"]) < 1.0 < float(cell["se_ratio_ci_upper"]) < 1.07
    assert 1.05 < float(cell["efficiency_reported_ratio"]) < 1.15


def test_the_efficiency_excess_shrinks_with_n_on_fresh_draws() -> None:
    """A wrong bound or an inefficient curve keeps its excess; the reported excess falls as 1/n.

    The reported ratio's excess over one falls by more than three times from n = 2,000 to
    n = 8,000, in sample and at five folds, and the in-sample ratio at n = 2,000 matches the
    registered cell.  The empirical ratio is too noisy at 300 draws to show the same alone.  The
    saturated mechanism makes the fit the NPMLE whatever the outcome learner, so the estimates,
    and their spread, are those of an efficient estimator.
    """
    registered = float(_summary().loc[CATEGORICAL, "efficiency_reported_ratio"])
    for arm in diagnostic.FOLDS:
        small, large = (_reported_efficiency(arm, n) for n in diagnostic.SIZES)
        assert large < small
        assert (small - 1.0) > 3.0 * (large - 1.0)
    assert abs(_reported_efficiency("in_sample", 2_000) - registered) < 0.01


def test_the_redesigned_overfitting_pair_passes() -> None:
    """Run 3's interpolating ensemble: the control clears its ceiling and the gain its floor."""
    summary = _summary()
    control = summary.loc[("crossfit_overfitting", "in_sample_control")]
    positive = summary.loc[("crossfit_overfitting", "cross_fitted_mtp_ltmle")]
    assert float(control["se_ratio_ci_upper"]) < OVERFIT_SE_CONTROL_CEILING
    assert float(control["coverage_gain_ci_lower"]) > OVERFIT_COVERAGE_GAIN
    assert bool(positive["property_passed"]) and bool(control["property_passed"])

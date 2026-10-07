"""The readings of the red cells of ``longitudinal-mtp``.

Two owners hold them.  ``binned-density-tail`` holds the cells that read a fixed-bin density
ratio in its tails; ``mtp-longitudinal-finite-sample`` holds two cells that read as
finite-sample limits.  Each test rebuilds a reading's numbers from the committed rows, or
from a probe on the law, so the reading cannot drift from the evidence.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_longitudinal_mtp as study
from tests.studies import longitudinal_mtp_common as common
from tests.studies import longitudinal_mtp_properties as properties
from tests.studies.evidence.property_verdicts import OVERFIT_SE_CONTROL_CEILING
from tests.studies.evidence.registry import Margins

pytestmark = pytest.mark.xdist_group("longitudinal_mtp_design")

#: The cells ``binned-density-tail`` owns.
TAIL_CELLS = (
    ("interval_calibration", "up__correctly_specified"),
    ("interval_calibration", "msm_mtp__correctly_specified"),
    ("interval_calibration", "randomized_mtp__correctly_specified"),
    ("crossfit_overfitting", "cross_fitted_mtp_ltmle"),
    ("crossfit_overfitting", "in_sample_control"),
)
#: The cells ``mtp-longitudinal-finite-sample`` owns.
FINITE_SAMPLE_CELLS = (
    ("interval_calibration", "categorical_mtp__correctly_specified"),
    ("root_n_and_efficiency", "up__n_500"),
)
#: The probe the tail reading quotes: one draw of 4,000 rows at seed 777.
PROBE_N, PROBE_SEED = 4_000, 777
#: A record, not a check: the categorical efficiency ratio with saturated learners, at
#: n = 2,000 and n = 8,000, 300 draws each (``_x12_s2_diag.py`` beside the plan).
CATEGORICAL_EFFICIENCY = {2_000: 1.066, 8_000: 1.024}


def _summary() -> pd.DataFrame:
    return pd.read_csv(study.STUDY.artifact("properties.csv")).set_index(["property", "cell"])


def _rows() -> pd.DataFrame:
    return pd.read_csv(
        study.STUDY.artifact("property-replicates.csv.gz"), float_precision="round_trip"
    )


def test_every_owned_cell_is_red_and_every_other_cell_passes() -> None:
    summary = _summary()
    owned = set(TAIL_CELLS) | set(FINITE_SAMPLE_CELLS)
    red = {key for key, passed in summary["passed"].items() if not passed}
    assert red == owned


@pytest.mark.parametrize(("family", "cell"), TAIL_CELLS[:3])
def test_the_tail_cells_are_conservative_and_unbiased(family: str, cell: str) -> None:
    """Rebuilt from the replication rows: coverage at or above nominal, and no bias."""
    rows = _rows()
    selected = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
    error = selected["estimate"] - selected["truth"]
    covered = selected["covered"].astype(bool)
    se_ratio = float(selected["std_error"].mean() / error.std(ddof=1))
    assert float(covered.mean()) >= 0.95
    assert se_ratio > 1.0
    assert abs(float(error.mean())) / float(error.std(ddof=1)) < Margins().standardized_bias
    summary = _summary().loc[(family, cell)]
    assert float(summary["coverage"]) == pytest.approx(float(covered.mean()), abs=1e-12)
    assert float(summary["efficiency_reported_ratio"]) > 1.05


def test_the_cross_fitted_tree_cells_read_the_same_inflation() -> None:
    """The positive arm's SE ratio is high, and the control's is high enough to miss its ceiling."""
    summary = _summary()
    positive = summary.loc[("crossfit_overfitting", "cross_fitted_mtp_ltmle")]
    control = summary.loc[("crossfit_overfitting", "in_sample_control")]
    assert float(positive["se_ratio"]) > 1.0 and float(positive["coverage"]) > 0.95
    assert float(control["se_ratio_ci_upper"]) > OVERFIT_SE_CONTROL_CEILING
    assert float(control["se_ratio"]) < float(positive["se_ratio"])


def test_the_binned_ratio_inflates_the_curve_on_a_probe() -> None:
    """At the study's 80 bins the tail ratio errs badly, and the curve's spread grows with it."""
    frame = common.sample_continuous(PROBE_N, PROBE_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = properties.continuous_fit(frame, "both_correct")
    curve = np.asarray(result.influence_curves[properties.UP], dtype=float)
    efficient = common.continuous_eif("up", frame) - common.continuous_eif("natural", frame)
    assert float(np.std(curve)) / float(np.std(efficient)) > 1.15
    w, a1, l2, a2 = (frame[c].to_numpy(dtype=float) for c in ("W", "A1", "L2", "A2"))
    true2 = common._node_ratio("up", 2, a2, common.mean2(w, a1, l2), a1)
    binned2 = np.asarray(result.fits["up"].node_ratio, dtype=float)[:, 1]
    assert float(np.max(np.abs(binned2 - true2))) > 2.0
    assert study.DENSITY_BINS == 80


def test_the_finite_sample_cells_read_as_limits_of_their_size() -> None:
    summary = _summary()
    categorical = summary.loc[("interval_calibration", "categorical_mtp__correctly_specified")]
    assert float(categorical["coverage"]) == pytest.approx(0.9435, abs=5e-5)
    assert CATEGORICAL_EFFICIENCY[8_000] < CATEGORICAL_EFFICIENCY[2_000]
    rung = summary.loc[("root_n_and_efficiency", "up__n_500")]
    assert float(rung["coverage_ci_lower"]) < Margins().coverage_floor < float(rung["coverage"])

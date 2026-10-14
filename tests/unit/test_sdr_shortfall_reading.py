"""The reading of the red cells of ``longitudinal-sdr``, owners ``X4-finite-sample`` and
``band-finite-sample``.

Each test rebuilds a reading's numbers from committed rows, so the reading cannot drift from the
evidence.  ``tests/unit/test_band_shortfall_reading.py`` rebuilds the band reading with the other
red bands.
"""

from __future__ import annotations

import pandas as pd

from tests.diagnostics.x4_both_correct import run as diagnostic
from tests.studies import canonical_longitudinal_sdr as study

#: The cells the two owners hold.
OWNED = {
    ("sequential_double_robustness", "both_correct__n_2000"),
    ("simultaneous_coverage", "six_regimens__simultaneous_band"),
}
CELL = ("sequential_double_robustness", "both_correct__n_2000")


def _summary() -> pd.DataFrame:
    return pd.read_csv(study.STUDY.artifact("properties.csv")).set_index(["property", "cell"])


def test_the_owned_cells_are_the_red_ones() -> None:
    summary = _summary()
    red = {key for key, passed in summary["property_passed"].items() if not passed}
    assert red == OWNED


def test_the_both_correct_cell_fails_on_coverage_alone() -> None:
    """The bias is inside its margin and the SE ratio inside the union-model band."""
    summary = _summary()
    cell = summary.loc[CELL]
    assert bool(cell["bias_equivalent"])
    assert round(float(cell["coverage"]), 3) == 0.911
    assert round(float(cell["coverage_ci_lower"]), 3) == 0.885 < 0.90
    assert round(float(cell["se_ratio"]), 3) == 0.946
    larger = summary.loc[("sequential_double_robustness", "both_correct__n_8000")]
    assert round(float(larger["coverage"]), 3) == 0.950
    assert round(float(larger["se_ratio"]), 3) == 0.999


def test_ltmle_shares_the_shortfall_on_the_same_draws() -> None:
    """``SDR`` and ``LTMLE`` cover equally on 1,000 fresh draws, so the defect is not SDR's.

    Both standard errors vary with the error they measure, which a Wald interval does not
    allow for at n = 2,000.
    """
    rows = pd.read_csv(diagnostic.HERE / "rows.csv.gz", float_precision="round_trip")
    assert len(rows) == diagnostic.REPLICATES
    reading = diagnostic.reading(rows).set_index("engine")
    committed = pd.read_csv(diagnostic.HERE / "reading.csv").set_index("engine")
    pd.testing.assert_frame_equal(reading, committed, check_exact=False, rtol=1e-5)
    rounded = reading.round(3)
    assert rounded.loc["sdr", "coverage"] == rounded.loc["ltmle", "coverage"] == 0.903
    assert rounded.loc["sdr", "se_ratio"] == 0.918
    assert rounded.loc["ltmle", "se_ratio"] == 0.917
    assert rounded.loc["sdr", "se_cv"] == 0.186
    assert rounded.loc["ltmle", "se_cv"] == 0.187
    assert rounded.loc["sdr", "error_se_correlation"] == 0.301
    assert rounded.loc["ltmle", "error_se_correlation"] == 0.321

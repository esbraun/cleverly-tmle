"""The reading of the red cell of ``longitudinal-sdr``, owner ``X39-below-tail-size``.

``both_correct__n_2000`` is a reporting cell: its n of 2,000 is below the exact-law tail size of
its contrast, so its coverage clause reports (``BELOW_TAIL_SIZE`` of
``tests/studies/longitudinal_sdr_properties.py``).  Each test rebuilds a reading's numbers from
committed rows, so the reading cannot drift from the evidence.
"""

from __future__ import annotations

import pandas as pd

from tests.diagnostics.x4_both_correct import run as diagnostic
from tests.studies import canonical_longitudinal_sdr as study
from tests.studies import longitudinal_sdr_properties as properties

#: The cells the owner holds.
OWNED = {("sequential_double_robustness", "both_correct__n_2000")}
CELL = ("sequential_double_robustness", "both_correct__n_2000")


def _summary() -> pd.DataFrame:
    return pd.read_csv(study.STUDY.artifact("properties.csv")).set_index(["property", "cell"])


def _rows() -> pd.DataFrame:
    rows = pd.read_csv(diagnostic.HERE / "rows.csv.gz", float_precision="round_trip")
    assert len(rows) == diagnostic.REPLICATES
    return rows


def test_the_owned_cells_are_the_red_ones() -> None:
    summary = _summary()
    red = {key for key, passed in summary["property_passed"].items() if not passed}
    assert red == OWNED


def test_the_both_correct_cell_fails_on_coverage_alone() -> None:
    """The bias is inside its margin and the SE ratio inside the union-model band."""
    assert "/".join(CELL) in properties.BELOW_TAIL_SIZE
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
    allow for below the tail size.
    """
    rows = _rows()
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


def test_the_strata_of_the_always_path_cell_rebuild() -> None:
    """Coverage falls in the draws where the always-path cell carries no within-cell variation.

    The split redraws each sample from its seed, so a change to the law or the sampler fails it.
    """
    rows = _rows()
    split = diagnostic.strata(rows).set_index("stratum")
    committed = pd.read_csv(diagnostic.HERE / "strata.csv").set_index("stratum")
    pd.testing.assert_frame_equal(split, committed, check_exact=False, rtol=1e-5)
    assert tuple(split.index) == diagnostic.STRATA
    assert int(split["draws"].sum()) == diagnostic.REPLICATES
    assert split["draws"].to_dict() == {
        "empty": 50,
        "one row": 149,
        "two or more, all Y = 1": 17,
        "two or more, all Y = 0": 312,
        "mixed": 472,
    }
    coverage = split["sdr_coverage"].round(3)
    assert coverage["mixed"] == 0.979
    assert coverage.drop("mixed").max() == 0.860 < 0.90


def test_a_draw_is_assigned_by_what_the_always_path_cell_holds() -> None:
    """A nonzero witness for each branch of :func:`stratum`."""
    path = {**diagnostic.ALWAYS_PATH}
    other = {**path, "W": 0.0}

    def frame(*outcomes: float, extra: int = 1) -> pd.DataFrame:
        records = [{**path, "Y": y} for y in outcomes] + [{**other, "Y": 1.0}] * extra
        return pd.DataFrame(records, columns=[*path, "Y"])

    assert diagnostic.stratum(frame()) == "empty"
    assert diagnostic.stratum(frame(0.0)) == "one row"
    assert diagnostic.stratum(frame(1.0, 1.0)) == "two or more, all Y = 1"
    assert diagnostic.stratum(frame(0.0, 0.0, 0.0)) == "two or more, all Y = 0"
    assert diagnostic.stratum(frame(0.0, 1.0)) == "mixed"

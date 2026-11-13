"""The heavy-ratio diagnostic's files, rebuilt from the artifacts and the code that made them.

``tests/diagnostics/fix5_heavy_ratios/run.py`` writes ``reading.csv`` from the committed
replications of the pricing and boundaries studies, ``quadrature.csv`` from the law-B
quadrature, and keeps ``probe_variance.csv``, the design probe's policy-weighted variances.
This module rebuilds ``reading.csv`` from the parquets, checks the numbers the diagnosis cites,
and checks the other two files against their own rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics.fix5_heavy_ratios import run
from tests.studies import mtp_tail_size_common as tail

HERE = Path(run.__file__).resolve().parent


def _reading() -> pd.DataFrame:
    return pd.read_csv(HERE / "reading.csv")


def test_the_reading_rebuilds_from_the_committed_replications() -> None:
    committed = _reading()
    rebuilt = run.committed_reading()
    assert list(committed.columns) == list(rebuilt.columns)
    for column in committed.columns:
        if not pd.api.types.is_numeric_dtype(committed[column]):
            assert committed[column].tolist() == rebuilt[column].tolist()
        else:
            np.testing.assert_allclose(committed[column], rebuilt[column], rtol=1e-5)


@pytest.mark.parametrize(
    ("cell", "estimand", "coverage", "se_ratio", "rms_ratio", "correlation"),
    [
        ("pricing_ltmle_tau5", "ey_regimen[price_down]", 0.917, 0.948, 0.983, -0.76),
        ("pricing_sdr_tau5", "ey_regimen[price_down]", 0.923, 0.952, 0.987, -0.76),
        (
            "pricing_ltmle_tau5",
            "ate_regimen[price_down vs natural_prices]",
            0.916,
            0.931,
            0.965,
            -0.75,
        ),
        ("p4_sdr__correctly_specified", "interval_calibration", 0.936, 0.941, 0.978, 0.12),
    ],
)
def test_the_cited_readings(
    cell: str, estimand: str, coverage: float, se_ratio: float, rms_ratio: float, correlation: float
) -> None:
    reading = _reading().set_index(["cell", "estimand"]).loc[(cell, estimand)]
    assert reading["coverage"] == pytest.approx(coverage, abs=5e-4)
    assert reading["se_ratio"] == pytest.approx(se_ratio, abs=5e-4)
    assert reading["rms_ratio"] == pytest.approx(rms_ratio, abs=5e-4)
    assert reading["error_se_correlation"] == pytest.approx(correlation, abs=5e-3)


def test_the_pricing_misses_are_one_sided_and_tied_to_the_se() -> None:
    reading = _reading()
    down = reading[reading["estimand"].str.contains("price_down")]
    assert (down["miss_above"] > 10 * down["miss_below"]).all()
    assert (down["coverage_q1"] < 0.75).all()
    assert (down["error_se_correlation"] < -0.7).all()
    up = reading[reading["estimand"] == "ey_regimen[price_up]"]
    assert (up["error_se_correlation"].abs() < 0.1).all()


def test_the_rms_ratio_stays_near_one_where_the_se_ratio_does_not() -> None:
    """The variance is not biased on average: the gap is the Jensen gap of a dispersed SE."""
    reading = _reading()
    jensen = 1.0 - reading["se_cv"] ** 2 / 2.0
    np.testing.assert_allclose(reading["se_ratio"] / reading["rms_ratio"], jensen, atol=0.01)


def test_the_quadrature_table_follows_its_rules() -> None:
    table = pd.read_csv(HERE / "quadrature.csv")
    assert set(zip(table["plan"], table["quantity"], table["order"], strict=True)) == set(
        run.QUADRATURE
    )
    np.testing.assert_allclose(table["mass"], 1.0, atol=1e-10)
    for row in table.itertuples():
        assert row.n_kappa == tail.tail_size(float(row.kappa))
    wide = table.pivot_table(index=["plan", "quantity"], columns="order", values="kappa")
    np.testing.assert_allclose(wide[40], wide[80], rtol=1e-8)
    p4 = table[(table["plan"] == "p4") & (table["order"] == 40)].set_index("quantity")
    assert p4.loc["ate", "kappa"] == pytest.approx(3260.86, abs=0.01)
    assert p4.loc["ey", "kappa"] == pytest.approx(2764.97, abs=0.01)


def test_the_diagonal_form_is_exact_on_every_plan() -> None:
    table = pd.read_csv(HERE / "diagonal.csv")
    assert set(zip(table["plan"], table["right"], strict=True)) == {
        (plan, right) for plan in ("p1", "p2", "p3", "p4") for right in (plan, "natural")
    }
    np.testing.assert_allclose(table["form"], table["moments"], rtol=1e-8)
    p4 = table[(table["plan"] == "p4") & (table["right"] == "p4")]
    assert float(p4["form"].iloc[0]) == pytest.approx(3.1357, abs=1e-4)


def test_one_quadrature_row_recomputes() -> None:
    table = pd.read_csv(HERE / "quadrature.csv").set_index(["plan", "quantity", "order"])
    moments = tail.law_b_moments("p4", "ate", 40)
    assert moments.kappa == pytest.approx(table.loc[("p4", "ate", 40), "kappa"], rel=1e-9)


def test_the_probe_file_holds_the_declared_draws() -> None:
    probe = pd.read_csv(HERE / "probe_variance.csv")
    expected = {(engine, rep) for engine in ("ltmle", "sdr") for rep in range(24)}
    assert set(zip(probe["est"], probe["rep"], strict=True)) == expected
    assert (probe["n"] == 5_000).all()
    assert (probe["variance"] > 0.0).all()

r"""The reading of the cross-fitted plug-in omitted-variable bound diagnostic (X26).

The diagnostic is ``tests/diagnostics/x26_crossfit_bias``.  On the binary law of the registered
study (``cf_y = cf_d = 0.5``, ``rho = 1``) it fitted each of 2,000 draws at ``n = 1000`` and
1,000 draws at ``n = 4000`` with five-fold cross-fitting and in sample.  Each row records the
raw plug-in :math:`\nu^2`, the second-order corrected :math:`\nu^2`, and the corrected bound
ends, each with its standard error, and the refusal key of the package.  Each test rebuilds a
number of the reading from the committed rows, so the reading cannot drift from the evidence.

The measured headline, for the cross-fitted ATE at ``n = 1000``:

=============  ========  =========  ========  =======
row            coverage  bias, SD   SE ratio  refused
=============  ========  =========  ========  =======
raw nu2        0.8845    +1.441     1.079     0
corrected nu2  0.8119    -0.072     0.732     6
lower bound    0.9564    -0.068     1.007     6
upper bound    0.9554    +0.032     1.012     6
=============  ========  =========  ========  =======

The correction removes the bias of the raw :math:`\nu^2`.  It also exposes that the corrected
:math:`\nu^2` standard error is too small at ``n = 1000``.  The bound rows, which the package
reports, cover near nominal in every cell.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.diagnostics.x26_crossfit_bias import run as diagnostic
from tests.studies import omitted_variable_bound_plugin as study
from tests.studies.evidence.seeds import replicate_seed


def _rows() -> pd.DataFrame:
    return pd.read_csv(diagnostic.HERE / "rows.csv.gz", float_precision="round_trip")


def _reading() -> pd.DataFrame:
    return diagnostic.reading(_rows())


def _cell(reading: pd.DataFrame, n: int, fitting: str, parameter: str, row: str) -> pd.Series:
    chosen = reading.loc[
        (reading["n"] == n)
        & (reading["fitting"] == fitting)
        & (reading["parameter"] == parameter)
        & (reading["row"] == row)
    ]
    assert len(chosen) == 1
    return chosen.iloc[0]


def test_the_rows_are_the_declared_design() -> None:
    """2,000 draws at n = 1000 and 1,000 at n = 4000, each fitted twice, three parameters."""
    rows = _rows()
    assert len(rows) == (2_000 + 1_000) * 2 * 3
    for n, count in diagnostic.ARMS.items():
        arm = rows.loc[rows["n"] == n]
        assert arm["replicate"].nunique() == count
        assert set(arm["fitting"]) == {"crossfit", "insample"}
        assert set(arm["parameter"]) == {"ate", "att", "atc"}
    truth = study.truths(diagnostic.SCENARIO)
    for token in diagnostic.PARAMETERS:
        chosen = rows.loc[rows["parameter"] == token]
        for name in diagnostic.ROWS:
            expected = truth[study.estimand_key(token, diagnostic.TRUTH_OF[name])]
            assert chosen[f"{name}_truth"].to_numpy() == pytest.approx(expected, rel=1e-12)


def test_the_seed_stream_is_not_the_registered_study() -> None:
    """No diagnostic seed equals a registered replicate seed of any scenario."""
    ours = {diagnostic.seed_of(n, k) for n in diagnostic.ARMS for k in range(50)}
    assert len(ours) == 100
    registered = {
        replicate_seed(study.STUDY, scenario, k)
        for scenario in study.STUDY.scenarios
        for k in range(50)
    }
    assert not ours & registered


def test_the_reading_is_rebuilt_from_the_rows() -> None:
    """The committed reading equals the reading recomputed from the committed rows."""
    committed = pd.read_csv(diagnostic.HERE / "reading.csv")
    rebuilt = _reading()
    assert len(rebuilt) == 2 * 2 * 3 * 4
    pd.testing.assert_frame_equal(rebuilt, committed, check_exact=False, rtol=1e-5, atol=1e-9)


def test_the_raw_cross_fitted_ate_nu2_misses_its_band_at_n_1000() -> None:
    """The raw cross-fitted ATE nu^2 covers 0.8845 and sits 1.441 SD above the truth.

    The 99% Wilson interval, 0.8648 to 0.9017, lies below the calibration floor of 0.92.  The
    SE ratio is 1.079.  The in-sample raw row on the same draws covers 0.969 with bias 0.430 SD.
    """
    reading = _reading()
    cell = _cell(reading, 1_000, "crossfit", "ate", "nu2_raw")
    assert cell["replicates"] == 2_000
    assert cell["refused"] == 0
    assert cell["coverage"] == pytest.approx(0.8845, abs=1e-12)
    assert cell["coverage_lower99"] == pytest.approx(0.864806, abs=1e-6)
    assert cell["coverage_upper99"] == pytest.approx(0.901652, abs=1e-6)
    assert cell["coverage_upper99"] < 0.92
    assert cell["bias_sd"] == pytest.approx(1.44065, abs=1e-5)
    assert cell["se_ratio"] == pytest.approx(1.07923, abs=1e-5)
    insample = _cell(reading, 1_000, "insample", "ate", "nu2_raw")
    assert insample["coverage"] == pytest.approx(0.969, abs=1e-12)
    assert insample["bias_sd"] == pytest.approx(0.429972, abs=1e-5)
    assert abs(insample["bias_sd"]) < cell["bias_sd"] / 3.0


def test_the_raw_bias_shrinks_but_remains_at_n_4000() -> None:
    """At n = 4000 the raw cross-fitted ATE nu^2 bias falls to 0.839 SD and coverage is 0.914.

    The raw ATT and ATC nu^2 fall from 0.995 and 0.973 SD to 0.571 and 0.560 SD.
    """
    reading = _reading()
    small = _cell(reading, 1_000, "crossfit", "ate", "nu2_raw")
    large = _cell(reading, 4_000, "crossfit", "ate", "nu2_raw")
    assert large["replicates"] == 1_000
    assert large["coverage"] == pytest.approx(0.914, abs=1e-12)
    assert large["bias_sd"] == pytest.approx(0.839218, abs=1e-5)
    assert large["se_ratio"] == pytest.approx(1.01252, abs=1e-5)
    assert 0.5 < large["bias_sd"] < small["bias_sd"]
    for token, before, after in (("att", 0.994704, 0.570926), ("atc", 0.972564, 0.560109)):
        assert _cell(reading, 1_000, "crossfit", token, "nu2_raw")["bias_sd"] == pytest.approx(
            before, abs=1e-5
        )
        assert _cell(reading, 4_000, "crossfit", token, "nu2_raw")["bias_sd"] == pytest.approx(
            after, abs=1e-5
        )


def test_the_corrected_nu2_has_no_bias_in_every_cell() -> None:
    """The corrected nu^2 bias is within 0.1 SD in every n, fitting and parameter cell.

    The largest absolute bias is 0.072 SD.  The raw rows carry 0.2 to 1.4 SD in the same cells.
    """
    reading = _reading()
    corrected = reading.loc[reading["row"] == "nu2"]
    assert len(corrected) == 12
    assert corrected["bias_sd"].abs().max() < 0.1
    assert corrected["bias_sd"].abs().max() == pytest.approx(0.0720772, abs=1e-5)
    raw = reading.loc[reading["row"] == "nu2_raw"]
    assert (raw["bias_sd"].to_numpy() > 0.2).all()


def test_the_corrected_nu2_se_ratio_at_each_n() -> None:
    """The corrected nu^2 SE ratio is below one: 0.73 to 0.75 cross-fitted and 0.90 to 0.92 in sample.

    This holds at n = 1000.  At n = 4000 the ratio is 0.95 to 0.96 cross-fitted and 0.96 to 0.98
    in sample, so the corrected nu^2 interval undercovers at n = 1000 (0.81 to 0.90) and covers
    0.926 to 0.934 at n = 4000.  The package reports the bound rows, not this interval.
    """
    reading = _reading()
    for fitting, low, high in (("crossfit", 0.7318, 0.7472), ("insample", 0.8970, 0.9197)):
        ratios = [
            _cell(reading, 1_000, fitting, t, "nu2")["se_ratio"] for t in ("ate", "att", "atc")
        ]
        assert min(ratios) == pytest.approx(low, abs=1e-4)
        assert max(ratios) == pytest.approx(high, abs=1e-4)
    for fitting, low, high in (("crossfit", 0.9473, 0.9608), ("insample", 0.9619, 0.9751)):
        ratios = [
            _cell(reading, 4_000, fitting, t, "nu2")["se_ratio"] for t in ("ate", "att", "atc")
        ]
        assert min(ratios) == pytest.approx(low, abs=1e-4)
        assert max(ratios) == pytest.approx(high, abs=1e-4)
    covers = reading.loc[(reading["row"] == "nu2") & (reading["n"] == 4_000), "coverage"]
    assert covers.min() == pytest.approx(0.926, abs=1e-12)
    assert covers.max() == pytest.approx(0.934, abs=1e-12)
    assert _cell(reading, 1_000, "crossfit", "ate", "nu2")["coverage"] == pytest.approx(
        0.811936, abs=1e-6
    )


def test_the_corrected_bound_rows_are_near_nominal_in_every_cell() -> None:
    """The corrected lower and upper rows cover 0.947 to 0.959, with SE ratio 0.986 to 1.019.

    This holds for all 12 cells of n, fitting and parameter.  The cross-fitted ATE at n = 1000
    covers 0.9564 (lower) and 0.9554 (upper) with bias -0.068 and 0.032 SD.
    """
    reading = _reading()
    bounds = reading.loc[reading["row"].isin(["lower", "upper"])]
    assert len(bounds) == 24
    assert bounds["coverage"].min() == pytest.approx(0.947, abs=1e-12)
    assert bounds["coverage"].max() == pytest.approx(0.959, abs=1e-12)
    assert bounds["se_ratio"].min() == pytest.approx(0.986493, abs=1e-5)
    assert bounds["se_ratio"].max() == pytest.approx(1.01883, abs=1e-5)
    lower = _cell(reading, 1_000, "crossfit", "ate", "lower")
    upper = _cell(reading, 1_000, "crossfit", "ate", "upper")
    assert lower["coverage"] == pytest.approx(0.956369, abs=1e-6)
    assert upper["coverage"] == pytest.approx(0.955366, abs=1e-6)
    assert lower["bias_sd"] == pytest.approx(-0.0682293, abs=1e-5)
    assert upper["bias_sd"] == pytest.approx(0.0317387, abs=1e-5)


def test_the_refusals_are_counted_per_cell() -> None:
    """The package refuses 6, 14 and 11 of 2,000 cross-fitted draws, and 1 in-sample ATT and ATC.

    These are the ATE, ATT and ATC at n = 1000.  The in-sample ATE and every n = 4000 cell have
    none.  A refused draw keeps its raw nu^2 and loses its corrected quantities.
    """
    rows = _rows()
    refused = rows["refusal"].fillna("").astype(str).ne("")
    counts = rows.loc[refused].groupby(["n", "fitting", "parameter"]).size().to_dict()
    assert counts == {
        (1_000, "crossfit", "ate"): 6,
        (1_000, "crossfit", "att"): 14,
        (1_000, "crossfit", "atc"): 11,
        (1_000, "insample", "att"): 1,
        (1_000, "insample", "atc"): 1,
    }
    assert rows.loc[refused, "nu2_raw"].notna().all()
    for name in ("nu2", "lower", "upper"):
        assert rows.loc[refused, name].isna().all()
        assert rows.loc[~refused, name].notna().all()
    reading = _reading()
    for (n, fitting, token), count in counts.items():
        cell = _cell(reading, n, fitting, token, "nu2")
        assert cell["refused"] == count
        assert cell["replicates"] == diagnostic.ARMS[n] - count


@pytest.mark.parametrize("n", sorted(diagnostic.ARMS))
def test_the_committed_rows_are_reproduced_by_a_refit(n: int) -> None:
    """Replicate 0 at each n refits to the committed estimates, standard errors and refusals."""
    rows = _rows()
    committed = rows.loc[(rows["n"] == n) & (rows["replicate"] == 0)]
    refit = pd.DataFrame(diagnostic.replicate_rows(n, 0))
    keys = ["fitting", "parameter"]
    committed = committed.set_index(keys).sort_index()
    refit = refit.set_index(keys).sort_index()
    columns = [f"{name}{suffix}" for name in diagnostic.ROWS for suffix in ("", "_se")]
    pd.testing.assert_frame_equal(
        refit[columns], committed[columns], check_exact=False, rtol=0, atol=1e-9
    )
    assert refit["refusal"].fillna("").tolist() == committed["refusal"].fillna("").tolist()

r"""The reading of the cross-fitted plug-in omitted-variable bound diagnostic (X26).

The diagnostic is ``tests/diagnostics/x26_crossfit_bias``.  It holds the evidence for the
``cross_fit`` refusal of ``cleverly.sensitivity._mechanism_stack``.  On the binary law of the
registered study (``n = 1000``, ``cf_y = cf_d = 0.5``, ``rho = 1``) it fitted each of 2,000
draws with five-fold cross-fitting and in sample.  It repeated the comparison on 1,000 draws at
``n = 4000``.  Each test rebuilds a number of the reading from the committed rows, so the
reading cannot drift from the evidence.

The measured headline, cross-fitted against in-sample on the same 2,000 draws at ``n = 1000``,
for the ATE :math:`\nu^2`:

=============  ========  ===========================  ========  =========
fitting        coverage  99% Wilson interval          bias, SD  SE ratio
=============  ========  ===========================  ========  =========
cross-fitted   0.8845    0.8648 to 0.9017             1.441     1.079
in sample      0.9690    0.9574 to 0.9775             0.430     1.034
=============  ========  ===========================  ========  =========

At ``n = 4000`` the cross-fitted ATE :math:`\nu^2` covers 0.914 with bias 0.839 SD.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cleverly.sensitivity import _mechanism_stack
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
            expected = truth[study.estimand_key(token, name)]
            assert chosen[f"{name}_truth"].to_numpy() == pytest.approx(expected, rel=1e-12)


def test_the_seed_stream_is_not_the_registered_study() -> None:
    """No diagnostic seed equals a registered replicate seed of any scenario."""
    ours = {diagnostic.seed_of(n, k) for n in diagnostic.ARMS for k in range(50)}
    assert len(ours) == 100
    registered = {
        replicate_seed(study.STUDY, scenario, k) for scenario in study.PARAMETERS for k in range(50)
    }
    assert not ours & registered


def test_the_reading_is_rebuilt_from_the_rows() -> None:
    """The committed reading equals the reading recomputed from the committed rows."""
    committed = pd.read_csv(diagnostic.HERE / "reading.csv")
    rebuilt = _reading()
    assert len(rebuilt) == 2 * 2 * 3 * 3
    pd.testing.assert_frame_equal(rebuilt, committed, check_exact=False, rtol=1e-5, atol=1e-9)


def test_the_cross_fitted_ate_nu2_misses_its_band_at_n_1000() -> None:
    """The cross-fitted ATE nu^2 covers 0.8845 and sits 1.441 SD above the truth.

    The 99% Wilson interval, 0.8648 to 0.9017, lies below the calibration floor of 0.92.  The
    SE ratio is 1.079.  The design-review pilot read 0.887, 1.44 SD and 1.07 on 1,600 draws.
    """
    cell = _cell(_reading(), 1_000, "crossfit", "ate", "nu2")
    assert cell["replicates"] == 2_000
    assert cell["coverage"] == pytest.approx(0.8845, abs=1e-12)
    assert cell["coverage_lower99"] == pytest.approx(0.864806, abs=1e-6)
    assert cell["coverage_upper99"] == pytest.approx(0.901652, abs=1e-6)
    assert cell["coverage_upper99"] < 0.92
    assert cell["bias_sd"] == pytest.approx(1.44065, abs=1e-5)
    assert cell["se_ratio"] == pytest.approx(1.07923, abs=1e-5)


def test_the_in_sample_fit_on_the_same_draws_is_not_biased_like_it() -> None:
    """In sample, the ATE nu^2 covers 0.969 with bias 0.430 SD and SE ratio 1.034.

    The in-sample interval is conservative, and its bias is under a third of the
    cross-fitted bias on the same 2,000 draws.
    """
    reading = _reading()
    insample = _cell(reading, 1_000, "insample", "ate", "nu2")
    crossfit = _cell(reading, 1_000, "crossfit", "ate", "nu2")
    assert insample["coverage"] == pytest.approx(0.969, abs=1e-12)
    assert insample["bias_sd"] == pytest.approx(0.429972, abs=1e-5)
    assert insample["se_ratio"] == pytest.approx(1.03388, abs=1e-5)
    assert insample["coverage_lower99"] > 0.92
    assert abs(insample["bias_sd"]) < crossfit["bias_sd"] / 3.0


def test_the_bias_shrinks_but_remains_at_n_4000() -> None:
    """At n = 4000 the cross-fitted ATE nu^2 bias falls to 0.839 SD and coverage rises to 0.914.

    The ATT and the ATC nu^2 fall from 0.995 and 0.973 SD to 0.571 and 0.560 SD.  The bias
    in standard deviations does not vanish at this ladder.
    """
    reading = _reading()
    small = _cell(reading, 1_000, "crossfit", "ate", "nu2")
    large = _cell(reading, 4_000, "crossfit", "ate", "nu2")
    assert large["replicates"] == 1_000
    assert large["coverage"] == pytest.approx(0.914, abs=1e-12)
    assert large["bias_sd"] == pytest.approx(0.839218, abs=1e-5)
    assert large["se_ratio"] == pytest.approx(1.01252, abs=1e-5)
    assert large["bias_sd"] < small["bias_sd"]
    assert large["bias_sd"] > 0.5
    for token, before, after in (("att", 0.994704, 0.570926), ("atc", 0.972564, 0.560109)):
        assert _cell(reading, 1_000, "crossfit", token, "nu2")["bias_sd"] == pytest.approx(
            before, abs=1e-5
        )
        assert _cell(reading, 4_000, "crossfit", token, "nu2")["bias_sd"] == pytest.approx(
            after, abs=1e-5
        )


def test_the_cross_fitted_bound_rows_stay_near_nominal() -> None:
    """The lower and upper bound rows of the cross-fitted ATE cover 0.950 and 0.951 at n = 1000.

    Their bias is -0.316 and 0.278 SD, and the SE ratios are 1.014 and 1.017.  The bound rows
    stay near nominal on these draws while the nu^2 row misses.
    """
    reading = _reading()
    lower = _cell(reading, 1_000, "crossfit", "ate", "lower")
    upper = _cell(reading, 1_000, "crossfit", "ate", "upper")
    assert lower["coverage"] == pytest.approx(0.95, abs=1e-12)
    assert upper["coverage"] == pytest.approx(0.951, abs=1e-12)
    assert lower["bias_sd"] == pytest.approx(-0.315521, abs=1e-5)
    assert upper["bias_sd"] == pytest.approx(0.277758, abs=1e-5)
    assert lower["se_ratio"] == pytest.approx(1.01431, abs=1e-5)
    assert upper["se_ratio"] == pytest.approx(1.01681, abs=1e-5)


def test_the_committed_rows_are_reproduced_by_a_refit() -> None:
    """Replicate 0 at n = 1000 refits to the committed estimates and standard errors.

    The refit also leaves the package's refusal in place afterwards.
    """
    rows = _rows()
    committed = rows.loc[(rows["n"] == 1_000) & (rows["replicate"] == 0)]
    refit = pd.DataFrame(diagnostic.replicate_rows(1_000, 0))
    assert _mechanism_stack.OUT_OF_FOLD_REFUSED is True
    keys = ["fitting", "parameter"]
    committed = committed.set_index(keys).sort_index()
    refit = refit.set_index(keys).sort_index()
    columns = [f"{name}{suffix}" for name in diagnostic.ROWS for suffix in ("", "_se")]
    pd.testing.assert_frame_equal(
        refit[columns], committed[columns], check_exact=False, rtol=0, atol=1e-9
    )

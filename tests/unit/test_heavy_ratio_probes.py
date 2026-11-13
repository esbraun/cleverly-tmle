"""The design probes of the heavy-ratio diagnosis, read back from their committed rows.

``tests/diagnostics/fix5_heavy_ratios/probes.py`` refits each probe from the shipped code
(``run.py --probes``, about seven CPU-hours) and writes one row per draw beside its summary.
The checks below rebuild every summary from its rows and pin the readings the documentation
cites, so a summary edited by hand, or a reading quoted from a different run, fails here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.diagnostics.fix5_heavy_ratios import probes
from tests.studies import canonical_mtp_pricing as pricing

HERE = Path(probes.__file__).resolve().parent


def _rows(name: str) -> pd.DataFrame:
    return pd.read_csv(probes.rows_path(name))


def _summary(name: str) -> pd.DataFrame:
    return pd.read_csv(HERE / f"{name}.csv")


@pytest.mark.parametrize("name", probes.PROBES)
def test_each_summary_rebuilds_from_its_rows(name: str) -> None:
    rebuilt = probes.summary_of(name, _rows(name))
    committed = _summary(name)
    assert list(rebuilt.columns) == list(committed.columns)
    for column in committed.columns:
        if not pd.api.types.is_numeric_dtype(committed[column]):
            assert rebuilt[column].tolist() == committed[column].tolist(), column
        else:
            np.testing.assert_allclose(
                rebuilt[column].to_numpy(float),
                committed[column].to_numpy(float),
                rtol=1e-5,
                atol=1e-9,
                err_msg=column,
            )


def test_probe1_refits_the_committed_draws() -> None:
    """Each probe-1 fit is the committed one: its estimate equals the committed row."""
    rows = _rows("probe1")
    committed = pd.read_parquet(
        Path(pricing.__file__).resolve().parents[1]
        / "canonical"
        / "mtp_pricing"
        / "replicates.parquet"
    )
    committed = committed[committed.implementation == "cleverly-mtp-pricing"]
    for est in ("ltmle", "sdr"):
        mine = rows[rows.est == est].set_index(["rep", "estimand"]).psi
        theirs = (
            committed[committed.scenario == f"pricing_{est}_tau5"]
            .set_index(["replicate", "estimand"])
            .estimate
        )
        np.testing.assert_allclose(mine, theirs.loc[mine.index], rtol=0, atol=1e-8)


def test_probe1_bootstrap_intervals_keep_the_coupling() -> None:
    """Linearized percentile, BCa, bootstrap-t and logit/atanh Wald intervals cover at most
    0.937, and their lowest Wald-SE quartile 0.70 to 0.81."""
    summary = _summary("probe1")
    others = summary[summary.interval != "wald"]
    assert round(float(others.coverage.max()), 3) == 0.937
    assert round(float(others.coverage_q1.min()), 2) == 0.70
    assert round(float(others.coverage_q1.max()), 2) == 0.81
    wald = summary[summary.interval == "wald"]
    assert (wald.coverage.round(3).tolist(), wald.coverage_q1.round(2).tolist()) == (
        [0.916, 0.917, 0.919, 0.923],
        [0.70, 0.70, 0.71, 0.72],
    )


def test_probe4_the_unprojected_pseudo_outcome_runs_narrow() -> None:
    """Law B ``p4``, ``SDR``: projected SE ratio 0.982 and coverage 0.950; unprojected 0.799
    and 0.904."""
    summary = _summary("probe4").set_index("kind")
    assert summary.loc["projected", ["se_ratio", "coverage"]].round(3).tolist() == [0.982, 0.950]
    assert summary.loc["unprojected", ["se_ratio", "coverage"]].round(3).tolist() == [0.799, 0.904]


def test_the_learner_screen_trades_level_for_coupling() -> None:
    """Against ``n Var(error)`` 2.07: d3 3.20, d5 2.39, depth 8 1.50 with correlations -0.26,
    -0.59 and -0.90; gap features 2.71, 2.09, 1.05; ``Q_5 (1 - Q_5)`` 2.36 (initial, -0.47) and
    2.44 (targeted, -0.98)."""
    summary = _summary("screen").set_index("variant")
    assert round(float(summary.truth.iloc[0]), 2) == 2.07
    means = summary["mean"].round(2).to_dict()
    assert means == {
        "d3": 3.20,
        "d3_gap": 2.71,
        "d5": 2.39,
        "d5_gap": 2.09,
        "d8": 1.50,
        "d8_gap": 1.05,
        "q1q_initial_d5": 2.35,
        "q1q_targeted_d5": 2.44,
    }
    corr = summary.corr_error.round(2).to_dict()
    assert [corr[k] for k in ("d3", "d5", "d8", "q1q_initial_d5", "q1q_targeted_d5")] == [
        -0.26,
        -0.59,
        -0.90,
        -0.47,
        -0.98,
    ]


def _span(frame: pd.DataFrame, column: str, digits: int) -> tuple[float, float]:
    return (round(float(frame[column].min()), digits), round(float(frame[column].max()), digits))


#: Each probe-5 configuration's readings under the max over its four rows (both estimators,
#: both quantities): coverage, lowest SE quartile, corr(error, SE), SE ratio and the share of
#: draws where the policy-weighted SE won.  Law P reproduces the design probe digit for digit;
#: law B reads the oracle binned density at the policy dose where the probe read the exact
#: ratio, so its last digit can differ from the probe's.
PROBE5_MAX = {
    "pricing_exact_d3": ((0.993, 0.995), (0.996, 1.0), (-0.52, -0.47), (1.24, 1.27), (0.88, 0.89)),
    "pricing_exact_d5": ((0.982, 0.986), (0.944, 0.96), (-0.66, -0.62), (1.13, 1.17), (0.79, 0.79)),
    "pricing_learned_d3": ((0.997, 1.0), (0.987, 1.0), (-0.52, -0.46), (1.36, 1.42), (0.9, 0.9)),
    "lawb_correct_d3": ((0.951, 0.971), (0.955, 0.995), (0.07, 0.13), (1.04, 1.11), (0.7, 0.76)),
    "lawb_primary_d3": ((0.947, 0.98), (0.947, 0.987), (-0.17, 0.11), (1.06, 1.07), (0.66, 0.68)),
}


@pytest.mark.parametrize("config", sorted(PROBE5_MAX))
def test_probe5_the_max_keeps_the_low_se_quartile(config: str) -> None:
    summary = _summary("probe5")
    rows = summary[summary.config == config]
    assert len(rows) == 4
    assert (rows.nonpositive == 0).all()
    assert (rows.max_equals_reported < 1e-12).all()
    readings = (
        _span(rows, "max_coverage", 3),
        _span(rows, "max_coverage_q1", 3),
        _span(rows, "max_corr", 2),
        _span(rows, "max_se_ratio", 2),
        _span(rows, "share_pw", 2),
    )
    assert readings == PROBE5_MAX[config]


def test_probe5_the_default_wald_rows_are_the_probes() -> None:
    """The draws are the probe's: on law P the default interval covers 0.916 to 0.923 on all
    1,000 draws, its lowest SE quartile 0.70 to 0.72, and its SE's CV is 0.27 to 0.28."""
    summary = _summary("probe5")
    rows = summary[summary.config == "pricing_exact_d3"]
    assert _span(rows, "wald_coverage", 3) == (0.916, 0.923)
    assert _span(rows, "wald_coverage_q1", 2) == (0.70, 0.72)
    assert _span(rows, "wald_cv", 2) == (0.27, 0.28)


def test_probe2_the_full_refit_bca_keeps_the_shortfall() -> None:
    """The full-refit BCa covers 31 of the 81 Wald misses of ``SDR``'s contrast and misses 1
    of 30 covered draws, which extrapolates to about 0.92."""
    row = _summary("probe2").set_index("estimand").loc[probes.DOWN_ATE]
    assert (row.missed, row.missed_covered, row.covered, row.covered_missed) == (81, 31, 30, 1)
    assert round(float(row.implied_coverage), 2) == 0.92
    assert row.failed_refits == 0


def test_probe6_the_held_out_screen_picks_d5() -> None:
    """Section 6.2's rule on the preview: worst-cell SE ratio under the max 1.050 for d5
    against 1.092 for d3 (``price_down`` at three nodes), no non-positive variance."""
    summary = _summary("probe6")
    assert probes.screen_choice(summary) == "d5"
    assert round(float(summary.d3_max_se_ratio.max()), 3) == 1.092
    assert round(float(summary.d5_max_se_ratio.max()), 3) == 1.050
    assert (summary.d3_nonpositive == 0).all() and (summary.d5_nonpositive == 0).all()
    # 1,800 fits, each read on its mean and its contrast.
    assert int(summary.replicates.sum()) == 2 * 1_800


#: Law P's held-out readings: Wald coverage and SE ratio, and the max under d3 and d5.
PROBE6_LAW_P = {
    "lawP:price_down:3": (
        (0.920, 0.950),
        (0.979, 0.981),
        (0.963, 0.967),
        (1.083, 1.092),
        (0.947, 0.960),
        (1.018, 1.019),
    ),
    "lawP:price_up:5": (
        (0.937, 0.943),
        (0.987, 1.006),
        (0.950, 0.953),
        (1.029, 1.041),
        (0.950, 0.953),
        (1.037, 1.050),
    ),
}


@pytest.mark.parametrize("config", sorted(PROBE6_LAW_P))
def test_probe6_law_p_readings(config: str) -> None:
    rows = _summary("probe6")
    rows = rows[rows.config == config]
    readings = tuple(
        _span(rows, column, 3)
        for column in (
            "wald_coverage",
            "wald_se_ratio",
            "d3_max_coverage",
            "d3_max_se_ratio",
            "d5_max_coverage",
            "d5_max_se_ratio",
        )
    )
    assert readings == PROBE6_LAW_P[config]

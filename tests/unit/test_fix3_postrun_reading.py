"""The post-run readings of the ``small-sample-wald`` red cells, rebuilt from committed rows.

``tests/diagnostics/fix3_postrun/`` fitted fresh draws of each red cell's law.  Every number
the evidence page (``small-sample-wald-intervals.md``) and the roadmap's
``X8-logit-small-stratum``, ``F3-difference-near-bound`` and ``F3-skewed-continuous`` rows quote
from that diagnosis is recomputed here, at the precision printed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.diagnostics.fix3_postrun import run as diagnostic
from tests.studies import small_sample_laws as laws

Z = float(stats.norm.ppf(0.975))


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(diagnostic.HERE / name)


def test_h3_curve_is_the_efficient_influence_function_at_large_n() -> None:
    """At n = 400,000 the fitted curve's mean per support point tracks H3's exact EIF, and
    ``sqrt(n)`` times the plug-in SE is within 1% of the efficiency SD."""
    big = _read("h3_big.csv.gz")
    exact = diagnostic.h3_efficient_curve()
    probabilities = laws.h3_probs()
    for name, rows in big.groupby("name"):
        curve = exact[str(name)]
        assert abs(float(np.sum(probabilities * curve))) < 1e-12
        efficiency_sd = float(np.sqrt(np.sum(probabilities * curve**2)))
        assert np.corrcoef(rows["fitted_mean"], curve[rows["cell"]])[0, 1] > 0.9999
        root_n_se = np.sqrt(diagnostic.H3_BIG_N) * float(rows["se"].iloc[0])
        assert abs(root_n_se / efficiency_sd - 1) < 0.01


def test_h3_shortfall_at_n_2000_vanishes_at_n_8000() -> None:
    """SE ratios 0.913, 0.945, 0.891 at n = 2,000 and 0.978, 1.025, 0.989 at 8,000; coverage at
    the empirical SD stays near 0.95 at both."""
    fits = _read("h3_fits.csv.gz")
    truth = laws.h3_truths()
    ratios, oracle = {}, {}
    for (n, name), rows in fits.groupby(["n", "name"]):
        error = rows["psi"] - truth[str(name)]
        spread = float(error.std(ddof=1))
        ratios[int(n), str(name)] = round(float(rows["se"].mean()) / spread, 3)
        oracle[int(n), str(name)] = float((error.abs() <= Z * spread).mean())
    assert fits.groupby("n")["r"].nunique().to_dict() == {2000: 3000, 8000: 1500}
    assert ratios == {
        (2000, "msm[(intercept)][V=2]"): 0.913,
        (2000, "msm[a][V=2]"): 0.945,
        (2000, "msm[W][V=2]"): 0.891,
        (8000, "msm[(intercept)][V=2]"): 0.978,
        (8000, "msm[a][V=2]"): 1.025,
        (8000, "msm[W][V=2]"): 0.989,
    }
    assert all(0.94 <= value <= 0.96 for value in oracle.values())


def test_h1_difference_reading() -> None:
    """On 3,000 fresh H1 draws the identity normal interval of the difference covers 0.930,
    the rule 0.931, with misses one-sided (0.055 above, 0.014 below), and the treated arm
    carries a median 0.907 of the difference's variance."""
    rows = _read("confirm.csv.gz")
    h1 = rows.loc[rows["label"] == "ho_sparse_treat"]
    assert h1["error"].isna().all() and h1["r"].nunique() == 3000
    ate = h1.loc[h1["estimand"] == "ate"]
    error = ate["psi"] - ate["truth"]
    assert round(float((error.abs() <= Z * ate["se"]).mean()), 3) == 0.930
    shown = ate["withheld"].isna()
    covered = (ate["rule_lower"] <= ate["truth"]) & (ate["truth"] <= ate["rule_upper"]) & shown
    assert round(float(covered.mean()), 3) == 0.931
    assert round(float((ate["rule_lower"] > ate["truth"]).mean()), 3) == 0.055
    assert round(float((ate["rule_upper"] < ate["truth"]).mean()), 3) == 0.014
    se = h1.pivot(index="r", columns="estimand", values="se")
    assert round(float(np.median(se["ey[1]"] ** 2 / se["ate"] ** 2)), 3) == 0.907


@pytest.mark.parametrize(
    ("name", "plug_in", "adjusted"),
    [("ey[1][V=2]", 0.934, 0.966), ("ey[0][V=2]", 0.956, 0.975), ("ate[V=2]", 0.944, 0.972)],
)
def test_h2_hand_built_curve_and_leverage(name: str, plug_in: float, adjusted: float) -> None:
    """The hand-built stratum-2 curve equals the fit's, and dividing each residual by the cell
    mean's leverage moves the SE ratio from ``plug_in`` to ``adjusted``."""
    rows = _read("h2_leverage.csv.gz")
    rows = rows.loc[rows["name"] == name]
    assert len(rows) == 2998
    assert float((rows["psi"] - rows["psi_hand"]).abs().max()) < 1e-13
    assert round(float(np.median(rows["se_hand"] / rows["se"])), 4) == 0.9997
    spread = float((rows["psi"] - rows["truth"]).std(ddof=1))
    assert round(float(rows["se"].mean()) / spread, 3) == plug_in
    assert round(float((rows["se"] * rows["se_adj_ratio"]).mean()) / spread, 3) == adjusted


def test_h2_smallest_cell_holds_about_six_rows() -> None:
    """The smallest of H2's six stratum-2 design cells (treated, ``W = 0``) holds about 6.5
    rows at n = 2,000, the saturated learner's leverage there being about 1/6.5."""
    cells = laws.h2_cell_probabilities(2)
    assert round(2_000 * min(cells.values()), 1) == 6.5
    assert min(cells, key=cells.__getitem__) == (1, 0)

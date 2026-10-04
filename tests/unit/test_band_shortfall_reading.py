"""The finite-sample reading of every red joint-coverage cell, rebuilt from the committed rows.

Each red band cell was read once by the declared diagnostic: the same replications, read again at
the design critical value ``c*`` of ``tests/unit/test_simultaneous_cell_design.py``.  A joint row
records the max-t statistic in ``estimate``, so the oracle band covers a replication exactly when
``estimate <= c*``, and the package band covers it when ``estimate <= std_error``.

The reading is finite-sample when the oracle band also under-covers.  The package critical value
then accounts for only the small gap between the two coverages.  For the strata and survival
cells ``c*`` comes from the exact law.  For a default-band shape it comes from the package's own
influence curves averaged over the design fits, so it is independent of the multiplier but not of
the curves; the source study's pointwise calibration cells are the evidence for those curves.

The owner rows of ``docs/roadmap.md`` and the study pages quote every number below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.studies import default_band_properties, stratified_law
from tests.studies.default_band_properties import design, design_correlation
from tests.studies.evidence.registry import ROOT
from tests.unit.test_clustered_crossfit_ltmle_design import BAND_DESIGN
from tests.unit.test_simultaneous_cell_design import EXACT


def _exact(cell: str, replicates: int) -> float:
    return design(EXACT[cell][0], replicates).critical


def _shape(label: str) -> float:
    shape = default_band_properties.SHAPE_BY_LABEL[label]
    return design(design_correlation(label), shape.replicates).critical


READINGS = {
    "strata": (
        "tmle3_stratified",
        lambda: _exact("canonical-stratified-tmle/strata", 2400),
        2.846,
        2.816,
        0.9046,
        0.9121,
    ),
    "crossfit_strata": (
        "tmle3_stratified",
        lambda: _exact("canonical-stratified-tmle/crossfit_strata", 2400),
        2.778,
        2.750,
        0.9192,
        0.9258,
    ),
    "all_reported": (
        "ltmle_survival",
        lambda: _exact("canonical-ltmle-survival/all_reported", 9600),
        2.622,
        2.605,
        0.9214,
        0.9241,
    ),
    "categorical_ltmle": (
        "default_bands",
        lambda: _shape("categorical_ltmle"),
        2.689,
        2.680,
        0.9250,
        0.9267,
    ),
    "categorical_ltmle_crossfit": (
        "default_bands",
        lambda: _shape("categorical_ltmle_crossfit"),
        2.689,
        2.680,
        0.9317,
        0.9325,
    ),
    "cde_z0": ("default_bands", lambda: _shape("cde_z0"), 2.331, 2.312, 0.9171, 0.9200),
    # The cross-fitted clustered LTMLE band. ``c*`` is the design critical value pinned before the
    # run from ten fits of the study's own subject.
    "clustered_regimens": (
        "lmtp_clustered_ltmle",
        lambda: BAND_DESIGN[1],
        2.409,
        2.387,
        0.9333,
        0.9392,
    ),
    "arms": (
        "drtmle_mar_multi_arm",
        lambda: _exact("multi-arm-mar-drtmle/arms", 2400),
        2.508,
        2.500,
        0.9337,
        0.9342,
    ),
}


@pytest.mark.parametrize("label", sorted(READINGS))
def test_the_red_band_reading_is_the_committed_rows_own(label: str) -> None:
    directory, critical, quoted_critical, package_critical, package, oracle = READINGS[label]
    rows = pd.read_csv(
        ROOT / "tests" / "canonical" / directory / "property-replicates.csv.gz",
        float_precision="round_trip",
    )
    band = rows.loc[rows["cell"] == f"{label}__simultaneous_band"]
    assert len(band) == int(band["requested_replicates"].iloc[0])
    c_star = critical()
    assert round(c_star, 3) == quoted_critical
    assert round(float(band["std_error"].mean()), 3) == package_critical
    # The package band's own verdict, read back from the statistic it recorded.
    assert (band["covered"] == (band["estimate"] <= band["std_error"]).astype(int)).all()
    assert round(float(band["covered"].mean()), 4) == package
    oracle_coverage = float((band["estimate"] <= c_star).mean())
    assert round(oracle_coverage, 4) == oracle
    # The finite-sample reading: the oracle band under-covers too, and the multiplier's lower
    # critical value explains less of the shortfall than the oracle band shows.
    assert oracle_coverage < 0.95
    assert oracle - package < 0.95 - oracle


def test_the_smallest_stratum_reading() -> None:
    """The ``strata-boundary-mean`` reading: a mean near 0.94 on few treated rows.

    At n = 2,000 the stratum V = 2 holds about 157 treated rows, and about 9 of them are expected
    to be non-events.  A Wald interval of a mean that close to 1 under-covers at that count.
    """
    v, _, a, y = np.asarray(stratified_law.SUPPORT, dtype=float).T
    probs = stratified_law.PROBS
    inside = v == 2
    assert round(stratified_law.TRUTH["ey[1][V=2]"], 4) == 0.9358
    treated_share = probs[inside & (a == 1)].sum() / probs[inside].sum()
    assert round(float(treated_share), 4) == 0.3938
    assert int(2000 * float(probs[inside].sum()) * float(treated_share)) == 157
    assert round(2000 * float(probs[inside & (a == 1) & (y == 0)].sum())) == 9


def test_the_multi_arm_missing_outcome_calibration_draws() -> None:
    """The ``F4-calibration-draws`` reading: one draw set, within chance of its siblings.

    The calibration cell and the band read the same 2,400 fits. Their empirical spread is 1.037
    times the mean standard error, while two independent cells of the same configuration and
    size sit at 0.997 and 0.989, with the same mean standard error. Pooled over all 4,400 fits
    the ratio is 1.020, and a Bartlett test across the three cells gives p = 0.20.
    """
    rows = pd.read_csv(
        ROOT / "tests" / "canonical" / "drtmle_mar_multi_arm" / "property-replicates.csv.gz",
        float_precision="round_trip",
    )
    ratios = {}
    for cell in ("ate__correctly_specified", "n_2000", "l3_ate_low__both_correct"):
        selected = rows.loc[rows["cell"] == cell]
        ratios[cell] = round(
            float(selected["estimate"].std(ddof=1) / selected["std_error"].mean()), 3
        )
        assert abs(float(selected["std_error"].mean()) - 0.0377) < 1e-4
    assert ratios == {
        "ate__correctly_specified": 1.037,
        "n_2000": 0.997,
        "l3_ate_low__both_correct": 0.989,
    }
    pooled = rows.loc[rows["cell"].isin(list(ratios))]
    z = (pooled["estimate"] - pooled["truth"]) / pooled["std_error"]
    assert len(z) == 4_400
    assert round(float(z.std(ddof=1)), 3) == 1.020
    # Bartlett on the standardized errors, so the three cells are compared on one scale.
    groups = [z.loc[pooled["cell"] == cell].to_numpy() for cell in ratios]
    assert round(float(stats.bartlett(*groups).pvalue), 2) == 0.20


def test_the_logit_slope_shortfall_in_the_smallest_stratum() -> None:
    """The ``X8-logit-small-stratum`` reading: the shipped unstratified fit shares the shortfall.

    The registered cells read 0.967 and 0.862.  On fresh draws the stratified fit reads 0.867 in
    stratum 2, and the unstratified fit on a sample of that stratum's size reads 0.896.  At four
    times the size both read within 0.01 of one.
    """
    registered = pd.read_csv(
        ROOT / "tests" / "canonical" / "npcausal_stratified_incremental" / "properties.csv",
        float_precision="round_trip",
    ).set_index("cell")
    assert round(float(registered.loc["logit_v1_a__correctly_specified", "se_ratio"]), 3) == 0.967
    assert round(float(registered.loc["logit_v2_a__correctly_specified", "se_ratio"]), 3) == 0.862
    rows = pd.read_csv(
        ROOT / "tests" / "diagnostics" / "x8_logit_small_stratum" / "rows.csv.gz",
        float_precision="round_trip",
    )
    ratios = {
        (str(arm), int(n)): round(
            float(group["std_error"].mean() / group["estimate"].std(ddof=1)), 3
        )
        for (arm, n), group in rows.groupby(["arm", "n"])
    }
    assert ratios == {
        ("pooled", 2000): 0.867,
        ("subset", 2000): 0.896,
        ("pooled", 8000): 1.006,
        ("subset", 8000): 0.997,
    }


def test_the_identity_msm_smallest_stratum() -> None:
    """The ``X8-identity-small-stratum`` reading, rebuilt from the committed rows.

    The calibration cell reads 0.964, while the primary draws give 0.983 and 0.985 for the same
    coefficient in the two implementations.  The paired ``msm[W][V=2]`` estimates agree to 6e-6
    with the same spread, and R reports 2.2% above its own spread.
    """
    here = ROOT / "tests" / "canonical" / "tmle3_stratified_msm"
    cells = pd.read_csv(here / "properties.csv", float_precision="round_trip").set_index("cell")
    assert round(float(cells.loc["identity_v2_a__correctly_specified", "se_ratio"]), 3) == 0.964
    summary = pd.read_csv(here / "summary.csv", float_precision="round_trip").set_index(
        ["estimand", "implementation"]
    )
    ratio = summary["se_ratio"].round(3)
    assert ratio[("msm[a][V=2]", "cleverly-stratified-msm")] == 0.983
    assert ratio[("msm[a][V=2]", "tmle3-msm-stratified")] == 0.985
    assert ratio[("msm[W][V=2]", "tmle3-msm-stratified")] == 1.022
    spread = summary["empirical_se"]
    assert (
        abs(
            spread[("msm[W][V=2]", "cleverly-stratified-msm")]
            - spread[("msm[W][V=2]", "tmle3-msm-stratified")]
        )
        < 2e-5
    )
    paired = pd.read_csv(here / "equivalence.csv", float_precision="round_trip").set_index(
        "estimand"
    )
    row = paired.loc["msm[W][V=2]"]
    assert abs(float(row["mean_difference"])) < 1e-5
    assert round(float(row["calibration_excess_upper"]), 4) == 0.0523
    assert row["comparison_conclusion"] == "inconclusive"


def _bias(rows: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(estimand), str(arm)): round(float((group["estimate"] - group["truth"]).mean()), 4)
        for (estimand, arm), group in rows.groupby(["estimand", "arm"])
    }


def test_the_stratified_drtmle_treatment_correct_bias() -> None:
    """The ``X8-drtmle-one-sided-bias`` reading, rebuilt from the committed diagnostic rows.

    Arm ``C`` reproduces the committed cells exactly.  Inside each stratum the shipped
    unstratified fit (``S``) and R ``drtmle`` with the same arrays (``R``) carry the same
    positive bias.  The stratified marginal inherits it and contracts faster than its spread.
    """
    here = ROOT / "tests" / "diagnostics" / "x8_drtmle_treatment_correct"
    validation = pd.read_csv(here / "validation.csv")
    assert (validation["replicates"] == 400).all()
    assert (validation["max_abs_estimate_difference"] == 0.0).all()
    bias = _bias(pd.read_csv(here / "rows.csv.gz", float_precision="round_trip"))
    for s in (0, 1, 2):
        name = f"ate[V={s}]"
        assert min(bias[(name, arm)] for arm in "CRS") > 0.009, name
    assert bias[("ate", "C")] == 0.0151
    assert bias[("ate", "M")] == 0.0029
    # Stratum 2 is mixed: on the same arrays the package's bias exceeds R's.  The subset fit
    # matches the package there, so the excess is DRTMLE at stratum size, not the strata.
    rows = pd.read_csv(here / "rows.csv.gz", float_precision="round_trip")
    wide = rows.pivot_table(index=["replicate", "estimand"], columns="arm", values="estimate")
    v2 = wide.xs("ate[V=2]", level="estimand")
    gap = v2["C"] - v2["R"]
    assert round(float(gap.mean()), 4) == 0.0080
    assert round(float(gap.std(ddof=1) / np.sqrt(len(gap))), 4) == 0.0018
    assert abs(float((v2["C"] - v2["S"]).mean())) < 1e-4
    contraction = pd.read_csv(here / "contraction-rows.csv.gz", float_precision="round_trip")
    assert _bias(contraction)[("ate", "C")] == 0.0058
    spread = contraction.loc[
        (contraction["arm"] == "C") & (contraction["estimand"] == "ate"), "estimate"
    ].std(ddof=1)
    assert round(float(spread), 4) == 0.0135


def test_the_stratified_drtmle_smallest_stratum() -> None:
    """The ``X8-drtmle-small-stratum`` reading: R drtmle shares each shortfall on the same draws."""
    here = ROOT / "tests" / "canonical" / "drtmle_stratified"
    summary = pd.read_csv(here / "summary.csv", float_precision="round_trip").set_index(
        ["estimand", "implementation"]
    )
    coverage = summary["coverage"].round(4)
    assert coverage[("ey[1][V=2]", "cleverly-stratified-drtmle")] == 0.9
    assert coverage[("ey[1][V=2]", "drtmle-r-stratified")] == 0.8925
    ratio = summary["se_ratio"].round(3)
    assert ratio[("ate[V=2]", "cleverly-stratified-drtmle")] == 0.985
    assert ratio[("ate[V=2]", "drtmle-r-stratified")] == 0.973
    paired = pd.read_csv(here / "equivalence.csv", float_precision="round_trip").set_index(
        "estimand"
    )
    assert abs(float(paired.loc["ey[1][V=2]", "mean_difference"])) < 1e-4
    cells = pd.read_csv(here / "properties.csv", float_precision="round_trip").set_index("cell")
    assert round(float(cells.loc["v2_ate__correctly_specified", "se_ratio_ci_lower"]), 4) == 0.9294

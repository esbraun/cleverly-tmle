"""Summarize ``sweep.csv`` into the repeated-draw numbers the notebook quotes.

Usage: ``python summarize.py``.  Output: ``summary.log``.  The ATE truth is the package value
``navigation_data(...)[1]["ate"]``, which ``truth.py`` confirms independently.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRUTH = 0.1628580376571836
LAW_NU2 = 4.8287  # truth.py
rows = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip")
n = len(rows)


def share(mask) -> str:
    p = float(np.mean(mask))
    return f"{p:.3f} (MC SE {np.sqrt(p * (1 - p) / n):.3f}; {int(np.sum(mask))} of {n})"


def quantiles(column, digits=2) -> list:
    return rows[column].quantile([0.05, 0.5, 0.95]).round(digits).tolist()


print(f"seeds {rows['seed'].min()} to {rows['seed'].max()}, {n} draws")
print()
print("Steps 5 and 6, the cross-fitted and the in-sample fit")
for label, name in (("cf", "cross-fitted"), ("is", "in-sample")):
    sd = rows[f"{label}_psi"].std(ddof=1)
    se = rows[f"{label}_se"].mean()
    print(f"  {name}")
    print(
        f"    bias {rows[f'{label}_psi'].mean() - TRUTH:+.4f} (MC SE {sd / np.sqrt(n):.4f});"
        f" empirical SD {sd:.4f}; mean SE {se:.4f}; SE/SD {se / sd:.2f}"
    )
    print(f"    coverage {share(rows[f'{label}_covers'])}")
ratio = rows["is_se"] / rows["cf_se"]
print(f"  in-sample SE / cross-fitted SE, 5% / 50% / 95%: {ratio.quantile([0.05, 0.5, 0.95]).round(2).tolist()}")
print(f"  in-sample SE / cross-fitted SE below 1/3: {share(ratio < 1 / 3)}")
print(f"  in-sample SE smaller than cross-fitted SE: {share(ratio < 1)}")
print(f"  |in-sample psi - cross-fitted psi| < 0.01: {share((rows['is_psi'] - rows['cf_psi']).abs() < 0.01)}")
print(
    "  in-sample miss in its SEs exceeds the cross-fitted miss in its SEs:"
    f" {share(rows['is_miss_in_se'] > rows['cf_miss_in_se'])}"
)
print()
print("Step 11, the assessment of the cross-fitted fit")
print(f"  no attention row: {share(rows['attention'].isna())}")
print(f"  propensity calibration slope, 5% / 50% / 95%: {quantiles('cal_slope')}")
print(f"  no unit truncated at the default bound: {share(rows['truncated'] == 0)}")
print(f"  smallest fitted g, 5% / 50% / 95%: {quantiles('min_g', 4)}")
print(f"  fitted share of g below 0.1, 5% / 50% / 95%: {quantiles('share_g_below_0.1', 4)}")
print(f"  treated ESS ratio, 5% / 50% / 95%: {quantiles('cf_ess_treated', 3)}")
print(f"  treated is the smaller ESS ratio: {share(rows['cf_ess_treated'] < rows['cf_ess_control'])}")
cf_min = np.minimum(rows["cf_ess_treated"], rows["cf_ess_control"])
print(f"  in-sample minimum ESS ratio above the cross-fitted one: {share(rows['is_ess_min'] > cf_min)}")
print()
print("Step 13, the sensitivity analysis of the cross-fitted fit")
print(f"  rv 5% / 50% / 95%: {quantiles('rv', 3)}")
print(f"  rva 5% / 50% / 95%: {quantiles('rva', 3)}")
print(
    f"  nu^2 (doubly robust) mean {rows['nu2'].mean():.3f}, 5% / 95%:"
    f" {rows['nu2'].quantile([0.05, 0.95]).round(2).tolist()}"
)
print(f"  nu^2 below the law's {LAW_NU2} (truth.py): {share(rows['nu2'] < LAW_NU2)}")
print(
    f"  mean shortfall {1 - rows['nu2'].mean() / LAW_NU2:.3f} in nu^2,"
    f" {1 - np.sqrt(rows['nu2'].mean() / LAW_NU2):.3f} in the bias scale"
)
print(f"  default bias-adjusted bounds exclude zero: {share(rows['bound_lower'] > 0)}")
print(f"  their 95% limits exclude zero: {share(rows['limit_lower'] > 0)}")
print(
    "  fitted share of g below 0.1 under the law's share, 0.0188 (truth.py):"
    f" {share(rows['share_g_below_0.1'] < 0.0188)}"
)
print()
print("Step 13, the shown draw (seed 34) at the law's nu^2 (truth.py)")
# Printed by the notebook's Step 13 cell: the estimate, sigma^2 and nu^2 of the fit, at the
# default strengths cf_y = cf_d = 0.03.  Bound and robustness value follow omitted_variable.py:
# bias <= sqrt(sigma^2 nu^2) * |rho| sqrt(cf_y cf_d / (1 - cf_d)), and at rho = 1 the equal
# strength r solving sqrt(sigma^2 nu^2) * r / sqrt(1 - r) = estimate.
SHOWN = {"estimate": 0.15731, "sigma2": 0.018093, "nu2": 4.5363, "cf_y": 0.03, "cf_d": 0.03}
for label, nu2 in (("fitted", SHOWN["nu2"]), ("law", LAW_NU2)):
    scale = np.sqrt(SHOWN["sigma2"] * nu2)
    bias = scale * np.sqrt(SHOWN["cf_y"] * SHOWN["cf_d"] / (1.0 - SHOWN["cf_d"]))
    t2 = (SHOWN["estimate"] / scale) ** 2
    rv = (np.sqrt(t2**2 + 4.0 * t2) - t2) / 2.0
    print(
        f"  {label:6s} nu^2 {nu2:.4f}: bias scale {scale:.5f}; bias <= {bias:.5f};"
        f" bounds ({SHOWN['estimate'] - bias:.4f}, {SHOWN['estimate'] + bias:.4f}); rv {rv:.3f}"
    )

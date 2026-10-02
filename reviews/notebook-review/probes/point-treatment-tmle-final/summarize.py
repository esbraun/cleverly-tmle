"""Summarize ``sweep.csv`` into the repeated-draw numbers the notebook quotes.

Usage: ``python summarize.py``.  Output: ``summary.log``.  The ATE truth is the package value
``navigation_data(...)[1]["ate"]``, which ``truth.py`` confirms independently.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRUTH = 0.1628580376571836
rows = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip")
n = len(rows)


def share(mask) -> str:
    p = float(np.mean(mask))
    return f"{p:.3f} (MC SE {np.sqrt(p * (1 - p) / n):.3f}; {int(np.sum(mask))} of {n})"


sd = rows["psi"].std(ddof=1)
print(f"seeds {rows['seed'].min()} to {rows['seed'].max()}, {n} draws")
print()
print("Step 6, the headline fit")
print(f"  bias {rows['psi'].mean() - TRUTH:+.4f} (MC SE {sd / np.sqrt(n):.4f})")
print(
    f"  empirical SD {sd:.4f}; mean SE {rows['se'].mean():.4f}; SE/SD {rows['se'].mean() / sd:.2f}"
)
print(f"  coverage {share(rows['covers'])}")
print()
print("Step 8, the learner combinations")
for label in ("flex_q_lin_g", "lin_q_flex_g", "both_linear"):
    error = (rows[f"{label}_psi"] - TRUTH).abs()
    print(
        f"  {label:13s} coverage {share(rows[f'{label}_covers'])}; mean |miss| {error.mean():.4f};"
        f" bias {(rows[f'{label}_psi'] - TRUTH).mean():+.4f}"
    )
both = (rows["both_linear_psi"] - TRUTH).abs()
one = np.maximum((rows["flex_q_lin_g_psi"] - TRUTH).abs(), (rows["lin_q_flex_g_psi"] - TRUTH).abs())
print(f"  both-linear miss larger than either one-flexible miss: {share(both > one)}")
print()
print("Step 9, the assessment and the truncation curve")
print(f"  no attention row: {share(rows['attention'].isna())}")
q = rows["cal_slope"].quantile([0.05, 0.5, 0.95]).round(2).tolist()
print(f"  propensity calibration slope, 5% / 50% / 95%: {q}")
print(f"  truncation at the default bound is zero: {share(rows['trunc_default'] == 0)}")
for bound in (0.05, 0.1, 0.2):
    move = rows[f"psi_move_{bound}"].abs()
    print(
        f"  bound {bound}: truncation positive {share(rows[f'trunc_{bound}'] > 0)};"
        f" median truncated share {rows[f'trunc_{bound}'].median():.4f};"
        f" median |psi move| {move.median():.5f}, max {move.max():.5f}"
    )
print()
print("Step 10, the sensitivity analysis")
print(f"  rv 5% / 50% / 95%: {rows['rv'].quantile([0.05, 0.5, 0.95]).round(3).tolist()}")
print(
    f"  nu^2 (doubly robust) mean {rows['nu2'].mean():.3f}, 5% / 95%:"
    f" {rows['nu2'].quantile([0.05, 0.95]).round(2).tolist()}"
)
print(f"  nu^2 below the law's 4.83 (truth.py): {share(rows['nu2'] < 4.83)}")
print(f"  discharge_risk cf_y clipped at 1: {share(rows['strong_cf_y'] == 1.0)}")
print(
    f"  discharge_risk raw outcome gain, min / max: {rows['strong_gain_y'].min():.3f} / {rows['strong_gain_y'].max():.3f}"
)
print(
    f"  medication_burden cf_y 5% / 50% / 95%: {rows['med_cf_y'].quantile([0.05, 0.5, 0.95]).round(3).tolist()}"
)
print(
    f"  medication_burden cf_d 5% / 50% / 95%: {rows['med_cf_d'].quantile([0.05, 0.5, 0.95]).round(3).tolist()}, max {rows['med_cf_d'].max():.3f}"
)
print(f"  medication_burden cf_y > cf_d: {share(rows['med_cf_y'] > rows['med_cf_d'])}")
print(f"  bias-adjusted bounds exclude zero: {share(rows['bound_lower'] > 0)}")
print(f"  95% limits of the bounds exclude zero: {share(rows['limit_lower'] > 0)}")
print(
    f"  bounds contain the truth: {share((rows['bound_lower'] <= TRUTH) & (TRUTH <= rows['bound_upper']))}"
)
print()
print("Step 3")
print(f"  unadjusted difference closer to the ATT: {share(rows['unadjusted_closer_to_att'])}")
print()
print("Step 10, the shown draw at the law's nu^2 (truth.py)")
# Printed by the notebook's Step 10 cell: the estimate, sigma^2 and nu^2 of the fit, and the
# medication_burden strengths.  Bound and robustness value follow omitted_variable.py:
# bias <= sqrt(sigma^2 nu^2) * |rho| sqrt(cf_y cf_d / (1 - cf_d)), and at rho = 1 the equal
# strength r solving sqrt(sigma^2 nu^2) * r / sqrt(1 - r) = estimate.
SHOWN = {"estimate": 0.16658, "sigma2": 0.017566, "nu2": 4.4183, "cf_y": 0.1122, "cf_d": 0.0587}
LAW_NU2 = 4.8286
for label, nu2 in (("fitted", SHOWN["nu2"]), ("law", LAW_NU2)):
    scale = np.sqrt(SHOWN["sigma2"] * nu2)
    bias = scale * np.sqrt(SHOWN["cf_y"] * SHOWN["cf_d"] / (1.0 - SHOWN["cf_d"]))
    t2 = (SHOWN["estimate"] / scale) ** 2
    rv = (np.sqrt(t2**2 + 4.0 * t2) - t2) / 2.0
    print(
        f"  {label:6s} nu^2 {nu2:.4f}: bias scale {scale:.5f}; bias <= {bias:.4f};"
        f" lower bound {SHOWN['estimate'] - bias:.4f}; rv {rv:.3f}"
    )
print(
    f"  mean doubly robust nu^2 over the draws {rows['nu2'].mean():.3f}: shortfall"
    f" {1 - rows['nu2'].mean() / LAW_NU2:.3f} in nu^2,"
    f" {1 - np.sqrt(rows['nu2'].mean() / LAW_NU2):.3f} in the bias scale"
)

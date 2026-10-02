"""Summarize ``sweep.csv`` into the repeated-draw numbers the notebook quotes.

Usage: ``python summarize.py``.  Output: ``summary.log``.  The ATE truth is the package value
``navigation_data(...)[1]["ate"]``, which ``truth.py`` confirms independently.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRUTH = 0.1628580376571836
BOUND = 5.0 / (np.sqrt(2000.0) * np.log(2000.0))
rows = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip")
n = len(rows)


def share(mask) -> str:
    p = float(np.mean(mask))
    return f"{p:.3f} (MC SE {np.sqrt(p * (1 - p) / n):.3f}; {int(np.sum(mask))} of {n})"


def quantiles(column, digits=2) -> list:
    return rows[column].quantile([0.05, 0.5, 0.95]).round(digits).tolist()


print(f"seeds {rows['seed'].min()} to {rows['seed'].max()}, {n} draws, n = 2000 each")
print()
print("Steps 6, 7 and 8: each fit against the truth")
for label, name in (
    ("ord", "ordinary TMLE"),
    ("dr", "DR-TMLE, Super Learner reductions"),
    ("const", "DR-TMLE, constant reductions"),
):
    psi = rows[f"{label}_psi"]
    sd = psi.std(ddof=1)
    se = rows[f"{label}_se"].mean()
    rmse = float(np.sqrt(np.mean((psi - TRUTH) ** 2)))
    print(f"  {name}")
    print(
        f"    bias {psi.mean() - TRUTH:+.5f} (MC SE {sd / np.sqrt(n):.5f}); empirical SD {sd:.5f};"
        f" mean SE {se:.5f}; SE/SD {se / sd:.2f}; RMSE {rmse:.5f}"
    )
    print(f"    coverage {share(rows[f'{label}_covers'])}")

print()
print("Paired comparison, DR-TMLE minus the ordinary TMLE")
diff = rows["dr_psi"] - rows["ord_psi"]
print(f"  mean difference {diff.mean():+.5f} (MC SE {diff.std(ddof=1) / np.sqrt(n):.5f})")
shift = diff / rows["ord_se"]
print(
    f"  shift in ordinary SEs, 5% / 50% / 95%: {shift.quantile([0.05, 0.5, 0.95]).round(2).tolist()}"
)
print(f"  |shift| below one ordinary SE: {share(shift.abs() < 1)}")
ratio = rows["dr_se"] / rows["ord_se"]
print(
    f"  SE ratio DR / ordinary, 5% / 50% / 95%: {ratio.quantile([0.05, 0.5, 0.95]).round(3).tolist()}"
)
nearer = (rows["dr_psi"] - TRUTH).abs() < (rows["ord_psi"] - TRUTH).abs()
print(f"  DR-TMLE nearer the truth than the ordinary TMLE: {share(nearer)}")
both = rows["dr_covers"] & rows["ord_covers"]
print(f"  both intervals cover: {share(both)}")
print(f"  only DR-TMLE covers: {share(rows['dr_covers'] & ~rows['ord_covers'])}")
print(f"  only the ordinary TMLE covers: {share(~rows['dr_covers'] & rows['ord_covers'])}")

print()
print("Step 8: constant reductions")
print(
    f"  constant fit passes score and correction checks: {share(rows['const_scores_passed'] & rows['const_corrections_passed'])}"
)
print(
    f"  spline fit passes score and correction checks: {share(rows['dr_scores_passed'] & rows['dr_corrections_passed'])}"
)
const_move = (rows["const_psi"] - rows["ord_psi"]).abs() / rows["ord_se"]
dr_move = (rows["dr_psi"] - rows["ord_psi"]).abs() / rows["ord_se"]
print(f"  |constant - ordinary| in ordinary SEs, median {const_move.median():.2f}")
print(f"  |spline - ordinary| in ordinary SEs, median {dr_move.median():.2f}")
spline_nearer = (rows["dr_psi"] - TRUTH).abs() < (rows["const_psi"] - TRUTH).abs()
print(f"  spline fit nearer the truth than the constant fit: {share(spline_nearer)}")

print()
print("Step 9: the assessment of the DR-TMLE fit")
print(f"  no attention row: {share(rows['attention'].isna())}")
print(f"  contract theorem: {share(rows['contract'] == 'theorem')}")
print(f"  nuisance report says look reasonable: {share(rows['nuisance_reasonable'])}")
print(f"  no unit truncated at the default bound {BOUND:.5f}: {share(rows['truncated'] == 0)}")
print(f"  smallest fitted g, 5% / 50% / 95%: {quantiles('min_fitted_g', 4)}")
print(f"  smallest true g on the draw's rows, 5% / 50% / 95%: {quantiles('min_true_g', 4)}")
print(f"  smallest true g on the draw's rows below the bound: {share(rows['min_true_g'] < BOUND)}")
for family in ("qr", "gr1", "gr2"):
    print(
        f"  {family}: spline best in at least 4 of 6 fits: {share(rows[f'{family}_spline_best'] >= 4)}"
    )

print()
print("Step 10: E-value of the DR-TMLE fit")
print(f"  point, 5% / 50% / 95%: {quantiles('evalue_point')}")
print(f"  limit, 5% / 50% / 95%: {quantiles('evalue_limit')}")
print(f"  limit above 2: {share(rows['evalue_limit'] > 2)}")

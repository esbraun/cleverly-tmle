"""Summarize ``sweep.csv`` into the repeated-draw numbers docs/examples/subgroup-effects.ipynb quotes.

Each line gives the share of draws on which one relation holds that the notebook's readings
state, with its Monte Carlo standard error.  A relation below 0.90 is worded "in this sample" on
the page (docs/development/example-notebooks.md, rules for markdown cells).

Usage: ``python summarize.py``.  Output: ``summary.log``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rows = pd.read_csv(HERE / "sweep.csv")
n = len(rows)
lines = []
GROUPS = ("family", "alone", "carer")


def share(mask) -> str:
    p = float(np.mean(mask))
    return f"{p:.2f} (MC SE {np.sqrt(p * (1 - p) / n):.3f}; {int(np.sum(mask))} of {n})"


def coverage(prefix: str, title: str) -> None:
    lines.append(title)
    for group in GROUPS:
        lines.append(f"  {group} term interval covers: {share(rows[f'{prefix}_covers_{group}'])}")


lines.append(
    f"seeds {rows['seed'].min()} to {rows['seed'].max()}, {n} draws; the page shows seed 100"
)
lines.append("")
lines.append("Why a simple comparison misleads")
lines.append(
    "  simple difference below the truth in every group: "
    + share((rows[[f"naive_gap_{g}" for g in GROUPS]] > 0).all(axis=1))
)
for group in GROUPS:
    lines.append(f"  {group} gap above 0.03: {share(rows[f'naive_gap_{group}'] > 0.03)}")
lines.append("")
coverage("indicators", "Projection on the group indicators")
lines.append(f"  every row supported: {share(rows['indicators_all_supported'])}")
lines.append(f"  every group's pointwise interval covers: {share(rows['group_points_cover'])}")
lines.append(f"  band covers all three groups: {share(rows['group_band_covers'])}")
lines.append(f"  band label supported: {share(rows['group_band_label'] == 'supported')}")
lines.append(f"  heterogeneity test p < 0.01: {share(rows['heterogeneity_p'] < 0.01)}")
lines.append(f"  heterogeneity test p < 0.05: {share(rows['heterogeneity_p'] < 0.05)}")
for group in GROUPS:
    error = rows[f"error_{group}"]
    lines.append(
        f"  {group} effect error: mean {error.mean():+.4f} (MC SE {error.std(ddof=1) / np.sqrt(n):.4f}),"
        f" SD {error.std(ddof=1):.3f}"
    )
lines.append("")
coverage("additive", "Indicators plus a straight line in age")
lines.append(f"  age slope interval covers 0.003: {share(rows['age_slope_covers'])}")
lines.append(f"  age slope interval excludes 0: {share(rows['age_slope_excludes_zero'])}")
lines.append(f"  band covers all nine points: {share(rows['age_band_covers'])}")
lines.append(f"  all nine pointwise intervals cover: {share(rows['age_points_cover'] == 9)}")
lines.append(f"  test p < 0.01: {share(rows['age_heterogeneity_p'] < 0.01)}")
lines.append("")
lines.append("Omitted-variable bound on the alone-minus-family difference")
lines.append(
    f"  bound at strength 0.03 stays above 0: {share(rows['alone_bound_lower_above_zero'])}"
)
q = rows["alone_rv"].quantile([0.05, 0.5, 0.95]).round(3).tolist()
lines.append(f"  robustness value 5% / 50% / 95%: {q}")
lines.append("")
lines.append("Refutation of the alone-minus-family difference")
lines.append(f"  placebo mean within 2 SDs of 0: {share(rows['placebo_within_2sd'])}")
lines.append(f"  random common cause within 0.01: {share(rows['random_common_cause_within_001'])}")
lines.append(f"  subset within 0.01: {share(rows['subset_within_001'])}")
lines.append("")
for z, words in ((0, "no"), (1, "yes")):
    coverage(f"direct{z}", f"Direct effect, visit held at {words}")
    lines.append(
        f"  every group's pointwise interval covers: {share(rows[f'direct{z}_points_cover'])}"
    )
    lines.append(f"  band covers all three groups: {share(rows[f'direct{z}_band_covers'])}")
lines.append("")
lines.append("Age slope inside each group (strata=)")
for group in GROUPS:
    lines.append(
        f"  {group} slope interval covers 0.003: {share(rows[f'strata_slope_covers_{group}'])}"
    )
lines.append("")
coverage("ratio", "Log risk ratio on the group indicators")
lines.append(f"  every group's ratio interval covers: {share(rows['ratio_points_cover'])}")
(HERE / "summary.log").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

"""Summarize ``sweep.csv`` into ``summary.log``, the numbers the longitudinal notebook quotes.

Usage: ``python summarize.py > summary.log``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TARGET = 0.361570  # truth.py
RULE_TRUTH = 0.321181  # truth.py
RULE_VS_ALWAYS = -0.040389  # truth.py
frame = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip")
n = len(frame)


def count(mask) -> str:
    hits = int(np.asarray(mask).sum())
    return f"{hits} of {n} ({hits / n:.3f})"


def spread(values) -> str:
    values = pd.Series(values).dropna()
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return (
        f"mean {values.mean():.4f}, median {q50:.4f}, 5% {q05:.4f}, 95% {q95:.4f}, "
        f"min {values.min():.4f}, max {values.max():.4f}"
    )


print(f"draws {n}, seeds {frame['seed'].min()} to {frame['seed'].max()}")
print()
print("Step 3, day-seven shares within discharge strata")
gap0 = frame["share_a10_engaged"] - frame["share_a10_not"]
gap1 = frame["share_a11_engaged"] - frame["share_a11_not"]
print("  engaged minus not, no discharge navigation:", spread(gap0))
print("  engaged minus not, discharge navigation:   ", spread(gap1))
print("  gap positive in both strata:", count((gap0 > 0) & (gap1 > 0)))
print("  pooled gap:", spread(frame["pooled_gap"]))
print("  engagement gap by discharge navigation > 0.5:", count(frame["eng_gap"] > 0.5))
print()
print("Step 6, clustered in-sample interval of always vs never")
covered = (frame["lo"] <= TARGET) & (TARGET <= frame["hi"])
print("  coverage:", count(covered))
print("  bias:", f"{frame['psi'].mean() - TARGET:+.4f}",
      f"(MC SE {frame['psi'].std(ddof=1) / np.sqrt(n):.4f})")
sd = frame["psi"].std(ddof=1)
print(f"  empirical SD {sd:.4f}; mean clustered SE {frame['se'].mean():.4f} "
      f"(ratio {frame['se'].mean() / sd:.3f}); mean SE without teams {frame['se_iid'].mean():.4f} "
      f"(ratio {frame['se_iid'].mean() / sd:.3f})")
iid_covered = (frame["psi_iid"] - 1.959964 * frame["se_iid"] <= TARGET) & (
    TARGET <= frame["psi_iid"] + 1.959964 * frame["se_iid"]
)
print("  coverage without teams:", count(iid_covered))
print("  |psi - truth| < SE:", count((frame["psi"] - TARGET).abs() < frame["se"]))
print()
print("Step 7, shortcuts")
print("  adjusting for engagement below truth:", count(frame["adj_psi"] < TARGET),
      spread(frame["adj_psi"] - TARGET))
print("  baseline only above truth:", count(frame["base_psi"] > TARGET),
      spread(frame["base_psi"] - TARGET))
print("  opposite directions:", count((frame["adj_psi"] < TARGET) & (frame["base_psi"] > TARGET)))
print()
print("Step 8, rule")
print("  rule vs never coverage:",
      count((frame["rule_lo"] <= RULE_TRUTH) & (RULE_TRUTH <= frame["rule_hi"])))
print("  rule vs always coverage:",
      count((frame["rva_lo"] <= RULE_VS_ALWAYS) & (RULE_VS_ALWAYS <= frame["rva_hi"])))
print("  rule vs always interval below zero:", count(frame["rva_hi"] < 0))
print("  rule share at node 2:", spread(frame["rule_share_t2"]))
print()
print("Step 9, diagnostics")
print("  max weight:", spread(frame["max_weight"]))
print("  no truncation at the default bound:", count(frame["share_trunc_max"] == 0))
print("  minimum Kish ratio:", spread(frame["min_eff_ratio"]))
print("  truncation-curve movement in SE:", spread(frame["curve_move_se"]))
print("  scores pass:", count(frame["scores_pass"]))
print("  calibration slopes:", f"{frame['cal_min'].min():.4f} to {frame['cal_max'].max():.4f}")
print("  regression slopes:", f"{frame['reg_min'].min():.4f} to {frame['reg_max'].max():.4f}")
print("  censoring auc node 1:", spread(frame["auc_censoring_1"]))
print("  censoring auc node 2:", spread(frame["auc_censoring_2"]))
both = frame[["auc_censoring_1", "auc_censoring_2"]]
print("  both censoring auc below 0.65:", count((both < 0.65).all(axis=1)))
print("  node 1 lower than node 2:", count(frame["auc_censoring_1"] < frame["auc_censoring_2"]))
print("  both censoring auc below every other auc:",
      count(both.max(axis=1) < frame["auc_min_other"]))
print("  lowest non-censoring auc:", spread(frame["auc_min_other"]))
print()
print("Step 10, covariate-drop moves in standard errors")
moves = frame[["move_age", "move_baseline_readiness", "move_engagement_day7"]]
for column in moves:
    print(f"  {column}:", spread(frame[column]), "positive:", count(frame[column] > 0))
print("  engagement move largest:", count(moves.idxmax(axis=1) == "move_engagement_day7"))
print("  baseline_readiness move smallest:",
      count(moves.idxmin(axis=1) == "move_baseline_readiness"))
print("  engagement move above age move:",
      count(frame["move_engagement_day7"] > frame["move_age"]))

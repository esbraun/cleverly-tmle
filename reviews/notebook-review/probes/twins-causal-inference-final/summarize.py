"""Summarize the committed TWINS sweeps into the numbers the notebook readings quote.

Usage: python summarize.py > summary.log
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

for booster in ("shallow", "current"):
    frame = pd.read_csv(HERE / f"sweep-{booster}.csv")
    ran = frame["sensitivity_ran"].astype(bool)
    print(f"== booster {booster}: {len(frame)} pair samples (seeds {sorted(frame.seed.tolist())})")
    print(f"sensitivity section ran without an exception: {int(ran.sum())}/{len(frame)}")
    failed = frame.loc[~ran, ["seed", "cf_g_min", "error"]]
    if len(failed):
        print("failures:")
        print(failed.to_string(index=False))
    print(f"max |gap| / package SE: {frame.gap_over_se.max():.2e}; below 0.001 on "
          f"{int((frame.gap_over_se < 1e-3).sum())}/{len(frame)}")
    print(f"max |score after targeting| (hand-built): {frame.max_score_after.max():.1e}")
    print(f"nu2 range (ran): {frame.nu2[ran].min():.3f} to {frame.nu2[ran].max():.3f}")
    print(f"min cross-fitted g: {frame.cf_g_min.min():.4f}")
    print(f"benchmark cf_y max: {frame.bench_cf_y[ran].max():.4f}")
    print(f"benchmark cf_d range: {frame.bench_cf_d[ran].min():.4f} to "
          f"{frame.bench_cf_d[ran].max():.4f}")
    print(f"RV range: {frame.rv[ran].min():.4f} to {frame.rv[ran].max():.4f}")
    print(f"propensity calibration slope range: {frame.propensity_cal_slope.min():.4f} to "
          f"{frame.propensity_cal_slope.max():.4f}")
    print(f"outcome calibration slope range: {frame.outcome_cal_slope.min():.4f} to "
          f"{frame.outcome_cal_slope.max():.4f}")
    flagged = frame.calibration_flags.fillna(0) > 0
    print(f"samples with at least one calibration flag: {int(flagged.sum())}/{len(frame)}")
    print()

ltmle = pd.read_csv(HERE / "ltmle-sweep.csv")
print(f"== LTMLE: {len(ltmle)} synthetic draws")
print(f"interval contains the truth: {int(ltmle.covers_truth.sum())}/{len(ltmle)}")
print(f"interval contains the naive contrast: {int(ltmle.covers_naive.sum())}/{len(ltmle)}")
print(f"first 40 draws, contains the naive contrast: {int(ltmle.covers_naive.head(40).sum())}/40")
print(f"mean naive - truth: {(ltmle.naive - ltmle.truth).mean():.4f}")
print(f"mean standard error: {ltmle.se.mean():.4f}")

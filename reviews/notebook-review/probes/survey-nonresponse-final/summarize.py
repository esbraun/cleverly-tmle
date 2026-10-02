"""Summarize ``sweep.csv`` into ``summary.log``, the numbers the survey notebook quotes."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRUE_ATE = 1.2
RESPONDENT_TARGET = 1.427936
frame = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip")
n = len(frame)


def share(column: str) -> str:
    hits = int(frame[column].sum())
    low, high = (
        hits / n - 1.96 * np.sqrt(hits / n * (1 - hits / n) / n),
        hits / n + 1.96 * np.sqrt(hits / n * (1 - hits / n) / n),
    )
    return f"{hits} of {n} ({hits / n:.3f}; Wald 95% {low:.3f} to {high:.3f})"


def spread(column: str) -> str:
    values = frame[column].dropna()
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return (
        f"n {len(values)}, mean {values.mean():.4f}, median {q50:.4f}, "
        f"5% {q05:.4f}, 95% {q95:.4f}, min {values.min():.4f}, max {values.max():.4f}"
    )


full_sd = frame["full_psi"].std()
cc_sd = frame["cc_psi"].std()
lines = [
    f"draws: {n}, seeds {frame['seed'].min()} to {frame['seed'].max()}",
    "",
    "Step 6 fit (missingness declared)",
    f"  covers 1.2: {share('full_covers')}",
    f"  bias {frame['full_psi'].mean() - TRUE_ATE:+.4f} "
    f"(MC SE {full_sd / np.sqrt(n):.4f}); empirical SD {full_sd:.4f}; "
    f"mean SE {frame['full_se'].mean():.4f}; SE/SD {frame['full_se'].mean() / full_sd:.3f}",
    "",
    "Step 7 complete-case fit",
    f"  covers 1.2: {share('cc_covers_truth')}",
    f"  covers 1.428 (population respondent-standardized target): "
    f"{share('cc_covers_population_target')}",
    f"  covers the draw's respondent effect: {share('cc_covers_sample_target')}",
    f"  mean estimate minus 1.427936: {frame['cc_psi'].mean() - RESPONDENT_TARGET:+.4f} "
    f"(MC SE {cc_sd / np.sqrt(n):.4f})",
    f"  mean of estimate minus the draw's respondent effect: "
    f"{(frame['cc_psi'] - frame['sample_target']).mean():+.4f}",
    f"  distance in SE from the draw's respondent effect: {spread('cc_distance_se')}",
    f"  |distance| >= 1.0 SE on {int((frame['cc_distance_se'].abs() >= 1.0).sum())} of {n}",
    f"  shift is most of the gap: {share('shift_is_most')}",
    f"  empirical SD {cc_sd:.4f}; mean SE {frame['cc_se'].mean():.4f}; "
    f"SE/SD {frame['cc_se'].mean() / cc_sd:.3f}",
    "",
    "Step 8 mild-law complete-case fit",
    f"  covers 1.2: {share('mild_covers')}",
    "",
    "Step 9 binary five-fold fits",
    f"  ate covers: {share('box_ate_covers')}",
    f"  rr covers: {share('box_rr_covers')}",
    f"  or covers: {share('box_or_covers')}",
    "",
    "Step 11 tipping gamma, arm_gamma {0: 0, 1: -1}",
    f"  point: {spread('tip_point')}",
    f"  point, largest shift in score units: {spread('tip_point_units')}",
    f"  point, shift / navigation-arm respondent SD: {spread('tip_point_sd_ratio')}",
    f"  use_ci=True: {spread('tip_ci')}",
    f"  use_ci=True, largest shift in score units: {spread('tip_ci_units')}",
    f"  scaler range: {spread('score_range')}",
    f"  corr(tip_point, scaler range) {frame['tip_point'].corr(frame['score_range']):.3f}",
    f"  CV of tip_point {frame['tip_point'].std() / frame['tip_point'].mean():.3f}; "
    f"CV of score-unit shift {frame['tip_point_units'].std() / frame['tip_point_units'].mean():.3f}",
]
(HERE / "summary.log").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print("\n".join(lines))

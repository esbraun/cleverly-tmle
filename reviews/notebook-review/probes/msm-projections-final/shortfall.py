"""Locate the cause of the Step 7 slope coverage shortfall, from sweep.csv and the law.

Reads the 4,000 draws in sweep.csv (seeds 30000 to 33999). Reports the excess kurtosis of the
slope z-statistic, the coverage a normal interval has at the recorded SE / SD ratio, the coverage
by quartile of the smallest fitted propensity, and the share of patients whose true probability
of `high` or `medium` is below 0.01 (the structural law of truth.py, 2e6 covariate draws).

Run: .venv/Scripts/python.exe reviews/notebook-review/probes/msm-projections-final/shortfall.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
TRUE_SLOPE = 0.265714285714  # truth.log: per-contact projection, uniform weight
COEF = np.array([[0.0, 0.0], [0.8, -0.4], [-0.5, 0.8]])


def main():
    sweep = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip")
    z = (sweep["slope"] - TRUE_SLOPE) / sweep["slope_se"]
    print(f"draws: {len(sweep)}")
    print(
        f"slope z-statistic: mean {z.mean():+.3f}, SD {z.std(ddof=1):.3f}, "
        f"excess kurtosis {stats.kurtosis(z):.2f}, skewness {stats.skew(z):+.2f}"
    )
    ratio = sweep["slope_se"].mean() / sweep["slope"].std(ddof=1)
    print(
        f"mean SE / empirical SD {ratio:.3f}; normal coverage at that ratio "
        f"2 Phi(1.96 x {ratio:.3f}) - 1 = {2 * stats.norm.cdf(1.959964 * ratio) - 1:.3f}"
    )
    quartile = pd.qcut(sweep["min_fitted_g"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    for label, group in sweep.groupby(quartile, observed=True):
        print(
            f"smallest fitted g {label} ({group['min_fitted_g'].min():.5f} to "
            f"{group['min_fitted_g'].max():.5f}): slope cover {group['slope_cover'].mean():.3f} "
            f"over {len(group)} draws"
        )
    misses = sweep.loc[~sweep["slope_cover"], "min_fitted_g"]
    median = sweep["min_fitted_g"].median()
    print(
        f"misses with smallest fitted g below the median: {(misses < median).mean():.3f} "
        f"of {len(misses)}"
    )

    rng = np.random.default_rng(20261002)
    w = rng.normal(size=(2_000_000, 2))
    logits = w @ COEF.T
    g = np.exp(logits - logits.max(1, keepdims=True))
    g /= g.sum(1, keepdims=True)
    either = (g[:, 1] < 0.01) | (g[:, 2] < 0.01)
    print(f"share with true g(medium) or g(high) below 0.01: {either.mean():.4f}")


if __name__ == "__main__":
    main()

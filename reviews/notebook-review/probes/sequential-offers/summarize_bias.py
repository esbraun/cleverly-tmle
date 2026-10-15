"""Summarize ``bias_check.csv`` into ``bias_check_summary.log``: bias with its Monte Carlo SE.

Usage: ``python summarize_bias.py > bias_check_summary.log``.  The truths come from
``offers_truth()``.  The bias of a fit is its mean estimate minus the truth over the draws, and
its Monte Carlo standard error is the standard deviation of the estimates over the square root of
the number of draws.  The interval column is the share of draws on which the estimate plus or
minus 1.96 standard errors contains the truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.datasets import OFFER_PLANS, offers_truth

HERE = Path(__file__).resolve().parent
TRUTH = offers_truth()
PLANS = tuple(OFFER_PLANS)
CONTRASTS = ("discount", "shipping", "win_back", "win_back_ship")
frame = pd.read_csv(HERE / "bias_check.csv", float_precision="round_trip")
n = len(frame)
print(f"draws {n}, seeds {frame['seed'].min()} to {frame['seed'].max()}, n = 20000 customers each")


def line(label: str, psi: pd.Series, se: pd.Series, truth: float) -> str:
    bias = psi.mean() - truth
    error = psi.std(ddof=1) / np.sqrt(n)
    covered = ((psi - 1.96 * se <= truth) & (truth <= psi + 1.96 * se)).mean()
    return (
        f"  {label:34s} bias {bias:+.4f} (MC SE {error:.4f}, z {bias / error:+.2f}); "
        f"SD {psi.std(ddof=1):.4f}; interval covers {covered:.3f}"
    )


for method in ("ltmle",):
    print()
    print(f"{method}: the four contrasts against none")
    for contrast in CONTRASTS:
        print(
            line(
                contrast,
                frame[f"{method}_contrast_{contrast}_psi"],
                frame[f"{method}_contrast_{contrast}_se"],
                TRUTH[f"ate_regimen[{contrast} vs none]"],
            )
        )
    print(f"{method}: the five levels")
    for plan in PLANS:
        print(
            line(
                plan,
                frame[f"{method}_level_{plan}_psi"],
                frame[f"{method}_level_{plan}_se"],
                TRUTH[f"ey_regimen[{plan}]"],
            )
        )
print()
print("ltmle: correlation of the four contrast estimates across draws")
print(
    frame[[f"ltmle_contrast_{c}_psi" for c in CONTRASTS]]
    .corr()
    .set_axis(CONTRASTS, axis=0)
    .set_axis(CONTRASTS, axis=1)
    .round(2)
    .to_string()
)

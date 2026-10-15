"""Summarize ``sweep.csv`` into ``summary.log``, the numbers the offers notebook quotes.

Usage: ``python summarize.py > summary.log``.  The truths come from ``offers_truth()``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.datasets import offers_truth

HERE = Path(__file__).resolve().parent
TRUTH = offers_truth()
CONTRASTS = ("discount", "shipping", "win_back", "win_back_ship")
METHODS = ("ltmle", "sdr", "gcomp", "ipw")
PAIRS = (("win_back", "discount"), ("win_back", "shipping"), ("win_back_ship", "win_back"))
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


def truth_of(contrast: str) -> float:
    return TRUTH[f"ate_regimen[{contrast} vs none]"]


print(f"draws {n}, seeds {frame['seed'].min()} to {frame['seed'].max()}")
print()
print("Coverage, bias, and SE ratio of each method and contrast")
for method in METHODS:
    for contrast in CONTRASTS:
        psi, se = frame[f"{method}_{contrast}_psi"], frame[f"{method}_{contrast}_se"]
        low, high = frame[f"{method}_{contrast}_lo"], frame[f"{method}_{contrast}_hi"]
        truth = truth_of(contrast)
        covered = (low <= truth) & (truth <= high)
        print(
            f"  {method:6s} {contrast:14s} coverage {count(covered)}; "
            f"bias {psi.mean() - truth:+.4f} (MC SE {psi.std(ddof=1) / np.sqrt(n):.4f}); "
            f"SD {psi.std(ddof=1):.4f}; mean SE {se.mean():.4f}; ratio {se.mean() / psi.std(ddof=1):.3f}"
        )
print()
print("Largest distance from the LTMLE estimate over the four contrasts")
for method in METHODS[1:]:
    gap = pd.concat(
        [
            (frame[f"{method}_{c}_psi"] - frame[f"ltmle_{c}_psi"]).abs().rename(c)
            for c in CONTRASTS
        ],
        axis=1,
    ).max(axis=1)
    print(f"  {method:6s}", spread(gap))
print()
print("Distance from the truth, mean absolute error over draws and contrasts")
for method in METHODS:
    error = np.mean(
        [(frame[f"{method}_{c}_psi"] - truth_of(c)).abs().mean() for c in CONTRASTS]
    )
    print(f"  {method:6s} {error:.4f}")
print()
print("Paired comparisons")
for better, other in PAIRS:
    label = f"{better}_minus_{other}"
    truth = truth_of(better) - truth_of(other)
    low, high = frame[f"{label}_lo"], frame[f"{label}_hi"]
    ratio = frame[f"{label}_unpaired_se"] / frame[f"{label}_se"]
    print(f"  {better} - {other}: truth {truth:.4f}")
    print("    paired coverage:", count((low <= truth) & (truth <= high)))
    print("    paired SE below unpaired SE:", count(frame[f"{label}_se"] < frame[f"{label}_unpaired_se"]))
    print("    unpaired SE over paired SE:", spread(ratio))
    print("    paired interval excludes zero:", count((low > 0) | (high < 0)))
print()
print("Synthetic deployment check, fit minus pilot")
gap = frame["level_psi"] - frame["pilot_mean"]
gap_se = np.hypot(frame["level_se"], frame["pilot_se"])
z = gap / gap_se
print("  difference:", spread(gap))
print("  difference in standard errors:", spread(z))
print("  |z| < 1.96:", count(z.abs() < 1.959964))
print("  |z| < 2:", count(z.abs() < 2.0))
print()
print("Support of the declared fit")
print("  minimum Kish ratio:", spread(frame["min_kish_ratio"]))
print("  minimum followers at any node:", spread(frame["min_followed"]))
print("  maximum weight:", spread(frame["max_weight"]))
print("  no weight truncated:", count(frame["max_truncated"] == 0))
for name in ("none", "discount", "shipping", "win_back", "win_back_ship"):
    print(f"  followers of {name} at week 3:", spread(frame[f"followed_{name}"]))
print()
print("Logs unavailable: estimated offer mechanism")
for contrast in CONTRASTS:
    truth = truth_of(contrast)
    low, high = frame[f"nologs_{contrast}_lo"], frame[f"nologs_{contrast}_hi"]
    shift = frame[f"nologs_{contrast}_psi"] - frame[f"ltmle_{contrast}_psi"]
    print(
        f"  {contrast:14s} coverage {count((low <= truth) & (truth <= high))}; "
        f"estimate minus declared fit: {spread(shift)}"
    )
print("  truncation-curve moves, in SEs of the fit:")
for contrast in CONTRASTS:
    for bound in (0.02, 0.05, 0.1):
        move = frame[f"trunc_{contrast}_{bound}"] / frame[f"nologs_{contrast}_se"]
        print(f"    {contrast:14s} bound {bound}: max |move| {move.abs().max():.3f}; mean {move.mean():+.3f}")
print("  truncated score cells at 0.1:", spread(sum(frame[f"cells_{c}_0.1"] for c in CONTRASTS)))
print()
print("Benchmark: drop engagement from the history of the static plans")
for contrast in ("discount", "shipping"):
    move = frame[f"drop_{contrast}_move"]
    in_se = move / frame[f"drop_{contrast}_se"]
    print(f"  {contrast}: move", spread(move))
    print(f"  {contrast}: move in SE", spread(in_se))
    print(f"  {contrast}: |move| below one SE:", count(in_se.abs() < 1))

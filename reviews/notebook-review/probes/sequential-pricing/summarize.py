"""Summarize ``sweep.csv`` into ``summary.log``, the numbers the pricing notebook quotes.

Usage: ``python summarize.py > summary.log``. The truths come from ``pricing_truth()``, and the
last block compares them with the independent transcription in ``truth.py``. The block on the
regression limits draws one large frame, with the size and the seed it prints.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.datasets import make_sequential_pricing, pricing_truth

HERE = Path(__file__).resolve().parent
PLANS = ("price_up", "price_down")
LADDER = ("up_6", "price_up", "up_25")
LIMIT_N, LIMIT_SEED = 400_000, 3_000_001
frame = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip")
n = len(frame)


def count(mask) -> str:
    hits = int(np.asarray(mask).sum())
    return f"{hits} of {n} ({hits / n:.3f})"


def spread(values) -> str:
    values = pd.Series(values).dropna()
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    return (
        f"mean {values.mean():.4f}, SD {values.std(ddof=1):.4f}, median {q50:.4f}, "
        f"5% {q05:.4f}, 95% {q95:.4f}, min {values.min():.4f}, max {values.max():.4f}"
    )


print(f"draws {n}, seeds {frame['seed'].min()} to {frame['seed'].max()}, n = 10000, complete data")
print(f"seconds per draw: mean {frame['seconds'].mean():.1f}")
print()
print("Coverage, bias, and SE ratio of LTMLE")
for prefix, outcome in (("y", "renewal"), ("r", "revenue")):
    for plan in PLANS:
        psi, se = frame[f"{prefix}_{plan}_psi"], frame[f"{prefix}_{plan}_se"]
        low, high = frame[f"{prefix}_{plan}_lo"], frame[f"{prefix}_{plan}_hi"]
        truth = frame[f"{prefix}_{plan}_truth"].iloc[0]
        covered = (low <= truth) & (truth <= high)
        print(
            f"  {outcome:8s} {plan:11s} coverage {count(covered)}; "
            f"bias {psi.mean() - truth:+.4f} (MC SE {psi.std(ddof=1) / np.sqrt(n):.4f}); "
            f"SD {psi.std(ddof=1):.4f}; mean SE {se.mean():.4f}; "
            f"ratio {se.mean() / psi.std(ddof=1):.3f}"
        )
print()
print("Support label of every fit")
for column in sorted(c for c in frame.columns if c.endswith("_label")):
    print(f"  {column}: supported on {count(frame[column] == 'supported')}")
print()
print("Warnings")
for column in ("renewal_warnings", "revenue_warnings", "ladder_warnings"):
    warned = frame[column].fillna("")
    print(f"  {column}: SparseInfluenceWarning on {count(warned.str.contains('SparseInfluence'))}")
    others = sorted(
        {w for cell in warned for w in cell.split(";") if w and w != "SparseInfluenceWarning"}
    )
    print(f"  {column}: other warnings {others}")
print(
    "  revenue price_down SparseInfluenceWarning, seed 1000:",
    "SparseInfluenceWarning" in str(frame.loc[frame["seed"] == 1000, "revenue_warnings"].iloc[0]),
)
print()
print("The revenue cut: the interval and the draws that span zero")
cut_low, cut_high = frame["r_price_down_lo"], frame["r_price_down_hi"]
print("  revenue price_down interval contains zero:", count((cut_low < 0) & (cut_high > 0)))
rise_low, rise_high = frame["r_price_up_lo"], frame["r_price_up_hi"]
print("  revenue price_up interval contains zero:", count((rise_low < 0) & (rise_high > 0)))
print()
print("Failure mode: the logistic regression with the plan's prices, against LTMLE")
for plan in PLANS:
    truth = frame[f"y_{plan}_truth"].iloc[0]
    ltmle, adjust, omit = frame[f"y_{plan}_psi"], frame[f"adjust_{plan}"], frame[f"omit_{plan}"]
    low, high = frame[f"y_{plan}_lo"], frame[f"y_{plan}_hi"]
    print(f"  {plan}: truth {truth:.4f}")
    print(f"    adjust: {spread(adjust)}")
    print(f"    omit: {spread(omit)}")
    print(
        "    adjust further from the truth than LTMLE:",
        count((adjust - truth).abs() > (ltmle - truth).abs()),
    )
    print("    adjust outside the LTMLE interval:", count((adjust < low) | (adjust > high)))
    print("    omit outside the LTMLE interval:", count((omit < low) | (omit > high)))
    print("    adjust between LTMLE and omit:", count((adjust - ltmle) * (adjust - omit) < 0))
print()
print("Node-3 support of the renewal fit")
for plan in PLANS:
    print(f"  {plan}: max weight {spread(frame[f'maxw_{plan}'])}")
    print(f"  {plan}: effective n over n {spread(frame[f'essr_{plan}'])}")
    print(f"  {plan}: no weight truncated: {count(frame[f'trunc_{plan}'] == 0)}")
print()
print("Factor ladder: a 6%, a 12%, and a 25% rise, node 3")
for plan in LADDER:
    low, high = frame[f"ladder_{plan}_lo"], frame[f"ladder_{plan}_hi"]
    truth = frame[f"ladder_{plan}_truth"].iloc[0]
    psi = frame[f"ladder_{plan}_psi"]
    print(
        f"  {plan:9s} coverage {count((low <= truth) & (truth <= high))}; "
        f"bias {psi.mean() - truth:+.4f} (MC SE {psi.std(ddof=1) / np.sqrt(n):.4f}); "
        f"interval width {spread(high - low)}"
    )
    print(f"  {plan:9s} max weight {spread(frame[f'ladder_{plan}_maxw'])}")
    print(f"  {plan:9s} effective n over n {spread(frame[f'ladder_{plan}_essr'])}")
    print(f"  {plan:9s} no weight truncated: {count(frame[f'ladder_{plan}_trunc'] == 0)}")
width = {p: frame[f"ladder_{p}_hi"] - frame[f"ladder_{p}_lo"] for p in LADDER}
print(
    "  width increases with the factor:",
    count((width["up_6"] < width["price_up"]) & (width["price_up"] < width["up_25"])),
)
weight = {p: frame[f"ladder_{p}_maxw"] for p in LADDER}
print(
    "  max weight increases with the factor:",
    count((weight["up_6"] < weight["price_up"]) & (weight["price_up"] < weight["up_25"])),
)
essr = {p: frame[f"ladder_{p}_essr"] for p in LADDER}
print(
    "  effective n decreases with the factor:",
    count((essr["up_6"] > essr["price_up"]) & (essr["price_up"] > essr["up_25"])),
)
print()
print(f"Limits of the regressions, one draw of n = {LIMIT_N}, seed {LIMIT_SEED}")
big, truths = make_sequential_pricing(LIMIT_N, seed=LIMIT_SEED)
from sklearn.linear_model import LogisticRegression  # noqa: E402

prices = [f"price_{k}" for k in (1, 2, 3)]
for label, columns in (
    ("adjust", ["usage", "engaged_1", "engaged_2", "engaged_3", *prices]),
    ("omit", ["usage", *prices]),
):
    model = LogisticRegression(max_iter=3000, C=1e6).fit(big[columns], big["renewed"])
    for plan in PLANS:
        moved = big[columns].copy()
        for node in (1, 2, 3):
            price = big[f"price_{node}"].to_numpy()
            engaged = big[f"engaged_{node}"].to_numpy()
            if plan == "price_up":
                moved[f"price_{node}"] = np.where(
                    price < (8 + 2 * engaged) / 1.12, 1.12 * price, price
                )
            else:
                moved[f"price_{node}"] = np.where(
                    price < (4 + 2 * engaged) / 0.85, price, 0.85 * price
                )
        limit = float(
            np.mean(model.predict_proba(moved)[:, 1] - model.predict_proba(big[columns])[:, 1])
        )
        exact = truths[f"ate_regimen[{plan} vs natural_prices]"]
        print(
            f"  {label} {plan}: limit {limit:.4f}, truth {exact:.4f}, gap {abs(limit - exact):.4f}"
        )
print()
print("The independent transcription truth.py against pricing_truth()")
import truth  # noqa: E402

independent = truth.truths()
exact = pricing_truth()
print("  same keys:", set(independent) == set(exact))
print(f"  largest difference: {max(abs(independent[k] - exact[k]) for k in exact):.1e}")
print()
print("Seed 1000, the draw the page shows")
print(frame[frame["seed"] == 1000].T.to_string())

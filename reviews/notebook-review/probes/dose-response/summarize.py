"""Summarize sweep.csv: per-pair coverage and support, and how often each page relation holds.

Reads sweep.csv (or sweep-jitter1.csv) and truth.csv beside this script (written by sweep.py) and prints, for each of
the 12 pairs, the bias, the SE ratio (mean SE over the SD of the estimates), the pointwise
coverage, the share of draws below the support floor, and the medians of the clever-covariate
and influence-curve Kish sizes; then the joint coverage of the band and the share of draws on
which each relation that the page narrates about its own draw holds.

Run: python reviews/notebook-review/probes/dose-response/summarize.py
(JITTER=1.0 in the environment reads sweep-jitter1.csv, as sweep.py does.)
"""

from pathlib import Path

from sweep import SWEEP

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def wilson(successes, n, z=2.576):
    """The 99% Wilson interval for a binomial share."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    sweep = pd.read_csv(SWEEP, float_precision="round_trip")
    truth = pd.read_csv(HERE / "truth.csv", float_precision="round_trip")
    sweep = sweep.merge(truth, on=["bandwidth", "dose"])
    sweep["cover"] = (sweep["lower"] <= sweep["true"]) & (sweep["true"] <= sweep["upper"])
    sweep["band_cover"] = (sweep["band_lower"] <= sweep["true"]) & (
        sweep["true"] <= sweep["band_upper"]
    )
    draws = sweep["seed"].nunique()
    print(f"draws: {draws} (seeds {sweep['seed'].min()} to {sweep['seed'].max()})")
    print()

    def per_pair(group):
        return pd.Series(
            {
                "true": group["true"].iloc[0],
                "bias": group["estimate"].mean() - group["true"].iloc[0],
                "se_ratio": group["se"].mean() / group["estimate"].std(ddof=1),
                "cover": group["cover"].mean(),
                "cover_above_floor": group.loc[~group["low_support"], "cover"].mean(),
                "below_floor": group["low_support"].mean(),
                "kish_median": group["kish"].median(),
                "ic_kish_median": group["ic_kish"].median(),
                "ic_kish_below_25": (group["ic_kish"] < 25).mean(),
            }
        )

    pairs = sweep.groupby(["bandwidth", "dose"]).apply(per_pair, include_groups=False)
    print("per pair:")
    print(pairs.round(3).to_string())
    print()

    by_draw = sweep.groupby("seed")
    band_joint = by_draw["band_cover"].all()
    low, high = wilson(band_joint.sum(), draws)
    print(f"band covers all 12 pairs: {band_joint.mean():.3f} (99% interval {low:.3f} to {high:.3f})")
    above = sweep[~sweep["low_support"]].groupby("seed")["band_cover"].all()
    print(f"band covers every pair above the floor: {above.mean():.3f}")
    print()

    def pair(frame, h, dose):
        return frame[(frame["bandwidth"] == h) & (frame["dose"] == dose)].iloc[0]

    relations = []
    for seed, frame in by_draw:
        flagged = sorted(
            zip(frame.loc[frame["low_support"], "bandwidth"], frame.loc[frame["low_support"], "dose"])
        )
        rise = {h: [pair(frame, h, d)["estimate"] for d in (5, 10, 15, 20)] for h in (2.5, 5.0)}
        width = {
            (h, d): pair(frame, h, d)["upper"] - pair(frame, h, d)["lower"]
            for h in (2.5, 5.0)
            for d in (15, 25)
        }
        relations.append(
            {
                "seed": seed,
                "every interval covers": frame["cover"].all(),
                "every band covers": frame["band_cover"].all(),
                "band separates 5% from 20% at bandwidth 5": pair(frame, 5.0, 5)["band_upper"]
                < pair(frame, 5.0, 20)["band_lower"],
                "band separates 5% from 20% at bandwidth 2.5": pair(frame, 2.5, 5)["band_upper"]
                < pair(frame, 2.5, 20)["band_lower"],
                "band separates 5% from 20% at both": (
                    pair(frame, 5.0, 5)["band_upper"] < pair(frame, 5.0, 20)["band_lower"]
                )
                and (pair(frame, 2.5, 5)["band_upper"] < pair(frame, 2.5, 20)["band_lower"]),
                "band separates 5% from 15% at bandwidth 5": pair(frame, 5.0, 5)["band_upper"]
                < pair(frame, 5.0, 15)["band_lower"],
                "both series rise from 5% to 20%": all(
                    np.all(np.diff(values) > 0) for values in rise.values()
                ),
                "rise 5% to 20% at bandwidth 5 is 0.12 to 0.14": 0.12
                < rise[5.0][3] - rise[5.0][0]
                < 0.14,
                "narrow band at 30% runs below 0 and above 1": pair(frame, 2.5, 30)["band_lower"]
                < 0
                < 1
                < pair(frame, 2.5, 30)["band_upper"],
                "flagged pairs are exactly the page's three": flagged
                == [(2.5, 30.0), (5.0, 25.0), (5.0, 30.0)],
                "both 30% pairs flagged": pair(frame, 2.5, 30)["low_support"]
                and pair(frame, 5.0, 30)["low_support"],
                "wide window at 25% flagged": bool(pair(frame, 5.0, 25)["low_support"]),
                "narrow window at 25% flagged": bool(pair(frame, 2.5, 25)["low_support"]),
                "some pair at 20% or below flagged": bool(
                    frame.loc[frame["dose"] <= 20, "low_support"].any()
                ),
                "wide interval narrower at 15%": width[(5.0, 15)] < width[(2.5, 15)],
                "wide interval wider at 25%": width[(5.0, 25)] > width[(2.5, 25)],
                "wide Kish smaller at 25%": pair(frame, 5.0, 25)["kish"]
                < pair(frame, 2.5, 25)["kish"],
                "quadrature shift below 0.05": frame["quadrature_shift"].iloc[0] < 0.05,
                "equations solved passed": frame["check_equations"].iloc[0] == "passed",
                "overlap needs attention": frame["check_overlap"].iloc[0] == "warning",
                "models check completed": frame["check_models"].iloc[0] == "completed",
                "influence Kish below 25 at 20%, both bandwidths": (
                    pair(frame, 2.5, 20)["ic_kish"] < 25
                )
                and (pair(frame, 5.0, 20)["ic_kish"] < 25),
            }
        )
    relations = pd.DataFrame(relations).set_index("seed")
    print("share of draws on which each relation holds:")
    for name, share in relations.mean().items():
        print(f"  {share:.3f}  {name}")
    print()

    # Does a small influence-curve Kish size mark the intervals that miss?
    print("pointwise coverage at doses 15 to 25 by influence-curve Kish size:")
    middle = sweep[(sweep["dose"] >= 15) & (sweep["dose"] <= 25)]
    bins = pd.cut(middle["ic_kish"], [0, 10, 25, 50, 100, np.inf])
    print(
        middle.groupby(bins, observed=True)["cover"]
        .agg(["mean", "size"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()

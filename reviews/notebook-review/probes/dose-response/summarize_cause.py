"""Compare the cause arms (cause.py) with the page's own arm (sweep.csv) on the same seeds.

Prints, for each arm and pair, the bias, the SE ratio (mean reported SE over the SD of the
estimates) and the pointwise coverage, on the seeds every arm holds. Also prints how often the
30-40% band renews least in the naive table on the redrawn engine (no fit; seeds 20000-20199).

Run: python reviews/notebook-review/probes/dose-response/summarize_cause.py > cause-summary.log
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sweep import simulate_renewals

HERE = Path(__file__).parent
ARMS = ["true_outcome", "more_bins", "flexible_density"]


def main():
    truth = pd.read_csv(HERE / "truth.csv", float_precision="round_trip")
    page = pd.read_csv(HERE / "sweep.csv", float_precision="round_trip").rename(
        columns={"se": "std_error"}
    )
    frames = {"page": page}
    for arm in ARMS:
        path = HERE / f"cause-{arm}.csv"
        if path.exists():
            frames[arm] = pd.read_csv(path, float_precision="round_trip")
    seeds = set.intersection(*(set(frame["seed"]) for frame in frames.values()))
    print(f"paired seeds: {len(seeds)}")
    for arm, frame in frames.items():
        frame = frame[frame["seed"].isin(seeds)].merge(truth, on=["bandwidth", "dose"])
        frame["cover"] = (frame["lower"] <= frame["true"]) & (frame["true"] <= frame["upper"])
        rows = frame.groupby(["bandwidth", "dose"]).apply(
            lambda g: pd.Series(
                {
                    "bias": g["estimate"].mean() - g["true"].iloc[0],
                    "se_ratio": g["std_error"].mean() / g["estimate"].std(ddof=1),
                    "cover": g["cover"].mean(),
                }
            ),
            include_groups=False,
        )
        print()
        print(f"arm: {arm}")
        print(rows.round(3).to_string())
    print()
    least = []
    for seed in range(20000, 20200):
        frame = simulate_renewals(4_000, seed, jitter=1.0)
        bands = pd.cut(frame["discount"], [0, 10, 20, 30, 40])
        share = frame.groupby(bands, observed=True)["renewed"].mean()
        least.append(share.iloc[-1] == share.min())
    print(f"redrawn engine: the 30-40% band renews least on {np.mean(least):.3f} of 200 draws")


if __name__ == "__main__":
    main()

"""Seed sweep of the dose-response page configuration (n = 4,000, the page's learners and fit).

Each draw refits the page's curve on ``simulate_renewals(n=4_000, seed=s)`` through
``CausalStudy``: six doses at bandwidths 2.5 and 5 on the range 2 to 32, the default eight
quadrature nodes per piece, shallow gradient boosting for the outcome, a 12-bin logistic hazard
for the discount, five folds and ``random_state=0``. It records, for each of the 12 pairs, the
estimate, its pointwise interval and band, the clever-covariate Kish size and support flag, and
the influence-curve Kish size (``diagnostics.influence()``), and for each draw the three lines of
``result.validate()``.

The truth is the smoothed curve of the structural equations, integrated on a 3,001-point grid
over a 1,000,000-row covariate draw (seed 20261126), not a value the library computes.

The truth does not depend on the engine, so one truth.csv serves both engines.

Run at cleverly-tmle e671bb7c (cleverly 0.2.8) on four cores:
python reviews/notebook-review/probes/dose-response/sweep.py 20000 20200 4
     JITTER=1.0 python reviews/notebook-review/probes/dose-response/sweep.py 20000 20200 4
Writes sweep.csv, or sweep-jitter1.csv with JITTER=1.0 (one row per draw and pair, resumable), and truth.csv beside this script; then
run summarize.py.
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, DoseResponseCurve, PointTreatment
from cleverly.dose_response import DoseResponse

HERE = Path(__file__).parent
CURVE = DoseResponse(doses=(5, 10, 15, 20, 25, 30), bandwidths=(2.5, 5), support=(2, 32))


def renewal_chance(discount, usage, tenure):
    lift = 1.6 * (1.0 - np.exp(-discount / 8.0))
    return 1.0 / (1.0 + np.exp(-(-1.2 + 0.9 * usage + 0.4 * tenure + lift)))


#: The SD of the engine's random jitter. The page as reviewed used 0.8; JITTER=1.0 in the
#: environment selects the redrawn page's engine and writes sweep-jitter1.csv instead.
JITTER = float(os.environ.get("JITTER", "0.8"))
SUFFIX = "" if JITTER == 0.8 else f"-jitter{JITTER:g}"


SWEEP = HERE / f"sweep{SUFFIX}.csv"
TRUTH = HERE / "truth.csv"


def simulate_renewals(n, seed, jitter=None):
    jitter = JITTER if jitter is None else jitter
    rng = np.random.default_rng(seed)
    usage = rng.normal(size=n)
    tenure = rng.normal(size=n)
    score = -0.3 - 0.45 * usage - 0.25 * tenure + rng.normal(scale=jitter, size=n)
    discount = 40.0 / (1.0 + np.exp(-score))
    renewed = rng.binomial(1, renewal_chance(discount, usage, tenure))
    return pd.DataFrame(
        {"usage": usage, "tenure": tenure, "discount": discount, "renewed": renewed}
    )


def smoothed_truth():
    rng = np.random.default_rng(20261126)
    usage = rng.normal(size=1_000_000)
    tenure = rng.normal(size=1_000_000)
    grid = np.linspace(*CURVE.support, 3001)
    theta = np.array([renewal_chance(a, usage, tenure).mean() for a in grid])
    rows = [
        {"bandwidth": h, "dose": a0, "true": np.trapezoid(CURVE.weight(h, a0, grid) * theta, grid)}
        for h, a0 in CURVE.pairs
    ]
    return pd.DataFrame(rows)


def one_draw(seed):
    frame = simulate_renewals(4_000, seed)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="renewed",
            treatment="discount",
            adjustment=("usage", "tenure"),
            treatment_kind="continuous",
        ),
    )
    effect = study.identify(DoseResponseCurve(CURVE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = effect.estimate(
            outcome_learner=HistGradientBoostingClassifier(
                max_depth=2, max_iter=100, random_state=0
            ),
            treatment_learner=LogisticRegression(max_iter=1000),
            density_bins=12,
            n_folds=5,
            random_state=0,
        )
    table = result.to_frame()
    influence = result.diagnostics.influence().to_frame().set_index("name")
    checks = {item.name: item.status.value for item in result.validate().items}
    rows = []
    for (h, a0), name in zip(CURVE.pairs, CURVE.names, strict=True):
        row = table[(table["bandwidth"] == h) & (table["dose"] == a0)].iloc[0]
        rows.append(
            {
                "seed": seed,
                "bandwidth": h,
                "dose": a0,
                "estimate": row["estimate"],
                "se": row["std_error"],
                "lower": row["lower"],
                "upper": row["upper"],
                "band_lower": row["band_lower"],
                "band_upper": row["band_upper"],
                "kish": row["kish"],
                "low_support": bool(row["low_support"]),
                "ic_kish": influence.loc[name, "kish"],
                "ic_kurtosis": influence.loc[name, "kurtosis"],
                "quadrature_shift": result.quadrature_shift,
                "band_critical": result.band.critical_value,
                "check_equations": checks.get("score_equations"),
                "check_overlap": checks.get("support"),
                "check_models": checks.get("nuisance_models"),
            }
        )
    return rows


def main():
    start, stop, workers = (int(x) for x in sys.argv[1:4])
    if not TRUTH.exists():
        smoothed_truth().to_csv(TRUTH, index=False, float_format="%.12g")
    done = set()
    if SWEEP.exists():
        done = set(pd.read_csv(SWEEP)["seed"].unique())
    todo = [s for s in range(start, stop) if s not in done]
    print(f"draws to run: {len(todo)} (already done: {len(done)})", flush=True)
    with Pool(workers) as pool:
        for count, rows in enumerate(pool.imap_unordered(one_draw, todo), start=1):
            frame = pd.DataFrame(rows)
            frame.to_csv(SWEEP, mode="a", header=not SWEEP.exists(), index=False)
            if count % 20 == 0:
                print(f"{count} of {len(todo)} draws", flush=True)


if __name__ == "__main__":
    main()

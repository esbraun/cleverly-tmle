"""Locate the cause of the under-coverage at dose 5 in sweep.csv, by swapping one nuisance model.

Each arm refits the page's configuration (sweep.py) on the same seeds with one change:

- ``true_outcome``: the outcome model is a logistic regression on the true form of the
  structural equation, (1 - exp(-discount / 8), usage, tenure), so the outcome model is right;
- ``more_bins``: the page's learners with ``density_bins=24`` instead of 12, a finer dose density.

The page's own arm is sweep.csv. Run:
python reviews/notebook-review/probes/dose-response/cause.py <arm> 20000 20200 4
Writes cause-<arm>.csv beside this script; summarize_cause.py compares the arms.
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from cleverly import CausalStudy, DoseResponseCurve, PointTreatment

from sweep import CURVE, simulate_renewals

HERE = Path(__file__).parent


def true_form(design):
    # The outcome learner receives (discount, usage, tenure), discount first.
    design = np.asarray(design, dtype=float)
    return np.column_stack([1.0 - np.exp(-design[:, 0] / 8.0), design[:, 1], design[:, 2]])


def learners(arm):
    boosted = HistGradientBoostingClassifier(max_depth=2, max_iter=100, random_state=0)
    if arm == "true_outcome":
        outcome = make_pipeline(FunctionTransformer(true_form), LogisticRegression(C=1e6))
        return outcome, 12
    if arm == "more_bins":
        return boosted, 24
    raise SystemExit(f"unknown arm {arm!r}")


def one_draw(job):
    arm, seed = job
    outcome, bins = learners(arm)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.identify(DoseResponseCurve(CURVE)).estimate(
            outcome_learner=outcome,
            treatment_learner=LogisticRegression(max_iter=1000),
            density_bins=bins,
            n_folds=5,
            random_state=0,
        )
    table = result.to_frame()
    table.insert(0, "seed", seed)
    return table[["seed", "bandwidth", "dose", "estimate", "std_error", "lower", "upper", "kish"]]


def main():
    arm = sys.argv[1]
    start, stop, workers = (int(x) for x in sys.argv[2:5])
    out = HERE / f"cause-{arm}.csv"
    done = set(pd.read_csv(out)["seed"].unique()) if out.exists() else set()
    todo = [(arm, s) for s in range(start, stop) if s not in done]
    print(f"{arm}: draws to run {len(todo)}", flush=True)
    with Pool(workers) as pool:
        for count, table in enumerate(pool.imap_unordered(one_draw, todo), start=1):
            table.to_csv(out, mode="a", header=not out.exists(), index=False)
            if count % 20 == 0:
                print(f"{arm}: {count} of {len(todo)}", flush=True)


if __name__ == "__main__":
    main()

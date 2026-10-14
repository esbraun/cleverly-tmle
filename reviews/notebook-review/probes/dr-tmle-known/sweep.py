"""Repeated-draw probe of the DR-TMLE tutorial's known-mechanism design. Not a registered study.

Command (from the repository root):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        PYTHONPATH=. .venv/bin/python reviews/notebook-review/probes/dr-tmle-known/sweep.py

Seeds: data seeds 1 to 200, one draw of ``navigation_data(n=2_000, seed=s)`` each.  Every fit
uses ``Runtime(random_state=s, n_jobs=1)``.  Two worker processes.

What it measures: the tutorial's Steps 6 and 7 on each draw.  The recorded offer probabilities
are the law's true propensity, declared with ``PointTreatment(treatment_probabilities=)``.  The
outcome regression is the tutorial's main-effects linear model, fitted in sample, and the reduced
regressions are the tutorial's linear and logistic models.  For the ordinary TMLE and for DR-TMLE
at the default guard it reports the mean error against the true ATE, the standard deviation of
the estimate, the mean standard error over that standard deviation, and the count of 95%
intervals that contain the true ATE.  It also counts the draws on which the DR-TMLE assessment
flags any row for attention.

Output: ``sweep.csv`` (one row per draw) and the summary on standard output, which
``summary.log`` keeps.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEEDS = range(1, 201)
COVARIATES = ["discharge_risk", "prior_utilization", "medication_burden", "age"]


def one_draw(seed: int) -> dict[str, float]:
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from cleverly import (
        ATE,
        CausalStudy,
        CrossFitting,
        DRTMLEMethod,
        ModelSpec,
        PointTreatment,
        Runtime,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import navigation_data, nonlinear_bounded_dgp

    frame, truth = navigation_data(n=2_000, seed=seed)
    g0 = nonlinear_bounded_dgp().propensity(frame[COVARIATES].to_numpy(dtype=float))
    frame["offer_probability"] = g0
    frame["usual_probability"] = 1.0 - g0
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=tuple(COVARIATES),
            treatment_probabilities={0: "usual_probability", 1: "offer_probability"},
        ),
    )
    effect = study.identify(ATE(reference=0))
    shared = {
        "models": ModelSpec(outcome_learner=LinearRegression(n_jobs=1)),
        "cross_fitting": CrossFitting(enabled=False),
        "targeting": Targeting(q_bounds=(0.0, 1.0)),
        "runtime": Runtime(random_state=seed, n_jobs=1),
    }
    ordinary = effect.estimate(method=TMLEMethod(**shared))["ate"]
    guarded_fit = effect.estimate(
        method=DRTMLEMethod(
            **shared,
            reduced_outcome_learner=LinearRegression(n_jobs=1),
            reduced_treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
        )
    )
    guarded = guarded_fit["ate"]
    attention = guarded_fit.assess().attention
    ate = truth["ate"]
    return {
        "seed": seed,
        "truth": ate,
        "tmle_psi": ordinary.psi,
        "tmle_se": ordinary.std_error,
        "tmle_cover": float(ordinary.ci[0] <= ate <= ordinary.ci[1]),
        "drtmle_psi": guarded.psi,
        "drtmle_se": guarded.std_error,
        "drtmle_cover": float(guarded.ci[0] <= ate <= guarded.ci[1]),
        "drtmle_attention": float(len(attention) > 0),
    }


def main() -> None:
    with Pool(2) as pool:
        rows = pool.map(one_draw, SEEDS, chunksize=5)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "sweep.csv", index=False)
    n = len(table)
    print(f"draws: {n}; n = 2,000 per draw")
    for name, label in (("tmle", "ordinary TMLE"), ("drtmle", "DR-TMLE")):
        error = table[f"{name}_psi"] - table["truth"]
        sd = float(np.std(table[f"{name}_psi"], ddof=1))
        se = float(table[f"{name}_se"].mean())
        covered = int(table[f"{name}_cover"].sum())
        print(
            f"{label:14s} mean error {error.mean():+.4f}  sd {sd:.4f}  "
            f"mean se {se:.4f}  mean se / sd {se / sd:.2f}  "
            f"intervals containing the ATE {covered} of {n}"
        )
    flagged = int(table["drtmle_attention"].sum())
    print(f"DR-TMLE draws with an assessment row needing attention: {flagged} of {n}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

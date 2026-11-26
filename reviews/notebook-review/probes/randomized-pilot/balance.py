"""Seed sweep of the randomized-pilot page's balance check, 200 draws at the page's size.

Run from the repository root:
    python reviews/notebook-review/probes/randomized-pilot/balance.py \
        > reviews/notebook-review/probes/randomized-pilot/balance.log

Each draw refits the page's in-sample TMLE with the recorded chances and reads
``sensitivity.covariate_balance``: the standardized difference of each covariate after weighting
by the recorded chances. The draws follow the offers as recorded, so every difference left after
weighting is a chance difference. The log counts how often one crosses the library's 0.1 line,
and how often the largest reaches the page's 0.118 (``medication_burden`` on seed 21).
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly import ATE, CausalStudy, CrossFitting, ModelSpec, PointTreatment, Runtime, TMLEMethod
from cleverly.datasets import make_randomized_pilot
from cleverly.sensitivity import covariate_balance

N = 3_000
DRAWS = 200
FIRST_SEED = 1_000
PAGE_LARGEST = 0.118
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")
CHANCES = {0: "chance_no_offer", 1: "chance_offer"}


def one(seed: int) -> dict[str, float]:
    warnings.simplefilter("ignore")
    frame, _ = make_randomized_pilot(n=N, seed=seed)
    design = PointTreatment(
        outcome="transition_success",
        treatment="transition_navigation",
        adjustment=COVARIATES,
        treatment_probabilities=CHANCES,
    )
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=LogisticRegression(max_iter=1000)),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )
    result = CausalStudy(frame, design=design).identify(ATE(reference=0)).estimate(method=method)
    table = covariate_balance(result).to_frame().set_index("covariate")
    return {"seed": seed, **table["smd_weighted"].to_dict()}


def main() -> None:
    with Pool(4) as pool:
        table = pd.DataFrame(pool.map(one, range(FIRST_SEED, FIRST_SEED + DRAWS)))
    weighted = table[list(COVARIATES)].abs()
    print(f"n = {N}; draws = {len(table)}; weighted standardized differences, absolute")
    for name in COVARIATES:
        column = weighted[name]
        print(
            f"{name:18s} SD {table[name].std():.4f}  above 0.1 on {int((column > 0.1).sum())} of "
            f"{len(table)}  largest {column.max():.3f}"
        )
    largest = weighted.max(axis=1)
    print(
        f"any covariate above 0.1 on {int((largest > 0.1).sum())} of {len(table)} draws; "
        f"median largest {np.median(largest):.3f}"
    )
    print(
        f"largest at or above {PAGE_LARGEST} on {int((largest >= PAGE_LARGEST).sum())} of "
        f"{len(table)} draws"
    )


if __name__ == "__main__":
    main()

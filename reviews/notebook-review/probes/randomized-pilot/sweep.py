"""Seed sweep of the randomized-pilot page's fits, 200 draws at the page's size.

Run from the repository root:
    python reviews/notebook-review/probes/randomized-pilot/sweep.py \
        > reviews/notebook-review/probes/randomized-pilot/sweep.log

Each draw refits the page's configurations with the page's learners: TMLE with the recorded
chances in sample and with out-of-fold predictions, DR-TMLE with the recorded chances in sample,
adaptive IPTW with the recorded chances and out-of-fold predictions, and TMLE with a fitted
(main-terms logistic) treatment model in sample. The CSV holds every estimate.
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    AdaptiveIPTWMethod,
    CausalStudy,
    CrossFitting,
    DRTMLEMethod,
    ModelSpec,
    PointTreatment,
    Runtime,
    TMLEMethod,
)
from cleverly.datasets import make_randomized_pilot, randomized_pilot_truth

N = 3_000
DRAWS = 200
FIRST_SEED = 1_000
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")
CHANCES = {0: "chance_no_offer", 1: "chance_offer"}
HERE = Path(__file__).parent
FITS = ("tmle_known_in", "tmle_known_xf", "drtmle_known_in", "aiptw_known_xf", "tmle_fitted_g_in")


def logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def one(seed: int) -> dict[str, float]:
    warnings.simplefilter("ignore")
    frame, truth = make_randomized_pilot(n=N, seed=seed)

    def effect(known: bool):
        design = PointTreatment(
            outcome="transition_success",
            treatment="transition_navigation",
            adjustment=COVARIATES,
            **({"treatment_probabilities": CHANCES} if known else {}),
        )
        return CausalStudy(frame, design=design).identify(ATE(reference=0))

    models = ModelSpec(outcome_learner=logistic(), treatment_learner=logistic())
    runtime = Runtime(random_state=seed, n_jobs=1)
    in_sample = CrossFitting(enabled=False)
    five = CrossFitting(n_folds=5)
    methods = {
        "tmle_known_in": (
            True,
            TMLEMethod(models=models, cross_fitting=in_sample, runtime=runtime),
        ),
        "tmle_known_xf": (True, TMLEMethod(models=models, cross_fitting=five, runtime=runtime)),
        "drtmle_known_in": (
            True,
            DRTMLEMethod(
                models=models,
                cross_fitting=in_sample,
                runtime=runtime,
                reduced_outcome_learner=LinearRegression(),
                reduced_treatment_learner=logistic(),
                reduced_mechanism_residual_learner=LinearRegression(),
            ),
        ),
        "aiptw_known_xf": (
            True,
            # The notebook passes models=ModelSpec() with
            # reduced_outcome_learner=LogisticRegression(). With recorded chances both resolve to
            # the same reduced logistic regression of success on the chance.
            AdaptiveIPTWMethod(models=models, cross_fitting=five, runtime=runtime),
        ),
        "tmle_fitted_g_in": (
            False,
            TMLEMethod(models=models, cross_fitting=in_sample, runtime=runtime),
        ),
    }
    row: dict[str, float] = {"seed": seed}
    offered = frame["transition_navigation"] == 1.0
    row["naive"] = float(
        frame.loc[offered, "transition_success"].mean()
        - frame.loc[~offered, "transition_success"].mean()
    )
    for name, (known, method) in methods.items():
        estimate = effect(known).estimate(method=method)["ate"]
        low, high = estimate.ci
        row[f"{name}_psi"] = estimate.psi
        row[f"{name}_se"] = estimate.std_error
        row[f"{name}_covered"] = float(low <= truth["ate"] <= high)
    return row


def main() -> None:
    with Pool(4) as pool:
        rows = pool.map(one, range(FIRST_SEED, FIRST_SEED + DRAWS))
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "sweep.csv", index=False)
    ate = randomized_pilot_truth()["ate"]
    print(f"true ATE {ate:.4f}; n = {N}; draws = {len(table)}")
    print(f"naive difference: mean {table['naive'].mean():.4f}")
    for name in FITS:
        psi = table[f"{name}_psi"]
        se = table[f"{name}_se"].mean()
        print(
            f"{name:17s} mean error {psi.mean() - ate:+.4f}  SD {psi.std():.4f}  "
            f"mean SE {se:.4f}  SE/SD {se / psi.std():.2f}  "
            f"covered {int(table[f'{name}_covered'].sum())} of {len(table)}"
        )


if __name__ == "__main__":
    main()

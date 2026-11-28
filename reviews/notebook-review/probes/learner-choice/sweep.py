"""Seed sweep of the fits that docs/examples/learner-choice.ipynb shows.

Each seed ``s`` draws ``navigation_data(n=1000, seed=s)`` and runs the page's fits with every
``random_state`` set to ``s``: the declared ensembles, the same ensembles with one candidate per
model (``meta_learner="discrete"``), and one shallow booster per model on the four program
columns plus 20 noise columns, without and with the covariate screen (``min_retain=4``).  It also runs
``screen_by_outcome`` on every row, as the page does, and records the columns it keeps.  The code of
each fit is the page's, with ``52`` replaced by ``s``; :data:`CONFIG` records the settings, and the
page's callback checks them against the notebook.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 9000 200 3``.  Output: ``sweep.csv`` (one row per seed), which ``summarize.py``
reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``, ``n_jobs=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")

#: The page's settings.  The callback asserts the notebook uses the same ones.
CONFIG = {
    "n": 1000,
    "outer_folds": 5,
    "inner_folds": 3,
    "q_bounds": (0.0, 1.0),
    "n_knots": 5,
    "booster": {"max_depth": 2, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0},
    "noise_columns": 20,
    "min_retain": 4,
}


def _booster(kind):
    return kind(**CONFIG["booster"])


def one_seed(seed: int) -> dict:
    from dataclasses import replace

    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    from cleverly import (
        ATE,
        CausalStudy,
        CrossFitting,
        ModelSpec,
        PointTreatment,
        Runtime,
        SuperLearner,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import navigation_data
    from cleverly.learners import screen_by_outcome

    start = time.perf_counter()
    frame, truth = navigation_data(n=CONFIG["n"], seed=seed)
    ate = truth["ate"]

    def identify(data, columns):
        design = PointTreatment(
            outcome="transition_score", treatment="transition_navigation", adjustment=columns
        )
        return CausalStudy(data, design=design).identify(ATE(reference=0))

    effect = identify(frame, COVARIATES)
    knots = CONFIG["n_knots"]
    outcome_ensemble = SuperLearner(
        library=[
            ("linear", LinearRegression()),
            ("splines", make_pipeline(SplineTransformer(n_knots=knots), RidgeCV())),
            ("boosting", _booster(HistGradientBoostingRegressor)),
        ],
        n_folds=CONFIG["inner_folds"],
        clip=(0.0, 1.0),
        random_state=seed,
    )
    treatment_ensemble = SuperLearner(
        library=[
            ("logistic", LogisticRegression(max_iter=1000)),
            (
                "splines",
                make_pipeline(SplineTransformer(n_knots=knots), LogisticRegression(max_iter=2000)),
            ),
            ("boosting", _booster(HistGradientBoostingClassifier)),
        ],
        n_folds=CONFIG["inner_folds"],
        random_state=seed,
    )
    declared_method = TMLEMethod(
        models=ModelSpec(outcome_learner=outcome_ensemble, treatment_learner=treatment_ensemble),
        cross_fitting=CrossFitting(n_folds=CONFIG["outer_folds"]),
        targeting=Targeting(q_bounds=CONFIG["q_bounds"]),
        runtime=Runtime(random_state=seed, n_jobs=1),
    )
    discrete_models = ModelSpec(
        outcome_learner=clone(outcome_ensemble).set_params(meta_learner="discrete"),
        treatment_learner=clone(treatment_ensemble).set_params(meta_learner="discrete"),
    )

    rng = np.random.default_rng(seed)
    count = CONFIG["noise_columns"]
    noise = pd.DataFrame(
        rng.normal(size=(len(frame), count)), columns=[f"record_{i:02d}" for i in range(count)]
    )
    wide = pd.concat([frame, noise], axis=1)
    wide_covariates = [*COVARIATES, *noise.columns]
    wide_effect = identify(wide, tuple(wide_covariates))
    plain = replace(
        declared_method,
        models=ModelSpec(
            outcome_learner=_booster(HistGradientBoostingRegressor),
            treatment_learner=_booster(HistGradientBoostingClassifier),
        ),
    )
    screened_models = replace(plain.models, screen_treatment=True, min_retain=CONFIG["min_retain"])

    row: dict = {"seed": seed, "truth": ate}
    fits = {
        "declared": effect.estimate(method=declared_method),
        "discrete": effect.estimate(method=replace(declared_method, models=discrete_models)),
        "unscreened": wide_effect.estimate(method=plain),
        "screened": wide_effect.estimate(method=replace(plain, models=screened_models)),
    }
    for label, fitted in fits.items():
        estimate = fitted["ate"]
        low, high = estimate.ci
        row[f"{label}_psi"] = estimate.psi
        row[f"{label}_se"] = estimate.std_error
        row[f"{label}_covers"] = bool(low <= ate <= high)
        support = fitted.diagnostics.support()
        row[f"{label}_at_cap"] = support.truncated["fraction"]
        row[f"{label}_min_g"] = support.propensity_quantiles["overall"][0.0]
    for label in ("declared", "discrete"):
        for model in fits[label].diagnostics.nuisance_models().models:
            for name, weight in model.learner_weights.items():
                row[f"{label}_{model.name}_{name}_weight"] = weight
            row[f"{label}_{model.name}_top"] = max(
                model.learner_weights, key=model.learner_weights.get
            )
    kept = screen_by_outcome(
        wide[wide_covariates].to_numpy(),
        wide["transition_navigation"].to_numpy(),
        wide["transition_score"].to_numpy(),
        min_retain=CONFIG["min_retain"],
    )
    kept_names = [name for name, keep in zip(wide_covariates, kept, strict=True) if keep]
    row["kept"] = ";".join(kept_names)
    row["kept_count"] = len(kept_names)
    for name in COVARIATES:
        row[f"kept_{name}"] = name in kept_names
    row["screened_minus_unscreened"] = row["screened_psi"] - row["unscreened_psi"]
    row["seconds"] = time.perf_counter() - start
    return row


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    seeds = range(first, first + count)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, seeds))
    pd.DataFrame(rows).to_csv(HERE / "sweep.csv", index=False)


if __name__ == "__main__":
    main()

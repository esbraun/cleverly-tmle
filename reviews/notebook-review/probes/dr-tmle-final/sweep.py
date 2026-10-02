"""Seed sweep of the DR-TMLE and ordinary TMLE fits that docs/examples/dr-tmle.ipynb shows.

Each seed ``s`` draws ``navigation_data(n=2000, seed=s)`` and runs the notebook's Step 6 fit
(DR-TMLE with Super Learner reductions), its Step 7 fit (the ordinary TMLE), its Step 8 fit
(DR-TMLE with constant reductions), its Step 9 assessment, and its Step 10 E-value, with every
``random_state`` set to ``s``.  The code of each fit is the notebook's, with ``55`` replaced by
``s``.

Usage: ``python sweep.py <first seed> <count> <workers>``.  The page cites
``python sweep.py 8000 200 12``.  Outputs: ``sweep.csv`` (one row per seed) and ``sweep.log``,
which ``summarize.py`` reads.  Every worker is single-threaded (``OMP_NUM_THREADS=1``,
``n_jobs=1``).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COVARIATES = ("discharge_risk", "prior_utilization", "medication_burden", "age")


def one_seed(seed: int) -> dict:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    from cleverly import (
        ATE,
        CausalStudy,
        CrossFitting,
        DRTMLEMethod,
        ModelSpec,
        PointTreatment,
        Runtime,
        SuperLearner,
        Targeting,
        TMLEMethod,
    )
    from cleverly.datasets import navigation_data
    from cleverly.datasets.synthetic import nonlinear_bounded_dgp

    start = time.perf_counter()
    frame, truth = navigation_data(n=2_000, seed=seed)
    ate = truth["ate"]
    effect = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="transition_score",
            treatment="transition_navigation",
            adjustment=COVARIATES,
        ),
    ).identify(ATE(reference=0))
    models = ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=seed),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=seed),
    )
    folds = CrossFitting(n_folds=3)
    declared_support = Targeting(q_bounds=(0.0, 1.0))
    runtime = Runtime(random_state=seed, n_jobs=1)

    def spline(final):
        return make_pipeline(SplineTransformer(n_knots=5, knots="quantile"), final)

    reduced_outcome = SuperLearner(
        library=[
            ("linear", LinearRegression(n_jobs=1)),
            ("spline", spline(LinearRegression(n_jobs=1))),
        ],
        task="regression",
        n_folds=3,
        random_state=seed,
        n_jobs=1,
    )
    reduced_treatment = SuperLearner(
        library=[
            ("logistic", LogisticRegression(max_iter=1000, random_state=seed)),
            ("spline", spline(LogisticRegression(max_iter=1000, random_state=seed))),
        ],
        task="classification",
        n_folds=3,
        random_state=seed,
        n_jobs=1,
    )
    drtmle = DRTMLEMethod(
        models=models,
        cross_fitting=folds,
        targeting=declared_support,
        runtime=runtime,
        reduced_outcome_learner=reduced_outcome,
        reduced_treatment_learner=reduced_treatment,
    )
    guarded = effect.estimate(method=drtmle)
    ordinary = effect.estimate(
        method=TMLEMethod(
            models=models, cross_fitting=folds, targeting=declared_support, runtime=runtime
        )
    )
    crude = effect.estimate(
        method=replace(
            drtmle,
            reduced_outcome_learner=DummyRegressor(),
            reduced_treatment_learner=DummyClassifier(strategy="prior"),
        )
    )
    row: dict = {"seed": seed}
    for label, fitted in (("dr", guarded), ("ord", ordinary), ("const", crude)):
        point = fitted["ate"]
        row |= {
            f"{label}_psi": point.psi,
            f"{label}_se": point.std_error,
            f"{label}_covers": point.ci[0] <= ate <= point.ci[1],
        }
    for label, fitted in (("dr", guarded), ("const", crude)):
        row[f"{label}_scores_passed"] = fitted.diagnostics.score_equations().passed
        row[f"{label}_corrections_passed"] = fitted.diagnostics.corrections().passed

    # Step 9.
    assessment = guarded.assess()
    row["attention"] = "|".join(item.name for item in assessment.attention)
    row["contract"] = assessment.report("corrections").contract
    row["nuisance_reasonable"] = "look reasonable" in assessment.report("nuisance_models").summary()
    support = assessment.report("support")
    row["truncated"] = support.truncated["count"]
    row["min_fitted_g"] = support.propensity_quantiles["overall"][0.0]
    reduced = guarded.extra["drtmle"].diagnostics
    for family in ("qr", "gr1", "gr2"):
        row[f"{family}_spline_best"] = sum(fit.best == "spline" for fit in reduced[family])

    # Step 10.
    evalue = guarded.sensitivity.evalue()
    row["evalue_point"] = evalue.point
    row["evalue_limit"] = evalue.limit

    # The law's propensity on the draw's own rows, for the support reading.
    g0 = nonlinear_bounded_dgp().propensity(frame.loc[:, list(COVARIATES)].to_numpy(dtype=float))
    row["min_true_g"] = float(g0.min())
    row["seconds"] = time.perf_counter() - start
    return row


def main() -> None:
    first, count, workers = (int(value) for value in sys.argv[1:4])
    import cleverly

    assert Path(cleverly.__file__).resolve().is_relative_to(HERE.parents[3] / "src")
    seeds = range(first, first + count)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one_seed, seeds))
    frame = pd.DataFrame(rows).sort_values("seed")
    frame.to_csv(HERE / "sweep.csv", index=False, float_format="%.10g")
    print(f"{len(frame)} seeds, {first} to {first + count - 1}; cleverly {cleverly.__file__}")
    print(f"median seconds per seed: {np.median(frame['seconds']):.1f}")


if __name__ == "__main__":
    main()

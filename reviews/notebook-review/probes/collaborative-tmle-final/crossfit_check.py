"""Fit the page's draw (seed 44) in sample and with five outer folds, under one declared range.

The synthetic score has no documented range, so the range ``q_bounds=(-20, 20)`` is a convention
chosen only to let the cross-fitted fit run.  The probe asks how far the fold layer moves the
C-TMLE and plain TMLE estimates with the page's linear learners.  Output: ``crossfit_check.log``.
"""

import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import (
    ATE,
    CausalStudy,
    CollaborativeTMLEMethod,
    CrossFitting,
    ModelSpec,
    PointTreatment,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import make_instrument

frame, truth = make_instrument(n=2_000, seed=44)
effect = CausalStudy(
    frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
).identify(ATE(reference=0))
models = ModelSpec(
    outcome_learner=LinearRegression(n_jobs=1),
    treatment_learner=LogisticRegression(max_iter=1000, random_state=44),
)
rt = Runtime(random_state=44, n_jobs=1)
for folds in (CrossFitting(enabled=False), CrossFitting(n_folds=5)):
    t = time.perf_counter()
    r = effect.estimate(
        method=CollaborativeTMLEMethod(
            models=models,
            cross_fitting=folds,
            targeting=Targeting(q_bounds=(-20.0, 20.0)),
            runtime=rt,
            strategy="greedy",
            selection_folds=3,
            selection_inner_folds=2,
        )
    )
    p = effect.estimate(
        method=TMLEMethod(
            models=models,
            cross_fitting=folds,
            targeting=Targeting(q_bounds=(-20.0, 20.0)),
            runtime=rt,
        )
    )
    print(
        folds,
        round(time.perf_counter() - t, 2),
        r["ate"].psi,
        r["ate"].plugin_std_error,
        p["ate"].psi,
        p["ate"].std_error,
    )
    try:
        print(r.diagnostics.nuisance_models().selection.summary())
    except Exception as e:
        print("selection:", type(e).__name__, e)

"""Review probe of the refutation step of docs/examples/point-treatment-tmle.ipynb (Step 10).

The notebook runs each refutation at 20 draws, because the fast suite runs its cells. This probe
refits the Step 6 fit and runs the page's two generated-outcome tests at 400 draws, and the
placebo, subset and random_common_cause tests at 100 draws.

Usage: ``python probe.py generated`` or ``python probe.py standard``, with the output sent to
``generated.log`` or ``standard.log``. The page quotes those logs. A review probe is not a
registered study.
"""

import sys
import time
import warnings

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    Inference,
    ModelSpec,
    PointTreatment,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.datasets import navigation_data, navigation_protocol
from cleverly.validation import BoundedAdjustmentOutcome

warnings.filterwarnings("ignore")

frame, truth = navigation_data(n=3_000, seed=21)
covariates = ["discharge_risk", "prior_utilization", "medication_burden", "age"]
effect = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="transition_score",
        treatment="transition_navigation",
        adjustment=tuple(covariates),
    ),
    protocol=navigation_protocol(),
).identify(ATE(reference=0))
method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=HistGradientBoostingRegressor(random_state=21),
        treatment_learner=HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=21
        ),
    ),
    cross_fitting=CrossFitting(n_folds=5),
    targeting=Targeting(q_bounds=(0.0, 1.0)),
    inference=Inference(alpha=0.05),
    runtime=Runtime(random_state=21, n_jobs=1),
)
result = effect.estimate(method=method)
print(f"Step 6 fit: estimate {result['ate'].psi:.4f}, truth {truth['ate']:.4f}", flush=True)

which = sys.argv[1]
if which == "generated":
    declared = BoundedAdjustmentOutcome(intercept=-0.5, effect=0.4, adjustment_scale=0.5)
    for name in ("simulated_outcome", "plasmode"):
        start = time.time()
        report = result.diagnostics.refute(
            tests=(name,), simulated_outcome=declared, n_replicates=400, random_state=61
        )
        test = report.tests[0]
        print(
            f"{name}: {test.n_replicates} draws, level {test.rung}, verdict {test.verdict}, "
            f"true value {test.declared_effect:.4f}, mean estimate {test.mean:.4f}, "
            f"{time.time() - start:.0f} s",
            flush=True,
        )
        for reading in test.readings:
            print(
                f"  {reading.name} {reading.estimate:.4f} in ({reading.lower:.4f}, "
                f"{reading.upper:.4f}), band {reading.band}, inside {reading.inside}",
                flush=True,
            )
else:
    start = time.time()
    report = result.diagnostics.refute(n_replicates=100, random_state=21)
    print(f"{time.time() - start:.0f} s for 300 refits")
    for test in report.tests:
        print(
            f"{test.name}: level {test.rung}, verdict {test.verdict}, {test.statistic_name} "
            f"{test.statistic:.4f}, {test.detail}",
            flush=True,
        )

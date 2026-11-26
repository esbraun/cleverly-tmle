"""Review probe of the generated-outcome lesson of docs/examples/population-impact.ipynb.

The notebook runs the three generated-outcome tests at 20 draws on the gain fit, because the slow
tier runs its cells. This probe refits that fit with the page's own data, learners and fit seed,
and runs the same tests at 400 draws, the count their level-2 bands need, with refutation seed 61.
The no-effect test runs again with seed 62, because its seed-61 run fails its band.

Usage: ``python reviews/notebook-review/probes/population-impact-refuters/probe.py`` with the
output sent to ``generated.log`` next to it. The page quotes that log. A review probe is not a
registered study.
"""

import time
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import (
    CausalStudy,
    CrossFitting,
    ModelSpec,
    PointTreatment,
    PopulationAttributableRisk,
    Runtime,
    TMLEMethod,
)
from cleverly.datasets import make_population_impact
from cleverly.validation import BernoulliAdjustmentOutcome

warnings.filterwarnings("ignore")

frame, truth = make_population_impact(n=4_000, seed=2031)
design = PointTreatment(
    outcome="top_box",
    treatment="transition_navigation",
    adjustment=("discharge_risk", "age", "prior_utilization"),
    missingness="responded",
)


def logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=2000)


method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=make_pipeline(
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), logistic()
        ),
        treatment_learner=logistic(),
        missingness_learner=logistic(),
    ),
    cross_fitting=CrossFitting(n_folds=10),
    runtime=Runtime(random_state=2031, n_jobs=1),
)
gain_fit = (
    CausalStudy(frame, design=design)
    .identify(PopulationAttributableRisk(reference=1))
    .estimate(method=method)
)
print(f"gain fit: par {gain_fit['par'].psi:.4f}", flush=True)

declared = {
    "dummy_outcome": BernoulliAdjustmentOutcome(effect=0.0, adjustment_scale=0.0),
    "simulated_outcome": BernoulliAdjustmentOutcome(effect=0.4),
}
for name, seed in (
    ("dummy_outcome", 61),
    ("simulated_outcome", 61),
    ("plasmode", 61),
    ("dummy_outcome", 62),
):
    start = time.time()
    report = gain_fit.diagnostics.refute(
        estimand="par",
        tests=(name,),
        dummy_outcome=declared["dummy_outcome"],
        simulated_outcome=declared["simulated_outcome"],
        n_replicates=400,
        random_state=seed,
    )
    test = report.tests[0]
    print(
        f"{name}, seed {seed}: {test.n_replicates} draws, level {test.rung}, "
        f"verdict {test.verdict}, "
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

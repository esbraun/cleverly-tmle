"""Review probe of the refutation step of docs/examples/longitudinal-survival.ipynb (Step 10).

The notebook runs each refutation at 20 draws, because the fast suite runs its cells. This probe
refits the same `RegimeContrast` fit and runs the page's three generated-outcome tests at 400
draws, and the placebo and subset tests at 100 draws under three seeds.

Usage: ``python probe.py > probe.log``. The page quotes ``probe.log``. A review probe is not a
registered study.
"""

import time
import warnings
from dataclasses import replace

from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    CausalStudy,
    CrossFitting,
    LongitudinalTreatment,
    ModelSpec,
    RegimeContrast,
    Runtime,
    TMLEMethod,
)
from cleverly.datasets import longitudinal_navigation_protocol, make_longitudinal_competing
from cleverly.validation import MultinomialHazardOutcome

warnings.filterwarnings("ignore")

frame, _ = make_longitudinal_competing(n=4_000, seed=53, censoring=True)
frame = frame.rename(
    columns={
        "W1": "age",
        "W2": "baseline_readiness",
        "A1": "navigation_p1",
        "C1": "enrolled_p1",
        "R1": "readmission_p1",
        "D1": "death_p1",
        "L2": "identified_needs",
        "A2": "navigation_p2",
        "C2": "enrolled_p2",
        "R2": "readmission_p2",
        "D2": "death_p2",
    }
)
design = LongitudinalTreatment(
    outcome={
        "readmission": ("readmission_p1", "readmission_p2"),
        "death": ("death_p1", "death_p2"),
    },
    censoring=("enrolled_p1", "enrolled_p2"),
    treatment=("navigation_p1", "navigation_p2"),
    baseline=("age", "baseline_readiness"),
    time_varying=((), ("identified_needs",)),
)
sequential = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LogisticRegression(max_iter=1000, random_state=41),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000, random_state=41),
        censoring_learner=LogisticRegression(max_iter=1000, random_state=41),
    ),
    cross_fitting=CrossFitting(enabled=False),
    runtime=Runtime(random_state=41, n_jobs=1),
)
fit = (
    CausalStudy(frame, design=design, protocol=longitudinal_navigation_protocol())
    .identify(RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2)))
    .estimate(method=sequential)
)
target = "ate_regimen[always vs never, readmission @ t=2]"
print(f"fit: {fit[target].psi:.4f}, SE {fit[target].std_error:.4f}")

independent = MultinomialHazardOutcome(
    intercepts=(-2.0, -3.0), effects=(0.0, 0.0), adjustment_scale=0.0
)
adjusted = MultinomialHazardOutcome(
    intercepts=(-2.0, -3.0), effects=(0.4, -0.6), adjustment_scale=0.5
)
generated = {
    "dummy_outcome": {"dummy_outcome": independent},
    "simulated_outcome": {"simulated_outcome": adjusted},
    "plasmode": {"simulated_outcome": replace(adjusted, history_effect=0.3)},
}
for name, declaration in generated.items():
    started = time.time()
    test = fit.diagnostics.refute(
        estimand=target, tests=(name,), n_replicates=400, random_state=61, **declaration
    ).tests[0]
    print(f"{name} at 400 draws, {time.time() - started:.0f} s, verdict {test.verdict}")
    print(f"  true value {test.declared_effect:.4f}, mean estimate {test.mean:.4f}")
    for reading in test.readings:
        print(
            f"  {reading.name} {reading.estimate:.4f} "
            f"({reading.lower:.4f}, {reading.upper:.4f}), band {reading.band}, "
            f"inside {reading.inside}"
        )
for seed in (61, 62, 63):
    started = time.time()
    tests = fit.diagnostics.refute(
        estimand=target, tests=("placebo", "subset"), n_replicates=100, random_state=seed
    ).tests
    print(f"placebo and subset at 100 draws, seed {seed}, {time.time() - started:.0f} s")
    for test in tests:
        print(
            f"  {test.name}: mean {test.mean:.4f}, SD {test.spread:.4f}, "
            f"{test.statistic_name} {test.statistic:.4f}, p = {test.statistic_pvalue:.4f}"
        )

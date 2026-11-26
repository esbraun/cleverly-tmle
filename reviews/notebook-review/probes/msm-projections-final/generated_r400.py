"""The page's generated-outcome checks at 400 refits each, the size their level-2 bands need.

Rebuilds the page's trend fit (``make_multi_arm(n=3000, seed=61)``, in-sample linear and logistic
learners, uniform weight, contacts 1, 2, 6) and runs ``dummy_outcome``, ``simulated_outcome`` and
``plasmode`` on the slope with ``n_replicates=400``, the same law and seeds the notebook's
``generated`` cell uses at 20. Prints the wall time, each test's level and verdict, and each
reading against its band, headed by the cleverly version and commit.

Run: .venv/bin/python reviews/notebook-review/probes/msm-projections-final/generated_r400.py > generated_r400.log
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import platform
import subprocess
import time

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (
    CausalStudy,
    CrossFitting,
    ModelSpec,
    MSMProjection,
    PointTreatment,
    Runtime,
    TMLEMethod,
)
from cleverly.datasets import make_multi_arm
from cleverly.msm import MSM
from cleverly.validation import GaussianAdjustmentOutcome, refute

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ARMS = ("low", "medium", "high")
CONTACTS = {"low": 1.0, "medium": 2.0, "high": 6.0}
SLOPE = "msm[assigned contacts]"


def provenance():
    """The commit, worktree state and package version the refits ran under."""

    def git(*args):
        return subprocess.run(
            ["git", "-C", WORKTREE, *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    clean = "clean" if not git("status", "--porcelain", "--", "src") else "modified src"
    return f"cleverly {cleverly.__version__}, commit {git('rev-parse', 'HEAD')[:12]} ({clean})"


def main():
    print(f"run under: {provenance()}")
    print(f"machine: {platform.machine()}, {os.cpu_count()} CPUs visible, one thread per fit")
    frame, truth = make_multi_arm(n=3_000, seed=61)
    population = np.array([truth[f"ey[{arm}]"] for arm in ARMS])
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000, random_state=61),
        ),
        cross_fitting=CrossFitting(enabled=False),
        runtime=Runtime(random_state=61, n_jobs=1),
    )

    def design(arm, data):
        return np.column_stack([np.ones(len(data)), np.full(len(data), CONTACTS[arm])])

    trend = MSM(design=design, terms=("(intercept)", "assigned contacts"), design_kind="known")
    result = study.identify(MSMProjection(trend)).estimate(method=method)
    print(f"slope on the page's draw: {result[SLOPE].psi:.4f}")
    law = GaussianAdjustmentOutcome(
        arm_effects={arm: float(mean) for arm, mean in zip(ARMS, population, strict=True)}
    )
    start = time.perf_counter()
    generated = refute(
        result,
        estimand=SLOPE,
        tests=("dummy_outcome", "simulated_outcome", "plasmode"),
        simulated_outcome=law,
        n_replicates=400,
    )
    elapsed = time.perf_counter() - start
    print(f"wall time for the three checks at 400 refits each: {elapsed:.0f} s")
    for test in generated.tests:
        print(
            f"{test.name}: {test.n_replicates} draws, level {test.rung}, verdict {test.verdict}, "
            f"true slope {test.declared_effect:.4f}, mean estimate {test.mean:.4f}"
        )
        for reading in test.readings:
            print(
                f"  {reading.name} {reading.estimate:.3f} in ({reading.lower:.3f}, "
                f"{reading.upper:.3f}), band {reading.band}, inside {reading.inside}"
            )


if __name__ == "__main__":
    main()

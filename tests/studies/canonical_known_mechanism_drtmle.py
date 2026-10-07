"""Registered evidence for complete-data DR-TMLE on a declared known treatment mechanism.

The subject is ``DRTMLE`` on data that declares its treatment mechanism.  This is Benkeser,
Carone, van der Laan and Gilbert (2017), Theorem 1, at the degenerate estimator
``g_n = g0``.  Under the ``"Q"`` guard the fit still fluctuates the mechanism along
``Q_r / g`` (the algorithm's step 6), so the targeted ``g*`` moves off the declaration on
purpose, as R ``drtmle`` does with a supplied ``gn``.

**The comparator** is R ``drtmle`` 1.1.2 at ``538a3a2`` with ``gn = list(1 - g0, g0)``,
through ``tests/canonical/known_mechanism_drtmle/run_study.R``.  Both sides read this
package's initial outcome regression as ``Qn``, fit the reduced regressions with the same
families (``glm_Qr = "gn"``, a linear regression on the mechanism, and ``glm_gr = "Qn"``, a
logistic regression on the outcome regression), and fit in sample (``cvFolds = 1``).
``tolg = 0.1`` lies below every declared value, and the runner refuses a sample where it
would bind.

**The scenarios** are the four guards, ``guard_none``, ``guard_q``, ``guard_g`` and
``guard_qg``, each reporting ``ey0``, ``ey1`` and ``ate`` with the wrong outcome regression of
:mod:`tests.studies.known_mechanism_law`.  All four read the same samples.

Publication policy is ``reporting``, with the red-cell route of
:mod:`tests.studies.canonical_known_mechanism` (owner ``X15-known-mechanism``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import known_mechanism_law as law
from tests.studies.canonical_known_mechanism import (
    COVARIATES,
    G_BOUNDS,
    PROBABILITIES,
    DeclaredOnly,
    OutcomeGLM,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import initial_estimates, primary_rows

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261151
RESAMPLING_SEED = 2026115101
IMPLEMENTATION = "cleverly-known-mechanism-drtmle"
REFERENCE = "drtmle-r-known-g"
ESTIMANDS = ("ey0", "ey1", "ate")
#: Scenario -> the guard both sides solve.
GUARDS: dict[str, tuple[str, ...]] = {
    "guard_none": (),
    "guard_q": ("Q",),
    "guard_g": ("g",),
    "guard_qg": ("Q", "g"),
}
OWNER = "guard_qg"
MAX_OUTER = 100

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "known_mechanism_accuracy": tuple(
        f"ate__drtmle_known__{name}" for name in ("none", "Q", "g", "Qg")
    ),
}

STUDY = StudyRecord(
    name="DR-TMLE on a declared known treatment mechanism",
    slug="known-treatment-mechanism-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "known_mechanism_drtmle",
    document="docs/technical-reference/method-evidence/known-treatment-mechanism.md",
    anchor="dr-tmle-on-a-declared-known-treatment-mechanism",
    scenarios=dict.fromkeys(GUARDS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={scenario: OWNER for scenario in GUARDS if scenario != OWNER},
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    modules=(
        "tests/studies/canonical_known_mechanism_drtmle.py",
        "tests/studies/known_mechanism_drtmle_properties.py",
        "tests/studies/canonical_known_mechanism.py",
        "tests/studies/known_mechanism_properties.py",
        "tests/studies/known_mechanism_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/drtmle/Dockerfile",
        "tests/canonical/known_mechanism_drtmle/run_study.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_known_mechanism_drtmle",
    properties_module="tests.studies.known_mechanism_drtmle_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "drtmle(Qn = list(qn0, qn1), gn = list(1 - g0, g0), glm_Qr = 'gn', glm_gr = 'Qn', "
        "reduction = 'univariate', maxIter = 100, tolIC = 1e-8, tolg = 0.1, cvFolds = 1); Qn is "
        "this package's initial outcome regression"
    ),
}

CONFIGURATION = {
    "construction": "complete-data DR-TMLE dividing by the declared mechanism",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "treatment_mechanism": "known (declared by column)",
    "outcome_learner": "unpenalized logistic regression on A, W1, W3 (omits W2)",
    "reduced_outcome_learner": "linear regression on the mechanism",
    "reduced_treatment_learner": "logistic regression on the outcome regression",
    "reduction": "univariate",
    "max_outer": MAX_OUTER,
}


def fit_cleverly(frame: pd.DataFrame, guard: tuple[str, ...], *, outcome: str = "wrong") -> Any:
    """Fit this study's DR-TMLE at one guard on one sample."""
    return (
        DRTMLE(
            guard=guard,
            reduction="univariate",
            outcome_learner=OutcomeGLM(outcome, 2),
            treatment_learner=DeclaredOnly(),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=2_000),
            cross_fit=False,
            estimands=ESTIMANDS,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_outer=MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=COVARIATES,
            treatment_probabilities=PROBABILITIES[2],
        )
        .single()
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in GUARDS:
        raise KeyError(scenario)
    truth = law.truth(2)
    return law.sample(n, seed), {name: truth[name] for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    return primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, GUARDS[scenario])
    sample = frame.loc[:, ["W1", "W2", "W3", "A", "Y"]].copy()
    sample["qn0"] = np.asarray(result.nuisance.outcome.arms[0.0], dtype=float)
    sample["qn1"] = np.asarray(result.nuisance.outcome.arms[1.0], dtype=float)
    sample["gn1"] = frame["p1"].to_numpy(dtype=float)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, cleverly_rows(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((scenario, replicate, n),) for scenario in GUARDS for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

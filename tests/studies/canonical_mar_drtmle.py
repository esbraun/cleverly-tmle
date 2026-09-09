"""Registered evidence study for randomized missing-outcome DR-TMLE."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.missing_outcome_study_helpers import (
    probabilities,
    sample_discrete,
    truths,
)
from tests.studies.point_study_helpers import initial_estimates, primary_rows

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20260923
SCENARIO = "binary_mar_randomized"
ESTIMANDS = ("ey0", "ey1", "ate")
G_BOUNDS = (0.01, 0.99)
NUISANCE_BOUND = 0.01
MAX_OUTER = 100
RANDOMIZATION = np.full(3, 0.5)
PROBS = probabilities(g=RANDOMIZATION)

STUDY = StudyRecord(
    name="randomized missing-outcome DR-TMLE",
    slug="mar-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_mar",
    document="docs/technical-reference/method-evidence/randomized-missing-outcome-dr-tmle.md",
    anchor="randomized-missing-outcome-dr-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    nuisance_count=3,
    resampling_seed=20260924,
    margins=Margins(),
    implementation="cleverly-mar-drtmle",
    reference="drtmle-r-mar",
    modules=(
        "tests/studies/canonical_mar_drtmle.py",
        "tests/studies/mar_drtmle_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/discrete_law_mar.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mar_drtmle",
    properties_module="tests.studies.mar_drtmle_properties",
    property_cells={
        "corrected_mar_inference": (
            "both_correct",
            "outcome_drift",
            "observation_drift",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ate__correctly_specified",
            "ate__shrunken_se_control",
            "ate__noise_control",
        ),
        "correction_necessity": (
            "five_reduction_cycle__closed_score",
            "five_reduction_cycle__initial_score_control",
        ),
    },
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "comparator_boundary": "joint treatment-response mechanism; both-correct limit only",
}

CONFIGURATION = {
    "construction": "Díaz and van der Laan randomized MAR DR-TMLE",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "known_treatment_probability": float(RANDOMIZATION[0]),
    "g_bounds": list(G_BOUNDS),
    "missingness_bound": NUISANCE_BOUND,
    "guard": ["Q", "g"],
    # Two names, because they are two different things.  ``reduction`` is the constructor
    # setting and only accepts ``"univariate"`` or ``"bivariate"``; a ``delta=`` fit replaces
    # it with Diaz and van der Laan's five regressions, and ``fitted_reduction`` is the family
    # ``ReducedFit.reduction`` then reports.  Publishing the fitted name alone read as a
    # constructor argument no caller could pass.
    "reduction": "univariate",
    "fitted_reduction": "missing_outcome",
    "max_outer": MAX_OUTER,
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_discrete(PROBS, n, seed), truths(PROBS, ESTIMANDS)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    law = mar.DiscreteLaw(PROBS)
    return (
        DRTMLE(
            randomized=True,
            cross_fit=False,
            outcome_learner=OracleOutcome(law),
            treatment_learner=OracleTreatment(law),
            missingness_learner=OracleMissingness(law),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=2_000),
            estimands=ESTIMANDS,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            nuisance_bound=NUISANCE_BOUND,
            max_outer=MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            # Read off the published record rather than left to the class default, so the
            # manifest cannot describe a fit the constructor did not make.
            guard=tuple(CONFIGURATION["guard"]),
            reduction=str(CONFIGURATION["reduction"]),
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_probabilities=np.full(len(frame), RANDOMIZATION[0]),
        )
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame)
    return primary_rows(
        result=result,
        truth=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    law = mar.DiscreteLaw(PROBS)
    levels = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    sample = frame.copy()
    sample["qn0"] = law.q[levels, 0]
    sample["qn1"] = law.q[levels, 1]
    # R `drtmle` takes one joint treatment-response mechanism, so the comparator is handed
    # ``P(A = a, Delta = 1 | W)``.  Built from :data:`RANDOMIZATION` rather than from a literal
    # 0.5, so a change to the trial's assignment probability reaches the comparator too.
    sample["gn0"] = (1.0 - RANDOMIZATION[levels]) * law.pi[levels, 0]
    sample["gn1"] = RANDOMIZATION[levels] * law.pi[levels, 1]
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference.items()
    ]
    return sample, truth_rows, cleverly_rows(frame, reference, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]

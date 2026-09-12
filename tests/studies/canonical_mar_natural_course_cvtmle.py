"""Registered evidence for stacked CV-TMLE of the MAR natural-course mean."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from cleverly import (
    CausalStudy,
    CrossFitting,
    Inference,
    ModelSpec,
    NaturalCourseMean,
    PointTreatment,
    Runtime,
    Targeting,
    TMLEMethod,
)
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_mar as mar
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.missing_outcome_study_helpers import (
    FailTreatment,
    efficiency_sd,
    sample_discrete,
)
from tests.studies.point_study_helpers import natural_course_initial_estimate, primary_rows

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20261109
SCENARIO = "binary_mar_natural_course_flexible"
ESTIMANDS = ("ey_obs",)
NUISANCE_BOUND = 0.01
TRUTH = float(mar.functional(mar.PROBS, "ey_obs"))
EFFICIENCY_SD = efficiency_sd(mar.PROBS, "ey_obs")


STUDY = StudyRecord(
    name="stacked missing-outcome natural-course CV-TMLE",
    slug="stacked-mar-natural-course-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_mar_natural_course_cvtmle",
    document=(
        "docs/technical-reference/method-evidence/stacked-missing-outcome-natural-course-cvtmle.md"
    ),
    anchor="stacked-missing-outcome-natural-course-cv-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261110,
    margins=Margins(),
    implementation="cleverly-stacked-mar-natural-course-cvtmle",
    reference=None,
    modules=(
        "tests/studies/canonical_mar_natural_course_cvtmle.py",
        "tests/studies/mar_natural_course_cvtmle_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/discrete_law_mar.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mar_natural_course_cvtmle",
    properties_module="tests.studies.mar_natural_course_cvtmle_properties",
    property_cells={
        "double_robustness": ("outcome_correct", "response_correct", "both_wrong"),
        "interval_calibration": (
            "ey_obs__flexible_learning",
            "ey_obs__shrunken_se_control",
        ),
        "crossfit_overfitting": (
            "stacked_mar_natural_course_cvtmle",
            "in_sample_control",
        ),
    },
    # The flexible learned-nuisance row does not claim efficiency-bound attainment.
    efficiency_bounds={"ey_obs": EFFICIENCY_SD},
    calibration_efficiency_ratio=False,
)

CONFIGURATION = {
    "construction": "one-repeat stacked MAR natural-course CV-TMLE",
    "source": (
        "Díaz, Carone, and van der Laan (2016), Section 2, equations 1-5; Levy (2018), Section 3.1"
    ),
    "cross_fit": True,
    "n_folds": 10,
    "repeats": 1,
    "stratify_folds": "none",
    "targeting_scheme": "pooled",
    "fold_evaluation": False,
    "split_plan": None,
    "fluctuation": "logistic",
    "targeting": "iterative",
    "target_weights": False,
    "simultaneous_intervals": False,
    "missingness_bound": NUISANCE_BOUND,
    "treatment_mechanism": "deliberately unavailable and asserted unused",
    "primary_nuisance_predictions": (
        "separate depth-five classification trees with minimum leaf size 25 for the "
        "outcome regression and response mechanism"
    ),
    "comparator_search": (
        "Rejected R tmle 2.1.1 because its maintained MAR API reports intervention-arm "
        "means, not the natural-course target; rejected tmle3 at ed72f8a because its "
        "generic treatment-specific outcome fit can use the Delta = 0 pseudo-outcomes when "
        "it predicts under Delta = 1, while this estimator fits that regression on "
        "respondents only; rejected zEpid 0.9.1 because its cross-fit TMLE targets inside "
        "each fold rather than with one pooled fluctuation; rejected Newey and Robins (2018) "
        "because their construction fits the outcome and inverse response regressions on "
        "distinct subsamples, which is a different estimator."
    ),
}


def flexible_outcome_learner() -> DecisionTreeClassifier:
    """The predeclared data-adaptive outcome learner."""
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=0)


def flexible_response_learner() -> DecisionTreeClassifier:
    """The separately fitted, predeclared data-adaptive response learner."""
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=1)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_discrete(mar.PROBS, n, seed), {"ey_obs": TRUTH}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def method(
    *,
    outcome_learner: Any,
    missingness_learner: Any,
    cross_fit: bool = True,
) -> TMLEMethod:
    """Build the exact public RM9 method declaration used by every study arm."""
    return TMLEMethod(
        models=ModelSpec(
            outcome_learner=outcome_learner,
            treatment_learner=FailTreatment(),
            missingness_learner=missingness_learner,
        ),
        cross_fitting=CrossFitting(
            enabled=cross_fit,
            n_folds=10,
            repeats=1,
            stratify_by="none" if cross_fit else "treatment",
            targeting_scheme="pooled",
            fold_evaluation=False,
            split_plan=None,
        ),
        targeting=Targeting(
            fluctuation="logistic",
            algorithm="iterative",
            nuisance_bound=NUISANCE_BOUND,
            target_weights=False,
            max_iter=100,
            tol=1e-10,
        ),
        inference=Inference(simultaneous=False),
        runtime=Runtime(random_state=0, n_jobs=1),
    )


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    outcome_learner: Any | None = None,
    missingness_learner: Any | None = None,
    cross_fit: bool = True,
    adjustment: tuple[str, ...] = ("W",),
) -> Any:
    configured = method(
        outcome_learner=flexible_outcome_learner() if outcome_learner is None else outcome_learner,
        missingness_learner=flexible_response_learner()
        if missingness_learner is None
        else missingness_learner,
        cross_fit=cross_fit,
    )
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=adjustment,
            missingness="Delta",
        ),
    ).estimate(NaturalCourseMean(), method=configured)


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame)
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials={"ey_obs": natural_course_initial_estimate(result)},
    )


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    return cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, list(REPLICATE_COLUMNS)]

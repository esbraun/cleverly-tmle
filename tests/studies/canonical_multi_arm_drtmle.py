"""Registered multi-arm DR-TMLE comparison against pinned R ``drtmle``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cleverly.data import CausalData
from cleverly.estimators import DRTMLE
from cleverly.learners.crossfit import Folds, check_integrity
from cleverly.utils.parallel import map_parallel
from cleverly.validation.score import DEFAULT_TOLERANCE, score_threshold
from tests.parallel import STUDY_JOBS
from tests.studies import multi_arm_common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import replicate_seed

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2000
SEED = 20260830
N_FOLDS = 5
MAX_OUTER = 100
SCENARIO = "multi_arm_binary_drtmle"

FIT_DIAGNOSTIC_SOURCE_COLUMNS = (
    "implementation",
    "scenario",
    "replicate",
    "n",
    "score_max",
    "solver_reported",
    "solver_passed",
    "bound_active",
)
FIT_DIAGNOSTIC_COLUMNS = (
    *FIT_DIAGNOSTIC_SOURCE_COLUMNS[:5],
    "score_threshold",
    "score_passed",
    *FIT_DIAGNOSTIC_SOURCE_COLUMNS[5:],
)

STUDY = StudyRecord(
    name="multi-arm point-treatment DR-TMLE",
    slug="canonical-multi-arm-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "multi_arm_drtmle",
    document="docs/technical-reference/method-evidence/multi-arm-dr-tmle.md",
    anchor="multi-arm-dr-tmle",
    scenarios={SCENARIO: multi_arm_common.ALL_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20261004,
    margins=Margins(),
    implementation="cleverly-multi-arm-drtmle",
    reference="drtmle-r-multi-arm",
    publication_policy="reporting",
    extra_artifacts=("fit-diagnostics.csv",),
    modules=(
        "tests/studies/multi_arm_common.py",
        "tests/studies/canonical_multi_arm_drtmle.py",
        "tests/studies/multi_arm_drtmle_properties.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_multi_arm_drtmle",
    properties_module="tests.studies.multi_arm_drtmle_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "treatment_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified",),
    },
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
}
CONFIGURATION = {
    "outcome_family": "binomial",
    "treatment_levels": list(multi_arm_common.LABELS),
    "reference": multi_arm_common.REFERENCE,
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "reduced_crossfit": "pooled",
    "reduction": "univariate",
    "guard": ["Q", "g"],
    "update_order": "drtmle",
    "qsteps": 2,
    "max_outer": MAX_OUTER,
    "g_bounds": list(multi_arm_common.G_BOUNDS),
    "comparison_scope": (
        "armwise extension supported by both implementations; the source theorem is binary"
    ),
}


class FixedFoldDRTMLE(DRTMLE):
    """DR-TMLE whose outer assignment is supplied to both implementations."""

    def __init__(self, assignment: np.ndarray, **kwargs: Any) -> None:
        self._assignment = np.asarray(assignment, dtype=np.int64)
        super().__init__(**kwargs)

    def _folds(self, data: CausalData, seed: int | None = None) -> Folds:
        del seed
        folds = Folds(self._assignment.copy(), int(self._assignment.max()) + 1)
        check_integrity(folds, cluster=data.cluster)
        return folds


def fixed_folds(treatment: np.ndarray, seed: int) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    assignment = np.empty(len(treatment), dtype=np.int64)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(treatment)), treatment)):
        assignment[test] = fold
    return assignment


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    frame, truth = multi_arm_common.draw_from_seed(scenario, n, seed)
    frame["fold"] = fixed_folds(frame["A"].to_numpy(), seed + 1)
    return frame, truth


def draw_scenario(scenario: str, n: int, replicate: int):  # type: ignore[no-untyped-def]
    seed = replicate_seed(STUDY, scenario, replicate)
    return draw_from_seed(scenario, n, seed)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    del scenario
    result = (
        FixedFoldDRTMLE(
            frame["fold"].to_numpy(dtype=np.int64),
            outcome_learner=LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs"),
            treatment_learner=LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs"),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs"),
            cross_fit=True,
            n_folds=N_FOLDS,
            estimands=("ey", "ate", "rr", "or"),
            reference=multi_arm_common.REFERENCE,
            simultaneous=False,
            g_bounds=multi_arm_common.G_BOUNDS,
            max_outer=MAX_OUTER,
            max_iter=100,
            tol=1e-10,
            random_state=0,
            guard=("Q", "g"),
            reduction="univariate",
            reduced_crossfit="pooled",
            update_order="drtmle",
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )
    score = result.diagnostics.score_equations()
    if not np.isfinite([float(row.score) for row in score.rows]).all():
        raise RuntimeError("DR-TMLE empirical score audit is non-finite")
    return result


def cleverly_rows(
    frame: pd.DataFrame,
    truth: dict[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return multi_arm_common.cleverly_rows(STUDY, fit_cleverly, frame, truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    sample = frame.copy()
    sample["A_code"] = pd.Categorical(sample["A"], categories=multi_arm_common.LABELS).codes
    nuisance = result.repeats[0].nuisance
    for code in range(3):
        sample[f"qn{code}"] = nuisance.outcome.arms[float(code)]
        sample[f"gn{code}"] = nuisance.propensity.arm(float(code))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    rows = multi_arm_common.rows_from_result(
        result,
        truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        n=n,
        estimands=STUDY.scenarios[scenario],
    )
    score = result.diagnostics.score_equations()
    score_max = max(abs(float(row.score)) for row in score.rows)
    solver_passed = all(not row.failure for row in score.rows if row.kind == "fluctuation")
    for row in rows:
        row.update(
            score_max=score_max,
            solver_reported=True,
            solver_passed=solver_passed,
            bound_active=False,
        )
    return sample, truth_row, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, fitted in outcomes for row in fitted])
    return samples, truths, rows


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    key = ["implementation", "scenario", "replicate"]
    diagnostics = rows.loc[:, list(FIT_DIAGNOSTIC_SOURCE_COLUMNS)].drop_duplicates()
    thresholds = (
        rows.groupby(key, sort=False)
        .apply(
            lambda group: score_threshold(
                group["std_error"],
                int(group["n"].iloc[0]),
                tolerance=DEFAULT_TOLERANCE,
            ),
            include_groups=False,
        )
        .rename("score_threshold")
    )
    diagnostics = diagnostics.merge(thresholds, on=key, how="left", validate="1:1")
    diagnostics["score_passed"] = diagnostics["score_max"] <= diagnostics["score_threshold"]
    reference = diagnostics["implementation"].eq(STUDY.reference)
    diagnostics.loc[reference, "solver_reported"] = False
    diagnostics.loc[reference, "solver_passed"] = np.nan
    return {
        "fit-diagnostics.csv": diagnostics.loc[:, list(FIT_DIAGNOSTIC_COLUMNS)].sort_values(
            ["scenario", "replicate", "implementation"], ignore_index=True
        )
    }


def scientific_failures(
    artifacts: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    diagnostics = artifacts["fit-diagnostics.csv"]
    reported = diagnostics["solver_reported"].astype(bool)
    return {
        "score audit (both implementations, shared bar)": diagnostics.loc[
            ~diagnostics["score_passed"].astype(bool)
        ],
        "solver health (subject only)": diagnostics.loc[
            reported & ~diagnostics["solver_passed"].fillna(True).astype(bool)
        ],
    }

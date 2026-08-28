"""Canonical stacked CV-TMLE evidence against pinned R ``tmle3``.

The ordinary-TMLE study supplies the laws and exact truths.  This study changes the
estimator construction: both implementations use the same ten outer folds, out-of-fold
GLM nuisance predictions, one common update over the stacked validation rows, and a
whole-sample plug-in evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_tmle import SCENARIO_ESTIMANDS
from tests.studies.canonical_tmle import draw_from_seed as canonical_tmle_draw_from_seed
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 1000
SEED = 20240820
N_FOLDS = 10
G_BOUNDS = (0.025, 0.975)

PROPERTY_CELLS = {
    "double_robustness": (
        "both_correct",
        "outcome_correct",
        "treatment_correct",
        "both_wrong",
    ),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "interval_calibration": ("correctly_specified",),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
    "crossfit_overfitting": ("stacked_cvtmle", "in_sample_control"),
}

STUDY = StudyRecord(
    name="stacked point-treatment CV-TMLE",
    slug="canonical-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle3_cvtmle",
    document="docs/technical-reference/method-evidence/stacked-point-treatment-cv-tmle.md",
    anchor="stacked-point-treatment-cv-tmle",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly-stacked-cvtmle",
    reference="tmle3-cvtmle",
    incomparable_se=frozenset({"paf"}),
    modules=(
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/cvtmle_properties.py",
        "tests/studies/stacked_cvtmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_cvtmle",
    properties_module="tests.studies.stacked_cvtmle_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "targeting_scheme": "pooled",
    "cv_evaluation": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "q_bounds": "sample outcome range",
    "folds": "identical treatment-stratified assignments supplied to both implementations",
}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from *this* study's declared seed.

    The laws come from the ordinary-TMLE study; the samples do not.  This row is separate
    evidence, and it would not be if it re-used another study's draws.
    """
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample from an explicit seed, for the published-seed audit.

    The laws are the ordinary study's, so the draw is too.  What belongs to this study is
    the seed that reaches here, which :func:`draw_scenario` supplies from its own record.
    """
    return canonical_tmle_draw_from_seed(scenario, n, seed)


def cv_fit(
    frame: pd.DataFrame,
    *,
    binary: bool,
    estimands: Sequence[str],
    n_folds: int,
    repeats: int = 1,
    cv_evaluation: bool,
) -> Any:
    """The cross-fitted point-treatment construction the three CV studies share.

    Every argument that separates the three rows is a keyword this takes, and every
    argument they agree on is written once here.  ``repeats=1`` is the estimator's own
    default (``tmle.py:375``, assigned plainly at ``:423``), so the two studies that leave
    it out build the identical estimator they built before.

    Each caller passes its *own* ``n_folds`` and ``estimands``.  Those two names collide
    across the callers and disagree in value: ``N_FOLDS`` is 10 in this module and 5 in
    ``repeated_crossfit``, and the estimand lists differ between ``SCENARIO_ESTIMANDS``
    and ``SUPPORTED``.  Resolving either by import would silently move a published row.
    """
    outcome = (
        LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs") if binary else LinearRegression()
    )
    treatment = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(
            outcome_learner=outcome,
            treatment_learner=treatment,
            cross_fit=True,
            n_folds=n_folds,
            repeats=repeats,
            targeting_scheme="pooled",
            cv_evaluation=cv_evaluation,
            estimands=estimands,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def fit_cleverly(frame: pd.DataFrame) -> Any:
    """The public stacked-validation construction matched to R ``cvtmle=TRUE``.

    This study sniffs the outcome type from the frame rather than taking a scenario, which
    is what the R side does with the same payload.
    """
    binary = set(frame["Y"].dropna().unique()).issubset({0, 1})
    scenario = "binary" if binary else "continuous"
    return cv_fit(
        frame,
        binary=binary,
        estimands=SCENARIO_ESTIMANDS[scenario],
        n_folds=N_FOLDS,
        cv_evaluation=False,
    )


def rows_from_result(
    record: StudyRecord,
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, estimate in result.estimates.items():
        reference = float(truth[name])
        low, high = estimate.ci
        inference_estimate = (
            float(estimate.log_psi)
            if estimate.scale == "ratio" and estimate.log_psi is not None
            else float(estimate.psi)
        )
        rows.append(
            {
                "implementation": record.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": inference_estimate,
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if estimate.scale == "ratio" else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    return rows_from_result(STUDY, fit_cleverly(frame), truth, scenario, replicate)


def fitted_rows(
    record: StudyRecord,
    sample: Callable[[str, int, int], tuple[pd.DataFrame, Mapping[str, float]]],
    rows: Callable[[pd.DataFrame, Mapping[str, float], str, int], list[dict[str, Any]]],
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
) -> pd.DataFrame:
    """Every declared replication of a study whose published file is rows alone.

    ``sample`` and ``rows`` are the adopting study's own ``draw_scenario`` and
    ``cleverly_rows``, passed rather than imported.  Taking the row builder rather than the
    fitter keeps the driver on the same path as
    ``test_refitting_a_committed_replication_reproduces_its_row``, which refits through
    ``cleverly_rows``: a driver that reached past it could publish a row the refit gate
    never checks.

    This study's own :func:`draw_and_fit` does not use it.  It publishes the sample and the
    truth as well, because the reference implementation reads the identical rows and the
    identical fold assignment from the artifacts.
    """

    def replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
        scenario, index, size = payload
        frame, truth = sample(scenario, size, index)
        return rows(frame, truth, scenario, index)

    payloads = [
        ((scenario, index, n),) for scenario in record.scenarios for index in range(replicates)
    ]
    outcomes = map_parallel(replicate, payloads, n_jobs=n_jobs)
    built = pd.DataFrame([row for records in outcomes for row in records])
    return built.loc[:, list(REPLICATE_COLUMNS)]


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    payload_frame = frame.copy()
    payload_frame.insert(0, "fold", result.nuisance.folds.assignment)
    payload_frame.insert(0, "replicate", replicate)
    payload_frame.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return payload_frame, truth_row, rows_from_result(STUDY, result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([frame for frame, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, records in outcomes for row in records])
    return samples, truths, rows.loc[:, list(REPLICATE_COLUMNS)]

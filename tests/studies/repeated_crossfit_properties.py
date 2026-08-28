"""Independent properties for repeated point-treatment cross-fitting."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from cleverly.datasets import DGP, nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.conftest import OracleOutcomeContinuous, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.cvtmle_properties import generate, summarize
from tests.studies.evidence.properties import REPLICATE_COLUMNS, control_row
from tests.studies.evidence.property_verdicts import (
    finish,
    fold_repeat_stability_verdicts,
    repeat_aggregation_verdicts,
    repeat_variance_verdicts,
)
from tests.studies.repeated_crossfit import G_BOUNDS, N_FOLDS, REPEATS, STUDY

STABILITY_N = 600
STABILITY_REPLICATES = 200
STABILITY_SAMPLE_SEED = 7
STABILITY_FOLD_SEED = 1000

DECISION_N = 1000
DECISION_REPLICATES = 400
DECISION_SEED = 20260925
DECISION_REPEATS = 5
TAIL_THRESHOLD = 1.7506860712521692
TAIL_TREATMENT_PROBABILITY = 0.12
OTHER_TREATMENT_PROBABILITY = 0.50
TAIL_EFFECT_INCREMENT = 8.0
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))


def split_instability_dgp() -> DGP:
    """A rare, weakly treated stratum with strong effect modification."""

    def propensity(w: np.ndarray) -> np.ndarray:
        tail = w[:, 0] > TAIL_THRESHOLD
        return np.where(tail, TAIL_TREATMENT_PROBABILITY, OTHER_TREATMENT_PROBABILITY)

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        tail = (w[:, 0] > TAIL_THRESHOLD).astype(float)
        baseline = 0.5 * w[:, 0] + 0.4 * w[:, 1] - 0.2 * w[:, 2]
        return baseline + a * (0.5 + TAIL_EFFECT_INCREMENT * tail)

    return DGP(
        name="repeat_split_instability",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        noise_scale=1.0,
    )


def _tmle(
    outcome_learner: Any,
    treatment_learner: Any,
    *,
    repeats: int,
    cv_evaluation: bool,
    random_state: int,
    estimand: Literal["ate", "att"],
) -> TMLE:
    return TMLE(
        outcome_learner=outcome_learner,
        treatment_learner=treatment_learner,
        cross_fit=True,
        n_folds=N_FOLDS,
        repeats=repeats,
        targeting_scheme="pooled",
        cv_evaluation=cv_evaluation,
        estimands=(estimand,),
        simultaneous=False,
        g_bounds=G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=random_state,
    )


def _fit(frame: pd.DataFrame, estimator: TMLE) -> Any:
    covariates = [column for column in frame.columns if column.startswith("W")]
    return estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=covariates,
    ).single()


def _row(
    *,
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: float,
    standard_error: float,
) -> dict[str, Any]:
    return control_row(
        property_name=property_name,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        standard_error=standard_error,
        critical=CRITICAL,
    )


def _stability_replicate(payload: tuple[int]) -> list[dict[str, Any]]:
    (replicate,) = payload
    frame, truth = nonlinear_dgp().sample(STABILITY_N, seed=STABILITY_SAMPLE_SEED)
    seed = STABILITY_FOLD_SEED + replicate
    configurations = (
        ("rowwise_three_draw_average", "positive", REPEATS, False),
        ("one_fixed_split", "control", 1, False),
        ("equal_fold_average", "control", 1, True),
    )
    rows: list[dict[str, Any]] = []
    for cell, role, repeats, cv_evaluation in configurations:
        result = _fit(
            frame,
            _tmle(
                LinearRegression(),
                LogisticRegression(max_iter=1000),
                repeats=repeats,
                cv_evaluation=cv_evaluation,
                random_state=seed,
                estimand="att",
            ),
        )
        estimate = result["att"]
        rows.append(
            _row(
                property_name="fold_repeat_stability",
                cell=cell,
                role=role,
                replicate=replicate,
                n=STABILITY_N,
                requested=STABILITY_REPLICATES,
                truth=float(truth["sample_att"]),
                estimate=float(estimate.psi),
                standard_error=float(estimate.std_error),
            )
        )
    return rows


def stability_rows(*, replicates: int, n_jobs: int) -> pd.DataFrame:
    """Fit the fixed-sample split-noise experiment."""
    outcomes = map_parallel(
        _stability_replicate,
        [((replicate,),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    return pd.DataFrame([row for records in outcomes for row in records]).loc[
        :, list(REPLICATE_COLUMNS)
    ]


def _sample_seed(replicate: int) -> int:
    sequence = np.random.SeedSequence(DECISION_SEED, spawn_key=(replicate,))
    return int(sequence.generate_state(1)[0])


def _draw_reports(result: Any, estimand: str) -> tuple[Any, ...]:
    reports = []
    for repeat in result.repeats:
        estimates, _ = result.estimator.retarget(
            result.data,
            repeat.nuisance,
            estimands=(estimand,),
            g_bounds=result.config.g_bounds,
            g_bounds_conditional=result.config.g_bounds_conditional,
        )
        reports.append(estimates[estimand])
    return tuple(reports)


def _dml_mean_standard_error(reports: tuple[Any, ...], mean: float) -> float:
    points = np.asarray([estimate.psi for estimate in reports], dtype=float)
    errors = np.asarray([estimate.std_error for estimate in reports], dtype=float)
    return float(np.sqrt(np.mean(errors**2 + (points - mean) ** 2)))


def _decision_rows(
    *,
    label: str,
    result: Any,
    truth: float,
    replicate: int,
) -> list[dict[str, Any]]:
    reports = _draw_reports(result, "ate")
    points = np.asarray([estimate.psi for estimate in reports], dtype=float)
    mean = float(np.mean(points))
    median = float(np.median(points))
    current_se = float(result["ate"].std_error)
    adjusted_se = _dml_mean_standard_error(reports, mean)
    if not np.isclose(float(result["ate"].psi), mean, rtol=0, atol=1e-12):
        raise RuntimeError("the repeated fit is not the arithmetic mean of its draw reports")

    aggregation_roles = {
        "oracle_mean": "positive",
        "oracle_median": "control",
        "stress_mean": "control",
        "stress_median": "positive",
    }
    variance_roles = {
        "oracle_averaged_ic": "positive",
        "oracle_dml_mean": "positive",
        "stress_averaged_ic": "control",
        "stress_dml_mean": "positive",
    }
    return [
        _row(
            property_name="repeat_aggregation",
            cell=f"{label}_mean",
            role=aggregation_roles[f"{label}_mean"],
            replicate=replicate,
            n=DECISION_N,
            requested=DECISION_REPLICATES,
            truth=truth,
            estimate=mean,
            standard_error=current_se,
        ),
        _row(
            property_name="repeat_aggregation",
            cell=f"{label}_median",
            role=aggregation_roles[f"{label}_median"],
            replicate=replicate,
            n=DECISION_N,
            requested=DECISION_REPLICATES,
            truth=truth,
            estimate=median,
            standard_error=current_se,
        ),
        _row(
            property_name="repeat_variance",
            cell=f"{label}_averaged_ic",
            role=variance_roles[f"{label}_averaged_ic"],
            replicate=replicate,
            n=DECISION_N,
            requested=DECISION_REPLICATES,
            truth=truth,
            estimate=mean,
            standard_error=current_se,
        ),
        _row(
            property_name="repeat_variance",
            cell=f"{label}_dml_mean",
            role=variance_roles[f"{label}_dml_mean"],
            replicate=replicate,
            n=DECISION_N,
            requested=DECISION_REPLICATES,
            truth=truth,
            estimate=mean,
            standard_error=adjusted_se,
        ),
    ]


def _decision_replicate(payload: tuple[int]) -> list[dict[str, Any]]:
    (replicate,) = payload
    dgp = split_instability_dgp()
    frame, truth = dgp.sample(DECISION_N, seed=_sample_seed(replicate))
    fold_seed = DECISION_SEED + replicate
    oracle = _fit(
        frame,
        _tmle(
            OracleOutcomeContinuous(dgp),
            OracleTreatment(dgp),
            repeats=DECISION_REPEATS,
            cv_evaluation=False,
            random_state=fold_seed,
            estimand="ate",
        ),
    )
    stress = _fit(
        frame,
        _tmle(
            DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            OracleTreatment(dgp),
            repeats=DECISION_REPEATS,
            cv_evaluation=False,
            random_state=fold_seed,
            estimand="ate",
        ),
    )
    for oracle_repeat, stress_repeat in zip(oracle.repeats, stress.repeats, strict=True):
        if not np.array_equal(
            oracle_repeat.nuisance.folds.assignment,
            stress_repeat.nuisance.folds.assignment,
        ):
            raise RuntimeError("the oracle and stress fits did not use identical fold draws")
    target = float(truth["ate"])
    return [
        *_decision_rows(label="oracle", result=oracle, truth=target, replicate=replicate),
        *_decision_rows(label="stress", result=stress, truth=target, replicate=replicate),
    ]


def decision_rows(*, replicates: int, n_jobs: int) -> pd.DataFrame:
    """Fit mean, median, and variance decisions on identical samples and folds."""
    outcomes = map_parallel(
        _decision_replicate,
        [((replicate,),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    return pd.DataFrame([row for records in outcomes for row in records]).loc[
        :, list(REPLICATE_COLUMNS)
    ]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    shared = generate(
        STUDY,
        "repeated",
        repeats=REPEATS,
        n_folds=N_FOLDS,
        n_jobs=n_jobs,
    )
    stability = stability_rows(replicates=STABILITY_REPLICATES, n_jobs=n_jobs)
    decisions = decision_rows(replicates=DECISION_REPLICATES, n_jobs=n_jobs)
    return pd.concat([shared, stability, decisions], ignore_index=True).loc[
        :, list(REPLICATE_COLUMNS)
    ]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    parts = summarize(
        rows,
        STUDY,
        "repeated",
        extra_columns=(
            "spread_ratio_one_split_ci_lower",
            "spread_ratio_one_split_ci_upper",
            "spread_ratio_equal_fold_ci_lower",
            "spread_ratio_equal_fold_ci_upper",
            "rmse_ratio_ci_lower",
            "rmse_ratio_ci_upper",
            "p90_error_ratio_ci_lower",
            "p90_error_ratio_ci_upper",
        ),
        return_parts=True,
    )
    if not isinstance(parts, tuple):
        raise RuntimeError("the shared CV-TMLE summary did not return its parts")
    summary, rates = parts
    fold_repeat_stability_verdicts(
        summary,
        rows,
        STUDY,
        positive_cell="rowwise_three_draw_average",
        one_split_cell="one_fixed_split",
        equal_fold_cell="equal_fold_average",
    )
    repeat_aggregation_verdicts(summary, rows, STUDY)
    repeat_variance_verdicts(summary, rows, STUDY)
    return finish(summary, rates)

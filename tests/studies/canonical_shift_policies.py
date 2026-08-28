"""Registered evidence study for continuous modified treatment policies."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from cleverly.datasets import ShiftDGP, shift_dgp
from cleverly.estimators import TMLE
from cleverly.exceptions import PositivityWarning
from cleverly.interventions import Shift
from cleverly.learners.density import bin_edges
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

LMTP_COMMIT = "f04a2b47f46debc515ce4ae778e05ebfde922c44"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20260903
SCENARIO = "continuous_modified_policy"
PRIMARY_CURVATURE = 0.15
POLICIES: tuple[tuple[float, float | None, str], ...] = (
    (0.0, None, "natural course"),
    (0.25, None, "+0.25"),
    (0.5, 3.0, "+0.5 capped"),
)
ESTIMANDS = (
    "ey_shift[natural course]",
    "ey_shift[+0.25]",
    "ey_shift[+0.5 capped]",
    "ate_shift[+0.25 vs natural course]",
    "ate_shift[+0.5 capped vs natural course]",
)
DENSITY_BINS = 20
ORACLE_DENSITY_BINS = 320
PRIMARY_DENSITY_BINS = ORACLE_DENSITY_BINS


class OracleShiftDensity(BaseEstimator):
    """Exact pooled-hazard probabilities for the study's conditional normal dose."""

    def __init__(self, dgp: ShiftDGP, edges: tuple[float, ...]) -> None:
        self.dgp = dgp
        self.edges = edges

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleShiftDensity:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        w = design[:, : self.dgp.n_latent]
        index = np.rint(design[:, self.dgp.n_latent]).astype(int)
        edges = np.asarray(self.edges, dtype=float)
        mean = np.asarray(self.dgp.dose_mean(w), dtype=float)
        last = len(edges) - 2
        lower = np.where(index == 0, -np.inf, edges[index])
        upper = np.where(index == last, np.inf, edges[index + 1])
        cdf_lower = norm.cdf((lower - mean) / self.dgp.dose_scale)
        cdf_upper = norm.cdf((upper - mean) / self.dgp.dose_scale)
        stopped = cdf_upper - cdf_lower
        survived = 1.0 - cdf_lower
        hazard = np.divide(stopped, survived, out=np.zeros_like(stopped), where=survived > 0)
        hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - hazard, hazard])


class OracleShiftOutcome(BaseEstimator):
    """Exact conditional outcome mean on the estimator's scaled outcome range."""

    def __init__(self, dgp: ShiftDGP) -> None:
        self.dgp = dgp

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleShiftOutcome:
        del sample_weight
        raw = self._raw(np.asarray(X, dtype=float))
        slope, intercept = np.polyfit(raw, np.asarray(y, dtype=float), 1)
        self.slope_ = float(slope)
        self.intercept_ = float(intercept)
        return self

    def _raw(self, design: np.ndarray) -> np.ndarray:
        return np.asarray(self.dgp.outcome_mean(design[:, 1:], design[:, 0]), dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return self.intercept_ + self.slope_ * self._raw(np.asarray(X, dtype=float))


class QuadraticShiftOutcome(BaseEstimator):
    """Correctly specified primary-law regression with all coefficients estimated."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> QuadraticShiftOutcome:
        self.model_ = LinearRegression().fit(
            self._design(np.asarray(X, dtype=float)),
            np.asarray(y, dtype=float),
            sample_weight=sample_weight,
        )
        return self

    @staticmethod
    def _design(values: np.ndarray) -> np.ndarray:
        return np.column_stack([values, values[:, 0] ** 2])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(np.asarray(X, dtype=float))))


def shifts(*, capped: bool = True) -> tuple[Shift, ...]:
    return tuple(
        Shift(delta, cap=(cap if capped or name != "+0.5 capped" else None), name=name)
        for delta, cap, name in POLICIES
    )


STUDY = StudyRecord(
    name="ordinary continuous modified treatment policies",
    slug="shift-policies",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_shift",
    document="docs/technical-reference/method-evidence/continuous-modified-treatment-policies.md",
    anchor="continuous-modified-treatment-policies",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260913,
    margins=Margins(),
    publication_policy="reporting",
    implementation="cleverly",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_shift_policies.py",
        "tests/studies/shift_policy_properties.py",
        "tests/studies/intervention_study_helpers.py",
        "tests/discrete_law_shift.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_shift_policies",
    properties_module="tests.studies.shift_policy_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "density_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified", "shrunken_se_control"),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
        "targeting_necessity": ("shift__targeted", "shift__untargeted"),
        "ratio_necessity": ("shift__declared", "shift__reversed_control"),
        "cap_necessity": ("shift__declared_cap", "shift__uncapped_control"),
        "natural_course_identity": ("natural__shift", "natural__mean"),
    },
)

REFERENCE_METADATA = {
    "lmtp_commit": LMTP_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "one-node continuous modified treatment policy",
}

CONFIGURATION = {
    "construction": "ordinary",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "density_bins": PRIMARY_DENSITY_BINS,
    "density_bins_when_load_bearing": ORACLE_DENSITY_BINS,
    "outcome_curvature": PRIMARY_CURVATURE,
    "policies": [name for _, _, name in POLICIES],
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = shift_dgp(curvature=PRIMARY_CURVATURE).sample(
        n, shifts=POLICIES, seed=seed, backend="pandas"
    )
    return frame, truth


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    dgp = shift_dgp(curvature=PRIMARY_CURVATURE)
    edges = tuple(float(value) for value in bin_edges(np.asarray(frame["A"]), PRIMARY_DENSITY_BINS))
    estimator = TMLE(
        shifts=shifts(),
        outcome_learner=QuadraticShiftOutcome(),
        treatment_learner=OracleShiftDensity(dgp, edges),
        cross_fit=False,
        simultaneous=False,
        density_bins=PRIMARY_DENSITY_BINS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    )
    return fit_shift_estimator(estimator, frame, dgp)


def fit_shift_estimator(estimator: TMLE, frame: pd.DataFrame, dgp: ShiftDGP) -> Any:
    """Fit a shift study while containing its expected uncapped-support warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        return estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(dgp.covariate_names),
        ).single()


def reversed_ratio_control(result: Any) -> Any:
    """Retarget one fit after replacing only the shift-density-ratio direction."""
    evaluated = result.nuisance.shifts
    if evaluated is None:  # pragma: no cover - a study contract guard
        raise AssertionError("a shift fit did not retain its evaluated policies")

    def reciprocal(values: np.ndarray) -> np.ndarray:
        bounded = np.clip(np.asarray(values, dtype=float), 0.2, 5.0)
        return np.clip(1.0 / bounded, 0.2, 5.0)

    reversed_shifts = replace(
        evaluated,
        ratio=reciprocal(evaluated.ratio),
        ratio_at=reciprocal(evaluated.ratio_at),
    )
    nuisance = replace(result.nuisance, shifts=reversed_shifts)
    estimates, fluctuations = result.estimator.retarget(
        result.data,
        nuisance,
        estimands=("ey_shift", "ate_shift"),
    )
    repeat = replace(
        result.repeats[0],
        nuisance=nuisance,
        fluctuations=fluctuations,
        psi={name: estimate.psi for name, estimate in estimates.items()},
    )
    return replace(result, estimates=estimates, repeats=(repeat,))


def initial_estimates(result: Any) -> dict[str, float]:
    """Return the untargeted shifted plug-ins under public parameter names."""
    nuisance = result.nuisance
    weights = np.asarray(result.data.weights, dtype=float)
    means = {
        name: float(
            np.average(
                nuisance.scaler.unscale_levels(nuisance.outcome.arms[float(index)]), weights=weights
            )
        )
        for index, (_, _, name) in enumerate(POLICIES)
    }
    reference = POLICIES[0][2]
    out = {f"ey_shift[{name}]": value for name, value in means.items()}
    for name, value in means.items():
        if name != reference:
            out[f"ate_shift[{name} vs {reference}]"] = value - means[reference]
    return out


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame)
    initials = initial_estimates(result)
    rows: list[dict[str, Any]] = []
    for name in ESTIMANDS:
        estimate = result[name]
        low, high = estimate.ci
        truth = float(reference[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": truth,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": initials[name],
            }
        )
    return rows


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
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

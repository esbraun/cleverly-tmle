"""Canonical ordinary longitudinal MSM projection evidence study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import RULE_LABEL, make_longitudinal, rule_arm_at_node_two
from cleverly.longitudinal import LTMLE
from cleverly.msm import MSM, solve_projection
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism, QuasiBinomialGLM
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

LTMLE_VERSION = "1.3-0"
LTMLE_SOURCE_COMMIT = "338c029dae9692ef20714125773da7037688993b"
LTMLE_TARBALL_SHA256 = "fb31d0dd6ab81687b81f3279b414c21e91c655e10aac12f73fc6723efd848aad"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_500
SEED = 20260829
SCENARIO = "censored_regimen_projection"
G_BOUNDS = (1e-8, 1.0)

REGIMENS: dict[str, Any] = {
    "never": 0,
    "always": 1,
    "early": (1, 0),
    RULE_LABEL: (1, lambda history: rule_arm_at_node_two(history["L2"])),
}
DURATION = {"never": 0.0, "always": 2.0, "early": 1.0, RULE_LABEL: 1.0}
PROJECTION_WEIGHT = {"never": 0.1, "always": 10.0, "early": 0.1, RULE_LABEL: 10.0}
TERMS = ("(intercept)", "duration")
ESTIMANDS = tuple(f"msm_regimen[{term}]" for term in TERMS)

STUDY = StudyRecord(
    name="ordinary longitudinal MSM projection",
    slug="longitudinal-msm",
    artifacts=ROOT / "tests" / "canonical" / "ltmle_msm",
    document="docs/technical-reference/method-evidence/ordinary-longitudinal-msm-projection.md",
    anchor="ordinary-longitudinal-msm-projection",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="ltmle projected regimen fits",
    modules=(
        "tests/studies/canonical_longitudinal_msm.py",
        "tests/studies/longitudinal_msm_properties.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_longitudinal_msm",
    properties_module="tests.studies.longitudinal_msm_properties",
    property_cells={
        "double_robustness": tuple(
            f"{term}__{configuration}"
            for term in TERMS
            for configuration in (
                "both_correct",
                "outcome_correct",
                "mechanism_correct",
                "both_wrong",
            )
        ),
        "root_n_and_efficiency": tuple(
            f"{term}__n_{size}" for term in TERMS for size in (500, 2000, 8000)
        ),
        "root_n_rate": tuple(
            f"{term}__{statistic}"
            for term in TERMS
            for statistic in ("empirical_sd", "reported_se")
        ),
        "interval_calibration": tuple(
            f"{term}__{cell}"
            for term in TERMS
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
        "type_i_error": ("duration__sharp_null",),
        "power": ("duration__alternative",),
        "targeting_necessity": ("duration__targeted", "duration__untargeted"),
        "projection_necessity": (
            "duration__declared_weights",
            "duration__uniform_weights",
        ),
    },
)

REFERENCE_METADATA = {
    "ltmle_version": LTMLE_VERSION,
    "ltmle_source_commit": LTMLE_SOURCE_COMMIT,
    "ltmle_tarball_sha256": LTMLE_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "fixed projection of correlated ltmle regimen estimates and ICs",
}

CONFIGURATION = {
    "construction": "ordinary pooled longitudinal MSM",
    "link": "identity",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "variance_method": "ic",
    "g_bounds": list(G_BOUNDS),
    "regimens": list(REGIMENS),
    "terms": list(TERMS),
    "duration": DURATION,
    "projection_weights": PROJECTION_WEIGHT,
}

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def declared_msm(
    durations: Mapping[str, float] = DURATION,
    projection_weights: Mapping[str, float] = PROJECTION_WEIGHT,
) -> MSM:
    """The two-term working model and its fixed regimen measure."""

    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        return np.column_stack(
            [np.ones(len(frame)), np.full(len(frame), float(durations[str(label)]))]
        )

    def weight(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        return np.full(len(frame), float(projection_weights[str(label)]))

    return MSM(design=design, terms=TERMS, weights=weight)


def project_means(means: Mapping[str, float]) -> np.ndarray:
    labels = tuple(REGIMENS)
    design = np.column_stack([np.ones(len(labels)), [DURATION[label] for label in labels]])
    weights = np.array([PROJECTION_WEIGHT[label] for label in labels])
    target = np.array([means[label] for label in labels])
    return np.linalg.solve(design.T @ (weights[:, None] * design), design.T @ (weights * target))


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, source_truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
    means = {label: float(source_truth[f"ey_regimen[{label}]"]) for label in REGIMENS}
    beta = project_means(means)
    return frame, dict(zip(ESTIMANDS, beta, strict=True))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(STUDY, scenario, replicate))


def fit_cleverly(frame: pd.DataFrame) -> Any:
    return LTMLE(
        REGIMENS,
        msm=declared_msm(),
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, **COLUMNS)


def initial_beta(result: Any, labels: tuple[str, ...] = tuple(REGIMENS)) -> np.ndarray:
    raw = np.column_stack(
        [result.scaler.unscale_levels(result.fits[label].steps[0].initial) for label in labels]
    )
    return solve_projection(
        result.msm.design,
        result.msm.weights,
        raw,
        result.data.weights,
        str(result.msm.link),
    ).beta


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    result = fit_cleverly(frame)
    initials = initial_beta(result)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(ESTIMANDS):
        estimate = result[name]
        low, high = estimate.ci
        target = float(reference[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": target,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= target <= high),
                "initial_estimate": float(initials[index]),
            }
        )
    return rows


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference.items()
    ]
    return sample, truths, cleverly_rows(frame, reference, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

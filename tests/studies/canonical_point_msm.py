"""Canonical point-treatment MSM projection evidence study."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.msm import MSM, solve_projection
from cleverly.utils.parallel import map_parallel
from tests import discrete_law as law
from tests.conftest import OracleOutcomeContinuous, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed

TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20260828
SCENARIO = "bounded_continuous_projection"
G_BOUNDS = (0.01, 0.99)
ESTIMANDS = tuple(f"msm[{term}]" for term in law.MSM_TERMS)
# Quadratic in W and additive in A.  The pinned R GLM sees W and W^2, so its nuisance
# regression is correctly specified, while the declared MSM deliberately omits W^2 and
# therefore remains a genuine projection rather than a saturated outcome model.
PRIMARY_Q = np.array([[0.25, 0.50], [0.35, 0.60], [0.55, 0.80]])
PROJECTION_WEIGHTS = np.array([[1.0 + 0.5 * a + 5.0 * w for a in range(2)] for w in range(3)])

STUDY = StudyRecord(
    name="ordinary point-treatment MSM projection",
    slug="point-msm",
    artifacts=ROOT / "tests" / "canonical" / "tmle3_msm",
    document="docs/technical-reference/method-evidence/point-treatment-msm-projection.md",
    anchor="point-treatment-msm-projection",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260830,
    margins=Margins(),
    implementation="cleverly",
    reference="tmle3",
    modules=(
        "tests/studies/canonical_point_msm.py",
        "tests/studies/point_msm_properties.py",
        "tests/discrete_law.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_point_msm",
    properties_module="tests.studies.point_msm_properties",
    property_cells={
        "double_robustness": tuple(
            f"{term}__{configuration}"
            for term in law.MSM_TERMS
            for configuration in (
                "both_correct",
                "outcome_correct",
                "mechanism_correct",
                "both_wrong",
            )
        ),
        "root_n_and_efficiency": tuple(
            f"{term}__n_{size}" for term in law.MSM_TERMS for size in (500, 2000, 8000)
        ),
        "root_n_rate": tuple(
            f"{term}__{statistic}"
            for term in law.MSM_TERMS
            for statistic in ("empirical_sd", "reported_se")
        ),
        "interval_calibration": tuple(
            f"{term}__{cell}"
            for term in law.MSM_TERMS
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
        "type_i_error": ("a__sharp_null",),
        "power": ("a__alternative",),
        "targeting_necessity": ("a__targeted", "a__untargeted"),
        "projection_necessity": ("W__declared_weights", "W__uniform_weights"),
    },
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "tmle3 Param_MSM with Gaussian identity-link projection",
}

CONFIGURATION = {
    "construction": "ordinary",
    "link": "identity",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "terms": list(law.MSM_TERMS),
    "projection_weights": "1 + 0.5 * treatment + 5 * W",
}


def declared_msm(*, uniform: bool = False) -> MSM:
    """The study's working design and fixed projection measure."""

    def design(arm: Any, frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=float)
        return np.column_stack([np.ones(len(w)), np.full(len(w), float(arm)), w])

    def weight(arm: Any, frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=float)
        return np.ones(len(w)) if uniform else 1.0 + 0.5 * float(arm) + 5.0 * w

    return MSM(design=design, terms=law.MSM_TERMS, weights=weight)


def truth() -> dict[str, float]:
    gram = np.einsum(
        "wap,waq,wa,w->pq", law.MSM_DESIGN, law.MSM_DESIGN, PROJECTION_WEIGHTS, law.P_W
    )
    moment = np.einsum("wap,wa,wa,w->p", law.MSM_DESIGN, PROJECTION_WEIGHTS, PRIMARY_Q, law.P_W)
    return dict(zip(ESTIMANDS, np.linalg.solve(gram, moment), strict=True))


def primary_law() -> law.DiscreteLaw:
    probs = np.empty_like(law.PROBS)
    for w, a, y in law.SUPPORT:
        arm = law.G[w] if a == 1 else 1.0 - law.G[w]
        outcome = PRIMARY_Q[w, a] if y == 1 else 1.0 - PRIMARY_Q[w, a]
        probs[w, a, y] = law.P_W[w] * arm * outcome
    return law.DiscreteLaw(probs)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw a continuous-outcome version of the finite-support MSM law."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    rng = np.random.default_rng(seed)
    w = rng.choice(3, size=n, p=law.P_W)
    a = rng.binomial(1, law.G[w]).astype(float)
    mean = PRIMARY_Q[w, a.astype(int)]
    concentration = 24.0
    y = rng.beta(mean * concentration, (1.0 - mean) * concentration)
    return pd.DataFrame({"Y": y, "A": a, "W": w.astype(float)}), truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(STUDY, scenario, replicate))


def fit_cleverly(frame: pd.DataFrame) -> Any:
    dgp = primary_law()
    return (
        TMLE(
            msm=declared_msm(),
            outcome_learner=OracleOutcomeContinuous(dgp),
            treatment_learner=OracleTreatment(dgp),
            cross_fit=False,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


def initial_beta(result: Any) -> np.ndarray:
    """Project the fitted initial arm regressions before fluctuation."""
    msm = result.nuisance.msm
    if msm is None:  # pragma: no cover - a study contract guard
        raise AssertionError("an MSM fit did not retain its evaluated working model")
    raw = np.column_stack(
        [
            result.nuisance.scaler.unscale_levels(result.nuisance.outcome.arms[arm])
            for arm in msm.arms
        ]
    )
    return solve_projection(msm.design, msm.weights, raw, result.data.weights, msm.link).beta


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

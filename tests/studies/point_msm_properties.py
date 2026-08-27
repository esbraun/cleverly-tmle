"""Independent properties for the point-treatment MSM projection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law as law
from tests.conftest import OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_point_msm import (
    G_BOUNDS,
    PROJECTION_WEIGHTS,
    STUDY,
    declared_msm,
    initial_beta,
)
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 1_200
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 1_600
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
TARGETING_REPLICATES = DOUBLE_ROBUST_REPLICATES
TARGETING_N = DOUBLE_ROBUST_N
PROJECTION_REPLICATES = DOUBLE_ROBUST_REPLICATES
PROJECTION_N = DOUBLE_ROBUST_N

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.25
PROJECTION_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

TERMS = law.MSM_TERMS
NAMES = {term: f"msm[{term}]" for term in TERMS}


def coefficients(probs: np.ndarray, *, uniform: bool = False) -> np.ndarray:
    p = np.asarray(probs)
    p_w = p.sum(axis=(1, 2))
    q = p[:, :, 1] / p.sum(axis=2)
    projection = np.ones_like(PROJECTION_WEIGHTS) if uniform else PROJECTION_WEIGHTS
    gram = np.einsum("wap,waq,wa,w->pq", law.MSM_DESIGN, law.MSM_DESIGN, projection, p_w)
    moment = np.einsum("wap,wa,wa,w->p", law.MSM_DESIGN, projection, q, p_w)
    return np.linalg.solve(gram, moment)


def influence_curves(probs: np.ndarray = law.PROBS, *, step: float = 1e-30) -> np.ndarray:
    base = np.asarray(probs, dtype=complex)
    curves = np.empty((len(law.SUPPORT), len(TERMS)))
    for point, support in enumerate(law.SUPPORT):
        mass = np.zeros_like(base)
        mass[support] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        curves[point] = np.imag(coefficients(perturbed)) / step
    return curves


TRUTH = dict(zip(NAMES.values(), coefficients(law.PROBS), strict=True))
EFFICIENCY_CURVES = influence_curves()
EFFICIENCY_SD = {
    term: float(np.sqrt(np.sum(law.PROBS.reshape(-1) * EFFICIENCY_CURVES[:, index] ** 2)))
    for index, term in enumerate(TERMS)
}


def probabilities(q: np.ndarray, *, g: np.ndarray = law.G) -> np.ndarray:
    """Build ``P(W, A, Y)`` from this law's fixed baseline and treatment mechanisms."""
    probs = np.empty_like(law.PROBS)
    for w, a, y in law.SUPPORT:
        arm = g[w] if a == 1 else 1.0 - g[w]
        outcome = q[w, a] if y == 1 else 1.0 - q[w, a]
        probs[w, a, y] = law.P_W[w] * arm * outcome
    return probs


NULL_Q = np.column_stack([0.25 + 0.2 * np.arange(3), 0.25 + 0.2 * np.arange(3)])
NULL_PROBS = probabilities(NULL_Q)
NULL_TRUTH = float(coefficients(NULL_PROBS)[TERMS.index("a")])

# Fixed, deliberately contrary nuisance functions make the both-wrong cell a stable negative
# control.  A sample-prior classifier was only weakly wrong for the intercept projection and
# made its discrimination verdict depend on the bootstrap stream.
WRONG_Q = 1.0 - law.Q
WRONG_G = np.array([0.75, 0.25, 0.75])
WRONG_PROBS = probabilities(WRONG_Q, g=WRONG_G)


def sample(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(law.SUPPORT), size=n, p=probs.reshape(-1))
    values = np.asarray(law.SUPPORT, dtype=float)[cells]
    return pd.DataFrame({"W": values[:, 0], "A": values[:, 1], "Y": values[:, 2]})


def _learners(probs: np.ndarray, configuration: str) -> tuple[Any, Any]:
    dgp = law.DiscreteLaw(probs)
    wrong = law.DiscreteLaw(WRONG_PROBS)
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    outcome = OracleOutcome(dgp) if q_correct else OracleOutcome(wrong)
    treatment = OracleTreatment(dgp) if g_correct else OracleTreatment(wrong)
    return outcome, treatment


def fit(
    frame: pd.DataFrame,
    probs: np.ndarray,
    configuration: str = "both_correct",
    *,
    uniform: bool = False,
) -> Any:
    outcome, treatment = _learners(probs, configuration)
    return (
        TMLE(
            msm=declared_msm(uniform=uniform),
            outcome_learner=outcome,
            treatment_learner=treatment,
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


def _fit_replication(
    payload: tuple[str, str, int, int, int, int, str],
) -> list[dict[str, Any]]:
    property_name, cell_suffix, replicate, n, requested, seed, configuration = payload
    probs = NULL_PROBS if property_name == "type_i_error" else law.PROBS
    frame = sample(probs, n, seed)
    result = fit(frame, probs, configuration)
    terms = (
        ("a",)
        if property_name
        in {
            "type_i_error",
            "power",
            "targeting_necessity",
        }
        else (("W",) if property_name == "projection_necessity" else TERMS)
    )
    rows: list[dict[str, Any]] = []
    for term in terms:
        name = NAMES[term]
        truth = NULL_TRUTH if property_name == "type_i_error" else float(TRUTH[name])
        role = (
            "control"
            if cell_suffix == "both_wrong"
            or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
            else "positive"
        )
        rows.append(
            replicate_row(
                property_name=property_name,
                cell=f"{term}__{cell_suffix}",
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=result[name],
                alpha=STUDY.margins.alpha,
            )
        )
        if property_name == "targeting_necessity":
            index = TERMS.index(term)
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{term}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=float(initial_beta(result)[index]),
                    standard_error=float(result[name].std_error),
                    critical=CRITICAL,
                )
            )
        if property_name == "projection_necessity":
            wrong = fit(frame, probs, configuration, uniform=True)
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{term}__uniform_weights",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=float(wrong[name].psi),
                    standard_error=float(wrong[name].std_error),
                    critical=CRITICAL,
                )
            )
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[tuple[str, str, int, int, str]] = []
    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
        specs.append(
            (
                "double_robustness",
                configuration,
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                configuration,
            )
        )
    for size in RATE_SIZES:
        specs.append(("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct"))
    specs.extend(
        [
            (
                "interval_calibration",
                "correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ("type_i_error", "sharp_null", NULL_N, NULL_REPLICATES, "both_correct"),
            ("power", "alternative", NULL_N, NULL_REPLICATES, "both_correct"),
            (
                "targeting_necessity",
                "targeted",
                TARGETING_N,
                TARGETING_REPLICATES,
                "mechanism_correct",
            ),
            (
                "projection_necessity",
                "declared_weights",
                PROJECTION_N,
                PROJECTION_REPLICATES,
                "both_correct",
            ),
        ]
    )
    payloads: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            payloads.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return payloads


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=TERMS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "projection_displacement"),
        rate_labels=TERMS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("a",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="projection_necessity",
        labels=("W",),
        arms=("declared_weights", "uniform_weights"),
        column="projection_displacement",
        threshold=PROJECTION_DISPLACEMENT,
    )
    return finish(summary, rates)

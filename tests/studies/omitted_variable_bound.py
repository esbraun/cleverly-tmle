"""Registered evidence for the standard error of the omitted-variable bound (RM22).

The subject is not an estimator of a causal parameter but the one-sided limits of the
bias-adjusted bounds of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026), Theorem 4.
Each bound is read as an estimate of a known population value, and its standard error is read
back off the reported limit, so the shared schema and the shared verdicts apply unchanged.

The law is :func:`cleverly.datasets.make_linear_ate`: four standard normal covariates, a
treatment mechanism ``expit(0.3 W1 - 0.2 W2 + 0.1 W3)``, a standard normal outcome noise and a
constant effect of 1.5.  Both nuisances are correctly specified main-effects GLMs.  Every truth
is a closed form: ``E[g / (1 - g)] = E[exp(eta)] = exp(0.07)`` for the linear index ``eta`` with
variance 0.14, and ``P(A = 1) = 1/2`` by symmetry, so ``nu^2`` is ``4 exp(0.07)`` for the ATT and
the ATC and ``2 + 2 exp(0.07)`` for the ATE, and ``sigma^2 = 1``.

The RM22 plan in ``docs/roadmap.md`` declared this study, its margins, its budget and its
red-cell route before any run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_linear_ate
from cleverly.estimators import TMLE
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import omitted_variable as omitted_variable_module
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 10_000
PRIMARY_N = 1_000
SEED = 20262201
RESAMPLING_SEED = 20262202

#: The large strength of the RM22 contract.  At the default 0.03 the share term moves no
#: standard-error ratio by 0.001, so the default strength could not show the correction.
CF_Y = 0.5
CF_D = 0.3
RHO = 1.0

#: The estimands the fit reports, and the two ends of each bound.
FITTED = ("att", "atc", "ate")
ENDS = ("lower", "upper")

#: The primary estimands, one per bound end: ``att_lower`` ... ``ate_upper``.
BOUNDS = tuple(f"{name}_{end}" for name in FITTED for end in ENDS)

SCENARIOS = {"linear": BOUNDS}

#: The two-sided 95% multiplier of each row's interval, and the one-sided multiplier the bound
#: reports its limit with.
TWO_SIDED = float(stats.norm.ppf(0.975))
ONE_SIDED = float(stats.norm.ppf(0.95))

#: The variance of the linear index of the treatment mechanism, ``0.3^2 + 0.2^2 + 0.1^2``.
INDEX_VARIANCE = 0.14

PROPERTY_CELLS = {
    "interval_calibration": (
        *(f"{bound}__correctly_specified" for bound in BOUNDS),
        "att_lower__inflated_se_control",
        "att_upper__inflated_se_control",
    ),
}

STUDY = StudyRecord(
    name="omitted-variable bound standard error",
    slug="omitted-variable-bound-se",
    artifacts=ROOT / "tests" / "canonical" / "omitted_variable_bound",
    document="docs/technical-reference/method-evidence/omitted-variable-bound-standard-error.md",
    anchor="omitted-variable-bound-standard-error",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-omitted-variable-bound",
    reference=None,
    modules=(
        "tests/studies/omitted_variable_bound.py",
        "tests/studies/omitted_variable_bound_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.omitted_variable_bound",
    properties_module="tests.studies.omitted_variable_bound_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "law": "make_linear_ate(n=1000): W ~ N(0, I_4), g = expit(0.3 W1 - 0.2 W2 + 0.1 W3), "
    "Y = 1.5 A + linear(W) + N(0, 1)",
    "fit": "in-sample TMLE, LinearRegression outcome, unpenalized main-effects "
    "LogisticRegression treatment, simultaneous=False, estimands ate, att and atc",
    "nu2_estimator": "auto, which resolves to doubly_robust",
    "strength": {"cf_y": CF_Y, "cf_d": CF_D, "rho": RHO},
    "primary_row": "estimate = the bound; std_error = (lower - ci_lower) / z_0.95 for a lower "
    "bound and (ci_upper - upper) / z_0.95 for an upper bound; interval = estimate +/- "
    "1.959964 std_error",
    "control": "the same fits, with the standard error from the curve of nu^2 without the "
    "conditioning-share term, which is the curve before RM22",
    "truth": "closed form: nu2 = 4 exp(0.07) for att and atc, 2 + 2 exp(0.07) for ate, "
    "sigma2 = 1, effect 1.5",
}


def truths() -> dict[str, float]:
    """The six population bounds, in closed form."""
    strength = math.sqrt(CF_Y * CF_D / (1.0 - CF_D)) * abs(RHO)
    odds = math.exp(INDEX_VARIANCE / 2.0)
    nu2 = {"att": 4.0 * odds, "atc": 4.0 * odds, "ate": 2.0 + 2.0 * odds}
    effect = 1.5
    values: dict[str, float] = {}
    for name in FITTED:
        bias = strength * math.sqrt(1.0 * nu2[name])
        values[f"{name}_lower"] = effect - bias
        values[f"{name}_upper"] = effect + bias
    return values


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit."""
    if scenario != "linear":
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    frame, _ = make_linear_ate(n=n, seed=seed)
    return frame, truths()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _unpenalized_logistic() -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


def fit(frame: pd.DataFrame) -> Any:
    """The declared in-sample TMLE with correctly specified GLMs.

    No warning is suppressed.  The declared fit raises none on this law, so a warning that a
    later package raises reaches the regeneration log instead of vanishing.
    """
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=_unpenalized_logistic(),
            cross_fit=False,
            simultaneous=False,
            estimands=FITTED,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


#: How closely the rebuilt curve must reproduce the package's own curve and standard error.
REBUILD_TOLERANCE = 1e-10


def _bias_curve(elements: Any, psi_nu2: Any) -> Any:
    """Theorem 4's curve of the maximal bias, from a curve of ``nu^2``."""
    return (elements.sigma2 * psi_nu2 + elements.nu2 * np.asarray(elements.psi_sigma2)) / (
        2.0 * elements.max_bias
    )


def bound_standard_errors(result: Any, name: str) -> dict[str, tuple[float, float]]:
    """The bound and its standard error at each end, with and without the share term.

    ``"reported"`` reads the package's limit back: ``(lower - ci_lower) / z_0.95``.
    ``"without_share_term"`` rebuilds the same curve after removing
    :func:`~cleverly.sensitivity.omitted_variable._conditioning_share_influence`, which is the
    curve the package reported before RM22.  For an unconditional parameter the two agree.
    """
    ov = omitted_variable_module
    bounds = ov.omitted_variable_bounds(result, name, cf_y=CF_Y, cf_d=CF_D, rho=RHO)
    reported = {
        "lower": (bounds.lower, (bounds.lower - bounds.ci_lower) / ONE_SIDED),
        "upper": (bounds.upper, (bounds.ci_upper - bounds.upper) / ONE_SIDED),
    }
    elements = ov.sensitivity_elements(result, name)
    assert elements.psi_nu2 is not None
    psi_nu2 = np.asarray(elements.psi_nu2, dtype=float)
    conditioning = ov.resolve_parameter(result, name).conditions_on
    term = np.zeros_like(psi_nu2)
    if conditioning is not None:
        data = result.data
        arms = list(result.repeats[0].nuisance.arms)
        indicator = np.asarray(data.treatment == conditioning, dtype=float)
        share = float(data.arm_fractions[arms.index(conditioning)])
        term = ov._conditioning_share_influence(elements.nu2, indicator, share, data.weights)
    psi_nu2 = psi_nu2 - term
    psi_bias = _bias_curve(elements, psi_nu2)
    curve = np.asarray(result[name].influence_curve, dtype=float)
    cluster = result.data.cluster
    strength = bounds.confounding_strength
    # The control is this rebuild minus the term, so the rebuild with the term has to be the
    # package's own curve and standard error.  A later package change that moved either one
    # would otherwise change what the control measures without a word.  For the ATE the term
    # is zero and the control is the package's curve outright.
    rebuilt = _bias_curve(elements, psi_nu2 + term)
    assert elements.psi_max_bias is not None
    if not np.allclose(rebuilt, elements.psi_max_bias, rtol=0.0, atol=REBUILD_TOLERANCE):
        raise RuntimeError(f"the rebuilt bias curve of {name!r} is not the package's")
    if conditioning is None and not np.array_equal(psi_bias, elements.psi_max_bias):
        raise RuntimeError(f"the control curve of {name!r} is not the package's curve")
    for end, sign in (("lower", -1.0), ("upper", 1.0)):
        rebuilt_se = float(np.sqrt(influence_variance(curve + sign * strength * rebuilt, cluster)))
        if abs(rebuilt_se - reported[end][1]) > REBUILD_TOLERANCE * max(1.0, rebuilt_se):
            raise RuntimeError(f"the rebuilt {end} standard error of {name!r} is not reported")
    without = {
        "lower": (
            bounds.lower,
            float(np.sqrt(influence_variance(curve - strength * psi_bias, cluster))),
        ),
        "upper": (
            bounds.upper,
            float(np.sqrt(influence_variance(curve + strength * psi_bias, cluster))),
        ),
    }
    return {
        **{f"reported_{end}": value for end, value in reported.items()},
        **{f"without_share_term_{end}": value for end, value in without.items()},
    }


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema, one row per bound end."""
    result = fit(frame)
    rows: list[dict[str, Any]] = []
    for name in FITTED:
        values = bound_standard_errors(result, name)
        for end in ENDS:
            estimate, std_error = values[f"reported_{end}"]
            bound = f"{name}_{end}"
            reference = float(truth[bound])
            low = estimate - TWO_SIDED * std_error
            high = estimate + TWO_SIDED * std_error
            rows.append(
                {
                    "implementation": STUDY.implementation,
                    "scenario": scenario,
                    "replicate": replicate,
                    "n": result.n,
                    "estimand": bound,
                    "truth": reference,
                    "estimate": float(estimate),
                    "inference_estimate": float(estimate),
                    "std_error": float(std_error),
                    "ci_lower": float(low),
                    "ci_upper": float(high),
                    "inference_scale": "identity",
                    "covered": int(low <= reference <= high),
                    "initial_estimate": math.nan,
                }
            )
    return rows


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, index, n = payload
    frame, truth = draw_scenario(scenario, n, index)
    return cleverly_rows(frame, truth, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Draw and fit every declared primary replication."""
    payloads = [
        ((scenario, index, n),) for scenario in STUDY.scenarios for index in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    built = pd.DataFrame([row for records in outcomes for row in records])
    return built.loc[:, list(REPLICATE_COLUMNS)]

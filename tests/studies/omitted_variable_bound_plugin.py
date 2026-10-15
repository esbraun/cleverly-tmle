r"""Registered evidence for the plug-in omitted-variable bound's standard error (X26).

The subject is the curve of the plug-in :math:`\hat\nu^2 = E_n[\hat\alpha^2]` that X26 stacks
with the score of a declared parametric treatment mechanism, and the one-sided limits of the
bias-adjusted bounds that the curve gives (Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis
2026, Theorem 4).  The mechanism term comes from the stacked estimating equation of the
logistic score and the second moment (Saul and Hudgens 2020, *J. Stat. Softw.* 92(2),
Section 2, Equations (1) and (2)).  A declared known mechanism has no mechanism term.

Each primary row reads one of three quantities as an estimate of a known population value:
:math:`\hat\nu^2` itself, with the standard error of its own curve, and the lower and upper
bound, with the standard error read back off the reported limit as in the RM22 study.

**The scenarios.**

``binary_logistic``
    :func:`cleverly.datasets.make_linear_ate` at ``n = 1000``.  In-sample TMLE, a
    ``LinearRegression`` outcome and ``LogisticRegression(C=numpy.inf, max_iter=2000)``.
``binary_logistic_crossfit``
    The same with ``cross_fit=True, n_folds=5``.  A cross-fitted continuous outcome needs a
    declared outcome support, so the fit passes ``q_bounds=(-40.0, 45.0)``.  The outcome is
    ``2 + 1.5 A`` plus a normal term with standard deviation ``sqrt(3.05)``, about 1.746, so
    each bound lies more than 23 standard deviations from both arm means.  No draw reaches it,
    and the bounds only fix the scale of the logistic fluctuation.
``binary_logistic_ridge``
    The same as ``binary_logistic`` with ``LogisticRegression(max_iter=1000)``: a fixed L2
    penalty with ``C = 1``, the idiom of the documentation.
``multinomial_logistic``
    The three-arm law :func:`draw_multinomial`.  :math:`W \sim N(0, I_3)`, the logits
    :math:`(0, 0.3 W_1 - 0.2 W_2, -0.2 W_1 + 0.3 W_2)`, and
    :math:`Y = 0.6\,\mathrm{step}[A] + W_1 - 0.5 W_2 + 0.2 W_3 + N(0, 1)` with steps
    ``(0, 1, 2.4)``.  In-sample, ``LogisticRegression(C=numpy.inf, max_iter=2000)``.  The
    package's three-arm law :func:`cleverly.datasets.make_multi_arm` is not used: its
    :math:`E[1/g^3]` is about :math:`e^{14}`, so the curve of :math:`\hat\nu^2` has a standard
    deviation of about 709 and no calibration cell could resolve it.
``binary_known``
    ``make_linear_ate`` with its own mechanism,
    ``cleverly.datasets.synthetic.linear_dgp().propensity``, declared through
    ``treatment_probabilities=``.  The plug-in then has no mechanism term.

**The truths.**  ``sigma^2 = 1`` on every law.  On ``make_linear_ate`` the closed forms of the
RM22 study hold: :math:`\nu^2 = 2 + 2e^{0.07}` for the ATE and :math:`4e^{0.07}` for the ATT
and the ATC, and the effect is 1.5.  On the three-arm law a two-dimensional Gauss-Hermite rule
with 80 nodes per axis gives :math:`p_a = E[g_a]`, :math:`E[1/g_a + 1/g_r]`,
:math:`(p_a + E[g_a^2/g_r]) / p_a^2` and :math:`(p_r + E[g_r^2/g_a]) / p_r^2`; the effects are
1.44, 1.44 and 0.6.

**The independent rebuild.**  Each replicate rebuilds the curve of :math:`\hat\nu^2` from the
captured coefficients, the folds and the data, and refuses the replicate unless it equals the
package's curve within :data:`REBUILD_TOLERANCE`.  It reads no package curve and imports no
package code of the stack.  Each control is the package's curve minus (inflated) or plus
(shrunken) a mechanism term that this module computed.

The X26 plan (rev 3.1, section 3.5 and section 8) declared this study, its margins, its
budget and its red-cell route before any run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_linear_ate
from cleverly.datasets.synthetic import linear_dgp
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import omitted_variable as omitted_variable_module
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 10_000
PRIMARY_N = 1_000
SEED = 20262601
RESAMPLING_SEED = 20262602

#: The strength of the X26 contract.  At (0.5, 0.3) the ATT control's predicted ratio, 1.089,
#: sits too close to the band edge 1.07 (plan section 3.4), so the study raises ``cf_d``.
CF_Y = 0.5
CF_D = 0.5
RHO = 1.0

#: The three quantities each parameter reports.
ROWS = ("nu2", "lower", "upper")

#: The parameters of each law: the study's token and the package's name, and the
#: representer's group, arm and the arm it is contrasted with.
BINARY_PARAMETERS: dict[str, tuple[str, str, float, float]] = {
    "ate": ("ate", "mean", 1.0, 0.0),
    "att": ("att", "att", 1.0, 0.0),
    "atc": ("atc", "atc", 1.0, 0.0),
}
MULTINOMIAL_PARAMETERS: dict[str, tuple[str, str, float, float]] = {
    "ate_2v0": ("ate[2.0 vs 0.0]", "mean", 2.0, 0.0),
    "att_2v0": ("att[2.0 vs 0.0]", "att", 2.0, 0.0),
    "atc_1v0": ("atc[1.0 vs 0.0]", "atc", 1.0, 0.0),
}
PARAMETERS: dict[str, dict[str, tuple[str, str, float, float]]] = {
    "binary_logistic": BINARY_PARAMETERS,
    "binary_logistic_crossfit": BINARY_PARAMETERS,
    "binary_logistic_ridge": BINARY_PARAMETERS,
    "multinomial_logistic": MULTINOMIAL_PARAMETERS,
    "binary_known": BINARY_PARAMETERS,
}

#: The scenarios whose mechanism is fitted, so the curve carries a mechanism term.
ESTIMATED = (
    "binary_logistic",
    "binary_logistic_crossfit",
    "binary_logistic_ridge",
    "multinomial_logistic",
)

#: The parameters with a control.  The ATE has none: at a correct model its mechanism term is
#: a reflection of the second-moment term, which keeps the variance (plan section 3.4).
CONTROLLED = ("att", "atc", "att_2v0", "atc_1v0")


def estimand_key(token: str, row: str) -> str:
    """The primary estimand of one parameter's row: ``plugin_att_nu2`` ... ."""
    return f"plugin_{token}_{row}"


def cell_label(scenario: str, token: str, row: str) -> str:
    """The calibration label of one row of one scenario: ``binary_logistic_att_nu2`` ... ."""
    return f"{scenario}_{token}_{row}"


SCENARIOS: dict[str, tuple[str, ...]] = {
    scenario: tuple(estimand_key(token, row) for token in parameters for row in ROWS)
    for scenario, parameters in PARAMETERS.items()
}


def control_kind(scenario: str) -> str:
    """The control of a scenario's ATT and ATC rows."""
    return "inflated_se_control" if scenario in ESTIMATED else "shrunken_se_control"


def _cells() -> tuple[str, ...]:
    cells: list[str] = []
    for scenario, parameters in PARAMETERS.items():
        for token in parameters:
            for row in ROWS:
                label = cell_label(scenario, token, row)
                cells.append(f"{label}__correctly_specified")
                if token in CONTROLLED:
                    cells.append(f"{label}__{control_kind(scenario)}")
    return tuple(cells)


PROPERTY_CELLS = {"interval_calibration": _cells()}

STUDY = StudyRecord(
    name="omitted-variable bound plug-in standard error",
    slug="omitted-variable-bound-plugin-se",
    artifacts=ROOT / "tests" / "canonical" / "omitted_variable_bound_plugin",
    document=(
        "docs/technical-reference/method-evidence/omitted-variable-bound-plugin-standard-error.md"
    ),
    anchor="omitted-variable-bound-plug-in-standard-error",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-omitted-variable-bound-plugin",
    reference=None,
    failure_bound={"cleverly-omitted-variable-bound-plugin": 0.01},
    modules=(
        "tests/studies/omitted_variable_bound_plugin.py",
        "tests/studies/omitted_variable_bound_plugin_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.omitted_variable_bound_plugin",
    properties_module="tests.studies.omitted_variable_bound_plugin_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

#: The declared outcome support of the cross-fitted scenario.  See the module docstring.
CROSSFIT_Q_BOUNDS = (-40.0, 45.0)

CONFIGURATION = {
    "laws": {
        "binary": "make_linear_ate(n=1000): W ~ N(0, I_4), g = expit(0.3 W1 - 0.2 W2 + 0.1 W3), "
        "Y = 2 + 1.5 A + W1 + 0.5 W2 - 0.8 W3 + 0.4 W4 + N(0, 1)",
        "multinomial": "W ~ N(0, I_3); logits (0, 0.3 W1 - 0.2 W2, -0.2 W1 + 0.3 W2); "
        "Y = 0.6 step[A] + W1 - 0.5 W2 + 0.2 W3 + N(0, 1), steps (0, 1, 2.4)",
    },
    "fits": {
        "binary_logistic": "in-sample TMLE, LinearRegression outcome, "
        "LogisticRegression(C=inf, max_iter=2000)",
        "binary_logistic_crossfit": "the same with cross_fit=True, n_folds=5, "
        "q_bounds=(-40.0, 45.0)",
        "binary_logistic_ridge": "the same as binary_logistic with "
        "LogisticRegression(max_iter=1000), so C = 1",
        "multinomial_logistic": "in-sample TMLE, LinearRegression outcome, "
        "LogisticRegression(C=inf, max_iter=2000), reference arm 0",
        "binary_known": "in-sample TMLE, LinearRegression outcome, the law's own mechanism "
        "linear_dgp().propensity declared through treatment_probabilities=",
    },
    "nu2_estimator": "plugin",
    "strength": {"cf_y": CF_Y, "cf_d": CF_D, "rho": RHO},
    "primary_rows": "nu2: estimate = nu2, std_error = sqrt(influence_variance(psi_nu2)); "
    "lower and upper: estimate = the bound, std_error = (lower - ci_lower) / z_0.95 and "
    "(ci_upper - upper) / z_0.95, each checked against the rebuilt curve; "
    "interval = estimate +/- 1.959964 std_error",
    "controls": {
        "inflated_se_control": "the estimated-mechanism scenarios' ATT and ATC rows, with the "
        "package curve minus the independently rebuilt mechanism term",
        "shrunken_se_control": "binary_known's ATT and ATC rows, with the package curve plus "
        "the mechanism term of a study-side LogisticRegression(C=inf) fit",
    },
    "truth": "sigma2 = 1. binary: nu2 = 2 + 2 exp(0.07) for ate, 4 exp(0.07) for att and atc, "
    "effect 1.5. multinomial: 80 x 80 Gauss-Hermite quadrature, effects 1.44, 1.44, 0.6",
}

#: The red-cell route, declared before any run (X26 plan section 3.5 and review F10).
RED_CELL_OWNER = "X26-plugin-finite-sample"
RED_CELL_POLICY = (
    "the study publishes under reporting; no replicate count, margin, law, learner or "
    "strength changes after a verdict is seen",
    "a red positive cell is diagnosed first: on 20 replicates of its scenario, the "
    "infinitesimal jackknife (unit witness W5) and the numerical sandwich "
    "(TestTheNumericalSandwich) are compared with the package curve. A mismatch is a defect, "
    "which is fixed and the study regenerated under a fresh declaration commit",
    "when they agree, the cell is rerun at n = 4000 with 5,000 replicates and published red "
    f"with owner {RED_CELL_OWNER} and the reading 'finite-sample, contracting' or "
    "'finite-sample, not contracting at n=4000', with a roadmap owner row",
    "a red positive cell reads as finite-sample only when its 99% SE-ratio interval contains "
    "1, its coverage is consistent with 0.95, and the diagnosis finds no defect; any other "
    "red positive cell is a defect suspect and never ends as an owner row unfixed",
    "a control that does not leave the band is reported as 'control underpowered', with its ratio",
)

CONFIGURATION["red_cell_policy"] = list(RED_CELL_POLICY)

#: The two-sided 95% multiplier of each row's interval, and the one-sided multiplier the bound
#: reports its limit with.
TWO_SIDED = float(stats.norm.ppf(0.975))
ONE_SIDED = float(stats.norm.ppf(0.95))

#: The variance of the linear index of the binary mechanism, ``0.3^2 + 0.2^2 + 0.1^2``.
INDEX_VARIANCE = 0.14

#: The three-arm law's arm steps and outcome coefficients.
MULTINOMIAL_STEPS = (0.0, 1.0, 2.4)
MULTINOMIAL_EFFECT_SCALE = 0.6

#: The nodes per axis of the quadrature behind the three-arm truths.
QUADRATURE_NODES = 80


def strength() -> float:
    """The multiplier on the bias scale, ``|rho| sqrt(cf_y cf_d / (1 - cf_d))``."""
    return abs(RHO) * math.sqrt(CF_Y * CF_D / (1.0 - CF_D))


def multinomial_logits(w: np.ndarray) -> np.ndarray:
    """``(n, 3)`` logits of the three-arm law, arm 0 first."""
    return np.column_stack(
        [np.zeros(w.shape[0]), 0.3 * w[:, 0] - 0.2 * w[:, 1], -0.2 * w[:, 0] + 0.3 * w[:, 1]]
    )


def multinomial_mechanism(w: np.ndarray) -> np.ndarray:
    """``(n, 3)`` treatment mechanism of the three-arm law."""
    eta = multinomial_logits(w)
    eta = eta - eta.max(axis=1, keepdims=True)
    expd = np.exp(eta)
    return np.asarray(expd / expd.sum(axis=1, keepdims=True), dtype=float)


def multinomial_second_moments(nodes: int = QUADRATURE_NODES) -> dict[str, float]:
    """The three-arm ``nu^2`` of each parameter, by Gauss-Hermite quadrature on ``(W1, W2)``.

    The mechanism reads ``W1`` and ``W2`` only, so a product rule on those two axes is exact
    up to the rule's error.
    """
    x, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    w1, w2 = np.meshgrid(x, x, indexing="ij")
    omega = np.outer(weights, weights).reshape(-1)
    g = multinomial_mechanism(np.column_stack([w1.reshape(-1), w2.reshape(-1)]))
    share = omega @ g
    return {
        "ate_2v0": float(omega @ (1.0 / g[:, 2] + 1.0 / g[:, 0])),
        "att_2v0": float((share[2] + omega @ (g[:, 2] ** 2 / g[:, 0])) / share[2] ** 2),
        "atc_1v0": float((share[0] + omega @ (g[:, 0] ** 2 / g[:, 1])) / share[0] ** 2),
    }


def _binary_second_moments() -> dict[str, float]:
    odds = math.exp(INDEX_VARIANCE / 2.0)
    return {"ate": 2.0 + 2.0 * odds, "att": 4.0 * odds, "atc": 4.0 * odds}


_MULTINOMIAL_NU2 = multinomial_second_moments()


def truths(scenario: str) -> dict[str, float]:
    """The population ``nu^2`` and both bounds of each parameter of ``scenario``."""
    if scenario not in PARAMETERS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    if scenario == "multinomial_logistic":
        nu2 = _MULTINOMIAL_NU2
        effects = {
            "ate_2v0": MULTINOMIAL_EFFECT_SCALE * MULTINOMIAL_STEPS[2],
            "att_2v0": MULTINOMIAL_EFFECT_SCALE * MULTINOMIAL_STEPS[2],
            "atc_1v0": MULTINOMIAL_EFFECT_SCALE * MULTINOMIAL_STEPS[1],
        }
    else:
        nu2 = _binary_second_moments()
        effects = dict.fromkeys(BINARY_PARAMETERS, 1.5)
    sigma2 = 1.0
    values: dict[str, float] = {}
    for token in PARAMETERS[scenario]:
        bias = strength() * math.sqrt(sigma2 * nu2[token])
        values[estimand_key(token, "nu2")] = nu2[token]
        values[estimand_key(token, "lower")] = effects[token] - bias
        values[estimand_key(token, "upper")] = effects[token] + bias
    return values


def draw_multinomial(n: int, seed: int) -> pd.DataFrame:
    """One sample of the three-arm law, columns ``Y, A, W1, W2, W3``."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 3))
    g = multinomial_mechanism(w)
    arm = (rng.random(n)[:, None] > np.cumsum(g, axis=1)).sum(axis=1)
    arm = np.minimum(arm, 2)
    steps = np.asarray(MULTINOMIAL_STEPS)[arm]
    outcome = (
        MULTINOMIAL_EFFECT_SCALE * steps
        + w[:, 0]
        - 0.5 * w[:, 1]
        + 0.2 * w[:, 2]
        + rng.standard_normal(n)
    )
    frame = pd.DataFrame(w, columns=["W1", "W2", "W3"])
    frame.insert(0, "A", arm.astype(float))
    frame.insert(0, "Y", outcome)
    return frame


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit."""
    truth = truths(scenario)
    if scenario == "multinomial_logistic":
        return draw_multinomial(n, seed), truth
    frame, _ = make_linear_ate(n=n, seed=seed)
    return frame, truth


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def known_mechanism(frame: pd.DataFrame) -> np.ndarray:
    """``(n,)`` ``P(A = 1 | W)`` of ``make_linear_ate``, the law's own mechanism."""
    covariates = frame.loc[:, ["W1", "W2", "W3", "W4"]].to_numpy(dtype=float)
    return np.asarray(linear_dgp().propensity(covariates), dtype=float)


def _treatment_learner(scenario: str) -> LogisticRegression:
    if scenario == "binary_logistic_ridge":
        return LogisticRegression(max_iter=1000)
    return LogisticRegression(C=np.inf, max_iter=2000)


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The declared TMLE of ``scenario``.

    No warning is suppressed.  The declared fits raise none on these laws, so a warning that a
    later package raises reaches the regeneration log instead of vanishing.
    """
    estimands: tuple[Literal["ate", "att", "atc"], ...] = ("ate", "att", "atc")
    if scenario == "binary_known":
        return (
            TMLE(
                outcome_learner=LinearRegression(),
                cross_fit=False,
                simultaneous=False,
                estimands=estimands,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                treatment_probabilities=known_mechanism(frame),
            )
            .single()
        )
    crossfit = scenario == "binary_logistic_crossfit"
    options: dict[str, Any] = {"cross_fit": crossfit}
    if crossfit:
        options.update(n_folds=5, q_bounds=CROSSFIT_Q_BOUNDS)
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=_treatment_learner(scenario),
            simultaneous=False,
            estimands=estimands,
            random_state=0,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


# ---------------------------------------------------------------- the independent rebuild

#: How closely the rebuilt curve, mechanism and representer must reproduce the package's
#: own, relative to the largest absolute value of the package's array (and at least 1).
REBUILD_TOLERANCE = 1e-8


def _close(rebuilt: np.ndarray, reported: np.ndarray) -> bool:
    scale = max(1.0, float(np.max(np.abs(reported))))
    return bool(np.max(np.abs(rebuilt - reported)) <= REBUILD_TOLERANCE * scale)


def _softmax_reference(design: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """``(n, K)`` probabilities with the reference arm first, from reference-form ``theta``."""
    eta = np.column_stack([np.zeros(design.shape[0]), design @ theta.T])
    eta = eta - eta.max(axis=1, keepdims=True)
    expd = np.exp(eta)
    return np.asarray(expd / expd.sum(axis=1, keepdims=True), dtype=float)


@dataclass(frozen=True)
class Logistic:
    """One fold's fitted logistic mechanism in reference form.

    Parameters
    ----------
    design : ndarray
        ``(n, d)`` intercept and covariates of every row.
    theta : ndarray
        ``(K - 1, d)`` coefficients against the lowest arm.
    penalty : ndarray
        ``(p, p)`` matrix ``P / C`` with ``p = (K - 1) d``.
    train : ndarray
        The rows the fold's model was fitted on.
    test : ndarray
        The rows the fold's model predicts.
    """

    design: np.ndarray
    theta: np.ndarray
    penalty: np.ndarray
    train: np.ndarray
    test: np.ndarray


def _penalty(c: float, arms: int, d: int) -> np.ndarray:
    """scikit-learn's L2 penalty in reference form, the intercept unpenalized."""
    p = (arms - 1) * d
    if math.isinf(c):
        return np.zeros((p, p))
    slopes = np.eye(d)
    slopes[0, 0] = 0.0
    between = np.ones((1, 1)) if arms == 2 else np.eye(arms - 1) - 1.0 / arms
    return np.asarray(np.kron(between, slopes) / c, dtype=float)


def _reference_theta(coef: np.ndarray, intercept: np.ndarray, arms: int) -> np.ndarray:
    full = np.column_stack([np.asarray(intercept).reshape(-1), np.asarray(coef)])
    if arms == 2:
        return np.asarray(full[:1], dtype=float)
    return np.asarray(full[1:] - full[0], dtype=float)


def _fold_rows(folds: Any, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds.n_folds == 1:
        everything = np.arange(n)
        return [(everything, everything)]
    return [(np.asarray(train), np.asarray(test)) for train, test in folds]


def captured_logistics(result: Any) -> list[Logistic]:
    """Each fold's mechanism, from the coefficients the fit captured, in fold order."""
    nuisance = result.repeats[0].nuisance
    arms = len(nuisance.arms)
    covariates = np.asarray(result.data.covariates, dtype=float)
    design = np.column_stack([np.ones(covariates.shape[0]), covariates])
    rows = _fold_rows(nuisance.folds, result.n)
    fits = nuisance.treatment_fits
    if fits is None or len(fits) != len(rows):
        raise RuntimeError("the fit captured no logistic mechanism for each fold")
    out = []
    for fit_record, (train, test) in zip(fits, rows, strict=True):
        if fit_record.steps or not fit_record.fit_intercept:
            raise RuntimeError("the declared mechanism is a bare LogisticRegression")
        theta = _reference_theta(fit_record.coef, fit_record.intercept, arms)
        out.append(
            Logistic(
                design=design,
                theta=theta,
                penalty=_penalty(float(fit_record.C), arms, design.shape[1]),
                train=train,
                test=test,
            )
        )
    return out


def study_logistic(result: Any) -> list[Logistic]:
    """A study-side in-sample ``LogisticRegression(C=inf)`` mechanism, for the known fits."""
    covariates = np.asarray(result.data.covariates, dtype=float)
    treatment = np.asarray(result.data.treatment, dtype=float)
    model = LogisticRegression(C=np.inf, max_iter=2000).fit(covariates, treatment)
    arms = len(result.repeats[0].nuisance.arms)
    everything = np.arange(covariates.shape[0])
    return [
        Logistic(
            design=np.column_stack([np.ones(covariates.shape[0]), covariates]),
            theta=_reference_theta(model.coef_, model.intercept_, arms),
            penalty=_penalty(math.inf, arms, covariates.shape[1] + 1),
            train=everything,
            test=everything,
        )
    ]


def _truncate(g: np.ndarray, bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The truncated mechanism and the cells it moved, by the package's documented rule.

    Two arms clip ``g_1`` and take ``g_0 = 1 - g_1``, marking both columns of a moved row.
    More arms clip column by column without renormalizing.
    """
    lower, upper = bounds
    if g.shape[1] == 2:
        one = np.clip(g[:, 1], lower, upper)
        moved = one != g[:, 1]
        return np.column_stack([1.0 - one, one]), np.column_stack([moved, moved])
    clipped = np.clip(g, lower, upper)
    return clipped, clipped != g


def representer(
    group: str, arm: int, versus: int, treatment: np.ndarray, g: np.ndarray, share: float | None
) -> tuple[np.ndarray, np.ndarray]:
    r"""``alpha`` at each row and its derivative in each column of ``g``.

    Columns are arm indices.  ``mean`` is the contrast :math:`1\{A=a\}/g_a - 1\{A=r\}/g_r`;
    ``att`` is :math:`(1\{A=a\} - 1\{A=r\} g_a / g_r) / p_a`; ``atc`` is
    :math:`(1\{A=a\} g_r / g_a - 1\{A=r\}) / p_r`.
    """
    on_a = (treatment == arm).astype(float)
    on_r = (treatment == versus).astype(float)
    derivative = np.zeros_like(g)
    if group == "mean":
        alpha = on_a / g[:, arm] - on_r / g[:, versus]
        derivative[:, arm] = -on_a / g[:, arm] ** 2
        derivative[:, versus] = on_r / g[:, versus] ** 2
        return alpha, derivative
    assert share is not None
    if group == "att":
        alpha = (on_a - on_r * g[:, arm] / g[:, versus]) / share
        derivative[:, arm] = -on_r / (g[:, versus] * share)
        derivative[:, versus] = on_r * g[:, arm] / (g[:, versus] ** 2 * share)
        return alpha, derivative
    alpha = (on_a * g[:, versus] / g[:, arm] - on_r) / share
    derivative[:, versus] = on_a / (g[:, arm] * share)
    derivative[:, arm] = -on_a * g[:, versus] / (g[:, arm] ** 2 * share)
    return alpha, derivative


def mechanism_term(
    folds: list[Logistic],
    treatment_index: np.ndarray,
    weights: np.ndarray,
    alpha: np.ndarray,
    derivative: np.ndarray,
) -> np.ndarray:
    r"""The stacked mechanism term :math:`\psi^\gamma` of :math:`E_n[w \alpha^2]`.

    For fold :math:`k`, :math:`J_k = \sum_{\mathrm{train}} w [\mathrm{diag}(g_{-r}) -
    g_{-r} g_{-r}^\top] \otimes x x^\top + P/C`,
    :math:`D_k = n^{-1} \sum_{\mathrm{test}} 2 w \alpha\, \partial_\theta \alpha`, and the row
    function is :math:`u_{k,i} = w_i x_i \otimes (e_{A_i} - g_k)_{-r} 1\{i \in
    \mathrm{train}_k\} - (w_i/n) P\theta_k/C`.  The term is
    :math:`\psi^\gamma_i = n \sum_k D_k^\top J_k^{-1} u_{k,i}`.  ``derivative`` is already zero
    on every truncated cell.
    """
    n = treatment_index.shape[0]
    term = np.zeros(n)
    for fold in folds:
        g = _softmax_reference(fold.design, fold.theta)
        arms = g.shape[1]
        d = fold.design.shape[1]
        reduced = g[:, 1:]
        information = np.zeros(((arms - 1) * d, (arms - 1) * d))
        train = fold.train
        x_train = fold.design[train]
        for m in range(arms - 1):
            for q in range(arms - 1):
                factor = reduced[train, m] * (float(m == q) - reduced[train, q])
                block = (x_train * (weights[train] * factor)[:, None]).T @ x_train
                information[m * d : (m + 1) * d, q * d : (q + 1) * d] = block
        information = information + fold.penalty
        # d alpha / d theta_m = x g_m (b_m - sum_c b_c g_c), reference column dropped.
        inner = (derivative * g).sum(axis=1)
        chain = g[:, 1:] * (derivative[:, 1:] - inner[:, None])
        test = fold.test
        coefficient = (2.0 * weights[test] * alpha[test])[:, None] * chain[test]
        gradient = np.einsum("nm,nd->md", coefficient, fold.design[test]).reshape(-1) / n
        direction = np.linalg.solve(information, gradient)
        indicator = np.zeros_like(g)
        indicator[np.arange(n), treatment_index] = 1.0
        residual = (indicator - g)[:, 1:]
        score = np.einsum("nm,nd->nmd", residual, fold.design).reshape(n, -1)
        term[train] += n * weights[train] * (score[train] @ direction)
        term -= weights * float(direction @ (fold.penalty @ fold.theta.reshape(-1)))
    return term


@dataclass(frozen=True)
class Rebuilt:
    """One parameter's curve of ``nu^2``, rebuilt without the package's curve.

    Parameters
    ----------
    nu2 : float
        The plug-in estimate.
    second_moment : ndarray
        ``w (alpha^2 - nu^2)``.
    share_term : ndarray
        The influence of the conditioning share, zero for the ATE.
    mechanism_term : ndarray
        The stacked mechanism term; zero for a declared known mechanism.
    """

    nu2: float
    second_moment: np.ndarray
    share_term: np.ndarray
    mechanism_term: np.ndarray

    @property
    def curve(self) -> np.ndarray:
        return self.second_moment + self.share_term + self.mechanism_term


def rebuild(result: Any, scenario: str, token: str, elements: Any) -> Rebuilt:
    """Rebuild the plug-in curve of one parameter and check it against the package's.

    The mechanism comes from the captured coefficients (an estimated mechanism) or from the
    law (a known one).  The representer, ``nu^2``, the share term and the mechanism term are
    this module's.  Any departure beyond :data:`REBUILD_TOLERANCE` raises ``RuntimeError``
    from this module, which stops the run: a departure is a defect, not a property of a sample.
    """
    _, group, arm_code, versus_code = PARAMETERS[scenario][token]
    nuisance = result.repeats[0].nuisance
    arms = tuple(float(a) for a in nuisance.arms)
    arm, versus = arms.index(arm_code), arms.index(versus_code)
    data = result.data
    n = result.n
    weights = np.asarray(data.weights, dtype=float)
    treatment_index = np.searchsorted(np.asarray(arms), np.asarray(data.treatment, dtype=float))
    stored = np.asarray(nuisance.propensity.values, dtype=float)
    folds: list[Logistic] = []
    if scenario == "binary_known":
        one = known_mechanism(_frame_of(result))
        g = np.column_stack([1.0 - one, one])
    else:
        folds = captured_logistics(result)
        g = np.zeros((n, len(arms)))
        for fold in folds:
            g[fold.test] = _softmax_reference(fold.design, fold.theta)[fold.test]
    if not _close(g, stored):
        raise RuntimeError(f"the rebuilt mechanism of {scenario!r} is not the package's")
    bounds = (
        result.config.g_bounds_conditional if group in ("att", "atc") else result.config.g_bounds
    )
    bounded, moved = _truncate(g, (float(bounds[0]), float(bounds[1])))
    conditioning = {"att": arm, "atc": versus}.get(group)
    share = None
    indicator = np.zeros(n)
    if conditioning is not None:
        indicator = (treatment_index == conditioning).astype(float)
        share = float(np.average(indicator, weights=weights))
    alpha, derivative = representer(group, arm, versus, treatment_index, bounded, share)
    if not _close(alpha, np.asarray(elements.riesz_representer, dtype=float)):
        raise RuntimeError(f"the rebuilt representer of {token!r} is not the package's")
    nu2 = float(np.average(alpha**2, weights=weights))
    if abs(nu2 - elements.nu2) > REBUILD_TOLERANCE * max(1.0, abs(elements.nu2)):
        raise RuntimeError(f"the rebuilt nu^2 of {token!r} is not the package's")
    share_term = np.zeros(n)
    if share is not None:
        share_term = -2.0 * nu2 * weights * (indicator - share) / share
    term = (
        mechanism_term(folds, treatment_index, weights, alpha, np.where(moved, 0.0, derivative))
        if folds
        else np.zeros(n)
    )
    rebuilt = Rebuilt(
        nu2=nu2,
        second_moment=weights * (alpha**2 - nu2),
        share_term=share_term,
        mechanism_term=term,
    )
    assert elements.psi_nu2 is not None
    if not _close(rebuilt.curve, np.asarray(elements.psi_nu2, dtype=float)):
        raise RuntimeError(f"the rebuilt curve of nu^2 of {token!r} is not the package's")
    return rebuilt


def _frame_of(result: Any) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(result.data.covariates, dtype=float), columns=list(result.data.covariate_names)
    )


def study_mechanism_term(result: Any, scenario: str, token: str) -> np.ndarray:
    r"""The mechanism term a study-side logistic fit adds to a known-mechanism curve.

    The representer and its derivative are the declared mechanism's; the score, the
    information and :math:`\partial g / \partial\theta` are those of an in-sample
    ``LogisticRegression(C=inf)`` on the same covariates.  This is the term the package would
    add if it treated the declared mechanism as fitted.
    """
    _, group, arm_code, versus_code = PARAMETERS[scenario][token]
    arms = tuple(float(a) for a in result.repeats[0].nuisance.arms)
    arm, versus = arms.index(arm_code), arms.index(versus_code)
    data = result.data
    weights = np.asarray(data.weights, dtype=float)
    treatment_index = np.searchsorted(np.asarray(arms), np.asarray(data.treatment, dtype=float))
    one = known_mechanism(_frame_of(result))
    g0 = np.column_stack([1.0 - one, one])
    conditioning = {"att": arm, "atc": versus}.get(group)
    share = None
    if conditioning is not None:
        share = float(np.average(treatment_index == conditioning, weights=weights))
    alpha, derivative = representer(group, arm, versus, treatment_index, g0, share)
    return mechanism_term(study_logistic(result), treatment_index, weights, alpha, derivative)


# ---------------------------------------------------------------- the rows


def _bias_curve(elements: Any, psi_nu2: np.ndarray) -> np.ndarray:
    """Theorem 4's curve of the maximal bias, from a curve of ``nu^2``."""
    return np.asarray(
        (elements.sigma2 * psi_nu2 + elements.nu2 * np.asarray(elements.psi_sigma2))
        / (2.0 * elements.max_bias),
        dtype=float,
    )


def _standard_errors(
    result: Any, name: str, elements: Any, psi_nu2: np.ndarray, strength_value: float
) -> dict[str, float]:
    """The standard error of ``nu^2`` and of each bound end, from one curve of ``nu^2``."""
    cluster = result.data.cluster
    curve = np.asarray(result[name].influence_curve, dtype=float)
    bias = _bias_curve(elements, psi_nu2)
    return {
        "nu2": float(np.sqrt(influence_variance(psi_nu2, cluster))),
        "lower": float(np.sqrt(influence_variance(curve - strength_value * bias, cluster))),
        "upper": float(np.sqrt(influence_variance(curve + strength_value * bias, cluster))),
    }


def parameter_rows(
    result: Any, scenario: str, token: str
) -> dict[str, tuple[float, float, float | None]]:
    """Each row's estimate, reported standard error and control standard error.

    The reported standard error of a bound end is read back off its limit, and it must equal
    the standard error of the rebuilt curve.  The control is ``None`` for a parameter without
    one.
    """
    ov = omitted_variable_module
    name = PARAMETERS[scenario][token][0]
    bounds = ov.omitted_variable_bounds(
        result, name, cf_y=CF_Y, cf_d=CF_D, rho=RHO, nu2_estimator="plugin"
    )
    elements = ov.sensitivity_elements(result, name, nu2_estimator="plugin")
    expected = "known" if scenario == "binary_known" else "logistic"
    if elements.limit_refusal is not None:
        # A sample on which the package withholds the limits (for example the
        # penalty_bias gate) is a failed replicate, counted against FAILURE_BOUND.
        raise DataError(f"the plug-in bound of {name!r} refused: {elements.limit_refusal.key}")
    if elements.plugin_mechanism != expected or elements.psi_nu2 is None:
        raise RuntimeError(f"the plug-in bound of {name!r} did not stack the {expected} mechanism")
    rebuilt = rebuild(result, scenario, token, elements)
    strength_value = bounds.confounding_strength
    reported = {
        "nu2": float(np.sqrt(influence_variance(elements.psi_nu2, result.data.cluster))),
        "lower": (bounds.lower - bounds.ci_lower) / ONE_SIDED,
        "upper": (bounds.ci_upper - bounds.upper) / ONE_SIDED,
    }
    own = _standard_errors(result, name, elements, rebuilt.curve, strength_value)
    for row in ("lower", "upper"):
        if abs(own[row] - reported[row]) > REBUILD_TOLERANCE * max(1.0, own[row]):
            raise RuntimeError(f"the rebuilt {row} standard error of {name!r} is not reported")
    estimates = {"nu2": float(elements.nu2), "lower": bounds.lower, "upper": bounds.upper}
    control: dict[str, float] | None = None
    if token in CONTROLLED:
        psi = np.asarray(elements.psi_nu2, dtype=float)
        if scenario in ESTIMATED:
            changed = psi - rebuilt.mechanism_term
        else:
            changed = psi + study_mechanism_term(result, scenario, token)
        control = _standard_errors(result, name, elements, changed, strength_value)
    return {
        row: (
            float(estimates[row]),
            float(reported[row]),
            None if control is None else float(control[row]),
        )
        for row in ROWS
    }


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema, one row per quantity."""
    result = fit(frame, scenario)
    rows: list[dict[str, Any]] = []
    for token in PARAMETERS[scenario]:
        values = parameter_rows(result, scenario, token)
        for row in ROWS:
            estimate, std_error, _ = values[row]
            key = estimand_key(token, row)
            reference = float(truth[key])
            low = estimate - TWO_SIDED * std_error
            high = estimate + TWO_SIDED * std_error
            rows.append(
                {
                    "implementation": STUDY.implementation,
                    "scenario": scenario,
                    "replicate": replicate,
                    "n": result.n,
                    "estimand": key,
                    "truth": reference,
                    "estimate": estimate,
                    "inference_estimate": estimate,
                    "std_error": std_error,
                    "ci_lower": float(low),
                    "ci_upper": float(high),
                    "inference_scale": "identity",
                    "covered": int(low <= reference <= high),
                    "initial_estimate": math.nan,
                }
            )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])

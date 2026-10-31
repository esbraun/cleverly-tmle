"""The law, the fits, the truth and the controls of the ``fold-evaluated-strata`` study (X28).

The study measures ``TMLE(cv_evaluation=True)`` with ``strata=`` (X11 PR E1, roadmap X28) on the
arm targets ``estimands=("ate", "ey")``. Each stratum estimate is the ``1/V`` average of the fold
plug-ins over the rows of each validation fold in the stratum, each fold's stratum curve divides
by the fold's own stratum share, and the marginal estimate is the share mixture of the stratum
estimates with its delta-method curve.

The law is the RM30 family of :mod:`tests.studies._learned_rule_law` with a steeper blip and a
proportion outcome: ``W1 ~ U(-1, 1)``, ``W2 ~ Bernoulli(0.5)``, ``logit g0(1 | W) = 0.3 W1 -
0.2 W2``, and ``Y | A, W ~ Beta(phi m, phi (1 - m))`` with mean
``m = Qbar0(A, W)``, ``logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + a (0.1 + 1.5 W1)`` and the
concentration ``phi = 12`` of :mod:`tests.studies.bounded_cv_laws`. The outcome regression is
the RM30 form; only the outcome's law around it changes, from Bernoulli to beta, so that its
conditional variance ``m (1 - m) / (1 + phi)`` leaves the share term of the marginal curve
about a quarter of the marginal variance (deviation D1 of the design record: with a Bernoulli
outcome the share term of the ``ate`` stays under 8% and a stronger blip puts stratum means
near 0 and 1, where the pilot's Wald intervals under-cover). The baseline stratum cuts ``W1``
at ``-0.6`` and ``0.1``, so the shares are 0.2, 0.35 and 0.45, unequal, and the stratum
effects are heterogeneous in sign. The ``weighted`` setting adds the known observation weight
``w = 1 + 1.5 W2``, which varies inside every stratum, so the fold's stratum weight mass
``W_{v,s}`` and its row count ``n_{v,s}`` are not proportional. The target of a weighted fit is
the parameter of the weight-tilted law.

The target is fixed and known, so the truth needs no refit. Each stratum mean
``E_w[Qbar0(a, W) | S = s]`` integrates ``Qbar0`` over ``W1`` on the stratum's interval by the
trapezoid rule on :data:`POINTS` points for each value of ``W2``, weighted by ``w(W2)``. The
marginal truth is the share mixture, and ``S`` is independent of ``W2``, so the weight leaves the
shares at the interval lengths. The same quadrature gives each parameter's efficient influence
curve SD (the noise-control scale) and the asymptotic SE ratio of the share-term control.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly.estimators import TMLE
from tests.studies import _learned_rule_law as base
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The outer folds of every fit.
N_FOLDS = 5

#: The edges of the baseline stratum on ``W1``: strata 0, 1 and 2 are ``[-1, -0.6)``,
#: ``[-0.6, 0.1)`` and ``[0.1, 1]``.
EDGES = (-1.0, -0.6, 0.1, 1.0)

#: The stratum codes, in the order the fit labels them (``S=0``, ``S=1``, ``S=2``).
STRATA = (0, 1, 2)

#: ``P(S = s)``: the interval lengths over the support length 2.
SHARES = (0.2, 0.35, 0.45)

#: The blip ``b(W1) = INTERCEPT + SLOPE * W1`` on the logit scale.
INTERCEPT = 0.1
SLOPE = 1.5

#: The beta concentration ``phi`` of the outcome, and the outcome support the fit declares.
#: Both equal :mod:`tests.studies.bounded_cv_laws`'s ``CONCENTRATION`` and ``Q_BOUNDS``, which
#: sample the same kind of proportion. The smallest beta shape ``phi min(m, 1 - m)`` of this law
#: is 1.09, above that module's floor of 0.70, so no draw rounds to 0 or 1.
PHI = 12.0
OUTCOME_BOUNDS = (0.0, 1.0)

#: The quadrature points of ``W1`` on each stratum's interval.
POINTS = 4001

#: The settings: rows only, and the known weight ``1 + 1.5 W2``.
SETTINGS = ("unweighted", "weighted")

#: The weight column of every sample. The ``unweighted`` fit does not read it.
WEIGHT = "w"

#: The adjustment columns, in the order the package builds the designs: the outcome learner
#: sees ``(A, W1, W2, S)`` and the treatment learner ``(W1, W2, S)``. ``S`` must be an
#: adjustment column to be a baseline stratum.
COVARIATES = ("W1", "W2", "S")

#: The targets the fit reports, by the short name a cell label carries.
TARGETS = {"ate": "ate", "ey0": "ey[0]", "ey1": "ey[1]"}

#: The stratum part of a cell label: one stratum code, or ``all`` for the marginal estimate.
STRATUM_KEYS: dict[str, int | None] = {"s0": 0, "s1": 1, "s2": 2, "all": None}

#: The parameter of every ``target_stratum`` label: 3 stratum parameters and the marginal one
#: for each of the three targets.
PARAMETER_LABELS = tuple(f"{target}_{stratum}" for target in TARGETS for stratum in STRATUM_KEYS)

#: The two-sided 95% normal multiplier of every interval the study writes itself.
CRITICAL = base.CRITICAL

#: The warning classes counted in ``solver_warnings``.
SOLVER_WARNINGS = base.SOLVER_WARNINGS


def parameter(label: str) -> str:
    """The reported parameter name of a ``target_stratum`` label, such as ``ate[S=0]``."""
    target, stratum = label.split("_")
    code = STRATUM_KEYS[stratum]
    name = TARGETS[target]
    return name if code is None else f"{name}[S={code}]"


#: Every reported parameter of a fit, in label order.
PARAMETERS = tuple(parameter(label) for label in PARAMETER_LABELS)


# --------------------------------------------------------------------------- the law


def blip(w1: Any) -> np.ndarray:
    """The blip ``0.1 + 1.5 W1`` on the logit scale."""
    return np.asarray(INTERCEPT + SLOPE * np.asarray(w1, dtype=float), dtype=float)


def qbar0(a: Any, w1: Any, w2: Any) -> np.ndarray:
    """``E[Y | A = a, W]``, from ``logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + a b(W1)``."""
    w1 = np.asarray(w1, dtype=float)
    return np.asarray(
        expit(0.2 + 0.5 * w1 - 0.3 * np.asarray(w2, dtype=float) + np.asarray(a) * blip(w1)),
        dtype=float,
    )


def weight(w2: Any) -> np.ndarray:
    """The known observation weight ``1 + 1.5 W2`` of the ``weighted`` setting."""
    return np.asarray(1.0 + 1.5 * np.asarray(w2, dtype=float), dtype=float)


def stratum_of(w1: Any) -> np.ndarray:
    """The baseline stratum code of each ``W1``."""
    return np.digitize(np.asarray(w1, dtype=float), EDGES[1:-1]).astype(np.int64)


def draw(n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows with columns ``W1, W2, S, A, Y, w``.

    The draws are the RM30 draws in the RM30 order (``W1``, ``W2``, ``A``, then ``Y``), with
    this law's outcome regression and a beta outcome around it; ``S`` and ``w`` are functions
    of ``W``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, base.g0(w1, w2)).astype(float)
    mean = qbar0(a, w1, w2)
    y = rng.beta(PHI * mean, PHI * (1.0 - mean))
    return pd.DataFrame(
        {"W1": w1, "W2": w2, "S": stratum_of(w1), "A": a, "Y": y, WEIGHT: weight(w2)}
    )


# ------------------------------------------------------------------------ the truth


def _target_parts(target: str, w1: np.ndarray, w2: float) -> tuple[np.ndarray, np.ndarray]:
    """The regression functional and the conditional residual variance of the curve.

    With ``v(a) = Qbar0(a)(1 - Qbar0(a)) / (1 + phi)``, the beta outcome's conditional variance:
    for ``ate`` they are ``Qbar0(1) - Qbar0(0)`` and ``sum_a v(a) / g0(a)``; for ``ey[a]``
    they are ``Qbar0(a)`` and ``v(a) / g0(a)``.
    """
    q1, q0 = qbar0(1.0, w1, w2), qbar0(0.0, w1, w2)
    g1 = base.g0(w1, w2)
    r1 = q1 * (1.0 - q1) / (1.0 + PHI) / g1
    r0 = q0 * (1.0 - q0) / (1.0 + PHI) / (1.0 - g1)
    if target == "ate":
        return q1 - q0, r1 + r0
    if target == "ey0":
        return q0, r0
    if target == "ey1":
        return q1, r1
    raise KeyError(target)


def _weights(setting: str) -> dict[float, float]:
    """``w(W2)`` at each ``W2``; 1 for the rows-only setting."""
    if setting not in SETTINGS:
        raise KeyError(setting)
    return {w2: float(weight(w2)) if setting == "weighted" else 1.0 for w2 in base.W2_LEVELS}


def _grid(code: int) -> np.ndarray:
    return np.linspace(EDGES[code], EDGES[code + 1], POINTS)


@dataclass(frozen=True)
class Moments:
    """One target's quadrature moments in one setting.

    Parameters
    ----------
    stratum_values : tuple of float
        ``psi_s``, the weight-tilted stratum means.
    value : float
        ``sum_s P(S = s) psi_s``.
    stratum_variances : tuple of float
        ``Var D_s``, the variance of each stratum parameter's efficient influence curve.
    variance : float
        ``Var D``, the marginal parameter's.
    within_variance : float
        The variance of the marginal curve without its share term ``(w / E w)(psi_S - psi)``.
    """

    stratum_values: tuple[float, ...]
    value: float
    stratum_variances: tuple[float, ...]
    variance: float
    within_variance: float


@cache
def moments(setting: str, target: str) -> Moments:
    """The truth and the curve variances of ``target`` in ``setting``, by quadrature.

    With ``t = w / E w``, the curve of the stratum parameter is
    ``t I(S = s) (R + f(W) - psi_s) / P(S = s)``, with ``R`` the residual term of the curve,
    whose conditional variance :func:`_target_parts` gives. The marginal curve is
    ``t (R + f(W) - psi)``, and the share-term control drops ``t (psi_S - psi)`` from it.
    """
    weights = _weights(setting)
    mean_weight = float(np.mean(list(weights.values())))
    values: list[float] = []
    for code in STRATA:
        grid = _grid(code)
        length = EDGES[code + 1] - EDGES[code]
        numerator = sum(
            0.5 * w * np.trapezoid(_target_parts(target, grid, w2)[0], grid) / length
            for w2, w in weights.items()
        )
        values.append(float(numerator / (0.5 * sum(weights.values()))))
    value = float(np.dot(SHARES, values))
    stratum_variances: list[float] = []
    variance = 0.0
    within = 0.0
    for code, share in zip(STRATA, SHARES, strict=True):
        grid = _grid(code)
        inside = 0.0
        for w2, w in weights.items():
            tilt = w / mean_weight
            f, residual = _target_parts(target, grid, w2)
            # The density of W1 is 1/2 and each W2 has probability 1/2.
            mass = 0.5 * tilt**2 / 2.0
            inside += mass * float(np.trapezoid(residual + (f - values[code]) ** 2, grid))
            variance += mass * float(np.trapezoid(residual + (f - value) ** 2, grid))
        within += inside
        stratum_variances.append(inside / share**2)
    return Moments(tuple(values), value, tuple(stratum_variances), variance, within)


def truths(setting: str) -> dict[str, float]:
    """The truth of every reported parameter in ``setting``."""
    out: dict[str, float] = {}
    for label in PARAMETER_LABELS:
        target, stratum = label.split("_")
        found = moments(setting, target)
        code = STRATUM_KEYS[stratum]
        out[parameter(label)] = found.value if code is None else found.stratum_values[code]
    return out


def efficiency_bound(setting: str, label: str) -> float:
    """The SD of ``sqrt(n)`` times the estimate of ``label``: its efficient curve's SD."""
    target, stratum = label.split("_")
    found = moments(setting, target)
    code = STRATUM_KEYS[stratum]
    variance = found.variance if code is None else found.stratum_variances[code]
    return float(np.sqrt(variance))


def share_term_ratio(setting: str, target: str) -> float:
    """The asymptotic SE ratio of the share-term control: ``sqrt(within / Var D)``."""
    found = moments(setting, target)
    return float(np.sqrt(found.within_variance / found.variance))


# ------------------------------------------------------------------------- the fits


def outcome_learner() -> Pipeline:
    """The declared outcome learner: every pairwise product of ``(A, W1, W2, S)``, quasibinomial.

    It contains the law's linear predictor, so it is correctly specified. The quasibinomial
    score solver fits a proportion, which scikit-learn's logistic classifier refuses.
    """
    return Pipeline(
        [
            ("features", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("model", QuasiBinomialGLM(tol=1e-10)),
        ]
    )


def drop_w1_learner() -> Pipeline:
    """The targeting-control outcome learner: RM30's ``drop_w1_learner()`` on a proportion.

    It keeps columns 0 and 2 of the design ``(A, W1, W2, S)``, that is ``A`` and ``W2``, so it
    drops ``S`` as well as ``W1``, and fits their pairwise products by the same quasibinomial
    solver. Its plug-in is nearly the same in every stratum.
    """
    return Pipeline(
        [
            ("keep", ColumnTransformer([("keep", "passthrough", [0, 2])])),
            ("features", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("model", QuasiBinomialGLM(tol=1e-10)),
        ]
    )


def estimator(fold_seed: int, *, learner: Any = None, n_folds: int = N_FOLDS) -> TMLE:
    """The declared estimator, with the fold seed as its ``random_state``."""
    return TMLE(
        outcome_learner=outcome_learner() if learner is None else learner,
        treatment_learner=base.treatment_learner(),
        cross_fit=True,
        n_folds=n_folds,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        q_bounds=OUTCOME_BOUNDS,
        estimands=("ate", "ey"),
        simultaneous=False,
        random_state=fold_seed,
    )


def fit(
    frame: pd.DataFrame,
    setting: str,
    fold_seed: int,
    *,
    learner: Any = None,
    n_folds: int = N_FOLDS,
) -> Any:
    """Fit the declared estimator to one sample of ``setting`` and return the single result."""
    if setting not in SETTINGS:
        raise KeyError(setting)
    columns: dict[str, Any] = {"weights": WEIGHT} if setting == "weighted" else {}
    return (
        estimator(fold_seed, learner=learner, n_folds=n_folds)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(COVARIATES),
            strata=["S"],
            **columns,
        )
        .single()
    )


@dataclass(frozen=True)
class Measured:
    """One fit with its solver-warning count.

    Parameters
    ----------
    result : Any
        The single result of the fit.
    warnings : int
        The solver warnings of the fit.
    """

    result: Any
    warnings: int


def measured(frame: pd.DataFrame, setting: str, fold_seed: int, *, learner: Any = None) -> Measured:
    """Fit one sample and count its solver warnings.

    A non-finite estimate or standard error stops the run with a ``RuntimeError``. A recordable
    fitting error from ``cleverly`` or a numerical library, the empty stratum-fold
    ``DataError`` among them, is one lost replication, which the failure bound counts.
    """
    with base.solver_warning_count() as count:
        result = fit(frame, setting, fold_seed, learner=learner)
    for name in PARAMETERS:
        estimate = result.estimates[name]
        if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
            raise RuntimeError(f"a non-finite estimate or standard error of {name}: {estimate}")
    return Measured(result, count[0])


# ---------------------------------------------------------------------- the controls


def stratum_values(result: Any) -> np.ndarray:
    """The value of ``S`` at every row of the fit.

    The fit's own codes (``result.data.strata``) number the strata in the order the data
    present them, not by the value of ``S``, so every control reads the values through
    ``strata_levels``.
    """
    data = result.data
    levels = np.array([int(level[0]) for level in data.strata_levels], dtype=np.int64)
    return levels[np.asarray(data.strata, dtype=np.int64)]


def fold_index(result: Any) -> np.ndarray:
    """The validation fold of every row of the fit."""
    index = np.full(result.data.n, -1, dtype=np.int64)
    for fold, (_, test) in enumerate(result.nuisance.folds):
        index[np.asarray(test)] = fold
    if np.any(index < 0):
        raise RuntimeError("the validation folds do not cover the sample")
    return index


def fold_shares(result: Any, code: int) -> np.ndarray:
    """``P^w_{n,v}(S = s)`` of each row's own fold: the fold's stratum weight mass over its mass."""
    strata = stratum_values(result)
    weights = np.asarray(result.data.weights, dtype=float)
    folds = fold_index(result)
    shares = np.empty(len(result.nuisance.folds), dtype=float)
    for fold in range(shares.size):
        rows = folds == fold
        shares[fold] = float(np.sum(weights[rows & (strata == code)]) / np.sum(weights[rows]))
    return shares[folds]


def curve_se(curve: np.ndarray) -> float:
    """``sqrt(sum_i phi_i^2) / n``, the variance rule of every stored fold-evaluated curve."""
    curve = np.asarray(curve, dtype=float)
    return float(np.sqrt(np.sum(np.square(curve))) / curve.size)


def unnormalised_se(result: Any, label: str) -> float:
    """The SE of the stratum curve without its ``1 / P_{n,v}(S = s)`` normalisation.

    The plan's invalid-inference control: the stored curve of the stratum parameter times each
    row's fold share, ``P^w_{n,v}(S = s)``, read with the same variance rule. It shrinks the
    stratum SE by about the share, so it must fall below the SE-ratio band.
    """
    code = STRATUM_KEYS[label.split("_")[1]]
    if code is None:
        raise ValueError(f"{label} is a marginal parameter, which has no stratum share")
    curve = np.asarray(result.estimates[parameter(label)].influence_curve, dtype=float)
    return curve_se(curve * fold_shares(result, code))


def share_dropped_se(result: Any, label: str) -> float:
    """The SE of the marginal curve without its share term ``(w_i / w_bar)(psi_S_i - psi)``.

    The control that tests the marginal derivation: ``sum_s P_n(S = s) phi_s``, the stratum part
    of the mixture's curve alone, built from the stored stratum curves and the weighted sample
    shares, read with the same variance rule.
    """
    target, stratum = label.split("_")
    if STRATUM_KEYS[stratum] is not None:
        raise ValueError(f"{label} is a stratum parameter, whose curve has no share term")
    data = result.data
    strata = stratum_values(result)
    weights = np.asarray(data.weights, dtype=float)
    curve = np.zeros(data.n, dtype=float)
    for code in STRATA:
        share = float(np.sum(weights[strata == code]) / np.sum(weights))
        name = parameter(f"{target}_s{code}")
        curve += share * np.asarray(result.estimates[name].influence_curve, dtype=float)
    return curve_se(curve)


def share_term(result: Any, label: str) -> np.ndarray:
    """The share term of the marginal curve, ``(w_i / w_bar)(psi_S_i - psi)``, from the report."""
    target, _ = label.split("_")
    data = result.data
    strata = stratum_values(result)
    weights = np.asarray(data.weights, dtype=float)
    stratum_psi = np.array(
        [float(result.estimates[parameter(f"{target}_s{code}")].psi) for code in STRATA]
    )
    psi = float(result.estimates[TARGETS[target]].psi)
    return np.asarray(weights / weights.mean() * (stratum_psi[strata] - psi), dtype=float)


def initial_arms(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """The out-of-fold initial predictions ``Qbar(0, W)`` and ``Qbar(1, W)`` at every row."""
    nuisance = result.nuisance
    control, treated = nuisance.outcome.levels
    scaler = nuisance.scaler
    q0 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[control]), dtype=float)
    q1 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[treated]), dtype=float)
    return q0, q1


def untargeted(result: Any, label: str) -> float:
    """The fit's estimate of ``label`` without its fluctuation.

    A stratum parameter is the ``1/V`` average over the folds of the weighted plug-in of the
    initial out-of-fold regression over the fold's rows in the stratum, as the fit averages its
    targeted plug-ins. A marginal parameter is the weighted share mixture of the three.
    """
    target, stratum = label.split("_")
    code = STRATUM_KEYS[stratum]
    data = result.data
    if code is None:
        strata = stratum_values(result)
        weights = np.asarray(data.weights, dtype=float)
        return float(
            sum(
                np.sum(weights[strata == s])
                / np.sum(weights)
                * untargeted(result, f"{target}_s{s}")
                for s in STRATA
            )
        )
    q0, q1 = initial_arms(result)
    values = {"ate": q1 - q0, "ey0": q0, "ey1": q1}[target]
    strata = stratum_values(result)
    weights = np.asarray(data.weights, dtype=float)
    folds = [np.asarray(test) for _, test in result.nuisance.folds]
    plug_ins = []
    for test in folds:
        rows = test[strata[test] == code]
        plug_ins.append(float(np.average(values[rows], weights=weights[rows])))
    value = float(np.mean(plug_ins))
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate of {label}: {value}")
    return value


# ----------------------------------------------------------- the empty-cell refusal


def empty_cell(frame: pd.DataFrame, setting: str, folds: Iterable[Any]) -> bool:
    """Whether some validation fold lacks a stratum, or an arm inside a stratum.

    The data condition of :func:`cleverly.estimators._strata.refuse_empty_stratum_folds`,
    restated on the frame: a cell whose positive weight mass is zero. The pilot reads its rate
    at each size and fold count.
    """
    strata = frame["S"].to_numpy()
    treatment = frame["A"].to_numpy()
    weights = frame[WEIGHT].to_numpy(dtype=float) if setting == "weighted" else np.ones(len(frame))
    for _, test in folds:
        test = np.asarray(test)
        for code in STRATA:
            rows = test[strata[test] == code]
            if np.sum(weights[rows]) <= 0.0:
                return True
            for arm in (0.0, 1.0):
                if np.sum(weights[rows][treatment[rows] == arm]) <= 0.0:
                    return True
    return False

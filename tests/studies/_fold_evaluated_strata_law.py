"""The law, the fits, the truth and the controls of the ``fold-evaluated-strata`` study (X28).

The study measures ``TMLE(cv_evaluation=True)`` with ``strata=`` (X11 PR E1, roadmap X28) on the
arm targets ``estimands=("ate", "ey")``. Each stratum estimate is the ``1/V`` average of the fold
plug-ins over the rows of each validation fold in the stratum, each fold's stratum curve divides
by the fold's own stratum share, and the marginal estimate is the share mixture of the stratum
estimates with its delta-method curve.

The law is the RM30 family of :mod:`tests.studies._learned_rule_law` with the blip
``0.1 + 1.5 W1``: ``W1 ~ U(-1, 1)``, ``W2 ~ Bernoulli(0.5)``, ``logit g0(1 | W) = 0.3 W1 -
0.2 W2`` and the outcome regression ``logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + a (0.1 +
1.5 W1)``. Two outcome laws share that mean model, one per pair of settings:

``unweighted``, ``weighted``
    The bounded outcome ``Y | A, W ~ Beta(phi m, phi (1 - m))`` with ``m = Qbar0(A, W)`` and
    the concentration ``phi = 12`` of :mod:`tests.studies.bounded_cv_laws`, fitted with the
    quasibinomial learner and ``q_bounds=(0, 1)``.
``binary_unweighted``, ``binary_weighted``
    The RM30 outcome ``Y | A, W ~ Bernoulli(m)``, fitted with RM30's logistic learner and
    ``q_bounds=None`` (the package refuses ``q_bounds`` on a binary outcome).

The beta law exists for the share-dropped control (deviation D1 of the design record). Its
conditional variance ``m (1 - m) / (1 + phi)`` leaves the share term about 27% of the marginal
``ate`` variance, so a fit that dropped the share term would read an SE ratio of 0.854 and fail
the band. Under the Bernoulli outcome the residual variance ``m (1 - m)`` swamps the share term
(SE ratios without it 0.974, 0.996 and 0.941), so no binary setting carries that control. The
binary settings cover alike at this mean model: the design review's pilot (label 20264731)
read 0.942 to 0.952 on every binary cell. Its sparse-event deficit at the steeper blip
``0.1 + 3.0 W1`` (0.918 for ``ey[1]`` in stratum 0, about six events) is generic Wald behaviour
that the stacked stratified fit shares (0.919 on the same draws), so the ``binary_sparse``
setting reports it as a diagnostic with no verdict.

The baseline stratum cuts ``W1`` at ``-0.6`` and ``0.1``, so the shares are 0.2, 0.35 and
0.45, unequal, and the stratum effects are heterogeneous in sign. The weighted settings add the
known observation weight ``w = 1 + 1.5 W2``, which varies inside every stratum, so the fold's
stratum weight mass ``W_{v,s}`` and its row count ``n_{v,s}`` are not proportional. The target
of a weighted fit is the parameter of the weight-tilted law.

The target is fixed and known, so the truth needs no refit. Each stratum mean
``E_w[Qbar0(a, W) | S = s]`` integrates ``Qbar0`` over ``W1`` on the stratum's interval by the
trapezoid rule on :data:`POINTS` points for each value of ``W2``, weighted by ``w(W2)``. The
marginal truth is the share mixture, and ``S`` is independent of ``W2``, so the weight leaves the
shares at the interval lengths. The truth reads the mean model only, so a binary setting's
truths are its bounded twin's. The same quadrature, with each outcome law's conditional variance,
gives each parameter's efficient influence curve SD (the noise-control scale) and the asymptotic
SE ratio of the share-term control.
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

#: The steeper slope of the ``binary_sparse`` diagnostic, the design record's first pilot law.
SPARSE_SLOPE = 3.0

#: The beta concentration ``phi`` of the outcome, and the outcome support the fit declares.
#: Both equal :mod:`tests.studies.bounded_cv_laws`'s ``CONCENTRATION`` and ``Q_BOUNDS``, which
#: sample the same kind of proportion. The smallest beta shape ``phi min(m, 1 - m)`` of this law
#: is 1.09, above that module's floor of 0.70, so no draw rounds to 0 or 1.
PHI = 12.0
OUTCOME_BOUNDS = (0.0, 1.0)

#: The quadrature points of ``W1`` on each stratum's interval.
POINTS = 4001


@dataclass(frozen=True)
class Setting:
    """One setting's outcome law and weight.

    Parameters
    ----------
    outcome : str
        ``"beta"`` for the bounded outcome, ``"binary"`` for the Bernoulli one.
    weighted : bool
        Whether the fit reads the known weight ``1 + 1.5 W2``.
    slope : float
        The slope of the blip.
    """

    outcome: str
    weighted: bool
    slope: float = SLOPE


#: The calibration settings: the bounded outcome rows only and weighted, then the binary one.
SETTING_LAWS = {
    "unweighted": Setting("beta", False),
    "weighted": Setting("beta", True),
    "binary_unweighted": Setting("binary", False),
    "binary_weighted": Setting("binary", True),
}
SETTINGS = tuple(SETTING_LAWS)

#: The settings of the bounded outcome, the only ones whose share term the controls can resolve.
BOUNDED_SETTINGS = ("unweighted", "weighted")

#: The descriptive sparse-event setting: binary, rows only, blip ``0.1 + 3.0 W1``.
SPARSE = "binary_sparse"

#: Every setting a sample is drawn for, by name.
LAWS = {**SETTING_LAWS, SPARSE: Setting("binary", False, SPARSE_SLOPE)}

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


def blip(w1: Any, slope: float = SLOPE) -> np.ndarray:
    """The blip ``0.1 + slope W1`` on the logit scale; the declared slope is 1.5."""
    return np.asarray(INTERCEPT + slope * np.asarray(w1, dtype=float), dtype=float)


def qbar0(a: Any, w1: Any, w2: Any, slope: float = SLOPE) -> np.ndarray:
    """``E[Y | A = a, W]``, from ``logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + a b(W1)``."""
    w1 = np.asarray(w1, dtype=float)
    return np.asarray(
        expit(0.2 + 0.5 * w1 - 0.3 * np.asarray(w2, dtype=float) + np.asarray(a) * blip(w1, slope)),
        dtype=float,
    )


def weight(w2: Any) -> np.ndarray:
    """The known observation weight ``1 + 1.5 W2`` of the ``weighted`` setting."""
    return np.asarray(1.0 + 1.5 * np.asarray(w2, dtype=float), dtype=float)


def stratum_of(w1: Any) -> np.ndarray:
    """The baseline stratum code of each ``W1``."""
    return np.digitize(np.asarray(w1, dtype=float), EDGES[1:-1]).astype(np.int64)


def draw(setting: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``setting`` with ``n`` rows and columns ``W1, W2, S, A, Y, w``.

    The draws are the RM30 draws in the RM30 order (``W1``, ``W2``, ``A``, then ``Y``), with
    the setting's outcome regression and its outcome law around it, beta or Bernoulli; ``S``
    and ``w`` are functions of ``W``. Every setting's weight column is filled, and only a
    weighted fit reads it.
    """
    spec = LAWS[setting]
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, base.g0(w1, w2)).astype(float)
    mean = qbar0(a, w1, w2, spec.slope)
    if spec.outcome == "binary":
        y = rng.binomial(1, mean).astype(float)
    else:
        y = rng.beta(PHI * mean, PHI * (1.0 - mean))
    return pd.DataFrame(
        {"W1": w1, "W2": w2, "S": stratum_of(w1), "A": a, "Y": y, WEIGHT: weight(w2)}
    )


# ------------------------------------------------------------------------ the truth


def _target_parts(
    target: str, w1: np.ndarray, w2: float, setting: str = "unweighted"
) -> tuple[np.ndarray, np.ndarray]:
    """The regression functional and the conditional residual variance of the curve.

    With ``v(a)`` the outcome's conditional variance, ``Qbar0(a)(1 - Qbar0(a)) / (1 + phi)``
    for the beta outcome and ``Qbar0(a)(1 - Qbar0(a))`` for the Bernoulli one: for ``ate`` they
    are ``Qbar0(1) - Qbar0(0)`` and ``sum_a v(a) / g0(a)``; for ``ey[a]`` they are ``Qbar0(a)``
    and ``v(a) / g0(a)``.
    """
    spec = LAWS[setting]
    q1, q0 = qbar0(1.0, w1, w2, spec.slope), qbar0(0.0, w1, w2, spec.slope)
    g1 = base.g0(w1, w2)
    scale = 1.0 if spec.outcome == "binary" else 1.0 / (1.0 + PHI)
    r1 = scale * q1 * (1.0 - q1) / g1
    r0 = scale * q0 * (1.0 - q0) / (1.0 - g1)
    if target == "ate":
        return q1 - q0, r1 + r0
    if target == "ey0":
        return q0, r0
    if target == "ey1":
        return q1, r1
    raise KeyError(target)


def _weights(setting: str) -> dict[float, float]:
    """``w(W2)`` at each ``W2``; 1 for a rows-only setting."""
    weighted = LAWS[setting].weighted
    return {w2: float(weight(w2)) if weighted else 1.0 for w2 in base.W2_LEVELS}


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
            0.5 * w * np.trapezoid(_target_parts(target, grid, w2, setting)[0], grid) / length
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
            f, residual = _target_parts(target, grid, w2, setting)
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
    """The bounded settings' outcome learner: every pairwise product of ``(A, W1, W2, S)``,
    quasibinomial.

    It contains the law's linear predictor, so it is correctly specified. The quasibinomial
    score solver fits a proportion, which scikit-learn's logistic classifier refuses. The
    binary settings fit RM30's :func:`~tests.studies._learned_rule_law.outcome_learner`, the
    logistic classifier on the same products (:func:`learner_of`).
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


def learner_of(setting: str) -> Pipeline:
    """The declared outcome learner of ``setting``: quasibinomial on a proportion, logistic on
    a binary outcome, each on the pairwise products of ``(A, W1, W2, S)``."""
    return base.outcome_learner() if LAWS[setting].outcome == "binary" else outcome_learner()


def q_bounds_of(setting: str) -> tuple[float, float] | None:
    """The declared outcome support: ``(0, 1)`` for a proportion, none for a binary outcome."""
    return None if LAWS[setting].outcome == "binary" else OUTCOME_BOUNDS


def estimator(
    fold_seed: int,
    setting: str = "unweighted",
    *,
    learner: Any = None,
    n_folds: int = N_FOLDS,
    cv_evaluation: bool = True,
) -> TMLE:
    """The declared estimator of ``setting``, with the fold seed as its ``random_state``.

    ``cv_evaluation=False`` is the stacked stratified fit the ``binary_sparse`` diagnostic
    pairs with the declared one: the same learners, the same folds, and the stratum plug-ins
    over the stacked out-of-fold predictions.
    """
    return TMLE(
        outcome_learner=learner_of(setting) if learner is None else learner,
        treatment_learner=base.treatment_learner(),
        cross_fit=True,
        n_folds=n_folds,
        cv_evaluation=cv_evaluation,
        targeting_scheme="pooled",
        g_bounds="auto",
        q_bounds=q_bounds_of(setting),
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
    cv_evaluation: bool = True,
) -> Any:
    """Fit the declared estimator to one sample of ``setting`` and return the single result."""
    columns: dict[str, Any] = {"weights": WEIGHT} if LAWS[setting].weighted else {}
    return (
        estimator(fold_seed, setting, learner=learner, n_folds=n_folds, cv_evaluation=cv_evaluation)
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


def measured(
    frame: pd.DataFrame,
    setting: str,
    fold_seed: int,
    *,
    learner: Any = None,
    cv_evaluation: bool = True,
) -> Measured:
    """Fit one sample and count its solver warnings.

    A non-finite estimate or standard error stops the run with a ``RuntimeError``. A recordable
    fitting error from ``cleverly`` or a numerical library, the empty stratum-fold
    ``DataError`` among them, is one lost replication, which the failure bound counts.
    """
    with base.solver_warning_count() as count:
        result = fit(frame, setting, fold_seed, learner=learner, cv_evaluation=cv_evaluation)
    for name in PARAMETERS:
        estimate = result.estimates[name]
        if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
            raise RuntimeError(f"a non-finite estimate or standard error of {name}: {estimate}")
    return Measured(result, count[0])


@dataclass(frozen=True)
class Paired:
    """The declared fit and the stacked stratified fit of one sample, on the same folds.

    Parameters
    ----------
    declared : Measured
        The fold-evaluated fit (``cv_evaluation=True``).
    stacked : Measured
        The stacked fit (``cv_evaluation=False``) with the same learners and fold seed.
    """

    declared: Measured
    stacked: Measured


def paired(frame: pd.DataFrame, setting: str, fold_seed: int) -> Paired:
    """Fit one sample twice, fold-evaluated and stacked, with the same fold seed.

    The fold seed draws the same outer folds for both fits, which ``fold_index`` of each checks;
    a mismatch stops the run.
    """
    declared = measured(frame, setting, fold_seed)
    stacked = measured(frame, setting, fold_seed, cv_evaluation=False)
    if not np.array_equal(fold_index(declared.result), fold_index(stacked.result)):
        raise RuntimeError("the stacked fit drew other folds than the declared one")
    return Paired(declared, stacked)


def stratum_arm_events(frame: pd.DataFrame, label: str) -> int:
    """The fewest rows with the rarer outcome value in any stratum-arm ``label`` reads.

    A stratum label reads its stratum, and a marginal label every stratum; ``ate`` reads both
    arms and ``ey[a]`` arm ``a``. In each such stratum-arm the count is the smaller of the rows
    with ``Y = 1`` and with ``Y = 0``, because a Wald interval for a mean near 0 or 1 rests on
    the rarer value. The ``binary_sparse`` diagnostic reports it on each row.
    """
    target, stratum = label.split("_")
    code = STRATUM_KEYS[stratum]
    strata = STRATA if code is None else (code,)
    arms = {"ate": (0.0, 1.0), "ey0": (0.0,), "ey1": (1.0,)}[target]
    s = frame["S"].to_numpy()
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("an event count needs a binary outcome")
    counts = []
    for stratum_code in strata:
        for arm in arms:
            cell = y[(s == stratum_code) & (a == arm)]
            events = int(np.sum(cell))
            counts.append(min(events, cell.size - events))
    return min(counts)


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

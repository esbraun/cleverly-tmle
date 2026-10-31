"""The fits, the truth harness and the controls of the ``learned-rule-strata`` study (X11 (m)).

The study measures the learned-rule value inside baseline strata (X11 PR E2):
``TMLE(learned_rule=LearnedRule(), cross_fit=True, n_folds=5, cv_evaluation=True)`` fitted with
``strata=["S"]``. Fold ``v``'s plug-in rule ``d_v`` is learned from its training rows, and
stratum ``s`` of fold ``v`` evaluates ``Psi_{s,d_v}(P) = E_P[Qbar(d_v(W), W) | S = s]`` on the
fold's validation rows. Each stratum estimate is the ``1/V`` average of the fold plug-ins over
the fold's rows in the stratum, each fold's stratum curve divides by the fold's own stratum
share, and the marginal estimate is the share mixture of the stratum estimates. That is the
construction of X11 PR E1 (roadmap X28) with the fold rule as the regime.

The law is E1's, read from :mod:`tests.studies._fold_evaluated_strata_law` and never edited
here: the RM30 covariates and treatment mechanism, ``logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2
+ a b(W1)``, the stratum ``S`` cut from ``W1`` at -0.6 and 0.1 (shares 0.2, 0.35 and 0.45), and
the known weight ``w = 1 + 1.5 W2`` in a weighted setting. Each outcome law puts the boundary of
the true rule ``d0(W) = I(b(W1) > 0)`` inside a different stratum:

``unweighted``, ``weighted``
    E1's bounded settings: ``Beta(12 m, 12 (1 - m))`` with the blip ``0.1 + 1.5 W1``, zero at
    ``W1 = -1/15``, inside stratum 1. Stratum 0 is asymptotically treat-no-one and stratum 2
    treat-everyone. Quasibinomial outcome learner, ``q_bounds=(0, 1)``.
``binary_steep``, ``binary_steep_weighted``
    E1's Bernoulli outcome law with the blip ``-1.2 + 3.0 W1``, zero at ``W1 = 0.4``, inside
    stratum 2. Strata 0 and 1 are asymptotically treat-no-one. RM30's logistic outcome learner,
    no ``q_bounds``. E1's own binary blip ``0.1 - 0.3 W1`` has its root in stratum 2 as well,
    but a tenth of the slope, and the learned-rule value under-covered there in the pilot: the
    weak-margin limit of RM30's ``weak_blip`` law (design record, D1).

Each blip has a nonzero slope at its root (1.5 bounded, 3.0 binary), so the rule's boundary
has probability zero and the law is not exceptional (the margin condition C3 of RM30). The
binary outcome's larger residual variance needs the steeper blip for the same rule noise.

The target of a fit is data-adaptive: ``psi_s = V^-1 sum_v Psi_{s,d_v}(P0)`` for each stratum
and ``sum_s P0(S = s) psi_s = V^-1 sum_v Psi_{d_v}(P0)`` for the marginal value. The harness
(:func:`harness`) refits each fold's rule from the fold's training rows with the fit's own
outcome learner, task and scaled outcome, requires it to equal the fit's rule on every
validation row (rule L4) and the fit's stored fold rule on every row, and integrates
``Qbar0(d_v(W), W)`` over each stratum's interval of ``W1`` by the trapezoid rule on
:data:`~tests.studies._fold_evaluated_strata_law.POINTS` points for each ``W2``, weighted by
``w(W2)``. Each stratum's grid carries that stratum's code as ``S``, because the learners
read ``S`` (:func:`grid_design`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.interventions import LearnedRule
from cleverly.interventions.learned import fold_rules
from cleverly.learners._fitting import predict_mean
from tests.studies import _fold_evaluated_strata_law as strata
from tests.studies import _learned_rule_law as base

#: The outer folds of every fit.
N_FOLDS = strata.N_FOLDS

#: The blip ``-1.2 + 3.0 W1`` of the two binary settings: zero at ``W1 = 0.4``, inside stratum
#: 2. Two weaker binary blips were piloted first (label 20266011, design record D1): E1's own
#: ``0.1 - 0.3 W1`` (root in stratum 2, slope 0.3) covered 0.911 to 0.933 with SE ratios 0.88 to
#: 0.97 in every stratum, and ``-0.6 + 1.5 W1`` read the boundary stratum at ratio 0.936 and
#: 0.954. That is the weak-margin finite-sample limit RM30 records at its ``weak_blip`` law
#: (roadmap F27): the five fold rules disagree on a quarter to a half of the boundary stratum,
#: so the rule-learning term the curve omits is not yet negligible. The stratum construction is
#: not what it measures.
BINARY_INTERCEPT = -1.2
BINARY_SLOPE = 3.0

#: The calibration settings. The bounded ones are E1's, read from its module; the binary ones
#: are E1's binary outcome law with the steeper blip, built from E1's ``Setting``.
LAWS = {
    "unweighted": strata.LAWS["unweighted"],
    "weighted": strata.LAWS["weighted"],
    "binary_steep": strata.Setting("binary", False, BINARY_SLOPE, BINARY_INTERCEPT),
    "binary_steep_weighted": strata.Setting("binary", True, BINARY_SLOPE, BINARY_INTERCEPT),
}
SETTINGS = tuple(LAWS)

#: The settings of the bounded outcome, the only ones whose share term a control can resolve.
BOUNDED_SETTINGS = strata.BOUNDED_SETTINGS

#: The stratum codes, and the shares ``P(S = s)``.
STRATA = strata.STRATA
SHARES = strata.SHARES

#: The marginal parameter of every fit.
ESTIMAND = base.ESTIMAND

#: The stratum part of a cell label: one stratum code, or ``all`` for the marginal value.
STRATUM_KEYS: dict[str, int | None] = {"s0": 0, "s1": 1, "s2": 2, "all": None}

#: Every cell label of one setting, strata first.
PARAMETER_LABELS = tuple(STRATUM_KEYS)

#: The two-sided 95% normal multiplier of every interval the study writes itself.
CRITICAL = base.CRITICAL

#: The column names a row carries beside the shared schema (:data:`~tests.studies.
#: _learned_rule_law.HARNESS_COLUMNS`): the descriptive oracle SE, the rows whose refit rule
#: equals the fit's, and the solver warnings of the fit and its refits.
HARNESS_COLUMNS = base.HARNESS_COLUMNS


def parameter(label: str) -> str:
    """The reported parameter name of a label: ``ey_learned_rule[learned rule][S=0]``, say."""
    code = STRATUM_KEYS[label]
    return ESTIMAND if code is None else f"{ESTIMAND}[S={code}]"


#: Every reported parameter of a fit, in label order.
PARAMETERS = tuple(parameter(label) for label in PARAMETER_LABELS)


def rule_root(setting: str) -> float:
    """The root ``-intercept / slope`` of the setting's blip: the true rule's boundary in ``W1``."""
    spec = LAWS[setting]
    return -spec.intercept / spec.slope


def boundary_stratum(setting: str) -> int:
    """The stratum whose interval holds the true rule's boundary."""
    return int(strata.stratum_of(rule_root(setting)))


def limit_rule(setting: str) -> Callable[[np.ndarray, float], np.ndarray]:
    """``d0(w1, w2) = I(b(W1) > 0)``: the rule of the declared learner's limit.

    Every declared outcome learner holds the law's logit-linear predictor, so its population
    projection is ``Qbar0`` and its plug-in rule tends to ``d0``.
    """
    spec = LAWS[setting]

    def rule(w1: np.ndarray, w2: float) -> np.ndarray:
        del w2
        return np.asarray(strata.blip(w1, spec.slope, spec.intercept) > 0.0, dtype=bool)

    return rule


# ---------------------------------------------------------------------------- the law


def draw(setting: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``setting``: E1's draw, in E1's order, with the setting's blip.

    A bounded setting's sample is E1's own (:func:`~tests.studies._fold_evaluated_strata_law.
    draw`). A binary setting repeats E1's draw with this study's blip: the same ``W1``, ``W2``,
    ``A`` and the same uniform stream for ``Y``, so it differs from E1's binary sample only in
    the treated rows' outcome mean.
    """
    if setting in strata.LAWS and strata.LAWS[setting] == LAWS[setting]:
        return strata.draw(setting, n, seed)
    spec = LAWS[setting]
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, base.g0(w1, w2)).astype(float)
    mean = strata.qbar0(a, w1, w2, spec.slope, spec.intercept)
    if spec.outcome != "binary":
        raise ValueError(f"{setting} is not a binary setting")
    y = rng.binomial(1, mean).astype(float)
    return pd.DataFrame(
        {
            "W1": w1,
            "W2": w2,
            "S": strata.stratum_of(w1),
            "A": a,
            "Y": y,
            strata.WEIGHT: strata.weight(w2),
        }
    )


def learner_of(setting: str) -> Any:
    """The declared outcome learner: E1's quasibinomial learner on a proportion and RM30's
    logistic learner on a binary outcome, each on the pairwise products of ``(A, W1, W2, S)``."""
    return base.outcome_learner() if LAWS[setting].outcome == "binary" else strata.outcome_learner()


def q_bounds_of(setting: str) -> tuple[float, float] | None:
    """The declared outcome support: ``(0, 1)`` for a proportion, none for a binary outcome."""
    return None if LAWS[setting].outcome == "binary" else strata.OUTCOME_BOUNDS


# --------------------------------------------------------------------- the quadrature


def _weights(setting: str) -> dict[float, float]:
    """``w(W2)`` at each ``W2``; 1 for a rows-only setting."""
    weighted = LAWS[setting].weighted
    return {w2: float(strata.weight(w2)) if weighted else 1.0 for w2 in base.W2_LEVELS}


def grid(code: int) -> np.ndarray:
    """The quadrature points of ``W1`` on stratum ``code``'s interval."""
    return np.linspace(strata.EDGES[code], strata.EDGES[code + 1], strata.POINTS)


def grid_design(code: int, w2: float) -> np.ndarray:
    """The covariates ``(W1, W2, S)`` at the grid points of stratum ``code`` for one ``W2``.

    ``S`` is ``code`` at every point. The stratum's interval is closed on the left and open
    on the right, so its last grid point is the next stratum's first, a point of measure zero
    whose code the trapezoid rule must not read: a rule that reads ``S`` would otherwise jump
    there, an error of about ``1e-5`` in the stratum value.
    """
    points = grid(code)
    return np.column_stack([points, np.full_like(points, w2), np.full_like(points, float(code))])


@dataclass(frozen=True)
class RuleMoments:
    """The stratum values and curve variances of one rule in one setting, by quadrature.

    Parameters
    ----------
    stratum_values : tuple of float
        ``Psi_{s,d}(P0)``, the weight-tilted stratum means of ``Qbar0(d(W), W)``.
    value : float
        ``sum_s P(S = s) Psi_{s,d}(P0)``, the marginal value.
    stratum_variances : tuple of float
        The variance of each stratum value's efficient influence curve at the rule.
    variance : float
        The marginal value's.
    within_variance : float
        The variance of the marginal curve without its share term ``(w / E w)(psi_S - psi)``.
    """

    stratum_values: tuple[float, ...]
    value: float
    stratum_variances: tuple[float, ...]
    variance: float
    within_variance: float


def rule_moments(setting: str, treat: Mapping[tuple[int, float], np.ndarray]) -> RuleMoments:
    """The values and variances of a rule given on the grid of each stratum and ``W2``.

    ``treat[code, w2]`` is the rule at :func:`grid` of stratum ``code`` for that ``W2``. With
    ``t = w / E w``, ``Q = Qbar0(d(W), W)``, ``g = g0(d(W) | W)`` and the outcome's conditional
    variance ``v`` (``Q (1 - Q) / (1 + phi)`` for the beta outcome, ``Q (1 - Q)`` for the
    Bernoulli one), the stratum curve is ``t I(S = s) (R + Q - psi_s) / P(S = s)`` with
    ``E[R^2 | W] = v / g``, and the marginal curve is ``t (R + Q - psi)``.
    """
    spec = LAWS[setting]
    weights = _weights(setting)
    mean_weight = float(np.mean(list(weights.values())))
    scale = 1.0 if spec.outcome == "binary" else 1.0 / (1.0 + strata.PHI)
    parts: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    values: list[float] = []
    for code in STRATA:
        points = grid(code)
        length = strata.EDGES[code + 1] - strata.EDGES[code]
        numerator = 0.0
        for w2, w in weights.items():
            rule = np.asarray(treat[code, w2], dtype=float)
            q = strata.qbar0(rule, points, w2, spec.slope, spec.intercept)
            treated = base.g0(points, w2)
            g = np.where(rule > 0.5, treated, 1.0 - treated)
            parts[code, w2] = (q, scale * q * (1.0 - q) / g)
            numerator += 0.5 * w * float(np.trapezoid(q, points)) / length
        values.append(float(numerator / (0.5 * sum(weights.values()))))
    value = float(np.dot(SHARES, values))
    stratum_variances: list[float] = []
    variance = 0.0
    within = 0.0
    for code, share in zip(STRATA, SHARES, strict=True):
        points = grid(code)
        inside = 0.0
        for w2, w in weights.items():
            q, residual = parts[code, w2]
            # The density of W1 is 1/2 and each W2 has probability 1/2.
            mass = 0.5 * (w / mean_weight) ** 2 / 2.0
            inside += mass * float(np.trapezoid(residual + (q - values[code]) ** 2, points))
            variance += mass * float(np.trapezoid(residual + (q - value) ** 2, points))
        within += inside
        stratum_variances.append(inside / share**2)
    return RuleMoments(tuple(values), value, tuple(stratum_variances), variance, within)


def limit_moments(setting: str) -> RuleMoments:
    """:func:`rule_moments` at the limit rule :func:`limit_rule`."""
    rule = limit_rule(setting)
    return rule_moments(
        setting, {(code, w2): rule(grid(code), w2) for code in STRATA for w2 in base.W2_LEVELS}
    )


def efficiency_bound(setting: str, label: str) -> float:
    """The SD of ``sqrt(n)`` times the estimate of ``label`` at the limit rule."""
    found = limit_moments(setting)
    code = STRATUM_KEYS[label]
    variance = found.variance if code is None else found.stratum_variances[code]
    return float(np.sqrt(variance))


def share_term_ratio(setting: str) -> float:
    """The asymptotic SE ratio of the share-term control at the limit rule."""
    found = limit_moments(setting)
    return float(np.sqrt(found.within_variance / found.variance))


# ------------------------------------------------------------------------- the fits


def estimator(fold_seed: int, setting: str = "unweighted", *, learner: Any = None) -> TMLE:
    """The declared estimator of ``setting``, with the fold seed as its ``random_state``."""
    return TMLE(
        outcome_learner=learner_of(setting) if learner is None else learner,
        treatment_learner=base.treatment_learner(),
        learned_rule=LearnedRule(),
        cross_fit=True,
        n_folds=N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        q_bounds=q_bounds_of(setting),
        simultaneous=False,
        random_state=fold_seed,
    )


def fit(frame: pd.DataFrame, setting: str, fold_seed: int, *, learner: Any = None) -> Any:
    """Fit the declared estimator to one sample of ``setting`` and return the single result."""
    columns: dict[str, Any] = {"weights": strata.WEIGHT} if LAWS[setting].weighted else {}
    return (
        estimator(fold_seed, setting, learner=learner)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=list(strata.COVARIATES),
            strata=["S"],
            **columns,
        )
        .single()
    )


# -------------------------------------------------------------------------- the harness


@dataclass(frozen=True)
class Truth:
    """What the harness computes for one fit.

    Parameters
    ----------
    truths : dict
        ``V^-1 sum_v Psi_{s,d_v}(P0)`` for each stratum parameter, and the share mixture for
        the marginal one, by reported name.
    oracle_ses : dict
        ``sqrt(V^-2 sum_v Var_P0 D_{s,v} / n_v)`` for each parameter, the curve variance at
        each fold's rule over that fold's validation rows; descriptive.
    fold_values : tuple of tuple of float
        ``Psi_{s,d_v}(P0)``, one tuple of stratum values per fold.
    rows_checked : int
        The validation rows whose refit rule equals the fit's rule.
    """

    truths: dict[str, float]
    oracle_ses: dict[str, float]
    fold_values: tuple[tuple[float, ...], ...]
    rows_checked: int


def outcome_task(result: Any) -> str:
    """The task the fit's outcome learner solved: classification for a binary outcome."""
    return "classification" if result.data.family == "binomial" else "regression"


def fold_rule(result: Any, learner: Any, fold: int) -> Callable[[np.ndarray], np.ndarray]:
    """Refit fold ``fold``'s plug-in rule from its training rows; return it on ``(W1, W2, S)``.

    The refit is the fit's own: ``learner`` through the package's ``fit_on_rows`` on the
    design ``(A, W1, W2, S)`` with the fit's weights, the scaled outcome and the fit's task,
    and the rule is ``Qbar_v(1, W) - Qbar_v(0, W) > 0`` of the clipped mean prediction.
    RM30's :func:`~tests.studies._learned_rule_law.fold_rule` reads a binary outcome as a class
    label, which a proportion outcome is not, so the study refits through this function on
    both outcome laws.
    """
    data = result.data
    nuisance = result.nuisance
    train, _ = list(nuisance.folds)[fold]
    task = outcome_task(result)
    model = fit_on_rows(
        learner,
        np.asarray(data.treatment_design(), dtype=float),
        np.asarray(nuisance.scaler.scale(np.asarray(data.outcome, dtype=float)), dtype=float),
        np.asarray(data.weights, dtype=float),
        np.asarray(train),
        task,  # type: ignore[arg-type]
        None,
    )

    def rule(covariates: np.ndarray) -> np.ndarray:
        ones = np.ones((covariates.shape[0], 1))
        treated = np.clip(predict_mean(model, np.hstack([ones, covariates]), task), 0, 1)  # type: ignore[arg-type]
        control = np.clip(predict_mean(model, np.hstack([0.0 * ones, covariates]), task), 0, 1)  # type: ignore[arg-type]
        return np.asarray(treated - control > 0.0, dtype=bool)

    return rule


def harness(result: Any, learner: Any, setting: str) -> Truth:
    """Refit each fold's rule, check it against the fit's rules, and integrate its values.

    Raises
    ------
    HarnessMismatch
        When a refit rule differs from the fit's rule on any validation row (rule L4), or from
        the fit's stored rule of that fold on any row.
    """
    covariates = np.asarray(result.data.covariates, dtype=float)
    fitted = base.rule_of(result)
    stored = fold_rules(result.nuisance)
    if stored is None:
        raise base.HarnessMismatch("the fit stores no fold rules")
    values: list[tuple[float, ...]] = []
    moments: list[RuleMoments] = []
    sizes: list[int] = []
    checked = 0
    for fold, (_, test) in enumerate(result.nuisance.folds):
        test = np.asarray(test)
        rule = fold_rule(result, learner, fold)
        refit = rule(covariates)
        disagree = int(np.sum(refit[test] != fitted[test]))
        if disagree:
            raise base.HarnessMismatch(
                f"fold {fold}: the refit rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        elsewhere = int(np.sum(refit != stored[:, fold]))
        if elsewhere:
            raise base.HarnessMismatch(
                f"fold {fold}: the refit rule differs from the stored fold rule on {elsewhere} "
                f"of {len(refit)} rows"
            )
        checked += len(test)
        found = rule_moments(
            setting,
            {(code, w2): rule(grid_design(code, w2)) for code in STRATA for w2 in base.W2_LEVELS},
        )
        moments.append(found)
        values.append(found.stratum_values)
        sizes.append(len(test))
    folds = len(values)
    stratum_truths = np.mean(np.asarray(values), axis=0)
    truths = {parameter(f"s{code}"): float(stratum_truths[code]) for code in STRATA}
    truths[ESTIMAND] = float(np.dot(SHARES, stratum_truths))
    oracle = {
        parameter(f"s{code}"): float(
            np.sqrt(sum(m.stratum_variances[code] / n for m, n in zip(moments, sizes, strict=True)))
            / folds
        )
        for code in STRATA
    }
    oracle[ESTIMAND] = float(
        np.sqrt(sum(m.variance / n for m, n in zip(moments, sizes, strict=True))) / folds
    )
    return Truth(truths=truths, oracle_ses=oracle, fold_values=tuple(values), rows_checked=checked)


@dataclass(frozen=True)
class Measured:
    """One fit with its truth and its solver-warning count.

    Parameters
    ----------
    result : Any
        The single result of the fit.
    truth : Truth
        The harness of the fit.
    warnings : int
        The solver warnings of the fit and of its refits.
    """

    result: Any
    truth: Truth
    warnings: int


def measured(frame: pd.DataFrame, setting: str, fold_seed: int, *, learner: Any = None) -> Measured:
    """Fit one sample, run its harness with the same learner, and count the solver warnings.

    Rule L4 stops the run on a non-finite estimate or standard error and on a harness mismatch,
    each a ``RuntimeError`` the phases do not record. A recordable fitting error from
    ``cleverly`` or a numerical library, the empty stratum-fold ``DataError`` among them, is one
    lost replication, which the failure bound counts.
    """
    chosen = learner_of(setting) if learner is None else learner
    with base.solver_warning_count() as count:
        result = fit(frame, setting, fold_seed, learner=chosen)
        truth = harness(result, chosen, setting)
    for name in PARAMETERS:
        estimate = result.estimates[name]
        if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
            raise RuntimeError(f"a non-finite estimate or standard error of {name} (rule L4)")
    return Measured(result, truth, count[0])


# ---------------------------------------------------------------------- the controls


def unnormalised_se(result: Any, label: str) -> float:
    """The SE of the stratum curve without its ``1 / P_{n,v}(S = s)`` normalisation.

    The plan's invalid-inference control, E1's: the stored stratum curve times each row's
    fold share, read with the same variance rule. It shrinks the SE by about the share.
    """
    code = STRATUM_KEYS[label]
    if code is None:
        raise ValueError(f"{label} is the marginal parameter, which has no stratum share")
    curve = np.asarray(result.estimates[parameter(label)].influence_curve, dtype=float)
    return strata.curve_se(curve * strata.fold_shares(result, code))


def share_dropped_se(result: Any) -> float:
    """The SE of the marginal curve without its share term ``(w_i / w_bar)(psi_S_i - psi)``.

    ``sum_s P_n(S = s) phi_s`` from the stored stratum curves and the weighted sample shares,
    read with the same variance rule.
    """
    data = result.data
    codes = strata.stratum_values(result)
    weights = np.asarray(data.weights, dtype=float)
    curve = np.zeros(data.n, dtype=float)
    for code in STRATA:
        share = float(np.sum(weights[codes == code]) / np.sum(weights))
        name = parameter(f"s{code}")
        curve += share * np.asarray(result.estimates[name].influence_curve, dtype=float)
    return strata.curve_se(curve)


def untargeted(result: Any, label: str) -> float:
    """The fit's estimate of ``label`` without its fluctuation.

    A stratum value is the ``1/V`` average over the folds of the weighted mean of the initial
    out-of-fold regression at the fit's rule, ``Qbar_v(d_i, W_i)``, over the fold's rows in the
    stratum. The marginal value is the weighted share mixture of the three.
    """
    code = STRATUM_KEYS[label]
    data = result.data
    codes = strata.stratum_values(result)
    weights = np.asarray(data.weights, dtype=float)
    if code is None:
        return float(
            sum(
                np.sum(weights[codes == s]) / np.sum(weights) * untargeted(result, f"s{s}")
                for s in STRATA
            )
        )
    q0, q1 = strata.initial_arms(result)
    plug_in = np.where(base.rule_of(result), q1, q0)
    folds = [np.asarray(test) for _, test in result.nuisance.folds]
    means = []
    for test in folds:
        rows = test[codes[test] == code]
        means.append(float(np.average(plug_in[rows], weights=weights[rows])))
    value = float(np.mean(means))
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate of {label} (rule L4): {value}")
    return value


def treated_share(result: Any, code: int) -> float:
    """The share of stratum ``code``'s rows that the fit's rule treats; descriptive."""
    codes = strata.stratum_values(result)
    return float(np.mean(base.rule_of(result)[codes == code]))

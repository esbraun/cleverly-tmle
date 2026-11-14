r"""Laws, fits, truths and harness of ``learned-rule-optimal-compositions`` (X11 (f3)).

The study reads ``TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"])``, the
fold-evaluated CV-TMLE of the value of the optimal rule, under the five compositions X11 PR 2
admits. Every fit uses RM30's settings (ten folds, ``cv_evaluation=True``, pooled targeting,
``g_bounds="auto"``, the plug-in rule) and the known treatment mechanism
(``treatment_probabilities=``), so each target is a fixed parameter of the law:

``weighted``
    RM30's ``non_exceptional`` law with the weight ``w = 1 + A Y`` (:data:`WEIGHT_C` = 1, the
    weight of witness W-f3w's fit). The weight depends jointly on the treatment and the
    outcome, so it moves every part of the weighted population :math:`dP_w = w\,dP/E[w]`:
    its ``W`` marginal is tilted by :math:`E[w \mid W] = 1 + g_0(1 \mid W)\bar Q_0(1, W)`, its
    mechanism is :math:`g_w(1 \mid W) = g_0(1 \mid W)(1 + \bar Q_0(1, W))/E[w \mid W]`, its
    treated regression has odds :math:`1 + c` times RM30's (so the interaction logistic
    learner stays correct, the treated intercept shifted by :math:`\log 2`), and its optimal
    rule is :math:`d^w_0 = 1\{W_1 > -0.1 - \log 2\}`, which treats 34.7% of the population
    that :math:`d_0` leaves untreated. The target is
    :math:`\Psi^w_{d^w_0}(P_w) = E_w\bar Q_w(d^w_0(W), W)`, and the fit is given the weighted
    population's own mechanism :math:`g_w` as ``treatment_probabilities=``. The design
    mechanism :math:`g_0` beside these weights is the Fix-16 defect, which no cell measures.
``strata``
    The same law with ``strata=["W2"]``: each stratum's target is
    :math:`E_0[\bar Q_0(d_0(W), W) \mid W_2 = s]` and the marginal one :math:`\Psi_{d_0}`.
``known_regimes``
    The same law beside ``Static(1)`` and ``Static(0)``: the contrasts
    :math:`\Psi_{d_0} - \Psi_{a}` (``ate_optimal_rule[learned rule vs always a]``).
``in_sample``
    The same law with ``cross_fit=False``: the in-sample TMLE of the one rule learned on every
    row, with RM30's learner marked parametric (condition C5a), target :math:`\Psi_{d_0}`.
``intermediate``
    The law of :mod:`tests.studies._learned_rule_intermediate_law` (binary ``Z`` after ``A``),
    each level ``z`` held fixed: the target is
    :math:`\Psi^z_{d^z_0} = E_0\bar Q_0(d^z_0(W), z, W)`.

Every truth is an adaptive quadrature (:func:`scipy.integrate.quad`) over ``W1`` for each
``W2``, split at the optimal rule's threshold, where the curve's variance integrand jumps
(:func:`label_limits`); so is the SD of each label's efficient curve at its optimal rule, which
sizes the noise control. The design test checks each against a second, independent quadrature.
The value of an arbitrary fold rule, known only on RM30's 4,001 grid points of ``W1``, uses the
trapezoid rule on that grid (:func:`rule_values`), descriptive only. Each row also carries the data-adaptive value of the fit's own
fold rules (refit with the package's ``fit_on_rows`` and checked against the fit's rule on every
validation row, rule L4), the regret ``truth - rule_value`` and the record's
``rule_disagreement``, all descriptive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule, Static
from tests.studies import _learned_rule_contrast_law as contrast
from tests.studies import _learned_rule_in_sample_law as in_sample
from tests.studies import _learned_rule_intermediate_law as zlaw
from tests.studies import _learned_rule_optimal_law as optimal

base = optimal.base

#: The RM30 law of every setting but the intermediate one.
LAW = "non_exceptional"

#: The reported optimal value and its name in a fit's estimates.
ESTIMAND = optimal.ESTIMAND

#: The five settings, in declaration order.
SETTINGS = ("weighted", "strata", "known_regimes", "in_sample", "intermediate")

#: Each setting's labels, and the name of each label's parameter in the study's rows. The
#: package reports the stratum estimates as ``<value>[W2=s]`` and the contrasts by regime; the
#: intermediate fit reports one result per level, both named :data:`ESTIMAND`, so the study names
#: the level ``<value>[Z=z]``.
PARAMETERS: dict[str, dict[str, str]] = {
    "weighted": {"weighted": ESTIMAND},
    "strata": {
        "strata_all": ESTIMAND,
        "strata_s0": f"{ESTIMAND}[W2=0.0]",
        "strata_s1": f"{ESTIMAND}[W2=1.0]",
    },
    "known_regimes": {
        "contrast_1": "ate_optimal_rule[learned rule vs always 1]",
        "contrast_0": "ate_optimal_rule[learned rule vs always 0]",
    },
    "in_sample": {"in_sample": ESTIMAND},
    "intermediate": {
        "intermediate_z0": f"{ESTIMAND}[Z=0]",
        "intermediate_z1": f"{ESTIMAND}[Z=1]",
    },
}

#: The nine calibration labels, setting by setting.
LABELS = tuple(label for setting in SETTINGS for label in PARAMETERS[setting])

#: The setting of each label.
LABEL_SETTING = {label: setting for setting in SETTINGS for label in PARAMETERS[setting]}

#: The estimates a fit of each setting reports, in the package's order. The known-regime fit
#: also reports the optimal value and each regime's value, which the study reads only to check
#: the fit (W-f3-id holds the identity with the learned-rule fit).
REPORTED: dict[str, tuple[str, ...]] = {
    "weighted": (ESTIMAND,),
    "strata": (ESTIMAND, f"{ESTIMAND}[W2=0.0]", f"{ESTIMAND}[W2=1.0]"),
    "known_regimes": (
        ESTIMAND,
        "ey_known_regime[always 1]",
        "ey_known_regime[always 0]",
        "ate_optimal_rule[learned rule vs always 1]",
        "ate_optimal_rule[learned rule vs always 0]",
    ),
    "in_sample": (ESTIMAND,),
    "intermediate": (ESTIMAND,),
}

#: The qualifiers each setting's support key must hold (a fit that keys otherwise measures
#: another configuration, so rule L4 stops the run).
QUALIFIERS: dict[str, frozenset[str]] = {
    "weighted": frozenset({"known_mechanism"}),
    "strata": frozenset({"known_mechanism", "strata"}),
    "known_regimes": frozenset({"known_mechanism", "known_regimes"}),
    "in_sample": frozenset({"known_mechanism"}),
    "intermediate": frozenset({"known_mechanism"}),
}

#: The study's weight is ``1 + WEIGHT_C * A * Y``: witness W-f3w's fit-level weight
#: (``test_the_fit_lands_on_the_weighted_optimal_value``, ``JOINT_C``), under which the weighted
#: optimal rule treats a third of the population that the unweighted one does not.
WEIGHT_C = 1.0

#: The columns each row carries beside the shared schema. ``rule_value`` is the data-adaptive
#: value of the fit's own fold rules for the row's parameter, ``regret`` the label's truth minus
#: it, and ``rule_disagreement`` the record's share of rows on which the fold rules disagree;
#: ``rule_rows_checked`` counts the validation rows whose refit rule equals the fit's rule, and
#: ``solver_warnings`` the solver warnings of the fit and its refits (rule L4). All descriptive.
HARNESS_COLUMNS = (
    "rule_rows_checked",
    "solver_warnings",
    "rule_value",
    "regret",
    "rule_disagreement",
)

#: The trapezoid mass of each grid point of ``W1`` for one level of ``W2``, times the density
#: 1/2 of ``W1`` and the probability 1/2 of the level.
MASS = optimal.MASS


# --------------------------------------------------------------------------- the laws


def weight(a: Any, y: Any) -> np.ndarray:
    """The study's observation weight ``1 + c A Y`` (:data:`WEIGHT_C`)."""
    return 1.0 + WEIGHT_C * np.asarray(a, dtype=float) * np.asarray(y, dtype=float)


def weight_given_w(w1: Any, w2: Any) -> np.ndarray:
    """``E[w | W] = 1 + c g0(1 | W) Qbar0(1, W)``, the tilt of the weighted ``W`` marginal."""
    return np.asarray(1.0 + WEIGHT_C * base.g0(w1, w2) * base.qbar0(1.0, w1, w2, LAW), dtype=float)


def weighted_mechanism(w1: Any, w2: Any) -> np.ndarray:
    """``g_w(1 | W) = g0(1 | W) E[w | A = 1, W] / E[w | W]``, the weighted population's mechanism.

    ``E[w | A = 1, W] = 1 + c Qbar0(1, W)`` and ``E[w | A = 0, W] = 1``.
    """
    treated = base.g0(w1, w2)
    lifted = treated * (1.0 + WEIGHT_C * base.qbar0(1.0, w1, w2, LAW))
    return np.asarray(lifted / weight_given_w(w1, w2), dtype=float)


def weighted_qbar(a: Any, w1: Any, w2: Any) -> np.ndarray:
    """``Qbar_w(a, W) = E[w Y | a, W] / E[w | a, W]``: odds ``1 + c`` times RM30's at ``a = 1``."""
    q = base.qbar0(a, w1, w2, LAW)
    lift = 1.0 + WEIGHT_C * np.asarray(a, dtype=float)
    return np.asarray(lift * q / (1.0 + (lift - 1.0) * q), dtype=float)


def draw(setting: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows of ``setting``'s law, with the weight column ``w`` where used.

    Every setting but ``intermediate`` draws RM30's ``non_exceptional`` law exactly
    (:func:`tests.studies._learned_rule_law.draw`); ``intermediate`` draws
    :func:`tests.studies._learned_rule_intermediate_law.draw`.
    """
    if setting not in SETTINGS:
        raise KeyError(f"no setting {setting!r}")
    if setting == "intermediate":
        return zlaw.draw(n, seed)
    frame = base.draw(LAW, n, seed)
    if setting == "weighted":
        frame = frame.assign(w=weight(frame["A"], frame["Y"]))
    return frame


def known_probabilities(frame: pd.DataFrame, setting: str) -> np.ndarray:
    """``[g(0 | W), g(1 | W)]`` for each row: the treatment mechanism of the setting's target.

    RM30's ``g0`` for every setting but ``weighted``, which passes the weighted population's
    :func:`weighted_mechanism` (supplying ``g0`` beside a weight in ``(A, Y)`` is Fix-16).
    """
    if setting != "weighted":
        return optimal.known_probabilities(frame)
    treated = weighted_mechanism(
        frame["W1"].to_numpy(dtype=float), frame["W2"].to_numpy(dtype=float)
    )
    return np.column_stack([1.0 - treated, treated])


# ------------------------------------------------------------------------ quadrature

#: The blip ``0.1 + W1`` of the ``non_exceptional`` law vanishes at this ``W1``: the optimal
#: rule is ``1{W1 > THRESHOLD}``.
THRESHOLD = -0.1


def weighted_threshold() -> float:
    """The weighted optimal rule's threshold, ``-0.1 - log(1 + c)``."""
    return THRESHOLD - float(np.log1p(WEIGHT_C))


def _average(
    integrand: Callable[[float, float], float], threshold: float, levels: tuple[float, ...]
) -> float:
    """``E[f(W1, W2) | W2 in levels]`` with ``W1 ~ U(-1, 1)``, split at ``threshold``.

    Adaptive quadrature on each side of the rule's threshold, where the curve's integrands
    jump, averaged over the given levels of ``W2`` (each of probability 1/2).
    """
    total = 0.0
    for w2 in levels:
        for low, high in ((-1.0, threshold), (threshold, 1.0)):
            piece, _ = quad(
                lambda w1, w2=w2: integrand(w1, w2),
                low,
                high,
                epsabs=1e-13,
                epsrel=1e-12,
                limit=200,
            )
            total += 0.5 * piece / len(levels)
    return total


def _rule(threshold: float) -> Callable[[float], float]:
    return lambda w1: float(w1 > threshold)


def _plain(threshold: float, levels: tuple[float, ...]) -> tuple[float, float]:
    """The value of ``1{W1 > threshold}`` on ``W2 in levels`` and its curve's variance.

    The variance is ``E[Q (1 - Q)/g + (Q - psi)^2 | W2 in levels]`` over ``P(W2 in levels)``,
    the curve of a stratum carrying ``1{W2 = s}/P(W2 = s)``.
    """
    rule = _rule(threshold)

    def q(w1: float, w2: float) -> float:
        return float(base.qbar0(rule(w1), w1, w2, LAW))

    def g(w1: float, w2: float) -> float:
        treated = float(base.g0(w1, w2))
        return treated if rule(w1) else 1.0 - treated

    value = _average(q, threshold, levels)
    second = _average(
        lambda w1, w2: q(w1, w2) * (1.0 - q(w1, w2)) / g(w1, w2) + (q(w1, w2) - value) ** 2,
        threshold,
        levels,
    )
    return value, second * len(base.W2_LEVELS) / len(levels)


def _contrast(threshold: float, arm: float) -> tuple[float, float]:
    """``Psi_d - Psi_a`` and ``Var(D_d - D_a)``, as
    :func:`tests.studies._learned_rule_contrast_law.curve_variances` states it."""
    rule = _rule(threshold)
    levels = base.W2_LEVELS

    def part(w1: float, w2: float) -> tuple[float, float, float, float, float]:
        treated = float(base.g0(w1, w2))
        d = rule(w1)
        q_d = float(base.qbar0(d, w1, w2, LAW))
        q_a = float(base.qbar0(arm, w1, w2, LAW))
        g_d = treated if d else 1.0 - treated
        g_a = treated if arm else 1.0 - treated
        return d, q_d, q_a, g_d, g_a

    def gap(w1: float, w2: float) -> float:
        _, q_d, q_a, _, _ = part(w1, w2)
        return q_d - q_a

    difference = _average(gap, threshold, levels)

    def second(w1: float, w2: float) -> float:
        d, q_d, q_a, g_d, g_a = part(w1, w2)
        residual = q_d * (1.0 - q_d) / g_d + q_a * (1.0 - q_a) / g_a if d != arm else 0.0
        return residual + (q_d - q_a - difference) ** 2

    return difference, _average(second, threshold, levels)


def _weighted(threshold: float) -> tuple[float, float]:
    """``Psi^w_d(P_w)`` of ``d = 1{W1 > threshold}`` and the variance of ``w D_w / E[w]``.

    ``Psi^w_d = E[E(w | W) Qbar_w(d, W)] / E[w]``. With ``D_w = 1{A = d}(Y - Qbar_w(d))/g_w(d)
    + Qbar_w(d) - psi``, the variance of the weighted fit's curve ``w D_w / E[w]`` under ``P``
    sums ``g0(a | W) P(y | a, W) (1 + c a y)^2 D_w(a, y, W)^2`` over the four cells of
    ``(A, Y)``, over ``E[w]^2``.
    """
    rule = _rule(threshold)
    levels = base.W2_LEVELS
    mass = _average(lambda w1, w2: float(weight_given_w(w1, w2)), threshold, levels)

    def tilted(w1: float, w2: float) -> float:
        return float(weight_given_w(w1, w2) * weighted_qbar(rule(w1), w1, w2))

    value = _average(tilted, threshold, levels) / mass

    def second(w1: float, w2: float) -> float:
        d = rule(w1)
        q_w = float(weighted_qbar(d, w1, w2))
        g_w1 = float(weighted_mechanism(w1, w2))
        g_w = g_w1 if d else 1.0 - g_w1
        treated = float(base.g0(w1, w2))
        out = 0.0
        for a in (0.0, 1.0):
            g_a = treated if a else 1.0 - treated
            q_a = float(base.qbar0(a, w1, w2, LAW))
            for y, p_y in ((0.0, 1.0 - q_a), (1.0, q_a)):
                curve = (a == d) * (y - q_w) / g_w + q_w - value
                out += g_a * p_y * (1.0 + WEIGHT_C * a * y) ** 2 * curve**2
        return out

    return value, _average(second, threshold, levels) / mass**2


def _level(z: float) -> tuple[float, float]:
    """The value of ``d^z_0`` at level ``z`` of the intermediate law and its curve's variance,
    ``E[Q (1 - Q)/(g(d) q0(z | d, W))] + Var Q`` with ``Q = Qbar0(d^z_0(W), z, W)``."""
    threshold = THRESHOLD + zlaw.BLIP_SHIFT * z
    rule = _rule(threshold)
    levels = base.W2_LEVELS

    def q(w1: float, w2: float) -> float:
        return float(zlaw.qbar0(rule(w1), z, w1, w2))

    value = _average(q, threshold, levels)

    def second(w1: float, w2: float) -> float:
        d = rule(w1)
        treated = float(zlaw.g0(w1, w2))
        g = treated if d else 1.0 - treated
        product = g * float(zlaw.q0(z, d, w1))
        return q(w1, w2) * (1.0 - q(w1, w2)) / product + (q(w1, w2) - value) ** 2

    return value, _average(second, threshold, levels)


def _grid_value(
    rule: Mapping[float, np.ndarray],
    qbar: Callable[..., np.ndarray],
    tilt: Callable[[float], np.ndarray],
    levels: tuple[float, ...],
) -> float:
    total = sum(float(np.sum(MASS * tilt(w2))) for w2 in levels)
    return (
        sum(
            float(np.sum(MASS * tilt(w2) * qbar(np.asarray(rule[w2], dtype=float), base.GRID, w2)))
            for w2 in levels
        )
        / total
    )


def _unweighted_qbar(a: Any, w1: Any, w2: Any) -> np.ndarray:
    return base.qbar0(a, w1, w2, LAW)


def _flat(w2: float) -> np.ndarray:
    return np.ones(base.GRID.size)


def _tilt(w2: float) -> np.ndarray:
    return weight_given_w(base.GRID, w2)


def rule_values(setting: str, rule: Mapping[float, np.ndarray]) -> dict[str, float]:
    """Each label's value of ``setting`` at a rule given on the grid (RM30 law), by trapezoid.

    ``weighted``: :math:`E[E(w \\mid W)\\bar Q_w(d, W)]/E[w]`; ``strata``: the value within
    each level of ``W2`` and the marginal value; ``known_regimes``: :math:`\\Psi_d - \\Psi_a`
    for each arm; ``in_sample``: :math:`\\Psi_d`. The intermediate setting reads
    :func:`tests.studies._learned_rule_intermediate_law.rule_value` at each level instead.
    """
    levels = base.W2_LEVELS
    if setting == "weighted":
        return {"weighted": _grid_value(rule, weighted_qbar, _tilt, levels)}
    marginal = _grid_value(rule, _unweighted_qbar, _flat, levels)
    if setting == "strata":
        return {
            "strata_all": marginal,
            "strata_s0": _grid_value(rule, _unweighted_qbar, _flat, (0.0,)),
            "strata_s1": _grid_value(rule, _unweighted_qbar, _flat, (1.0,)),
        }
    if setting == "known_regimes":
        return {
            f"contrast_{arm}": marginal - contrast.static_value(float(arm), LAW) for arm in (1, 0)
        }
    if setting == "in_sample":
        return {"in_sample": marginal}
    raise KeyError(f"no grid value for setting {setting!r}")


@cache
def label_limits() -> dict[str, dict[str, float]]:
    """Each label's truth and the SD of ``sqrt(n)`` times its estimate at the optimal rule.

    The truth is the label's parameter at its optimal rule (:math:`d^w_0` for ``weighted``,
    :math:`d^z_0` for an intermediate level, :math:`d_0` otherwise), by adaptive quadrature
    split at the rule's threshold. ``bound`` is the SD of the label's efficient curve there:
    the weighted curve ``w D_w / E[w]``; the stratum curve ``1{W2 = s}/P(W2 = s) (...)``; the
    contrast curve :math:`D_{d_0} - D_a`; and the level curve with ``1{Z = z}/q0``. It sizes
    the noise controls only.
    """
    out: dict[str, dict[str, float]] = {}

    def put(label: str, value: float, variance: float) -> None:
        out[label] = {"truth": value, "bound": float(np.sqrt(variance))}

    put("weighted", *_weighted(weighted_threshold()))
    put("strata_all", *_plain(THRESHOLD, base.W2_LEVELS))
    for s in base.W2_LEVELS:
        put(f"strata_s{int(s)}", *_plain(THRESHOLD, (s,)))
    for arm in (1, 0):
        put(f"contrast_{arm}", *_contrast(THRESHOLD, float(arm)))
    out["in_sample"] = dict(out["strata_all"])
    for z in zlaw.LEVELS:
        put(f"intermediate_z{int(z)}", *_level(z))
    return out


def truth(label: str) -> float:
    """The fixed truth of ``label``."""
    return label_limits()[label]["truth"]


def unweighted_truth() -> float:
    """:math:`\\Psi_{d_0}(P_0)`, the unweighted optimal value, which the target control's
    ``target_gap_control`` is read against."""
    return label_limits()["strata_all"]["truth"]


@cache
def unweighted_rule_truth() -> float:
    """:math:`\\Psi^w_{d_0}(P_w)`, the weighted value of the unweighted optimal rule, which the
    target control's ``unweighted_rule_control`` is read against: where a fit that learned its
    rule without the weights would land."""
    return _weighted(THRESHOLD)[0]


# ------------------------------------------------------------------------------ the fit


def estimator(setting: str, fold_seed: int) -> TMLE:
    """The declared estimator of ``setting``, with rule L2's fold seed.

    RM30's settings with ``estimands=["ey_optimal_rule"]``: ten folds, the fold-evaluated
    report, pooled targeting, ``g_bounds="auto"``. ``known_regimes`` adds ``Static(1)`` and
    ``Static(0)``; ``in_sample`` sets ``cross_fit=False, cv_evaluation=False`` with RM30's
    learner marked parametric; ``intermediate`` uses the intermediate law's learners.
    """
    settings: dict[str, Any] = {
        "learned_rule": LearnedRule(),
        "estimands": ["ey_optimal_rule"],
        "n_folds": base.N_FOLDS,
        "cv_evaluation": True,
        "targeting_scheme": "pooled",
        "g_bounds": "auto",
        "outcome_learner": base.outcome_learner(),
        "treatment_learner": base.treatment_learner(),
        "random_state": fold_seed,
    }
    if setting == "known_regimes":
        settings["interventions"] = [Static(1), Static(0)]
    elif setting == "in_sample":
        settings.update(
            cross_fit=False, cv_evaluation=False, outcome_learner=in_sample.InteractionLogistic()
        )
    elif setting == "intermediate":
        settings.update(
            outcome_learner=zlaw.declared_learner(),
            intermediate_learner=zlaw.intermediate_learner(),
        )
    elif setting not in ("weighted", "strata"):
        raise KeyError(f"no setting {setting!r}")
    return TMLE(**settings)


def learner(setting: str) -> Any:
    """A fresh copy of ``setting``'s outcome learner, for the harness refit."""
    return {
        "in_sample": in_sample.InteractionLogistic,
        "intermediate": zlaw.declared_learner,
    }.get(setting, base.outcome_learner)()


def fit(frame: pd.DataFrame, setting: str, fold_seed: int) -> Any:
    """Fit ``setting`` to one sample with the known mechanism.

    Returns the single result, or the result set keyed by level for ``intermediate``.
    """
    arguments: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": list(base.COVARIATES),
        "treatment_probabilities": known_probabilities(frame, setting),
    }
    if setting == "weighted":
        arguments["weights"] = "w"
    elif setting == "strata":
        arguments["strata"] = ["W2"]
    elif setting == "intermediate":
        arguments["intermediate"] = "Z"
    results = estimator(setting, fold_seed).fit(frame, **arguments)
    return results if setting == "intermediate" else results.single()


# -------------------------------------------------------------------------- the harness


@dataclass(frozen=True)
class Measured:
    """One fit of one setting, its refit fold rules and its solver warnings.

    Parameters
    ----------
    setting : str
        The setting.
    results : Mapping
        Each label's result: the one fit, or the level's result for ``intermediate``.
    rule_values : Mapping
        Each label's value at the fit's own fold rules, averaged over the folds.
    rows_checked : Mapping
        Each label's validation rows whose refit rule equals the fit's rule.
    solver_warnings : int
        The solver warnings of the fit and its refits.
    """

    setting: str
    results: Mapping[str, Any]
    rule_values: Mapping[str, float]
    rows_checked: Mapping[str, int]
    solver_warnings: int

    def estimate(self, label: str) -> Any:
        """The label's reported estimate."""
        name = ESTIMAND if self.setting == "intermediate" else PARAMETERS[self.setting][label]
        return self.results[label].estimates[name]


def _grid_harness(result: Any, setting: str) -> tuple[dict[str, float], int]:
    """Refit each fold's rule, check it on the validation rows, and average each label's value.

    The refit is :func:`tests.studies._learned_rule_law.fold_rule`, the package's own
    ``fit_on_rows`` on the fold's training rows with the fit's weights, so the deterministic
    learner reproduces the fitted rule exactly. An in-sample fit's one pseudo-fold learned its
    rule on every row.
    """
    covariates = np.asarray(result.data.covariates, dtype=float)
    fitted = base.rule_of(result)
    grid = {w2: np.column_stack([base.GRID, np.full_like(base.GRID, w2)]) for w2 in base.W2_LEVELS}
    values: dict[str, list[float]] = {label: [] for label in PARAMETERS[setting]}
    checked = 0
    for fold, (_, test) in enumerate(result.nuisance.folds):
        rule = base.fold_rule(result, learner(setting), fold)
        disagree = int(np.sum(rule(covariates[test]) != fitted[test]))
        if disagree:
            raise base.HarnessMismatch(
                f"fold {fold}: the refit rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        checked += len(test)
        for label, value in rule_values(
            setting, {w2: rule(points) for w2, points in grid.items()}
        ).items():
            values[label].append(value)
    return {label: float(np.mean(v)) for label, v in values.items()}, checked


def _check(result: Any, setting: str) -> None:
    """Rule L4: refuse a fit that reports other parameters, keys otherwise, or is not finite."""
    names = tuple(result.estimates)
    # The stratum estimates come in the order the strata appear in the draw.
    if sorted(names) != sorted(REPORTED[setting]):
        raise base.HarnessMismatch(f"a {setting} fit reports {names}, not {REPORTED[setting]}")
    qualifiers = frozenset(result.config.support_key.qualifiers)
    if qualifiers != QUALIFIERS[setting]:
        raise base.HarnessMismatch(f"a {setting} fit keys {sorted(qualifiers)}")
    for name, estimate in result.estimates.items():
        if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
            raise RuntimeError(f"a non-finite estimate or SE of {name} (rule L4): {estimate}")


def measured(frame: pd.DataFrame, setting: str, fold_seed: int) -> Measured:
    """Fit one sample of ``setting``, run its harness, and count the solver warnings of both.

    Rule L4 stops the run on a non-finite estimate or SE, on a harness mismatch, on a fit that
    reports other parameters or levels, and on a support key with other qualifiers: each raises
    an error that the replication loops do not record.
    """
    with base.solver_warning_count() as count:
        results = fit(frame, setting, fold_seed)
        if setting == "intermediate":
            truths = zlaw.harness(results, learner(setting), estimand=ESTIMAND)
    if setting == "intermediate":
        if tuple(results) != zlaw.LEVELS:
            raise base.HarnessMismatch(f"the fit reports levels {tuple(results)}")
        labels = {f"intermediate_z{int(z)}": results[z] for z in zlaw.LEVELS}
        for result in labels.values():
            _check(result, setting)
        values = {f"intermediate_z{int(z)}": truths[z].truth for z in zlaw.LEVELS}
        checked = {f"intermediate_z{int(z)}": truths[z].rows_checked for z in zlaw.LEVELS}
    else:
        _check(results, setting)
        with base.solver_warning_count() as refit_count:
            values, rows = _grid_harness(results, setting)
        count[0] += refit_count[0]
        labels = dict.fromkeys(PARAMETERS[setting], results)
        checked = dict.fromkeys(PARAMETERS[setting], rows)
    return Measured(
        setting=setting,
        results=labels,
        rule_values=values,
        rows_checked=checked,
        solver_warnings=count[0],
    )


def harness_columns(fitted: Measured, label: str) -> dict[str, Any]:
    """The harness columns of one label's row."""
    (split,) = fitted.results[label].extra["learned_rule"].splits
    return {
        "rule_rows_checked": fitted.rows_checked[label],
        "solver_warnings": fitted.solver_warnings,
        "rule_value": fitted.rule_values[label],
        "regret": truth(label) - fitted.rule_values[label],
        "rule_disagreement": float(split.rule_disagreement),
    }


def _weighted_untargeted(result: Any, stratum: float | None = None) -> float:
    """The fold average (``1/V``) of each fold's weighted mean of ``Qbar_v(d_i, W_i)``.

    Each fold's plug-in is read off the out-of-fold outcome regression at the fit's own rule,
    over the fold's validation rows (those with ``W2 = stratum`` when one is given), weighted by
    the fit's observation weights, as the fit averages its targeted plug-ins.
    """
    nuisance = result.nuisance
    control, treated = nuisance.outcome.levels
    scaler = nuisance.scaler
    q0 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[control]), dtype=float)
    q1 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[treated]), dtype=float)
    plug_in = np.where(base.rule_of(result), q1, q0)
    data = result.data
    weights = np.ones(data.n) if data.weights is None else np.asarray(data.weights, dtype=float)
    covariates = np.asarray(data.covariates, dtype=float)
    means = []
    for _, test in nuisance.folds:
        rows = np.asarray(test)
        if stratum is not None:
            rows = rows[covariates[rows, base.COVARIATES.index("W2")] == stratum]
        means.append(float(np.average(plug_in[rows], weights=weights[rows])))
    value = float(np.mean(means))
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate (rule L4): {value}")
    return value


def untargeted_estimate(fitted: Measured, label: str) -> float:
    """The label's fit without its fluctuation: the fold average of its initial plug-ins.

    The RM30 reader (:func:`tests.studies._learned_rule_law.untargeted_estimate`) for the
    in-sample, intermediate and marginal stratified values; the weighted fold means of
    :func:`_weighted_untargeted` for the weighted value and each stratum; and the contrast
    reader (:func:`tests.studies._learned_rule_contrast_law.untargeted`) for the contrasts.
    """
    result = fitted.results[label]
    if label == "weighted":
        return _weighted_untargeted(result)
    if label in ("strata_s0", "strata_s1"):
        return _weighted_untargeted(result, float(label[-1]))
    if fitted.setting == "known_regimes":
        return float(contrast.untargeted(result, label))
    return float(base.untargeted_estimate(result))

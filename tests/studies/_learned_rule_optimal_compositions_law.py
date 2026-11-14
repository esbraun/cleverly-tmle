r"""Laws, fits, truths and harness of ``learned-rule-optimal-compositions`` (X11 (f3)).

The study reads ``TMLE(learned_rule=LearnedRule(), estimands=["ey_optimal_rule"])``, the
fold-evaluated CV-TMLE of the value of the optimal rule, under the five compositions X11 PR 2
admits. Every fit uses RM30's settings (ten folds, ``cv_evaluation=True``, pooled targeting,
``g_bounds="auto"``, the plug-in rule) and the known treatment mechanism
(``treatment_probabilities=``), so each target is a fixed parameter of the law:

``weighted``
    RM30's ``non_exceptional`` law with the known weight ``w = 1 + 1.5 * 1{W1 > 0}``. The weight
    is a function of ``W`` alone, so ``g_w = g0``, ``Qbar_w = Qbar0`` and ``d0^w = d0``; the
    target is :math:`\Psi^w_{d_0}(P_w) = E[w\bar Q_0(d_0(W), W)]/E[w]`.
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

Every truth is the trapezoid rule on RM30's 4,001 points of ``W1`` for each ``W2``
(:func:`label_limits`), and so is the SD of each label's efficient curve at its optimal rule,
which sizes the noise control. Each row also carries the data-adaptive value of the fit's own
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

#: The study's weight, ``1 + 1.5 * 1{W1 > 0}``: the weight of ``learned-rule-compositions``
#: and of witness W-f3w.
WEIGHT_STEP = 1.5

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


def weight(w1: Any) -> np.ndarray:
    """The study's observation weight ``1 + 1.5 * 1{W1 > 0}``."""
    return 1.0 + WEIGHT_STEP * (np.asarray(w1, dtype=float) > 0.0)


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
        frame = frame.assign(w=weight(frame["W1"]))
    return frame


def known_probabilities(frame: pd.DataFrame) -> np.ndarray:
    """``[g0(0 | W), g0(1 | W)]`` for each row: RM30's mechanism, shared by both laws."""
    return optimal.known_probabilities(frame)


# ------------------------------------------------------------------------ quadrature


def _value(
    rule: Mapping[float, np.ndarray], density: Callable[[float], np.ndarray]
) -> tuple[float, float]:
    """``E_P Qbar0(d(W), W)`` and the variance of the curve under the tilt ``density``.

    ``density(w2)`` is the (unnormalized) density of ``W`` at each grid point, relative to
    :data:`MASS`. With ``h = density / E[density]``, ``Q = Qbar0(d, W)`` and ``g = g0(d | W)``,
    the value is ``E[h Q]`` and the variance of ``h (1{A = d}(Y - Q)/g + Q - psi)`` is
    ``E[h^2 (Q (1 - Q)/g + (Q - psi)^2)]``.
    """
    total = sum(float(np.sum(MASS * density(w2))) for w2 in base.W2_LEVELS)
    value = 0.0
    for w2 in base.W2_LEVELS:
        q = base.qbar0(np.asarray(rule[w2], dtype=float), base.GRID, w2, LAW)
        value += float(np.sum(MASS * density(w2) * q)) / total
    second = 0.0
    for w2 in base.W2_LEVELS:
        treat = np.asarray(rule[w2], dtype=float)
        q = base.qbar0(treat, base.GRID, w2, LAW)
        treated = base.g0(base.GRID, w2)
        g = np.where(treat > 0.5, treated, 1.0 - treated)
        h = density(w2) / total
        second += float(np.sum(MASS * h**2 * (q * (1.0 - q) / g + (q - value) ** 2)))
    return value, second


def _weighted(w2: float) -> np.ndarray:
    return weight(base.GRID)


def _stratum(s: float) -> Callable[[float], np.ndarray]:
    return lambda w2: np.full(base.GRID.size, float(w2 == s))


def _everyone(w2: float) -> np.ndarray:
    return np.ones(base.GRID.size)


def rule_values(setting: str, rule: Mapping[float, np.ndarray]) -> dict[str, float]:
    """Each label's value of ``setting`` at a rule given on the grid (RM30 law).

    ``weighted``: :math:`E[w\\bar Q_0(d, W)]/E[w]`; ``strata``: the value within each level of
    ``W2`` and the marginal value; ``known_regimes``: :math:`\\Psi_d - \\Psi_a` for each arm;
    ``in_sample``: :math:`\\Psi_d`. The intermediate setting reads
    :func:`tests.studies._learned_rule_intermediate_law.rule_value` at each level instead.
    """
    if setting == "weighted":
        return {"weighted": _value(rule, _weighted)[0]}
    if setting == "strata":
        return {
            "strata_all": _value(rule, _everyone)[0],
            "strata_s0": _value(rule, _stratum(0.0))[0],
            "strata_s1": _value(rule, _stratum(1.0))[0],
        }
    if setting == "known_regimes":
        value = _value(rule, _everyone)[0]
        return {f"contrast_{arm}": value - contrast.static_value(float(arm), LAW) for arm in (1, 0)}
    if setting == "in_sample":
        return {"in_sample": _value(rule, _everyone)[0]}
    raise KeyError(f"no grid value for setting {setting!r}")


@cache
def label_limits() -> dict[str, dict[str, float]]:
    """Each label's truth and the SD of ``sqrt(n)`` times its estimate at the optimal rule.

    The truth is the label's parameter at :math:`d_0` (at :math:`d^z_0` for an intermediate
    level), by the trapezoid rule. ``bound`` is the SD of the label's efficient curve there: the
    tilted curve for ``weighted``; the stratum curve ``1{W2 = s}/P(W2 = s) (...)`` for a stratum,
    whose variance is the within-stratum one over ``P(W2 = s) = 1/2``; the contrast curve
    :math:`D_{d_0} - D_a` of :func:`tests.studies._learned_rule_contrast_law.curve_variances`;
    and the level curve of :func:`tests.studies._learned_rule_intermediate_law.level_limit`. It
    sizes the noise controls only.
    """
    d0 = optimal.optimal_rule(optimal.law_qbar(LAW))
    out: dict[str, dict[str, float]] = {}
    value, variance = _value(d0, _weighted)
    out["weighted"] = {"truth": value, "bound": float(np.sqrt(variance))}
    value, variance = _value(d0, _everyone)
    out["strata_all"] = {"truth": value, "bound": float(np.sqrt(variance))}
    out["in_sample"] = dict(out["strata_all"])
    for s in base.W2_LEVELS:
        value, variance = _value(d0, _stratum(s))
        out[f"strata_s{int(s)}"] = {"truth": value, "bound": float(np.sqrt(variance))}
    for arm in (1, 0):
        _, difference = contrast.curve_variances(d0, float(arm), LAW)
        out[f"contrast_{arm}"] = {
            "truth": out["strata_all"]["truth"] - contrast.static_value(float(arm), LAW),
            "bound": float(np.sqrt(difference)),
        }
    for z in zlaw.LEVELS:
        limit = zlaw.level_limit(z)
        out[f"intermediate_z{int(z)}"] = {"truth": limit["truth"], "bound": limit["bound"]}
    return out


def truth(label: str) -> float:
    """The fixed truth of ``label``."""
    return label_limits()[label]["truth"]


def unweighted_truth() -> float:
    """:math:`\\Psi_{d_0}(P_0)`, the value the weighted target control is read against."""
    return label_limits()["strata_all"]["truth"]


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
        "treatment_probabilities": known_probabilities(frame),
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


def untargeted_estimate(fitted: Measured, label: str) -> float:
    """The label's fit without its fluctuation, where the shared reader applies, else ``nan``.

    The RM30 reader averages the untargeted plug-ins of the learned-rule value over the folds; it
    applies to the weighted, in-sample and intermediate fits, whose one parameter is that value.
    The stratum and contrast parameters have no such reader here, so their column is ``nan``.
    """
    if fitted.setting in ("weighted", "in_sample", "intermediate"):
        return float(base.untargeted_estimate(fitted.results[label]))
    return float("nan")

r"""The truth of a learned-rule fit beside a known static regime (X11 PR G, part (c)).

``TMLE(learned_rule=..., interventions=[Static(a)])`` reports the learned-rule value, the known
regime's value :math:`\Psi_{a}(P_0) = E_0\{\bar Q_0(a, W)\}` and their contrast
:math:`\tilde\psi_{0n} - \Psi_a(P_0)`.  The first term is data-adaptive: it is the fold average
of the values of the rules the fit learned, which the RM30 harness
:func:`tests.studies._learned_rule_law.harness` refits and integrates.  The second is fixed: it
is the same trapezoid rule on the RM30 grid of ``W1``, for each ``W2``, applied to the rule that
assigns ``a`` everywhere.

The module reads the RM30 laws and the harness of :mod:`tests.studies._learned_rule_law`
without editing them.  ``tests/unit/test_learned_rule_contrast_truth.py`` checks the static value
and the contrast truth against a closed form and against Monte Carlo draws.

The ``learned-rule-contrast`` study (:mod:`tests.studies.learned_rule_contrast`) fits each draw
beside ``Static(1)`` and ``Static(0)`` under one or both reports and reads five parameters off
each fit (:data:`TAGS`).  :func:`harness` refits each fold's rule once, checks it against the
fit's rule on every validation row (rule L4), and integrates on the RM30 grid the value of each
fold rule and the variances of each fold's curves, so every parameter's truth and oracle SE come
from one refit.  :func:`measured` fits the reports of one draw on one fold seed and refuses a
pair that drew other folds or rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule, Static
from tests.studies import _learned_rule_law as base

#: The two reports of a learned-rule fit the study reads, by their ``cv_evaluation``.
REPORTS = {"fold_evaluated": True, "stacked": False}

#: The known regimes of every study fit, in ``interventions=`` order: ``Static(1)`` is column 1
#: of the regime set and ``Static(0)`` column 2.
ARMS = (1, 0)

#: The short tag of each reported parameter, and its name in ``result.estimates``. The study
#: reads all five on every fit: the learned-rule value, each known regime's value, and each
#: contrast, rule minus regime.
TAGS = {
    "value": "ey_learned_rule[learned rule]",
    "known_1": "ey_known_regime[always 1]",
    "known_0": "ey_known_regime[always 0]",
    "contrast_1": "ate_learned_rule[learned rule vs always 1]",
    "contrast_0": "ate_learned_rule[learned rule vs always 0]",
}

#: The arm of each tag that reads a known regime, or ``None`` for the learned-rule value.
TAG_ARMS: dict[str, int | None] = {
    "value": None,
    "known_1": 1,
    "known_0": 0,
    "contrast_1": 1,
    "contrast_0": 0,
}

#: The contrast tags, rule minus regime.
CONTRAST_TAGS = ("contrast_1", "contrast_0")

#: The two laws the study draws, RM30's.
SCENARIOS = ("non_exceptional", "misspecified_limit")

#: The columns each row carries beside the shared schema. ``oracle_se`` is descriptive.
#: ``rule_rows_checked`` counts the validation rows whose refit rule equals the fit's rule, and
#: ``solver_warnings`` the solver warnings of the fits and the refits of the draw (rule L4).
HARNESS_COLUMNS = base.HARNESS_COLUMNS


def estimator(
    learner: Any, fold_seed: int, arms: tuple[int, ...] = (1,), *, cv_evaluation: bool = True
) -> TMLE:
    """The RM30 estimator beside ``Static(a)`` for each ``a`` of ``arms``.

    The settings are those of :func:`tests.studies._learned_rule_law.estimator`, and every
    other setting is the constructor's default, ``simultaneous=True`` among them.
    ``cv_evaluation=False`` is the stacked report (X11 (e)).
    """
    return TMLE(
        learned_rule=LearnedRule(),
        interventions=[Static(arm) for arm in arms],
        n_folds=base.N_FOLDS,
        cv_evaluation=cv_evaluation,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


def fit(
    frame: pd.DataFrame,
    learner: Any,
    fold_seed: int,
    arms: tuple[int, ...] = (1,),
    *,
    cv_evaluation: bool = True,
) -> Any:
    """Fit :func:`estimator` to one sample and return the single result."""
    return (
        estimator(learner, fold_seed, arms, cv_evaluation=cv_evaluation)
        .fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES))
        .single()
    )


def static_value(arm: float, law: str) -> float:
    r"""``Psi_a(P0) = E Qbar0(a, W)``, the value of the known regime ``Static(arm)``.

    Parameters
    ----------
    arm : float
        The arm the regime assigns everywhere, 0 or 1.
    law : str
        A law of :data:`tests.studies._learned_rule_law.BLIPS`.

    Returns
    -------
    float
        The value, by the trapezoid rule of
        :func:`tests.studies._learned_rule_law.rule_value` on the RM30 grid.

    Raises
    ------
    ValueError
        If ``arm`` is neither 0 nor 1.
    """
    if arm not in (0.0, 1.0):
        raise ValueError(f"a static regime of this law assigns arm 0 or 1; got {arm}")
    grid = base.fixed_rule_grid(lambda w1, _w2: np.full(w1.shape, bool(arm)))
    value, _ = base.rule_value(grid, law)
    return value


@dataclass(frozen=True)
class ContrastTruth:
    """The truth of one contrast fit against one known static regime.

    Parameters
    ----------
    learned : tests.studies._learned_rule_law.Truth
        The harness truth of the learned-rule value, :math:`\\tilde\\psi_{0n}`.
    known : float
        :math:`\\Psi_a(P_0)`, the fixed value of the known regime.
    contrast : float
        The truth of the reported contrast: ``learned.truth - known``, or its negative when
        ``reference=`` names the learned rule.
    """

    learned: base.Truth
    known: float
    contrast: float


def contrast_truth(
    result: Any,
    learner: Any,
    law: str,
    arm: float,
    *,
    column: int = 1,
    reference_is_rule: bool = False,
) -> ContrastTruth:
    """Return the truth of ``ate_learned_rule`` for a fit beside ``Static(arm)``.

    Parameters
    ----------
    result : TMLEResult
        A learned-rule fit of one split with known regimes, on a draw of ``law``.
    learner : estimator
        The fit's outcome learner, which the harness refits on each fold's training rows.
    law : str
        The law the fit's data were drawn from.
    arm : float
        The arm of the known static regime, 0 or 1.
    column : int, default=1
        The regime-set column of that regime: 1 for the first regime of ``interventions=``.
    reference_is_rule : bool, default=False
        Whether ``reference=`` names the learned rule, which reports the regime minus the rule.

    Returns
    -------
    ContrastTruth
        The learned-rule truth, the known value and the contrast truth.

    Raises
    ------
    ValueError
        If the fit's column ``column`` does not assign ``arm`` on every row.
    tests.studies._learned_rule_law.HarnessMismatch
        If a refitted fold rule differs from the fit's rule on a validation row.
    """
    values = np.asarray(result.nuisance.regimes.values, dtype=float)
    if not np.all(values[:, int(arm), column] == 1.0):
        raise ValueError(f"column {column} of the fit's regime set is not Static({arm:g})")
    learned = base.harness(result, learner, law)
    known = static_value(arm, law)
    difference = learned.truth - known
    return ContrastTruth(
        learned=learned, known=known, contrast=-difference if reference_is_rule else difference
    )


# ------------------------------------------------------------- the study's harness


def curve_variances(treat: Mapping[float, np.ndarray], arm: float, law: str) -> tuple[float, float]:
    r"""``Var D_K`` and ``Var(D_d - D_K)`` at the rule ``treat`` and the known ``Static(arm)``.

    With :math:`Q_r = \bar Q_0(r(W), W)`, :math:`g_r = g_0(r(W) \mid W)` and
    :math:`D_r = 1\{A = r(W)\}(Y - Q_r)/g_r + Q_r - \Psi_r`, the residual terms of
    :math:`D_d` and :math:`D_K` cancel where :math:`d(W) = a` and are disjoint where it is
    not, so

    .. math::

        \mathrm{Var}(D_d - D_K) = E\big[1\{d \ne a\}\{Q_d(1 - Q_d)/g_d + Q_a(1 - Q_a)/g_a\}
        \big] + \mathrm{Var}(Q_d - Q_a),

    and :math:`\mathrm{Var}\,D_K = E[Q_a(1 - Q_a)/g_a] + \mathrm{Var}\,Q_a`. Both integrals
    use the trapezoid rule of :func:`tests.studies._learned_rule_law.rule_value`.
    """
    known_second = known_mean = residual = gap_second = gap_mean = 0.0
    for w2 in base.W2_LEVELS:
        rule = np.asarray(treat[w2], dtype=float)
        treated = base.g0(base.GRID, w2)
        q_d = base.qbar0(rule, base.GRID, w2, law)
        q_a = base.qbar0(arm, base.GRID, w2, law)
        g_d = np.where(rule > 0.5, treated, 1.0 - treated)
        g_a = treated if arm else 1.0 - treated
        differ = (rule > 0.5) != bool(arm)

        def mean(values: np.ndarray) -> float:
            return float(0.5 * np.trapezoid(values, base.GRID) / 2.0)

        known_second += mean(q_a * (1.0 - q_a) / g_a + q_a**2)
        known_mean += mean(q_a)
        residual += mean(differ * (q_d * (1.0 - q_d) / g_d + q_a * (1.0 - q_a) / g_a))
        gap_second += mean((q_d - q_a) ** 2)
        gap_mean += mean(q_d - q_a)
    return known_second - known_mean**2, residual + gap_second - gap_mean**2


def tag_variance(tag: str, treat: Mapping[float, np.ndarray], law: str) -> float:
    """The variance of one fold's curve of the parameter ``tag`` at the rule ``treat``."""
    arm = TAG_ARMS[tag]
    if arm is None:
        return base.rule_value(treat, law)[1]
    known, difference = curve_variances(treat, float(arm), law)
    return known if tag.startswith("known") else difference


def efficiency_bounds(treat: Mapping[float, np.ndarray], law: str) -> dict[str, float]:
    """The SD of ``sqrt(n)`` times each parameter's estimate at the limit rule ``treat``."""
    return {tag: float(np.sqrt(tag_variance(tag, treat, law))) for tag in TAGS}


@dataclass(frozen=True)
class Harness:
    """The refit fold rules of one draw: each fold's value and curve variances.

    Parameters
    ----------
    fold_values : tuple of float
        ``Psi_{d_v}(P0)`` for each fold.
    fold_variances : Mapping
        Each tag to the variance of its curve in each fold, at that fold's rule.
    sizes : tuple of int
        Each fold's validation rows.
    rows_checked : int
        The validation rows whose refit rule equals the fit's rule.
    """

    fold_values: tuple[float, ...]
    fold_variances: Mapping[str, tuple[float, ...]]
    sizes: tuple[int, ...]
    rows_checked: int


def harness(result: Any, learner: Any, law: str) -> Harness:
    """Refit each fold's rule, check it against the fit's rule, and integrate its moments.

    The refit is :func:`tests.studies._learned_rule_law.fold_rule`, the package's own
    ``fit_on_rows`` on the fit's training rows, so a deterministic learner reproduces the fitted
    rule exactly.

    Raises
    ------
    tests.studies._learned_rule_law.HarnessMismatch
        When the refit rule differs from the fit's rule on any validation row.
    """
    covariates = np.asarray(result.data.covariates, dtype=float)
    fitted = base.rule_of(result)
    grid = {w2: np.column_stack([base.GRID, np.full_like(base.GRID, w2)]) for w2 in base.W2_LEVELS}
    values: list[float] = []
    variances: dict[str, list[float]] = {tag: [] for tag in TAGS}
    sizes: list[int] = []
    checked = 0
    for fold, (_, test) in enumerate(result.nuisance.folds):
        rule = base.fold_rule(result, learner, fold)
        disagree = int(np.sum(rule(covariates[test]) != fitted[test]))
        if disagree:
            raise base.HarnessMismatch(
                f"fold {fold}: the refit rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        checked += len(test)
        treat = {w2: rule(points) for w2, points in grid.items()}
        values.append(base.rule_value(treat, law)[0])
        for tag in TAGS:
            variances[tag].append(tag_variance(tag, treat, law))
        sizes.append(len(test))
    return Harness(
        fold_values=tuple(values),
        fold_variances={tag: tuple(v) for tag, v in variances.items()},
        sizes=tuple(sizes),
        rows_checked=checked,
    )


def report_weights(result: Any) -> np.ndarray:
    """The fold weights of the fit's report, computed from its folds and checked on its record.

    ``1/V`` for the fold-evaluated report and ``n_v / n`` for the stacked one (the fit has no
    weights).

    Raises
    ------
    tests.studies._learned_rule_law.HarnessMismatch
        When the record's ``fold_weights`` differ from the weights computed here.
    """
    (split,) = result.extra["learned_rule"].splits
    sizes = np.array([len(test) for _, test in result.nuisance.folds], dtype=float)
    weights = (
        sizes / sizes.sum() if split.scheme == "stacked" else np.full(sizes.size, 1 / sizes.size)
    )
    if not np.allclose(split.fold_weights, weights, rtol=0.0, atol=1e-12):
        raise base.HarnessMismatch(
            f"the record's {split.scheme} fold weights {split.fold_weights} are not {weights}"
        )
    return weights


def truths(result: Any, refit: Harness, law: str) -> dict[str, tuple[float, float]]:
    """Each tag's truth and oracle SE for the report of ``result``.

    The learned-rule value's truth is ``sum_v omega_v Psi_{d_v}(P0)`` with the report's fold
    weights; a known regime's is ``Psi_a(P0)``, the same in every fold; a contrast's is their
    difference. The oracle SE is ``sqrt(sum_v omega_v^2 Var D_v / n_v)`` of each curve.
    """
    weights = report_weights(result)
    value = float(np.dot(weights, refit.fold_values))
    out: dict[str, tuple[float, float]] = {}
    for tag, arm in TAG_ARMS.items():
        variances = np.asarray(refit.fold_variances[tag])
        oracle = float(np.sqrt(np.sum(weights**2 * variances / np.asarray(refit.sizes))))
        if arm is None:
            truth = value
        elif tag.startswith("known"):
            truth = static_value(float(arm), law)
        else:
            truth = value - static_value(float(arm), law)
        out[tag] = (truth, oracle)
    return out


def check_known_columns(result: Any) -> None:
    """Refuse a fit whose known columns are not ``Static(1)`` then ``Static(0)``."""
    values = np.asarray(result.nuisance.regimes.values, dtype=float)
    for column, arm in enumerate(ARMS, start=1):
        if not np.all(values[:, arm, column] == 1.0):
            raise base.HarnessMismatch(f"column {column} of the regime set is not Static({arm})")
    names = tuple(result.estimates)
    if names != tuple(TAGS.values()):
        raise base.HarnessMismatch(f"the fit reports {names}, not {tuple(TAGS.values())}")


def same_partition(first: Any, second: Any) -> bool:
    """Whether two fits drew the same validation folds and used the same rule on every row."""
    folds = [np.asarray(test) for _, test in first.nuisance.folds]
    others = [np.asarray(test) for _, test in second.nuisance.folds]
    return (
        len(folds) == len(others)
        and all(np.array_equal(a, b) for a, b in zip(folds, others, strict=True))
        and bool(np.array_equal(base.rule_of(first), base.rule_of(second)))
    )


@dataclass(frozen=True)
class Measured:
    """The fits of one draw under each requested report, with their shared refit.

    Parameters
    ----------
    results : Mapping
        Each report of :data:`REPORTS` that was fitted to its result.
    refit : Harness
        The fold rules' moments, refit once: every report drew the same folds and rules.
    solver_warnings : int
        The solver warnings of every fit and refit of the draw.
    """

    results: Mapping[str, Any]
    refit: Harness
    solver_warnings: int


def measured(
    frame: pd.DataFrame,
    law: str,
    learner: Any,
    fold_seed: int,
    reports: Sequence[str] = tuple(REPORTS),
) -> Measured:
    """Fit one draw beside ``Static(1)`` and ``Static(0)`` under each report, and refit its rules.

    Every report uses the same fold seed, so the same folds and the same fold rules, which is
    checked. Rule L4 stops the run on a non-finite estimate or SE and on a harness mismatch:
    each raises an error that the replication loops do not record.
    """
    with base.solver_warning_count() as count:
        results = {
            report: fit(frame, learner, fold_seed, ARMS, cv_evaluation=REPORTS[report])
            for report in reports
        }
        first = results[reports[0]]
        refit = harness(first, learner, law)
    for report, result in results.items():
        check_known_columns(result)
        if not same_partition(first, result):
            raise base.HarnessMismatch(f"the {report} fit drew other folds or rules")
        for name, estimate in result.estimates.items():
            if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
                raise RuntimeError(f"a non-finite estimate or SE of {name} (rule L4): {estimate}")
    return Measured(results=results, refit=refit, solver_warnings=count[0])


def untargeted(result: Any, tag: str) -> float:
    """The fit's parameter ``tag`` without its fluctuation: the initial plug-ins, fold-weighted.

    Each fold's plug-in of a regime is the mean of ``Qbar_v(r(W_i), W_i)`` over its validation
    rows, read off the out-of-fold outcome regression at the regime's arm: the fit's own rule for
    the learned-rule value, ``a`` for ``Static(a)``. The folds are weighted by the report's fold
    weights, and a contrast is the difference of the two.
    """
    nuisance = result.nuisance
    control, treated = nuisance.outcome.levels
    scaler = nuisance.scaler
    q0 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[control]), dtype=float)
    q1 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[treated]), dtype=float)
    weights = report_weights(result)

    def plug_in(values: np.ndarray) -> float:
        return float(
            np.dot(weights, [np.mean(values[np.asarray(test)]) for _, test in nuisance.folds])
        )

    learned = plug_in(np.where(base.rule_of(result), q1, q0))
    arm = TAG_ARMS[tag]
    if arm is None:
        value = learned
    else:
        known = plug_in(q1 if arm else q0)
        value = known if tag.startswith("known") else learned - known
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate of {tag} (rule L4): {value}")
    return value


def contrast_parts(result: Any, tag: str) -> tuple[float, float]:
    """The SEs of the learned-rule value and of the known regime a contrast ``tag`` reads."""
    arm = TAG_ARMS[tag]
    if arm is None or not tag.startswith("contrast"):
        raise ValueError(f"{tag} is not a contrast")
    learned = float(result.estimates[TAGS["value"]].std_error)
    known = float(result.estimates[TAGS[f"known_{arm}"]].std_error)
    return learned, known


def correlation_one_se(result: Any, tag: str) -> float:
    """``|se_L - se_K|``: the contrast's SE if the two estimates had correlation one."""
    learned, known = contrast_parts(result, tag)
    return abs(learned - known)


def independent_se(result: Any, tag: str) -> float:
    """``sqrt(se_L^2 + se_K^2)``: the contrast's SE if the two estimates were independent."""
    learned, known = contrast_parts(result, tag)
    return float(np.hypot(learned, known))

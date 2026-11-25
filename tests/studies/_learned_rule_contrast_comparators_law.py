r"""The truth of a learned rule beside stochastic regimes and the natural course (T58).

The ``learned-rule-contrast-comparators`` study
(:mod:`tests.studies.learned_rule_contrast_comparators`) fits each RM30 draw twice on one fold
seed, under both reports:

- **fit C (combined)**: ``TMLE(learned_rule=LearnedRule(), interventions=[Static(0), coin,
  offer, NaturalCourse()])``, the engine form of
  ``LearnedRuleContrast(against=(Static(0), coin, offer, NaturalCourse()))``. ``coin`` assigns
  treatment with probability one half to everyone, and ``offer`` with probability one half
  where ``W1 > 0`` and never elsewhere. ``Static(0)`` anchors the fit to T44;
* **fit N (natural course alone)**: ``TMLE(learned_rule=LearnedRule(),
  interventions=[NaturalCourse()])``, the engine form of
  ``LearnedRuleContrast(against=(NaturalCourse(),))``.

Every comparator's value is fixed: :math:`\Psi_K = E_0 \sum_a g^\star(a \mid W) \bar Q_0(a, W)`
for a known regime and :math:`\mu_0 = E_0 Y` for the natural course, each by the trapezoid rule
on the RM30 grid of ``W1`` for each ``W2``. The learned value is data-adaptive, and
:func:`harness` refits each fold's rule with :func:`tests.studies._learned_rule_law.fold_rule`,
as :mod:`tests.studies._learned_rule_contrast_law` does for T44.

Each curve is written, given ``(A, W)``, as ``c(A, W) (Y - Qbar0(A, W)) + m(A, W)``:

- a rule ``d``: ``c = 1{A = d}/g0(A | W)`` and ``m = Qbar0(d(W), W)``;
- a known regime with density ``g*``: ``c = g*(A | W)/g0(A | W)`` and
  ``m = sum_a g*(a | W) Qbar0(a, W)``;
- the natural course: ``c = 1`` and ``m = Qbar0(A, W)``.

A contrast's ``c`` and ``m`` are the differences, so its variance is
``E sum_a g0(a | W) {c^2 Qbar0 (1 - Qbar0) + m^2} - (E sum_a g0(a | W) m)^2``
(:func:`curve_variance`). The same quadrature gives each parameter's efficiency bound at the
limit rule and each fold's oracle SE.

``offer`` jumps at ``W1 = 0``, a grid point. The quadrature takes the mean of its two sides
there, which keeps the trapezoid rule second order (:func:`offer_grid`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule, NaturalCourse, Static, Stochastic
from tests.studies import _learned_rule_law as base

#: The two reports of a learned-rule fit the study reads, by their ``cv_evaluation``.
REPORTS = {"fold_evaluated": True, "stacked": False}

#: The two fits of a draw.
FITS = ("combined", "natural_course")

#: The label the natural course reports under.
NATURAL_COURSE = "natural course"

#: The comparators of each fit: ``("static", 0)``, ``("stochastic", name)`` or
#: ``("natural", None)``.
COMPARATORS: dict[str, tuple[tuple[str, Any], ...]] = {
    "combined": (("static", 0), ("stochastic", "coin"), ("stochastic", "offer"), ("natural", None)),
    "natural_course": (("natural", None),),
}

#: Each tag to its fit and its name in ``result.estimates``. Fit C reports nine parameters
#: and fit N three; fit N's ``ey_natural_course`` is fit C's number (it reads no fluctuation),
#: which :func:`measured` checks, so it carries no tag of its own.
TAGS: dict[str, tuple[str, str]] = {
    "value": ("combined", "ey_learned_rule[learned rule]"),
    "known_0": ("combined", "ey_known_regime[always 0]"),
    "coin": ("combined", "ey_known_regime[coin]"),
    "offer": ("combined", "ey_known_regime[offer]"),
    "natural": ("combined", f"ey_natural_course[{NATURAL_COURSE}]"),
    "contrast_0": ("combined", "ate_learned_rule[learned rule vs always 0]"),
    "contrast_coin": ("combined", "ate_learned_rule[learned rule vs coin]"),
    "contrast_offer": ("combined", "ate_learned_rule[learned rule vs offer]"),
    "contrast_natural": ("combined", f"ate_learned_rule[learned rule vs {NATURAL_COURSE}]"),
    "alone_value": ("natural_course", "ey_learned_rule[learned rule]"),
    "alone_contrast_natural": (
        "natural_course",
        f"ate_learned_rule[learned rule vs {NATURAL_COURSE}]",
    ),
}

#: The comparator each tag reads: ``None`` for a learned value, else the comparator's key.
TAG_COMPARATOR: dict[str, str | None] = {
    "value": None,
    "known_0": "static_0",
    "coin": "coin",
    "offer": "offer",
    "natural": "natural",
    "contrast_0": "static_0",
    "contrast_coin": "coin",
    "contrast_offer": "offer",
    "contrast_natural": "natural",
    "alone_value": None,
    "alone_contrast_natural": "natural",
}

#: The contrast tags, rule minus comparator.
CONTRAST_TAGS = tuple(tag for tag in TAGS if "contrast" in tag)

#: The value tags of a learned rule.
VALUE_TAGS = ("value", "alone_value")

#: The row label of each tag: its reported name, prefixed for fit N, whose names repeat fit C's.
LABELS = {
    tag: name if fit == "combined" else f"natural course alone: {name}"
    for tag, (fit, name) in TAGS.items()
}

#: The two laws the study draws, RM30's.
SCENARIOS = ("non_exceptional", "misspecified_limit")

#: The columns each row carries beside the shared schema.
HARNESS_COLUMNS = base.HARNESS_COLUMNS


# --------------------------------------------------------------------- the regimes


def coin_density(w: pd.DataFrame) -> np.ndarray:
    """``g*(. | W)`` of ``coin``: treat with probability one half."""
    p = np.full(len(w), 0.5)
    return np.column_stack([1.0 - p, p])


def offer_density(w: pd.DataFrame) -> np.ndarray:
    """``g*(. | W)`` of ``offer``: treat with probability one half where ``W1 > 0``."""
    p = 0.5 * (np.asarray(w["W1"], dtype=float) > 0.0)
    return np.column_stack([1.0 - p, p])


def regimes(fit: str) -> list[Any]:
    """The ``interventions=`` list of fit ``fit``."""
    out: list[Any] = []
    for kind, key in COMPARATORS[fit]:
        if kind == "static":
            out.append(Static(key))
        elif kind == "stochastic":
            density = coin_density if key == "coin" else offer_density
            out.append(Stochastic(density, name=key, density_kind="known"))
        else:
            out.append(NaturalCourse(name=NATURAL_COURSE))
    return out


def estimator(learner: Any, fold_seed: int, fit: str, *, cv_evaluation: bool = True) -> TMLE:
    """The RM30 estimator of fit ``fit``; every other setting is the constructor's default."""
    return TMLE(
        learned_rule=LearnedRule(),
        interventions=regimes(fit),
        n_folds=base.N_FOLDS,
        cv_evaluation=cv_evaluation,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


def fit_one(
    frame: pd.DataFrame, learner: Any, fold_seed: int, fit: str, *, cv_evaluation: bool = True
) -> Any:
    """Fit :func:`estimator` to one sample and return the single result."""
    return (
        estimator(learner, fold_seed, fit, cv_evaluation=cv_evaluation)
        .fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES))
        .single()
    )


# --------------------------------------------------------------------- the quadrature


def offer_grid(w2: float) -> np.ndarray:
    """``g*(1 | W)`` of ``offer`` on the grid, with the mean of both sides at ``W1 = 0``."""
    del w2
    p = 0.5 * (base.GRID > 0.0)
    p[base.GRID == 0.0] = 0.25
    return p


#: ``g*(1 | W)`` of each known stochastic regime on the grid, for each ``W2``.
STOCHASTIC_GRID: dict[str, Callable[[float], np.ndarray]] = {
    "coin": lambda w2: np.full(base.GRID.shape, 0.5),
    "offer": offer_grid,
}


def _mean(values: np.ndarray) -> float:
    """``E`` over ``W1`` on the grid for one ``W2``, times that level's probability."""
    return float(0.5 * np.trapezoid(values, base.GRID) / 2.0)


Parts = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _parts(comparator: str | None, rule: np.ndarray | None, w2: float, law: str) -> Parts:
    """``(c(0), c(1), m(0), m(1))`` of one curve on the grid at ``W2 = w2``."""
    q = (base.qbar0(0.0, base.GRID, w2, law), base.qbar0(1.0, base.GRID, w2, law))
    treated = base.g0(base.GRID, w2)
    g = (1.0 - treated, treated)
    if comparator is None:
        if rule is None:
            raise ValueError("a learned value needs its rule")
        d = np.asarray(rule, dtype=bool)
        q_d = np.where(d, q[1], q[0])
        return (~d) / g[0], d / g[1], q_d, q_d
    if comparator == "natural":
        ones = np.ones_like(base.GRID)
        return ones, ones, q[0], q[1]
    p = np.zeros_like(base.GRID) if comparator == "static_0" else STOCHASTIC_GRID[comparator](w2)
    m = p * q[1] + (1.0 - p) * q[0]
    return (1.0 - p) / g[0], p / g[1], m, m


def curve_variance(tag: str, treat: Mapping[float, np.ndarray] | None, law: str) -> float:
    """The variance of the curve of ``tag`` at the rule ``treat`` and the true law.

    ``treat`` is the learned rule on the grid, as
    :func:`tests.studies._learned_rule_law.rule_value` reads it; a comparator's value reads
    none.
    """
    comparator = TAG_COMPARATOR[tag]
    second = mean = 0.0
    for w2 in base.W2_LEVELS:
        q = (base.qbar0(0.0, base.GRID, w2, law), base.qbar0(1.0, base.GRID, w2, law))
        treated = base.g0(base.GRID, w2)
        g = (1.0 - treated, treated)
        if tag in VALUE_TAGS:
            c0, c1, m0, m1 = _parts(None, None if treat is None else treat[w2], w2, law)
        else:
            c0, c1, m0, m1 = _parts(comparator, None, w2, law)
            if "contrast" in tag:
                if treat is None:
                    raise ValueError(f"{tag} needs the learned rule")
                r0, r1, n0, n1 = _parts(None, treat[w2], w2, law)
                c0, c1, m0, m1 = r0 - c0, r1 - c1, n0 - m0, n1 - m1
        second += _mean(
            g[0] * (c0**2 * q[0] * (1.0 - q[0]) + m0**2)
            + g[1] * (c1**2 * q[1] * (1.0 - q[1]) + m1**2)
        )
        mean += _mean(g[0] * m0 + g[1] * m1)
    return second - mean**2


def comparator_value(comparator: str, law: str) -> float:
    """The fixed value of one comparator: ``Psi_K`` of a known regime, or ``mu0 = E Y``."""
    total = 0.0
    for w2 in base.W2_LEVELS:
        _, _, m0, m1 = _parts(comparator, None, w2, law)
        treated = base.g0(base.GRID, w2)
        total += _mean((1.0 - treated) * m0 + treated * m1)
    return total


def efficiency_bounds(treat: Mapping[float, np.ndarray], law: str) -> dict[str, float]:
    """The SD of ``sqrt(n)`` times each tag's estimate at the limit rule ``treat``."""
    return {tag: float(np.sqrt(curve_variance(tag, treat, law))) for tag in TAGS}


# --------------------------------------------------------------------- the harness


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
            variances[tag].append(curve_variance(tag, treat, law))
        sizes.append(len(test))
    return Harness(
        fold_values=tuple(values),
        fold_variances={tag: tuple(v) for tag, v in variances.items()},
        sizes=tuple(sizes),
        rows_checked=checked,
    )


def report_weights(result: Any) -> np.ndarray:
    """The fold weights of the fit's report, computed from its folds and checked on its record.

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

    A learned value's truth is ``sum_v omega_v Psi_{d_v}(P0)`` with the report's fold weights;
    a comparator's is its fixed value; a contrast's is their difference. The oracle SE is
    ``sqrt(sum_v omega_v^2 Var D_v / n_v)`` of each curve.
    """
    weights = report_weights(result)
    value = float(np.dot(weights, refit.fold_values))
    out: dict[str, tuple[float, float]] = {}
    for tag, comparator in TAG_COMPARATOR.items():
        variances = np.asarray(refit.fold_variances[tag])
        oracle = float(np.sqrt(np.sum(weights**2 * variances / np.asarray(refit.sizes))))
        if comparator is None:
            truth = value
        elif "contrast" in tag:
            truth = value - comparator_value(comparator, law)
        else:
            truth = comparator_value(comparator, law)
        out[tag] = (truth, oracle)
    return out


def check_fit(result: Any, fit: str) -> None:
    """Refuse a fit whose regime columns or reported names are not the declared ones."""
    values = np.asarray(result.nuisance.regimes.values, dtype=float)
    known = [key for kind, key in COMPARATORS[fit] if kind != "natural"]
    if values.shape[2] != 1 + len(known):
        raise base.HarnessMismatch(f"the {fit} fit's regime set has {values.shape[2]} columns")
    for column, key in enumerate(known, start=1):
        if key == 0 and not np.all(values[:, 0, column] == 1.0):
            raise base.HarnessMismatch(f"column {column} of the {fit} fit is not Static(0)")
    names = tuple(result.estimates)
    expected = tuple(name for f, name in TAGS.values() if f == fit)
    if fit == "natural_course":
        expected = (expected[0], TAGS["natural"][1], expected[1])
    if names != expected:
        raise base.HarnessMismatch(f"the {fit} fit reports {names}, not {expected}")


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
    """The fits of one draw, by ``(fit, report)``, with their shared refit.

    Parameters
    ----------
    results : Mapping
        Each fitted ``(fit, report)`` pair to its result.
    refit : Harness
        The fold rules' moments, refit once: every fit drew the same folds and rules.
    solver_warnings : int
        The solver warnings of every fit and refit of the draw.
    """

    results: Mapping[tuple[str, str], Any]
    refit: Harness
    solver_warnings: int

    def result(self, tag: str, report: str) -> Any:
        """The result that reports ``tag`` under ``report``."""
        return self.results[TAGS[tag][0], report]


def measured(
    frame: pd.DataFrame,
    law: str,
    learner: Any,
    fold_seed: int,
    reports: Sequence[str] = tuple(REPORTS),
    fits: Sequence[str] = FITS,
) -> Measured:
    """Fit one draw under each fit and report on one fold seed, and refit its rules once.

    Every fit uses the same fold seed, so the same folds and the same fold rules, which is
    checked, as is the natural course's value being the same number in both fits. Rule L4 stops
    the run on a non-finite estimate or SE and on a harness mismatch.
    """
    with base.solver_warning_count() as count:
        results = {
            (fit, report): fit_one(frame, learner, fold_seed, fit, cv_evaluation=REPORTS[report])
            for fit in fits
            for report in reports
        }
        first = results[fits[0], reports[0]]
        refit = harness(first, learner, law)
    natural = TAGS["natural"][1]
    for (fit, report), result in results.items():
        check_fit(result, fit)
        if not same_partition(first, result):
            raise base.HarnessMismatch(f"the {fit} {report} fit drew other folds or rules")
        for name, estimate in result.estimates.items():
            if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
                raise RuntimeError(f"a non-finite estimate or SE of {name} (rule L4): {estimate}")
        other = results.get(("combined", report))
        if fit == "natural_course" and other is not None:
            here, there = result.estimates[natural], other.estimates[natural]
            if (here.psi, here.std_error) != (there.psi, there.std_error):
                raise base.HarnessMismatch(f"the natural course differs between the {report} fits")
    return Measured(results=results, refit=refit, solver_warnings=count[0])


def _plug_ins(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The out-of-fold ``(Qbar_v(0, W), Qbar_v(1, W))``, ``W1`` and ``Y`` on every row."""
    nuisance = result.nuisance
    control, treated = nuisance.outcome.levels
    scaler = nuisance.scaler
    q0 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[control]), dtype=float)
    q1 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[treated]), dtype=float)
    w1 = np.asarray(result.data.covariates, dtype=float)[:, 0]
    y = np.asarray(result.data.outcome, dtype=float)
    return q0, q1, w1, y


def untargeted(result: Any, tag: str) -> float:
    """The fit's parameter ``tag`` without its fluctuation, fold-weighted by the report.

    A rule's or a known regime's fold plug-in is the mean of ``sum_a g*(a | W) Qbar_v(a, W)``
    over the fold's validation rows; the natural course's is the fold's mean outcome, which has
    no fluctuation. A contrast is the difference.
    """
    q0, q1, w1, y = _plug_ins(result)
    weights = report_weights(result)
    folds = result.nuisance.folds

    def plug_in(values: np.ndarray) -> float:
        return float(np.dot(weights, [np.mean(values[np.asarray(test)]) for _, test in folds]))

    def of(comparator: str) -> float:
        if comparator == "natural":
            return plug_in(y)
        if comparator == "static_0":
            p = np.zeros_like(q0)
        elif comparator == "coin":
            p = np.full_like(q0, 0.5)
        else:
            p = 0.5 * (w1 > 0.0)
        return plug_in(p * q1 + (1.0 - p) * q0)

    learned = plug_in(np.where(base.rule_of(result), q1, q0))
    comparator = TAG_COMPARATOR[tag]
    if comparator is None:
        value = learned
    elif "contrast" in tag:
        value = learned - of(comparator)
    else:
        value = of(comparator)
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate of {tag} (rule L4): {value}")
    return value


def contrast_parts(result: Any, tag: str) -> tuple[float, float]:
    """The SEs of the learned value and of the comparator a contrast ``tag`` reads."""
    if tag not in CONTRAST_TAGS:
        raise ValueError(f"{tag} is not a contrast")
    fit = TAGS[tag][0]
    value = "value" if fit == "combined" else "alone_value"
    comparator = {
        "static_0": "known_0",
        "coin": "coin",
        "offer": "offer",
        "natural": "natural",
    }[str(TAG_COMPARATOR[tag])]
    learned = float(result.estimates[TAGS[value][1]].std_error)
    known = float(result.estimates[TAGS[comparator][1]].std_error)
    return learned, known


def correlation_one_se(result: Any, tag: str) -> float:
    """``|se_L - se_K|``: the contrast's SE if the two estimates had correlation one."""
    learned, known = contrast_parts(result, tag)
    return abs(learned - known)


def independent_se(result: Any, tag: str) -> float:
    """``sqrt(se_L^2 + se_K^2)``: the contrast's SE if the two estimates were independent."""
    learned, known = contrast_parts(result, tag)
    return float(np.hypot(learned, known))

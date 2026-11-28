"""Validate the replay of a learned-rule fit and of a fit of the value of the optimal rule.

A refit relearns the rule inside each training fold, so a perturbed draw would estimate the
value of other rules.  The learned-rule replay keeps the fit's own folds, its repeat seeds and
each row's realized rule (X82 part F, learned rules (o); X11 (o)).  Every cell then evaluates
the value of these fixed rules fold-wise with the ``1/V`` weights of the learned-rule record.
The reading is confounding of the value of these fixed rules, not of the learned-rule
procedure.

The value of the optimal rule moves with the rule, so its replay relearns the rule in every
cell on the fit's own folds and repeat seeds (X11 (f3)), and each cell estimates the optimal
value of its perturbed law.

Both replays cover every alias of their fit: the rule's value, each known regime and natural
course declared beside it and their contrasts (X11 (f4b), (f4c)), each baseline stratum
(X11 (f4a)), and a level of an intermediate, where every rule holds the intermediate at the
fit's level (X11 (j4)).  The known comparators read the baseline covariates alone, which no
cell moves, so each is frozen at the densities the fit stored; the natural course carries no
array.  The natural-course value itself is ``E[Y]`` and is refused (:func:`natural_course_refusal`).
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np

from .._refusal import Refusal, RefusalReason
from ..exceptions import CapabilityError, DataError
from ..interventions import RegimeSet, Static
from ..interventions.base import split_natural_course
from ..interventions.learned import (
    NATURAL_COURSE_TARGET,
    OPTIMAL_CONTRAST_TARGET,
    OPTIMAL_RULE_TARGET,
    LearnedRule,
    LearnedRuleRecord,
    freeze_learned_rule,
    frozen_rule_replay,
    reports_optimal_rule,
)
from ..study import _matches_registered_point_identification
from ..targets import TARGETS
from ..targets.base import parameter_name
from ._simulated_confounding_common import (
    _population_alias,
    check_alias,
    check_only_declared_axis,
    point_study_or_refuse,
    replay_error,
)
from ._simulated_confounding_fixed import _BaselineRows, _checked_regimes, _FrozenRegime

#: The targets of a learned-rule fit, all on the ``learned_rule`` axis.
LEARNED_FAMILY = frozenset(
    {"ey_learned_rule", "ey_known_regime", "ate_learned_rule", NATURAL_COURSE_TARGET}
)

#: The targets of a fit of the value of the optimal rule, all on the ``learned_rule`` axis.
OPTIMAL_FAMILY = frozenset(
    {OPTIMAL_RULE_TARGET, "ey_known_regime", OPTIMAL_CONTRAST_TARGET, NATURAL_COURSE_TARGET}
)

RuleReplay = Literal["frozen", "relearned"]


def replay_family(result: Any) -> frozenset[str]:
    """The targets whose aliases the replay of this learned-rule fit covers.

    Parameters
    ----------
    result : TMLEResult
        A fit on the ``learned_rule`` axis.

    Returns
    -------
    frozenset of str
        :data:`OPTIMAL_FAMILY` for a fit of the value of the optimal rule, else
        :data:`LEARNED_FAMILY`.
    """
    return OPTIMAL_FAMILY if reports_optimal_rule(result) else LEARNED_FAMILY


def natural_course_refusal(result: Any, estimand: str, key: Any) -> Refusal:
    """The refusal of the natural-course alias beside a learned or optimal rule (R2).

    Parameters
    ----------
    result : TMLEResult
        The learned-rule fit.
    estimand : str
        The requested ``ey_natural_course[...]`` alias.
    key : ParameterKey
        Its structured key.

    Returns
    -------
    Refusal
        An ``INVALID_DECLARATION`` record whose alternative names the contrast with the
        natural course, in the same population as the request.
    """
    stem = OPTIMAL_CONTRAST_TARGET if reports_optimal_rule(result) else "ate_learned_rule"
    rule = result.estimator.learned_rule
    contrast = _population_alias(result, key, parameter_name(stem, arm=rule.name, versus=key.value))
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"simulated_confounding refuses {estimand!r}, the value of the natural course beside "
        "the rule: it is E[Y], with no counterfactual treatment dependence for a simulated "
        "common cause to move.",
        alternative=(
            f"Replay the contrast {contrast!r}, which compares the rule's value with the "
            "natural course."
        ),
    )


def _same_comparators(declared: Any, stored: Any) -> bool:
    """Compare comparator declarations by identity, or by value for a ``Static`` arm."""
    left, right = tuple(declared), tuple(stored)
    return len(left) == len(right) and all(
        a is b or (type(a) is Static and type(b) is Static and a == b)
        for a, b in zip(left, right, strict=True)
    )


def _checked_fit(
    result: Any, estimand: str, key: Any, family: frozenset[str], error: str
) -> tuple[LearnedRule, LearnedRuleRecord]:
    """The checks both learned-rule validators share; return the rule and its record.

    The functional's target and the requested alias are each in the fit's family and on the
    ``learned_rule`` axis, and the identification matches the one the registered design
    reconstructs for the functional's own target.  The rule comes first in the functional's
    interventions and the comparators follow in the estimator's order.  Every stored split
    holds the rule's column then each known regime, and fold sizes that match the record.
    The record's level of the intermediate is the fit's and the functional's.
    """
    identified = result.identified_effect
    functional = identified.functional
    estimator = result.estimator
    record = result.extra.get("learned_rule")
    rule = getattr(estimator, "learned_rule", None)
    registered = TARGETS.get(functional.target)
    if (
        key.estimand not in family
        or functional.target not in family
        or registered is None
        or getattr(TARGETS.get(key.estimand), "parameter_axis", None) != "learned_rule"
        or registered.parameter_axis != "learned_rule"
        or functional.axis != "learned_rule"
        or key.axis != "learned_rule"
        or result.config.parameter_axis != "learned_rule"
        or key.estimand not in result.config.estimands
        or not set(estimator.estimands or ()) <= family
        or estimator.reference != functional.reference
        or not _matches_registered_point_identification(
            identified.identification,
            registered.identification,
            identified,
            result.data,
            "learned_rule",
        )
    ):
        raise replay_error(error)
    check_only_declared_axis(result, key, "learned_rule", error, intermediate=True)
    point_study_or_refuse(result, error)
    if type(rule) is not LearnedRule or type(record) is not LearnedRuleRecord:
        raise replay_error(error)
    known = tuple(estimator.interventions or ())
    natural = getattr(estimator, "natural_course", None)
    declared = tuple(functional.interventions)
    others, declared_natural = split_natural_course(list(declared[1:]))
    known_names = tuple(str(item.name) for item in known)
    natural_name = None if natural is None else str(natural.name)
    comparators = (*known_names, *(() if natural_name is None else (natural_name,)))
    repeats = tuple(result.repeats)
    level = result.intermediate_value
    if (
        not declared
        or declared[0] != rule
        or not _same_comparators(others, known)
        or not (declared_natural is natural or declared_natural == natural)
        or record.name != rule.name
        or record.estimand
        != (OPTIMAL_RULE_TARGET if family is OPTIMAL_FAMILY else "ey_learned_rule")
        or record.known_regimes != known_names
        or record.natural_course != natural_name
        or record.intermediate_level != level
        or functional.intermediate != level
        or not repeats
        or result.n_repeats != len(repeats)
        or estimator.repeats != len(repeats)
        or record.n_splits != len(repeats)
        or any(
            repeat.nuisance.regimes is None
            or repeat.nuisance.regimes.names != (rule.name, *known_names)
            or np.asarray(repeat.nuisance.regimes.values).shape
            != (result.data.n, result.data.n_arms, 1 + len(known))
            or tuple(int(np.asarray(test).size) for _, test in repeat.nuisance.folds)
            != split.fold_sizes
            for repeat, split in zip(repeats, record.splits, strict=True)
        )
    ):
        raise replay_error(error)
    # The structured key names the rule, a known regime, or the contrast of the rule with
    # one comparator; the alias is composed from those names and the population.
    if key.estimand in (OPTIMAL_RULE_TARGET, "ey_learned_rule"):
        valid = key.value == rule.name and key.reference is None
    elif key.estimand == "ey_known_regime":
        valid = key.value in known_names and key.reference is None
    else:
        valid = key.value == rule.name and key.reference in comparators
    if not valid:
        raise replay_error(error)
    alias = parameter_name(key.estimand, arm=key.value, versus=key.reference)
    check_alias(result, estimand, key, alias, error)
    return rule, record


def _freeze_comparators(result: Any, error: str) -> Any:
    """Return a copy of the estimator whose known comparators hold the fit's densities (R5).

    The known regimes read the baseline covariates alone, which no cell moves, so the arrays
    a refit would compute are the arrays the fit stored.  Freezing them guards against a
    stateful callback, as the fixed-regime replay does.  The declared regimes are evaluated
    again on the stored data, and every stored split's columns ``1`` to ``K`` must equal
    them.  The natural course carries no array and stays as declared.
    """
    estimator = result.estimator
    known = tuple(estimator.interventions or ())
    if not known:
        return estimator
    data = result.data
    try:
        expected = _checked_regimes(RegimeSet.evaluate(known, data), data)
        for repeat in result.repeats:
            stored = np.asarray(repeat.nuisance.regimes.values, dtype=float)[:, :, 1:]
            if not np.array_equal(stored, expected.values):
                raise DataError(
                    "declared regime densities disagree with a stored cross-fitting draw"
                )
    except DataError as cause:
        raise CapabilityError(
            f"{error}: {cause.refusal.condition}",
            reason=RefusalReason.INVALID_DECLARATION,
        ) from cause
    replay = copy.copy(estimator)
    baseline = _BaselineRows.from_data(data)
    replay.interventions = tuple(
        _FrozenRegime(
            name,
            expected.values[:, :, index].copy(),
            baseline,
            density_kind=known[index].density_kind,
        )
        for index, name in enumerate(expected.names)
    )
    return replay


def validate_learned_rule_replay(
    result: Any, estimand: str, key: Any
) -> tuple[Any, tuple[Any], RuleReplay]:
    """Check a learned-rule fit and return its frozen-rule replay and folds.

    Parameters
    ----------
    result : TMLEResult
        The learned-rule fit.
    estimand : str
        The requested alias: the rule's value, a known regime beside it, or a contrast,
        marginal or in a baseline stratum, with or without a level of an intermediate.
    key : ParameterKey
        The structured key of the alias.

    Returns
    -------
    tuple
        The replay estimator, which carries a :class:`FrozenLearnedRule` and the frozen
        known comparators, the one realized outer split that every cell refits on, and
        ``"frozen"``.

    Raises
    ------
    CapabilityError
        If the stored provenance of the fit disagrees with itself.
    """
    error = "simulated_confounding found inconsistent learned-rule parameter metadata"
    _checked_fit(result, estimand, key, LEARNED_FAMILY, error)
    if result.n_repeats != 1:
        raise replay_error(error)
    values = np.asarray(result.nuisance.regimes.values, dtype=float)[:, :, 0]
    # One-hot over the arms: each row's realized rule assigns exactly one arm.
    if not np.array_equal(values.sum(axis=1), np.ones(result.data.n)) or not (
        np.all((values == 0.0) | (values == 1.0))
    ):
        raise replay_error(error)
    replay = frozen_rule_replay(_freeze_comparators(result, error), freeze_learned_rule(result))
    return replay, (result.nuisance.folds,), "frozen"


def validate_optimal_rule_replay(
    result: Any, estimand: str, key: Any
) -> tuple[Any, tuple[Any, ...], RuleReplay]:
    """Check a fit of the value of the optimal rule and return its relearning replay.

    Every cell refits the stored estimator, which carries the declared
    :class:`~cleverly.interventions.LearnedRule`, so the cell relearns the rule on the
    perturbed table.  The cell keeps the fit's folds, one stored split per repeat, and the
    fit's repeat seeds, so the anchor cell reproduces the fit and a nonzero cell differs from it
    only through the perturbed table and the rules that table teaches (X11 (f3)).  The known
    comparators are frozen at the densities the fit stored.

    Parameters
    ----------
    result : TMLEResult
        The fit of the value of the optimal rule.
    estimand : str
        The requested alias: the optimal value, a known regime beside it, or a contrast,
        marginal or in a baseline stratum, with or without a level of an intermediate.
    key : ParameterKey
        The structured key of the alias.

    Returns
    -------
    tuple
        The replay estimator, the outer split of every repeat, and ``"relearned"``.

    Raises
    ------
    CapabilityError
        If the stored provenance of the fit disagrees with itself.
    """
    error = "simulated_confounding found inconsistent optimal-rule parameter metadata"
    _checked_fit(result, estimand, key, OPTIMAL_FAMILY, error)
    replay = _freeze_comparators(result, error)
    return replay, tuple(repeat.nuisance.folds for repeat in result.repeats), "relearned"

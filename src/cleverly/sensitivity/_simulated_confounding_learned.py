"""Validate the replay of a learned-rule fit and of a fit of the value of the optimal rule.

A refit relearns the rule inside each training fold, so a perturbed draw would estimate the
value of other rules.  The learned-rule replay keeps the fit's own folds and each row's
realized rule (X82 part F, learned rules (o); X11 (o)).  Every cell then evaluates the value
of these fixed rules fold-wise with the ``1/V`` weights of the learned-rule record.  The
reading is confounding of the value of these fixed rules, not of the learned-rule procedure.

The value of the optimal rule moves with the rule, so its replay relearns the rule in every
cell on the fit's own folds and repeat seeds (X11 (f3)), and each cell estimates the optimal
value of its perturbed law.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._refusal import RefusalReason
from ..exceptions import CapabilityError
from ..interventions.learned import (
    OPTIMAL_RULE_TARGET,
    LearnedRule,
    LearnedRuleRecord,
    freeze_learned_rule,
    frozen_rule_replay,
)
from ..targets.base import parameter_name
from ._simulated_confounding_common import (
    check_alias,
    check_only_declared_axis,
    check_registered_target,
    check_replay_declaration,
    point_study_or_refuse,
)

LEARNED_RULE_TARGETS = frozenset({"ey_learned_rule"})

#: The target whose replay relearns the rule in every cell (X11 (f3)).  It is not in
#: :data:`LEARNED_RULE_TARGETS`, which routes to the frozen replay.
OPTIMAL_RULE_REPLAY_TARGETS = frozenset({OPTIMAL_RULE_TARGET})


def validate_learned_rule_replay(result: Any, estimand: str, key: Any) -> tuple[Any, tuple[Any]]:
    """Check a learned-rule fit and return its frozen-rule replay and folds.

    Parameters
    ----------
    result : TMLEResult
        The learned-rule fit.
    estimand : str
        The requested alias.
    key : ParameterKey
        The structured key of the alias.

    Returns
    -------
    tuple
        The replay estimator, which carries a :class:`FrozenLearnedRule`, and the one
        realized outer split that every cell refits on.

    Raises
    ------
    CapabilityError
        If the stored provenance of the fit disagrees with itself.
    """
    error = "simulated_confounding found inconsistent learned-rule parameter metadata"
    estimator = result.estimator
    functional = result.identified_effect.functional
    record = result.extra.get("learned_rule")
    rule = getattr(estimator, "learned_rule", None)
    check_registered_target(result, key, "learned_rule", error)
    check_replay_declaration(result, key, error)
    check_only_declared_axis(result, key, "learned_rule", error)
    point_study_or_refuse(result, error)
    nuisance = result.nuisance
    regimes = nuisance.regimes
    if (
        type(rule) is not LearnedRule
        or type(record) is not LearnedRuleRecord
        or tuple(functional.interventions) != (rule,)
        or result.n_repeats != 1
        or len(result.nuisances) != 1
        or regimes is None
        or regimes.names != (rule.name,)
        or np.asarray(regimes.values).shape != (result.data.n, result.data.n_arms, 1)
        or key.value != rule.name
        or record.name != rule.name
        or record.n_splits != 1
        or tuple(int(np.asarray(test).size) for _, test in nuisance.folds)
        != record.splits[0].fold_sizes
    ):
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
    values = np.asarray(regimes.values, dtype=float)
    # One-hot over the arms: each row's realized rule assigns exactly one arm.
    if not np.array_equal(values[:, :, 0].sum(axis=1), np.ones(result.data.n)) or not (
        np.all((values == 0.0) | (values == 1.0))
    ):
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
    # A stratum alias is composed from the marginal alias of the rule's value.
    check_alias(result, estimand, key, parameter_name(key.estimand, arm=rule.name), error)
    return frozen_rule_replay(estimator, freeze_learned_rule(result)), (nuisance.folds,)


def validate_optimal_rule_replay(
    result: Any, estimand: str, key: Any
) -> tuple[Any, tuple[Any, ...]]:
    """Check a fit of the value of the optimal rule and return its relearning replay.

    Every cell refits the stored estimator, which carries the declared
    :class:`~cleverly.interventions.LearnedRule`, so the cell relearns the rule on the
    perturbed table.  The cell keeps the fit's folds, one stored split per repeat, and the
    fit's repeat seeds, so the anchor cell reproduces the fit and a nonzero cell differs from it
    only through the perturbed table and the rules that table teaches (X11 (f3)).

    Parameters
    ----------
    result : TMLEResult
        The fit of the value of the optimal rule.
    estimand : str
        The requested alias.
    key : ParameterKey
        The structured key of the alias.

    Returns
    -------
    tuple
        The stored estimator, and the outer split of every repeat.

    Raises
    ------
    CapabilityError
        If the stored provenance of the fit disagrees with itself.
    """
    error = "simulated_confounding found inconsistent optimal-rule parameter metadata"
    estimator = result.estimator
    functional = result.identified_effect.functional
    record = result.extra.get("learned_rule")
    rule = getattr(estimator, "learned_rule", None)
    check_registered_target(result, key, "learned_rule", error)
    check_replay_declaration(result, key, error)
    check_only_declared_axis(result, key, "learned_rule", error)
    point_study_or_refuse(result, error)
    repeats = tuple(result.repeats)
    regimes = result.nuisance.regimes
    if (
        type(rule) is not LearnedRule
        or type(record) is not LearnedRuleRecord
        or record.estimand != OPTIMAL_RULE_TARGET
        or tuple(functional.interventions) != (rule,)
        or result.n_repeats != len(repeats)
        or estimator.repeats != len(repeats)
        or record.n_splits != len(repeats)
        or regimes is None
        or regimes.names != (rule.name,)
        or key.value != rule.name
        or record.name != rule.name
        or any(
            tuple(int(np.asarray(test).size) for _, test in repeat.nuisance.folds)
            != split.fold_sizes
            for repeat, split in zip(repeats, record.splits, strict=True)
        )
    ):
        raise CapabilityError(error, reason=RefusalReason.INVALID_DECLARATION)
    check_alias(result, estimand, key, parameter_name(key.estimand, arm=rule.name), error)
    return estimator, tuple(repeat.nuisance.folds for repeat in repeats)

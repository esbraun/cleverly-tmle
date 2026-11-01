"""Validate and freeze the realized fold rules of a learned-rule fit for a replay.

A refit relearns the rule inside each training fold, so a perturbed draw would estimate the
value of other rules.  The replay keeps the fit's own folds and each row's realized rule
(X82 part F, learned rules (o); X11 (o)).  Every cell then evaluates the value of these
fixed rules fold-wise with the ``1/V`` weights of the learned-rule record.  The reading is
confounding of the value of these fixed rules, not of the learned-rule procedure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._refusal import RefusalReason
from ..exceptions import CapabilityError
from ..interventions.learned import (
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

"""Interventions and the regimes a fit targets.

See :mod:`cleverly.interventions.base` for what a regime is and which kinds are
deliberately refused, :mod:`cleverly.interventions.incremental` for the one whose
:math:`g^\\star` is built out of the estimated mechanism, and
:mod:`cleverly.interventions.support` for the overlap diagnostics a regime needs that an
arm-level positivity report does not give. :mod:`cleverly.interventions.learned` holds the
rule learned inside each training fold, whose fold-average value is a target of its own.
"""

from __future__ import annotations

from .base import (
    Intervention,
    RegimeSet,
    Rule,
    Static,
    Stochastic,
    as_interventions,
    refuse_mixed_interventions,
)
from .incremental import (
    Incremental,
    IncrementalSupport,
    IPSISet,
    check_incremental_support,
)
from .learned import LearnedRule, LearnedRuleRecord
from .policy import (
    POLICY_TYPES,
    ModifiedPolicy,
    Piece,
    Piecewise,
    Policy,
    PolicySet,
    PolicySupport,
    Randomizer,
    RiskRatioTilt,
    Scale,
    Shift,
    check_policy_support,
)
from .support import RegimeSupport, SupportReport, check_support

__all__ = [
    "POLICY_TYPES",
    "IPSISet",
    "Incremental",
    "IncrementalSupport",
    "Intervention",
    "LearnedRule",
    "LearnedRuleRecord",
    "ModifiedPolicy",
    "Piece",
    "Piecewise",
    "Policy",
    "PolicySet",
    "PolicySupport",
    "Randomizer",
    "RegimeSet",
    "RegimeSupport",
    "RiskRatioTilt",
    "Rule",
    "Scale",
    "Shift",
    "Static",
    "Stochastic",
    "SupportReport",
    "as_interventions",
    "check_incremental_support",
    "check_policy_support",
    "check_support",
    "refuse_mixed_interventions",
]

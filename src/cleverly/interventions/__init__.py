"""Interventions and the regimes a fit targets.

See :mod:`cleverly.interventions.base` for what a regime is and which kinds are
deliberately refused, and :mod:`cleverly.interventions.support` for the overlap
diagnostics a regime needs that an arm-level positivity report does not give.
"""

from __future__ import annotations

from .base import (
    Intervention,
    RegimeSet,
    Rule,
    Static,
    Stochastic,
    as_interventions,
    refuse_unsupported,
)
from .support import RegimeSupport, SupportReport, check_support

__all__ = [
    "Intervention",
    "RegimeSet",
    "RegimeSupport",
    "Rule",
    "Static",
    "Stochastic",
    "SupportReport",
    "as_interventions",
    "check_support",
    "refuse_unsupported",
]

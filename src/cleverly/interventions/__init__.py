"""Interventions and the regimes a fit targets.

See :mod:`cleverly.interventions.base` for what a regime is and which kinds are
deliberately refused, :mod:`cleverly.interventions.incremental` for the one whose
:math:`g^\\star` is built out of the estimated mechanism, and
:mod:`cleverly.interventions.support` for the overlap diagnostics a regime needs that an
arm-level positivity report does not give.
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
from .incremental import (
    Incremental,
    IncrementalSupport,
    IPSISet,
    check_incremental_support,
)
from .shift import Shift, ShiftSet, ShiftSupport, check_shift_support
from .support import RegimeSupport, SupportReport, check_support

__all__ = [
    "IPSISet",
    "Incremental",
    "IncrementalSupport",
    "Intervention",
    "RegimeSet",
    "RegimeSupport",
    "Rule",
    "Shift",
    "ShiftSet",
    "ShiftSupport",
    "Static",
    "Stochastic",
    "SupportReport",
    "as_interventions",
    "check_incremental_support",
    "check_shift_support",
    "check_support",
    "refuse_unsupported",
]

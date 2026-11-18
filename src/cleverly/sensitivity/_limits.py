"""The limits a sensitivity bound reads from the interval rule of the estimate it bounds.

:func:`~cleverly.inference.interval_reference` gives the rule at the bound's end point and
standard error. Two cases need a reading it leaves undefined, and both bounds take them here,
once: an estimate whose status withholds its interval still reports the bound's plug-in
limits as a guarded diagnostic, and a bound of a binary mean on or past the edge of ``(0, 1)``
has no logit.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .._inference_status import UNREPORTED_INTERVAL_STATUSES
from ..inference.influence import IntervalReference, ParameterEstimate, interval_reference

__all__ = ["bound_limits"]


def bound_limits(estimate: ParameterEstimate, **options: Any) -> IntervalReference:
    """The limits of a bound end under ``estimate``'s interval rule.

    ``options`` are those of :func:`~cleverly.inference.interval_reference`. An estimate
    whose status withholds its interval (``"constant_arm_cell_plugin"``,
    ``"out_of_range_plugin"``) gives no interval to read, and its bound's limits are a
    diagnostic that the bound's accessors refuse: they take the identity scale with the
    estimate's own reference, as the plug-in limits did before the bounds read the rule. A
    bound of a binary mean on or past the edge of ``(0, 1)`` has no logit; its limit takes
    the identity scale, which lies past that edge and so states no less than the range.
    """
    if estimate.inference in UNREPORTED_INTERVAL_STATUSES:
        estimate = replace(
            estimate,
            inference="influence_curve",
            bounds=None,
            outcome_family=None,
            components=None,
        )
    read = interval_reference(estimate, **options)
    if estimate.interval_scale == "range_logit" and not (
        np.isfinite(read.lower) and np.isfinite(read.upper)
    ):
        read = interval_reference(replace(estimate, bounds=None), **options)
    return read

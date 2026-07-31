"""Longitudinal TMLE: time-varying treatment, time-varying confounding, censoring.

The container and the regimens it is indexed by live here; the estimator that consumes
them follows.  See :mod:`cleverly.longitudinal.data` for the node ordering the container
enforces and the readings of the data it refuses rather than guesses at.
"""

from __future__ import annotations

from .data import LongitudinalData
from .regimen import Regimen, resolve_regimens

__all__ = ["LongitudinalData", "Regimen", "resolve_regimens"]

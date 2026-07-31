"""Longitudinal TMLE: time-varying treatment, time-varying confounding, censoring.

>>> from cleverly.longitudinal import LTMLE
>>> from cleverly.datasets import make_longitudinal
>>> frame, truth = make_longitudinal(n=2000, seed=0)
>>> result = LTMLE({"always": 1, "never": 0}, n_folds=5, random_state=0).fit(
...     frame,
...     outcome="Y",
...     treatment=["A1", "A2"],
...     baseline=["W1", "W2"],
...     time_varying=[[], ["L2"]],
...     censoring=["C1", "C2"],
... )
>>> print(result.summary())                                        # doctest: +SKIP

See :mod:`cleverly.longitudinal.sequential` for the recursion and the influence
function, and :mod:`cleverly.longitudinal.estimator` for the assumptions it identifies
the parameter under.
"""

from __future__ import annotations

from .data import LongitudinalData
from .estimator import LTMLE, LongitudinalConfig, LongitudinalResult, ltmle
from .regimen import Regimen, resolve_regimens
from .sequential import (
    LongitudinalError,
    Mechanism,
    RegimenFit,
    SequentialStep,
    fit_mechanism,
    fit_regimen,
)

__all__ = [
    "LTMLE",
    "LongitudinalConfig",
    "LongitudinalData",
    "LongitudinalError",
    "LongitudinalResult",
    "Mechanism",
    "Regimen",
    "RegimenFit",
    "SequentialStep",
    "fit_mechanism",
    "fit_regimen",
    "ltmle",
    "resolve_regimens",
]

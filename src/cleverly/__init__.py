"""cleverly: targeted maximum likelihood estimation for Python.

Quickstart
----------
>>> from cleverly import TMLE
>>> from cleverly.datasets import make_nonlinear_ate
>>> frame, truth = make_nonlinear_ate(n=1000, seed=0)
>>> result = TMLE(random_state=0).fit(frame, outcome="Y", treatment="A").single()
>>> print(result.summary())                                        # doctest: +SKIP

The estimator takes pandas or polars dataframes interchangeably and returns results
in whichever backend it was given.  Every result carries its influence curves, so
:attr:`~cleverly.estimators.base.TMLEResult.sensitivity` and
:attr:`~cleverly.estimators.base.TMLEResult.validation` need no refitting.
"""

from __future__ import annotations

from ._version import __version__
from .data import CausalData
from .estimators import (
    ALL_ESTIMANDS,
    CTMLE,
    DEFAULT_ESTIMANDS,
    TMLE,
    CTMLESelection,
    CVTargeting,
    TMLEResult,
    TMLEResultSet,
    tmle,
)
from .estimators.serialize import load
from .exceptions import (
    CleverlyError,
    ConvergenceWarning,
    DataError,
    NotFittedError,
    PositivityWarning,
    WeightingWarning,
)
from .inference import ParameterEstimate
from .learners import SuperLearner
from .longitudinal import (
    LTMLE,
    LongitudinalData,
    LongitudinalError,
    LongitudinalResult,
    Regimen,
    ltmle,
)
from .provenance import Provenance
from .targets import TARGETS, Identification, Target, register

__all__ = [
    "ALL_ESTIMANDS",
    "CTMLE",
    "DEFAULT_ESTIMANDS",
    "LTMLE",
    "TARGETS",
    "TMLE",
    "CTMLESelection",
    "CVTargeting",
    "CausalData",
    "CleverlyError",
    "ConvergenceWarning",
    "DataError",
    "Identification",
    "LongitudinalData",
    "LongitudinalError",
    "LongitudinalResult",
    "NotFittedError",
    "ParameterEstimate",
    "PositivityWarning",
    "Provenance",
    "Regimen",
    "SuperLearner",
    "TMLEResult",
    "TMLEResultSet",
    "Target",
    "WeightingWarning",
    "__version__",
    "load",
    "ltmle",
    "register",
    "tmle",
]

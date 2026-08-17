"""cleverly: targeted maximum likelihood estimation for Python.

Quickstart
----------
>>> from cleverly import ATE, CausalStudy, PointTreatment
>>> from cleverly.datasets import make_nonlinear_ate
>>> frame, truth = make_nonlinear_ate(n=1000, seed=0)
>>> study = CausalStudy(
...     frame,
...     design=PointTreatment(
...         outcome="Y",
...         treatment="A",
...         adjustment=["W1", "W2", "W3", "W4"],
...     ),
... )
>>> result = study.identify(ATE()).estimate(random_state=0)
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
    DRTMLE,
    TMLE,
    CTMLEOutcomeAdaptiveFit,
    CTMLEPreorder,
    CTMLESelection,
    CTMLEStrategy,
    CVTargeting,
    TMLEResult,
    TMLEResultSet,
    tmle,
)
from .estimators.serialize import load
from .exceptions import (
    CapabilityError,
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
from .methods import (
    CrossFitting,
    Inference,
    MethodAvailability,
    ModelSpec,
    Runtime,
    Targeting,
    TMLEMethod,
)
from .provenance import Provenance
from .study import ATE, CausalStudy, IdentifiedEffect, ParameterKey, PointTreatment
from .targets import TARGETS, Identification, Target, register
from .variable_importance import (
    VariableImportanceEntry,
    VariableImportanceResult,
    variable_importance,
)

__all__ = [
    "ALL_ESTIMANDS",
    "ATE",
    "CTMLE",
    "DEFAULT_ESTIMANDS",
    "DRTMLE",
    "LTMLE",
    "TARGETS",
    "TMLE",
    "CTMLEOutcomeAdaptiveFit",
    "CTMLEPreorder",
    "CTMLESelection",
    "CTMLEStrategy",
    "CVTargeting",
    "CapabilityError",
    "CausalData",
    "CausalStudy",
    "CleverlyError",
    "ConvergenceWarning",
    "CrossFitting",
    "DataError",
    "Identification",
    "IdentifiedEffect",
    "Inference",
    "LongitudinalData",
    "LongitudinalError",
    "LongitudinalResult",
    "MethodAvailability",
    "ModelSpec",
    "NotFittedError",
    "ParameterEstimate",
    "ParameterKey",
    "PointTreatment",
    "PositivityWarning",
    "Provenance",
    "Regimen",
    "Runtime",
    "SuperLearner",
    "TMLEMethod",
    "TMLEResult",
    "TMLEResultSet",
    "Target",
    "Targeting",
    "VariableImportanceEntry",
    "VariableImportanceResult",
    "WeightingWarning",
    "__version__",
    "load",
    "ltmle",
    "register",
    "tmle",
    "variable_importance",
]

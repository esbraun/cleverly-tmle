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
from .assessment import (
    AssessmentCapability,
    AssessmentStatus,
    DiagnosticReport,
    Replayability,
    ValidationReport,
)
from .estimators.serialize import load
from .exceptions import (
    CapabilityError,
    CleverlyError,
    ConvergenceWarning,
    DataError,
    MethodConfigurationError,
    NotFittedError,
    PositivityWarning,
    WeightingWarning,
)
from .inference import ParameterEstimate
from .learners import SuperLearner
from .methods import (
    CollaborativeTMLEMethod,
    CrossFitting,
    DRTMLEMethod,
    EstimationMethod,
    Inference,
    MethodAvailability,
    ModelSpec,
    Runtime,
    Targeting,
    TMLEMethod,
)
from .provenance import Provenance
from .study import (
    ATC,
    ATE,
    ATT,
    BackdoorMeanContrast,
    CausalResult,
    CausalStudy,
    ControlledDirectEffect,
    CounterfactualMean,
    Estimand,
    ExplicitAdjustmentProvider,
    IdentificationProvider,
    IdentifiedEffect,
    IncrementalEffect,
    IncrementalMean,
    LongitudinalTreatment,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    MSMProjection,
    NaturalCourseMean,
    OddsRatio,
    ParameterKey,
    PointTreatment,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
    RegimeContrast,
    RegimeMean,
    RiskRatio,
)
from .variable_importance import (
    VariableImportanceEntry,
    VariableImportanceResult,
    variable_importance,
)

__all__ = [
    "ATC",
    "ATE",
    "ATT",
    "AssessmentCapability",
    "AssessmentStatus",
    "BackdoorMeanContrast",
    "CapabilityError",
    "CausalResult",
    "CausalStudy",
    "CleverlyError",
    "CollaborativeTMLEMethod",
    "ControlledDirectEffect",
    "ConvergenceWarning",
    "CounterfactualMean",
    "CrossFitting",
    "DRTMLEMethod",
    "DataError",
    "DiagnosticReport",
    "Estimand",
    "EstimationMethod",
    "ExplicitAdjustmentProvider",
    "IdentificationProvider",
    "IdentifiedEffect",
    "IncrementalEffect",
    "IncrementalMean",
    "Inference",
    "LongitudinalTreatment",
    "MSMProjection",
    "MethodAvailability",
    "MethodConfigurationError",
    "ModelSpec",
    "ModifiedTreatmentPolicy",
    "ModifiedTreatmentPolicyEffect",
    "NaturalCourseMean",
    "NotFittedError",
    "OddsRatio",
    "ParameterEstimate",
    "ParameterKey",
    "PointTreatment",
    "PopulationAttributableFraction",
    "PopulationAttributableRisk",
    "PositivityWarning",
    "Provenance",
    "RegimeContrast",
    "RegimeMean",
    "Replayability",
    "RiskRatio",
    "Runtime",
    "SuperLearner",
    "TMLEMethod",
    "Targeting",
    "ValidationReport",
    "VariableImportanceEntry",
    "VariableImportanceResult",
    "WeightingWarning",
    "__version__",
    "load",
    "variable_importance",
]

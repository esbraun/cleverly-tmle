"""cleverly: targeted maximum likelihood estimation for Python.

Quickstart
----------
>>> from cleverly import ATE, CausalStudy, PointTreatment
>>> from cleverly.datasets import make_nonlinear_ate
>>> from sklearn.linear_model import LinearRegression, LogisticRegression
>>> frame, _ = make_nonlinear_ate(n=200, seed=0)
>>> study = CausalStudy(
...     frame,
...     design=PointTreatment(
...         outcome="Y",
...         treatment="A",
...         adjustment=["W1", "W2", "W3", "W4"],
...     ),
... )
>>> result = study.identify(ATE()).estimate(
...     outcome_learner=LinearRegression(),
...     treatment_learner=LogisticRegression(max_iter=1000),
...     cross_fit=False,
...     random_state=0,
... )
>>> sorted(result.estimates)
['ate']

The estimator takes pandas or polars dataframes interchangeably and returns results
in whichever backend it was given. Every result carries the fitted artifacts used by
its cache-only validation and diagnostic operations. Other assessments declare when
they retarget or refit.
"""

from __future__ import annotations

from ._version import __version__
from .assessment import (
    AssessmentCapability,
    AssessmentReport,
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
    MonteCarloWarning,
    NotFittedError,
    ParameterSpaceWarning,
    PositivityWarning,
    SparseInfluenceWarning,
    VersionMismatchWarning,
    WeightingWarning,
)
from .inference import ParameterEstimate
from .learners import SplitPlan, SuperLearner
from .longitudinal import Repeated
from .methods import (
    AdaptiveIPTWMethod,
    CollaborativeTMLEMethod,
    CrossFitting,
    DRTMLEMethod,
    EstimationMethod,
    GComputationMethod,
    Inference,
    IPSIMethod,
    IPWMethod,
    MethodAvailability,
    ModelSpec,
    ParametricGFormulaMethod,
    Runtime,
    SDRMethod,
    Targeting,
    TMLEMethod,
)
from .protocol import StudyProtocol
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
    LearnedRuleValue,
    LongitudinalTreatment,
    LongPanel,
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
    TimeToEvent,
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
    "AdaptiveIPTWMethod",
    "AssessmentCapability",
    "AssessmentReport",
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
    "GComputationMethod",
    "IPSIMethod",
    "IPWMethod",
    "IdentificationProvider",
    "IdentifiedEffect",
    "IncrementalEffect",
    "IncrementalMean",
    "Inference",
    "LearnedRuleValue",
    "LongPanel",
    "LongitudinalTreatment",
    "MSMProjection",
    "MethodAvailability",
    "MethodConfigurationError",
    "ModelSpec",
    "ModifiedTreatmentPolicy",
    "ModifiedTreatmentPolicyEffect",
    "MonteCarloWarning",
    "NaturalCourseMean",
    "NotFittedError",
    "OddsRatio",
    "ParameterEstimate",
    "ParameterKey",
    "ParameterSpaceWarning",
    "ParametricGFormulaMethod",
    "PointTreatment",
    "PopulationAttributableFraction",
    "PopulationAttributableRisk",
    "PositivityWarning",
    "Provenance",
    "RegimeContrast",
    "RegimeMean",
    "Repeated",
    "Replayability",
    "RiskRatio",
    "Runtime",
    "SDRMethod",
    "SparseInfluenceWarning",
    "SplitPlan",
    "StudyProtocol",
    "SuperLearner",
    "TMLEMethod",
    "Targeting",
    "TimeToEvent",
    "ValidationReport",
    "VariableImportanceEntry",
    "VariableImportanceResult",
    "VersionMismatchWarning",
    "WeightingWarning",
    "__version__",
    "load",
    "variable_importance",
]

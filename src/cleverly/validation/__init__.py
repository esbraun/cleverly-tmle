"""Validation: score diagnostics, nuisance quality, refutation and simulation studies."""

from __future__ import annotations

from .drtmle import (
    IDENTITY_TOLERANCE,
    MARGIN_ACTIVE,
    CorrectionCheck,
    CorrectionRow,
    correction_check,
)
from .nuisance import (
    NuisanceDiagnostics,
    NuisanceModelReport,
    RepeatSpreadRow,
    nuisance_diagnostics,
)
from .outcome_processes import (
    BernoulliAdjustmentOutcome,
    BoundedAdjustmentOutcome,
    LogisticHazardOutcome,
    MultinomialHazardOutcome,
)
from .refute import (
    DEFAULT_OUTCOME_REPLICATES,
    DEFAULT_TESTS,
    BandReading,
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    EmpiricalRefitRecord,
    EquivalenceBands,
    GaussianAdjustmentOutcome,
    GaussianIndependentOutcome,
    GaussianNoise,
    RefutationResult,
    RefutationTest,
    RelativeGaussianNoise,
    refute,
)
from .score import DEFAULT_TOLERANCE, ScoreCheck, ScoreCheckRow, score_check
from .simulation import (
    CoverageStudy,
    DirectoryCheckpoint,
    EstimandSummary,
    ReplicationCheckpoint,
    ReplicationFailure,
    ReplicationRecord,
    StudyResult,
    summarize_replications,
)

__all__ = [
    "DEFAULT_OUTCOME_REPLICATES",
    "DEFAULT_TESTS",
    "DEFAULT_TOLERANCE",
    "IDENTITY_TOLERANCE",
    "MARGIN_ACTIVE",
    "BandReading",
    "BernoulliAdjustmentOutcome",
    "BootstrapMeasurementError",
    "BoundedAdjustmentOutcome",
    "CorrectionCheck",
    "CorrectionRow",
    "CoverageStudy",
    "DirectoryCheckpoint",
    "EmpiricalInclusionRule",
    "EmpiricalRefitRecord",
    "EquivalenceBands",
    "EstimandSummary",
    "GaussianAdjustmentOutcome",
    "GaussianIndependentOutcome",
    "GaussianNoise",
    "LogisticHazardOutcome",
    "MultinomialHazardOutcome",
    "NuisanceDiagnostics",
    "NuisanceModelReport",
    "RefutationResult",
    "RefutationTest",
    "RelativeGaussianNoise",
    "RepeatSpreadRow",
    "ReplicationCheckpoint",
    "ReplicationFailure",
    "ReplicationRecord",
    "ScoreCheck",
    "ScoreCheckRow",
    "StudyResult",
    "correction_check",
    "nuisance_diagnostics",
    "refute",
    "score_check",
    "summarize_replications",
]

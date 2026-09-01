"""Validation: score diagnostics, nuisance quality, refutation and simulation studies."""

from __future__ import annotations

from .drtmle import (
    IDENTITY_TOLERANCE,
    MARGIN_ACTIVE,
    CorrectionCheck,
    CorrectionRow,
    correction_check,
)
from .nuisance import NuisanceDiagnostics, NuisanceModelReport, nuisance_diagnostics
from .refute import (
    DEFAULT_OUTCOME_REPLICATES,
    DEFAULT_TESTS,
    EmpiricalInclusionRule,
    GaussianAdjustmentOutcome,
    GaussianIndependentOutcome,
    GaussianNoise,
    GeneratedOutcomeRecord,
    RefutationResult,
    RefutationTest,
    refute,
)
from .score import DEFAULT_TOLERANCE, ScoreCheck, ScoreCheckRow, score_check
from .simulation import (
    CoverageStudy,
    EstimandSummary,
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
    "CorrectionCheck",
    "CorrectionRow",
    "CoverageStudy",
    "EmpiricalInclusionRule",
    "EstimandSummary",
    "GaussianAdjustmentOutcome",
    "GaussianIndependentOutcome",
    "GaussianNoise",
    "GeneratedOutcomeRecord",
    "NuisanceDiagnostics",
    "NuisanceModelReport",
    "RefutationResult",
    "RefutationTest",
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

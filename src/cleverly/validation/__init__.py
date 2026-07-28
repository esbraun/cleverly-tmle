"""Validation: score diagnostics, nuisance quality, refutation and simulation studies."""

from __future__ import annotations

from .api import ValidationSuite
from .nuisance import NuisanceDiagnostics, NuisanceModelReport, nuisance_diagnostics
from .refute import DEFAULT_TESTS, RefutationResult, RefutationTest, refute
from .score import DEFAULT_TOLERANCE, ScoreCheck, ScoreCheckRow, score_check
from .simulation import CoverageStudy, EstimandSummary, StudyResult

__all__ = [
    "DEFAULT_TESTS",
    "DEFAULT_TOLERANCE",
    "CoverageStudy",
    "EstimandSummary",
    "NuisanceDiagnostics",
    "NuisanceModelReport",
    "RefutationResult",
    "RefutationTest",
    "ScoreCheck",
    "ScoreCheckRow",
    "StudyResult",
    "ValidationSuite",
    "nuisance_diagnostics",
    "refute",
    "score_check",
]

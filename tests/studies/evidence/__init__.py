"""Shared machinery for method-evidence studies.

A *method* evidence study asks the repeated-sampling question the estimand manifest does
not: applied to samples from a known law, does a complete estimator's bias and uncertainty
behave the way its source theory predicts, and -- where a canonical implementation exists --
does this one match it? The Technical appendix's method grid and the dedicated pages under
``docs/technical-reference/method-evidence/`` are the reader-facing side;
:mod:`tests.studies.evidence.registry` is the declaration a study writes to get all of this.

The one rule the whole package is built around: **an accept-decision must be bounded by a
margin declared in advance, never by a test against zero.**  Monte Carlo evidence accumulates
by adding replications, so a verdict has to get easier -- or at worst stay put -- as
replications grow.  A significance test does the reverse: it converges on rejecting every
estimator whose finite-sample remainder is not identically zero, which is all of them, so the
study would go red for the one reason that is not a defect.  See
:func:`~tests.studies.evidence.inference.standardized_bias_verdict` and
:func:`~tests.studies.evidence.performance.independent_performance_tests`.
"""

from __future__ import annotations

from tests.studies.evidence.claims import load, matches, value
from tests.studies.evidence.comparison import equivalence
from tests.studies.evidence.inference import (
    BiasVerdict,
    Interval,
    bootstrap,
    clopper_pearson,
    coverage_for_se_ratio,
    lower_bound,
    percentile_interval,
    se_ratio_for_coverage,
    standardized_bias_verdict,
    student_interval,
    upper_bound,
)
from tests.studies.evidence.manifest import hashes, provenance, write_manifest
from tests.studies.evidence.pairing import Paired, paired_wide
from tests.studies.evidence.performance import independent_performance_tests, summarize
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord, registered
from tests.studies.evidence.schema import (
    REPLICATE_COLUMNS,
    truth_on_inference_scale,
    validate_replicates,
)
from tests.studies.evidence.seeds import replicate_seed

__all__ = [
    "REPLICATE_COLUMNS",
    "ROOT",
    "BiasVerdict",
    "Interval",
    "Margins",
    "Paired",
    "StudyRecord",
    "bootstrap",
    "clopper_pearson",
    "coverage_for_se_ratio",
    "equivalence",
    "hashes",
    "independent_performance_tests",
    "load",
    "lower_bound",
    "matches",
    "paired_wide",
    "percentile_interval",
    "provenance",
    "registered",
    "replicate_seed",
    "se_ratio_for_coverage",
    "standardized_bias_verdict",
    "student_interval",
    "summarize",
    "truth_on_inference_scale",
    "upper_bound",
    "validate_replicates",
    "value",
    "write_manifest",
]

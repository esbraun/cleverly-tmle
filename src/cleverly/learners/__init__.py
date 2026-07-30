"""Nuisance-model machinery: cross-fitting, screening and the Super Learner."""

from __future__ import annotations

from ._fitting import (
    accepts_groups,
    fit_learner,
    infer_task,
    predict_mean,
    supports_sample_weight,
)
from ._threads import get_thread_limit, set_thread_limit, thread_limit
from .crossfit import Folds, make_folds, resolve_n_folds
from .density import (
    ConditionalDensity,
    DensityDiagnostics,
    bin_edges,
    fit_conditional_density,
)
from .library import LIBRARY_PRESETS, has_lightgbm, resolve_library
from .screeners import CorrelationScreener, screen_by_correlation
from .super_learner import SuperLearner, SuperLearnerDiagnostics

__all__ = [
    "LIBRARY_PRESETS",
    "ConditionalDensity",
    "CorrelationScreener",
    "DensityDiagnostics",
    "Folds",
    "SuperLearner",
    "SuperLearnerDiagnostics",
    "accepts_groups",
    "bin_edges",
    "fit_conditional_density",
    "fit_learner",
    "get_thread_limit",
    "has_lightgbm",
    "infer_task",
    "make_folds",
    "predict_mean",
    "resolve_library",
    "resolve_n_folds",
    "screen_by_correlation",
    "set_thread_limit",
    "supports_sample_weight",
    "thread_limit",
]

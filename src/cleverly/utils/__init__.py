"""Internal helpers shared across estimators."""

from __future__ import annotations

from .bounds import OutcomeScaler, bound, expit, logit, resolve_g_bounds, shrink_probabilities
from .frames import as_frame, column_array, frame_from_dict, matrix_from_columns, resolve_backend
from .parallel import map_parallel, resolve_n_jobs

__all__ = [
    "OutcomeScaler",
    "as_frame",
    "bound",
    "column_array",
    "expit",
    "frame_from_dict",
    "logit",
    "map_parallel",
    "matrix_from_columns",
    "resolve_backend",
    "resolve_g_bounds",
    "resolve_n_jobs",
    "shrink_probabilities",
]

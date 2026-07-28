"""Shared type aliases.

Kept in a private module so the public namespace stays focused on estimators.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ArrayLike",
    "Backend",
    "BoolArray",
    "Estimand",
    "Family",
    "FloatArray",
    "FluctuationKind",
    "GBounds",
    "IntArray",
    "Learner",
    "TargetingMethod",
    "TargetingScheme",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ArrayLike = Any

#: Anything implementing the scikit-learn ``fit``/``predict`` (or
#: ``predict_proba``) protocol, including :class:`cleverly.SuperLearner`,
#: :class:`sklearn.pipeline.Pipeline`, and grid-search wrappers.  Typed as
#: ``Any`` deliberately: scikit-learn estimators are structurally, not
#: nominally, typed and requiring a Protocol here would reject valid learners.
Learner = Any

Backend = Literal["pandas", "polars"]
Family = Literal["gaussian", "binomial", "auto"]
FluctuationKind = Literal["logistic", "linear"]
TargetingMethod = Literal["iterative", "one_step"]
TargetingScheme = Literal["pooled", "fold"]
Estimand = Literal["ey1", "ey0", "ate", "att", "atc", "rr", "or"]

#: Propensity-score truncation: ``"auto"`` for the sample-size dependent
#: default, a single float ``lo`` meaning ``[lo, 1 - lo]``, or an explicit pair.
GBounds = Literal["auto"] | float | tuple[float, float]

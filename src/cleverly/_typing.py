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
    "ParameterAxis",
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

#: What a fit's parameters are indexed *by*: a treatment arm, a declared regime, a
#: declared shift, or a coefficient of a declared working model.  The four partition the
#: target registry -- see :attr:`cleverly.Target.parameter_axis` for why they are
#: exclusive rather than cumulative.
#:
#: The first three also declare what "counterfactual" means for the fit.  ``"msm"`` does
#: not: its counterfactuals are still the arms, and what moves is the *summary* the fit
#: reports of them.  It is an axis all the same, because a summary's coefficients are not
#: indexed by anything the other three name.
ParameterAxis = Literal["arm", "regime", "shift", "msm"]

#: Propensity-score truncation: ``"auto"`` for the sample-size dependent
#: default, a single float ``lo`` meaning ``[lo, 1 - lo]``, or an explicit pair.
GBounds = Literal["auto"] | float | tuple[float, float]

"""Shared type aliases.

Kept in a private module so the public namespace stays focused on estimators.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Backend",
    "BoolArray",
    "CumulativeGBounds",
    "Estimand",
    "Family",
    "FloatArray",
    "FluctuationKind",
    "FoldStrata",
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

#: What the outer cross-fitting folds are balanced on.  ``"treatment"`` is the
#: long-standing behaviour and the default; ``"treatment+outcome"`` crosses in the
#: outcome so that a rare event cannot leave a fold with none of them.
FoldStrata = Literal["treatment", "treatment+outcome"]
#: Every estimand name :class:`~cleverly.estimators.TMLE`'s ``estimands=`` accepts, which is
#: every key of the target registry.  Which of them a *particular* fit can report depends on
#: its outcome family, its arm count and which parameter axis it declared, and that is
#: checked at runtime by :func:`~cleverly.targets.resolve_estimands` against the registry
#: itself.  This alias is the static half of the same statement and cannot be derived from
#: the registry: a ``Literal`` has to be written out, and importing ``cleverly.targets`` here
#: would invert the dependency of the type aliases on the package.
#:
#: So it is a hand-maintained copy, and the way a hand-maintained copy stays honest is a gate
#: rather than care -- ``tests/unit/test_registry.py`` compares its members with ``TARGETS``
#: in both directions.  It had already drifted once without one: ``ey``, ``ey_obs``, ``par``
#: and ``paf`` were reportable and absent from here, as was every non-arm axis, so annotating
#: a correct call was a type error.
Estimand = Literal[
    "ate",
    "att",
    "atc",
    "ey",
    "ey1",
    "ey0",
    "ey_obs",
    "par",
    "paf",
    "rr",
    "or",
    "ey_regime",
    "ate_regime",
    "ey_ipsi",
    "ate_ipsi",
    "ey_shift",
    "ate_shift",
    "msm",
]

#: What a fit's parameters are indexed *by*: a treatment arm, a declared regime, a
#: declared shift, a declared tilt of the mechanism, or a coefficient of a declared
#: working model.  The five partition the target registry -- see
#: :attr:`cleverly.Target.parameter_axis` for why they are exclusive rather than
#: cumulative.
#:
#: The first four also declare what "counterfactual" means for the fit.  ``"msm"`` does
#: not: its counterfactuals are still the arms, and what moves is the *summary* the fit
#: reports of them.  It is an axis all the same, because a summary's coefficients are not
#: indexed by anything the other four name.
#:
#: ``"ipsi"`` is the one whose intervention is a functional of the observed-data law: its
#: ``q_delta`` is built out of the estimated mechanism, so it carries an extra influence
#: curve term and a second score equation.  That is why it is not a kind of ``"regime"``.
ParameterAxis = Literal["arm", "regime", "shift", "ipsi", "msm"]

#: Propensity-score truncation: ``"auto"`` for the sample-size dependent
#: default, a single float ``lo`` meaning ``[lo, 1 - lo]``, or an explicit pair.
GBounds = Literal["auto"] | float | tuple[float, float]

#: Bounds on an estimated cumulative longitudinal treatment-and-censoring
#: probability.  There is deliberately no ``"auto"`` member: cleverly has no
#: data-adaptive or depth-adaptive rule for choosing this bound.  LTMLE's package
#: default is an explicit fixed pair, recorded in :mod:`cleverly.utils.bounds`.
CumulativeGBounds = float | tuple[float, float]

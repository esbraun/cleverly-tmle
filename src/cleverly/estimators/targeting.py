"""Building the clever covariate and solving for the fluctuation.

These two steps used to be private methods on :class:`~cleverly.TMLE`
(``_submodel`` and ``_solve_rows``), which made them awkward for the code that
legitimately needs them: :mod:`cleverly.sensitivity.positivity` reaches for the
clever covariate to report its largest value,
:mod:`cleverly.sensitivity.omitted_variable` re-solves the fluctuation against a
tilted propensity, and :class:`~cleverly.CTMLE` does both while searching over
candidate propensity models.  All three were calling through
``result.estimator._submodel``, so a result whose estimator had been dropped
silently reported ``nan``.

Neither step needs an estimator.  ``build_submodel`` is a pure function of the
data and the nuisance estimates; ``solve_submodel`` needs only the seven settings
collected here into :class:`TargetingSpec`, which :class:`~cleverly.estimators
.base.TMLEConfig` carries on every result.  Pulling them out is what lets a
serialised result keep working -- everything reached through ``retarget`` needs
this module and nothing else.

This module deliberately does not import :mod:`cleverly.estimators.base`, so
``base`` can hold a :class:`TargetingSpec` without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray, FluctuationKind, TargetingMethod
from ..data.causal_data import CausalData
from ..fluctuation.iterative import Fluctuation, InitialFit, solve_fluctuation
from ..fluctuation.one_step import solve_one_step
from ..fluctuation.submodel import Submodel, TargetGroup, submodel_for
from ._nuisance import NuisanceEstimates
from .direct_effect import clever_covariate_inputs

__all__ = ["TargetingSpec", "build_submodel", "solve_submodel"]


@dataclass(frozen=True)
class TargetingSpec:
    """Everything the targeting step needs that is not data.

    Kept as one object so that re-solving a fluctuation -- which the truncation
    curve, the omitted-variable tilt and the C-TMLE search all do -- needs the
    settings of the fit it is re-solving, not the estimator that produced it.
    """

    targeting: TargetingMethod = "iterative"
    fluctuation: FluctuationKind = "logistic"
    target_weights: bool = False
    alpha: float = 0.9995
    max_iter: int = 20
    tol: float = 1e-10
    step_size: float = 1e-3


def build_submodel(
    data: CausalData,
    nuisance: NuisanceEstimates,
    group: TargetGroup,
    *,
    bounds: tuple[float, float],
    nuisance_bound: float,
    intermediate_value: float | None = None,
    missingness_override: FloatArray | None = None,
) -> Submodel:
    """Clever covariates for one estimand family, at the given truncation bounds.

    ``bounds`` truncates the propensity and ``nuisance_bound`` the missingness and
    intermediate mechanisms.  Both are applied here rather than stored on
    :class:`~cleverly.estimators._nuisance.NuisanceEstimates`, which keeps the raw
    predictions, so that a sensitivity sweep can re-truncate without refitting.

    ``missingness_override`` replaces the estimated ``P(Delta = 1 | A, W)`` with a
    supplied one, which is how the MNAR tilt moves the missingness mechanism
    without touching any other nuisance.
    """
    lower = float(nuisance_bound)
    propensity = nuisance.bounded_propensity(bounds)
    missingness = (
        nuisance.bounded_missingness(lower)
        if missingness_override is None
        else np.clip(np.asarray(missingness_override, dtype=float), lower, 1.0)
    )
    intermediate_density, selection = clever_covariate_inputs(
        data, nuisance, intermediate_value, lower
    )
    return submodel_for(
        group,
        data.treatment,
        propensity,
        arms=nuisance.arms,
        # Read lazily: `treated_fraction` names no quantity on a continuous treatment and
        # raises, and every builder is called through this one signature -- so evaluating
        # it eagerly would refuse a shift fit on behalf of a builder that discards it.
        treated_fraction=None if data.is_continuous_treatment else data.treated_fraction,
        missingness=missingness,
        intermediate_density=intermediate_density,
        selection=selection,
        regimes=None if nuisance.regimes is None else nuisance.regimes.values,
        shifts=None if nuisance.shifts is None else nuisance.shifts.design,
    )


def solve_submodel(
    scaled: FloatArray,
    initial: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    observed: BoolArray,
    spec: TargetingSpec,
    *,
    warn: bool = True,
) -> Fluctuation:
    """Solve the fluctuation on the given rows, by whichever method ``spec`` names."""
    if spec.targeting == "one_step":
        return solve_one_step(
            scaled,
            initial,
            submodel,
            weights,
            observed,
            target_weights=spec.target_weights,
            alpha=spec.alpha,
            step_size=spec.step_size,
            tol=spec.tol,
            warn=warn,
        )
    return solve_fluctuation(
        scaled,
        initial,
        submodel,
        weights,
        observed,
        kind=spec.fluctuation,
        target_weights=spec.target_weights,
        alpha=spec.alpha,
        max_iter=spec.max_iter,
        tol=spec.tol,
        warn=warn,
    )

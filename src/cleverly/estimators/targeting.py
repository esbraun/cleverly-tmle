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

from dataclasses import dataclass, replace

import numpy as np

from .._typing import BoolArray, FloatArray, FluctuationKind, TargetingMethod
from ..data.causal_data import CausalData
from ..fluctuation.iterative import Fluctuation, InitialFit, solve_fluctuation
from ..fluctuation.mechanism import (
    MechanismFluctuation,
    mechanism_covariate,
    mechanism_score,
    solve_mechanism,
)
from ..fluctuation.one_step import solve_one_step
from ..fluctuation.submodel import Submodel, TargetGroup, submodel_for
from ._nuisance import NuisanceEstimates
from .direct_effect import clever_covariate_inputs

__all__ = ["TargetingSpec", "build_submodel", "solve_submodel", "solve_with_mechanism"]

#: A round that leaves the mechanism score above this share of the previous round's has
#: reached the fixed point of the alternation; further rounds move nothing.
_STALL_FACTOR = 0.5

#: A relative mechanism score above this is genuinely unsolved and is reported as a
#: failure.  Well clear of the fixed point the alternation settles at in practice
#: (~1e-10), and well below the point at which the score would matter beside a standard
#: error -- :func:`~cleverly.validation.score_check` makes that comparison properly.
_UNSOLVED = 1e-6


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
        # The covariate needs only h * phi; the two factors are wanted apart one layer up,
        # in the projection itself. See cleverly.msm.MSMSet.weighted_design.
        msm=None if nuisance.msm is None else nuisance.msm.weighted_design,
        # The tilt's covariate, precomputed from the *untruncated* mechanism. The
        # `propensity` above still arrives bounded and the ipsi builder discards it:
        # g is inside that estimand, so a bound there would move the parameter.
        incremental=None if nuisance.incremental is None else nuisance.incremental.weights,
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


def solve_with_mechanism(
    data: CausalData,
    nuisance: NuisanceEstimates,
    group: TargetGroup,
    spec: TargetingSpec,
    *,
    bounds: tuple[float, float],
    nuisance_bound: float,
    scaled: FloatArray,
    weights: FloatArray,
    observed: BoolArray,
    max_outer: int = 20,
    warn: bool = True,
) -> tuple[Submodel, Fluctuation, NuisanceEstimates]:
    r"""Target ``Qbar`` and the treatment mechanism together, alternating until both settle.

    For a group whose parameter is defined *through* the mechanism -- today only ``ipsi``
    -- solving the outcome score equation leaves a second one, in the tangent space of
    the treatment mechanism, unsolved.  :mod:`cleverly.fluctuation.mechanism` derives it.
    Neither can be solved once and left: the mechanism covariate reads the targeted
    :math:`\bar Q^*` and the outcome covariate reads the targeted :math:`g^*`.

    The loop is coordinate ascent on one joint likelihood.  The outcome quasi-likelihood
    and :math:`P_n \log g_\epsilon(A \mid W)` are separate factors of the likelihood of
    :math:`(A, Y) \mid W`, and each step is the MLE within its own submodel holding the
    other fixed, so the joint value never decreases.  That is why the outcome
    fluctuation *continues* from the current :math:`\bar Q^*` rather than restarting from
    :math:`\bar Q^0`: restarting would break the monotonicity and with it the argument
    that this terminates.

    Returns the final submodel, the outcome fluctuation carrying the mechanism
    fluctuation on it, and a :class:`NuisanceEstimates` whose tilt has been re-evaluated
    at the targeted mechanism.  The caller uses that third value for the estimates and
    keeps the original for ``result.nuisance``, so the reported nuisances stay the ones
    that were actually fitted.
    """
    if nuisance.incremental is None:
        raise ValueError(
            f"group {group!r} targets the treatment mechanism, so the fit must carry the "
            "intervention that defines it; this NuisanceEstimates has none"
        )
    current = nuisance
    tilt = nuisance.incremental
    fit = nuisance.outcome
    submodel = build_submodel(data, current, group, bounds=bounds, nuisance_bound=nuisance_bound)
    fluctuation = solve_submodel(scaled, fit, submodel, weights, observed, spec, warn=warn)
    trace: list[tuple[int, float, float, float]] = []
    mechanism: MechanismFluctuation | None = None
    previous: float | None = None

    for outer in range(1, max_outer + 1):
        covariate = mechanism_covariate(group, fluctuation.targeted, tilt)
        mechanism = solve_mechanism(
            data.treatment, tilt.propensity, covariate, weights, tol=spec.tol
        )
        current = current.retilted(mechanism.propensity)
        assert current.incremental is not None
        tilt = current.incremental

        submodel = build_submodel(
            data, current, group, bounds=bounds, nuisance_bound=nuisance_bound
        )
        fluctuation = solve_submodel(
            scaled, fluctuation.targeted, submodel, weights, observed, spec, warn=False
        )
        # The mechanism covariate moved when Qbar did, so `mechanism.score` is now stale.
        # Re-evaluate it at the fluctuated pair before testing convergence: a loop that
        # tested the stale one would exit having solved the outcome equation and left the
        # mechanism equation open, which is precisely what the delta=1 identity catches.
        settled, scale = mechanism_score(
            data.treatment,
            tilt.propensity,
            mechanism_covariate(group, fluctuation.targeted, tilt),
            weights,
        )
        mechanism = replace(mechanism, score=settled, score_scale=scale)
        joint = float(fluctuation.loglik) + float(mechanism.loglik or 0.0)
        settled_score = mechanism.relative_score
        trace.append((outer, fluctuation.relative_score_norm, settled_score, joint))
        if max(fluctuation.relative_score_norm, settled_score) <= spec.tol:
            break
        # The alternation converges linearly, so the score falls by a roughly constant
        # factor each round until it reaches the fixed point of the coupled system and
        # stops moving. Iterating past that buys nothing: stop when a round no longer
        # improves the mechanism score materially, and let the *size* of what is left
        # decide whether it counts as a failure.
        if previous is not None and settled_score > _STALL_FACTOR * previous:
            break
        previous = settled_score

    assert mechanism is not None
    failure = mechanism.failure
    if failure is None and trace and trace[-1][2] > _UNSOLVED:
        # Reported, never raised, on the terms every other targeting failure is: a
        # sensitivity sweep that pushes a fit into bad territory must still return.
        failure = "max_iter_reached"
    mechanism = replace(mechanism, trace=tuple(trace), failure=failure)
    return submodel, replace(fluctuation, mechanism=mechanism), current

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

from .._typing import BoolArray, FloatArray, FluctuationKind, IntArray, TargetingMethod
from ..data.causal_data import CausalData
from ..fluctuation.iterative import Fluctuation, InitialFit, solve_fluctuation
from ..fluctuation.mechanism import (
    MechanismFluctuation,
    mechanism_covariate,
    mechanism_score,
    solve_mechanism,
)
from ..fluctuation.one_step import solve_one_step
from ..fluctuation.submodel import Submodel, TargetGroup, restrict, submodel_for
from ..msm import solve_projection
from ..utils.bounds import OutcomeScaler
from ._nuisance import NuisanceEstimates
from .direct_effect import clever_covariate_inputs

__all__ = [
    "ProjectionFluctuation",
    "TargetingSpec",
    "build_submodel",
    "needs_projection",
    "reported_beta",
    "solve_submodel",
    "solve_with_mechanism",
    "solve_with_projection",
]

#: A round that leaves the mechanism score above this share of the previous round's has
#: reached the fixed point of the alternation; further rounds move nothing.
#:
#: Deliberately close to one.  The alternation converges *linearly*, at a rate set by how
#: strongly the two nuisances are coupled, and that rate is not always fast: measured here
#: at 0.15 per round on a well-conditioned fit and 0.52 on one whose tilts are collinear
#: enough to make the mechanism solve ill-conditioned. A threshold that treated 0.52 as a
#: stall would stop a loop that was halving its score every round and had another twelve
#: rounds of progress in it -- which it did, and which the score check caught.
_STALL_FACTOR = 0.95

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
    reference: float | None = None,
    msm_beta: FloatArray | None = None,
) -> Submodel:
    """Clever covariates for one estimand family, at the given truncation bounds.

    ``bounds`` truncates the propensity and ``nuisance_bound`` the missingness and
    intermediate mechanisms.  Both are applied here rather than stored on
    :class:`~cleverly.estimators._nuisance.NuisanceEstimates`, which keeps the raw
    predictions, so that a sensitivity sweep can re-truncate without refitting.

    ``missingness_override`` replaces the estimated ``P(Delta = 1 | A, W)`` with a
    supplied one, which is how the MNAR tilt moves the missingness mechanism
    without touching any other nuisance.

    ``reference`` is the arm the conditional-effect fluctuations contrast against, and is
    ignored by every other group.  ``None`` means the lowest arm, which is what a binary
    fit has always used and what the arm-indexed estimands default to.

    ``msm_beta`` is the working model's current coefficients, which a **non-identity link**
    puts inside the clever covariate through :math:`dm/d\\eta`.  It is ignored under the
    identity link, where the covariate is free of :math:`\\beta`, and required under any
    other -- which is why the ``msm`` group's targeting alternates
    (:func:`solve_with_projection`) rather than building this once.
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
        # Read lazily: `arm_fractions` names no quantity on a continuous treatment and
        # raises, and every builder is called through this one signature -- so evaluating
        # it eagerly would refuse a shift fit on behalf of a builder that discards it.
        arm_fractions=None if data.is_continuous_treatment else data.arm_fractions,
        reference=reference,
        missingness=missingness,
        intermediate_density=intermediate_density,
        selection=selection,
        regimes=None if nuisance.regimes is None else nuisance.regimes.values,
        shifts=None if nuisance.shifts is None else nuisance.shifts.design,
        # The covariate needs only h * (dm/deta) * phi; the factors are wanted apart one
        # layer up, in the projection itself. See cleverly.msm.MSMSet.weighted_design_at,
        # which is the identity link's beta-free array whenever the link is the identity.
        msm=None if nuisance.msm is None else nuisance.msm.weighted_design_at(msm_beta),
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


@dataclass(frozen=True)
class ProjectionFluctuation:
    """The working model's coefficients, and how the alternation that found them went.

    A sibling of :class:`~cleverly.fluctuation.mechanism.MechanismFluctuation` in the same
    sense: the other half of a targeting step that has two halves, carried on the outcome
    fluctuation that was solved beside it.  It is a different *kind* of thing, though, and
    the difference is the whole reason this is a separate class rather than a reuse.  A
    mechanism fluctuation is a nuisance that was tilted, and its half of the alternation is
    a likelihood maximisation.  This is the **reported parameter**, solved for by weighted
    least squares -- so the two steps here maximise nothing in common and the coordinate
    ascent argument that makes ``solve_with_mechanism`` terminate does not carry over.  See
    :func:`solve_with_projection` for what does.

    Attributes
    ----------
    beta:
        The coefficients the **report** is taken at: the projection of the targeted fit,
        which is the same solve on the same predictions that
        :func:`~cleverly.inference.influence.msm_coefficients` runs.  On a pooled fit it
        is also the ``beta`` the returned covariate was built at, to within ``tol`` -- that
        is what the alternation converges to.  Under fold-wise targeting each fold had its
        own, recorded in :attr:`folds`, and this is still the one the coefficients are
        reported at.  Kept so that a caller rebuilding the covariate --
        ``res.sensitivity.positivity()`` does -- does not have to guess at it.
    trace:
        One ``(outer, relative outcome score, relative shift in beta)`` row per round.
        Kept for the reason the mechanism's trace is: a loop that stalls should be visible
        rather than inferred.
    converged, failure:
        Whether the shift in ``beta`` reached ``tol``, and why it stopped if not.
        Reported, never raised, on the terms every other targeting failure is.
    """

    beta: FloatArray
    trace: tuple[tuple[int, float, float], ...] = ()
    converged: bool = True
    failure: str | None = None
    #: Per-fold detail under ``targeting_scheme="fold"``, where each fold ran its own
    #: alternation and so had its own covariate.  Empty for a pooled fit.
    folds: tuple[ProjectionFluctuation, ...] = ()

    @property
    def n_outer(self) -> int:
        return len(self.trace)


def needs_projection(nuisance: NuisanceEstimates, group: TargetGroup) -> bool:
    """Whether this fit's targeting has a projection half to alternate with.

    A property of the declared *link* rather than of the group: an identity-link working
    model has a clever covariate free of ``beta``, so its targeting is one fluctuation and
    goes down exactly the path it went down before links existed.
    """
    return group == "msm" and nuisance.msm is not None and nuisance.msm.link != "identity"


def reported_beta(
    nuisance: NuisanceEstimates, targeted: InitialFit, weights: FloatArray
) -> FloatArray | None:
    """The coefficients a targeted fit reports, or ``None`` if it has no working model.

    The projection of ``targeted``, which is what
    :func:`~cleverly.inference.influence.msm_coefficients` reports and therefore the only
    ``beta`` a diagnostic should rebuild the covariate at.  One function rather than the
    two callers each writing the solve, for the reason there is one
    :func:`~cleverly.msm.solve_projection`.
    """
    msm = nuisance.msm
    if msm is None:
        return None
    predictions = _raw_arms(targeted, msm.arms, nuisance.scaler)
    return solve_projection(msm.design, msm.weights, predictions, weights, msm.link).beta


def solve_with_projection(
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
    rows: IntArray | None = None,
    max_outer: int = 50,
    warn: bool = True,
) -> tuple[Submodel, Fluctuation]:
    r"""Target ``Qbar`` and solve the working model's projection, alternating until both settle.

    Under a non-identity link the clever covariate is
    :math:`h\,(dm/d\eta)\,\varphi / g`, which reads :math:`\beta`; and :math:`\beta` is
    the projection of the *targeted* :math:`\bar Q^*`, which reads the covariate.  Neither
    can be solved once and left, so this alternates:

    .. code-block:: text

        beta <- projection of Qbar^0
        repeat:  covariate at beta  ->  fluctuate Qbar^0  ->  beta <- projection of Qbar*

    **Each round restarts from** :math:`\bar Q^0` rather than continuing from the current
    :math:`\bar Q^*`, which is the opposite of what :func:`solve_with_mechanism` does, and
    deliberately.  That loop must continue, because its argument for terminating is that
    each step maximises its own factor of one joint likelihood and so cannot decrease it.
    Here there is no joint criterion to preserve -- the projection is a least-squares
    solve, not a likelihood -- so the useful thing to have instead is a clean fixed point:
    at exit, :math:`\bar Q^*` is *the* fluctuation of :math:`\bar Q^0` along
    :math:`H_{\hat\beta}` and :math:`\hat\beta` is *the* projection of that
    :math:`\bar Q^*`.  ``epsilon`` stays one interpretable coefficient vector rather than a
    running sum along a path of submodels, and the identity link is exactly the case that
    exits after one round -- which is why it is short-circuited before this is ever called
    rather than handled inside it.

    Convergence is judged on the shift in :math:`\beta`, relative to its size, and it is
    fast: measured on a three-armed binomial process the shift falls by a factor of
    ``1e-3`` per round under the log link and ``1e-4`` under the logit, so the loop exits
    in four rounds.  That is not luck, and it is the contrast with
    :func:`solve_with_mechanism` again -- there the two nuisances enter each other's
    covariates directly, while here :math:`\beta` reaches the covariate only through the
    smooth factor :math:`dm/d\eta`, which barely moves once the fluctuation is small.  The
    stall rule inherited from :data:`_STALL_FACTOR` is therefore slack here rather than
    load-bearing, and is kept so that a badly conditioned fit stops rather than spins.

    ``rows`` restricts the whole alternation to a subset -- one validation fold, for
    ``targeting_scheme="fold"``.  Each fold then gets its *own* :math:`\beta`, which is
    what makes fold-wise targeting mean here what it means everywhere else: no row
    contributes to any coefficient that fluctuates it.  The covariate is built on the full
    sample at that fold's :math:`\beta` and then restricted, rather than rebuilt from
    sliced nuisances, because the covariate is row-wise and the two agree -- and because
    the sliced version would have to re-derive arm fractions and bounds from a subsample
    that is not the population they describe.
    """
    if nuisance.msm is None:  # pragma: no cover - guarded by needs_projection
        raise ValueError(
            f"group {group!r} targets a working model's coefficients, so the fit must "
            "carry the model that defines them; this NuisanceEstimates has none"
        )
    index = None if rows is None else np.asarray(rows)
    msm = nuisance.msm if index is None else nuisance.msm.subset(index)
    initial = nuisance.outcome if index is None else _slice(nuisance.outcome, index)
    y = scaled if index is None else scaled[index]
    w = weights if index is None else weights[index]
    seen = observed if index is None else observed[index]
    scaler = nuisance.scaler

    beta = solve_projection(
        msm.design, msm.weights, _raw_arms(initial, msm.arms, scaler), w, msm.link
    ).beta
    trace: list[tuple[int, float, float]] = []
    previous: float | None = None
    submodel = _projection_submodel(data, nuisance, group, bounds, nuisance_bound, beta, index)
    fluctuation = solve_submodel(y, initial, submodel, w, seen, spec, warn=warn)

    for outer in range(1, max_outer + 1):
        solved = solve_projection(
            msm.design,
            msm.weights,
            _raw_arms(fluctuation.targeted, msm.arms, scaler),
            w,
            msm.link,
        )
        shift = float(np.max(np.abs(solved.beta - beta))) / (1.0 + float(np.max(np.abs(beta))))
        trace.append((outer, fluctuation.relative_score_norm, shift))
        if shift <= spec.tol:
            break
        # The alternation contracts linearly, so the shift falls by a roughly constant
        # factor each round until it reaches the fixed point and stops moving. Stop when a
        # round no longer improves it materially, and let the *size* of what is left decide
        # whether that counts as a failure.
        if previous is not None and shift > _STALL_FACTOR * previous:
            break
        previous = shift
        beta = solved.beta
        submodel = _projection_submodel(data, nuisance, group, bounds, nuisance_bound, beta, index)
        # From Qbar^0 again, not from the current Qbar*: see above.
        fluctuation = solve_submodel(y, initial, submodel, w, seen, spec, warn=False)

    last = trace[-1][2] if trace else 0.0
    failure = None if last <= _UNSOLVED else "max_iter_reached"
    projection = ProjectionFluctuation(
        beta=beta, trace=tuple(trace), converged=bool(last <= spec.tol), failure=failure
    )
    return submodel, replace(fluctuation, projection=projection)


def _projection_submodel(
    data: CausalData,
    nuisance: NuisanceEstimates,
    group: TargetGroup,
    bounds: tuple[float, float],
    nuisance_bound: float,
    beta: FloatArray,
    index: IntArray | None,
) -> Submodel:
    """The clever covariate at ``beta``, on the whole sample or on one fold's rows."""
    submodel = build_submodel(
        data, nuisance, group, bounds=bounds, nuisance_bound=nuisance_bound, msm_beta=beta
    )
    return submodel if index is None else restrict(submodel, index)


def _slice(fit: InitialFit, index: IntArray) -> InitialFit:
    """One fold's rows of an initial fit."""
    return fit.map_arms(lambda values: values[index])


def _raw_arms(fit: InitialFit, arms: tuple[float, ...], scaler: OutcomeScaler) -> FloatArray:
    """``(n, K)`` predictions on the *outcome's own* scale, arms in the model's order.

    The projection is solved where its coefficients are reported, which for this estimand
    is the raw scale -- :func:`~cleverly.inference.influence.msm_coefficients` sets out
    why a coefficient vector has no single scale to map back with.  The score equation is
    indifferent to which of the two is used, since the residual rescales by the same
    factor, but ``m`` under a link is not: ``expit`` of a linear predictor is a
    probability, and a probability is not a scaled outcome.
    """
    stacked = np.column_stack([fit.arms[level] for level in arms])
    if scaler.is_identity:
        return stacked
    return np.asarray(scaler.lower + scaler.range * stacked, dtype=float)


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
    max_outer: int = 50,
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

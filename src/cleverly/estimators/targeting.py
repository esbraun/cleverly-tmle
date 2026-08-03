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

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .._typing import BoolArray, FloatArray, FluctuationKind, IntArray, TargetingMethod
from ..data.causal_data import CausalData
from ..fluctuation._score import relative_score, score_columns, score_scale
from ..fluctuation.iterative import (
    Fluctuation,
    InitialFit,
    TargetingFailure,
    solve_fluctuation,
)
from ..fluctuation.mechanism import (
    MechanismFluctuation,
    mechanism_covariate,
    mechanism_score,
    solve_bounded_mechanism,
    solve_mechanism,
)
from ..fluctuation.one_step import solve_one_step
from ..fluctuation.reduced import reduced_mechanism_covariate, reduced_outcome_submodel
from ..fluctuation.submodel import Submodel, TargetGroup, restrict, submodel_for
from ..msm import solve_projection
from ..utils.bounds import OutcomeScaler
from ._nuisance import NuisanceEstimates, Propensity
from .direct_effect import clever_covariate_inputs
from .reduced import ReducedSet

__all__ = [
    "ProjectionFluctuation",
    "ReductionExit",
    "ReductionFluctuation",
    "ReductionOrder",
    "ReductionSpec",
    "TargetingSpec",
    "build_submodel",
    "needs_projection",
    "needs_reduction",
    "reported_beta",
    "solve_submodel",
    "solve_with_mechanism",
    "solve_with_projection",
    "solve_with_reduction",
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

#: An **absolute** score below ``_NEGLIGIBLE / n`` counts an equation as solved, whatever
#: its relative score says.  This exists because the relative test is the wrong instrument
#: for two of the three, and the reason is structural rather than a matter of taste.
#:
#: :func:`~cleverly.fluctuation._score.relative_score` divides by ``mean|w h|``, which
#: :func:`~cleverly.fluctuation._score.score_scale` documents as "the largest the score
#: could be" -- a sound normalisation for a covariate of order one.  Equation (10)'s
#: covariate is :math:`g_{r,2}/g_{r,1}` and :math:`g_{r,2}` vanishes exactly where the
#: mechanism is right, so on the fits anybody wants that denominator is ``1e-3`` to
#: ``1e-2``: measured at ``9.4e-3`` and ``2.4e-3`` on a 400-row ``linear`` fit and
#: ``5.1e-2`` and ``4.9e-3`` on a ``nonlinear`` one.  Asking for ``spec.tol`` *of that* is
#: asking for an absolute score near ``1e-13``, six orders below the point at which the
#: score stops mattering.  Equation (9) cannot reach ``spec.tol`` for a different reason
#: and one already written down as item 5: its covariate reads the very mechanism it
#: tilts, so a solve zeroes the score at the pre-tilt covariate and leaves a residual at
#: the post-tilt one.  Neither is a solver that failed; both are rulers with the wrong
#: zero, and on the round a 400-row fit gave up at they read ``2.3e-8`` and ``3.9e-8``
#: together -- so relaxing either alone stops nothing, which was measured before this was
#: applied to all three.
#:
#: The bar here is the one the package already applies to the fit it reports:
#: :func:`~cleverly.validation.score_check` passes a fluctuation whose score is under
#: ``DEFAULT_TOLERANCE * se / sqrt(n)``, and with ``se = O(n**-0.5)`` on the scaled outcome
#: that is this constant over ``n``.  Asymptotic linearity asks for ``P_n D = o(n**-0.5)``
#: and nothing more; machine zero was never the requirement.  Where the covariate *is* of
#: order one -- equation (8), whose ``1/g`` is bounded below by the truncation -- the
#: relative test is the tighter of the two and still does the stopping, so a
#: well-conditioned fit exits exactly where it used to.
_NEGLIGIBLE = 1e-3


def _solved(relative: float, absolute: float, tol: float, negligible: float) -> bool:
    """Whether one equation is solved, on whichever of the two rulers it can meet.

    The relative test is the tighter one wherever the covariate is of order one, so it
    goes on doing the stopping for equation (8) and nothing about a well-conditioned fit
    changes.  The absolute test is what lets equations (9) and (10) stop at all: both have
    a covariate that is small or that moves under its own solve, and neither can reach
    ``tol`` *of its own magnitude* -- which is a fact about the derivation, recorded as
    items 5 to 7 of ``docs/roadmap.md``, rather than a solver that needs more rounds.
    """
    return relative <= tol or absolute <= negligible


#: Which of :func:`solve_with_reduction`'s three exits fired.  ``"tolerance"`` is the loop
#: reaching ``spec.tol``, ``"stall"`` is the dual rule below finding neither the objective
#: climbing nor the score improving, and ``"cap"`` is running out of ``max_outer`` rounds.
#:
#: Recorded because nothing else on the record carries it and the three are not otherwise
#: distinguishable: ``rounds`` cannot separate a stall from a cap without ``max_outer``,
#: which is a function default rather than a field, and ``failure`` is a statement about
#: score magnitude rather than about how the loop ended -- see
#: :attr:`ReductionFluctuation.exit_reason`.
ReductionExit = Literal["tolerance", "stall", "cap"]

#: Which order the three equations are solved in within a round.  Two routes to one stated
#: exit rather than two estimators: the 2016 working paper's step 7 states its own
#: termination as the three empirical means being approximately zero, so the order it writes
#: down is one way of reaching a fixed point and not part of what Theorem 1 assumes about the
#: collection returned.  That is ``docs/roadmap.md``'s item 22, whose theoretical half closed
#: on reading the paper and whose numerical half -- *do the two routes reach the same fixed
#: point on real data* -- is a measurement, and needs the second route to exist here.
#:
#: ``"cleverly"`` is this package's own and the default, bit for bit what it always was:
#: equation (9), refit, equation (10), equation (8), refit.  ``"paper"`` is
#: ``docs/drtmle-theorem-concordance.md`` §6's steps 2 to 6 -- equation (8), refit
#: :math:`g_{r,1}` and :math:`g_{r,2}` at the **once-updated** outcome regression, equation
#: (10), refit :math:`Q_r` at the **twice-updated** one, equation (9).  Neither the exit
#: test, the stall rule nor the closing pass differs between them, deliberately: what is in
#: question is the route, and running one arm of the comparison under a different stopping
#: rule or a different reporting convention would confound the two.
ReductionOrder = Literal["cleverly", "paper"]


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
    return scaler.unscale_levels(np.column_stack([fit.arms[level] for level in arms]))


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


@dataclass(frozen=True)
class ReductionSpec:
    """What a doubly-robust targeting step needs that the cached arrays cannot carry.

    A third sibling of :class:`TargetingSpec`, and the one place this module's
    estimator-free design bends.  Every other targeting step here is arithmetic on fitted
    predictions, so a serialised result can re-run it with nothing but its settings.  This
    one **refits learners inside the alternation** -- equations (9) and (10) are stated at
    starred :math:`Q_r^*` and :math:`g_r^*`, and ``drtmle``'s algorithm maps initial
    estimates of the outcome regression, the mechanism *and the reduced regressions* into
    estimates that satisfy them.  Holding the reductions at their initial fit would solve a
    different equation, and whether that one suffices is a question for a theorem rather
    than a matter of taste.

    So the learners arrive as a callable rather than as an import: this module stays free of
    :mod:`cleverly.learners` and of the estimator, and
    :class:`~cleverly.DRTMLE` supplies a closure over the ones it resolved.  What that
    costs is stated where a reader meets it -- a truncation curve on a doubly-robust fit
    costs a fit per point rather than a fraction of one.

    Attributes
    ----------
    refit:
        Takes a :class:`~cleverly.estimators._nuisance.NuisanceEstimates` carrying the
        *current* targeted regression and mechanism, and returns the reductions relative to
        them.  It is handed a whole object rather than two arrays for the reason
        :func:`~cleverly.estimators.reduced.fit_reduced` takes one: ``folds`` is read off
        it, so it cannot be given a mechanism and a split that came from different
        constructions.
    guard:
        Which extra equations are solved, in ``drtmle``'s vocabulary and **crossed** the way
        that package crosses it: ``"Q"`` guards against a misspecified outcome regression
        and adds equation (9), which fluctuates ``g``; ``"g"`` guards against a misspecified
        mechanism and adds equation (10), which fluctuates ``Qbar``.  An empty guard is a
        plain TMLE and never reaches here -- a fit that wants one carries no reductions at
        all, which is what makes it bit-for-bit the ordinary path rather than the ordinary
        path recovered by a loop that happens to exit early.
    """

    refit: Callable[[NuisanceEstimates], ReducedSet]
    guard: tuple[str, ...] = ("Q", "g")
    #: Which route through the round the alternation takes -- see :data:`ReductionOrder`.
    #: It rides here rather than being a keyword of :func:`solve_with_reduction` because it
    #: is the *estimator's* declaration, exactly as ``guard`` is, and because that keeps
    #: :meth:`~cleverly.TMLE._solve_reduction` free of a setting only one subclass has.
    order: ReductionOrder = "cleverly"


@dataclass(frozen=True)
class ReductionFluctuation:
    """Equation (10)'s fluctuation, and how the doubly-robust alternation went.

    The third sibling of :class:`~cleverly.fluctuation.mechanism.MechanismFluctuation` and
    :class:`ProjectionFluctuation`, and closest to the first: both are halves of a targeting
    step that has more than one, and both are carried on the outcome fluctuation solved
    beside them.  Equation (9)'s half is a ``MechanismFluctuation`` and lives in
    ``Fluctuation.mechanism``, exactly where ``ipsi`` puts its own, so
    :func:`~cleverly.validation.score_check` reports it with no changes.

    Attributes
    ----------
    bounds:
        The mechanism truncation the two extra covariates divided by.  On record because the
        influence curve is built from these same quantities and must divide by the same
        thing -- a curve reconstructed at a different bound would be a curve for an
        estimator nobody ran.
    reduced:
        The reduced-dimension regressions the equations were finally solved against --
        refitted against the targeted pair, so **not** the ones on
        ``result.nuisance.reduced``, which stay the initial fit exactly as
        ``result.nuisance.propensity`` stays the initial mechanism.  The influence curve
        reads these.
    guard:
        Which extra equations this fluctuation solved, in ``ReductionSpec.guard``'s
        vocabulary, carried on the record because it is also **which correction the
        reported curve subtracts** -- :func:`~cleverly.estimators.tmle.correction_parts`
        reads it off here, so a fit reloaded from disk selects the same terms its
        estimator did.  It was on this record and read by the validation layer alone while
        the curve subtracted both, which was ``docs/roadmap.md``'s item 23.
    epsilon, score, score_scale, score_initial, names:
        Equation (10)'s fluctuation, reported on the same footing as the outcome
        fluctuation's so the two can sit in one table.  ``epsilon`` is the **last round's**
        step and not a running total, exactly as the outcome fluctuation's is in every
        alternation here -- so it is near zero at any converged fixed point and is a poor
        thing to assert "the equation did something" against.  ``score_initial`` is the
        exception and is the **first** round's: a "before" taken from the exiting round
        would be zero for the same reason and would say nothing about what was solved.
    trace:
        One ``(outer, outcome score, reduced score, mechanism score, joint loglik)`` row per
        round.  The joint value is what makes the loop terminate rather than merely settle:
        equation (9) is a weighted logistic MLE of ``A`` given ``W`` and equations (8) and
        (10) are the outcome quasi-likelihood, separate factors of the likelihood of
        ``(A, Y) | W``, so each step maximises its own factor with the other held fixed.
        Refitting the reductions between rounds does not break that -- it changes the
        *direction* of the next submodel, not the value at the point it passes through.
    """

    reduced: ReducedSet
    guard: tuple[str, ...]
    bounds: tuple[float, float]
    epsilon: FloatArray
    score: FloatArray
    score_scale: FloatArray
    score_initial: FloatArray
    names: tuple[str, ...] = ()
    trace: tuple[tuple[int, float, float, float, float], ...] = field(default_factory=tuple)
    #: How many rounds of the alternation ran -- the rounds that **refitted**.  The trace
    #: carries one further row for the closing pass, which refits nothing, so this is what
    #: "did the loop terminate on its own" means and what the outer cap is compared against.
    rounds: int = 0
    converged: bool = True
    failure: str | None = None
    #: Which of the loop's three exits fired.  **Not** a restatement of :attr:`failure`, and
    #: the two disagree in both directions: ``failure`` is set from the *closing pass's*
    #: scores against :data:`_UNSOLVED` and never reads the round count, so a fit that ran
    #: out of rounds but scores well reports ``None`` while one that stalled at round three
    #: with a bad score reports ``"max_iter_reached"`` -- a name that on its own would have a
    #: reader believe the cap was reached.  Which exit fired is a fact about the loop, it is
    #: recoverable from nowhere else on this record, and it is what a claim of the form "such
    #: a fit runs to the outer cap" has to be checked against.
    exit_reason: ReductionExit = "tolerance"
    #: Whether the closing pass's mechanism stage stopped on its step cap rather than on
    #: ``spec.tol``.  Equation (9)'s covariate reads the very mechanism it tilts, so each
    #: solve leaves a residual at the post-tilt covariate that iterating shrinks without
    #: removing; the cap binding is an expected outcome of that rather than a fault.  It is
    #: on the record so that a reader can see the stage stopped counting rather than infer
    #: from :attr:`closing` that it converged.
    closing_capped: bool = False
    #: How many rounds' equation-(10) solves reported a failure of their own.  Not rare and
    #: not a defect in the solver: :math:`g_{r,2}` vanishes exactly where the mechanism is
    #: right, so on any fit whose :math:`\hat g` is nearly right that covariate is nearly
    #: zero and its Hessian near-singular -- measured at ``mean|h| = 1e-3`` with a singular
    #: Hessian in a third of the rounds on a 2000-row ``glm`` fit.  It is why this loop
    #: stops on a statistical tolerance rather than a numerical one, and it is reported so
    #: that a reader can see the equation was hard rather than infer it from the round count.
    ill_conditioned: int = 0
    #: How many solves the closing pass took.  It re-solves all three equations at the
    #: reductions this record carries, which the alternation itself never does -- it solves
    #: at one refit and reports at another.  Zero would mean the pass did not run.
    closing: int = 0

    @property
    def relative_score(self) -> float:
        """Equation (10)'s largest score component relative to its possible magnitude."""
        return relative_score(self.score, self.score_scale)

    @property
    def n_outer(self) -> int:
        """The refitting rounds, not counting the closing pass's row in :attr:`trace`."""
        return self.rounds

    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.names, np.asarray(self.epsilon).tolist(), strict=True))


def needs_reduction(nuisance: NuisanceEstimates, group: TargetGroup) -> bool:
    """Whether this fit's targeting has extra score equations to solve beside the outcome's.

    A property of the *nuisances* rather than of the group, and it has to be: the group is
    still ``"mean"`` -- the report is still ``ey1``, ``ey0`` and ``ate``, a different
    estimator behind the same parameters exactly as :class:`~cleverly.CTMLE` is -- so a
    predicate keyed on the group name alone would divert every ordinary fit in the package.
    """
    return group == "mean" and nuisance.reduced is not None


def solve_with_reduction(
    data: CausalData,
    nuisance: NuisanceEstimates,
    group: TargetGroup,
    spec: TargetingSpec,
    *,
    reduction: ReductionSpec,
    bounds: tuple[float, float],
    nuisance_bound: float,
    scaled: FloatArray,
    weights: FloatArray,
    observed: BoolArray,
    max_outer: int = 50,
    warn: bool = True,
) -> tuple[Submodel, Fluctuation]:
    r"""Solve the outcome equation and the two extra ones together, alternating until all settle.

    A plain TMLE solves equation (8) and stops.  Doubly-robust inference (van der Laan 2014;
    Benkeser, Carone, van der Laan & Gilbert 2017) adds

    .. math::

        (9)  \quad & P_n[\, Q_r(a, W)/g^*(a|W)\,\{1_a - g^*(a|W)\}\,] = 0 \\
        (10) \quad & P_n[\, 1_a\,g_{r,2}(a|W)/g_{r,1}(a|W)\,\{Y - \bar Q^*(a, W)\}\,] = 0

    and none of the three can be solved once and left: (9) fluctuates :math:`g` along a
    covariate reading the very :math:`g^*` it moves, (8) and (10) fluctuate :math:`\bar Q`
    along covariates reading :math:`g^*` and the reductions, and the reductions are
    themselves regressions *on* the current pair.

    .. code-block:: text

        prime:  Qbar* <- fluctuation of Qbar^0 along 1_a/g          (equation 8)
        repeat:
            if "Q" in guard:  g* <- logistic tilt along Qr/g*       (equation 9)
                              refit Qr, gr1, gr2 at (Qbar*, g*)
            if "g" in guard:  Qbar* <- fluctuate along gr2/gr1      (equation 10)
            Qbar* <- fluctuate along 1_a/g*                         (equation 8)
            re-evaluate all three scores at the pair the round exits at

    **That is one of two routes to the same stated exit, and the other is the working
    paper's own.**  ``reduction.order`` selects between them -- see :data:`ReductionOrder`.
    The paper's step 7 states its termination as the three empirical means being
    approximately zero, so its six-step order is one way of reaching a fixed point rather
    than something Theorem 1 assumes about the collection returned; ``"paper"`` implements
    it beside this one so that *whether the two reach the same fixed point on real data* is
    a run rather than an argument (``docs/roadmap.md``'s item 22).  What the second route
    does **not** get is a second stopping rule, a second stall test or a second closing pass:
    it shares all three, because the question is the route and a comparison in which two
    things differ answers nothing.

    Three things about that shape are decisions rather than transcription.

    **The outcome fluctuation continues from** :math:`\bar Q^*` **rather than restarting from**
    :math:`\bar Q^0`, which is :func:`solve_with_mechanism`'s choice and not
    :func:`solve_with_projection`'s.  The argument is the one
    :mod:`cleverly.fluctuation.mechanism` writes out and it carries over: equation (9) is a
    weighted logistic MLE of :math:`A \mid W` and equations (8) and (10) are the outcome
    quasi-likelihood, separate factors of the likelihood of :math:`(A, Y) \mid W`, so each
    step maximises its own factor with the others held fixed and the joint value never
    decreases.  Restarting would break the monotonicity and with it the reason this
    terminates.

    **Equations (8) and (10) are solved one after the other rather than as one wider
    submodel**, which is ``drtmle``'s ``Qsteps = 2`` -- backfitting, "found to be more
    stable in simulations".  It is also what keeps a ``Submodel`` column belonging to one
    arm, which :mod:`cleverly.sensitivity.omitted_variable` reads.

    **Every score in sight goes stale, and the convergence test is taken after the round
    rather than during it.**  The mechanism covariate reads :math:`\bar Q^*`, which the two
    outcome steps then move; equation (10)'s covariate is fixed within a round but its
    residual is not, since equation (8) fluctuates :math:`\bar Q` afterwards.  A loop
    testing the scores where they were solved would exit having solved one equation and
    left another open -- the failure :func:`~cleverly.fluctuation.mechanism.mechanism_score`
    exists for, here three times over.

    **The loop is followed by a closing pass, and it is not an optimisation.**  Every round
    solves equation (10) at that round's *first* refit of the reductions and equation (9) at
    the *previous* round's second, then refits again before the record is built --
    ``drtmle``'s ordering, kept.  So neither extra equation is solved at the arrays the
    reported curve is built from, and that curve's empirical mean is zero only insofar as
    the loop converged.  Mean zero is what makes the estimator asymptotically linear with
    the curve it reports: on a 800-row fit stopped after one round the largest reported
    curve mean was ``3.7e-3`` against a standard error of ``0.105``.
    :func:`_close_at_frozen_reductions` re-solves all three at the reductions the record
    carries, which brings that to ``5.8e-7``, and moves a converged fit's ``psi`` and
    standard error by nothing.

    **This loop does not reliably converge, and the reason is structural rather than
    numerical.**  Equation (10)'s covariate is :math:`g_{r,2}/g_{r,1}`, and :math:`g_{r,2}`
    vanishes exactly where the mechanism is right -- so on a fit whose :math:`\hat g` is
    nearly right, which is every fit anybody wants, that covariate is nearly zero and its
    Newton solve is near-singular: observed at ``mean|h| = 1e-3``, ``|epsilon|`` reaching
    280 and a singular Hessian in a third of the rounds on one unseeded draw.  Such a fit
    exits at ``max_outer`` and reports ``failure = "max_iter_reached"``.  ``drtmle``
    sidesteps the question entirely by capping at three iterations and never claiming to
    converge.  :attr:`ReductionFluctuation.ill_conditioned` reports the conditioning;
    ``docs/roadmap.md`` lists this under *What is still open*, and
    :class:`~cleverly.DRTMLE`'s module docstring says what turns on it.

    **Swept over 96 fits** -- four processes by two sizes by twelve seeds, the table under
    *How the alternation exits* in ``docs/drtmle-investigation-log.md`` -- which replaced the
    six-fit claim that stood here.  It
    said the loop "converged in 15 to 45 rounds" on six seeded draws of *one* process, and
    so ran to the cap only on a minority.  Running out of rounds is a minority (8 of 96),
    but converging is rarer: **2 of 96 reached the tolerance and 86 stalled**.  The
    conditioning is also worst on ``linear`` -- 5 of 12 draws at ``n = 600`` and 9 of 12 at
    ``n = 1,200``, against 0 of 12 for ``nonlinear`` -- which is what "vanishes where the
    mechanism is right" predicts, the easy process being the ill-conditioned one.

    The exit test used to be a *relative* score alone, dividing by a ``mean|h|`` of order
    ``1e-3`` and so reading an absolutely negligible score as a large one; :data:`_NEGLIGIBLE`
    and :func:`_solved` are what that became and say why.

    Returns the final outcome submodel and the equation-(8) fluctuation, carrying equation
    (9)'s tilt on :attr:`~cleverly.fluctuation.Fluctuation.mechanism` and equation (10)'s on
    ``.reduction``.  **Two values rather than** :func:`solve_with_mechanism`'s **three**:
    that one re-derives its nuisances because ``ipsi``'s estimand is a functional of
    :math:`g`, and this estimand is :math:`E[\bar Q^*(a, W)]`, which reads no mechanism at
    all.  Nothing the alternation moves belongs on ``result.nuisance``.
    """
    if nuisance.reduced is None:
        raise ValueError(
            f"group {group!r} solves the doubly-robust score equations, so the fit must "
            "carry the reduced-dimension regressions that define them; this "
            "NuisanceEstimates has none"
        )
    guard = tuple(reduction.guard)
    order = reduction.order
    if not guard:
        raise ValueError(
            "an empty guard solves no extra equation and is a plain TMLE; such a fit must "
            "carry no reduced regressions at all rather than reach this alternation"
        )
    arms = nuisance.arms
    if len(arms) != 2:
        raise ValueError(f"the doubly-robust equations are derived for two arms; got {list(arms)}")
    upper = arms[1]
    indicator = (np.asarray(data.treatment, dtype=float) == float(upper)).astype(float)
    mask = np.asarray(observed, dtype=bool)

    reduced = nuisance.reduced
    targeted_g = nuisance.propensity.arm(upper)
    current = nuisance
    submodel = build_submodel(data, current, group, bounds=bounds, nuisance_bound=nuisance_bound)
    fluctuation = solve_submodel(
        scaled, nuisance.outcome, submodel, weights, observed, spec, warn=warn
    )

    mechanism: MechanismFluctuation | None = None
    extra: Fluctuation | None = None
    extra_submodel: Submodel | None = None
    # Equation (10)'s violation before any round, kept from the first solve rather than the
    # last. Every other field here is the exiting round's, which is right for a score and
    # wrong for a "before": at a converged fixed point the last round starts where it ends,
    # so the last round's `score_initial` is zero and says nothing about what was solved.
    first_initial: FloatArray | None = None
    ill_conditioned = 0
    trace: list[tuple[int, float, float, float, float]] = []
    previous: float | None = None
    previous_joint: float | None = None
    # The cap is what happens when neither break below fires, so it is the starting value
    # rather than a fourth case to detect after the loop.
    exit_reason: ReductionExit = "cap"

    # The latest targeted outcome regression, which is not always ``fluctuation.targeted``:
    # under the paper's order equation (10) is solved *after* equation (8), so the round
    # ends on ``extra``'s. One variable rather than the two orders each reading a different
    # field, since the tail below and the closing pass both need "the current Qbar".
    targeted_q = fluctuation.targeted

    for outer in range(1, max_outer + 1):
        if order == "paper":
            # Steps 2 to 6 of the working paper's recursion, in its order --
            # `docs/drtmle-theorem-concordance.md` §6. The two refits are the paper's own
            # steps 3 and 5 and they are what the order is *about*: the reductions are
            # taken at two different vintages of the outcome regression, the mechanism
            # half at the once-updated one and Qr at the twice-updated one, where this
            # package's order refits all three together twice a round.
            submodel = build_submodel(
                data, current, group, bounds=bounds, nuisance_bound=nuisance_bound
            )
            fluctuation = solve_submodel(  # step 2: equation (8), along H_1(g^k)
                scaled, targeted_q, submodel, weights, observed, spec, warn=False
            )
            targeted_q = fluctuation.targeted
            if "g" in guard:
                # Step 3, at g^k and the once-updated Qbar. `replace` rather than the whole
                # set: Qr is step 5's and must not arrive early, which is the difference
                # between the two orders rather than an optimisation.
                once = reduction.refit(_reduction_inputs(current, targeted_q, targeted_g))
                reduced = replace(reduced, gr1=once.gr1, gr2=once.gr2)
                extra_submodel = reduced_outcome_submodel(data.treatment, reduced, bounds=bounds)
                extra = solve_submodel(  # step 4: equation (10), along H_2
                    scaled, targeted_q, extra_submodel, weights, observed, spec, warn=False
                )
                targeted_q = extra.targeted
                if first_initial is None:
                    first_initial = np.asarray(extra.score_initial)
                if extra.failure is not None:
                    ill_conditioned += 1
            if "Q" in guard:
                twice = reduction.refit(  # step 5, at g^k and the twice-updated Qbar
                    _reduction_inputs(current, targeted_q, targeted_g)
                )
                reduced = replace(reduced, qr=twice.qr)
                mechanism = solve_bounded_mechanism(  # step 6: equation (9), along H_3
                    indicator,
                    targeted_g,
                    reduced_mechanism_covariate(reduced, targeted_g, bounds=bounds),
                    weights,
                    bounds=bounds,
                    tol=spec.tol,
                )
                targeted_g = mechanism.propensity
                current = _retargeted_mechanism(nuisance, targeted_g, arms)
        else:
            if "Q" in guard:
                mechanism = solve_bounded_mechanism(
                    indicator,
                    targeted_g,
                    reduced_mechanism_covariate(reduced, targeted_g, bounds=bounds),
                    weights,
                    bounds=bounds,
                    tol=spec.tol,
                )
                # The **truncated** tilt, which is what makes the next round's offset, every
                # later covariate and the reported correction read one array. Carrying the
                # raw one forward is what left a clipped row outside the bounds for the rest
                # of the fit, and it was the load-bearing half of item 20.
                targeted_g = mechanism.propensity
                current = _retargeted_mechanism(nuisance, targeted_g, arms)
                reduced = reduction.refit(_reduction_inputs(current, targeted_q, targeted_g))

            if "g" in guard:
                extra_submodel = reduced_outcome_submodel(data.treatment, reduced, bounds=bounds)
                extra = solve_submodel(
                    scaled, targeted_q, extra_submodel, weights, observed, spec, warn=False
                )
                if first_initial is None:
                    first_initial = np.asarray(extra.score_initial)
                if extra.failure is not None:
                    ill_conditioned += 1

            submodel = build_submodel(
                data, current, group, bounds=bounds, nuisance_bound=nuisance_bound
            )
            fluctuation = solve_submodel(
                scaled,
                targeted_q if extra is None else extra.targeted,
                submodel,
                weights,
                observed,
                spec,
                warn=False,
            )
            targeted_q = fluctuation.targeted
            if "Q" in guard:
                # Qr is a regression of the outcome residual, so the step just taken moved
                # its target. Refit before the score below is read, or the loop tests
                # equation (9) at a covariate the exiting pair no longer implies.
                reduced = reduction.refit(_reduction_inputs(current, targeted_q, targeted_g))

        # Equation (8)'s score, at the pair the round *exits* at rather than the pair it was
        # solved at. Under this package's order those are the same state and this is a
        # bit-for-bit no-op; under the paper's, equation (8) is solved first and steps 4 and
        # 6 then move both the regression it fluctuated and the mechanism it divides by. One
        # call rather than a branch, so that the docstring's "re-evaluate all three scores at
        # the pair the round exits at" is structurally true of all three rather than true of
        # equation (8) by accident of where it sits.
        submodel = build_submodel(
            data, current, group, bounds=bounds, nuisance_bound=nuisance_bound
        )
        fluctuation = _restated_outcome_score(
            fluctuation,
            scaled=scaled,
            targeted=targeted_q,
            submodel=submodel,
            weights=weights,
            mask=mask,
        )

        reduced_score = 0.0
        reduced_absolute = 0.0
        if extra_submodel is not None:
            # At the *final* reductions, not the ones equation (10) was solved along. The
            # influence curve reads `reduction.reduced`, so a score taken at any other set
            # would report an equation the reported curve does not contain -- which is how
            # the per-estimand row of `score_check` came to disagree with the per-equation
            # rows by two orders of magnitude before this was written down.
            extra_submodel = reduced_outcome_submodel(data.treatment, reduced, bounds=bounds)
            settled = score_columns(
                scaled, targeted_q.observed, extra_submodel.observed, weights, mask
            )
            scale = score_scale(extra_submodel.observed, weights, mask)
            assert extra is not None
            extra = replace(extra, score=settled, score_scale=scale)
            reduced_score = relative_score(settled, scale)
            reduced_absolute = float(np.max(np.abs(settled))) if settled.size else 0.0

        mechanism_relative = 0.0
        mechanism_absolute = 0.0
        if mechanism is not None:
            settled_g, scale_g = mechanism_score(
                indicator,
                targeted_g,
                reduced_mechanism_covariate(reduced, targeted_g, bounds=bounds),
                weights,
            )
            mechanism = replace(mechanism, score=settled_g, score_scale=scale_g)
            mechanism_relative = mechanism.relative_score
            mechanism_absolute = float(np.max(np.abs(settled_g))) if settled_g.size else 0.0

        joint = float(fluctuation.loglik) + float(
            0.0 if mechanism is None else (mechanism.loglik or 0.0)
        )
        worst = max(fluctuation.relative_score_norm, reduced_score, mechanism_relative)
        trace.append(
            (outer, fluctuation.relative_score_norm, reduced_score, mechanism_relative, joint)
        )
        # Each equation is judged on whichever of the two rulers it can actually meet, and
        # all three get the same pair rather than equation (10) getting a special case:
        # measured on a 400-row fit, the round the loop gave up on had equation (10) at
        # 2.3e-8 relative *and* equation (9) at 3.9e-8, so relaxing one alone changes
        # nothing. `worst` stays the *relative* max above, because it is what the stall
        # rule measures progress with and a mixed quantity would make "improving" mean two
        # things on alternate rounds.
        negligible = _NEGLIGIBLE / float(data.n)
        if (
            _solved(fluctuation.relative_score_norm, fluctuation.score_norm, spec.tol, negligible)
            and _solved(reduced_score, reduced_absolute, spec.tol, negligible)
            and _solved(mechanism_relative, mechanism_absolute, spec.tol, negligible)
        ):
            exit_reason = "tolerance"
            break
        # The stall rule watches the *objective as well as* the score, which is where this
        # loop parts company with `solve_with_mechanism`. There the mechanism score falls by
        # a roughly constant factor every round; here three coupled equations make it
        # non-monotone for the first few -- measured on a 600-row `glm` fit, the mechanism
        # score went 2.8e-2, 2.9e-2, 1.7e-2, 1.8e-2 before descending cleanly to 7e-9, while
        # the joint likelihood rose at every one of those rounds. A score-only rule stopped
        # that fit at round 2 with two equations open and reported the interval anyway. So
        # stall only when *neither* has moved: the objective is what the coordinate ascent
        # is climbing, and the score is what the exit is about.
        climbing = previous_joint is None or joint > previous_joint + spec.tol * (1.0 + abs(joint))
        improving = previous is None or worst <= _STALL_FACTOR * previous
        if not climbing and not improving:
            exit_reason = "stall"
            break
        previous = worst if previous is None else min(previous, worst)
        previous_joint = joint

    rounds = len(trace)
    if order == "paper":
        # The paper's round ends on equation (10)'s update of Qbar, so the state the loop
        # left is `targeted_q` and not the outcome fluctuation's own field. The closing pass
        # starts from `fluctuation.targeted`, so hand it the former. Guarded on the order
        # rather than done unconditionally because under the other the two are one object
        # and the default path is a regression surface.
        fluctuation = replace(fluctuation, targeted=targeted_q)
    closing = _close_at_frozen_reductions(
        data,
        nuisance,
        group,
        spec,
        reduced=reduced,
        guard=guard,
        bounds=bounds,
        nuisance_bound=nuisance_bound,
        scaled=scaled,
        weights=weights,
        observed=observed,
        mask=mask,
        indicator=indicator,
        arms=arms,
        targeted_g=targeted_g,
        fluctuation=fluctuation,
        mechanism=mechanism,
        extra=extra,
    )
    submodel = closing.submodel
    fluctuation = closing.fluctuation
    mechanism = closing.mechanism
    extra = closing.extra
    extra_submodel = closing.extra_submodel
    trace.append(
        (
            rounds + 1,
            fluctuation.relative_score_norm,
            closing.reduced_score,
            closing.mechanism_score,
            closing.joint,
        )
    )

    worst = max(fluctuation.relative_score_norm, closing.reduced_score, closing.mechanism_score)
    unsolved: TargetingFailure | None = None if worst <= _UNSOLVED else "max_iter_reached"
    failure = unsolved
    if mechanism is not None:
        mechanism = replace(
            mechanism,
            trace=tuple((row[0], row[1], row[3], row[4]) for row in trace),
            failure=mechanism.failure or unsolved,
        )
    record = ReductionFluctuation(
        reduced=reduced,
        guard=guard,
        bounds=(float(bounds[0]), float(bounds[1])),
        epsilon=np.zeros(0) if extra is None else extra.epsilon,
        score=np.zeros(0) if extra is None else extra.score,
        score_scale=np.zeros(0) if extra is None else np.asarray(extra.score_scale),
        score_initial=np.zeros(0) if first_initial is None else first_initial,
        names=() if extra_submodel is None else extra_submodel.names,
        trace=tuple(trace),
        rounds=rounds,
        converged=bool(worst <= spec.tol),
        failure=failure,
        exit_reason=exit_reason,
        ill_conditioned=ill_conditioned,
        closing=closing.steps,
        closing_capped=closing.capped,
    )
    return submodel, replace(fluctuation, mechanism=mechanism, reduction=record)


def _restated_outcome_score(
    fluctuation: Fluctuation,
    *,
    scaled: FloatArray,
    targeted: InitialFit,
    submodel: Submodel,
    weights: FloatArray,
    mask: BoolArray,
) -> Fluctuation:
    r"""Equation (8)'s score, re-read at the pair the round exits at rather than solved at.

    The other two equations are already restated in the round's tail -- equation (10) at the
    final reductions and equation (9) at the final mechanism -- and this makes the third the
    same, so that :func:`solve_with_reduction`'s *"re-evaluate all three scores at the pair
    the round exits at"* is a property of the loop rather than of where equation (8) happens
    to sit in it.

    **Called unconditionally, and under the default order that is a bit-for-bit no-op.**
    :func:`~cleverly.fluctuation.iterative.solve_fluctuation` computes the ``score`` it
    returns *after* its loop, by this very expression at the iterate it returns as
    ``.targeted`` (and :func:`~cleverly.fluctuation.one_step.solve_one_step` and the linear
    branch do the same) -- so re-evaluating it at the same submodel, weights and mask returns
    the identical float64 array.  Under this package's order equation (8) is solved last and
    nothing moves after it, so the restatement recovers what was already there; under the
    paper's it is solved **first** and steps 4 and 6 then move both the regression it
    fluctuated and the mechanism its covariate divides by.

    That the two cases can share one call is what removes the hazard rather than guarding it.
    While this was conditional on the order, deleting it was a change **no assertion about a
    reported fit could see**: it was run, and 68 of ``tests/unit/test_drtmle_fit.py``'s 69
    tests still passed, because :func:`_close_at_frozen_reductions` re-solves all three
    equations at the reductions the record carries and makes the reported fit identical
    either way.  That is ``docs/roadmap.md`` item 12's shape in a second place, and it is
    lesson 12 of the investigation log: the closing pass is an anaesthetic, so a defect in
    how the loop *exits* has to be caught at the loop rather than at the fit.  What keeps the
    no-op a no-op is
    ``tests/unit/test_fluctuation_score.py``, which pins the solver-side identity this rests
    on -- a mid-loop score recorded in place of the post-loop one would break it, and that
    mutation is one line away in :mod:`cleverly.fluctuation.iterative`.

    **Not applicable to the closing pass.**  That stage builds its fluctuation from a
    *four-column joint* solve over equations (8) and (10) and then overwrites ``score`` with
    the two-column submodel's, so its ``score`` is deliberately not this expression at the
    submodel the solve ran along, and restating it there would change a number rather than
    recover one.
    """
    return replace(
        fluctuation,
        score=score_columns(scaled, targeted.observed, submodel.observed, weights, mask),
        score_scale=score_scale(submodel.observed, weights, mask),
    )


@dataclass(frozen=True)
class _Closing:
    """What the closing pass left behind, so the caller assembles one record from one place."""

    submodel: Submodel
    fluctuation: Fluctuation
    mechanism: MechanismFluctuation | None
    extra: Fluctuation | None
    extra_submodel: Submodel | None
    reduced_score: float
    mechanism_score: float
    joint: float
    steps: int
    capped: bool


def _close_at_frozen_reductions(
    data: CausalData,
    nuisance: NuisanceEstimates,
    group: TargetGroup,
    spec: TargetingSpec,
    *,
    reduced: ReducedSet,
    guard: tuple[str, ...],
    bounds: tuple[float, float],
    nuisance_bound: float,
    scaled: FloatArray,
    weights: FloatArray,
    observed: BoolArray,
    mask: BoolArray,
    indicator: FloatArray,
    arms: tuple[float, ...],
    targeted_g: FloatArray,
    fluctuation: Fluctuation,
    mechanism: MechanismFluctuation | None,
    extra: Fluctuation | None,
    max_steps: int = 20,
) -> _Closing:
    r"""Re-solve the three equations at the reductions the influence curve will read.

    The alternation solves equation (9) at the *previous* round's reductions and equation
    (10) at the current round's *first* refit, and then refits once more before the record
    is built -- ``drtmle``'s ordering, kept.  So neither extra equation is solved at the
    arrays the reported curve is built from, and the curve's empirical mean is zero only
    insofar as the loop converged.  Mean zero is the property the whole estimator rests on,
    which is why "insofar as it converged" is not good enough: measured on a 800-row fit
    stopped after one round, the largest reported curve mean was ``3.7e-3`` against a
    standard error of ``0.105``, and nothing in the output said so.

    **Freezing the reductions makes the system triangular, and that is why this is two
    stages rather than another alternation.**  With :math:`Q_r`, :math:`g_{r,1}` and
    :math:`g_{r,2}` held fixed:

    .. code-block:: text

        (9)   Qr(a,W)/g*(a|W) (1_a - g*(a|W))       reads g* only -- no Qbar anywhere
        (10)  1_a gr2(a|W)/gr1(a|W) (Y - Qbar*)     reads Qbar* only -- no g
        (8)   1_a/g*(a|W),  residual (Y - Qbar*)    reads both

    So equation (9) is solved first and nothing downstream can disturb it -- moving
    :math:`\bar Q^*` cannot change a term :math:`\bar Q` does not appear in -- and then (8)
    and (10) are solved for :math:`\bar Q^*` at that fixed :math:`g^*`, **jointly**, in one
    Newton solve over all four columns rather than by backfitting them.  Backfitting them
    converges at a rate set by how collinear the two covariates are, and that rate is not
    always usable: measured on the exact law, twenty backfitting steps left equation (10)
    at ``3.9e-4`` where one joint solve lands at ``1.5e-12``.  The *returned* submodel is
    still the two-column ``mean`` one -- ``sensitivity/omitted_variable.py`` reads
    :meth:`~cleverly.fluctuation.submodel.Submodel.column_for`, and a four-column submodel
    must not escape this function.

    No learner is refitted here, so the pass costs arithmetic; the whole point is that the
    reductions do *not* move.

    **What it does not achieve, and the docstring says so rather than the reader
    discovering it.**  Equation (9)'s covariate is :math:`Q_r/g^*`: it reads the very
    mechanism it tilts, so one :func:`~cleverly.fluctuation.mechanism.solve_mechanism` call
    zeroes the score at the covariate built from the *pre-tilt* :math:`g^*` and leaves a
    residual at the post-tilt one.  Iterating shrinks that; it does not remove it.  Whether
    that stage stopped on ``max_steps`` or on the tolerance is reported as ``capped``, so
    that a reader is not left to infer convergence from a step count that has no other way
    of saying which it was.

    **"All three equations are solved at the arrays the curve is built from" is true, and it
    is worth recording that it was false until piece B1b.**  The arrays were always the
    same arrays; the *expressions* were not.  This stage used to solve
    :math:`P_n[H_g (A - g^*)] = 0` at the **raw** tilted mechanism, while
    :func:`~cleverly.inference.influence.reduced_corrections` truncates :math:`g^*` inside
    its residual as well as in its denominator -- so the two coincided on every row the
    truncation left alone and differed on every row it clipped, and one clipped row of 600
    was enough to leave the reported curve uncentred by ``5.8e-4`` while this stage recorded
    ``8e-11``.  That was item 20 in ``docs/roadmap.md``, it accounted for item 11 as well,
    and the measurements are in ``docs/drtmle-investigation-log.md``.

    :func:`~cleverly.fluctuation.mechanism.solve_bounded_mechanism` is what closed it: it
    solves the score at the **truncated** tilt, which is the expression the reported curve
    carries, and this stage carries that truncated array forward.  Two things about the fix
    are easy to undo and both were measured before being written down.  Carrying
    ``mechanism.propensity`` forward is not incidental -- the raw array's clipped rows used
    to stay outside the bounds for the rest of the fit, which is *how* the disagreement
    persisted -- and clipping *after* an unconstrained solve is not the same fix: it makes
    the two expressions agree while solving neither, at ``6.8e-06`` against a bar near
    ``4e-06`` where the bound binds at the fixed point.

    :func:`~cleverly.validation.drtmle.correction_check` is the instrument that showed it,
    and it stays: it recomputes the mean of the term the curve carries from the state this
    function returns and reports its difference from the score recorded here, per arm.  What
    changed is that the difference is now zero to roundoff rather than reported as itself.
    """
    steps = 0
    if "Q" in guard:
        for _ in range(max_steps):
            steps += 1
            solved = solve_bounded_mechanism(
                indicator,
                targeted_g,
                reduced_mechanism_covariate(reduced, targeted_g, bounds=bounds),
                weights,
                bounds=bounds,
                tol=spec.tol,
            )
            targeted_g = solved.propensity
            settled, scale = mechanism_score(
                indicator,
                targeted_g,
                reduced_mechanism_covariate(reduced, targeted_g, bounds=bounds),
                weights,
            )
            mechanism = replace(solved, score=settled, score_scale=scale)
            if mechanism.relative_score <= spec.tol:
                break

    # The stage ran out of steps exactly when it left the score above tolerance, so this is
    # read off the score rather than off which way the loop above ended -- one statement of
    # what happened rather than two that could drift apart.
    capped = "Q" in guard and mechanism is not None and mechanism.relative_score > spec.tol

    current = _retargeted_mechanism(nuisance, targeted_g, arms) if "Q" in guard else nuisance
    submodel = build_submodel(data, current, group, bounds=bounds, nuisance_bound=nuisance_bound)
    extra_submodel: Submodel | None = None
    reduced_score = 0.0

    if "g" not in guard:
        steps += 1
        fluctuation = solve_submodel(
            scaled, fluctuation.targeted, submodel, weights, observed, spec, warn=False
        )
        return _Closing(
            submodel=submodel,
            fluctuation=fluctuation,
            mechanism=mechanism,
            extra=extra,
            extra_submodel=None,
            reduced_score=0.0,
            mechanism_score=0.0 if mechanism is None else mechanism.relative_score,
            joint=float(fluctuation.loglik)
            + float(0.0 if mechanism is None else (mechanism.loglik or 0.0)),
            steps=steps,
            capped=capped,
        )

    extra_submodel = reduced_outcome_submodel(data.treatment, reduced, bounds=bounds)
    # **Jointly**, not by backfitting them. Equations (8) and (10) fluctuate the same
    # `Qbar` along two covariates, so one Newton solve over all four columns drives all
    # four scores to machine precision at once, where alternating them converges at a rate
    # set by how collinear the covariates are -- measured on the exact law at 3.9e-4 after
    # twenty backfitting steps against 1.5e-12 from one joint solve. The *returned*
    # submodel stays the two-column `mean` one either way: `column_for` is what
    # `sensitivity/omitted_variable.py` reads, and a four-column submodel must not escape.
    steps += 1
    joint = solve_submodel(
        scaled,
        fluctuation.targeted,
        _stacked(submodel, extra_submodel),
        weights,
        observed,
        spec,
        warn=False,
    )
    width = submodel.dim
    settled = score_columns(scaled, joint.targeted.observed, submodel.observed, weights, mask)
    settled_q = score_columns(
        scaled, joint.targeted.observed, extra_submodel.observed, weights, mask
    )
    scale_q = score_scale(extra_submodel.observed, weights, mask)
    fluctuation = replace(
        joint,
        epsilon=joint.epsilon[:width],
        names=submodel.names,
        score=settled,
        score_scale=score_scale(submodel.observed, weights, mask),
    )
    extra = replace(
        joint,
        epsilon=joint.epsilon[width:],
        names=extra_submodel.names,
        score=settled_q,
        score_scale=scale_q,
    )
    reduced_score = relative_score(settled_q, scale_q)

    return _Closing(
        submodel=submodel,
        fluctuation=fluctuation,
        mechanism=mechanism,
        extra=extra,
        extra_submodel=extra_submodel,
        reduced_score=reduced_score,
        mechanism_score=0.0 if mechanism is None else mechanism.relative_score,
        joint=float(fluctuation.loglik)
        + float(0.0 if mechanism is None else (mechanism.loglik or 0.0)),
        steps=steps,
        capped=capped,
    )


def _retargeted_mechanism(
    nuisance: NuisanceEstimates, targeted: FloatArray, arms: tuple[float, ...]
) -> NuisanceEstimates:
    """``nuisance`` with the mechanism replaced by the tilted one, for the covariate only.

    Built here and thrown away with the alternation: the targeted mechanism belongs on the
    fluctuation, never on ``result.nuisance``, so that the nuisance diagnostics go on
    describing the model that was fitted.  The complement form is
    :meth:`~cleverly.estimators._nuisance.Propensity.bounded`'s two-arm rule arriving one
    step earlier -- the tilt moves one probability and the other arm is its complement.
    """
    g1 = np.asarray(targeted, dtype=float).reshape(-1)
    return replace(nuisance, propensity=Propensity(np.column_stack([1.0 - g1, g1]), arms))


def _reduction_inputs(
    nuisance: NuisanceEstimates, targeted: InitialFit, mechanism: FloatArray
) -> NuisanceEstimates:
    """The nuisances a refit of the reduced regressions is taken *relative to*.

    The targeted pair rather than the initial one, which is the whole of what makes this an
    alternation rather than three equations solved at arrays fixed in advance.  ``folds``,
    the scaler and the weights travel unchanged, so the refit is out of fold on the same
    split the primary fits used.
    """
    del mechanism  # already written onto `nuisance` by `_retargeted_mechanism`
    return replace(nuisance, outcome=targeted)


def _stacked(first: Submodel, second: Submodel) -> Submodel:
    """The two outcome covariates side by side, for one Newton solve over both equations.

    Used only inside :func:`_close_at_frozen_reductions` and never returned: it has four
    columns, and :attr:`~cleverly.fluctuation.submodel.Submodel.arm_columns` on a wider
    submodel would say something ``sensitivity/omitted_variable.py``'s Riesz representer
    does not expect.  The arm mapping is carried over from ``first`` so the object is
    well formed; nothing reads it here.
    """
    return Submodel(
        np.hstack([first.observed, second.observed]),
        {arm: np.hstack([first.arms[arm], second.arms[arm]]) for arm in first.arms},
        first.names + second.names,
        first.group,
        dict(first.arm_columns),
        dict(first.contrast_columns),
    )

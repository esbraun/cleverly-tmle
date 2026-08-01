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
    "ReductionFluctuation",
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
    converged: bool = True
    failure: str | None = None
    #: How many rounds' equation-(10) solves reported a failure of their own.  Not rare and
    #: not a defect in the solver: :math:`g_{r,2}` vanishes exactly where the mechanism is
    #: right, so on any fit whose :math:`\hat g` is nearly right that covariate is nearly
    #: zero and its Hessian near-singular -- measured at ``mean|h| = 1e-3`` with a singular
    #: Hessian in a third of the rounds on a 2000-row ``glm`` fit.  It is why this loop
    #: stops on a statistical tolerance rather than a numerical one, and it is reported so
    #: that a reader can see the equation was hard rather than infer it from the round count.
    ill_conditioned: int = 0

    @property
    def relative_score(self) -> float:
        """Equation (10)'s largest score component relative to its possible magnitude."""
        return relative_score(self.score, self.score_scale)

    @property
    def n_outer(self) -> int:
        return len(self.trace)

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

    for outer in range(1, max_outer + 1):
        if "Q" in guard:
            mechanism = solve_mechanism(
                indicator,
                targeted_g,
                reduced_mechanism_covariate(reduced, targeted_g, bounds=bounds),
                weights,
                tol=spec.tol,
            )
            targeted_g = mechanism.propensity
            current = _retargeted_mechanism(nuisance, targeted_g, arms)
            reduced = reduction.refit(_reduction_inputs(current, fluctuation.targeted, targeted_g))

        if "g" in guard:
            extra_submodel = reduced_outcome_submodel(data.treatment, reduced, bounds=bounds)
            extra = solve_submodel(
                scaled, fluctuation.targeted, extra_submodel, weights, observed, spec, warn=False
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
            fluctuation.targeted if extra is None else extra.targeted,
            submodel,
            weights,
            observed,
            spec,
            warn=False,
        )
        if "Q" in guard:
            # Qr is a regression of the outcome residual, so the step just taken moved its
            # target. Refit before the score below is read, or the loop tests equation (9)
            # at a covariate the exiting pair no longer implies.
            reduced = reduction.refit(_reduction_inputs(current, fluctuation.targeted, targeted_g))

        reduced_score = 0.0
        if extra_submodel is not None:
            # At the *final* reductions, not the ones equation (10) was solved along. The
            # influence curve reads `reduction.reduced`, so a score taken at any other set
            # would report an equation the reported curve does not contain -- which is how
            # the per-estimand row of `score_check` came to disagree with the per-equation
            # rows by two orders of magnitude before this was written down.
            extra_submodel = reduced_outcome_submodel(data.treatment, reduced, bounds=bounds)
            settled = score_columns(
                scaled, fluctuation.targeted.observed, extra_submodel.observed, weights, mask
            )
            scale = score_scale(extra_submodel.observed, weights, mask)
            assert extra is not None
            extra = replace(extra, score=settled, score_scale=scale)
            reduced_score = relative_score(settled, scale)

        mechanism_relative = 0.0
        if mechanism is not None:
            settled_g, scale_g = mechanism_score(
                indicator,
                targeted_g,
                reduced_mechanism_covariate(reduced, targeted_g, bounds=bounds),
                weights,
            )
            mechanism = replace(mechanism, score=settled_g, score_scale=scale_g)
            mechanism_relative = mechanism.relative_score

        joint = float(fluctuation.loglik) + float(
            0.0 if mechanism is None else (mechanism.loglik or 0.0)
        )
        worst = max(fluctuation.relative_score_norm, reduced_score, mechanism_relative)
        trace.append(
            (outer, fluctuation.relative_score_norm, reduced_score, mechanism_relative, joint)
        )
        if worst <= spec.tol:
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
            break
        previous = worst if previous is None else min(previous, worst)
        previous_joint = joint

    last = trace[-1] if trace else (0, 0.0, 0.0, 0.0, 0.0)
    worst = max(last[1], last[2], last[3])
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
        converged=bool(worst <= spec.tol),
        failure=failure,
        ill_conditioned=ill_conditioned,
    )
    return submodel, replace(fluctuation, mechanism=mechanism, reduction=record)


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

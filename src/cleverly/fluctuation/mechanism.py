r"""Fluctuating the treatment mechanism, for parameters that are defined through it.

Every other fluctuation in this package moves the outcome regression.  That is enough
whenever the target parameter is a functional of :math:`\bar Q` and the covariate
distribution alone: the efficient influence function then lives entirely in the tangent
space those two span, and one submodel through :math:`\bar Q^0` reaches all of it.

An :class:`~cleverly.interventions.Incremental` intervention is not such a parameter.
Its :math:`q_\delta = \delta g / (\delta g + 1 - g)` is built out of the mechanism, so

.. math::

    \Psi(\delta) = E\!\left[\frac{\delta g \bar Q(1, W) + (1 - g)\bar Q(0, W)}
                                 {\delta g + 1 - g}\right]

mentions :math:`g` directly and its influence function carries a term
:math:`\partial m/\partial g \cdot (A - g)` that no fluctuation of :math:`\bar Q` can
touch -- it is a score in the tangent space of the *treatment* mechanism.  A TMLE that
solved only the :math:`\bar Q` equation would report an influence curve it had not made
mean zero, which is the one thing the method is for.

So the mechanism gets a submodel of its own,

.. math::

    \operatorname{logit} g_\epsilon(W) = \operatorname{logit} \hat g(W)
                                       + \epsilon^\top H_g(W),
    \qquad
    \frac{\partial}{\partial \epsilon} \log g_\epsilon(A \mid W)
        = H_g(W)\,\{A - g_\epsilon(W)\},

whose score at the solution is exactly the missing term once
:math:`H_g = (\partial m/\partial g)` -- for the incremental tilt,
:math:`\delta\{\bar Q(1,W) - \bar Q(0,W)\}/D_\delta^2`, which is
:attr:`~cleverly.interventions.IPSISet.derivative` times the blip.

**It is the same solver.**  A weighted logistic MLE with an offset and no intercept is
what :func:`~cleverly.fluctuation.iterative.solve_fluctuation` already runs; only
:math:`(y, \text{offset})` change, from :math:`(Y, \operatorname{logit}\bar Q^0)` to
:math:`(A, \operatorname{logit}\hat g)`.  Sharing
:func:`~cleverly.fluctuation.iterative._newton_logistic` rather than writing a second
Newton loop is deliberate: two solvers would mean two line searches, two failure
vocabularies and two definitions of "converged" for one idea.

**The tilt is logistic whatever ``fluctuation=`` says.**  A linear tilt of a probability
is not a probability, and here the tilted mechanism is not merely a regression target --
it is substituted back into :math:`\Psi`, so a value outside :math:`(0, 1)` would not be
an inaccuracy but a nonsense.  ``fluctuation="linear"`` therefore applies to the outcome
regression only.

**Why the two alternate.**  :math:`H_g` is a function of the targeted :math:`\bar Q^*`
and the outcome covariate :math:`h` is a function of the targeted :math:`g^*`, so neither
can be solved once and left.  The alternation is coordinate ascent on one joint
likelihood -- :math:`P_n \log g_\epsilon(A \mid W)` and the outcome quasi-likelihood are
separate factors of the likelihood of :math:`(A, Y) \mid W`, and each step maximises its
own factor with the other held fixed -- so the joint value never decreases and the loop
converges to a stationary point rather than merely being hoped to settle.
:func:`~cleverly.estimators.targeting.solve_with_mechanism` runs it and records the
per-iteration trace, because a loop that stalls should be visible in
:func:`~cleverly.validation.score_check` rather than inferred.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize
from scipy.special import expit, logit

from .._typing import FloatArray
from ._score import relative_score, score_columns, score_scale
from .iterative import InitialFit, TargetingFailure, _newton_logistic

if TYPE_CHECKING:  # pragma: no cover - `interventions` does not import `fluctuation`
    from ..interventions import IPSISet

__all__ = [
    "MECHANISM_BUILDERS",
    "MechanismFluctuation",
    "apply_mechanism_tilt",
    "mechanism_covariate",
    "mechanism_score",
    "needs_mechanism",
    "register_mechanism",
    "solve_bounded_mechanism",
    "solve_mechanism",
]

#: How far from 0 and 1 an initial mechanism is held before its logit is taken.  This is
#: **not** a truncation of the mechanism in the ``g_bounds`` sense -- it does not enter
#: any clever covariate and it cannot move the estimand at a row where the propensity is
#: interior.  It exists because ``logit(0)`` is ``-inf``, and a row whose fitted
#: propensity is exactly 0 or 1 would otherwise put a non-finite offset into the Newton
#: solve.  Such a row is one the fluctuation cannot move anyway: it sits at the boundary
#: of the model, where the submodel through it is degenerate.
_LOGIT_GUARD = 1e-12


@dataclass(frozen=True)
class MechanismFluctuation:
    """The targeted treatment mechanism, and how the solve for it went.

    A sibling of :class:`~cleverly.fluctuation.iterative.Fluctuation` rather than a
    subclass: it reports the same kind of thing about a different equation, and the two
    are carried together on a fit that solves both.

    Attributes
    ----------
    propensity:
        ``(n,)``, the targeted :math:`g^*(1 \\mid W)`.  This is what
        :math:`\\Psi(\\delta)` is evaluated at, and what the influence curve's
        :math:`(A - g)` is taken against.  It is **not** written back onto
        :class:`~cleverly.estimators._nuisance.NuisanceEstimates`, which keeps the
        initial cross-fitted mechanism exactly as it keeps the initial
        :math:`\\bar Q` -- so ``retarget`` stays idempotent and the nuisance diagnostics
        keep describing the model that was actually fitted.
    epsilon:
        One coefficient per tilt.
    score, score_scale, score_initial:
        :math:`P_n[H_g (A - g^*)]` after targeting, the largest magnitude it could have
        had, and its value before.  Reported on the same footing as the outcome
        fluctuation's so that :func:`~cleverly.validation.score_check` can put the two
        equations in one table.
    """

    propensity: FloatArray
    epsilon: FloatArray
    score: FloatArray
    score_scale: FloatArray
    score_initial: FloatArray
    #: Further mechanisms moved by the **same** tilt at the **same** ``epsilon``, in the
    #: order they were passed as ``carry``, and truncated exactly when :attr:`propensity`
    #: was.  Empty on every fit that did not ask for one --
    #: :class:`~cleverly.DRTMLE` with ``reduced_crossfit="nested"`` is the only caller.
    #: Nothing here reads them: they take the tilt and contribute nothing to it.
    carried: tuple[FloatArray, ...] = ()
    converged: bool = True
    n_iter: int = 0
    epsilon_std_error: FloatArray | None = None
    hessian_condition: float | None = None
    loglik: float | None = None
    failure: TargetingFailure | None = None
    #: One ``(outer, q_score, g_score, joint_loglik)`` row per alternation step, filled
    #: in by :func:`~cleverly.estimators.targeting.solve_with_mechanism`.  Kept because a
    #: non-monotone joint likelihood is a bug in the alternation and nothing else would
    #: show it.
    trace: tuple[tuple[int, float, float, float], ...] = field(default_factory=tuple)

    @property
    def relative_score(self) -> float:
        """Largest score component relative to its maximum possible magnitude."""
        return relative_score(self.score, self.score_scale)


#: Builders of the mechanism clever covariate, keyed by target group.  A registry rather
#: than a branch so that a group needing a mechanism fluctuation declares itself here
#: instead of being added to a list inside the estimator, exactly as
#: :data:`~cleverly.fluctuation.submodel.SUBMODEL_BUILDERS` works for the outcome side.
MECHANISM_BUILDERS: dict[str, Callable[..., FloatArray]] = {}


def register_mechanism(group: str, builder: Callable[..., FloatArray]) -> Callable[..., FloatArray]:
    """Declare that ``group``'s score equation has a treatment-mechanism half."""
    if group in MECHANISM_BUILDERS:
        raise ValueError(f"a mechanism covariate for group {group!r} is already registered")
    MECHANISM_BUILDERS[group] = builder
    return builder


def ipsi_mechanism_covariate(targeted: InitialFit, incremental: IPSISet) -> FloatArray:
    r"""``(n, R)``: :math:`\delta_r \{\bar Q^*(1, W) - \bar Q^*(0, W)\} / D_{\delta_r}^2`.

    The blip weighted by :math:`\partial q_\delta / \partial g`.  It reads the *targeted*
    outcome regression, which is why the alternation exists: with the initial
    :math:`\bar Q^0` here the mechanism would be tilted toward the wrong equation.

    The blip is on the ``[0, 1]`` outcome scale, as every other clever covariate here is.
    The whole influence curve is mapped back by ``ctx.finish``, and it is linear in the
    outcome, so no term needs unscaling on its own.
    """
    derivative = np.asarray(incremental.derivative, dtype=float)
    arms = targeted.arms
    blip = np.asarray(arms[1.0], dtype=float) - np.asarray(arms[0.0], dtype=float)
    return np.asarray(derivative * blip[:, None], dtype=float)


register_mechanism("ipsi", ipsi_mechanism_covariate)


def needs_mechanism(group: str) -> bool:
    """Whether ``group``'s targeting is unfinished when the outcome fluctuation converges."""
    return group in MECHANISM_BUILDERS


def mechanism_covariate(group: str, targeted: InitialFit, carrier: IPSISet) -> FloatArray:
    """The mechanism clever covariate for ``group``, by registry lookup."""
    try:
        builder = MECHANISM_BUILDERS[group]
    except KeyError:
        raise ValueError(
            f"group {group!r} has no mechanism covariate; registered groups are "
            f"{sorted(MECHANISM_BUILDERS)}. Use register_mechanism() to add one."
        ) from None
    return builder(targeted, carrier)


def mechanism_score(
    treatment: FloatArray,
    propensity: FloatArray,
    covariate: FloatArray,
    weights: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """``(score, scale)`` for the mechanism equation, without solving it.

    The alternation needs this: after the outcome fluctuation moves, the mechanism
    covariate moves with it, so the score that was solved a moment ago is stale.  Testing
    convergence against a stale score is how a two-equation loop exits having solved one
    of them, which is exactly the failure the ``delta = 1`` identity catches.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g = np.asarray(propensity, dtype=float).reshape(-1)
    h = np.asarray(covariate, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    everywhere = np.ones(a.size, dtype=bool)
    return score_columns(a, g, h, w, everywhere), score_scale(h, w, everywhere)


def apply_mechanism_tilt(
    propensity: FloatArray,
    covariate: FloatArray,
    epsilon: FloatArray,
    *,
    bounds: tuple[float, float] | None = None,
) -> FloatArray:
    r""":math:`\operatorname{expit}(\operatorname{logit}\hat g + H_g\epsilon)`, optionally clipped.

    The mechanism tilt written once, so that the two solvers below and every caller moving
    a *further* array by the same fluctuation apply one map rather than three copies of it.
    Unlike the outcome fluctuation there is no step sequence to reproduce: the Newton
    iteration here is on :math:`\epsilon` alone and the array is formed once from the final
    coefficient, which is why this can be a function of ``(base, H, epsilon)`` and
    :attr:`~cleverly.fluctuation.iterative.Fluctuation.carried` cannot.

    ``bounds`` is passed on the branch :func:`solve_bounded_mechanism` takes, and left
    ``None`` on the branch it does not, so a carried array is truncated exactly when the
    array it accompanies was.
    """
    g = np.asarray(propensity, dtype=float).reshape(-1)
    offset = logit(np.clip(g, _LOGIT_GUARD, 1.0 - _LOGIT_GUARD))
    tilt = np.asarray(expit(offset + np.asarray(covariate, dtype=float) @ epsilon), dtype=float)
    return tilt if bounds is None else np.clip(tilt, float(bounds[0]), float(bounds[1]))


def solve_mechanism(
    treatment: FloatArray,
    propensity: FloatArray,
    covariate: FloatArray,
    weights: FloatArray,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
    carry: Sequence[FloatArray] = (),
) -> MechanismFluctuation:
    r"""Tilt ``propensity`` along ``covariate`` until :math:`P_n[H_g (A - g^*)] = 0`.

    ``propensity`` is :math:`\hat g(1 \mid W)` as an ``(n,)`` array and ``covariate`` the
    ``(n, R)`` matrix :math:`H_g`.  The returned mechanism is
    :math:`\operatorname{expit}(\operatorname{logit}\hat g + \epsilon^\top H_g)`.

    An all-zero covariate is not a failure and not a special case: the score is already
    zero, :math:`\epsilon` is zero and the mechanism comes back untouched.  That happens
    at :math:`\delta = 1`, where the blip weight is finite but the parameter does not
    depend on :math:`g` in a way targeting could improve, and wherever the outcome
    regression finds no treatment effect at all.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g = np.asarray(propensity, dtype=float).reshape(-1)
    h = np.asarray(covariate, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if h.ndim != 2 or h.shape[0] != a.size:
        raise ValueError(f"mechanism covariate must be ({a.size}, R); got {h.shape}")
    everywhere = np.ones(a.size, dtype=bool)

    offset = logit(np.clip(g, _LOGIT_GUARD, 1.0 - _LOGIT_GUARD))
    before = score_columns(a, g, h, w, everywhere)

    epsilon, converged, detail = _newton_logistic(h, a, offset, w, max_iter=max_iter, tol=tol)
    tilted = np.asarray(expit(offset + h @ epsilon), dtype=float)

    failure: TargetingFailure | None = detail.failure
    if failure is None and not converged:
        failure = "max_iter_reached"
    return MechanismFluctuation(
        propensity=tilted,
        carried=tuple(apply_mechanism_tilt(base, h, epsilon) for base in carry),
        epsilon=epsilon,
        score=score_columns(a, tilted, h, w, everywhere),
        score_scale=score_scale(h, w, everywhere),
        score_initial=before,
        converged=converged,
        n_iter=max_iter,
        epsilon_std_error=detail.epsilon_std_error,
        hessian_condition=detail.hessian_condition,
        loglik=detail.loglik,
        failure=failure,
    )


#: Share of rows pinned against the truncation past which an *unsolved* bounded mechanism
#: equation is called ``"bounds_pinned"`` rather than ``"max_iter_reached"``.  The same 1%
#: :func:`~cleverly.fluctuation.iterative._classify` uses on the outcome side, and
#: for the same reason: below it, a state with a few rows at the boundary is one a solver
#: could still have moved, so the failure is the solver's rather than the geometry's.
_PINNED_SHARE = 0.01


def solve_bounded_mechanism(
    treatment: FloatArray,
    propensity: FloatArray,
    covariate: FloatArray,
    weights: FloatArray,
    *,
    bounds: tuple[float, float],
    max_iter: int = 50,
    tol: float = 1e-12,
    carry: Sequence[FloatArray] = (),
) -> MechanismFluctuation:
    r"""Tilt ``propensity`` until the score **at the truncated mechanism** is zero.

    :func:`solve_mechanism` solves :math:`P_n[H_g(A - g^*)] = 0` at the raw
    :math:`\operatorname{expit}` tilt.  This solves

    .. math::

        F(\epsilon) = P_n\bigl[w\,H_g\,\{1_a - \bar g_\epsilon\}\bigr] = 0,
        \qquad
        \bar g_\epsilon = \operatorname{clip}
            \bigl(\operatorname{expit}(\operatorname{logit}\hat g + H_g\epsilon),\;
                  \text{lo},\,\text{hi}\bigr),

    which is a different equation on exactly the rows the bound clips.  **That difference is
    the whole reason this function exists**: :math:`\bar g_\epsilon` is the mechanism
    :func:`~cleverly.inference.influence.reduced_correction_parts` divides by *and* subtracts,
    so :math:`F` **is** the pair of per-arm correction means and a root makes the reported
    curve's centring an identity rather than a second thing to check.  Solving the raw score
    instead leaves the two agreeing on every row the bound leaves alone and parting company on
    every row it clips -- one clipped row of 600 was enough to leave a curve uncentred at
    ``5.8e-04`` while the solver recorded ``1e-09``.  That was ``docs/roadmap.md``'s item 20,
    and this is piece B1b.

    **The unconstrained solve is tried first and returned untouched when nothing clips**, which
    is not an optimisation but the guarantee that this changes no fit it should not.  Where the
    clip is slack on every row it is the identity, so the unconstrained root *is* the bounded
    root -- and a draw where the bound never binds therefore comes back bit for bit what
    :func:`solve_mechanism` gives, down to ``hessian_condition`` and ``loglik``.  Every module
    that fits at inert bounds (``tests/unit/test_influence_gateaux_drtmle.py`` at ``1e-12``
    with ``rtol=0``, and ``tests/unit/test_theorem_drtmle.py``) is on that branch by
    construction rather than by measurement.

    **Why not fluctuate inside the bounds instead.**  A smooth bounded submodel --
    :math:`\text{lo} + (\text{hi} - \text{lo})\operatorname{expit}(\operatorname{logit} u_0 +
    H_g\epsilon)` -- never leaves the bounds and needs no projection, and
    ``docs/drtmle/theorem-concordance.md`` §7 prefers it *to post-fit clipping*.  It was
    prototyped and it loses twice.  It is a **different submodel on every fit**, not only on
    the clipping ones: at inert bounds of ``1e-6`` it moved a no-clip fixture's ``psi`` by
    ``2.7e-03`` standard errors where this function moves it by zero.  And where the bound does
    bind it left the final score at ``1.5e-07`` against this branch's ``2.1e-10``, because its
    derivative :math:`(\text{hi} - \text{lo})u(1 - u)` collapses near the bounds.  §7's stated
    reason -- that a projection applied *after* an unconstrained optimisation does not solve
    the clipped state's first-order condition -- is an argument against clipping afterwards,
    which is what this function does not do.

    **A root need not exist**, and that is reported rather than approximated.  With every row
    pinned, :math:`\partial\bar g_\epsilon/\partial\epsilon` is zero everywhere and no
    :math:`\epsilon` moves :math:`F` at all; the failure is ``"bounds_pinned"``, whose existing
    wording in :mod:`cleverly.fluctuation.iterative` already says exactly this.  Returning the
    last iterate as though it were a solution is the one outcome this must not have.

    Parameters
    ----------
    bounds:
        The ``g_bounds`` truncation, the same pair
        :func:`~cleverly.fluctuation.reduced.reduced_mechanism_covariate` built ``covariate``'s
        denominator at and :func:`~cleverly.inference.influence.reduced_correction_parts`
        reads.  Passing a different pair here would put the residual and the denominator at two
        mechanisms again, which is the defect rather than a variant of it.

    Notes
    -----
    Not used by ``ipsi``, which calls :func:`solve_mechanism` through
    :func:`~cleverly.estimators.targeting.solve_with_mechanism`.  That estimand is a functional
    of :math:`g` itself and truncating it would move :math:`\Psi(\delta)` rather than
    regularise a denominator, so ``g_bounds`` is refused there outright -- and that path is a
    regression surface, which is why this is a sibling function rather than a keyword on that
    one.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    h = np.asarray(covariate, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    lower, upper = float(bounds[0]), float(bounds[1])
    if not 0.0 < lower < upper < 1.0:
        raise ValueError(f"the mechanism truncation must satisfy 0 < lo < hi < 1; got {bounds}")

    plain = solve_mechanism(a, propensity, h, w, max_iter=max_iter, tol=tol, carry=carry)
    raw = np.asarray(plain.propensity, dtype=float)
    if not np.any((raw < lower) | (raw > upper)):
        # The fast path returns the unconstrained solve untouched, so a carried array is
        # untruncated here too: the whole point of this branch is that the estimator *is*
        # the unconstrained one, and clipping only the carried arrays would make the two
        # constructions differ by a truncation rather than by a construction.
        return plain

    g = np.asarray(propensity, dtype=float).reshape(-1)
    offset = logit(np.clip(g, _LOGIT_GUARD, 1.0 - _LOGIT_GUARD))
    everywhere = np.ones(a.size, dtype=bool)

    def at(epsilon: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``(raw tilt, truncated tilt)`` at ``epsilon``."""
        tilt = np.asarray(expit(offset + h @ epsilon), dtype=float)
        return tilt, np.clip(tilt, lower, upper)

    def residual(epsilon: FloatArray) -> FloatArray:
        return score_columns(a, at(epsilon)[1], h, w, everywhere)

    warm = np.asarray(plain.epsilon, dtype=float)
    if warm.size == 0 or float(np.max(np.abs(residual(warm)))) <= tol:
        # Already a root of the bounded equation, which is not a rare case: an all-zero
        # covariate is one -- `Q_r` vanishes row by row wherever the outcome regression is
        # right -- and it must not be handed to a root finder with no derivative to work
        # with.  The mechanism still comes back *truncated*, since that is the array the
        # curve reads.
        return replace(
            plain,
            propensity=at(warm)[1],
            carried=tuple(apply_mechanism_tilt(base, h, warm, bounds=bounds) for base in carry),
            score=residual(warm),
            converged=True,
            failure=None,
        )

    # Warm-started from the unconstrained root, which is the nearest thing to the answer
    # already paid for: the two equations differ only on the clipped rows.  `hybr` is
    # MINPACK's Powell hybrid, a trust region rather than a plain Newton, which is what this
    # equation wants: `F` is only piecewise smooth, so a step taken with one active set can
    # land in another, and a hand-rolled damped Newton stalled at `1.9e-04` on a fixture
    # where a root exists at `1e-17`.  The *verdict* below is still this module's own -- the
    # solver proposes an iterate and the score decides whether it is a solution -- so there
    # is one definition of converged here rather than two.
    solved = optimize.root(residual, np.asarray(plain.epsilon, dtype=float), method="hybr")
    epsilon = np.asarray(solved.x, dtype=float)
    score = residual(epsilon)
    converged = bool(score.size == 0 or np.max(np.abs(score)) <= tol)

    tilt, truncated = at(epsilon)
    failure: TargetingFailure | None = None
    if not converged:
        # Named from the endpoint, as `iterative._classify` names the outcome
        # side's: a state whose rows are pinned against the bound is one no epsilon can move,
        # since the clip is flat there and contributes nothing to the derivative.  A pinned
        # share is only a failure *given* an unsolved score -- under this convention rows sit
        # at the bound on ordinary fits whose equation is solved exactly.
        pinned = float(np.mean((tilt <= lower) | (tilt >= upper)))
        failure = "bounds_pinned" if pinned > _PINNED_SHARE else "max_iter_reached"
    return MechanismFluctuation(
        propensity=truncated,
        carried=tuple(apply_mechanism_tilt(base, h, epsilon, bounds=bounds) for base in carry),
        epsilon=epsilon,
        score=score,
        score_scale=score_scale(h, w, everywhere),
        score_initial=plain.score_initial,
        converged=converged,
        n_iter=max_iter,
        epsilon_std_error=plain.epsilon_std_error,
        hessian_condition=plain.hessian_condition,
        loglik=plain.loglik,
        failure=failure,
    )

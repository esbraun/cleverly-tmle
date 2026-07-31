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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import expit, logit

from .._typing import FloatArray
from ._score import relative_score, score_columns, score_scale
from .iterative import InitialFit, TargetingFailure, _newton_logistic

if TYPE_CHECKING:  # pragma: no cover - `interventions` does not import `fluctuation`
    from ..interventions import IPSISet

__all__ = [
    "MECHANISM_BUILDERS",
    "MechanismFluctuation",
    "mechanism_covariate",
    "needs_mechanism",
    "register_mechanism",
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


def solve_mechanism(
    treatment: FloatArray,
    propensity: FloatArray,
    covariate: FloatArray,
    weights: FloatArray,
    *,
    max_iter: int = 50,
    tol: float = 1e-12,
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

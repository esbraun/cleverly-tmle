r"""Separable effects of a held baseline treatment on two competing causes.

A separable intervention gives the component of the treatment that acts on one cause the
arm :math:`a_D` and the component that acts on the other cause the arm :math:`a_Y`
(Stensrud, Young, Didelez, Robins and Hernán 2022, estimand (5), identification (8) in
Section 5.2 and its censored form (10) in Section 5.4, arXiv v3 numbering).  With the first
cause :math:`D` counted before the second cause :math:`Y` within a node, each node's draw is
two binary factors: :math:`D` with probability :math:`\lambda_D(k \mid a, W)` among the units
at risk, then :math:`Y` with probability :math:`h_Y(k \mid a, W)` among those also free of
:math:`D`.  The separable law composes them across the two arms,

.. math::

    \tilde\lambda_D(k) = \lambda_D(k \mid a_D), \qquad
    \tilde\lambda_Y(k) = h_Y(k \mid a_Y)\,(1 - \lambda_D(k \mid a_D)), \qquad
    \tilde S(k) = \prod_{s \le k} (1 - h_Y(s \mid a_Y))(1 - \lambda_D(s \mid a_D)),

and the parameter of cause :math:`j` at horizon :math:`t` is
:math:`\psi_j(t) = E_W \sum_{k \le t} \tilde S(k - 1)\,\tilde\lambda_j(k)`.  At
:math:`a_Y = a_D` it is the cumulative incidence of that static arm.

The efficient influence curve follows from the chain rule through the composed hazards
(``X91-D-design.md`` section 2; Martinussen and Stensrud 2023 give the continuous-time
counterpart).  With :math:`c_l(k) = 1\{l = j\} - (\tilde F_j(t) - \tilde F_j(k)) / \tilde S(k)`
and :math:`S(k - 1 \mid a)` the factual survival at level :math:`a`, its factor clever
covariates are

.. math::

    H_Y(k) = \frac{\tilde S(k-1)\,(1 - \lambda_D(k \mid a_D))\,c_Y(k)}
        {g(a_Y) G_k(a_Y)\,S(k-1 \mid a_Y)\,(1 - \lambda_D(k \mid a_Y))}, \qquad
    H_D(k) = \frac{\tilde S(k-1)\,(c_D(k) - h_Y(k \mid a_Y)\,c_Y(k))}
        {g(a_D) G_k(a_D)\,S(k-1 \mid a_D)},

read at :math:`A = a_Y` and :math:`A = a_D` respectively, and the curve is
:math:`\sum_k [H_D R_k (dN_D - \lambda_D) + H_Y R_k (1 - dN_D)(dN_Y - h_Y)] + \tilde F_j(t)
- \psi_j(t)`.

The state is held as :math:`\beta_D = \operatorname{logit} \lambda_D` and
:math:`\beta_Y = \operatorname{logit} h_Y`, and each factor is fluctuated on its own logistic
submodel with one coefficient per target.  Solving both factors' score equations separately
leaves a correct factor where it is, so the estimate is consistent when the mechanism and
either :math:`\lambda_D` or :math:`h_Y` (at both arms) are consistent, as well as when both
hazards are (design section 2a).  The shipped softmax fluctuation of
:mod:`cleverly.longitudinal.hazard` solves the same scores but loses that robustness, which is
why separable fits target here.  The targeting is the one-step walk of
:func:`~cleverly.longitudinal.hazard.walk_curve` (Rytgaard and van der Laan 2024, Section 3.4)
over the :math:`2M` factor scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit, xlogy

from .._typing import BoolArray, FloatArray, IntArray
from .._warnings import warn as _warn
from ..exceptions import ConvergenceWarning
from ..fluctuation._score import relative_score
from .hazard import SIGMA_FLOOR, HazardFit

__all__ = [
    "FACTORS",
    "SeparableInputs",
    "SeparableState",
    "SeparableTarget",
    "influence",
    "scores",
    "separable_direction",
    "walk_loss",
    "walk_separable",
]

#: The two factors of a node, in score order within each target.
FACTORS = ("competing", "event")

#: The shortest step before the walk gives up, as in the shipped hazard walk.
_MIN_STEP = 1e-14


@dataclass(frozen=True)
class SeparableTarget:
    """One reported parameter of a separable fit.

    Parameters
    ----------
    label : str
        The regimen's label.
    levels : tuple of int
        ``(a_D, a_Y)``: the level codes whose components act on the first and the second
        cause.
    role : {"competing", "event"}
        Whether the parameter is the incidence of the first cause (``"competing"``) or the
        second (``"event"``).
    cause : str
        The cause label.
    horizon : int
        The horizon, counted from one.
    """

    label: str
    levels: tuple[int, int]
    role: str
    cause: str
    horizon: int


def _per(value: FloatArray, denominator: FloatArray) -> FloatArray:
    """``value / denominator``, and zero where the denominator is zero."""
    return np.asarray(
        np.divide(
            value,
            denominator,
            out=np.zeros(np.broadcast_shapes(value.shape, denominator.shape)),
            where=denominator > 0.0,
        ),
        dtype=float,
    )


@dataclass(frozen=True)
class SeparableState:
    """The two factors at every unit, node and level, as clipped logits.

    Parameters
    ----------
    beta_d : FloatArray
        ``(n, T, L)`` :math:`\\operatorname{logit} \\lambda_D(k \\mid a, W)`.  A held cell is
        :math:`\\pm\\infty`.
    beta_y : FloatArray
        ``(n, T, L)`` :math:`\\operatorname{logit} h_Y(k \\mid a, W)`.
    """

    beta_d: FloatArray
    beta_y: FloatArray
    lam_d: FloatArray = field(init=False, repr=False)
    h_y: FloatArray = field(init=False, repr=False)
    survival: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lam_d = expit(np.asarray(self.beta_d, dtype=float))
        h_y = expit(np.asarray(self.beta_y, dtype=float))
        object.__setattr__(self, "lam_d", lam_d)
        object.__setattr__(self, "h_y", h_y)
        object.__setattr__(self, "survival", np.cumprod((1.0 - lam_d) * (1.0 - h_y), axis=1))

    @classmethod
    def from_hazards(
        cls,
        hazards: FloatArray,
        first: int,
        bounds: tuple[float, float],
        held: tuple[BoolArray, BoolArray, BoolArray, BoolArray],
    ) -> SeparableState:
        """The state of ``(n, T, L, 2)`` cause-specific hazards fitted in the declared order.

        ``hazards[..., first]`` is :math:`\\lambda_D`, and the other cause over
        :math:`1 - \\lambda_D` is :math:`h_Y`.  ``held`` holds the ``(T, L)`` cells whose
        factor is held: empty and full :math:`D`, then empty and full :math:`Y` among the
        :math:`D`-free.
        """
        lam = np.asarray(hazards, dtype=float)
        lam_d = lam[..., first]
        h_y = _per(lam[..., 1 - first], 1.0 - lam_d)
        with np.errstate(divide="ignore"):
            beta_d = np.clip(np.log(lam_d) - np.log1p(-lam_d), *bounds)
            beta_y = np.clip(np.log(h_y) - np.log1p(-h_y), *bounds)
        empty_d, full_d, empty_y, full_y = (np.asarray(cell, dtype=bool)[None] for cell in held)
        beta_d = np.where(empty_d, -np.inf, np.where(full_d, np.inf, beta_d))
        beta_y = np.where(empty_y, -np.inf, np.where(full_y, np.inf, beta_y))
        return cls(beta_d, beta_y)

    def moved(
        self, delta_d: FloatArray, delta_y: FloatArray, bounds: tuple[float, float]
    ) -> SeparableState:
        """The state after adding each factor's increment, then clipping; held cells stay."""

        def step(beta: FloatArray, delta: FloatArray) -> FloatArray:
            return np.where(np.isinf(beta), beta, np.clip(beta + delta, *bounds))

        return SeparableState(step(self.beta_d, delta_d), step(self.beta_y, delta_y))

    def composed(self, levels: tuple[int, int]) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(n, T)`` composed survival, ``(n, T)`` survival before each node, and
        ``(n, T, 2)`` composed incidence by role, under the arms ``(a_D, a_Y)``.

        Each incidence is a running sum of non-negative products, so it is non-decreasing over
        the nodes in floating point.
        """
        a_d, a_y = levels
        lam_d = self.lam_d[:, :, a_d]
        h_y = self.h_y[:, :, a_y]
        survival = np.cumprod((1.0 - h_y) * (1.0 - lam_d), axis=1)
        before = np.concatenate([np.ones_like(survival[:, :1]), survival[:, :-1]], axis=1)
        incidence = np.stack(
            [
                np.cumsum(before * lam_d, axis=1),
                np.cumsum(before * (h_y * (1.0 - lam_d)), axis=1),
            ],
            axis=2,
        )
        return survival, before, incidence


@dataclass(frozen=True)
class SeparableInputs:
    """Everything the separable targeting reads besides the factors.

    Parameters
    ----------
    at_risk : BoolArray
        ``(n, T)`` :math:`R_k`: uncensored through ``k`` and event-free before ``k``.
    event_d : FloatArray
        ``(n, T)`` :math:`dN_D(k)` on the rows at risk, zero elsewhere.
    event_y : FloatArray
        ``(n, T)`` :math:`dN_Y(k)` on the rows at risk, zero elsewhere.
    observed : IntArray
        ``(n,)`` dense code of the baseline treatment each unit received.
    inverse : FloatArray
        ``(n, T, L)`` :math:`1 / (g(a \\mid W) G_k(a, W))` at every level, after
        ``g_bounds``.
    weights : FloatArray
        ``(n,)`` observation weights, mean one.
    alpha : float
        The probability bound of each factor.
    """

    at_risk: BoolArray
    event_d: FloatArray
    event_y: FloatArray
    observed: IntArray
    inverse: FloatArray
    weights: FloatArray
    alpha: float

    @property
    def n(self) -> int:
        return int(self.at_risk.shape[0])

    @property
    def bounds(self) -> tuple[float, float]:
        """The clip interval of every factor logit."""
        upper = float(np.log(self.alpha / (1.0 - self.alpha)))
        return -upper, upper

    @property
    def rows(self) -> IntArray:
        return np.arange(self.n)

    def rows_d(self) -> BoolArray:
        """``(n, T)`` the rows of the competing factor: at risk."""
        return self.at_risk

    def rows_y(self) -> BoolArray:
        """``(n, T)`` the rows of the event factor: at risk and free of the competing cause."""
        return self.at_risk & (self.event_d == 0.0)


def _covariates(
    state: SeparableState, inputs: SeparableInputs, target: SeparableTarget
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(n, T)`` :math:`H_D` at :math:`a_D` and :math:`H_Y` at :math:`a_Y` for every unit, and
    ``(n,)`` the plug-in :math:`\\tilde F_j(t \\mid W)`.

    Both covariates are zero after the horizon.
    """
    a_d, a_y = target.levels
    survival, before, incidence = state.composed(target.levels)
    t = target.horizon
    j = 0 if target.role == FACTORS[0] else 1
    remaining = _per(incidence[:, t - 1, j][:, None] - incidence[:, :, j], survival)
    c_d = (1.0 if j == 0 else 0.0) - remaining
    c_y = (1.0 if j == 1 else 0.0) - remaining
    factual = np.concatenate(
        [np.ones_like(state.survival[:, :1]), state.survival[:, :-1]], axis=1
    )  # (n, T, L): S(k - 1 | a)
    lam_d = state.lam_d
    h_y = state.h_y
    weight_y = _per(
        before * (1.0 - lam_d[:, :, a_d]) * inputs.inverse[:, :, a_y],
        factual[:, :, a_y] * (1.0 - lam_d[:, :, a_y]),
    )
    weight_d = _per(before * inputs.inverse[:, :, a_d], factual[:, :, a_d])
    h_event = weight_y * c_y
    h_competing = weight_d * (c_d - h_y[:, :, a_y] * c_y)
    h_event[:, t:] = 0.0
    h_competing[:, t:] = 0.0
    return h_competing, h_event, incidence[:, t - 1, j]


def _residuals(state: SeparableState, inputs: SeparableInputs) -> tuple[FloatArray, FloatArray]:
    """``(n, T)`` each factor's residual at the observed level, zero off its rows."""
    rows, levels = inputs.rows, inputs.observed
    lam_d = state.lam_d[rows, :, levels]
    h_y = state.h_y[rows, :, levels]
    residual_d = np.where(inputs.rows_d(), inputs.event_d - lam_d, 0.0)
    residual_y = np.where(inputs.rows_y(), inputs.event_y - h_y, 0.0)
    return residual_d, residual_y


def _terms(
    state: SeparableState, inputs: SeparableInputs, targets: Sequence[SeparableTarget]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(2M, n)`` factor score terms, ``(2M,)`` scales and ``(M, n)`` plug-ins.

    The factors of target ``m`` sit at rows ``2m`` (competing) and ``2m + 1`` (event).
    """
    residual_d, residual_y = _residuals(state, inputs)
    rows_d, rows_y = inputs.rows_d(), inputs.rows_y()
    weight = np.abs(inputs.weights)
    terms = np.zeros((2 * len(targets), inputs.n))
    scale = np.zeros(2 * len(targets))
    plug = np.zeros((len(targets), inputs.n))
    for m, target in enumerate(targets):
        h_d, h_y, plug[m] = _covariates(state, inputs, target)
        a_d, a_y = target.levels
        at_d = (inputs.observed == a_d)[:, None]
        at_y = (inputs.observed == a_y)[:, None]
        terms[2 * m] = np.where(at_d, h_d * residual_d, 0.0).sum(axis=1)
        terms[2 * m + 1] = np.where(at_y, h_y * residual_y, 0.0).sum(axis=1)
        scale[2 * m] = np.mean(weight * np.where(at_d & rows_d, np.abs(h_d), 0.0).sum(axis=1))
        scale[2 * m + 1] = np.mean(weight * np.where(at_y & rows_y, np.abs(h_y), 0.0).sum(axis=1))
    return terms, scale, plug


def scores(
    state: SeparableState, inputs: SeparableInputs, targets: Sequence[SeparableTarget]
) -> tuple[FloatArray, FloatArray]:
    """``(2M,)`` weighted factor scores :math:`P_n D^{(1)}` and their scales.

    Parameters
    ----------
    state : SeparableState
        The current factors.
    inputs : SeparableInputs
        The targeting inputs.
    targets : sequence of SeparableTarget
        The targets, in report order.

    Returns
    -------
    tuple of FloatArray
        The scores and the per-score scale, ``mean(|w| sum_k |H|)`` over the factor's rows.
    """
    terms, scale, _ = _terms(state, inputs, targets)
    return np.asarray((terms * inputs.weights).mean(axis=-1), dtype=float), scale


def influence(
    state: SeparableState, inputs: SeparableInputs, targets: Sequence[SeparableTarget]
) -> tuple[FloatArray, FloatArray]:
    """The plug-in levels and the weighted influence curves of every target.

    Parameters
    ----------
    state : SeparableState
        The reported factors.
    inputs : SeparableInputs
        The targeting inputs.
    targets : sequence of SeparableTarget
        The targets, in report order.

    Returns
    -------
    tuple of FloatArray
        ``(M,)`` levels and ``(M, n)`` influence curves :math:`(w / E[w]) D^*(P_w)`.
    """
    terms, _, plug = _terms(state, inputs, targets)
    levels = np.average(plug, axis=1, weights=inputs.weights)
    hazard = terms[0::2] + terms[1::2]
    curves = hazard + plug - levels[:, None]
    return np.asarray(levels, dtype=float), np.asarray(curves * inputs.weights, dtype=float)


def separable_direction(
    state: SeparableState,
    inputs: SeparableInputs,
    targets: Sequence[SeparableTarget],
    coefficients: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """``(n, T, L)`` increments of :math:`\\beta_D` and :math:`\\beta_Y` per unit coefficient.

    Target ``m``'s competing covariate moves :math:`\\beta_D` at its level :math:`a_D` with
    coefficient ``coefficients[2m]``, and its event covariate moves :math:`\\beta_Y` at
    :math:`a_Y` with ``coefficients[2m + 1]``.  No target moves :math:`\\lambda_D` at
    :math:`a_Y` or :math:`h_Y` at :math:`a_D` unless the two levels agree.

    Parameters
    ----------
    state : SeparableState
        The current factors.
    inputs : SeparableInputs
        The targeting inputs.
    targets : sequence of SeparableTarget
        The targets, in report order.
    coefficients : FloatArray
        ``(2M,)`` coefficient of every factor score.

    Returns
    -------
    tuple of FloatArray
        The increments of the two factor logits.
    """
    delta_d = np.zeros_like(state.lam_d)
    delta_y = np.zeros_like(state.h_y)
    for m, target in enumerate(targets):
        h_d, h_y, _ = _covariates(state, inputs, target)
        a_d, a_y = target.levels
        delta_d[:, :, a_d] += coefficients[2 * m] * h_d
        delta_y[:, :, a_y] += coefficients[2 * m + 1] * h_y
    return delta_d, delta_y


def walk_loss(state: SeparableState, inputs: SeparableInputs) -> float:
    """The negative weighted log-likelihood of both factors on their rows.

    Parameters
    ----------
    state : SeparableState
        The current factors.
    inputs : SeparableInputs
        The targeting inputs.

    Returns
    -------
    float
        The loss.
    """
    rows, levels = inputs.rows, inputs.observed
    lam_d = state.lam_d[rows, :, levels]
    h_y = state.h_y[rows, :, levels]
    term_d = xlogy(inputs.event_d, lam_d) + xlogy(1.0 - inputs.event_d, 1.0 - lam_d)
    term_y = xlogy(inputs.event_y, h_y) + xlogy(1.0 - inputs.event_y, 1.0 - h_y)
    weight = inputs.weights[:, None]
    counted = (inputs.weights != 0.0)[:, None]
    total = np.where(inputs.rows_d() & counted, weight * term_d, 0.0).sum()
    total += np.where(inputs.rows_y() & counted, weight * term_y, 0.0).sum()
    return -float(total)


def walk_separable(
    initial: SeparableState,
    inputs: SeparableInputs,
    targets: Sequence[SeparableTarget],
    *,
    step_size: float,
    tol: float,
    max_steps: int = 20_000,
    warn: bool = True,
) -> tuple[SeparableState, HazardFit]:
    """Walk the universal least favorable submodel over every factor score.

    The direction is :math:`\\Sigma^{-1} s / \\lVert s \\rVert_\\Sigma` with :math:`s` the
    ``2M`` factor scores and the fixed :math:`\\Sigma = \\operatorname{diag}(P_n D_f^2)` of
    each factor's part of the initial influence curve.  A step is accepted only when the
    :math:`\\Sigma`-norm of the score does not rise; otherwise it is halved.

    Parameters
    ----------
    initial : SeparableState
        The initial factors.
    inputs : SeparableInputs
        The targeting inputs.
    targets : sequence of SeparableTarget
        The targets, in report order.
    step_size : float
        The first step length.
    tol : float
        Relative-score tolerance.
    max_steps : int
        The step cap.
    warn : bool
        Whether a walk that stops short raises a ``ConvergenceWarning``.

    Returns
    -------
    tuple
        The targeted state and its record, whose ``targets`` holds one
        ``(label, cause, horizon, factor)`` entry per score.
    """
    bounds = inputs.bounds
    state = initial
    levels, _ = influence(state, inputs, targets)
    terms, _, _ = _terms(state, inputs, targets)
    weighted = terms * inputs.weights
    centred = weighted - weighted.mean(axis=1, keepdims=True)
    sigma = np.maximum(np.mean(centred**2, axis=1), SIGMA_FLOOR)
    score, scale = scores(state, inputs, targets)
    first = score.copy()
    norm = float(np.sqrt(np.sum(score**2 / sigma)))
    trace = [relative_score(score, scale)]
    norms = [norm]
    losses = [walk_loss(state, inputs)]
    dx = float(step_size)
    steps = 0
    while trace[-1] > tol and steps < max_steps and norm > 0.0:
        coefficients = score / sigma / norm
        delta_d, delta_y = separable_direction(state, inputs, targets, coefficients)
        candidate = state.moved(dx * delta_d, dx * delta_y, bounds)
        candidate_score, candidate_scale = scores(candidate, inputs, targets)
        candidate_norm = float(np.sqrt(np.sum(candidate_score**2 / sigma)))
        if candidate_norm > norm:
            dx *= 0.5
            if dx < _MIN_STEP:
                break
            continue
        state, score, scale, norm = candidate, candidate_score, candidate_scale, candidate_norm
        steps += 1
        trace.append(relative_score(score, scale))
        norms.append(norm)
        losses.append(walk_loss(state, inputs))
    relative = relative_score(score, scale)
    converged = bool(relative <= tol)
    if not converged and warn:
        _warn(
            f"one-step separable targeting stopped after {steps} step(s) with a relative score "
            f"of {relative:.3g} > {tol:g}. Try a smaller step_size, or check "
            "result.diagnostics for a positivity problem",
            ConvergenceWarning,
            stacklevel=3,
        )
    record_targets = tuple(
        (target.label, target.cause, target.horizon, factor)
        for target in targets
        for factor in FACTORS
    )
    return state, HazardFit(
        algorithm="one_step",
        targets=record_targets,  # type: ignore[arg-type]
        converged=converged,
        n_iter=steps,
        score_initial=first,
        score=score,
        score_scale=scale,
        trace=tuple(trace),
        initial_levels=levels,
        sigma=sigma,
        norm_trace=tuple(norms),
        walk_loss_trace=tuple(losses),
        failure=None
        if converged
        else "max_iter_reached"
        if steps >= max_steps
        else "line_search_exhausted",
        tol=tol,
    )

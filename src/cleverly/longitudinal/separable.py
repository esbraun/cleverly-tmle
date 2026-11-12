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
over the :math:`2M` factor scores.  When the walk's relative score stops halving over
``_STALL`` steps, the classical iterative TMLE takes over: the covariates are frozen, each
factor's fluctuation coefficients are fitted by maximum likelihood on that factor's rows, the
state moves, and the covariates are recomputed.  If that stops short, Newton steps on the full
Jacobian of the factor scores, whose covariates move with the state, finish it.  Every phase
moves each factor only along its own covariates at its own level, and the state with the
smallest relative score is reported.

The fallbacks matter for cross-fitted saturated learners.  When the targets that move a factor
at a level span most of its cells in every fold, the score equations move those cells toward
their own fold's empirical hazard: the root lies far from the cross-fitted start, by up to two
logits in small cells.  The walk's fixed-length steps crawl over that distance.  A full Newton
step at frozen covariates overshoots there, because it ignores how the covariates change with
the state; the exact fluctuation fit of the iterative TMLE is damped by its own likelihood and
does not.  Nearly collinear targets leave the iterative TMLE with a residual that only the full
Jacobian's small directions solve.  Where scores have more than one root, the reported one
depends on this path.  When the rows of a cell hold no event of a factor, the score of that
cell is solved only at a hazard of zero, outside the bound set by ``alpha``: the fit then stops
with ``failure="cell_separated"``, as separation stops a logistic fit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.special import expit, log_expit, xlogy

from .._typing import BoolArray, FloatArray, IntArray
from .._warnings import warn as _warn
from ..exceptions import CapabilityError, ConvergenceWarning, DataError, RefusalReason
from ..fluctuation._score import relative_score
from ..fluctuation.iterative import TargetingFailure
from .data import LongitudinalData
from .hazard import SIGMA_FLOOR, HazardFit, event_indicators
from .regimen import Plan, Regimen, Separable
from .sequential import RegimenFit

__all__ = [
    "FACTORS",
    "Separable",
    "SeparableInputs",
    "SeparablePlan",
    "SeparableState",
    "SeparableTarget",
    "influence",
    "resolve_separable",
    "scores",
    "separable_direction",
    "split_separable",
    "target_separable",
    "walk_loss",
    "walk_separable",
]

#: The two factors of a node, in score order within each target.
FACTORS = ("competing", "event")

#: The shortest step before the walk gives up, as in the shipped hazard walk.
_MIN_STEP = 1e-14
#: Accepted walk steps over which the relative score must halve; otherwise the walk hands
#: over to the iterative TMLE.
_STALL = 50
#: Steps of each fallback phase without a lower relative score, by at least the share
#: below, before the phase stops.
_PATIENCE = 6
#: The share of the best relative score a step must reach to count as progress.
_PROGRESS = 0.9
#: The most steps of the iterative TMLE and of the Newton steps on the full score Jacobian.
_ITERATIVE_STEPS = 30
_JACOBIAN_STEPS = 20
#: The central-difference length of the full score Jacobian, per unit coefficient.
_JACOBIAN_STEP = 1e-6
#: The logit distance from the clip within which a row counts as at the clip: rounding only,
#: because the state's clip sets a pinned logit to the bound itself.
_AT_CLIP = 1e-9
#: The share of the score norm that rows at the clip must carry for a stopped fit to read
#: ``cell_separated``.
_BOUND_SHARE = 0.5
#: Newton-Raphson iterations of one factor's fluctuation fit at frozen covariates, and the
#: largest logit move of a full step that ends the fit.
_FIT_ITERATIONS = 100
_FIT_CHANGE = 1e-9


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


@dataclass(frozen=True)
class SeparablePlan:
    """A resolved separable intervention.

    Parameters
    ----------
    label : str
        The regimen's label.
    first : int
        The cause index counted first.
    levels : tuple of int
        ``(a_D, a_Y)``: the level codes of the first and the second cause.
    declared : Separable
        The declaration.
    """

    label: str
    first: int
    levels: tuple[int, int]
    declared: Separable


def separable_words(declared: Separable) -> str:
    """The separable intervention in plain words, for a summary.

    Parameters
    ----------
    declared : Separable
        The declaration.

    Returns
    -------
    str
        For example ``"the part acting on relapse at 1, the part acting on death at 0;
        death counted first within a time step"``.
    """
    parts = ", ".join(
        f"the part acting on {cause} at {level}" for cause, level in declared.arms.items()
    )
    return f"{parts}; {declared.first} counted first within a time step"


def split_separable(spec: Any) -> tuple[Any, tuple[tuple[str, Separable], ...]]:
    """Separate the ``Separable`` entries of a ``regimens=`` mapping from the rest.

    Parameters
    ----------
    spec : Any
        The ``regimens=`` argument.

    Returns
    -------
    tuple
        The remaining specification, and ``(label, Separable)`` of each separable entry in
        order.  A specification without one comes back unchanged with an empty tuple.
    """
    if not isinstance(spec, Mapping) or not any(isinstance(v, Separable) for v in spec.values()):
        if isinstance(spec, Separable):
            raise DataError(
                "a Separable needs a label and a static reference arm: pass "
                "regimens={'a0': 0, 'separable': Separable(...)}"
            )
        return spec, ()
    rest = {label: plan for label, plan in spec.items() if not isinstance(plan, Separable)}
    found = tuple((str(label), plan) for label, plan in spec.items() if isinstance(plan, Separable))
    if not rest:
        raise DataError(
            "a fit with a Separable regimen needs at least one static arm in regimens=, "
            "which is the reference of the separable direct and indirect contrasts; add, "
            "for example, 'a0': 0"
        )
    return rest, found


def resolve_separable(
    data: LongitudinalData, separables: Sequence[tuple[str, Separable]]
) -> tuple[SeparablePlan, ...]:
    """Resolve each separable intervention against the panel's causes and levels.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel.
    separables : sequence of tuple
        ``(label, Separable)`` of each entry.

    Returns
    -------
    tuple of SeparablePlan
        One plan per entry, in order.

    Raises
    ------
    DataError
        If the panel does not hold exactly two causes, an arm names an unknown cause or
        level, or two entries declare different first causes.
    """
    causes = tuple(data.cause_labels)
    if len(causes) != 2:
        raise DataError(
            "a Separable regimen needs exactly two causes, the event of interest and one "
            f"competing event; this panel holds {len(causes) or 1}. Stensrud et al. (2022) "
            "state the separable effects for one competing event"
        )
    out = []
    firsts = set()
    for label, declared in separables:
        keys = {str(key) for key in declared.arms}
        if keys != set(causes):
            raise DataError(
                f"Separable {label!r} names causes {sorted(keys)}, but the panel's causes "
                f"are {sorted(causes)}"
            )
        codes = {}
        for cause, level in declared.arms.items():
            # The panel's own encoding, so a level matches exactly as a static arm's does.
            try:
                encoded = data.encode_assignment((level,) * data.n_decisions, label)
            except DataError as error:
                raise DataError(
                    f"Separable {label!r} gives cause {cause!r} the level {level!r}, which "
                    f"the treatment does not take: {error}"
                ) from error
            codes[str(cause)] = int(encoded[0, 0])
        first = causes.index(str(declared.first))
        firsts.add(first)
        second = causes[1 - first]
        out.append(SeparablePlan(label, first, (codes[causes[first]], codes[second]), declared))
    if len(firsts) > 1:
        raise CapabilityError(
            "every Separable regimen of one fit must declare the same first cause: the "
            "initial hazards are fitted once, in that order.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            roadmap="X91",
            alternative="Fit the two orders in two fits.",
        )
    return tuple(out)


def refuse_separable_companions(regimens: Sequence[Any]) -> None:
    """Refuse a regimen beside a ``Separable`` that is not one static level.

    Parameters
    ----------
    regimens : sequence
        The resolved regimens declared beside the separable entries.

    Raises
    ------
    CapabilityError
        If a regimen holds a rule, a policy, a shift or a tilt.
    """
    for regimen in regimens:
        if not isinstance(regimen, Regimen):
            label = getattr(regimen, "label", repr(regimen))
            raise CapabilityError(
                f"regimen {label!r} is not a static arm. Beside a Separable regimen, the "
                "factor-wise targeting reads every arm as one level of the baseline "
                "treatment; a rule, a policy, a shift or a tilt mixes levels across units, "
                "and its factor-wise form is not built.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                roadmap="X91",
                alternative="Fit that regimen in its own fit, without the Separable.",
            )


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
    #: Per-state memo of the covariates for one inputs object (held under ``"inputs"`` and
    #: checked by identity), so the scores and the walk direction at one state compute
    #: them once.
    cache: dict[Any, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        lam_d = expit(np.asarray(self.beta_d, dtype=float))
        h_y = expit(np.asarray(self.beta_y, dtype=float))
        object.__setattr__(self, "lam_d", lam_d)
        object.__setattr__(self, "h_y", h_y)
        object.__setattr__(self, "cache", {})
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

        Parameters
        ----------
        hazards : FloatArray
            ``(n, T, L, 2)`` cause-specific hazards by unit, node, level and cause.
        first : int
            The index of the cause counted first within a node.
        bounds : tuple of float
            The clip interval of every factor logit.
        held : tuple of BoolArray
            The ``(T, L)`` held cells, in the order above.

        Returns
        -------
        SeparableState
            The factors, with held cells at :math:`\\pm\\infty`.
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
        """The state after adding each factor's increment, then clipping; held cells stay.

        Parameters
        ----------
        delta_d : FloatArray
            ``(n, T, L)`` increment of :math:`\\beta_D`.
        delta_y : FloatArray
            ``(n, T, L)`` increment of :math:`\\beta_Y`.
        bounds : tuple of float
            The clip interval of every factor logit.

        Returns
        -------
        SeparableState
            The moved state, with an empty cache.
        """

        def step(beta: FloatArray, delta: FloatArray) -> FloatArray:
            return np.where(np.isinf(beta), beta, np.clip(beta + delta, *bounds))

        return SeparableState(step(self.beta_d, delta_d), step(self.beta_y, delta_y))

    def composed(self, levels: tuple[int, int]) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(n, T)`` composed survival, ``(n, T)`` survival before each node, and
        ``(n, T, 2)`` composed incidence by role, under the arms ``(a_D, a_Y)``.

        Each incidence is a running sum of non-negative products, so it is non-decreasing over
        the nodes in floating point.

        Parameters
        ----------
        levels : tuple of int
            The level codes ``(a_D, a_Y)``.

        Returns
        -------
        tuple of FloatArray
            The composed survival, the survival before each node, and the incidence by role.
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
        """The number of units."""
        return int(self.at_risk.shape[0])

    @property
    def bounds(self) -> tuple[float, float]:
        """The clip interval of every factor logit."""
        upper = float(np.log(self.alpha / (1.0 - self.alpha)))
        return -upper, upper

    @property
    def rows(self) -> IntArray:
        """``(n,)`` the unit index, for reading each unit's observed level."""
        return np.arange(self.n)

    def rows_d(self) -> BoolArray:
        """``(n, T)`` the rows of the competing factor: at risk."""
        return self.at_risk

    def rows_y(self) -> BoolArray:
        """``(n, T)`` the rows of the event factor: at risk and free of the competing cause."""
        return self.at_risk & (self.event_d == 0.0)


def _level_block(
    state: SeparableState, inputs: SeparableInputs, levels: tuple[int, int]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``(n, T)`` composed survival, ``(n, T, 2)`` composed incidence, and the ``(n, T)``
    weights of :math:`H_Y` at :math:`a_Y` and of :math:`H_D` at :math:`a_D`, shared by every
    target with these levels."""
    key = ("levels", levels)
    if state.cache.get("inputs") is not inputs:
        state.cache.clear()
        state.cache["inputs"] = inputs
    if key not in state.cache:
        a_d, a_y = levels
        survival, before, incidence = state.composed(levels)
        factual = np.concatenate(
            [np.ones_like(state.survival[:, :1]), state.survival[:, :-1]], axis=1
        )  # (n, T, L): S(k - 1 | a)
        lam_d = state.lam_d
        weight_y = _per(
            before * (1.0 - lam_d[:, :, a_d]) * inputs.inverse[:, :, a_y],
            factual[:, :, a_y] * (1.0 - lam_d[:, :, a_y]),
        )
        weight_d = _per(before * inputs.inverse[:, :, a_d], factual[:, :, a_d])
        state.cache[key] = (survival, incidence, weight_y, weight_d)
    return state.cache[key]


def _covariates(
    state: SeparableState, inputs: SeparableInputs, target: SeparableTarget
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(n, T)`` :math:`H_D` at :math:`a_D` and :math:`H_Y` at :math:`a_Y` for every unit, and
    ``(n,)`` the plug-in :math:`\\tilde F_j(t \\mid W)`.

    Both covariates are zero after the horizon.
    """
    if state.cache.get("inputs") is not inputs:
        state.cache.clear()
        state.cache["inputs"] = inputs
    key = ("target", target)
    if key in state.cache:
        return state.cache[key]
    survival, incidence, weight_y, weight_d = _level_block(state, inputs, target.levels)
    t = target.horizon
    j = 0 if target.role == FACTORS[0] else 1
    remaining = _per(incidence[:, t - 1, j][:, None] - incidence[:, :, j], survival)
    c_d = (1.0 if j == 0 else 0.0) - remaining
    c_y = (1.0 if j == 1 else 0.0) - remaining
    h_event = weight_y * c_y
    h_competing = weight_d * (c_d - state.h_y[:, :, target.levels[1]] * c_y)
    h_event[:, t:] = 0.0
    h_competing[:, t:] = 0.0
    out = (h_competing, h_event, incidence[:, t - 1, j])
    state.cache[key] = out
    return out


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


def _factor_fit(
    state: SeparableState,
    inputs: SeparableInputs,
    targets: Sequence[SeparableTarget],
    factor: int,
    covariates: Sequence[tuple[FloatArray, FloatArray, FloatArray]],
) -> FloatArray:
    """The ``(n, T, L)`` increment of one factor's logits from its fluctuation fit at frozen
    covariates.

    With the covariates of ``state`` frozen, the factor's coefficients enter its binomial
    log-likelihood as a logistic regression with the current logits as offset, one column per
    target.  The likelihood is concave in the coefficients, so Newton-Raphson with step halving
    reaches its maximum; duplicate or zero columns are resolved by the minimum-norm least-squares
    step.  The fit reads the logits without the clip, so a cell that starts at the clip (an
    out-of-fold start from a training fold with no event) moves inward when its own rows hold
    events, and a cell whose rows hold none heads outward until the state's clip stops it.  This
    is one step of the classical iterative TMLE (van der Laan and Rubin 2006).
    """
    rows, observed = inputs.rows, inputs.observed
    beta = state.beta_d if factor == 0 else state.beta_y
    events = inputs.event_d if factor == 0 else inputs.event_y
    factor_rows = inputs.rows_d() if factor == 0 else inputs.rows_y()
    offset = beta[rows, :, observed]
    used = factor_rows & np.isfinite(offset) & (inputs.weights != 0.0)[:, None]
    offset = np.where(used, offset, 0.0)
    weight = np.where(used, inputs.weights[:, None], 0.0)
    design = np.stack(
        [
            np.where((observed == target.levels[factor])[:, None], covariates[m][factor], 0.0)
            for m, target in enumerate(targets)
        ]
    )  # (M, n, T)
    flat = design.reshape(len(targets), -1)

    def logits(coefficients: FloatArray) -> FloatArray:
        return np.asarray(offset + np.tensordot(coefficients, design, axes=1), dtype=float)

    def loss(coefficients: FloatArray) -> float:
        eta = logits(coefficients)
        return -float(np.sum(weight * (events * log_expit(eta) + (1.0 - events) * log_expit(-eta))))

    coefficients = np.zeros(len(targets))
    current = loss(coefficients)
    for _ in range(_FIT_ITERATIONS):
        p = expit(logits(coefficients))
        gradient = flat @ (weight * (events - p)).reshape(-1)
        curvature = (weight * p * (1.0 - p)).reshape(-1)
        information = (flat * curvature) @ flat.T
        step = np.linalg.lstsq(information, gradient, rcond=1e-13)[0]
        if float(gradient @ step) <= 0.0:
            break
        slack = 1e-12 * max(1.0, abs(current))
        length = 1.0
        trial = loss(coefficients + step)
        while trial > current + slack and length > _MIN_STEP:
            length *= 0.5
            trial = loss(coefficients + length * step)
        if trial > current + slack:
            break
        coefficients = coefficients + length * step
        current = trial
        # Newton's steps shrink quadratically, so once a full step moves no logit by more than
        # _FIT_CHANGE the next would move them by about its square: the fit has converged,
        # whatever the scale of the weights.  The halving test allows the rounding of the loss.
        moved = np.abs(np.tensordot(step, design, axes=1))[used]
        if length == 1.0 and (moved.size == 0 or float(moved.max()) <= _FIT_CHANGE):
            break
    delta = np.zeros_like(beta)
    for m, target in enumerate(targets):
        delta[:, :, target.levels[factor]] += coefficients[m] * covariates[m][factor]
    return delta


def _iterative_direction(
    state: SeparableState,
    inputs: SeparableInputs,
    targets: Sequence[SeparableTarget],
    score: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The increments of both factor logits from their fluctuation fits at frozen covariates.

    The two factors' fits are separate, because a coefficient of one factor does not move the
    other factor's likelihood.
    """
    covariates = [_covariates(state, inputs, target) for target in targets]
    return (
        _factor_fit(state, inputs, targets, 0, covariates),
        _factor_fit(state, inputs, targets, 1, covariates),
    )


def _jacobian_direction(
    state: SeparableState,
    inputs: SeparableInputs,
    targets: Sequence[SeparableTarget],
    score: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """The Newton increments on the full score Jacobian, with the covariates moving.

    Each of the ``2M`` coefficients moves its own factor along its own covariate at its own
    level, the change of all ``2M`` scores is read by a central difference, and the step is the
    minimum-norm least-squares solution of ``J c = -s``.  A central difference is accurate to
    about the square of its length, which is what lets the solve keep the Jacobian's directions
    down to ``rcond = 1e-10``; nearly collinear targets put the root along them.  It costs
    ``4M`` score evaluations, so the walk calls it only after the iterative TMLE stops short.
    """
    covariates = [_covariates(state, inputs, target) for target in targets]
    directions: list[tuple[int, int]] = [
        (m, factor) for m in range(len(targets)) for factor in (0, 1)
    ]

    def increments(weights: FloatArray) -> tuple[FloatArray, FloatArray]:
        delta_d = np.zeros_like(state.lam_d)
        delta_y = np.zeros_like(state.h_y)
        for weight, (m, factor) in zip(weights, directions, strict=True):
            if weight != 0.0:
                level = targets[m].levels[factor]
                (delta_d if factor == 0 else delta_y)[:, :, level] += weight * covariates[m][factor]
        return delta_d, delta_y

    jacobian = np.empty((score.size, len(directions)))
    for k in range(len(directions)):
        unit = np.zeros(len(directions))
        unit[k] = _JACOBIAN_STEP
        up = scores(state.moved(*increments(unit), inputs.bounds), inputs, targets)[0]
        down = scores(state.moved(*increments(-unit), inputs.bounds), inputs, targets)[0]
        jacobian[:, k] = (up - down) / (2.0 * _JACOBIAN_STEP)
    coefficients = np.linalg.lstsq(jacobian, -score, rcond=1e-10)[0]
    return increments(coefficients)


def _separated(
    state: SeparableState,
    inputs: SeparableInputs,
    targets: Sequence[SeparableTarget],
    sigma: FloatArray,
) -> bool:
    """Whether rows the clip holds carry most of the remaining score.

    The rows whose factor logit at their observed level sits on either clip are read together,
    events and non-events, so their part of the score is the net residual of the cells the clip
    holds.  The fluctuation fit of the iterative TMLE reads the logits without the clip, so it
    takes a cell whose root lies inside the clip off the clip, and leaves on it only a cell whose
    rows hold no event (or only events), whose root lies at a hazard of zero (or one).  A cell
    near the clip but inside it is not counted: at its root its net residual vanishes, and only
    rounding is left.  The fit reads ``cell_separated`` when that part has at least
    :data:`_BOUND_SHARE` of the whole score's :math:`\\Sigma`-norm.  The solver offsets that
    part by moving other cells off their own solution, so the rest of the score can be as large
    as the whole and does not decide.
    """
    low, high = inputs.bounds
    rows, observed = inputs.rows, inputs.observed

    def near(beta: FloatArray, factor_rows: BoolArray) -> BoolArray:
        logit_at = beta[rows, :, observed]
        finite = np.isfinite(logit_at) & factor_rows
        lower = logit_at <= low + _AT_CLIP
        upper = logit_at >= high - _AT_CLIP
        return np.asarray(finite & (lower | upper), dtype=bool)

    near_d = near(state.beta_d, inputs.rows_d())
    near_y = near(state.beta_y, inputs.rows_y())
    if not (near_d.any() or near_y.any()):
        return False
    score, _ = scores(state, inputs, targets)
    residual_d, residual_y = _residuals(state, inputs)
    pinned = np.zeros_like(score)
    for m, target in enumerate(targets):
        h_d, h_y, _ = _covariates(state, inputs, target)
        a_d, a_y = target.levels
        at_d = (observed == a_d)[:, None] & near_d
        at_y = (observed == a_y)[:, None] & near_y
        pinned[2 * m] = np.mean(inputs.weights * np.where(at_d, h_d * residual_d, 0.0).sum(axis=1))
        pinned[2 * m + 1] = np.mean(
            inputs.weights * np.where(at_y, h_y * residual_y, 0.0).sum(axis=1)
        )
    total = float(np.sqrt(np.sum(score**2 / sigma)))
    heading_part = float(np.sqrt(np.sum(pinned**2 / sigma)))
    return heading_part >= _BOUND_SHARE * total


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
        The cap on steps taken, counting the steps of an abandoned fallback tail.
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
        if steps >= _STALL and trace[-1] > 0.5 * trace[-1 - _STALL]:
            break
    # The fallbacks, each from the best state so far: the classical iterative TMLE, then Newton
    # steps on the full score Jacobian.  Both take full steps, because a step that raises the
    # score can lead to the root; each keeps the best state and stops after a few steps
    # without a new best.  The record describes the reported (best) state, so the steps of an
    # abandoned tail are dropped.
    best = (trace[-1], state, score, scale, norm)
    best_steps = spent = steps
    phase: Literal["walk", "iterative", "jacobian"] = "walk"
    fallback = False
    fallbacks: tuple[tuple[Literal["iterative", "jacobian"], Any, int], ...] = (
        ("iterative", _iterative_direction, _ITERATIVE_STEPS),
        ("jacobian", _jacobian_direction, _JACOBIAN_STEPS),
    )
    for name, direction, cap in fallbacks:
        if best[0] <= tol or spent >= max_steps or best[4] <= 0.0:
            break
        if fallback and _separated(best[1], inputs, targets, sigma):
            break
        fallback = True
        _, state, score, scale, norm = best
        steps = best_steps
        del trace[steps + 1 :], norms[steps + 1 :], losses[steps + 1 :]
        mark, since, taken = best[0], 0, 0
        while best[0] > tol and spent < max_steps and since < _PATIENCE and taken < cap:
            delta_d, delta_y = direction(state, inputs, targets, score)
            state = state.moved(delta_d, delta_y, bounds)
            score, scale = scores(state, inputs, targets)
            norm = float(np.sqrt(np.sum(score**2 / sigma)))
            steps += 1
            spent += 1
            taken += 1
            trace.append(relative_score(score, scale))
            norms.append(norm)
            losses.append(walk_loss(state, inputs))
            if trace[-1] < best[0]:
                best, best_steps, phase = (trace[-1], state, score, scale, norm), steps, name
            if trace[-1] < _PROGRESS * mark:
                mark, since = trace[-1], 0
            else:
                since += 1
    _, state, score, scale, norm = best
    steps = best_steps
    del trace[steps + 1 :], norms[steps + 1 :], losses[steps + 1 :]
    relative = relative_score(score, scale)
    converged = bool(relative <= tol)
    failure: TargetingFailure | None = None
    if not converged:
        if spent >= max_steps:
            failure = "max_iter_reached"
        elif not fallback:
            failure = "line_search_exhausted"
        elif _separated(state, inputs, targets, sigma):
            failure = "cell_separated"
        else:
            failure = "newton_stalled"
    if failure is not None and warn:
        advice = {
            "cell_separated": (
                "A hazard the targets move can solve its score only at zero or one, as when "
                "one fold's cell holds no event of a cause; the bound set by alpha stops it"
            ),
            "newton_stalled": (
                "The iterative TMLE and the Newton steps stopped short of the root; check "
                "result.diagnostics for a positivity problem"
            ),
        }.get(
            failure, "Try a smaller step_size, or check result.diagnostics for a positivity problem"
        )
        _warn(
            f"one-step separable targeting stopped after {steps} step(s) with a relative score "
            f"of {relative:.3g} > {tol:g}. {advice}",
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
        targets=record_targets,
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
        failure=failure,
        tol=tol,
        phase=phase,
    )


def _held_cells(
    data: LongitudinalData,
    at_risk: BoolArray,
    event_d: FloatArray,
    event_y: FloatArray,
    n_levels: int,
) -> tuple[BoolArray, BoolArray, BoolArray, BoolArray]:
    """``(T, L)`` cells whose factor is held: empty and full :math:`D`, then empty and full
    :math:`Y` among the :math:`D`-free, each among the rows of positive weight at that level."""
    observed = np.asarray(np.nan_to_num(data.node_codes(1), nan=0.0), dtype=np.int64)
    positive = (np.asarray(data.weights, dtype=float) > 0.0)[:, None, None]
    at_level = observed[:, None, None] == np.arange(n_levels)[None, None, :]
    rows_d = at_risk[:, :, None] & positive & at_level
    rows_y = rows_d & (event_d == 0.0)[:, :, None]
    d = (event_d > 0.0)[:, :, None]
    y = (event_y > 0.0)[:, :, None]
    present_d = rows_d.any(axis=0)
    present_y = rows_y.any(axis=0)
    empty_d = present_d & ~(rows_d & d).any(axis=0)
    full_d = present_d & ~(rows_d & ~d).any(axis=0)
    empty_y = present_y & ~(rows_y & y).any(axis=0)
    full_y = present_y & ~(rows_y & ~y).any(axis=0)
    return empty_d, full_d, empty_y, full_y


def target_separable(
    data: LongitudinalData,
    plans: Sequence[Plan],
    separables: Sequence[SeparablePlan],
    hazards: FloatArray,
    cumulative: Sequence[FloatArray],
    observed_cumulative: tuple[FloatArray, FloatArray],
    *,
    horizons: Sequence[int],
    alpha: float,
    tol: float,
    step_size: float,
) -> dict[tuple[str, str | None, int], RegimenFit]:
    """Target every regimen of a separable fit and report one fit per regimen, cause and horizon.

    The static arms are targeted as separable interventions with equal levels, in the same
    walk, so one set of factors carries every reported curve.

    Parameters
    ----------
    data : LongitudinalData
        A two-cause survival panel of a held design.
    plans : sequence of Plan
        The static arms' plans, in report order.
    separables : sequence of SeparablePlan
        The separable interventions, in report order after the static arms.
    hazards : FloatArray
        ``(n, T, L, 2)`` initial cause-specific hazards fitted in the declared order.
    cumulative : sequence of FloatArray
        One ``(n, T)`` bounded :math:`g G_k` per level.
    observed_cumulative : tuple of FloatArray
        The unbounded and bounded ``(n, T)`` :math:`g G_k` at each unit's observed level.
    horizons : sequence of int
        The reported horizons, counted from one.
    alpha : float
        The probability bound of each factor.
    tol : float
        Relative-score tolerance.
    step_size : float
        The first step of the walk.

    Returns
    -------
    dict
        One :class:`~cleverly.longitudinal.sequential.RegimenFit` per
        ``(regimen, cause, horizon)``.

    Raises
    ------
    CapabilityError
        If a level whose component acts on the second cause holds a node where every event
        was the first cause, so the second cause's hazard there is not identified.
    """
    causes = tuple(data.cause_labels)
    first = separables[0].first
    second = 1 - first
    at_risk, events = event_indicators(data)
    inverse = np.stack([1.0 / np.asarray(c, dtype=float) for c in cumulative], axis=2)
    observed = np.asarray(np.nan_to_num(data.node_codes(1), nan=0.0), dtype=np.int64)
    weights = np.asarray(data.weights, dtype=float)
    inputs = SeparableInputs(
        at_risk=at_risk,
        event_d=events[:, :, first],
        event_y=events[:, :, second],
        observed=observed,
        inverse=inverse,
        weights=weights,
        alpha=alpha,
    )
    held = _held_cells(data, at_risk, inputs.event_d, inputs.event_y, inverse.shape[2])
    regimens: list[tuple[str, tuple[int, int], Any]] = [
        (plan.label, (int(plan.values[0, 0]),) * 2, plan) for plan in plans
    ]
    regimens += [(sep.label, sep.levels, sep) for sep in separables]
    horizons = tuple(horizons)
    last = max(horizons)
    full_d, full_y = held[1], held[3]
    targets: list[SeparableTarget] = []
    for label, levels, _ in regimens:
        a_d, a_y = levels
        if a_d != a_y:
            blocked = np.flatnonzero(full_d[:last, a_y])
            if blocked.size:
                raise CapabilityError(
                    f"Separable {label!r} reads the hazard of {causes[second]!r} among those "
                    f"free of {causes[first]!r} at level code {a_y}, but every unit at risk "
                    f"there had {causes[first]!r} at node {int(blocked[0]) + 1}, so that "
                    "hazard is not identified (positivity (7) of Stensrud et al. 2022).",
                    reason=RefusalReason.INVALID_DECLARATION,
                    alternative="Fit fewer horizons, or coarsen the time grid.",
                )
            dropped = np.flatnonzero(full_y[:last, a_d])
            if dropped.size:
                node = int(dropped[0]) + 1
                kept = tuple(t for t in horizons if t <= node)
                _warn(
                    f"Separable {label!r}: no unit at level code {a_d} stays event-free past "
                    f"node {node}, so the horizons after it are not reported for this regimen "
                    f"({[t for t in horizons if t > node]}).",
                    stacklevel=3,
                )
            else:
                kept = horizons
        else:
            kept = horizons
        for j, cause in ((first, causes[first]), (second, causes[second])):
            role = FACTORS[0] if j == first else FACTORS[1]
            targets.extend(SeparableTarget(label, levels, role, cause, t) for t in kept)
    start = SeparableState.from_hazards(hazards, first, inputs.bounds, held)
    state, record = walk_separable(start, inputs, targets, step_size=step_size, tol=tol)
    levels_out, curves = influence(state, inputs, targets)
    by_label = {label: spec for label, _, spec in regimens}
    out: dict[tuple[str, str | None, int], RegimenFit] = {}
    for m, target in enumerate(targets):
        a_d, a_y = target.levels
        _, before, _ = state.composed(target.levels)
        factual = np.concatenate(
            [np.ones_like(state.survival[:, :1]), state.survival[:, :-1]], axis=1
        )
        reads = (observed == a_d) | (observed == a_y)
        weight = _per(
            before * inputs.inverse[inputs.rows, :, observed],
            factual[inputs.rows, :, observed],
        )
        clever = np.where(at_risk & reads[:, None] & (before > 0.0), weight, 0.0)
        clever[:, target.horizon :] = 0.0
        spec = by_label[target.label]
        regimen = spec.regimen if isinstance(spec, Plan) else spec
        out[(target.label, target.cause, target.horizon)] = RegimenFit(
            regimen=regimen,
            psi_scaled=float(levels_out[m]),
            influence_curve_scaled=np.asarray(curves[m], dtype=float),
            horizon=target.horizon,
            cause=target.cause,
            steps=(),
            cumulative_unbounded=observed_cumulative[0],
            cumulative=observed_cumulative[1],
            assignment=np.repeat(observed[:, None].astype(float), data.n_times, axis=1),
            obs_weights=weights,
            policy=(),
            hazard=record,
            hazard_clever=clever,
            hazard_constant_nodes=_constant_nodes(held, target),
        )
    return out


def _constant_nodes(
    held: tuple[BoolArray, BoolArray, BoolArray, BoolArray], target: SeparableTarget
) -> tuple[int, ...]:
    """The nodes up to the horizon with a held cell the target reads: :math:`h_Y` at
    :math:`a_Y`, :math:`\\lambda_D` at :math:`a_D`, and a full :math:`D` cell at :math:`a_Y`.

    A full cell of :math:`h_Y` at :math:`a_D` is not flagged: the horizons after it are
    dropped, and at its own node :math:`h_Y(k \\mid a_D)` enters no kept target."""
    empty_d, full_d, empty_y, full_y = held
    a_d, a_y = target.levels
    t = target.horizon
    flagged = (
        empty_y[:t, a_y] | full_y[:t, a_y] | empty_d[:t, a_d] | full_d[:t, a_d] | full_d[:t, a_y]
    )
    return tuple(int(k) + 1 for k in np.flatnonzero(flagged))

r"""The drift-corrected survival TMLE of Díaz (2019).

Díaz, I. (2019).  Statistical inference for data-adaptive doubly robust estimators with
survival outcomes.  *Statistics in Medicine* 38(15), 2735-2748 (arXiv:1709.00401v3).  The
parameter is the treatment-specific risk :math:`F_a(\tau) = 1 - \theta_a(\tau)`, with
:math:`\theta_a(\tau) = E_W S(\tau \mid a, W)`, for one binary baseline treatment, baseline
covariates, one event type and right censoring.

Hazard ``LTMLE`` is doubly robust consistent, but its normal limit needs every nuisance to be
consistent at a rate.  Section 5 of the source adds a *drift* correction: for each horizon and
each arm it fits univariate regressions of residuals on scalar indices of the initial fits
(eqs. (7), (8)), builds the clever covariates of its eq. (9), and fluctuates the treatment
mechanism :math:`g_A`, the censoring hazard :math:`g_R` and the event hazard :math:`h` along the
logistic submodels (14) to (16) until every :math:`\varepsilon` is below
:math:`10^{-4} n^{-3/5}`.  The influence curve is :math:`D - D_L - D_R - D_A` (Theorem 2).

The source orders the event at :math:`t` before censoring at :math:`t`, and the container orders
censoring at node :math:`k` before the event there.  So the source's :math:`1 - g_R(m)` is the
container's retention :math:`c_{m+1}`, :math:`G(t)` is the container's :math:`G_t`, and the
source's :math:`R_0` is the container's :math:`C_1`, which a
:meth:`~cleverly.longitudinal.LongitudinalData.from_time_to_event` container holds at one.
:func:`paper_view` relabels once, and every later formula uses the source's indexing.

Three choices differ from the source, each stated in the design (X91 part F, rev 2.1):

- a declared factor (a known treatment mechanism, a declared retention or a node with no
  censoring) is never fluctuated, its residual regressions on the hazard side are set to their
  population value zero, and its score is not subtracted from the curve;
- every denominator is floored: the cumulative :math:`g_A G` and :math:`q` at ``g_bounds[0]``
  (``q`` also capped at ``1 - g_bounds[0]``), the products :math:`B_k`, :math:`D_k` at
  :math:`10^{-8}`.  The products :math:`U`, :math:`V` and the ratios of :math:`B`, :math:`D` are
  formed as forward partial products, which is the same algebra without a division;
- the loop stops after ``max_iter`` iterations with a warning.

Four signs and indices differ from eq. (9) as the design transcribed it, each pinned by the
exact-law drift identity (check 5 in ``tests/unit/test_survival_drtmle.py``):
:math:`D_R = +\sum_k A J_k H_R(k)\{R_k - g_R(k)\}`, since censoring lowers the cumulative
:math:`g_A G` while treatment raises it; the :math:`e_R` sum enters :math:`H_L` with the sign
opposite to the :math:`e_A` term, for the same reason; the :math:`e_A` term's products are
:math:`D_0(t) B_0(t - 1)`, on the index :math:`C_h(0) = S(\tau)` that :math:`q` uses; and
:math:`e_A` itself is regressed on :math:`S(\tau)`, not on :math:`M = \sum_t S(t)`, because its
term in the remainder telescopes to :math:`S_0(\tau) - \hat S(\tau)`.  A re-read of the source
(``X91-sources.md`` section 6a.4) traces the first three to slips in its supplementary proof.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np
from scipy.special import expit, logit

from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from .._typing import BoolArray, FloatArray, Learner
from .._warnings import warn as _warn
from ..exceptions import CapabilityError, ConvergenceWarning, LongitudinalError
from ..inference.multiplier import DEFAULT_BAND_SCALE, DEFAULT_KIND, DEFAULT_N_REPLICATES
from ..learners.crossfit import Folds
from ..utils.bounds import bound
from ._kernel import SmoothFit, smooth
from .collaborative import all_censored_nodes, all_censored_refusal, eligible_censoring
from .data import LongitudinalData
from .estimator import (
    DEFAULT_LTMLE_G_BOUNDS,
    LTMLE,
    LongitudinalResult,
    _fit_key,
    _frozen_plans,
    _HazardBlock,
    _HazardReplay,
    _LongitudinalReplayRecipe,
    _Reported,
)
from .hazard import HazardFit, HazardState, InitialHazards, fit_hazards, hazard_inputs, level_label
from .regimen import DynamicRegimen, Plan, resolve_regimens
from .separable import split_separable
from .sequential import Mechanism, NodeBound, RegimenFit

if TYPE_CHECKING:
    from .._summary import Detail, Summary

__all__ = [
    "DENOMINATOR_FLOOR",
    "DriftRecord",
    "SurvivalDRTMLE",
    "drift_design_refusal",
    "paper_view",
    "target_drift",
]

#: The floor of the products :math:`B_k(t)`, :math:`D_k(t)` in a denominator, for underflow.
DENOMINATOR_FLOOR: Final = 1e-8
#: The guards the drift loop accepts: both blocks, or none (hazard ``LTMLE``).
_GUARDS: Final = (("Q", "g"), ())
#: Newton settings of the one- and two-parameter logistic fits of :math:`\varepsilon`.
_NEWTON_MAX_ITER: Final = 100
_NEWTON_TOL: Final = 1e-12
_NEWTON_HALVINGS: Final = 40


# ------------------------------------------------------------------------------ the view


@dataclass(frozen=True)
class PaperView:
    """One arm of a survival panel in the source's indexing.

    Every array has one row per unit.  Column ``t - 1`` of a hazard-side array is time
    :math:`t = 1, \\ldots, K`; column ``m`` of a censoring-side array is time
    :math:`m = 0, \\ldots, K - 1`.

    Parameters
    ----------
    arm : FloatArray
        ``(n,)`` the indicator :math:`1\\{A = a\\}`, written :math:`A` in the source.
    g_a : FloatArray
        ``(n,)`` the initial :math:`g_A(W) = P(A = a \\mid W)`.
    g_r : FloatArray
        ``(n, K)`` the initial :math:`g_R(m, W)`, one minus the container's retention at node
        ``m + 1``, at level ``a``.
    h : FloatArray
        ``(n, K)`` the initial event hazard :math:`h(t, W)` at level ``a``.
    at_risk : BoolArray
        ``(n, K)`` :math:`I_t`: uncensored before :math:`t` and event-free before :math:`t`.
    events : FloatArray
        ``(n, K)`` :math:`L_t` on the rows at risk, zero elsewhere.
    eligible : BoolArray
        ``(n, K)`` :math:`J_m`: at risk of censoring at :math:`m`.
    censored : FloatArray
        ``(n, K)`` :math:`R_m` on the eligible rows, zero elsewhere.
    held_treatment : bool
        Whether :math:`g_A` is declared.
    held_censoring : BoolArray
        ``(K,)`` whether :math:`g_R(m)` is declared or held at zero.
    held_hazard : BoolArray
        ``(K,)`` whether the hazard cell at level ``a`` and time :math:`t` is held at zero or
        one, so that no fluctuation moves it.
    """

    arm: FloatArray
    g_a: FloatArray
    g_r: FloatArray
    h: FloatArray
    at_risk: BoolArray
    events: FloatArray
    eligible: BoolArray
    censored: FloatArray
    held_treatment: bool
    held_censoring: BoolArray
    held_hazard: BoolArray

    @property
    def n(self) -> int:
        """The number of units."""
        return int(self.arm.shape[0])

    @property
    def n_times(self) -> int:
        """The number of times :math:`K`."""
        return int(self.h.shape[1])


def paper_view(
    data: LongitudinalData,
    mechanism: Mechanism,
    initial: InitialHazards,
    code: int,
    *,
    alpha: float,
    g_bounds: tuple[float, float],
) -> PaperView:
    """Relabel one arm of a held survival panel into the source's indexing (section 3).

    Parameters
    ----------
    data : LongitudinalData
        A held survival panel with one cause.
    mechanism : Mechanism
        The fitted mechanism, evaluated at every level.
    initial : InitialHazards
        The initial hazards.
    code : int
        The level code of the arm.
    alpha : float
        The probability bound of every hazard; an estimated :math:`g` is clipped into it too.
    g_bounds : tuple of float
        The cumulative mechanism bounds.

    Returns
    -------
    PaperView
        The arm in the source's indexing.
    """
    levels = [level_label(c) for c in range(len(data.treatment_levels[0]))]
    cumulative = [
        mechanism.cumulative(data, _level_plan(data, c), g_bounds) for c in range(len(levels))
    ]
    inputs = hazard_inputs(data, (_level_plan(data, code),), cumulative, alpha=alpha)
    start = HazardState.from_hazards(initial.hazards, inputs.bounds, inputs.empty, inputs.full)
    label = levels[code]
    n, n_times = data.n, data.n_times
    observed = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
    arm = (observed == code).astype(float)
    shrink = 1.0 - alpha
    held_treatment = 1 in mechanism.known_treatment_nodes
    g_a = np.asarray(mechanism.treatment[0][label], dtype=float)
    if not held_treatment:
        g_a = np.clip(g_a, shrink, 1.0 - shrink)
    g_r = np.zeros((n, n_times))
    held_censoring = np.ones(n_times, dtype=bool)
    if data.censoring_names:
        held_nodes = set(mechanism.no_censoring_nodes) | set(mechanism.known_censoring_nodes)
        for m in range(n_times):
            retention = np.asarray(mechanism.censoring[m][label], dtype=float)
            held = (m + 1) in held_nodes
            held_censoring[m] = held
            g_r[:, m] = 1.0 - retention if held else np.clip(1.0 - retention, shrink, 1.0 - shrink)
        eligible, censored = eligible_censoring(data)
    else:
        eligible = np.zeros((n, n_times), dtype=bool)
        censored = np.zeros((n, n_times), dtype=bool)
    return PaperView(
        arm=arm,
        g_a=g_a,
        g_r=g_r,
        h=np.asarray(start.hazards[:, :, code, 0], dtype=float),
        at_risk=np.asarray(inputs.at_risk, dtype=bool),
        events=np.asarray(inputs.events[:, :, 0], dtype=float),
        eligible=np.asarray(eligible, dtype=bool),
        censored=np.asarray(censored, dtype=float),
        held_treatment=held_treatment,
        held_censoring=held_censoring,
        held_hazard=np.asarray(inputs.empty[:, code, 0] | inputs.full[:, code], dtype=bool),
    )


def _level_plan(data: LongitudinalData, code: int) -> Plan:
    from .regimen import Regimen

    level = data.treatment_levels[0][code]
    return Plan(
        Regimen(level_label(code), (level,) * data.n_decisions),
        np.full((data.n, data.n_times), float(code)),
    )


# -------------------------------------------------------------------------- the state


@dataclass(frozen=True)
class _State:
    """The current :math:`(\\tilde g_A, \\tilde g_R, \\tilde h)` of one arm."""

    g_a: FloatArray  # (n,)
    g_r: FloatArray  # (n, K)
    h: FloatArray  # (n, K)

    def survival(self, tau: int) -> FloatArray:
        """``(n, tau + 1)`` :math:`S(t)`, :math:`t = 0, \\ldots, \\tau`."""
        out = np.ones((self.h.shape[0], tau + 1))
        out[:, 1:] = np.cumprod(1.0 - self.h[:, :tau], axis=1)
        return out

    def retention(self, tau: int) -> FloatArray:
        """``(n, tau + 1)`` :math:`G(t) = \\prod_{m < t} \\{1 - g_R(m)\\}`, :math:`t \\le \\tau`."""
        out = np.ones((self.h.shape[0], tau + 1))
        out[:, 1:] = np.cumprod(1.0 - self.g_r[:, :tau], axis=1)
        return out

    def pi(self, tau: int) -> FloatArray:
        """``(n, tau + 1)`` the raw cumulative :math:`g_A G(t)`."""
        return np.asarray(self.g_a[:, None] * self.retention(tau), dtype=float)


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """``numerator / denominator``, and zero where the denominator is zero.

    Every survival ratio here has a numerator that is zero wherever its denominator is.
    """
    return np.asarray(
        np.divide(
            numerator,
            denominator,
            out=np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape)),
            where=denominator > 0.0,
        ),
        dtype=float,
    )


# ------------------------------------------------------------------- the reductions


@dataclass(frozen=True)
class ReductionFit:
    """One auxiliary regression of the drift correction.

    Parameters
    ----------
    family : str
        ``"e_R"``, ``"e_L"``, ``"e_A"``, ``"u"``, ``"v"``, ``"b"``, ``"d"`` or ``"q"``.
    index : int
        The first subscript: :math:`t` of :math:`C_g(t)` or :math:`k` of :math:`C_h(k)`; ``-1``
        for :math:`e_A` and :math:`q`.
    time : int
        The argument :math:`m`, or ``-1`` for a regression without one.
    bandwidth : float or None
        The chosen bandwidth, ``None`` for the subpopulation mean.
    degenerate : bool
        Whether the regression is the subpopulation mean.
    rows : int
        The subpopulation rows.
    """

    family: str
    index: int
    time: int
    bandwidth: float | None
    degenerate: bool
    rows: int


#: ``regress(index, outcome, rows)``: a regression evaluated at every row.
Regression = Callable[[FloatArray, FloatArray, BoolArray], FloatArray]


class _Reductions:
    """The Step 1 regressions of one arm and horizon, fitted once and held fixed."""

    def __init__(
        self,
        view: PaperView,
        initial: _State,
        tau: int,
        g_bounds: tuple[float, float],
        regress: Regression | None = None,
    ) -> None:
        self.view = view
        self.regress = regress
        self.tau = tau
        self.records: list[ReductionFit] = []
        self._cache: dict[tuple[str, int, int], FloatArray] = {}
        survival = initial.survival(tau)
        pi = bound(initial.pi(tau), *g_bounds)
        self.c_g = pi  # column t: C_g(t), t = 1..tau
        self.c_h = _ratio(survival[:, tau : tau + 1], survival)  # column k: C_h(k), k = 0..tau
        self.s_tau = survival[:, tau]
        self.initial = initial
        self.pi = pi

    def fit(
        self,
        family: str,
        index: int,
        time: int,
        x: FloatArray,
        y: FloatArray,
        rows: BoolArray,
    ) -> FloatArray:
        key = (family, index, time)
        if key not in self._cache:
            if self.regress is not None:
                self._cache[key] = np.asarray(self.regress(x, y, rows), dtype=float)
                self.records.append(ReductionFit(family, index, time, None, False, int(rows.sum())))
            else:
                result: SmoothFit = smooth(x, y, rows)
                self._cache[key] = result.values
                self.records.append(
                    ReductionFit(
                        family, index, time, result.bandwidth, result.degenerate, result.rows
                    )
                )
        return self._cache[key]

    # -- the families of section 4.2

    def e_r(self, k: int) -> FloatArray:
        view = self.view
        if view.held_censoring[k]:
            return np.zeros(view.n)
        rows = view.eligible[:, k] & (view.arm > 0.0)
        outcome = _ratio(view.censored[:, k] - self.initial.g_r[:, k], self.pi[:, k + 1])
        return self.fit("e_R", k, -1, self.c_h[:, k], outcome, rows)

    def e_l(self, t: int) -> FloatArray:
        view = self.view
        rows = view.at_risk[:, t - 1] & (view.arm > 0.0)
        outcome = self.c_h[:, t] * (view.events[:, t - 1] - self.initial.h[:, t - 1])
        return self.fit("e_L", t, -1, self.c_g[:, t], outcome, rows)

    def e_a(self) -> FloatArray:
        view = self.view
        if view.held_treatment:
            return np.zeros(view.n)
        outcome = (view.arm - self.initial.g_a) / self.initial.g_a
        return self.fit("e_A", -1, -1, self.s_tau, outcome, np.ones(view.n, dtype=bool))

    def q(self, g_lower: float) -> FloatArray:
        view = self.view
        fitted = self.fit("q", -1, -1, self.s_tau, view.arm, np.ones(view.n, dtype=bool))
        return np.asarray(fitted, dtype=float)

    def u(self, t: int, m: int) -> FloatArray:
        view = self.view
        rows = view.eligible[:, m] & (view.arm > 0.0)
        return self.fit("u", t, m, self.c_g[:, t], view.censored[:, m], rows)

    def v(self, t: int, m: int) -> FloatArray:
        view = self.view
        rows = view.at_risk[:, m - 1] & (view.arm > 0.0)
        return self.fit("v", t, m, self.c_g[:, t], view.events[:, m - 1], rows)

    def b(self, k: int, m: int) -> FloatArray:
        view = self.view
        rows = view.at_risk[:, m - 1] & (view.arm > 0.0)
        return self.fit("b", k, m, self.c_h[:, k], view.events[:, m - 1], rows)

    def d(self, k: int, m: int) -> FloatArray:
        view = self.view
        rows = view.eligible[:, m] & (view.arm > 0.0)
        return self.fit("d", k, m, self.c_h[:, k], view.censored[:, m], rows)


@dataclass(frozen=True)
class _Coefficients:
    """The parts of eq. (9) that the loop holds fixed: reductions and their products.

    ``a[:, t - 1]`` is :math:`U_t(t) V_t(t - 1) e_L(t)`; ``r[k][:, t - 1]`` is
    :math:`[V_t(t-1)/V_t(k)][U_t(t)/U_t(k)] e_L(t)` for :math:`t > k`; ``l[t - 1][:, k]`` is
    :math:`-[B_k(k)/B_k(t-1)][D_k(k)/D_k(t)] e_R(k)`; ``c[:, t - 1]`` is
    :math:`e_A / \\{q D_0(t) B_0(t-1)\\}`.
    """

    a: FloatArray
    r: tuple[FloatArray, ...]
    l: tuple[FloatArray, ...]
    c: FloatArray
    hits: dict[str, int]


def _forward(values: Sequence[FloatArray], n: int) -> FloatArray:
    """The product :math:`\\prod (1 - v)` over ``values``, one when empty."""
    out = np.ones(n)
    for value in values:
        out = out * (1.0 - value)
    return out


def _a_term_index(t: int) -> int:
    """The index :math:`C_h(k)` of the products in the :math:`e_A` term of :math:`H_L(t)`.

    :math:`k = 0`, whose :math:`C_h(0) = S(\\tau)` is the index of :math:`q`, so that
    :math:`q D_0(t) B_0(t - 1)` estimates :math:`P(A = 1, I_t = 1)` given one index.  The
    transcription :math:`D_t(t) B_t(t - 1)` conditions at :math:`t = \\tau` on the constant
    :math:`C_h(\\tau)` and leaves a first-order term in the drift identity (check 5).
    """
    return 0


def fit_coefficients(
    reductions: _Reductions, tau: int, g_lower: float, *, drift: bool = True
) -> _Coefficients:
    """Fit every Step 1 regression of one arm and horizon and form the fixed products.

    Parameters
    ----------
    reductions : _Reductions
        The regressions of the arm and horizon.
    tau : int
        The horizon.
    g_lower : float
        The floor of :math:`q` and the cap :math:`1 - g_{\\text{lower}}`.
    drift : bool
        ``False`` forces every drift covariate to zero, a check device.

    Returns
    -------
    _Coefficients
        The fixed parts of eq. (9).
    """
    view = reductions.view
    n = view.n
    hits = {"q": 0, "B": 0, "D": 0}
    a = np.zeros((n, tau))
    c = np.zeros((n, tau))
    r = [np.zeros((n, tau)) for _ in range(tau)]
    l_terms = [np.zeros((n, tau)) for _ in range(tau)]
    if not drift:
        return _Coefficients(a, tuple(r), tuple(l_terms), c, hits)
    e_l = {t: reductions.e_l(t) for t in range(1, tau + 1)}
    if np.any([np.any(value != 0.0) for value in e_l.values()]):
        for t in range(1, tau + 1):
            u = [reductions.u(t, m) for m in range(t)]  # m = 0..t-1
            v = [reductions.v(t, m) for m in range(1, t)]  # m = 1..t-1, at index m - 1
            a[:, t - 1] = _forward(u, n) * _forward(v, n) * e_l[t]
            for k in range(t):
                # [V_t(t-1)/V_t(k)] = prod_{m=k+1}^{t-1}; [U_t(t)/U_t(k)] = prod_{m=k}^{t-1}.
                r[k][:, t - 1] = _forward(v[k:], n) * _forward(u[k:], n) * e_l[t]
    e_r = {k: reductions.e_r(k) for k in range(tau)}
    e_a = reductions.e_a()
    q_raw = reductions.q(g_lower)
    q = np.clip(q_raw, g_lower, 1.0 - g_lower)
    hits["q"] = int(np.sum(q != q_raw))
    for t in range(1, tau + 1):
        for k in range(t):
            if not np.any(e_r[k] != 0.0):
                continue
            # B_k(k)/B_k(t-1) = 1/prod_{m=k+1}^{t-1}; D_k(k)/D_k(t) = 1/prod_{m=k}^{t-1}.
            b_prod = _forward([reductions.b(k, m) for m in range(k + 1, t)], n)
            d_prod = _forward([reductions.d(k, m) for m in range(k, t)], n)
            hits["B"] += int(np.sum(b_prod < DENOMINATOR_FLOOR))
            hits["D"] += int(np.sum(d_prod < DENOMINATOR_FLOOR))
            l_terms[t - 1][:, k] = -e_r[k] / (
                np.maximum(b_prod, DENOMINATOR_FLOOR) * np.maximum(d_prod, DENOMINATOR_FLOOR)
            )
        if np.any(e_a != 0.0):
            index = _a_term_index(t)
            d_full = _forward([reductions.d(index, m) for m in range(t)], n)
            b_full = _forward([reductions.b(index, m) for m in range(1, t)], n)
            hits["B"] += int(np.sum(b_full < DENOMINATOR_FLOOR))
            hits["D"] += int(np.sum(d_full < DENOMINATOR_FLOOR))
            c[:, t - 1] = e_a / (
                q * np.maximum(d_full, DENOMINATOR_FLOOR) * np.maximum(b_full, DENOMINATOR_FLOOR)
            )
    return _Coefficients(a, tuple(r), tuple(l_terms), c, hits)


# ------------------------------------------------------------------ the covariates


@dataclass(frozen=True)
class Covariates:
    """The clever covariates of eq. (9) at one state.

    Parameters
    ----------
    h_a : FloatArray
        ``(n,)`` :math:`H_A`.
    h_r : FloatArray
        ``(n, tau)`` :math:`H_R(k)`, :math:`k = 0, \\ldots, \\tau - 1`.
    h_l : FloatArray
        ``(n, tau)`` :math:`H_L(t)` at column ``t - 1``.
    z : FloatArray
        ``(n, tau)`` :math:`Z(t)` at column ``t - 1``.
    floor_hits : int
        The cumulative :math:`g_A G` values the ``g_bounds`` floor moved.
    """

    h_a: FloatArray
    h_r: FloatArray
    h_l: FloatArray
    z: FloatArray
    floor_hits: int


def clever_covariates(
    state: _State, coefficients: _Coefficients, tau: int, g_bounds: tuple[float, float]
) -> Covariates:
    """Eq. (9) at ``state`` with the fixed reductions.

    Parameters
    ----------
    state : _State
        The current :math:`(\\tilde g_A, \\tilde g_R, \\tilde h)`.
    coefficients : _Coefficients
        The fixed parts.
    tau : int
        The horizon.
    g_bounds : tuple of float
        The bounds of the cumulative :math:`g_A G`.

    Returns
    -------
    Covariates
        :math:`H_A`, :math:`H_R`, :math:`H_L` and :math:`Z`.
    """
    raw = state.pi(tau)
    pi = bound(raw, *g_bounds)
    hits = int(np.sum(pi[:, 1:] != raw[:, 1:]))
    survival = state.survival(tau)
    n = raw.shape[0]
    h_a = (coefficients.a / pi[:, 1:]).sum(axis=1)
    h_r = np.zeros((n, tau))
    for k in range(tau):
        # [1/(g_A G(k+1))] [G(k)/G(t)] = pi_k / (pi_{k+1} pi_t).
        weight = raw[:, k : k + 1] / (pi[:, k + 1 : k + 2] * pi[:, 1:])
        h_r[:, k] = (coefficients.r[k] * weight).sum(axis=1)
    h_l = np.zeros((n, tau))
    for t in range(1, tau + 1):
        inner = (
            _ratio(survival[:, t - 1 : t], survival[:, :t]) * coefficients.l[t - 1][:, :t]
        ).sum(axis=1) + survival[:, t - 1] * coefficients.c[:, t - 1]
        h_l[:, t - 1] = _ratio(survival[:, tau], survival[:, t]) * inner
    z = _ratio(survival[:, tau : tau + 1], survival[:, 1:]) / pi[:, 1:]
    return Covariates(h_a, h_r, h_l, z, hits)


# --------------------------------------------------------------------- the submodels


def _logistic_epsilon(y: FloatArray, offset: FloatArray, x: FloatArray) -> FloatArray:
    """The unpenalized logistic maximum-likelihood :math:`\\varepsilon` of ``y`` on ``x``.

    A column identically zero on the rows keeps :math:`\\varepsilon = 0`.

    Parameters
    ----------
    y : FloatArray
        ``(N,)`` binary outcomes.
    offset : FloatArray
        ``(N,)`` logit offsets.
    x : FloatArray
        ``(N, p)`` covariates.

    Returns
    -------
    FloatArray
        ``(p,)`` the fitted coefficients, non-finite when Newton fails.
    """
    p = x.shape[1]
    eps = np.zeros(p)
    if y.size == 0:
        return eps
    live = np.flatnonzero(np.any(x != 0.0, axis=0))
    if live.size == 0:
        return eps
    design = x[:, live]
    beta = np.zeros(live.size)

    def loglik(b: FloatArray) -> float:
        eta = offset + design @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    current = loglik(beta)
    for _ in range(_NEWTON_MAX_ITER):
        mu = expit(offset + design @ beta)
        gradient = design.T @ (y - mu)
        hessian = (design * (mu * (1.0 - mu))[:, None]).T @ design
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            return np.full(p, np.nan)
        size = 1.0
        accepted = False
        for _ in range(_NEWTON_HALVINGS):
            trial = beta + size * step
            value = loglik(trial)
            if value >= current - 1e-14 * max(1.0, abs(current)):
                accepted = True
                break
            size *= 0.5
        if not accepted:
            # No halving raised the likelihood: the current point is the maximum to rounding.
            break
        beta, current = trial, value
        if np.max(np.abs(size * step)) < _NEWTON_TOL:
            break
    eps[live] = beta
    return eps


def fluctuate_once(
    view: PaperView,
    state: _State,
    covariates: Covariates,
    tau: int,
    alpha: float,
) -> tuple[_State, FloatArray]:
    """Fit the submodels (14) to (16) from ``state`` and apply them.

    Parameters
    ----------
    view : PaperView
        The arm.
    state : _State
        The current state.
    covariates : Covariates
        Eq. (9) at ``state``.
    tau : int
        The horizon.
    alpha : float
        The probability bound of every fluctuated probability.

    Returns
    -------
    tuple
        The new state and :math:`(\\varepsilon_A, \\varepsilon_R, \\varepsilon_{L,1},
        \\varepsilon_{L,2})`.
    """
    clip = float(logit(alpha))
    treated = view.arm > 0.0
    eps = np.zeros(4)
    g_a = state.g_a
    if not view.held_treatment:
        eps[0] = _logistic_epsilon(view.arm, logit(state.g_a), covariates.h_a[:, None])[0]
    estimated = [m for m in range(tau) if not view.held_censoring[m]]
    if estimated:
        ys, offsets, xs = [], [], []
        for m in estimated:
            rows = view.eligible[:, m] & treated
            ys.append(view.censored[rows, m])
            offsets.append(logit(state.g_r[rows, m]))
            xs.append(covariates.h_r[rows, m])
        eps[1] = _logistic_epsilon(
            np.concatenate(ys), np.concatenate(offsets), np.concatenate(xs)[:, None]
        )[0]
    nodes = [t for t in range(1, tau + 1) if not view.held_hazard[t - 1]]
    if nodes:
        ys, offsets, xs = [], [], []
        for t in nodes:
            rows = view.at_risk[:, t - 1] & treated
            ys.append(view.events[rows, t - 1])
            offsets.append(logit(state.h[rows, t - 1]))
            xs.append(np.column_stack([covariates.h_l[rows, t - 1], covariates.z[rows, t - 1]]))
        eps[2:] = _logistic_epsilon(np.concatenate(ys), np.concatenate(offsets), np.vstack(xs))
    if not np.all(np.isfinite(eps)):
        return state, eps
    if not view.held_treatment:
        g_a = expit(np.clip(logit(state.g_a) + eps[0] * covariates.h_a, -clip, clip))
    g_r = state.g_r.copy()
    for m in estimated:
        g_r[:, m] = expit(
            np.clip(logit(state.g_r[:, m]) + eps[1] * covariates.h_r[:, m], -clip, clip)
        )
    h = state.h.copy()
    for t in nodes:
        shift = eps[2] * covariates.h_l[:, t - 1] + eps[3] * covariates.z[:, t - 1]
        h[:, t - 1] = expit(np.clip(logit(state.h[:, t - 1]) + shift, -clip, clip))
    return _State(g_a, g_r, h), eps


# ------------------------------------------------------------------------ the curve


@dataclass(frozen=True)
class DriftCurve:
    """The targeted risk and its influence curve terms at one arm and horizon.

    Parameters
    ----------
    risk : float
        :math:`1 - \\tilde\\theta`.
    eif : FloatArray
        ``(n,)`` the efficient influence function of the risk at the targeted state.
    d_l : FloatArray
        ``(n,)`` :math:`D_L` of the survival, at the targeted state.
    d_r : FloatArray
        ``(n,)`` :math:`D_R` of the survival over the estimated censoring nodes.
    d_a : FloatArray
        ``(n,)`` :math:`D_A` of the survival, zero when :math:`g_A` is declared.
    curve : FloatArray
        ``(n,)`` the corrected influence curve of the risk,
        :math:`-(D - D_L - D_R - D_A)` with :math:`D` the survival EIF.
    """

    risk: float
    eif: FloatArray
    d_l: FloatArray
    d_r: FloatArray
    d_a: FloatArray
    curve: FloatArray


def corrected_curve(
    view: PaperView,
    state: _State,
    covariates: Covariates,
    tau: int,
    g_bounds: tuple[float, float],
) -> DriftCurve:
    """The risk and the corrected curve of Theorem 2 with the rule of 4.5a.

    Parameters
    ----------
    view : PaperView
        The arm.
    state : _State
        The targeted state.
    covariates : Covariates
        Eq. (9) at ``state``.
    tau : int
        The horizon.
    g_bounds : tuple of float
        The bounds of the cumulative :math:`g_A G`.

    Returns
    -------
    DriftCurve
        The risk, the EIF and the three scores.
    """
    survival = state.survival(tau)
    pi = bound(state.pi(tau), *g_bounds)
    arm = view.arm
    residual = np.where(view.at_risk[:, :tau], view.events[:, :tau] - state.h[:, :tau], 0.0)
    theta = float(np.mean(survival[:, tau]))
    weight = _ratio(survival[:, tau : tau + 1], survival[:, 1:]) / pi[:, 1:]
    survival_eif = -(arm[:, None] * weight * residual).sum(axis=1) + survival[:, tau] - theta
    d_l = -(arm[:, None] * covariates.h_l * residual).sum(axis=1)
    d_r = np.zeros(view.n)
    for m in range(tau):
        if view.held_censoring[m]:
            continue
        rows = view.eligible[:, m]
        d_r += np.where(
            rows, arm * covariates.h_r[:, m] * (view.censored[:, m] - state.g_r[:, m]), 0.0
        )
    d_a = np.zeros(view.n) if view.held_treatment else -covariates.h_a * (arm - state.g_a)
    corrected = survival_eif - d_l - d_r - d_a
    return DriftCurve(1.0 - theta, -survival_eif, d_l, d_r, d_a, -corrected)


# ------------------------------------------------------------------------- the loop


#: The printed names of :attr:`DriftRecord.scores`, in order.
SCORE_LABELS: Final = ("efficient score", "treatment drift", "censoring drift", "hazard drift")


@dataclass(frozen=True, eq=False)
class DriftRecord:
    """The drift correction of one arm at one horizon.

    Parameters
    ----------
    arm : str
        The regimen label.
    horizon : int
        The horizon, counted from one.
    guard : tuple of str
        ``("Q", "g")``, or ``()`` for hazard ``LTMLE``.
    held_factors : tuple of str
        The declared or held factors, never fluctuated: ``"treatment"`` and
        ``"censoring node k"``.
    curve_terms : tuple of str
        Which of ``"D_L"``, ``"D_R"`` and ``"D_A"`` the curve subtracts.
    censoring_terms : tuple of int
        The censoring nodes, counted from one in the container, that :math:`D_R` sums over.
    n_iter : int
        The iterations the loop ran.
    converged : bool
        Whether the stop rule was met before ``max_iter``.
    threshold : float
        The stop rule :math:`10^{-4} n^{-3/5}`.
    epsilon : FloatArray
        ``(n_iter, 4)`` :math:`(\\varepsilon_A, \\varepsilon_R, \\varepsilon_{L,1},
        \\varepsilon_{L,2})` per iteration.
    scores : tuple of float
        :math:`(P_n D, P_n D_A, P_n D_R, P_n D_L)` at the stop, of the survival.
    reductions : tuple of ReductionFit
        Each auxiliary regression, in fit order.
    denominator_hits : tuple of tuple
        ``(function, count)`` of the floored :math:`q`, :math:`B` and :math:`D` values.
    g_floor_hits : int
        The cumulative :math:`g_A G` values the ``g_bounds`` floor moved at the stop.
    n_folds : int
        The outer folds of the initial fits.
    """

    arm: str
    horizon: int
    guard: tuple[str, ...]
    held_factors: tuple[str, ...]
    curve_terms: tuple[str, ...]
    censoring_terms: tuple[int, ...]
    n_iter: int
    converged: bool
    threshold: float
    epsilon: FloatArray
    scores: tuple[float, float, float, float]
    reductions: tuple[ReductionFit, ...]
    denominator_hits: tuple[tuple[str, int], ...]
    g_floor_hits: int
    n_folds: int

    @property
    def degenerate(self) -> int:
        """The regressions that fell back to the subpopulation mean."""
        return sum(fit.degenerate for fit in self.reductions)

    def to_frame(self, data: LongitudinalData | None = None) -> Any:
        """One row per iteration: the four :math:`\\varepsilon`.

        Parameters
        ----------
        data : LongitudinalData or None
            When given, the rows come back in that data's frame type.

        Returns
        -------
        dict or frame
            The columns ``iteration``, ``eps_A``, ``eps_R``, ``eps_L1`` and ``eps_L2``.
        """
        eps = np.asarray(self.epsilon, dtype=float).reshape(-1, 4)
        payload: dict[str, Any] = {
            "iteration": list(range(1, eps.shape[0] + 1)),
            "eps_A": eps[:, 0].tolist(),
            "eps_R": eps[:, 1].tolist(),
            "eps_L1": eps[:, 2].tolist(),
            "eps_L2": eps[:, 3].tolist(),
        }
        if data is None:
            return payload
        return data.frame_like(payload)

    def summary(self, *, detail: Detail = "short") -> Summary:
        """A printable report of the drift correction.

        Parameters
        ----------
        detail : {"short", "full"}, default "short"
            Accepted for symmetry with the other records; both print the same.

        Returns
        -------
        Summary
            The loop, the held factors and the scores at the stop.
        """
        from .._summary import Cell, Summary, SummaryTable, check_detail

        check_detail(detail)
        rows = tuple(
            (Cell(label, label), Cell(float(value), f"{value:.2e}"))
            for label, value in zip(SCORE_LABELS, self.scores, strict=True)
        )
        table = SummaryTable(("score", "mean at the stop"), rows, kind="other")
        header = (
            ("Arm", self.arm),
            ("Horizon", str(self.horizon)),
            ("Iterations", f"{self.n_iter} ({'stopped' if self.converged else 'cap reached'})"),
            ("Held factors", ", ".join(self.held_factors) or "none"),
            ("Curve subtracts", ", ".join(self.curve_terms) or "nothing"),
            ("Regressions", f"{len(self.reductions)} ({self.degenerate} at the mean)"),
            ("Floored denominators", str(sum(count for _, count in self.denominator_hits))),
        )
        return Summary.build(title="Drift correction", header=header, tables=(table,))

    def _repr_html_(self) -> str:
        return self.summary().as_html()

    def __repr__(self) -> str:
        return (
            f"DriftRecord(arm={self.arm!r}, horizon={self.horizon}, n_iter={self.n_iter}, "
            f"converged={self.converged})"
        )


@dataclass(frozen=True)
class DriftFit:
    """The targeted state, the curve and the record of one arm at one horizon.

    Parameters
    ----------
    state : _State
        The targeted state.
    curve : DriftCurve
        The risk and the corrected curve.
    covariates : Covariates
        Eq. (9) at the targeted state.
    record : DriftRecord
        The record.
    """

    state: _State
    curve: DriftCurve
    covariates: Covariates
    record: DriftRecord


def stop_threshold(n: int) -> float:
    """The stop rule of the source, :math:`10^{-4} n^{-3/5}`.

    Parameters
    ----------
    n : int
        The rows.

    Returns
    -------
    float
        The threshold of :math:`\\max |\\varepsilon|`.
    """
    return 1e-4 * float(n) ** (-0.6)


def target_drift(
    view: PaperView,
    tau: int,
    *,
    label: str,
    g_bounds: tuple[float, float],
    alpha: float,
    max_iter: int,
    n_folds: int,
    drift: bool = True,
    warn: bool = True,
    regress: Regression | None = None,
) -> DriftFit:
    """Run Steps 1 to 5 of Section 5 for one arm at one horizon.

    Parameters
    ----------
    view : PaperView
        The arm.
    tau : int
        The horizon.
    label : str
        The regimen label, for the record and the warning.
    g_bounds : tuple of float
        The cumulative mechanism bounds.
    alpha : float
        The probability bound of every fluctuated probability.
    max_iter : int
        The iteration cap.
    n_folds : int
        The outer folds, for the record.
    drift : bool
        ``False`` forces :math:`H_A`, :math:`H_R`, :math:`H_L` to zero, so only :math:`Z`
        moves the hazards: a check device.
    warn : bool
        Whether a loop that reaches ``max_iter`` warns.
    regress : callable or None
        ``regress(x, y, rows)`` in place of the kernel smoother, a check device for
        population regressions on a finite law.

    Returns
    -------
    DriftFit
        The targeted state, the curve and the record.
    """
    state = _State(view.g_a.copy(), view.g_r.copy(), view.h.copy())
    reductions = _Reductions(view, state, tau, g_bounds, regress)
    coefficients = fit_coefficients(reductions, tau, g_bounds[0], drift=drift)
    threshold = stop_threshold(view.n)
    path: list[FloatArray] = []
    converged = False
    for _ in range(max_iter):
        covariates = clever_covariates(state, coefficients, tau, g_bounds)
        state, eps = fluctuate_once(view, state, covariates, tau, alpha)
        path.append(eps)
        if not np.all(np.isfinite(eps)):
            raise LongitudinalError(
                f"the drift correction of {label!r} at horizon {tau} met a logistic submodel "
                f"with no finite maximum likelihood at iteration {len(path)}, so no targeted "
                "state exists. Check the overlap of the arms and of censoring in "
                "result.diagnostics, or bind g_bounds further from zero"
            )
        if float(np.max(np.abs(eps))) < threshold:
            converged = True
            break
    covariates = clever_covariates(state, coefficients, tau, g_bounds)
    curve = corrected_curve(view, state, covariates, tau, g_bounds)
    if not converged and warn:
        _warn(
            f"the drift correction of {label!r} at horizon {tau} stopped after {len(path)} "
            f"iteration(s) with max |epsilon| = {float(np.max(np.abs(path[-1]))):.3g} above "
            f"{threshold:.3g}. Raise max_iter, or check result.drift for a floored denominator",
            ConvergenceWarning,
            stacklevel=3,
        )
    estimated = tuple(m + 1 for m in range(tau) if not view.held_censoring[m])
    held = (("treatment",) if view.held_treatment else ()) + tuple(
        f"censoring node {m + 1}" for m in range(tau) if view.held_censoring[m]
    )
    terms = ("D_L",) + (("D_R",) if estimated else ()) + (() if view.held_treatment else ("D_A",))
    survival_eif = -curve.eif
    record = DriftRecord(
        arm=label,
        horizon=tau,
        guard=("Q", "g"),
        held_factors=held,
        curve_terms=terms,
        censoring_terms=estimated,
        n_iter=len(path),
        converged=converged,
        threshold=threshold,
        epsilon=np.asarray(path, dtype=float).reshape(-1, 4),
        scores=(
            float(np.mean(survival_eif)),
            float(np.mean(curve.d_a)),
            float(np.mean(curve.d_r)),
            float(np.mean(curve.d_l)),
        ),
        reductions=tuple(reductions.records),
        denominator_hits=tuple(sorted(coefficients.hits.items())),
        g_floor_hits=covariates.floor_hits,
        n_folds=n_folds,
    )
    return DriftFit(state, curve, covariates, record)


# ------------------------------------------------------------------------- refusals

_ESTIMATOR_ALTERNATIVE = (
    "Fit LTMLE(survival_model='hazard') for the hazard TMLE without the drift correction."
)


def _gap(text: str, roadmap: str, alternative: str = _ESTIMATOR_ALTERNATIVE) -> Refusal:
    return Refusal(RefusalReason.IMPLEMENTATION_GAP, text, alternative=alternative, roadmap=roadmap)


def _invalid(text: str) -> Refusal:
    return Refusal(RefusalReason.INVALID_DECLARATION, text)


SEQUENTIAL_REFUSAL = _gap(
    "SurvivalDRTMLE fits one baseline treatment held over every node with baseline covariates "
    "only. Díaz (2019) has baseline covariates and one decision, and names time-dependent "
    "confounding of censoring as future work.",
    "F5",
    "Fit LTMLE, which supports a treatment at several nodes and later covariates.",
)
DECLARED_CENSORING_REFUSAL = _gap(
    "SurvivalDRTMLE with an estimated treatment model cannot read declared censoring "
    "probabilities. A declared retention column holds one value per unit, at the arm the unit "
    "received, while the treatment covariate H_A divides by the retention under each arm on "
    "every row; where retention depends on treatment the drift score is then biased.",
    "X62",
    "Declare the treatment probabilities too (treatment_probabilities=), or let the censoring "
    "model be estimated.",
)
CAUSES_REFUSAL = _gap(
    "SurvivalDRTMLE fits a time-to-event outcome with one event type, and this fit declares two "
    "or more causes. Díaz (2019) has one event type, and no source derives drift terms for "
    "cause-specific hazards.",
    "X62",
)
ARMS_REFUSAL = _gap(
    "SurvivalDRTMLE fits static arms of a binary baseline treatment. Díaz (2019) defines the "
    "treatment mechanism for a binary treatment and a static arm, so a rule, a known "
    "stochastic policy and a treatment with more than two levels are not built.",
    "X62",
)
POLICY_REFUSAL = _gap(
    "SurvivalDRTMLE does not fit a modified treatment policy or an odds tilt. Each changes the "
    "clever covariates of the drift correction, and no source derives them.",
    "F5",
)
DOSE_REFUSAL = _gap(
    "SurvivalDRTMLE needs a categorical baseline treatment. A continuous dose has no drift "
    "terms in the source.",
    "X62",
    "Fit LTMLE with a modified treatment policy for a dose.",
)
OUTCOME_REFUSAL = _gap(
    "SurvivalDRTMLE fits a time-to-event outcome, and this fit declares an end-of-study or "
    "repeated outcome. Díaz (2019) derives the drift terms for the event hazard.",
    "X62",
    "Fit LTMLE for an end-of-study or repeated outcome.",
)
ELIMINATE_REFUSAL = _gap(
    "SurvivalDRTMLE does not take eliminate=. An eliminated cause adds a coarsening factor that "
    "the drift terms of the source do not include.",
    "X62",
)
SEPARABLE_REFUSAL = _gap(
    "SurvivalDRTMLE does not fit a Separable regimen. A separable effect needs the factor-wise "
    "hazard targeting of LTMLE, and no source gives its drift terms.",
    "X62",
    "Fit the Separable regimen with LTMLE(survival_model='hazard', targeting='one_step').",
)
SURVIVAL_MODEL_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "SurvivalDRTMLE fluctuates the event hazard, as Díaz (2019) does, so survival_model is "
    "'hazard' and there is no pseudo-outcome learner.",
    alternative="Fit LTMLE(survival_model='iterated_mean') for the iterated-mean recursion.",
)
ONE_STEP_REFUSAL = _gap(
    "SurvivalDRTMLE runs the per-horizon iteration of three logistic submodels of Díaz (2019). "
    "targeting='one_step' would need a universal submodel for the drift terms, which no source "
    "gives.",
    "X62",
    "Keep targeting='iterative', or fit LTMLE(targeting='one_step') for a monotone curve.",
)
Q_BOUNDS_REFUSAL = _invalid(
    "SurvivalDRTMLE does not take q_bounds=. The outcome is an event indicator, and each "
    "hazard is bounded by alpha."
)
DOSE_SETTINGS_REFUSAL = _invalid(
    "SurvivalDRTMLE does not take density_bins= or ratio='classifier'. Both belong to a "
    "continuous node or a policy ratio, and the fit reads static arms of a binary treatment."
)
COMPETING_LEARNER_REFUSAL = _gap(
    "SurvivalDRTMLE does not take competing_learner=. It fits one event type, and an "
    "eliminated cause adds a factor the drift terms do not include.",
    "X62",
)
BAND_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "SurvivalDRTMLE builds no simultaneous band. Each horizon is targeted separately, and a "
    "band needs the joint curve of separately targeted horizons, which no source treats.",
    alternative="Leave simultaneous=False; each horizon keeps its own interval.",
    roadmap="X62",
)
ISOTONIC_REFUSAL = _gap(
    "SurvivalDRTMLE does not take isotonic=True. Each horizon is a separate fit, and projecting "
    "them onto a monotone curve has no source.",
    "X62",
)
PROJECTION_REFUSAL = _gap(
    "SurvivalDRTMLE does not take msm=, node_bounds= or ratio_trim=. None of them is in the "
    "source.",
    "X62",
)
RULES_REFUSAL = _invalid(
    "SurvivalDRTMLE stops by the rule of Díaz (2019), max |epsilon| below 1e-4 n^(-3/5), so "
    "stopping=, step_bound=, iterate=, step_size= and tol= keep their defaults."
)
GUARD_REFUSAL = _gap(
    "SurvivalDRTMLE fluctuates the treatment, censoring and event models together, or none of "
    "them (guard=()). One block alone is not described in the source.",
    "X62",
    "Pass guard=('Q', 'g') for the drift correction, or guard=() for hazard LTMLE.",
)
POINT_SETTINGS_REFUSAL = _invalid(
    "SurvivalDRTMLE does not take the reduced-regression settings of point DRTMLE "
    "(reduction=, reduced_outcome_learner=, reduced_treatment_learner=, "
    "reduced_mechanism_residual_learner=, reduced_crossfit=, update_order=, evaluation=). "
    "Its regressions use the fixed kernel smoother of the drift correction."
)
RANDOMIZED_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "SurvivalDRTMLE reads a known treatment mechanism from the design, not from randomized=.",
    alternative="Declare treatment_probabilities= on the TimeToEvent design.",
)
WEIGHTS_REFUSAL = _gap(
    "SurvivalDRTMLE does not take observation weights. Weighted reductions and a weighted "
    "stopping rule are not in the source and are not measured.",
    "X62",
)
BOOTSTRAP_REFUSAL = _gap(
    "SurvivalDRTMLE draws no bootstrap and no repeated split. Each draw would rerun every "
    "reduction, and neither is measured.",
    "X62",
    "Fit with repeats=1 and n_bootstrap=0; the interval is the influence-curve one.",
)
RMST_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "A SurvivalDRTMLE result has no restricted mean survival time. It needs every node up to "
    "the horizon targeted with its own drift equations, and only the requested horizons are.",
    alternative="Fit LTMLE(survival_model='hazard', targeting='one_step') for the RMST.",
    roadmap="X62",
)
TRUNCATION_REFUSAL = _gap(
    "A SurvivalDRTMLE result has no truncation curve. Each bound would rerun the reductions "
    "and the drift loop, and no study measures that replay.",
    "X62",
    "Fit SurvivalDRTMLE at each bound for a curve.",
)
REPLAY_REFUSAL = _gap(
    "This replay reruns the reductions and the drift loop of a SurvivalDRTMLE result, and no "
    "study measures it.",
    "X62",
    NO_ALTERNATIVE,
)

#: The point DRTMLE settings this engine refuses, with their defaults.
_POINT_SETTINGS: Final = {
    "reduction": "univariate",
    "reduced_outcome_learner": None,
    "reduced_treatment_learner": None,
    "reduced_mechanism_residual_learner": None,
    "reduced_crossfit": "foldwise",
    "update_order": "drtmle",
    "evaluation": None,
}


def drift_design_refusal(data: LongitudinalData, regimens: Any) -> Refusal | None:
    """The refusal a prepared panel and its regimens meet, or ``None`` when it fits them.

    One function for the estimator and for ``CausalStudy.available_methods()``.

    Parameters
    ----------
    data : LongitudinalData
        The prepared panel, with any eliminated cause applied.
    regimens : Any
        The ``regimens=`` argument, or the resolved regimens.

    Returns
    -------
    Refusal or None
        The first refusal of the design, in the order of section 1 of the design.
    """
    _, separable = split_separable(regimens)
    if separable:
        return SEPARABLE_REFUSAL
    if not data.treatment_held and data.n_decisions > 1:
        return SEQUENTIAL_REFUSAL
    if any(names for names in data.time_varying_names):
        return SEQUENTIAL_REFUSAL
    if data.node_kind(1) != "categorical":
        return DOSE_REFUSAL
    if not data.is_survival:
        return OUTCOME_REFUSAL
    if data.eliminated:
        return ELIMINATE_REFUSAL
    if len(data.cause_labels) > 1:
        return CAUSES_REFUSAL
    if len(data.treatment_levels[0]) != 2:
        return ARMS_REFUSAL
    resolved = resolve_regimens(
        regimens,
        data.n_decisions,
        held_nodes=data.n_times if data.treatment_held else None,
    )
    for regimen in resolved:
        if isinstance(regimen, DynamicRegimen):
            if regimen.has_mtp or regimen.has_tilt:
                return POLICY_REFUSAL
            return ARMS_REFUSAL
    if data.is_weighted:
        return WEIGHTS_REFUSAL
    known = data.known_mechanisms
    if (
        known is not None
        and all(values is None for values in known.treatment)
        and any(values is not None for values in known.censoring)
    ):
        return DECLARED_CENSORING_REFUSAL
    return None


# ------------------------------------------------------------------------ the estimator


def _hazard_record(fit: DriftFit, label: str, initial_risk: float) -> HazardFit:
    """The targeting record a :class:`RegimenFit` carries, read by the shared diagnostics."""
    record = fit.record
    return HazardFit(
        algorithm="per_horizon",
        targets=((label, None, record.horizon),),
        converged=record.converged,
        n_iter=record.n_iter,
        score_initial=np.array([np.nan]),
        score=np.array([record.scores[0]]),
        score_scale=np.ones(1),
        trace=tuple(float(np.max(np.abs(eps))) for eps in record.epsilon),
        initial_levels=np.array([initial_risk]),
        failure=None if record.converged else "max_iter_reached",
        tol=record.threshold,
    )


def run_drift(
    data: LongitudinalData,
    plans: Sequence[Plan],
    mechanism: Mechanism,
    initial: InitialHazards,
    *,
    horizons: Sequence[int],
    g_bounds: tuple[float, float],
    alpha: float,
    max_iter: int,
    n_folds: int,
) -> tuple[dict[str, RegimenFit], dict[tuple[str, int], DriftRecord]]:
    """Run the drift correction for every static arm at every horizon.

    Parameters
    ----------
    data : LongitudinalData
        A held survival panel with one cause and a binary treatment.
    plans : sequence of Plan
        The static plans.
    mechanism : Mechanism
        The fitted mechanism, evaluated at every level.
    initial : InitialHazards
        The initial hazards.
    horizons : sequence of int
        The reported horizons.
    g_bounds : tuple of float
        The cumulative mechanism bounds.
    alpha : float
        The probability bound.
    max_iter : int
        The iteration cap of each loop.
    n_folds : int
        The outer folds, for the record.

    Returns
    -------
    tuple
        The fits keyed by ``_fit_key`` and the records keyed by ``(label, horizon)``.
    """
    rows = np.arange(data.n)
    observed = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
    pairs = [
        mechanism.cumulative_with_unbounded(data, _level_plan(data, code), g_bounds)
        for code in range(len(data.treatment_levels[0]))
    ]
    unbounded = np.stack([pair[0] for pair in pairs], axis=2)[rows, :, observed]
    bounded = np.stack([pair[1] for pair in pairs], axis=2)[rows, :, observed]
    fits: dict[str, RegimenFit] = {}
    records: dict[tuple[str, int], DriftRecord] = {}
    for plan in plans:
        code = int(np.asarray(plan.values)[0, 0])
        view = paper_view(data, mechanism, initial, code, alpha=alpha, g_bounds=g_bounds)
        initial_survival = np.cumprod(1.0 - view.h, axis=1)
        for t in horizons:
            fit = target_drift(
                view,
                int(t),
                label=plan.label,
                g_bounds=g_bounds,
                alpha=alpha,
                max_iter=max_iter,
                n_folds=n_folds,
            )
            records[(plan.label, int(t))] = fit.record
            clever = np.where(view.at_risk & (view.arm > 0.0)[:, None], 1.0 / bounded, 0.0)
            clever[:, int(t) :] = 0.0
            fits[_fit_key(plan.label, None, int(t), True)] = RegimenFit(
                regimen=plan.regimen,
                psi_scaled=fit.curve.risk,
                influence_curve_scaled=np.asarray(fit.curve.curve, dtype=float),
                horizon=int(t),
                cause=None,
                steps=(),
                cumulative_unbounded=unbounded,
                cumulative=bounded,
                assignment=np.asarray(plan.values),
                obs_weights=np.asarray(data.weights, dtype=float),
                policy=plan.policy,
                hazard=_hazard_record(
                    fit, plan.label, 1.0 - float(np.mean(initial_survival[:, int(t) - 1]))
                ),
                hazard_clever=clever,
                hazard_constant_nodes=tuple(k + 1 for k in range(int(t)) if view.held_hazard[k]),
            )
    return fits, records


class SurvivalDRTMLE(LTMLE):  # numpydoc ignore=PR01,PR02
    r"""Drift-corrected TMLE of a treatment-specific risk curve (Díaz 2019).

    For each horizon and each static arm of a binary baseline treatment, the fit starts from
    the hazard ``LTMLE`` nuisances (out of fold by default), fits univariate kernel
    regressions of residuals on scalar indices of those fits, and fluctuates the treatment
    mechanism, the censoring hazard and the event hazard together until every fluctuation
    parameter is below :math:`10^{-4} n^{-3/5}`.  The interval is the Wald interval from the
    corrected influence curve of Theorem 2 of the source.  The module docstring of
    :mod:`cleverly.longitudinal.drift` states the algorithm and its departures.

    Parameters
    ----------
    regimens : mapping or sequence
        Static arms of the binary baseline treatment, such as ``{"a1": 1, "a0": 0}``.
    reference : str or None, default=None
        The regimen every contrast is taken against; the first declared by default.
    horizons : sequence of int or None, default=None
        The nodes the curve is reported at.  ``None`` uses the nodes at which both arms have
        an event.  Each horizon runs its own correction, at a cost that grows with the
        square of the horizon, so naming the horizons you need saves time.
    outcome_learner : estimator or None, default=None
        The learner of the initial event hazard.  ``None`` uses the default Super Learner.
    treatment_learner : estimator or None, default=None
        The learner of the initial treatment mechanism.
    censoring_learner : estimator or None, default=None
        The learner of the initial censoring hazard.
    n_folds : int, default=10
        The outer split of the initial fits.  ``1`` fits them in sample (Theorem 2 of the
        source, which needs a Donsker condition no fit can check).
    learner_folds : int, default=5
        The internal folds of a Super Learner.
    g_bounds : float or tuple of float, default=(0.01, 1.0)
        The bound of each cumulative mechanism product; its lower end also floors the
        denominators of the drift covariates.
    alpha : float, default=0.9995
        The probability bound of every fluctuated probability.
    alpha_sig : float, default=0.05
        The significance level of the intervals.
    guard : tuple of str, default=("Q", "g")
        ``("Q", "g")`` runs the drift correction.  ``()`` runs hazard ``LTMLE``.
    max_iter : int, default=200
        The iteration cap of each drift loop.  The source states none.
    hazard_pooling : {"node", "pooled"}, default="node"
        How the initial hazard pools the nodes.
    random_state : int or None, default=None
        Seeds the outer split and the learners.
    run_id : str or None, default=None
        A label for the run record.
    n_jobs : int, default=1
        Workers of the hazard learner.
    n_multiplier : int, default=10_000
        The multiplier draws of a band.  The fit builds no band (``simultaneous=True``
        raises), so this and the next two keywords change nothing.  They take the package's
        default band and any value, as on the other engines.
    multiplier_kind : {"mammen", "rademacher", "normal"}, default="mammen"
        The multiplier distribution of a band.
    band_scale : {"logit", "identity"}, default="logit"
        The scale of a band.

    See Also
    --------
    cleverly.longitudinal.LTMLE : Hazard targeting without the drift correction.
    cleverly.estimators.DRTMLE : The drift-corrected TMLE of a point-treatment mean.

    Notes
    -----
    The other keywords of :class:`~cleverly.longitudinal.LTMLE` and of point
    :class:`~cleverly.estimators.DRTMLE` are accepted only at their defaults, and every other
    value raises before any learner with the reason the design records.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LogisticRegression
    >>> from cleverly.longitudinal import LongitudinalData, SurvivalDRTMLE
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> w1 = rng.normal(size=n)
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w1)))
    >>> event = rng.binomial(1, 1 / (1 + np.exp(0.5 - 0.6 * w1 + 0.5 * a)))
    >>> time = np.where(event == 1, rng.integers(1, 3, size=n), 2).astype(float)
    >>> frame = pd.DataFrame({"time": time, "event": event, "A": a, "W1": w1})
    >>> data = LongitudinalData.from_time_to_event(
    ...     frame, time="time", event="event", treatment="A", baseline=["W1"]
    ... )
    >>> result = SurvivalDRTMLE(
    ...     {"a1": 1, "a0": 0},
    ...     reference="a0",
    ...     outcome_learner=LogisticRegression(),
    ...     treatment_learner=LogisticRegression(),
    ...     n_folds=2,
    ...     random_state=0,
    ... ).fit(data)
    >>> sorted(result.drift)
    [('a0', 1), ('a0', 2), ('a1', 1), ('a1', 2)]
    >>> result.inference_status
    'influence_curve'
    """

    #: The support-record engine, read in place of :attr:`_estimator`, which keeps ``"tmle"``
    #: for the status and the hazard machinery.
    _support_engine: ClassVar[tuple[str, ...]] = ("drtmle", "univariate")

    def __init__(
        self,
        regimens: Any,
        *,
        reference: str | None = None,
        horizons: Sequence[int] | None = None,
        outcome_learner: Learner | None = None,
        treatment_learner: Learner | None = None,
        censoring_learner: Learner | None = None,
        n_folds: int = 10,
        learner_folds: int = 5,
        g_bounds: Any = DEFAULT_LTMLE_G_BOUNDS,
        alpha: float = 0.9995,
        alpha_sig: float = 0.05,
        guard: Sequence[str] = ("Q", "g"),
        max_iter: int = 200,
        hazard_pooling: str = "node",
        random_state: int | None = None,
        run_id: str | None = None,
        n_jobs: int = 1,
        survival_model: str = "hazard",
        targeting: str = "iterative",
        simultaneous: bool = False,
        n_multiplier: int = DEFAULT_N_REPLICATES,
        multiplier_kind: str = DEFAULT_KIND,
        band_scale: str = DEFAULT_BAND_SCALE,
        repeats: int = 1,
        n_bootstrap: int = 0,
        bootstrap_resampling: Any = "auto",
        bootstrap_interval: Any = "percentile",
        stopping: Any = "relative",
        step_bound: float | None = None,
        step_search: bool = True,
        iterate: Any = "last",
        step_size: float = 1e-3,
        tol: float = 1e-10,
        q_bounds: tuple[float, float] | None = None,
        msm: Any = None,
        node_bounds: Mapping[Any, Any] | None = None,
        ratio_trim: float | None = None,
        isotonic: bool = False,
        pseudo_learner: Learner | None = None,
        competing_learner: Learner | None = None,
        density_bins: int | None = None,
        ratio: str = "density",
        reduction: str = "univariate",
        reduced_outcome_learner: Learner | None = None,
        reduced_treatment_learner: Learner | None = None,
        reduced_mechanism_residual_learner: Learner | None = None,
        reduced_crossfit: str = "foldwise",
        update_order: str = "drtmle",
        evaluation: Any = None,
        randomized: bool = False,
        **refused: Any,
    ) -> None:
        self.guard = tuple(guard)
        self.reduction = reduction
        self.reduced_outcome_learner = reduced_outcome_learner
        self.reduced_treatment_learner = reduced_treatment_learner
        self.reduced_mechanism_residual_learner = reduced_mechanism_residual_learner
        self.reduced_crossfit = reduced_crossfit
        self.update_order = update_order
        self.evaluation = evaluation
        self.randomized = randomized
        self._pending: dict[tuple[str, int], DriftRecord] | None = None
        super().__init__(
            regimens,
            reference=reference,
            horizons=horizons,
            msm=msm,
            outcome_learner=outcome_learner,
            pseudo_learner=pseudo_learner,
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            competing_learner=competing_learner,
            n_folds=n_folds,
            learner_folds=learner_folds,
            g_bounds=g_bounds,
            q_bounds=q_bounds,
            alpha=alpha,
            alpha_sig=alpha_sig,
            simultaneous=simultaneous,
            n_multiplier=n_multiplier,
            multiplier_kind=multiplier_kind,
            band_scale=band_scale,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            run_id=run_id,
            n_jobs=n_jobs,
            n_bootstrap=n_bootstrap,
            bootstrap_resampling=bootstrap_resampling,
            bootstrap_interval=bootstrap_interval,
            density_bins=density_bins,
            ratio=ratio,
            repeats=repeats,
            stopping=stopping,
            step_bound=step_bound,
            step_search=step_search,
            iterate=iterate,
            survival_model=survival_model,
            targeting=targeting,
            step_size=step_size,
            isotonic=isotonic,
            ratio_trim=ratio_trim,
            hazard_pooling=hazard_pooling,
            node_bounds=node_bounds,
            **refused,
        )

    # -- refusals

    def _validate_settings(self) -> None:
        """Refuse every setting section 1 of the design refuses, then run the shared checks."""
        refusal = self._settings_refusal()
        if refusal is not None:
            raise CapabilityError.from_refusal(refusal)
        if isinstance(self.max_iter, bool) or not (
            isinstance(self.max_iter, (int, np.integer)) and self.max_iter >= 1
        ):
            raise ValueError(f"max_iter must be an integer of at least 1; got {self.max_iter!r}")
        super()._validate_settings()

    def _settings_refusal(self) -> Refusal | None:
        """The first refused setting, in the order of the design's table."""
        if self.survival_model != "hazard" or self.pseudo_learner is not None:
            return SURVIVAL_MODEL_REFUSAL
        if self.competing_learner is not None:
            return COMPETING_LEARNER_REFUSAL
        if self.density_bins is not None or self.ratio != "density":
            return DOSE_SETTINGS_REFUSAL
        if self.targeting != "iterative":
            return ONE_STEP_REFUSAL
        if self.q_bounds is not None:
            return Q_BOUNDS_REFUSAL
        if self.simultaneous:
            return BAND_REFUSAL
        if self.isotonic:
            return ISOTONIC_REFUSAL
        if self.node_bounds is not None or self.ratio_trim is not None or self.msm is not None:
            return PROJECTION_REFUSAL
        if (
            self.stopping != "relative"
            or self.step_bound is not None
            or self.step_search is not True
            or self.iterate != "last"
            or self.step_size != 1e-3
            or self.tol != 1e-10
        ):
            return RULES_REFUSAL
        if self.guard not in _GUARDS:
            return GUARD_REFUSAL
        if any(
            getattr(self, name) is not default and getattr(self, name) != default
            for name, default in _POINT_SETTINGS.items()
        ):
            return POINT_SETTINGS_REFUSAL
        if self.randomized:
            return RANDOMIZED_REFUSAL
        if (
            self.repeats != 1
            or self.n_bootstrap
            or self.bootstrap_resampling != "auto"
            or self.bootstrap_interval != "percentile"
        ):
            return BOOTSTRAP_REFUSAL
        return None

    def _refuse_prepared(self, data: LongitudinalData) -> None:
        """Refuse a design section 1 of the design refuses, and a node every row leaves at.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.
        """
        refusal = drift_design_refusal(data, self.regimens)
        if refusal is not None:
            raise CapabilityError.from_refusal(refusal)
        horizons = self._horizons(data)
        nodes = all_censored_nodes(data, max(horizons))
        if nodes:
            grid = data.time_grid
            time = float(grid[nodes[0] - 1]) if grid is not None else float(nodes[0])
            raise CapabilityError.from_refusal(all_censored_refusal(nodes[0], time))

    def _horizons(self, data: LongitudinalData) -> tuple[int, ...]:
        """The requested horizons, or by default the nodes at which both arms have an event.

        Parameters
        ----------
        data : LongitudinalData
            The prepared panel.

        Returns
        -------
        tuple of int
            The horizons, counted from one.
        """
        if self.horizons is not None or not data.is_survival or data.event is None:
            return super()._horizons(data)
        observed = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
        _, events = _event_rows(data)
        both = np.ones(data.n_times, dtype=bool)
        for code in range(len(data.treatment_levels[0])):
            both &= events[observed == code].any(axis=0)
        nodes = tuple(int(k) + 1 for k in np.flatnonzero(both))
        return nodes or (data.n_times,)

    # -- the correction

    def _hazard_block(
        self,
        prepared: LongitudinalData,
        plans: tuple[Plan, ...],
        levels: tuple[Plan, ...],
        folds: Folds,
        *,
        learners: tuple[Learner, Learner, Learner, Learner, Learner | None],
        horizons: tuple[int, ...],
        bounds: tuple[float, float],
        targeting: dict[str, Any],
        replay_recipe: _LongitudinalReplayRecipe,
        node_bounds: dict[str | None, dict[int, NodeBound]] | None,
        separable: tuple[Any, ...],
    ) -> _HazardBlock:
        """Fit the inherited nuisances, then run the drift correction per arm and horizon.

        Under ``guard=()`` the block is hazard ``LTMLE``'s own.

        Parameters
        ----------
        prepared : LongitudinalData
            The prepared panel.
        plans : tuple of Plan
            The static plans.
        levels : tuple of Plan
            One static plan per treatment level.
        folds : Folds
            The realized outer split.
        learners : tuple
            The resolved learners.
        horizons : tuple of int
            The reported horizons.
        bounds : tuple of float
            The cumulative mechanism bounds.
        targeting : dict
            The fluctuation settings.
        replay_recipe : _LongitudinalReplayRecipe
            The draw's replay recipe.
        node_bounds : dict or None
            Always ``None``: the constructor refuses node bounds.
        separable : tuple
            Always empty: the design refusal refuses a separable regimen.

        Returns
        -------
        _HazardBlock
            The mechanism, the fits, the initial hazards and the recipe.
        """
        if not self.guard:
            self._pending = None
            return super()._hazard_block(
                prepared,
                plans,
                levels,
                folds,
                learners=learners,
                horizons=horizons,
                bounds=bounds,
                targeting=targeting,
                replay_recipe=replay_recipe,
                node_bounds=node_bounds,
                separable=separable,
            )
        outcome_learner, pseudo_learner, treatment_learner, censoring_learner, competing = learners
        mechanism = self._fit_mechanism(
            prepared,
            (*plans, *levels),
            treatment_learner=treatment_learner,
            censoring_learner=censoring_learner,
            competing_learner=competing,
            folds=folds,
        )
        parametric = self._parametric(outcome_learner, pseudo_learner, mechanism, horizons, plans)
        initial = fit_hazards(
            prepared,
            levels,
            learner=outcome_learner,
            folds=folds,
            n_jobs=self.n_jobs,
            hazard_pooling=self.hazard_pooling,
        )
        replay_recipe = replace(
            replay_recipe,
            parametric=parametric,
            omissions=(),
            hazard=_HazardReplay(
                _frozen_plans(levels),
                "per_horizon",
                float(self.step_size),
                targeting["alpha"],
                targeting["max_iter"],
                targeting["tol"],
                (),
            ),
        )
        fits, records = run_drift(
            prepared,
            plans,
            mechanism,
            initial,
            horizons=horizons,
            g_bounds=bounds,
            alpha=targeting["alpha"],
            max_iter=int(self.max_iter),
            n_folds=folds.n_folds,
        )
        self._pending = records
        return _HazardBlock(mechanism, fits, initial, replay_recipe, plans, parametric)

    def _finish(
        self,
        result: LongitudinalResult,
        reported: _Reported,
        fits: Mapping[str, RegimenFit],
    ) -> LongitudinalResult:
        """Attach the drift records.

        Parameters
        ----------
        result : LongitudinalResult
            The result.
        reported : _Reported
            The names, estimates and contributing fits.
        fits : mapping of str to RegimenFit
            The fits.

        Returns
        -------
        LongitudinalResult
            The result with :attr:`~cleverly.longitudinal.LongitudinalResult.drift` set.
        """
        records, self._pending = self._pending, None
        return replace(result, drift=records)

    def _refit_template(self) -> SurvivalDRTMLE:
        """The configuration a result keeps, without the pending records of this fit.

        Returns
        -------
        SurvivalDRTMLE
            The template.
        """
        template = super()._refit_template()
        assert isinstance(template, SurvivalDRTMLE)
        template._pending = None
        return template


def _event_rows(data: LongitudinalData) -> tuple[BoolArray, BoolArray]:
    """``(n, T)`` the rows at risk and the rows with an event at each node."""
    from .hazard import event_indicators

    at_risk, events = event_indicators(data)
    return at_risk, events[:, :, 0] > 0.0

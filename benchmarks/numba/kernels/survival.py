r"""Discrete-time survival and competing risks: the same recursion, run many more times.

A survival fit is the longitudinal recursion with two changes, and the second is the one
that costs.

*The pseudo-outcome composes.*  ``Z_t = 1{cause j at t} + (1 - 1{any event at t}) Q*_{t+1}``
-- a cause-specific numerator against an all-cause survival factor.  Per node that is a few
more multiplies and nothing to benchmark.

*Every horizon is its own backward pass.*  The cumulative risk at ``k`` starts the
recursion at ``k`` rather than at ``T``, so reporting the whole curve is ``T(T+1)/2``
node-regressions per regimen per cause where the end-of-study fit is ``T``.  At ``T = 100``
that is 5,050 nodes against 100, and every one of them rebuilds the masks the last one
already had.  This is the kernel where the ``O(T^2)`` mask construction of
:mod:`.longitudinal` becomes ``O(T^3)``, and where the risk set is *shrinking* -- so a
representation that walks all ``n`` rows at every node does work that the data says is not
there.

Four implementations, answering four different questions:

``numpy``
    the shipped shape: dense ``(n,)`` arrays at every node of every horizon, masks rebuilt.
``numpy_shared_masks``
    the same numbers with the masks and the cumulative product built once and reused
    across the horizons.  They do not depend on the horizon at all, so this is not an
    approximation -- it is the same arithmetic with the redundancy removed, and it is the
    "improve the numpy" arm.
``numba``
    the whole per-horizon recursion fused and compiled.
``numba_parallel``
    ``prange`` over ``(regimen, cause, horizon)`` cells, which are mutually independent
    once the node predictions exist.  This is the axis with by far the most work on it in
    any real survival fit, and the reason a survival benchmark is not just a longitudinal
    one with a bigger ``T``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import Regime, make_survival
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_mapping
from . import KernelSpec, register

__all__ = [
    "build",
    "numba_incidence",
    "numba_incidence_parallel",
    "numpy_incidence",
    "numpy_incidence_shared",
]

_ALPHA = 0.9995
_MAX_NEWTON = 20
_NEWTON_TOL = 1e-10


def build(
    n: int = 50_000,
    n_times: int = 20,
    n_regimens: int = 2,
    n_causes: int = 1,
    n_horizons: int = 0,
    incidence: float = 0.06,
    regime: Regime = "moderate",
    g_lower: float = 0.01,
    seed: int = 20260803,
) -> dict[str, Any]:
    """``n_horizons=0`` means every horizon, which is what a risk curve reports."""
    fixture = make_survival(
        n, n_times=n_times, n_regimens=n_regimens, n_causes=n_causes,
        incidence=incidence, regime=regime, seed=seed,
    )
    horizons = (
        tuple(range(1, n_times + 1))
        if n_horizons <= 0
        else tuple(np.unique(np.linspace(1, n_times, n_horizons).astype(int)).tolist())
    )
    return {"fixture": fixture, "horizons": horizons, "g_bounds": (g_lower, 1.0 - g_lower)}


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


# --------------------------------------------------------------------------- numpy


def _masks_and_cumulative(fixture, assignment, bounds):
    """``following[t]``, the all-cause event-free mask and the cumulative mechanism.

    None of the three depends on the horizon, which is exactly why building them inside
    the horizon loop -- as the shipped shape effectively does, by rebuilding the masks at
    every node of every pass -- is redundant work rather than necessary work.
    """
    base = fixture.base
    lower, upper = bounds
    n, times = base.n, base.n_times
    followed = np.ones(n, dtype=bool)
    running = np.ones(n)
    following = np.empty((times, n), dtype=bool)
    cumulative = np.empty((times, n))
    for t in range(times):
        followed = (
            followed
            & (base.treated[:, t] == assignment[:, t])
            & (base.uncensored[:, t] == 1.0)
        )
        g1 = np.clip(base.treatment_probability[:, t], lower, upper)
        running = running * np.where(assignment[:, t] == 1.0, g1, 1.0 - g1)
        running = running * np.clip(base.censoring_probability[:, t], lower, upper)
        following[t] = followed
        cumulative[t] = running
    return following, cumulative


def _one_horizon(fixture, following, cumulative, event, cause_event, horizon, weights):
    """One backward pass from ``horizon`` down to node 1, returning psi and the curve."""
    base = fixture.base
    n = base.n
    carried = np.zeros(n)
    clever_terms = []
    targeted_first = None
    for t in range(horizon - 1, -1, -1):
        # `following(t)` reads the event at t-1: a unit that has the event *at* t is in
        # node t's regression, because it is the observation that the event happened.
        # `following(t)` reads the event at `t - 1` and `at_risk(t)` reads it at `t`: a
        # unit that has the event at `t` **is** in node `t`'s regression -- it is the
        # observation that the event happened -- and is not in node `t + 1`'s.
        event_free_before = np.ones(n, dtype=bool) if t == 0 else (event[:, t - 1] == 0.0)
        follow = following[t] & event_free_before
        at_risk = (
            np.ones(n, dtype=bool)
            if t == 0
            else (following[t - 1] & (event[:, t] == 0.0))
        )
        pseudo = cause_event[:, t] + (1.0 - event[:, t]) * carried
        initial = np.where(at_risk, base.initial[t], 0.0)
        denominator = np.where(at_risk, cumulative[t], 1.0)
        counterfactual = np.where(at_risk, 1.0 / denominator, 0.0)
        clever = np.where(follow, counterfactual, 0.0)

        shrunk = np.clip(initial, 1.0 - _ALPHA, _ALPHA)
        offset = np.log(shrunk / (1.0 - shrunk))
        epsilon = 0.0
        total = weights[follow].sum()
        if total > 0.0:
            for _ in range(_MAX_NEWTON):
                p = _expit(offset + clever * epsilon)
                gradient = float((weights * clever * (pseudo - p))[follow].sum())
                if abs(gradient) / total <= _NEWTON_TOL:
                    break
                hessian = float((weights * clever * clever * p * (1.0 - p))[follow].sum())
                if hessian <= 0.0 or not np.isfinite(hessian):
                    break
                epsilon += gradient / hessian
        targeted = np.clip(
            _expit(offset + counterfactual * epsilon), 1.0 - _ALPHA, _ALPHA
        )
        clever_terms.append((clever, pseudo, targeted))
        carried = np.where(at_risk, targeted, 0.0)
        targeted_first = targeted

    psi = float(np.average(targeted_first, weights=weights))
    influence = targeted_first - psi
    for clever, pseudo, targeted in clever_terms:
        influence = influence + clever * (pseudo - targeted)
    return psi, weights * influence


def numpy_incidence(inputs: dict[str, Any], *, share_masks: bool = False) -> dict[str, Any]:
    """The cumulative incidence at every requested horizon, per regimen and cause."""
    fixture = inputs["fixture"]
    base = fixture.base
    weights = base.weights
    out: dict[str, Any] = {}
    shared: dict[int, Any] = {}
    for r in range(base.n_regimens):
        for j, cause in enumerate(fixture.causes):
            for horizon in inputs["horizons"]:
                if share_masks:
                    if r not in shared:
                        shared[r] = _masks_and_cumulative(
                            fixture, base.assignment[r], inputs["g_bounds"]
                        )
                    following, cumulative = shared[r]
                else:
                    following, cumulative = _masks_and_cumulative(
                        fixture, base.assignment[r], inputs["g_bounds"]
                    )
                psi, curve = _one_horizon(
                    fixture, following, cumulative, fixture.event,
                    fixture.cause_event[j], horizon, weights,
                )
                key = f"{base.labels[r]}|{cause}|{horizon}"
                out[f"psi_{key}"] = np.array([psi])
                out[f"ic_{key}"] = curve
    return out


def numpy_incidence_shared(inputs: dict[str, Any]) -> dict[str, Any]:
    """The same numbers with the horizon-independent masks built once per regimen."""
    return numpy_incidence(inputs, share_masks=True)


# --------------------------------------------------------------------------- numba


@njit()
def _build_masks(
    treatment_probability, censoring_probability, treated, uncensored, assignment,
    lower, upper, following, cumulative,
):
    n, times = treated.shape
    for i in range(n):
        followed = True
        running = 1.0
        for t in range(times):
            if treated[i, t] != assignment[i, t] or uncensored[i, t] != 1.0:
                followed = False
            g = min(max(treatment_probability[i, t], lower), upper)
            if assignment[i, t] != 1.0:
                g = 1.0 - g
            running *= g * min(max(censoring_probability[i, t], lower), upper)
            following[t, i] = followed
            cumulative[t, i] = running


@njit()
def _horizon_pass(
    following, cumulative, event, cause_event, initial, weights, horizon,
    targeted, clever, pseudo,
):
    """One horizon's backward pass, fused.  Returns the plug-in; fills the work arrays."""
    n = weights.shape[0]
    for i in range(n):
        pseudo[horizon - 1, i] = 0.0
    for t in range(horizon - 1, -1, -1):
        total = 0.0
        for i in range(n):
            at_risk = True if t == 0 else (following[t - 1, i] and event[i, t] == 0.0)
            event_free = True if t == 0 else event[i, t - 1] == 0.0
            follow = following[t, i] and event_free
            covariate = 1.0 / cumulative[t, i] if at_risk else 0.0
            clever[t, i] = covariate if follow else 0.0
            carried = 0.0 if t == horizon - 1 else targeted[t + 1, i]
            if t < horizon - 1:
                carried = targeted[t + 1, i] if (following[t, i] and event[i, t + 1] == 0.0) else 0.0
            pseudo[t, i] = cause_event[i, t] + (1.0 - event[i, t]) * carried
            if follow:
                total += weights[i]
        epsilon = 0.0
        if total > 0.0:
            for _ in range(_MAX_NEWTON):
                gradient = 0.0
                hessian = 0.0
                for i in range(n):
                    if clever[t, i] == 0.0:
                        continue
                    at_risk = True if t == 0 else (following[t - 1, i] and event[i, t] == 0.0)
                    base = initial[t, i] if at_risk else 0.0
                    base = min(max(base, 1.0 - _ALPHA), _ALPHA)
                    eta = np.log(base / (1.0 - base)) + clever[t, i] * epsilon
                    if eta > 700.0:
                        eta = 700.0
                    elif eta < -700.0:
                        eta = -700.0
                    p = 1.0 / (1.0 + np.exp(-eta))
                    gradient += weights[i] * clever[t, i] * (pseudo[t, i] - p)
                    hessian += weights[i] * clever[t, i] * clever[t, i] * p * (1.0 - p)
                if abs(gradient) / total <= _NEWTON_TOL:
                    break
                if hessian <= 0.0 or not np.isfinite(hessian):
                    break
                epsilon += gradient / hessian
        for i in range(n):
            at_risk = True if t == 0 else (following[t - 1, i] and event[i, t] == 0.0)
            base = initial[t, i] if at_risk else 0.0
            base = min(max(base, 1.0 - _ALPHA), _ALPHA)
            covariate = 1.0 / cumulative[t, i] if at_risk else 0.0
            eta = np.log(base / (1.0 - base)) + covariate * epsilon
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            q = 1.0 / (1.0 + np.exp(-eta))
            targeted[t, i] = min(max(q, 1.0 - _ALPHA), _ALPHA)

    total_weight = 0.0
    plug_in = 0.0
    for i in range(n):
        total_weight += weights[i]
        plug_in += weights[i] * targeted[0, i]
    return plug_in / total_weight


@njit()
def _cell(
    following, cumulative, event, cause_event, initial, weights, horizon, times,
    psi_out, influence_out, index,
):
    n = weights.shape[0]
    targeted = np.empty((times, n))
    clever = np.zeros((times, n))
    pseudo = np.zeros((times, n))
    plug_in = _horizon_pass(
        following, cumulative, event, cause_event, initial, weights, horizon,
        targeted, clever, pseudo,
    )
    psi_out[index] = plug_in
    for i in range(n):
        value = targeted[0, i] - plug_in
        for t in range(horizon):
            value += clever[t, i] * (pseudo[t, i] - targeted[t, i])
        influence_out[index, i] = weights[i] * value


@njit()
def _incidence_serial(
    treatment_probability, censoring_probability, treated, uncensored, initial,
    assignment, event, cause_event, weights, horizons, lower, upper,
):
    n, times = treated.shape
    n_regimens = assignment.shape[0]
    n_causes = cause_event.shape[0]
    cells = n_regimens * n_causes * horizons.shape[0]
    psi = np.empty(cells)
    influence = np.empty((cells, n))
    for r in range(n_regimens):
        following = np.empty((times, n), dtype=np.bool_)
        cumulative = np.empty((times, n))
        _build_masks(
            treatment_probability, censoring_probability, treated, uncensored,
            assignment[r], lower, upper, following, cumulative,
        )
        for j in range(n_causes):
            for h in range(horizons.shape[0]):
                index = (r * n_causes + j) * horizons.shape[0] + h
                _cell(
                    following, cumulative, event, cause_event[j], initial, weights,
                    horizons[h], times, psi, influence, index,
                )
    return psi, influence


@pjit()
def _incidence_parallel(
    treatment_probability, censoring_probability, treated, uncensored, initial,
    assignment, event, cause_event, weights, horizons, lower, upper,
):
    """``prange`` over ``(regimen, cause, horizon)``.

    The masks are rebuilt per cell here rather than shared per regimen, which is the price
    of the flat parallel axis: sharing them would need a barrier between the mask pass and
    the cell pass, and at ``T`` nodes the mask pass is ``O(T n)`` against the cell's
    ``O(T^2 n)``.  Paying it back is what the flat axis buys, and the serial kernel keeps
    the shared form so the two costs are separable in the report.
    """
    n, times = treated.shape
    n_regimens = assignment.shape[0]
    n_causes = cause_event.shape[0]
    n_horizons = horizons.shape[0]
    cells = n_regimens * n_causes * n_horizons
    psi = np.empty(cells)
    influence = np.empty((cells, n))
    for index in prange(cells):
        h = index % n_horizons
        j = (index // n_horizons) % n_causes
        r = index // (n_horizons * n_causes)
        following = np.empty((times, n), dtype=np.bool_)
        cumulative = np.empty((times, n))
        _build_masks(
            treatment_probability, censoring_probability, treated, uncensored,
            assignment[r], lower, upper, following, cumulative,
        )
        _cell(
            following, cumulative, event, cause_event[j], initial, weights,
            horizons[h], times, psi, influence, index,
        )
    return psi, influence


def _run(inputs: dict[str, Any], kernel: Any) -> dict[str, Any]:
    fixture = inputs["fixture"]
    base = fixture.base
    lower, upper = inputs["g_bounds"]
    horizons = np.asarray(inputs["horizons"], dtype=np.int64)
    psi, influence = kernel(
        base.treatment_probability, base.censoring_probability, base.treated,
        base.uncensored, base.initial, base.assignment, fixture.event,
        fixture.cause_event, base.weights, horizons, float(lower), float(upper),
    )
    out: dict[str, Any] = {}
    index = 0
    for r in range(base.n_regimens):
        for cause in fixture.causes:
            for horizon in inputs["horizons"]:
                key = f"{base.labels[r]}|{cause}|{horizon}"
                out[f"psi_{key}"] = np.array([psi[index]])
                out[f"ic_{key}"] = influence[index]
                index += 1
    return out


def numba_incidence(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _incidence_serial)


def numba_incidence_parallel(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _incidence_parallel)


_IMPLEMENTATIONS: dict[str, Any] = {
    "numpy": numpy_incidence,
    "numpy_shared_masks": numpy_incidence_shared,
}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_incidence
    _IMPLEMENTATIONS["numba_parallel"] = numba_incidence_parallel

register(
    KernelSpec(
        name="survival_incidence",
        estimator="survival",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=compare_mapping,
        tolerance=(1e-8, 1e-9),
        parallel_axis="horizons",
        note="T(T+1)/2 backward passes per regimen per cause, all of them independent",
        dimensions={
            "n": 50_000,
            "n_times": 20,
            "n_regimens": 2,
            "n_causes": 1,
            "n_horizons": 0,
            "incidence": 0.06,
            "regime": "moderate",
            "g_lower": 0.01,
            "seed": 20260803,
        },
        amortise=False,
    )
)

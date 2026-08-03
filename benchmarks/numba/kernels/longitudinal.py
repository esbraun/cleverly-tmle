r"""The sequential backward recursion, with the node regressions cached.

``fit_regimen`` walks the nodes backwards and at each one fits a regression, builds masks,
divides by the cumulative mechanism, solves a one-column fluctuation and carries the
targeted prediction back.  The regression is a learner and is out of scope; everything
else is the package's own, and there is more of it than the flat structure suggests.

**The masks are quadratic in the node count, and that is the finding to check.**
``at_risk(t)`` is ``uncensored_through(t-1) & followed_through(t-1) & event_free(t-1)``, and
``followed_through`` is itself a loop of ``t`` ``&`` operations over ``n`` rows.  Called at
every node, that is ``O(T^2 n)`` boolean work per regimen -- and a survival fit runs one
backward pass *per horizon*, which makes it ``O(T^3 n)``.  At ``T = 2`` nobody notices.  At
the ``T = 100`` a discrete-time survival fit reaches, ten thousand full-length boolean
passes is not a rounding error.

So this kernel measures three things that a "does numba help" framing would run together:

* ``numpy`` -- the shipped shape: masks rebuilt per node, one ``np.where`` per array.
* ``numpy_prefix`` -- the same algorithm with the running masks carried *down* the nodes
  instead of rebuilt, which is an ``O(T n)`` algorithm rather than an ``O(T^2 n)`` one and
  needs no compiler at all.  If this closes the gap, the answer is "fix the algorithm".
* ``numba`` / ``numba_parallel`` -- the whole recursion fused, with the regimens as the
  parallel axis.

**Regimens are the safe parallel axis and time is not.**  After the node predictions exist
the regimens share nothing: each has its own masks, its own cumulative product and its own
fluctuation.  The recursion over ``t`` is genuinely sequential -- node ``t`` regresses on
what node ``t+1`` produced -- so it is left alone; parallelising it would require a scan
formulation that does not exist for a targeted recursion, and inventing one would be a
different estimator rather than a faster one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import Regime, make_longitudinal
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_mapping
from . import KernelSpec, register

__all__ = [
    "build",
    "numba_recursion",
    "numba_recursion_parallel",
    "numpy_recursion",
    "numpy_recursion_prefix",
]

_ALPHA = 0.9995
#: What a masked-out row's prediction is filled with.  Nothing reads it -- every use is
#: masked first -- but it has to be finite or the fluctuation's arithmetic produces nan.
_FILLER = 0.0
#: Newton iterations for the one-column node fluctuation.  The package caps at 20 and
#: converges in two or three; fixing it here keeps every implementation doing the same
#: work, which is what makes the timings comparable.
_MAX_NEWTON = 20
_NEWTON_TOL = 1e-10


def build(
    n: int = 100_000,
    n_times: int = 5,
    n_regimens: int = 4,
    regime: Regime = "moderate",
    dynamic: bool = False,
    g_lower: float = 0.01,
    seed: int = 20260803,
) -> dict[str, Any]:
    fixture = make_longitudinal(
        n, n_times=n_times, n_regimens=n_regimens, regime=regime, dynamic=dynamic, seed=seed
    )
    return {"fixture": fixture, "g_bounds": (g_lower, 1.0 - g_lower)}


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


# --------------------------------------------------------------------------- numpy


def _cumulative(fixture, assignment, bounds):
    """``(n, T)`` product of the bounded treatment and censoring factors up to each node."""
    lower, upper = bounds
    running = np.ones(fixture.n)
    columns = []
    for t in range(fixture.n_times):
        g1 = np.clip(fixture.treatment_probability[:, t], lower, upper)
        running = running * np.where(assignment[:, t] == 1.0, g1, 1.0 - g1)
        running = running * np.clip(fixture.censoring_probability[:, t], lower, upper)
        columns.append(running.copy())
    return np.column_stack(columns)


def _node_fluctuation(pseudo, initial, clever, weights, mask):
    """The package's one-column logistic Newton at a node, with an offset and no intercept."""
    shrunk = np.clip(initial, 1.0 - _ALPHA, _ALPHA)
    offset = np.log(shrunk / (1.0 - shrunk))
    epsilon = 0.0
    total = weights[mask].sum()
    if total <= 0.0:
        return 0.0
    for _ in range(_MAX_NEWTON):
        p = _expit(offset + clever * epsilon)
        gradient = float((weights * clever * (pseudo - p))[mask].sum())
        if abs(gradient) / total <= _NEWTON_TOL:
            break
        hessian = float((weights * clever * clever * p * (1.0 - p))[mask].sum())
        if hessian <= 0.0 or not np.isfinite(hessian):
            break
        epsilon += gradient / hessian
    return epsilon


def numpy_recursion(inputs: dict[str, Any], *, prefix: bool = False) -> dict[str, Any]:
    """The shipped recursion's shape, with the node regressions read from the fixture.

    ``prefix=False`` rebuilds ``followed_through`` and ``uncensored_through`` at every
    node, which is what ``LongitudinalData`` does; ``prefix=True`` carries them down
    instead.  The two compute the same masks -- the second is a prefix scan of the first
    -- so they agree exactly and differ only in how much boolean work they do.
    """
    fixture = inputs["fixture"]
    bounds = inputs["g_bounds"]
    n, times = fixture.n, fixture.n_times
    weights = fixture.weights
    out: dict[str, Any] = {}

    for r in range(fixture.n_regimens):
        assignment = fixture.assignment[r]
        cumulative = _cumulative(fixture, assignment, bounds)
        followed_prefix = np.ones(n, dtype=bool)
        uncensored_prefix = np.ones(n, dtype=bool)
        masks = []
        for t in range(times):
            if prefix:
                followed_prefix = followed_prefix & (fixture.treated[:, t] == assignment[:, t])
                uncensored_prefix = uncensored_prefix & (fixture.uncensored[:, t] == 1.0)
                masks.append((followed_prefix.copy(), uncensored_prefix.copy()))
            else:
                followed = np.ones(n, dtype=bool)
                uncensored = np.ones(n, dtype=bool)
                for s in range(t + 1):
                    followed = followed & (fixture.treated[:, s] == assignment[:, s])
                    uncensored = uncensored & (fixture.uncensored[:, s] == 1.0)
                masks.append((followed, uncensored))

        carried = fixture.outcome
        targeted_first = None
        clever_terms = []
        for t in range(times - 1, -1, -1):
            following = masks[t][0] & masks[t][1]
            at_risk = np.ones(n, dtype=bool) if t == 0 else (masks[t - 1][0] & masks[t - 1][1])
            initial = np.where(at_risk, fixture.initial[t], _FILLER)
            denominator = np.where(at_risk, cumulative[:, t], 1.0)
            counterfactual = np.where(at_risk, 1.0 / denominator, 0.0)
            clever = np.where(following, counterfactual, 0.0)
            epsilon = _node_fluctuation(carried, initial, clever, weights, following)
            shrunk = np.clip(initial, 1.0 - _ALPHA, _ALPHA)
            targeted = np.clip(
                _expit(np.log(shrunk / (1.0 - shrunk)) + counterfactual * epsilon),
                1.0 - _ALPHA,
                _ALPHA,
            )
            clever_terms.append((clever, carried, targeted))
            carried = np.where(at_risk, targeted, _FILLER)
            targeted_first = targeted

        psi = float(np.average(targeted_first, weights=weights))
        influence = targeted_first - psi
        for clever, pseudo, targeted in clever_terms:
            influence = influence + clever * (pseudo - targeted)
        out[f"psi_{fixture.labels[r]}"] = np.array([psi])
        out[f"ic_{fixture.labels[r]}"] = weights * influence
    return out


def numpy_recursion_prefix(inputs: dict[str, Any]) -> dict[str, Any]:
    """The same recursion with the masks carried rather than rebuilt: ``O(T n)``, not ``O(T^2 n)``."""
    return numpy_recursion(inputs, prefix=True)


# --------------------------------------------------------------------------- numba


@njit()
def _recursion_one(
    treatment_probability, censoring_probability, treated, uncensored, outcome, initial,
    assignment, weights, lower, upper, psi_out, influence_out, index,
):
    """One regimen's whole backward pass, fused.

    Masks are carried forward once (the ``O(T n)`` form), the cumulative product is built
    in the same pass, and each node's fluctuation, update and influence term are computed
    without materialising an intermediate per node beyond what the recursion has to keep --
    the targeted prediction at each node, which the influence curve reads afterwards.
    """
    n = outcome.shape[0]
    times = treated.shape[1]

    following = np.ones((times, n), dtype=np.bool_)
    cumulative = np.ones((times, n))
    followed = np.ones(n, dtype=np.bool_)
    running = np.ones(n)
    for t in range(times):
        g1 = treatment_probability[:, t]
        c1 = censoring_probability[:, t]
        for i in range(n):
            if treated[i, t] != assignment[i, t] or uncensored[i, t] != 1.0:
                followed[i] = False
            g = min(max(g1[i], lower), upper)
            if assignment[i, t] != 1.0:
                g = 1.0 - g
            c = min(max(c1[i], lower), upper)
            running[i] = running[i] * g * c
            cumulative[t, i] = running[i]
            following[t, i] = followed[i]

    targeted = np.empty((times, n))
    clever = np.empty((times, n))
    pseudo = np.empty((times, n))
    carried = outcome.copy()

    for t in range(times - 1, -1, -1):
        for i in range(n):
            pseudo[t, i] = carried[i]
        # `at_risk` at node t is `following` at node t-1, which is the identity the
        # recursion closes on; at the first node everyone is at risk.
        epsilon = 0.0
        total = 0.0
        for i in range(n):
            at_risk = True if t == 0 else following[t - 1, i]
            covariate = 1.0 / cumulative[t, i] if at_risk else 0.0
            clever[t, i] = covariate if following[t, i] else 0.0
            if following[t, i]:
                total += weights[i]
        if total > 0.0:
            for _ in range(_MAX_NEWTON):
                gradient = 0.0
                hessian = 0.0
                for i in range(n):
                    if not following[t, i]:
                        continue
                    at_risk = True if t == 0 else following[t - 1, i]
                    base = initial[t, i] if at_risk else _FILLER
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
            at_risk = True if t == 0 else following[t - 1, i]
            base = initial[t, i] if at_risk else _FILLER
            base = min(max(base, 1.0 - _ALPHA), _ALPHA)
            # The update is applied at the *counterfactual* covariate -- no follower
            # indicator -- exactly as the arm path reads `submodel.arms[a]` rather than
            # `.observed`. Reading `clever` here would leave every node after the first
            # un-updated for the rows that did not follow.
            covariate = 1.0 / cumulative[t, i] if at_risk else 0.0
            eta = np.log(base / (1.0 - base)) + covariate * epsilon
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            q = 1.0 / (1.0 + np.exp(-eta))
            q = min(max(q, 1.0 - _ALPHA), _ALPHA)
            targeted[t, i] = q
            carried[i] = q if at_risk else _FILLER

    total_weight = 0.0
    plug_in = 0.0
    for i in range(n):
        total_weight += weights[i]
        plug_in += weights[i] * targeted[0, i]
    plug_in /= total_weight
    psi_out[index] = plug_in
    for i in range(n):
        value = targeted[0, i] - plug_in
        for t in range(times):
            value += clever[t, i] * (pseudo[t, i] - targeted[t, i])
        influence_out[index, i] = weights[i] * value


@njit()
def _recursion_serial(
    treatment_probability, censoring_probability, treated, uncensored, outcome, initial,
    assignment, weights, lower, upper,
):
    n_regimens = assignment.shape[0]
    psi = np.empty(n_regimens)
    influence = np.empty((n_regimens, outcome.shape[0]))
    for r in range(n_regimens):
        _recursion_one(
            treatment_probability, censoring_probability, treated, uncensored, outcome,
            initial, assignment[r], weights, lower, upper, psi, influence, r,
        )
    return psi, influence


@pjit()
def _recursion_parallel(
    treatment_probability, censoring_probability, treated, uncensored, outcome, initial,
    assignment, weights, lower, upper,
):
    """One regimen per thread.

    The regimens share every nuisance prediction and nothing else -- separate masks,
    separate cumulative product, separate fluctuation -- so this axis needs no reduction
    and no locking.  It is also the axis a real fit has the most of: a curve over four
    static plans and a dynamic rule is five, and a working model over a grid is more.
    """
    n_regimens = assignment.shape[0]
    psi = np.empty(n_regimens)
    influence = np.empty((n_regimens, outcome.shape[0]))
    for r in prange(n_regimens):
        _recursion_one(
            treatment_probability, censoring_probability, treated, uncensored, outcome,
            initial, assignment[r], weights, lower, upper, psi, influence, r,
        )
    return psi, influence


def _run(inputs: dict[str, Any], kernel: Any) -> dict[str, Any]:
    fixture = inputs["fixture"]
    lower, upper = inputs["g_bounds"]
    psi, influence = kernel(
        fixture.treatment_probability,
        fixture.censoring_probability,
        fixture.treated,
        fixture.uncensored,
        fixture.outcome,
        fixture.initial,
        fixture.assignment,
        fixture.weights,
        float(lower),
        float(upper),
    )
    out: dict[str, Any] = {}
    for r, label in enumerate(fixture.labels):
        out[f"psi_{label}"] = np.array([psi[r]])
        out[f"ic_{label}"] = influence[r]
    return out


def numba_recursion(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _recursion_serial)


def numba_recursion_parallel(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _recursion_parallel)


_IMPLEMENTATIONS: dict[str, Any] = {
    "numpy": numpy_recursion,
    "numpy_prefix_masks": numpy_recursion_prefix,
}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_recursion
    _IMPLEMENTATIONS["numba_parallel"] = numba_recursion_parallel

register(
    KernelSpec(
        name="ltmle_backward_recursion",
        estimator="ltmle",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=compare_mapping,
        # A Newton solved to a gradient tolerance rather than to machine zero: two
        # implementations can stop on either side of the last step, so the coefficient
        # agrees to the tolerance and not beyond it.
        tolerance=(1e-8, 1e-9),
        parallel_axis="regimens",
        note="masks are O(T^2 n) as written; the regimens are independent once cached",
        dimensions={
            "n": 100_000,
            "n_times": 5,
            "n_regimens": 4,
            "regime": "moderate",
            "dynamic": False,
            "g_lower": 0.01,
            "seed": 20260803,
        },
        amortise=True,
    )
)

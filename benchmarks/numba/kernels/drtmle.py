r"""DR-TMLE's extra targeting equations, and the cost that turned out to dominate them.

``DRTMLE`` solves two further score equations against reduced-dimension regressions and
alternates between them and the mechanism until both are solved.  The package-owned
arithmetic per round is: build ``Q_r`` and ``g_{r,2}``'s covariates, solve two logistic
fluctuations, re-tilt, recompute the two relative scores, test the stall factor.  That is
what ``numpy_reduction_round`` measures and what the compiled arms are compared against.

**But that is not where a DR-TMLE ``retarget`` spends its time, and the profile is worth
recording here rather than only in the inventory.**  At ``n = 20,000`` with ``glm``
learners, ``DRTMLE.retarget`` measures **16.1 s against a 7.3 s full fit** -- more than
twice the fit it is supposed to be a cheap re-run of.  Of the 72 s spent in three such
calls, **41 s is inside :mod:`threadpoolctl`**: ``_make_controller_from_path`` is entered
3.1 million times and ``_find_libraries_with_dl_iterate_phdr`` 12,432 times.

The cause is structural rather than accidental.  ``ReductionSpec.refit`` re-fits the
reduced regressions inside the alternation -- correctly, since ``g_{r,2}`` is a functional
of the mechanism being tilted -- and every learner fit enters
:func:`cleverly.learners.thread_limit`, which constructs a fresh
``threadpoolctl.ThreadpoolController`` and walks every shared library the process has
loaded.  Measured here at **1.5 ms per entry**.  With thousands of small fits per
``retarget``, the thread-limiting costs more than the fitting.

That is a real finding and it is not a numba finding, which is why it has a measurement
arm of its own (:func:`thread_limit_overhead`) rather than a paragraph.  No compiled kernel
addresses it; hoisting the controller out of the per-fit path does.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..fixtures import Regime, make_targeting
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_mapping
from . import KernelSpec, register

__all__ = [
    "build",
    "numba_reduction_round",
    "numba_reduction_round_parallel",
    "numpy_reduction_round",
    "thread_limit_overhead",
]

_ALPHA = 0.9995
_MAX_NEWTON = 20
_NEWTON_TOL = 1e-10


def build(
    n: int = 100_000,
    n_rounds: int = 8,
    regime: Regime = "moderate",
    seed: int = 20260803,
) -> dict[str, Any]:
    """A fit plus the two reduced regressions, as cached predictions.

    ``Q_r`` (the outcome-reduction) and ``g_{r,2}`` (the mechanism-reduction) both vanish
    row by row at correct nuisances, which is exactly the regime in which a Gateaux check
    goes blind -- so the fixture puts them at a *non-zero* value, drawn around a small but
    real bias.  A fixture at the truth would make every implementation agree for a reason
    that has nothing to do with whether they compute the same thing.
    """
    fixture = make_targeting(n, n_arms=2, regime=regime, seed=seed)
    rng = np.random.default_rng(seed + 17)
    reduced_outcome = 0.05 * rng.standard_normal((n, 2))
    reduced_mechanism = np.clip(0.15 + 0.05 * rng.standard_normal((n, 2)), 1e-3, 1.0 - 1e-3)
    return {
        "fixture": fixture,
        "reduced_outcome": np.ascontiguousarray(reduced_outcome),
        "reduced_mechanism": np.ascontiguousarray(reduced_mechanism),
        "n_rounds": n_rounds,
    }


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


# --------------------------------------------------------------------------- numpy


def numpy_reduction_round(inputs: dict[str, Any]) -> dict[str, Any]:
    """``n_rounds`` of the alternation's package-owned arithmetic, per arm.

    One round is: covariates from the current mechanism, a one-column fluctuation of each
    of the two equations, a re-tilt of the mechanism, and the two relative scores that the
    stall test reads.  The *learner* refit that a real round also does is out of the timed
    region by construction -- it is a nuisance fit -- and its own cost is the subject of
    :func:`thread_limit_overhead`.
    """
    fixture = inputs["fixture"]
    y = fixture.outcome
    weights = fixture.weights
    mask = fixture.observed
    indicator = fixture.treatment_indicator
    q_reduced = inputs["reduced_outcome"]
    g_reduced = inputs["reduced_mechanism"]

    mechanism = fixture.propensity.copy()
    initial = np.clip(fixture.initial_observed, 1.0 - _ALPHA, _ALPHA)
    offset = np.log(initial / (1.0 - initial))

    epsilons = np.zeros((inputs["n_rounds"], 2))
    scores = np.zeros((inputs["n_rounds"], 2))
    for r in range(inputs["n_rounds"]):
        for arm in (0, 1):
            g = np.clip(mechanism[:, arm], 1e-3, 1.0 - 1e-3)
            # Equation (8)-shaped: the ordinary 1/g covariate carrying the
            # reduction.
            covariate = indicator[:, arm] / g * (1.0 + q_reduced[:, arm])
            epsilon = 0.0
            total = weights[mask].sum()
            for _ in range(_MAX_NEWTON):
                p = _expit(offset + covariate * epsilon)
                gradient = float((weights * covariate * (y - p))[mask].sum())
                if abs(gradient) / total <= _NEWTON_TOL:
                    break
                hessian = float((weights * covariate * covariate * p * (1.0 - p))[mask].sum())
                if hessian <= 0.0 or not np.isfinite(hessian):
                    break
                epsilon += gradient / hessian
            epsilons[r, arm] = epsilon
            # Equation (10)-shaped: the mechanism is tilted along g_{r,2}/g_{r,1}, and the
            # score of that equation is what the stall factor is measured on.
            tilt = g_reduced[:, arm] / np.maximum(g, 1e-3)
            mechanism[:, arm] = np.clip(
                _expit(np.log(g / (1.0 - g)) + 1e-3 * tilt), 1e-3, 1.0 - 1e-3
            )
            residual = np.zeros_like(y)
            residual[mask] = y[mask] - _expit(offset + covariate * epsilon)[mask]
            scores[r, arm] = float(np.mean(weights * residual * covariate))
    return {"epsilons": epsilons, "scores": scores, "mechanism": mechanism}


# --------------------------------------------------------------------------- numba


@njit()
def _round_arm(y, offset, weights, mask, indicator, mechanism, q_reduced, g_reduced, arm):
    n = y.shape[0]
    epsilon = 0.0
    total = 0.0
    for i in range(n):
        if mask[i]:
            total += weights[i]
    for _ in range(_MAX_NEWTON):
        gradient = 0.0
        hessian = 0.0
        for i in range(n):
            if not mask[i]:
                continue
            g = min(max(mechanism[i, arm], 1e-3), 1.0 - 1e-3)
            covariate = indicator[i, arm] / g * (1.0 + q_reduced[i, arm])
            eta = offset[i] + covariate * epsilon
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            p = 1.0 / (1.0 + np.exp(-eta))
            gradient += weights[i] * covariate * (y[i] - p)
            hessian += weights[i] * covariate * covariate * p * (1.0 - p)
        if abs(gradient) / total <= _NEWTON_TOL:
            break
        if hessian <= 0.0 or not np.isfinite(hessian):
            break
        epsilon += gradient / hessian

    score = 0.0
    for i in range(n):
        g = min(max(mechanism[i, arm], 1e-3), 1.0 - 1e-3)
        covariate = indicator[i, arm] / g * (1.0 + q_reduced[i, arm])
        if mask[i]:
            eta = offset[i] + covariate * epsilon
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            p = 1.0 / (1.0 + np.exp(-eta))
            score += weights[i] * (y[i] - p) * covariate
        tilt = g_reduced[i, arm] / max(g, 1e-3)
        eta_g = np.log(g / (1.0 - g)) + 1e-3 * tilt
        if eta_g > 700.0:
            eta_g = 700.0
        elif eta_g < -700.0:
            eta_g = -700.0
        updated = 1.0 / (1.0 + np.exp(-eta_g))
        mechanism[i, arm] = min(max(updated, 1e-3), 1.0 - 1e-3)
    return epsilon, score / n


@njit()
def _rounds_serial(y, offset, weights, mask, indicator, mechanism, q_reduced, g_reduced, n_rounds):
    epsilons = np.zeros((n_rounds, 2))
    scores = np.zeros((n_rounds, 2))
    for r in range(n_rounds):
        for arm in range(2):
            epsilons[r, arm], scores[r, arm] = _round_arm(
                y, offset, weights, mask, indicator, mechanism, q_reduced, g_reduced, arm
            )
    return epsilons, scores, mechanism


@pjit()
def _rounds_parallel(
    y, offset, weights, mask, indicator, mechanism, q_reduced, g_reduced, n_rounds
):
    """``prange`` over the arms, which is the only independent axis the alternation has.

    The *rounds* are strictly sequential -- each reads the mechanism the last one tilted --
    so the parallel arm here is two-wide on a two-arm fit and ``K``-wide on a multi-arm
    one.  Two-wide parallelism on four cores is a poor bargain and the report is expected
    to say so; running it is what turns that expectation into a measurement.
    """
    epsilons = np.zeros((n_rounds, 2))
    scores = np.zeros((n_rounds, 2))
    for r in range(n_rounds):
        for arm in prange(2):
            epsilons[r, arm], scores[r, arm] = _round_arm(
                y, offset, weights, mask, indicator, mechanism, q_reduced, g_reduced, arm
            )
    return epsilons, scores, mechanism


def _run(inputs: dict[str, Any], kernel: Any) -> dict[str, Any]:
    fixture = inputs["fixture"]
    initial = np.clip(fixture.initial_observed, 1.0 - _ALPHA, _ALPHA)
    epsilons, scores, mechanism = kernel(
        fixture.outcome,
        np.log(initial / (1.0 - initial)),
        fixture.weights,
        fixture.observed,
        fixture.treatment_indicator,
        fixture.propensity.copy(),
        inputs["reduced_outcome"],
        inputs["reduced_mechanism"],
        int(inputs["n_rounds"]),
    )
    return {"epsilons": epsilons, "scores": scores, "mechanism": mechanism}


def numba_reduction_round(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _rounds_serial)


def numba_reduction_round_parallel(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _rounds_parallel)


# ------------------------------------------------------- the non-numba finding


def thread_limit_overhead(repeats: int = 200) -> dict[str, float]:
    """Seconds to enter and leave ``cleverly.learners.thread_limit`` once.

    Not a kernel and deliberately not registered as one: there is nothing to compile and
    nothing to parallelise.  It is here because the DR-TMLE profile put 57% of a
    ``retarget`` inside it, and a benchmark suite that reported only the arithmetic it
    could compile would have left that unmeasured.

    The context manager builds a fresh ``threadpoolctl.ThreadpoolController`` per call,
    which walks the process's loaded shared objects through ``dl_iterate_phdr``.  The cost
    is per *entry*, so it scales with the number of learner fits and not with ``n``.
    """
    from threadpoolctl import threadpool_limits

    from cleverly.learners._threads import thread_limit

    start = time.perf_counter()
    for _ in range(repeats):
        with thread_limit():
            pass
    per_entry = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        with threadpool_limits(limits=1):
            pass
    raw = (time.perf_counter() - start) / repeats
    return {"thread_limit_seconds": per_entry, "threadpool_limits_seconds": raw}


_IMPLEMENTATIONS: dict[str, Any] = {"numpy": numpy_reduction_round}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_reduction_round
    _IMPLEMENTATIONS["numba_parallel"] = numba_reduction_round_parallel

register(
    KernelSpec(
        name="drtmle_reduction_rounds",
        estimator="drtmle",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=compare_mapping,
        tolerance=(1e-8, 1e-9),
        parallel_axis="arms",
        note=(
            "the alternation's own arithmetic; the profile puts 57% of a real retarget in "
            "threadpoolctl instead, which no compiled kernel addresses"
        ),
        dimensions={
            "n": 100_000,
            "n_rounds": 8,
            "regime": "moderate",
            "seed": 20260803,
        },
        amortise=True,
    )
)

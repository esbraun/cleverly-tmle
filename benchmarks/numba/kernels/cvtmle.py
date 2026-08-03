r"""CV-TMLE's fold loop: the one kernel whose interesting axis is a task, not a row.

``TMLE._solve_by_fold`` is a plain serial ``for`` over ``nuisance.folds``.  Each iteration
restricts the covariate and the fit to that fold's rows, solves its own fluctuation, and
the pieces are stitched back by index.  The folds are independent *by construction* -- a
fold's ``epsilon`` is fitted only against rows whose nuisance predictions came from models
that never saw them, which is the whole point of the scheme -- so this is a task-parallel
axis sitting unused.

Measured on a real ``retarget`` at ``n = 20,000`` with ``library="glm"``, the absolute cost
grows with the fold count while the share falls, because the *fit* gets more expensive
faster than the targeting does:

===== ============ ==============
folds ``retarget`` share of a fit
===== ============ ==============
2     72.5 ms      16.2%
5     85.7 ms      10.4%
10    115.8 ms     8.5%
20    156.1 ms     6.2%
===== ============ ==============

**Which parallelism, though, is the question this kernel exists to answer.**  Folds cut
each task's row count by ``1/F``, so within-fold row parallelism has less and less to work
with as the fold count rises, while task parallelism has more and more tasks.  Somewhere
those cross, and the crossing point is not guessable -- it depends on the thread-pool
overhead against the per-fold work, both of which are properties of the machine.  So four
arms:

``numpy``
    the shipped serial loop.
``numpy_threads``
    the same loop over a :class:`~concurrent.futures.ThreadPoolExecutor`.  **Threads, not
    processes**, and that is the interesting part: the per-fold body is numpy, which
    releases the GIL for its BLAS calls and its ufuncs but not for the Python around them.
    Whether that is enough is exactly what this arm measures, and it is the cheap option --
    processes would have to pickle a slice of every array per fold.
``numba``
    the fold loop compiled, still serial.
``numba_parallel``
    ``prange`` over folds, each fold's solve independent, no reduction.

The stitching is included in the timed region for all four.  It is ``O(n)`` fancy-indexing
that the parallel arms still have to do serially, so leaving it out would report a speed-up
on the part that parallelises and hide the part that does not.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..fixtures import Regime, make_targeting
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_mapping
from . import KernelSpec, register

__all__ = [
    "build",
    "numba_fold_targeting",
    "numba_fold_targeting_parallel",
    "numpy_fold_targeting",
    "numpy_fold_targeting_threads",
]

_ALPHA = 0.9995
_MAX_NEWTON = 20
_NEWTON_TOL = 1e-12


def build(
    n: int = 100_000,
    n_folds: int = 10,
    n_arms: int = 2,
    regime: Regime = "moderate",
    seed: int = 20260803,
) -> dict[str, Any]:
    fixture = make_targeting(n, n_arms=n_arms, n_folds=n_folds, regime=regime, seed=seed)
    # The fold membership as an index list, built once: the package builds it once too
    # (`nuisance.folds` iterates train/test pairs), and rebuilding it inside the timed
    # region would measure `np.flatnonzero` rather than the targeting.
    members = [np.flatnonzero(fixture.folds == f) for f in range(fixture.n_folds)]
    starts = np.zeros(fixture.n_folds + 1, dtype=np.int64)
    for f, index in enumerate(members):
        starts[f + 1] = starts[f] + index.size
    return {
        "fixture": fixture,
        "members": members,
        "flat_members": np.concatenate(members).astype(np.int64),
        "starts": starts,
    }


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


# --------------------------------------------------------------------------- numpy


def _one_fold(fixture, index):
    """One fold's fluctuation and its targeted predictions, on that fold's rows only."""
    y = fixture.outcome[index]
    weights = fixture.weights[index]
    mask = fixture.observed[index]
    h = fixture.covariate_observed[index]
    h_arms = fixture.covariate_arms[index]
    initial = np.clip(fixture.initial_observed[index], 1.0 - _ALPHA, _ALPHA)
    arms = np.clip(fixture.initial_arms[index], 1.0 - _ALPHA, _ALPHA)
    offset = np.log(initial / (1.0 - initial))

    epsilon = np.zeros(h.shape[1])
    total = weights[mask].sum()
    for _ in range(_MAX_NEWTON):
        p = _expit(offset + h @ epsilon)
        gradient = h[mask].T @ (weights[mask] * (y[mask] - p[mask]))
        if total <= 0.0 or np.max(np.abs(gradient)) / total <= _NEWTON_TOL:
            break
        variance = np.zeros_like(p)
        variance[mask] = weights[mask] * p[mask] * (1.0 - p[mask])
        hessian = h.T @ (h * variance[:, None])
        try:
            epsilon = epsilon + np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
    targeted = np.clip(_expit(offset + h @ epsilon), 1.0 - _ALPHA, _ALPHA)
    targeted_arms = np.clip(
        _expit(np.log(arms / (1.0 - arms)) + np.einsum("nkp,p->nk", h_arms, epsilon)),
        1.0 - _ALPHA,
        _ALPHA,
    )
    return epsilon, targeted, targeted_arms


def _stitch(fixture, members, results):
    """Reassemble the fold-wise pieces into full-length arrays, by index.

    Serial in every implementation, deliberately: it is what the package does
    (``fluctuation.stitch``), it is ``O(n)`` fancy-indexing, and excluding it would report
    a speed-up on the half that parallelises while hiding the half that does not.
    """
    observed = np.empty(fixture.n)
    arms = np.empty_like(fixture.initial_arms)
    epsilons = np.empty((len(members), fixture.dim))
    for f, index in enumerate(members):
        epsilon, targeted, targeted_arms = results[f]
        observed[index] = targeted
        arms[index] = targeted_arms
        epsilons[f] = epsilon
    return {"epsilon": epsilons, "targeted_observed": observed, "targeted_arms": arms}


def numpy_fold_targeting(inputs: dict[str, Any]) -> dict[str, Any]:
    """The shipped shape: a serial ``for`` over the folds, then one stitch."""
    fixture = inputs["fixture"]
    members = inputs["members"]
    return _stitch(fixture, members, [_one_fold(fixture, index) for index in members])


def numpy_fold_targeting_threads(inputs: dict[str, Any]) -> dict[str, Any]:
    """The same loop over a thread pool.

    Threads rather than processes because the per-fold body is numpy: its BLAS calls and
    ufuncs release the GIL, and processes would have to pickle a slice of six arrays per
    fold to gain the same thing.  Whether the Python *between* those calls holds the GIL
    long enough to spoil it is the measurement.

    The worker count is read from numba's -- a slightly odd source, and the right one:
    this arm is measured under the same ``--num-cores`` sweep as the compiled ones, and
    :mod:`..resources` sets that count on numba's pool for every mode.
    """
    from ..implementations.numba_parallel import effective_threads

    fixture = inputs["fixture"]
    members = inputs["members"]
    workers = max(1, effective_threads())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda index: _one_fold(fixture, index), members))
    return _stitch(fixture, members, results)


# --------------------------------------------------------------------------- numba


@njit()
def _fold_solve(y, weights, mask, h, h_arms, initial, arms, rows, epsilon, targeted, targeted_arms):
    """One fold's Newton and update, reading its rows through an index vector.

    Indexed rather than sliced: the fold's rows are scattered through the sample, so a
    contiguous copy would be an ``O(n_fold * p)`` gather per implementation and would put a
    different amount of memory traffic in each arm.  Reading through ``rows`` is what the
    numpy path's fancy-indexing does, done once.
    """
    count = rows.shape[0]
    columns = h.shape[1]
    n_arms = arms.shape[1]
    gradient = np.zeros(columns)
    hessian = np.zeros((columns, columns))
    for j in range(columns):
        epsilon[j] = 0.0

    total = 0.0
    for r in range(count):
        i = rows[r]
        if mask[i]:
            total += weights[i]

    for _ in range(_MAX_NEWTON):
        for j in range(columns):
            gradient[j] = 0.0
            for k in range(columns):
                hessian[j, k] = 0.0
        for r in range(count):
            i = rows[r]
            if not mask[i]:
                continue
            base = min(max(initial[i], 1.0 - _ALPHA), _ALPHA)
            eta = np.log(base / (1.0 - base))
            for j in range(columns):
                eta += h[i, j] * epsilon[j]
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            p = 1.0 / (1.0 + np.exp(-eta))
            residual = weights[i] * (y[i] - p)
            variance = weights[i] * p * (1.0 - p)
            for j in range(columns):
                gradient[j] += residual * h[i, j]
                for k in range(columns):
                    hessian[j, k] += variance * h[i, j] * h[i, k]
        largest = 0.0
        for j in range(columns):
            if abs(gradient[j]) > largest:
                largest = abs(gradient[j])
        if total <= 0.0 or largest / total <= _NEWTON_TOL:
            break
        step = np.linalg.solve(hessian, gradient)
        finite = True
        for j in range(columns):
            if not np.isfinite(step[j]):
                finite = False
        if not finite:
            break
        for j in range(columns):
            epsilon[j] += step[j]

    for r in range(count):
        i = rows[r]
        base = min(max(initial[i], 1.0 - _ALPHA), _ALPHA)
        eta = np.log(base / (1.0 - base))
        for j in range(columns):
            eta += h[i, j] * epsilon[j]
        if eta > 700.0:
            eta = 700.0
        elif eta < -700.0:
            eta = -700.0
        q = 1.0 / (1.0 + np.exp(-eta))
        targeted[i] = min(max(q, 1.0 - _ALPHA), _ALPHA)
        for k in range(n_arms):
            base_k = min(max(arms[i, k], 1.0 - _ALPHA), _ALPHA)
            eta_k = np.log(base_k / (1.0 - base_k))
            for j in range(columns):
                eta_k += h_arms[i, k, j] * epsilon[j]
            if eta_k > 700.0:
                eta_k = 700.0
            elif eta_k < -700.0:
                eta_k = -700.0
            q_k = 1.0 / (1.0 + np.exp(-eta_k))
            targeted_arms[i, k] = min(max(q_k, 1.0 - _ALPHA), _ALPHA)


@njit()
def _folds_serial(y, weights, mask, h, h_arms, initial, arms, flat_members, starts, n_folds):
    epsilons = np.zeros((n_folds, h.shape[1]))
    targeted = np.empty(y.shape[0])
    targeted_arms = np.empty_like(arms)
    for f in range(n_folds):
        _fold_solve(
            y,
            weights,
            mask,
            h,
            h_arms,
            initial,
            arms,
            flat_members[starts[f] : starts[f + 1]],
            epsilons[f],
            targeted,
            targeted_arms,
        )
    return epsilons, targeted, targeted_arms


@pjit()
def _folds_parallel(y, weights, mask, h, h_arms, initial, arms, flat_members, starts, n_folds):
    """``prange`` over folds.

    Every output slot is written by exactly one fold -- the folds partition the rows -- so
    there is no shared accumulator and no reduction.  The stitch is *implicit* here: each
    fold writes straight into the full-length output at its own indices, which is what the
    partition makes safe and is why this arm has no separate stitching pass.
    """
    epsilons = np.zeros((n_folds, h.shape[1]))
    targeted = np.empty(y.shape[0])
    targeted_arms = np.empty_like(arms)
    for f in prange(n_folds):
        _fold_solve(
            y,
            weights,
            mask,
            h,
            h_arms,
            initial,
            arms,
            flat_members[starts[f] : starts[f + 1]],
            epsilons[f],
            targeted,
            targeted_arms,
        )
    return epsilons, targeted, targeted_arms


def _run(inputs: dict[str, Any], kernel: Any) -> dict[str, Any]:
    fixture = inputs["fixture"]
    epsilons, targeted, targeted_arms = kernel(
        fixture.outcome,
        fixture.weights,
        fixture.observed,
        np.ascontiguousarray(fixture.covariate_observed),
        np.ascontiguousarray(fixture.covariate_arms),
        fixture.initial_observed,
        fixture.initial_arms,
        inputs["flat_members"],
        inputs["starts"],
        int(fixture.n_folds),
    )
    return {
        "epsilon": epsilons,
        "targeted_observed": targeted,
        "targeted_arms": targeted_arms,
    }


def numba_fold_targeting(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _folds_serial)


def numba_fold_targeting_parallel(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _folds_parallel)


_IMPLEMENTATIONS: dict[str, Any] = {
    "numpy": numpy_fold_targeting,
    "numpy_threads": numpy_fold_targeting_threads,
}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_fold_targeting
    _IMPLEMENTATIONS["numba_parallel"] = numba_fold_targeting_parallel

register(
    KernelSpec(
        name="cvtmle_fold_targeting",
        estimator="cvtmle",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=compare_mapping,
        # A Newton per fold, stopped on a gradient tolerance: the coefficient agrees to
        # that tolerance and the targeted predictions to rather better, since `expit`
        # compresses.
        tolerance=(1e-7, 1e-9),
        parallel_axis="folds",
        note="a serial Python loop over folds that are independent by construction",
        dimensions={
            "n": 100_000,
            "n_folds": 10,
            "n_arms": 2,
            "regime": "moderate",
            "seed": 20260803,
        },
        amortise=True,
    )
)

r"""The universal least-favourable walk: many small passes, in a Python loop.

``solve_one_step`` walks ``logit Q_{t+dt} = logit Q_t + dt h^T d_t`` until the score
vanishes.  Every step of that walk, in Python, does:

.. code-block:: text

    score_columns   -> a zeros(n), a masked write, an (n, p) product, a mean
    apply_logistic  -> for the observed fit and *each of K arms*:
                       logit (n), an (n, p) @ (p,) product, expit (n), a clip (n)
    quasi_loglik    -> a clip (n), two logs (n), a weighted sum
    and again for the trial step, because a rejected step is computed and thrown away

At ``K = 2`` that is roughly twenty full-length array passes and a dozen temporaries per
step, and the walk takes tens to thousands of steps -- the package caps it at 20,000.  So
this is the one kernel in the package where the interpreter genuinely is in the inner
loop, and where the roadmap's own asymptotic table already puts it at **82% of a
``library="glm"`` fit's per-row cost**, eleven times the Newton solver that answers the
same question.

That makes it the strongest *algorithmic* candidate here in the way the multiplier
bootstrap is the strongest *memory* one, and it is worth being clear about which question
each arm answers:

* ``numba`` fuses the passes into one loop over rows per step and keeps the whole walk
  inside the compiled function, so the per-step interpreter cost goes to zero.  This is
  the honest "what does compiling buy" number.
* ``numba_parallel`` puts a ``prange`` over rows inside each step.  The steps themselves
  are strictly sequential -- each reads the fit the last one produced -- so rows are the
  only axis, and each step ends in a reduction across threads.
* ``numpy`` is the shipped path, transcribed.

**The stopping rule has to be identical, and that is the trap.**  Halve the step on an
overshoot, accept otherwise, stop at ``relative_score <= tol`` or ``max_steps``.  A
compiled kernel that converged in a different number of steps would be measured against a
different amount of work, so the validator compares ``n_iter`` and ``converged`` exactly
(within one step, per :func:`~..validation.compare_solver`) and not only the coefficients.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import Regime, make_targeting
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_solver
from . import KernelSpec, register

__all__ = [
    "build",
    "numba_one_step",
    "numba_one_step_deferred",
    "numba_one_step_parallel",
    "numpy_one_step",
    "numpy_one_step_deferred",
]

#: The package's own shrink bound, so a step here moves the fit as far as a step there.
_ALPHA = 0.9995


def build(
    n: int = 100_000,
    n_arms: int = 2,
    regime: Regime = "moderate",
    step_size: float = 1e-3,
    max_steps: int = 2000,
    tol: float = 1e-10,
    seed: int = 20260803,
) -> dict[str, Any]:
    """A fit ready to be walked.

    ``max_steps`` defaults well below the package's 20,000 so that one measurement is
    seconds rather than minutes; the *cost per step* is what scales, and the runner sweeps
    the cap to show the walk length is the dimension it looks like.
    """
    fixture = make_targeting(n, n_arms=n_arms, regime=regime, seed=seed)
    return {
        "fixture": fixture,
        "step_size": step_size,
        "max_steps": max_steps,
        "tol": tol,
    }


def _shrink(values: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(values, 1.0 - alpha, alpha)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------- numpy


def _score_columns(y, q, h, weights, mask):
    residual = np.zeros_like(y)
    residual[mask] = y[mask] - q[mask]
    return ((weights * residual)[:, None] * h).mean(axis=0)


def _score_scale(h, weights, mask):
    contribution = np.zeros_like(h)
    contribution[mask] = np.abs(weights[mask])[:, None] * np.abs(h[mask])
    return contribution.mean(axis=0)


def _relative(score, scale):
    if score.size == 0:
        return 0.0
    return float(np.max(np.abs(score) / np.maximum(scale, 1e-300)))


def numpy_one_step(inputs: dict[str, Any], *, defer_arms: bool = False) -> dict[str, Any]:
    """The shipped walk, transcribed from :mod:`cleverly.fluctuation.one_step`.

    Transcribed rather than called for the reason the multiplier reference is: the package
    function also builds a :class:`~cleverly.fluctuation.Fluctuation`, records a trace,
    diagnoses a failure mode and may warn, none of which a compiled kernel would do and
    none of which is the arithmetic under test.

    ``defer_arms`` is the algorithmic arm, and it is here because the profile pointed at
    it rather than at the interpreter.  The walk's score reads the fit at the *observed*
    treatment only; the ``K`` counterfactual arms are updated every trial step and never
    read until the walk ends.  Since ``logit`` is additive along the submodel, updating
    them once at the accumulated ``epsilon`` gives the same array -- measured at
    ``2.6e-15`` after fifty steps -- for ``1/(K+1)`` of the transcendental work.  On this
    fixture that is 8.5 ms a step against 5.1 ms.

    **It is the same array only while the shrink bound does not bind.**  ``shrunk(alpha)``
    is applied after each step, so once an arm's prediction is pinned at ``alpha`` the
    incremental path clamps repeatedly and the deferred path clamps once.  That is a real
    divergence under a severe positivity regime, and the run reports it rather than the
    benchmark hiding it: the correctness gate is against the shipped path, and this arm
    failing it at ``regime="severe"`` is the finding, not a bug in the harness.
    """
    fixture = inputs["fixture"]
    y = fixture.outcome
    weights = fixture.weights
    mask = fixture.observed
    h = fixture.covariate_observed
    h_arms = fixture.covariate_arms
    tol = inputs["tol"]
    max_steps = inputs["max_steps"]

    observed = _shrink(fixture.initial_observed, _ALPHA)
    arms = _shrink(fixture.initial_arms, _ALPHA)
    epsilon = np.zeros(h.shape[1])
    dx = float(inputs["step_size"])
    scale = _score_scale(h, weights, mask)

    score = _score_columns(y, observed, h, weights, mask)
    norm = float(np.linalg.norm(score))
    steps = 0

    while steps < max_steps and _relative(score, scale) > tol:
        if norm == 0.0:
            break
        direction = score / norm
        move = dx * direction
        candidate_observed = _shrink(_expit(_logit(observed) + h @ move), _ALPHA)
        if not defer_arms:
            candidate_arms = _shrink(
                _expit(_logit(arms) + np.einsum("nkp,p->nk", h_arms, move)), _ALPHA
            )
        candidate_score = _score_columns(y, candidate_observed, h, weights, mask)
        candidate_norm = float(np.linalg.norm(candidate_score))
        if candidate_norm > norm:
            dx *= 0.5
            if dx < 1e-14:
                break
            continue
        epsilon = epsilon + move
        observed = candidate_observed
        if not defer_arms:
            arms = candidate_arms
        score, norm = candidate_score, candidate_norm
        steps += 1

    if defer_arms:
        arms = _shrink(
            _expit(_logit(arms) + np.einsum("nkp,p->nk", h_arms, epsilon)), _ALPHA
        )
    relative = _relative(score, scale)
    return {
        "epsilon": epsilon,
        "targeted_observed": observed,
        "targeted_arms": arms,
        "score": score,
        "n_iter": steps,
        "converged": bool(relative <= tol),
    }


# --------------------------------------------------------------------------- numba


@njit(inline="always")
def _move_arm(h_arms, arms, move, i, k, columns):
    """One arm's prediction after a move along the submodel, for one row.

    Written once and shared by the serial and the parallel step so the two cannot drift:
    the pair differ in how they walk the rows, and nothing else about them should differ
    at all.
    """
    shift = 0.0
    for j in range(columns):
        shift += h_arms[i, k, j] * move[j]
    p = arms[i, k]
    eta = np.log(p / (1.0 - p)) + shift
    if eta > 700.0:
        eta = 700.0
    elif eta < -700.0:
        eta = -700.0
    q = 1.0 / (1.0 + np.exp(-eta))
    if q > _ALPHA:
        return _ALPHA
    if q < 1.0 - _ALPHA:
        return 1.0 - _ALPHA
    return q


@njit()
def _apply_arms(h_arms, arms, epsilon, out_arms):
    """The deferred arm update: one pass at the accumulated ``epsilon``."""
    for i in range(arms.shape[0]):
        for k in range(arms.shape[1]):
            out_arms[i, k] = _move_arm(h_arms, arms, epsilon, i, k, epsilon.shape[0])


@njit(inline="always")
def _fused_step_serial(
    y, weights, mask, h, h_arms, observed, arms, move, out_observed, out_arms, score,
    update_arms,
):
    """One trial step: apply the move and take the resulting score, in a single row pass.

    The fusion is the point.  numpy needs one pass to ``logit``, one to multiply, one to
    ``expit``, one to clip and one more to form the residual, each allocating; here a row
    is loaded once, walked through all of that in registers, and its contribution added to
    the score accumulator before the next row is touched.
    """
    rows = y.shape[0]
    columns = h.shape[1]
    n_arms = arms.shape[1]
    for j in range(columns):
        score[j] = 0.0
    for i in range(rows):
        shift = 0.0
        for j in range(columns):
            shift += h[i, j] * move[j]
        p = observed[i]
        eta = np.log(p / (1.0 - p)) + shift
        if eta > 700.0:
            eta = 700.0
        elif eta < -700.0:
            eta = -700.0
        q = 1.0 / (1.0 + np.exp(-eta))
        if q > _ALPHA:
            q = _ALPHA
        elif q < 1.0 - _ALPHA:
            q = 1.0 - _ALPHA
        out_observed[i] = q
        if update_arms:
            for k in range(n_arms):
                out_arms[i, k] = _move_arm(h_arms, arms, move, i, k, columns)
        if mask[i]:
            residual = weights[i] * (y[i] - q)
            for j in range(columns):
                score[j] += residual * h[i, j]
    for j in range(columns):
        score[j] /= rows


@pjit(inline="always")
def _fused_step_parallel(
    y, weights, mask, h, h_arms, observed, arms, move, out_observed, out_arms, score, partial,
    update_arms,
):
    """The same step with a ``prange`` over rows and a thread-local score reduction.

    Rows are the only axis available: the walk is sequential in ``t`` by construction --
    each step's direction is the score at the fit the previous step produced -- so
    parallelising over steps would not be an optimisation of this algorithm but a
    different algorithm.
    """
    rows = y.shape[0]
    columns = h.shape[1]
    n_arms = arms.shape[1]
    n_threads = partial.shape[0]
    span = (rows + n_threads - 1) // n_threads
    for t in prange(n_threads):
        start = t * span
        stop = min(rows, start + span)
        for j in range(columns):
            partial[t, j] = 0.0
        for i in range(start, stop):
            shift = 0.0
            for j in range(columns):
                shift += h[i, j] * move[j]
            p = observed[i]
            eta = np.log(p / (1.0 - p)) + shift
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            q = 1.0 / (1.0 + np.exp(-eta))
            if q > _ALPHA:
                q = _ALPHA
            elif q < 1.0 - _ALPHA:
                q = 1.0 - _ALPHA
            out_observed[i] = q
            if update_arms:
                for k in range(n_arms):
                    out_arms[i, k] = _move_arm(h_arms, arms, move, i, k, columns)
            if mask[i]:
                residual = weights[i] * (y[i] - q)
                for j in range(columns):
                    partial[t, j] += residual * h[i, j]
    for j in range(columns):
        total = 0.0
        for t in range(n_threads):
            total += partial[t, j]
        score[j] = total / rows


def _make_walk(step_kernel, decorator, parallel: bool):
    """Build the walk around a step kernel.

    The whole walk lives inside one compiled function: a Python loop calling a compiled
    step would pay the dispatch and the argument unboxing per step, which at a thousand
    steps is most of what compiling was meant to remove.
    """

    @decorator()
    def walk(
        y, weights, mask, h, h_arms, observed, arms, step_size, max_steps, tol, n_threads,
        defer_arms,
    ):
        rows = y.shape[0]
        columns = h.shape[1]
        n_arms = arms.shape[1]

        scale = np.zeros(columns)
        for i in range(rows):
            if mask[i]:
                for j in range(columns):
                    scale[j] += abs(weights[i]) * abs(h[i, j])
        for j in range(columns):
            scale[j] /= rows

        current_observed = np.empty(rows)
        current_arms = np.empty((rows, n_arms))
        for i in range(rows):
            value = observed[i]
            current_observed[i] = min(max(value, 1.0 - _ALPHA), _ALPHA)
            for k in range(n_arms):
                arm_value = arms[i, k]
                current_arms[i, k] = min(max(arm_value, 1.0 - _ALPHA), _ALPHA)

        score = np.zeros(columns)
        for i in range(rows):
            if mask[i]:
                residual = weights[i] * (y[i] - current_observed[i])
                for j in range(columns):
                    score[j] += residual * h[i, j]
        for j in range(columns):
            score[j] /= rows

        trial_observed = np.empty(rows)
        trial_arms = np.empty((rows, n_arms))
        trial_score = np.zeros(columns)
        partial = np.zeros((n_threads, columns))
        epsilon = np.zeros(columns)
        move = np.zeros(columns)

        norm = 0.0
        for j in range(columns):
            norm += score[j] * score[j]
        norm = np.sqrt(norm)

        dx = step_size
        steps = 0
        while steps < max_steps:
            relative = 0.0
            for j in range(columns):
                candidate = abs(score[j]) / max(scale[j], 1e-300)
                if candidate > relative:
                    relative = candidate
            if relative <= tol or norm == 0.0:
                break
            for j in range(columns):
                move[j] = dx * score[j] / norm
            if parallel:
                _fused_step_parallel(
                    y, weights, mask, h, h_arms, current_observed, current_arms,
                    move, trial_observed, trial_arms, trial_score, partial,
                    not defer_arms,
                )
            else:
                _fused_step_serial(
                    y, weights, mask, h, h_arms, current_observed, current_arms,
                    move, trial_observed, trial_arms, trial_score,
                    not defer_arms,
                )
            trial_norm = 0.0
            for j in range(columns):
                trial_norm += trial_score[j] * trial_score[j]
            trial_norm = np.sqrt(trial_norm)
            if trial_norm > norm:
                dx *= 0.5
                if dx < 1e-14:
                    break
                continue
            for j in range(columns):
                epsilon[j] += move[j]
                score[j] = trial_score[j]
            norm = trial_norm
            for i in range(rows):
                current_observed[i] = trial_observed[i]
            if not defer_arms:
                for i in range(rows):
                    for k in range(n_arms):
                        current_arms[i, k] = trial_arms[i, k]
            steps += 1

        if defer_arms:
            _apply_arms(h_arms, current_arms, epsilon, current_arms)

        relative = 0.0
        for j in range(columns):
            candidate = abs(score[j]) / max(scale[j], 1e-300)
            if candidate > relative:
                relative = candidate
        return epsilon, current_observed, current_arms, score, steps, relative

    return walk


_WALK_SERIAL = _make_walk(_fused_step_serial, njit, parallel=False)
_WALK_PARALLEL = _make_walk(_fused_step_parallel, pjit, parallel=True)


def _run(
    inputs: dict[str, Any], walk: Any, n_threads: int, *, defer_arms: bool = False
) -> dict[str, Any]:
    fixture = inputs["fixture"]
    epsilon, observed, arms, score, steps, relative = walk(
        fixture.outcome,
        fixture.weights,
        fixture.observed,
        np.ascontiguousarray(fixture.covariate_observed),
        np.ascontiguousarray(fixture.covariate_arms),
        fixture.initial_observed,
        fixture.initial_arms,
        float(inputs["step_size"]),
        int(inputs["max_steps"]),
        float(inputs["tol"]),
        int(n_threads),
        bool(defer_arms),
    )
    return {
        "epsilon": epsilon,
        "targeted_observed": observed,
        "targeted_arms": arms,
        "score": score,
        "n_iter": int(steps),
        "converged": bool(relative <= inputs["tol"]),
    }


def numpy_one_step_deferred(inputs: dict[str, Any]) -> dict[str, Any]:
    """numpy, with the counterfactual arms updated once at the end rather than per step."""
    return numpy_one_step(inputs, defer_arms=True)


def numba_one_step(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _WALK_SERIAL, 1)


def numba_one_step_deferred(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _WALK_SERIAL, 1, defer_arms=True)


def numba_one_step_parallel(inputs: dict[str, Any]) -> dict[str, Any]:
    from ..implementations.numba_parallel import effective_threads

    return _run(inputs, _WALK_PARALLEL, effective_threads())


def numba_one_step_parallel_deferred(inputs: dict[str, Any]) -> dict[str, Any]:
    from ..implementations.numba_parallel import effective_threads

    return _run(inputs, _WALK_PARALLEL, effective_threads(), defer_arms=True)


_IMPLEMENTATIONS: dict[str, Any] = {
    "numpy": numpy_one_step,
    "numpy_deferred_arms": numpy_one_step_deferred,
}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_one_step
    _IMPLEMENTATIONS["numba_deferred_arms"] = numba_one_step_deferred
    _IMPLEMENTATIONS["numba_parallel"] = numba_one_step_parallel
    _IMPLEMENTATIONS["numba_parallel_deferred_arms"] = numba_one_step_parallel_deferred

register(
    KernelSpec(
        name="one_step_walk",
        estimator="tmle",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=compare_solver,
        # A walk of a thousand steps accumulates its rounding; what has to agree tightly
        # is the *fit*, and the coefficient is a sum of a thousand increments.
        tolerance=(1e-9, 1e-9),
        parallel_axis="rows",
        note="tens of full-length numpy passes per step, in a Python loop, for up to 20,000 steps",
        dimensions={
            "n": 100_000,
            "n_arms": 2,
            "regime": "moderate",
            "step_size": 1e-3,
            "max_steps": 2000,
            "tol": 1e-10,
            "seed": 20260803,
        },
        amortise=True,
    )
)

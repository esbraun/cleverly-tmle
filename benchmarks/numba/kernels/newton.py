r"""The negative controls: kernels measured in order to be rejected.

Two of them, and both are here because a reader would otherwise reasonably assume they had
been overlooked.

**The targeting Newton solve.**  ``benchmarks/bench_tmle.py`` already answers this and
finds a wash; it is re-run inside this harness so the answer is on the same axes as
everything else -- same statistics, same thread control, same correctness gate -- rather
than being quoted from a different instrument.  The reason it is a wash is structural and
worth restating, because it is also the reason several kernels here are *not* negative
controls: the loop's inner work is ``x @ epsilon``, ``x.T @ (w r)`` and ``x.T @ (x v)``,
which are BLAS calls, plus one vectorised ``expit``.  A compiler pays where the interpreter
is in the inner loop.  Here it is not; in the one-step walk it is.

**The MSM Gram contraction.**  ``solve_projection``'s Jacobian term is a four-operand
``einsum`` that reshapes into one ``dgemm``.  The package already fixed the real problem
there -- ``optimize=False`` made it fourteen times slower than the same arithmetic through
BLAS -- and what is left is a matrix product.  It is included to show that a kernel *whose
share is real* can still be the wrong candidate: the share is the reason to look, and the
shape is the reason to stop.

Keeping these in the suite is not decoration.  A benchmark that only reports its positive
results cannot be used to check that those results are not an artefact of the harness; if
the fused walk were fast because the reference was crippled, the Newton control would be
fast for the same reason and it is not.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import Regime, make_targeting
from ..implementations.numba_parallel import PARALLEL_AVAILABLE
from ..implementations.numba_serial import njit
from ..validation import compare_arrays, compare_solver
from . import KernelSpec, register

__all__ = ["build_gram", "build_newton", "numpy_gram", "numpy_newton", "numba_newton"]

_ALPHA = 0.9995


# --------------------------------------------------------------- the Newton solve


def build_newton(
    n: int = 100_000,
    n_arms: int = 2,
    regime: Regime = "moderate",
    max_iter: int = 50,
    tol: float = 1e-12,
    seed: int = 20260803,
) -> dict[str, Any]:
    fixture = make_targeting(n, n_arms=n_arms, regime=regime, seed=seed)
    initial = np.clip(fixture.initial_observed, 1.0 - _ALPHA, _ALPHA)
    return {
        "x": np.ascontiguousarray(fixture.covariate_observed),
        "y": fixture.outcome,
        "offset": np.log(initial / (1.0 - initial)),
        "weights": fixture.weights,
        "max_iter": max_iter,
        "tol": tol,
    }


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def numpy_newton(inputs: dict[str, Any]) -> dict[str, Any]:
    """The shipped weighted-logistic Newton with an offset, no intercept, no line search.

    The backtracking line search, the conditioning diagnostics and the failure
    classification are stripped out for the same reason ``bench_tmle.py`` strips them: they
    are the same work in both implementations and none of it is the arithmetic under test.
    """
    x, y, offset, weights = inputs["x"], inputs["y"], inputs["offset"], inputs["weights"]
    epsilon = np.zeros(x.shape[1])
    total_weight = weights.sum()
    converged = False
    iterations = 0
    for _ in range(inputs["max_iter"]):
        p = _expit(offset + x @ epsilon)
        gradient = x.T @ (weights * (y - p))
        if np.max(np.abs(gradient)) / total_weight <= inputs["tol"]:
            converged = True
            break
        variance = weights * p * (1.0 - p)
        hessian = x.T @ (x * variance[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        epsilon = epsilon + step
        iterations += 1
    return {"epsilon": epsilon, "n_iter": iterations, "converged": converged}


@njit()
def _newton(x, y, offset, weights, max_iter, tol):
    rows, columns = x.shape
    epsilon = np.zeros(columns)
    gradient = np.zeros(columns)
    hessian = np.zeros((columns, columns))
    total_weight = 0.0
    for i in range(rows):
        total_weight += weights[i]
    converged = False
    iterations = 0
    for _ in range(max_iter):
        for j in range(columns):
            gradient[j] = 0.0
            for k in range(columns):
                hessian[j, k] = 0.0
        for i in range(rows):
            eta = offset[i]
            for j in range(columns):
                eta += x[i, j] * epsilon[j]
            if eta > 700.0:
                eta = 700.0
            elif eta < -700.0:
                eta = -700.0
            p = 1.0 / (1.0 + np.exp(-eta))
            residual = weights[i] * (y[i] - p)
            variance = weights[i] * p * (1.0 - p)
            for j in range(columns):
                gradient[j] += residual * x[i, j]
                for k in range(columns):
                    hessian[j, k] += variance * x[i, j] * x[i, k]
        largest = 0.0
        for j in range(columns):
            if abs(gradient[j]) > largest:
                largest = abs(gradient[j])
        if largest / total_weight <= tol:
            converged = True
            break
        step = np.linalg.solve(hessian, gradient)
        for j in range(columns):
            epsilon[j] += step[j]
        iterations += 1
    return epsilon, iterations, converged


def numba_newton(inputs: dict[str, Any]) -> dict[str, Any]:
    epsilon, iterations, converged = _newton(
        inputs["x"], inputs["y"], inputs["offset"], inputs["weights"],
        int(inputs["max_iter"]), float(inputs["tol"]),
    )
    return {"epsilon": epsilon, "n_iter": int(iterations), "converged": bool(converged)}


# ------------------------------------------------------ the MSM Gram contraction


def build_gram(
    n: int = 100_000, n_arms: int = 4, n_terms: int = 4, seed: int = 20260803
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "design": np.ascontiguousarray(rng.standard_normal((n, n_arms, n_terms))),
        "h": np.ascontiguousarray(rng.random((n, n_arms)) + 0.5),
        "weights": np.ones(n),
    }


def numpy_gram(inputs: dict[str, Any]) -> np.ndarray:
    """``einsum('ijp,ijq,ij,i->pq', design, design, h, w)`` with ``optimize=True``.

    The package settled on ``optimize=True`` here after measuring the unoptimised spelling
    at fourteen times slower -- numpy's own nested-loop kernel rather than a pairwise
    contraction through BLAS.  What that leaves is a ``dgemm``, which is why this is a
    control.
    """
    design, h, weights = inputs["design"], inputs["h"], inputs["weights"]
    return np.asarray(
        np.einsum("ijp,ijq,ij,i->pq", design, design, h, weights, optimize=True)
    )


@njit()
def _gram(design, h, weights):
    rows, arms, terms = design.shape
    out = np.zeros((terms, terms))
    for i in range(rows):
        for j in range(arms):
            factor = h[i, j] * weights[i]
            for p in range(terms):
                value = design[i, j, p] * factor
                for q in range(terms):
                    out[p, q] += value * design[i, j, q]
    return out


def numba_gram(inputs: dict[str, Any]) -> np.ndarray:
    return _gram(inputs["design"], inputs["h"], inputs["weights"])


_NEWTON: dict[str, Any] = {"numpy": numpy_newton}
_GRAM: dict[str, Any] = {"numpy": numpy_gram}
if PARALLEL_AVAILABLE:
    _NEWTON["numba"] = numba_newton
    _GRAM["numba"] = numba_gram

register(
    KernelSpec(
        name="newton_targeting",
        estimator="tmle",
        build=build_newton,
        implementations=_NEWTON,
        compare=compare_solver,
        tolerance=(1e-10, 1e-8),
        parallel_axis=None,
        negative_control=True,
        note="already BLAS-bound; bench_tmle.py measured a wash and this reproduces it here",
        dimensions={
            "n": 100_000,
            "n_arms": 2,
            "regime": "moderate",
            "max_iter": 50,
            "tol": 1e-12,
            "seed": 20260803,
        },
    )
)

register(
    KernelSpec(
        name="msm_gram",
        estimator="msm",
        build=build_gram,
        implementations=_GRAM,
        compare=compare_arrays,
        tolerance=(1e-8, 1e-11),
        parallel_axis=None,
        negative_control=True,
        note="a four-operand einsum that reshapes into one dgemm once optimize=True",
        dimensions={"n": 100_000, "n_arms": 4, "n_terms": 4, "seed": 20260803},
    )
)

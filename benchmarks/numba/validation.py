"""Correctness gates.  No timing here is reported until its implementation passes one.

The comparison is deliberately **not** bitwise.  A ``prange`` reduction sums in a
different order from a serial one and a fused kernel evaluates ``expit(logit(q) + h e)``
without materialising the intermediate, so the last bits move; demanding equality would
reject every implementation worth having and teach the reader nothing.  What is demanded
is agreement to a tolerance the kernel declares, plus -- and this is the part a tolerance
alone does not give -- agreement on the *algorithmic* facts a float comparison cannot see:
the iteration count, the convergence flag, which candidate was selected, which rows were
masked.

:func:`compare_arrays` is the ordinary gate.  :func:`compare_solver` is the one for
anything that iterates, because two solvers can agree on the answer to ``1e-14`` while
one of them took four times as many steps to get there, and a benchmark that reports the
faster per-step cost of the slower solver is measuring the wrong thing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Verdict",
    "compare_arrays",
    "compare_mapping",
    "compare_scalar",
    "compare_solver",
    "check",
]


@dataclass(frozen=True)
class Verdict:
    """The outcome of one implementation's correctness gate."""

    correct: bool
    max_abs_error: float
    max_rel_error: float
    reason: str = ""

    def __bool__(self) -> bool:
        return self.correct


def _errors(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    a = np.asarray(reference, dtype=float).reshape(-1)
    b = np.asarray(candidate, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return float("inf"), float("inf")
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all():
        # A non-finite in one and not the other is a disagreement, not a rounding
        # difference, and must not be dropped by the mask that handles the rest.
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            return float("inf"), float("inf")
    a, b = a[finite], b[finite]
    if a.size == 0:
        return 0.0, 0.0
    absolute = np.abs(a - b)
    relative = absolute / np.maximum(np.abs(a), 1e-300)
    return float(absolute.max()), float(relative.max())


def compare_arrays(reference: Any, candidate: Any) -> tuple[float, float]:
    """Largest absolute and relative disagreement between two arrays."""
    return _errors(np.asarray(reference), np.asarray(candidate))


def compare_scalar(reference: Any, candidate: Any) -> tuple[float, float]:
    return _errors(np.atleast_1d(np.asarray(reference, dtype=float)),
                   np.atleast_1d(np.asarray(candidate, dtype=float)))


def compare_mapping(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[float, float]:
    """Worst disagreement across a dict of named arrays.

    A missing or extra key is infinite error rather than a skipped comparison: an
    implementation that computes four of five estimands and agrees on the four is not
    correct, and averaging over the keys it happens to have is how that gets missed.
    """
    if set(reference) != set(candidate):
        return float("inf"), float("inf")
    worst = (0.0, 0.0)
    for key in reference:
        errors = _errors(np.asarray(reference[key]), np.asarray(candidate[key]))
        worst = (max(worst[0], errors[0]), max(worst[1], errors[1]))
    return worst


def compare_solver(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[float, float]:
    """Numerical *and* algorithmic agreement for anything that iterates.

    The numeric fields are compared as usual.  The fields that are not numbers --
    ``converged``, ``n_iter``, ``failure`` -- are compared exactly, and a disagreement
    returns infinite error, because a solver that stopped for a different reason has not
    computed the same thing however close its answer looks.

    ``n_iter`` is allowed to differ by one.  Two implementations of the same walk can
    cross the tolerance on either side of a single step when the last increment is at the
    rounding scale, and rejecting that would reject a correct kernel for a reason that has
    nothing to do with it.  Anything wider is a different stopping rule.
    """
    if set(reference) != set(candidate):
        return float("inf"), float("inf")
    for key in ("converged", "failure", "selected", "method"):
        if key in reference and reference[key] != candidate[key]:
            return float("inf"), float("inf")
    if "n_iter" in reference and abs(int(reference["n_iter"]) - int(candidate["n_iter"])) > 1:
        return float("inf"), float("inf")
    numeric = {
        key: value
        for key, value in reference.items()
        if key not in ("converged", "failure", "selected", "method", "n_iter")
    }
    return compare_mapping(numeric, {key: candidate[key] for key in numeric})


def check(
    reference: Any,
    candidate: Any,
    *,
    compare: Any,
    tolerance: tuple[float, float],
) -> Verdict:
    """Run a kernel's comparison and turn it into a pass or a stated failure."""
    atol, rtol = tolerance
    try:
        absolute, relative = compare(reference, candidate)
    except Exception as error:  # noqa: BLE001 - the reason is the useful output
        return Verdict(False, float("inf"), float("inf"), f"{type(error).__name__}: {error}")
    ok = bool(absolute <= atol or relative <= rtol)
    reason = (
        ""
        if ok
        else f"max_abs={absolute:.3g} > atol={atol:g} and max_rel={relative:.3g} > rtol={rtol:g}"
    )
    return Verdict(ok, absolute, relative, reason)

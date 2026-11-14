"""A held-treatment law that probability weights realise exactly, for the Markov-order tests.

One baseline decision ``A`` is held over three nodes.  Every node is binary, in the order
``W, L1, A, L2, C2, L3, C3, Y``: ``L_t`` reads ``W, L_{t-1}, A``, the retention ``C_t`` at
nodes 2 and 3 reads ``W, L_t, A``, and ``Y`` reads ``W, L3, A``.  So every nuisance reads
``W``, the node's own covariate and the held arm: the order-0 restriction holds, *with* the
arm in each window, since every node's arm is ``A``.

One row per support point carries its probability as a weight (``NaN`` after a unit leaves),
so a weighted saturated fit is the oracle.  Both halves of a window that keeps the arm are
first-order here.  Retention reads ``A``, so a censoring design without it is wrong for every
regimen.  A rule or a policy of ``L1`` at node 1 leaves both arms among the followers, so an
outcome design without it is wrong there.  A static plan's outcome design needs no arm.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd

from cleverly import LongitudinalTreatment

_LEVELS = (0.2, 0.45, 0.75)


def _q(*bits: int) -> float:
    """A conditional probability fixed by the bits it reads; flipping any bit moves it."""
    return _LEVELS[(sum(b * (1 + i % 2) for i, b in enumerate(bits)) + len(bits)) % 3]


def _bern(p: float, x: int) -> float:
    return p if x else 1.0 - p


def _l(t: int, w: int, l_prev: int, a: int) -> float:
    return _q(w, l_prev, a, t)


def _a(w: int, l1: int) -> float:
    return _q(w, l1, 1)


def _retain(t: int, w: int, l_now: int, a: int) -> float:
    return 0.6 + 0.3 * _q(a, l_now, w, t)


def _y(w: int, l3: int, a: int) -> float:
    return _q(w, l3, a, 0)


def frame() -> pd.DataFrame:
    """One weighted row per support point; the weights sum to one."""
    rows = []
    for w, l1, a, l2, l3, y in itertools.product((0, 1), repeat=6):
        head = 0.5 * _bern(_l(1, w, 0, 0), l1) * _bern(_a(w, l1), a) * _bern(_l(2, w, l1, a), l2)
        r2, r3 = _retain(2, w, l2, a), _retain(3, w, l3, a)
        tail = _bern(_l(3, w, l2, a), l3)
        point = {"W": w, "L1": l1, "A": a, "L2": l2, "L3": l3, "C1": 1}
        rows.append(
            {
                **point,
                "C2": 1,
                "C3": 1,
                "Y": y,
                "wt": head * r2 * tail * r3 * _bern(_y(w, l3, a), y),
            }
        )
        if y == 0:  # one row for every (W, L1, A, L2, L3) that leaves at node 3
            rows.append({**point, "C2": 1, "C3": 0, "Y": np.nan, "wt": head * r2 * tail * (1 - r3)})
        if y == 0 and l3 == 0:  # one row for every (W, L1, A, L2) that leaves at node 2
            rows.append(
                {**point, "L3": np.nan, "C2": 0, "C3": 0, "Y": np.nan, "wt": head * (1 - r2)}
            )
    out = pd.DataFrame(rows)
    assert abs(out["wt"].sum() - 1.0) < 1e-12
    return out


def design(order: int | None) -> LongitudinalTreatment:
    """The held design of :func:`frame`, under ``markov_order=order``."""
    return LongitudinalTreatment(
        outcome="Y",
        treatment="A",
        baseline=["W"],
        time_varying=[["L1"], ["L2"], ["L3"]],
        censoring=["C1", "C2", "C3"],
        weights="wt",
        markov_order=order,
    )


def truth(treat: Callable[[int, int], float]) -> float:
    """``E[Y^d]`` with no censoring, when ``d`` treats with probability ``treat(W, L1)``."""
    total = 0.0
    for w, l1, a, l2, l3 in itertools.product((0, 1), repeat=5):
        weight = 0.5 * _bern(_l(1, w, 0, 0), l1) * _bern(treat(w, l1), a)
        weight *= _bern(_l(2, w, l1, a), l2) * _bern(_l(3, w, l2, a), l3)
        total += weight * _y(w, l3, a)
    return total

"""Three-node laws that a finite sample realises exactly, for the Markov-order tests.

Every node is binary, in the order ``W, L1, A1, L2, A2, L3, A3, Y``.  Every conditional
probability is a multiple of ``1/4``, so ``N = 2 * 4**7`` rows laid out in the cell
proportions make the empirical law equal to the law.  At ``P_n = P_0`` a TMLE whose
mechanisms are right returns the truth whatever its outcome regressions, and a TMLE whose
outcome regressions are right returns the truth whatever its mechanisms.

Two laws differ in what each conditional reads.

| law | ``L_t`` reads | ``A_t`` reads | ``Y`` reads |
| --- | --- | --- | --- |
| ``"markov1"`` | ``W, L_{t-1}, A_{t-1}`` | ``W, L_{t-1}, L_t`` | ``W, L2, L3, A3`` |
| ``"one_family"`` | ``W, L_{t-1}, A_{t-1}`` | ``W, A_{t-1}, L_t`` | ``W, L1, L3, A3`` |

Under ``"markov1"`` every nuisance reads the order-1 window, and the mechanisms and the
outcome regressions both read a column outside the order-0 window.  Under ``"one_family"``
the mechanisms read the order-0 window and the outcome regressions read ``L1``, outside it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

#: Rows in the realised sample: one factor of two for ``W`` and seven of four.
N = 2 * 4**7

NODES = ("W", "L1", "A1", "L2", "A2", "L3", "A3", "Y")

_QUARTERS = (0.25, 0.5, 0.75)


def _q(*bits: int) -> float:
    """A conditional probability in quarters, fixed by the bits it reads."""
    # Each bit carries a weight of 1 or 2, never 0 mod 3, so flipping any bit moves the value.
    return _QUARTERS[(sum(b * (1 + i % 2) for i, b in enumerate(bits)) + len(bits)) % 3]


def _l_prob(w: int, l_prev: int, a_prev: int, t: int) -> float:
    return _q(w, l_prev, a_prev, t)


def _a_prob(law: str, w: int, l_prev: int, l_now: int, a_prev: int, t: int) -> float:
    if law == "markov1":
        return _q(w, l_prev, l_now, 1 - t % 2)
    return _q(w, a_prev, l_now, t % 2)


def _y_prob(law: str, point: dict[str, int]) -> float:
    early = point["L2"] if law == "markov1" else point["L1"]
    return _q(point["W"], early, point["L3"], point["A3"], 1)


def _bern(p: float, x: int) -> float:
    return p if x == 1 else 1.0 - p


def mass(law: str, point: dict[str, int]) -> float:
    """``P`` of one support point under ``law``."""
    total = 0.5
    l_prev, a_prev = 0, 0
    for t in (1, 2, 3):
        l_now, a_now = point[f"L{t}"], point[f"A{t}"]
        total *= _bern(_l_prob(point["W"], l_prev, a_prev, t), l_now)
        total *= _bern(_a_prob(law, point["W"], l_prev, l_now, a_prev, t), a_now)
        l_prev, a_prev = l_now, a_now
    return total * _bern(_y_prob(law, point), point["Y"])


def frame(law: str) -> pd.DataFrame:
    """The ``N``-row sample whose empirical law is ``law``."""
    rows: list[tuple[int, ...]] = []
    for bits in itertools.product((0, 1), repeat=len(NODES)):
        point = dict(zip(NODES, bits, strict=True))
        count = mass(law, point) * N
        whole = round(count)
        if abs(count - whole) > 1e-9:  # pragma: no cover - guards the constants
            raise AssertionError("keep every conditional a multiple of 1/4")
        rows.extend([bits] * whole)
    out = pd.DataFrame(np.array(rows, dtype=float), columns=list(NODES))
    assert len(out) == N
    return out


#: A rule reads the history frame ``[W, L_1, ..., L_t]``.
Rule = Callable[[Any], Any]


def truth(law: str, plan: tuple[int | Rule, int | Rule, int | Rule]) -> float:
    """``E[Y^d]`` by the g-formula over the support, for constant or rule nodes.

    A rule node is a callable of the dict ``{"W", "L1", ..., "L_t"}``.
    """
    total = 0.0
    for w, l1, l2, l3 in itertools.product((0, 1), repeat=4):
        history = {"W": w, "L1": l1, "L2": l2, "L3": l3}
        arms = []
        for t, node in enumerate(plan, start=1):
            seen = {k: v for k, v in history.items() if k == "W" or int(k[1]) <= t}
            arms.append(int(node(seen)) if callable(node) else int(node))
        weight = 0.5
        l_prev, a_prev = 0, 0
        for t, l_now in enumerate((l1, l2, l3), start=1):
            weight *= _bern(_l_prob(w, l_prev, a_prev, t), l_now)
            l_prev, a_prev = l_now, arms[t - 1]
        point = {"W": w, "L1": l1, "L2": l2, "L3": l3, "A3": arms[2]}
        total += weight * _y_prob(law, point)
    return total

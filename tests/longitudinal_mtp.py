"""The estimator side of :mod:`tests.discrete_law_longitudinal_mtp`.

The law module states each plan as maps of support indices and must not import the
library (``tests/unit/test_oracle_independence.py``).  This module declares the same plans
as policy objects, for a fit on a continuous, a categorical or a vector reading of the law.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from cleverly.interventions import (
    ModifiedPolicy,
    Piece,
    Randomizer,
    RiskRatioTilt,
    Scale,
    Shift,
)
from cleverly.learners import density as density_module
from cleverly.longitudinal import DynamicRegimen

from . import discrete_law_longitudinal_mtp as law

INF = np.inf


def _earlier(h: Any) -> np.ndarray:
    return np.asarray(h["A1"], dtype=float)


def _up_piecewise(top: float) -> ModifiedPolicy:
    """``a + 1`` below ``top``, the identity above: ``Shift(1, cap=top)`` written as pieces."""
    return ModifiedPolicy(
        "up",
        pieces=(
            Piece(-INF, top - 0.5, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
            Piece(top - 0.5, INF),
        ),
        policy_kind="known",
    )


def _boundary(h: Any) -> np.ndarray:
    return np.where(_earlier(h) >= 2.0, 2.5, -1.0)


HISTORY = ModifiedPolicy(
    "history",
    pieces=(
        Piece(-INF, _boundary, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
        Piece(_boundary, INF),
    ),
    policy_kind="known",
)

DECREASING = ModifiedPolicy(
    "decreasing",
    pieces=(Piece(-INF, INF, lambda a, h: 3.0 - a, lambda b, h: 3.0 - b, lambda b, h: -1.0),),
    policy_kind="known",
)

RANDOM = ModifiedPolicy(
    "random",
    pieces={
        "up": (
            Piece(-INF, 2.5, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
            Piece(2.5, INF),
        ),
        "stay": (Piece(-INF, INF),),
    },
    randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
    policy_kind="known",
)


def _vector_map(a: Any, h: Any) -> list[Any]:
    return [(value[0], 0) if tuple(value) == (1, 1) else tuple(value) for value in a]


VECTOR = ModifiedPolicy("vector", apply=_vector_map, policy_kind="known")

NATURAL = Shift(0.0, cap=None)
UP = Shift(1.0, cap=3.0)


def regimens(labels: tuple[str, ...]) -> dict[str, Any]:
    """The plans of :data:`tests.discrete_law_longitudinal_mtp.PLANS` on the unit grid."""
    table = {
        "natural": NATURAL,
        "up": UP,
        "history": DynamicRegimen("history", (NATURAL, HISTORY)),
        "decreasing": DynamicRegimen("decreasing", (UP, DECREASING)),
        "random": RANDOM,
        "up at 2": DynamicRegimen("up at 2", (NATURAL, UP)),
        "up then history": DynamicRegimen("up then history", (UP, HISTORY)),
        "vector": VECTOR,
    }
    return {label: table[label] for label in labels}


def scaled_regimens() -> dict[str, Any]:
    """The natural course and ``Scale(2, cap=8)`` at node 2 of the scaled law."""
    return {
        "natural": NATURAL,
        "up at 2": DynamicRegimen("up at 2", (NATURAL, Scale(2.0, cap=8.0))),
    }


@contextmanager
def exact_bins() -> Iterator[None]:
    """Make the binned density read the law's own bins, one support point per bin.

    The estimator reads bin edges from sample quantiles, which do not isolate the support
    points of a coarse law.  The grid is the law's, chosen by the dose values a node holds.
    """
    original = density_module.bin_edges

    def law_edges(values: Any, n_bins: int) -> np.ndarray:
        present = set(np.unique(np.asarray(values, dtype=float)).tolist())
        for grid in (law.UNIT_GRID, law.SCALED_GRID):
            if present <= set(grid.doses):
                return np.asarray(grid.edges, dtype=float)
        return original(values, n_bins)

    density_module.bin_edges = law_edges  # type: ignore[assignment]
    try:
        yield
    finally:
        density_module.bin_edges = original  # type: ignore[assignment]


def _treat_if_l2(a: Any, h: Any) -> Any:
    return np.where(np.asarray(h["L2"], dtype=float) == 1.0, 1, np.asarray(a))


#: The binary-node plan of the survival and competing-risk laws: a risk-ratio tilt of one half
#: at node 1, and "treat every unit with ``L2 = 1``" at node 2.  The law side is
#: :data:`tests.discrete_law_longitudinal_mtp.SURVIVAL_NODE1` and ``SURVIVAL_NODE2``.
SURVIVAL_PLAN = DynamicRegimen(
    "mtp",
    (
        RiskRatioTilt(0.5),
        ModifiedPolicy("treat if L2", apply=_treat_if_l2, policy_kind="known"),
    ),
)

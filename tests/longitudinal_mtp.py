"""The estimator side of :mod:`tests.discrete_law_longitudinal_mtp`.

The law module states each plan as maps of support indices and must not import the
library (``tests/unit/test_oracle_independence.py``).  This module declares the same plans
as policy objects, for a fit on a continuous, a categorical or a vector reading of the law.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from cleverly.interventions import (
    JointPolicy,
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
def exact_bins(grids: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Make the binned density read the law's own bins, one support point per bin.

    The estimator reads bin edges from sample quantiles, which do not isolate the support
    points of a coarse law.  ``grids`` maps a dose column to its law grid, and the density
    of that column reads it by name.  A dose that ``grids`` does not name reads the first
    law grid that holds its values, so a geometric grid is never displaced by the unit grid
    for a column it is declared for.
    """
    original = density_module.bin_edges
    declared = dict(grids or {})

    def law_edges(values: Any, n_bins: int, *, name: str | None = None) -> np.ndarray:
        if name is not None and name in declared:
            return np.asarray(declared[name].edges, dtype=float)
        present = set(np.unique(np.asarray(values, dtype=float)).tolist())
        for grid in (law.UNIT_GRID, law.SCALED_GRID):
            if present <= set(grid.doses):
                return np.asarray(grid.edges, dtype=float)
        return original(values, n_bins, name=name)

    # The modules that read edges from every at-risk row for a restricted fit (``fit_rows``)
    # import the function by name, so each of their references is replaced too.
    from cleverly.interventions import joint_density
    from cleverly.longitudinal import sequential

    readers = (density_module, joint_density, sequential)
    for module in readers:
        module.bin_edges = law_edges  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for module in readers:
            module.bin_edges = original  # type: ignore[attr-defined]


def _treat_if_l2(a: Any, h: Any) -> Any:
    return np.where(np.asarray(h["L2"], dtype=float) == 1.0, 1, np.asarray(a))


#: The binary-node plan of the survival and competing-risk laws: a risk-ratio tilt of 0.25
#: at node 1, and "treat every unit with ``L2 = 1``" at node 2.  The law side is
#: :data:`tests.discrete_law_longitudinal_mtp.SURVIVAL_NODE1` and ``SURVIVAL_NODE2``.
SURVIVAL_PLAN = DynamicRegimen(
    "mtp",
    (
        RiskRatioTilt(0.25),
        ModifiedPolicy("treat if L2", apply=_treat_if_l2, policy_kind="known"),
    ),
)


# ------------------------------------------------------------------ mixed vector nodes

#: The fit's reading of :func:`tests.discrete_law_longitudinal_mtp.mixed_frame`: two mixed
#: nodes, each a binary flag and a continuous dose, on the classifier route.
MIXED_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": [["F1", "D1"], ["F2", "D2"]],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["D1", "D2"],
}

#: The top of the mixed law's dose grid, which every dose policy caps at.
MIXED_TOP = 2.0


def _set_flag(a: Any, h: Any) -> np.ndarray:
    return np.ones(len(a))


def _flag_if_dosed(dose: str) -> Any:
    """A flag map that reads the natural dose: one where the dose is at least one."""

    def assign(a: Any, h: Any) -> np.ndarray:
        return np.where(np.asarray(h[dose], dtype=float) >= 1.0, 1, np.asarray(a))

    return assign


def _flagged_boundary(flag: str) -> Any:
    def boundary(h: Any) -> np.ndarray:
        return np.where(np.asarray(h[flag], dtype=float) == 1.0, MIXED_TOP - 0.5, -INF)

    return boundary


def _up_if_flagged(flag: str) -> ModifiedPolicy:
    """``d + 1`` where the natural flag is one and ``d`` is below the top, else ``d``."""
    boundary = _flagged_boundary(flag)
    return ModifiedPolicy(
        f"up if {flag}",
        pieces=(
            Piece(-INF, boundary, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
            Piece(boundary, INF),
        ),
        policy_kind="known",
    )


def _joint(label: str, time: int) -> JointPolicy:
    flag, dose = f"F{time}", f"D{time}"
    up = Shift(1.0, cap=MIXED_TOP)
    if label == "natural":
        return JointPolicy(label, components={}, policy_kind="known")
    if label == "dose up":
        return JointPolicy(label, components={dose: up}, policy_kind="known")
    if label == "flag gated":
        return JointPolicy(
            label,
            components={
                flag: ModifiedPolicy("flag", apply=_set_flag, policy_kind="known"),
                dose: _up_if_flagged(flag),
            },
            policy_kind="known",
        )
    if label == "dose gated":
        return JointPolicy(label, components={dose: _up_if_flagged(flag)}, policy_kind="known")
    if label == "flag set, dose up":
        # W17: the truth is the "flag gated (shifted flag)" plan, with the top dose held by
        # the cap, an identity piece that the joint density reads explicitly.
        return JointPolicy(
            label,
            components={
                flag: ModifiedPolicy("flag", apply=_set_flag, policy_kind="known"),
                dose: up,
            },
            policy_kind="known",
        )
    if label == "flag reads dose":
        return JointPolicy(
            label,
            components={
                flag: ModifiedPolicy("flag", apply=_flag_if_dosed(dose), policy_kind="known"),
                dose: up,
            },
            policy_kind="known",
        )
    if label == "random":
        return JointPolicy(
            label,
            components={"up": {dose: up}, "stay": {}},
            randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
            policy_kind="known",
        )
    raise KeyError(label)


def mixed_regimens(labels: tuple[str, ...]) -> dict[str, Any]:
    """The plans of :data:`tests.discrete_law_longitudinal_mtp.MIXED_PLANS` as joint policies."""
    return {label: DynamicRegimen(label, (_joint(label, 1), _joint(label, 2))) for label in labels}

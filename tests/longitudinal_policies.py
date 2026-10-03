"""The estimator side of :mod:`tests.discrete_law_longitudinal_policy`.

The oracle module states the policies as tables and the parameter as a g-formula, and it
imports nothing from ``cleverly``.  This module turns the same tables into what ``LTMLE`` is
handed: :class:`~cleverly.interventions.Stochastic` nodes inside
:class:`~cleverly.longitudinal.DynamicRegimen` plans.  The end-of-study policies return a
dataframe keyed by label, so the data layer's sorted level order and the tables' raw order
cannot be confused silently.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from cleverly.interventions import Stochastic
from cleverly.longitudinal import DynamicRegimen

from .discrete_law_longitudinal_policy import (
    _CODE,
    ARM_LABELS,
    PLANS,
    SURVIVAL_Q1,
    SURVIVAL_Q2,
    _draw,
)

__all__ = [
    "policy_node",
    "recorded_regimen",
    "regimen",
    "regimens",
    "survival_regimen",
]


def _node_one_density(table: np.ndarray) -> Callable[[Any], pd.DataFrame]:
    def density(frame: Any) -> pd.DataFrame:
        w = np.asarray(frame["W"], dtype=int)
        return pd.DataFrame(table[w], columns=list(ARM_LABELS))

    return density


def _node_two_density(table: np.ndarray) -> Callable[[Any], pd.DataFrame]:
    def density(frame: Any) -> pd.DataFrame:
        w = np.asarray(frame["W"], dtype=int)
        l2 = np.asarray(frame["L2"], dtype=int)
        a1 = np.array([_CODE[label] for label in frame["A1"]], dtype=int)
        return pd.DataFrame(table[w, a1, l2], columns=list(ARM_LABELS))

    return density


def policy_node(table: np.ndarray, node: int, name: str) -> Stochastic:
    """The estimator side of one policy table: a known :class:`Stochastic` node."""
    density = _node_one_density(table) if node == 1 else _node_two_density(table)
    return Stochastic(density, name, density_kind="known")


def _rule_two(frame: Any) -> Any:
    return np.where(np.asarray(frame["L2"]) == 1, "high", "low")


def regimen(label: str) -> Any:
    """The estimator side of :data:`PLANS`: what ``LTMLE`` is handed as ``regimens=``."""
    nodes: list[Any] = []
    has_rule = False
    for node, (kind, value) in enumerate(PLANS[label], start=1):
        if kind == "label":
            nodes.append(ARM_LABELS[value])
        elif kind == "rule":
            nodes.append(_rule_two)
            has_rule = True
        else:
            nodes.append(policy_node(value, node, f"{label}{node}"))
    if not any(kind == "policy" for kind, _ in PLANS[label]):
        return ARM_LABELS[PLANS[label][0][1]]
    return DynamicRegimen(label, tuple(nodes), rule_kind="known" if has_rule else None)


def regimens(labels: tuple[str, ...] = tuple(PLANS)) -> dict[str, Any]:
    """``regimens=`` for a fit over ``labels``, the reference first."""
    return {label: regimen(label) for label in labels}


def recorded_regimen(label: str = "mix") -> DynamicRegimen:
    """The rule regimen that reads the recorded randomizer: Theorem 3 as stated.

    A rule frame holds no earlier treatment, so the node-2 rule recomputes ``A1`` from ``W``
    and ``E1``.  On the rows that follow the rule that is the arm the unit received.
    """
    first_table = PLANS[label][0][1]
    second_table = PLANS[label][1][1]

    def first(frame: Any) -> Any:
        w = np.asarray(frame["W"], dtype=int)
        e1 = np.asarray(frame["E1"], dtype=int)
        return np.array([ARM_LABELS[_draw(first_table[a], b)] for a, b in zip(w, e1, strict=True)])

    def second(frame: Any) -> Any:
        w = np.asarray(frame["W"], dtype=int)
        e1 = np.asarray(frame["E1"], dtype=int)
        e2 = np.asarray(frame["E2"], dtype=int)
        l2 = np.asarray(frame["L2"], dtype=int)
        out = []
        for wi, e1i, e2i, l2i in zip(w, e1, e2, l2, strict=True):
            a1 = _draw(first_table[wi], e1i)
            out.append(ARM_LABELS[_draw(second_table[wi, a1, l2i], e2i)])
        return np.array(out)

    return DynamicRegimen(f"{label} (recorded)", (first, second), rule_kind="known")


def survival_regimen(label: str = "draw") -> DynamicRegimen:
    """The estimator side of the survival policy, as binary level columns ``(0, 1)``."""

    def first(frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=int)
        p = SURVIVAL_Q1[w]
        return np.column_stack([1.0 - p, p])

    def second(frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=int)
        a1 = np.asarray(frame["A1"], dtype=float).astype(int)
        l2 = np.asarray(frame["L2"], dtype=int)
        p = SURVIVAL_Q2[w, a1, l2]
        return np.column_stack([1.0 - p, p])

    return DynamicRegimen(
        label,
        (
            Stochastic(first, "s1", density_kind="known"),
            Stochastic(second, "s2", density_kind="known"),
        ),
    )

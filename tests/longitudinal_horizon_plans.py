"""The estimator side of the X39 plans: the plans of :mod:`tests.discrete_law_longitudinal_horizon`
written as the regimens ``LTMLE`` and ``SDR`` take.

The oracle states each plan as a density over its history tree, and this module states the same
plan as labels, :class:`~cleverly.longitudinal.DynamicRegimen` rules and known
:class:`~cleverly.interventions.Stochastic` nodes over data frames.  Writing a plan twice, in two
representations, is deliberate: a slip in one is a wrong number, not a mistake that cancels on both
sides of an exact witness.

A rule node reads the history frame ``[W, L_1, ..., L_t]``.  A policy node reads the policy frame,
which also holds the earlier arms ``A_1, ..., A_{t-1}``.  Each policy density is one table lookup
per node, vectorized over rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from cleverly.interventions import Stochastic
from cleverly.longitudinal import DynamicRegimen

#: The plans whose density is not an indicator at some node.
POLICY_PLANS = frozenset(
    {"mix", "coin", "mix_noprev", "mixlast", "coinlast", "taper", "uniform", "mix_node1"}
    | {"mix_norot", "mix_reversed"}
)


def _column(frame: Any, name: str) -> np.ndarray:
    return np.asarray(frame[name], dtype=float)


def _density(frame: Any, rows: np.ndarray, m: int) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(range(m)), index=getattr(frame, "index", None))


def _mix_rows(t: int, m: int, frame: Any, *, previous: bool = True) -> np.ndarray:
    n = len(frame)
    if m == 2:
        p = np.full(n, 0.5) if t == 1 else np.where(_column(frame, f"L{t}") == 1.0, 0.75, 0.5)
        return np.column_stack([1.0 - p, p])
    if t == 1:
        return np.tile([0.25, 0.25, 0.5], (n, 1))
    base = np.array([0.5, 0.25, 0.25])
    table = np.stack([np.roll(base, k) for k in range(3)])
    shift = _column(frame, f"L{t}").astype(int)
    if previous:
        shift = shift + _column(frame, f"A{t - 1}").astype(int)
    return np.asarray(table[shift % 3])


def _policy(name: str, t: int, K: int, m: int, every: set[int]) -> Stochastic:
    """The known density of policy plan ``name`` at node ``t``."""

    def density(frame: Any) -> pd.DataFrame:
        n = len(frame)
        if name in ("mix", "coin", "mixlast", "coinlast", "taper"):
            rows = _mix_rows(t, m, frame)
        elif name == "mix_noprev":
            rows = _mix_rows(t, m, frame, previous=t not in every)
        elif name == "uniform":
            rows = np.full((n, m), 1.0 / m)
        elif name == "mix_node1":
            rows = _mix_rows(1, m, frame)
        elif name == "mix_norot":
            if t == 1:
                rows = _mix_rows(1, m, frame)
            else:
                rows = np.tile([0.5, 0.5] if m == 2 else [0.5, 0.25, 0.25], (n, 1))
        elif name == "mix_reversed":
            rows = _mix_rows(t, m, frame)[:, ::-1]
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(name)
        return _density(frame, rows, m)

    return Stochastic(density, f"{name}{t}", density_kind="known")


def _respond_rule(t: int, top: int, *, reverse: bool = False) -> Any:
    def rule(history: Any) -> np.ndarray:
        on = _column(history, "W") >= 2 if t == 1 else _column(history, f"L{t}") == 1.0
        return np.where(on != reverse, top, 0)

    return rule


def _lagged_rule(t: int, top: int) -> Any:
    def rule(history: Any) -> np.ndarray:
        on = _column(history, "W") >= 2 if t <= 2 else _column(history, f"L{t - 1}") == 1.0
        return np.where(on, top, 0)

    return rule


def regimen(name: str, K: int, m: int, nodes: Iterable[int] | None = None) -> Any:
    """The regimen ``LTMLE`` takes for plan ``name``.

    ``nodes`` limits a mutated plan (``lagged``, ``mix_noprev``) to those nodes, as
    :func:`tests.discrete_law_longitudinal_horizon.plan` does.
    """
    top = m - 1
    every = set(range(1, K + 1)) if nodes is None else set(nodes)
    if name in ("low", "never"):
        return 0
    if name in ("high", "always"):
        return top
    if name == "switch":
        return DynamicRegimen(name, tuple(top if t < K else 0 for t in range(1, K + 1)))
    if name.startswith("dur") and m == 2:
        j = int(name[3:])
        return DynamicRegimen(name, tuple(int(t <= j) for t in range(1, K + 1)))
    if name in ("respond", "reversed"):
        rules = tuple(_respond_rule(t, top, reverse=name == "reversed") for t in range(1, K + 1))
        return DynamicRegimen(name, rules, rule_kind="known")
    if name == "lagged":
        rules = tuple(
            _lagged_rule(t, top) if t in every and t > 1 else _respond_rule(t, top)
            for t in range(1, K + 1)
        )
        return DynamicRegimen(name, rules, rule_kind="known")
    if name in ("mixlast", "coinlast"):
        nodes_: tuple[Any, ...] = (*(_policy(name, t, K, m, every) for t in range(1, K)), 0)
        return DynamicRegimen(name, nodes_)
    if name == "taper":
        nodes_ = (top, *(_policy(name, t, K, m, every) for t in range(2, K + 1)))
        return DynamicRegimen(name, nodes_)
    if name in POLICY_PLANS:
        return DynamicRegimen(name, tuple(_policy(name, t, K, m, every) for t in range(1, K + 1)))
    raise ValueError(f"unknown plan {name!r}")


def regimens(names: Iterable[str], K: int, m: int) -> dict[str, Any]:
    """``{name: regimen}`` in the order given; the first is the reference."""
    return {name: regimen(name, K, m) for name in names}

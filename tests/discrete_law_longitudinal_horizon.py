r"""A finite longitudinal law over three to five decisions, with censoring that grows over time.

The K-node counterpart of :mod:`tests.discrete_law_longitudinal`.  Every node is discrete --

.. code-block:: text

    W -> A1 -> C1 -> L2 -> A2 -> C2 -> ... -> LK -> AK -> CK -> Y

-- with ``W`` uniform on ``{0, 1, 2, 3}``, binary ``L_t`` and ``C_t`` and ``Y``, and ``m``
treatment arms.  The support is finite, so one row per observed support point, weighted by its
mass, *is* the law (:func:`weighted_frame`).  On that frame a single-fold fit estimates the law's
own parameter, which lets the exact witnesses of the X39 studies assert equalities instead of
statistical statements.

The law family is indexed by :class:`Law` ``(B, s, c0, gamma_C)``.  ``B`` scales the
confounding of every treatment logit and ``s`` sharpens it by a share per node, so positivity
weakens over time.  ``c0`` is the retention intercept, and ``gamma_C`` scales the history terms of
the retention logit: it is 1 on every study law and 4 on the censoring-witness law.  The
conditionals:

=========  ==========================================================================================
node       conditional
=========  ==========================================================================================
``W``      uniform on ``{0, 1, 2, 3}``; ``z = (W - 1.5) / 1.5``
``L_t``    ``logit = -0.4 + 0.5 z + 0.9 e(A_{t-1}) + 0.7 L_{t-1} - 0.2 (t - 2)`` for ``t >= 2``;
           ``L_1 = 0``
``A_t``    softmax over the arms; arm ``k >= 1`` has logit
           ``B (1 + s (t - 1)) (-0.3 k + 0.7 z gamma_k + 0.9 (L_t - 0.5) delta_k + 1{A_{t-1} = k})``
           with ``gamma_{m-1} = delta_{m-1} = 1`` and ``gamma_k = -0.5``, ``delta_k = 0.5`` otherwise
           (the indicator term is absent at ``t = 1``)
``C_t``    ``logit = c0 - 0.35 (t - 1) + gamma_C (0.3 z - 0.5 L_t - 0.4 e(A_t))``
``Y``      ``logit = -0.6 + 0.4 z + 0.9 z^2 - 0.6 1{z > 0.5} + sum_t 0.22 e(A_t)(1 + 0.1 (t - 1))
           + 0.8 L_K + 0.4 L_{K-1} - 0.3 L_K e(A_K)``
=========  ==========================================================================================

with ``e(a) = a / (m - 1)``.  The truths read ``L`` and ``Y`` only, so every law of the family
has the same truths.

The g-formula and the efficient influence function are written out longhand over the history
tree, and :func:`truth_from_masses` recomputes the g-formula from the support masses alone, so a
complex-step derivative of it checks :func:`eif`.  Like every ``tests/discrete_law*.py`` oracle,
this module imports nothing from ``cleverly`` (``tests/unit/test_oracle_independence.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

#: The tail criterion: the relative SD of the plug-in variance, ``sqrt((kappa - 1) / n)``, is at
#: most this.
C_KAPPA = 0.25

#: Baseline levels.
W_LEVELS = 4


@dataclass(frozen=True)
class Law:
    """One law of the family.

    Parameters
    ----------
    B : float
        The confounding scale of every treatment logit.
    s : float
        The share by which confounding sharpens per node.
    c0 : float
        The retention intercept.
    gamma_C : float
        The scale of the history terms of the retention logit.
    null_outcome : bool
        When true, ``Y`` does not read any arm directly (the sharp-null outcome table).
    """

    B: float = 1.0
    s: float = 0.1
    c0: float = 3.5
    gamma_C: float = 1.0
    null_outcome: bool = False

    @property
    def key(self) -> tuple[float, float, float, float]:
        return (self.B, self.s, self.c0, self.gamma_C)


#: The realistic law: full confounding that sharpens by 10 percent per node.
L_R = Law(1.0, 0.1, 3.5)
#: The censoring-witness law: ``L_R`` with the retention history terms scaled by four.
L_C = Law(1.0, 0.1, 3.5, 4.0)

#: The grid of fallback laws: ``B`` in {0.25, 0.5, 0.75, 1}, ``s`` in {0, 0.05, 0.1}, ``c0`` in
#: {3.5, 4.5, 5.5}.
GRID = tuple(
    Law(B, s, c0)
    for B in (0.25, 0.5, 0.75, 1.0)
    for s in (0.0, 0.05, 0.1)
    for c0 in (3.5, 4.5, 5.5)
)


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _z(w: Any) -> Any:
    return (np.asarray(w, dtype=float) - 1.5) / 1.5


def _effect(a: Any, m: int) -> Any:
    return np.asarray(a, dtype=float) / (m - 1)


# ----------------------------------------------------------------------------- conditionals
def treatment_probabilities(t: int, m: int, law: Law, w: Any, l_t: Any, a_prev: Any) -> np.ndarray:
    """``P(A_t = k | W, L_t, A_{t-1})`` as an ``(n, m)`` array."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    l_t = np.broadcast_to(np.asarray(l_t, dtype=float), w.shape)
    a_prev = np.broadcast_to(np.asarray(a_prev), w.shape)
    scale = law.B * (1.0 + law.s * (t - 1))
    logits = np.zeros((w.size, m))
    for k in range(1, m):
        top = k == m - 1
        gamma = 1.0 if top else -0.5
        delta = 1.0 if top else 0.5
        lagged = (a_prev == k).astype(float) if t > 1 else 0.0
        logits[:, k] = scale * (-0.3 * k + 0.7 * _z(w) * gamma + 0.9 * (l_t - 0.5) * delta + lagged)
    logits -= logits.max(axis=1, keepdims=True)
    expd = np.exp(logits)
    return np.asarray(expd / expd.sum(axis=1, keepdims=True))


def retention_probability(t: int, m: int, law: Law, w: Any, l_t: Any, a_t: Any) -> np.ndarray:
    """``P(C_t = 1 | W, L_t, A_t)``."""
    history = 0.3 * _z(w) - 0.5 * np.asarray(l_t, dtype=float) - 0.4 * _effect(a_t, m)
    return np.asarray(_expit(law.c0 - 0.35 * (t - 1) + law.gamma_C * history))


def covariate_probability(t: int, m: int, w: Any, a_prev: Any, l_prev: Any) -> np.ndarray:
    """``P(L_t = 1 | W, A_{t-1}, L_{t-1})`` for ``t >= 2``."""
    return np.asarray(
        _expit(
            -0.4
            + 0.5 * _z(w)
            + 0.9 * _effect(a_prev, m)
            + 0.7 * np.asarray(l_prev, dtype=float)
            - 0.2 * (t - 2)
        )
    )


def outcome_probability(
    K: int, m: int, w: Any, arms: list[Any], covariates: list[Any], *, null: bool = False
) -> np.ndarray:
    """``P(Y = 1 | W, A_1..A_K, L_1..L_K)``; ``covariates[0]`` is ``L_1 = 0``."""
    z = _z(w)
    linear = -0.6 + 0.4 * z + 0.9 * z**2 - 0.6 * (z > 0.5)
    l_last = np.asarray(covariates[K - 1], dtype=float)
    l_before = np.asarray(covariates[K - 2], dtype=float)
    linear = linear + 0.8 * l_last + 0.4 * l_before
    if not null:
        for index in range(K):
            linear = linear + 0.22 * _effect(arms[index], m) * (1.0 + 0.1 * index)
        linear = linear - 0.3 * l_last * _effect(arms[K - 1], m)
    return np.asarray(_expit(linear))


# ----------------------------------------------------------------------------- the history tree
@dataclass(frozen=True)
class Histories:
    """Every history ``H_t = (W, L_2..L_t, A_1..A_{t-1})`` per node, as columns.

    Node ``t`` holds ``4 (2m)^{t-1}`` histories.  Index ``h`` at node ``t + 1`` is
    ``(h_t m + a_t) 2 + l_{t+1}``, so a child is found by arithmetic.

    Parameters
    ----------
    K : int
        Decision nodes.
    m : int
        Arms.
    W : dict
        Node -> the baseline of each history.
    L : dict
        Node -> ``{s: L_s}`` for ``2 <= s <= t``.
    A : dict
        Node -> ``{s: A_s}`` for ``1 <= s < t``.
    """

    K: int
    m: int
    W: dict[int, np.ndarray]
    L: dict[int, dict[int, np.ndarray]]
    A: dict[int, dict[int, np.ndarray]]

    def size(self, t: int) -> int:
        return int(self.W[t].size)

    def l_now(self, t: int) -> np.ndarray:
        return self.L[t][t] if t > 1 else np.zeros(self.size(t))

    def a_prev(self, t: int) -> np.ndarray:
        return self.A[t][t - 1] if t > 1 else np.zeros(self.size(t), dtype=int)


@cache
def histories(K: int, m: int) -> Histories:
    """The history tree of ``(K, m)``."""
    W = {1: np.arange(W_LEVELS, dtype=float)}
    L: dict[int, dict[int, np.ndarray]] = {1: {}}
    A: dict[int, dict[int, np.ndarray]] = {1: {}}
    for t in range(1, K):
        size = W[t].size
        parent = np.repeat(np.arange(size), 2 * m)
        arm = np.tile(np.repeat(np.arange(m), 2), size)
        cov = np.tile(np.array([0.0, 1.0]), size * m)
        W[t + 1] = W[t][parent]
        L[t + 1] = {s: values[parent] for s, values in L[t].items()}
        L[t + 1][t + 1] = cov
        A[t + 1] = {s: values[parent] for s, values in A[t].items()}
        A[t + 1][t] = arm
    return Histories(K, m, W, L, A)


def history_index(t: int, m: int, w: Any, covariates: dict[int, Any], arms: dict[int, Any]) -> Any:
    """The node-``t`` index of the histories given as columns (``covariates`` keyed 2..t)."""
    index = np.asarray(w, dtype=int)
    for s in range(1, t):
        index = (index * m + np.asarray(arms[s], dtype=int)) * 2 + np.asarray(
            covariates[s + 1], dtype=int
        )
    return index


@dataclass(frozen=True)
class Mechanism:
    """The law's conditionals on the history tree.

    Parameters
    ----------
    g : dict
        Node -> ``(N_t, m)`` treatment probabilities.
    c : dict
        Node -> ``(N_t, m)`` retention probability given each arm.
    l_next : dict
        Node ``t < K`` -> ``(N_t, m)`` probability that ``L_{t+1} = 1`` given each arm.
    y : numpy.ndarray
        ``(N_K, m)`` outcome probability given each arm at node ``K``.
    """

    g: dict[int, np.ndarray]
    c: dict[int, np.ndarray]
    l_next: dict[int, np.ndarray]
    y: np.ndarray


@cache
def conditionals(K: int, m: int, law: Law) -> Mechanism:
    """The conditionals of ``law`` on the history tree of ``(K, m)``."""
    tree = histories(K, m)
    g, c, l_next = {}, {}, {}
    for t in range(1, K + 1):
        w, l_t, a_prev = tree.W[t], tree.l_now(t), tree.a_prev(t)
        g[t] = treatment_probabilities(t, m, law, w, l_t, a_prev)
        c[t] = np.column_stack([retention_probability(t, m, law, w, l_t, a) for a in range(m)])
        if t < K:
            l_next[t] = np.column_stack(
                [covariate_probability(t + 1, m, w, a, l_t) for a in range(m)]
            )
    w = tree.W[K]
    covs = [np.zeros(w.size)] + [tree.L[K][s] for s in range(2, K + 1)]
    y = np.column_stack(
        [
            outcome_probability(
                K,
                m,
                w,
                [tree.A[K][s] for s in range(1, K)] + [np.full(w.size, a)],
                covs,
                null=law.null_outcome,
            )
            for a in range(m)
        ]
    )
    return Mechanism(g, c, l_next, y)


# ----------------------------------------------------------------------------- plans
#: A plan maps ``(t, tree)`` to the ``(N_t, m)`` intervention density on node ``t``'s histories.
Plan = Callable[[int, Histories], np.ndarray]


def _indicator(size: int, m: int, arm: Any) -> np.ndarray:
    out = np.zeros((size, m))
    out[np.arange(size), np.broadcast_to(np.asarray(arm, dtype=int), (size,))] = 1.0
    return out


def _mix(t: int, tree: Histories, *, previous: bool = True) -> np.ndarray:
    """The known stochastic plan: ``coin`` at ``m = 2`` and the rotated ``mix`` at ``m = 3``."""
    size, m = tree.size(t), tree.m
    if m == 2:
        p = np.full(size, 0.5) if t == 1 else np.where(tree.l_now(t) == 1, 0.75, 0.5)
        return np.column_stack([1.0 - p, p])
    if t == 1:
        return np.tile([0.25, 0.25, 0.5], (size, 1))
    base = np.array([0.5, 0.25, 0.25])
    rolls = np.stack([np.roll(base, k) for k in range(3)])
    shift = tree.l_now(t).astype(int) + (tree.a_prev(t).astype(int) if previous else 0)
    return np.asarray(rolls[shift % 3])


def _respond_on(t: int, tree: Histories) -> np.ndarray:
    return np.asarray(tree.W[t] >= 2) if t == 1 else np.asarray(tree.l_now(t) == 1)


def plan(name: str, K: int, m: int, nodes: Iterable[int] | None = None) -> Plan:
    """The plan ``name``.  ``nodes`` limits a mutated plan to those nodes.

    The mutated plans (``lagged``, ``mix_noprev``) apply their change at ``nodes`` (default every
    node where the change exists) and the declared plan elsewhere.
    """
    top = m - 1
    every = set(range(1, K + 1)) if nodes is None else set(nodes)

    def static(arm: int) -> Plan:
        return lambda t, tree: _indicator(tree.size(t), m, arm)

    def respond(t: int, tree: Histories) -> np.ndarray:
        return _indicator(tree.size(t), m, np.where(_respond_on(t, tree), top, 0))

    def lagged(t: int, tree: Histories) -> np.ndarray:
        if t not in every or t == 1:
            return respond(t, tree)
        on = tree.W[t] >= 2 if t == 2 else tree.L[t][t - 1] == 1
        return _indicator(tree.size(t), m, np.where(on, top, 0))

    def reversed_rule(t: int, tree: Histories) -> np.ndarray:
        return _indicator(tree.size(t), m, np.where(_respond_on(t, tree), 0, top))

    def mix(t: int, tree: Histories) -> np.ndarray:
        return _mix(t, tree)

    def mix_noprev(t: int, tree: Histories) -> np.ndarray:
        return _mix(t, tree, previous=t not in every)

    def mixlast(t: int, tree: Histories) -> np.ndarray:
        return mix(t, tree) if t < K else _indicator(tree.size(t), m, 0)

    def switch(t: int, tree: Histories) -> np.ndarray:
        return _indicator(tree.size(t), m, top if t < K else 0)

    def taper(t: int, tree: Histories) -> np.ndarray:
        return _indicator(tree.size(t), m, top) if t == 1 else mix(t, tree)

    def uniform(t: int, tree: Histories) -> np.ndarray:
        return np.full((tree.size(t), m), 1.0 / m)

    def mix_node1(t: int, tree: Histories) -> np.ndarray:
        return _mix(1, histories(1, m))[np.asarray(tree.W[t], dtype=int)]

    def mix_norot(t: int, tree: Histories) -> np.ndarray:
        if t == 1:
            return mix(t, tree)
        base = [0.5, 0.5] if m == 2 else [0.5, 0.25, 0.25]
        return np.tile(base, (tree.size(t), 1))

    def mix_reversed(t: int, tree: Histories) -> np.ndarray:
        return mix(t, tree)[:, ::-1]

    table: dict[str, Plan] = {
        "low": static(0),
        "never": static(0),
        "high": static(top),
        "always": static(top),
        "respond": respond,
        "lagged": lagged,
        "reversed": reversed_rule,
        "mix": mix,
        "coin": mix,
        "mix_noprev": mix_noprev,
        "mixlast": mixlast,
        "coinlast": mixlast,
        "switch": switch,
        "taper": taper,
        "uniform": uniform,
        "mix_node1": mix_node1,
        "mix_norot": mix_norot,
        "mix_reversed": mix_reversed,
    }
    if m == 2:
        for j in range(K + 1):
            table[f"dur{j}"] = _duration(j, m)
    if name not in table:
        raise ValueError(f"unknown plan {name!r}")
    return table[name]


def _duration(j: int, m: int) -> Plan:
    """Treat through node ``j``, then stop."""
    return lambda t, tree: _indicator(tree.size(t), m, int(t <= j))


#: The plans whose density is not an indicator at some node.
POLICY_PLANS = frozenset(
    {"mix", "coin", "mix_noprev", "mixlast", "coinlast", "taper", "uniform", "mix_node1"}
    | {"mix_norot", "mix_reversed"}
)


def policy_nodes(name: str, K: int) -> frozenset[int]:
    """The nodes at which plan ``name`` is a known stochastic node."""
    if name in ("mixlast", "coinlast"):
        return frozenset(range(1, K))
    if name == "taper":
        return frozenset(range(2, K + 1))
    if name in POLICY_PLANS:
        return frozenset(range(1, K + 1))
    return frozenset()


# ----------------------------------------------------------------------------- the g-formula
@dataclass(frozen=True)
class GFormula:
    """The g-formula of one plan on one law.

    Parameters
    ----------
    psi : float
        The plan mean.
    qa : dict
        Node -> ``(N_t, m)`` regression ``E[Qbar_{t+1} | H_t, A_t = a, C_t = 1]``.
    qbar : dict
        Node -> ``(N_t,)`` plan-integrated regression ``Qbar_t``.
    q : dict
        Node -> ``(N_t, m)`` intervention density.
    """

    psi: float
    qa: dict[int, np.ndarray]
    qbar: dict[int, np.ndarray]
    q: dict[int, np.ndarray]


def gformula(K: int, m: int, plan_: Plan | str, law: Law = L_R) -> GFormula:
    """The sequential regression of ``plan_`` under ``law``."""
    tree = histories(K, m)
    density = plan(plan_, K, m) if isinstance(plan_, str) else plan_
    mech = conditionals(K, m, law)
    q = {t: density(t, tree) for t in range(1, K + 1)}
    qa: dict[int, np.ndarray] = {K: mech.y}
    qbar: dict[int, np.ndarray] = {K: np.sum(q[K] * mech.y, axis=1)}
    for t in range(K - 1, 0, -1):
        child = qbar[t + 1].reshape(tree.size(t), m, 2)
        p1 = mech.l_next[t]
        qa[t] = (1.0 - p1) * child[:, :, 0] + p1 * child[:, :, 1]
        qbar[t] = np.sum(q[t] * qa[t], axis=1)
    psi = float(np.mean(qbar[1]))
    return GFormula(psi, qa, qbar, q)


def truth(K: int, m: int, plan_: Plan | str, law: Law = L_R) -> float:
    """``E[Y^q]``; law-free in ``(B, s, c0, gamma_C)``, so ``law`` matters for the null law only."""
    return gformula(K, m, plan_, law).psi


# ----------------------------------------------------------------------------- the support
@dataclass(frozen=True)
class Support:
    """The observed support points, in the row order of :func:`weighted_frame`.

    A row ends at node ``end``: censored there (``kind`` 0) or, at ``K``, uncensored with
    ``Y = 1`` (``kind`` 1) or ``Y = 0`` (``kind`` 2).

    Parameters
    ----------
    K : int
        Decision nodes.
    m : int
        Arms.
    end : numpy.ndarray
        The last node each row reached.
    kind : numpy.ndarray
        0 censored at ``end``; 1 ``Y = 1``; 2 ``Y = 0``.
    path : numpy.ndarray
        ``(rows, K)`` history index per node, ``-1`` beyond ``end``.
    arm : numpy.ndarray
        ``(rows, K)`` observed arm per node, ``-1`` beyond ``end``.
    """

    K: int
    m: int
    end: np.ndarray
    kind: np.ndarray
    path: np.ndarray
    arm: np.ndarray = field(repr=False)

    @property
    def rows(self) -> int:
        return int(self.end.size)


@cache
def support(K: int, m: int) -> Support:
    """The observed support of ``(K, m)``, depth first: baseline, then each arm, then ``L``."""
    ends: list[int] = []
    kinds: list[int] = []
    paths: list[list[int]] = []
    arms: list[list[int]] = []

    def walk(t: int, index: int, path: list[int], taken: list[int]) -> None:
        for a in range(m):
            here = [*path, index]
            took = [*taken, a]
            ends.append(t)
            kinds.append(0)
            paths.append(here)
            arms.append(took)
            if t == K:
                for kind in (1, 2):
                    ends.append(t)
                    kinds.append(kind)
                    paths.append(here)
                    arms.append(took)
                continue
            for cov in (0, 1):
                walk(t + 1, (index * m + a) * 2 + cov, here, took)

    for w in range(W_LEVELS):
        walk(1, w, [], [])
    rows = len(ends)
    path = np.full((rows, K), -1, dtype=int)
    arm = np.full((rows, K), -1, dtype=int)
    for i, (p, a) in enumerate(zip(paths, arms, strict=True)):
        path[i, : len(p)] = p
        arm[i, : len(a)] = a
    return Support(K, m, np.asarray(ends), np.asarray(kinds), path, arm)


def support_size(K: int, m: int) -> int:
    """``sum_t 4m(2m)^{t-1} + 2 * 4m(2m)^{K-1}``."""
    return sum(4 * m * (2 * m) ** (t - 1) for t in range(1, K + 1)) + 2 * 4 * m * (2 * m) ** (K - 1)


def masses(K: int, m: int, law: Law = L_R) -> np.ndarray:
    """The probability of each support point under ``law``."""
    sup = support(K, m)
    mech = conditionals(K, m, law)
    rows = np.arange(sup.rows)
    mass = np.full(sup.rows, 1.0 / W_LEVELS)
    for t in range(1, K + 1):
        reached = sup.end >= t
        h, a = sup.path[reached, t - 1], sup.arm[reached, t - 1]
        factor = mech.g[t][h, a]
        stays = mech.c[t][h, a]
        last = sup.end[reached] == t
        censored = last & (sup.kind[reached] == 0)
        factor = factor * np.where(censored, 1.0 - stays, stays)
        if t < K:
            cov = np.where(last, 0, sup.path[np.flatnonzero(reached), np.minimum(t, K - 1)] % 2)
            p1 = mech.l_next[t][h, a]
            factor = factor * np.where(last, 1.0, np.where(cov == 1, p1, 1.0 - p1))
        else:
            py = mech.y[h, a]
            factor = factor * np.where(
                sup.kind[reached] == 1, py, np.where(sup.kind[reached] == 2, 1.0 - py, 1.0)
            )
        mass[rows[reached]] *= factor
    return mass


def weighted_frame(K: int, m: int, law: Law = L_R) -> pd.DataFrame:
    """One row per observed support point, with its mass in column ``w``.

    Columns follow :func:`draw`: ``W``, then per node ``L{t}`` (``t >= 2``), ``A{t}``,
    ``g{t}_{k}``, ``pc{t}`` and ``C{t}``, then ``Y`` and ``w``.  A node a row never reached is
    missing.
    """
    sup = support(K, m)
    tree = histories(K, m)
    mech = conditionals(K, m, law)
    columns: dict[str, np.ndarray] = {"W": tree.W[1][sup.path[:, 0]]}
    for t in range(1, K + 1):
        reached = sup.end >= t
        h = np.where(reached, sup.path[:, t - 1], 0)
        a = np.where(reached, sup.arm[:, t - 1], 0)
        if t > 1:
            columns[f"L{t}"] = np.where(reached, tree.L[t][t][h], np.nan)
        columns[f"A{t}"] = np.where(reached, a.astype(float), np.nan)
        for k in range(m):
            columns[f"g{t}_{k}"] = np.where(reached, mech.g[t][h, k], np.nan)
        columns[f"pc{t}"] = np.where(reached, mech.c[t][h, a], np.nan)
        censored = (sup.end == t) & (sup.kind == 0)
        columns[f"C{t}"] = np.where(reached, np.where(censored, 0.0, 1.0), np.nan)
    columns["Y"] = np.where(sup.kind == 1, 1.0, np.where(sup.kind == 2, 0.0, np.nan))
    columns["w"] = masses(K, m, law)
    return pd.DataFrame(columns)


# ----------------------------------------------------------------------------- influence curves
def _ratios(K: int, m: int, g: GFormula, law: Law) -> np.ndarray:
    """``(rows, K)`` cumulative ratio ``R_t`` per support row: zero once unfollowed or censored."""
    sup = support(K, m)
    mech = conditionals(K, m, law)
    out = np.zeros((sup.rows, K))
    running = np.ones(sup.rows)
    for t in range(1, K + 1):
        reached = sup.end >= t
        h = np.where(reached, sup.path[:, t - 1], 0)
        a = np.where(reached, sup.arm[:, t - 1], 0)
        uncensored = reached & ~((sup.end == t) & (sup.kind == 0))
        running = running * np.where(
            uncensored, g.q[t][h, a] / (mech.g[t][h, a] * mech.c[t][h, a]), 0.0
        )
        out[:, t - 1] = running
    return out


def curve(
    K: int,
    m: int,
    plan_: Plan | str,
    law: Law,
    regressions: dict[int, tuple[np.ndarray, np.ndarray]],
    ratios: np.ndarray | None = None,
) -> np.ndarray:
    """The longhand curve at given node regressions, per support row.

    ``regressions[t] = (qa_observed, qbar)``: per row, the node-``t`` regression at the row's
    observed arm and the plan-integrated value the node carries.  Rows that never reached node
    ``t`` may hold anything there.  ``ratios`` defaults to the law's true cumulative ratios.
    The curve is ``sum_t R_t (Qbar_{t+1} - Q^A_t) + Qbar_1 - psi`` with ``Qbar_{K+1} = Y`` and
    ``psi`` the mass-weighted mean of ``Qbar_1``.
    """
    sup = support(K, m)
    weight = masses(K, m, law)
    ratio = _ratios(K, m, gformula(K, m, plan_, law), law) if ratios is None else ratios
    y = np.where(sup.kind == 1, 1.0, 0.0)
    first = np.asarray(regressions[1][1], dtype=float)
    psi = float(np.sum(weight * first) / np.sum(weight))
    out = first - psi
    for t in range(1, K + 1):
        qa_obs = np.asarray(regressions[t][0], dtype=float)
        following = np.asarray(regressions[t + 1][1], dtype=float) if t < K else y
        term = ratio[:, t - 1] * (following - qa_obs)
        out = out + np.where(ratio[:, t - 1] != 0.0, term, 0.0)
    return np.asarray(out)


def oracle_regressions(
    K: int, m: int, plan_: Plan | str, law: Law = L_R
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The true node regressions of ``plan_`` per support row, in :func:`curve`'s form."""
    sup = support(K, m)
    g = gformula(K, m, plan_, law)
    out = {}
    for t in range(1, K + 1):
        reached = sup.end >= t
        h = np.where(reached, sup.path[:, t - 1], 0)
        a = np.where(reached, sup.arm[:, t - 1], 0)
        out[t] = (g.qa[t][h, a], g.qbar[t][h])
    return out


def eif(K: int, m: int, plan_: Plan | str, law: Law = L_R) -> np.ndarray:
    """The efficient influence function of ``E[Y^q]`` per support row."""
    return curve(K, m, plan_, law, oracle_regressions(K, m, plan_, law))


def oracle_regression(
    K: int, m: int, plan_: Plan | str, law: Law, t: int
) -> Callable[[Any, dict[int, Any], dict[int, Any], Any], np.ndarray]:
    """``Q^A_{0,t}(a, H_t)`` as a function of ``(W, {s: L_s}, {s: A_s}, a)`` columns."""
    qa = gformula(K, m, plan_, law).qa[t]

    def regression(w: Any, covariates: dict[int, Any], arms: dict[int, Any], a: Any) -> np.ndarray:
        index = history_index(t, m, w, covariates, arms)
        return np.asarray(qa[index, np.asarray(a, dtype=int)])

    return regression


# ----------------------------------------------------------------------------- the g-formula from masses
def truth_from_masses(K: int, m: int, plan_: Plan | str, mass: np.ndarray) -> Any:
    """The g-formula computed from support masses alone (complex masses allowed).

    The conditionals are ratios of summed masses, so a complex-step perturbation of one row's
    mass gives the Gateaux derivative of the plan mean in that row's direction.
    """
    sup = support(K, m)
    tree = histories(K, m)
    density = plan(plan_, K, m) if isinstance(plan_, str) else plan_
    q = {t: density(t, tree) for t in range(1, K + 1)}
    dtype = np.result_type(mass, float)
    reach = {}
    cont = {}
    for t in range(1, K + 1):
        reached = sup.end >= t
        h = sup.path[reached, t - 1]
        a = sup.arm[reached, t - 1]
        node = np.zeros(tree.size(t), dtype=dtype)
        np.add.at(node, h, mass[reached])
        reach[t] = node
        uncensored = ~((sup.end[reached] == t) & (sup.kind[reached] == 0))
        both = np.zeros((tree.size(t), m), dtype=dtype)
        np.add.at(both, (h[uncensored], a[uncensored]), mass[reached][uncensored])
        cont[t] = both
    ones = sup.kind == 1
    ymass = np.zeros((tree.size(K), m), dtype=dtype)
    np.add.at(ymass, (sup.path[ones, K - 1], sup.arm[ones, K - 1]), mass[ones])
    qbar = np.sum(q[K] * (ymass / cont[K]), axis=1)
    for t in range(K - 1, 0, -1):
        child = reach[t + 1].reshape(tree.size(t), m, 2)
        p1 = child[:, :, 1] / cont[t]
        nxt = qbar.reshape(tree.size(t), m, 2)
        qa = (1.0 - p1) * nxt[:, :, 0] + p1 * nxt[:, :, 1]
        qbar = np.sum(q[t] * qa, axis=1)
    return np.sum(reach[1] * qbar) / np.sum(reach[1])


# ----------------------------------------------------------------------------- diagnostics
def summary(weight: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """``sigma``, ``kappa`` and ``n_kappa`` of a mean-zero curve under ``weight``."""
    p = weight / np.sum(weight)
    variance = float(np.sum(p * values**2))
    kappa = float(np.sum(p * values**4) / variance**2)
    return {
        "sigma": float(np.sqrt(variance)),
        "kappa": kappa,
        "n_kappa": float(np.ceil((kappa - 1.0) / C_KAPPA**2)),
    }


def cumulative_follow(K: int, m: int, plan_: Plan | str, law: Law) -> np.ndarray:
    """Per support row ending at ``K`` uncensored and followed: ``prod_s g_s c_s``; else NaN."""
    sup = support(K, m)
    mech = conditionals(K, m, law)
    g = gformula(K, m, plan_, law)
    product = np.ones(sup.rows)
    followed = np.ones(sup.rows, dtype=bool)
    for t in range(1, K + 1):
        reached = sup.end >= t
        h = np.where(reached, sup.path[:, t - 1], 0)
        a = np.where(reached, sup.arm[:, t - 1], 0)
        product = product * np.where(reached, mech.g[t][h, a] * mech.c[t][h, a], 1.0)
        followed &= ~reached | (g.q[t][h, a] > 0)
    final = (sup.end == K) & (sup.kind != 0) & followed
    return np.where(final, product, np.nan)


def diagnostics(
    K: int, m: int, plan_: Plan | str, law: Law = L_R, bound: float = 0.01
) -> dict[str, float]:
    """The section 5.1 diagnostics of one plan mean.

    ``sigma``, ``kappa``, ``n_kappa`` of the EIF; ``pmin`` the smallest cumulative follow
    probability over followed paths; ``ess_K`` the node-``K`` effective sample fraction
    ``1 / E[R_K^2]``; ``binding`` the followed mass share whose cumulative probability lies below
    ``bound``.
    """
    weight = masses(K, m, law)
    g = gformula(K, m, plan_, law)
    out = {"psi": g.psi, **summary(weight, eif(K, m, plan_, law))}
    ratio = _ratios(K, m, g, law)[:, K - 1]
    out["ess_K"] = 1.0 / float(np.sum(weight * ratio**2))
    cum = cumulative_follow(K, m, plan_, law)
    ok = ~np.isnan(cum)
    out["pmin"] = float(np.min(cum[ok]))
    out["binding"] = float(np.sum(weight[ok & (cum < bound)]) / np.sum(weight[ok]))
    return out


def contrast(
    K: int, m: int, first: Plan | str, second: Plan | str, law: Law = L_R
) -> dict[str, float]:
    """The diagnostics of ``psi(first) - psi(second)``, with the paired-curve correlation."""
    weight = masses(K, m, law)
    p = weight / np.sum(weight)
    da, db = eif(K, m, first, law), eif(K, m, second, law)
    va, vb = float(np.sum(p * da**2)), float(np.sum(p * db**2))
    out = {"psi": truth(K, m, first, law) - truth(K, m, second, law), **summary(weight, da - db)}
    out["corr"] = float(np.sum(p * da * db) / np.sqrt(va * vb))
    out["inflation"] = float(np.sqrt((va + vb) / np.sum(p * (da - db) ** 2)))
    return out


def msm_coefficients(K: int, law: Law = L_R) -> dict[str, dict[str, float]]:
    """The unit-weight identity-link projection of ``dur0..durK`` on ``(1, duration)``, ``m = 2``."""
    weight = masses(K, 2, law)
    design = np.column_stack([np.ones(K + 1), np.arange(K + 1)])
    operator = np.linalg.solve(design.T @ design, design.T)
    psis = np.array([truth(K, 2, f"dur{j}", law) for j in range(K + 1)])
    curves = np.column_stack([eif(K, 2, f"dur{j}", law) for j in range(K + 1)])
    beta = operator @ psis
    ic = curves @ operator.T
    return {
        name: {"psi": float(beta[i]), **summary(weight, ic[:, i])}
        for i, name in enumerate(("msm_intercept", "msm_duration"))
    }


# ----------------------------------------------------------------------------- sampling
def draw(K: int, m: int, n: int, seed: int, law: Law = L_R) -> pd.DataFrame:
    """``n`` units drawn from ``law``, with the law's probabilities as ``g{t}_{k}`` and ``pc{t}``.

    The column layout is :func:`weighted_frame`'s without ``w``.
    """
    rng = np.random.default_rng(seed)
    w = rng.integers(0, W_LEVELS, n).astype(float)
    alive = np.ones(n, dtype=bool)
    columns: dict[str, np.ndarray] = {"W": w}
    arms: list[np.ndarray] = []
    covs: list[np.ndarray] = [np.zeros(n)]
    a_prev = np.zeros(n, dtype=int)
    l_prev = np.zeros(n)
    for t in range(1, K + 1):
        if t > 1:
            l_t = (rng.random(n) < covariate_probability(t, m, w, a_prev, l_prev)).astype(float)
            columns[f"L{t}"] = np.where(alive, l_t, np.nan)
            covs.append(l_t)
        else:
            l_t = np.zeros(n)
        g = treatment_probabilities(t, m, law, w, l_t, a_prev)
        a_t = (rng.random(n)[:, None] > np.cumsum(g, axis=1)[:, :-1]).sum(axis=1)
        columns[f"A{t}"] = np.where(alive, a_t.astype(float), np.nan)
        for k in range(m):
            columns[f"g{t}_{k}"] = np.where(alive, g[:, k], np.nan)
        pc = retention_probability(t, m, law, w, l_t, a_t)
        c_t = (rng.random(n) < pc).astype(float)
        columns[f"pc{t}"] = np.where(alive, pc, np.nan)
        columns[f"C{t}"] = np.where(alive, c_t, np.nan)
        alive = alive & (c_t == 1)
        arms.append(a_t)
        a_prev, l_prev = a_t, l_t
    py = outcome_probability(K, m, w, arms, covs, null=law.null_outcome)
    columns["Y"] = np.where(alive, (rng.random(n) < py).astype(float), np.nan)
    return pd.DataFrame(columns)


#: The columns ``LTMLE`` and ``SDR`` read.
def roles(K: int, m: int) -> dict[str, Any]:
    """The ``fit`` keywords of a frame from :func:`draw` or :func:`weighted_frame`."""
    return {
        "outcome": "Y",
        "treatment": [f"A{t}" for t in range(1, K + 1)],
        "baseline": ["W"],
        "time_varying": [[]] + [[f"L{t}"] for t in range(2, K + 1)],
        "censoring": [f"C{t}" for t in range(1, K + 1)],
    }


def probability_columns(K: int, m: int) -> dict[str, Any]:
    """``treatment_probabilities=`` and ``censoring_probabilities=`` for the law's columns."""
    return {
        "treatment_probabilities": {
            f"A{t}": {k: f"g{t}_{k}" for k in range(m)} for t in range(1, K + 1)
        },
        "censoring_probabilities": {f"C{t}": f"pc{t}" for t in range(1, K + 1)},
    }

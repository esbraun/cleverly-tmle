r"""The laws of the registered study ``sequential-baselines`` (X41), with exact truths.

The module imports nothing from ``cleverly``: every truth here is an independent statement of
the g-formula, so agreement with an estimate is evidence about the estimator.

======  ======================================================================================
law     structure
======  ======================================================================================
``D2``  :math:`T = 2`, discrete.  Binary :math:`W`; binary :math:`L_1` on :math:`W`; binary
        :math:`A_1` on :math:`L_1`; retention :math:`C_1` on :math:`(L_1, A_1)`; binary
        :math:`L_2` on :math:`(L_1, A_1)`; three-arm :math:`A_2` on :math:`(L_2, A_1)`;
        retention :math:`C_2` on :math:`(L_2, A_2)`; binary :math:`Y` with
        :math:`\operatorname{logit} = b_0 + b_W W + b_L L_2 + b_A[A_2] + b_{LA} L_2 1\{A_2 = 2\}`
``D2B`` ``D2`` with a binary :math:`A_2` (the R ``ltmle`` parity law: ``ltmle`` refuses a
        non-binary A node)
``D2S`` ``D2`` with an absorbing event :math:`Y_1` after node 1 on :math:`(L_1, A_1)` and
        :math:`Y_2` on :math:`(W, L_2, A_2)`
``D2C`` ``D2S`` with two competing causes at each node, multinomial on the same terms
``G``   :math:`T = 3`, continuous.  :math:`W \sim U(-1, 1)`, binary :math:`V`;
        :math:`L_t = 0.3 W + 0.5 L_{t-1} - 0.4 A_{t-1} + U_t`; binary :math:`A_t` logistic
        in :math:`(W, V, L_t, A_{t-1})`; retention logistic in
        :math:`(W, L_t, A_t)`; :math:`Y = 1 + 0.5 W + 0.3 V + 0.6 L_3 - 0.8 A_3 + U_Y`.
        Every noise term is :math:`U(-1, 1)`
``X``   ``G`` with :math:`0.4 L_3^2` in :math:`Y` and :math:`0.5 (L_t^2 - 0.5)` in the logit
        of :math:`A_t` and of :math:`C_t`: main-terms GLMs are wrong for every regression
``GM``  :math:`T = 2`, continuous doses :math:`A_t = 0.5 L_t + U_{A_t}`, no censoring, and
        :math:`Y` linear in :math:`(W, L_2, A_2)`; the modified policy lowers each dose by
        0.5 where the result stays in its conditional support
======  ======================================================================================

**Correct models.**  On ``D2`` every mechanism is main-terms logistic (multinomial at the
three-arm node) in the shipped history design, so the main-terms ``GLM`` is exact.  The node-2
outcome regression is main-terms logistic in :math:`(W, L_2)` among the followers of a static
plan or of the rule :data:`D2_RULE`, which reads :math:`L_1` and :math:`L_2` only; at the policy
node it needs :math:`L_2 \times` arm, which ``GLM(interactions=2)`` holds.  The node-1
regression is a function of :math:`(W, L_1)`, which ``GLM(interactions=2)`` saturates.  The
design test checks every one of these claims at population weights, to ``1e-10``.

On ``G`` every conditional mean of the recursion is linear in the history for a static plan
and for the rule :math:`A_t = V`, so main-terms ``GLM(family="gaussian")`` is exact, and every
mechanism is main-terms logistic.  On ``GM`` the policy lowers a dose where
:math:`U_{A_t} \ge -0.5`, which has probability 0.75 whatever the history, so every
conditional mean of the recursion stays linear.

**Truths.**  The ``D2`` family's truths are finite sums over the support (:func:`D2Law.truth`),
exact to rounding.  The ``G``, ``X`` and ``GM`` truths are closed forms: every
:math:`L_t` is a linear combination of independent uniforms and of the assigned arms, so a
mean is a linear combination of known means and ``X``'s quadratic term reads the known
variance :math:`1/3` of each uniform.  :func:`continuous_truth` states them.  The design test
checks each against a Monte Carlo of :math:`10^6` draws.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

# ------------------------------------------------------------------------------ the D2 family

#: ``P(W = 1)``.
P_W = 0.5
#: ``logit P(L_1 = 1 | W)``: intercept and the ``W`` slope.
L1_COEF = (-0.3, 0.8)
#: ``logit P(A_1 = 1 | L_1)``: intercept and the ``L_1`` slope.
G1_COEF = (-0.2, 0.7)
#: ``logit P(C_1 = 1 | L_1, A_1)``: intercept, ``L_1`` and ``A_1``.
C1_COEF = (2.2, 0.4, -0.5)
#: ``logit P(L_2 = 1 | L_1, A_1)``.
L2_COEF = (-0.4, 0.9, 0.6)
#: The three-arm node: the logits of arms 1 and 2 against arm 0, each with intercept,
#: ``L_2`` and ``A_1``.
G2_COEF = ((0.1, 0.4, -0.3), (0.0, -0.5, 0.6))
#: The binary node of ``D2B``: ``logit P(A_2 = 1 | L_2, A_1)``.
G2_BINARY_COEF = (-0.1, 0.5, 0.4)
#: ``logit P(C_2 = 1 | L_2, A_2)``: intercept, ``L_2``, ``1{A_2 = 1}``, ``1{A_2 = 2}``.
C2_COEF = (2.0, 0.3, -0.3, -0.5)
#: The outcome: ``b_0``, ``b_W``, ``b_L``, ``b_A`` per arm, ``b_LA``.
Y_COEF = (-0.4, 0.5, 0.7, (0.0, -0.6, 0.4), -0.9)
#: The node-1 event of the survival variants: intercept, ``L_1``, ``A_1``.
Y1_COEF = (-1.6, 0.5, -0.4)
#: The second cause of ``D2C``, per node: the node-1 coefficients ``(intercept, L_1, A_1)``
#: and the node-2 offset added to :data:`Y_COEF`'s intercept.
SECOND_CAUSE_NODE1 = (-2.0, -0.3, 0.3)
SECOND_CAUSE_OFFSET = -0.9

#: The observation weight of the ``weights`` cells.
WEIGHT_SLOPE = 0.5


def observation_weight(w: Any) -> Any:
    """The known weight :math:`w = 1 + 0.5 W` of the weighted cells."""
    return 1.0 + WEIGHT_SLOPE * np.asarray(w, dtype=float)


def _bern(p: float, value: int) -> float:
    return float(p if value == 1 else 1.0 - p)


def _softmax(logits: Sequence[float]) -> np.ndarray:
    values = np.exp(np.asarray(logits, dtype=float))
    return np.asarray(values / values.sum(), dtype=float)


#: A plan node: ``("arm", a)``, ``("rule", f)`` with ``f(history) -> arm``, or
#: ``("policy", q)`` with ``q(history) -> probabilities over the arms``.
Node = tuple[str, Any]


def _assignment(node: Node, history: Mapping[str, int], arms: int) -> np.ndarray:
    """The intervention density over the arms at one history."""
    kind, value = node
    if kind == "arm":
        return np.eye(arms)[int(value)]
    if kind == "rule":
        return np.eye(arms)[int(value(history))]
    if kind == "policy":
        density = np.asarray(value(history), dtype=float)
        if density.shape != (arms,) or abs(density.sum() - 1.0) > 1e-12:
            raise ValueError(f"a policy density must have {arms} entries summing to one")
        return density
    raise ValueError(f"unknown plan node {kind!r}")


@dataclass(frozen=True)
class D2Law:
    """One law of the ``D2`` family.

    Parameters
    ----------
    name : str
        ``"D2"``, ``"D2B"``, ``"D2S"`` or ``"D2C"``.
    arms : int
        The arms of node 2: 3, or 2 on ``D2B``.
    events : int
        0 for an end-of-study outcome, 1 for one absorbing event, 2 for two competing causes.
    """

    name: str
    arms: int = 3
    events: int = 0

    # -- conditionals ------------------------------------------------------------------
    def p_l1(self, w: int) -> float:
        return float(expit(L1_COEF[0] + L1_COEF[1] * w))

    def p_a1(self, l1: int) -> float:
        return float(expit(G1_COEF[0] + G1_COEF[1] * l1))

    def p_c1(self, l1: int, a1: int) -> float:
        return float(expit(C1_COEF[0] + C1_COEF[1] * l1 + C1_COEF[2] * a1))

    def p_l2(self, l1: int, a1: int) -> float:
        return float(expit(L2_COEF[0] + L2_COEF[1] * l1 + L2_COEF[2] * a1))

    def g2(self, l2: int, a1: int) -> np.ndarray:
        """``P(A_2 = a | L_2, A_1)`` over the arms."""
        if self.arms == 2:
            p = float(expit(G2_BINARY_COEF[0] + G2_BINARY_COEF[1] * l2 + G2_BINARY_COEF[2] * a1))
            return np.array([1.0 - p, p])
        logits = [0.0] + [c[0] + c[1] * l2 + c[2] * a1 for c in G2_COEF]
        return _softmax(logits)

    def p_c2(self, l2: int, a2: int) -> float:
        return float(
            expit(C2_COEF[0] + C2_COEF[1] * l2 + C2_COEF[2] * (a2 == 1) + C2_COEF[3] * (a2 == 2))
        )

    def _outcome_logit(self, w: int, l2: int, a2: int) -> float:
        b0, bw, bl, ba, bla = Y_COEF
        return float(b0 + bw * w + bl * l2 + ba[a2] + bla * l2 * (a2 == 2))

    def outcome(self, w: int, l2: int, a2: int) -> np.ndarray:
        """``P(cause j at node 2 | W, L_2, A_2)`` for ``j = 0..events`` (0 is no event).

        For an end-of-study law it is ``(1 - P(Y = 1), P(Y = 1))``.
        """
        eta = self._outcome_logit(w, l2, a2)
        if self.events == 2:
            return _softmax([0.0, eta, eta + SECOND_CAUSE_OFFSET])
        p = float(expit(eta))
        return np.array([1.0 - p, p])

    def first_event(self, l1: int, a1: int) -> np.ndarray:
        """``P(cause j at node 1 | L_1, A_1)``; only for a survival or competing law."""
        eta = Y1_COEF[0] + Y1_COEF[1] * l1 + Y1_COEF[2] * a1
        if self.events == 2:
            second = SECOND_CAUSE_NODE1[0] + SECOND_CAUSE_NODE1[1] * l1
            second += SECOND_CAUSE_NODE1[2] * a1
            return _softmax([0.0, eta, second])
        p = float(expit(eta))
        return np.array([1.0 - p, p])

    # -- the support -------------------------------------------------------------------
    @cached_property
    def support(self) -> tuple[tuple[tuple[Any, ...], ...], np.ndarray]:
        """Every observable path with its probability, in a fixed order.

        A path is ``(w, l1, a1, c1, j1, l2, a2, c2, j2)`` with ``None`` for a node the unit
        never reached; ``j1`` is ``None`` on an end-of-study law.
        """
        points: list[tuple[Any, ...]] = []
        mass: list[float] = []
        causes = range(self.events + 1) if self.events else (None,)
        for w, l1, a1 in itertools.product((0, 1), repeat=3):
            base = _bern(P_W, w) * _bern(self.p_l1(w), l1) * _bern(self.p_a1(l1), a1)
            p_c1 = self.p_c1(l1, a1)
            points.append((w, l1, a1, 0, None, None, None, None, None))
            mass.append(base * (1.0 - p_c1))
            for j1 in causes:
                p_j1 = 1.0 if j1 is None else float(self.first_event(l1, a1)[j1])
                stem = base * p_c1 * p_j1
                if j1 not in (None, 0):
                    points.append((w, l1, a1, 1, j1, None, None, None, None))
                    mass.append(stem)
                    continue
                for l2 in (0, 1):
                    branch = stem * _bern(self.p_l2(l1, a1), l2)
                    density = self.g2(l2, a1)
                    for a2 in range(self.arms):
                        reach = branch * density[a2]
                        p_c2 = self.p_c2(l2, a2)
                        points.append((w, l1, a1, 1, j1, l2, a2, 0, None))
                        mass.append(reach * (1.0 - p_c2))
                        probs = self.outcome(w, l2, a2)
                        for j2 in range(len(probs)):
                            points.append((w, l1, a1, 1, j1, l2, a2, 1, j2))
                            mass.append(reach * p_c2 * probs[j2])
        total = np.asarray(mass)
        if abs(total.sum() - 1.0) > 1e-12:
            raise AssertionError(f"{self.name}: the support mass is {total.sum()}")
        return tuple(points), total

    def columns(self) -> list[str]:
        """The frame's columns, in node order."""
        names = ["W", "L1", "A1", "C1"]
        names += self._event_columns(1)
        names += ["L2", "A2", "C2"]
        names += self._event_columns(2) if self.events else ["Y"]
        return names

    def _event_columns(self, node: int) -> list[str]:
        if self.events == 0:
            return []
        if self.events == 1:
            return [f"Y{node}"]
        return [f"R{node}", f"D{node}"]

    def outcome_argument(self) -> Any:
        """``outcome=`` for a fit on this law."""
        if self.events == 0:
            return "Y"
        if self.events == 1:
            return ["Y1", "Y2"]
        return {"relapse": ["R1", "R2"], "death": ["D1", "D2"]}

    def fit_columns(self) -> dict[str, Any]:
        """The node declaration a fit takes."""
        return {
            "outcome": self.outcome_argument(),
            "treatment": ["A1", "A2"],
            "baseline": ["W"],
            "time_varying": [["L1"], ["L2"]],
            "censoring": ["C1", "C2"],
        }

    def _rows(self) -> np.ndarray:
        """The support as a ``(points, columns)`` float matrix, ``nan`` where unreached."""
        points, _ = self.support
        out = np.full((len(points), len(self.columns())), np.nan)
        for index, (w, l1, a1, c1, j1, l2, a2, c2, j2) in enumerate(points):
            values: list[Any] = [w, l1, a1, c1]
            if self.events == 1:
                values.append(j1 if c1 == 1 else None)
            elif self.events == 2:
                values += [None, None] if j1 is None else [float(j1 == 1), float(j1 == 2)]
            values += [l2, a2, c2]
            if self.events == 2:
                values += [None, None] if j2 is None else [float(j2 == 1), float(j2 == 2)]
            else:
                values.append(j2)
            out[index] = [np.nan if value is None else float(value) for value in values]
        return out

    @cached_property
    def support_frame(self) -> pd.DataFrame:
        """One row per support point, with its probability in column ``prob``."""
        frame = pd.DataFrame(self._rows(), columns=self.columns())
        frame["prob"] = self.support[1]
        return frame

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` independent rows."""
        rng = np.random.default_rng(seed)
        _, mass = self.support
        cells = rng.choice(len(mass), size=n, p=mass)
        return pd.DataFrame(self._rows()[cells], columns=self.columns())

    # -- the g-formula -----------------------------------------------------------------
    def truth(
        self,
        plan: Sequence[Node],
        *,
        horizon: int = 2,
        cause: int = 1,
        weight: Callable[[int], float] | None = None,
    ) -> float:
        """The g-formula of ``plan`` at ``horizon``, for cause ``cause``.

        An end-of-study law reads ``horizon = 2`` and ``P(Y = 1)``.  ``weight(w)`` tilts the
        law of ``W`` by a known observation weight, which is the weighted target.
        """
        first, second = plan
        p_w = np.array([1.0 - P_W, P_W])
        if weight is not None:
            p_w = p_w * np.array([weight(0), weight(1)])
            p_w = p_w / p_w.sum()
        total = 0.0
        for w in (0, 1):
            for l1 in (0, 1):
                pi1 = _assignment(first, {"W": w, "L1": l1}, 2)
                for a1 in (0, 1):
                    stem = p_w[w] * _bern(self.p_l1(w), l1) * pi1[a1]
                    if stem == 0.0:
                        continue
                    alive = 1.0
                    if self.events:
                        events = self.first_event(l1, a1)
                        total += stem * float(events[cause])
                        alive = float(events[0])
                    if horizon == 1:
                        continue
                    for l2 in (0, 1):
                        history = {"W": w, "L1": l1, "L2": l2, "A1": a1}
                        pi2 = _assignment(second, history, self.arms)
                        for a2 in range(self.arms):
                            reach = stem * alive * _bern(self.p_l2(l1, a1), l2) * pi2[a2]
                            total += reach * float(self.outcome(w, l2, a2)[cause])
        return float(total)

    def follower_share(self, plan: Sequence[Node]) -> float:
        """The probability that a unit follows ``plan`` and stays uncensored through node 2."""
        points, mass = self.support
        share = 0.0
        for (w, l1, a1, c1, j1, l2, a2, c2, _), p in zip(points, mass, strict=True):
            if c1 != 1 or j1 not in (None, 0) or c2 != 1:
                continue
            pi1 = _assignment(plan[0], {"W": w, "L1": l1}, 2)[a1]
            pi2 = _assignment(plan[1], {"W": w, "L1": l1, "L2": l2, "A1": a1}, self.arms)[a2]
            share += p * float(pi1 > 0) * float(pi2 > 0)
        return share


D2 = D2Law("D2")
D2B = D2Law("D2B", arms=2)
D2S = D2Law("D2S", events=1)
D2C = D2Law("D2C", events=2)
DISCRETE_LAWS = {law.name: law for law in (D2, D2B, D2S, D2C)}


def d2_rule_first(history: Mapping[str, Any]) -> Any:
    """The rule's node 1: treat when :math:`L_1 = 1`."""
    return history["L1"]


def d2_rule_second(history: Mapping[str, Any]) -> Any:
    """The rule's node 2: arm 2 when :math:`L_2 = 1`, arm 0 otherwise."""
    return 2 * np.asarray(history["L2"])


def d2b_rule_second(history: Mapping[str, Any]) -> Any:
    """The binary law's rule at node 2: treat when :math:`L_2 = 1`."""
    return history["L2"]


#: The history-free policy of node 2 (arms 0, 1, 2), with mass on arm 2.
D2_POLICY = (0.2, 0.3, 0.5)

#: The ``D2`` plans, as oracle nodes.
D2_PLANS: dict[str, tuple[Node, Node]] = {
    "static_12": (("arm", 1), ("arm", 2)),
    "static_00": (("arm", 0), ("arm", 0)),
    "rule": (("rule", d2_rule_first), ("rule", d2_rule_second)),
    "policy": (("arm", 1), ("policy", lambda history: D2_POLICY)),
}
#: The ``D2B`` plans (the parity law has a binary node 2).
D2B_PLANS: dict[str, tuple[Node, Node]] = {
    "always": (("arm", 1), ("arm", 1)),
    "never": (("arm", 0), ("arm", 0)),
    "rule": (("rule", d2_rule_first), ("rule", d2b_rule_second)),
}


# ------------------------------------------------------------------------------ G, X and GM

#: ``P(V = 1)``.
P_V = 0.4
#: ``L_t = 0.3 W + 0.5 L_{t-1} - 0.4 A_{t-1} + U_t``.
L_W, L_LAG, L_A = 0.3, 0.5, -0.4
#: ``logit P(A_t = 1 | W, V, L_t, A_{t-1})``: intercept, ``W``, ``V``, ``L_t``, ``A_{t-1}``.
G_TREATMENT = (0.1, 0.2, -0.3, 0.3, 0.4)
#: ``logit P(C_t = 1 | W, L_t, A_t)``: intercept, ``W``, ``L_t``, ``A_t``.
G_RETENTION = (2.5, 0.2, 0.2, -0.3)
#: ``Y = 1 + 0.5 W + 0.3 V + 0.6 L_3 - 0.8 A_3 + U_Y``.
G_OUTCOME = (1.0, 0.5, 0.3, 0.6, -0.8)
#: Law ``X``'s quadratic terms: ``0.4 L_3^2`` in ``Y`` and ``0.5 (L_t^2 - 0.5)`` in each logit.
X_OUTCOME_SQUARE = 0.4
X_LOGIT_SQUARE = 0.5
X_LOGIT_CENTRE = 0.5
#: The variance of :math:`U(-1, 1)`.
UNIFORM_VARIANCE = 1.0 / 3.0
G_TIMES = 3

#: Law ``GM``: ``L_1 = 0.4 W + U``, ``A_t = 0.5 L_t + U``,
#: ``L_2 = 0.3 W + 0.5 L_1 + 0.4 A_1 + U`` and ``Y = 1 + 0.5 W + 0.6 L_2 + 0.7 A_2 + U``.
GM_L1_W = 0.4
GM_DOSE_L = 0.5
GM_L2 = (0.3, 0.5, 0.4)
GM_OUTCOME = (1.0, 0.5, 0.6, 0.7)
#: How far the policy lowers a dose, and the share of units it moves (``U >= -0.5``).
GM_STEP = 0.5
GM_MOVED = 0.75


@dataclass(frozen=True)
class ContinuousLaw:
    """Law ``G`` or ``X``.

    Parameters
    ----------
    name : str
        ``"G"`` or ``"X"``.
    quadratic : bool
        Whether the quadratic terms of ``X`` are present.
    """

    name: str
    quadratic: bool = False

    def treatment_logit(self, w: Any, v: Any, l: Any, a_prev: Any) -> Any:
        c = G_TREATMENT
        eta = c[0] + c[1] * w + c[2] * v + c[3] * l + c[4] * a_prev
        if self.quadratic:
            eta = eta + X_LOGIT_SQUARE * (np.asarray(l) ** 2 - X_LOGIT_CENTRE)
        return eta

    def retention_logit(self, w: Any, l: Any, a: Any) -> Any:
        c = G_RETENTION
        eta = c[0] + c[1] * w + c[2] * l + c[3] * a
        if self.quadratic:
            eta = eta + X_LOGIT_SQUARE * (np.asarray(l) ** 2 - X_LOGIT_CENTRE)
        return eta

    def outcome_mean(self, w: Any, v: Any, l3: Any, a3: Any) -> Any:
        c = G_OUTCOME
        mean = c[0] + c[1] * w + c[2] * v + c[3] * l3 + c[4] * a3
        if self.quadratic:
            mean = mean + X_OUTCOME_SQUARE * np.asarray(l3) ** 2
        return mean

    def columns(self) -> dict[str, Any]:
        """The node declaration a fit takes."""
        return {
            "outcome": "Y",
            "treatment": [f"A{t}" for t in range(1, G_TIMES + 1)],
            "baseline": ["W", "V"],
            "time_varying": [[f"L{t}"] for t in range(1, G_TIMES + 1)],
            "censoring": [f"C{t}" for t in range(1, G_TIMES + 1)],
        }

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` independent rows; a censored unit's later nodes are ``nan``."""
        rng = np.random.default_rng(seed)
        w = rng.uniform(-1.0, 1.0, n)
        v = rng.binomial(1, P_V, n).astype(float)
        out: dict[str, np.ndarray] = {"W": w, "V": v}
        alive = np.ones(n, dtype=bool)
        l_prev = np.zeros(n)
        a_prev = np.zeros(n)
        for t in range(1, G_TIMES + 1):
            l = L_W * w + L_LAG * l_prev + L_A * a_prev + rng.uniform(-1.0, 1.0, n)
            a = rng.binomial(1, expit(self.treatment_logit(w, v, l, a_prev))).astype(float)
            c = rng.binomial(1, expit(self.retention_logit(w, l, a))).astype(float)
            out[f"L{t}"] = np.where(alive, l, np.nan)
            out[f"A{t}"] = np.where(alive, a, np.nan)
            out[f"C{t}"] = np.where(alive, c, np.nan)
            alive = alive & (c == 1)
            l_prev, a_prev = l, a
        y = self.outcome_mean(w, v, l_prev, a_prev) + rng.uniform(-1.0, 1.0, n)
        out["Y"] = np.where(alive, y, np.nan)
        order = ["W", "V"]
        for t in range(1, G_TIMES + 1):
            order += [f"L{t}", f"A{t}", f"C{t}"]
        return pd.DataFrame(out)[[*order, "Y"]]

    # -- the support of Y --------------------------------------------------------------
    def l3_loadings(self) -> tuple[float, tuple[float, float, float]]:
        """``L_3``'s loading on ``W`` and on ``(U_1, U_2, U_3)``, and its arm loadings."""
        w_load = L_W * (1.0 + L_LAG + L_LAG**2)
        return w_load, (L_LAG**2, L_LAG, 1.0)

    def arm_shift(self, arms: Sequence[float]) -> float:
        """``L_3``'s mean shift under arms ``(a_1, a_2)``: ``-0.2 a_1 - 0.4 a_2``."""
        return L_A * (L_LAG * arms[0] + arms[1])

    def outcome_support(self) -> tuple[float, float]:
        r"""The exact support of :math:`Y`.

        ``G``: :math:`Y` is linear in independent uniforms and binary arms, so each end sums
        the absolute loadings.  ``X``: :math:`Y` is convex in the continuous coordinates, so
        its maximum sits at a vertex of the cube (enumerated) and its minimum solves a
        one-dimensional problem in :math:`L_3` (the remaining coordinates enter linearly
        through :math:`L_3` and :math:`W`), solved on a fine grid refined by the closed-form
        stationary point.  Both ends are exact to rounding.
        """
        w_load, u_load = self.l3_loadings()
        c = G_OUTCOME
        best_low, best_high = np.inf, -np.inf
        for v, a1, a2, a3 in itertools.product((0, 1), repeat=4):
            shift = self.arm_shift((a1, a2))
            base = c[0] + c[2] * v + c[4] * a3
            if not self.quadratic:
                spread = abs(c[1] + c[3] * w_load) + c[3] * sum(u_load) + 1.0
                centre = base + c[3] * shift
                best_low = min(best_low, centre - spread)
                best_high = max(best_high, centre + spread)
                continue
            # Vertices of (W, U1, U2, U3) for the maximum; U_Y = +-1 enters linearly.
            for signs in itertools.product((-1.0, 1.0), repeat=4):
                w = signs[0]
                l3 = w_load * w + sum(u * s for u, s in zip(u_load, signs[1:], strict=True))
                l3 += shift
                y = base + c[1] * w + c[3] * l3 + X_OUTCOME_SQUARE * l3**2
                best_high = max(best_high, y + 1.0)
            # The minimum: for each W in [-1, 1], L_3 ranges over an interval; minimize the
            # convex quadratic in L_3 on it, then minimize over W (a piecewise smooth convex
            # function of W, evaluated on a grid and at its kinks).
            reach = sum(u_load)
            grid = np.linspace(-1.0, 1.0, 200_001)
            low_l = w_load * grid + shift - reach
            high_l = w_load * grid + shift + reach
            stationary = -c[3] / (2.0 * X_OUTCOME_SQUARE)
            l_star = np.clip(stationary, low_l, high_l)
            y = base + c[1] * grid + c[3] * l_star + X_OUTCOME_SQUARE * l_star**2
            best_low = min(best_low, float(y.min()) - 1.0)
        return float(best_low), float(best_high)

    def q_bounds(self) -> tuple[float, float]:
        """The declared ``q_bounds``: the support of :math:`Y`, rounded outward to 1e-6."""
        low, high = self.outcome_support()
        return float(np.floor(low * 1e6) / 1e6), float(np.ceil(high * 1e6) / 1e6)


G = ContinuousLaw("G")
X = ContinuousLaw("X", quadratic=True)


def continuous_truth(law: ContinuousLaw, arms: Sequence[Any]) -> float:
    r"""The mean of :math:`Y` under arms ``(a_1, a_2, a_3)`` on ``G`` or ``X``.

    Each arm is ``0``, ``1`` or ``"V"`` (the rule :math:`A_t = V`).  :math:`L_3` under fixed
    arms is :math:`0.525 W + 0.25 U_1 + 0.5 U_2 + U_3 + s(a)` with
    :math:`s(a) = -0.2 a_1 - 0.4 a_2`, so :math:`E[L_3 \mid V] = s(a)` and
    :math:`E[L_3^2 \mid V] = (0.525^2 + 0.25^2 + 0.5^2 + 1) / 3 + s(a)^2`.
    """
    w_load, u_load = law.l3_loadings()
    variance = UNIFORM_VARIANCE * (w_load**2 + sum(u**2 for u in u_load))
    c = G_OUTCOME
    total = 0.0
    for v, p in ((0, 1.0 - P_V), (1, P_V)):
        realised = [float(v) if arm == "V" else float(arm) for arm in arms]
        shift = law.arm_shift(realised[:2])
        mean = c[0] + c[2] * v + c[3] * shift + c[4] * realised[2]
        if law.quadratic:
            mean += X_OUTCOME_SQUARE * (variance + shift**2)
        total += p * mean
    return float(total)


#: The ``G`` and ``X`` plans: ``(a_1, a_2, a_3)``.
G_PLANS: dict[str, tuple[Any, Any, Any]] = {
    "treat_all": (1, 1, 1),
    "treat_none": (0, 0, 0),
    "rule": ("V", "V", "V"),
}


def g_truths(law: ContinuousLaw) -> dict[str, float]:
    """Every level and the ``treat_all vs treat_none`` contrast of ``law``."""
    out = {label: continuous_truth(law, arms) for label, arms in G_PLANS.items()}
    out["contrast"] = out["treat_all"] - out["treat_none"]
    return out


def gm_sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of law ``GM`` (no censoring)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, n)
    l1 = GM_L1_W * w + rng.uniform(-1.0, 1.0, n)
    a1 = GM_DOSE_L * l1 + rng.uniform(-1.0, 1.0, n)
    l2 = GM_L2[0] * w + GM_L2[1] * l1 + GM_L2[2] * a1 + rng.uniform(-1.0, 1.0, n)
    a2 = GM_DOSE_L * l2 + rng.uniform(-1.0, 1.0, n)
    c = GM_OUTCOME
    y = c[0] + c[1] * w + c[2] * l2 + c[3] * a2 + rng.uniform(-1.0, 1.0, n)
    return pd.DataFrame({"W": w, "L1": l1, "A1": a1, "L2": l2, "A2": a2, "Y": y})


GM_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [["L1"], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}


def gm_boundary(l: Any) -> Any:
    """The lowest dose the policy moves: :math:`0.5 L_t - 0.5`, so the result stays above
    the conditional support's lower end :math:`0.5 L_t - 1`."""
    return GM_DOSE_L * np.asarray(l, dtype=float) - GM_STEP


def gm_truths() -> dict[str, float]:
    r"""The natural-course mean, the policy mean and their contrast on ``GM``.

    Under the policy :math:`E[A_t^d \mid L_t] = 0.5 L_t - 0.5 \times 0.75`, every other node is
    linear, and :math:`E[W] = E[U] = 0`.
    """
    shift = -GM_STEP * GM_MOVED
    a1 = shift
    l2 = GM_L2[2] * a1
    a2 = GM_DOSE_L * l2 + shift
    c = GM_OUTCOME
    policy = c[0] + c[2] * l2 + c[3] * a2
    natural = c[0]
    return {"natural": natural, "down": policy, "contrast": policy - natural}


def gm_outcome_support() -> tuple[float, float]:
    """The exact support of :math:`Y` on ``GM``: a linear form in independent uniforms."""
    # Loadings of Y on (W, U_L1, U_A1, U_L2, U_A2, U_Y).
    c = GM_OUTCOME
    l1 = np.array([GM_L1_W, 1.0, 0.0, 0.0, 0.0, 0.0])
    a1 = GM_DOSE_L * l1 + np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    l2 = GM_L2[0] * np.eye(6)[0] + GM_L2[1] * l1 + GM_L2[2] * a1 + np.eye(6)[3]
    a2 = GM_DOSE_L * l2 + np.eye(6)[4]
    y = c[1] * np.eye(6)[0] + c[2] * l2 + c[3] * a2 + np.eye(6)[5]
    spread = float(np.abs(y).sum())
    return c[0] - spread, c[0] + spread


__all__ = [
    "D2",
    "D2B",
    "D2B_PLANS",
    "D2C",
    "D2S",
    "D2_PLANS",
    "D2_POLICY",
    "DISCRETE_LAWS",
    "GM_COLUMNS",
    "G_PLANS",
    "ContinuousLaw",
    "D2Law",
    "G",
    "X",
    "continuous_truth",
    "g_truths",
    "gm_boundary",
    "gm_outcome_support",
    "gm_sample",
    "gm_truths",
    "observation_weight",
]

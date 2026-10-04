"""Laws, truths, plans and learners of the ``longitudinal-mtp`` study.

Three laws, each with its truth computed before any run.

**The continuous law.**  Two nodes, a dose on ``[0, 6]`` at each:

.. code-block:: text

    W ~ Bernoulli(1/2)
    A1 | W ~ N(2.5 + 0.6 W, 1) truncated to [0, 6]
    L2 | W, A1 ~ Bernoulli(expit(-1 + 0.35 A1 - 0.4 W))
    A2 | W, A1, L2 ~ N(1.2 + 0.45 A1 + 0.5 L2 - 0.2 W, 1) truncated to [0, 6]
    Y | ... ~ Bernoulli(expit(-2.2 + 0.25 A1 + 0.45 A2 - 0.03 A2^2 + 0.5 L2 + 0.3 W))

Every outcome probability lies in ``[0.09, 0.85]``.  Every policy keeps the dose inside the
support (each is capped below the upper end), so every density ratio is bounded, the
condition of Díaz, Williams, Hoffman and Schenck (2023), Theorem 3.  The truth is the
g-formula of their Theorem 1 with each node's policy reading the intervened earlier dose
(Definition 1), by composite Gauss-Legendre quadrature split at every point where a policy
or its reading of the history jumps; :func:`continuous_truth` checks the quadrature by
doubling its order (agreement below ``1e-10``).

**The categorical law.**  Integer levels ``0..5`` at each node, the shape of ``lmtp``'s
``man/lmtp_tmle.Rd`` Examples 2.1 and 2.3: the policy ``(a - 1) * (a - 1 >= 1) + a * (a - 1
< 1)`` and its variant gated on ``L_t``.  **The binary law** has 0/1 nodes, for the
risk-ratio tilt of ``lmtp::ipsi``.  Both are finite and their truths are finite sums.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.interventions import ModifiedPolicy, Piece, RiskRatioTilt, Scale, Shift
from cleverly.longitudinal import DynamicRegimen
from tests.studies.fractional_glm import QuasiBinomialGLM

LOWER, UPPER = 0.0, 6.0
SD = 1.0
CAP = 5.5
STEP = 0.5
FACTOR = 1.25
#: Node 2 of the history plan moves only where the earlier dose is above this.
HISTORY_THRESHOLD = 3.0


def mean1(w: Any) -> Any:
    return 2.5 + 0.6 * np.asarray(w, dtype=float)


def p_l2(w: Any, a1: Any) -> Any:
    return expit(-1.0 + 0.35 * np.asarray(a1, dtype=float) - 0.4 * np.asarray(w, dtype=float))


def mean2(w: Any, a1: Any, l2: Any) -> Any:
    return (
        1.2
        + 0.45 * np.asarray(a1, dtype=float)
        + 0.5 * np.asarray(l2, dtype=float)
        - 0.2 * np.asarray(w, dtype=float)
    )


def outcome_mean(w: Any, a1: Any, l2: Any, a2: Any) -> Any:
    a2 = np.asarray(a2, dtype=float)
    return expit(
        -2.2
        + 0.25 * np.asarray(a1, dtype=float)
        + 0.45 * a2
        - 0.03 * a2**2
        + 0.5 * np.asarray(l2, dtype=float)
        + 0.3 * np.asarray(w, dtype=float)
    )


def truncated_pdf(a: Any, mean: Any) -> Any:
    """The normal density truncated to ``[LOWER, UPPER]``."""
    a = np.asarray(a, dtype=float)
    mean = np.asarray(mean, dtype=float)
    mass = norm.cdf((UPPER - mean) / SD) - norm.cdf((LOWER - mean) / SD)
    inside = (a >= LOWER) & (a <= UPPER)
    return np.where(inside, norm.pdf((a - mean) / SD) / (SD * mass), 0.0)


def truncated_draw(rng: np.random.Generator, mean: Any) -> np.ndarray:
    """One truncated-normal draw per mean, by the inverse distribution function."""
    mean = np.asarray(mean, dtype=float)
    low = norm.cdf((LOWER - mean) / SD)
    high = norm.cdf((UPPER - mean) / SD)
    u = rng.uniform(low, high)
    return np.clip(mean + SD * norm.ppf(u), LOWER, UPPER)


def sample_continuous(n: int, seed: int, *, noise: bool = False) -> pd.DataFrame:
    """``n`` rows of the continuous law."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    a1 = truncated_draw(rng, mean1(w))
    l2 = rng.binomial(1, p_l2(w, a1)).astype(float)
    a2 = truncated_draw(rng, mean2(w, a1, l2))
    y = rng.binomial(1, outcome_mean(w, a1, l2, a2)).astype(float)
    frame = pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y})
    if noise:
        frame.insert(1, "U", rng.normal(size=n))
    return frame


# ---------------------------------------------------------------------------- policies


def _up(a: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return np.where(a + STEP > CAP, a, a + STEP)


def _scale(a: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return np.where(a * FACTOR > CAP, a, a * FACTOR)


def _identity(a: Any) -> Any:
    return np.asarray(a, dtype=float)


def _history(a2: Any, a1: Any) -> Any:
    a2 = np.asarray(a2, dtype=float)
    moves = (np.asarray(a1, dtype=float) > HISTORY_THRESHOLD) & (a2 + STEP <= CAP)
    return np.where(moves, a2 + STEP, a2)


#: Each continuous plan as its two node maps, ``d1(a1)`` and ``d2(a2, a1)`` with ``a1`` the
#: intervened earlier dose.
CONTINUOUS_MAPS: dict[str, tuple[Callable[[Any], Any], Callable[[Any, Any], Any]]] = {
    "natural": (_identity, lambda a2, a1: _identity(a2)),
    "up": (_up, lambda a2, a1: _up(a2)),
    "scale at 2": (_identity, lambda a2, a1: _scale(a2)),
    "up then history": (_up, _history),
}
CONTINUOUS_LABELS = tuple(CONTINUOUS_MAPS)
#: Every dose at which an integrand of :func:`continuous_truth` jumps.
_BREAKS = (CAP - STEP, CAP / FACTOR, HISTORY_THRESHOLD, HISTORY_THRESHOLD - STEP)


def _boundary(h: Any) -> np.ndarray:
    """``u(h)``: the history policy's moving piece ends at ``CAP - STEP`` where A1 is high."""
    return np.where(np.asarray(h["A1"], dtype=float) > HISTORY_THRESHOLD, CAP - STEP, LOWER - 1.0)


def history_map(a: Any, h: Any) -> Any:
    """The history policy's moving piece, ``a + STEP``."""
    return np.asarray(a, dtype=float) + STEP


def history_inverse(b: Any, h: Any) -> Any:
    """The inverse of :func:`history_map`."""
    return np.asarray(b, dtype=float) - STEP


def history_derivative(b: Any, h: Any) -> Any:
    """The derivative of :func:`history_inverse`."""
    return np.ones_like(np.asarray(b, dtype=float))


HISTORY_POLICY = ModifiedPolicy(
    "history",
    pieces=(
        Piece(-np.inf, _boundary, history_map, history_inverse, history_derivative),
        Piece(_boundary, np.inf),
    ),
    policy_kind="known",
)
NATURAL = Shift(0.0, cap=None)
UP = Shift(STEP, cap=CAP)
SCALE = Scale(FACTOR, cap=CAP)


def continuous_regimens() -> dict[str, Any]:
    """The continuous plans as policy objects, in :data:`CONTINUOUS_LABELS` order."""
    return {
        "natural": NATURAL,
        "up": UP,
        "scale at 2": DynamicRegimen("scale at 2", (NATURAL, SCALE)),
        "up then history": DynamicRegimen("up then history", (UP, HISTORY_POLICY)),
    }


def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes and weights on ``[LOWER, UPPER]``, split at the breaks."""
    points = sorted({LOWER, UPPER, *(b for b in _BREAKS if LOWER < b < UPPER)})
    base, weights = np.polynomial.legendre.leggauss(order)
    nodes, scaled = [], []
    for low, high in itertools.pairwise(points):
        half = 0.5 * (high - low)
        nodes.append(low + half * (base + 1.0))
        scaled.append(half * weights)
    return np.concatenate(nodes), np.concatenate(scaled)


def _continuous_mean(label: str, order: int) -> float:
    d1, d2 = CONTINUOUS_MAPS[label]
    nodes, weights = _gauss_legendre(order)
    psi = 0.0
    for w in (0.0, 1.0):
        a1d = d1(nodes)
        inner = np.zeros_like(nodes)
        for l2 in (0.0, 1.0):
            p = p_l2(w, a1d) if l2 else 1.0 - p_l2(w, a1d)
            a2 = nodes[None, :]
            a2d = d2(a2, a1d[:, None])
            density = truncated_pdf(a2, mean2(w, a1d[:, None], l2))
            q = outcome_mean(w, a1d[:, None], l2, a2d)
            inner = inner + p * np.sum(weights[None, :] * density * q, axis=1)
        psi += 0.5 * float(np.sum(weights * truncated_pdf(nodes, mean1(w)) * inner))
    return psi


@cache
def continuous_truth(label: str) -> float:
    """``E[Y]`` under a continuous plan, by quadrature checked against twice the order."""
    value = _continuous_mean(label, 96)
    check = _continuous_mean(label, 192)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} moved by {abs(value - check):.3g}")
    return value


# ---------------------------------------------------------------------------- finite laws


@dataclass(frozen=True)
class FiniteLaw:
    """A two-node law with ``K`` levels per node, every cell a multiple of ``1 / N``.

    Parameters
    ----------
    name : str
        The law's name.
    levels : tuple of int
        The treatment levels.
    p_w : ndarray
        ``P(W = w)``.
    g1 : ndarray
        ``g_1(a | w)``, indexed ``[w, a]``.
    p_l2 : ndarray
        ``P(L2 = 1 | w, a1)``.
    g2 : ndarray
        ``g_2(a | w, a1, l2)``, indexed ``[w, a1, l2, a]``.
    q : ndarray
        ``E[Y | w, a1, l2, a2]``.
    """

    name: str
    levels: tuple[int, ...]
    p_w: np.ndarray
    g1: np.ndarray
    p_l2: np.ndarray
    g2: np.ndarray
    q: np.ndarray

    @property
    def support(self) -> tuple[tuple[int, int, int, int, int], ...]:
        k = len(self.levels)
        return tuple(itertools.product(range(2), range(k), range(2), range(k), range(2)))

    def cell(self, w: int, a1: int, l2: int, a2: int, y: int) -> float:
        p_l = self.p_l2[w, a1] if l2 else 1.0 - self.p_l2[w, a1]
        p_y = self.q[w, a1, l2, a2] if y else 1.0 - self.q[w, a1, l2, a2]
        return float(self.p_w[w] * self.g1[w, a1] * p_l * self.g2[w, a1, l2, a2] * p_y)

    @property
    def probs(self) -> np.ndarray:
        values = np.array([self.cell(*point) for point in self.support])
        if abs(values.sum() - 1.0) > 1e-12:
            raise AssertionError(f"the {self.name} law does not sum to one")
        return values

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cells = rng.choice(len(self.support), size=n, p=self.probs)
        points = np.asarray(self.support, dtype=int)[cells]
        levels = np.asarray(self.levels)
        return pd.DataFrame(
            {
                "W": points[:, 0].astype(float),
                "A1": levels[points[:, 1]],
                "L2": points[:, 2].astype(float),
                "A2": levels[points[:, 3]],
                "Y": points[:, 4].astype(float),
            }
        )

    def mean(self, plan: tuple[Any, Any]) -> float:
        """``E[Y]`` under a plan of ``(probability, map)`` branches per node.

        A node-1 map is ``[w][a1]`` and a node-2 map ``[w][a1][l2][a2]``, each entry a level
        index; node 2 reads the intervened earlier level.
        """
        k = len(self.levels)
        psi = 0.0
        for w in range(2):
            for a1 in range(k):
                for p_e, map1 in plan[0]:
                    a1d = map1[w][a1]
                    for l2 in range(2):
                        p_l = self.p_l2[w, a1d] if l2 else 1.0 - self.p_l2[w, a1d]
                        for a2 in range(k):
                            for p_f, map2 in plan[1]:
                                a2d = map2[w][a1d][l2][a2]
                                psi += (
                                    self.p_w[w]
                                    * self.g1[w, a1]
                                    * p_e
                                    * p_l
                                    * self.g2[w, a1d, l2, a2]
                                    * p_f
                                    * self.q[w, a1d, l2, a2d]
                                )
        return float(psi)


def _rotated(base: list[float], shift: int) -> np.ndarray:
    return np.roll(np.asarray(base, dtype=float), shift)


def _categorical_law() -> FiniteLaw:
    base = [4.0, 4.0, 3.0, 3.0, 3.0, 3.0]
    g1 = np.array([_rotated(base, w) for w in range(2)]) / 20.0
    p_l2 = np.array([[0.25 + 0.1 * ((a + w) % 3) for a in range(6)] for w in range(2)])
    g2 = np.array(
        [
            [[_rotated(base, w + a1 + 2 * l2) / 20.0 for l2 in range(2)] for a1 in range(6)]
            for w in range(2)
        ]
    )
    q = np.array(
        [
            [
                [
                    [0.15 + 0.1 * a2 + 0.04 * a1 + 0.05 * l2 + 0.05 * w for a2 in range(6)]
                    for l2 in range(2)
                ]
                for a1 in range(6)
            ]
            for w in range(2)
        ]
    )
    return FiniteLaw("categorical", tuple(range(6)), np.array([0.5, 0.5]), g1, p_l2, g2, q)


def _binary_law() -> FiniteLaw:
    g1 = np.array([[0.65, 0.35], [0.4, 0.6]])
    p_l2 = np.array([[0.3, 0.55], [0.45, 0.7]])
    g2 = np.array(
        [
            [[[0.6, 0.4], [0.35, 0.65]], [[0.5, 0.5], [0.3, 0.7]]],
            [[[0.55, 0.45], [0.4, 0.6]], [[0.45, 0.55], [0.25, 0.75]]],
        ]
    )
    q = np.array(
        [
            [[[0.25, 0.4], [0.35, 0.55]], [[0.3, 0.5], [0.45, 0.65]]],
            [[[0.3, 0.45], [0.4, 0.6]], [[0.35, 0.6], [0.5, 0.75]]],
        ]
    )
    return FiniteLaw("binary", (0, 1), np.array([0.5, 0.5]), g1, p_l2, g2, q)


CATEGORICAL_LAW = _categorical_law()
BINARY_LAW = _binary_law()


def _node1(law: FiniteLaw, assign: Callable[[int, int], int]) -> Any:
    return tuple(tuple(assign(w, a) for a in range(len(law.levels))) for w in range(2))


def _node2(law: FiniteLaw, assign: Callable[[int, int, int, int], int]) -> Any:
    k = len(law.levels)
    return tuple(
        tuple(
            tuple(tuple(assign(w, a1, l2, a) for a in range(k)) for l2 in range(2))
            for a1 in range(k)
        )
        for w in range(2)
    )


def _minus_one(a: int) -> int:
    """``(a - 1) * (a - 1 >= 1) + a * (a - 1 < 1)``, lmtp's Example 2.1 policy."""
    return a - 1 if a - 1 >= 1 else a


_LAW = CATEGORICAL_LAW
CATEGORICAL_PLANS: dict[str, tuple[Any, Any]] = {
    "natural": (
        ((1.0, _node1(_LAW, lambda w, a: a)),),
        ((1.0, _node2(_LAW, lambda w, a1, l, a: a)),),
    ),
    "minus one": (
        ((1.0, _node1(_LAW, lambda w, a: _minus_one(a))),),
        ((1.0, _node2(_LAW, lambda w, a1, l, a: _minus_one(a))),),
    ),
    "gated": (
        ((1.0, _node1(_LAW, lambda w, a: a)),),
        ((1.0, _node2(_LAW, lambda w, a1, l, a: _minus_one(a) if l == 1 else a)),),
    ),
}
CATEGORICAL_LABELS = tuple(CATEGORICAL_PLANS)

#: ``RiskRatioTilt(0.5)``: keep the treatment with probability one half, else set it to 0.
_KEEP1 = _node1(BINARY_LAW, lambda w, a: a)
_ZERO1 = _node1(BINARY_LAW, lambda w, a: 0)
_KEEP2 = _node2(BINARY_LAW, lambda w, a1, l, a: a)
_ZERO2 = _node2(BINARY_LAW, lambda w, a1, l, a: 0)
TILT_PLANS: dict[str, tuple[Any, Any]] = {
    "natural": (((1.0, _KEEP1),), ((1.0, _KEEP2),)),
    "rr 0.5": (((0.5, _KEEP1), (0.5, _ZERO1)), ((0.5, _KEEP2), (0.5, _ZERO2))),
}
TILT_LABELS = tuple(TILT_PLANS)


def _minus_one_labels(a: Any, h: Any) -> Any:
    values = np.asarray(a, dtype=float)
    return np.where(values - 1 >= 1, values - 1, values)


def _gated_labels(a: Any, h: Any) -> Any:
    values = np.asarray(a, dtype=float)
    gate = np.asarray(h["L2"], dtype=float) == 1.0
    return np.where(gate & (values - 1 >= 1), values - 1, values)


MINUS_ONE = ModifiedPolicy("minus one", apply=_minus_one_labels, policy_kind="known")
GATED = ModifiedPolicy("gated", apply=_gated_labels, policy_kind="known")


def categorical_regimens() -> dict[str, Any]:
    return {
        "natural": NATURAL,
        "minus one": MINUS_ONE,
        "gated": DynamicRegimen("gated", (NATURAL, GATED)),
    }


def tilt_regimens() -> dict[str, Any]:
    return {"natural": RiskRatioTilt(1.0), "rr 0.5": RiskRatioTilt(0.5)}


# ---------------------------------------------------------------------------- learners


class OracleDoseHazard(BaseEstimator):
    """The exact pooled-hazard probabilities of the continuous law's dose, per node.

    The density layer hands this the history design followed by the bin block, the bin index
    and its drop-first indicators.  The history is ``[W]`` at node 1 and ``[W, L2, A1]`` at
    node 2, so the width tells the node.  The hazard of bin ``b`` is the conditional
    probability that the dose stops in ``b`` given it reached ``b``, under the truncated
    normal of :func:`mean1` or :func:`mean2`.  ``marginal=True`` is the misspecified
    mechanism: the same family at the marginal mean, which ignores the history.

    Parameters
    ----------
    edges1 : tuple of float
        The bin edges of node 1.
    edges2 : tuple of float
        The bin edges of node 2.
    marginal : bool
        Whether to ignore the history.
    extra : int
        Leading noise columns the history carries before ``W``, read past.
    """

    def __init__(
        self,
        edges1: tuple[float, ...],
        edges2: tuple[float, ...],
        marginal: bool = False,
        extra: int = 0,
    ) -> None:
        self.edges1 = edges1
        self.edges2 = edges2
        self.marginal = marginal
        self.extra = extra

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleDoseHazard:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        bins = len(self.edges1) - 1
        width = design.shape[1] - (bins - 1)
        history = design[:, :width]
        index = np.rint(design[:, width]).astype(int)
        if width == 1 + self.extra:
            edges = np.asarray(self.edges1, dtype=float)
            w = history[:, self.extra]
            mean = np.full(len(w), 2.8) if self.marginal else mean1(w)
        elif width == 3 + self.extra:
            edges = np.asarray(self.edges2, dtype=float)
            w, l2, a1 = (history[:, self.extra + k] for k in range(3))
            mean = np.full(len(w), 2.6) if self.marginal else mean2(w, a1, l2)
        else:
            raise ValueError(f"unexpected oracle design width {design.shape[1]}")
        last = len(edges) - 2
        lower = np.where(index == 0, LOWER, edges[index])
        upper = np.where(index == last, UPPER, edges[np.minimum(index + 1, last + 1)])
        mass = norm.cdf((UPPER - mean) / SD) - norm.cdf((LOWER - mean) / SD)
        below = (norm.cdf((lower - mean) / SD) - norm.cdf((LOWER - mean) / SD)) / mass
        reached = (norm.cdf((upper - mean) / SD) - norm.cdf((LOWER - mean) / SD)) / mass
        survived = 1.0 - below
        stopped = reached - below
        hazard = np.divide(stopped, survived, out=np.ones_like(stopped), where=survived > 1e-15)
        hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - hazard, hazard])


class CorrectOutcome(BaseEstimator):
    """The final node's correctly specified logistic model, ``[W, L2, A1, A2, A2^2]``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CorrectOutcome:
        self.model_ = QuasiBinomialGLM().fit(self._design(X), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return np.column_stack([matrix, matrix[:, -1] ** 2])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class RichPseudo(BaseEstimator):
    """Least squares on a rich basis of ``[W, A1]``, for the first node's pseudo-outcome.

    The first node's regression is an integral of the final one and has no closed form.  A
    degree-five polynomial in the dose, its interactions with ``W``, and a step at the history
    policy's threshold, fitted by least squares, approximates it to well below the sampling
    error at the study's sizes.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RichPseudo:
        self.model_ = LinearRegression().fit(self._design(X), y, sample_weight=sample_weight)
        return self

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w, a = matrix[:, 0], matrix[:, -1]
        step = (a > HISTORY_THRESHOLD - STEP).astype(float)
        high = (a > HISTORY_THRESHOLD).astype(float)
        powers = [a**k for k in range(1, 6)]
        return np.column_stack([w, *powers, *(w * p for p in powers[:3]), step, high, w * step])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)


def continuous_learners(
    configuration: str, edges: tuple[Any, Any], *, extra: int = 0
) -> tuple[Any, Any, Any]:
    """``(outcome, pseudo, dose hazard)`` of one nuisance configuration."""
    edges1, edges2 = (tuple(float(v) for v in e) for e in edges)
    oracle = OracleDoseHazard(edges1, edges2, extra=extra)
    if configuration == "primary":
        return QuasiBinomialGLM(), LinearRegression(), oracle
    if configuration in {"overfit_crossfit", "overfit_control"}:
        return (
            DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
            DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            oracle,
        )
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    outcome = CorrectOutcome() if q_correct else DummyClassifier(strategy="prior")
    pseudo = RichPseudo() if q_correct else DummyRegressor(strategy="mean")
    hazard = oracle if g_correct else OracleDoseHazard(edges1, edges2, marginal=True, extra=extra)
    return outcome, pseudo, hazard


# ---------------------------------------------------------------------------- efficiency bounds
#
# The efficiency bound of a calibration label is the standard deviation of its efficient
# influence function under the law.  On the continuous law it is computed from the closed-form
# nuisances, never from a fit: the final regression is ``outcome_mean``, the first is
# :func:`first_regression` by quadrature, and each node's ratio is Equation (3) of Díaz et al.
# (2023) on the truncated-normal density.  The standard deviation is a Monte Carlo average over
# :data:`BOUND_DRAWS` draws, whose relative error is far below the study's SE-ratio band.

BOUND_DRAWS = 400_000
BOUND_SEED = 20261049
#: Plans used only by property cells: the node-2 shift alone, for the randomized policy and
#: the working model.
CONTINUOUS_MAPS["up at 2"] = (_identity, lambda a2, a1: _up(a2))


def _node_ratio(
    label: str, node: int, a: np.ndarray, mean: np.ndarray, earlier: np.ndarray
) -> np.ndarray:
    """Equation (3) on the truncated normal: the ratio of node ``node`` of a plan at dose ``a``."""
    g = truncated_pdf(a, mean)
    safe = np.where(g > 0.0, g, 1.0)
    kind = {
        ("natural", 1): "identity",
        ("natural", 2): "identity",
        ("up", 1): "up",
        ("up", 2): "up",
        ("scale at 2", 1): "identity",
        ("scale at 2", 2): "scale",
        ("up then history", 1): "up",
        ("up then history", 2): "history",
        ("up at 2", 1): "identity",
        ("up at 2", 2): "up",
    }[(label, node)]
    if kind == "identity":
        return np.ones_like(a)
    if kind == "up":
        moved = truncated_pdf(a - STEP, mean) * (a <= CAP)
        return np.where(g > 0.0, moved / safe, 0.0) + (a + STEP > CAP)
    if kind == "scale":
        moved = truncated_pdf(a / FACTOR, mean) / FACTOR * (a <= CAP)
        return np.where(g > 0.0, moved / safe, 0.0) + (a * FACTOR > CAP)
    shifted = np.where(g > 0.0, truncated_pdf(a - STEP, mean) * (a <= CAP) / safe, 0.0) + (
        a + STEP > CAP
    )
    return np.where(earlier > HISTORY_THRESHOLD, shifted, 1.0)


def first_regression(label: str, w: np.ndarray, a1: np.ndarray, order: int = 96) -> np.ndarray:
    """``m_1(a1, w) = E[m_2(d_2(A2, H_2), H_2) | A1 = a1, W = w]`` by quadrature, per row."""
    _, d2 = CONTINUOUS_MAPS[label]
    nodes, weights = _gauss_legendre(order)
    out = np.zeros(len(a1))
    for l2 in (0.0, 1.0):
        p = p_l2(w, a1) if l2 else 1.0 - p_l2(w, a1)
        a2 = nodes[None, :]
        density = truncated_pdf(a2, mean2(w[:, None], a1[:, None], l2))
        q = outcome_mean(w[:, None], a1[:, None], l2, d2(a2, a1[:, None]))
        out = out + p * np.sum(weights[None, :] * density * q, axis=1)
    return out


def continuous_eif(label: str, frame: pd.DataFrame) -> np.ndarray:
    """The efficient influence function of ``E[Y]`` under a continuous plan, at every row."""
    d1, d2 = CONTINUOUS_MAPS[label]
    w = frame["W"].to_numpy(dtype=float)
    a1 = frame["A1"].to_numpy(dtype=float)
    l2 = frame["L2"].to_numpy(dtype=float)
    a2 = frame["A2"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    r1 = _node_ratio(label, 1, a1, mean1(w), a1)
    r2 = _node_ratio(label, 2, a2, mean2(w, a1, l2), a1)
    m2_observed = outcome_mean(w, a1, l2, a2)
    m2_policy = outcome_mean(w, a1, l2, d2(a2, a1))
    m1_observed = first_regression(label, w, a1)
    m1_policy = first_regression(label, w, d1(a1))
    psi = continuous_truth(label)
    return m1_policy - psi + r1 * (m2_policy - m1_observed) + r1 * r2 * (y - m2_observed)


@cache
def continuous_bound_sample() -> pd.DataFrame:
    return sample_continuous(BOUND_DRAWS, BOUND_SEED)


@cache
def continuous_eif_draws(label: str) -> np.ndarray:
    return continuous_eif(label, continuous_bound_sample())


def continuous_contrast_sd(label: str, reference: str = "natural") -> float:
    """The efficiency bound of ``E[Y^label] - E[Y^reference]``, as a standard deviation."""
    return float(np.std(continuous_eif_draws(label) - continuous_eif_draws(reference)))


def finite_functional(law: FiniteLaw, probs: Any, plan: tuple[Any, Any]) -> Any:
    """:meth:`FiniteLaw.mean` written over cell probabilities, for the complex step."""
    k = len(law.levels)
    cells = np.asarray(probs).reshape(2, k, 2, k, 2)
    total = cells.sum()
    psi: Any = 0.0
    for w in range(2):
        p_w = cells[w].sum() / total
        for a1 in range(k):
            g1 = cells[w, a1].sum() / cells[w].sum()
            for p_e, map1 in plan[0]:
                a1d = map1[w][a1]
                reached = cells[w, a1d].sum()
                for l2 in range(2):
                    stratum = cells[w, a1d, l2].sum()
                    p_l = stratum / reached
                    for a2 in range(k):
                        g2 = cells[w, a1d, l2, a2].sum() / stratum
                        for p_f, map2 in plan[1]:
                            a2d = map2[w][a1d][l2][a2]
                            q = cells[w, a1d, l2, a2d, 1] / cells[w, a1d, l2, a2d].sum()
                            psi = psi + p_w * g1 * p_e * p_l * g2 * p_f * q
    return psi


def finite_contrast_sd(law: FiniteLaw, plan: tuple[Any, Any], reference: tuple[Any, Any]) -> float:
    """The efficiency bound of a contrast of two plans on a finite law, by complex step."""
    probs = law.probs
    curve = np.zeros(len(probs))
    for point in range(len(probs)):
        base = probs.astype(complex)
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * 1e-30) * base + 1j * 1e-30 * mass
        value = finite_functional(law, perturbed, plan) - finite_functional(
            law, perturbed, reference
        )
        curve[point] = float(np.imag(value) / 1e-30)
    return float(np.sqrt(np.sum(probs * curve**2)))

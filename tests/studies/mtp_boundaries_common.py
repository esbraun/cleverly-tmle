"""Laws, plans, truths and learners of the ``mtp-boundaries`` study.

**Law B.**  Two nodes, each dose on a support that depends on the history:

.. code-block:: text

    W1 ~ N(0, 1) truncated to [-2.5, 2.5];  W2 ~ Bernoulli(1/2);  u = 6 + 2 W2
    A1 | W ~ N(u - 1.6 + 0.4 W1, 1.2^2) truncated to [0, u]
    L2 | W, A1 ~ Bernoulli(expit(-0.8 + 0.3 A1 - 0.5 W1))
    A2 | W, A1, L2 ~ N(u - 1.9 + 0.3 (A1 - u + 1.6) + 0.6 L2 - 0.3 W1, 1.2^2) truncated to [0, u]
    Y | ... ~ Bernoulli(expit(-2 + 0.2 A1 + 0.4 A2 - 0.03 A2^2 + 0.5 L2 + 0.3 W1))

1.2 is a standard deviation.  Every outcome probability lies in ``[0.060, 0.899]``, because W1
is bounded.  **Law B'** adds an atom at node 2: ``A2 = 3`` (a list price) with probability
``p(h) = expit(-1.4 + 0.4 L2 + 0.2 W1)``, in ``[0.130, 0.378]``, and otherwise the truncated
normal of law B.  The declared support of each dose is ``Support(0, u(h))``
(:data:`SUPPORT`).  Every draw is an inverse-distribution-function draw, never a clip, so no
dose is tied with another except the declared atom.

**Plans** (each at both nodes unless stated).

=========  ==========================================================================
label      map
=========  ==========================================================================
natural    ``Shift(0, cap=None)``
p1         ``Scale(0.9, cap=None)``: nonbinding, maps ``[0, u]`` into itself
p2         ``a + 1`` on ``[0, u - 1)`` and held on ``[u - 1, u]``: the boundary binds
p3         ``u (a / u)^0.8``: one piece with a Jacobian that varies with the dose
p4         ``a + 1`` on ``[0, 3)`` and ``a - 1.5`` on ``[3, u]``: two inverse branches
p5         law B', node 2 only: ``max(3, 0.65 a)``, the interval ``[0, 3 / 0.65)``
           sent to the declared atom 3 (``Piece(..., to_atom=3)``)
=========  ==========================================================================

**Truths.**  The g-formula of Díaz, Williams, Hoffman and Schenck (2023), Theorem 1, with each
node's natural dose drawn given the intervened history (Definition 1), by composite
Gauss-Legendre quadrature over ``W1``, ``A1`` and ``A2``, split at every break of a map and with
the first segment of each dose graded (``a = b t^5``) so that the ``a^0.8`` of p3 is smooth in
``t``.  The atom adds a finite term.  :func:`truth` checks the quadrature against twice the
order (agreement below ``1e-10``).

**Efficiency bounds.**  The efficient influence function of Díaz et al. (2023), Theorem 2, is
evaluated from the closed-form nuisances (:func:`eif`): the final regression is
:func:`outcome_mean`, the first is :func:`first_regression` by quadrature, and each ratio is
Equation (3) on the truncated normal (:func:`node_ratio`).  :data:`BOUNDS` pins the standard
deviation of each contrast over :data:`BOUND_DRAWS` draws; the bound sizes only the noise
control and is not a claim.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from cleverly.interventions import ModifiedPolicy, Piece, Scale, Shift, Support
from cleverly.longitudinal import DynamicRegimen
from tests.studies.fractional_glm import QuasiBinomialGLM

SD = 1.2
W_BOUND = 2.5
ATOM = 3.0
#: The maps' constants.
SCALE_FACTOR = 0.9
STEP = 1.0
POWER = 0.8
BRANCH_SPLIT = 3.0
BRANCH_UP = 1.0
BRANCH_DOWN = 1.5
ATOM_SCALE = 0.65
ATOM_KNEE = ATOM / ATOM_SCALE

REFERENCE = "natural"
#: The plans read on law B, and on law B'.
B_LABELS = ("natural", "p1", "p2", "p3", "p4")
B_ATOM_LABELS = ("natural", "p5")
LAWS = ("B", "B_atom")


# ---------------------------------------------------------------------------- the law


def upper(w2: Any) -> Any:
    """The upper end of the support, ``u = 6 + 2 W2``."""
    return 6.0 + 2.0 * np.asarray(w2, dtype=float)


def mean1(w1: Any, w2: Any) -> Any:
    return upper(w2) - 1.6 + 0.4 * np.asarray(w1, dtype=float)


def p_l2(w1: Any, a1: Any) -> Any:
    return expit(-0.8 + 0.3 * np.asarray(a1, dtype=float) - 0.5 * np.asarray(w1, dtype=float))


def mean2(w1: Any, w2: Any, a1: Any, l2: Any) -> Any:
    u = upper(w2)
    return (
        u
        - 1.9
        + 0.3 * (np.asarray(a1, dtype=float) - u + 1.6)
        + 0.6 * np.asarray(l2, dtype=float)
        - 0.3 * np.asarray(w1, dtype=float)
    )


def p_atom(w1: Any, l2: Any) -> Any:
    """``p(h)``: the probability of the atom ``A2 = 3`` on law B'."""
    return expit(-1.4 + 0.4 * np.asarray(l2, dtype=float) + 0.2 * np.asarray(w1, dtype=float))


def outcome_mean(w1: Any, a1: Any, l2: Any, a2: Any) -> Any:
    a2 = np.asarray(a2, dtype=float)
    return expit(
        -2.0
        + 0.2 * np.asarray(a1, dtype=float)
        + 0.4 * a2
        - 0.03 * a2**2
        + 0.5 * np.asarray(l2, dtype=float)
        + 0.3 * np.asarray(w1, dtype=float)
    )


def tn_mass(mean: Any, top: Any) -> Any:
    return norm.cdf((np.asarray(top) - mean) / SD) - norm.cdf((0.0 - np.asarray(mean)) / SD)


def tn_pdf(a: Any, mean: Any, top: Any) -> Any:
    """The normal density with SD 1.2 truncated to ``[0, top]``."""
    a = np.asarray(a, dtype=float)
    inside = (a >= 0.0) & (a <= top)
    return np.where(inside, norm.pdf((a - mean) / SD) / (SD * tn_mass(mean, top)), 0.0)


def tn_cdf(a: Any, mean: Any, top: Any) -> Any:
    """The distribution function of :func:`tn_pdf`."""
    a = np.clip(np.asarray(a, dtype=float), 0.0, top)
    return (norm.cdf((a - mean) / SD) - norm.cdf(-np.asarray(mean) / SD)) / tn_mass(mean, top)


def tn_draw(rng: np.random.Generator, mean: Any, top: Any) -> np.ndarray:
    """One draw per row, by the inverse distribution function (no clip)."""
    mean = np.asarray(mean, dtype=float)
    low = norm.cdf(-mean / SD)
    high = norm.cdf((np.asarray(top) - mean) / SD)
    return np.asarray(mean + SD * norm.ppf(rng.uniform(low, high)), dtype=float)


def w1_draw(rng: np.random.Generator, n: int) -> np.ndarray:
    low, high = norm.cdf(-W_BOUND), norm.cdf(W_BOUND)
    return np.asarray(norm.ppf(rng.uniform(low, high, n)), dtype=float)


def w1_pdf(w1: Any) -> Any:
    return norm.pdf(w1) / (norm.cdf(W_BOUND) - norm.cdf(-W_BOUND))


def sample(n: int, seed: int, *, atom: bool = False) -> pd.DataFrame:
    """``n`` rows of law B, or of law B' with ``atom=True``."""
    rng = np.random.default_rng(seed)
    w1 = w1_draw(rng, n)
    w2 = rng.integers(0, 2, n).astype(float)
    u = upper(w2)
    a1 = tn_draw(rng, mean1(w1, w2), u)
    l2 = rng.binomial(1, p_l2(w1, a1)).astype(float)
    a2 = tn_draw(rng, mean2(w1, w2, a1, l2), u)
    if atom:
        a2 = np.where(rng.random(n) < p_atom(w1, l2), ATOM, a2)
    y = rng.binomial(1, outcome_mean(w1, a1, l2, a2)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A1": a1, "L2": l2, "A2": a2, "Y": y})


def sample_b(n: int, seed: int) -> pd.DataFrame:
    return sample(n, seed, atom=False)


def sample_b_atom(n: int, seed: int) -> pd.DataFrame:
    return sample(n, seed, atom=True)


SAMPLERS: dict[str, Callable[[int, int], pd.DataFrame]] = {"B": sample_b, "B_atom": sample_b_atom}


# ---------------------------------------------------------------------------- the plans


def _upper_of(h: Any) -> np.ndarray:
    return np.asarray(upper(np.asarray(h["W2"], dtype=float)), dtype=float)


def _knee(h: Any) -> np.ndarray:
    return _upper_of(h) - STEP


def _p3_map(a: Any, h: Any) -> Any:
    u = _upper_of(h)
    return u * (np.clip(np.asarray(a, dtype=float), 0.0, None) / u) ** POWER


def _p3_inverse(b: Any, h: Any) -> Any:
    u = _upper_of(h)
    return u * (np.clip(np.asarray(b, dtype=float), 0.0, None) / u) ** (1.0 / POWER)


def _p3_derivative(b: Any, h: Any) -> Any:
    u = _upper_of(h)
    return (1.0 / POWER) * (np.clip(np.asarray(b, dtype=float), 0.0, None) / u) ** (
        1.0 / POWER - 1.0
    )


def _plus(delta: float) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    return (
        lambda a, h: np.asarray(a, dtype=float) + delta,
        lambda b, h: np.asarray(b, dtype=float) - delta,
        lambda b, h: np.ones_like(np.asarray(b, dtype=float)),
    )


#: The declared support of each dose: ``[0, 6 + 2 W2]``.
SUPPORT = Support(0.0, _upper_of)
NATURAL = Shift(0.0, cap=None)
P1 = Scale(SCALE_FACTOR, cap=None, name="p1")
P2 = ModifiedPolicy(
    "p2",
    pieces=(Piece(-np.inf, _knee, *_plus(STEP)), Piece(_knee, np.inf)),
    policy_kind="known",
)
P3 = ModifiedPolicy(
    "p3",
    pieces=(Piece(-np.inf, np.inf, _p3_map, _p3_inverse, _p3_derivative),),
    policy_kind="known",
)
P4 = ModifiedPolicy(
    "p4",
    pieces=(
        Piece(-np.inf, BRANCH_SPLIT, *_plus(BRANCH_UP)),
        Piece(BRANCH_SPLIT, np.inf, *_plus(-BRANCH_DOWN)),
    ),
    policy_kind="known",
)
P5_NODE = ModifiedPolicy(
    "p5",
    pieces=(
        Piece(-np.inf, ATOM_KNEE, to_atom=ATOM),
        Piece(
            ATOM_KNEE,
            np.inf,
            lambda a, h: ATOM_SCALE * np.asarray(a, dtype=float),
            lambda b, h: np.asarray(b, dtype=float) / ATOM_SCALE,
            lambda b, h: np.full_like(np.asarray(b, dtype=float), 1.0 / ATOM_SCALE),
        ),
    ),
    policy_kind="known",
)


def regimens(labels: tuple[str, ...]) -> dict[str, Any]:
    """The plans as policy objects, in ``labels`` order."""
    every = {
        "natural": NATURAL,
        "p1": P1,
        "p2": P2,
        "p3": P3,
        "p4": P4,
        "p5": DynamicRegimen("p5", (NATURAL, P5_NODE)),
    }
    return {label: every[label] for label in labels}


# The same maps as arrays, for the truth and the closed-form ratios.  ``u`` is per row.


def _identity(a: Any, u: Any) -> Any:
    return np.asarray(a, dtype=float)


def _scale(a: Any, u: Any) -> Any:
    return SCALE_FACTOR * np.asarray(a, dtype=float)


def _bind(a: Any, u: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return np.where(a < u - STEP, a + STEP, a)


def _power(a: Any, u: Any) -> Any:
    return u * (np.clip(np.asarray(a, dtype=float), 0.0, None) / u) ** POWER


def _branches(a: Any, u: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return np.where(a < BRANCH_SPLIT, a + BRANCH_UP, a - BRANCH_DOWN)


def _to_atom(a: Any, u: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return np.where(a < ATOM_KNEE, ATOM, ATOM_SCALE * a)


#: Each plan as its two node maps ``d_t(a, u)``.
MAPS: dict[str, tuple[Callable[[Any, Any], Any], Callable[[Any, Any], Any]]] = {
    "natural": (_identity, _identity),
    "p1": (_scale, _scale),
    "p2": (_bind, _bind),
    "p3": (_power, _power),
    "p4": (_branches, _branches),
    "p5": (_identity, _to_atom),
}
LAW_OF: dict[str, str] = {"p1": "B", "p2": "B", "p3": "B", "p4": "B", "p5": "B_atom"}


def _breaks(u: float) -> tuple[float, ...]:
    """Every dose at which a map of any plan jumps or changes form, inside ``(0, u)``."""
    return tuple(sorted(b for b in {u - STEP, BRANCH_SPLIT, ATOM_KNEE} if 0.0 < b < u))


# ---------------------------------------------------------------------------- quadrature


def _graded(order: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre on ``[low, high]`` after ``a = low + (high - low) t^5``."""
    base, weights = np.polynomial.legendre.leggauss(order)
    t = 0.5 * (base + 1.0)
    span = high - low
    return low + span * t**5, 0.5 * weights * span * 5.0 * t**4


def _plain(order: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    base, weights = np.polynomial.legendre.leggauss(order)
    half = 0.5 * (high - low)
    return low + half * (base + 1.0), half * weights


@cache
def dose_nodes(
    u: float, order: int, extra: tuple[float, ...] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Composite nodes and weights on ``[0, u]``, split at :func:`_breaks` and ``extra``.

    The first segment is graded.
    """
    points = tuple(sorted({0.0, *_breaks(u), *(b for b in extra if 0.0 < b < u), u}))
    nodes, weights = [], []
    for k, (low, high) in enumerate(itertools.pairwise(points)):
        x, w = (_graded if k == 0 else _plain)(order, low, high)
        nodes.append(x)
        weights.append(w)
    return np.concatenate(nodes), np.concatenate(weights)


@cache
def w1_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = _plain(order, -W_BOUND, W_BOUND)
    return x, w * w1_pdf(x)


def _inner(
    label: str, w1: np.ndarray, w2: float, a1d: np.ndarray, l2: float, order: int, atom: bool
) -> np.ndarray:
    """``E[m_2(H_2, d_2(A_2))]`` over ``A2`` given ``(W, A1 = a1d, L2)``, per ``(w1, a1d)``."""
    _, d2 = MAPS[label]
    u = float(upper(w2))
    a2, weight = dose_nodes(u, order)
    w1b = w1[:, None, None]
    a1b = a1d[..., None]
    density = tn_pdf(a2[None, None, :], mean2(w1b, w2, a1b, l2), u)
    q = outcome_mean(w1b, a1b, l2, d2(a2[None, None, :], u))
    continuous = np.sum(weight * density * q, axis=-1)
    if not atom:
        return continuous
    p = p_atom(w1[:, None], l2)
    at_atom = outcome_mean(w1[:, None], a1d, l2, d2(np.full_like(a1d, ATOM), u))
    return (1.0 - p) * continuous + p * at_atom


def _mean(label: str, law: str, order: int) -> float:
    d1, _ = MAPS[label]
    atom = law == "B_atom"
    w1, weight_w = w1_nodes(order)
    total = 0.0
    for w2 in (0.0, 1.0):
        u = float(upper(w2))
        a1, weight_a = dose_nodes(u, order)
        density = tn_pdf(a1[None, :], mean1(w1[:, None], w2), u)
        a1d = np.broadcast_to(d1(a1, u)[None, :], (w1.size, a1.size))
        inner = np.zeros_like(density)
        for l2 in (0.0, 1.0):
            p = p_l2(w1[:, None], a1d)
            inner = inner + (p if l2 else 1.0 - p) * _inner(label, w1, w2, a1d, l2, order, atom)
        total += 0.5 * float(np.sum(weight_w[:, None] * weight_a[None, :] * density * inner))
    return total


@cache
def truth(label: str, law: str) -> float:
    """``E[Y]`` under a plan on a law, by quadrature checked against twice the order."""
    value = _mean(label, law, 40)
    check = _mean(label, law, 80)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} on {law} moved by {value - check:.3g}")
    return value


def contrast(label: str) -> float:
    """The truth of ``E[Y^label] - E[Y^natural]`` on the plan's law."""
    law = LAW_OF[label]
    return truth(label, law) - truth(REFERENCE, law)


# ---------------------------------------------------------------------------- design numbers


def share_at_least(threshold: Callable[[float], float], node: int, order: int = 80) -> float:
    """``P(A_t >= threshold(u))`` under law B, by quadrature."""
    w1, weight_w = w1_nodes(order)
    total = 0.0
    for w2 in (0.0, 1.0):
        u = float(upper(w2))
        a1, weight_a = dose_nodes(u, order)
        g1 = tn_pdf(a1[None, :], mean1(w1[:, None], w2), u)
        if node == 1:
            tail = 1.0 - tn_cdf(threshold(u), mean1(w1[:, None], w2), u)
            total += 0.5 * float(np.sum(weight_w[:, None] * tail))
            continue
        for l2 in (0.0, 1.0):
            p = p_l2(w1[:, None], a1[None, :])
            tail = 1.0 - tn_cdf(threshold(u), mean2(w1[:, None], w2, a1[None, :], l2), u)
            total += 0.5 * float(
                np.sum(weight_w[:, None] * weight_a[None, :] * g1 * (p if l2 else 1 - p) * tail)
            )
    return total


def mass_where(
    predicate: Callable[[np.ndarray, float], np.ndarray],
    node: int,
    jumps: tuple[float, ...],
    order: int = 80,
) -> float:
    """``P(predicate(A_t, u))`` under law B (natural course), by quadrature.

    ``jumps`` are the doses where the predicate changes, so the quadrature splits there.
    """
    w1, weight_w = w1_nodes(order)
    total = 0.0
    for w2 in (0.0, 1.0):
        u = float(upper(w2))
        a, weight_a = dose_nodes(u, order, jumps)
        hit = predicate(a, u).astype(float)
        g1 = tn_pdf(a[None, :], mean1(w1[:, None], w2), u)
        if node == 1:
            total += 0.5 * float(np.sum(weight_w[:, None] * weight_a[None, :] * g1 * hit))
            continue
        for l2 in (0.0, 1.0):
            p = p_l2(w1[:, None], a[None, :])
            g2 = tn_pdf(a[None, None, :], mean2(w1[:, None, None], w2, a[None, :, None], l2), u)
            inner = np.sum(weight_a * g2 * hit, axis=-1)
            total += 0.5 * float(
                np.sum(weight_w[:, None] * weight_a[None, :] * g1 * (p if l2 else 1 - p) * inner)
            )
    return total


def atom_history_expectation(
    function: Callable[[np.ndarray, np.ndarray], np.ndarray], order: int = 80
) -> float:
    """``E[function(p(h), F_c(3 / 0.65 | h))]`` over the node-2 history of law B'."""
    w1, weight_w = w1_nodes(order)
    total = 0.0
    for w2 in (0.0, 1.0):
        u = float(upper(w2))
        a1, weight_a = dose_nodes(u, order)
        g1 = tn_pdf(a1[None, :], mean1(w1[:, None], w2), u)
        for l2 in (0.0, 1.0):
            p = p_l2(w1[:, None], a1[None, :])
            share = tn_cdf(ATOM_KNEE, mean2(w1[:, None], w2, a1[None, :], l2), u)
            value = function(np.broadcast_to(p_atom(w1[:, None], l2), share.shape), share)
            total += 0.5 * float(
                np.sum(weight_w[:, None] * weight_a[None, :] * g1 * (p if l2 else 1 - p) * value)
            )
    return total


def atom_ratio(p: Any, share: Any) -> Any:
    """The P5 ratio at the atom: ``(p + (1 - p) F_c(3 / 0.65 | h)) / p``."""
    return (p + (1.0 - p) * share) / p


def atom_ratio_supremum(points: int = 401) -> float:
    """The largest P5 atom ratio over the node-2 histories of law B', on a fine grid."""
    best = 0.0
    w1 = np.linspace(-W_BOUND, W_BOUND, points)[:, None]
    for w2 in (0.0, 1.0):
        u = float(upper(w2))
        a1 = np.linspace(0.0, u, points)[None, :]
        for l2 in (0.0, 1.0):
            share = tn_cdf(ATOM_KNEE, mean2(w1, w2, a1, l2), u)
            best = max(best, float(np.max(atom_ratio(p_atom(w1, l2), share))))
    return best


def outcome_bounds() -> tuple[float, float]:
    """The exact range of the outcome probability over the support box."""
    a2 = np.linspace(0.0, 8.0, 80_001)
    quadratic = 0.4 * a2 - 0.03 * a2**2
    low = -2.0 + float(quadratic.min()) - 0.3 * W_BOUND
    high = -2.0 + 0.2 * 8.0 + float(quadratic.max()) + 0.5 + 0.3 * W_BOUND
    return float(expit(low)), float(expit(high))


def atom_probability_bounds() -> tuple[float, float]:
    return float(expit(-1.4 - 0.2 * W_BOUND)), float(expit(-1.4 + 0.4 + 0.2 * W_BOUND))


# ---------------------------------------------------------------------------- closed forms


def induced_density(label: str, node: int, a: Any, mean: Any, u: Any) -> np.ndarray:
    """The density the plan induces at node ``node``, against Lebesgue measure, at ``a``.

    On law B' at node 2 this is the continuous part's own density ``f`` (the factor
    ``1 - p`` of the observed and the induced laws cancels in the ratio).
    """
    a = np.asarray(a, dtype=float)
    kind = MAPS[label][node - 1]
    if kind is _identity:
        return tn_pdf(a, mean, u)
    if kind is _scale:
        return tn_pdf(a / SCALE_FACTOR, mean, u) / SCALE_FACTOR
    if kind is _bind:
        moved = tn_pdf(a - STEP, mean, u) * (a - STEP < u - STEP)
        return moved + tn_pdf(a, mean, u) * (a >= u - STEP)
    if kind is _power:
        source = u * (np.clip(a, 0.0, None) / u) ** (1.0 / POWER)
        slope = (1.0 / POWER) * (np.clip(a, 0.0, None) / u) ** (1.0 / POWER - 1.0)
        return tn_pdf(source, mean, u) * slope * (a <= u)
    if kind is _branches:
        first = tn_pdf(a - BRANCH_UP, mean, u) * (a - BRANCH_UP < BRANCH_SPLIT)
        second = tn_pdf(a + BRANCH_DOWN, mean, u) * (a + BRANCH_DOWN >= BRANCH_SPLIT)
        return first + second
    if kind is _to_atom:
        return tn_pdf(a / ATOM_SCALE, mean, u) / ATOM_SCALE * (a / ATOM_SCALE >= ATOM_KNEE)
    raise KeyError(label)


def node_ratio(label: str, node: int, a: Any, mean: Any, u: Any, p: Any = None) -> np.ndarray:
    """Equation (3) on the truncated normal; with ``p``, the mixed ratio of law B' node 2."""
    a = np.asarray(a, dtype=float)
    g = tn_pdf(a, mean, u)
    safe = np.where(g > 0.0, g, 1.0)
    continuous = np.where(g > 0.0, induced_density(label, node, a, mean, u) / safe, 0.0)
    if p is None or node == 1:
        return continuous
    at_atom = np.abs(a - ATOM) <= 1e-12
    if MAPS[label][1] is _to_atom:
        mass = atom_ratio(p, tn_cdf(ATOM_KNEE, mean, u))
    else:
        mass = np.ones_like(a)
    return np.where(at_atom, mass, continuous)


def first_regression(
    label: str, law: str, w1: np.ndarray, w2: np.ndarray, a1: np.ndarray, order: int = 40
) -> np.ndarray:
    """``m_1(w, a1) = E[m_2(H_2, d_2(A_2)) | A1 = a1, W = w]`` by quadrature, per row."""
    _, d2 = MAPS[label]
    atom = law == "B_atom"
    out = np.zeros(len(a1))
    for value in (0.0, 1.0):
        u = float(upper(value))
        a2, weight = dose_nodes(u, order)
        for rows in np.array_split(np.flatnonzero(w2 == value), max(1, len(a1) // 20_000)):
            if not rows.size:
                continue
            w, a = w1[rows][:, None], a1[rows][:, None]
            for l2 in (0.0, 1.0):
                density = tn_pdf(a2[None, :], mean2(w, value, a, l2), u)
                q = outcome_mean(w, a, l2, d2(a2[None, :], u))
                inner = np.sum(weight * density * q, axis=1)
                if atom:
                    pa = p_atom(w[:, 0], l2)
                    inner = (1 - pa) * inner + pa * outcome_mean(
                        w[:, 0], a[:, 0], l2, d2(np.full(rows.size, ATOM), u)
                    )
                p = p_l2(w[:, 0], a[:, 0])
                out[rows] += (p if l2 else 1 - p) * inner
    return out


def eif(label: str, law: str, frame: pd.DataFrame) -> np.ndarray:
    """The efficient influence function of ``E[Y]`` under a plan, at every row."""
    d1, d2 = MAPS[label]
    w1 = frame["W1"].to_numpy(dtype=float)
    w2 = frame["W2"].to_numpy(dtype=float)
    a1 = frame["A1"].to_numpy(dtype=float)
    l2 = frame["L2"].to_numpy(dtype=float)
    a2 = frame["A2"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    u = upper(w2)
    p = p_atom(w1, l2) if law == "B_atom" else None
    r1 = node_ratio(label, 1, a1, mean1(w1, w2), u)
    r2 = node_ratio(label, 2, a2, mean2(w1, w2, a1, l2), u, p)
    m2_observed = outcome_mean(w1, a1, l2, a2)
    m2_policy = outcome_mean(w1, a1, l2, d2(a2, u))
    m1_observed = first_regression(label, law, w1, w2, a1)
    m1_policy = first_regression(label, law, w1, w2, d1(a1, u))
    psi = truth(label, law)
    return m1_policy - psi + r1 * (m2_policy - m1_observed) + r1 * r2 * (y - m2_observed)


BOUND_DRAWS = 200_000
BOUND_SEED = 20264019


@cache
def bound_sample(law: str) -> pd.DataFrame:
    return SAMPLERS[law](BOUND_DRAWS, BOUND_SEED)


def contrast_sd(label: str, draws: int = BOUND_DRAWS) -> float:
    """The efficiency bound of ``E[Y^label] - E[Y^natural]``, as a standard deviation."""
    law = LAW_OF[label]
    frame = bound_sample(law).iloc[:draws]
    return float(np.std(eif(label, law, frame) - eif(REFERENCE, law, frame)))


# ---------------------------------------------------------------------------- learners


class OracleHazard(BaseEstimator):
    """The exact pooled-hazard probabilities of each dose, and the exact atom probability.

    The density layer hands this the history followed by the bin index (``bin_design =
    "index"``).  The history is ``[W1, W2]`` at node 1 and ``[W1, W2, L2, A1]`` at node 2, so a
    design of width 3 is node 1's hazard and one of width 5 is node 2's.  On law B' the atom
    classifier at node 2 reads the history alone (width 4) with label 1 at the atom, and gets
    ``[1 - p(h), p(h)]``.  The hazard of bin ``b`` is the probability that the dose stops in
    ``b`` given it reached ``b``, under the truncated normal on ``[0, u(h)]``; on law B' that is
    the law of the rows with no atom, which is the same truncated normal.

    Parameters
    ----------
    edges1 : tuple of float
        The bin edges of node 1.
    edges2 : tuple of float
        The bin edges of node 2 (of the rows with no atom, on law B').
    """

    bin_design = "index"

    def __init__(self, edges1: tuple[float, ...] = (), edges2: tuple[float, ...] = ()) -> None:
        self.edges1 = edges1
        self.edges2 = edges2

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleHazard:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        width = design.shape[1]
        if width == 4:
            p = p_atom(design[:, 0], design[:, 2])
            return np.column_stack([1.0 - p, p])
        history, index = design[:, :-1], np.rint(design[:, -1]).astype(int)
        w1, w2 = history[:, 0], history[:, 1]
        u = upper(w2)
        if width == 3:
            edges, mean = np.asarray(self.edges1, dtype=float), mean1(w1, w2)
        elif width == 5:
            edges = np.asarray(self.edges2, dtype=float)
            mean = mean2(w1, w2, history[:, 3], history[:, 2])
        else:
            raise ValueError(f"unexpected oracle design width {width}")
        last = len(edges) - 2
        lower = np.where(index == 0, 0.0, edges[index])
        top = np.where(index == last, np.inf, edges[np.minimum(index + 1, last + 1)])
        below = tn_cdf(lower, mean, u)
        reached = tn_cdf(np.minimum(top, u), mean, u)
        survived = 1.0 - below
        hazard = np.divide(
            reached - below, survived, out=np.ones_like(below), where=survived > 1e-15
        )
        hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - hazard, hazard])


class CorrectOutcome(BaseEstimator):
    """The final node's correctly specified logistic model: the design plus ``A2^2``.

    The design is ``[W1, W2, L2, A1, A2]``, with the atom indicator after it on law B', so
    ``A2`` is column 4 on both laws.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CorrectOutcome:
        self.model_ = QuasiBinomialGLM().fit(self._design(X), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return np.column_stack([matrix, matrix[:, 4] ** 2])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class WrongOutcome(CorrectOutcome):
    """The necessity pairs' outcome model: main terms without ``L2`` and without ``A2^2``."""

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return np.delete(matrix, 2, axis=1)


class RichPseudo(BaseEstimator):
    """Least squares on a rich smooth basis of ``[W1, W2, A1]``, for the first node.

    The first node's regression has no closed form, and every plan's map at node 2 reads only
    ``u(W2)``, so the regression is smooth in ``(W1, A1)`` within each ``W2``.  The basis is
    a degree-five polynomial in ``A1``, a cubic in ``W1``, their products up to total degree
    four, each repeated for ``W2 = 1``.  ``tests/unit/test_mtp_boundaries_design.py`` checks
    its approximation error against :func:`first_regression` by quadrature.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RichPseudo:
        self.model_ = LinearRegression().fit(self._design(X), y, sample_weight=sample_weight)
        return self

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w1, w2, a = matrix[:, 0], matrix[:, 1], matrix[:, -1] / 4.0
        terms = [w1**i * a**j for i in range(4) for j in range(6) if 0 < i + j <= 5 and i <= 3]
        block = np.column_stack(terms)
        return np.column_stack([block, w2[:, None] * block, w2])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)


#: The efficiency bound of each plan's contrast, pinned from :func:`contrast_sd`;
#: ``tests/unit/test_mtp_boundaries_design.py`` recomputes them.
BOUNDS: dict[str, float] = {
    "p1": 0.3797404449688521,
    "p2": 0.6228356141639579,
    "p3": 0.20240929541788777,
    "p4": 1.6237844351483597,
    "p5": 1.1033747551684423,
}

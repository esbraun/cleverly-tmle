"""Law P, its plans, truths and learners: the pricing law of ``mtp-pricing``.

**Law P.**  ``tau`` in ``{2, 3, 5}`` price nodes with engagement feedback:

.. code-block:: text

    W1 ~ N(0, 1) truncated to [-2.5, 2.5];  A_0 = c_0 = 6;  L_0 = 0
    for t = 1..tau, with c_t = 6 + 2 L_t:
      L_t ~ Bernoulli(expit(0.3 + 0.6 W1 - 0.35 (A_{t-1} - c_{t-1}) + 0.8 L_{t-1}))
      A_t ~ N(c_t + 0.3 (A_{t-1} - c_{t-1}) + 0.3 W1, 1.5^2) truncated to [4 + 2 L_t, 8 + 2 L_t]
    Y ~ Bernoulli(expit(0.5 - 0.45 (A_tau - c_tau) + 0.9 L_tau + 0.3 W1
                        - 0.1 sum_{s < tau} (A_s - c_s)))
    revenue R = Y A_tau, declared support [0, 10]

Engaged customers get, and tolerate, higher prices; a high price lowers the next node's
engagement.  The declared support of ``A_t`` is ``[4 + 2 L_t, 8 + 2 L_t]`` (:func:`support`).
Every draw is an inverse-distribution-function draw.

**Plans.**  ``natural_prices``; ``price_up``: ``1.12 a`` on ``[l_t, u_t / 1.12)`` and held on
``[u_t / 1.12, u_t]``, so the upper boundary binds; ``price_down``: held on ``[l_t, l_t / 0.85)`` and
``0.85 a`` on ``[l_t / 0.85, u_t]``, so the lower boundary binds.  Each is a
:class:`~cleverly.interventions.ModifiedPolicy` per node, reading that node's ``L_t``.

**Truths.**  The g-formula under Definition 1 of Díaz, Williams, Hoffman and Schenck (2023).
At ``tau = 2`` by Gauss-Legendre quadrature split at every break (:func:`quadrature_truth`);
at ``tau = 3`` and 5 by :data:`TRUTH_DRAWS` draws of Definition 1 with common random numbers
across plans (:func:`monte_carlo_truth`), pinned in :data:`TRUTHS` with their Monte Carlo SE.
The fast tier cross-checks the Monte Carlo against the quadrature at ``tau = 2``.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from cleverly.interventions import ModifiedPolicy, Piece, Shift, Support
from cleverly.longitudinal import DynamicRegimen
from tests.studies.fractional_glm import QuasiBinomialGLM

SD = 1.5
W_BOUND = 2.5
UP = 1.12
DOWN = 0.85
HORIZONS = (2, 3, 5)
REFERENCE = "natural_prices"
LABELS = ("natural_prices", "price_up", "price_down")
REVENUE_BOUNDS = (0.0, 10.0)


# ---------------------------------------------------------------------------- the law


def lower(l: Any) -> Any:
    return 4.0 + 2.0 * np.asarray(l, dtype=float)


def upper(l: Any) -> Any:
    return 8.0 + 2.0 * np.asarray(l, dtype=float)


def centre(l: Any) -> Any:
    return 6.0 + 2.0 * np.asarray(l, dtype=float)


def p_engaged(w1: Any, gap: Any, l_prev: Any) -> Any:
    """``P(L_t = 1)`` given ``W1``, the previous price gap and the previous engagement."""
    return expit(0.3 + 0.6 * np.asarray(w1) - 0.35 * np.asarray(gap) + 0.8 * np.asarray(l_prev))


def price_mean(w1: Any, l: Any, gap: Any) -> Any:
    return centre(l) + 0.3 * np.asarray(gap) + 0.3 * np.asarray(w1)


def purchase(w1: Any, l_last: Any, gap_last: Any, lag: Any) -> Any:
    """``P(Y = 1)``: ``gap_last = A_tau - c_tau`` and ``lag = sum_{s < tau} (A_s - c_s)``."""
    return expit(
        0.5
        - 0.45 * np.asarray(gap_last)
        + 0.9 * np.asarray(l_last)
        + 0.3 * np.asarray(w1)
        - 0.1 * np.asarray(lag)
    )


def tn_mass(mean: Any, low: Any, high: Any) -> Any:
    return norm.cdf((np.asarray(high) - mean) / SD) - norm.cdf((np.asarray(low) - mean) / SD)


def tn_pdf(a: Any, mean: Any, low: Any, high: Any) -> Any:
    a = np.asarray(a, dtype=float)
    inside = (a >= low) & (a <= high)
    return np.where(inside, norm.pdf((a - mean) / SD) / (SD * tn_mass(mean, low, high)), 0.0)


def tn_draw(rng: np.random.Generator, mean: Any, low: Any, high: Any, u: Any = None) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    a = norm.cdf((np.asarray(low) - mean) / SD)
    b = norm.cdf((np.asarray(high) - mean) / SD)
    v = rng.random(mean.shape) if u is None else u
    return np.asarray(mean + SD * norm.ppf(a + v * (b - a)), dtype=float)


def w1_draw(rng: np.random.Generator, n: int) -> np.ndarray:
    low, high = norm.cdf(-W_BOUND), norm.cdf(W_BOUND)
    return np.asarray(norm.ppf(rng.uniform(low, high, n)), dtype=float)


def columns(tau: int) -> list[str]:
    return ["W1", *itertools.chain.from_iterable((f"L{t}", f"A{t}") for t in range(1, tau + 1))]


def sample(n: int, seed: int, tau: int) -> pd.DataFrame:
    """``n`` rows of law P at ``tau`` nodes, with the purchase ``Y`` and the revenue ``R``."""
    rng = np.random.default_rng(seed)
    w1 = w1_draw(rng, n)
    out: dict[str, np.ndarray] = {"W1": w1}
    gap, l_prev, lag = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(1, tau + 1):
        l = (rng.random(n) < p_engaged(w1, gap, l_prev)).astype(float)
        a = tn_draw(rng, price_mean(w1, l, gap), lower(l), upper(l))
        out[f"L{t}"], out[f"A{t}"] = l, a
        if t < tau:
            lag = lag + (a - centre(l))
        gap, l_prev = a - centre(l), l
    y = (rng.random(n) < purchase(w1, l_prev, gap, lag)).astype(float)
    frame = pd.DataFrame(out)
    frame["Y"] = y
    frame["R"] = y * out[f"A{tau}"]
    return frame


# ---------------------------------------------------------------------------- the plans


def up_map(a: Any, l: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return np.where(a < upper(l) / UP, UP * a, a)


def down_map(a: Any, l: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return np.where(a < lower(l) / DOWN, a, DOWN * a)


def identity_map(a: Any, l: Any) -> Any:
    return np.asarray(a, dtype=float)


MAPS: dict[str, Callable[[Any, Any], Any]] = {
    "natural_prices": identity_map,
    "price_up": up_map,
    "price_down": down_map,
}


def support(t: int) -> Support:
    """The declared support of ``A_t``: ``[4 + 2 L_t, 8 + 2 L_t]``."""
    column = f"L{t}"
    return Support(
        lambda h: lower(np.asarray(h[column], dtype=float)),
        lambda h: upper(np.asarray(h[column], dtype=float)),
    )


def _scaled(factor: float) -> tuple[Callable[..., Any], ...]:
    return (
        lambda a, h: factor * np.asarray(a, dtype=float),
        lambda b, h: np.asarray(b, dtype=float) / factor,
        lambda b, h: np.full_like(np.asarray(b, dtype=float), 1.0 / factor),
    )


def node_policy(label: str, t: int) -> Any:
    """The plan's policy at node ``t``, reading ``L_t`` from the policy frame."""
    column = f"L{t}"
    if label == "natural_prices":
        return Shift(0.0, cap=None)

    def knee_up(h: Any) -> np.ndarray:
        return np.asarray(upper(np.asarray(h[column], dtype=float)) / UP, dtype=float)

    def knee_down(h: Any) -> np.ndarray:
        return np.asarray(lower(np.asarray(h[column], dtype=float)) / DOWN, dtype=float)

    if label == "price_up":
        pieces = (Piece(-np.inf, knee_up, *_scaled(UP)), Piece(knee_up, np.inf))
    elif label == "price_down":
        pieces = (Piece(-np.inf, knee_down), Piece(knee_down, np.inf, *_scaled(DOWN)))
    else:
        raise KeyError(label)
    return ModifiedPolicy(f"{label} at {t}", pieces=pieces, policy_kind="known")


def regimens(labels: tuple[str, ...], tau: int) -> dict[str, Any]:
    return {
        label: DynamicRegimen(label, tuple(node_policy(label, t) for t in range(1, tau + 1)))
        for label in labels
    }


def induced(label: str, b: Any, mean: Any, l: Any) -> np.ndarray:
    """The density the plan induces at ``b``, given the node's conditional mean and ``L_t``."""
    b = np.asarray(b, dtype=float)
    low, high = lower(l), upper(l)
    if label == "natural_prices":
        return tn_pdf(b, mean, low, high)
    if label == "price_up":
        moved = tn_pdf(b / UP, mean, low, high) / UP * (b / UP < high / UP)
        return moved + tn_pdf(b, mean, low, high) * (b >= high / UP)
    if label == "price_down":
        moved = tn_pdf(b / DOWN, mean, low, high) / DOWN * (b / DOWN >= low / DOWN)
        return tn_pdf(b, mean, low, high) * (b < low / DOWN) + moved
    raise KeyError(label)


def ratio(label: str, b: Any, mean: Any, l: Any) -> np.ndarray:
    """Equation (3) on the truncated normal: the plan's node ratio at ``b``."""
    g = tn_pdf(b, mean, lower(l), upper(l))
    safe = np.where(g > 0.0, g, 1.0)
    return np.asarray(np.where(g > 0.0, induced(label, b, mean, l) / safe, 0.0), dtype=float)


# ---------------------------------------------------------------------------- truths


def _plain(order: int, low: Any, high: Any) -> tuple[np.ndarray, np.ndarray]:
    base, weights = np.polynomial.legendre.leggauss(order)
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    half = 0.5 * (high - low)
    return (
        (low + half)[..., None] + half[..., None] * base,
        half[..., None] * weights,
    )


def _dose_nodes(l: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    low, high = float(lower(l)), float(upper(l))
    breaks = sorted({low, high, high / UP, low / DOWN})
    nodes, weights = [], []
    for a, b in itertools.pairwise(breaks):
        x, w = _plain(order, a, b)
        nodes.append(x)
        weights.append(w)
    return np.concatenate(nodes), np.concatenate(weights)


def _quadrature(label: str, order: int, revenue: bool) -> float:
    d = MAPS[label]
    w1, ww = _plain(order, -W_BOUND, W_BOUND)
    ww = ww * norm.pdf(w1) / (norm.cdf(W_BOUND) - norm.cdf(-W_BOUND))
    total = 0.0
    for l1 in (0.0, 1.0):
        p1 = p_engaged(w1, 0.0, 0.0)
        mass1 = ww * (p1 if l1 else 1.0 - p1)
        a1, wa1 = _dose_nodes(l1, order)
        g1 = tn_pdf(a1[None, :], price_mean(w1[:, None], l1, 0.0), lower(l1), upper(l1))
        gap1 = d(a1, l1) - centre(l1)
        for l2 in (0.0, 1.0):
            p2 = p_engaged(w1[:, None], gap1[None, :], l1)
            mass2 = mass1[:, None] * wa1[None, :] * g1 * (p2 if l2 else 1.0 - p2)
            a2, wa2 = _dose_nodes(l2, order)
            mean2 = price_mean(w1[:, None, None], l2, gap1[None, :, None])
            g2 = tn_pdf(a2[None, None, :], mean2, lower(l2), upper(l2))
            assigned = d(a2, l2)
            value = purchase(
                w1[:, None, None], l2, assigned[None, None, :] - centre(l2), gap1[None, :, None]
            )
            if revenue:
                value = value * assigned[None, None, :]
            total += float(np.sum(mass2[..., None] * g2 * wa2 * value))
    return total


@cache
def quadrature_truth(label: str, revenue: bool = False) -> float:
    """``E[Y]`` (or ``E[R]``) under a plan at ``tau = 2``, checked against twice the order."""
    value = _quadrature(label, 48, revenue)
    check = _quadrature(label, 96, revenue)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} moved by {value - check:.3g}")
    return value


def simulate(
    n: int, seed: int, tau: int, labels: tuple[str, ...] = LABELS
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Definition 1 under each plan on common random numbers: ``(P(Y = 1), A_tau^d)`` per row."""
    rng = np.random.default_rng(seed)
    w1 = w1_draw(rng, n)
    uniforms = [(rng.random(n), rng.random(n)) for _ in range(tau)]
    out = {}
    for label in labels:
        d = MAPS[label]
        gap, l_prev, lag = np.zeros(n), np.zeros(n), np.zeros(n)
        for t, (u_l, u_a) in enumerate(uniforms, start=1):
            l = (u_l < p_engaged(w1, gap, l_prev)).astype(float)
            a = d(tn_draw(rng, price_mean(w1, l, gap), lower(l), upper(l), u_a), l)
            if t < tau:
                lag = lag + (a - centre(l))
            gap, l_prev = a - centre(l), l
        out[label] = (purchase(w1, l_prev, gap, lag), gap + centre(l_prev))
    return out


TRUTH_DRAWS = 10_000_000
TRUTH_SEED = 20264029


def monte_carlo_truth(tau: int, draws: int = TRUTH_DRAWS, chunk: int = 1_000_000) -> dict[str, Any]:
    """Each plan's mean purchase and revenue, the contrasts, and their Monte Carlo SEs."""
    sums: dict[str, list[np.ndarray]] = {}
    for k, start in enumerate(range(0, draws, chunk)):
        size = min(chunk, draws - start)
        drawn = simulate(size, TRUTH_SEED + 7919 * k + tau, tau)
        for label, (p, a) in drawn.items():
            sums.setdefault(f"ey[{label}]", []).append(p)
            sums.setdefault(f"er[{label}]", []).append(p * a)
        for label in LABELS[1:]:
            sums.setdefault(f"ate[{label}]", []).append(drawn[label][0] - drawn[REFERENCE][0])
    out: dict[str, Any] = {}
    for key, parts in sums.items():
        values = np.concatenate(parts)
        out[key] = (float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size)))
    return out


#: The pinned truths ``(value, Monte Carlo SE)`` of :func:`monte_carlo_truth` at ``tau = 3``
#: and 5, and the quadrature at ``tau = 2`` (SE 0).  ``tests/unit/test_mtp_pricing_design.py``
#: recomputes the quadrature and checks the pins against a fresh Monte Carlo.
TRUTHS: dict[int, dict[str, tuple[float, float]]] = {
    3: {
        "ey[natural_prices]": (0.7300633578965745, 4.360678078009672e-05),
        "er[natural_prices]": (5.375254148007253, 0.0004565233766058205),
        "ey[price_up]": (0.6297434945405513, 4.6702031290524995e-05),
        "er[price_up]": (5.063056517860076, 0.0004982491629918984),
        "ey[price_down]": (0.8450017801875289, 2.8457540642018193e-05),
        "er[price_down]": (5.500908099872063, 0.0003717156414812079),
        "ate[price_up]": (-0.1003198633560232, 1.933676665300479e-05),
        "ate[price_down]": (0.1149384222909546, 2.3933175742207858e-05),
    },
    5: {
        "ey[natural_prices]": (0.7303246627692517, 4.384029630703548e-05),
        "er[natural_prices]": (5.380321906049831, 0.00045475636824327415),
        "ey[price_up]": (0.5978794711965534, 4.78857793043328e-05),
        "er[price_up]": (4.808130518123396, 0.0004928982511069234),
        "ey[price_down]": (0.8696531115457695, 2.501219007503007e-05),
        "er[price_down]": (5.665477331385306, 0.00036228577224033006),
        "ate[price_up]": (-0.13244519157269827, 2.0622550113812617e-05),
        "ate[price_down]": (0.13932844877651784, 2.630663689807715e-05),
    },
}


def truth(tau: int, key: str) -> float:
    """``key`` is ``ey[<plan>]``, ``er[<plan>]`` (revenue) or ``ate[<plan>]``."""
    if tau == 2:
        kind, label = key[:-1].split("[")
        if kind == "ate":
            return quadrature_truth(label) - quadrature_truth(REFERENCE)
        return quadrature_truth(label, revenue=kind == "er")
    return TRUTHS[tau][key][0]


# ---------------------------------------------------------------------------- designs


def node_columns(tau: int, t: int) -> dict[str, int]:
    """The column of each variable in the history design of node ``t``.

    The history is ``[W1, L_1..L_t, A_1..A_{t-1}]``; the outcome design appends ``A_t``.
    """
    index = {"W1": 0}
    for s in range(1, t + 1):
        index[f"L{s}"] = s
    for s in range(1, t):
        index[f"A{s}"] = t + s
    index[f"A{t}"] = 2 * t
    return index


def _node_of_outcome_design(width: int) -> int:
    # [W1, L_1..L_t, A_1..A_t] has 1 + 2 t columns.
    if (width - 1) % 2:
        raise ValueError(f"unexpected outcome design width {width}")
    return (width - 1) // 2


def _gaps(design: np.ndarray, t: int) -> list[np.ndarray]:
    index = node_columns(t, t)
    return [design[:, index[f"A{s}"]] - centre(design[:, index[f"L{s}"]]) for s in range(1, t + 1)]


class FinalOutcome(BaseEstimator):
    """The correctly specified purchase model on ``[A_tau - c_tau, L_tau, W1, lag]``.

    With ``revenue=True`` the target is ``R / 10`` on the recursion's ``[0, 1]`` scale; the
    model then fits ``Y = 10 z / A_tau`` and predicts ``A_tau P(Y = 1) / 10``.
    """

    def __init__(self, revenue: bool = False) -> None:
        self.revenue = revenue

    @staticmethod
    def _design(X: Any) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(X, dtype=float)
        t = _node_of_outcome_design(x.shape[1])
        gaps = _gaps(x, t)
        lag = np.sum(gaps[:-1], axis=0) if t > 1 else np.zeros(len(x))
        index = node_columns(t, t)
        return (
            np.column_stack([gaps[-1], x[:, index[f"L{t}"]], x[:, 0], lag]),
            x[:, index[f"A{t}"]],
        )

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> FinalOutcome:
        design, price = self._design(X)
        target = np.asarray(y, dtype=float)
        if self.revenue:
            target = np.clip(REVENUE_BOUNDS[1] * target / price, 0.0, 1.0)
        self.model_ = QuasiBinomialGLM().fit(design, target, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        design, price = self._design(X)
        p = np.asarray(self.model_.predict(design), dtype=float)
        return price * p / REVENUE_BOUNDS[1] if self.revenue else p

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def restricted_cubic(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted (natural) cubic spline basis: ``x`` and ``len(knots) - 2`` terms."""
    k = knots
    scale = (k[-1] - k[0]) ** 2

    def cube(v: np.ndarray) -> np.ndarray:
        return np.clip(v, 0.0, None) ** 3

    terms = [x]
    for j in range(len(k) - 2):
        terms.append(
            (
                cube(x - k[j])
                - cube(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                + cube(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])
            )
            / scale
        )
    return np.column_stack(terms)


SPLINE_QUANTILES = (0.05, 0.275, 0.5, 0.725, 0.95)


class SplinePseudo(BaseEstimator):
    """The pseudo-outcome regression at a node ``t < tau``: a quasibinomial GLM on splines.

    A natural cubic spline basis with 5 knots (at the training quantiles 0.05, 0.275, 0.5,
    0.725 and 0.95) of each of ``A_t``, ``A_{t-1}`` and ``W1``, with ``L_t``, ``L_{t-1}`` and
    their products with the spline of ``A_t``, and the linear lag ``sum_{s < t-1} (A_s -
    c_s)``, which the purchase reads and the spline of ``A_{t-1}`` does not hold.
    """

    def _basis(self, X: Any, fit: bool) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        t = _node_of_outcome_design(x.shape[1])
        index = node_columns(t, t)
        names = [f"A{t}", "W1"] + ([f"A{t - 1}"] if t > 1 else [])
        if fit:
            self.knots_ = {
                name: np.quantile(x[:, index[name]], SPLINE_QUANTILES) for name in names
            }
        splines = {name: restricted_cubic(x[:, index[name]], self.knots_[name]) for name in names}
        engaged = [x[:, index[f"L{t}"]]] + ([x[:, index[f"L{t - 1}"]]] if t > 1 else [])
        blocks = [*splines.values(), *(e[:, None] for e in engaged)]
        blocks += [e[:, None] * splines[f"A{t}"] for e in engaged]
        if t > 2:
            gaps = _gaps(x, t)
            blocks.append(np.sum(gaps[:-2], axis=0)[:, None])
        return np.column_stack(blocks)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SplinePseudo:
        self.model_ = QuasiBinomialGLM().fit(
            self._basis(X, True), np.asarray(y, dtype=float), sample_weight=sample_weight
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._basis(X, False)), dtype=float)


class WrongOutcome(BaseEstimator):
    """The pairs' outcome model at every node: a quasibinomial GLM on ``[A_t, W1]``.

    It omits ``L_t`` and the lagged prices (``lawP_wrong.log``: bias 0.096 at ``tau = 3``
    and 0.128 at ``tau = 5`` for the plug-in under ``up``).
    """

    def _design(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        t = _node_of_outcome_design(x.shape[1])
        return np.column_stack([x[:, node_columns(t, t)[f"A{t}"]], x[:, 0]])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WrongOutcome:
        self.model_ = QuasiBinomialGLM().fit(
            self._design(X), np.asarray(y, dtype=float), sample_weight=sample_weight
        )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _history_node(width: int) -> int:
    # [W1, L_1..L_t, A_1..A_{t-1}] has 2 t columns.
    if width % 2:
        raise ValueError(f"unexpected history width {width}")
    return width // 2


def _node_law(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, L_t)`` of the node's dose given its history design."""
    t = _history_node(history.shape[1])
    index = node_columns(t, t)
    l = history[:, index[f"L{t}"]]
    if t == 1:
        gap = np.zeros(len(history))
    else:
        gap = history[:, index[f"A{t - 1}"]] - centre(history[:, index[f"L{t - 1}"]])
    return price_mean(history[:, 0], l, gap), l


class OracleRatio(BaseEstimator):
    """The exact ratio on the classifier route: ``u = r / (1 + r)`` of the stacked label.

    The stacked design is ``[A_t, H_t]``.  ``fit`` reads which plan the label-1 block holds
    by applying each candidate map of :data:`MAPS` to the label-0 doses, and the
    predictions are the exact odds of that plan, Equation (3) on the truncated normal.  The
    stack's ``u / (1 - u)`` then returns the closed-form ratio, so no ratio is estimated.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleRatio:
        x = np.asarray(X, dtype=float)
        label = np.asarray(y, dtype=float)
        natural, assigned = x[label == 0.0], x[label == 1.0]
        if natural.shape != assigned.shape:
            raise ValueError("the stack holds unequal blocks")
        _, l = _node_law(natural[:, 1:])
        for name, d in MAPS.items():
            if np.allclose(d(natural[:, 0], l), assigned[:, 0], rtol=0, atol=1e-12):
                self.plan_ = name
                break
        else:
            raise ValueError("the stacked doses match no declared plan")
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        mean, l = _node_law(x[:, 1:])
        r = ratio(self.plan_, x[:, 0], mean, l)
        u = r / (1.0 + r)
        return np.column_stack([1.0 - u, u])


class SplineHazard(BaseEstimator):
    """The declared learned density: a logistic pooled hazard on a spline of the bin.

    The density layer hands this the history followed by the bin index (``bin_design =
    "index"``).  The hazard's log-odds is a B-spline basis of the bin position (8 degrees of
    freedom), the history, and the products of the two, fitted by an unpenalized-in-effect
    logistic regression (``C = 1e6``).
    """

    bin_design = "index"

    def __init__(self, n_bins: int = 320, df: int = 8) -> None:
        self.n_bins = n_bins
        self.df = df

    def _design(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        history, position = x[:, :-1], x[:, -1:] / max(self.n_bins - 1, 1)
        basis = self.spline_.transform(np.clip(position, 0.0, 1.0))
        products = (basis[:, :, None] * history[:, None, :]).reshape(len(x), -1)
        return np.column_stack([basis, history, products])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SplineHazard:
        self.spline_ = SplineTransformer(
            n_knots=self.df - 2, degree=3, include_bias=False
        ).fit(np.array([[0.0], [1.0]]))
        self.model_ = LogisticRegression(C=1e6, max_iter=2_000).fit(
            self._design(X), np.asarray(y), sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._design(X)), dtype=float)


def piece_indicators(a: np.ndarray, l: np.ndarray) -> np.ndarray:
    """The breakpoints of the declared support's plans: ``1.12 l``, ``u / 1.12``, ``l / 0.85``,
    ``0.85 u``."""
    low, high = lower(l), upper(l)
    return np.column_stack(
        [a >= UP * low, a >= high / UP, a >= low / DOWN, a >= DOWN * high]
    ).astype(float)


class QuadraticLogistic(BaseEstimator):
    """Classifier candidate 1: a logistic regression on ``[a, a^2, h, a h, 1{piece}]``.

    The stacked design is ``[A_t, H_t]``.  ``1{piece}`` holds :func:`piece_indicators` at
    the node's declared support.
    """

    def _design(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        a, h = x[:, 0], x[:, 1:]
        t = _history_node(h.shape[1])
        l = h[:, node_columns(t, t)[f"L{t}"]]
        return np.column_stack([a, a**2, h, a[:, None] * h, piece_indicators(a, l)])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> QuadraticLogistic:
        self.model_ = LogisticRegression(C=1e6, max_iter=2_000).fit(
            self._design(X), np.asarray(y), sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._design(X)), dtype=float)


def boosted(depth: int = 2) -> Any:
    """Classifier candidate 2: a shallow, regularized boosted classifier.

    The beta brief forbids a bare ``HistGradientBoostingClassifier`` below 10,000 rows, so
    the trees are shallow, the leaves hold at least 50 rows, the leaf values are shrunk by an
    L2 penalty, and no early stopping draws a random validation split.  ``depth=4`` is the
    declared more local alternative of ``classifier_split``.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=depth,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


#: The declared classifier candidates of the learner screen.
CLASSIFIER_CANDIDATES: dict[str, Callable[[], Any]] = {
    "quadratic_logistic": QuadraticLogistic,
    "boosted_depth2": lambda: boosted(2),
}

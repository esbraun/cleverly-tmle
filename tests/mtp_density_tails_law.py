"""Point laws for the density route of a modified treatment policy, with closed-form ratios.

Six laws share ``W1``, ``W2``, the outcome ``Y = Q0(A, W) + N(0, 1)`` with
``Q0(a, w) = 0.5 a - 0.1 a^2 + w1 + 0.5 w2 + 0.3 a w1``, and a shift of one:

- ``G``: ``A | W ~ N(c, 1.5^2)``, ``c = 1 + 0.5 W1 + 0.5 W2``, ``Shift(1)``.  Unbounded ratio.
- ``T``: ``G`` truncated to ``[-2, 4.5]``, ``Shift(1, cap=4.5)``.  Bounded ratio.
- ``C``: ``(1 - 0.0015) N(c, 1.5^2) + 0.0015 N(20, 1.5^2)``: a few doses far beyond a gap.
- ``S``: ``A = c + 1.5 t_5``: heavy tails.
- ``B``: ``W1`` a standard normal truncated to ``[-2.5, 2.5]``, ``u = 6 + 2 W2``,
  ``A | W ~ N(u - 1.6 + 0.4 W1, 1.2^2)`` truncated to ``[0, u]``; the policy adds one below
  ``u - 1`` and holds the dose on ``[u - 1, u]``.  The fit declares ``Support(0, u(W))``.
- ``U``: as ``B`` with ``u = 7 + 0.8 W1`` and mean ``u - 1.6 + 0.4 W2``: every bin near the
  top straddles some row's support edge.

For each law the module gives a sampler, the conditional CDF and density, the true ratio
``r0 = g^d / g``, the efficient influence curve ``D0 = r0 (Y - Q0) + Q0(d(A), W) - psi0``, an
oracle hazard learner (each row's exact bin probabilities, so only the representation is left),
the correct and a wrong outcome regression, and the pre-fix representation (equal-mass grid read
as a histogram) as a reference reading.  It shares no code with ``src/``.

``EFFICIENCY_SD`` is ``SD(D0)`` from Monte Carlo over 4,000,000 draws; the witnesses use it only
to scale the influence-curve gap ``SD(IC - D0) / SD(D0)``, which no constant in ``psi0`` moves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student
from sklearn.base import BaseEstimator

SIGMA = 1.5
DELTA = 1.0
LOW, HIGH = -2.0, 4.5
CONTAMINATION, FAR, FAR_SD = 0.0015, 20.0, 1.5
DEGREES = 5
#: Laws B and U.
BOUNDED_SD, W1_BOUND = 1.2, 2.5

LAWS = ("G", "T", "C", "S", "B", "U")
SUPPORTED = ("B", "U")

#: ``(psi0, SD(D0))`` by Monte Carlo, 4,000,000 draws.
TRUTH = {
    "G": (0.7620878893447987, 2.1802671430413594),
    "T": (0.8012982641621987, 2.163865726954183),
    "C": (0.710847761584088, 2.57631874283126),
    "S": (0.6111472701877791, 2.5332099962281647),
    "B": (-0.4047355661123545, 2.954624574578488),
    "U": (-0.2683273712975322, 2.572458112748248),
}
EFFICIENCY_SD = {law: sd for law, (_, sd) in TRUTH.items()}


def q0(a: Any, w1: Any, w2: Any) -> np.ndarray:
    """The true outcome regression."""
    a = np.asarray(a, dtype=float)
    return np.asarray(0.5 * a - 0.1 * a**2 + w1 + 0.5 * w2 + 0.3 * a * w1, dtype=float)


def centre(w1: Any, w2: Any) -> np.ndarray:
    """``c(W)`` of laws G, T, C and S."""
    return 1.0 + 0.5 * np.asarray(w1, dtype=float) + 0.5 * np.asarray(w2, dtype=float)


def upper(law: str, w1: Any, w2: Any) -> np.ndarray:
    """``u(W)``, the upper end of the support of laws B and U."""
    if law == "B":
        return 6.0 + 2.0 * np.asarray(w2, dtype=float)
    return 7.0 + 0.8 * np.asarray(w1, dtype=float)


def _bounded_mean(law: str, w1: Any, w2: Any) -> np.ndarray:
    shift = np.asarray(w1, dtype=float) if law == "B" else np.asarray(w2, dtype=float)
    return upper(law, w1, w2) - 1.6 + 0.4 * shift


def cdf(law: str, x: Any, w1: Any, w2: Any) -> np.ndarray:
    """``P(A <= x | W)``."""
    x = np.asarray(x, dtype=float)
    if law in SUPPORTED:
        mu, u = _bounded_mean(law, w1, w2), upper(law, w1, w2)
        clipped = np.clip(x, 0.0, u)
        mass = norm.cdf((u - mu) / BOUNDED_SD) - norm.cdf(-mu / BOUNDED_SD)
        return (norm.cdf((clipped - mu) / BOUNDED_SD) - norm.cdf(-mu / BOUNDED_SD)) / mass
    c = centre(w1, w2)
    if law in ("G", "T"):
        out = norm.cdf((x - c) / SIGMA)
        if law == "T":
            low, high = norm.cdf((LOW - c) / SIGMA), norm.cdf((HIGH - c) / SIGMA)
            out = np.clip((out - low) / (high - low), 0.0, 1.0)
        return np.asarray(out, dtype=float)
    if law == "C":
        return np.asarray(
            (1 - CONTAMINATION) * norm.cdf((x - c) / SIGMA)
            + CONTAMINATION * norm.cdf((x - FAR) / FAR_SD),
            dtype=float,
        )
    return np.asarray(student.cdf((x - c) / SIGMA, DEGREES), dtype=float)


def pdf(law: str, x: Any, w1: Any, w2: Any) -> np.ndarray:
    """``g(x | W)``."""
    x = np.asarray(x, dtype=float)
    if law in SUPPORTED:
        mu, u = _bounded_mean(law, w1, w2), upper(law, w1, w2)
        mass = norm.cdf((u - mu) / BOUNDED_SD) - norm.cdf(-mu / BOUNDED_SD)
        inside = (x >= 0.0) & (x <= u)
        return np.where(inside, norm.pdf((x - mu) / BOUNDED_SD) / (BOUNDED_SD * mass), 0.0)
    c = centre(w1, w2)
    if law == "G":
        return np.asarray(norm.pdf((x - c) / SIGMA) / SIGMA, dtype=float)
    if law == "T":
        low, high = norm.cdf((LOW - c) / SIGMA), norm.cdf((HIGH - c) / SIGMA)
        inside = (x >= LOW) & (x <= HIGH)
        return np.where(inside, norm.pdf((x - c) / SIGMA) / SIGMA / (high - low), 0.0)
    if law == "C":
        return np.asarray(
            (1 - CONTAMINATION) * norm.pdf((x - c) / SIGMA) / SIGMA
            + CONTAMINATION * norm.pdf((x - FAR) / FAR_SD) / FAR_SD,
            dtype=float,
        )
    return np.asarray(student.pdf((x - c) / SIGMA, DEGREES) / SIGMA, dtype=float)


def draw(law: str, n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of ``W1``, ``W2``, ``A`` and ``Y``."""
    rng = np.random.default_rng(seed)
    if law in SUPPORTED:
        w1 = norm.ppf(rng.uniform(norm.cdf(-W1_BOUND), norm.cdf(W1_BOUND), n))
        w2 = rng.integers(0, 2, n).astype(float)
        mu, u = _bounded_mean(law, w1, w2), upper(law, w1, w2)
        quantile = rng.uniform(norm.cdf(-mu / BOUNDED_SD), norm.cdf((u - mu) / BOUNDED_SD))
        a = mu + BOUNDED_SD * norm.ppf(quantile)
    else:
        w1 = rng.standard_normal(n)
        w2 = rng.integers(0, 2, n).astype(float)
        c = centre(w1, w2)
        if law == "G":
            a = c + SIGMA * rng.standard_normal(n)
        elif law == "T":
            low, high = norm.cdf((LOW - c) / SIGMA), norm.cdf((HIGH - c) / SIGMA)
            a = c + SIGMA * norm.ppf(low + (high - low) * rng.random(n))
        elif law == "C":
            far = rng.random(n) < CONTAMINATION
            a = np.where(
                far, FAR + FAR_SD * rng.standard_normal(n), c + SIGMA * rng.standard_normal(n)
            )
        else:
            a = c + SIGMA * rng.standard_t(DEGREES, n)
    y = q0(a, w1, w2) + rng.standard_normal(n)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


def assign(law: str, a: Any, w1: Any, w2: Any) -> np.ndarray:
    """``d(a, W)``: one more, held at the cap on T and on ``[u - 1, u]`` on B and U."""
    a = np.asarray(a, dtype=float)
    if law == "T":
        return np.where(a + DELTA <= HIGH, a + DELTA, a)
    if law in SUPPORTED:
        return np.where(a < upper(law, w1, w2) - DELTA, a + DELTA, a)
    return a + DELTA


def ratio_from(law: str, density: Any, a: Any, w1: Any, w2: Any) -> np.ndarray:
    """``r = g(a - 1) / g(a)`` on the moved piece, plus one where the policy holds the dose."""
    a = np.asarray(a, dtype=float)
    below = np.asarray(density(a), dtype=float)
    above = np.asarray(density(a - DELTA), dtype=float)
    held = np.zeros(a.size, dtype=bool)
    if law == "T":
        above = above * (a <= HIGH)
        held = a > HIGH - DELTA
    elif law in SUPPORTED:
        above = above * (a >= DELTA)
        held = a >= upper(law, w1, w2) - DELTA
    out = np.where(below > 0.0, above / np.where(below > 0.0, below, 1.0), 0.0)
    return np.asarray(out + held, dtype=float)


def r0(law: str, a: Any, w1: Any, w2: Any) -> np.ndarray:
    """The true ratio."""
    return ratio_from(law, lambda v: pdf(law, v, w1, w2), a, w1, w2)


def columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``W1``, ``W2``, ``A`` and ``Y`` as arrays."""
    w1, w2, a, y = (frame[name].to_numpy(dtype=float) for name in ("W1", "W2", "A", "Y"))
    return w1, w2, a, y


def eif(law: str, frame: pd.DataFrame) -> np.ndarray:
    """``D0`` at every row."""
    w1, w2, a, y = columns(frame)
    psi = TRUTH[law][0]
    return r0(law, a, w1, w2) * (y - q0(a, w1, w2)) + q0(assign(law, a, w1, w2), w1, w2) - psi


def bin_probabilities(law: str, edges: Any, w1: Any, w2: Any) -> np.ndarray:
    """Exact ``(n, B)`` bin masses, the outer bins open to the tails."""
    e = np.asarray(edges, dtype=float)
    w1 = np.asarray(w1, dtype=float)[:, None]
    w2 = np.asarray(w2, dtype=float)[:, None]
    full = cdf(law, e[None, :], w1, w2)
    full[:, 0] = 0.0
    full[:, -1] = 1.0
    p = np.clip(np.diff(full, axis=1), 0.0, None)
    return np.asarray(p / p.sum(axis=1, keepdims=True), dtype=float)


def through_clip(p: Any) -> np.ndarray:
    """Exact bin masses passed through the package's hazard clip ``[1e-12, 1 - 1e-12]``.

    The same log-space product a fitted density applies, written out here, so that a bin
    outside a row's support carries about ``1e-12`` of its mass, not an exact zero.
    """
    p = np.asarray(p, dtype=float)
    below = np.cumsum(p, axis=1)[:, :-1] - p[:, :-1]
    survival = 1.0 - below
    hazard = np.divide(p[:, :-1], survival, out=np.ones_like(survival), where=survival > 1e-15)
    hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
    log_survival = np.cumsum(np.log1p(-hazard), axis=1)
    leading = np.log(hazard)
    leading[:, 1:] += log_survival[:, :-1]
    return np.asarray(np.exp(np.column_stack([leading, log_survival[:, -1]])), dtype=float)


class OracleHazard(BaseEstimator):  # type: ignore[misc]
    """Each row's exact pooled hazards on fixed edges; the design is ``[W1, W2, bin]``.

    ``bin_design = "index"`` makes the package pass the bin index alone.
    """

    bin_design = "index"

    def __init__(self, edges: tuple[float, ...] = (), law: str = "G") -> None:
        self.edges = edges
        self.law = law

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleHazard:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        e = np.asarray(self.edges, dtype=float)
        index = np.rint(design[:, 2]).astype(int)
        last = e.size - 2
        lower = np.where(index == 0, -np.inf, e[index])
        upper_edge = np.where(index == last, np.inf, e[np.minimum(index + 1, last + 1)])
        reached = cdf(self.law, upper_edge, design[:, 0], design[:, 1])
        below = cdf(self.law, lower, design[:, 0], design[:, 1])
        survival = 1.0 - below
        hazard = np.divide(
            reached - below, survival, out=np.ones_like(below), where=survival > 1e-15
        )
        hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - hazard, hazard])


class CorrectOutcome(BaseEstimator):  # type: ignore[misc]
    """OLS on ``[1, A, A^2, W1, W2, A W1]``: the true regression's span.  Design ``[A, W1, W2]``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CorrectOutcome:
        self.coef_ = np.linalg.lstsq(self._basis(X), np.asarray(y, dtype=float), rcond=None)[0]
        return self

    @staticmethod
    def _basis(x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        a, w1, w2 = x[:, 0], x[:, 1], x[:, 2]
        return np.column_stack([np.ones(a.size), a, a**2, w1, w2, a * w1])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self._basis(X) @ self.coef_, dtype=float)


def wrong_q(a: Any, w1: Any, w2: Any, frame: pd.DataFrame) -> np.ndarray:
    """OLS of ``Y`` on ``[1, A, W1, W2]`` fitted on ``frame``, evaluated at ``(a, W)``."""
    fw1, fw2, fa, fy = columns(frame)
    design = np.column_stack([np.ones(fa.size), fa, fw1, fw2])
    coef = np.linalg.lstsq(design, fy, rcond=None)[0]
    a = np.asarray(a, dtype=float)
    return np.asarray(
        np.column_stack([np.ones(a.size), a, np.asarray(w1), np.asarray(w2)]) @ coef, dtype=float
    )


def equal_mass_edges(values: Any, n_bins: int) -> np.ndarray:
    """The pre-fix grid: ``n_bins`` equal-mass bins, outer edges padded by a hair."""
    a = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(a, np.linspace(0.0, 1.0, n_bins + 1)))
    pad = max(float(edges[-1] - edges[0]), 1.0) * 1e-9
    edges[0] -= pad
    edges[-1] += pad
    return np.asarray(edges, dtype=float)


def histogram(p: Any, edges: Any, values: Any) -> np.ndarray:
    """The pre-fix reading: each row's bin probability over the bin width, zero outside."""
    p = np.asarray(p, dtype=float)
    e = np.asarray(edges, dtype=float)
    v = np.asarray(values, dtype=float)
    index = np.clip(np.digitize(v, e) - 1, 0, e.size - 2)
    inside = (v >= e[0]) & (v <= e[-1])
    out = p[np.arange(v.size), index] / np.diff(e)[index]
    return np.asarray(np.where(inside, out, 0.0), dtype=float)


def ratio_error(law: str, ratio: Any, frame: pd.DataFrame) -> float:
    """``||r_n - r0||_n / SD(D0)`` over the sample."""
    w1, w2, a, _ = columns(frame)
    gap = np.asarray(ratio, dtype=float) - r0(law, a, w1, w2)
    return float(np.sqrt(np.mean(gap**2)) / EFFICIENCY_SD[law])

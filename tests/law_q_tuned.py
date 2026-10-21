r"""Law Q of X40 (plan section 5.3) at two nodes, by quadrature, for the tuned-cap tests.

The implementation never enters this module (``tests/unit/test_mtp_tuned_caps.py`` checks
its imports).  It writes the law, the parameter under a capped shift whose cap is a tuner,
the tuner's population value, and the influence curve of the population target

.. math::

    D_\Psi = D_\Phi(\theta_0) + \sum_t \partial_t\Phi(\theta_0)' \mathrm{IF}_t

longhand, each from its own formula: the LMTP curve :math:`D_\Phi` of Díaz, Williams,
Hoffman and Schenck (2023), Theorem 2; the influence function of a quantile and of a
pseudo-true linear quantile regression; and the derivative
:math:`E[W_{1:t-1} g_t(e_t \mid H_t) \{m_t(c_t) - m_t(e_t)\} \partial c_t/\partial\theta_t]`.
Every integral is Gauss-Legendre quadrature, split at every point where its integrand
jumps, so a central difference of :math:`\Psi` along a smooth path is accurate far below
the ``1e-6`` the identity test reads.

The law: :math:`W \sim N(0, 1)` truncated to ``[-2.5, 2.5]``; :math:`A_1 \mid W \sim
N(0.5 W, 1)` truncated to ``[-4, 4]``; :math:`L_2 \sim \mathrm{Bernoulli}(\mathrm{expit}(0.4 A_1 +
0.3 W))`; :math:`A_2 \sim N(0.4 A_1 + 0.6 L_2 + 0.3 W, 1)` truncated to ``[-4, 4]``; and
:math:`Y \sim \mathrm{Beta}(30 m, 30 (1 - m))` with :math:`m = \mathrm{expit}(3 (A_2 - s) +
0.5 L_2 + 0.3 A_1 + 0.2 W)`, :math:`s` the median of :math:`A_2` minus 0.5.  A path tilts
one factor: :math:`p_\epsilon \propto p \exp(\epsilon\, \tanh(\cdot))` for the doses and
:math:`W`, and :math:`\mathrm{logit}\, p_\epsilon = \mathrm{logit}\, p + \epsilon` for
:math:`L_2`.  The policy at both nodes is ``Shift(1.0, cap=c_t(h))``: :math:`d(a) = a + 1`
when :math:`a + 1 \le c_t(h)`, and :math:`a` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

DELTA = 1.0
W_RANGE = (-2.5, 2.5)
A_RANGE = (-4.0, 4.0)
_K = 20
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(_K)


def expit(x: np.ndarray) -> np.ndarray:
    return special.expit(x)


def _segments(lo: float, hi: float, splits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``[lo, hi]``, split at ``splits`` per batch row.

    ``splits`` is ``(B, S)``.  Returns ``(B, (S + 1) K)`` nodes and weights.  A split outside
    the range is clipped to its end, which gives a segment of zero weight.
    """
    edges = np.sort(np.clip(splits, lo, hi), axis=-1)
    batch = edges.shape[:-1]
    edges = np.concatenate([np.full((*batch, 1), lo), edges, np.full((*batch, 1), hi)], axis=-1)
    left, right = edges[..., :-1], edges[..., 1:]
    half = 0.5 * (right - left)
    mid = 0.5 * (right + left)
    x = mid[..., None] + half[..., None] * _NODES
    w = half[..., None] * _WEIGHTS
    return x.reshape((*batch, -1)), w.reshape((*batch, -1))


@dataclass(frozen=True)
class Tilt:
    """The path coordinates: one exponent per tilted factor."""

    w: float = 0.0
    a1: float = 0.0
    l2: float = 0.0
    a2: float = 0.0


@dataclass(frozen=True)
class Tuner:
    """A tuner of the oracle: a dose quantile, or a linear quantile regression.

    ``features`` names the columns of :math:`\\tilde H` after the intercept, from
    ``("W",)`` at node 1 and from ``("W", "L2", "A1")`` at node 2.  Empty for a quantile.
    """

    q: float = 0.5
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class LawQ:
    """Law Q, optionally tilted, with the tuners of the two nodes.

    Parameters
    ----------
    tuners : tuple of Tuner
        The tuner of node 1 and of node 2.
    tilt : Tilt
        The path coordinates.
    shift : float or None
        The outcome's offset ``s``; ``None`` computes the median of ``A2`` minus 0.5 at the
        untilted law.
    """

    tuners: tuple[Tuner, Tuner] = (Tuner(), Tuner())
    tilt: Tilt = field(default_factory=Tilt)
    shift: float | None = None
    factor: float | None = None

    # ------------------------------------------------------------ the map

    def edge(self, c: np.ndarray) -> np.ndarray:
        """The last moved dose ``e = c - DELTA``, or ``c / f`` of a ``Scale(f)``."""
        return c - DELTA if self.factor is None else c / self.factor

    def move(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """``d(x)``: ``x + DELTA`` (``f x``) where that is at most ``c``, else ``x``."""
        moved = x + DELTA if self.factor is None else self.factor * x
        return np.where(moved <= c, moved, x)

    def pre(self, a: np.ndarray) -> np.ndarray:
        """The dose the map sends to ``a`` below the cap."""
        return a - DELTA if self.factor is None else a / self.factor

    @property
    def slope(self) -> float:
        """:math:`|e'|`, the derivative of the edge in the cap: 1 for a shift, 1/f for a scale."""
        return 1.0 if self.factor is None else 1.0 / self.factor

    @property
    def floor(self) -> float:
        """The lowest dose a moved unit reaches: ``-4 + DELTA``, or ``-4 f``."""
        return A_RANGE[0] + DELTA if self.factor is None else A_RANGE[0] * self.factor

    def induced(self, g: np.ndarray, gpre: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
        """The density of the assigned dose at ``a``, from ``g(a)`` and ``g(pre(a))``."""
        return gpre * self.slope * (a <= c) + g * (a > self.edge(c))

    # ------------------------------------------------------------ factors

    @cached_property
    def w_grid(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = W_RANGE
        # Two segments, so the W law is integrated to double precision.
        x, wts = _segments(lo, hi, np.array([[0.0]]))
        x, wts = x[0], wts[0]
        density = stats.norm.pdf(x) * np.exp(self.tilt.w * np.tanh(x))
        mass = density * wts
        return x, mass / mass.sum()

    def g1(self, a: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self._dose_density(a, 0.5 * w, self.tilt.a1, self._z1(w))

    def g2(self, a: np.ndarray, w: np.ndarray, a1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        mean = 0.4 * a1 + 0.6 * l2 + 0.3 * w
        return self._dose_density(a, mean, self.tilt.a2, self._z(mean, self.tilt.a2))

    def p_l2(self, w: np.ndarray, a1: np.ndarray) -> np.ndarray:
        return expit(0.4 * a1 + 0.3 * w + self.tilt.l2)

    def m(self, w: np.ndarray, a1: np.ndarray, l2: np.ndarray, a2: np.ndarray) -> np.ndarray:
        return expit(3.0 * (a2 - self.s) + 0.5 * l2 + 0.3 * a1 + 0.2 * w)

    @staticmethod
    def _dose_density(a: np.ndarray, mean: np.ndarray, eps: float, z: np.ndarray) -> np.ndarray:
        lo, hi = A_RANGE
        inside = (a >= lo) & (a <= hi)
        raw = stats.norm.pdf(a - mean) * np.exp(eps * np.tanh(a))
        return np.where(inside, raw / z, 0.0)

    @staticmethod
    def _z(mean: np.ndarray, eps: float) -> np.ndarray:
        mean = np.asarray(mean, dtype=float)
        x, wts = _segments(*A_RANGE, (mean - 0.0)[..., None] * np.ones(1))
        raw = stats.norm.pdf(x - mean[..., None]) * np.exp(eps * np.tanh(x))
        return np.sum(raw * wts, axis=-1)

    def _z1(self, w: np.ndarray) -> np.ndarray:
        return self._z(0.5 * np.asarray(w, dtype=float), self.tilt.a1)

    def cdf1(self, t: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self._cdf(t, 0.5 * w, self.tilt.a1)

    def cdf2(self, t: np.ndarray, w: np.ndarray, a1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        return self._cdf(t, 0.4 * a1 + 0.6 * l2 + 0.3 * w, self.tilt.a2)

    def _cdf(self, t: np.ndarray, mean: np.ndarray, eps: float) -> np.ndarray:
        t, mean = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(mean, dtype=float))
        lo, hi = A_RANGE
        top = np.clip(t, lo, hi)
        x, wts = _segments(lo, 0.0, np.zeros((*t.shape, 0)))
        # Map the reference nodes on [lo, 0] to [lo, top] per element.
        scale = (top - lo) / (0.0 - lo)
        x = lo + (x - lo) * scale[..., None]
        wts = wts * scale[..., None]
        raw = stats.norm.pdf(x - mean[..., None]) * np.exp(eps * np.tanh(x))
        return np.sum(raw * wts, axis=-1) / self._z(mean, eps)

    @cached_property
    def s(self) -> float:
        if self.shift is not None:
            return float(self.shift)
        base = LawQ(tuners=self.tuners, shift=0.0)
        return float(base.marginal_quantile(2, 0.5)) - 0.5

    # ------------------------------------------------------------ marginals

    def _observed_h2(self) -> tuple[np.ndarray, ...]:
        """``(w, a1, l2, mass)`` of the observed node-2 history, flattened."""
        w, fw = self.w_grid
        a1, wa = _segments(*A_RANGE, np.repeat((0.5 * w)[:, None], 1, axis=1))
        mass = fw[:, None] * self.g1(a1, w[:, None]) * wa
        ww = np.broadcast_to(w[:, None], a1.shape)
        p = self.p_l2(ww, a1)
        out_w = np.concatenate([ww.ravel(), ww.ravel()])
        out_a = np.concatenate([a1.ravel(), a1.ravel()])
        out_l = np.concatenate([np.zeros(a1.size), np.ones(a1.size)])
        out_m = np.concatenate([(mass * (1.0 - p)).ravel(), (mass * p).ravel()])
        return out_w, out_a, out_l, out_m

    def marginal_cdf(self, node: int, t: float) -> float:
        if node == 1:
            w, fw = self.w_grid
            return float(np.sum(fw * self.cdf1(np.full(w.size, t), w)))
        w, a1, l2, mass = self._observed_h2()
        return float(np.sum(mass * self.cdf2(np.full(w.size, t), w, a1, l2)))

    def marginal_density(self, node: int, t: float) -> float:
        if node == 1:
            w, fw = self.w_grid
            return float(np.sum(fw * self.g1(np.full(w.size, t), w)))
        w, a1, l2, mass = self._observed_h2()
        return float(np.sum(mass * self.g2(np.full(w.size, t), w, a1, l2)))

    def marginal_quantile(self, node: int, q: float) -> float:
        return float(
            optimize.brentq(lambda t: self.marginal_cdf(node, t) - q, -4.0, 4.0, xtol=1e-14)
        )

    # ------------------------------------------------------------ tuners

    def _features(self, node: int, names: tuple[str, ...], h: dict[str, np.ndarray]) -> np.ndarray:
        columns = [np.ones_like(h["W"])] + [h[name] for name in names]
        return np.stack(columns, axis=-1)

    @cached_property
    def theta(self) -> tuple[np.ndarray, np.ndarray]:
        """The population tuner of each node: the quantile, or the pseudo-true beta."""
        return (self._theta(1), self._theta(2))

    def _theta(self, node: int) -> np.ndarray:
        tuner = self.tuners[node - 1]
        if not tuner.features:
            return np.array([self.marginal_quantile(node, tuner.q)])
        h, mass = self._history(node)
        x = self._features(node, tuner.features, h)
        cdf = self.cdf1 if node == 1 else self.cdf2
        density = self.g1 if node == 1 else self.g2
        args = (h["W"],) if node == 1 else (h["W"], h["A1"], h["L2"])
        beta = np.zeros(x.shape[-1])
        for _ in range(60):
            fit = x @ beta
            score = x.T @ (mass * (tuner.q - cdf(fit, *args)))
            jacobian = (x * (mass * density(fit, *args))[:, None]).T @ x
            step = np.linalg.solve(jacobian, score)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-15:
                break
        return beta

    def _history(self, node: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if node == 1:
            w, fw = self.w_grid
            return {"W": w}, fw
        w, a1, l2, mass = self._observed_h2()
        return {"W": w, "A1": a1, "L2": l2}, mass

    def cap(
        self, node: int, h: dict[str, np.ndarray], theta: np.ndarray | None = None
    ) -> np.ndarray:
        """``c_t(h)`` at the population tuner, or at ``theta``."""
        tuner = self.tuners[node - 1]
        value = self.theta[node - 1] if theta is None else theta
        if not tuner.features:
            return np.full(np.shape(h["W"]), float(value[0]))
        return self._features(node, tuner.features, h) @ value

    def gradient(self, node: int, h: dict[str, np.ndarray]) -> np.ndarray:
        tuner = self.tuners[node - 1]
        if not tuner.features:
            return np.ones((*np.shape(h["W"]), 1))
        return self._features(node, tuner.features, h)

    # ------------------------------------------------------------ the parameter

    def q1(self, w: np.ndarray, b: np.ndarray, theta2: np.ndarray | None = None) -> np.ndarray:
        """``m_1(b, w) = E[m_2(d_2(A_2), H_2) | A_1 = b, W = w]``, elementwise."""
        w, b = np.broadcast_arrays(np.asarray(w, dtype=float), np.asarray(b, dtype=float))
        total = np.zeros(w.shape)
        for l2 in (0.0, 1.0):
            ll = np.full(w.shape, l2)
            p = self.p_l2(w, b)
            weight = p if l2 else 1.0 - p
            c2 = self.cap(2, {"W": w, "A1": b, "L2": ll}, theta2)
            e2 = self.edge(c2)
            x, wts = _segments(*A_RANGE, e2[..., None])
            moved = self.move(x, c2[..., None])
            g = self.g2(x, w[..., None], b[..., None], ll[..., None])
            value = self.m(w[..., None], b[..., None], ll[..., None], moved)
            total = total + weight * np.sum(g * value * wts, axis=-1)
        return total

    def psi(self, theta: tuple[np.ndarray, np.ndarray] | None = None) -> float:
        """:math:`\\Phi(P, \\theta)`: the mean under the maps at ``theta`` (default the tuners)."""
        theta1 = self.theta[0] if theta is None else theta[0]
        theta2 = self.theta[1] if theta is None else theta[1]
        w, fw = self.w_grid
        c1 = self.cap(1, {"W": w}, theta1)
        a1, wa = _segments(*A_RANGE, self.edge(c1)[:, None])
        moved = self.move(a1, c1[:, None])
        inner = self.g1(a1, w[:, None]) * self.q1(w[:, None], moved, theta2) * wa
        return float(np.sum(fw * inner.sum(axis=1)))

    @property
    def tuned_psi(self) -> float:
        """:math:`\\Psi(P) = \\Phi(P, \\theta(P))`."""
        return self.psi()

    # ------------------------------------------------------------ the curve

    def derivative(self, node: int) -> np.ndarray:
        """The analytic :math:`\\partial\\Phi / \\partial\\theta_t` at the population tuner."""
        w, fw = self.w_grid
        if node == 1:
            h = {"W": w}
            c1 = self.cap(1, h)
            e1 = self.edge(c1)
            jump = self.slope * self.g1(e1, w) * (self.q1(w, c1) - self.q1(w, e1))
            return np.sum((fw * jump)[:, None] * self.gradient(1, h), axis=0)
        c1 = self.cap(1, {"W": w})
        a1, wa = _segments(*A_RANGE, self.edge(c1)[:, None])
        b = self.move(a1, c1[:, None])
        ww = np.broadcast_to(w[:, None], a1.shape)
        mass = fw[:, None] * self.g1(a1, w[:, None]) * wa
        total = 0.0
        for l2 in (0.0, 1.0):
            ll = np.full(a1.shape, l2)
            p = self.p_l2(ww, b)
            weight = mass * (p if l2 else 1.0 - p)
            h = {"W": ww, "A1": b, "L2": ll}
            c2 = self.cap(2, h)
            e2 = self.edge(c2)
            jump = (
                self.slope
                * self.g2(e2, ww, b, ll)
                * (self.m(ww, b, ll, c2) - self.m(ww, b, ll, e2))
            )
            total = total + np.sum((weight * jump)[..., None] * self.gradient(2, h), axis=(0, 1))
        return np.asarray(total)

    def ratio1(self, a1: np.ndarray, w: np.ndarray) -> np.ndarray:
        """:math:`r_1 = g^d_1 / g_1` at the observed first dose."""
        c1 = self.cap(1, {"W": w})
        g = self.g1(a1, w)
        induced = self.induced(g, self.g1(self.pre(a1), w), a1, c1)
        return np.where(g > 0.0, induced / np.where(g > 0.0, g, 1.0), 0.0)

    def influence(self, node: int, h: dict[str, np.ndarray], a: np.ndarray) -> np.ndarray:
        """``(..., p)`` the tuner's influence function at observed dose ``a``."""
        tuner = self.tuners[node - 1]
        theta = self.theta[node - 1]
        if not tuner.features:
            density = self.marginal_density(node, float(theta[0]))
            return (-((a <= theta[0]).astype(float) - tuner.q) / density)[..., None]
        hist, mass = self._history(node)
        x = self._features(node, tuner.features, hist)
        fit = x @ theta
        density = self.g1 if node == 1 else self.g2
        args = (hist["W"],) if node == 1 else (hist["W"], hist["A1"], hist["L2"])
        jacobian = (x * (mass * density(fit, *args))[:, None]).T @ x
        here = self._features(node, tuner.features, h)
        score = here * (tuner.q - (a <= here @ theta).astype(float))[..., None]
        return np.linalg.solve(jacobian, score.reshape(-1, x.shape[-1]).T).T.reshape(score.shape)

    def score_pairing(self, path: str, *, tuning: bool = True) -> float:
        """:math:`E[D_\\Psi s]` for the score ``s`` of one path, by quadrature.

        ``tuning=False`` drops the tuners' term, the mutation M-e4 of the identity.
        """
        derivative = (self.derivative(1), self.derivative(2)) if tuning else None
        psi = self.tuned_psi
        w, fw = self.w_grid
        c1 = self.cap(1, {"W": w})
        splits = np.stack([self.edge(c1), c1, np.full(w.size, self.floor)], axis=1)
        a1, wa = _segments(*A_RANGE, splits)
        ww = np.broadcast_to(w[:, None], a1.shape)
        mass1 = fw[:, None] * self.g1(a1, w[:, None]) * wa
        moved1 = self.move(a1, c1[:, None])
        term1 = self.q1(ww, moved1) - psi
        if derivative is not None:
            term1 = term1 + self.influence(1, {"W": ww}, a1) @ derivative[0]
        s_w = np.tanh(ww) - float(np.sum(fw * np.tanh(w)))
        e_tanh1 = self._conditional_tanh1(w)
        s_a1 = np.tanh(a1) - e_tanh1[:, None]
        total = np.sum(mass1 * term1 * self._pick(path, w=s_w, a1=s_a1))
        r1 = self.ratio1(a1, ww)
        q_obs = self.q1(ww, a1)
        for l2 in (0.0, 1.0):
            ll = np.full(a1.shape, l2)
            p = self.p_l2(ww, a1)
            mass2 = mass1 * (p if l2 else 1.0 - p)
            h = {"W": ww, "A1": a1, "L2": ll}
            c2 = self.cap(2, h)
            a2, wb = _segments(
                *A_RANGE, np.stack([self.edge(c2), c2, np.full(c2.shape, self.floor)], axis=-1)
            )
            g2 = self.g2(a2, ww[..., None], a1[..., None], ll[..., None])
            moved2 = self.move(a2, c2[..., None])
            value = self.m(ww[..., None], a1[..., None], ll[..., None], moved2)
            term2 = r1[..., None] * (value - q_obs[..., None])
            if derivative is not None:
                h2 = {key: np.broadcast_to(v[..., None], a2.shape) for key, v in h.items()}
                term2 = term2 + self.influence(2, h2, a2) @ derivative[1]
            s_l2 = np.broadcast_to((l2 - p)[..., None], a2.shape)
            s_a2 = np.tanh(a2) - self._conditional_tanh2(ww, a1, ll)[..., None]
            s = self._pick(
                path,
                w=np.broadcast_to(s_w[..., None], a2.shape),
                a1=np.broadcast_to(s_a1[..., None], a2.shape),
                l2=s_l2,
                a2=s_a2,
            )
            total = total + np.sum(mass2[..., None] * g2 * wb * term2 * s)
        return float(total)

    def spreads(self) -> tuple[float, float]:
        """The SD of the tuning term and the SD of :math:`D_\\Phi`, by quadrature.

        :math:`D_\\Phi` has three orthogonal pieces: :math:`Q^d_1(W, d(A_1)) - \\psi`,
        :math:`r_1 \\{m(d(A_2)) - Q_1(W, A_1)\\}` and :math:`r_1 r_2 (Y - m)`, the last with
        conditional variance :math:`m (1 - m) / 31` under the Beta law of ``Y``.
        """
        derivative = (self.derivative(1), self.derivative(2))
        psi = self.tuned_psi
        w, fw = self.w_grid
        c1 = self.cap(1, {"W": w})
        splits = np.stack([self.edge(c1), c1, np.full(w.size, self.floor)], axis=1)
        a1, wa = _segments(*A_RANGE, splits)
        ww = np.broadcast_to(w[:, None], a1.shape)
        mass1 = fw[:, None] * self.g1(a1, w[:, None]) * wa
        moved1 = self.move(a1, c1[:, None])
        piece0 = self.q1(ww, moved1) - psi
        tune1 = self.influence(1, {"W": ww}, a1) @ derivative[0]
        phi = float(np.sum(mass1 * piece0**2))
        term = 0.0
        r1 = self.ratio1(a1, ww)
        q_obs = self.q1(ww, a1)
        for l2 in (0.0, 1.0):
            ll = np.full(a1.shape, l2)
            p = self.p_l2(ww, a1)
            mass2 = mass1 * (p if l2 else 1.0 - p)
            h = {"W": ww, "A1": a1, "L2": ll}
            c2 = self.cap(2, h)
            a2, wb = _segments(
                *A_RANGE, np.stack([self.edge(c2), c2, np.full(c2.shape, self.floor)], axis=-1)
            )
            w3, a13, l3 = ww[..., None], a1[..., None], ll[..., None]
            g2 = self.g2(a2, w3, a13, l3)
            moved2 = self.move(a2, c2[..., None])
            piece1 = r1[..., None] * (self.m(w3, a13, l3, moved2) - q_obs[..., None])
            induced = self.induced(g2, self.g2(self.pre(a2), w3, a13, l3), a2, c2[..., None])
            r2 = np.where(g2 > 0.0, induced / np.where(g2 > 0.0, g2, 1.0), 0.0)
            mean = self.m(w3, a13, l3, a2)
            h2 = {key: np.broadcast_to(v[..., None], a2.shape) for key, v in h.items()}
            tune = tune1[..., None] + self.influence(2, h2, a2) @ derivative[1]
            joint = mass2[..., None] * g2 * wb
            noise = (r1[..., None] * r2) ** 2 * mean * (1.0 - mean) / 31.0
            phi += float(np.sum(joint * (piece1**2 + noise)))
            term += float(np.sum(joint * tune**2))
        return float(np.sqrt(term)), float(np.sqrt(phi))

    def efficient_sd(self) -> float:
        """The SD of :math:`D_\\Psi = D_\\Phi + \\sum_t \\partial_t\\Phi' \\mathrm{IF}_t`, by quadrature.

        Unlike :meth:`spreads`, the pieces are added before they are squared, so the
        correlation between the tuning term and :math:`D_\\Phi` is in the answer.
        """
        derivative = (self.derivative(1), self.derivative(2))
        psi = self.tuned_psi
        w, fw = self.w_grid
        c1 = self.cap(1, {"W": w})
        splits = np.stack([self.edge(c1), c1, np.full(w.size, self.floor)], axis=1)
        a1, wa = _segments(*A_RANGE, splits)
        ww = np.broadcast_to(w[:, None], a1.shape)
        mass1 = fw[:, None] * self.g1(a1, w[:, None]) * wa
        moved1 = self.move(a1, c1[:, None])
        piece0 = self.q1(ww, moved1) - psi
        tune1 = self.influence(1, {"W": ww}, a1) @ derivative[0]
        r1 = self.ratio1(a1, ww)
        q_obs = self.q1(ww, a1)
        second = 0.0
        mean = 0.0
        for l2 in (0.0, 1.0):
            ll = np.full(a1.shape, l2)
            p = self.p_l2(ww, a1)
            mass2 = mass1 * (p if l2 else 1.0 - p)
            h = {"W": ww, "A1": a1, "L2": ll}
            c2 = self.cap(2, h)
            a2, wb = _segments(
                *A_RANGE, np.stack([self.edge(c2), c2, np.full(c2.shape, self.floor)], axis=-1)
            )
            w3, a13, l3 = ww[..., None], a1[..., None], ll[..., None]
            g2 = self.g2(a2, w3, a13, l3)
            moved2 = self.move(a2, c2[..., None])
            piece1 = r1[..., None] * (self.m(w3, a13, l3, moved2) - q_obs[..., None])
            induced = self.induced(g2, self.g2(self.pre(a2), w3, a13, l3), a2, c2[..., None])
            r2 = np.where(g2 > 0.0, induced / np.where(g2 > 0.0, g2, 1.0), 0.0)
            mu = self.m(w3, a13, l3, a2)
            h2 = {key: np.broadcast_to(v[..., None], a2.shape) for key, v in h.items()}
            tune2 = self.influence(2, h2, a2) @ derivative[1]
            joint = mass2[..., None] * g2 * wb
            noise = (r1[..., None] * r2) ** 2 * mu * (1.0 - mu) / 31.0
            pre = piece0[..., None] + tune1[..., None] + piece1 + tune2
            second += float(np.sum(joint * (pre**2 + noise)))
            mean += float(np.sum(joint * pre))
        return float(np.sqrt(second - mean**2))

    @staticmethod
    def _pick(path: str, **scores: np.ndarray) -> np.ndarray:
        if path in scores:
            return scores[path]
        # Node 1's terms depend on (W, A1) only, so the L2 and A2 scores have mean zero
        # given them and contribute nothing there.
        return np.zeros_like(next(iter(scores.values())))

    def _conditional_tanh1(self, w: np.ndarray) -> np.ndarray:
        a, wts = _segments(*A_RANGE, (0.5 * w)[:, None])
        return np.sum(self.g1(a, w[:, None]) * np.tanh(a) * wts, axis=-1)

    def _conditional_tanh2(self, w: np.ndarray, a1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        mean = 0.4 * a1 + 0.6 * l2 + 0.3 * w
        a, wts = _segments(*A_RANGE, mean[..., None])
        return np.sum(
            self.g2(a, w[..., None], a1[..., None], l2[..., None]) * np.tanh(a) * wts, axis=-1
        )

    def tilted(self, path: str, eps: float) -> LawQ:
        return LawQ(tuners=self.tuners, tilt=Tilt(**{path: eps}), shift=self.s, factor=self.factor)


def pathwise_derivative(law: LawQ, path: str, step: float = 2e-3) -> float:
    """``d/de Psi(P_e)`` at zero by the four-point central difference."""

    def at(eps: float) -> float:
        return law.tilted(path, eps).tuned_psi

    return (8.0 * (at(step) - at(-step)) - (at(2 * step) - at(-2 * step))) / (12.0 * step)


def sample_node1(n: int, seed: int) -> pd.DataFrame:
    """The ``W`` and ``A1`` columns of :func:`sample` at the same seed, drawn alone."""
    rng = np.random.default_rng(seed)
    w = stats.truncnorm.rvs(-2.5, 2.5, size=n, random_state=rng)
    mean1 = 0.5 * w
    a1 = stats.truncnorm.rvs(-4.0 - mean1, 4.0 - mean1, loc=mean1, size=n, random_state=rng)
    return pd.DataFrame({"W": w, "A1": a1})


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` draws of law Q."""
    rng = np.random.default_rng(seed)
    w = stats.truncnorm.rvs(-2.5, 2.5, size=n, random_state=rng)
    mean1 = 0.5 * w
    a1 = stats.truncnorm.rvs(-4.0 - mean1, 4.0 - mean1, loc=mean1, size=n, random_state=rng)
    l2 = (rng.uniform(size=n) < expit(0.4 * a1 + 0.3 * w)).astype(float)
    mean2 = 0.4 * a1 + 0.6 * l2 + 0.3 * w
    a2 = stats.truncnorm.rvs(-4.0 - mean2, 4.0 - mean2, loc=mean2, size=n, random_state=rng)
    m = expit(3.0 * (a2 - LawQ().s) + 0.5 * l2 + 0.3 * a1 + 0.2 * w)
    y = rng.beta(30.0 * m, 30.0 * (1.0 - m))
    return pd.DataFrame({"W": w, "A1": a1, "L2": l2, "A2": a2, "Y": y})

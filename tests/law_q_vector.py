r"""Law Q with a flag at node 2: a tuned cap inside a ``JointPolicy``, by quadrature (M5).

The law extends ``tests/law_q_tuned.py``.  Node 1 is the dose :math:`A_1`.  Node 2 is the
vector :math:`(F_2, A_2)`: a binary flag :math:`F_2 \sim \mathrm{Bernoulli}(\mathrm{expit}(0.6
L_2 - 0.4 A_1 + 0.2 W - 0.2))` and the dose :math:`A_2 \sim N(0.4 A_1 + 0.6 L_2 + 0.3 W +
0.8 F_2 - 0.4, 1)` truncated to ``[-4, 4]``.  The outcome mean is :math:`m = \mathrm{expit}(3
(A_2 - s) + 0.5 L_2 + 0.3 A_1 + 0.2 W + 0.7 F_2)`, with :math:`s` the median of :math:`A_2`
minus 0.5.  The policy is ``Shift(1.0, cap=c_1(W))`` at node 1, and at node 2 a
``JointPolicy`` that sets the flag to 1 and moves the dose by ``Shift(1.0, cap=c_2(h))``.
The tuner of :math:`c_2` reads what the dose map reads: :math:`W, L_2, A_1` and the natural
flag :math:`F_2`.

The chain rule gives :math:`D_\Psi = D_\Phi + \sum_t \partial_t\Phi' \mathrm{IF}_t` with

.. math::

    \partial_2\Phi = E\bigl[W_1\, g_2(e_2 \mid H_2, F_2) |e'|
        \{m(H_2, 1, c_2) - m(H_2, 1, e_2)\} \partial c_2/\partial\theta_2\bigr],

where :math:`m` reads the assigned flag 1 and the natural flag enters only through the
dose's law and the tuner.  ``assigned_flag=False`` reads the natural flag there, which is
the mutation of a probe that sets only the tuned component.

The module imports no ``cleverly`` code (``tests/unit/test_mtp_tuned_components.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .law_q_tuned import A_RANGE, DELTA, W_RANGE, Tuner, _segments, expit

FLAGS = (0.0, 1.0)


@dataclass(frozen=True)
class Tilt:
    """The path coordinates: one exponent per tilted factor."""

    w: float = 0.0
    a1: float = 0.0
    l2: float = 0.0
    f2: float = 0.0
    a2: float = 0.0


@dataclass(frozen=True)
class LawQV:
    """Law Q with the flag, optionally tilted, with the tuners of the two nodes.

    Parameters
    ----------
    tuners : tuple of Tuner
        The tuner of node 1 (features from ``("W",)``) and of the dose at node 2 (from
        ``("W", "L2", "A1", "F2")``).
    tilt : Tilt
        The path coordinates.
    shift : float or None
        The outcome's offset ``s``; ``None`` computes it at the untilted law.
    factor : float or None
        ``Scale(factor)`` in place of ``Shift(1.0)`` at both nodes.
    assigned_flag : bool
        Whether the derivative of node 2 reads the assigned flag (the law) or the natural
        flag (the mutation).
    """

    tuners: tuple[Tuner, Tuner] = (Tuner(), Tuner())
    tilt: Tilt = field(default_factory=Tilt)
    shift: float | None = None
    factor: float | None = None
    assigned_flag: bool = True

    # ------------------------------------------------------------ the map

    def edge(self, c: np.ndarray) -> np.ndarray:
        return c - DELTA if self.factor is None else c / self.factor

    def move(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        moved = x + DELTA if self.factor is None else self.factor * x
        return np.where(moved <= c, moved, x)

    def pre(self, a: np.ndarray) -> np.ndarray:
        return a - DELTA if self.factor is None else a / self.factor

    @property
    def slope(self) -> float:
        return 1.0 if self.factor is None else 1.0 / self.factor

    @property
    def floor(self) -> float:
        return A_RANGE[0] + DELTA if self.factor is None else A_RANGE[0] * self.factor

    def induced(self, g: np.ndarray, gpre: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
        return gpre * self.slope * (a <= c) + g * (a > self.edge(c))

    # ------------------------------------------------------------ factors

    @cached_property
    def w_grid(self) -> tuple[np.ndarray, np.ndarray]:
        x, wts = _segments(*W_RANGE, np.array([[0.0]]))
        x, wts = x[0], wts[0]
        mass = stats.norm.pdf(x) * np.exp(self.tilt.w * np.tanh(x)) * wts
        return x, mass / mass.sum()

    @staticmethod
    def _density(a: np.ndarray, mean: np.ndarray, eps: float) -> np.ndarray:
        lo, hi = A_RANGE
        inside = (a >= lo) & (a <= hi)
        raw = stats.norm.pdf(a - mean) * np.exp(eps * np.tanh(a))
        return np.where(inside, raw / LawQV._z(mean, eps), 0.0)

    @staticmethod
    def _z(mean: np.ndarray, eps: float) -> np.ndarray:
        mean = np.asarray(mean, dtype=float)
        x, wts = _segments(*A_RANGE, mean[..., None] * np.ones(1))
        raw = stats.norm.pdf(x - mean[..., None]) * np.exp(eps * np.tanh(x))
        return np.sum(raw * wts, axis=-1)

    @staticmethod
    def _cdf(t: np.ndarray, mean: np.ndarray, eps: float) -> np.ndarray:
        t, mean = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(mean, dtype=float))
        lo, hi = A_RANGE
        top = np.clip(t, lo, hi)
        x, wts = _segments(lo, 0.0, np.zeros((*t.shape, 0)))
        scale = (top - lo) / (0.0 - lo)
        x = lo + (x - lo) * scale[..., None]
        wts = wts * scale[..., None]
        raw = stats.norm.pdf(x - mean[..., None]) * np.exp(eps * np.tanh(x))
        return np.sum(raw * wts, axis=-1) / LawQV._z(mean, eps)

    @staticmethod
    def mean2(w: np.ndarray, a1: np.ndarray, l2: np.ndarray, f2: np.ndarray) -> np.ndarray:
        return 0.4 * a1 + 0.6 * l2 + 0.3 * w + 0.8 * f2 - 0.4

    def g1(self, a: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self._density(a, 0.5 * w, self.tilt.a1)

    def g2(self, a: np.ndarray, w: Any, a1: Any, l2: Any, f2: Any) -> np.ndarray:
        return self._density(a, self.mean2(w, a1, l2, f2), self.tilt.a2)

    def cdf1(self, t: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self._cdf(t, 0.5 * w, self.tilt.a1)

    def cdf2(self, t: np.ndarray, w: Any, a1: Any, l2: Any, f2: Any) -> np.ndarray:
        return self._cdf(t, self.mean2(w, a1, l2, f2), self.tilt.a2)

    def p_l2(self, w: np.ndarray, a1: np.ndarray) -> np.ndarray:
        return expit(0.4 * a1 + 0.3 * w + self.tilt.l2)

    def p_f2(self, w: np.ndarray, a1: np.ndarray, l2: np.ndarray) -> np.ndarray:
        return expit(0.6 * l2 - 0.4 * a1 + 0.2 * w - 0.2 + self.tilt.f2)

    def m(self, w: Any, a1: Any, l2: Any, f2: Any, a2: Any) -> np.ndarray:
        return expit(3.0 * (a2 - self.s) + 0.5 * l2 + 0.3 * a1 + 0.2 * w + 0.7 * f2)

    @cached_property
    def s(self) -> float:
        if self.shift is not None:
            return float(self.shift)
        base = LawQV(tuners=self.tuners, shift=0.0)
        return float(base.marginal_quantile(2, 0.5)) - 0.5

    # ------------------------------------------------------------ the observed history

    def _observed_h2(self) -> dict[str, np.ndarray]:
        """``W, A1, L2, F2`` of the observed node-2 history and its mass, flattened."""
        w, fw = self.w_grid
        a1, wa = _segments(*A_RANGE, (0.5 * w)[:, None])
        mass = fw[:, None] * self.g1(a1, w[:, None]) * wa
        ww = np.broadcast_to(w[:, None], a1.shape)
        out: dict[str, list[np.ndarray]] = {"W": [], "A1": [], "L2": [], "F2": [], "mass": []}
        for l2 in FLAGS:
            pl = self.p_l2(ww, a1)
            ll = np.full(a1.shape, l2)
            pf = self.p_f2(ww, a1, ll)
            for f2 in FLAGS:
                weight = mass * (pl if l2 else 1.0 - pl) * (pf if f2 else 1.0 - pf)
                out["W"].append(ww.ravel())
                out["A1"].append(a1.ravel())
                out["L2"].append(ll.ravel())
                out["F2"].append(np.full(a1.size, f2))
                out["mass"].append(weight.ravel())
        return {key: np.concatenate(value) for key, value in out.items()}

    def marginal_cdf(self, node: int, t: float) -> float:
        if node == 1:
            w, fw = self.w_grid
            return float(np.sum(fw * self.cdf1(np.full(w.size, t), w)))
        h = self._observed_h2()
        values = self.cdf2(np.full(h["W"].size, t), h["W"], h["A1"], h["L2"], h["F2"])
        return float(np.sum(h["mass"] * values))

    def marginal_density(self, node: int, t: float) -> float:
        if node == 1:
            w, fw = self.w_grid
            return float(np.sum(fw * self.g1(np.full(w.size, t), w)))
        h = self._observed_h2()
        values = self.g2(np.full(h["W"].size, t), h["W"], h["A1"], h["L2"], h["F2"])
        return float(np.sum(h["mass"] * values))

    def marginal_quantile(self, node: int, q: float) -> float:
        return float(
            optimize.brentq(lambda t: self.marginal_cdf(node, t) - q, -4.0, 4.0, xtol=1e-14)
        )

    # ------------------------------------------------------------ tuners

    @staticmethod
    def _features(names: tuple[str, ...], h: dict[str, np.ndarray]) -> np.ndarray:
        columns = [np.ones_like(np.asarray(h["W"], dtype=float))] + [
            np.asarray(h[name], dtype=float) for name in names
        ]
        return np.stack(np.broadcast_arrays(*columns), axis=-1)

    @cached_property
    def theta(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._theta(1), self._theta(2))

    def _history(self, node: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if node == 1:
            w, fw = self.w_grid
            return {"W": w}, fw
        h = self._observed_h2()
        return {key: h[key] for key in ("W", "A1", "L2", "F2")}, h["mass"]

    def _node_law(self, node: int, h: dict[str, np.ndarray]) -> tuple[Any, Any]:
        if node == 1:
            return (lambda t: self.cdf1(t, h["W"]), lambda t: self.g1(t, h["W"]))
        args = (h["W"], h["A1"], h["L2"], h["F2"])
        return (lambda t: self.cdf2(t, *args), lambda t: self.g2(t, *args))

    def _theta(self, node: int) -> np.ndarray:
        tuner = self.tuners[node - 1]
        if not tuner.features:
            return np.array([self.marginal_quantile(node, tuner.q)])
        h, mass = self._history(node)
        x = self._features(tuner.features, h)
        cdf, density = self._node_law(node, h)
        beta = np.zeros(x.shape[-1])
        for _ in range(60):
            fit = x @ beta
            score = x.T @ (mass * (tuner.q - cdf(fit)))
            jacobian = (x * (mass * density(fit))[:, None]).T @ x
            step = np.linalg.solve(jacobian, score)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-15:
                break
        return beta

    def cap(
        self, node: int, h: dict[str, np.ndarray], theta: np.ndarray | None = None
    ) -> np.ndarray:
        tuner = self.tuners[node - 1]
        value = self.theta[node - 1] if theta is None else theta
        if not tuner.features:
            return np.full(np.shape(h["W"]), float(value[0]))
        return self._features(tuner.features, h) @ value

    def gradient(self, node: int, h: dict[str, np.ndarray]) -> np.ndarray:
        tuner = self.tuners[node - 1]
        if not tuner.features:
            return np.ones((*np.shape(h["W"]), 1))
        return self._features(tuner.features, h)

    # ------------------------------------------------------------ the parameter

    def q1(self, w: np.ndarray, b: np.ndarray, theta2: np.ndarray | None = None) -> np.ndarray:
        """``E[m(H_2, 1, d_2(A_2)) | A_1 = b, W = w]``, elementwise."""
        w, b = np.broadcast_arrays(np.asarray(w, dtype=float), np.asarray(b, dtype=float))
        total = np.zeros(w.shape)
        for l2 in FLAGS:
            ll = np.full(w.shape, l2)
            pl = self.p_l2(w, b)
            pf = self.p_f2(w, b, ll)
            for f2 in FLAGS:
                ff = np.full(w.shape, f2)
                weight = (pl if l2 else 1.0 - pl) * (pf if f2 else 1.0 - pf)
                c2 = self.cap(2, {"W": w, "A1": b, "L2": ll, "F2": ff}, theta2)
                x, wts = _segments(*A_RANGE, self.edge(c2)[..., None])
                moved = self.move(x, c2[..., None])
                g = self.g2(x, w[..., None], b[..., None], ll[..., None], ff[..., None])
                value = self.m(w[..., None], b[..., None], ll[..., None], 1.0, moved)
                total = total + weight * np.sum(g * value * wts, axis=-1)
        return total

    def psi(self, theta: tuple[np.ndarray, np.ndarray] | None = None) -> float:
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
        for l2 in FLAGS:
            ll = np.full(a1.shape, l2)
            pl = self.p_l2(ww, b)
            pf = self.p_f2(ww, b, ll)
            for f2 in FLAGS:
                ff = np.full(a1.shape, f2)
                weight = mass * (pl if l2 else 1.0 - pl) * (pf if f2 else 1.0 - pf)
                h = {"W": ww, "A1": b, "L2": ll, "F2": ff}
                c2 = self.cap(2, h)
                e2 = self.edge(c2)
                read = 1.0 if self.assigned_flag else ff
                jump = (
                    self.slope
                    * self.g2(e2, ww, b, ll, ff)
                    * (self.m(ww, b, ll, read, c2) - self.m(ww, b, ll, read, e2))
                )
                total = total + np.sum(
                    (weight * jump)[..., None] * self.gradient(2, h), axis=(0, 1)
                )
        return np.asarray(total)

    def ratio1(self, a1: np.ndarray, w: np.ndarray) -> np.ndarray:
        c1 = self.cap(1, {"W": w})
        g = self.g1(a1, w)
        induced = self.induced(g, self.g1(self.pre(a1), w), a1, c1)
        return np.where(g > 0.0, induced / np.where(g > 0.0, g, 1.0), 0.0)

    def influence(self, node: int, h: dict[str, np.ndarray], a: np.ndarray) -> np.ndarray:
        """``(..., p)`` the tuner's influence function at the observed dose ``a``."""
        tuner = self.tuners[node - 1]
        theta = self.theta[node - 1]
        if not tuner.features:
            density = self.marginal_density(node, float(theta[0]))
            return (-((a <= theta[0]).astype(float) - tuner.q) / density)[..., None]
        hist, mass = self._history(node)
        x = self._features(tuner.features, hist)
        _, density = self._node_law(node, hist)
        jacobian = (x * (mass * density(x @ theta))[:, None]).T @ x
        here = self._features(tuner.features, h)
        score = here * (tuner.q - (a <= here @ theta).astype(float))[..., None]
        flat = np.linalg.solve(jacobian, score.reshape(-1, x.shape[-1]).T).T
        return flat.reshape(score.shape)

    def score_pairing(self, path: str, *, tuning: bool = True) -> float:
        """:math:`E[D_\\Psi s]` for the score ``s`` of one path, by quadrature."""
        derivative = (self.derivative(1), self.derivative(2)) if tuning else None
        psi = self.tuned_psi
        w, fw = self.w_grid
        c1 = self.cap(1, {"W": w})
        splits = np.stack([self.edge(c1), c1, np.full(w.size, self.floor)], axis=1)
        a1, wa = _segments(*A_RANGE, splits)
        ww = np.broadcast_to(w[:, None], a1.shape)
        mass1 = fw[:, None] * self.g1(a1, w[:, None]) * wa
        term1 = self.q1(ww, self.move(a1, c1[:, None])) - psi
        if derivative is not None:
            term1 = term1 + self.influence(1, {"W": ww}, a1) @ derivative[0]
        s_w = np.tanh(ww) - float(np.sum(fw * np.tanh(w)))
        s_a1 = np.tanh(a1) - self._conditional_tanh(0.5 * w, self.tilt.a1)[:, None]
        scores1 = {"w": s_w, "a1": s_a1}
        total = float(np.sum(mass1 * term1 * scores1.get(path, np.zeros_like(s_w))))
        r1 = self.ratio1(a1, ww)
        q_obs = self.q1(ww, a1)
        for l2 in FLAGS:
            ll = np.full(a1.shape, l2)
            pl = self.p_l2(ww, a1)
            pf = self.p_f2(ww, a1, ll)
            for f2 in FLAGS:
                ff = np.full(a1.shape, f2)
                mass2 = mass1 * (pl if l2 else 1.0 - pl) * (pf if f2 else 1.0 - pf)
                h = {"W": ww, "A1": a1, "L2": ll, "F2": ff}
                c2 = self.cap(2, h)
                edges = np.stack([self.edge(c2), c2, np.full(c2.shape, self.floor)], axis=-1)
                a2, wb = _segments(*A_RANGE, edges)
                w3, a13, l3, f3 = ww[..., None], a1[..., None], ll[..., None], ff[..., None]
                g2 = self.g2(a2, w3, a13, l3, f3)
                value = self.m(w3, a13, l3, 1.0, self.move(a2, c2[..., None]))
                term2 = r1[..., None] * (value - q_obs[..., None])
                if derivative is not None:
                    h2 = {key: np.broadcast_to(v[..., None], a2.shape) for key, v in h.items()}
                    term2 = term2 + self.influence(2, h2, a2) @ derivative[1]
                mean = self.mean2(ww, a1, ll, ff)
                scores = {
                    "w": np.broadcast_to(s_w[..., None], a2.shape),
                    "a1": np.broadcast_to(s_a1[..., None], a2.shape),
                    "l2": np.broadcast_to((l2 - pl)[..., None], a2.shape),
                    "f2": np.broadcast_to((f2 - pf)[..., None], a2.shape),
                    "a2": np.tanh(a2) - self._conditional_tanh(mean, self.tilt.a2)[..., None],
                }
                total += float(np.sum(mass2[..., None] * g2 * wb * term2 * scores[path]))
        return total

    def _conditional_tanh(self, mean: np.ndarray, eps: float) -> np.ndarray:
        mean = np.asarray(mean, dtype=float)
        a, wts = _segments(*A_RANGE, mean[..., None])
        return np.sum(self._density(a, mean[..., None], eps) * np.tanh(a) * wts, axis=-1)

    def tilted(self, path: str, eps: float) -> LawQV:
        return LawQV(
            tuners=self.tuners,
            tilt=Tilt(**{path: eps}),
            shift=self.s,
            factor=self.factor,
            assigned_flag=self.assigned_flag,
        )


def pathwise_derivative(law: LawQV, path: str, step: float = 2e-3) -> float:
    """``d/de Psi(P_e)`` at zero by the four-point central difference."""

    def at(eps: float) -> float:
        return law.tilted(path, eps).tuned_psi

    return (8.0 * (at(step) - at(-step)) - (at(2 * step) - at(-2 * step))) / (12.0 * step)


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` draws of the law: ``W, A1, L2, F2, A2, Y``."""
    rng = np.random.default_rng(seed)
    w = stats.truncnorm.rvs(-2.5, 2.5, size=n, random_state=rng)
    mean1 = 0.5 * w
    a1 = stats.truncnorm.rvs(-4.0 - mean1, 4.0 - mean1, loc=mean1, size=n, random_state=rng)
    l2 = (rng.uniform(size=n) < expit(0.4 * a1 + 0.3 * w)).astype(float)
    f2 = (rng.uniform(size=n) < expit(0.6 * l2 - 0.4 * a1 + 0.2 * w - 0.2)).astype(float)
    mean2 = LawQV.mean2(w, a1, l2, f2)
    a2 = stats.truncnorm.rvs(-4.0 - mean2, 4.0 - mean2, loc=mean2, size=n, random_state=rng)
    law = LawQV()
    y = rng.beta(30.0 * law.m(w, a1, l2, f2, a2), 30.0 * (1.0 - law.m(w, a1, l2, f2, a2)))
    return pd.DataFrame({"W": w, "A1": a1, "L2": l2, "F2": f2, "A2": a2, "Y": y})

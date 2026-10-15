r"""The point-treatment law of the X40 tuned-cap tests: node 1 of law Q with its own outcome.

The implementation never enters this module (``tests/unit/test_point_tuned_caps.py`` checks
its imports).  The law is the first node of :mod:`tests.law_q_tuned`: :math:`W \sim N(0, 1)`
truncated to ``[-2.5, 2.5]`` and :math:`A \mid W \sim N(0.5 W, 1)` truncated to ``[-4, 4]``,
with the outcome :math:`Y \sim \mathrm{Beta}(30 m, 30 (1 - m))`,
:math:`m(a, w) = \mathrm{expit}(3 (a - s) + 0.3 w)` and :math:`s` the median of :math:`A`
minus 0.5.  The policy is ``Shift(1.0, cap=c(w))``.  The parameter, the tuner and the curve

.. math::

    D_\Psi = r(A, W)\{Y - m(A, W)\} + m(d(A, W), W) - \Psi + \partial\Phi' \mathrm{IF}

are written longhand by Gauss-Legendre quadrature split at every jump, with
:math:`r = g^d / g` and :math:`\partial\Phi = E[g(e \mid W)\{m(c, W) - m(e, W)\}
\partial c/\partial\theta]`.  The residual term is mean zero given :math:`(A, W)`, so it
pairs with no score of the :math:`W` or :math:`A` law; the identity of T-e4 reads those two
paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from .law_q_tuned import A_RANGE, DELTA, LawQ, Tilt, Tuner, _segments, expit

__all__ = ["PointLawQ", "pathwise_derivative", "sample"]


@dataclass(frozen=True)
class PointLawQ(LawQ):
    """The point law, optionally tilted along the ``W`` or the ``A`` path.

    Parameters
    ----------
    tuner : Tuner
        The tuner of the cap: a quantile, or a linear quantile regression on ``("W",)``.
    tilt : Tilt
        The path coordinates; only ``w`` and ``a1`` apply.
    shift : float or None
        The outcome's offset ``s``; ``None`` computes it at the untilted law.
    """

    tuner: Tuner = field(default_factory=Tuner)
    tilt: Tilt = field(default_factory=Tilt)
    shift: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tuners", (self.tuner, Tuner()))

    @cached_property
    def s(self) -> float:
        if self.shift is not None:
            return float(self.shift)
        base = PointLawQ(tuner=self.tuner, shift=0.0)
        return float(base.marginal_quantile(1, 0.5)) - 0.5

    def m_point(self, w: np.ndarray, a: np.ndarray) -> np.ndarray:
        """:math:`m(a, w) = E[Y \\mid A = a, W = w]`."""
        return expit(3.0 * (np.asarray(a) - self.s) + 0.3 * np.asarray(w))

    @cached_property
    def theta(self) -> tuple[np.ndarray, np.ndarray]:
        """The population tuner of the one node; the second entry is unused."""
        return (self._theta(1), np.zeros(1))

    def q1(self, w: np.ndarray, b: np.ndarray, theta2: np.ndarray | None = None) -> np.ndarray:
        """The outcome regression at dose ``b``: the point law has no later node."""
        w, b = np.broadcast_arrays(np.asarray(w, dtype=float), np.asarray(b, dtype=float))
        return self.m_point(w, b)

    def _node(self) -> tuple[np.ndarray, ...]:
        """``(w, a, mass, moved)`` on the observed ``(W, A)`` law, split at ``e`` and ``c``."""
        w, fw = self.w_grid
        c = self.cap(1, {"W": w})
        splits = np.stack([c - DELTA, c, np.full(w.size, A_RANGE[0] + DELTA)], axis=1)
        a, wa = _segments(*A_RANGE, splits)
        ww = np.broadcast_to(w[:, None], a.shape)
        mass = fw[:, None] * self.g1(a, w[:, None]) * wa
        moved = np.where(a + DELTA <= c[:, None], a + DELTA, a)
        return ww, a, mass, moved

    def score_pairing(self, path: str, *, tuning: bool = True) -> float:
        """:math:`E[D_\\Psi s]` for the score of the ``"w"`` or the ``"a1"`` path.

        ``tuning=False`` drops the tuner's term, the mutation M-e4 of the identity.
        """
        ww, a, mass, moved = self._node()
        term = self.m_point(ww, moved) - self.tuned_psi
        if tuning:
            term = term + self.influence(1, {"W": ww}, a) @ self.derivative(1)
        w, fw = self.w_grid
        if path == "w":
            score = np.tanh(ww) - float(np.sum(fw * np.tanh(w)))
        elif path == "a1":
            score = np.tanh(a) - self._conditional_tanh1(w)[:, None]
        else:
            raise ValueError(f"the point law has the paths 'w' and 'a1'; got {path!r}")
        return float(np.sum(mass * term * score))

    def spreads(self) -> tuple[float, float]:
        """The SD of the tuner's term and the SD of :math:`D_\\Phi`, by quadrature."""
        ww, a, mass, moved = self._node()
        tune = self.influence(1, {"W": ww}, a) @ self.derivative(1)
        plug = self.m_point(ww, moved) - self.tuned_psi
        ratio = self.ratio1(a, ww)
        mean = self.m_point(ww, a)
        noise = ratio**2 * mean * (1.0 - mean) / 31.0
        return (
            float(np.sqrt(np.sum(mass * tune**2))),
            float(np.sqrt(np.sum(mass * (plug**2 + noise)))),
        )

    def tilted(self, path: str, eps: float) -> PointLawQ:  # type: ignore[override]
        return PointLawQ(tuner=self.tuner, tilt=Tilt(**{path: eps}), shift=self.s)


def pathwise_derivative(law: PointLawQ, path: str, step: float = 2e-3) -> float:
    """``d/de Psi(P_e)`` at zero by the four-point central difference."""

    def at(eps: float) -> float:
        return law.tilted(path, eps).tuned_psi

    return (8.0 * (at(step) - at(-step)) - (at(2 * step) - at(-2 * step))) / (12.0 * step)


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` draws of the point law: columns ``W``, ``A`` and ``Y``."""
    rng = np.random.default_rng(seed)
    w = stats.truncnorm.rvs(-2.5, 2.5, size=n, random_state=rng)
    mean = 0.5 * w
    a = stats.truncnorm.rvs(-4.0 - mean, 4.0 - mean, loc=mean, size=n, random_state=rng)
    m = expit(3.0 * (a - PointLawQ().s) + 0.3 * w)
    y = rng.beta(30.0 * m, 30.0 * (1.0 - m))
    return pd.DataFrame({"W": w, "A": a, "Y": y})

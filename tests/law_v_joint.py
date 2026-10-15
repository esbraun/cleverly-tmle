r"""Law V: one vector node with tuned caps inside a ``JointPolicy``, by quadrature.

The node is :math:`(F, A, B)` after the baseline :math:`W \sim N(0, 1)` truncated to
``[-2.5, 2.5]``.  The flag is :math:`F \sim \mathrm{Bernoulli}(\mathrm{expit}(0.5 W -
0.1))`.  The doses are :math:`A \sim N(0.5 W + 0.6 F - 0.2, 1)` and :math:`B \sim N(0.4 W +
0.5 A + 0.4 F - 0.3, 1)`, both truncated to ``[-4, 4]``.  The outcome mean is :math:`m =
\mathrm{expit}(1.2 (A - 0.5) + 1.2 (B - 0.5) + 0.3 W + 0.5 F)`.

A policy is a list of branches of known probability.  Each branch sets the flag (or keeps
it) and moves ``A``, ``B`` or both by ``Shift(1.0, cap=c(h))``.  The tuner of ``A`` reads
:math:`W, F`; the tuner of ``B`` reads :math:`W, F` and the natural :math:`A`, which is what
each dose map may read in the order ``(A, B)``.  With :math:`\Psi = \sum_b p_b \Phi_b(\theta)`
the chain rule gives one Leibniz term per branch and tuned component:

.. math::

    \partial_A\Phi = \sum_b p_b E\bigl[g_A(e_A \mid W, F) |e'| \int g_B(b \mid W, F, e_A)
        \{m(v^b[A = c_A]) - m(v^b[A = e_A])\}\,db\; \partial c_A/\partial\theta_A\bigr],

where :math:`v^b` is the vector branch ``b`` assigns at the natural :math:`A = e_A`, and the
same with the roles of the doses swapped for ``B``.  The mutations: drop a component's
term, drop a branch's term or its probability, or read the natural flag in the term.

The module imports no ``cleverly`` code (``tests/unit/test_mtp_joint_tuning.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .law_q_tuned import A_RANGE, DELTA, W_RANGE, Tuner, _segments, expit
from .law_q_vector import LawQV

FLAGS = (0.0, 1.0)
DOSES = ("A", "B")


@dataclass(frozen=True)
class Tilt:
    """The path coordinates: one exponent per tilted factor."""

    w: float = 0.0
    f: float = 0.0
    a: float = 0.0
    b: float = 0.0


@dataclass(frozen=True)
class Branch:
    """One branch of the policy.

    Parameters
    ----------
    probability : float
        The known branch probability.
    flag : float or None
        The flag the branch sets, ``None`` for the natural flag.
    tuned : tuple of str
        The doses the branch moves by the tuned shift; the others keep their natural value.
    """

    probability: float
    flag: float | None
    tuned: tuple[str, ...]


@dataclass(frozen=True)
class Curve:
    """Which terms the curve carries: the law, or one of its mutations.

    Parameters
    ----------
    doses : tuple of str
        The doses whose tuning term the curve carries.
    skip : tuple of int
        The branches whose terms the curve drops.
    weighted : bool
        Whether a branch's term carries its probability.
    assigned_flag : bool
        Whether the term reads the flag the branch assigns or the natural flag.
    """

    doses: tuple[str, ...] = DOSES
    skip: tuple[int, ...] = ()
    weighted: bool = True
    assigned_flag: bool = True


@dataclass(frozen=True)
class LawV:
    """Law V, optionally tilted, with a policy and the tuners of its two doses.

    Parameters
    ----------
    branches : tuple of Branch
        The policy.
    tuners : tuple of Tuner
        The tuner of ``A`` (features from ``("W", "F")``) and of ``B`` (from
        ``("W", "F", "A")``).
    tilt : Tilt
        The path coordinates.
    """

    branches: tuple[Branch, ...] = (Branch(1.0, 1.0, DOSES),)
    tuners: tuple[Tuner, Tuner] = (Tuner(), Tuner())
    tilt: Tilt = field(default_factory=Tilt)

    # ------------------------------------------------------------ the map

    @staticmethod
    def edge(c: Any) -> Any:
        return c - DELTA

    @staticmethod
    def move(x: Any, c: Any) -> Any:
        return np.where(x + DELTA <= c, x + DELTA, x)

    # ------------------------------------------------------------ factors

    @cached_property
    def w_grid(self) -> tuple[np.ndarray, np.ndarray]:
        x, wts = _segments(*W_RANGE, np.array([[0.0]]))
        x, wts = x[0], wts[0]
        mass = stats.norm.pdf(x) * np.exp(self.tilt.w * np.tanh(x)) * wts
        return x, mass / mass.sum()

    def p_f(self, w: Any) -> Any:
        return expit(0.5 * w - 0.1 + self.tilt.f)

    @staticmethod
    def mean_a(w: Any, f: Any) -> Any:
        return 0.5 * w + 0.6 * f - 0.2

    @staticmethod
    def mean_b(w: Any, f: Any, a: Any) -> Any:
        return 0.4 * w + 0.5 * a + 0.4 * f - 0.3

    def g(self, dose: str, x: Any, h: dict[str, Any]) -> Any:
        """The density of ``dose`` at ``x`` given its history ``h``."""
        if dose == "A":
            return LawQV._density(x, self.mean_a(h["W"], h["F"]), self.tilt.a)
        return LawQV._density(x, self.mean_b(h["W"], h["F"], h["A"]), self.tilt.b)

    def cdf(self, dose: str, x: Any, h: dict[str, Any]) -> Any:
        if dose == "A":
            return LawQV._cdf(x, self.mean_a(h["W"], h["F"]), self.tilt.a)
        return LawQV._cdf(x, self.mean_b(h["W"], h["F"], h["A"]), self.tilt.b)

    def conditional_tanh(self, dose: str, h: dict[str, Any]) -> Any:
        mean = self.mean_a(h["W"], h["F"]) if dose == "A" else self.mean_b(h["W"], h["F"], h["A"])
        eps = self.tilt.a if dose == "A" else self.tilt.b
        # The grid repeats each history along the later doses, so each distinct mean is
        # integrated once.
        unique, inverse = np.unique(np.asarray(mean, dtype=float), return_inverse=True)
        x, wts = _segments(*A_RANGE, unique[:, None])
        values = np.sum(LawQV._density(x, unique[:, None], eps) * np.tanh(x) * wts, axis=-1)
        return values[inverse].reshape(np.shape(mean))

    @staticmethod
    def m(w: Any, f: Any, a: Any, b: Any) -> Any:
        return expit(1.2 * (a - 0.5) + 1.2 * (b - 0.5) + 0.3 * w + 0.5 * f)

    # ------------------------------------------------------------ the tuners

    def _history(self, dose: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """The history a dose's law conditions on, flattened, and its mass."""
        w, fw = self.w_grid
        ws, fs, mass = [], [], []
        for f in FLAGS:
            pf = self.p_f(w)
            ws.append(w)
            fs.append(np.full(w.size, f))
            mass.append(fw * (pf if f else 1.0 - pf))
        h = {"W": np.concatenate(ws), "F": np.concatenate(fs)}
        weight = np.concatenate(mass)
        if dose == "A":
            return h, weight
        a, wa = _segments(*A_RANGE, self.mean_a(h["W"], h["F"])[:, None])
        out = {
            "W": np.broadcast_to(h["W"][:, None], a.shape).ravel(),
            "F": np.broadcast_to(h["F"][:, None], a.shape).ravel(),
            "A": a.ravel(),
        }
        return out, (
            weight[:, None] * self.g("A", a, {k: v[:, None] for k, v in h.items()}) * wa
        ).ravel()

    def marginal_cdf(self, dose: str, t: float) -> float:
        h, mass = self._history(dose)
        return float(np.sum(mass * self.cdf(dose, np.full(mass.size, t), h)))

    def marginal_density(self, dose: str, t: float) -> float:
        h, mass = self._history(dose)
        return float(np.sum(mass * self.g(dose, np.full(mass.size, t), h)))

    @staticmethod
    def _features(names: tuple[str, ...], h: dict[str, Any]) -> np.ndarray:
        columns = [np.ones_like(np.asarray(h["W"], dtype=float))] + [
            np.asarray(h[name], dtype=float) for name in names
        ]
        return np.stack(np.broadcast_arrays(*columns), axis=-1)

    def tuner(self, dose: str) -> Tuner:
        return self.tuners[DOSES.index(dose)]

    @cached_property
    def theta(self) -> dict[str, np.ndarray]:
        return {dose: self._theta(dose) for dose in DOSES}

    def _theta(self, dose: str) -> np.ndarray:
        tuner = self.tuner(dose)
        if not tuner.features:
            return np.array(
                [
                    optimize.brentq(
                        lambda t: self.marginal_cdf(dose, t) - tuner.q, -4.0, 4.0, xtol=1e-14
                    )
                ]
            )
        h, mass = self._history(dose)
        x = self._features(tuner.features, h)
        beta = np.zeros(x.shape[-1])
        for _ in range(60):
            fit = x @ beta
            score = x.T @ (mass * (tuner.q - self.cdf(dose, fit, h)))
            jacobian = (x * (mass * self.g(dose, fit, h))[:, None]).T @ x
            step = np.linalg.solve(jacobian, score)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-15:
                break
        return beta

    def cap(self, dose: str, h: dict[str, Any], theta: np.ndarray | None = None) -> Any:
        tuner = self.tuner(dose)
        value = self.theta[dose] if theta is None else theta
        if not tuner.features:
            return np.full(np.shape(h["W"]), float(value[0]))
        return self._features(tuner.features, h) @ value

    def gradient(self, dose: str, h: dict[str, Any]) -> np.ndarray:
        tuner = self.tuner(dose)
        if not tuner.features:
            return np.ones((*np.shape(h["W"]), 1))
        return self._features(tuner.features, h)

    def influence(self, dose: str, h: dict[str, Any], x: Any) -> np.ndarray:
        """``(..., p)`` the tuner's influence function at the observed dose ``x``."""
        tuner = self.tuner(dose)
        theta = self.theta[dose]
        if not tuner.features:
            density = self.marginal_density(dose, float(theta[0]))
            return (-((x <= theta[0]).astype(float) - tuner.q) / density)[..., None]
        hist, mass = self._history(dose)
        design = self._features(tuner.features, hist)
        jacobian = (design * (mass * self.g(dose, design @ theta, hist))[:, None]).T @ design
        here = self._features(tuner.features, h)
        score = here * (tuner.q - (x <= here @ theta).astype(float))[..., None]
        flat = np.linalg.solve(jacobian, score.reshape(-1, design.shape[-1]).T).T
        return flat.reshape(score.shape)

    # ------------------------------------------------------------ the parameter

    def assigned(self, branch: Branch, a: Any, b: Any, c_a: Any, c_b: Any) -> tuple[Any, Any]:
        """The doses a branch assigns at the natural ``a, b``, with caps ``c_a, c_b``."""
        return (
            self.move(a, c_a) if "A" in branch.tuned else a,
            self.move(b, c_b) if "B" in branch.tuned else b,
        )

    def _grid(self, f: float, theta: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """The quadrature of ``(W, A, B)`` at flag ``f``, split at the caps and edges."""
        theta = self.theta if theta is None else theta
        w, fw = self.w_grid
        pf = self.p_f(w)
        ff = np.full(w.shape, f)
        c_a = self.cap("A", {"W": w, "F": ff}, theta["A"])
        a, wa = _segments(*A_RANGE, np.stack([self.edge(c_a), c_a], axis=-1))
        w2 = np.broadcast_to(w[:, None], a.shape)
        h_b = {"W": w2, "F": np.full(a.shape, f), "A": a}
        c_b = self.cap("B", h_b, theta["B"])
        b, wb = _segments(*A_RANGE, np.stack([self.edge(c_b), c_b], axis=-1))
        mass = (
            (fw * (pf if f else 1.0 - pf))[:, None, None]
            * (self.g("A", a, {"W": w2, "F": f}) * wa)[..., None]
            * self.g("B", b, {k: v[..., None] for k, v in h_b.items()})
            * wb
        )
        shape = b.shape
        return {
            "W": np.broadcast_to(w[:, None, None], shape),
            "F": np.full(shape, f),
            "A": np.broadcast_to(a[..., None], shape),
            "B": b,
            "c_a": np.broadcast_to(c_a[:, None, None], shape),
            "c_b": np.broadcast_to(c_b[..., None], shape),
            "mass": mass,
        }

    def value(self, grid: dict[str, np.ndarray]) -> np.ndarray:
        """``sum_b p_b m(v^b)`` at the natural values of the grid."""
        total = np.zeros(grid["B"].shape)
        for branch in self.branches:
            a, b = self.assigned(branch, grid["A"], grid["B"], grid["c_a"], grid["c_b"])
            flag = grid["F"] if branch.flag is None else branch.flag
            total = total + branch.probability * self.m(grid["W"], flag, a, b)
        return total

    def psi(self, theta: dict[str, np.ndarray] | None = None) -> float:
        return float(
            sum(
                np.sum(grid["mass"] * self.value(grid))
                for grid in (self._grid(f, theta) for f in FLAGS)
            )
        )

    @property
    def tuned_psi(self) -> float:
        return self.psi()

    # ------------------------------------------------------------ the curve

    def _weight(self, index: int, branch: Branch, curve: Curve) -> float:
        if index in curve.skip:
            return 0.0
        return branch.probability if curve.weighted else 1.0

    def derivative(self, dose: str, curve: Curve = Curve()) -> np.ndarray:
        """The analytic :math:`\\partial\\Phi / \\partial\\theta` of ``dose``'s tuner."""
        w, fw = self.w_grid
        p = self.gradient(dose, {"W": w, "F": w, "A": w}).shape[-1]
        total = np.zeros(p)
        for f in FLAGS:
            pf = self.p_f(w)
            mass_wf = fw * (pf if f else 1.0 - pf)
            h_a = {"W": w, "F": np.full(w.shape, f)}
            c_a = self.cap("A", h_a)
            e_a = self.edge(c_a)
            if dose == "A":
                # B's law and B's cap read the natural A, here at the edge e_A.
                h_b = {**h_a, "A": e_a}
                c_b = self.cap("B", h_b)
                b, wb = _segments(*A_RANGE, np.stack([self.edge(c_b), c_b], axis=-1))
                g_b = self.g("B", b, {k: v[:, None] for k, v in h_b.items()}) * wb
                jump = np.zeros(b.shape)
                for index, branch in enumerate(self.branches):
                    if "A" not in branch.tuned:
                        continue
                    flag = f if branch.flag is None or not curve.assigned_flag else branch.flag
                    _, moved = self.assigned(branch, e_a[:, None], b, c_a[:, None], c_b[:, None])
                    jump = jump + self._weight(index, branch, curve) * (
                        self.m(w[:, None], flag, c_a[:, None], moved)
                        - self.m(w[:, None], flag, e_a[:, None], moved)
                    )
                inner = self.g("A", e_a, h_a) * np.sum(g_b * jump, axis=-1)
                total = total + np.sum((mass_wf * inner)[:, None] * self.gradient("A", h_a), axis=0)
                continue
            a, wa = _segments(*A_RANGE, np.stack([e_a, c_a], axis=-1))
            w2 = np.broadcast_to(w[:, None], a.shape)
            h_b = {"W": w2, "F": np.full(a.shape, f), "A": a}
            c_b = self.cap("B", h_b)
            e_b = self.edge(c_b)
            jump = np.zeros(a.shape)
            for index, branch in enumerate(self.branches):
                if "B" not in branch.tuned:
                    continue
                flag = f if branch.flag is None or not curve.assigned_flag else branch.flag
                moved, _ = self.assigned(branch, a, e_b, c_a[:, None], c_b)
                jump = jump + self._weight(index, branch, curve) * (
                    self.m(w2, flag, moved, c_b) - self.m(w2, flag, moved, e_b)
                )
            inner = (
                mass_wf[:, None]
                * self.g("A", a, {"W": w2, "F": f})
                * wa
                * self.g("B", e_b, h_b)
                * jump
            )
            total = total + np.sum(inner[..., None] * self.gradient("B", h_b), axis=(0, 1))
        return total

    def tuned_doses(self) -> tuple[str, ...]:
        return tuple(dose for dose in DOSES if any(dose in b.tuned for b in self.branches))

    def score_pairing(self, path: str, curve: Curve = Curve()) -> float:
        """:math:`E[D_\\Psi s]` for the score ``s`` of one path, by quadrature.

        The residual :math:`r (Y - m)` pairs to zero with every path here, none of which
        moves the outcome's law, so the pairing reads :math:`m(v^d) - \\Psi` and the tuning
        terms.
        """
        derivatives = {
            dose: self.derivative(dose, curve) for dose in self.tuned_doses() if dose in curve.doses
        }
        psi = self.tuned_psi
        w, fw = self.w_grid
        mean_tanh = float(np.sum(fw * np.tanh(w)))
        total = 0.0
        for f in FLAGS:
            grid = self._grid(f)
            term = self.value(grid) - psi
            if "A" in derivatives:
                term = term + self.influence("A", grid, grid["A"]) @ derivatives["A"]
            if "B" in derivatives:
                term = term + self.influence("B", grid, grid["B"]) @ derivatives["B"]
            if path == "w":
                score = np.tanh(grid["W"]) - mean_tanh
            elif path == "f":
                score = f - self.p_f(grid["W"])
            else:
                dose = path.upper()
                score = np.tanh(grid[dose]) - self.conditional_tanh(dose, grid)
            total += float(np.sum(grid["mass"] * term * score))
        return total

    def tilted(self, path: str, eps: float) -> LawV:
        return LawV(branches=self.branches, tuners=self.tuners, tilt=Tilt(**{path: eps}))


def pathwise_derivative(law: LawV, path: str, step: float = 2e-3) -> float:
    """``d/de Psi(P_e)`` at zero by the four-point central difference."""

    def at(eps: float) -> float:
        return law.tilted(path, eps).tuned_psi

    return (8.0 * (at(step) - at(-step)) - (at(2 * step) - at(-2 * step))) / (12.0 * step)


def sample(n: int, seed: int) -> pd.DataFrame:
    """``n`` draws of the law: ``W, F, A, B, Y``."""
    rng = np.random.default_rng(seed)
    w = stats.truncnorm.rvs(-2.5, 2.5, size=n, random_state=rng)
    f = (rng.uniform(size=n) < expit(0.5 * w - 0.1)).astype(float)
    mean_a = LawV.mean_a(w, f)
    a = stats.truncnorm.rvs(-4.0 - mean_a, 4.0 - mean_a, loc=mean_a, size=n, random_state=rng)
    mean_b = LawV.mean_b(w, f, a)
    b = stats.truncnorm.rvs(-4.0 - mean_b, 4.0 - mean_b, loc=mean_b, size=n, random_state=rng)
    m = LawV.m(w, f, a, b)
    y = rng.beta(30.0 * m, 30.0 * (1.0 - m))
    return pd.DataFrame({"W": w, "F": f, "A": a, "B": b, "Y": y})

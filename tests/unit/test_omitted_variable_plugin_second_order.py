r"""The second-order correction of the plug-in :math:`\nu^2` (X26 plan, section 9).

:class:`~cleverly.sensitivity._second_order.PluginStack` estimates the :math:`O(1/n)` bias of
:math:`\hat\nu^2 = E_n[w\hat\alpha^2]` and of its variance estimate.  The witnesses here are
deterministic.

* W10 compares each closed form of the engine with a central difference of the rows.
* W11 compares the root bias of a saturated logistic model with the Cox and Snell value per
  cell (Firth 1993, Section 2), and the bias of a misspecified model on a random design with an
  exact multinomial enumeration.
* W12 compares :math:`n\hat B` with the Richardson limit of :math:`n(E\hat\nu^2 - \nu^2)` from
  an exact binomial enumeration of a fixed design, in sample and cross-fitted, with weight
  levels, the ATT share, a known mechanism, three arms and a ridge.  It does the same for the
  variance estimate in sample.  A cell at a truncation bound has a bias of order
  :math:`n^{-1/2}`, which no smooth correction covers, and the package refuses it.

The engine reads each fixed design through ``mass`` (the expected count of each row) and
``strata`` (the design point, within which the units are centred).  Each witness evaluates
the engine at the law, so :math:`n\hat B` does not depend on :math:`n`.

:class:`TestTheWitnessHasTeeth` commits the mutations M13 to M20 of the X26 plan.  Each floor
is the smallest deviation measured under that mutation, rounded down.
"""

from __future__ import annotations

import contextlib
import functools
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import cleverly.sensitivity._mechanism_stack as ms
import cleverly.sensitivity._second_order as so
import cleverly.sensitivity.omitted_variable as ov
from cleverly.estimators import TMLE
from cleverly.exceptions import PositivityWarning
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity._second_order import PluginStack, StackFold, representer_derivatives

#: The tolerance of a Richardson limit against :math:`n\hat B`, relative to :math:`n\hat B`.
#: The largest deviation measured was 2.8e-4 (W12e, three arms); the remainder of the
#: extrapolation is of order :math:`n^{-3}`.
LIMIT_RTOL = 1e-3

#: The tolerance of a closed form against a central difference, relative to its largest entry.
CLOSED_RTOL = 1e-6


def expit(z: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: Any) -> Any:
    return np.log(p / (1.0 - p))


# ------------------------------------------------------------------------- fixed designs


@dataclass(frozen=True)
class Design:
    """A fixed design: cells with a known mechanism, rows per cell and weight level, folds.

    Parameters
    ----------
    g : tuple of tuple of float
        ``(cells, K)`` mechanism of each cell.
    group, arm, versus, conditioning : the representer, by column.
    levels : tuple of float
        The weight levels of each cell, ``m`` rows each, normalised to mean one.
    folds : int
        1 for an in-sample fit.
    known : bool
        Whether the mechanism is declared rather than fitted.
    C : float
        The L2 penalty on the slopes of the saturated model.
    """

    g: tuple[tuple[float, ...], ...] = ((0.7, 0.3), (0.4, 0.6))
    group: str = "mean"
    arm: int = 1
    versus: int | None = 0
    conditioning: int | None = None
    levels: tuple[float, ...] = (1.0,)
    folds: int = 1
    known: bool = False
    C: float = float("inf")

    @property
    def cells(self) -> int:
        return len(self.g)

    @property
    def k(self) -> int:
        return len(self.g[0])

    @property
    def weights(self) -> np.ndarray:
        levels = np.asarray(self.levels, dtype=float)
        return levels / levels.mean()

    def design(self) -> np.ndarray:
        """``(cells, cells)`` saturated design: an intercept and one indicator per other cell."""
        return np.column_stack([np.ones(self.cells), np.eye(self.cells)[:, 1:]])

    def theta(self) -> np.ndarray:
        """``(K - 1, cells)`` coefficients that reproduce ``g`` exactly."""
        eta = np.log(np.asarray(self.g)[:, 1:] / np.asarray(self.g)[:, [0]])
        return np.asarray(np.linalg.solve(self.design(), eta).T)

    def penalty(self) -> np.ndarray:
        r = (self.k - 1) * self.cells
        if np.isinf(self.C):
            return np.zeros((r, r))
        slopes = np.eye(self.cells)
        slopes[0, 0] = 0.0
        arm = np.ones((1, 1)) if self.k == 2 else np.eye(self.k - 1) - 1.0 / self.k
        return np.asarray(np.kron(arm, slopes) / self.C)

    def n(self, m: int) -> int:
        return m * self.cells * len(self.levels)

    def stack(self, m: int, *, theta: np.ndarray | None = None) -> PluginStack:
        """The engine at the law: one row per fold, cell, weight level and arm."""
        rows = [
            (f, c, l, a)
            for f in range(self.folds)
            for c in range(self.cells)
            for l in range(len(self.levels))
            for a in range(self.k)
        ]
        f, c, l, a = (np.array(column) for column in zip(*rows, strict=True))
        g = np.asarray(self.g)
        mass = (m / self.folds) * g[c, a]
        strata = (f * self.cells + c) * len(self.levels) + l
        design = self.design()[c]
        coefficients = self.theta() if theta is None else theta
        folds = []
        if not self.known:
            for k in range(self.folds):
                train = np.ones(len(rows), bool) if self.folds == 1 else f != k
                test = np.ones(len(rows), bool) if self.folds == 1 else f == k
                folds.append(StackFold(design, coefficients, self.penalty(), train, test))
        return PluginStack(
            folds,
            a,
            self.k,
            self.weights[l],
            self.group,
            self.arm,
            self.versus,
            self.conditioning,
            known=g[c] if self.known else None,
            mass=mass,
            strata=strata,
        )

    def truth(self) -> float:
        """``nu^2`` at the law."""
        stack = self.stack(1)
        return float(stack.root[stack.layout.nu])


def _alpha2(design: Design, a: np.ndarray, g: np.ndarray, share: Any = None) -> np.ndarray:
    """``alpha^2`` of arm column ``a`` (broadcast) at mechanism ``g`` (``(..., K)``)."""
    onehot = np.eye(design.k)[a]
    shape = np.broadcast_shapes(onehot.shape, g.shape)
    flat_on = np.broadcast_to(onehot, shape).reshape(-1, design.k)
    flat_g = np.broadcast_to(g, shape).reshape(-1, design.k)
    flat_share = None if share is None else np.broadcast_to(share, shape[:-1]).reshape(-1)
    alpha, _, _ = representer_derivatives(
        design.group, design.arm, design.versus, flat_on, flat_g, flat_share
    )
    return np.asarray(alpha**2).reshape(shape[:-1])


def _binomial(m: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Counts ``1..m-1`` and their probabilities: an empty arm leaves ``g_hat`` at 0 or 1."""
    k = np.arange(1, m)
    return k, stats.binom.pmf(k, m, p)


def exact_mean_bias(design: Design, m: int) -> float:
    """``E nu2_hat - nu2`` for a mean or ATE representer of a binary separable design.

    Each cell's estimate depends on its own counts, so the expectation is a sum over cells,
    folds and weight levels.  With two weight levels the cell's counts are enumerated jointly.
    """
    assert design.k == 2 and design.conditioning is None and not design.known
    w = design.weights
    levels = len(w)
    train = m if design.folds == 1 else m * (design.folds - 1) // design.folds
    total = 0.0
    for _, p in design.g:
        grids = [_binomial(train, p) for _ in range(levels)]
        counts = np.meshgrid(*[k for k, _ in grids], indexing="ij")
        prob = functools.reduce(np.multiply, np.meshgrid(*[q for _, q in grids], indexing="ij"))
        treated = sum(w[l] * counts[l] for l in range(levels))
        g1 = treated / (train * w.sum())
        g_hat = np.stack([1.0 - g1, g1], axis=-1)
        a1 = _alpha2(design, np.array(1), g_hat)
        a0 = _alpha2(design, np.array(0), g_hat)
        if design.folds == 1:
            value = sum(w[l] * (counts[l] * a1 + (m - counts[l]) * a0) for l in range(levels))
        else:
            value = w.sum() * m * (p * a1 + (1.0 - p) * a0)
        total += float(np.sum(prob * value) / prob.sum())
    return total / design.n(m) - design.truth()


def exact_three_arm_bias(design: Design, m: int) -> float:
    """``E nu2_hat - nu2`` in sample for three arms: a trinomial enumeration per cell."""
    assert design.k == 3 and design.folds == 1 and len(design.levels) == 1
    total = 0.0
    for g in np.asarray(design.g):
        k0, k1 = np.meshgrid(np.arange(1, m), np.arange(1, m), indexing="ij")
        k2 = m - k0 - k1
        ok = k2 >= 1
        counts = np.stack([k0[ok], k1[ok], k2[ok]], axis=-1)
        prob = stats.multinomial.pmf(counts, m, g)
        g_hat = counts / m
        value = sum(counts[:, a] * _alpha2(design, np.array(a), g_hat) for a in range(3))
        total += float(np.sum(prob * value) / prob.sum())
    return total / design.n(m) - design.truth()


def _joint(design: Design, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The joint counts ``(k_1, k_2)`` of the two cells of an unweighted binary design."""
    assert design.cells == 2 and design.k == 2 and len(design.levels) == 1
    (k1, q1), (k2, q2) = (_binomial(m, p) for _, p in design.g)
    a, b = np.meshgrid(k1, k2, indexing="ij")
    return a, b, np.outer(q1, q2)


def exact_share_bias(design: Design, m: int) -> float:
    """``E nu2_hat - nu2`` in sample for an ATT or ATC: the share couples the two cells."""
    assert design.folds == 1 and design.conditioning is not None
    k1, k2, prob = _joint(design, m)
    n = design.n(m)
    counts = (k1, k2)
    on = sum(k if design.conditioning == 1 else m - k for k in counts)
    share = on / n
    value = 0.0
    for c, k in enumerate(counts):
        g = np.asarray(design.g[c])
        g_hat = (
            np.broadcast_to(g, (*k.shape, 2))
            if design.known
            else np.stack([1.0 - k / m, k / m], axis=-1)
        )
        value = (
            value
            + k * _alpha2(design, np.array(1), g_hat, share)
            + (m - k) * _alpha2(design, np.array(0), g_hat, share)
        )
    return float(np.sum(prob * value / n) / prob.sum()) - design.truth()


def _ridge_roots(design: Design, k1: np.ndarray, k2: np.ndarray, m: int) -> np.ndarray:
    """``(..., 2)`` penalized roots of the two-cell logistic model, by Newton's method.

    The design is ``(1, 0)`` and ``(1, 1)`` and the penalty acts on the slope, so the score and
    the 2-by-2 information are written out.
    """
    slope = design.penalty()[1, 1]
    theta = np.zeros((*k1.shape, 2))
    for _ in range(60):
        g0 = expit(theta[..., 0])
        g1 = expit(theta[..., 0] + theta[..., 1])
        s0 = (k1 - m * g0) + (k2 - m * g1)
        s1 = (k2 - m * g1) - slope * theta[..., 1]
        q0, q1 = m * g0 * (1.0 - g0), m * g1 * (1.0 - g1)
        a, b, d = q0 + q1, q1, q1 + slope
        det = a * d - b * b
        step = np.stack([(d * s0 - b * s1) / det, (a * s1 - b * s0) / det], axis=-1)
        theta = theta + step
        if np.max(np.abs(step)) < 1e-13:
            break
    return theta


def exact_ridge_bias(design: Design, m: int) -> float:
    """``E nu2_hat - nu2(theta*)`` in sample under a fixed L2 penalty on the slope."""
    assert design.folds == 1 and design.conditioning is None
    k1, k2, prob = _joint(design, m)
    theta = _ridge_roots(design, k1, k2, m)
    g1 = np.stack([expit(theta[..., 0]), expit(theta[..., 0] + theta[..., 1])], axis=-1)
    value = 0.0
    for c, k in enumerate((k1, k2)):
        g_hat = np.stack([1.0 - g1[..., c], g1[..., c]], axis=-1)
        value = (
            value
            + k * _alpha2(design, np.array(1), g_hat)
            + (m - k) * _alpha2(design, np.array(0), g_hat)
        )
    return float(np.sum(prob * value / design.n(m)) / prob.sum()) - design.truth()


def exact_variance_bias(design: Design, m: int) -> tuple[float, float]:
    """``E v_hat - v`` in sample for the saturated ATE, and ``v``.

    With a saturated model in sample the plug-in curve is the doubly robust one
    (``TestTheExactLawCurve`` W2): ``psi = 2 (1/g_hat + 1/(1 - g_hat)) - alpha_hat^2 - nu_hat``.
    """
    assert design.group == "mean" and design.versus == 0 and design.folds == 1
    k1, k2, prob = _joint(design, m)
    n = design.n(m)

    def second_moment(counts: tuple[Any, Any], g_hats: list[Any]) -> tuple[Any, Any]:
        nu = (
            sum(
                k * _alpha2(design, np.array(1), g) + (m - k) * _alpha2(design, np.array(0), g)
                for k, g in zip(counts, g_hats, strict=True)
            )
            / n
        )
        total = 0.0
        for k, g in zip(counts, g_hats, strict=True):
            centre = 2.0 * (1.0 / g[..., 1] + 1.0 / g[..., 0])
            for a, rows in ((1, k), (0, m - k)):
                total = total + rows * (centre - _alpha2(design, np.array(a), g) - nu) ** 2
        return total / n, nu

    g_hats = [np.stack([1.0 - k / m, k / m], axis=-1) for k in (k1, k2)]
    v_hat, _ = second_moment((k1, k2), g_hats)
    law = [np.asarray(row) for row in design.g]
    expected = (np.array(m * law[0][1]), np.array(m * law[1][1]))
    v, _ = second_moment(expected, law)
    return float(np.sum(prob * v_hat) / prob.sum()) - float(v), float(v)


def richardson(bias: Callable[[Design, int], float], design: Design, m: int) -> float:
    """The limit of ``n e(n)`` from ``n``, ``2n`` and ``4n``, its two leading remainders removed.

    ``n e(n) = L + a / n + b / n^2 + O(n^{-3})`` for a smooth function of binomial
    proportions, so two Richardson steps leave ``O(n^{-3})``.
    """
    values = [design.n(r) * bias(design, r) for r in (m, 2 * m, 4 * m)]
    first = [2.0 * values[1] - values[0], 2.0 * values[2] - values[1]]
    return (4.0 * first[1] - first[0]) / 3.0


#: The rows per cell at which :func:`scaled_point` evaluates the engine.  Under a fixed penalty
#: ``n B`` moves with ``n`` through ``(J + P / C)^{-1}``, so the witness reads its limit.
LAW_ROWS = 10**7


def scaled_point(design: Design, m: int = LAW_ROWS) -> float:
    """``n B`` at the law: the engine's point bias, scaled by the rows it stands for."""
    return design.n(m) * design.stack(m).bias(variance=False).point


def exact_random_design(n: int) -> tuple[float, float]:
    """``E theta_hat - theta*`` and ``E nu2_hat - nu2`` on a random two-point design.

    The rows are independent draws of ``x`` in ``{1, 2}`` with probability one half each and
    of ``A`` with ``P(A = 1 | x) = 0.3, 0.6``.  The model ``expit(theta x)`` has no intercept, so
    it is misspecified, and its score derivative ``-x^2 g (1 - g)`` varies with the row: the
    term ``E[grad s Q s]`` of the bias is not zero.  The counts of the four cells are
    multinomial, and each draw's root is found by Newton's method.
    """
    probs = np.array(
        [0.5 * RANDOM_G[0], 0.5 * (1 - RANDOM_G[0]), 0.5 * RANDOM_G[1], 0.5 * (1 - RANDOM_G[1])]
    )
    a, b, c = np.meshgrid(np.arange(n + 1), np.arange(n + 1), np.arange(n + 1), indexing="ij")
    d = n - a - b - c
    ok = d >= 0
    counts = np.stack([a[ok], b[ok], c[ok], d[ok]], axis=-1).astype(float)
    # Every draw needs both arms, or the root is infinite.
    treated = counts[:, 0] + counts[:, 2]
    keep = (treated > 0) & (treated < n)
    counts = counts[keep]
    prob = stats.multinomial.pmf(counts, n, probs)
    theta = np.zeros(len(counts))
    for _ in range(80):
        g1, g2 = expit(theta), expit(2.0 * theta)
        score = (counts[:, 0] - (counts[:, 0] + counts[:, 1]) * g1) + 2.0 * (
            counts[:, 2] - (counts[:, 2] + counts[:, 3]) * g2
        )
        information = (counts[:, 0] + counts[:, 1]) * g1 * (1 - g1) + 4.0 * (
            counts[:, 2] + counts[:, 3]
        ) * g2 * (1 - g2)
        step = score / information
        theta = theta + step
        if np.max(np.abs(step)) < 1e-13:
            break
    g1, g2 = expit(theta), expit(2.0 * theta)
    nu = (
        counts[:, 0] / g1**2
        + counts[:, 1] / (1 - g1) ** 2
        + counts[:, 2] / g2**2
        + counts[:, 3] / (1 - g2) ** 2
    ) / n
    total = prob.sum()
    stack = random_design_stack(1)
    star, nu_star = stack.root[0], stack.root[stack.layout.nu]
    return float(prob @ theta / total - star), float(prob @ nu / total - nu_star)


#: The mechanism of the random two-point design of :func:`exact_random_design`.
RANDOM_G = (0.3, 0.6)


def _random_theta() -> float:
    """The root of the population score of ``expit(theta x)``."""
    theta = 0.0
    for _ in range(100):
        g1, g2 = expit(theta), expit(2.0 * theta)
        score = 0.5 * (RANDOM_G[0] - g1) + 0.5 * 2.0 * (RANDOM_G[1] - g2)
        theta += score / (0.5 * g1 * (1 - g1) + 2.0 * g2 * (1 - g2))
    return theta


def random_design_stack(n: float) -> PluginStack:
    """The engine at the law of the random design: four independent cells, no strata."""
    x = np.array([1.0, 1.0, 2.0, 2.0])
    a = np.array([1, 0, 1, 0])
    g = np.array([RANDOM_G[0], RANDOM_G[0], RANDOM_G[1], RANDOM_G[1]])
    mass = n * 0.5 * np.where(a == 1, g, 1.0 - g)
    everything = np.ones(4, bool)
    fold = StackFold(
        x[:, None], np.array([[_random_theta()]]), np.zeros((1, 1)), everything, everything
    )
    return PluginStack([fold], a, 2, np.ones(4), "mean", 1, 0, None, mass=mass)


# ---------------------------------------------------------------------------- W10 closed forms


def _random_stack(
    k: int, group: str, folds: int, *, cluster: bool = False, seed: int = 7
) -> PluginStack:
    """A small random sample with weights, a ridge and ``folds`` folds, at random coefficients.

    The coefficients are not a root, which the closed forms do not need.
    """
    rng = np.random.default_rng(seed)
    n = 36
    design = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    theta = 0.3 * rng.normal(size=(k - 1, 3))
    labels = np.arange(n) % folds
    stack_folds = [
        StackFold(
            design,
            theta + 0.05 * f,
            0.2 * np.eye((k - 1) * 3),
            np.ones(n, bool) if folds == 1 else labels != f,
            np.ones(n, bool) if folds == 1 else labels == f,
        )
        for f in range(folds)
    ]
    arms = rng.integers(0, k, size=n)
    arms[:k] = np.arange(k)
    versus = None if group == "ey" else 0
    conditioning = {"att": k - 1, "atc": 0}.get(group)
    return PluginStack(
        stack_folds,
        arms,
        k,
        rng.uniform(0.5, 2.0, size=n),
        "mean" if group in ("ey", "ate") else group,
        k - 1,
        versus,
        conditioning,
        cluster=np.arange(n) // 3 if cluster else None,
    )


W10_CASES = [
    (k, group, folds) for k in (2, 3) for group in ("ey", "ate", "att", "atc") for folds in (1, 3)
]


def _w10_id(case: tuple[int, str, int]) -> str:
    return f"K{case[0]}-{case[1]}-folds{case[2]}"


class TestTheClosedForms:
    """W10: each closed form of the engine against a central difference of the rows."""

    STEP = 1e-6

    def point(self, stack: PluginStack) -> np.ndarray:
        rng = np.random.default_rng(1)
        eta = stack.root + 0.01 * rng.normal(size=stack.root.size)
        return np.asarray(eta)

    def central(self, function: Callable[[np.ndarray], Any], eta: np.ndarray, a: Any) -> Any:
        return (function(eta + self.STEP * a) - function(eta - self.STEP * a)) / (2 * self.STEP)

    @pytest.mark.parametrize("case", W10_CASES, ids=_w10_id)
    def test_the_jacobian_vector_product(self, case: tuple[int, str, int]) -> None:
        stack = _random_stack(*case)
        eta = self.point(stack)
        a = np.random.default_rng(2).normal(size=eta.size)
        numeric = self.central(stack.rows, eta, a)
        np.testing.assert_allclose(
            stack.jvp(eta, a), numeric, rtol=0, atol=CLOSED_RTOL * np.abs(numeric).max()
        )
        per_row = np.random.default_rng(3).normal(size=(stack.n, eta.size))
        rowwise = stack.jvp(eta, per_row)
        for i in (0, 7, stack.n - 1):
            np.testing.assert_allclose(rowwise[i], stack.jvp(eta, per_row[i])[i], rtol=1e-12)

    @pytest.mark.parametrize("case", W10_CASES, ids=_w10_id)
    def test_the_second_directional_derivative(self, case: tuple[int, str, int]) -> None:
        """``G``, ``e``, ``grad^2 g`` and ``dJ/dtheta`` enter here, through the rows."""
        stack = _random_stack(*case)
        eta = self.point(stack)
        a = np.random.default_rng(4).normal(size=eta.size)
        numeric = self.central(lambda e: stack.jvp(e, a), eta, a)
        np.testing.assert_allclose(
            stack.second_directional(eta, a),
            numeric,
            rtol=0,
            atol=CLOSED_RTOL * np.abs(numeric).max(),
        )

    @pytest.mark.parametrize("case", W10_CASES, ids=_w10_id)
    def test_the_vector_jacobian_product(self, case: tuple[int, str, int]) -> None:
        stack = _random_stack(*case)
        eta = self.point(stack)
        gamma = np.random.default_rng(5).normal(size=eta.size)
        columns = [self.central(stack.rows, eta, unit) for unit in np.eye(eta.size)]
        numeric = np.einsum("jit,t->ij", np.array(columns), gamma)
        np.testing.assert_allclose(
            stack.vjp(eta, gamma), numeric, rtol=0, atol=CLOSED_RTOL * np.abs(numeric).max()
        )

    @pytest.mark.parametrize("case", W10_CASES, ids=_w10_id)
    def test_the_rows_are_complex_analytic(self, case: tuple[int, str, int]) -> None:
        """The augmented blocks are differentiated by a complex step of the closed forms."""
        stack = _random_stack(*case)
        eta = self.point(stack)
        a = np.random.default_rng(6).normal(size=eta.size)
        step = 1e-30
        np.testing.assert_allclose(
            stack.rows(eta + 1j * step * a).imag / step, stack.jvp(eta, a), rtol=1e-12, atol=1e-12
        )
        gamma = np.random.default_rng(8).normal(size=eta.size)
        numeric = stack.vjp(eta + 1j * step * a, gamma.astype(complex)).imag / step
        expected = self.central(lambda e: stack.vjp(e, gamma), eta, a)
        np.testing.assert_allclose(
            numeric, expected, rtol=0, atol=CLOSED_RTOL * np.abs(expected).max()
        )

    @pytest.mark.parametrize("case", W10_CASES, ids=_w10_id)
    def test_the_information_is_the_summed_jacobian(self, case: tuple[int, str, int]) -> None:
        """``A`` is the derivative of the summed rows, with ``-P/C`` on each fold block."""
        stack = _random_stack(*case)
        eta = self.point(stack)
        columns = [
            self.central(lambda e: stack.rows(e).sum(axis=0) + stack.penalty(e), eta, unit)
            for unit in np.eye(eta.size)
        ]
        numeric = np.column_stack(columns)
        np.testing.assert_allclose(
            stack.information(eta), numeric, rtol=0, atol=CLOSED_RTOL * np.abs(numeric).max()
        )

    @pytest.mark.parametrize("group", ["ey", "ate", "att", "atc"])
    def test_the_representer_derivatives(self, group: str) -> None:
        """``b`` and ``e`` against central differences in each column of a three-arm ``g``."""
        rng = np.random.default_rng(9)
        g = rng.dirichlet(np.ones(3), size=20)
        onehot = np.eye(3)[rng.integers(0, 3, size=20)]
        versus = None if group == "ey" else 0
        name = "mean" if group in ("ey", "ate") else group
        share = None if name == "mean" else 0.4

        def call(values: np.ndarray) -> tuple[Any, Any, Any]:
            return representer_derivatives(name, 2, versus, onehot, values, share)

        _, first, second = call(g)
        for c in range(3):
            unit = np.zeros(3)
            unit[c] = self.STEP
            plus, minus = call(g + unit), call(g - unit)
            np.testing.assert_allclose(
                first[:, c], (plus[0] - minus[0]) / (2 * self.STEP), rtol=1e-7, atol=1e-9
            )
            np.testing.assert_allclose(
                second[:, :, c], (plus[1] - minus[1]) / (2 * self.STEP), rtol=1e-7, atol=1e-9
            )


# ---------------------------------------------------------------------- W11 and W12


#: The saturated in-sample ATE, the reference design of every W12 witness.
BASE = Design()

W12: dict[str, tuple[Design, Callable[[Design, int], float], int]] = {
    "W12a-in-sample": (BASE, exact_mean_bias, 100),
    "W12a-cross-fitted": (Design(folds=5), exact_mean_bias, 200),
    "W12b-weighted-in-sample": (Design(levels=(1.0, 3.0)), exact_mean_bias, 100),
    "W12b-weighted-cross-fitted": (Design(levels=(1.0, 3.0), folds=5), exact_mean_bias, 100),
    "W12c-att-share": (Design(group="att", conditioning=1), exact_share_bias, 100),
    "W12c-atc-share": (Design(group="atc", conditioning=0), exact_share_bias, 100),
    "W12d-known-att": (Design(group="att", conditioning=1, known=True), exact_share_bias, 100),
    "W12e-three-arms": (
        Design(g=((0.5, 0.3, 0.2), (0.3, 0.3, 0.4)), arm=2),
        exact_three_arm_bias,
        100,
    ),
    "W12f-ridge": (Design(C=0.5), exact_ridge_bias, 100),
}


@functools.cache
def exact_limit(name: str) -> float:
    design, bias, m = W12[name]
    return richardson(bias, design, m)


def w12_deviation(name: str) -> float:
    """``|n B - lim n e(n)|`` relative to the limit."""
    design, _, _ = W12[name]
    limit = exact_limit(name)
    return abs(scaled_point(design) - limit) / abs(limit)


@functools.cache
def exact_variance_limit() -> tuple[float, float]:
    limit = richardson(lambda design, m: exact_variance_bias(design, m)[0], BASE, 200)
    return limit, exact_variance_bias(BASE, 200)[1]


def variance_deviation() -> float:
    limit, _ = exact_variance_limit()
    m = LAW_ROWS
    return abs(BASE.n(m) * BASE.stack(m).bias().variance - limit) / abs(limit)


def cox_snell_deviation() -> float:
    """The root bias of the saturated model against ``(2g - 1) / (2 m g (1 - g))`` per cell."""
    m = 400
    root = BASE.stack(m).bias(variance=False).root
    g = np.array([row[1] for row in BASE.g])
    cell = (2.0 * g - 1.0) / (2.0 * m * g * (1.0 - g))
    expected = np.array([cell[0], cell[1] - cell[0]])
    return float(np.max(np.abs(root[:2] - expected) / np.abs(expected)))


@functools.cache
def random_design_limits() -> tuple[float, float]:
    """The Richardson limits of ``n (E theta_hat - theta*)`` and ``n (E nu2_hat - nu2)``."""
    sizes = (40, 80, 160)
    values = np.array([[n * e for e in exact_random_design(n)] for n in sizes])
    first = 2.0 * values[1:] - values[:-1]
    return tuple(float(v) for v in (4.0 * first[1] - first[0]) / 3.0)  # type: ignore[return-value]


def random_design_deviation() -> float:
    n = 10**7
    root = random_design_stack(n).bias(variance=False).root
    predicted = (n * root[0], n * root[-1])
    return float(
        max(abs(p - e) / abs(e) for p, e in zip(predicted, random_design_limits(), strict=True))
    )


class TestTheCoxSnellBias:
    """W11: the general Rilstone-Srivastava-Ullah path reduces to Cox and Snell."""

    def test_the_saturated_root_bias_is_the_cox_snell_bias(self) -> None:
        assert cox_snell_deviation() < 1e-10

    def test_a_misspecified_model_on_a_random_design(self) -> None:
        """The ``E[grad s Q s]`` term, and the bias of ``nu^2``, on independent rows."""
        assert random_design_deviation() < RANDOM_RTOL


#: The tolerance of the random-design witness: three sizes up to n = 160 leave a remainder of
#: order n^-3 times a large constant.  The largest deviation measured was 3.0e-3.
RANDOM_RTOL = 1e-2


class TestTheExactLawBias:
    """W12: ``n B`` is the limit of ``n (E nu2_hat - nu2)`` on exact fixed-design laws."""

    @pytest.mark.parametrize("name", list(W12))
    def test_the_correction_is_the_exact_bias(self, name: str) -> None:
        assert w12_deviation(name) < LIMIT_RTOL

    def test_the_variance_correction_is_the_exact_bias(self) -> None:
        """The augmented stack, with the noise of gamma, against an exact enumeration."""
        _, v = exact_variance_limit()
        stack = BASE.stack(200)
        assert stack.bias().second_moment == pytest.approx(v, rel=1e-12)
        assert variance_deviation() < LIMIT_RTOL

    def test_every_in_sample_term_is_of_order_one(self) -> None:
        """``D beta``, ``tr(G V) / 2`` and ``c`` each move ``n B`` by more than 4."""
        parts = _parts(BASE)
        assert min(abs(v) for v in parts.values()) > 4.0, parts

    def test_the_known_ate_has_no_correction(self) -> None:
        design = Design(known=True)
        assert abs(scaled_point(design)) < 1e-6
        assert abs(exact_share_bias(Design(group="att", conditioning=1, known=True), 100)) > 0.0


def _parts(design: Design) -> dict[str, float]:
    """``n`` times each in-sample term of the bias of ``nu^2``, read through the assembly."""
    captured: dict[str, float] = {}
    original = so._assemble

    def spy(q: Any, first: Any, curvature: Any, penalty: Any, layout: Any) -> Any:
        if q.shape[0] == layout.size and not captured:
            bracket_theta = np.zeros_like(first)
            bracket_theta[layout.thetas] = (first + 0.5 * curvature + penalty)[layout.thetas]
            captured["D beta"] = float((q @ bracket_theta)[layout.nu])
            captured["tr(GV)/2"] = float(q[layout.nu, layout.nu] * 0.5 * curvature[layout.nu])
            captured["c"] = float(q[layout.nu, layout.nu] * first[layout.nu])
        return original(q, first, curvature, penalty, layout)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(so, "_assemble", spy)
        m = LAW_ROWS
        design.stack(m).bias(variance=False)
    return {key: design.n(m) * value for key, value in captured.items()}


# ----------------------------------------------------------------------- a bound at a cell edge


def exact_clipped_bias(m: int, lower: float) -> float:
    """``E nu2_hat - nu2`` in sample for :data:`BASE` with ``g_hat`` clipped below at ``lower``."""
    k1, k2, prob = _joint(BASE, m)
    value = 0.0
    for k in (k1, k2):
        g1 = np.maximum(k / m, lower)
        g_hat = np.stack([1.0 - g1, g1], axis=-1)
        value = (
            value
            + k * _alpha2(BASE, np.array(1), g_hat)
            + (m - k) * _alpha2(BASE, np.array(0), g_hat)
        )
    return float(np.sum(prob * value / BASE.n(m)) / prob.sum()) - BASE.truth()


def at_bound_excess(m: int) -> float:
    r""":math:`\sqrt n\,|E\hat\nu^2 - \nu^2 - \hat B|` when a bound sits at the true cell."""
    lower = BASE.g[0][1]
    smooth = scaled_point(BASE) / BASE.n(m)
    return float(np.sqrt(BASE.n(m)) * abs(exact_clipped_bias(m, lower) - smooth))


class TestABoundAtACellHasNoSmoothCorrection:
    """The truncation kink adds a bias of order :math:`n^{-1/2}`, so the package refuses it."""

    def test_the_excess_bias_is_of_order_root_n(self) -> None:
        small, large = at_bound_excess(400), at_bound_excess(1600)
        assert min(small, large) > AT_BOUND_FLOOR
        assert large / small == pytest.approx(1.0, abs=0.1)


#: The floor of :func:`at_bound_excess`.  The values measured were 2.50 at m = 400 and 2.42 at
#: m = 1600.
AT_BOUND_FLOOR = 2.0


# ---------------------------------------------------------------------------- package wiring


def wiring_frame(n: int, seed: int) -> pd.DataFrame:
    """A two-arm frame with a logistic mechanism in two covariates and a linear outcome."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, 2))
    arm = (rng.random(n) < expit(0.2 + 0.4 * w[:, 0] - 0.3 * w[:, 1])).astype(float)
    outcome = arm + w[:, 0] + 0.5 * w[:, 1] + rng.normal(size=n)
    return pd.DataFrame({"Y": outcome, "A": arm, "W1": w[:, 0], "W2": w[:, 1]})


def cell_frame(m: int, g1: tuple[float, float]) -> pd.DataFrame:
    """Two covariate cells of ``m`` rows with treated shares ``g1`` exactly."""
    x = np.repeat([0.0, 1.0], m)
    arm = np.concatenate([np.arange(m) < round(m * p) for p in g1]).astype(float)
    outcome = arm + x + np.sin(np.arange(2 * m))
    return pd.DataFrame({"Y": outcome, "A": arm, "X": x})


@functools.cache
def wiring_fit(cross_fit: bool, *, seed: int = 3, n: int = 1000) -> Any:
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(C=np.inf, max_iter=2000),
            cross_fit=cross_fit,
            n_folds=5,
            q_bounds=(-20.0, 20.0),
            g_bounds=(0.01, 0.99),
            estimands=("ate", "att"),
            simultaneous=False,
            random_state=0,
        )
        .fit(wiring_frame(n, seed), outcome="Y", treatment="A")
        .single()
    )


def cell_fit(m: int, g1: tuple[float, float], bounds: tuple[float, float]) -> Any:
    """An in-sample fit of :func:`cell_frame`; a bound that clips a cell is deliberate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        return _cell_fit(m, g1, bounds)


def _cell_fit(m: int, g1: tuple[float, float], bounds: tuple[float, float]) -> Any:
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(C=np.inf, max_iter=2000),
            cross_fit=False,
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
            g_bounds=bounds,
        )
        .fit(cell_frame(m, g1), outcome="Y", treatment="A")
        .single()
    )


def elements_and_engine(result: Any, estimand: str) -> tuple[Any, PluginStack | None]:
    """The plug-in elements of ``estimand`` and the stack the package built for them."""
    engines: list[PluginStack] = []
    original = ms._engine

    def spy(*arguments: Any, **keywords: Any) -> PluginStack:
        engine = original(*arguments, **keywords)
        engines.append(engine)
        return engine

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ms, "_engine", spy)
        elements = ov.sensitivity_elements(result, estimand, nu2_estimator="plugin")
    return elements, (engines[-1] if engines else None)


WIRING_CASES = [(cf, estimand) for cf in (False, True) for estimand in ("ate", "att")]


class TestThePackageReadsTheStack:
    """The bound's point, curve and variance are the stack's, on fitted results."""

    @pytest.mark.parametrize("case", WIRING_CASES, ids=lambda c: f"{c[1]}-cf{int(c[0])}")
    def test_the_stack_curve_is_the_shipped_curve(self, case: tuple[bool, str]) -> None:
        elements, engine = elements_and_engine(wiring_fit(case[0]), case[1])
        assert engine is not None and elements.limit_refusal is None
        np.testing.assert_allclose(
            engine.bias(variance=False).curve, elements.psi_nu2, rtol=1e-8, atol=1e-10
        )

    @pytest.mark.parametrize("case", WIRING_CASES, ids=lambda c: f"{c[1]}-cf{int(c[0])}")
    def test_the_point_is_the_moment_less_the_correction(self, case: tuple[bool, str]) -> None:
        elements, engine = elements_and_engine(wiring_fit(case[0]), case[1])
        assert engine is not None
        bias = engine.bias()
        assert elements.nu2_correction == pytest.approx(bias.point, rel=1e-12)
        assert elements.nu2 + elements.nu2_correction == pytest.approx(
            engine.root[engine.layout.nu], rel=1e-12
        )
        assert elements.nu2_variance_correction == pytest.approx(
            bias.variance / engine.n, rel=1e-12
        )
        corrected = influence_variance(elements.psi_nu2, None) - elements.nu2_variance_correction
        assert corrected > 0.0

    def test_an_unadmitted_learner_reports_no_correction(self) -> None:
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                cross_fit=False,
                estimands=("ate",),
                simultaneous=False,
                random_state=0,
                treatment_learner=make_pipeline(PolynomialFeatures(2), LogisticRegression()),
            )
            .fit(wiring_frame(300, 4), outcome="Y", treatment="A")
            .single()
        )
        elements, engine = elements_and_engine(result, "ate")
        assert engine is None and elements.nu2_correction is None
        bound = ov.omitted_variable_bounds(
            result, "ate", nu2_estimator="plugin", cf_y=0.3, cf_d=0.3
        )
        assert bound.limit_refusal is not None and bound.limit_refusal.roadmap == "F26"
        assert "plug-in nu^2, uncorrected" in bound.summary()
        assert bound.to_dict()["nu2_correction"] is None

    def test_the_doubly_robust_bound_carries_no_correction(self) -> None:
        bound = ov.omitted_variable_bounds(
            wiring_fit(False), "ate", nu2_estimator="doubly_robust", cf_y=0.3, cf_d=0.3
        )
        assert "nu2_correction" not in bound.to_dict()


# ------------------------------------------------------------------------------- the refusals


class TestTheCorrectionRefuses:
    """``corrected_nonpositive``, ``correction_unstable`` and ``truncation_active``."""

    def test_corrected_nonpositive(self) -> None:
        assert ms.correction_refusal(1.0, 1.5, 0.01).key == "corrected_nonpositive"  # type: ignore[union-attr]
        assert ms.correction_refusal(1.0, 0.1, -1e-6).key == "corrected_nonpositive"  # type: ignore[union-attr]

    def test_correction_unstable_at_its_threshold(self) -> None:
        """``B^2 / nu^2 = 0.01`` against ``0.25 SE``: refused at SE 0.0399, kept at 0.0401."""
        assert ms.correction_refusal(1.0, 0.1, 0.0399**2).key == "correction_unstable"  # type: ignore[union-attr]
        assert ms.correction_refusal(1.0, 0.1, 0.0401**2) is None

    def test_a_fit_meets_the_stability_rule(self) -> None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ms, "CORRECTION_STABILITY", 1e-9)
            elements, _ = elements_and_engine(wiring_fit(False), "ate")
        assert elements.limit_refusal.key == "correction_unstable"
        assert elements.psi_nu2 is not None and elements.psi_max_bias is None

    def test_a_clipped_cell_refuses(self) -> None:
        """``g = 0.2`` clipped at 0.3: the logit lies 7.8 SE inside the bound, so only the
        clipping refuses it."""
        elements, _ = elements_and_engine(cell_fit(400, (0.2, 0.5), (0.3, 0.7)), "ate")
        assert elements.limit_refusal.key == "truncation_active"
        assert "move the mechanism of 400 rows" in elements.limit_refusal.text

    def test_a_logit_near_a_bound_refuses(self) -> None:
        """``g = 0.2`` at 400 rows has logit SE 0.125; the bound 0.15 lies 2.8 SE away."""
        elements, _ = elements_and_engine(cell_fit(400, (0.2, 0.5), (0.15, 0.85)), "ate")
        assert elements.limit_refusal.key == "truncation_active"
        assert "standard errors from a truncation bound" in elements.limit_refusal.text

    def test_a_logit_far_from_the_bounds_is_corrected(self) -> None:
        """The bound 0.1 lies 6.5 SE from the logit of 0.2."""
        elements, _ = elements_and_engine(cell_fit(400, (0.2, 0.5), (0.1, 0.9)), "ate")
        assert elements.limit_refusal is None and elements.nu2_correction is not None


# ------------------------------------------------------------------------------- the mutations


def _drop_theta_bracket(q: Any, first: Any, curvature: Any, penalty: Any, layout: Any) -> Any:
    """M13: drop ``D beta``, the root's own bias, from the bracket."""
    bracket = first + 0.5 * curvature + penalty
    bracket[layout.thetas] = 0.0
    return q @ bracket


def _drop_trace(q: Any, first: Any, curvature: Any, penalty: Any, layout: Any) -> Any:
    """M14: drop ``tr(G V) / 2``, the curvature of the second-moment row."""
    curvature = curvature.copy()
    curvature[layout.nu] = 0.0
    return q @ (first + 0.5 * curvature + penalty)


def _drop_covariance(q: Any, first: Any, curvature: Any, penalty: Any, layout: Any) -> Any:
    """M15: drop ``c``, the covariance of the second-moment row with the score."""
    first = first.copy()
    first[layout.nu] = 0.0
    return q @ (first + 0.5 * curvature + penalty)


def _model_based(q: Any, meat: Any, layout: Any) -> Any:
    """M16: the model-based ``V_theta theta = Q_theta theta``, right only without weights."""
    v = q @ meat @ q.T
    v[layout.thetas, layout.thetas] = q[layout.thetas, layout.thetas]
    return v


def _flip_root_curvature(q: Any, first: Any, curvature: Any, penalty: Any, layout: Any) -> Any:
    """M17: flip the sign of the curvature of the score rows, the ``B1`` term."""
    curvature = curvature.copy()
    curvature[layout.thetas] *= -1.0
    return q @ (first + 0.5 * curvature + penalty)


MUTATIONS: dict[str, tuple[str, Any]] = {
    "M13-drop-root-bias": ("_assemble", _drop_theta_bracket),
    "M14-drop-trace": ("_assemble", _drop_trace),
    "M15-drop-covariance": ("_assemble", _drop_covariance),
    "M16-model-based-variance": ("_sandwich", _model_based),
    "M17-flip-root-curvature": ("_assemble", _flip_root_curvature),
    "M18-score-on-every-row": ("_training_mask", lambda fold: np.ones_like(fold.train)),
    "M19-fixed-share": ("_stacks_share", lambda conditional: False),
}

#: The witness each mutation fails, and the floor of its deviation (the smallest measured,
#: rounded down).
TEETH: dict[str, tuple[str, float]] = {
    "M13-drop-root-bias": ("W12a-in-sample", 0.3),  # measured 0.326
    "M14-drop-trace": ("W12a-in-sample", 2.0),  # measured 2.674
    "M15-drop-covariance": ("W12a-in-sample", 1.9),  # measured 2.000
    "M16-model-based-variance": ("W12b-weighted-in-sample", 0.5),  # measured 0.541
    "M17-flip-root-curvature": ("W12a-in-sample", 0.6),  # measured 0.652
    "M18-score-on-every-row": ("W12a-cross-fitted", 0.7),  # measured 0.733
    "M19-fixed-share": ("W12c-att-share", 0.3),  # measured 0.326
}


@contextlib.contextmanager
def mutated(name: str) -> Iterator[None]:
    hook, replacement = MUTATIONS[name]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(so, hook, replacement)
        yield


def _rewrapped(function: Any) -> Any:
    return lambda *arguments: function(*arguments)


class TestTheWitnessHasTeeth:
    """Each mutation of the engine fails the witness that names it; a rewrap fails none."""

    @pytest.mark.parametrize("name", list(MUTATIONS))
    def test_the_mutation_fails_its_witness(self, name: str) -> None:
        witness, floor = TEETH[name]
        with mutated(name):
            deviation = w12_deviation(witness)
        assert deviation > floor > LIMIT_RTOL

    def test_the_flipped_curvature_fails_cox_snell(self) -> None:
        with mutated("M17-flip-root-curvature"):
            assert cox_snell_deviation() > 1.0

    def test_the_model_based_variance_holds_without_weights(self) -> None:
        """M16's control: the information equality holds on the unweighted in-sample law."""
        with mutated("M16-model-based-variance"):
            assert w12_deviation("W12a-in-sample") < LIMIT_RTOL

    def test_m0_a_rewrap_of_every_hook_passes(self) -> None:
        with pytest.MonkeyPatch.context() as patch:
            for hook in ("_assemble", "_sandwich", "_training_mask", "_stacks_share"):
                patch.setattr(so, hook, _rewrapped(getattr(so, hook)))
            for name in ("W12a-in-sample", "W12a-cross-fitted", "W12c-att-share"):
                assert w12_deviation(name) < LIMIT_RTOL

    def test_m20_a_clipped_cell_read_as_unclipped_passes_the_rule(self) -> None:
        """M20: with the clipping ignored, the 7.8 SE margin lets the clipped fit through."""
        original = ms.truncation_refusal

        def unclipped(clipped: Any, *rest: Any) -> Any:
            return original(np.zeros_like(clipped), *rest)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ms, "truncation_refusal", unclipped)
            elements, _ = elements_and_engine(cell_fit(400, (0.2, 0.5), (0.3, 0.7)), "ate")
        assert elements.limit_refusal is None

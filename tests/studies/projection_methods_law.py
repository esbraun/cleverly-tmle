r"""Law P of the ``projection-methods`` study and its exact readings (P1 PR 3 plan, 5.1).

Law P is law L of :mod:`tests.studies.drtmle_one_wrong` with effect modification by ``W1`` on
the logit scale:

- ``W1, W2 ~ N(0, 1)`` and ``g_0(1 | W) = expit(0.6 W1 + 0.3 W2)``, law L's mechanism;
- ``Qbar_0(a, W) = expit(-0.2 + a (0.5 + 0.6 W1) - W1 + 2 W2)``;
- complete binary outcomes, ``V = W1`` and the basis ``Linear(["W1"])``.

The wrong outcome regression is law L's own wrong learner, the logistic GLM on ``(A, W1)``,
which omits ``W2`` and the modification.  The slow mechanism of the ``dr_slow`` ladder is the
correct GLM plus the logit offset ``n^(-1/3) W2 (1 + W1)``, a constructed error that breaks the
level sets of ``g_0``.

Every number of plan 5.1 is computed here by ``120 x 120`` Gauss--Hermite quadrature on the
``g_0`` index ``u = (0.6 W1 + 0.3 W2) / sqrt(0.45)`` and its orthogonal coordinate ``v``.  The two
are independent standard normals, so a mean given ``u`` (the population reduced regression on
the mechanism) is a sum over ``v`` alone.  The influence-curve second moments integrate ``A`` and
``Y`` out exactly: four ``(a, y)`` cells, each with mass ``g_0(a | W) P(Y = y | a, W)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from tests.studies import drtmle_one_wrong as law_l

#: The basis terms of every fit.
TERMS = ("(intercept)", "W1")
#: Quadrature nodes per axis (plan 5.1).
NODES = 120
#: ``logit Qbar_0(a, W) = -0.2 + a (0.5 + 0.6 W1) - W1 + 2 W2``.
OUTCOME = {"intercept": -0.2, "a": 0.5, "a_w1": 0.6, "w1": -1.0, "w2": 2.0}
#: The ladder of the slow mechanism (plan 5.2).
LADDER = (1_000, 4_000, 16_000)
#: The size of every 4k group, at which the SDs are stated.
N = 4_000


def g0(w1: Any, w2: Any) -> Any:
    """``P(A = 1 | W)``, law L's mechanism."""
    return law_l.g0(w1, w2)


def qbar0(a: Any, w1: Any, w2: Any) -> Any:
    """``E[Y | A = a, W]`` of law P."""
    a = np.asarray(a, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    return expit(
        OUTCOME["intercept"]
        + a * (OUTCOME["a"] + OUTCOME["a_w1"] * w1)
        + OUTCOME["w1"] * w1
        + OUTCOME["w2"] * np.asarray(w2, dtype=float)
    )


def slow_offset(w1: Any, w2: Any, n: int) -> Any:
    """The slow mechanism's logit offset ``n^(-1/3) W2 (1 + W1)``."""
    return law_l.slow(n) * np.asarray(w2, dtype=float) * (1.0 + np.asarray(w1, dtype=float))


def sample(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n`` rows of law P, with the declared mechanism in ``p0`` and ``p1``."""
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    probability = g0(w1, w2)
    a = rng.binomial(1, probability).astype(float)
    y = rng.binomial(1, qbar0(a, w1, w2)).astype(float)
    return pd.DataFrame(
        {"Y": y, "A": a, "W1": w1, "W2": w2, "p0": 1.0 - probability, "p1": probability}
    )


# ------------------------------------------------------------------------------ quadrature


class Grid:
    """The quadrature grid on ``(u, v)``, its weights and the covariates at its nodes."""

    def __init__(self, nodes: int = NODES) -> None:
        x, w = law_l.gauss_hermite(nodes)
        self.node_weights = w / w.sum()
        scale = math.sqrt(0.45)
        along = np.array([0.6, 0.3]) / scale
        across = np.array([-0.3, 0.6]) / scale
        u, v = np.meshgrid(x, x, indexing="ij")
        self.weights = np.outer(self.node_weights, self.node_weights)
        self.w1 = u * along[0] + v * across[0]
        self.w2 = u * along[1] + v * across[1]
        self.phi = (np.ones_like(self.w1), self.w1)

    def mean(self, values: np.ndarray) -> float:
        """``E[values(W)]``."""
        return float((self.weights * values).sum())

    def given_index(self, values: np.ndarray) -> np.ndarray:
        """``E[values(W) | g_0(W)]`` at each node: the mean over ``v`` at fixed ``u``."""
        return np.broadcast_to(
            (values * self.node_weights[None, :]).sum(axis=1)[:, None], values.shape
        )


def logistic_limit(grid: Grid, design: Any, target: dict[int, np.ndarray]) -> np.ndarray:
    """The population logistic fit over both arms, weighted by ``g_0(a | W)``.

    ``design(a)`` is the list of the fit's columns at arm ``a`` (an intercept first); Newton's
    method solves ``sum_a E[g_0(a | W) x_a (target_a - expit(x_a' b))] = 0``.
    """
    g1 = g0(grid.w1, grid.w2)
    mass = {1: g1, 0: 1.0 - g1}
    beta = np.zeros(len(design(0)))
    for _ in range(100):
        score = np.zeros_like(beta)
        hessian = np.zeros((len(beta), len(beta)))
        for a in (0, 1):
            columns = design(a)
            p = expit(sum(b * c for b, c in zip(beta, columns, strict=True)))
            for i, ci in enumerate(columns):
                score[i] += grid.mean(mass[a] * ci * (target[a] - p))
                for j, cj in enumerate(columns):
                    hessian[i, j] += grid.mean(mass[a] * ci * cj * p * (1.0 - p))
        step = np.linalg.solve(hessian, score)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13:
            return beta
    raise RuntimeError("the population logistic fit did not converge")


def wrong_columns(grid: Grid, a: int) -> list[np.ndarray]:
    """The wrong GLM's columns ``(1, A, W1)`` at arm ``a``."""
    return [np.ones_like(grid.w1), np.full_like(grid.w1, float(a)), grid.w1]


class LawP:
    """Every plan 5.1 reading of law P, computed once."""

    def __init__(self, nodes: int = NODES) -> None:
        grid = Grid(nodes)
        self.grid = grid
        g1 = g0(grid.w1, grid.w2)
        self.g = {1: g1, 0: 1.0 - g1}
        self.q0 = {a: qbar0(float(a), grid.w1, grid.w2) for a in (0, 1)}
        phi = grid.phi
        self.gram = np.array([[grid.mean(p * q) for q in phi] for p in phi])
        self.gram_inverse = np.linalg.inv(self.gram)
        tau = self.q0[1] - self.q0[0]
        self.beta = self.gram_inverse @ np.array([grid.mean(p * tau) for p in phi])
        self.wrong_limit = logistic_limit(grid, lambda a: wrong_columns(grid, a), self.q0)
        self.q_wrong = {
            a: expit(
                sum(b * c for b, c in zip(self.wrong_limit, wrong_columns(grid, a), strict=True))
            )
            for a in (0, 1)
        }
        # Q_{r,j,a} = E[phi_j (Qbar_0 - Qbar)(a, W) | g_0], the weighted reduced regression.
        self.qr = {
            a: [grid.given_index(p * (self.q0[a] - self.q_wrong[a])) for p in phi] for a in (0, 1)
        }

    def _terms(self, q: dict[int, np.ndarray], a: int, y: int, subtract: bool) -> list[np.ndarray]:
        """``M (IC)`` at cell ``(a, y)``: the plug-in, the residual and, if asked, ``-D_A``."""
        phi = self.grid.phi
        fitted = self.beta[0] * phi[0] + self.beta[1] * phi[1]
        sign = 1.0 if a == 1 else -1.0
        out = []
        for k, p in enumerate(phi):
            term = p * (q[1] - q[0] - fitted) + sign * p / self.g[a] * (y - q[a])
            if subtract:
                for b in (0, 1):
                    s_b = 1.0 if b == 1 else -1.0
                    took = 1.0 if b == a else 0.0
                    term = term - s_b * self.qr[b][k] / self.g[b] * (took - self.g[b])
            out.append(term)
        return out

    def covariance(self, q: dict[int, np.ndarray], subtract: bool) -> np.ndarray:
        """``E[IC IC']`` of the coefficients at outcome regression ``q`` and the true ``g``."""
        out = np.zeros((2, 2))
        for a in (0, 1):
            for y in (0, 1):
                mass = self.g[a] * (self.q0[a] if y else 1.0 - self.q0[a])
                terms = self._terms(q, a, y, subtract)
                curve = [
                    sum(self.gram_inverse[j, k] * terms[k] for k in range(2)) for j in range(2)
                ]
                for i in range(2):
                    for j in range(2):
                        out[i, j] += self.grid.mean(mass * curve[i] * curve[j])
        return out

    def sd(self, covariance: np.ndarray, n: int = N) -> np.ndarray:
        """The SD of each coefficient at ``n`` rows."""
        return np.sqrt(np.diag(covariance) / n)

    @property
    def efficient(self) -> np.ndarray:
        """The efficient covariance: the curve at the true ``Qbar_0``."""
        return self.covariance(self.q0, subtract=False)

    @property
    def dr_known(self) -> np.ndarray:
        """The covariance of ``IC_beta`` with the declared ``g`` and the wrong ``Qbar``: ``D* - D_A``."""
        return self.covariance(self.q_wrong, subtract=True)

    @property
    def d_star(self) -> np.ndarray:
        """The covariance of ``D*`` alone at the wrong ``Qbar``: the TMLE projection's curve."""
        return self.covariance(self.q_wrong, subtract=False)

    def plugin_displacement(self) -> np.ndarray:
        """The plug-in at the wrong ``Qbar`` minus the truth, in SDs of ``IC_beta`` at 4,000."""
        tau = self.q_wrong[1] - self.q_wrong[0]
        plugin = self.gram_inverse @ np.array([self.grid.mean(p * tau) for p in self.grid.phi])
        return (plugin - self.beta) / self.sd(self.dr_known)

    def conditioning_norms(self) -> dict[int, np.ndarray]:
        """``L_2(F_{j,a})``, ``F = phi_j (Qbar_0 - Qbar) - E[. | g_0]``: H1 fails when nonzero."""
        out = {}
        for a in (1, 0):
            values = []
            for p in self.grid.phi:
                f = p * (self.q0[a] - self.q_wrong[a])
                values.append(math.sqrt(self.grid.mean((f - self.grid.given_index(f)) ** 2)))
            out[a] = np.array(values)
        return out

    def remainder_displacement(self, offset: np.ndarray, n: int) -> np.ndarray:
        r"""The ``R_3`` displacement of ``beta`` in SDs of ``IC_beta`` at ``n`` under ``offset``.

        ``offset`` is the mechanism's logit error times ``n^{1/3}``, so the estimated
        mechanism is ``expit(logit g_0 + n^{-1/3} offset)``.  The term is
        ``M^{-1} sum_a s(a) E[F_{j,a} (g_0 - g_n)(a | W) / g_0(a | W)]``, the remainder proxy
        convention of :func:`tests.studies.drtmle_one_wrong.remainder_proxy`.
        """
        g1n = expit(logit(self.g[1]) + law_l.slow(n) * offset)
        gn = {1: g1n, 0: 1.0 - g1n}
        total = np.zeros(2)
        for a in (0, 1):
            sign = 1.0 if a == 1 else -1.0
            for k, p in enumerate(self.grid.phi):
                f = p * (self.q0[a] - self.q_wrong[a])
                f = f - self.grid.given_index(f)
                total[k] += sign * self.grid.mean(f * (self.g[a] - gn[a]) / self.g[a])
        return math.sqrt(n) * (self.gram_inverse @ total) / np.sqrt(np.diag(self.dr_known))

    def ladder(self, offset: str = "product") -> np.ndarray:
        """The ladder's displacements, ``(rung, term)``, for one of the four offsets of 5.1."""
        w1, w2 = self.grid.w1, self.grid.w2
        offsets = {"product": w2 * (1.0 + w1), "w2": w2, "w1w2": w1 * w2, "w1": w1}
        return np.array([self.remainder_displacement(offsets[offset], n) for n in LADDER])

    def conservative_ratio(self) -> np.ndarray:
        """The predicted SE ratio of ``dr_glm_4k``: the reported over the true SD per coefficient.

        With the correct GLM mechanism on ``(1, W1, W2)`` and the wrong ``Qbar``, the estimator's
        curve is ``IC_beta`` minus its projection on the mechanism GLM's score ``X (A - g_0)``
        (``theorem.md``, "A parametric mechanism makes the interval conservative"), while the
        fit reports ``IC_beta`` itself.
        """
        grid = self.grid
        g1 = self.g[1]
        design = [np.ones_like(grid.w1), grid.w1, grid.w2]
        information = np.array(
            [[grid.mean(x * z * g1 * (1.0 - g1)) for z in design] for x in design]
        )
        cross = np.zeros((2, 3))
        for a in (0, 1):
            residual = (1.0 if a == 1 else 0.0) - g1
            # Y integrated out: E[Y - Qbar | a, W] = Qbar_0 - Qbar at the wrong regression.
            terms = self._terms(self.q_wrong, a, 0, subtract=True)
            sign = 1.0 if a == 1 else -1.0
            terms = [
                t + sign * p / self.g[a] * self.q0[a] for t, p in zip(terms, grid.phi, strict=True)
            ]
            curve = [sum(self.gram_inverse[j, k] * terms[k] for k in range(2)) for j in range(2)]
            for j in range(2):
                for m in range(3):
                    cross[j, m] += grid.mean(self.g[a] * curve[j] * design[m] * residual)
        variance = np.diag(self.dr_known)
        reduction = np.array([c @ np.linalg.solve(information, c) for c in cross])
        return np.sqrt(variance / (variance - reduction))

    def correlation(self) -> float:
        """The correlation of the two coefficients' ``IC_beta``."""
        v = self.dr_known
        return float(v[0, 1] / math.sqrt(v[0, 0] * v[1, 1]))


_LAW: dict[int, LawP] = {}


def law(nodes: int = NODES) -> LawP:
    """The readings of law P at ``nodes``, computed once per process."""
    if nodes not in _LAW:
        _LAW[nodes] = LawP(nodes)
    return _LAW[nodes]


def truth() -> dict[str, float]:
    """The projection coefficients, keyed by the reported name."""
    beta = law().beta
    return {f"cate_projection[{term}]": float(b) for term, b in zip(TERMS, beta, strict=True)}


def design_table() -> dict[str, Any]:
    """Every plan 5.1 reading, by name."""
    p = law()
    return {
        "beta": tuple(p.beta),
        "gram": p.gram,
        "wrong_limit": tuple(p.wrong_limit),
        "efficient_sd": tuple(p.sd(p.efficient)),
        "dr_known_sd": tuple(p.sd(p.dr_known)),
        "d_star_sd": tuple(p.sd(p.d_star)),
        "d_star_ratio": tuple(p.sd(p.d_star) / p.sd(p.dr_known)),
        "plugin_displacement": tuple(p.plugin_displacement()),
        "conditioning_norms": {a: tuple(v) for a, v in p.conditioning_norms().items()},
        "conservative_ratio": tuple(p.conservative_ratio()),
        "correlation": p.correlation(),
        "ladder": {name: p.ladder(name) for name in ("product", "w2", "w1w2", "w1")},
    }


def quadrature_error() -> float:
    """The largest change of ``beta`` from 120 to 80 nodes per axis."""
    return float(np.max(np.abs(law(NODES).beta - law(80).beta)))

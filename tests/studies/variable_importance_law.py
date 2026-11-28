"""The laws of the ``variable-importance`` study, with exact truths and the Gaussian oracle.

``W ~ N(0, 1)``. Ten binary candidates ``X1..X10`` and a three-level ``Z`` follow a pairwise
log-linear law given ``W``::

    P(x, z | W) ∝ exp( Σ_j x_j (a_j + c_j W) + z (a_z + c_z W) + d_z[z]
                       + Σ_{j<k} J_jk x_j x_k + Σ_j K_j x_j z )

Every conditional is then a main-terms logit (a multinomial logit for ``Z``) in ``W``, the other
candidates and ``Z`` as a number. The outcome is ``Y ~ Bernoulli(expit(b0 + Σ β_j X_j + β_z Z +
γ W))``, so ``LogisticRegression`` is correctly specified for every learner role. Sampling
enumerates the 3,072 states.

The twelve parameters are the controlled ATEs of ``X1..X10`` and of ``Z`` at 1 and 2 against
0, each adjusted for ``W`` and every other candidate (``adjust_for_other_candidates=True``).
:meth:`Law.exact` computes them and the exact covariance of their efficient influence curves by
enumeration and 40-node Gauss-Hermite quadrature: ``Σ_jk = E[H_j H_k Q(1 - Q)] + Cov(P_j, P_k)``,
with ``H`` the clever covariate and ``P`` the plug-in contrast; the cross term is zero. A null
row's truth is exactly 0. :meth:`Law.exact_unadjusted` computes the truths of the
``adjust_for_other_candidates=False`` run, the ``W``-adjusted risk differences, which the
adjustment witness reads.

:func:`oracle` applies the step-down max-T, Benjamini-Hochberg and Benjamini-Yekutieli to
``N(mu_n, R)`` draws on the exact ``R``: the planning values and the oracle columns of the
evidence page. :func:`step_down` is this module's own step-down, written apart from
:mod:`cleverly.variable_importance`, and the study's identity gate compares the two.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

#: The number of binary candidates.
BINARY = 10
#: The candidate columns, in fit order.
CANDIDATES: tuple[str, ...] = (*(f"X{j + 1}" for j in range(BINARY)), "Z")
#: The twelve parameters, in fit order: one per binary candidate and two for ``Z``.
NAMES: tuple[str, ...] = (*(f"X{j + 1}" for j in range(BINARY)), "Z:1", "Z:2")
#: A truth below this in absolute value is a null.
NULL_TOLERANCE = 1e-12
#: Quadrature nodes for the exact moments.
QUADRATURE_NODES = 40
#: The signal strength of the mixed laws.
STRONG = 0.6


@dataclass(frozen=True)
class Law:
    """One pairwise log-linear law of the candidates and its logistic outcome.

    Parameters
    ----------
    name : str
        The law's name.
    beta : tuple of float
        The outcome coefficients of ``X1..X10``.
    beta_z : float
        The outcome coefficient of ``Z``.
    J : tuple of tuple of float
        The symmetric pairwise interactions of the binary candidates.
    K : tuple of float
        The interactions of each binary candidate with ``Z``.
    a : tuple of float
        The main terms of the binary candidates.
    c : tuple of float
        Their slopes in ``W``.
    a_z : float
        The main term of ``Z``.
    c_z : float
        The slope of ``Z`` in ``W``.
    d_z : tuple of float
        The level offsets of ``Z``.
    b0 : float
        The outcome intercept.
    gamma : float
        The outcome's slope in ``W``.
    gated : bool
        Whether the law's cells are gated; ``collinear_pair`` is reported.
    """

    name: str
    beta: tuple[float, ...]
    beta_z: float
    J: tuple[tuple[float, ...], ...]
    K: tuple[float, ...]
    a: tuple[float, ...] = (0.0,) * BINARY
    c: tuple[float, ...] = (0.5,) * BINARY
    a_z: float = -0.3  # numpydoc ignore=PR01
    c_z: float = 0.4  # numpydoc ignore=PR01
    d_z: tuple[float, float, float] = (0.0, 0.0, -0.5)
    b0: float = -0.4  # numpydoc ignore=PR01
    gamma: float = 0.8  # numpydoc ignore=PR01
    gated: bool = True
    _states: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        xs = np.array(list(itertools.product((0, 1), repeat=BINARY)), dtype=float)
        states = np.vstack([np.column_stack([xs, np.full(len(xs), z)]) for z in (0, 1, 2)])
        object.__setattr__(self, "_states", states)

    @property
    def states(self) -> np.ndarray:
        """Every joint state ``(x_1..x_10, z)``, one row per state."""
        return self._states

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.beta, dtype=float),
            np.asarray(self.J, dtype=float),
            np.asarray(self.K, dtype=float),
            np.asarray(self.a, dtype=float),
            np.asarray(self.c, dtype=float),
        )

    def probabilities(self, w: Sequence[float] | np.ndarray) -> np.ndarray:
        """``P(state | W = w)``, one row per ``w``."""
        _, J, K, a, c = self._arrays()
        x, z = self.states[:, :BINARY], self.states[:, BINARY].astype(int)
        const = (
            x @ a
            + z * self.a_z
            + np.asarray(self.d_z)[z]
            + 0.5 * np.einsum("sj,jk,sk->s", x, J, x)
            + (x @ K) * z
        )
        slope = x @ c + z * self.c_z
        energy = const[None, :] + np.asarray(w, dtype=float)[:, None] * slope[None, :]
        energy -= energy.max(axis=1, keepdims=True)
        p = np.exp(energy)
        return np.asarray(p / p.sum(axis=1, keepdims=True), dtype=float)

    def linear_predictor(self, w: float) -> np.ndarray:
        """The outcome's log-odds at every state, at ``W = w``."""
        beta = np.asarray(self.beta, dtype=float)
        x, z = self.states[:, :BINARY], self.states[:, BINARY]
        return np.asarray(self.b0 + x @ beta + self.beta_z * z + self.gamma * w, dtype=float)

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """``n`` rows from ``numpy.random.default_rng(seed)``: the candidates, ``Z``, ``W``, ``Y``."""
        rng = np.random.default_rng(seed)
        w = rng.normal(size=n)
        p = self.probabilities(w)
        u = rng.random(n)
        index = np.minimum((p.cumsum(axis=1) < u[:, None]).sum(axis=1), len(self.states) - 1)
        state = self.states[index]
        beta = np.asarray(self.beta, dtype=float)
        eta = self.b0 + state[:, :BINARY] @ beta + self.beta_z * state[:, BINARY] + self.gamma * w
        y = rng.binomial(1, expit(eta))
        frame = pd.DataFrame(state[:, :BINARY].astype(int), columns=list(CANDIDATES[:BINARY]))
        frame["Z"] = state[:, BINARY].astype(int)
        frame["W"] = w
        frame["Y"] = y
        return frame

    def _partners(self) -> tuple[np.ndarray, np.ndarray]:
        """For each state, the state with ``x_j`` flipped, and the states with ``z`` set."""
        size = 2**BINARY
        index = np.arange(len(self.states))
        xi, zi = index % size, index // size
        flip = np.stack([zi * size + (xi ^ (1 << (BINARY - 1 - j))) for j in range(BINARY)], axis=1)
        zset = np.stack([level * size + xi for level in range(3)], axis=1)
        return flip, zset

    def exact(self) -> tuple[np.ndarray, np.ndarray, float]:
        """The twelve truths, the exact influence-curve covariance, the smallest propensity.

        Returns
        -------
        tuple
            ``(psi, covariance, smallest_propensity)``: ``psi`` and the ``(12, 12)``
            covariance in :data:`NAMES` order, and the smallest conditional propensity of any
            candidate level at any state and node.
        """
        nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
        weights = weights / weights.sum()
        flip, zset = self._partners()
        m = len(NAMES)
        x, z = self.states[:, :BINARY], self.states[:, BINARY].astype(int)
        ehh = np.zeros((m, m))
        plug = np.zeros(m)
        plug2 = np.zeros((m, m))
        smallest = 1.0
        for node, weight in zip(nodes, weights, strict=True):
            p = self.probabilities([node])[0]
            q = expit(self.linear_predictor(node))
            h = np.zeros((len(p), m))
            contrast = np.zeros((len(p), m))
            for j in range(BINARY):
                partner = p[flip[:, j]]
                on = x[:, j] == 1
                g = np.where(on, p / (p + partner), partner / (p + partner))
                smallest = min(smallest, float(g.min()), float((1 - g).min()))
                h[:, j] = np.where(on, 1 / g, -1 / (1 - g))
                q_partner = q[flip[:, j]]
                contrast[:, j] = np.where(on, q - q_partner, q_partner - q)
            pz = p[zset]
            gz = pz / pz.sum(axis=1, keepdims=True)
            smallest = min(smallest, float(gz.min()))
            qz = q[zset]
            for column, level in ((BINARY, 1), (BINARY + 1, 2)):
                h[:, column] = (z == level) / gz[:, level] - (z == 0) / gz[:, 0]
                contrast[:, column] = qz[:, level] - qz[:, 0]
            variance = q * (1 - q)
            ehh += weight * (h * (p * variance)[:, None]).T @ h
            plug += weight * p @ contrast
            plug2 += weight * (contrast * p[:, None]).T @ contrast
        covariance = ehh + plug2 - np.outer(plug, plug)
        return plug, covariance, smallest

    def exact_unadjusted(self) -> np.ndarray:
        """The truths of the ``adjust_for_other_candidates=False`` run, in :data:`NAMES` order.

        Each is the ``W``-adjusted contrast ``E_W[E(Y | X_j = 1, W) - E(Y | X_j = 0, W)]``, and
        for ``Z`` at level ``k`` against 0 the same with ``Z`` in place of ``X_j``.
        """
        nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
        weights = weights / weights.sum()
        x, z = self.states[:, :BINARY], self.states[:, BINARY].astype(int)
        out = np.zeros(len(NAMES))
        for node, weight in zip(nodes, weights, strict=True):
            p = self.probabilities([node])[0]
            q = expit(self.linear_predictor(node))

            def mean(mask: np.ndarray, p: np.ndarray = p, q: np.ndarray = q) -> float:
                return float((p * q)[mask].sum() / p[mask].sum())

            for j in range(BINARY):
                out[j] += weight * (mean(x[:, j] == 1) - mean(x[:, j] == 0))
            for column, level in ((BINARY, 1), (BINARY + 1, 2)):
                out[column] += weight * (mean(z == level) - mean(z == 0))
        return out

    def propensity_quantiles(
        self, n: int = 20_000, seed: int = 0, quantiles: Sequence[float] = (0.0, 0.001, 0.01)
    ) -> list[float]:
        """Quantiles of the smallest conditional propensity of each row's candidates."""
        rng = np.random.default_rng(seed)
        w = rng.normal(size=n)
        p = self.probabilities(w)
        u = rng.random(n)
        index = np.minimum((p.cumsum(axis=1) < u[:, None]).sum(axis=1), len(self.states) - 1)
        flip, zset = self._partners()
        rows = np.arange(n)
        values = []
        for j in range(BINARY):
            own, partner = p[rows, index], p[rows, flip[index, j]]
            g = own / (own + partner)
            values.append(np.minimum(g, 1 - g))
        pz = p[rows[:, None], zset[index]]
        values.append((pz / pz.sum(axis=1, keepdims=True)).min(axis=1))
        joined = np.concatenate(values)
        return [float(np.quantile(joined, level)) for level in quantiles]


def _pairs(pairs: Sequence[tuple[int, int]], value: float) -> tuple[np.ndarray, np.ndarray]:
    """Interactions ``J`` on the pairs, with main terms that centre each pair."""
    J = np.zeros((BINARY, BINARY))
    a = np.zeros(BINARY)
    for i, k in pairs:
        J[i, k] = J[k, i] = value
        a[i] -= value / 2
        a[k] -= value / 2
    return J, a


def _tuple2(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in matrix)


def _tuple(vector: np.ndarray | Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in vector)


def build_laws() -> dict[str, Law]:
    """The six laws of the study (design rev 2, section 4)."""
    zero = np.zeros(BINARY)
    none = np.zeros((BINARY, BINARY))
    strong = np.array([STRONG] * 3 + [-STRONG] + [0.0] * 6)
    five_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    laws: dict[str, Law] = {}
    laws["independent_null"] = Law(
        "independent_null", _tuple(zero), 0.0, _tuple2(none), _tuple(zero)
    )
    laws["independent_mixed"] = Law(
        "independent_mixed", _tuple(strong), 0.35, _tuple2(none), _tuple(zero)
    )
    J, a = _pairs(five_pairs, 3.0)
    laws["negative_pairs_null"] = Law(
        "negative_pairs_null", _tuple(zero), 0.0, _tuple2(J), _tuple(zero), a=_tuple(a)
    )
    J, a = _pairs([(4, 5), (6, 7), (8, 9)], 3.0)
    J2, a2 = _pairs([(0, 4), (1, 6), (2, 8), (3, 9)], 1.5)
    K = np.zeros(BINARY)
    K[0] = 0.6
    laws["nulls_beside_signals"] = Law(
        "nulls_beside_signals", _tuple(strong), 0.0, _tuple2(J + J2), _tuple(K), a=_tuple(a + a2)
    )
    J = np.zeros((BINARY, BINARY))
    a = np.zeros(BINARY)
    nulls, signals = range(4, 10), range(4)
    for i in nulls:
        for k in nulls:
            if i < k:
                J[i, k] = J[k, i] = -1.0
    for s in signals:
        for i in nulls:
            J[s, i] = J[i, s] = 1.0
    a[list(nulls)] = 0.5
    a[list(signals)] = -3.0
    beta = np.array([0.36, 0.36, 0.65, 0.65] + [0.0] * 6)
    laws["dobriban_like"] = Law(
        "dobriban_like", _tuple(beta), 0.0, _tuple2(J), _tuple(zero), a=_tuple(a)
    )
    J, a = _pairs([(0, 1)], 6.0)
    laws["collinear_pair"] = Law(
        "collinear_pair",
        _tuple([0.6] + [0.0] * 9),
        0.0,
        _tuple2(J),
        _tuple(zero),
        a=_tuple(a),
        gated=False,
    )
    return laws


#: The six laws, in the study's order.
LAWS: Mapping[str, Law] = build_laws()


# ------------------------------------------------------------------ procedures


def step_down(statistics: np.ndarray, null_draws: np.ndarray) -> np.ndarray:
    """Step-down max-T adjusted p-values by an explicit loop over the steps.

    ``statistics`` is ``(datasets, m)`` or ``(m,)`` of absolute statistics, ``null_draws`` is
    ``(B, m)`` of absolute null draws. The result is monotone in rank and carries no floor at
    the rows' own p-values (the caller applies it).
    """
    t = np.atleast_2d(np.asarray(statistics, dtype=float))
    out = np.empty_like(t)
    for row in range(t.shape[0]):
        order = np.argsort(-t[row], kind="stable")
        running = 0.0
        for k in range(t.shape[1]):
            remaining = order[k:]
            share = float(np.mean(null_draws[:, remaining].max(axis=1) >= t[row, order[k]]))
            running = max(running, share)
            out[row, order[k]] = running
    return out if np.ndim(statistics) == 2 else out[0]


def bh_reject(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg rejections at level ``q``, row by row of ``pvalues``."""
    p = np.atleast_2d(np.asarray(pvalues, dtype=float))
    m = p.shape[-1]
    ordered = np.sort(p, axis=-1)
    passes = ordered <= q * np.arange(1, m + 1) / m
    k = np.where(passes.any(axis=-1), m - np.argmax(passes[..., ::-1], axis=-1), 0)
    cut = np.where(
        k > 0, np.take_along_axis(ordered, np.maximum(k - 1, 0)[..., None], -1)[..., 0], -1.0
    )
    return p <= cut[..., None]


def by_level(q: float, m: int) -> float:
    """The Benjamini-Yekutieli level: ``q`` over the harmonic number of ``m``."""
    return q / float(np.sum(1.0 / np.arange(1, m + 1)))


def correlation(covariance: np.ndarray) -> np.ndarray:
    """The correlation matrix of a covariance matrix."""
    scale = np.sqrt(np.diag(covariance))
    return np.asarray(covariance / np.outer(scale, scale), dtype=float)


def oracle(
    law: Law,
    n: int,
    *,
    alphas: Sequence[float] = (0.05, 0.10),
    datasets: int = 8_000,
    draws: int = 10_000,
    seed: int,
) -> dict[float, dict[str, Any]]:
    """The Gaussian oracle of one law at ``n``: max-T, BH and BY on ``N(mu_n, R)``.

    ``mu_n = psi / (sd / sqrt(n))`` and ``R`` are exact. Returns, for each level, the max-T
    FWER, its FDR and per-row power, and the FDR and power of BH and BY.
    """
    psi, covariance, _ = law.exact()
    sd = np.sqrt(np.diag(covariance))
    exact_r = correlation(covariance)
    mu = psi / (sd / np.sqrt(n))
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(exact_r)
    statistics = mu + rng.standard_normal((datasets, len(mu))) @ factor.T
    null_draws = np.abs(rng.standard_normal((draws, len(mu))) @ factor.T)
    absolute = np.abs(statistics)
    pvalues = 2.0 * norm.sf(absolute)
    adjusted = np.maximum(step_down(absolute, null_draws), pvalues)
    null = np.abs(psi) < NULL_TOLERANCE
    out: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        reading: dict[str, Any] = {}
        for name, rejected in (
            ("max_t", adjusted <= alpha),
            ("bh", bh_reject(pvalues, alpha)),
            ("by", bh_reject(pvalues, by_level(alpha, len(mu)))),
        ):
            false = rejected[:, null].sum(axis=1)
            total = rejected.sum(axis=1)
            reading[f"{name}_fwer"] = float(np.mean(false > 0))
            reading[f"{name}_fdr"] = float(
                np.mean(np.where(total > 0, false / np.maximum(total, 1), 0.0))
            )
            reading[f"{name}_power"] = {
                NAMES[j]: float(np.mean(rejected[:, j])) for j in np.flatnonzero(~null)
            }
        out[alpha] = reading
    return out


def fixed_draw_fwer(law: Law, *, draws: int, seed: int, fresh: int, fresh_seed: int) -> float:
    """The complete-null FWER of the first step's critical value from one fixed draw matrix.

    The study fixes ``multiplier_seed`` across replicates, so the Monte Carlo error of the
    critical value is common to every replicate (design review R2-3). This reads it on the
    exact ``R``: the critical value at level 0.05 of ``draws`` draws from ``seed``, and its
    exact rejection rate estimated on ``fresh`` independent draws.
    """
    _, covariance, _ = law.exact()
    factor = np.linalg.cholesky(correlation(covariance))
    m = factor.shape[0]
    fixed = np.abs(np.random.default_rng(seed).standard_normal((draws, m)) @ factor.T).max(axis=1)
    critical = float(np.quantile(fixed, 0.95, method="higher"))
    rng = np.random.default_rng(fresh_seed)
    maxima = np.abs(rng.standard_normal((fresh, m)) @ factor.T).max(axis=1)
    return float(np.mean(maxima >= critical))

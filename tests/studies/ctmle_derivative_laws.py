r"""The declared laws of the C-TMLE0 study, ``ctmle-derivative``.

``W = (W1, W2, W3, W4)`` is iid ``N(0, 1)``.  ``W1`` enters the treatment law only (an
instrument), ``W2`` and ``W3`` are confounders, and ``W4`` enters the outcome only.  The
outcome is binary with

.. math::

    \operatorname{logit} \bar Q_0(a, W) = c_a + u^\top (W_2, W_3, W_4),

a main-terms GLM in ``[I(A = a), W]``, so an unpenalized logistic regression on the outcome
design is correctly specified.  Every arm shares the slope ``u``, so an arm mean is a
one-dimensional Gauss-Hermite integral over the normal index ``u^T W``.

The treatment law is main-terms logistic (a softmax with arm 0 the baseline at three arms),
so the internal lasso is consistent for it.  The ``outcome_correct`` law adds the logit term
``v12 W1 W2``, which no main-terms model reads.

The missing-outcome law keeps the primary law and adds a logistic response mechanism
``pi(A, W)``.  The outcome is missing at random, so every truth is the primary law's.

The clustered law keeps the primary law's marginals.  ``Y = 1{U < Qbar_0(A, W)}``, where
``U = Phi(sqrt(rho) B + sqrt(1 - rho) E)`` with ``B`` shared within a cluster.  ``U`` is
uniform for every row, so ``E[Y | A, W] = Qbar_0(A, W)``.  The treatment draws are
independent given ``W``, so ``P(A = 1 | W)`` stays main-terms logistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, softmax
from scipy.stats import norm

#: The treatment bounds every cell declares.
G_BOUNDS = (0.025, 0.975)

#: Gauss-Hermite nodes of every arm mean.
HERMITE_NODES = 80

#: The covariate names, in design order.
COVARIATES = ("W1", "W2", "W3", "W4")


@dataclass(frozen=True)
class DerivativeLaw:
    """One declared law.

    Parameters
    ----------
    name : str
        The law's name in the study.
    intercepts : tuple of float
        ``c_a`` per arm, in arm order.  Two entries for a binary treatment.
    slope : tuple of float
        ``u`` on ``(W2, W3, W4)``, shared by every arm.
    scores : tuple of tuple of float
        One row per non-baseline arm: the treatment logit's coefficients on ``W``.
    interaction : float
        The coefficient of ``W1 W2`` in every non-baseline arm's score.
    response : tuple of float or None
        ``(intercept, on A, on W2, on W4)`` of the response mechanism's logit, or ``None``
        for a law with no missing outcome.
    clusters : int or None
        The cluster size of the clustered law, or ``None``.
    rho : float
        The within-cluster correlation of the copula's normal scores.
    """

    name: str
    intercepts: tuple[float, ...]
    slope: tuple[float, float, float]
    scores: tuple[tuple[float, float, float, float], ...]
    interaction: float = 0.0
    response: tuple[float, float, float, float] | None = None
    clusters: int | None = None
    rho: float = 0.0
    arms: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.intercepts) - 1:
            raise ValueError("a law needs one score row per non-baseline arm")
        object.__setattr__(self, "arms", tuple(float(a) for a in range(len(self.intercepts))))

    @property
    def k(self) -> int:
        return len(self.intercepts)

    def mechanism(self, w: np.ndarray) -> np.ndarray:
        """``(n, K)``: ``P(A = a | W)`` per arm."""
        logits = [np.zeros(w.shape[0])]
        for row in self.scores:
            logits.append(w @ np.asarray(row) + self.interaction * w[:, 0] * w[:, 1])
        values = np.column_stack(logits)
        return np.asarray(softmax(values, axis=1), dtype=float)

    def outcome_probability(self, arm: int, w: np.ndarray) -> np.ndarray:
        """``Qbar_0(arm, W)``."""
        return np.asarray(
            expit(self.intercepts[arm] + w[:, 1:] @ np.asarray(self.slope)), dtype=float
        )

    def response_probability(self, a: np.ndarray, w: np.ndarray) -> np.ndarray:
        """``pi(A, W) = P(Delta = 1 | A, W)``; ones on a law with no missing outcome."""
        if self.response is None:
            return np.ones(w.shape[0])
        c, on_a, on_w2, on_w4 = self.response
        return np.asarray(expit(c + on_a * a + on_w2 * w[:, 1] + on_w4 * w[:, 3]), dtype=float)

    def arm_means(self) -> np.ndarray:
        """``E[Qbar_0(a, W)]`` per arm, by Gauss-Hermite over ``u^T W ~ N(0, |u|^2)``."""
        nodes, weights = hermegauss(HERMITE_NODES)
        weights = weights / weights.sum()
        scale = float(np.linalg.norm(self.slope))
        return np.array([float(weights @ expit(c + scale * nodes)) for c in self.intercepts])

    def truth(self) -> dict[str, float]:
        """Every estimand the study reports on this law."""
        means = self.arm_means()
        out = {f"ey[{a:g}]": float(m) for a, m in zip(self.arms, means, strict=True)}
        if self.k == 2:
            out.update(
                ate=float(means[1] - means[0]),
                ey1=float(means[1]),
                ey0=float(means[0]),
                rr=float(means[1] / means[0]),
            )
        else:
            for a in range(1, self.k):
                out[f"ate[{a}-0]"] = float(means[a] - means[0])
        return out

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """One dataset of ``n`` rows from the law, with a cluster id on the clustered law."""
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((n, 4))
        g = self.mechanism(w)
        cumulative = np.cumsum(g, axis=1)
        a = (rng.uniform(size=(n, 1)) > cumulative).sum(axis=1).astype(float)
        q = np.choose(a.astype(int), [self.outcome_probability(j, w) for j in range(self.k)])
        if self.clusters is None:
            u = rng.uniform(size=n)
            cluster = None
        else:
            if n % self.clusters:
                raise ValueError(f"n = {n} is not a whole number of clusters of {self.clusters}")
            cluster = np.repeat(np.arange(n // self.clusters), self.clusters)
            shared = rng.standard_normal(n // self.clusters)[cluster]
            own = rng.standard_normal(n)
            u = norm.cdf(np.sqrt(self.rho) * shared + np.sqrt(1.0 - self.rho) * own)
        y = (u < q).astype(float)
        frame = pd.DataFrame(w, columns=list(COVARIATES))
        frame["A"] = a
        if self.response is not None:
            observed = rng.uniform(size=n) < self.response_probability(a, w)
            frame["Delta"] = observed.astype(float)
            y = np.where(observed, y, np.nan)
        frame["Y"] = y
        if cluster is not None:
            frame["cluster"] = cluster
        return frame

    def eif_sd(self, draws: int, seed: int, estimand: str = "ate") -> tuple[float, float]:
        """The efficient influence curve's SD at the truth, and its Monte Carlo error.

        Complete data, binary treatment, ``ate``: the variance is
        ``E[Q1 (1 - Q1) / g1 + Q0 (1 - Q0) / g0 + (Q1 - Q0 - psi)^2]``.  The error is the
        delta-method standard error of the SD from the per-draw terms.
        """
        if self.k != 2 or estimand != "ate" or self.response is not None:
            raise ValueError("eif_sd covers the complete-data binary ate only")
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((draws, 4))
        g = self.mechanism(w)
        q0, q1 = self.outcome_probability(0, w), self.outcome_probability(1, w)
        psi = self.truth()["ate"]
        terms = q1 * (1 - q1) / g[:, 1] + q0 * (1 - q0) / g[:, 0] + (q1 - q0 - psi) ** 2
        variance = float(terms.mean())
        error = float(terms.std(ddof=1) / np.sqrt(draws)) / (2.0 * np.sqrt(variance))
        return float(np.sqrt(variance)), error


_SLOPE = (0.6, 0.5, 0.7)
_SCORE = ((0.6, 0.5, -0.4, 0.0),)

#: The primary law: main-terms logistic treatment, GLM-correct outcome.
PRIMARY = DerivativeLaw("primary", (-0.3, 0.2), _SLOPE, _SCORE)
#: The primary law with no treatment effect; ``Qbar`` still depends on ``W``.
SHARP_NULL = DerivativeLaw("sharp_null", (-0.3, -0.3), _SLOPE, _SCORE)
#: The power cell's law: the primary law at the declared contrast.
ALTERNATIVE = DerivativeLaw("alternative", (-0.3, 0.07), _SLOPE, _SCORE)
#: A treatment law with a ``W1 W2`` logit term, which the main-terms lasso cannot read.
OUTCOME_CORRECT = DerivativeLaw("outcome_correct", (-0.3, 0.2), _SLOPE, _SCORE, interaction=0.8)
#: Three arms: a main-terms multinomial treatment law and a GLM-correct outcome.
THREE_ARM = DerivativeLaw(
    "three_arm",
    (-0.3, 0.2, 0.5),
    _SLOPE,
    ((0.5, 0.4, -0.3, 0.0), (-0.4, 0.3, 0.3, 0.0)),
)
#: The primary law with outcomes missing at random.
MISSING_OUTCOME = DerivativeLaw(
    "missing_outcome", (-0.3, 0.2), _SLOPE, _SCORE, response=(1.2, -0.3, 0.4, 0.3)
)
#: The primary law in 100 clusters of 20, dependent through a Gaussian copula.
CLUSTERED = DerivativeLaw("clustered", (-0.3, 0.2), _SLOPE, _SCORE, clusters=20, rho=0.3)

LAWS = {
    law.name: law
    for law in (
        PRIMARY,
        SHARP_NULL,
        ALTERNATIVE,
        OUTCOME_CORRECT,
        THREE_ARM,
        MISSING_OUTCOME,
        CLUSTERED,
    )
}

#: The power cell's size and replicates, and the calibration cells' replicates.
POWER_N = 2_000
POWER_REPLICATES = 1_200
CALIBRATION_REPLICATES = 2_400
#: The efficient influence curve's SD of the alternative law's ``ate`` at ``10^6`` draws (seed
#: 1), and its Monte Carlo error.
ALTERNATIVE_EIF_SD = 1.0005
ALTERNATIVE_EIF_SD_ERROR = 0.00021
#: The KL gap of the main-terms logistic projection of the treatment law, at ``10^5`` draws:
#: below the first floor on the primary law, above the second on ``outcome_correct``.
PRIMARY_KL_CEILING = 1e-3
OUTCOME_CORRECT_KL_FLOOR = 1e-2
#: The largest share of mechanism values the three-arm law may put outside ``G_BOUNDS``.
TRUNCATION_CEILING = 0.01

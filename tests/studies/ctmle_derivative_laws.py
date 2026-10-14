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
``beta W2 |W2|``, which no main-terms model reads.

The missing-outcome law keeps the primary law and adds a logistic response mechanism
``pi(A, W)``.  The outcome is missing at random, so every truth is the primary law's.

The clustered law keeps the primary law's marginals.  ``Y = 1{U < Qbar_0(A, W)}``, where
``U = Phi(sqrt(rho) B + sqrt(1 - rho) E)`` with ``B`` shared within a cluster.  The treatment
is ``A = 1{V < g_0(W)}`` with ``V`` built the same way from its own shared score and
``treatment_rho``.  ``U`` and ``V`` are uniform and independent of ``W`` for every row, so
``E[Y | A, W] = Qbar_0(A, W)`` and ``P(A = 1 | W)`` stays main-terms logistic.  Correlated
treatments within a cluster correlate the clever covariates, so the influence curve's
cluster design effect is about 2.5 (:data:`CLUSTER_DESIGN_EFFECT`).  Residual correlation
alone would leave it near 1, because ``E[H | W] = 0``.
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
    curvature : float
        The coefficient of ``W2 |W2|`` in every non-baseline arm's score.  ``W2`` is a
        confounder, so the term the main-terms lasso cannot read moves the outcome.
    response : tuple of float or None
        ``(intercept, on A, on W2, on W4)`` of the response mechanism's logit, or ``None``
        for a law with no missing outcome.
    clusters : int or None
        The cluster size of the clustered law, or ``None``.
    rho : float
        The within-cluster correlation of the outcome copula's normal scores.
    treatment_rho : float
        The within-cluster correlation of the treatment copula's normal scores.
    """

    name: str
    intercepts: tuple[float, ...]
    slope: tuple[float, float, float]
    scores: tuple[tuple[float, float, float, float], ...]
    curvature: float = 0.0
    response: tuple[float, float, float, float] | None = None
    clusters: int | None = None
    rho: float = 0.0
    treatment_rho: float = 0.0
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
            logits.append(w @ np.asarray(row) + self.curvature * w[:, 1] * np.abs(w[:, 1]))
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
                out[f"ate[{float(a)!r} vs 0.0]"] = float(means[a] - means[0])
        return out

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """One dataset of ``n`` rows from the law, with a cluster id on the clustered law."""
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((n, 4))
        g = self.mechanism(w)
        cumulative = np.cumsum(g, axis=1)
        cluster = None
        if self.clusters is None:
            v = rng.uniform(size=(n, 1))
        else:
            if n % self.clusters:
                raise ValueError(f"n = {n} is not a whole number of clusters of {self.clusters}")
            cluster = np.repeat(np.arange(n // self.clusters), self.clusters)
            v = _copula(rng, cluster, self.treatment_rho)[:, None]
        a = (v > cumulative).sum(axis=1).astype(float)
        q = np.choose(a.astype(int), [self.outcome_probability(j, w) for j in range(self.k)])
        u = rng.uniform(size=n) if cluster is None else _copula(rng, cluster, self.rho)
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

        For the contrast of arm ``a`` against arm 0, with ``pi`` the response mechanism, the
        variance is ``E[Qa (1 - Qa) / (ga pia) + Q0 (1 - Q0) / (g0 pi0) + (Qa - Q0 - psi)^2]``.
        ``estimand`` is ``"ate"`` on a binary law and ``"ate[a vs 0.0]"`` otherwise.  The
        error is the delta-method standard error of the SD from the per-draw terms.  The
        clustered law has no such bound, because its rows are dependent.
        """
        if self.clusters is not None:
            raise ValueError("eif_sd has no bound for a law with dependent rows")
        if estimand not in self.truth() or not estimand.startswith("ate"):
            raise ValueError(f"{estimand!r} is not a contrast of this law")
        arm = 1 if estimand == "ate" else int(float(estimand[4:].split(" vs ", 1)[0]))
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((draws, 4))
        g = self.mechanism(w)
        q0, qa = self.outcome_probability(0, w), self.outcome_probability(arm, w)
        pi0 = self.response_probability(np.zeros(draws), w)
        pia = self.response_probability(np.full(draws, float(arm)), w)
        psi = self.truth()[estimand]
        terms = (
            qa * (1 - qa) / (g[:, arm] * pia)
            + q0 * (1 - q0) / (g[:, 0] * pi0)
            + (qa - q0 - psi) ** 2
        )
        variance = float(terms.mean())
        error = float(terms.std(ddof=1) / np.sqrt(draws)) / (2.0 * np.sqrt(variance))
        return float(np.sqrt(variance)), error


def _copula(rng: np.random.Generator, cluster: np.ndarray, rho: float) -> np.ndarray:
    """Uniforms whose normal scores share a cluster component of variance ``rho``."""
    shared = rng.standard_normal(int(cluster.max()) + 1)[cluster]
    own = rng.standard_normal(cluster.size)
    return np.asarray(norm.cdf(np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * own), dtype=float)


_SLOPE = (0.6, 0.5, 0.7)
_SCORE = ((0.6, 0.5, -0.4, 0.0),)

#: The primary law: main-terms logistic treatment, GLM-correct outcome.
PRIMARY = DerivativeLaw("primary", (-0.3, 0.2), _SLOPE, _SCORE)
#: The primary law with no treatment effect; ``Qbar`` still depends on ``W``.
SHARP_NULL = DerivativeLaw("sharp_null", (-0.3, -0.3), _SLOPE, _SCORE)
#: The power cell's law: the primary law at the declared contrast.
ALTERNATIVE = DerivativeLaw("alternative", (-0.3, 0.07), _SLOPE, _SCORE)
#: The primary law plus ``W2 |W2|`` in the treatment logit, which the main-terms lasso cannot
#: read.  :mod:`tests.studies.ctmle_derivative_dr_pilot` sizes the coefficient.
OUTCOME_CURVATURE = 0.65
OUTCOME_CORRECT = DerivativeLaw(
    "outcome_correct", (-0.3, 0.2), _SLOPE, _SCORE, curvature=OUTCOME_CURVATURE
)
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
#: The primary law in 100 clusters of 20, with treatment and outcome dependent within a
#: cluster through two Gaussian copulas.
CLUSTERED = DerivativeLaw(
    "clustered", (-0.3, 0.2), _SLOPE, _SCORE, clusters=20, rho=0.5, treatment_rho=0.5
)

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
#: The efficient influence curve's SD of each calibration law's contrast, at ``10^6`` draws
#: (seed 1), and its Monte Carlo error.  The study publishes each calibration cell's empirical
#: and reported SD against it.
PRIMARY_EIF_SD = 0.99989
PRIMARY_EIF_SD_ERROR = 0.00021
THREE_ARM_EIF_SD = 1.23389
THREE_ARM_EIF_SD_ERROR = 0.00033
MISSING_OUTCOME_EIF_SD = 1.18333
MISSING_OUTCOME_EIF_SD_ERROR = 0.00027
#: The KL gap of the main-terms logistic projection of the treatment law, at ``10^5`` draws:
#: below this ceiling on the primary law.  The pilot's floor on ``outcome_correct`` is twice it.
PRIMARY_KL_CEILING = 1e-3
#: The largest share of mechanism values the three-arm law may put outside ``G_BOUNDS``.
TRUNCATION_CEILING = 0.01
#: The smallest influence-curve cluster design effect the clustered law may have.  The IID
#: control's SE ratio is then about ``1 / sqrt(2) = 0.71``, below the family's 0.80 ceiling.
CLUSTER_DESIGN_EFFECT = 2.0

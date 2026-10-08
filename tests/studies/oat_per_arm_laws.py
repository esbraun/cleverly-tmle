r"""The declared laws of the per-arm outcome-adaptive C-TMLE study, ``ctmle-oat-per-arm``.

Fixed before any registered run.  ``W = (W1, W2, W3)`` is iid ``N(0, 1)``.  The outcome is
binary with

.. math::

    \operatorname{logit} \bar Q_0(a, W) = c_a + u_{a1} W_1 + u_{a2} W_2,

so each arm reads its own combination of ``(W1, W2)``: an ``A x W`` interaction on the logit
scale.  The treatment law is a softmax with arm 0 the baseline,
``score_a = v_{a1} W_1 + v_{a3} W_3``.  ``W3`` enters only the treatment law, so it is an
instrument.  The three conditions of the X17 design hold on both laws:

* each arm has its own outcome index, so the per-arm design ``Qbar_0(a, W)`` and the shared
  design ``[Qbar_0(a, W): a]`` have different limits;
* ``g_0`` reads ``W3``, which no ``Qbar_0(a, .)`` reads, so the per-arm projection
  ``P(A = a | Qbar_0(a, W))`` is not ``g_0``;
* the projection lies inside the declared ``G_BOUNDS = (0.01, 0.99)``.

The law-level numbers (a ``10^6`` Monte Carlo with 40-node Gauss-Hermite projections) are in
``docs/technical-reference/method-evidence/outcome-adaptive-per-arm-c-tmle.md``.

Every truth is a quadrature.  An arm mean is a one-dimensional Gauss-Hermite integral over
the normal index ``u_a . (W1, W2)``.  ``ey_obs`` is a Gauss-Hermite integral over
``(W1, W2, W3)``.  A weight-tilted truth splits ``W1`` at zero and integrates each half line
by Gauss-Legendre.

The treatment learner of every cell is :func:`cubic_logit_learner`: an unpenalized logistic
regression on ``(l, l^2, l^3)`` with ``l = logit(q)``, ``q`` clipped to
``[1e-6, 1 - 1e-6]``.  The ``drtmle`` runner fits the same model with ``glm``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

#: The treatment bounds every cell declares.  The arm-2 projection of the three-arm law
#: reaches 0.0255, so the conventional ``(0.025, 0.975)`` would nearly bind on it.
G_BOUNDS = (0.01, 0.99)

#: Where the learner basis clips ``q`` before the logit.  The ``drtmle`` runner uses the same.
LOGIT_CLIP = 1e-6

#: Quadrature sizes.
HERMITE_NODES = 80
JOINT_HERMITE_NODES = 40
LEGENDRE_NODES = 200
#: The upper end of each half line of ``W1``, in standard deviations.
HALF_LINE = 9.0

#: The declared known weight of the ``weighted`` cell: ``w(W) = 0.5 + 1{W1 > 0}``.
WEIGHT_BASE = 0.5


def logit_basis(q: Any) -> np.ndarray:
    """``(l, l^2, l^3)`` with ``l = logit(clip(q, 1e-6, 1 - 1e-6))``, one row per prediction.

    Refuses a design of more than one column: the per-arm design is one prediction.
    """
    values = np.asarray(q, dtype=float)
    if values.ndim == 2 and values.shape[1] != 1:
        raise ValueError(
            f"the cubic logit basis reads one prediction column; got {values.shape[1]}"
        )
    values = values.reshape(-1)
    lq = logit(np.clip(values, LOGIT_CLIP, 1.0 - LOGIT_CLIP))
    return np.column_stack([lq, lq**2, lq**3])


def cubic_logit_learner() -> Pipeline:
    """The treatment learner of every cell: a cubic in ``logit q``, unpenalized.

    ``C=np.inf`` is the unpenalized fit; sklearn 1.8 deprecates ``penalty=None``.  The
    Newton-Cholesky solver at a tight tolerance converges to the maximum-likelihood fit that
    R's ``glm`` reaches, which the paired ``drtmle`` cell needs.

    Returns
    -------
    Pipeline
        The basis transform, then the logistic regression.
    """
    return Pipeline(
        [
            ("basis", FunctionTransformer(logit_basis)),
            (
                "model",
                LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000),
            ),
        ]
    )


def main_terms_outcome() -> LogisticRegression:
    """An unpenalized logistic regression on the main terms of the outcome design.

    Misspecified on both laws, because it has no ``A x W`` term.  It is the outcome model
    of the ``robustness_contract/outcome_wrong`` control.
    """
    return LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000)


def arm_interactions(design: Any, *, indicators: int) -> np.ndarray:
    """``[I, W, I x W]`` from an outcome design whose first ``indicators`` columns are arms.

    The binary treatment design is ``[A, W]`` and the ``K``-arm design is ``K - 1``
    drop-first indicators followed by ``W``, as
    :meth:`~cleverly.data.CausalData.treatment_design` builds them.
    """
    matrix = np.asarray(design, dtype=float)
    arms = matrix[:, :indicators]
    covariates = matrix[:, indicators:]
    products = (arms[:, :, None] * covariates[:, None, :]).reshape(matrix.shape[0], -1)
    return np.column_stack([arms, covariates, products])


def interaction_outcome(k: int) -> Pipeline:
    """The correctly specified outcome model: an unpenalized logistic GLM on ``[I, W, I x W]``.

    Parameters
    ----------
    k : int
        The number of arms.

    Returns
    -------
    Pipeline
        The interaction transform, then the logistic regression.
    """
    return Pipeline(
        [
            (
                "interactions",
                FunctionTransformer(arm_interactions, kw_args={"indicators": k - 1}),
            ),
            ("model", main_terms_outcome()),
        ]
    )


@dataclass(frozen=True)
class OatLaw:
    """One law of the study.

    Parameters
    ----------
    name : str
        The law's name.
    c : tuple of float
        The outcome intercept of each arm.
    u : tuple of tuple of float
        The ``(W1, W2)`` outcome coefficients of each arm.
    v : tuple of tuple of float
        The ``(W1, W3)`` treatment scores of each arm; arm 0 is ``(0, 0)``.
    weighted : bool
        Whether the truth is under the weight-tilted law of :data:`WEIGHT_BASE`, and the
        sample carries the weight column ``wt``.
    labels : tuple of str
        The treatment labels, in arm order.  The first is the reference.
    """

    name: str
    c: tuple[float, ...]
    u: tuple[tuple[float, float], ...]
    v: tuple[tuple[float, float], ...]
    weighted: bool = False
    labels: tuple[str, ...] = field(default=())

    @property
    def k(self) -> int:
        """The number of arms."""
        return len(self.c)

    @property
    def arm_labels(self) -> tuple[str, ...]:
        """The treatment labels, defaulting to the arm indices as text."""
        return self.labels or tuple(str(a) for a in range(self.k))

    def outcome_probability(self, arm: int, w1: Any, w2: Any) -> np.ndarray:
        """``Qbar_0(arm, W)``."""
        u = self.u[arm]
        return np.asarray(expit(self.c[arm] + u[0] * np.asarray(w1) + u[1] * np.asarray(w2)))

    def mechanism(self, w1: Any, w3: Any) -> np.ndarray:
        """``g_0(a | W)``, shape ``(..., K)``, by a softmax with arm 0 the baseline."""
        w1 = np.asarray(w1, dtype=float)
        w3 = np.asarray(w3, dtype=float)
        v = np.asarray(self.v, dtype=float)
        scores = w1[..., None] * v[:, 0] + w3[..., None] * v[:, 1]
        scores = scores - scores.max(axis=-1, keepdims=True)
        expo = np.exp(scores)
        return np.asarray(expo / expo.sum(axis=-1, keepdims=True))

    def sample(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        """``n`` iid rows and the law's truth.

        Parameters
        ----------
        n : int
            Rows.
        seed : int
            The draw's seed.

        Returns
        -------
        tuple of DataFrame and dict
            Columns ``W1, W2, W3, A, Y`` (and ``wt`` on a weighted law), and the truth.
        """
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((n, 3))
        probabilities = self.mechanism(w[:, 0], w[:, 2])
        cumulative = probabilities.cumsum(axis=1)
        draw = rng.random(n)
        arm = np.minimum((draw[:, None] > cumulative).sum(axis=1), self.k - 1)
        q = np.column_stack([self.outcome_probability(a, w[:, 0], w[:, 1]) for a in range(self.k)])
        y = (rng.random(n) < q[np.arange(n), arm]).astype(float)
        frame = pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2]})
        if self.k == 2:
            frame["A"] = arm.astype(float)
        else:
            frame["A"] = np.asarray(self.arm_labels, dtype=object)[arm]
        frame["Y"] = y
        if self.weighted:
            frame["wt"] = weight(frame["W1"].to_numpy())
        return frame, self.truth()

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        """The coverage-study form of :meth:`sample`."""
        return self.sample(n, seed)

    def arm_means(self) -> np.ndarray:
        """``E[Y(a)]`` for each arm, under the weight-tilted law when :attr:`weighted`."""
        if self.weighted:
            w1, w2, mass = _half_line_grid()
            share = weight(w1)
            total = float(np.sum(mass * share))
            return np.array(
                [
                    float(np.sum(mass * share * self.outcome_probability(a, w1, w2)) / total)
                    for a in range(self.k)
                ]
            )
        nodes, weights = hermegauss(HERMITE_NODES)
        weights = weights / weights.sum()
        means = []
        for a in range(self.k):
            scale = float(np.hypot(*self.u[a]))
            means.append(float(np.sum(weights * expit(self.c[a] + scale * nodes))))
        return np.array(means)

    def observed_mean(self) -> float:
        """``E[Y] = E[sum_a g_0(a | W) Qbar_0(a, W)]``, under the tilted law when weighted."""
        nodes, weights = hermegauss(JOINT_HERMITE_NODES)
        weights = weights / weights.sum()
        w1, w2, w3 = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        mass = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        if self.weighted:
            # The tilt reads W1 only, so the W1 axis moves to the half-line grid.
            h1, h2, half = _half_line_grid()
            g = self.mechanism(h1[..., None], nodes[None, None, :])
            q = np.stack([self.outcome_probability(a, h1, h2) for a in range(self.k)], axis=-1)
            share = weight(h1)
            joint = np.einsum("ijka,ija->ijk", g, q)
            tilted = half[..., None] * share[..., None] * weights[None, None, :]
            return float(np.sum(tilted * joint) / np.sum(half * share))
        g = self.mechanism(w1, w3)
        q = np.stack([self.outcome_probability(a, w1, w2) for a in range(self.k)], axis=-1)
        return float(np.sum(mass[..., None] * g * q))

    def truth(self) -> dict[str, float]:
        """Every parameter the study reports, named as the fit names it.

        Returns
        -------
        dict
            The arm means, the contrasts against arm 0, and ``ey_obs``, ``par`` and ``paf``.
        """
        means = self.arm_means()
        observed = self.observed_mean()
        labels = self.arm_labels
        out: dict[str, float] = {}
        if self.k == 2:
            out.update(
                {
                    "ey0": float(means[0]),
                    "ey1": float(means[1]),
                    "ate": float(means[1] - means[0]),
                    "rr": float(means[1] / means[0]),
                    "or": float(_odds(means[1]) / _odds(means[0])),
                    "ey_obs": observed,
                    "par": float(observed - means[0]),
                    "paf": float(1.0 - means[0] / observed),
                }
            )
            return out
        reference = labels[0]
        for a, label in enumerate(labels):
            out[f"ey[{label}]"] = float(means[a])
        for a, label in enumerate(labels[1:], start=1):
            pair = f"{label} vs {reference}"
            out[f"ate[{pair}]"] = float(means[a] - means[0])
            out[f"rr[{pair}]"] = float(means[a] / means[0])
            out[f"or[{pair}]"] = float(_odds(means[a]) / _odds(means[0]))
        out["ey_obs"] = observed
        out[f"par[{reference}]"] = float(observed - means[0])
        out[f"paf[{reference}]"] = float(1.0 - means[0] / observed)
        return out


def _odds(p: float) -> float:
    return p / (1.0 - p)


def weight(w1: Any) -> np.ndarray:
    """The declared known weight ``w(W) = 0.5 + 1{W1 > 0}``."""
    return WEIGHT_BASE + (np.asarray(w1, dtype=float) > 0.0).astype(float)


def _half_line_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ``(W1, W2)`` grid with ``W1`` split at zero, and the normal mass of each node.

    ``W1`` takes Gauss-Legendre nodes on ``[-9, 0]`` and on ``[0, 9]``, each weighted by the
    standard normal density; ``W2`` takes Gauss-Hermite nodes.  The weight jumps at
    ``W1 = 0``, so no node straddles the jump.
    """
    x, wx = leggauss(LEGENDRE_NODES)
    half = HALF_LINE / 2.0
    positive = half * (x + 1.0)
    w1 = np.concatenate([-positive[::-1], positive])
    w1_mass = np.concatenate([wx[::-1], wx]) * half * np.exp(-0.5 * w1**2) / np.sqrt(2.0 * np.pi)
    nodes, weights = hermegauss(JOINT_HERMITE_NODES)
    weights = weights / weights.sum()
    grid1, grid2 = np.meshgrid(w1, nodes, indexing="ij")
    mass = w1_mass[:, None] * weights[None, :]
    return grid1, grid2, mass


BINARY_ACTIVE = OatLaw(
    name="binary_active",
    c=(-0.3, 0.3),
    u=((0.8, -0.8), (0.8, 0.8)),
    v=((0.0, 0.0), (0.5, 1.6)),
)

#: The sharp-null twin of :data:`BINARY_ACTIVE`: arm 1 reads arm 0's index, so ``ate = 0``.
#: The treatment law is unchanged.
BINARY_NULL = OatLaw(
    name="binary_sharp_null",
    c=(-0.3, -0.3),
    u=((0.8, -0.8), (0.8, -0.8)),
    v=((0.0, 0.0), (0.5, 1.6)),
)

#: :data:`BINARY_ACTIVE` under the declared known weight.
BINARY_WEIGHTED = OatLaw(
    name="binary_active_weighted",
    c=BINARY_ACTIVE.c,
    u=BINARY_ACTIVE.u,
    v=BINARY_ACTIVE.v,
    weighted=True,
)

THREE_ARM_ACTIVE = OatLaw(
    name="three_arm_active",
    c=(-0.4, 0.0, 0.4),
    u=((0.7, 0.7), (0.7, -0.7), (-0.8, 0.5)),
    v=((0.0, 0.0), (0.9, 1.0), (-0.3, -1.0)),
)

LAWS: Mapping[str, OatLaw] = {
    law.name: law for law in (BINARY_ACTIVE, BINARY_NULL, BINARY_WEIGHTED, THREE_ARM_ACTIVE)
}

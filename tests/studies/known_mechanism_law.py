"""The declared laws of the known-treatment-mechanism studies.

Fixed before any registered run.

**Point treatment.**  ``W1 ~ N(0, 1)``, ``W2 ~ Bernoulli(0.5)`` (the randomization stratum)
and ``W3 ~ U(-1, 1)``.  The binary mechanism is ``g0 = 0.25`` at ``W2 = 0`` and ``0.70`` at
``W2 = 1``.  The three-arm mechanism is ``(0.2, 0.3, 0.5)`` and ``(0.4, 0.4, 0.2)`` by ``W2``.
The outcome is binary with

.. math::

    \\operatorname{logit} \\bar Q_0(a, W) = -0.5 + b_a + 0.9 W_1 - 0.6 W_1^2 + 0.5 W_2
        + c_a W_2 + d_a W_3

with ``b = (0, 0.8, 0.4)``, ``c = (0, 0.8, -0.5)`` and ``d = (0, 0.7, 0.4)``.  Arms 0 and 1
are the two-arm law.  Every truth is a quadrature: Gauss-Hermite with 200 nodes on ``W1``,
Gauss-Legendre with 60 nodes on ``W3``, summed over ``W2``.

The **wrong outcome regression** is the main-terms logistic regression on ``(A, W1, W3)``.  It
omits the randomization stratum ``W2``, ``W1^2`` and both interactions.  The plan named the
main terms in ``(A, W1, W2, W3)``.  Its population bias under an intercept-only mechanism is
0.0039, about 0.8 times the accuracy margin at ``n = 2000``, and raising the allocation
contrast to ``(0.2, 0.8)`` makes it zero.  Without ``W2`` the bias is 0.0790
(:data:`CONTROL_BIAS`), 15.6 times the margin of a quarter of the control's spread
(:func:`control_sd`), so the control can fail.  The correct outcome regression is the
logistic regression on the true terms.  The correct estimated mechanism is the logistic
regression on ``W2``, and the wrong one is intercept-only.

**Sequential randomization.**  A two-node SMART: ``L0 ~ Bernoulli(0.5)``,
``A1 ~ Bernoulli(0.5)``, censoring after node one with retention 0.9 at ``L0 = 0`` and 0.8
at ``L0 = 1``, ``L1 ~ Bernoulli(expit(-0.4 + 0.8 L0 + 0.6 A1))``, ``A2 ~ Bernoulli(0.3)`` at
``L1 = 0`` and ``Bernoulli(0.6)`` at ``L1 = 1``, and a binary ``Y`` with
``logit P(Y = 1) = -0.8 + 0.5 L0 + 0.7 A1 + 0.9 L1 + 0.6 A2 + 0.8 A1 A2 - 0.7 L1 A2``.  The
truth is an exact enumeration.  The wrong outcome regressions keep the main terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit

#: ``P(A = 1 | W2)`` of the two-arm law, at ``W2 = 0`` and ``W2 = 1``.
BINARY_MECHANISM = (0.25, 0.70)
#: ``P(A = a | W2)`` of the three-arm law, arms in code order, at ``W2 = 0`` and ``W2 = 1``.
THREE_ARM_MECHANISM = ((0.2, 0.3, 0.5), (0.4, 0.4, 0.2))
#: The arm coefficients of the outcome law.
ARM_MAIN = (0.0, 0.8, 0.4)
ARM_BY_W2 = (0.0, 0.8, -0.5)
ARM_BY_W3 = (0.0, 0.7, 0.4)
#: The odds multiplier of the incremental cell.
DELTA = 2.0
#: The population bias of the intercept-only mechanism with the wrong outcome regression, by
#: the quadrature of :func:`control_bias` (recorded before any run).
CONTROL_BIAS = 0.0790275
#: Quadrature sizes.
HERMITE_NODES = 200
LEGENDRE_NODES = 60


def outcome_probability(arm: Any, w1: Any, w2: Any, w3: Any) -> Any:
    """``Qbar0(a, W)`` of the point law, at any arm code 0, 1 or 2."""
    a = np.asarray(arm, dtype=int)
    main = np.asarray(ARM_MAIN)[a]
    by_w2 = np.asarray(ARM_BY_W2)[a]
    by_w3 = np.asarray(ARM_BY_W3)[a]
    w1 = np.asarray(w1, dtype=float)
    return expit(
        -0.5 + main + 0.9 * w1 - 0.6 * w1**2 + 0.5 * np.asarray(w2) + by_w2 * w2 + by_w3 * w3
    )


def mechanism(w2: Any, arms: int = 2) -> np.ndarray:
    """``(n, K)`` true mechanism by ``W2``, arms in code order."""
    stratum = np.asarray(w2, dtype=float).astype(int)
    if arms == 2:
        one = np.asarray(BINARY_MECHANISM)[stratum]
        return np.column_stack([1.0 - one, one])
    return np.asarray(THREE_ARM_MECHANISM, dtype=float)[stratum]


def sample(n: int, seed: int, *, arms: int = 2) -> pd.DataFrame:
    """One draw of the point law, with the declared probability columns ``p0``, ``p1`` (``p2``)."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    w3 = rng.uniform(-1.0, 1.0, size=n)
    g = mechanism(w2, arms)
    a = (rng.uniform(size=n)[:, None] > np.cumsum(g, axis=1)).sum(axis=1).astype(float)
    y = rng.binomial(1, outcome_probability(a.astype(int), w1, w2, w3)).astype(float)
    frame = pd.DataFrame({"W1": w1, "W2": w2, "W3": w3, "A": a, "Y": y})
    for code in range(arms):
        frame[f"p{code}"] = g[:, code]
    return frame


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, wx = hermegauss(HERMITE_NODES)
    wx = wx / wx.sum()
    u, wu = leggauss(LEGENDRE_NODES)
    wu = wu / wu.sum()
    w1, w3 = np.meshgrid(x, u, indexing="ij")
    mass = np.outer(wx, wu).ravel()
    w1, w3 = w1.ravel(), w3.ravel()
    return (
        np.concatenate([w1, w1]),
        np.concatenate([np.zeros_like(w1), np.ones_like(w1)]),
        np.concatenate([w3, w3]),
        np.concatenate([0.5 * mass, 0.5 * mass]),
    )


def truth(arms: int = 2) -> dict[str, float]:
    """Every parameter the point studies report, by quadrature.

    ``ey[a]`` and ``ate[a vs 0]`` at every arm count, and at two arms also ``ey0``, ``ey1``,
    ``ate``, ``rr``, ``or``, ``att``, ``atc`` and the incremental mean at :data:`DELTA`.
    """
    w1, w2, w3, mass = _grid()
    g = mechanism(w2, arms)
    q = np.column_stack([outcome_probability(arm, w1, w2, w3) for arm in range(arms)])
    means = mass @ q
    out = {f"ey[{float(arm)}]": float(means[arm]) for arm in range(arms)}
    for arm in range(1, arms):
        out[f"ate[{float(arm)} vs 0.0]"] = float(means[arm] - means[0])
    if arms == 2:
        g1 = g[:, 1]
        blip = q[:, 1] - q[:, 0]
        out.update(
            ey0=float(means[0]),
            ey1=float(means[1]),
            ate=float(means[1] - means[0]),
            rr=float(means[1] / means[0]),
            or_=float((means[1] / (1 - means[1])) / (means[0] / (1 - means[0]))),
            att=float(np.sum(mass * g1 * blip) / np.sum(mass * g1)),
            atc=float(np.sum(mass * (1 - g1) * blip) / np.sum(mass * (1 - g1))),
        )
        out["or"] = out.pop("or_")
        tilted = DELTA * g1 / (DELTA * g1 + 1 - g1)
        out["ey_ipsi"] = float(mass @ (tilted * q[:, 1] + (1 - tilted) * q[:, 0]))
    return out


def ate_efficiency_sd(arms: int = 2, arm: int = 1) -> float:
    """The exact standard deviation of the efficient curve of ``ate[arm vs 0]``.

    The bound is the same whether the mechanism is known or estimated (Hahn 1998, Theorems 1
    and 2 compared), so it serves the known-mechanism cells too.
    """
    w1, w2, w3, mass = _grid()
    g = mechanism(w2, arms)
    q1 = outcome_probability(arm, w1, w2, w3)
    q0 = outcome_probability(0, w1, w2, w3)
    psi = float(mass @ (q1 - q0))
    variance = mass @ (q1 * (1 - q1) / g[:, arm] + q0 * (1 - q0) / g[:, 0] + (q1 - q0 - psi) ** 2)
    return float(np.sqrt(variance))


def wrong_outcome_known_sd() -> float:
    """The exact SD of ``D*(Qbar_inf, g0)``: the known-mechanism ATE at the wrong regression's limit.

    The known-mechanism TMLE is consistent here, and this is its influence curve's spread.  It
    exceeds :func:`ate_efficiency_sd`, because the outcome regression is wrong.
    """
    w1, w2, w3, mass = _grid()
    g1 = mechanism(w2, 2)[:, 1]
    beta = _wrong_outcome_limit()
    one = expit(np.column_stack([np.ones_like(w1), np.ones_like(w1), w1, w3]) @ beta)
    zero = expit(np.column_stack([np.ones_like(w1), np.zeros_like(w1), w1, w3]) @ beta)
    q1 = outcome_probability(1, w1, w2, w3)
    q0 = outcome_probability(0, w1, w2, w3)
    psi = truth(2)["ate"]
    blip = one - zero - psi
    variance = mass @ (
        (q1 * (1 - q1) + (q1 - one) ** 2) / g1
        + (q0 * (1 - q0) + (q0 - zero) ** 2) / (1 - g1)
        + blip**2
        + 2 * blip * ((q1 - one) - (q0 - zero))
    )
    return float(np.sqrt(variance))


def _wrong_outcome_limit() -> np.ndarray:
    """The population main-terms logistic fit on ``(1, A, W1, W3)``, by weighted maximum likelihood."""
    from scipy.optimize import minimize

    w1, w2, w3, mass = _grid()
    g1 = mechanism(w2, 2)[:, 1]
    designs, weights, labels = [], [], []
    for arm in (0.0, 1.0):
        share = g1 if arm else 1.0 - g1
        q = outcome_probability(int(arm), w1, w2, w3)
        design = np.column_stack([np.ones_like(w1), np.full_like(w1, arm), w1, w3])
        designs += [design, design]
        weights += [mass * share * q, mass * share * (1 - q)]
        labels += [np.ones_like(w1), np.zeros_like(w1)]
    x = np.vstack(designs)
    y = np.concatenate(labels)
    w = np.concatenate(weights)

    def loss(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(-np.sum(w * (y * eta - np.logaddexp(0.0, eta))))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return -(x.T @ (w * (y - expit(x @ beta))))

    return minimize(
        loss, np.zeros(x.shape[1]), jac=gradient, method="BFGS", options={"gtol": 1e-12}
    ).x


def control_bias() -> float:
    """The population bias of the intercept-only mechanism with the wrong outcome regression.

    The intercept-only mechanism converges to the marginal arm share, so the clever covariate
    of each arm is a constant and the fluctuation of a logistic outcome regression with an arm
    intercept does not move.  The limit is the plug-in of the population main-terms fit,
    which this function computes as a weighted logistic maximum likelihood on the quadrature
    grid.
    """
    w1, _, w3, mass = _grid()
    beta = _wrong_outcome_limit()
    one = np.column_stack([np.ones_like(w1), np.ones_like(w1), w1, w3])
    zero = np.column_stack([np.ones_like(w1), np.zeros_like(w1), w1, w3])
    plug_in = float(mass @ (expit(one @ beta) - expit(zero @ beta)))
    return plug_in - truth(2)["ate"]


def control_sd() -> float:
    """The SD of the intercept-only-mechanism estimator's curve at its limit.

    The mechanism converges to the marginal share ``p = P(A = 1)``, so the curve is
    ``A / p (Y - Qbar_1) - (1 - A) / (1 - p) (Y - Qbar_0) + Qbar_1 - Qbar_0 - psi`` at the wrong
    outcome regression's limit.  The accuracy margin of the control cell is a quarter of its
    spread, which this sizes before any run.
    """
    w1, w2, w3, mass = _grid()
    g1 = mechanism(w2, 2)[:, 1]
    share = float(mass @ g1)
    beta = _wrong_outcome_limit()
    one = expit(np.column_stack([np.ones_like(w1), np.ones_like(w1), w1, w3]) @ beta)
    zero = expit(np.column_stack([np.ones_like(w1), np.zeros_like(w1), w1, w3]) @ beta)
    q1 = outcome_probability(1, w1, w2, w3)
    q0 = outcome_probability(0, w1, w2, w3)
    psi = float(mass @ (one - zero))
    blip = one - zero - psi
    variance = mass @ (
        g1 * (q1 * (1 - q1) + (q1 - one) ** 2) / share**2
        + (1 - g1) * (q0 * (1 - q0) + (q0 - zero) ** 2) / (1 - share) ** 2
        + blip**2
        + 2 * blip * (g1 * (q1 - one) / share - (1 - g1) * (q0 - zero) / (1 - share))
    )
    return float(np.sqrt(variance))


# ----------------------------------------------------------------------------- learners


def outcome_features(design: Any, kind: str, arms: int = 2) -> np.ndarray:
    """The outcome regression's features from the ``[arm block, W1, W2, W3]`` design.

    ``kind="correct"`` gives the true terms and ``kind="wrong"`` the main terms in
    ``(A, W1, W3)``.  The arm block is one column at two arms and two drop-first indicators at
    three.
    """
    values = np.asarray(design, dtype=float)
    width = 1 if arms == 2 else 2
    block = values[:, :width]
    w1, w2, w3 = values[:, width], values[:, width + 1], values[:, width + 2]
    if kind == "wrong":
        return np.column_stack([block, w1, w3])
    if kind != "correct":
        raise ValueError(kind)
    return np.column_stack([block, w1, w1**2, w2, block * w2[:, None], block * w3[:, None]])


def treatment_features(design: Any, kind: str) -> np.ndarray:
    """``W2`` for the correct mechanism, nothing for the intercept-only one."""
    values = np.asarray(design, dtype=float)
    if kind == "correct":
        return values[:, [1]]
    if kind == "intercept":
        return np.zeros((values.shape[0], 1))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- SMART law

#: ``P(C1 = 1 | L0)`` at ``L0 = 0`` and ``L0 = 1``.
SMART_RETENTION = (0.9, 0.8)
#: ``P(A2 = 1 | L1)`` at ``L1 = 0`` and ``L1 = 1``.
SMART_SECOND_STAGE = (0.3, 0.6)
#: ``P(A1 = 1)``.
SMART_FIRST_STAGE = 0.5


def smart_l1(l0: Any, a1: Any) -> Any:
    """``P(L1 = 1 | L0, A1)``."""
    return expit(-0.4 + 0.8 * np.asarray(l0) + 0.6 * np.asarray(a1))


def smart_outcome(l0: Any, a1: Any, l1: Any, a2: Any) -> Any:
    """``P(Y = 1 | L0, A1, L1, A2)``."""
    return expit(-0.8 + 0.5 * l0 + 0.7 * a1 + 0.9 * l1 + 0.6 * a2 + 0.8 * a1 * a2 - 0.7 * l1 * a2)


def smart_sample(n: int, seed: int) -> pd.DataFrame:
    """One draw of the SMART law, with the declared probability columns.

    ``g1_0``, ``g1_1``, ``g2_0``, ``g2_1`` are the node mechanisms and ``r1``, ``r2`` the
    retention probabilities.  Rows censored after node one are missing from ``L1`` on, and
    ``C2`` is one for every row still observed.
    """
    rng = np.random.default_rng(seed)
    l0 = rng.binomial(1, 0.5, size=n).astype(float)
    a1 = rng.binomial(1, SMART_FIRST_STAGE, size=n).astype(float)
    retention = np.asarray(SMART_RETENTION)[l0.astype(int)]
    c1 = rng.binomial(1, retention).astype(float)
    l1 = rng.binomial(1, smart_l1(l0, a1)).astype(float)
    g2 = np.asarray(SMART_SECOND_STAGE)[l1.astype(int)]
    a2 = rng.binomial(1, g2).astype(float)
    y = rng.binomial(1, smart_outcome(l0, a1, l1, a2)).astype(float)
    alive = c1 == 1.0
    return pd.DataFrame(
        {
            "L0": l0,
            "A1": a1,
            "C1": c1,
            "L1": np.where(alive, l1, np.nan),
            "A2": np.where(alive, a2, np.nan),
            "C2": np.where(alive, 1.0, np.nan),
            "Y": np.where(alive, y, np.nan),
            "g1_0": 1.0 - SMART_FIRST_STAGE,
            "g1_1": SMART_FIRST_STAGE,
            "g2_0": np.where(alive, 1.0 - g2, np.nan),
            "g2_1": np.where(alive, g2, np.nan),
            "r1": retention,
            "r2": np.where(alive, 1.0, np.nan),
        }
    )


def smart_truth(plans: dict[str, Sequence[int]]) -> dict[str, float]:
    """``E[Y^{a1, a2}]`` of each static plan, by exact enumeration."""
    out = {}
    for label, (a1, a2) in plans.items():
        total = 0.0
        for l0 in (0, 1):
            for l1 in (0, 1):
                p_l1 = smart_l1(l0, a1) if l1 else 1.0 - smart_l1(l0, a1)
                total += 0.5 * float(p_l1) * float(smart_outcome(l0, a1, l1, a2))
        out[label] = total
    return out

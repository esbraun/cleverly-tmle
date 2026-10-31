r"""The finite-support laws of the X82 SG coverage cells, with their exact bound inputs.

The O cells of ``ovb-limit-coverage`` read one baseline law, the law of
``tests/unit/test_omitted_variable_part_g.py``, in three versions:

=========  ===========================================================  ======================
version    support                                                      cells
=========  ===========================================================  ======================
complete   ``(w, a, y)``                                                O3 regime, O4 MSM
response   ``(w, a, delta, y)``, ``Y`` missing at random given ``A, W``  O1
mediated   ``(w, a, z, y)``, complete outcome                           O2
=========  ===========================================================  ======================

O5 reads :mod:`tests.discrete_law_shift`.  Each law is a table of cell probabilities, so
:func:`exact_frame` *is* the law (one row per cell, weighted by its probability), and
:func:`sample` draws ``n`` rows from it.  The closed forms below give ``psi``, ``sigma^2`` and
``nu^2`` of every primary parameter from the tables alone: no fitted nuisance enters them.

The covariate enters each fit as two indicators, ``W1`` and ``W2``, so a logistic model with
every pairwise interaction is saturated in each nuisance (:func:`saturated_classifier`).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import tests.discrete_law_shift as shift_law

#: ``P(W = w)``.
P_W = np.array([0.5, 0.3, 0.2])

#: ``P(A = 1 | W = w)``.
G1 = np.array([0.4, 0.6, 0.25])

#: ``P(Delta = 1 | A = a, W = w)``, indexed ``[w, a]``.
PI = np.array([[0.7, 0.9], [0.5, 0.8], [0.85, 0.6]])

#: ``P(Z = 1 | A = a, W = w)``, indexed ``[w, a]``.
QZ = np.array([[0.3, 0.6], [0.5, 0.4], [0.7, 0.2]])

VERSIONS = ("complete", "response", "mediated")


def outcome_risk(a: float, z: float, w: float) -> float:
    """``P(Y = 1 | A = a, Z = z, W = w)``, with an ``A`` by ``W`` interaction."""
    return float(expit(-0.5 + 0.8 * a + 0.6 * z + 0.4 * w - 0.3 * a * w))


def _risk(a: int, w: int, version: str) -> float:
    """``Qbar(a, w)``: the risk given ``A, W``, with ``Z`` integrated out unless it is mediated."""
    if version == "mediated":
        raise ValueError("the mediated version conditions on Z; read outcome_risk directly")
    return outcome_risk(a, 0, w)


def exact_frame(version: str) -> pd.DataFrame:
    """One row per support point of ``version``, weighted by its probability."""
    if version not in VERSIONS:
        raise ValueError(f"unknown version {version!r}; choose from {VERSIONS}")
    rows = []
    zs = range(2) if version == "mediated" else (0,)
    ds = range(2) if version == "response" else (1,)
    for w, a, z, d, y in itertools.product(range(3), range(2), zs, ds, range(2)):
        if d == 0 and y == 1:
            continue
        p = P_W[w] * (G1[w] if a else 1.0 - G1[w])
        if version == "mediated":
            p *= QZ[w, a] if z else 1.0 - QZ[w, a]
        if version == "response":
            p *= PI[w, a] if d else 1.0 - PI[w, a]
        if d:
            risk = outcome_risk(a, z, w)
            p *= risk if y else 1.0 - risk
        row = {"W1": float(w == 1), "W2": float(w == 2), "A": float(a)}
        if version == "mediated":
            row["Z"] = float(z)
        if version == "response":
            row["D"] = float(d)
        row["Y"] = float(y) if d else np.nan
        row["weight"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def shift_exact_frame() -> pd.DataFrame:
    """The law of :mod:`tests.discrete_law_shift`, one weighted row per cell."""
    rows = []
    for w, a, y in itertools.product(range(3), range(len(shift_law.DOSES)), range(2)):
        risk = shift_law.Q[w, a]
        p = shift_law.P_W[w] * shift_law.G[w, a] * (risk if y else 1.0 - risk)
        rows.append(
            {
                "W1": float(w == 1),
                "W2": float(w == 2),
                "A": float(shift_law.DOSES[a]),
                "Y": float(y),
                "weight": p,
            }
        )
    return pd.DataFrame(rows)


def sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """``n`` independent rows of the law that ``frame`` weights."""
    weights = frame["weight"].to_numpy(dtype=float)
    index = rng.choice(len(frame), size=n, p=weights / weights.sum())
    return frame.iloc[index].drop(columns="weight").reset_index(drop=True)


def saturated_classifier() -> Pipeline:
    """A logistic model on every pairwise product of its inputs, practically unpenalized.

    Given ``A`` (or a dose's atom indicators), ``Z`` and the two ``W`` indicators, the
    pairwise products hold every ``A`` by ``W`` cell, so the model is saturated in the cells
    each nuisance of these laws reads.
    """
    return make_pipeline(
        PolynomialFeatures(2, interaction_only=True, include_bias=False),
        LogisticRegression(C=1e8, max_iter=10_000),
    )


# ------------------------------------------------------------------- closed forms


def _w_mass() -> np.ndarray:
    return P_W


def _g(a: int) -> np.ndarray:
    return G1 if a else 1.0 - G1


def arm_mean(a: int) -> float:
    """``E[Qbar(a, W)]`` on the complete and response versions."""
    return float(sum(P_W[w] * _risk(a, w, "complete") for w in range(3)))


def sigma2(version: str) -> float:
    """``E[R (Y - Qbar)^2]``, with ``R = Delta`` on the response version and 1 otherwise."""
    total = 0.0
    for w, a in itertools.product(range(3), range(2)):
        mass = P_W[w] * _g(a)[w]
        if version == "mediated":
            for z in range(2):
                pz = QZ[w, a] if z else 1.0 - QZ[w, a]
                risk = outcome_risk(a, z, w)
                total += mass * pz * risk * (1.0 - risk)
            continue
        risk = _risk(a, w, "complete")
        observed = PI[w, a] if version == "response" else 1.0
        total += mass * observed * risk * (1.0 - risk)
    return float(total)


def mediated_sigma2(z: int) -> float:
    """``E[1{Z = z} (Y - Qbar(A, z, W))^2]``: the residual on the rows the CDE at ``z`` reads."""
    total = 0.0
    for w, a in itertools.product(range(3), range(2)):
        pz = QZ[w, a] if z else 1.0 - QZ[w, a]
        risk = outcome_risk(a, z, w)
        total += P_W[w] * _g(a)[w] * pz * risk * (1.0 - risk)
    return float(total)


def response_nu2(estimand: str) -> float:
    """``E[alpha^2]`` of ``ey1`` or ``ate`` with ``alpha = Delta [A / (g pi_1) - ...]``."""
    one = sum(P_W[w] / (G1[w] * PI[w, 1]) for w in range(3))
    zero = sum(P_W[w] / ((1.0 - G1[w]) * PI[w, 0]) for w in range(3))
    if estimand == "ey1":
        return float(one)
    if estimand == "ate":
        return float(one + zero)
    raise KeyError(estimand)


def mediated_psi(z: int) -> float:
    """The controlled direct effect ``E[Qbar(1, z, W) - Qbar(0, z, W)]``."""
    return float(sum(P_W[w] * (outcome_risk(1, z, w) - outcome_risk(0, z, w)) for w in range(3)))


def mediated_nu2(z: int) -> float:
    """``E[alpha^2]`` with ``alpha = 1{Z = z} [A / (g q_z(1)) - (1 - A) / ((1 - g) q_z(0))]``."""
    total = 0.0
    for w in range(3):
        for a in range(2):
            qz = QZ[w, a] if z else 1.0 - QZ[w, a]
            total += P_W[w] / (_g(a)[w] * qz)
    return float(total)


def regime_psi(weights: np.ndarray) -> float:
    """``E[sum_a c(a, W) Qbar(a, W)]`` for a weight table ``c`` indexed ``[w, a]``."""
    return float(
        sum(P_W[w] * weights[w, a] * _risk(a, w, "complete") for w in range(3) for a in range(2))
    )


def regime_nu2(weights: np.ndarray) -> float:
    """``E[(c(A, W) / g(A | W))^2] = sum_w P(w) sum_a c(a, w)^2 / g(a | w)``."""
    return float(sum(P_W[w] * weights[w, a] ** 2 / _g(a)[w] for w in range(3) for a in range(2)))


def msm_terms(design: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""The MSM coefficient, its Gram matrix, and the per-cell design.

    ``design(a, w)`` returns :math:`\varphi(a, w)`; the weight :math:`h` is one.  The
    coefficient is :math:`M^{-1} E[\sum_a \varphi(a, W) \bar Q(a, W)]` with
    :math:`M = E[\sum_a \varphi \varphi^\top]`.
    """
    phi = np.array([[design(a, w) for a in range(2)] for w in range(3)], dtype=float)
    gram = sum(P_W[w] * np.outer(phi[w, a], phi[w, a]) for w in range(3) for a in range(2))
    target = sum(P_W[w] * phi[w, a] * _risk(a, w, "complete") for w in range(3) for a in range(2))
    beta = np.linalg.solve(gram, target)
    return beta, np.asarray(gram), phi


def msm_nu2(design: Any, coefficient: int) -> float:
    """``E[alpha_j^2]`` with ``alpha_j = e_j' M^{-1} phi(A, W) / g(A | W)``."""
    _, gram, phi = msm_terms(design)
    b = np.linalg.solve(gram, np.eye(len(gram))[coefficient])
    return float(
        sum(P_W[w] * float(phi[w, a] @ b) ** 2 / _g(a)[w] for w in range(3) for a in range(2))
    )


def shift_terms(label: str) -> tuple[float, float, float]:
    """``(psi, sigma^2, nu^2)`` of ``ey_policy[label]`` on the shift law."""
    g = shift_law.G
    q = shift_law.Q
    target = shift_law.POLICIES[label]
    psi = float(
        sum(
            shift_law.P_W[w] * g[w, a] * q[w, target[a]]
            for w in range(3)
            for a in range(len(shift_law.DOSES))
        )
    )
    induced = np.zeros_like(g)
    for source, moved in enumerate(target):
        induced[:, moved] += g[:, source]
    nu2 = float(
        sum(
            shift_law.P_W[w] * induced[w, a] ** 2 / g[w, a]
            for w in range(3)
            for a in range(len(shift_law.DOSES))
        )
    )
    s2 = float(
        sum(
            shift_law.P_W[w] * g[w, a] * q[w, a] * (1.0 - q[w, a])
            for w in range(3)
            for a in range(len(shift_law.DOSES))
        )
    )
    return psi, s2, nu2

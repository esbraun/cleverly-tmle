r"""A variant of :mod:`tests.discrete_law_cde` whose controlled effect is not linear in ``W``.

The shipped CDE law has :math:`\tau_z(w) = \bar Q(1, z, w) - \bar Q(0, z, w)` exactly linear in
``w`` at both levels, so a projection onto ``Linear(W)`` cannot tell averaging the contrast over
every row from averaging it over the ``Z = z`` rows: the plug-in stratum term vanishes.  This
variant sets ``Qbar(1, 1, 0) = 0.75`` and ``Qbar(1, 1, 1) = 0.25`` at ``w = 1``, which breaks
the linearity at both levels and keeps every cell count integral at ``N = 1024``.

Every other constant is the parent's, so ``g``, ``q_z`` and ``pi`` are unchanged and the
oracle learners of :mod:`tests.conftest` read the variant through :class:`DiscreteLaw`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tests import discrete_law_cde as parent

N = parent.N
LEVELS = parent.LEVELS
SUPPORT = parent.SUPPORT
OBSERVED_ZERO, OBSERVED_ONE, UNOBSERVED = (
    parent.OBSERVED_ZERO,
    parent.OBSERVED_ONE,
    parent.UNOBSERVED,
)
P_W, G, QZ, PI = parent.P_W, parent.G, parent.QZ, parent.PI

#: ``Qbar(a, z, w)``, indexed ``[w, a, z]``: the parent's with ``w = 1, a = 1`` changed.
QBAR = parent.QBAR.copy()
QBAR[1, 1, 0] = 0.75
QBAR[1, 1, 1] = 0.25


def _cell_counts(qbar: np.ndarray) -> np.ndarray:
    counts = np.empty((3, 2, 2, 3))
    for w, a, z, k in SUPPORT:
        arm = G[w] if a == 1 else 1.0 - G[w]
        level = QZ[w, a] if z == 1 else 1.0 - QZ[w, a]
        base = P_W[w] * arm * level
        if k == UNOBSERVED:
            counts[w, a, z, k] = base * (1.0 - PI[w, a]) * N
        else:
            outcome = qbar[w, a, z] if k == OBSERVED_ONE else 1.0 - qbar[w, a, z]
            counts[w, a, z, k] = base * PI[w, a] * outcome * N
    rounded = np.rint(counts)
    if np.max(np.abs(counts - rounded)) > 1e-6:  # pragma: no cover - guards the constants
        raise AssertionError("the variant's cell probabilities are not multiples of 1/N")
    return rounded.astype(int)


COUNTS = _cell_counts(QBAR)
PROBS = COUNTS / N


def frame(*, complete: bool = False) -> pd.DataFrame:
    """The ``N``-row sample whose empirical law is the variant, as :func:`parent.frame`.

    With ``complete=True`` the unobserved rows are dropped, which leaves the law of the
    complete rows: the same ``Qbar``, with ``g`` and ``q_z`` reweighted by ``pi``.
    """
    counts = [COUNTS[w, a, z, k] for w, a, z, k in SUPPORT]
    cells = np.repeat(np.arange(len(SUPPORT)), counts)
    columns = np.array(SUPPORT, dtype=float)[cells]
    kind = columns[:, 3]
    out = pd.DataFrame(
        {
            "W": columns[:, 0],
            "A": columns[:, 1],
            "Z": columns[:, 2],
            "Y": np.where(kind == UNOBSERVED, np.nan, kind),
            "Delta": np.where(kind == UNOBSERVED, 0.0, 1.0),
        }
    )
    if complete:
        out = out[out["Delta"] == 1.0].drop(columns="Delta").reset_index(drop=True)
    return out


def complete_probs() -> np.ndarray:
    """The variant's cell probabilities on its complete rows, ``Delta = 0`` cells emptied."""
    p = PROBS.copy()
    p[..., UNOBSERVED] = 0.0
    return p / p.sum()


def law(*, complete: bool = False) -> parent.DiscreteLaw:
    """The variant duck-typed as a ``DGP`` for the oracle learners."""
    return parent.DiscreteLaw(complete_probs() if complete else PROBS)


def _design() -> np.ndarray:
    """``phi(w) = (1, w)`` at ``w = 0, 1, 2``: the ``Linear(W)`` basis."""
    return np.column_stack([np.ones(3), np.arange(3.0)])


def projection(p_w: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """``M^{-1} E[phi tau]`` under the covariate law ``p_w``."""
    phi = _design()
    gram = phi.T @ (p_w[:, None] * phi)
    return np.linalg.solve(gram, phi.T @ (p_w * tau))


def log_ratio_projection(p_w: np.ndarray, q_t: np.ndarray, q_r: np.ndarray) -> np.ndarray:
    """The root of ``E[phi{(Q_t + Q_r) expit(phi' b) - Q_t}] = 0`` under ``p_w``, by Newton."""
    phi = _design()
    beta = np.zeros(2)
    for _ in range(100):
        m = 1.0 / (1.0 + np.exp(-(phi @ beta)))
        total = q_t + q_r
        score = phi.T @ (p_w * (total * m - q_t))
        info = phi.T @ ((p_w * total * m * (1.0 - m))[:, None] * phi)
        step = np.linalg.solve(info, score)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-15:
            break
    return beta


def truth(level: int, scale: str = "difference", probs: Any = None) -> np.ndarray:
    """``beta_z`` under ``Linear(W)``: the projection of the CDE at ``Z = level``."""
    p = PROBS if probs is None else np.asarray(probs)
    dgp = parent.DiscreteLaw(p)
    p_w = p.sum(axis=(1, 2, 3))
    q = dgp.q[:, :, level]
    if scale == "difference":
        return projection(p_w, q[:, 1] - q[:, 0])
    return log_ratio_projection(p_w, q[:, 1], q[:, 0])


def plugin_stratum(level: int, probs: Any = None) -> np.ndarray:
    """The CDE contrast projected under ``P(W | Z = z)``: the plug-in stratum mutation."""
    p = PROBS if probs is None else np.asarray(probs)
    dgp = parent.DiscreteLaw(p)
    stratum = p[:, :, level, :].sum(axis=(1, 2))
    q = dgp.q[:, :, level]
    return projection(stratum / stratum.sum(), q[:, 1] - q[:, 0])


def observed_z_prediction(probs: Any = None) -> np.ndarray:
    """The projection of the contrast predicted at each row's observed intermediate.

    Each row's ``Qbar(1, Z_i, W_i) - Qbar(0, Z_i, W_i)`` at its own recorded ``Z_i``, instead
    of at the fixed level: the counterfactual ``Z = z`` block of the outcome design dropped.
    ``Z_i`` follows its observed law given ``W``, a mixture over the observed arm, so the
    answer is the same at both levels and is neither CDE nor the total effect.
    """
    p = PROBS if probs is None else np.asarray(probs)
    dgp = parent.DiscreteLaw(p)
    p_w = p.sum(axis=(1, 2, 3))
    z_one = p[:, :, 1, :].sum(axis=(1, 2)) / p_w  # P(Z = 1 | W)
    tau = dgp.q[:, 1, :] - dgp.q[:, 0, :]
    return projection(p_w, z_one * tau[:, 1] + (1.0 - z_one) * tau[:, 0])

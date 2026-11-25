r"""The ``cde_obs`` law of the ``projection-evidence`` study and its exact readings (P1 PR 2, 5.1).

The law is the ``conditional-projection`` observational law with a binary intermediate ``Z``
between the treatment and the outcome:

- ``W ~ U(-1, 1)^3`` and ``g(1 | W) = expit(0.4 W1 - 0.5 W2 + 0.3 W3)``
  (:func:`tests.studies.conditional_projection.propensity`);
- ``q_1(a, W) = P(Z = 1 | A = a, W) = expit(-0.2 + 0.8 a + 1.2 W3 - 0.4 W1)``;
- ``logit Qbar(a, z, W) = -0.3 + 0.6 W1 - 0.4 W2 + a blip(W) + z (0.5 - 0.9 a)``, with the
  shipped ``blip(W) = 0.3 + 0.5 W1 + 0.4 sin(pi W3)``;
- complete outcomes, ``V = W1`` and the basis ``Linear(["W1"])``.

The target at level ``z`` is the projection of ``tau_z(W) = Qbar(1, z, W) - Qbar(0, z, W)`` (or
of its log ratio) onto ``(1, W1)`` under the law of ``W``.  The ``z (0.5 - 0.9 a)`` term makes
``tau_z`` depend on ``z``, and the ``1.2 W3`` term in ``q_1`` ties the intermediate to the
nonlinear part of the blip, so averaging over the ``Z = z`` rows moves the ``V = W1`` projection.

Every number of plan 5.1 is computed here by Gauss--Legendre quadrature over ``U(-1, 1)^3``: the
truths, the efficient SDs at ``n``, and the limits of the plug-in stratum, of the total effect,
of the plug-in at the declared wrong ``Qbar``, and of the TMLE at that ``Qbar`` with the full
clever covariate, with ``q_z`` omitted and with the selection ``1{Z = z}`` dropped.  A TMLE limit
solves the population score equation of its covariate, ``E[H (Y - Q*_eps)] = 0``, by Newton's
method on the quadrature nodes, and projects ``Q*_eps(1, z, W) - Q*_eps(0, z, W)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from tests.studies import conditional_projection as cp

#: The modifier and basis of every ``cde_obs`` fit.
TERMS = ("(intercept)", "W1")
LEVELS = (0, 1)
#: The CDE band grid (plan 5.1, N4).
BAND_GRID_W1 = np.linspace(-0.8, 0.8, 5)
#: The quadrature nodes per axis of the design numbers (the plan read 32).
NODES = 32
CRITICAL = float(stats.norm.ppf(0.975))


def intermediate_probability(w: np.ndarray, a: Any) -> np.ndarray:
    """``q_1(a, W) = expit(-0.2 + 0.8 a + 1.2 W3 - 0.4 W1)``."""
    return expit(-0.2 + 0.8 * np.asarray(a, dtype=float) + 1.2 * w[..., 2] - 0.4 * w[..., 0])


def intermediate_density(w: np.ndarray, a: Any, z: int) -> np.ndarray:
    """``q_z(a, W) = P(Z = z | A = a, W)``."""
    q1 = intermediate_probability(w, a)
    return q1 if z == 1 else 1.0 - q1


def outcome_logit(w: np.ndarray, a: Any, z: Any) -> np.ndarray:
    """``logit Qbar(a, z, W)``."""
    a = np.asarray(a, dtype=float)
    z = np.asarray(z, dtype=float)
    return -0.3 + 0.6 * w[..., 0] - 0.4 * w[..., 1] + a * cp.blip(w) + z * (0.5 - 0.9 * a)


def declared_error(w: np.ndarray, a: Any) -> np.ndarray:
    """The declared wrong ``Qbar``'s logit error: ``conditional_projection``'s ``"missing"`` one."""
    return cp.declared_error("missing", w, a)


def sample(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` rows: ``Y``, ``A``, ``Z``, ``W1``, ``W2``, ``W3``."""
    w = rng.uniform(-1.0, 1.0, (n, 3))
    a = rng.binomial(1, cp.propensity(w))
    z = rng.binomial(1, intermediate_probability(w, a))
    y = rng.binomial(1, expit(outcome_logit(w, a, z))).astype(float)
    return pd.DataFrame(
        {"Y": y, "A": a, "Z": z.astype(float), "W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2]}
    )


# ------------------------------------------------------------------------------ quadrature


def grid(nodes: int = NODES) -> tuple[np.ndarray, np.ndarray]:
    """``U(-1, 1)^3`` nodes and probability weights."""
    return cp._grid(nodes)


def design(w: np.ndarray) -> np.ndarray:
    """The basis ``(1, W1)`` at the rows of ``w``."""
    return np.column_stack([np.ones(len(w)), w[:, 0]])


def arm_means(w: np.ndarray, z: int, error: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``(Qbar(1, z, W), Qbar(0, z, W))``, at the declared wrong ``Qbar`` when ``error``."""
    shift = (lambda a: declared_error(w, a)) if error else (lambda a: np.zeros(len(w)))
    return (
        expit(outcome_logit(w, 1, z) + shift(1)),
        expit(outcome_logit(w, 0, z) + shift(0)),
    )


def truth(z: int, scale: str = "difference", *, nodes: int = NODES) -> np.ndarray:
    """The coefficients of the projection at level ``z`` on ``scale``."""
    w, mass = grid(nodes)
    q_t, q_r = arm_means(w, z)
    return cp.projection(design(w), mass, q_t, q_r, scale)


def truth_names(z: int, scale: str = "difference", *, nodes: int = NODES) -> dict[str, float]:
    """:func:`truth` keyed by the names a fit reports."""
    stem = "cate_projection" if scale == "difference" else "log_crr_projection"
    return {
        f"{stem}[{term}]": float(v)
        for term, v in zip(TERMS, truth(z, scale, nodes=nodes), strict=True)
    }


def efficient_covariance(
    z: int, n: int, scale: str = "difference", *, nodes: int = NODES
) -> np.ndarray:
    r"""The efficient covariance at ``n`` of the projection's coefficients at level ``z``.

    On the difference scale the curve is ``M^{-1} phi (tau_z - phi'beta + sum_a s_a 1{A = a}
    1{Z = z} / (g_a q_z(a)) (Y - Qbar(a, z)))``.  On the log ratio it is the logistic
    estimating equation's curve, with Jacobian ``E[phi phi' (Q_t + Q_r) e (1 - e)]`` and the
    pseudo-outcome noise read through ``(1 - e)`` on the treated arm and ``e`` on the reference.
    """
    w, mass = grid(nodes)
    phi = design(w)
    q_t, q_r = arm_means(w, z)
    g1 = cp.propensity(w)
    d_t = g1 * intermediate_density(w, 1, z)
    d_r = (1.0 - g1) * intermediate_density(w, 0, z)
    beta = cp.projection(phi, mass, q_t, q_r, scale)
    if scale == "difference":
        jacobian = phi.T @ (mass[:, None] * phi)
        second = (q_t - q_r - phi @ beta) ** 2 + q_t * (1 - q_t) / d_t + q_r * (1 - q_r) / d_r
    else:
        e = expit(phi @ beta)
        jacobian = phi.T @ ((mass * (q_t + q_r) * e * (1 - e))[:, None] * phi)
        plug = (q_t + q_r) * e - q_t
        second = plug**2 + (1 - e) ** 2 * q_t * (1 - q_t) / d_t + e**2 * q_r * (1 - q_r) / d_r
    middle = phi.T @ ((mass * second)[:, None] * phi)
    inverse = np.linalg.inv(jacobian)
    return inverse @ middle @ inverse / n


def efficient_sd(z: int, n: int, scale: str = "difference") -> np.ndarray:
    """The efficient SDs of the coefficients at level ``z`` and ``n`` rows."""
    return np.sqrt(np.diag(efficient_covariance(z, n, scale)))


def plugin_stratum_limit(z: int, *, nodes: int = NODES) -> np.ndarray:
    """The limit of ``M_z^{-1} P_n[1{Z = z} phi (Q_t - Q_r)]``: the projection under ``W | Z = z``."""
    w, mass = grid(nodes)
    g1 = cp.propensity(w)
    share = g1 * intermediate_density(w, 1, z) + (1 - g1) * intermediate_density(w, 0, z)
    q_t, q_r = arm_means(w, z)
    return cp.projection(design(w), mass * share, q_t, q_r, "difference")


def total_effect_limit(*, nodes: int = NODES) -> np.ndarray:
    """The projection of the total effect ``E[Y(1) - Y(0) | W]`` onto ``(1, W1)``."""
    w, mass = grid(nodes)
    totals = []
    for a in (1, 0):
        q1 = intermediate_probability(w, a)
        totals.append(q1 * expit(outcome_logit(w, a, 1)) + (1 - q1) * expit(outcome_logit(w, a, 0)))
    return cp.projection(design(w), mass, totals[0], totals[1], "difference")


def plugin_wrong_limit(z: int, *, nodes: int = NODES) -> np.ndarray:
    """The plug-in projection at the declared wrong ``Qbar``."""
    w, mass = grid(nodes)
    q_t, q_r = arm_means(w, z, error=True)
    return cp.projection(design(w), mass, q_t, q_r, "difference")


#: The TMLE covariate variants whose limits the design reads.
COVARIATES = ("full", "no_density", "no_selection")


def tmle_wrong_limit(z: int, covariate: str = "full", *, nodes: int = NODES) -> np.ndarray:
    r"""The difference-scale TMLE limit at the declared wrong ``Qbar``, true mechanisms.

    The observed covariate of arm ``a`` is ``s_a 1{A = a} 1{Z = z} phi / (g_a q_z(a))``;
    ``"no_density"`` drops ``q_z``, and ``"no_selection"`` drops ``1{Z = z}``, so every row of
    arm ``a`` is fluctuated whatever its ``Z``.  The counterfactual covariate at ``(a, z)`` is
    ``s_a phi / (g_a q_z(a))`` (without ``q_z`` for ``"no_density"``).  The fluctuation is
    logistic: ``Q*_eps = expit(logit Qbar_w + eps' H)``.  ``eps`` solves the population score
    ``E[H (Y - Q*_eps)] = 0``, integrated over the nodes, both arms and both intermediate values.
    """
    if covariate not in COVARIATES:
        raise KeyError(covariate)
    w, mass = grid(nodes)
    phi = design(w)
    g1 = cp.propensity(w)
    gram = phi.T @ (mass[:, None] * phi)
    m_inverse = np.linalg.inv(gram)

    def counterfactual(a: int) -> np.ndarray:
        g_a = g1 if a == 1 else 1 - g1
        density = 1.0 if covariate == "no_density" else intermediate_density(w, a, z)
        sign = 1.0 if a == 1 else -1.0
        return sign * (phi @ m_inverse) / (g_a * density)[:, None]

    cells: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for a in (0, 1):
        g_a = g1 if a == 1 else 1 - g1
        h = counterfactual(a)
        for level in (0, 1):
            if covariate != "no_selection" and level != z:
                continue
            weight = mass * g_a * intermediate_density(w, a, level)
            truth_q = expit(outcome_logit(w, a, level))
            start = outcome_logit(w, a, level) + declared_error(w, a)
            cells.append((weight, h, start, truth_q))
    epsilon = np.zeros(phi.shape[1])
    for _ in range(100):
        score = np.zeros_like(epsilon)
        jacobian = np.zeros((len(epsilon), len(epsilon)))
        for weight, h, start, truth_q in cells:
            q = expit(start + h @ epsilon)
            score += h.T @ (weight * (truth_q - q))
            jacobian -= h.T @ ((weight * q * (1 - q))[:, None] * h)
        step = np.linalg.solve(jacobian, score)
        epsilon = epsilon - step
        if np.max(np.abs(step)) < 1e-14:
            break
    targeted = [
        expit(outcome_logit(w, a, z) + declared_error(w, a) + counterfactual(a) @ epsilon)
        for a in (1, 0)
    ]
    return cp.projection(phi, mass, targeted[0], targeted[1], "difference")


def standardized(limit: np.ndarray, z: int, n: int = 4_000) -> np.ndarray:
    """``(limit - truth) / SD`` at level ``z`` and ``n``."""
    return (limit - truth(z)) / efficient_sd(z, n)


def smallest_density_product(*, nodes: int = NODES) -> float:
    """The smallest ``g_a(W) q_z(a, W)`` over the nodes, arms and levels."""
    w, _ = grid(nodes)
    g1 = cp.propensity(w)
    return float(
        min(
            np.min(g_a * intermediate_density(w, a, z))
            for a, g_a in ((1, g1), (0, 1 - g1))
            for z in LEVELS
        )
    )


# ---------------------------------------------------------------------------- the bands


#: The Monte Carlo seed of the band design numbers: a local constant, not a record seed (the
#: study's record seeds are 20261511 and 20261512).
BAND_DESIGN_SEED = 20261513


def pointwise_joint_coverage(
    covariance: np.ndarray,
    grid_design: np.ndarray,
    *,
    draws: int = 400_000,
    seed: int = BAND_DESIGN_SEED,
) -> tuple[float, float]:
    """``(joint coverage of the pointwise 95% intervals, max-t critical value)`` over a grid.

    By Monte Carlo with a fixed seed: coefficient errors are drawn from ``covariance`` and read
    at the grid's design rows (the curve's covariance has the coefficients' rank, so it is
    drawn through them).  The coverage is the share of draws whose every standardized value lies
    inside 1.96, and the critical value is the 0.95 quantile of the largest absolute one.
    """
    rng = np.random.default_rng(seed)
    errors = rng.multivariate_normal(np.zeros(len(covariance)), covariance, size=draws)
    values = errors @ grid_design.T
    sd = np.sqrt(np.einsum("ip,pq,iq->i", grid_design, covariance, grid_design))
    largest = np.max(np.abs(values / sd), axis=1)
    return float(np.mean(largest <= CRITICAL)), float(np.quantile(largest, 0.95))


def cde_band_reading(z: int = 0, n: int = 4_000) -> tuple[float, float]:
    """The pointwise joint coverage and critical value on the CDE band grid at level ``z``."""
    grid_design = np.column_stack([np.ones(len(BAND_GRID_W1)), BAND_GRID_W1])
    return pointwise_joint_coverage(efficient_covariance(z, n), grid_design)


def proj_band_grid() -> pd.DataFrame:
    """The ``proj_obs`` band grid ``{-0.8, 0, 0.8}^2`` over ``(W1, W3)``: 9 points."""
    values = (-0.8, 0.0, 0.8)
    points = [(a, b) for a in values for b in values]
    return pd.DataFrame(points, columns=["W1", "W3"])


def proj_band_reading(n: int = 4_000) -> tuple[float, float]:
    """The pointwise joint coverage and critical value on the ``proj_obs`` grid at ``n``."""
    grid_frame = proj_band_grid()
    grid_design = np.column_stack(
        [np.ones(len(grid_frame)), grid_frame["W1"].to_numpy(), grid_frame["W3"].to_numpy()]
    )
    return pointwise_joint_coverage(cp.efficient_covariance("observational", n), grid_design)


def design_table() -> dict[int, Any]:
    """Every number of the plan 5.1 table, by level: truths, SDs and standardized limits."""
    out: dict[int, Any] = {}
    total = total_effect_limit()
    for z in LEVELS:
        out[z] = {
            "beta": tuple(truth(z)),
            "beta_log_ratio": tuple(truth(z, "log_ratio")),
            "sd": tuple(efficient_sd(z, 4_000)),
            "sd_log_ratio": tuple(efficient_sd(z, 4_000, "log_ratio")),
            "plugin_stratum": tuple(standardized(plugin_stratum_limit(z), z)),
            "total_effect": tuple(standardized(total, z)),
            "plugin_wrong": tuple(standardized(plugin_wrong_limit(z), z)),
            **{
                f"tmle_{name}": tuple(standardized(tmle_wrong_limit(z, name), z))
                for name in COVARIATES
            },
        }
    return out


def quadrature_error(z: int = 0) -> float:
    """The largest change in a ``cde_obs`` truth when the nodes per axis drop from 48 to 32."""
    return float(np.max(np.abs(truth(z, nodes=48) - truth(z, nodes=32))))


def logit_safe(p: np.ndarray) -> np.ndarray:
    """``logit`` clipped away from 0 and 1."""
    return logit(np.clip(p, 1e-12, 1 - 1e-12))


__all__ = [
    "BAND_GRID_W1",
    "LEVELS",
    "TERMS",
    "cde_band_reading",
    "design_table",
    "efficient_sd",
    "proj_band_grid",
    "proj_band_reading",
    "sample",
    "tmle_wrong_limit",
    "truth",
    "truth_names",
]

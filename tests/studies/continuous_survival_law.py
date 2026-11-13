r"""The law of the ``continuous-time-survival`` study, with its truths (X91 part G).

Hage et al. (2025) scenario 1 with two stated departures (``X91-G-design.md`` section 10).
:math:`X_1, X_2` indicate the levels of a three-level factor with equal shares,
:math:`X_3 \sim N(0, 1)`, and :math:`X_4 = 1\{X_3 < 1\}` is carried as a column.  Given
:math:`(Z, X)` the two causes have constant hazards

.. math::

    h_1 = \exp(-1.5 + X_1 - X_2 + 0.5 X_3 - Z + \beta Z X_4), \qquad
    h_2 = \exp(-2 - X_1 + X_2 - 0.5 X_3 - 0.5 Z),

and :math:`Z \sim \operatorname{Bern}\{\operatorname{expit}(\omega_1 X_1 + \omega_2 X_2 +
\omega_3 X_3)\}` with :math:`\omega = (1, -1, 1)`, or :math:`(6, -1, 1)` in the positivity cell
(Hage S2).  Censoring is exponential with rate :math:`0.15`, independent of :math:`(Z, X)` as in
Hage, or :math:`0.15 \exp(0.5 X_3 + 0.5 Z)` in the dependent-censoring cell.

**Departures from Hage.** (a) The arm-specific term :math:`\beta Z X_4` in the cause-1 log
hazard: of the default library only the Cox ``"interaction"`` model contains it, so the library
holds the truth only through that member.  Hage's scenario 3 adds :math:`-4 \cdot 1\{X_3 < 1\}`
without saying it is arm-specific.  The term is not the study's wrong-model witness: the
``"main"`` model's treated risk of cause 1 at the middle time sits within 0.002 of the truth on a
draw of 20,000, so a library of ``"treatment"`` and ``"main"`` would be wrong in name only.  The
wrong event hazard is the ``"treatment"`` model alone, which reads that risk as 0.288 against
0.235.  (b) The dependent-censoring hazard, since Hage's censoring
is independent of :math:`X` and :math:`Z`.

**Truth.** With constant hazards :math:`F_j(t \mid z, x) = h_j / h \cdot (1 - e^{-h t})`,
:math:`h = h_1 + h_2`, so :math:`E\,F_j(t \mid z, X)` is a sum over the three factor levels of an
integral over :math:`X_3`, split at :math:`X_3 = 1` where :math:`X_4` jumps and cut at :math:`\pm 12`, each piece by
adaptive quadrature against the normal density (:func:`truth`).  The target times are the
25th, 50th and 75th percentiles of the observed event times under the base law
(:data:`TIMES`), read once from a large fixed-seed draw and rounded to two decimals.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.special import expit

#: The arm-specific coefficient of :math:`Z X_4` in the cause-1 log hazard.
BETA = 1.0
#: The treatment model's coefficients on :math:`(X_1, X_2, X_3)`.
OMEGA = (1.0, -1.0, 1.0)
#: Hage S2's coefficients, which push the treatment probability towards 0 and 1.
OMEGA_POSITIVITY = (6.0, -1.0, 1.0)
#: The base censoring rate.
CENSORING_RATE = 0.15
#: The dependent-censoring coefficients on :math:`(X_3, Z)`.
CENSORING_DEPENDENCE = (0.5, 0.5)
#: The target times: the quartiles of the observed event times under the base law.
TIMES = (0.46, 1.12, 2.32)
ARMS = {"treated": 1, "control": 0}
REFERENCE = "control"
CAUSES = {1: "cause1", 2: "cause2"}
BASELINE = ("X1", "X2", "X3", "X4")


def rates(
    x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, z: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """The cause hazards :math:`(h_1, h_2)` given the arm and the covariates."""
    x4 = (np.asarray(x3) < 1.0).astype(float)
    h1 = np.exp(-1.5 + x1 - x2 + 0.5 * x3 - 1.0 * z + BETA * z * x4)
    h2 = np.exp(-2.0 - x1 + x2 - 0.5 * x3 - 0.5 * z)
    return np.asarray(h1, dtype=float), np.asarray(h2, dtype=float)


def sample(
    n: int,
    seed: int | np.random.Generator,
    *,
    positivity: bool = False,
    dependent_censoring: bool = False,
) -> pd.DataFrame:
    """One draw as a frame with ``time``, ``event``, ``Z`` and ``X1`` to ``X4``.

    Parameters
    ----------
    n : int
        The number of units.
    seed : int or numpy.random.Generator
        The draw's seed.
    positivity : bool
        Draw the treatment with :data:`OMEGA_POSITIVITY`.
    dependent_censoring : bool
        Censor at the rate that depends on :math:`X_3` and :math:`Z`.

    Returns
    -------
    pandas.DataFrame
        One row per unit.
    """
    rng = np.random.default_rng(seed)
    factor = rng.integers(0, 3, n)
    x1, x2 = (factor == 1).astype(float), (factor == 2).astype(float)
    x3 = rng.normal(size=n)
    w = OMEGA_POSITIVITY if positivity else OMEGA
    z = rng.binomial(1, expit(w[0] * x1 + w[1] * x2 + w[2] * x3)).astype(float)
    h1, h2 = rates(x1, x2, x3, z)
    event_time = rng.exponential(1.0 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    rate = np.full(n, CENSORING_RATE)
    if dependent_censoring:
        rate = rate * np.exp(CENSORING_DEPENDENCE[0] * x3 + CENSORING_DEPENDENCE[1] * z)
    censor_time = rng.exponential(1.0 / rate)
    return pd.DataFrame(
        {
            "time": np.minimum(event_time, censor_time),
            "event": np.where(event_time <= censor_time, cause, 0),
            "Z": z,
            "X1": x1,
            "X2": x2,
            "X3": x3,
            "X4": (x3 < 1.0).astype(float),
        }
    )


def risk(z: int, cause: int, t: float) -> float:
    """:math:`E\\,F_j(t \\mid z, X)`, by quadrature over :math:`X_3` split at 1."""
    total = 0.0
    for x1, x2 in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):

        def integrand(x3: float, x1: float = x1, x2: float = x2) -> float:
            h1, h2 = rates(np.array(x1), np.array(x2), np.array(x3), z)
            h = h1 + h2
            hj = h1 if cause == 1 else h2
            return float(hj / h * (1.0 - np.exp(-h * t)) * stats.norm.pdf(x3))

        for low, high in ((-12.0, 1.0), (1.0, 12.0)):
            value, _ = integrate.quad(integrand, low, high, epsabs=1e-13, epsrel=1e-12)
            total += value / 3.0
    return total


def level_name(arm: str, cause: str, t: float) -> str:
    """The engine's name of an arm's risk of one cause at one time."""
    return f"cif_regimen[{arm}, {cause} @ t={t:g}]"


def difference_name(cause: str, t: float) -> str:
    """The engine's name of the risk difference of one cause at one time."""
    return f"ate_regimen[treated vs control, {cause} @ t={t:g}]"


def truth(times: tuple[float, ...] = TIMES) -> dict[str, float]:
    """Every arm's risk of each cause at each time, and the risk differences.

    Parameters
    ----------
    times : tuple of float
        The target times.

    Returns
    -------
    dict of str to float
        The truths keyed by the engine's parameter names.
    """
    out: dict[str, float] = {}
    for arm, z in ARMS.items():
        for code, cause in CAUSES.items():
            for t in times:
                out[level_name(arm, cause, t)] = risk(z, code, t)
    for cause in CAUSES.values():
        for t in times:
            out[difference_name(cause, t)] = (
                out[level_name("treated", cause, t)] - out[level_name("control", cause, t)]
            )
    return out


def event_time_quartiles(n: int = 2_000_000, seed: int = 2026911951) -> tuple[float, ...]:
    """The quartiles of the observed event times under the base law, from one large draw."""
    frame = sample(n, seed)
    observed = frame.loc[frame["event"] > 0, "time"].to_numpy()
    return tuple(float(v) for v in np.round(np.quantile(observed, [0.25, 0.5, 0.75]), 2))


TRUTH: Mapping[str, float] = truth()
#: The first target time.  Its rows are published but not gated: in the committed pilot
#: (aa19c6d6, cross-fitted) the influence-curve standard error there reads up to 10% below the
#: sampling spread at n = 1,000 (5% at 4,000) and coverage runs 0.878 to 0.956 at 1,000 (0.926
#: to 0.957 at 4,000), although the estimate is unbiased and efficient (X91-G-design.md 15).
EARLY_TIME = TIMES[0]
#: The targets the record gates: every one at the two later times.
GATED: tuple[str, ...] = tuple(name for name in TRUTH if f"@ t={EARLY_TIME:g}]" not in name)
#: The first target time's targets, published as measured.
EARLY: tuple[str, ...] = tuple(name for name in TRUTH if name not in GATED)


def _x3_rule(nodes: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normal-density weights over :math:`X_3`, split at 1 and cut at :math:`\\pm 12`."""
    points, weights = [], []
    for low, high in ((-12.0, 1.0), (1.0, 12.0)):
        x, w = np.polynomial.legendre.leggauss(nodes)
        mid, half = (high + low) / 2.0, (high - low) / 2.0
        points.append(mid + half * x)
        weights.append(half * w * stats.norm.pdf(mid + half * x))
    return np.concatenate(points), np.concatenate(weights)


def efficiency_sd(cause: int, t: float, nodes: int = 200) -> dict[str, float]:
    r""":math:`\sqrt{E D^2}` of each arm's risk of ``cause`` at ``t`` and of its difference.

    On the base law (censoring at the constant rate :data:`CENSORING_RATE`, treatment by
    :data:`OMEGA`), the efficient influence function of :math:`E F_j(t \mid a, X)` is
    :math:`F_j(t \mid a, X) - \psi` plus a martingale integral, so

    .. math::

        E D_a^2 = \operatorname{Var} F_j(t \mid a, X) + E_X \frac{1}{\pi_a(X)} \int_0^t
        \sum_l h_l\, c_l(u)^2\, \frac{S(u)}{S_c(u)}\, du,
        \qquad c_l(u) = 1\{l = j\} - \frac{F_j(t) - F_j(u)}{S(u)},

    with constant hazards :math:`(F_j(t) - F_j(u)) / S(u) = (h_j / h)(1 - e^{-h (t - u)})`.  The
    difference adds the two martingale terms to :math:`\operatorname{Var}\{F_j(t \mid 1, X) -
    F_j(t \mid 0, X)\}`, since the arms' martingales are orthogonal to each other and to every
    function of :math:`X`.  Gauss-Legendre rules in :math:`u` and in :math:`X_3` evaluate it.

    Parameters
    ----------
    cause : int
        The cause code, 1 or 2.
    t : float
        The target time.
    nodes : int
        Nodes per rule piece.

    Returns
    -------
    dict of str to float
        ``{"treated": ..., "control": ..., "difference": ...}``.
    """
    x3, w3 = _x3_rule(nodes)
    u, wu = np.polynomial.legendre.leggauss(nodes)
    u, wu = t / 2.0 * (u + 1.0), t / 2.0 * wu
    martingale: dict[int, float] = {}
    level: dict[int, np.ndarray] = {}
    # Three factor levels with equal shares, each over the same X3 rule.
    weights = np.concatenate([w3 / 3.0] * 3)
    for z in (0, 1):
        totals, levels = [], []
        for x1, x2 in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
            h1, h2 = rates(np.full_like(x3, x1), np.full_like(x3, x2), x3, z)
            h = h1 + h2
            hj = h1 if cause == 1 else h2
            p1 = expit(OMEGA[0] * x1 + OMEGA[1] * x2 + OMEGA[2] * x3)
            pi = p1 if z == 1 else 1.0 - p1
            g = (hj / h)[:, None] * (1.0 - np.exp(-h[:, None] * (t - u[None, :])))
            terms = hj[:, None] * (1.0 - g) ** 2 + (h - hj)[:, None] * g**2
            ratio = np.exp(-(h[:, None] - CENSORING_RATE) * u[None, :])
            totals.append((terms * ratio) @ wu / pi)
            levels.append(hj / h * (1.0 - np.exp(-h * t)))
        martingale[z] = float(weights @ np.concatenate(totals))
        level[z] = np.concatenate(levels)

    def variance(values: np.ndarray) -> float:
        mean = float(weights @ values)
        return float(weights @ (values - mean) ** 2)

    return {
        "treated": float(np.sqrt(variance(level[1]) + martingale[1])),
        "control": float(np.sqrt(variance(level[0]) + martingale[0])),
        "difference": float(np.sqrt(variance(level[1] - level[0]) + martingale[1] + martingale[0])),
    }

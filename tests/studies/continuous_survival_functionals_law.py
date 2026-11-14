r"""Truths of the ``continuous-survival-functionals`` study (X6 part (c)).

The law is the registered study's base law, :mod:`tests.studies.continuous_survival_law`, read at
the restriction time :data:`TAU`, its upper event-time quartile.  With constant cause-specific
hazards :math:`h_1, h_2` and :math:`h = h_1 + h_2` given :math:`(z, x)`,

.. math::

    \int_0^\tau F_j(t \mid z, x)\, dt = \frac{h_j}{h}\left\{\tau - \frac{1 - e^{-h\tau}}{h}\right\},
    \qquad
    \int_0^\tau S(t \mid z, x)\, dt = \frac{1 - e^{-h\tau}}{h},

so each restricted mean time lost (RMTL) and restricted mean survival time (RMST) is an
integral over :math:`X` taken by the same quadrature as the risks (:func:`restricted`).  A risk
ratio's truth is the ratio of the two arms' risks.

The gated targets (:data:`GATED`) are each arm's risk of each cause and each risk difference at
the median and upper quartile, the risk ratio of each cause there, the RMTL of each arm and cause,
the RMST of each arm, and their differences at :data:`TAU`: 25 in all.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import integrate, stats
from scipy.special import expit

from tests.studies import continuous_survival_law as law

#: The restriction time: the upper quartile of the observed event times.
TAU = law.TIMES[2]
#: The target times whose risks, risk differences and risk ratios are gated.
GATED_TIMES = law.TIMES[1:]


def rmtl_name(arm: str, cause: str, tau: float = TAU) -> str:
    """The engine's name of an arm's RMTL of one cause."""
    return f"rmtl_regimen[{arm}, {cause} @ t={tau:g}]"


def rmtl_difference_name(cause: str, tau: float = TAU) -> str:
    """The engine's name of one cause's RMTL difference."""
    return f"rmtl_regimen[treated vs control, {cause} @ t={tau:g}]"


def rmst_name(arm: str, tau: float = TAU) -> str:
    """The engine's name of an arm's RMST."""
    return f"rmst_regimen[{arm} @ t={tau:g}]"


def rmst_difference_name(tau: float = TAU) -> str:
    """The engine's name of the RMST difference."""
    return f"rmst_regimen[treated vs control @ t={tau:g}]"


def ratio_name(cause: str, t: float) -> str:
    """The engine's name of one cause's risk ratio at one time."""
    return f"rr_regimen[treated vs control, {cause} @ t={t:g}]"


def restricted(z: int, cause: int | None, tau: float = TAU) -> float:
    """:math:`E \\int_0^\\tau F_j(t \\mid z, X) dt`, or the RMST when ``cause`` is ``None``."""
    total = 0.0
    for x1, x2 in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):

        def integrand(x3: float, x1: float = x1, x2: float = x2) -> float:
            h1, h2 = law.rates(np.array(x1), np.array(x2), np.array(x3), z)
            h = h1 + h2
            alive = (1.0 - np.exp(-h * tau)) / h
            lost = alive if cause is None else (h1 if cause == 1 else h2) / h * (tau - alive)
            return float(lost * stats.norm.pdf(x3))

        for low, high in ((-12.0, 1.0), (1.0, 12.0)):
            value, _ = integrate.quad(integrand, low, high, epsabs=1e-13, epsrel=1e-12)
            total += value / 3.0
    return total


def truth() -> dict[str, float]:
    """Every gated target's truth, keyed by the engine's parameter names."""
    out: dict[str, float] = {}
    for t in GATED_TIMES:
        for arm in law.ARMS:
            for cause in law.CAUSES.values():
                name = law.level_name(arm, cause, t)
                out[name] = law.TRUTH[name]
        for cause in law.CAUSES.values():
            out[law.difference_name(cause, t)] = law.TRUTH[law.difference_name(cause, t)]
            out[ratio_name(cause, t)] = (
                law.TRUTH[law.level_name("treated", cause, t)]
                / law.TRUTH[law.level_name("control", cause, t)]
            )
    for arm, z in law.ARMS.items():
        for code, cause in law.CAUSES.items():
            out[rmtl_name(arm, cause)] = restricted(z, code)
    for cause in law.CAUSES.values():
        out[rmtl_difference_name(cause)] = (
            out[rmtl_name("treated", cause)] - out[rmtl_name("control", cause)]
        )
    for arm, z in law.ARMS.items():
        out[rmst_name(arm)] = restricted(z, None)
    out[rmst_difference_name()] = out[rmst_name("treated")] - out[rmst_name("control")]
    return out


def rmst_difference_efficiency_sd(tau: float = TAU, nodes: int = 200) -> float:
    r""":math:`\sqrt{E D^2}` of the RMST difference at ``tau`` on the base law.

    The efficient influence function of :math:`E \int_0^\tau S(t \mid a, X) dt` is the centred
    conditional RMST plus :math:`-1\{A = a\} / \{\pi_a(X)\} \int_0^\tau R(u) / \{S(u) S_c(u^-)\}
    dM(u)`, with :math:`R(u) = \int_u^\tau S(t) dt` and :math:`M` the all-cause counting-process
    martingale.  With a constant total hazard, :math:`R(u) / S(u) = (1 - e^{-h(\tau - u)}) / h`, so

    .. math::

        E D^2 = \operatorname{Var}\{\mathrm{RMST}(1, X) - \mathrm{RMST}(0, X)\}
        + \sum_a E_X \frac{1}{\pi_a(X)} \int_0^\tau h \left(\frac{R(u)}{S(u)}\right)^2
        \frac{S(u)}{S_c(u)}\, du,

    since the arms' martingales are orthogonal to each other and to every function of :math:`X`.
    The study reads it only to size the calibration cell's noise control, and claims no
    efficiency.

    Parameters
    ----------
    tau : float
        The restriction time.
    nodes : int
        Nodes per rule piece.

    Returns
    -------
    float
        The standard deviation.
    """
    x3, w3 = law._x3_rule(nodes)
    u, wu = np.polynomial.legendre.leggauss(nodes)
    u, wu = tau / 2.0 * (u + 1.0), tau / 2.0 * wu
    weights = np.concatenate([w3 / 3.0] * 3)
    martingale = 0.0
    level: dict[int, np.ndarray] = {}
    for z in (0, 1):
        totals, levels = [], []
        for x1, x2 in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
            h1, h2 = law.rates(np.full_like(x3, x1), np.full_like(x3, x2), x3, z)
            h = h1 + h2
            p1 = expit(law.OMEGA[0] * x1 + law.OMEGA[1] * x2 + law.OMEGA[2] * x3)
            pi = p1 if z == 1 else 1.0 - p1
            remaining = (1.0 - np.exp(-h[:, None] * (tau - u[None, :]))) / h[:, None]
            ratio = np.exp(-(h[:, None] - law.CENSORING_RATE) * u[None, :])
            totals.append((h[:, None] * remaining**2 * ratio) @ wu / pi)
            levels.append((1.0 - np.exp(-h * tau)) / h)
        martingale += float(weights @ np.concatenate(totals))
        level[z] = np.concatenate(levels)
    contrast = level[1] - level[0]
    mean = float(weights @ contrast)
    return float(np.sqrt(weights @ (contrast - mean) ** 2 + martingale))


TRUTH: Mapping[str, float] = truth()
#: The 25 gated targets, in the order of :data:`TRUTH`.
GATED: tuple[str, ...] = tuple(TRUTH)

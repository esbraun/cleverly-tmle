r"""A randomized pilot of the navigation offer, with recorded offer chances and an exact truth.

The pilot drew who got the navigation offer at random.  The chance of an offer depended on the
patient's discharge-risk group, and the pilot recorded each patient's chance of each arm.

======================  =================================================================
node                    law
======================  =================================================================
:math:`S`               ``discharge_risk``, standard normal
:math:`P`               ``prior_utilization``, standard normal
:math:`M`, :math:`G`    ``medication_burden`` and ``age``, standard normal, unused by the law
:math:`H`               ``high_risk``, :math:`1\{S > 0\}`
:math:`A`               ``transition_navigation``, Bernoulli with
                        :math:`P(A = 1 \mid H) = 0.2 + 0.6 H`
:math:`Y`               ``transition_success``, Bernoulli with
                        :math:`\operatorname{expit}\{-0.2 + A (0.4 + 0.6 H) - 1.2 H
                        - 0.3 S + 0.5 S^2 - 0.5 P\}`
======================  =================================================================

The columns ``chance_no_offer`` and ``chance_offer`` hold :math:`P(A = 0 \mid H)` and
:math:`P(A = 1 \mid H)`, the design's offer chances.  Each counterfactual mean is a
two-dimensional integral over :math:`(S, P)`, computed by Gauss-Legendre quadrature on each side
of :math:`S = 0` and Gauss-Hermite quadrature in :math:`P`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import Backend
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = ["PILOT_OFFER_CHANCES", "make_randomized_pilot", "randomized_pilot_truth"]

#: The pilot's chance of an offer in each discharge-risk group, keyed by ``high_risk``.
PILOT_OFFER_CHANCES: dict[int, float] = {0: 0.2, 1: 0.8}


def _success_probability(offer: Any, risk: Any, utilization: Any) -> Any:
    """``P(transition_success = 1 | A, S, P)`` of the pilot law."""
    a = np.asarray(offer, dtype=float)
    r = np.asarray(risk, dtype=float)
    high = (r > 0.0).astype(float)
    return expit(
        -0.2 + a * (0.4 + 0.6 * high) - 1.2 * high - 0.3 * r + 0.5 * r**2 - 0.5 * utilization
    )


def randomized_pilot_truth() -> dict[str, float]:
    r"""Return the exact counterfactual means and the average effect of the pilot law.

    Returns
    -------
    dict of str to float
        ``ey0`` and ``ey1``, the share of successful transitions if no one or everyone got the
        offer, and ``ate``, their difference.

    See Also
    --------
    cleverly.datasets.make_randomized_pilot : Draws data from this law.

    Notes
    -----
    The success probability is smooth in :math:`S` on each side of the risk threshold
    :math:`S = 0`, so each side is integrated with 200 Gauss-Legendre nodes on ``[0, 12]``
    against the standard normal density.  The mass beyond 12 is below ``1e-32``.
    :math:`P` enters through a standard normal integral, taken with 80 Gauss-Hermite nodes.

    Examples
    --------
    >>> from cleverly.datasets import randomized_pilot_truth
    >>> truth = randomized_pilot_truth()
    >>> round(truth["ate"], 4)
    0.142
    """
    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(200)
    half = 6.0 * (legendre_nodes + 1.0)
    half_weights = 6.0 * legendre_weights * np.exp(-0.5 * half**2) / np.sqrt(2.0 * np.pi)
    risk = np.concatenate([-half, half])
    risk_weights = np.concatenate([half_weights, half_weights])
    hermite_nodes, hermite_weights = np.polynomial.hermite_e.hermegauss(80)
    hermite_weights = hermite_weights / hermite_weights.sum()
    truth: dict[str, float] = {}
    for arm in (0, 1):
        grid = _success_probability(arm, risk[:, None], hermite_nodes[None, :])
        truth[f"ey{arm}"] = float(risk_weights @ grid @ hermite_weights)
    truth["ate"] = truth["ey1"] - truth["ey0"]
    return truth


def make_randomized_pilot(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""Draw the randomized pilot of the navigation offer, with its recorded offer chances.

    Parameters
    ----------
    n : int
        Number of discharges.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per discharge: ``transition_success`` (1 or 0), ``transition_navigation``
        (the offer, 1 or 0), ``high_risk`` (1 when ``discharge_risk`` is above 0),
        ``discharge_risk``, ``prior_utilization``, ``medication_burden``, ``age``, and the
        recorded chances ``chance_no_offer`` and ``chance_offer``.
    truth : dict of str to float
        The exact values of :func:`randomized_pilot_truth`.

    See Also
    --------
    cleverly.datasets.randomized_pilot_truth : The exact truth of this law.
    cleverly.datasets.navigation_protocol : The protocol of the program these data describe.

    Notes
    -----
    The offer chance is 0.2 in the low-risk group and 0.8 in the high-risk group
    (:data:`PILOT_OFFER_CHANCES`).  The high-risk group also has a lower success rate, so a
    plain comparison of offered and other patients is confounded by the risk group.  The
    module docstring gives the law.

    Examples
    --------
    >>> from cleverly.datasets import make_randomized_pilot
    >>> frame, truth = make_randomized_pilot(n=500, seed=3)
    >>> frame.shape
    (500, 9)
    >>> sorted(frame.groupby("high_risk")["chance_offer"].first().round(2))
    [0.2, 0.8]
    """
    rng = np.random.default_rng(seed)
    risk, utilization, medication, age = rng.standard_normal((4, n))
    high = (risk > 0.0).astype(float)
    chance = np.where(high == 1.0, PILOT_OFFER_CHANCES[1], PILOT_OFFER_CHANCES[0])
    offer = (rng.random(n) < chance).astype(float)
    success = (rng.random(n) < _success_probability(offer, risk, utilization)).astype(float)
    columns = {
        "transition_success": success,
        "transition_navigation": offer,
        "high_risk": high,
        "discharge_risk": risk,
        "prior_utilization": utilization,
        "medication_burden": medication,
        "age": age,
        "chance_no_offer": 1.0 - chance,
        "chance_offer": chance,
    }
    return frame_from_dict(columns, backend=backend), randomized_pilot_truth()

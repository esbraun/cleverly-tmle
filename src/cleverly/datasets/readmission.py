r"""A discharge program run by navigator teams, with a known truth and the protocol of its question.

A hospital discharges adults home.  Each patient belongs to one navigator team.  At discharge the
team decides whether to offer the patient navigation, and on day 30 whether to make a follow-up
call.  The outcome is an unplanned readmission by day 90.  Some patients leave follow-up before
day 30 or before day 90, and a readmission before day 30 enters the history of the call.

=========================  ========================================================================
column                     law
=========================  ========================================================================
``team``                   the navigator team, ``team_size`` consecutive rows per team
``lives_alone``            Bernoulli(0.35)
``discharge_risk``         standard normal
``offer``                  Bernoulli, logit linear in ``lives_alone`` and ``discharge_risk``
``followed_30``            Bernoulli, logit linear in ``discharge_risk`` and ``offer``
``readmitted_30``          among ``followed_30 = 1``: Bernoulli, logit linear in the baseline
                           and ``offer``
``call``                   among ``followed_30 = 1``: Bernoulli, logit linear in
                           ``discharge_risk``, ``offer`` and ``readmitted_30``
``followed_90``            among ``followed_30 = 1``: one after a readmission by day 30, else
                           Bernoulli, logit linear in ``discharge_risk`` and ``call``
``readmitted_90``          among ``followed_90 = 1``: one after a readmission by day 30, else
                           Bernoulli with mean :math:`p + \delta u \min(p, 1 - p)`
=========================  ========================================================================

Every column after a patient leaves follow-up is missing, which is the form
:class:`~cleverly.longitudinal.LongitudinalData` requires.  Nobody dies in the simulated program.

**The team component.**  :math:`p` is the logistic day-31-to-90 readmission probability given the
history, and :math:`u` is a latent of the patient's team and discharge offer: two independent
Rademacher draws per team, one read by the offered patients and one by the others, with
:math:`\delta = 0.95`.  Two draws rather than one carry the team into a contrast of the offer, as
:func:`~cleverly.datasets.longitudinal._shared_within_clusters` explains for ``make_longitudinal``.
The construction is that of the study law ``tests/studies/clustered_longitudinal_laws.py``, which
the registered study ``clustered-cross-fitted-ltmle`` reads.

**Truth.**  :math:`E_u[p + \delta u \min(p, 1 - p)] = p` for a latent with mean zero, and the latent
enters no node but the last outcome.  So the law of the last outcome given the history is the
logistic one, and the mean under the plan :math:`(a_1, a_2)` is

.. math::

    E[Y_{a_1 a_2}] = E_W\bigl[h(W, a_1) + \{1 - h(W, a_1)\}\, p(W, a_1, a_2)\bigr],

with :math:`h` the day-30 readmission probability.  :func:`team_readmission_truth` evaluates it by
enumeration over ``lives_alone`` and a Gauss--Hermite rule over ``discharge_risk``, to well under
``1e-12``.  ``tests/unit/test_datasets_readmission.py`` compares it with a Monte Carlo draw of the
intervened law.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np

from .._typing import Backend
from ..protocol import StudyProtocol
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict
from .navigation import longitudinal_navigation_protocol

__all__ = ["make_team_readmission", "team_readmission_protocol", "team_readmission_truth"]

#: The coefficients of the law, each on the logit scale.
_LAW = {
    "lives_alone": 0.35,
    # offer at discharge
    "offer_intercept": -0.2,
    "offer_alone": 0.4,
    "offer_risk": 0.7,
    # still followed on day 30
    "followed_30_intercept": 2.4,
    "followed_30_risk": -0.3,
    "followed_30_offer": 0.3,
    # readmitted by day 30
    "early_intercept": -1.7,
    "early_alone": 0.3,
    "early_risk": 0.8,
    "early_offer": -0.5,
    # the day-30 call
    "call_intercept": -0.4,
    "call_risk": 0.5,
    "call_offer": 1.0,
    "call_early": 0.6,
    # still followed on day 90, among patients not readmitted by day 30
    "followed_90_intercept": 2.4,
    "followed_90_risk": -0.2,
    "followed_90_call": 0.3,
    # readmitted between day 31 and day 90, among patients not readmitted by day 30
    "late_intercept": -1.5,
    "late_alone": 0.3,
    "late_risk": 0.7,
    "late_offer": -0.3,
    "late_call": -0.5,
}

#: The scale of the team component of the day-90 outcome.
TEAM_DELTA = 0.95

#: The plans of the question, as (offer at discharge, call on day 30).
_PLANS = {"always": (1.0, 1.0), "never": (0.0, 0.0), "offer_only": (1.0, 0.0)}


def _early(alone: Any, risk: Any, offer: Any) -> Any:
    """``P(readmitted_30 = 1 | lives_alone, discharge_risk, offer)``."""
    return expit(
        _LAW["early_intercept"]
        + _LAW["early_alone"] * alone
        + _LAW["early_risk"] * risk
        + _LAW["early_offer"] * offer
    )


def _late(alone: Any, risk: Any, offer: Any, call: Any) -> Any:
    """``P(readmitted_90 = 1 | history, readmitted_30 = 0)`` before the team component."""
    return expit(
        _LAW["late_intercept"]
        + _LAW["late_alone"] * alone
        + _LAW["late_risk"] * risk
        + _LAW["late_offer"] * offer
        + _LAW["late_call"] * call
    )


@cache
def team_readmission_truth(offer: float, call: float, nodes: int = 64) -> float:
    r"""Return the share readmitted by day 90 had every patient followed the plan ``(offer, call)``.

    Parameters
    ----------
    offer : float
        The discharge offer of the plan, 0 or 1.
    call : float
        The day-30 call of the plan, 0 or 1.
    nodes : int, default=64
        The Gauss--Hermite nodes over ``discharge_risk``.

    Returns
    -------
    float
        :math:`E[Y_{a_1 a_2}]`, by enumeration over ``lives_alone`` and quadrature over
        ``discharge_risk``.

    Examples
    --------
    >>> from cleverly.datasets import team_readmission_truth
    >>> round(team_readmission_truth(1.0, 1.0) - team_readmission_truth(0.0, 0.0), 4)
    -0.1301
    """
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    total = 0.0
    for alone, mass in ((1.0, _LAW["lives_alone"]), (0.0, 1.0 - _LAW["lives_alone"])):
        early = _early(alone, points, offer)
        late = _late(alone, points, offer, call)
        total += mass * float(np.sum(weights * (early + (1.0 - early) * late)))
    return total


def make_team_readmission(
    n_teams: int = 100,
    team_size: int = 40,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Draw the team-run discharge program: a discharge offer, a day-30 call, and dropout.

    Parameters
    ----------
    n_teams : int, default=100
        The number of navigator teams.
    team_size : int, default=40
        The patients of each team.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per patient, with the columns ``team``, ``lives_alone``, ``discharge_risk``,
        ``offer``, ``followed_30``, ``readmitted_30``, ``call``, ``followed_90`` and
        ``readmitted_90`` in time order.
    truth : dict of str to float
        ``ey_regimen[...]`` for the plans ``always``, ``never`` and ``offer_only``, and
        ``ate_regimen[always vs never]``, under the names a fit reports them by.

    See Also
    --------
    cleverly.datasets.team_readmission_protocol : The study description of the program.
    cleverly.datasets.make_longitudinal : The two-decision law without teams.

    Examples
    --------
    >>> from cleverly.datasets import make_team_readmission
    >>> frame, truth = make_team_readmission(n_teams=4, team_size=10, seed=0)
    >>> frame.shape
    (40, 9)
    >>> round(truth["ate_regimen[always vs never]"], 4)
    -0.1301
    """
    if n_teams < 1 or team_size < 1:
        raise ValueError("n_teams and team_size must be positive")
    rng = np.random.default_rng(seed)
    n = n_teams * team_size
    team = np.arange(n) // team_size
    alone = rng.binomial(1, _LAW["lives_alone"], n).astype(float)
    risk = rng.standard_normal(n)
    offer = rng.binomial(
        1,
        expit(_LAW["offer_intercept"] + _LAW["offer_alone"] * alone + _LAW["offer_risk"] * risk),
    ).astype(float)
    followed_30 = rng.binomial(
        1,
        expit(
            _LAW["followed_30_intercept"]
            + _LAW["followed_30_risk"] * risk
            + _LAW["followed_30_offer"] * offer
        ),
    ).astype(float)
    early = rng.binomial(1, _early(alone, risk, offer)).astype(float)
    call = rng.binomial(
        1,
        expit(
            _LAW["call_intercept"]
            + _LAW["call_risk"] * risk
            + _LAW["call_offer"] * offer
            + _LAW["call_early"] * early
        ),
    ).astype(float)
    stays = rng.binomial(
        1,
        expit(
            _LAW["followed_90_intercept"]
            + _LAW["followed_90_risk"] * risk
            + _LAW["followed_90_call"] * call
        ),
    ).astype(float)
    followed_90 = np.where(early == 1.0, 1.0, stays)

    latent = rng.choice((-1.0, 1.0), size=(n_teams, 2))
    shared = latent[team, offer.astype(int)]
    late = _late(alone, risk, offer, call)
    mean = late + TEAM_DELTA * shared * np.minimum(late, 1.0 - late)
    readmitted = np.where(early == 1.0, 1.0, rng.binomial(1, mean).astype(float))

    seen_30 = followed_30 == 1.0
    seen_90 = seen_30 & (followed_90 == 1.0)
    payload = {
        "team": team.astype(float),
        "lives_alone": alone,
        "discharge_risk": risk,
        "offer": offer,
        "followed_30": followed_30,
        "readmitted_30": np.where(seen_30, early, np.nan),
        "call": np.where(seen_30, call, np.nan),
        "followed_90": np.where(seen_30, followed_90, np.nan),
        "readmitted_90": np.where(seen_90, readmitted, np.nan),
    }
    truth = {
        f"ey_regimen[{label}]": team_readmission_truth(*plan) for label, plan in _PLANS.items()
    }
    truth["ate_regimen[always vs never]"] = truth["ey_regimen[always]"] - truth["ey_regimen[never]"]
    return frame_from_dict(payload, backend=backend), truth


def team_readmission_protocol() -> StudyProtocol:
    """Return the study description of the team-run discharge program.

    Returns
    -------
    StudyProtocol
        :func:`~cleverly.datasets.longitudinal_navigation_protocol` with the day-30 call as the
        second decision and readmission by day 90 as the outcome.

    See Also
    --------
    cleverly.datasets.make_team_readmission : The simulated program this description fits.

    Examples
    --------
    >>> from cleverly.datasets import team_readmission_protocol
    >>> team_readmission_protocol().horizon
    '90 days after discharge'
    """
    program = longitudinal_navigation_protocol()
    return replace(
        program,
        treatment_strategies=(
            "Offer navigation at discharge and make the day-30 follow-up call",
            program.treatment_strategies[1],
        ),
        treatment_versions=(
            "The declared discharge navigation contact and the day-30 call by the patient's team",
            "Usual discharge support without navigation contacts or a day-30 call",
        ),
        outcome="Unplanned readmission between discharge and day 90",
        horizon="90 days after discharge",
        intercurrent_event_handling=(
            "Count a readmission before day 30 as the outcome, and keep it in the history of the "
            "day-30 call",
            "Treat loss to follow-up as censoring, so the target is the readmission share had "
            "every patient stayed in follow-up (hypothetical strategy)",
            "Count death before day 90 as a readmission (composite strategy)",
            program.intercurrent_event_handling[2],
        ),
    )

r"""A weekly-offer campaign with an order recorded every week, its exact truths, and its protocol.

A subscription retailer decides each week, for three weeks, whether to send a customer a
free-shipping code.  The retailer's scoring system favours customers who did not order the week
before and who are not engaged, so last week's order drives this week's offer.  The outcome is
recorded every week: did the customer place an order that week?  It can return to zero, so it is a
repeated outcome, not an event.  Customers who withdraw tracking consent are censored.

The module holds the law, the plans of the tutorial ``docs/examples/weekly-outcomes.ipynb``, the
exact truths, and the protocol.  The tutorial, the tests, and the truths read one definition of
each plan.

The truths are computed by **enumeration, not simulation**.  The baseline tenure band takes three
values and every later node is binary, so a counterfactual mean is a finite sum over paths.
``tests/unit/test_datasets_weekly_orders.py`` compares each truth with a Monte Carlo draw of the
intervened law and rejects deliberately wrong truths.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np

from .._typing import Backend, FloatArray
from ..protocol import StudyProtocol
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict
from .offers import offers_protocol

__all__ = [
    "ORDER_PLANS",
    "ORDER_TILTS",
    "ORDER_WEEKS",
    "make_weekly_orders",
    "offer_after_no_order",
    "weekly_orders_protocol",
    "weekly_orders_truth",
]

#: The number of weekly decisions.  The order of each week is recorded after its decision.
ORDER_WEEKS = 3

#: The tenure bands, each with probability one third: under a year, one to two years, longer.
_TENURE = (0.0, 1.0, 2.0)

#: The coefficients of the law, each on the logit scale.
_LAW: dict[str, float] = {
    # an order in week 0, the week before the campaign, given tenure
    "y0_intercept": -0.4,
    "y0_tenure": 0.4,
    # engagement in week 1, given tenure and the week-0 order
    "l1_intercept": -0.3,
    "l1_tenure": 0.4,
    "l1_order": 0.8,
    # engagement in week t >= 2, given tenure and last week's engagement, offer and order
    "l_intercept": -0.7,
    "l_tenure": 0.3,
    "l_engaged": 1.1,
    "l_offer": 0.7,
    "l_order": 0.8,
    # the scoring system: the offer in week t, given tenure, this week's engagement, and last
    # week's order and offer.  Last week's order lowers the chance of an offer sharply.
    "a_intercept": 0.8,
    "a_tenure": 0.1,
    "a_engaged": -0.4,
    "a_order": -1.4,
    "a_offer": 0.2,
    # staying tracked after the send of week t, given tenure, engagement and the offer
    "c_intercept": 2.8,
    "c_tenure": 0.2,
    "c_engaged": 0.4,
    "c_offer": -0.5,
    # an order in week t, given tenure, engagement, the offer, last week's order, and an offer in
    # two weeks running (the second code in a row works less well)
    "y_intercept": -1.5,
    "y_tenure": 0.3,
    "y_engaged": 1.0,
    "y_offer": 0.9,
    "y_order": 1.2,
    "y_repeat": -0.7,
}


def offer_after_no_order(previous_order: Any) -> Any:
    """Return the win-back offer: a code for a customer who did not order last week.

    Parameters
    ----------
    previous_order : array-like of float
        Last week's order, ``1`` for an order and ``0`` for none.  In week 1 it is the order of
        week 0, the week before the campaign.

    Returns
    -------
    ndarray of float
        ``1.0`` (send the code) where ``previous_order`` is ``0``, and ``0.0`` elsewhere.

    See Also
    --------
    ORDER_PLANS : The three plans of the weekly-outcomes tutorial.

    Examples
    --------
    >>> from cleverly.datasets import offer_after_no_order
    >>> offer_after_no_order([0, 1, 0]).tolist()
    [1.0, 0.0, 1.0]
    """
    return 1.0 - np.asarray(previous_order, dtype=float)


def _fixed(level: float) -> Callable[[Any], Any]:
    """Return the plan that sends ``level`` every week."""

    def plan(previous_order: Any) -> Any:
        return np.full(np.shape(previous_order), level, dtype=float)

    plan.__name__ = "always" if level else "never"
    return plan


#: The three plans of the tutorial, each a rule from last week's order to this week's offer.  The
#: key is the name a fit reports.
ORDER_PLANS: dict[str, Callable[[Any], Any]] = {
    "never": _fixed(0.0),
    "always": _fixed(1.0),
    "win_back": offer_after_no_order,
}

#: The odds tilts of the tutorial: the name a fit reports, and the factor on the odds of an offer.
#: A factor of one keeps the scoring system, the natural course.
ORDER_TILTS: dict[str, float] = {"natural course": 1.0, "odds x2": 2.0}


def _order_week0(tenure: Any) -> Any:
    return expit(_LAW["y0_intercept"] + _LAW["y0_tenure"] * np.asarray(tenure))


def _engaged(week: int, tenure: Any, engaged: Any, offer: Any, order: Any) -> Any:
    """The chance of engagement in ``week``, given last week's engagement, offer and order."""
    if week == 1:
        return expit(_LAW["l1_intercept"] + _LAW["l1_tenure"] * tenure + _LAW["l1_order"] * order)
    return expit(
        _LAW["l_intercept"]
        + _LAW["l_tenure"] * tenure
        + _LAW["l_engaged"] * engaged
        + _LAW["l_offer"] * offer
        + _LAW["l_order"] * order
    )


def _offered(tenure: Any, engaged: Any, order: Any, offer: Any) -> Any:
    """The scoring system: the chance of an offer, given this week's engagement and last week."""
    return expit(
        _LAW["a_intercept"]
        + _LAW["a_tenure"] * tenure
        + _LAW["a_engaged"] * engaged
        + _LAW["a_order"] * order
        + _LAW["a_offer"] * offer
    )


def _tracked(tenure: Any, engaged: Any, offer: Any) -> Any:
    return expit(
        _LAW["c_intercept"]
        + _LAW["c_tenure"] * tenure
        + _LAW["c_engaged"] * engaged
        + _LAW["c_offer"] * offer
    )


def _ordered(tenure: Any, engaged: Any, offer: Any, order: Any, last_offer: Any) -> Any:
    return expit(
        _LAW["y_intercept"]
        + _LAW["y_tenure"] * tenure
        + _LAW["y_engaged"] * engaged
        + _LAW["y_offer"] * offer
        + _LAW["y_order"] * order
        + _LAW["y_repeat"] * offer * last_offer
    )


#: The chance of an offer under a plan, from ``(tenure, engaged, last order, last offer)``.
_OfferChance = Callable[[float, float, float, float], float]


def _plan_chance(rule: Callable[[Any], Any]) -> _OfferChance:
    def chance(tenure: float, engaged: float, order: float, offer: float) -> float:
        return float(np.asarray(rule(np.array([order])))[0])

    return chance


def _tilt_chance(delta: float) -> _OfferChance:
    def chance(tenure: float, engaged: float, order: float, offer: float) -> float:
        base = float(_offered(tenure, engaged, order, offer))
        return delta * base / (delta * base + 1.0 - base)

    return chance


def _curve(chance: _OfferChance) -> tuple[float, ...]:
    r"""Return :math:`E[Y_h]` for ``h = 1, 2, 3`` under the offer chance ``chance``, by enumeration.

    The state carried from week to week is last week's engagement, offer and order, given tenure.
    Tracking loss is absent under the intervention (the hypothetical strategy).
    """
    curve = np.zeros(ORDER_WEEKS)
    for tenure in _TENURE:
        first = float(_order_week0(tenure))
        state = {(0.0, 0.0, 0.0): 1.0 - first, (0.0, 0.0, 1.0): first}
        for week in range(1, ORDER_WEEKS + 1):
            following: dict[tuple[float, float, float], float] = {}
            for (engaged_last, offer_last, order_last), mass in state.items():
                p_engaged = float(_engaged(week, tenure, engaged_last, offer_last, order_last))
                for engaged, p_l in ((0.0, 1.0 - p_engaged), (1.0, p_engaged)):
                    p_offer = chance(tenure, engaged, order_last, offer_last)
                    for offer, p_a in ((0.0, 1.0 - p_offer), (1.0, p_offer)):
                        weight = mass * p_l * p_a
                        if weight == 0.0:
                            continue
                        p_order = float(_ordered(tenure, engaged, offer, order_last, offer_last))
                        curve[week - 1] += weight * p_order / len(_TENURE)
                        for order, p_y in ((0.0, 1.0 - p_order), (1.0, p_order)):
                            key = (engaged, offer, order)
                            following[key] = following.get(key, 0.0) + weight * p_y
            state = following
    return tuple(float(value) for value in curve)


@cache
def _truth_table() -> tuple[tuple[str, float], ...]:
    table: dict[str, float] = {}
    for reference, curves in (
        ("never", {name: _curve(_plan_chance(rule)) for name, rule in ORDER_PLANS.items()}),
        ("natural course", {name: _curve(_tilt_chance(d)) for name, d in ORDER_TILTS.items()}),
    ):
        for name, curve in curves.items():
            for week, value in enumerate(curve, start=1):
                table[f"ey_regimen[{name} @ t={week}]"] = value
        for name, curve in curves.items():
            if name == reference:
                continue
            for week, value in enumerate(curve, start=1):
                table[f"ate_regimen[{name} vs {reference} @ t={week}]"] = (
                    value - curves[reference][week - 1]
                )
    return tuple(table.items())


def weekly_orders_truth() -> dict[str, float]:
    """Return the exact weekly order share under each plan and tilt, and each contrast.

    Returns
    -------
    dict of str to float
        ``ey_regimen[<plan> @ t=<week>]`` for each key of :data:`ORDER_PLANS` and
        :data:`ORDER_TILTS` and each week from 1 to 3; ``ate_regimen[<plan> vs never @ t=<week>]``
        for the plans other than ``never``; and ``ate_regimen[odds x2 vs natural course @
        t=<week>]``.  These are the labels a fit reports.

    See Also
    --------
    make_weekly_orders : The generator whose law these truths enumerate.
    ORDER_PLANS : The plans this function evaluates.

    Notes
    -----
    Under a plan or a tilt, tracking loss is removed, and the offer of each week is drawn from the
    plan's chance given the customer's path so far.  A tilt by a factor :math:`\\delta` multiplies
    the odds of the scoring system's chance :math:`\\pi` of an offer, so the offer chance is
    :math:`\\delta \\pi / (\\delta \\pi + 1 - \\pi)`.

    Examples
    --------
    >>> from cleverly.datasets import weekly_orders_truth
    >>> truth = weekly_orders_truth()
    >>> round(truth["ate_regimen[always vs never @ t=1]"], 3)
    0.178
    >>> round(truth["ate_regimen[odds x2 vs natural course @ t=3]"], 3)
    0.013
    """
    return dict(_truth_table())


def make_weekly_orders(
    n: int,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""Draw a three-week offer campaign with a weekly order, as a wide frame with exact truths.

    Each week the retailer's scoring system decides whether to send the customer a free-shipping
    code, and the order of that week is recorded.  The frame has one row per customer.

    Parameters
    ----------
    n : int
        Number of customers.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    frame : dataframe
        A wide frame with the columns ``tenure``, ``order_0``, and, for each week ``t`` from 1 to
        3 in time order, ``engaged_t``, ``offer_t``, ``tracked_t`` and ``order_t``.
        ``tracked_t = 0`` sets ``order_t`` and every later column to missing.
    truth : dict of str to float
        The result of :func:`weekly_orders_truth`.

    See Also
    --------
    weekly_orders_truth : The exact truths this function returns.
    weekly_orders_protocol : The protocol of the question the frame answers.
    cleverly.Repeated : The outcome declaration that reads ``order_1`` to ``order_3``.

    Notes
    -----
    The law, for weeks :math:`t = 1, 2, 3`, with :math:`L_t` engagement, :math:`A_t` the offer,
    :math:`C_t` staying tracked, and :math:`Y_t` the order of week :math:`t`:

    ======================== ================================================================
    variable                 law
    ======================== ================================================================
    ``tenure`` :math:`W`     0, 1 or 2 (under a year, one to two years, longer), each 1/3
    ``order_0`` :math:`Y_0`  Bernoulli, logit :math:`-0.4 + 0.4 W`
    ``engaged_1``            Bernoulli, logit :math:`-0.3 + 0.4 W + 0.8 Y_0`
    ``engaged_t``, t >= 2    Bernoulli, logit :math:`-0.7 + 0.3 W + 1.1 L_{t-1} + 0.7 A_{t-1}
                             + 0.8 Y_{t-1}`
    ``offer_t``              Bernoulli, logit :math:`0.8 + 0.1 W - 0.4 L_t - 1.4 Y_{t-1} +
                             0.2 A_{t-1}`
    ``tracked_t``            Bernoulli, logit :math:`2.8 + 0.2 W + 0.4 L_t - 0.5 A_t`
    ``order_t``              Bernoulli, logit :math:`-1.5 + 0.3 W + 1.0 L_t + 0.9 A_t +
                             1.2 Y_{t-1} - 0.7 A_t A_{t-1}`
    ======================== ================================================================

    :math:`A_0 = L_0 = 0`.  Last week's order lowers the chance of an offer and raises the chance
    of an order, so it confounds each later offer.  Tracking loss reads the week's record and
    offer, never the week's order, so it is missing at random given the history.

    Examples
    --------
    >>> from cleverly.datasets import make_weekly_orders
    >>> frame, truth = make_weekly_orders(200, seed=0)
    >>> list(frame.columns)[:6]
    ['tenure', 'order_0', 'engaged_1', 'offer_1', 'tracked_1', 'order_1']
    >>> round(truth["ate_regimen[win_back vs never @ t=2]"], 3)
    0.089
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, len(_TENURE), size=n).astype(float)
    order = (rng.random(n) < _order_week0(tenure)).astype(float)
    payload: dict[str, FloatArray] = {"tenure": tenure, "order_0": order.copy()}
    engaged = np.zeros(n)
    offer = np.zeros(n)
    observed = np.ones(n, dtype=bool)
    for week in range(1, ORDER_WEEKS + 1):
        engaged_now = (rng.random(n) < _engaged(week, tenure, engaged, offer, order)).astype(float)
        offer_now = (rng.random(n) < _offered(tenure, engaged_now, order, offer)).astype(float)
        stays = (rng.random(n) < _tracked(tenure, engaged_now, offer_now)).astype(float)
        order_now = (rng.random(n) < _ordered(tenure, engaged_now, offer_now, order, offer)).astype(
            float
        )
        payload[f"engaged_{week}"] = np.where(observed, engaged_now, np.nan)
        payload[f"offer_{week}"] = np.where(observed, offer_now, np.nan)
        payload[f"tracked_{week}"] = np.where(observed, stays, np.nan)
        observed = observed & (stays == 1.0)
        payload[f"order_{week}"] = np.where(observed, order_now, np.nan)
        engaged, offer, order = engaged_now, offer_now, order_now
    return frame_from_dict(payload, backend=backend), weekly_orders_truth()


def weekly_orders_protocol() -> StudyProtocol:
    """Return the protocol of the weekly-orders campaign.

    The record starts from :func:`~cleverly.datasets.offers_protocol` and changes the fields the
    weekly outcome and the free-shipping offer change.

    Returns
    -------
    StudyProtocol
        The protocol that compares three offer plans and an odds tilt over three weekly sends, on
        the order of each week.

    See Also
    --------
    make_weekly_orders : The synthetic law that draws data for this protocol.
    cleverly.datasets.offers_protocol : The protocol this record starts from.

    Examples
    --------
    >>> from cleverly.datasets import weekly_orders_protocol
    >>> protocol = weekly_orders_protocol()
    >>> len(protocol.treatment_strategies)
    5
    >>> protocol.horizon
    'Weeks 1 to 3, each week ending after its send'
    """
    base = offers_protocol()
    return replace(
        base,
        eligibility=(
            *base.eligibility[:4],
            "A subscription term that is fixed at the first weekly send and runs through week 3",
        ),
        treatment_strategies=(
            "Send no code in weeks 1 to 3",
            "Send the free-shipping code every week",
            "Win-back: send the code in a week only when the customer did not order the week "
            "before",
            "Keep the scoring system's chance of a code each week",
            "Double the odds of a code under the scoring system each week",
        ),
        treatment_versions=(
            "No code sent",
            "The declared free-shipping code, one send per week, valid until the end of that week",
            "The declared free-shipping code after a week without an order, no code otherwise",
            "The scoring system's draw of the declared code",
            "A draw of the declared code with the odds of the scoring system's chance doubled",
        ),
        outcome="An order in each week, recorded weekly; a customer can order in any week",
        horizon="Weeks 1 to 3, each week ending after its send",
        intercurrent_event_handling=(
            "Tracking-consent withdrawal censors the customer from that week on. The strategy is "
            "hypothetical: the orders had tracking continued",
            "Each code expires at the end of its week",
            "None: the term runs through week 3, so no account closes before the last week",
        ),
        assumption_rationale=(
            "The scoring system reads tenure, the week's engagement score, and last week's order "
            "and code, all recorded, which gives sequential exchangeability for the codes",
            "Tenure and weekly engagement are the recorded causes of consent loss",
            *base.assumption_rationale[2:],
        ),
    )

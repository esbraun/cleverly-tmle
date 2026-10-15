r"""A weekly-offer campaign with a known truth, and the protocol that states its question.

A subscription retailer sends one offer per week for three weeks: no offer, a 15% discount code,
or a free-shipping code.  The CRM draws each offer at random, with logged probabilities that
depend on the customer's engagement that week.  Engagement responds to the earlier offers, and it
drives both the later offers and the week-4 outcome, a full-price order.  So engagement is a
time-varying confounder affected by earlier treatment, and the offer mechanism is known.  Customers
who withdraw tracking consent are censored.

The module holds the law, the plans of the tutorial, the exact truth, and the protocol.  The
tutorial notebook ``docs/examples/sequential-offers.ipynb``, the tests, and the truth read one
definition of each plan.

The truth is computed by **enumeration and quadrature, not simulation**.  Under a deterministic
plan the offers follow from the engagement path, and consent loss is absent under the
intervention (the hypothetical strategy), so what is left is a sum over the sixteen engagement
paths of Gauss--Hermite integrals over tenure.  ``tests/unit/test_datasets_offers.py`` compares
that sum with a Monte Carlo draw of the intervened law, and rejects three wrong truths.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np

from .._typing import Backend, FloatArray
from ..protocol import StudyProtocol
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = [
    "OFFER_LEVELS",
    "OFFER_PLANS",
    "OFFER_WEEKS",
    "make_sequential_offers",
    "offers_protocol",
    "offers_truth",
    "win_back",
    "win_back_then_ship",
]

#: The offer levels, in the order of the logged-probability columns.
OFFER_LEVELS = ("none", "discount", "shipping")

#: The number of weekly sends.  The outcome is the full-price order in the week after the last.
OFFER_WEEKS = 3

#: The logged probabilities of ``(none, discount, shipping)``, by the engagement of that week.
#: The smallest is 0.25, so the cumulative product over three weeks is at least 0.0156, above the
#: default ``g_bounds`` lower bound of 0.01.
_LOGGED = {
    1: (0.40, 0.25, 0.35),
    0: (0.30, 0.45, 0.25),
}

#: The coefficients of the law.  Each is a logit-scale coefficient except where named otherwise.
_LAW = {
    # engagement in week 1: expit(e1_intercept + e1_tenure * tenure)
    "e1_intercept": 0.2,
    "e1_tenure": 0.5,
    # engagement in week t + 1, given engagement and the offer of week t
    "next_intercept": -0.4,
    "next_engaged": 1.0,
    "next_discount": 2.2,
    "next_shipping": 0.5,
    "next_tenure": 0.3,
    # staying under consent after the week-t decision
    "consent_intercept": 2.6,
    "consent_engaged": 0.8,
    "consent_tenure": 0.3,
    # a full-price order in week 4, given week-4 engagement, tenure and the fatigue count
    "y_intercept": -0.9,
    "y_engaged": 2.0,
    "y_tenure": 0.4,
    "y_fatigue": 0.5,
    "y_interaction": 0.5,
}


def win_back(engaged: Any) -> Any:
    """Return the win-back offer: the discount for a customer who is not engaged, else nothing.

    Parameters
    ----------
    engaged : array-like of int
        Engagement this week, ``1`` for engaged and ``0`` for not engaged.

    Returns
    -------
    ndarray of str
        ``"discount"`` where ``engaged`` is ``0`` and ``"none"`` elsewhere.

    See Also
    --------
    win_back_then_ship : The same rule with free shipping for engaged customers.
    OFFER_PLANS : All five plans of the offers tutorial.

    Examples
    --------
    >>> from cleverly.datasets import win_back
    >>> win_back([0, 1, 0]).tolist()
    ['discount', 'none', 'discount']
    """
    return np.where(np.asarray(engaged) == 0, "discount", "none")


def win_back_then_ship(engaged: Any) -> Any:
    """Return the win-back-then-ship offer: the discount if not engaged, else free shipping.

    Parameters
    ----------
    engaged : array-like of int
        Engagement this week, ``1`` for engaged and ``0`` for not engaged.

    Returns
    -------
    ndarray of str
        ``"discount"`` where ``engaged`` is ``0`` and ``"shipping"`` elsewhere.

    See Also
    --------
    win_back : The same rule with no offer for engaged customers.
    OFFER_PLANS : All five plans of the offers tutorial.

    Examples
    --------
    >>> from cleverly.datasets import win_back_then_ship
    >>> win_back_then_ship([0, 1, 0]).tolist()
    ['discount', 'shipping', 'discount']
    """
    return np.where(np.asarray(engaged) == 0, "discount", "shipping")


def _always(level: str) -> Callable[[Any], Any]:
    """Return the plan that sends ``level`` every week."""

    def plan(engaged: Any) -> Any:
        return np.full(np.shape(engaged), level, dtype=object)

    plan.__name__ = f"always_{level}"
    return plan


#: The five plans of the tutorial, each a rule from the week's engagement to the offer level.  The
#: first three are static, and the last two read engagement.  The key is the name a fit reports.
OFFER_PLANS: dict[str, Callable[[Any], Any]] = {
    "none": _always("none"),
    "discount": _always("discount"),
    "shipping": _always("shipping"),
    "win_back": win_back,
    "win_back_ship": win_back_then_ship,
}


def _logged_probabilities(engaged: FloatArray) -> FloatArray:
    """The logged ``(none, discount, shipping)`` probabilities, one row per customer."""
    engaged = np.asarray(engaged)
    return np.where(
        (engaged == 1)[:, None],
        np.asarray(_LOGGED[1])[None, :],
        np.asarray(_LOGGED[0])[None, :],
    )


def _first_engagement(tenure: FloatArray) -> FloatArray:
    return expit(_LAW["e1_intercept"] + _LAW["e1_tenure"] * tenure)


def _next_engagement(engaged: Any, offer: Any, tenure: FloatArray) -> FloatArray:
    offer = np.asarray(offer)
    return expit(
        _LAW["next_intercept"]
        + _LAW["next_engaged"] * np.asarray(engaged)
        + _LAW["next_discount"] * (offer == "discount")
        + _LAW["next_shipping"] * (offer == "shipping")
        + _LAW["next_tenure"] * tenure
    )


def _consent(engaged: Any, tenure: FloatArray) -> FloatArray:
    return expit(
        _LAW["consent_intercept"]
        + _LAW["consent_engaged"] * np.asarray(engaged)
        + _LAW["consent_tenure"] * tenure
    )


def _fatigue_step(engaged: Any, offer: Any) -> Any:
    """One if the offer is a discount sent to an engaged customer, else zero."""
    return (np.asarray(offer) == "discount") & (np.asarray(engaged) == 1)


def _outcome_probability(final: Any, tenure: FloatArray, fatigue: Any) -> FloatArray:
    final = np.asarray(final)
    return expit(
        _LAW["y_intercept"]
        + _LAW["y_engaged"] * final
        + _LAW["y_tenure"] * tenure
        - _LAW["y_fatigue"] * np.asarray(fatigue)
        + _LAW["y_interaction"] * final * tenure
    )


def _plan_mean(
    rule: Callable[[Any], Any],
    nodes: int = 48,
    *,
    read: Callable[[tuple[int, ...], int], int] | None = None,
) -> float:
    r"""Return :math:`E[Y_d]` for the plan ``rule`` by enumeration and quadrature.

    The sixteen engagement paths :math:`(L_1, \dots, L_4)` are summed, and tenure is integrated
    by a ``nodes``-point Gauss--Hermite rule.  ``read`` picks the engagement the rule reads at a
    week from the path so far, and the default is the week's own :math:`L_t`.
    """
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    total = 0.0
    for path in itertools.product((0, 1), repeat=OFFER_WEEKS + 1):
        first = _first_engagement(points)
        probability = first if path[0] == 1 else 1.0 - first
        fatigue = np.zeros_like(points)
        for week in range(OFFER_WEEKS):
            seen = path[week] if read is None else read(path, week)
            offer = np.asarray(rule(np.array([seen])))[0]
            fatigue = fatigue + float(_fatigue_step(path[week], offer))
            step = _next_engagement(path[week], offer, points)
            probability = probability * (step if path[week + 1] == 1 else 1.0 - step)
        total += float(
            np.sum(weights * probability * _outcome_probability(path[-1], points, fatigue))
        )
    return total


@cache
def _truth_table(nodes: int) -> tuple[tuple[str, float], ...]:
    means = {name: _plan_mean(rule, nodes) for name, rule in OFFER_PLANS.items()}
    table = {f"ey_regimen[{name}]": value for name, value in means.items()}
    for name, value in means.items():
        if name != "none":
            table[f"ate_regimen[{name} vs none]"] = value - means["none"]
    return tuple(table.items())


def offers_truth(nodes: int = 48) -> dict[str, float]:
    r"""Return the exact week-4 full-price share under each plan, and each contrast with no offers.

    Under a deterministic plan the offers follow from the engagement path, and the intervention
    removes consent loss.  The mean is a sum over the sixteen engagement paths of an integral over
    tenure, which a Gauss--Hermite rule evaluates.

    Parameters
    ----------
    nodes : int, default=48
        Number of Gauss--Hermite nodes for tenure.

    Returns
    -------
    dict of str to float
        ``ey_regimen[<plan>]`` for each key of :data:`OFFER_PLANS`, and
        ``ate_regimen[<plan> vs none]`` for each plan except ``none``.  These are the labels a fit
        reports.

    See Also
    --------
    make_sequential_offers : The generator whose law this truth integrates.
    OFFER_PLANS : The plans this function evaluates.

    Notes
    -----
    The rule of a plan reads the engagement of the week it acts in, so the enumeration needs no
    offer path beyond the one the rule gives.

    Examples
    --------
    >>> from cleverly.datasets import offers_truth
    >>> truth = offers_truth()
    >>> sorted(truth)[:2]
    ['ate_regimen[discount vs none]', 'ate_regimen[shipping vs none]']
    >>> truth["ate_regimen[win_back vs none]"] > 0.05
    True
    """
    return dict(_truth_table(int(nodes)))


def make_sequential_offers(
    n: int,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
    policy: str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""Draw a weekly-offer campaign as a long panel, with the exact truth of its plans.

    Customers receive an offer in each of three weeks: ``"none"``, ``"discount"``, or
    ``"shipping"``.  The outcome ``full_price`` is a full-price order in week 4.  The frame has one
    row per customer and week while the customer is observed.

    Parameters
    ----------
    n : int
        Number of customers.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.
    policy : str or None, default=None
        ``None`` draws the logged campaign.  A key of :data:`OFFER_PLANS` draws a synthetic
        deployment check: a fresh cohort in which the CRM sends that plan's offers and every
        customer stays tracked.

    Returns
    -------
    panel : dataframe
        A long panel with columns ``customer``, ``week``, ``tenure``, ``engaged``, ``offer``,
        ``consent``, ``p_none``, ``p_discount``, ``p_shipping``, and ``full_price``.
    truth : dict of str to float
        The result of :func:`offers_truth`.

    See Also
    --------
    cleverly.datasets.offers_truth : The exact truth this function returns.
    cleverly.datasets.offers_protocol : The protocol of the question the panel answers.
    cleverly.LongPanel : The design that reads this panel.

    Notes
    -----
    The law, for weeks :math:`t = 1, 2, 3`, with :math:`L_t` the engagement of week :math:`t` and
    :math:`A_t` the offer:

    ============================ ==============================================================
    variable                     law
    ============================ ==============================================================
    ``tenure``                   :math:`N(0, 1)`
    ``engaged`` :math:`L_1`      Bernoulli, logit :math:`0.2 + 0.5\,\text{tenure}`
    ``offer`` :math:`A_t`        logged: engaged (none 0.40, discount 0.25, shipping 0.35); not
                                 engaged (none 0.30, discount 0.45, shipping 0.25)
    ``engaged`` :math:`L_{t+1}`  Bernoulli, logit :math:`-0.4 + 1.0 L_t + 2.2\,[A_t =
                                 \text{discount}] + 0.5\,[A_t = \text{shipping}] +
                                 0.3\,\text{tenure}`
    ``consent``                  Bernoulli, logit :math:`2.6 + 0.8 L_t + 0.3\,\text{tenure}`
    ``full_price``               Bernoulli, logit :math:`-0.9 + 2.0 L_4 + 0.4\,\text{tenure} -
                                 0.5 D + 0.5 L_4\,\text{tenure}`
    ============================ ==============================================================

    :math:`D` counts the discounts sent to engaged customers.  A discount re-engages a customer
    who is not engaged, and it wears on a customer who already is.  So the ordering of the plans is
    not monotone in the number of discounts.  Each code expires at the end of its week, so no code
    is redeemable in week 4.  Account closure is outside the law.

    Engagement :math:`L_4` is the week-4 engagement.  It enters the outcome and no column.

    The panel carries the logged probabilities in ``p_none``, ``p_discount``, and ``p_shipping``,
    so a design can declare them as the known offer mechanism.  A ``policy`` draw sets the
    probability of the deployed offer to one.

    Examples
    --------
    >>> from cleverly.datasets import make_sequential_offers
    >>> panel, truth = make_sequential_offers(200, seed=0)
    >>> list(panel.columns)[:5]
    ['customer', 'week', 'tenure', 'engaged', 'offer']
    >>> int(panel["week"].max())
    3
    >>> sorted(panel["offer"].unique())
    ['discount', 'none', 'shipping']
    >>> round(truth["ate_regimen[win_back vs none]"], 3)
    0.072
    """
    if policy is not None and policy not in OFFER_PLANS:
        raise ValueError(f"policy must be None or one of {sorted(OFFER_PLANS)}, got {policy!r}")
    rng = np.random.default_rng(seed)
    tenure = rng.standard_normal(n)
    engaged = (rng.random(n) < _first_engagement(tenure)).astype(int)
    observed = np.ones(n, dtype=bool)
    fatigue = np.zeros(n)
    weeks: list[dict[str, Any]] = []
    for week in range(1, OFFER_WEEKS + 1):
        draw_offer, draw_consent, draw_next = rng.random(n), rng.random(n), rng.random(n)
        if policy is None:
            probabilities = _logged_probabilities(engaged)
            index = (draw_offer[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
            index = np.minimum(index, len(OFFER_LEVELS) - 1)
            offer = np.asarray(OFFER_LEVELS, dtype=object)[index]
            consent = (draw_consent < _consent(engaged, tenure)).astype(int)
        else:
            offer = np.asarray(OFFER_PLANS[policy](engaged), dtype=object)
            probabilities = np.stack([(offer == level).astype(float) for level in OFFER_LEVELS], 1)
            consent = np.ones(n, dtype=int)
        weeks.append(
            {
                "customer": np.flatnonzero(observed),
                "week": np.full(int(observed.sum()), week),
                "tenure": tenure[observed],
                "engaged": engaged[observed],
                "offer": offer[observed],
                "consent": consent[observed],
                "p_none": probabilities[observed, 0],
                "p_discount": probabilities[observed, 1],
                "p_shipping": probabilities[observed, 2],
            }
        )
        fatigue = fatigue + _fatigue_step(engaged, offer)
        observed = observed & (consent == 1)
        engaged = (draw_next < _next_engagement(engaged, offer, tenure)).astype(int)
    bought = rng.random(n) < _outcome_probability(engaged, tenure, fatigue)
    outcome = np.where(observed, bought.astype(float), np.nan)
    payload = {name: np.concatenate([week[name] for week in weeks]) for name in weeks[0]}
    order = np.lexsort((payload["week"], payload["customer"]))
    payload = {name: values[order] for name, values in payload.items()}
    payload["full_price"] = outcome[payload["customer"]]
    return frame_from_dict(payload, backend=backend), offers_truth()


def offers_protocol() -> StudyProtocol:
    """Return the protocol of the weekly-offer campaign.

    Returns
    -------
    StudyProtocol
        The protocol that compares five offer plans over three weekly sends, on a full-price order
        in week 4.

    See Also
    --------
    make_sequential_offers : The synthetic law that draws data for this protocol.
    cleverly.StudyProtocol : The record this function fills.
    cleverly.datasets.longitudinal_navigation_protocol : The protocol of the health program.

    Notes
    -----
    The record states time zero, eligibility, the treatment versions, the outcome horizon, and the
    interference unit.  The available history and the confounders are not protocol fields.  The
    assumption rationale names the confounders, and a design prints the history at each node.

    Examples
    --------
    >>> from cleverly.datasets import offers_protocol
    >>> protocol = offers_protocol()
    >>> len(protocol.treatment_strategies)
    5
    >>> protocol.horizon
    'Week 4, the week after the third weekly send'
    """
    return StudyProtocol(
        target_population="Active subscribers at the start of the campaign window",
        eligibility=(
            "Active subscription at the first weekly send",
            "Marketing consent at the first weekly send",
            "At least one prior order",
            "One account per customer, so shared-household accounts are excluded",
            "An account that stays open through week 4, so a closure-free window is a "
            "condition of the target population",
        ),
        time_zero=(
            "The first weekly send, after the week-1 engagement score and before the first offer"
        ),
        treatment_strategies=(
            "Send no offer in weeks 1 to 3",
            "Send the 15% discount code every week",
            "Send the free-shipping code every week",
            "Win-back: send the discount code when the customer is not engaged, otherwise "
            "send no offer",
            "Win-back then ship: send the discount code when the customer is not engaged, "
            "otherwise send the free-shipping code",
        ),
        treatment_versions=(
            "No code sent",
            "The declared 15% code, one send per week, valid until the end of that week",
            "The declared free-shipping code, one send per week, valid until the end of that week",
            "The declared 15% code for customers who are not engaged at the send, no code "
            "otherwise",
            "The declared 15% code for customers who are not engaged at the send, and the "
            "declared free-shipping code otherwise",
        ),
        outcome="A full-price order in week 4, an order placed without a code",
        horizon="Week 4, the week after the third weekly send",
        intercurrent_event_handling=(
            "Tracking-consent withdrawal before week 4 censors the customer. The strategy is "
            "hypothetical: the order had tracking continued",
            "Each code expires at the end of its week, so no code is redeemable in week 4",
            "Account closure before week 4 is outside the target population",
        ),
        interference_unit="Individual customer",
        assumption_rationale=(
            "Randomization with logged probabilities gives sequential exchangeability for the "
            "offers, given tenure and the weekly engagement score",
            "Tenure and weekly engagement are the recorded confounders, and they cover the "
            "measured causes of consent loss",
            "Declared code templates, one send per week, support consistency",
            "One account per customer and per-customer codes support no interference",
        ),
    )

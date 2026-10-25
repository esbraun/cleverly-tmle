r"""A monthly renewal-price program with a known truth, and the protocol that states its question.

A subscription retailer sets a renewal price for each customer in each of three months.  The
pricing engine reads the customer's usage, this month's engagement, and the gap between its last
price and the last month's centre.  It draws a price inside a band that moves with engagement.
Engagement responds to the last price gap, and it drives both the next price and the month-3
renewal.  So engagement is a time-varying confounder affected by earlier prices, and the price
engine carries part of each price into the next one.

The module holds the law, the plans of the tutorial, the exact truth, and the protocol.  The
tutorial notebook ``docs/examples/sequential-pricing.ipynb``, the tests, and the truth read one
definition of each plan.

The truth is computed by **quadrature, not simulation**.  A plan maps the price the engine would
set to a new price.  The mean outcome under the plan is a sum over the eight engagement paths of a
Gauss--Legendre integral over usage and the three natural prices, split at every break of the plan.
The same integral with the gap in the engagement model, in the price model, or in both held at its
natural value gives the decomposition of the effect by path.
``tests/unit/test_datasets_pricing.py`` compares the integral with a Monte Carlo draw of the
intervened law, with the law of the registered study ``mtp-pricing``, and with four wrong truths.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import norm

from .._typing import Backend
from ..interventions import ModifiedPolicy, Piece, Shift, Support
from ..longitudinal import DynamicRegimen
from ..protocol import StudyProtocol
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = [
    "PRICE_BANDS",
    "PRICE_NODES",
    "PRICE_PLANS",
    "make_sequential_pricing",
    "price_rise",
    "pricing_protocol",
    "pricing_truth",
]

#: The number of monthly renewal prices.  The outcome is the renewal at the third price.
PRICE_NODES = 3

#: The reference plan: every customer keeps the price the pricing engine sets.
_REFERENCE = "natural_prices"

#: The factor of the question's price rise and the factor of its price cut.
_RISE = 1.12
_CUT = 0.85

#: The coefficients of the law.  Each is on the raw variable: a logit-scale coefficient for
#: engagement and renewal, and a coefficient on the price mean for the price.  The price gap of a
#: month is the price minus that month's centre, ``centre + centre_engaged * engaged``.
_LAW = {
    # usage: N(0, 1) truncated to [-usage_bound, usage_bound]
    "usage_bound": 2.5,
    # engagement in month t: expit(intercept + usage * usage + gap * last gap + lag * last engaged)
    "engaged_intercept": 0.3,
    "engaged_usage": 0.6,
    "engaged_gap": -0.35,
    "engaged_lag": 0.8,
    # the centre and the band of month t: centre = base + engaged * engaged
    "centre_base": 6.0,
    "centre_engaged": 2.0,
    "band_below": 2.0,
    "band_above": 2.0,
    # the price: N(centre + gap * last gap + usage * usage, sd^2) truncated to the band
    "price_gap": 0.3,
    "price_usage": 0.3,
    "price_sd": 1.5,
    # renewal: expit(intercept + gap * last gap + engaged * month-3 engaged + usage * usage
    #                + lag * sum of the gaps of months 1 and 2)
    "renew_intercept": 0.5,
    "renew_gap": -0.45,
    "renew_engaged": 0.9,
    "renew_usage": 0.3,
    "renew_lag": -0.1,
}

#: The default quadrature orders ``(price, usage)``.  Orders ``(12, 32)`` agree with these to 1e-12.
_NODES = (8, 24)


def _centre(engaged: Any) -> Any:
    return _LAW["centre_base"] + _LAW["centre_engaged"] * np.asarray(engaged, dtype=float)


def _lower(engaged: Any) -> Any:
    return _centre(engaged) - _LAW["band_below"]


def _upper(engaged: Any) -> Any:
    return _centre(engaged) + _LAW["band_above"]


def _engagement(usage: Any, gap: Any, last_engaged: Any) -> Any:
    """``P(engaged_t = 1)``, given the usage, the last price gap and the last engagement."""
    return expit(
        _LAW["engaged_intercept"]
        + _LAW["engaged_usage"] * np.asarray(usage)
        + _LAW["engaged_gap"] * np.asarray(gap)
        + _LAW["engaged_lag"] * np.asarray(last_engaged)
    )


def _price_mean(usage: Any, engaged: Any, gap: Any) -> Any:
    return (
        _centre(engaged)
        + _LAW["price_gap"] * np.asarray(gap)
        + _LAW["price_usage"] * np.asarray(usage)
    )


def _renewal(usage: Any, last_engaged: Any, last_gap: Any, earlier_gaps: Any) -> Any:
    """``P(renewed = 1)``, given the usage, the month-3 engagement, and the price gaps."""
    return expit(
        _LAW["renew_intercept"]
        + _LAW["renew_gap"] * np.asarray(last_gap)
        + _LAW["renew_engaged"] * np.asarray(last_engaged)
        + _LAW["renew_usage"] * np.asarray(usage)
        + _LAW["renew_lag"] * np.asarray(earlier_gaps)
    )


def _mass(mean: Any, low: Any, high: Any) -> Any:
    sd = _LAW["price_sd"]
    return norm.cdf((np.asarray(high) - mean) / sd) - norm.cdf((np.asarray(low) - mean) / sd)


def _density(price: Any, mean: Any, low: Any, high: Any) -> Any:
    """The density of the engine's price: a normal truncated to the band."""
    sd = _LAW["price_sd"]
    price = np.asarray(price, dtype=float)
    inside = (price >= low) & (price <= high)
    return np.where(inside, norm.pdf((price - mean) / sd) / (sd * _mass(mean, low, high)), 0.0)


def _draw_price(rng: np.random.Generator, mean: Any, low: Any, high: Any) -> Any:
    sd = _LAW["price_sd"]
    mean = np.asarray(mean, dtype=float)
    below = norm.cdf((np.asarray(low) - mean) / sd)
    above = norm.cdf((np.asarray(high) - mean) / sd)
    return mean + sd * norm.ppf(below + rng.random(mean.shape) * (above - below))


# ------------------------------------------------------------------------------------ the plans


def _rise_label(factor: float) -> str:
    return "price_up" if factor == _RISE else f"up_{round(100 * (factor - 1))}"


_Map = Callable[[Any, Any], Any]


def _scaled(factor: float) -> tuple[_Map, _Map, _Map]:
    """The map, its inverse, and the derivative of the inverse, for the scale ``factor``."""
    return (
        lambda price, history: factor * np.asarray(price, dtype=float),
        lambda moved, history: np.asarray(moved, dtype=float) / factor,
        lambda moved, history: np.full_like(np.asarray(moved, dtype=float), 1.0 / factor),
    )


def _engaged_column(node: int) -> str:
    return f"engaged_{node}"


def _price_column(node: int) -> str:
    return f"price_{node}"


def _node_policy(kind: str, factor: float, node: int, label: str) -> Any:
    """The plan's policy at ``node``.  It reads ``engaged_<node>``, recorded before the price."""
    column = _engaged_column(node)
    if kind == "natural":
        return Shift(0.0, cap=None)

    def knee_up(history: Any) -> Any:
        return np.asarray(_upper(np.asarray(history[column], dtype=float)) / factor, dtype=float)

    def knee_down(history: Any) -> Any:
        return np.asarray(_lower(np.asarray(history[column], dtype=float)) / factor, dtype=float)

    forward, inverse, derivative = _scaled(factor)
    if kind == "up":
        pieces = (
            Piece(-np.inf, knee_up, forward, inverse, derivative),
            Piece(knee_up, np.inf),
        )
    else:
        pieces = (
            Piece(-np.inf, knee_down),
            Piece(knee_down, np.inf, forward, inverse, derivative),
        )
    return ModifiedPolicy(f"{label} at {node}", pieces=pieces, policy_kind="known")


def _regimen(kind: str, factor: float, label: str) -> DynamicRegimen:
    return DynamicRegimen(
        label,
        tuple(_node_policy(kind, factor, node, label) for node in range(1, PRICE_NODES + 1)),
    )


def price_rise(factor: float, label: str | None = None) -> DynamicRegimen:
    r"""Return the plan that raises each price by ``factor`` and keeps the price in the band.

    At each month the plan multiplies the engine's price :math:`a` by ``factor`` where the raised
    price stays below that month's band ceiling :math:`u_t`, that is, where
    :math:`a < u_t / \text{factor}`.  Elsewhere it keeps the engine's price.  The map is piecewise
    smooth and invertible, so the plan is a modified treatment policy.  A ceiling at :math:`u_t`
    would put a point mass there, and its map would not be invertible.

    Parameters
    ----------
    factor : float
        The price factor, greater than 1.
    label : str or None, default=None
        The name a fit reports for the plan.  ``None`` gives ``"price_up"`` for the factor 1.12
        and ``"up_<percent>"`` for any other factor, such as ``"up_25"`` for 1.25.

    Returns
    -------
    DynamicRegimen
        One policy for each of the three months, each reading ``engaged_<month>``.

    See Also
    --------
    PRICE_PLANS : The three plans of the tutorial.
    PRICE_BANDS : The band each month's price must stay in.
    pricing_truth : The exact truth of a plan with this factor.

    Examples
    --------
    >>> from cleverly.datasets import price_rise
    >>> plan = price_rise(1.25)
    >>> plan.label
    'up_25'
    >>> len(plan.plan)
    3
    """
    factor = float(factor)
    if not factor > 1.0:
        raise ValueError(f"a price rise needs a factor above 1, got {factor!r}")
    return _regimen("up", factor, label or _rise_label(factor))


def _band(column: str) -> Support:
    """The band ``[4 + 2 e, 8 + 2 e]`` of a price, where ``e`` is the engagement in ``column``."""

    def lower(history: Any) -> Any:
        return _lower(np.asarray(history[column], dtype=float))

    def upper(history: Any) -> Any:
        return _upper(np.asarray(history[column], dtype=float))

    return Support(lower, upper)


#: The band each month's price takes, keyed by the price column: ``[4 + 2 e, 8 + 2 e]`` for the
#: month's engagement ``e``.  Pass it as ``continuous_treatment=`` to the design.
PRICE_BANDS: dict[str, Support] = {
    _price_column(node): _band(_engaged_column(node)) for node in range(1, PRICE_NODES + 1)
}

#: The three plans of the tutorial, by the name a fit reports: the engine's own prices, a 12% rise
#: that keeps the engine's price where the rise would leave the band, and a 15% cut that does the
#: same where the cut would leave it.
PRICE_PLANS: dict[str, DynamicRegimen] = {
    _REFERENCE: _regimen("natural", 1.0, _REFERENCE),
    "price_up": price_rise(_RISE),
    "price_down": _regimen("down", _CUT, "price_down"),
}


# ------------------------------------------------------------------------------------ the truth


class _Plan(NamedTuple):
    """A plan for the quadrature: ``kind`` is ``natural``, ``up`` or ``down``."""

    kind: str
    factor: float

    def assign(self, price: Any, engaged: Any) -> Any:
        price = np.asarray(price, dtype=float)
        if self.kind == "up":
            return np.where(price < _upper(engaged) / self.factor, self.factor * price, price)
        if self.kind == "down":
            return np.where(price < _lower(engaged) / self.factor, price, self.factor * price)
        return price

    def breaks(self, engaged: float) -> list[float]:
        """The band ends, the breaks of the question's two plans, and the break of this plan.

        Every plan splits at the same default breaks, so the contrast of two plans cancels the
        quadrature error of the shared pieces.
        """
        low, high = float(_lower(engaged)), float(_upper(engaged))
        points = {low, high, high / _RISE, low / _CUT}
        if self.kind == "up":
            points.add(high / self.factor)
        elif self.kind == "down":
            points.add(low / self.factor)
        return sorted(points)


def _legendre(order: int, low: Any, high: Any) -> tuple[Any, Any]:
    base, weights = np.polynomial.legendre.leggauss(order)
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    half = 0.5 * (high - low)
    return (low + half)[..., None] + half[..., None] * base, half[..., None] * weights


def _price_nodes(plan: _Plan, engaged: float, order: int) -> tuple[Any, Any]:
    nodes, weights = [], []
    for low, high in itertools.pairwise(plan.breaks(engaged)):
        x, w = _legendre(order, low, high)
        nodes.append(x)
        weights.append(w)
    return np.concatenate(nodes), np.concatenate(weights)


def _mean_outcomes(
    plan: _Plan,
    orders: tuple[int, int],
    *,
    engagement_held: bool,
    engine_held: bool,
    horizon: int = PRICE_NODES,
) -> tuple[float, float]:
    """``(E[renewed], E[revenue])`` under the plan, by quadrature, with ``horizon`` prices.

    ``engagement_held`` reads the natural price gap in the engagement model, and ``engine_held``
    reads it in the next price mean.  The renewal model reads the assigned gaps in every case.
    The tests call the integral at ``horizon=2`` to compare it with the registered study.
    """
    price_order, usage_order = orders
    bound = _LAW["usage_bound"]
    usage, usage_weights = _legendre(usage_order, -bound, bound)
    usage_weights = usage_weights * norm.pdf(usage) / (norm.cdf(bound) - norm.cdf(-bound))
    totals = [0.0, 0.0]

    def visit(
        node: int,
        last_engaged: float,
        mass: Any,
        u: Any,
        assigned_gap: Any,
        natural_gap: Any,
        earlier: Any,
    ) -> None:
        engagement_gap = natural_gap if engagement_held else assigned_gap
        price_gap = natural_gap if engine_held else assigned_gap
        chance = _engagement(u, engagement_gap, last_engaged)
        for engaged in (0.0, 1.0):
            reached = mass * (chance if engaged else 1.0 - chance)
            price, weight = _price_nodes(plan, engaged, price_order)
            density = _density(
                price,
                _price_mean(u[..., None], engaged, price_gap[..., None]),
                _lower(engaged),
                _upper(engaged),
            )
            assigned = plan.assign(price, engaged)
            joint = reached[..., None] * weight * density
            if node < horizon:
                visit(
                    node + 1,
                    engaged,
                    joint,
                    u[..., None],
                    assigned - _centre(engaged),
                    price - _centre(engaged),
                    earlier[..., None] + (assigned - _centre(engaged)),
                )
            else:
                renewal = _renewal(
                    u[..., None], engaged, assigned - _centre(engaged), earlier[..., None]
                )
                totals[0] += float(np.sum(joint * renewal))
                totals[1] += float(np.sum(joint * renewal * assigned))

    zero = np.zeros_like(usage)
    visit(1, 0.0, usage_weights, usage, zero, zero, zero)
    return totals[0], totals[1]


def _factor_plans(factors: Iterable[float]) -> dict[str, _Plan]:
    plans = {
        _REFERENCE: _Plan("natural", 1.0),
        "price_up": _Plan("up", _RISE),
        "price_down": _Plan("down", _CUT),
    }
    for factor in factors:
        plans[_rise_label(float(factor))] = _Plan("up", float(factor))
    return plans


@cache
def _truth_table(
    factors: tuple[float, ...], nodes: tuple[int, int]
) -> tuple[tuple[str, float], ...]:
    plans = _factor_plans(factors)
    free = {
        label: _mean_outcomes(plan, nodes, engagement_held=False, engine_held=False)
        for label, plan in plans.items()
    }
    table: dict[str, float] = {}
    for label, (renew, revenue) in free.items():
        table[f"ey_regimen[{label}]"] = renew
        table[f"ey_revenue[{label}]"] = revenue
    for label, (renew, revenue) in free.items():
        if label != _REFERENCE:
            table[f"ate_regimen[{label} vs {_REFERENCE}]"] = renew - free[_REFERENCE][0]
            table[f"ate_revenue[{label} vs {_REFERENCE}]"] = revenue - free[_REFERENCE][1]
    held = {
        "ate_engagement_held": (True, False),
        "ate_engine_held": (False, True),
        "ate_both_held": (True, True),
    }
    for name, (engagement_held, engine_held) in held.items():
        for label in ("price_up", "price_down"):
            renew, _ = _mean_outcomes(
                plans[label], nodes, engagement_held=engagement_held, engine_held=engine_held
            )
            table[f"{name}[{label}]"] = renew - free[_REFERENCE][0]
    return tuple(table.items())


def pricing_truth(
    factors: Iterable[float] = (1.06, 1.25), nodes: tuple[int, int] = _NODES
) -> dict[str, float]:
    r"""Return the exact mean renewal and revenue under each plan, and the effect of each plan.

    A plan maps the price the engine would set to a new price.  The mean outcome under the plan
    is a sum over the eight engagement paths of a Gauss--Legendre integral over usage and the three
    natural prices.  The integral splits each price band at every break of the plan.  The
    engagement and the engine's next price respond to the assigned prices, as the law has them.

    Parameters
    ----------
    factors : iterable of float, default=(1.06, 1.25)
        The factors of the extra price rises, beside the plans of :data:`PRICE_PLANS`.  A factor
        gives the plan ``price_rise(factor)``.
    nodes : tuple of int, default=(8, 24)
        The quadrature orders for one piece of a price band and for usage.

    Returns
    -------
    dict of str to float
        ``ey_regimen[<plan>]`` and ``ey_revenue[<plan>]``, the mean renewal and the mean revenue
        under each plan.  ``ate_regimen[<plan> vs natural_prices]`` and
        ``ate_revenue[<plan> vs natural_prices]``, the effect of each plan on both.  These are
        the labels a fit reports.  ``ate_engagement_held[<plan>]``, ``ate_engine_held[<plan>]``
        and ``ate_both_held[<plan>]`` for ``price_up`` and ``price_down``, the decomposition below.

    See Also
    --------
    make_sequential_pricing : The generator whose law this truth integrates.
    PRICE_PLANS : The plans this function evaluates.
    price_rise : A plan with another factor.

    Notes
    -----
    An outcome regression on the usage, the engagement, and the prices can hold two things at the
    values the engine's own prices produced.  The decomposition keys give the effect on renewal when
    the plan's prices enter the renewal model and one or both of these stay natural.

    * ``ate_engagement_held``: the engagement model reads the natural price gap.
    * ``ate_engine_held``: the engine's next price mean reads the natural price gap.
    * ``ate_both_held``: both read the natural gap.  This is the limit of the estimate from a
      logistic regression of the renewal on the usage, the engagement, and the three prices,
      with the prices replaced by the plan's prices.

    Examples
    --------
    >>> from cleverly.datasets import pricing_truth
    >>> truth = pricing_truth()
    >>> round(truth["ate_regimen[price_up vs natural_prices]"], 4)
    -0.1003
    >>> truth["ate_regimen[price_down vs natural_prices]"] > 0.1
    True
    """
    orders = (int(nodes[0]), int(nodes[1]))
    return dict(_truth_table(tuple(float(factor) for factor in factors), orders))


# ------------------------------------------------------------------------------------ the data


def make_sequential_pricing(
    n: int,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""Draw three monthly renewal prices and the renewal, with the exact truth of the plans.

    A subscription retailer's pricing engine sets a renewal price in each of three months.  The
    outcome ``renewed`` is the renewal at the third price, and ``revenue`` is ``renewed`` times
    that price.  The frame has one wide row per customer and no missing value.

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
        A wide frame with columns ``usage``, ``engaged_1``, ``price_1``, ``engaged_2``,
        ``price_2``, ``engaged_3``, ``price_3``, ``renewed``, and ``revenue``.
    truth : dict of str to float
        The result of :func:`pricing_truth`.

    See Also
    --------
    cleverly.datasets.pricing_truth : The exact truth this function returns.
    cleverly.datasets.pricing_protocol : The protocol of the question the frame answers.
    cleverly.datasets.PRICE_PLANS : The plans of the tutorial.

    Notes
    -----
    The law, for months :math:`t = 1, 2, 3`, with :math:`W` the usage, :math:`L_t` the engagement,
    :math:`A_t` the price, :math:`c_t = 6 + 2 L_t` the month's centre, and :math:`g_t = A_t - c_t`
    the price gap.  The start is :math:`A_0 = c_0 = 6` and :math:`L_0 = 0`.

    ============================ ===========================================================
    variable                     law
    ============================ ===========================================================
    ``usage`` :math:`W`          :math:`N(0, 1)` truncated to :math:`[-2.5, 2.5]`
    ``engaged_t`` :math:`L_t`    Bernoulli, logit :math:`0.3 + 0.6 W - 0.35\,g_{t-1} +
                                 0.8 L_{t-1}`
    ``price_t`` :math:`A_t`      :math:`N(c_t + 0.3\,g_{t-1} + 0.3 W,\ 1.5^2)` truncated to
                                 :math:`[4 + 2 L_t,\ 8 + 2 L_t]`
    ``renewed`` :math:`Y`        Bernoulli, logit :math:`0.5 - 0.45\,g_3 + 0.9 L_3 + 0.3 W -
                                 0.1 (g_1 + g_2)`
    ``revenue``                  :math:`Y A_3`, with the declared support :math:`[0, 10]`
    ============================ ===========================================================

    A price above the centre lowers the next engagement, and an engaged customer takes a higher
    price.  The engine carries 0.3 of each price gap into the next price.  The generator does not
    round a price.  A rounded price could leave a plan's price one unit in the last place outside
    the band that the design declares.  Each column equals the same column of the law of the
    registered study ``mtp-pricing`` at three nodes, row for row.  There is no censoring: every
    customer has every column.

    Examples
    --------
    >>> from cleverly.datasets import make_sequential_pricing
    >>> frame, truth = make_sequential_pricing(200, seed=0)
    >>> list(frame.columns)[:3]
    ['usage', 'engaged_1', 'price_1']
    >>> frame.shape
    (200, 9)
    >>> round(truth["ate_regimen[price_up vs natural_prices]"], 4)
    -0.1003
    """
    rng = np.random.default_rng(seed)
    bound = _LAW["usage_bound"]
    usage = np.asarray(
        norm.ppf(rng.uniform(norm.cdf(-bound), norm.cdf(bound), n)),
        dtype=float,
    )
    payload: dict[str, Any] = {"usage": usage}
    gap, last_engaged, earlier = np.zeros(n), np.zeros(n), np.zeros(n)
    price = np.zeros(n)
    for node in range(1, PRICE_NODES + 1):
        engaged = (rng.random(n) < _engagement(usage, gap, last_engaged)).astype(float)
        price = _draw_price(rng, _price_mean(usage, engaged, gap), _lower(engaged), _upper(engaged))
        payload[_engaged_column(node)] = engaged
        payload[_price_column(node)] = price
        if node < PRICE_NODES:
            earlier = earlier + (price - _centre(engaged))
        gap, last_engaged = price - _centre(engaged), engaged
    renewed = (rng.random(n) < _renewal(usage, last_engaged, gap, earlier)).astype(float)
    payload["renewed"] = renewed
    payload["revenue"] = renewed * price
    return frame_from_dict(payload, backend=backend), pricing_truth()


def pricing_protocol() -> StudyProtocol:
    """Return the protocol of the monthly renewal-price program.

    Returns
    -------
    StudyProtocol
        The protocol that compares the engine's own prices with a price rise and a price cut over
        three monthly prices, on the renewal at the third price.

    See Also
    --------
    make_sequential_pricing : The synthetic law that draws data for this protocol.
    cleverly.StudyProtocol : The record this function fills.
    cleverly.datasets.offers_protocol : The protocol of the weekly-offer program.

    Notes
    -----
    The record states time zero, eligibility, the treatment versions, the outcome horizon, and the
    interference unit.  The available history and the confounders are not protocol fields.  The
    assumption rationale names the confounders, and a design prints the history at each node.

    Examples
    --------
    >>> from cleverly.datasets import pricing_protocol
    >>> protocol = pricing_protocol()
    >>> len(protocol.treatment_strategies)
    3
    >>> protocol.horizon
    'Month 3, at the renewal after the third monthly price'
    """
    return StudyProtocol(
        target_population="Active monthly subscribers at the start of a three-month term",
        eligibility=(
            "Active subscription at the first monthly renewal price",
            "A renewal price set by the pricing engine in each of months 1 to 3",
            "One account per customer, so shared-household accounts are excluded",
            "A three-month term that starts at time zero, with no exit before the month-3 renewal",
        ),
        time_zero=(
            "The first monthly renewal price, after the usage score and the month-1 engagement "
            "score and before the engine sets the price"
        ),
        treatment_strategies=(
            "The pricing engine's own price in months 1 to 3",
            "Raise each monthly price by a declared factor where the raised price stays inside "
            "that month's price band, otherwise keep the engine's price. The question uses a "
            "12% rise, and the support step also reads 6% and 25%",
            "Cut each monthly price by 15% where the cut price stays inside that month's price "
            "band, otherwise keep the engine's price",
        ),
        treatment_versions=(
            "The renewal price the engine quotes, not rounded",
            "The declared factor on the engine's price, quoted for one month",
            "The 15% cut on the engine's price, quoted for one month",
        ),
        outcome=(
            "Renewal at the month-3 price, and month-3 revenue: the renewal times the month-3 price"
        ),
        horizon="Month 3, at the renewal after the third monthly price",
        intercurrent_event_handling=(
            "None: the three-month term has no exit before the month-3 renewal",
            "Every customer in the data has all three prices and the renewal, so the data have "
            "no censoring",
        ),
        interference_unit="Individual customer",
        assumption_rationale=(
            "The pricing engine logs every input it reads, so usage and the monthly "
            "engagement score cover the common causes of the price, the later engagement, and "
            "the renewal. This is sequential randomization given the recorded history",
            "A price band that moves with engagement is declared for each month, so the "
            "policy keeps every price inside the support of the engine's price",
            "A declared factor on the quoted price, for one month, makes each price one "
            "well-defined action, so a customer's recorded outcome is what that price produces",
            "One account per customer and no reference-price spillover support no interference",
        ),
    )

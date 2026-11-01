"""The weekly-orders law: exact truths, the frame layout, and the protocol.

:func:`cleverly.datasets.weekly_orders_truth` enumerates every path of the binary law.  A forward
Monte Carlo draw of the intervened law, written here from the documented equations, checks each
mean.  The draw uses no code path of the enumeration, so it checks the law's bookkeeping and the
plan and tilt definitions.

These checks are blind to a term that vanishes at the truth, so three wrong truths must fail the
same comparison: a rule that reads this week's order draw (which no rule can see at the send), a truth weighted by
staying tracked (the complete-case law instead of the hypothetical strategy), and a tilt that
doubles the chance of an offer instead of its odds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import expit

from cleverly.datasets import (
    ORDER_PLANS,
    ORDER_TILTS,
    ORDER_WEEKS,
    make_weekly_orders,
    offer_after_no_order,
    weekly_orders_protocol,
    weekly_orders_truth,
)

#: Customers per plan in the Monte Carlo comparison.  A weekly share near 0.6 has a standard error
#: of about 0.0007, so four standard errors are about 0.003.
MONTE_CARLO_N = 500_000

#: An offer chance from ``(tenure, engaged, last order, last offer, this week's order draw)``.
#: The last argument lets a mutation read the order of the same week.
OfferChance = Callable[..., np.ndarray]


def _scoring(tenure, engaged, order, offer):
    """The documented scoring system, written out independently of the module."""
    return expit(0.8 + 0.1 * tenure - 0.4 * engaged - 1.4 * order + 0.2 * offer)


def _forward(
    chance: OfferChance, *, seed: int, weight_by_tracking: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the intervened law forward and return each week's mean and standard error."""
    rng = np.random.default_rng(seed)
    n = MONTE_CARLO_N
    tenure = rng.integers(0, 3, size=n).astype(float)
    order = (rng.random(n) < expit(-0.4 + 0.4 * tenure)).astype(float)
    engaged = np.zeros(n)
    offer = np.zeros(n)
    tracked = np.ones(n)
    means, errors = np.zeros(ORDER_WEEKS), np.zeros(ORDER_WEEKS)
    for week in range(1, ORDER_WEEKS + 1):
        if week == 1:
            p_engaged = expit(-0.3 + 0.4 * tenure + 0.8 * order)
        else:
            p_engaged = expit(-0.7 + 0.3 * tenure + 1.1 * engaged + 0.7 * offer + 0.8 * order)
        engaged_now = (rng.random(n) < p_engaged).astype(float)
        order_draw = rng.random(n)
        offer_now = (rng.random(n) < chance(tenure, engaged_now, order, offer, order_draw)).astype(
            float
        )
        p_order = expit(
            -1.5
            + 0.3 * tenure
            + 1.0 * engaged_now
            + 0.9 * offer_now
            + 1.2 * order
            - 0.7 * offer_now * offer
        )
        order_now = (order_draw < p_order).astype(float)
        tracked = tracked * (
            rng.random(n) < expit(2.8 + 0.2 * tenure + 0.4 * engaged_now - 0.5 * offer_now)
        )
        kept = order_now[tracked == 1] if weight_by_tracking else order_now
        means[week - 1] = kept.mean()
        errors[week - 1] = kept.std() / np.sqrt(kept.size)
        engaged, offer, order = engaged_now, offer_now, order_now
    return means, errors


def _plan(rule):
    return lambda tenure, engaged, order, offer, draw: rule(order)


def _tilt(delta):
    def chance(tenure, engaged, order, offer, draw):
        base = _scoring(tenure, engaged, order, offer)
        return delta * base / (delta * base + 1.0 - base)

    return chance


@pytest.fixture(scope="module")
def truth() -> dict[str, float]:
    return weekly_orders_truth()


def _curve(truth: dict[str, float], name: str) -> np.ndarray:
    return np.array([truth[f"ey_regimen[{name} @ t={week}]"] for week in (1, 2, 3)])


def _assert_matches(truth_curve: np.ndarray, means: np.ndarray, errors: np.ndarray) -> None:
    assert np.all(np.abs(truth_curve - means) <= 4.0 * errors), (truth_curve, means, errors)


def _assert_rejects(wrong_curve: np.ndarray, truth_curve: np.ndarray, errors: np.ndarray) -> None:
    # The wrong truth must sit more than six Monte Carlo standard errors from the right one in some
    # week.  A Monte Carlo mean then lands within four standard errors of the wrong truth only by
    # an error of more than two standard errors, so the comparison rejects it with probability
    # above 0.97.
    assert np.max(np.abs(wrong_curve - truth_curve) / errors) > 6.0


def test_the_truth_names_every_plan_tilt_week_and_contrast(truth: dict[str, float]) -> None:
    expected = set()
    for name in (*ORDER_PLANS, *ORDER_TILTS):
        expected |= {f"ey_regimen[{name} @ t={week}]" for week in (1, 2, 3)}
    for name in ("always", "win_back"):
        expected |= {f"ate_regimen[{name} vs never @ t={week}]" for week in (1, 2, 3)}
    expected |= {f"ate_regimen[odds x2 vs natural course @ t={week}]" for week in (1, 2, 3)}
    assert set(truth) == expected
    for week in (1, 2, 3):
        assert truth[f"ate_regimen[always vs never @ t={week}]"] == pytest.approx(
            truth[f"ey_regimen[always @ t={week}]"] - truth[f"ey_regimen[never @ t={week}]"],
            abs=1e-15,
        )
    assert make_weekly_orders(20, seed=1)[1] == truth


@pytest.mark.parametrize("index,name", list(enumerate(ORDER_PLANS)))
def test_each_plan_curve_matches_a_forward_draw(
    truth: dict[str, float], index: int, name: str
) -> None:
    means, errors = _forward(_plan(ORDER_PLANS[name]), seed=700 + index)
    _assert_matches(_curve(truth, name), means, errors)


@pytest.mark.parametrize("index,name", list(enumerate(ORDER_TILTS)))
def test_each_tilt_curve_matches_a_forward_draw(
    truth: dict[str, float], index: int, name: str
) -> None:
    means, errors = _forward(_tilt(ORDER_TILTS[name]), seed=710 + index)
    _assert_matches(_curve(truth, name), means, errors)


def test_the_natural_course_matches_the_observed_weekly_share_among_all_customers(
    truth: dict[str, float],
) -> None:
    # The natural course keeps the scoring system and removes tracking loss.  So the generator's
    # own draw, weighted by one over the documented chance of staying tracked, must reproduce
    # the tilt-one curve.  This checks the generator against the enumeration.
    frame, _ = make_weekly_orders(MONTE_CARLO_N, seed=720)
    curve = _curve(truth, "natural course")
    for week in (1, 2, 3):
        observed = frame[f"order_{week}"].notna().to_numpy()
        # Inverse probability of staying tracked through this week, from the documented model.
        weight = np.ones(len(frame))
        for step in range(1, week + 1):
            stay = expit(
                2.8
                + 0.2 * frame["tenure"].to_numpy()
                + 0.4 * frame[f"engaged_{step}"].to_numpy()
                - 0.5 * frame[f"offer_{step}"].to_numpy()
            )
            weight = weight / np.where(np.isnan(stay), 1.0, stay)
        values = np.where(observed, frame[f"order_{week}"].to_numpy() * weight, 0.0)
        mean = values.mean()
        error = values.std() / np.sqrt(values.size)
        assert abs(mean - curve[week - 1]) <= 4.0 * error, (week, mean, curve[week - 1], error)


def test_three_wrong_truths_fail_the_forward_comparison(truth: dict[str, float]) -> None:
    right = _curve(truth, "win_back")
    _, errors = _forward(_plan(ORDER_PLANS["win_back"]), seed=730)

    # 1. A rule that reads this week's order draw, which no rule can see at the send: it sends the
    #    code when the customer would not order this week without one.
    def reads_this_week(tenure, engaged, order, offer, draw):
        # The order of this week under no code, from the documented order model and the draw.
        p_no_code = expit(-1.5 + 0.3 * tenure + 1.0 * engaged + 1.2 * order)
        return 1.0 - (draw < p_no_code).astype(float)

    wrong, _ = _forward(reads_this_week, seed=731)
    _assert_rejects(wrong, right, errors)

    # 2. The complete-case law: the share among customers still tracked, under the plan.
    complete_case, _ = _forward(_plan(ORDER_PLANS["win_back"]), seed=732, weight_by_tracking=True)
    _assert_rejects(complete_case, right, errors)

    # 3. A tilt that doubles the chance of an offer instead of its odds.
    def doubled_chance(tenure, engaged, order, offer, draw):
        return np.minimum(2.0 * _scoring(tenure, engaged, order, offer), 1.0)

    doubled, doubled_errors = _forward(doubled_chance, seed=733)
    _assert_rejects(doubled, _curve(truth, "odds x2"), doubled_errors)


def test_the_frame_is_wide_in_time_order_with_monotone_censoring() -> None:
    frame, _ = make_weekly_orders(3_000, seed=5)
    columns = ["tenure", "order_0"]
    for week in (1, 2, 3):
        columns += [f"engaged_{week}", f"offer_{week}", f"tracked_{week}", f"order_{week}"]
    assert list(frame.columns) == columns
    assert set(frame["tenure"].unique()) == {0.0, 1.0, 2.0}
    for week in (1, 2, 3):
        lost = frame[f"tracked_{week}"] == 0
        # Every column after the week's tracking flag is missing once tracking stops.
        later = columns[columns.index(f"tracked_{week}") + 1 :]
        assert frame.loc[lost, later].isna().all().all()
        kept = frame[f"tracked_{week}"] == 1
        assert frame.loc[kept, f"order_{week}"].notna().all()
    # Last week's order lowers the chance of an offer.
    week_two = frame[frame["tracked_1"] == 1]
    rates = week_two.groupby("order_1")["offer_2"].mean()
    assert rates[1.0] < rates[0.0] - 0.15


def test_the_win_back_rule_sends_the_code_after_a_week_without_an_order() -> None:
    assert offer_after_no_order([0.0, 1.0]).tolist() == [1.0, 0.0]
    assert ORDER_PLANS["never"](np.array([0.0, 1.0])).tolist() == [0.0, 0.0]
    assert ORDER_PLANS["always"](np.array([0.0, 1.0])).tolist() == [1.0, 1.0]


def test_the_protocol_names_the_five_strategies_and_the_weekly_outcome() -> None:
    protocol = weekly_orders_protocol()
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 5
    assert "recorded weekly" in protocol.outcome
    assert protocol.eligibility[-1].endswith("runs through week 3")

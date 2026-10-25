"""The three-decision navigation program: exact truths, blips, layout, and the score's range.

:func:`cleverly.datasets.recovery_score_truth` is a finite sum over ``lives_alone`` and the two
symptom flags.  A Monte Carlo draw of the intervened law checks every plan mean and every blip.
Exact-law checks are blind to a term that vanishes at the truth, so two wrong truths must fail the
same comparison: one that drops the offer's effect on the day-10 symptoms, and one that drops the
product of the day-10 offer with ``lives_alone``, the term the tutorial's wrong model misses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import (
    RECOVERY_BOUNDS,
    RECOVERY_PLANS,
    make_recovery_score,
    recovery_score_protocol,
    recovery_score_truth,
)
from cleverly.datasets import recovery as law

#: Draws in each Monte Carlo comparison.
MONTE_CARLO_N = 1_000_000
#: The floor added to four Monte Carlo standard errors. The truths are exact sums.
FLOOR = 1e-9

Plan = tuple[float, float, float]
Day10 = Callable[[Any, Any], Any]
Day20 = Callable[[Any, Any], Any]


def _intervened(
    plan: Plan,
    alone: np.ndarray,
    rng: np.random.Generator,
    day10: Day10 = law._symptoms_day10,
    day20: Day20 = law._symptoms_day20,
) -> np.ndarray:
    """Draw the day-30 score of each patient in ``alone`` with every offer set by ``plan``."""
    o1, o2, o3 = plan
    n = alone.size
    risk = rng.uniform(0.0, 1.0, n)
    s1 = np.asarray(rng.binomial(1, day10(alone, o1)), dtype=float)
    s2 = np.asarray(rng.binomial(1, day20(alone, o2)), dtype=float)
    mean = law._score_mean(risk, alone, o1, s1, o2, s2, o3)
    return np.asarray(mean + rng.uniform(-law._NOISE, law._NOISE, n), dtype=float)


def _plan_draws(plan: Plan, seed: int, **wrong: Any) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alone = rng.binomial(1, law._ALONE, MONTE_CARLO_N).astype(float)
    return _intervened(plan, alone, rng, **wrong)


def _tolerance(draws: np.ndarray) -> float:
    return 4.0 * draws.std(ddof=1) / np.sqrt(draws.size) + FLOOR


@pytest.mark.parametrize("plan", list(RECOVERY_PLANS))
def test_each_plan_mean_matches_a_monte_carlo_draw(plan: str) -> None:
    draws = _plan_draws(RECOVERY_PLANS[plan], seed=21)
    assert abs(draws.mean() - recovery_score_truth(*RECOVERY_PLANS[plan])) < _tolerance(draws)


@pytest.mark.parametrize("node", [0, 1, 2])
@pytest.mark.parametrize("alone", [0.0, 1.0])
def test_each_blip_matches_a_paired_monte_carlo_draw(node: int, alone: float) -> None:
    """The effect of one offer with the others withheld, within one ``lives_alone`` group.

    Both plans read the same uniform draws (common random numbers), so the paired difference has
    a small standard error.
    """
    plan: Plan = (float(node == 0), float(node == 1), float(node == 2))
    group = np.full(MONTE_CARLO_N, alone)
    treated = _intervened(plan, group, np.random.default_rng(31))
    control = _intervened((0.0, 0.0, 0.0), group, np.random.default_rng(31))
    difference = treated - control
    _, truth = make_recovery_score(10, seed=0)
    column = ("offer_discharge", "offer_day10", "offer_day20")[node]
    expected = truth[f"blip[{column}:1.0:intercept]"]
    if alone == 1.0 and node < 2:
        expected += truth[f"blip[{column}:1.0:lives_alone]"]
    assert abs(difference.mean() - expected) < _tolerance(difference)


def _without_day10_effect(alone: Any, offer: Any) -> Any:
    """A wrong law: the discharge offer does not change the day-10 symptoms."""
    del offer
    return law._symptoms_day10(alone, 0.0)


def _without_day20_product(alone: Any, offer: Any) -> Any:
    """A wrong law: the day-10 offer lowers the day-20 symptoms equally for every patient."""
    return law._symptoms_day20(alone, offer) - law._LAW["s2_offer_alone"] * offer * alone


@pytest.mark.parametrize(
    "wrong",
    [{"day10": _without_day10_effect}, {"day20": _without_day20_product}],
    ids=["no-day10-effect", "no-day20-product"],
)
def test_a_wrong_law_fails_the_same_comparison(wrong: dict[str, Any]) -> None:
    draws = _plan_draws(RECOVERY_PLANS["always"], seed=21, **wrong)
    gap = abs(draws.mean() - recovery_score_truth(*RECOVERY_PLANS["always"]))
    assert gap > 10.0 * _tolerance(draws)


def test_the_truth_dictionary_uses_the_names_a_fit_reports() -> None:
    _, truth = make_recovery_score(10, seed=0)
    assert set(truth) == {
        "ey_regimen[always]",
        "ey_regimen[never]",
        "ate_regimen[always vs never]",
        "blip[offer_discharge:1.0:intercept]",
        "blip[offer_discharge:1.0:lives_alone]",
        "blip[offer_day10:1.0:intercept]",
        "blip[offer_day10:1.0:lives_alone]",
        "blip[offer_day20:1.0:intercept]",
    }
    assert truth["ate_regimen[always vs never]"] == pytest.approx(
        truth["ey_regimen[always]"] - truth["ey_regimen[never]"]
    )
    # Under the linear blip model the plan contrast is the sum of the mean blips.
    share = law._ALONE
    blips = sum(
        truth[f"blip[{column}:1.0:intercept]"] for column in ("offer_discharge", "offer_day10")
    )
    blips += truth["blip[offer_day20:1.0:intercept]"]
    blips += share * (
        truth["blip[offer_discharge:1.0:lives_alone]"] + truth["blip[offer_day10:1.0:lives_alone]"]
    )
    assert truth["ate_regimen[always vs never]"] == pytest.approx(blips)
    # The lesson of the tutorial: the offers help patients who live alone more.
    assert truth["blip[offer_discharge:1.0:lives_alone]"] > 0.0
    assert truth["blip[offer_day10:1.0:lives_alone]"] > 0.0


def test_the_layout_and_the_score_range() -> None:
    frame, _ = make_recovery_score(50_000, seed=3)
    assert list(frame.columns) == [
        "lives_alone",
        "discharge_risk",
        "offer_discharge",
        "symptoms_day10",
        "offer_day10",
        "symptoms_day20",
        "offer_day20",
        "recovery_day30",
    ]
    assert not frame.isna().any().any()
    lower, upper = RECOVERY_BOUNDS
    # The smallest and largest means are 10 + 8 and 83.3 - 8 away from the bounds.
    assert frame["recovery_day30"].between(lower + 10.0, upper - 16.0).all()
    for column in ("lives_alone", "offer_discharge", "symptoms_day10", "offer_day10"):
        assert set(frame[column].unique()) <= {0.0, 1.0}


def test_the_discharge_offer_goes_mostly_to_patients_who_live_alone() -> None:
    """The imbalance that makes the main-terms symptom model carry the offer's effect across."""
    frame, _ = make_recovery_score(50_000, seed=4)
    share = frame.groupby("lives_alone")["offer_discharge"].mean()
    assert share[1.0] > 0.5 > 0.2 > share[0.0]


def test_the_protocol_names_the_three_decisions() -> None:
    protocol = recovery_score_protocol()
    assert protocol.treatment_strategies[0].startswith("Offer navigation at discharge, on day 10")
    assert "0 to 100" in protocol.outcome


def test_a_nonpositive_size_is_refused() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        make_recovery_score(0)

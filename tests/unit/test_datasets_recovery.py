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


FRAME_N = 400_000


@pytest.fixture(scope="module")
def large_frame() -> Any:
    frame, _ = make_recovery_score(FRAME_N, seed=11)
    return frame


def _offer_probability(frame: Any, node: int) -> np.ndarray:
    """The known probability of each patient's offer at one decision, from the law's table."""
    p = law._LAW
    risk = frame["discharge_risk"].to_numpy()
    alone = frame["lives_alone"].to_numpy()
    if node == 0:
        logit = p["o1_intercept"] + p["o1_risk"] * risk + p["o1_alone"] * alone
    else:
        prefix = f"o{node + 1}"
        symptoms = frame[("symptoms_day10", "symptoms_day20")[node - 1]].to_numpy()
        earlier = frame[("offer_discharge", "offer_day10")[node - 1]].to_numpy()
        logit = (
            p[f"{prefix}_intercept"]
            + p[f"{prefix}_risk"] * risk
            + p[f"{prefix}_alone"] * alone
            + p[f"{prefix}_symptoms"] * symptoms
            + p[f"{prefix}_offer"] * earlier
        )
    return np.asarray(1.0 / (1.0 + np.exp(-logit)), dtype=float)


@pytest.mark.parametrize("plan", list(RECOVERY_PLANS))
def test_weighting_the_drawn_frame_recovers_each_plan_mean(large_frame: Any, plan: str) -> None:
    """The end-to-end witness: the sampler draws the law whose plan means the truths report.

    Inverse probability weighting with the known offer probabilities is unbiased for the plan
    mean, so a sampler that drifts from the law the truths describe moves this estimate.
    """
    columns = ("offer_discharge", "offer_day10", "offer_day20")
    weight = np.ones(len(large_frame))
    for node, (column, value) in enumerate(zip(columns, RECOVERY_PLANS[plan], strict=True)):
        offered = _offer_probability(large_frame, node)
        chance = offered if value == 1.0 else 1.0 - offered
        weight *= (large_frame[column].to_numpy() == value) / chance
    terms = weight * large_frame["recovery_day30"].to_numpy()
    assert abs(terms.mean() - recovery_score_truth(*RECOVERY_PLANS[plan])) < _tolerance(terms)


@pytest.mark.parametrize("day", [10, 20])
def test_each_symptom_flag_follows_its_law_in_every_cell_of_the_record(
    large_frame: Any, day: int
) -> None:
    """Each flag depends on living alone and the latest offer only, as the truths assume."""
    flag = f"symptoms_day{day}"
    record = ["lives_alone", "offer_discharge"]
    if day == 20:
        record += ["symptoms_day10", "offer_day10"]
    helper = law._symptoms_day10 if day == 10 else law._symptoms_day20
    offer = "offer_discharge" if day == 10 else "offer_day10"
    for key, cell in large_frame.groupby(record):
        values = dict(zip(record, key, strict=True))
        expected = float(helper(values["lives_alone"], values[offer]))
        observed = cell[flag].to_numpy()
        se = np.sqrt(expected * (1.0 - expected) / observed.size)
        assert abs(observed.mean() - expected) < 4.0 * se + FLOOR, key


def test_a_regression_of_the_drawn_score_recovers_the_score_law(large_frame: Any) -> None:
    """The score's mean is linear in the record, so least squares recovers each coefficient."""
    names = [
        ("y_risk", "discharge_risk"),
        ("y_alone", "lives_alone"),
        ("y_offer_discharge", "offer_discharge"),
        ("y_symptoms_day10", "symptoms_day10"),
        ("y_offer_day10", "offer_day10"),
        ("y_symptoms_day20", "symptoms_day20"),
        ("y_offer_day20", "offer_day20"),
    ]
    design = np.column_stack(
        [np.ones(len(large_frame))] + [large_frame[column].to_numpy() for _, column in names]
    )
    score = large_frame["recovery_day30"].to_numpy()
    coefficients, *_ = np.linalg.lstsq(design, score, rcond=None)
    residual = score - design @ coefficients
    covariance = residual.var() * np.linalg.inv(design.T @ design)
    expected = [law._LAW["y_intercept"]] + [law._LAW[key] for key, _ in names]
    se = np.sqrt(np.diag(covariance))
    assert np.all(np.abs(coefficients - expected) < 4.0 * se + FLOOR)


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
    binary = (
        "lives_alone",
        "offer_discharge",
        "symptoms_day10",
        "offer_day10",
        "symptoms_day20",
        "offer_day20",
    )
    for column in binary:
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

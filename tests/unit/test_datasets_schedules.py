"""The contact-schedule and service-dose laws: their exact truths and their draws.

:func:`cleverly.datasets.contact_schedules_truth` and
:func:`cleverly.datasets.navigation_doses_truth` integrate by quadrature.  Monte Carlo checks hold
each: the conditional mean averaged over drawn covariates and doses under the intervened law (no
outcome noise), and an importance-weighted mean of drawn outcomes, which also checks the outcome,
schedule and dose draws.  Each tolerance is four Monte Carlo standard errors.

An exact-law check is blind to a term that vanishes at the truth, so deliberately wrong laws must
fail the same comparison: a tilt that ignores the factors, a score without the schedule step, a
dose law without the interaction of the two doses, and a plan that ignores its cap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly.datasets import (
    DOSE_PLANS,
    SCHEDULE_LEVELS,
    contact_schedules_truth,
    make_contact_schedules,
    make_navigation_doses,
    navigation_doses_truth,
)
from cleverly.datasets import schedules as law
from cleverly.utils.bounds import expit

MONTE_CARLO_N = 1_000_000
TILTS = {
    "current practice": {"low": 1.0},
    "medium and high doubled": {"medium": 2.0, "high": 2.0},
    "high doubled": {"high": 2.0},
}


def _within(value: float, draws: np.ndarray) -> bool:
    error = float(draws.std(ddof=1) / np.sqrt(draws.size))
    return abs(value - float(draws.mean())) < 4.0 * error


# ----------------------------------------------------------------- contact schedules


def _schedule_oracle_draws(seed: int) -> dict[str, np.ndarray]:
    """For each truth key, the per-row terms whose mean estimates it, from drawn covariates."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((MONTE_CARLO_N, 3))
    g = law._schedule_probabilities(w)
    m = law._score_mean(w)
    terms = {f"ey[{level}]": m[:, index] for index, level in enumerate(SCHEDULE_LEVELS)}
    tilted: dict[str, np.ndarray] = {}
    for name, factors in TILTS.items():
        delta = law._tilt_factors(factors)
        q = g * delta
        q = q / q.sum(axis=1, keepdims=True)
        tilted[name] = np.sum(q * m, axis=1)
        terms[f"ey_ipsi[{name}]"] = tilted[name]
    for name in list(TILTS)[1:]:
        terms[f"ate_ipsi[{name} vs current practice]"] = tilted[name] - tilted["current practice"]
    return terms


def test_the_schedule_truth_matches_the_score_mean_over_drawn_covariates() -> None:
    truth = contact_schedules_truth(TILTS)
    terms = _schedule_oracle_draws(20261025)
    assert set(truth) == set(terms)
    for key, draws in terms.items():
        assert _within(truth[key], draws), key


def test_the_schedule_truth_matches_the_weighted_means_of_drawn_scores() -> None:
    # Weight each drawn score by the tilted chance of its own schedule over the true chance.
    frame, truth = make_contact_schedules(MONTE_CARLO_N, seed=20261026, tilts=TILTS)
    w = frame[["discharge_risk", "age", "baseline_support_need"]].to_numpy()
    code = {label: index for index, label in enumerate(SCHEDULE_LEVELS)}
    level = np.array([code[label] for label in frame["cadence"]])
    score = frame["transition_score"].to_numpy()
    assert score.min() > 0.0 and score.max() < 1.0
    g = law._schedule_probabilities(w)
    observed = g[np.arange(score.size), level]
    for name, factors in TILTS.items():
        delta = law._tilt_factors(factors)
        tilted = delta[level] / (g @ delta)
        assert _within(truth[f"ey_ipsi[{name}]"], tilted * score), name
    for index, label in enumerate(SCHEDULE_LEVELS):
        draws = (level == index) * score / observed
        assert _within(truth[f"ey[{label}]"], draws), label


def _ignore_factors(factors: Any) -> np.ndarray:
    return np.ones(len(SCHEDULE_LEVELS))


def _no_schedule_step(w: np.ndarray) -> np.ndarray:
    base = -0.3 + 0.6 * w[..., 0] - 0.3 * w[..., 1] + 0.2 * w[..., 2]
    return np.asarray(expit(base[..., None] + np.zeros(3)), dtype=float)


@pytest.mark.parametrize(
    ("name", "wrong"), [("_tilt_factors", _ignore_factors), ("_score_mean", _no_schedule_step)]
)
def test_a_wrong_schedule_law_fails_the_monte_carlo_comparison(
    name: str, wrong: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    terms = _schedule_oracle_draws(20261025)
    monkeypatch.setattr(law, name, wrong)
    mutated = contact_schedules_truth(TILTS)
    assert any(not _within(mutated[key], terms[key]) for key in terms)


def test_the_schedule_draw_is_seeded_and_refuses_unknown_levels() -> None:
    first, _ = make_contact_schedules(200, seed=11)
    second, _ = make_contact_schedules(200, seed=11)
    assert first.equals(second)
    assert list(first.columns) == [
        "transition_score",
        "cadence",
        "discharge_risk",
        "age",
        "baseline_support_need",
    ]
    with pytest.raises(ValueError, match="unknown schedule levels"):
        contact_schedules_truth({"now": {"weekly": 2.0}})
    with pytest.raises(ValueError, match="positive"):
        contact_schedules_truth({"now": {"high": 0.0}})


def test_the_schedule_truth_is_pinned() -> None:
    truth = contact_schedules_truth(TILTS)
    # Independent check: the Monte Carlo comparisons above, and 64 Gauss-Hermite nodes per axis.
    assert truth["ey[medium]"] == pytest.approx(0.5444866271, abs=1e-9)
    assert truth["ate_ipsi[medium and high doubled vs current practice]"] == pytest.approx(
        0.0146501827, abs=1e-9
    )
    assert truth["ate_ipsi[high doubled vs current practice]"] == pytest.approx(
        0.0208135470, abs=1e-9
    )


# ----------------------------------------------------------------- service doses


def _dose_oracle_draws(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((MONTE_CARLO_N, 3))
    hours_mean, minutes_mean = law._dose_means(w)
    hours = hours_mean + law._HOURS_SD * rng.standard_normal(MONTE_CARLO_N)
    minutes = minutes_mean + law._MINUTES_SD * rng.standard_normal(MONTE_CARLO_N)
    means = {
        name: law._success_probability(
            w,
            law._shifted(hours, plan.get("navigator_hours")),
            law._shifted(minutes, plan.get("pharmacist_minutes")),
        )
        for name, plan in DOSE_PLANS.items()
    }
    terms = {f"ey_policy[{name}]": values for name, values in means.items()}
    for name, values in means.items():
        if name != "current practice":
            terms[f"ate_policy[{name} vs current practice]"] = values - means["current practice"]
    return terms


def test_the_dose_truth_matches_the_success_probability_over_drawn_doses() -> None:
    truth = navigation_doses_truth()
    terms = _dose_oracle_draws(20261027)
    assert set(truth) == set(terms)
    for key, draws in terms.items():
        assert _within(truth[key], draws), key


def test_the_dose_truth_matches_the_mean_of_drawn_outcomes() -> None:
    frame, truth = make_navigation_doses(MONTE_CARLO_N, seed=20261028)
    success = frame["transition_success"].to_numpy()
    assert set(np.unique(success)) == {0.0, 1.0}
    assert _within(truth["ey_policy[current practice]"], success)


def _no_interaction(w: np.ndarray, hours: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    h = hours - 4.0
    m = (minutes - 40.0) / 10.0
    eta = -0.2 - 0.6 * w[..., 0] + 0.3 * w[..., 1] - 0.3 * w[..., 2] + 0.3 * h + 0.25 * m
    return np.asarray(expit(eta - 0.1 * h**2), dtype=float)


def _no_cap(dose: np.ndarray, step: tuple[float, float] | None) -> np.ndarray:
    return dose if step is None else dose + step[0]


@pytest.mark.parametrize(
    ("name", "wrong"), [("_success_probability", _no_interaction), ("_shifted", _no_cap)]
)
def test_a_wrong_dose_law_fails_the_monte_carlo_comparison(
    name: str, wrong: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    terms = _dose_oracle_draws(20261027)
    monkeypatch.setattr(law, name, wrong)
    law._plan_means.cache_clear()
    try:
        mutated = navigation_doses_truth()
    finally:
        law._plan_means.cache_clear()
    assert any(not _within(mutated[key], terms[key]) for key in terms)


def test_the_two_doses_interact() -> None:
    # Raising both adds more than raising each alone: the interaction term at work.
    truth = navigation_doses_truth()
    both = truth["ate_policy[both vs current practice]"]
    alone = (
        truth["ate_policy[more navigator time vs current practice]"]
        + truth["ate_policy[more pharmacist time vs current practice]"]
    )
    assert both - alone > 0.01


def test_the_dose_draw_is_seeded_and_names_its_columns() -> None:
    first, _ = make_navigation_doses(200, seed=11)
    second, _ = make_navigation_doses(200, seed=11)
    assert first.equals(second)
    assert list(first.columns) == [
        "transition_success",
        "navigator_hours",
        "pharmacist_minutes",
        "discharge_risk",
        "age",
        "medication_burden",
    ]


def test_the_dose_truth_is_pinned() -> None:
    truth = navigation_doses_truth()
    assert truth["ate_policy[both vs current practice]"] == pytest.approx(0.0731333475, abs=1e-8)
    assert truth["ey_policy[current practice]"] == pytest.approx(0.4423390844, abs=1e-8)

"""The discharge-practices law: exact truths, the observed draw, and wrong truths that must fail.

:func:`cleverly.datasets.discharge_practices_truth` is a sum over the 32 combinations of the
five practices inside a product Gauss--Hermite rule.  A Monte Carlo draw of the covariates and the
practices checks each effect without that enumeration: ``adjusted[c]`` averages the outcome
model's difference at the observed other practices, and ``total[c]`` sets ``c`` and redraws the
follow-up booking from the navigation offer it then has.

These checks are blind to a term that vanishes at the truth, so wrong truths must fail the same
comparisons: a navigation total whose booking ignores the offer, and a navigation effect with the
booking left to follow the offer where it should stay as observed.  The two practices with no
effect have truths of exactly zero, which the enumeration gives whatever the generator draws, so
the observed draw is checked against the outcome model directly, with a model that omits a real
practice as the control that must fail.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import (
    PRACTICES,
    discharge_practices_truth,
    make_discharge_practices,
)
from cleverly.datasets import discharge_practices as law

pytestmark = pytest.mark.xdist_group("datasets_discharge_practices")

#: Patients in each Monte Carlo draw.  An effect's standard error is below 0.0002 here, so four
#: standard errors are under 0.001, far below the 0.034 gap between navigation's two effects.
MONTE_CARLO_N = 1_000_000
#: The floor added to four Monte Carlo standard errors: the quadrature error of the truth, which
#: 40 and 60 Gauss--Hermite nodes per axis agree to below this bound (checked in this module).
QUADRATURE_FLOOR = 1e-10

NAVIGATION = "transition_navigation"
FOLLOWUP = "followup_booked"


def _draw(seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Covariates and the five practices of ``MONTE_CARLO_N`` patients under current practice."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(MONTE_CARLO_N, len(law.COVARIATES)))
    practices: dict[str, np.ndarray] = {}
    for name, model in law.CHANCES.items():
        practices[name] = rng.binomial(1, model(w, practices)).astype(float)
    return w, practices


@pytest.fixture(scope="module")
def population() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    return _draw(10)


def _mean(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std() / np.sqrt(values.size))


def _adjusted(
    population: tuple[np.ndarray, dict[str, np.ndarray]], name: str
) -> tuple[float, float]:
    """``adjusted[name]`` by Monte Carlo: the other practices stay as drawn."""
    w, practices = population
    high = law.outcome_probability(w, {**practices, name: np.ones(len(w))})
    low = law.outcome_probability(w, {**practices, name: np.zeros(len(w))})
    return _mean(high - low)


def _total(
    population: tuple[np.ndarray, dict[str, np.ndarray]], name: str, *, seed: int
) -> tuple[float, float]:
    """``total[name]`` by Monte Carlo: set ``name``, and redraw what it causes."""
    w, practices = population
    means = []
    for level in (1.0, 0.0):
        rng = np.random.default_rng(seed)  # common numbers for the two arms
        setting = {**practices, name: np.full(len(w), level)}
        if name == NAVIGATION:
            setting[FOLLOWUP] = rng.binomial(1, law.CHANCES[FOLLOWUP](w, setting)).astype(float)
        means.append(law.outcome_probability(w, setting))
    return _mean(means[0] - means[1])


def _agrees(value: float, estimate: tuple[float, float]) -> bool:
    mean, error = estimate
    return abs(value - mean) <= 4 * error + QUADRATURE_FLOOR


# ------------------------------------------------------------------------------------ the truth


def test_the_truth_is_converged_in_the_quadrature() -> None:
    coarse, fine = law._truth(40), law._truth(60)
    assert coarse.keys() == fine.keys()
    for key, value in coarse.items():
        assert abs(value - fine[key]) < QUADRATURE_FLOOR, key


def test_the_truth_has_the_values_the_tutorial_narrates() -> None:
    truth = discharge_practices_truth()
    assert round(truth["adjusted[transition_navigation]"], 4) == 0.0165
    assert round(truth["total[transition_navigation]"], 4) == 0.0506
    assert round(truth["total[followup_booked]"], 4) == 0.0984
    assert round(truth["total[medication_review]"], 4) == 0.0493
    for name in ("teach_back", "home_health_referral"):
        assert truth[f"adjusted[{name}]"] == truth[f"total[{name}]"] == 0.0
    # The shares pin each practice's mechanism, which the Monte Carlo checks reuse.
    assert round(truth["stayed_home_rate"], 4) == 0.7474
    shares = {name: round(truth[f"share[{name}]"], 4) for name in PRACTICES}
    assert shares == {
        "transition_navigation": 0.4344,
        "followup_booked": 0.4668,
        "medication_review": 0.4538,
        "teach_back": 0.5242,
        "home_health_referral": 0.3073,
    }


@pytest.mark.parametrize("name", PRACTICES)
def test_each_adjusted_effect_agrees_with_a_monte_carlo_draw(
    population: tuple[np.ndarray, dict[str, np.ndarray]], name: str
) -> None:
    assert _agrees(discharge_practices_truth()[f"adjusted[{name}]"], _adjusted(population, name))


@pytest.mark.parametrize("name", PRACTICES)
def test_each_total_effect_agrees_with_a_monte_carlo_draw(
    population: tuple[np.ndarray, dict[str, np.ndarray]], name: str
) -> None:
    estimate = _total(population, name, seed=20)
    assert _agrees(discharge_practices_truth()[f"total[{name}]"], estimate)


def _with_navigation_adjusted(name: str, nodes: int = 40, *, navigation_too: bool = True) -> float:
    """The effect a fit estimates that adjusts for the covariates and the navigation offer.

    Written apart from the module's enumeration: the outcome regression on ``(name, navigation,
    W)`` averages the outcome model over the other practices' law given those three, and the
    contrast is averaged over the observed law of ``(navigation, W)``.  Given ``(navigation, W)``
    every practice but the booking depends on ``W`` only, and the booking on navigation and
    ``W``, so the other practices do not depend on ``name``.

    With ``navigation_too=False`` the regression also averages over the navigation offer given
    ``(name, W)``, which is what a fit adjusting for ``W`` alone estimates.
    """
    w, mass = law._grid(nodes)
    others = [other for other in PRACTICES if other not in (name, NAVIGATION)]
    navigation_chance = law.CHANCES[NAVIGATION](w, {})
    if not navigation_too:
        return _covariates_only(name, w, mass, others, navigation_chance)
    effect = np.zeros(len(w))
    for navigation in (0.0, 1.0):
        weight = navigation_chance if navigation == 1.0 else 1.0 - navigation_chance
        regression = {0.0: np.zeros(len(w)), 1.0: np.zeros(len(w))}
        for level in (0.0, 1.0):
            for levels in itertools.product((0.0, 1.0), repeat=len(others)):
                setting = {
                    NAVIGATION: navigation,
                    name: level,
                    **dict(zip(others, levels, strict=True)),
                }
                chance = np.ones(len(w))
                for other in others:
                    p = law.CHANCES[other](w, setting)
                    chance = chance * (p if setting[other] == 1.0 else 1.0 - p)
                regression[level] += chance * law.outcome_probability(w, setting)
        effect += weight * (regression[1.0] - regression[0.0])
    return float(mass @ effect)


def _covariates_only(
    name: str, w: np.ndarray, mass: np.ndarray, others: list[str], navigation_chance: np.ndarray
) -> float:
    """``E_W[E(Y | name = 1, W) - E(Y | name = 0, W)]``, the navigation offer averaged out."""
    regression = {}
    for level in (0.0, 1.0):
        numerator, denominator = np.zeros(len(w)), np.zeros(len(w))
        for navigation in (0.0, 1.0):
            prior = navigation_chance if navigation == 1.0 else 1.0 - navigation_chance
            setting = {NAVIGATION: navigation, name: level}
            p = law.CHANCES[name](w, setting)
            prior = prior * (p if level == 1.0 else 1.0 - p)
            for levels in itertools.product((0.0, 1.0), repeat=len(others)):
                setting.update(zip(others, levels, strict=True))
                chance = prior.copy()
                for other in others:
                    q = law.CHANCES[other](w, setting)
                    chance = chance * (q if setting[other] == 1.0 else 1.0 - q)
                numerator += chance * law.outcome_probability(w, setting)
                denominator += chance
        regression[level] = numerator / denominator
    return float(mass @ (regression[1.0] - regression[0.0]))


def test_the_identity_check_can_fail_for_the_booking() -> None:
    """Adjusting the booking for the covariates alone misses its confounder, the offer."""
    truth = discharge_practices_truth()["adjusted[followup_booked]"]
    wrong = _with_navigation_adjusted(FOLLOWUP, navigation_too=False)
    assert round(wrong, 4) == 0.1037
    assert abs(wrong - truth) > 0.005


@pytest.mark.parametrize("name", [name for name in PRACTICES if name != NAVIGATION])
def test_the_other_practices_need_not_adjust_for_one_another(name: str) -> None:
    """Their ``adjusted`` effect is also what a fit adjusting for the navigation offer gives."""
    truth = discharge_practices_truth()
    assert abs(_with_navigation_adjusted(name) - truth[f"adjusted[{name}]"]) < 1e-12
    assert abs(truth[f"total[{name}]"] - truth[f"adjusted[{name}]"]) < 1e-12


def test_navigations_two_effects_differ_by_the_booked_visits() -> None:
    truth = discharge_practices_truth()
    gap = truth["total[transition_navigation]"] - truth["adjusted[transition_navigation]"]
    assert gap > 0.03


# --------------------------------------------------------------------------- the wrong truths


def test_a_booking_that_ignores_the_offer_fails_the_total_check(
    population: tuple[np.ndarray, dict[str, np.ndarray]],
) -> None:
    """A navigation total whose booking keeps its observed value is the adjusted effect."""
    wrong = discharge_practices_truth()["adjusted[transition_navigation]"]
    assert not _agrees(wrong, _total(population, NAVIGATION, seed=20))


def test_a_booking_that_follows_the_offer_fails_the_adjusted_check(
    population: tuple[np.ndarray, dict[str, np.ndarray]],
) -> None:
    wrong = discharge_practices_truth()["total[transition_navigation]"]
    assert not _agrees(wrong, _adjusted(population, NAVIGATION))


# ------------------------------------------------------------------------- the observed draw


@pytest.fixture(scope="module")
def observed() -> Any:
    frame, _ = make_discharge_practices(MONTE_CARLO_N, seed=30)
    return frame


def test_the_draw_has_the_documented_columns_and_shares(observed: Any) -> None:
    assert list(observed.columns) == ["stayed_home", *PRACTICES, *law.COVARIATES]
    truth = discharge_practices_truth()
    assert _agrees(truth["stayed_home_rate"], _mean(observed["stayed_home"].to_numpy()))
    for name in PRACTICES:
        assert _agrees(truth[f"share[{name}]"], _mean(observed[name].to_numpy())), name


def _residual_association(frame: Any, name: str, omit: str | None = None) -> tuple[float, float]:
    """The mean of the outcome residual times the centred practice ``name``.

    The residual is taken from the law's outcome model, with ``omit`` set to zero when given.
    It averages zero when the draw follows that model, whatever ``name`` is.
    """
    w = frame[list(law.COVARIATES)].to_numpy()
    practices = {other: frame[other].to_numpy() for other in PRACTICES}
    centred = practices[name] - practices[name].mean()
    if omit is not None:
        practices[omit] = np.zeros(len(frame))
    residual = frame["stayed_home"].to_numpy() - law.outcome_probability(w, practices)
    return _mean(residual * centred)


@pytest.mark.parametrize("name", PRACTICES)
def test_the_draw_follows_the_outcome_model(observed: Any, name: str) -> None:
    """The draw's outcome follows the law's outcome model, whatever practice is centred.

    The generator calls that model itself, so this cannot catch a wrong coefficient in it (the
    pinned truths do). It catches a draw that pairs outcomes with the wrong rows or columns.
    """
    assert _agrees(0.0, _residual_association(observed, name))


def test_an_outcome_model_without_the_booking_fails_the_residual_check(observed: Any) -> None:
    assert not _agrees(0.0, _residual_association(observed, FOLLOWUP, omit=FOLLOWUP))


def test_the_backends_draw_the_same_values() -> None:
    pytest.importorskip("polars")
    pandas_frame, _ = make_discharge_practices(200, seed=4, backend="pandas")
    polars_frame, _ = make_discharge_practices(200, seed=4, backend="polars")
    assert list(polars_frame.columns) == list(pandas_frame.columns)
    for column in pandas_frame.columns:
        np.testing.assert_array_equal(polars_frame[column].to_numpy(), pandas_frame[column])

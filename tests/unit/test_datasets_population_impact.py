"""The population-impact law: exact truth, the observed draw, and three wrong truths.

:func:`cleverly.datasets.population_impact_truth` is a sum over the two visit levels inside a
product Gauss--Hermite rule.  A Monte Carlo draw of each intervened law checks every rate and
contrast.  A second draw, of the observed program data, checks the generator: a Horvitz--Thompson
mean with the true response probabilities must recover the current-practice rate, so the draw's
masking is missing at random as documented.

These checks are blind to a term that vanishes at the truth, so three wrong truths must fail the
same comparisons: a visit that ignores the offer, a current-practice rate taken over respondents
only, and a controlled rate read at the other visit level.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_population_impact, population_impact_truth
from cleverly.datasets import population_impact as law

pytestmark = pytest.mark.xdist_group("datasets_population_impact")

#: Patients per intervened arm.  A rate's standard error is at most 0.5 / sqrt(N) = 0.00035, so
#: four standard errors are about 0.0014 for a rate and 0.002 for a contrast of two arms.
MONTE_CARLO_N = 2_000_000
#: The floor added to four Monte Carlo standard errors: the quadrature error of the truth, which
#: 40 and 60 Gauss--Hermite nodes per axis agree to below this bound (checked in this module).
QUADRATURE_FLOOR = 1e-10


def _draw_covariates(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(MONTE_CARLO_N, len(law.COVARIATES)))


def _intervened(
    a: float,
    *,
    held: float | None = None,
    seed: int,
    visit: Callable[[np.ndarray, float], np.ndarray] = law.visit_probability,
) -> np.ndarray:
    """Top-box draws with every patient at offer level ``a``, the visit free or ``held``."""
    rng = np.random.default_rng(seed + 1)
    w = _draw_covariates(seed)
    if held is None:
        z = rng.binomial(1, visit(w, a)).astype(float)
        chance = np.where(
            z == 1.0, law.outcome_probability(w, a, 1.0), law.outcome_probability(w, a, 0.0)
        )
    else:
        chance = law.outcome_probability(w, a, held)
    return rng.binomial(1, chance).astype(float)


def _natural_course(seed: int) -> np.ndarray:
    """Top-box draws under current practice: the offer and the visit follow the law."""
    rng = np.random.default_rng(seed + 1)
    w = _draw_covariates(seed)
    a = rng.binomial(1, law.propensity(w)).astype(float)
    z = rng.binomial(
        1, np.where(a == 1.0, law.visit_probability(w, 1.0), law.visit_probability(w, 0.0))
    )
    chance = np.select(
        [(a == 1.0) & (z == 1), (a == 1.0) & (z == 0), z == 1],
        [
            law.outcome_probability(w, 1.0, 1.0),
            law.outcome_probability(w, 1.0, 0.0),
            law.outcome_probability(w, 0.0, 1.0),
        ],
        law.outcome_probability(w, 0.0, 0.0),
    )
    return rng.binomial(1, chance).astype(float)


def _mean(draws: np.ndarray) -> tuple[float, float]:
    return float(draws.mean()), float(draws.std() / np.sqrt(draws.size))


@pytest.fixture(scope="module")
def monte_carlo() -> dict[str, tuple[float, float]]:
    """The mean and standard error of each intervened rate, keyed like the truth."""
    return {
        "ey_obs": _mean(_natural_course(10)),
        "ey1": _mean(_intervened(1.0, seed=20)),
        "ey0": _mean(_intervened(0.0, seed=30)),
        "cde_ey1[visit=0]": _mean(_intervened(1.0, held=0.0, seed=40)),
        "cde_ey0[visit=0]": _mean(_intervened(0.0, held=0.0, seed=50)),
        "cde_ey1[visit=1]": _mean(_intervened(1.0, held=1.0, seed=60)),
        "cde_ey0[visit=1]": _mean(_intervened(0.0, held=1.0, seed=70)),
    }


def _agrees(value: float, mean: float, error: float) -> bool:
    return abs(value - mean) <= 4 * error + QUADRATURE_FLOOR


def _difference_agrees(
    value: float, first: tuple[float, float], second: tuple[float, float]
) -> bool:
    gap = abs(value - (first[0] - second[0]))
    return gap <= 4 * float(np.hypot(first[1], second[1])) + QUADRATURE_FLOOR


# ------------------------------------------------------------------------------------ the truth


def test_the_truth_is_converged_in_the_quadrature() -> None:
    coarse, fine = law._truth(40), law._truth(60)
    assert set(coarse) == set(fine)
    for key, value in coarse.items():
        assert abs(value - fine[key]) < QUADRATURE_FLOOR, key


def test_the_derived_truths_follow_from_the_rates() -> None:
    truth = population_impact_truth()
    assert truth["ate"] == pytest.approx(truth["ey1"] - truth["ey0"], abs=1e-15)
    assert truth["rr"] == pytest.approx(truth["ey1"] / truth["ey0"], rel=1e-14)
    odds = {a: truth[f"ey{a}"] / (1.0 - truth[f"ey{a}"]) for a in (0, 1)}
    assert truth["or"] == pytest.approx(odds[1] / odds[0], rel=1e-14)
    for reference in (0, 1):
        rate = truth[f"ey{reference}"]
        assert truth[f"par[reference={reference}]"] == pytest.approx(
            truth["ey_obs"] - rate, abs=1e-15
        )
        assert truth[f"paf[reference={reference}]"] == pytest.approx(
            1.0 - rate / truth["ey_obs"], abs=1e-15
        )
    for z in (0, 1):
        assert truth[f"cde_ate[visit={z}]"] == pytest.approx(
            truth[f"cde_ey1[visit={z}]"] - truth[f"cde_ey0[visit={z}]"], abs=1e-15
        )
        assert truth[f"cde_rr[visit={z}]"] == pytest.approx(
            truth[f"cde_ey1[visit={z}]"] / truth[f"cde_ey0[visit={z}]"], rel=1e-14
        )
    # The gain over current practice is the effect among the patients not offered today, times
    # their share; the tutorial's failure mode reads this identity.
    assert -truth["par[reference=1]"] == pytest.approx(
        (1.0 - truth["offered_share"]) * truth["atc"], abs=1e-14
    )
    assert truth["par[reference=0]"] == pytest.approx(
        truth["offered_share"] * truth["att"], abs=1e-14
    )


def test_the_truth_has_the_margins_the_tutorial_narrates() -> None:
    """Current practice reaches the patients who gain most, and the visit carries some effect."""
    truth = population_impact_truth()
    assert truth["att"] - truth["atc"] >= 0.04
    assert truth["ate"] + truth["par[reference=1]"] >= 0.08
    for z in (0, 1):
        assert 0.03 <= truth["ate"] - truth[f"cde_ate[visit={z}]"] <= 0.06
        assert truth[f"cde_ate[visit={z}]"] >= 0.1
    assert 0.70 <= truth["response_rate"] <= 0.80


@pytest.mark.parametrize(
    "key",
    [
        "ey_obs",
        "ey1",
        "ey0",
        "cde_ey1[visit=0]",
        "cde_ey0[visit=0]",
        "cde_ey1[visit=1]",
        "cde_ey0[visit=1]",
    ],
)
def test_each_rate_agrees_with_a_monte_carlo_draw(
    monte_carlo: dict[str, tuple[float, float]], key: str
) -> None:
    mean, error = monte_carlo[key]
    assert _agrees(population_impact_truth()[key], mean, error)


def test_each_contrast_agrees_with_a_monte_carlo_draw(
    monte_carlo: dict[str, tuple[float, float]],
) -> None:
    """The contrasts are tested directly, because two level errors can add."""
    truth = population_impact_truth()
    assert _difference_agrees(truth["ate"], monte_carlo["ey1"], monte_carlo["ey0"])
    for reference in (0, 1):
        assert _difference_agrees(
            truth[f"par[reference={reference}]"],
            monte_carlo["ey_obs"],
            monte_carlo[f"ey{reference}"],
        )
    for z in (0, 1):
        assert _difference_agrees(
            truth[f"cde_ate[visit={z}]"],
            monte_carlo[f"cde_ey1[visit={z}]"],
            monte_carlo[f"cde_ey0[visit={z}]"],
        )


# ------------------------------------------------------------------------- the observed draw


@pytest.fixture(scope="module")
def observed() -> Any:
    frame, _ = make_population_impact(MONTE_CARLO_N, seed=80)
    return frame


def test_the_draw_masks_exactly_the_unreturned_surveys(observed: Any) -> None:
    frame = observed
    assert list(frame.columns) == [
        "top_box",
        "transition_navigation",
        *law.COVARIATES,
        "pcp_visit_7d",
        "responded",
    ]
    assert frame["top_box"].isna().equals(frame["responded"] == 0)
    truth = population_impact_truth()
    for column, key in (("transition_navigation", "offered_share"), ("responded", "response_rate")):
        mean, error = _mean(frame[column].to_numpy())
        assert _agrees(truth[key], mean, error), column
    # Both arms reach both visit levels, so the controlled direct effect has positivity.
    for arm in (0.0, 1.0):
        share = frame.loc[frame["transition_navigation"] == arm, "pcp_visit_7d"].mean()
        assert 0.2 <= share <= 0.8


def _horvitz_thompson(frame: Any) -> tuple[float, float]:
    """The current-practice rate from respondents, weighted by the true response probabilities."""
    w = frame[list(law.COVARIATES)].to_numpy()
    a = frame["transition_navigation"].to_numpy()
    chance = np.where(a == 1.0, law.response_probability(w, 1.0), law.response_probability(w, 0.0))
    terms = np.where(frame["responded"].to_numpy() == 1.0, frame["top_box"].to_numpy(), 0.0)
    return _mean(np.nan_to_num(terms) / chance)


def test_the_observed_draw_recovers_current_practice_under_missing_at_random(
    observed: Any,
) -> None:
    mean, error = _horvitz_thompson(observed)
    assert _agrees(population_impact_truth()["ey_obs"], mean, error)


# --------------------------------------------------------------------------- the wrong truths


def test_a_visit_that_ignores_the_offer_fails_the_monte_carlo_check(
    monte_carlo: dict[str, tuple[float, float]],
) -> None:
    w, mass = law._grid(40)

    def free(a: float) -> float:
        visit = law.visit_probability(w, 0.0)
        mean = visit * law.outcome_probability(w, a, 1.0)
        return float(mass @ (mean + (1.0 - visit) * law.outcome_probability(w, a, 0.0)))

    wrong = free(1.0) - free(0.0)
    assert not _difference_agrees(wrong, monte_carlo["ey1"], monte_carlo["ey0"])


def test_a_respondent_only_current_rate_fails_the_horvitz_thompson_check(
    observed: Any,
) -> None:
    frame = observed
    wrong = float(frame.loc[frame["responded"] == 1, "top_box"].mean())
    mean, error = _horvitz_thompson(frame)
    assert not _agrees(wrong, mean, error)


def test_a_controlled_rate_at_the_other_visit_level_fails_the_monte_carlo_check(
    monte_carlo: dict[str, tuple[float, float]],
) -> None:
    truth = population_impact_truth()
    for z, other in ((0, 1), (1, 0)):
        mean, error = monte_carlo[f"cde_ey1[visit={z}]"]
        assert not _agrees(truth[f"cde_ey1[visit={other}]"], mean, error)

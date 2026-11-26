"""The home-support law: exact group truths, the drawn frame, and deliberately wrong truths.

:func:`cleverly.datasets.home_support_truth` is closed form.  A Monte Carlo draw of the
covariates, with each row's exact potential-outcome chances (no Bernoulli noise), checks every
group effect, direct effect, risk ratio and projection coefficient.  A second set of checks reads
the drawn columns themselves: an inverse-probability-weighted group effect with the true offer
chance, and the share with a 7-day visit in each arm, so a frame wired from the wrong column
fails even when the law's functions are right.

Exact-law checks are blind to a term that vanishes at the truth, so three wrong truths must fail
the same comparisons: the direct effect at visit 1 without the offer-by-visit term, the total
effect without the part the visit adds, and the group effects with "alone" and "paid carer"
swapped.  Two mutated frames must fail the drawn-column checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.bases import Additive, Indicators, Linear, Modifiers, resolve
from cleverly.datasets import (
    HOME_SUPPORT_LEVELS,
    home_support_effect,
    home_support_truth,
    make_home_support,
)
from cleverly.datasets import home_support as law

pytestmark = pytest.mark.xdist_group("datasets_home_support")

#: Rows of the Monte Carlo covariate draw.  A group mean of exact chances has a standard error
#: near 0.07 / sqrt(80,000), about 2.5e-4, against wrong truths 0.023 to 0.12 away.
MONTE_CARLO_N = 400_000
#: Standard errors a Monte Carlo value may sit from its truth.
TOLERANCE_SE = 4.0


@pytest.fixture(scope="module")
def covariates() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261126)
    return {
        "age": rng.uniform(50.0, 90.0, MONTE_CARLO_N),
        "risk": rng.uniform(0.0, 1.0, MONTE_CARLO_N),
        "level": rng.choice(np.asarray(HOME_SUPPORT_LEVELS, dtype=object), MONTE_CARLO_N),
    }


def _arm_mean(w: dict[str, np.ndarray], a: float, held: float | None = None) -> np.ndarray:
    """Each row's chance of staying home with the offer at ``a``, the visit free or ``held``."""
    args = (w["level"], w["age"], w["risk"])
    if held is not None:
        return law.home_chance(a, held, *args)
    visit = law.visit_chance(a, w["risk"])
    return visit * law.home_chance(a, 1.0, *args) + (1.0 - visit) * law.home_chance(a, 0.0, *args)


def _mean_and_se(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _within(value: float, se: float, truth: float) -> bool:
    return abs(value - truth) <= TOLERANCE_SE * se + 1e-12


def _group_effects(w: dict[str, np.ndarray], held: float | None = None) -> dict[str, Any]:
    effect = _arm_mean(w, 1.0, held) - _arm_mean(w, 0.0, held)
    return {level: _mean_and_se(effect[w["level"] == level]) for level in HOME_SUPPORT_LEVELS}


def test_the_frame_has_the_documented_columns_and_levels() -> None:
    frame, truth = make_home_support(n=2_000, seed=0)
    assert list(frame.columns) == [
        "age",
        "discharge_risk",
        "home_support",
        "navigation",
        "pcp_visit_7d",
        "home_30",
    ]
    assert set(frame["home_support"]) == set(HOME_SUPPORT_LEVELS)
    for column in ("navigation", "pcp_visit_7d", "home_30"):
        assert set(frame[column].unique()) <= {0, 1}
    assert frame["age"].between(50, 90).all() and frame["discharge_risk"].between(0, 1).all()
    assert truth == home_support_truth()
    again, _ = make_home_support(n=2_000, seed=0)
    pd.testing.assert_frame_equal(frame, again)
    with pytest.raises(ValueError, match="positive"):
        make_home_support(n=0)


def test_the_frame_builds_in_polars() -> None:
    pl = pytest.importorskip("polars")
    frame, _ = make_home_support(n=50, seed=1, backend="polars")
    assert isinstance(frame, pl.DataFrame)
    assert set(frame["home_support"].to_list()) <= set(HOME_SUPPORT_LEVELS)


def test_every_chance_stays_inside_its_documented_range() -> None:
    """The corners of the covariate box hold the extremes of each linear or logistic chance."""
    corners = pd.MultiIndex.from_product(
        [HOME_SUPPORT_LEVELS, (50.0, 90.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        names=["level", "age", "risk", "offer", "visit"],
    ).to_frame(index=False)
    level, age, risk = corners["level"].to_numpy(), corners["age"], corners["risk"]
    offer = law.offer_chance(level, age, risk)
    visit = law.visit_chance(corners["offer"], risk)
    home = law.home_chance(corners["offer"], corners["visit"], level, age, risk)
    assert (round(offer.min(), 2), round(offer.max(), 2)) == (0.10, 0.92)
    assert (round(visit.min(), 2), round(visit.max(), 2)) == (0.20, 0.65)
    assert (round(home.min(), 2), round(home.max(), 2)) == (0.08, 0.72)


def test_each_group_effect_and_direct_effect_matches_its_truth(
    covariates: dict[str, np.ndarray],
) -> None:
    truth = home_support_truth()
    for level, (value, se) in _group_effects(covariates).items():
        assert _within(value, se, truth[f"group_effect[{level}]"]), level
    for z in (0, 1):
        for level, (value, se) in _group_effects(covariates, held=float(z)).items():
            assert _within(value, se, truth[f"direct_effect[visit={z}][{level}]"]), (z, level)


def test_the_rates_without_the_offer_and_the_risk_ratios_match_their_truth(
    covariates: dict[str, np.ndarray],
) -> None:
    truth = home_support_truth()
    untreated, treated = _arm_mean(covariates, 0.0), _arm_mean(covariates, 1.0)
    for level in HOME_SUPPORT_LEVELS:
        rows = covariates["level"] == level
        rate, se = _mean_and_se(untreated[rows])
        assert _within(rate, se, truth[f"untreated_rate[{level}]"]), level
        ratio = float(np.mean(treated[rows]) / rate)
        # The delta method's standard error of a ratio of two correlated means, bounded above.
        ratio_se = ratio * (se / rate + _mean_and_se(treated[rows])[1] / np.mean(treated[rows]))
        assert _within(ratio, ratio_se, truth[f"risk_ratio[{level}]"]), level


def test_the_projection_coefficients_are_the_least_squares_fit_of_the_row_effects(
    covariates: dict[str, np.ndarray],
) -> None:
    """Regress each row's exact effect on the basis a fit uses, and compare with the truth."""
    truth = home_support_truth()
    rows = pd.DataFrame({"home_support": covariates["level"], "age": covariates["age"]})
    modifiers = Modifiers(
        ["home_support", "age"],
        levels={"home_support": HOME_SUPPORT_LEVELS},
        domains={"age": (50, 90)},
    )
    effect = _arm_mean(covariates, 1.0) - _arm_mean(covariates, 0.0)
    for prefix, basis in (
        ("indicators", Indicators("home_support")),
        ("additive", Additive(Indicators("home_support"), Linear(["age"]))),
    ):
        basis = resolve(basis, modifiers)
        design = np.asarray(basis.evaluate(rows, modifiers), dtype=float)
        coefficients = np.linalg.lstsq(design, effect, rcond=None)[0]
        expected = [truth[f"{prefix}:cate_projection[{term}]"] for term in basis.terms]
        np.testing.assert_allclose(coefficients, expected, atol=2e-3)
        # The law is exactly additive in the group and age, so the fitted line leaves only the
        # variation by discharge risk, which averages to zero within every cell.
    np.testing.assert_allclose(
        home_support_effect(covariates["level"][:1000], covariates["age"][:1000]),
        design[:1000] @ np.asarray(expected),
        atol=1e-12,
    )


def test_the_truth_keys_agree_with_each_other() -> None:
    """The projection keys are the group truths read as a reference level and differences."""
    truth = home_support_truth()
    family = truth["group_effect[family]"]
    assert truth["indicators:cate_projection[(intercept)]"] == family
    assert truth["indicators:cate_projection[home_support[alone]]"] == pytest.approx(
        truth["group_effect[alone]"] - family
    )
    for z in (0, 1):
        assert truth[f"direct[visit={z}]:cate_projection[(intercept)]"] == pytest.approx(
            truth[f"direct_effect[visit={z}][family]"]
        )
    assert np.exp(truth["log_ratio:log_crr_projection[(intercept)]"]) == pytest.approx(
        truth["risk_ratio[family]"]
    )
    assert np.exp(
        truth["log_ratio:log_crr_projection[(intercept)]"]
        + truth["log_ratio:log_crr_projection[home_support[alone]]"]
    ) == pytest.approx(truth["risk_ratio[alone]"])
    # The documented gaps: total minus direct is 0.053 at visit 0 and 0.013 at visit 1.
    for level in HOME_SUPPORT_LEVELS:
        total = truth[f"group_effect[{level}]"]
        assert total - truth[f"direct_effect[visit=0][{level}]"] == pytest.approx(0.053)
        assert total - truth[f"direct_effect[visit=1][{level}]"] == pytest.approx(0.013)


def _wrong_direct_visit_one(level: str) -> float:
    """The direct effect at visit 1 with the ``0.04 A Z`` term dropped."""
    return float(home_support_truth()[f"direct_effect[visit=0][{level}]"])


def _wrong_total(level: str) -> float:
    """The total effect without the part that the visit adds."""
    return float(home_support_truth()[f"direct_effect[visit=0][{level}]"])


def _wrong_swapped(level: str) -> float:
    """The group effects with "alone" and "paid carer" swapped."""
    other = {"alone": "paid carer", "paid carer": "alone"}.get(level, level)
    return float(home_support_truth()[f"group_effect[{other}]"])


@pytest.mark.parametrize(
    ("wrong", "held", "levels"),
    [
        pytest.param(_wrong_direct_visit_one, 1.0, HOME_SUPPORT_LEVELS, id="no-offer-by-visit"),
        pytest.param(_wrong_total, None, HOME_SUPPORT_LEVELS, id="no-part-through-visit"),
        pytest.param(_wrong_swapped, None, ("alone", "paid carer"), id="swapped-levels"),
    ],
)
def test_a_deliberately_wrong_truth_fails_the_same_check(
    covariates: dict[str, np.ndarray], wrong: Any, held: float | None, levels: tuple[str, ...]
) -> None:
    effects = _group_effects(covariates, held=held)
    for level in levels:
        value, se = effects[level]
        assert not _within(value, se, wrong(level)), level


@pytest.fixture(scope="module")
def drawn() -> pd.DataFrame:
    frame, _ = make_home_support(n=MONTE_CARLO_N, seed=7)
    return frame


def _ipw_group_effects(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Each group's effect from the drawn columns, weighted by the true chance of the offer."""
    g = law.offer_chance(frame["home_support"].to_numpy(), frame["age"], frame["discharge_risk"])
    a, y = frame["navigation"].to_numpy(), frame["home_30"].to_numpy()
    terms = a * y / g - (1 - a) * y / (1 - g)
    level = frame["home_support"].to_numpy()
    return {name: _mean_and_se(terms[level == name]) for name in HOME_SUPPORT_LEVELS}


def _ipw_direct_effects(frame: pd.DataFrame, z: int) -> dict[str, tuple[float, float]]:
    """Each group's effect with the visit held at ``z``, weighted by the true offer and visit chances."""
    level = frame["home_support"].to_numpy()
    risk = frame["discharge_risk"].to_numpy()
    g = law.offer_chance(level, frame["age"], risk)
    a, visit = frame["navigation"].to_numpy(), frame["pcp_visit_7d"].to_numpy()
    y = frame["home_30"].to_numpy()
    held = visit == z
    p1 = law.visit_chance(1.0, risk) if z else 1.0 - law.visit_chance(1.0, risk)
    p0 = law.visit_chance(0.0, risk) if z else 1.0 - law.visit_chance(0.0, risk)
    terms = held * (a * y / (g * p1) - (1 - a) * y / ((1 - g) * p0))
    return {name: _mean_and_se(terms[level == name]) for name in HOME_SUPPORT_LEVELS}


def _visit_share_matches(frame: pd.DataFrame) -> bool:
    """Whether the drawn visits give E{Z(1)} = 0.575 and E{Z(0)} = 0.275 by weighting.

    The arms' raw visit shares differ from these, because the offer and the visit both read
    discharge risk; weighting by the true chance of the offer removes that.
    """
    g = law.offer_chance(frame["home_support"].to_numpy(), frame["age"], frame["discharge_risk"])
    a, z = frame["navigation"].to_numpy(), frame["pcp_visit_7d"].to_numpy()
    checks = []
    for terms, expected in ((a * z / g, 0.575), ((1 - a) * z / (1 - g), 0.275)):
        value, se = _mean_and_se(terms)
        checks.append(_within(value, se, expected))
    return all(checks)


def test_the_drawn_columns_recover_the_group_effects_and_the_visit_shares(
    drawn: pd.DataFrame,
) -> None:
    truth = home_support_truth()
    for level, (value, se) in _ipw_group_effects(drawn).items():
        assert _within(value, se, truth[f"group_effect[{level}]"]), level
    for z in (0, 1):
        for level, (value, se) in _ipw_direct_effects(drawn, z).items():
            assert _within(value, se, truth[f"direct_effect[visit={z}][{level}]"]), (z, level)
    assert _visit_share_matches(drawn)
    shares = drawn["home_support"].value_counts(normalize=True)
    for level, share in zip(HOME_SUPPORT_LEVELS, (0.5, 0.3, 0.2), strict=True):
        assert abs(shares[level] - share) < 0.005


def test_a_mutated_frame_fails_the_drawn_column_checks(
    drawn: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation controls on the drawn frame.

    A group column shuffled after the draw, a visit that ignores the offer, and an outcome drawn
    from the chance of a visit rather than the drawn visit each fail a drawn-column check.
    """
    truth = home_support_truth()
    shuffled = drawn.assign(
        home_support=np.random.default_rng(0).permutation(drawn["home_support"].to_numpy())
    )
    value, se = _ipw_group_effects(shuffled)["alone"]
    assert not _within(value, se, truth["group_effect[alone]"])
    original = law.visit_chance
    with monkeypatch.context() as patch:
        patch.setattr(law, "visit_chance", lambda offer, risk: original(0.5, risk))
        blind, _ = make_home_support(n=MONTE_CARLO_N, seed=7)
    assert not _visit_share_matches(blind)
    # The outcome drawn from the chance of a visit instead of the drawn visit keeps every total
    # effect but erases the visit's own part, so the direct effect at visit 0 moves by 0.053.
    home = law.home_chance
    with monkeypatch.context() as patch:
        patch.setattr(
            law,
            "home_chance",
            lambda offer, visit, level, age, risk: home(
                offer, original(offer, risk), level, age, risk
            ),
        )
        unwired, _ = make_home_support(n=MONTE_CARLO_N, seed=7)
    value, se = _ipw_group_effects(unwired)["alone"]
    assert _within(value, se, truth["group_effect[alone]"])
    value, se = _ipw_direct_effects(unwired, 0)["alone"]
    assert not _within(value, se, truth["direct_effect[visit=0][alone]"])

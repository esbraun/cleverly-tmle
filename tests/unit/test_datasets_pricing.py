"""The monthly renewal-price law: exact truth, decomposition by path, plans, and the protocol.

The truth in :func:`cleverly.datasets.pricing_truth` is a Gauss--Legendre integral over usage and
the three natural prices, summed over the eight engagement paths.  A Monte Carlo draw of the
intervened law checks it level by level and contrast by contrast.  The registered study
``mtp-pricing`` holds the same law at other horizons, so the generator equals that law row for row
and the integral at two prices equals that study's quadrature.

Exact-law checks are blind to a term that vanishes at the truth.  The decomposition holds the price
gap of the engagement model, of the engine's price model, or of both at its natural value.  Four
wrong truths must fail the Monte Carlo comparison: an engine that reads the natural last price, an
engagement model that reads it, both, and a renewal model that drops the lag term.  Each held value
must pass a Monte Carlo draw that holds the same path.

The inequality gates keep the lesson of the tutorial when a retune moves a coefficient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import tests.studies.mtp_pricing_common as study_law
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import (
    PRICE_BANDS,
    PRICE_NODES,
    PRICE_PLANS,
    make_sequential_pricing,
    price_rise,
    pricing_protocol,
    pricing_truth,
)
from cleverly.datasets import pricing as law
from cleverly.interventions import Shift

#: The documented size and seed of the tutorial.  The seed is the first of the swept range.
DOCUMENTED_N = 10_000
NOTEBOOK_SEED = 1000
#: Draws in the Monte Carlo comparisons.  The standard error of a contrast is about 6e-5.
MONTE_CARLO_N = 1_000_000
#: The draw that approximates the limit of the regression without engagement.  The sweep probe
#: ``reviews/notebook-review/probes/sequential-pricing/summarize.py`` records the same size and seed.
LIMIT_N = 400_000
LIMIT_SEED = 3_000_001
#: The floor added to four Monte Carlo standard errors: the quadrature error of the truth.
QUADRATURE_FLOOR = 1e-9

PLANS = law._factor_plans((1.06, 1.25))
STUDY_PLANS = ("natural_prices", "price_up", "price_down")


# --------------------------------------------------------------------------- Monte Carlo helpers


def _held_monte_carlo(
    label: str, n: int, seed: int, *, engagement_held: bool = False, engine_held: bool = False
) -> tuple[float, float]:
    """Mean and standard error of the renewal contrast of ``label``, with a path held.

    Draws common random numbers for the plan and for the engine's own prices.  A held path reads
    the natural price gap in the engagement model, in the engine's price mean, or in both.
    """
    rng = np.random.default_rng(seed)
    usage = study_law.w1_draw(rng, n)
    uniforms = [(rng.random(n), rng.random(n)) for _ in range(PRICE_NODES)]
    renewal = {}
    for name in (label, "natural_prices"):
        assign = study_law.MAPS[name]
        gap_assigned, gap_natural = np.zeros(n), np.zeros(n)
        last_engaged, earlier = np.zeros(n), np.zeros(n)
        for node, (u_engaged, u_price) in enumerate(uniforms, start=1):
            read = gap_natural if engagement_held else gap_assigned
            engaged = (u_engaged < study_law.p_engaged(usage, read, last_engaged)).astype(float)
            read = gap_natural if engine_held else gap_assigned
            price = study_law.tn_draw(
                rng,
                study_law.price_mean(usage, engaged, read),
                study_law.lower(engaged),
                study_law.upper(engaged),
                u_price,
            )
            moved = assign(price, engaged)
            if node < PRICE_NODES:
                earlier = earlier + (moved - study_law.centre(engaged))
            gap_assigned = moved - study_law.centre(engaged)
            gap_natural = price - study_law.centre(engaged)
            last_engaged = engaged
        renewal[name] = study_law.purchase(usage, last_engaged, gap_assigned, earlier)
    difference = renewal[label] - renewal["natural_prices"]
    return float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(n))


@pytest.fixture(scope="module")
def monte_carlo() -> dict[str, tuple[float, float]]:
    """Mean and standard error of each plan's renewal, revenue, and contrast, at three prices."""
    drawn = study_law.simulate(MONTE_CARLO_N, seed=2026, tau=PRICE_NODES)
    out = {}
    for label, (renew, price) in drawn.items():
        for key, values in ((f"ey[{label}]", renew), (f"er[{label}]", renew * price)):
            out[key] = (float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size)))
        if label != "natural_prices":
            difference = renew - drawn["natural_prices"][0]
            out[f"ate[{label}]"] = (
                float(difference.mean()),
                float(difference.std(ddof=1) / np.sqrt(difference.size)),
            )
            revenue = renew * price - drawn["natural_prices"][0] * drawn["natural_prices"][1]
            out[f"ate_revenue[{label}]"] = (
                float(revenue.mean()),
                float(revenue.std(ddof=1) / np.sqrt(revenue.size)),
            )
    return out


@pytest.fixture(scope="module")
def regression_limits() -> dict[str, float]:
    """The large-sample limit of the regression with and without engagement, by plan."""
    frame, _ = make_sequential_pricing(LIMIT_N, seed=LIMIT_SEED)
    out = {}
    for name, with_engagement in (("both_held", True), ("omit", False)):
        columns = ["usage"]
        columns += [f"engaged_{k}" for k in range(1, 4)] if with_engagement else []
        columns += [f"price_{k}" for k in range(1, 4)]
        model = LogisticRegression(C=1e6, max_iter=3000).fit(frame[columns], frame["renewed"])
        for label in ("price_up", "price_down"):
            moved = frame[columns].copy()
            for node in range(1, 4):
                moved[f"price_{node}"] = PLANS[label].assign(
                    frame[f"price_{node}"].to_numpy(), frame[f"engaged_{node}"].to_numpy()
                )
            out[f"{name}[{label}]"] = float(
                np.mean(
                    model.predict_proba(moved)[:, 1] - model.predict_proba(frame[columns])[:, 1]
                )
            )
    return out


# ------------------------------------------------------------------------------------ the data


def test_the_columns_equal_the_registered_study_law_row_for_row() -> None:
    names = {"usage": "W1"}
    for node in range(1, 4):
        names[f"engaged_{node}"] = f"L{node}"
        names[f"price_{node}"] = f"A{node}"
    names |= {"renewed": "Y", "revenue": "R"}
    for seed in (0, NOTEBOOK_SEED):
        frame, _ = make_sequential_pricing(DOCUMENTED_N, seed=seed)
        expected = study_law.sample(DOCUMENTED_N, seed, PRICE_NODES)
        assert list(frame.columns) == list(names)
        for column, source in names.items():
            np.testing.assert_allclose(
                frame[column].to_numpy(), expected[source].to_numpy(), rtol=1e-12, atol=0
            )
        for column in ("engaged_1", "engaged_2", "engaged_3", "renewed"):
            assert (frame[column].to_numpy() == expected[names[column]].to_numpy()).all()


def test_the_frame_is_complete_and_inside_the_declared_supports() -> None:
    frame, truth = make_sequential_pricing(DOCUMENTED_N, seed=NOTEBOOK_SEED)
    assert frame.shape == (DOCUMENTED_N, 9)
    assert not frame.isna().any().any()
    assert (frame["usage"].abs() <= 2.5).all()
    for node in range(1, 4):
        engaged = frame[f"engaged_{node}"].to_numpy()
        assert set(np.unique(engaged)) == {0.0, 1.0}
        low, high = 4 + 2 * engaged, 8 + 2 * engaged
        assert (frame[f"price_{node}"].to_numpy() >= low).all()
        assert (frame[f"price_{node}"].to_numpy() <= high).all()
        band = PRICE_BANDS[f"price_{node}"]
        assert callable(band.lower) and callable(band.upper)
        np.testing.assert_array_equal(band.lower(frame), low)
        np.testing.assert_array_equal(band.upper(frame), high)
    np.testing.assert_allclose(frame["revenue"], frame["renewed"] * frame["price_3"], atol=1e-12)
    assert 0 < frame["revenue"].max() <= 10.0
    assert truth == pricing_truth()
    other, _ = make_sequential_pricing(DOCUMENTED_N, seed=NOTEBOOK_SEED)
    assert frame.equals(other)
    assert not frame.equals(make_sequential_pricing(DOCUMENTED_N, seed=NOTEBOOK_SEED + 1)[0])


def test_the_generator_has_no_censoring_keyword() -> None:
    with pytest.raises(TypeError):
        make_sequential_pricing(10, seed=0, censoring=True)  # type: ignore[call-arg]


# ------------------------------------------------------------------------------------ the truth


def test_the_truth_names_every_plan_contrast_and_path() -> None:
    truth = pricing_truth()
    labels = ("natural_prices", "price_up", "price_down", "up_6", "up_25")
    expected = {f"{kind}[{label}]" for kind in ("ey_regimen", "ey_revenue") for label in labels}
    for label in labels[1:]:
        expected |= {
            f"ate_regimen[{label} vs natural_prices]",
            f"ate_revenue[{label} vs natural_prices]",
        }
    for kind in ("ate_engagement_held", "ate_engine_held", "ate_both_held"):
        expected |= {f"{kind}[price_up]", f"{kind}[price_down]"}
    assert set(truth) == expected
    for label in labels[1:]:
        assert truth[f"ate_regimen[{label} vs natural_prices]"] == pytest.approx(
            truth[f"ey_regimen[{label}]"] - truth["ey_regimen[natural_prices]"], abs=1e-15
        )
        assert truth[f"ate_revenue[{label} vs natural_prices]"] == pytest.approx(
            truth[f"ey_revenue[{label}]"] - truth["ey_revenue[natural_prices]"], abs=1e-15
        )
    assert set(pricing_truth(factors=())) == {
        key for key in truth if "up_6" not in key and "up_25" not in key
    }
    assert round(truth["ey_regimen[natural_prices]"], 4) == 0.7301
    assert round(truth["ey_revenue[natural_prices]"], 4) == 5.3765


@pytest.mark.parametrize("label", STUDY_PLANS)
def test_the_truth_is_converged_in_the_quadrature(label: str) -> None:
    coarse = law._mean_outcomes(PLANS[label], (8, 24), engagement_held=False, engine_held=False)
    finer = law._mean_outcomes(PLANS[label], (12, 32), engagement_held=False, engine_held=False)
    assert coarse == pytest.approx(finer, abs=1e-11)


@pytest.mark.parametrize("label", STUDY_PLANS)
@pytest.mark.parametrize("revenue", [False, True])
def test_the_integral_at_two_prices_equals_the_registered_study_quadrature(
    label: str, revenue: bool
) -> None:
    """The study checks its quadrature at orders 48 and 96. This test reads order 48."""
    value = law._mean_outcomes(
        PLANS[label], (12, 32), engagement_held=False, engine_held=False, horizon=2
    )[int(revenue)]
    assert value == pytest.approx(study_law._quadrature(label, 48, revenue), abs=1e-10)


@pytest.mark.parametrize("label", ("price_up", "price_down"))
def test_the_truth_matches_the_monte_carlo_of_the_intervened_law(
    monte_carlo: dict[str, tuple[float, float]], label: str
) -> None:
    truth = pricing_truth()
    for key, exact in (
        (f"ey[{label}]", truth[f"ey_regimen[{label}]"]),
        (f"er[{label}]", truth[f"ey_revenue[{label}]"]),
        (f"ate[{label}]", truth[f"ate_regimen[{label} vs natural_prices]"]),
        (f"ate_revenue[{label}]", truth[f"ate_revenue[{label} vs natural_prices]"]),
    ):
        value, error = monte_carlo[key]
        assert abs(value - exact) < 4 * error + QUADRATURE_FLOOR, key
    for key, exact in (
        ("ey[natural_prices]", truth["ey_regimen[natural_prices]"]),
        ("er[natural_prices]", truth["ey_revenue[natural_prices]"]),
    ):
        value, error = monte_carlo[key]
        assert abs(value - exact) < 4 * error + QUADRATURE_FLOOR, key


def test_the_truth_matches_the_pinned_ten_million_draw_truths_of_the_study() -> None:
    truth = pricing_truth()
    for key, (value, error) in study_law.TRUTHS[PRICE_NODES].items():
        kind, label = key[:-1].split("[")
        exact = {
            "ey": truth[f"ey_regimen[{label}]"],
            "er": truth[f"ey_revenue[{label}]"],
            "ate": truth.get(f"ate_regimen[{label} vs natural_prices]", 0.0),
        }[kind]
        assert abs(value - exact) < 4 * error + QUADRATURE_FLOOR, key


@pytest.mark.parametrize("label", ("price_up", "price_down"))
@pytest.mark.parametrize(
    ("name", "flags"),
    [
        ("ate_engagement_held", {"engagement_held": True}),
        ("ate_engine_held", {"engine_held": True}),
        ("ate_both_held", {"engagement_held": True, "engine_held": True}),
    ],
)
def test_each_held_value_matches_a_monte_carlo_that_holds_the_same_path(
    label: str, name: str, flags: dict[str, bool]
) -> None:
    value, error = _held_monte_carlo(label, MONTE_CARLO_N, 17, **flags)
    assert abs(value - pricing_truth()[f"{name}[{label}]"]) < 4 * error + QUADRATURE_FLOOR


# ------------------------------------------------------------------------- the mutation controls


@pytest.mark.parametrize("label", ("price_up", "price_down"))
def test_the_monte_carlo_rejects_a_truth_that_holds_a_path(
    monte_carlo: dict[str, tuple[float, float]], label: str
) -> None:
    """A truth that reads the natural price in the engine or the engagement model is wrong.

    The mutation is the held integral itself.  The draw must reject it by more than four standard
    errors, so a path term that vanished from the truth would fail here.  Raise ``MONTE_CARLO_N``
    if the control lacks power.
    """
    truth = pricing_truth()
    value, error = monte_carlo[f"ate[{label}]"]
    assert abs(value - truth[f"ate_regimen[{label} vs natural_prices]"]) < 4 * error + 1e-9
    for name in ("ate_engagement_held", "ate_engine_held", "ate_both_held"):
        assert abs(value - truth[f"{name}[{label}]"]) > 4 * error, name


@pytest.mark.parametrize("label", ("price_up", "price_down"))
def test_the_monte_carlo_rejects_a_renewal_model_without_the_lag_term(
    monte_carlo: dict[str, tuple[float, float]], label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(law._LAW, "renew_lag", 0.0)
    plan = PLANS[label]
    mutated = (
        law._mean_outcomes(plan, (8, 24), engagement_held=False, engine_held=False)[0]
        - law._mean_outcomes(
            PLANS["natural_prices"], (8, 24), engagement_held=False, engine_held=False
        )[0]
    )
    value, error = monte_carlo[f"ate[{label}]"]
    assert abs(value - mutated) > 4 * error, "the Monte Carlo cannot see the lag term"


# ------------------------------------------------------------------------------------ the gates


def test_the_gates_keep_the_lesson_of_the_tutorial() -> None:
    """Inequalities on the exact values, so a retune cannot erase the lesson."""
    truth = pricing_truth()
    up, down = "price_up", "price_down"
    ate = {label: truth[f"ate_regimen[{label} vs natural_prices]"] for label in (up, down)}
    assert ate[up] <= -0.05
    assert ate[down] >= 0.05
    for label in (up, down):
        # The regression that holds both paths misses the truth.
        assert abs(truth[f"ate_both_held[{label}]"] - ate[label]) >= 0.015
        # Each path alone supplies a share of the gap, in the direction that shrinks the effect.
        for path in ("engagement", "engine"):
            held = truth[f"ate_{path}_held[{label}]"]
            assert abs(held - ate[label]) >= 0.005, (path, label)
            assert abs(held) < abs(ate[label]), (path, label)
        both = truth[f"ate_both_held[{label}]"]
        assert abs(both) < abs(truth[f"ate_engagement_held[{label}]"])
        assert abs(both) < abs(truth[f"ate_engine_held[{label}]"])
    # The response saturates: a rise of 25% moves the mean less than 25/6 times a rise of 6%.
    ratio = (
        truth["ate_regimen[up_25 vs natural_prices]"] / truth["ate_regimen[up_6 vs natural_prices]"]
    )
    assert 1.0 < ratio <= 3.0
    assert ratio < 25 / 6


def test_the_regression_without_engagement_misses_the_effect_by_more_than_the_regression_with_it(
    regression_limits: dict[str, float],
) -> None:
    truth = pricing_truth()
    for label in ("price_up", "price_down"):
        exact = truth[f"ate_regimen[{label} vs natural_prices]"]
        # The regression with engagement converges to the both-held value, not to the truth.
        assert regression_limits[f"both_held[{label}]"] == pytest.approx(
            truth[f"ate_both_held[{label}]"], abs=1e-3
        )
        # The omit limit has no closed form.  It is an approximation from one draw.
        assert abs(regression_limits[f"omit[{label}]"] - exact) >= 0.05
        assert abs(regression_limits[f"omit[{label}]"]) < abs(
            regression_limits[f"both_held[{label}]"]
        )


def test_the_held_shares_stay_above_the_study_floor() -> None:
    frame, _ = make_sequential_pricing(400_000, seed=11)
    for node in range(1, 4):
        engaged = frame[f"engaged_{node}"].to_numpy()
        price = frame[f"price_{node}"].to_numpy()
        up_held = np.mean(price >= (8 + 2 * engaged) / 1.12)
        down_held = np.mean(price < (4 + 2 * engaged) / 0.85)
        assert 0.15 <= up_held <= 0.25, (node, up_held)
        assert 0.15 <= down_held <= 0.25, (node, down_held)
    last = frame["engaged_3"].to_numpy()
    for factor, share in ((1.06, 0.086), (1.25, 0.45)):
        held = np.mean(frame["price_3"].to_numpy() >= (8 + 2 * last) / factor)
        assert held == pytest.approx(share, abs=0.01)


# ------------------------------------------------------------------------------------ the plans


def _declared_price(policy: Any, price: np.ndarray, history: Any) -> np.ndarray:
    """The dose the declared policy assigns: the map of the piece that holds each price."""
    out = np.array(price, dtype=float)
    if isinstance(policy, Shift):
        assert policy.delta == 0.0 and policy.cap is None
        return out
    for piece in policy.pieces:
        low = np.broadcast_to(
            piece.lower(history) if callable(piece.lower) else piece.lower, price.shape
        )
        high = np.broadcast_to(
            piece.upper(history) if callable(piece.upper) else piece.upper, price.shape
        )
        inside = (price >= low) & (price < high)
        if not piece.is_identity:
            out = np.where(inside, piece.map(price, history), out)
    return out


@pytest.mark.parametrize("label", STUDY_PLANS)
def test_the_declared_policies_assign_the_prices_that_the_truth_assigns(label: str) -> None:
    """The fit reads the declared pieces and the truth reads the plan map: they must agree."""
    frame, _ = make_sequential_pricing(DOCUMENTED_N, seed=3)
    regimen = PRICE_PLANS[label]
    assert regimen.label == label
    assert len(regimen.plan) == PRICE_NODES
    for node, policy in enumerate(regimen.plan, start=1):
        price = frame[f"price_{node}"].to_numpy()
        engaged = frame[f"engaged_{node}"].to_numpy()
        declared = _declared_price(policy, price, frame)
        np.testing.assert_allclose(declared, PLANS[label].assign(price, engaged), atol=1e-12)
        assert (declared >= 4 + 2 * engaged - 1e-12).all()
        assert (declared <= 8 + 2 * engaged + 1e-12).all()
        # A plan other than the engine's own prices moves some prices and holds others.
        if label != "natural_prices":
            assert 0.1 < np.mean(declared == price) < 0.3
            assert np.mean(declared != price) > 0.7


def test_a_plan_reads_the_engagement_of_its_own_month() -> None:
    frame, _ = make_sequential_pricing(DOCUMENTED_N, seed=3)
    for node, policy in enumerate(PRICE_PLANS["price_up"].plan, start=1):
        shifted = frame.copy()
        shifted[f"engaged_{node}"] = 1.0 - frame[f"engaged_{node}"]
        price = frame[f"price_{node}"].to_numpy()
        assert not np.allclose(
            _declared_price(policy, price, frame), _declared_price(policy, price, shifted)
        )


def test_price_rise_names_the_plan_and_refuses_a_cut() -> None:
    assert price_rise(1.12).label == "price_up"
    assert price_rise(1.06).label == "up_6"
    assert price_rise(1.25).label == "up_25"
    assert price_rise(1.25, "bold").label == "bold"
    for factor in (1.0, 0.9):
        with pytest.raises(ValueError, match="factor above 1"):
            price_rise(factor)


def test_the_plans_pass_the_identification_checks_of_the_design() -> None:
    frame, _ = make_sequential_pricing(2_000, seed=NOTEBOOK_SEED)
    design = LongitudinalTreatment(
        outcome="renewed",
        treatment=[f"price_{node}" for node in range(1, 4)],
        baseline=["usage"],
        time_varying=[[f"engaged_{node}"] for node in range(1, 4)],
        continuous_treatment=PRICE_BANDS,
    )
    effect = CausalStudy(frame, design=design, protocol=pricing_protocol()).identify(
        RegimeContrast(PRICE_PLANS, reference="natural_prices")
    )
    lines = effect.summary(detail="full").section("Assumptions").lines
    assert any("strong sequential randomization" in line for line in lines)


# ---------------------------------------------------------------------------------- the protocol


def test_the_protocol_states_three_strategies_and_a_complete_window() -> None:
    protocol = pricing_protocol()
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 3
    assert protocol.horizon == "Month 3, at the renewal after the third monthly price"
    assert any("no censoring" in entry for entry in protocol.intercurrent_event_handling)
    assert protocol.fingerprint == pricing_protocol().fingerprint
    assert len(protocol.fingerprint) == 16
    rationale = " ".join(protocol.assumption_rationale)
    assert "sequential randomization" in rationale and "no interference" in rationale

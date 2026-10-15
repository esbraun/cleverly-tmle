"""The weekly-offer law: exact truth, the logged mechanism, the pilot draw, and the panel layout.

The truth in :func:`cleverly.datasets.offers_truth` is a sum over engagement paths of Gauss--Hermite
integrals.  A Monte Carlo draw of the intervened law checks it level by level and contrast by
contrast.  A second draw, of the logged campaign, checks the estimation sample: a
Horvitz--Thompson estimate with the logged offer probabilities and the true consent probabilities
must recover each level.  The enumeration in this module reuses the law's step functions and the
same path loop, so it checks the bookkeeping of the sum and not the law.

These checks are blind to a term that vanishes at the truth, so three wrong truths must fail the
Monte Carlo comparison: a rule that reads last week's engagement, a fatigue count that drops the
week-3 discount, and a truth weighted by consent (the complete-case law instead of the
hypothetical strategy).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pytest

from cleverly import CausalStudy, LongPanel
from cleverly.datasets import (
    OFFER_LEVELS,
    OFFER_PLANS,
    OFFER_WEEKS,
    make_sequential_offers,
    offers_protocol,
    offers_truth,
    win_back,
    win_back_then_ship,
)
from cleverly.datasets import offers as law

#: The documented size and seed of the tutorial.  The seed is the first of the swept range.
DOCUMENTED_N = 20_000
NOTEBOOK_SEED = 1000
#: Customers per plan in the Monte Carlo comparison.  The standard error is about 0.0007 for a
#: level and 0.001 for a contrast, so four standard errors are 0.004 for a contrast.  A contrast
#: gap of 0.006 is rejected with probability about 0.98.
MONTE_CARLO_N = 500_000
#: Customers in the draw of the logged campaign.  The Horvitz--Thompson standard error is about
#: 0.004 for a level.
LOGGED_N = 1_000_000
#: The floor added to four Monte Carlo standard errors: the quadrature error of the truth, which
#: 48 and 96 Gauss--Hermite nodes agree to below this bound (checked in this module).
QUADRATURE_FLOOR = 1e-6

PLAN_NAMES = tuple(OFFER_PLANS)


def _enumerated_mean(
    rule: Callable[[Any], Any],
    *,
    lagged_rule: bool = False,
    drop_week_three: bool = False,
    weight_by_consent: bool = False,
    nodes: int = 48,
) -> float:
    """Re-derive a plan mean by enumeration, with optional deliberate errors.

    Without an error flag this equals :func:`offers_truth` to machine precision.  Each flag is one
    mutation control: the rule reads ``L_{t-1}`` instead of ``L_t``, the week-3 discount leaves the
    fatigue count, or the paths are weighted by the probability of staying consented.
    """
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    numerator = denominator = 0.0
    for path in itertools.product((0, 1), repeat=OFFER_WEEKS + 1):
        first = law._first_engagement(points)
        mass = first if path[0] == 1 else 1.0 - first
        fatigue = np.zeros_like(points)
        stays = np.ones_like(points)
        for week in range(OFFER_WEEKS):
            seen = path[max(week - 1, 0)] if lagged_rule else path[week]
            offer = np.asarray(rule(np.array([seen])))[0]
            if not (drop_week_three and week == OFFER_WEEKS - 1):
                fatigue = fatigue + float(law._fatigue_step(path[week], offer))
            step = law._next_engagement(path[week], offer, points)
            mass = mass * (step if path[week + 1] == 1 else 1.0 - step)
            stays = stays * law._consent(path[week], points)
        weight = weights * mass * (stays if weight_by_consent else 1.0)
        numerator += float(np.sum(weight * law._outcome_probability(path[-1], points, fatigue)))
        denominator += float(np.sum(weight))
    return numerator / denominator if weight_by_consent else numerator


@pytest.fixture(scope="module")
def monte_carlo() -> dict[str, tuple[float, float]]:
    """The mean and standard error of the week-4 outcome in a deployed-plan cohort, per plan."""
    draws = {}
    for index, name in enumerate(PLAN_NAMES):
        panel, _ = make_sequential_offers(MONTE_CARLO_N, seed=500 + index, policy=name)
        outcome = panel.loc[panel["week"] == OFFER_WEEKS, "full_price"].to_numpy()
        draws[name] = (float(outcome.mean()), float(outcome.std() / np.sqrt(outcome.size)))
    return draws


# ------------------------------------------------------------------------------------ the truth


def test_the_truth_names_every_plan_and_every_contrast() -> None:
    truth = offers_truth()
    expected = {f"ey_regimen[{name}]" for name in PLAN_NAMES}
    expected |= {f"ate_regimen[{name} vs none]" for name in PLAN_NAMES if name != "none"}
    assert set(truth) == expected
    for name in PLAN_NAMES:
        if name != "none":
            assert truth[f"ate_regimen[{name} vs none]"] == pytest.approx(
                truth[f"ey_regimen[{name}]"] - truth["ey_regimen[none]"], abs=1e-15
            )
    assert make_sequential_offers(50, seed=1)[1] == truth


def test_the_truth_is_converged_in_the_quadrature_and_matches_an_independent_enumeration() -> None:
    truth = offers_truth()
    finer = offers_truth(nodes=96)
    for name in PLAN_NAMES:
        key = f"ey_regimen[{name}]"
        assert abs(truth[key] - finer[key]) < QUADRATURE_FLOOR
        assert _enumerated_mean(OFFER_PLANS[name]) == pytest.approx(truth[key], abs=1e-12)


def test_the_truth_orders_the_plans_as_the_tutorial_narrates() -> None:
    """The margins of the design: win-back beats no offers and the discount every week."""
    truth = offers_truth()
    mean = {name: truth[f"ey_regimen[{name}]"] for name in PLAN_NAMES}
    assert mean["win_back"] - mean["none"] >= 0.05
    assert mean["win_back"] - mean["discount"] >= 0.08
    # The ordering is not monotone in the number of discounts: the discount every week loses to
    # no offers, and the win-back rule, which discounts the customers who are not engaged, wins.
    assert mean["discount"] < mean["none"] < mean["win_back"]
    assert mean["win_back_ship"] > mean["win_back"]
    assert mean["shipping"] > mean["none"]


@pytest.mark.parametrize("name", PLAN_NAMES)
def test_the_truth_agrees_with_a_monte_carlo_draw_of_the_intervened_law(
    monte_carlo: dict[str, tuple[float, float]], name: str
) -> None:
    mean, error = monte_carlo[name]
    assert abs(mean - offers_truth()[f"ey_regimen[{name}]"]) <= 4 * error + QUADRATURE_FLOOR


@pytest.mark.parametrize("name", [name for name in PLAN_NAMES if name != "none"])
def test_the_truth_agrees_with_a_monte_carlo_draw_of_each_contrast(
    monte_carlo: dict[str, tuple[float, float]], name: str
) -> None:
    """The contrast is tested directly, because two level errors of 0.003 can add to 0.006."""
    mean, error = monte_carlo[name]
    reference, reference_error = monte_carlo["none"]
    gap = abs((mean - reference) - offers_truth()[f"ate_regimen[{name} vs none]"])
    assert gap <= 4 * float(np.hypot(error, reference_error)) + QUADRATURE_FLOOR


@pytest.fixture(scope="module")
def logged_draw() -> Any:
    """The customers of a logged draw whose outcome is observed, three consecutive rows each."""
    panel, _ = make_sequential_offers(LOGGED_N, seed=900)
    observed = panel.groupby("customer")["full_price"].transform("first").notna()
    kept = panel[observed]
    assert len(kept) % OFFER_WEEKS == 0
    return kept


def _horvitz_thompson(kept: Any, *, consent_weights: bool = True) -> dict[str, tuple[float, float]]:
    """A Horvitz--Thompson estimate of each level from a logged draw, with its standard error.

    The weight of a customer who followed the plan and stayed consenting is one over the product
    of the logged probability of the planned offer and the true probability of consent, at each
    week.  Nothing is fitted, so a mismatch between the logged law and the truth shows as a gap.
    The flag ``consent_weights=False`` drops the consent factor, which is a deliberate error.
    """
    engaged, tenure = kept["engaged"].to_numpy(), kept["tenure"].to_numpy()
    consent = law._consent(engaged, tenure) if consent_weights else np.ones(len(kept))
    estimates = {}
    for name, rule in OFFER_PLANS.items():
        planned = np.asarray(rule(engaged), dtype=object)
        probability = np.take_along_axis(
            kept[[f"p_{level}" for level in OFFER_LEVELS]].to_numpy(),
            np.array([OFFER_LEVELS.index(level) for level in planned])[:, None],
            axis=1,
        )[:, 0]
        factor = np.where(kept["offer"].to_numpy() == planned, 1.0 / (probability * consent), 0.0)
        weight = factor.reshape(-1, OFFER_WEEKS).prod(axis=1)
        outcome = kept["full_price"].to_numpy().reshape(-1, OFFER_WEEKS)[:, 0]
        # Customers lost to consent contribute zero to the sum, and count in the denominator.
        terms = np.zeros(LOGGED_N)
        terms[: weight.size] = weight * outcome
        estimates[name] = (float(terms.mean()), float(terms.std() / np.sqrt(LOGGED_N)))
    return estimates


@pytest.fixture(scope="module")
def logged_estimates(logged_draw: Any) -> dict[str, tuple[float, float]]:
    return _horvitz_thompson(logged_draw)


@pytest.mark.parametrize("name", PLAN_NAMES)
def test_the_truth_agrees_with_a_horvitz_thompson_estimate_on_the_logged_campaign(
    logged_estimates: dict[str, tuple[float, float]], name: str
) -> None:
    """The estimation sample, which the ``policy=`` draws above do not touch, has the truth's law.

    The two branches of the generator share the engagement, outcome and fatigue lines.  They
    differ in the offer and the consent lines, which this estimate reads through the logged
    ``p_*`` columns and the true consent probabilities.  The standard error is about 0.004, so
    this check rejects a gross divergence, such as a consent line that reads the wrong week.
    A finer check is ``reviews/notebook-review/probes/sequential-offers/logged_branch_check.py``,
    which uses 15 million logged customers.
    """
    mean, error = logged_estimates[name]
    assert abs(mean - offers_truth()[f"ey_regimen[{name}]"]) <= 4 * error + QUADRATURE_FLOOR


def test_a_logged_estimate_without_the_consent_weights_fails_the_comparison(
    logged_draw: Any,
) -> None:
    """Mutation control: dropping the consent factor moves every level by more than 10 SE."""
    wrong = _horvitz_thompson(logged_draw, consent_weights=False)
    truth = offers_truth()
    for name, (mean, error) in wrong.items():
        assert abs(mean - truth[f"ey_regimen[{name}]"]) > 10 * error, name


MUTATIONS = {
    "the rule reads last week's engagement": {"lagged_rule": True},
    "the week-3 discount leaves the fatigue count": {"drop_week_three": True},
    "the truth is weighted by consent": {"weight_by_consent": True},
}


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_a_wrong_truth_fails_the_monte_carlo_comparison(
    monte_carlo: dict[str, tuple[float, float]], mutation: str
) -> None:
    """Each deliberate error moves some plan's value by more than four standard errors.

    The comparison above passes the right truth and would pass a truth with a term that
    vanishes at the plans it checks.  This control names the plans where each term does not
    vanish, and shows the comparison rejects the wrong value there.
    """
    rejected = []
    for name in PLAN_NAMES:
        wrong = _enumerated_mean(OFFER_PLANS[name], **MUTATIONS[mutation])
        mean, error = monte_carlo[name]
        if abs(mean - wrong) > 4 * error + QUADRATURE_FLOOR:
            rejected.append(name)
    assert rejected
    if "last week's" in mutation:
        assert set(rejected) == {"win_back", "win_back_ship"}
    if "week-3 discount" in mutation:
        assert rejected == ["discount"]
    if "consent" in mutation:
        assert len(rejected) >= 4


# ------------------------------------------------------------------ the plans and the protocol


def test_the_rules_read_the_week_s_engagement() -> None:
    assert win_back([0, 1, 0, 1]).tolist() == ["discount", "none", "discount", "none"]
    assert win_back_then_ship([0, 1]).tolist() == ["discount", "shipping"]
    for name in ("none", "discount", "shipping"):
        assert set(OFFER_PLANS[name]([0, 1, 1]).tolist()) == {name}
    assert set(OFFER_LEVELS) == {"none", "discount", "shipping"}


def test_the_protocol_pairs_each_strategy_with_a_version() -> None:
    protocol = offers_protocol()
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == len(PLAN_NAMES)
    assert offers_protocol().fingerprint == protocol.fingerprint
    assert any("redeemable in week 4" in event for event in protocol.intercurrent_event_handling)
    assert any("hypothetical" in event for event in protocol.intercurrent_event_handling)


# ------------------------------------------------------------------------ the logged draw


@pytest.fixture(scope="module")
def logged() -> Any:
    panel, _ = make_sequential_offers(DOCUMENTED_N, seed=NOTEBOOK_SEED)
    return panel


def test_the_panel_is_long_with_one_row_per_customer_week_while_observed(logged: Any) -> None:
    assert list(logged.columns) == [
        "customer",
        "week",
        "tenure",
        "engaged",
        "offer",
        "consent",
        "p_none",
        "p_discount",
        "p_shipping",
        "full_price",
    ]
    assert not logged.duplicated(["customer", "week"]).any()
    rows = logged.groupby("customer")["week"].agg(["count", "max"])
    # A customer's rows are the weeks up to the first lost consent, and the last row of a customer
    # who left has consent 0.
    left = logged.loc[logged["consent"] == 0].groupby("customer")["week"].first()
    assert (rows.loc[left.index, "max"] == left).all()
    stayed = rows.index.difference(left.index)
    assert (rows.loc[stayed, "count"] == OFFER_WEEKS).all()
    outcome = logged.groupby("customer")["full_price"].first()
    assert outcome.loc[left.index].isna().all()
    assert outcome.loc[stayed].notna().all()
    # Account closure is outside the law: consent is the only reason a customer leaves.
    assert 0.10 < len(left) / DOCUMENTED_N < 0.20


def test_the_logged_probabilities_are_the_laws_and_the_offers_follow_them(logged: Any) -> None:
    probabilities = logged[["p_none", "p_discount", "p_shipping"]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    for engaged, expected in ((1, (0.40, 0.25, 0.35)), (0, (0.30, 0.45, 0.25))):
        rows = logged[logged["engaged"] == engaged]
        np.testing.assert_allclose(
            rows[["p_none", "p_discount", "p_shipping"]].to_numpy(),
            np.broadcast_to(expected, (len(rows), 3)),
        )
        for week in range(1, OFFER_WEEKS + 1):
            cell = rows[rows["week"] == week]
            for level, share in zip(OFFER_LEVELS, expected, strict=True):
                observed = float((cell["offer"] == level).mean())
                error = np.sqrt(share * (1 - share) / len(cell))
                assert abs(observed - share) < 4 * error, (engaged, week, level)
    # The smallest cumulative product over the weeks is above the default g_bounds lower bound.
    assert probabilities.min() ** OFFER_WEEKS > 0.01


def test_every_static_plan_keeps_a_thin_but_usable_follower_share(logged: Any) -> None:
    """At the documented size each plan keeps at least 1.5% of the customers throughout."""
    wide = logged.pivot(index="customer", columns="week")
    for name, rule in OFFER_PLANS.items():
        followed = np.ones(len(wide), dtype=bool)
        for week in range(1, OFFER_WEEKS + 1):
            engaged = np.nan_to_num(wide[("engaged", week)].to_numpy()).astype(int)
            followed &= wide[("offer", week)].to_numpy() == np.asarray(rule(engaged), dtype=object)
            followed &= wide[("consent", week)].to_numpy() == 1
        assert followed.mean() >= 0.015, name


def test_the_consent_loss_depends_on_the_recorded_history_only(logged: Any) -> None:
    """Retention is a function of engagement and tenure, the recorded history."""
    first = logged[logged["week"] == 1]
    for engaged in (0, 1):
        rows = first[first["engaged"] == engaged]
        expected = float(law._consent(engaged, rows["tenure"].to_numpy()).mean())
        error = np.sqrt(expected * (1 - expected) / len(rows))
        assert abs(float(rows["consent"].mean()) - expected) < 4 * error


def test_the_same_seed_gives_the_same_panel_and_another_seed_does_not() -> None:
    first, _ = make_sequential_offers(500, seed=9)
    second, _ = make_sequential_offers(500, seed=9)
    other, _ = make_sequential_offers(500, seed=10)
    assert first.equals(second)
    assert not first.equals(other)


def test_the_polars_backend_gives_the_same_columns() -> None:
    panel, _ = make_sequential_offers(200, seed=2, backend="polars")
    pandas_panel, _ = make_sequential_offers(200, seed=2)
    assert isinstance(panel, pl.DataFrame)
    assert panel.columns == list(pandas_panel.columns)
    assert panel.shape == pandas_panel.shape


def test_a_long_panel_design_reads_the_panel_with_the_logged_probabilities(logged: Any) -> None:
    design = LongPanel(
        unit="customer",
        time="week",
        treatment="offer",
        baseline=["tenure"],
        time_varying=["engaged"],
        censoring="consent",
        outcome="full_price",
        treatment_probabilities={
            "none": "p_none",
            "discount": "p_discount",
            "shipping": "p_shipping",
        },
    )
    data = CausalStudy(logged, design=design).data
    assert data.n == DOCUMENTED_N
    assert data.known_mechanisms is not None
    # The covariate history of each decision: the baseline and every engagement score so far.
    # Earlier offers are part of the history too, and the identification summary lists them.
    assert data.history_names(1) == ("tenure", "engaged@1")
    assert data.history_names(2) == ("tenure", "engaged@1", "engaged@2")
    assert data.history_names(3) == ("tenure", "engaged@1", "engaged@2", "engaged@3")
    assert data.treatment_levels[0] == ("discount", "none", "shipping")


# --------------------------------------------------------------------------- the pilot draw


@pytest.mark.parametrize("name", PLAN_NAMES)
def test_a_pilot_cohort_follows_the_plan_and_loses_nobody(name: str) -> None:
    panel, truth = make_sequential_offers(3000, seed=21, policy=name)
    assert truth == offers_truth()
    assert (panel.groupby("customer")["week"].count() == OFFER_WEEKS).all()
    assert (panel["consent"] == 1).all()
    assert panel["full_price"].notna().all()
    expected = np.asarray(OFFER_PLANS[name](panel["engaged"].to_numpy()), dtype=object)
    assert (panel["offer"].to_numpy() == expected).all()
    # The pilot's logged probability of the deployed offer is one.
    deployed = panel[[f"p_{level}" for level in OFFER_LEVELS]].to_numpy()
    assert (deployed.max(axis=1) == 1.0).all()
    assert (deployed.sum(axis=1) == 1.0).all()
    assert (deployed.argmax(axis=1) == [OFFER_LEVELS.index(o) for o in expected]).all()


def test_an_unknown_pilot_plan_is_refused() -> None:
    with pytest.raises(ValueError, match="policy must be None or one of"):
        make_sequential_offers(10, seed=1, policy="half price")

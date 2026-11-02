"""The navigation program of the treatment-rules tutorial: exact values, layout, and fold truth.

:func:`cleverly.datasets.make_treatment_rules` computes the value of each static plan and of the
best rule in closed form, and :func:`cleverly.datasets.treatment_rule_value` integrates any policy
on a grid.  A Monte Carlo draw of the intervened law checks both.  Exact-law checks are blind to a
term that vanishes at the truth, so each comparison also has a wrong value that must fail it: the
value of the best rule with the sign of the offer's effect reversed, and a random-offer value that
ignores the offer's chance.  The fold truth of a stacked fit weights each fold by its share of
the rows, and on an unequal split plan the ``1/V`` weighting must fail it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    LearnedRuleValue,
    ModelSpec,
    PointTreatment,
    Runtime,
    SplitPlan,
    TMLEMethod,
)
from cleverly.datasets import learned_rule_truth, make_treatment_rules, treatment_rule_value
from cleverly.datasets import treatment_rules as law
from cleverly.learners import random_partition

#: Draws in each Monte Carlo comparison.
MONTE_CARLO_N = 2_000_000


def _intervened(policy, n: int, seed: int) -> np.ndarray:
    """Draw ``home_30`` for ``n`` patients whose offer follows ``policy``."""
    rng = np.random.default_rng(seed)
    need = rng.uniform(-1.0, 1.0, n)
    alone = rng.binomial(1, 0.5, n).astype(float)
    chance = np.asarray(policy(pd.DataFrame({"need": need, "lives_alone": alone})), dtype=float)
    offer = rng.binomial(1, chance).astype(float)
    return rng.binomial(1, law._home(offer, need, alone)).astype(float)


def _agrees(value: float, draws: np.ndarray) -> bool:
    """``value`` lies within four Monte Carlo standard errors of the draws' mean."""
    se = draws.std(ddof=1) / np.sqrt(draws.size)
    return bool(abs(draws.mean() - value) < 4.0 * se)


POLICIES = {
    "ey_none": lambda data: np.zeros(len(data)),
    "ey_all": lambda data: np.ones(len(data)),
    "ey_best_rule": lambda data: (data["need"] > law.BEST_RULE_CUTOFF).astype(float),
}


@pytest.mark.parametrize("key", sorted(POLICIES))
def test_each_exact_value_matches_a_monte_carlo_draw(key: str) -> None:
    _, truth = make_treatment_rules(n=10, seed=0)
    draws = _intervened(POLICIES[key], MONTE_CARLO_N, seed=101)
    assert _agrees(truth[key], draws)
    # The grid rule agrees with the closed form, to its documented error.
    assert treatment_rule_value(POLICIES[key]) == pytest.approx(truth[key], abs=1e-5)


def test_a_reversed_effect_fails_the_comparison() -> None:
    """The comparison can see the sign of the offer's effect."""
    _, truth = make_treatment_rules(n=10, seed=0)
    draws = _intervened(POLICIES["ey_best_rule"], MONTE_CARLO_N, seed=102)
    # Offering below the cutoff rather than above it gives the worst threshold rule.
    worst = treatment_rule_value(lambda data: data["need"] <= law.BEST_RULE_CUTOFF)
    assert not _agrees(worst, draws)
    assert truth["ey_best_rule"] - worst > 0.05


def test_a_random_offer_value_reads_the_chance_of_the_offer() -> None:
    def random_offer(data):
        return np.where(data["need"] > 0, 0.5, 0.0)

    value = treatment_rule_value(random_offer)
    draws = _intervened(random_offer, MONTE_CARLO_N, seed=103)
    assert _agrees(value, draws)
    # The value is linear in the chance: half of offering to every patient above 0.
    eligible = treatment_rule_value(lambda data: (data["need"] > 0).astype(float))
    none = treatment_rule_value(POLICIES["ey_none"])
    assert value == pytest.approx(0.5 * eligible + 0.5 * none, abs=1e-12)
    # A value that offers every eligible patient fails the comparison.
    assert not _agrees(eligible, draws)


def test_the_best_rule_beats_every_other_threshold() -> None:
    _, truth = make_treatment_rules(n=10, seed=0)
    for cutoff in (-1.0, -0.5, -0.2, 0.0, 0.3, 1.0):
        assert law._threshold_value(cutoff) < truth["ey_best_rule"]
    assert truth["ey_best_rule"] > truth["ey_all"] > truth["ey_none"]
    assert truth["ate"] == truth["ey_all"] - truth["ey_none"]


def test_the_draw_has_the_documented_layout_and_seed() -> None:
    frame, _ = make_treatment_rules(n=500, seed=3)
    assert list(frame.columns) == ["need", "lives_alone", "offer", "home_30"]
    assert frame["need"].between(-1.0, 1.0).all()
    for column in ("lives_alone", "offer", "home_30"):
        assert set(frame[column]) <= {0.0, 1.0}
    again, _ = make_treatment_rules(n=500, seed=3)
    pd.testing.assert_frame_equal(frame, again)


def test_the_offer_is_confounded_by_need_and_living_alone() -> None:
    frame, _ = make_treatment_rules(n=200_000, seed=4)
    offered = frame.groupby("offer")[["need", "lives_alone"]].mean()
    assert (offered.loc[1.0] > offered.loc[0.0]).all()


def test_a_policy_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        treatment_rule_value(lambda data: np.full(len(data), 2.0))
    with pytest.raises(ValueError, match="one probability"):
        treatment_rule_value(lambda data: np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        make_treatment_rules(n=0)


def _learned_fit(seed: int):
    frame, _ = make_treatment_rules(n=1_000, seed=seed)
    outcome = make_pipeline(
        PolynomialFeatures(2, interaction_only=True, include_bias=False),
        LogisticRegression(C=1e6, max_iter=5000),
    )
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=outcome, treatment_learner=LogisticRegression()),
        cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True),
        runtime=Runtime(random_state=1, n_jobs=1),
    )
    design = PointTreatment(
        outcome="home_30", treatment="offer", adjustment=("need", "lives_alone")
    )
    result = CausalStudy(frame, design=design).estimate(LearnedRuleValue(), method=method)
    return result, outcome


def _independent_fold_values(result, outcome, frame: pd.DataFrame) -> list[float]:
    """Refit each fold's model with scikit-learn on the frame, and integrate its rule."""
    columns = ["offer", "need", "lives_alone"]
    fold_values = []
    for train, _test in result.nuisance.folds:
        model = clone(outcome).fit(frame.iloc[train][columns], frame.iloc[train]["home_30"])

        def rule(data: pd.DataFrame, model=model) -> np.ndarray:
            rows = data[["need", "lives_alone"]]
            treated = model.predict_proba(rows.assign(offer=1.0)[columns])[:, 1]
            control = model.predict_proba(rows.assign(offer=0.0)[columns])[:, 1]
            return (treated > control).astype(float)

        fold_values.append(treatment_rule_value(rule))
    return fold_values


def test_the_fold_truth_refits_the_rules_the_fit_used() -> None:
    result, outcome = _learned_fit(seed=5)
    value = learned_rule_truth(result, outcome)
    frame, _ = make_treatment_rules(n=1_000, seed=5)
    fold_values = _independent_fold_values(result, outcome, frame)
    # The folds learn different rules, so a bug that gave every fold one model would show.
    assert len(set(np.round(fold_values, 10))) > 1
    assert value == pytest.approx(float(np.mean(fold_values)), abs=1e-9)
    _, truth = make_treatment_rules(n=10, seed=0)
    # Each fold rule is a threshold near the best one, so its value sits just below the best.
    assert truth["ey_none"] < value <= truth["ey_best_rule"]
    assert truth["ey_best_rule"] - value < 0.01


def test_the_stacked_truth_weights_each_fold_by_its_share_of_the_rows() -> None:
    """X11 (e): on an unequal split plan the stacked target weights fold v by n_v / n."""
    frame, _ = make_treatment_rules(n=1_000, seed=5)
    sizes = np.random.default_rng(0).integers(2, 40, 200)
    labels = np.repeat(np.arange(sizes.size), sizes)[: len(frame)]
    frame = frame.assign(clinic=labels)
    draw = random_partition(len(frame), 5, cluster=labels, seed=2)
    outcome = make_pipeline(
        PolynomialFeatures(2, interaction_only=True, include_bias=False),
        LogisticRegression(C=1e6, max_iter=5000),
    )
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=outcome, treatment_learner=LogisticRegression()),
        cross_fitting=CrossFitting(n_folds=5, split_plan=SplitPlan.from_folds([draw])),
        runtime=Runtime(random_state=1, n_jobs=1),
    )
    design = PointTreatment(
        outcome="home_30", treatment="offer", adjustment=("need", "lives_alone"), cluster="clinic"
    )
    result = CausalStudy(frame, design=design).estimate(LearnedRuleValue(), method=method)
    (split,) = result.extra["learned_rule"].splits
    assert split.scheme == "stacked"
    shares = np.asarray(split.fold_sizes, dtype=float) / len(frame)
    # Nonzero: the plan's folds are unequal, so n_v / n is not 1/V.
    assert np.max(np.abs(shares - 0.2)) > 0.02, shares
    np.testing.assert_allclose(split.fold_weights, shares, rtol=0, atol=1e-15)
    fold_values = _independent_fold_values(result, outcome, frame)
    value = learned_rule_truth(result, outcome)
    assert value == pytest.approx(float(np.dot(shares, fold_values)), abs=1e-9)
    # The 1/V mutation, the fold-evaluated weighting, is a different number on this plan: the
    # draw gives a gap of about 5.8e-4, and the declared floor is 2e-4.
    assert abs(float(np.mean(fold_values)) - value) > 2e-4


def test_the_fold_truth_refuses_a_fit_of_several_splits() -> None:
    frame, _ = make_treatment_rules(n=400, seed=2)
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(C=1e6, max_iter=5000),
            treatment_learner=LogisticRegression(),
        ),
        cross_fitting=CrossFitting(n_folds=5, repeats=2),
        runtime=Runtime(random_state=1, n_jobs=1),
    )
    design = PointTreatment(
        outcome="home_30", treatment="offer", adjustment=("need", "lives_alone")
    )
    result = CausalStudy(frame, design=design).estimate(LearnedRuleValue(), method=method)
    with pytest.raises(ValueError, match="one split"):
        learned_rule_truth(result, LogisticRegression(C=1e6, max_iter=5000))


def test_the_fold_truth_refuses_a_fit_of_another_target() -> None:
    frame, _ = make_treatment_rules(n=400, seed=2)
    design = PointTreatment(
        outcome="home_30", treatment="offer", adjustment=("need", "lives_alone")
    )
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=LogisticRegression(), treatment_learner=LogisticRegression()
        ),
        cross_fitting=CrossFitting(enabled=False),
    )
    result = CausalStudy(frame, design=design).estimate(ATE(), method=method)
    with pytest.raises(ValueError, match="LearnedRuleValue"):
        learned_rule_truth(result, LogisticRegression())


def test_the_fold_truth_refuses_a_learner_that_learns_another_rule() -> None:
    """A refit with another learner changes the rule, and the check catches it."""
    result, _ = _learned_fit(seed=5)
    main_effects = LogisticRegression(C=1e6, max_iter=5000)
    with pytest.raises(ValueError, match="differs"):
        learned_rule_truth(result, main_effects)


def test_the_fold_truth_refuses_a_weighted_fit() -> None:
    """A weighted fit targets the weighted population, which this helper does not integrate."""
    frame, _ = make_treatment_rules(n=400, seed=3)
    frame = frame.assign(w=1.0 + 0.5 * frame["lives_alone"])
    outcome = LogisticRegression(C=1e6, max_iter=5000)
    method = TMLEMethod(
        models=ModelSpec(outcome_learner=outcome, treatment_learner=LogisticRegression()),
        cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True),
        runtime=Runtime(random_state=1, n_jobs=1),
    )
    design = PointTreatment(
        outcome="home_30", treatment="offer", adjustment=("need", "lives_alone"), weights="w"
    )
    result = CausalStudy(frame, design=design).estimate(LearnedRuleValue(), method=method)
    with pytest.raises(ValueError, match="unweighted"):
        learned_rule_truth(result, outcome)

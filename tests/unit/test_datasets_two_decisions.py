"""The two-decision program of the learned-regimen tutorial: law, exact values and fold truth.

:func:`cleverly.datasets.make_two_decision_rules` is the law of the registered study
``learned-regimen-ltmle`` under program names, so a draw must equal the study's draw and each
exact value the study's quadrature.  A Monte Carlo draw of the intervened law checks the values
independently, and a wrong value must fail that check: the best plan with the second decision's
effect reversed.  :func:`cleverly.datasets.learned_regimen_truth` must equal the study harness,
and it must refuse a fit whose stored rules it cannot reproduce.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from cleverly.datasets import (
    TWO_DECISION_COLUMNS,
    learned_regimen_truth,
    make_two_decision_rules,
)
from cleverly.datasets import two_decisions as program
from cleverly.longitudinal import LTMLE, LearnedRegimen
from tests.studies import _learned_regimen_law as study

#: The study's column names, keyed to the program's.
STUDY_NAMES = {
    "W1": "readiness",
    "W2": "lives_alone",
    "A1": "navigation",
    "C1": "tracked_14",
    "L2": "self_care_14",
    "A2": "telemonitoring",
    "C2": "tracked_30",
    "Y": "home_30",
}

MONTE_CARLO_N = 2_000_000


@pytest.mark.parametrize("seed", [7, 2026])
def test_a_draw_is_the_study_draw_renamed(seed: int) -> None:
    frame, _ = make_two_decision_rules(2_000, seed=seed)
    expected = study.draw(2_000, seed).rename(columns=STUDY_NAMES)
    pd.testing.assert_frame_equal(frame, expected[list(frame.columns)])


def test_the_navigation_cutoffs_are_where_the_two_arms_cross() -> None:
    """Each cutoff is a root of the gap between the arms' values, under the best second rule or
    with telemonitoring for nobody, read from the study's quadrature on a fine local grid."""
    _, truth = make_two_decision_rules(5, seed=0)
    best_second = study.optimal_pair()
    for alone, name in ((0.0, "with_others"), (1.0, "alone")):
        for key, slope, offset in (
            (f"navigation_cutoff_{name}", best_second.slope[alone], best_second.offset[alone]),
            (
                f"navigation_cutoff_without_telemonitoring_{name}",
                {0.0: 0.0, 1.0: 0.0},
                {a1: np.full(study.GRID.size, -0.5) for a1 in (0.0, 1.0)},
            ),
        ):
            values = study._arm_values(slope, offset, alone)
            gap = values[1.0] - values[0.0]
            crossing = int(np.argmax(gap > 0.0))
            assert gap[crossing - 1] <= 0.0 < gap[crossing]
            assert study.GRID[crossing - 1] <= truth[key] <= study.GRID[crossing]
    # Ignoring the second decision moves the first rule's cutoff a long way.
    assert truth["navigation_cutoff_without_telemonitoring_with_others"] == pytest.approx(-0.05)
    assert truth["navigation_cutoff_with_others"] > 0.2


def test_the_exact_values_are_the_study_quadrature() -> None:
    _, truth = make_two_decision_rules(5, seed=0)
    statics = study.static_values()
    assert truth["ey_best_rules"] == pytest.approx(study.law_limit()["truth"], abs=1e-12)
    assert truth["ey_neither"] == pytest.approx(statics[0.0, 0.0], abs=1e-12)
    assert truth["ey_navigation_only"] == pytest.approx(statics[1.0, 0.0], abs=1e-12)
    assert truth["ey_telemonitoring_only"] == pytest.approx(statics[0.0, 1.0], abs=1e-12)
    assert truth["ey_both"] == pytest.approx(statics[1.0, 1.0], abs=1e-12)
    assert truth["telemonitoring_cutoff_without_navigation"] == pytest.approx(0.1 / 1.2)
    assert truth["telemonitoring_cutoff_after_navigation"] == pytest.approx(0.7 / 1.2)
    # The best first rule is the one the study's optimal pair holds on the same grid.
    best = study.optimal_pair()
    for alone, name in ((0.0, "with_others"), (1.0, "alone")):
        cutoff = truth[f"navigation_cutoff_{name}"]
        np.testing.assert_array_equal(best.first[alone], (cutoff < study.GRID).astype(float))


def _intervened(first_cutoff: dict[float, float], sign: float, seed: int) -> np.ndarray:
    """``home_30`` under a pair: navigation above a readiness cutoff, telemonitoring where
    ``sign`` times the second decision's effect is positive; nobody leaves follow-up."""
    rng = np.random.default_rng(seed)
    readiness = rng.uniform(-1.0, 1.0, MONTE_CARLO_N)
    alone = rng.binomial(1, 0.5, MONTE_CARLO_N).astype(float)
    cutoff = np.where(alone > 0.5, first_cutoff[1.0], first_cutoff[0.0])
    navigation = (readiness >= cutoff).astype(float)
    self_care = program._l2_mean(navigation, readiness, alone) + rng.standard_normal(MONTE_CARLO_N)
    effect = -0.1 + 1.2 * self_care - 0.6 * navigation
    telemonitoring = (sign * effect > 0.0).astype(float)
    chance = program._home(navigation, telemonitoring, self_care, readiness, alone)
    return rng.binomial(1, chance).astype(float)


def _agrees(value: float, draws: np.ndarray) -> bool:
    se = draws.std(ddof=1) / np.sqrt(draws.size)
    return bool(abs(draws.mean() - value) < 4.0 * se)


def test_the_best_value_agrees_with_a_monte_carlo_draw_and_a_reversed_rule_does_not() -> None:
    _, truth = make_two_decision_rules(5, seed=0)
    cutoffs = {0.0: truth["navigation_cutoff_with_others"], 1.0: truth["navigation_cutoff_alone"]}
    assert _agrees(truth["ey_best_rules"], _intervened(cutoffs, 1.0, seed=1))
    assert not _agrees(truth["ey_best_rules"], _intervened(cutoffs, -1.0, seed=1))
    # A fixed plan: navigation for everyone and telemonitoring for no one.
    everyone = {0.0: -2.0, 1.0: -2.0}
    rng_draws = _intervened(everyone, 0.0, seed=2)
    assert _agrees(truth["ey_navigation_only"], rng_draws)


def _fit(frame: pd.DataFrame, random_state: int = 11):
    return LTMLE(
        LearnedRegimen(),
        n_folds=10,
        random_state=random_state,
        outcome_learner=study.outcome_learner(),
        pseudo_learner=study.declared_node1(),
        treatment_learner=study.nuisance_learner(),
        censoring_learner=study.nuisance_learner(),
    ).fit(frame, **TWO_DECISION_COLUMNS)


@pytest.mark.parametrize(("seed", "random_state"), [(7, 11), (2026, 0)])
def test_the_fold_truth_is_the_study_harness(seed: int, random_state: int) -> None:
    frame, _ = make_two_decision_rules(2_000, seed=seed)
    result = _fit(frame, random_state)
    value = learned_regimen_truth(result, frame)
    study_frame = study.draw(2_000, seed)
    expected = study.harness(
        study.fit(study_frame, "declared", random_state), study_frame, "declared"
    )
    assert value == pytest.approx(expected.truth, abs=1e-12)
    # The fold rules differ from the best pair, so the truth is not the best value.
    assert abs(value - study.law_limit()["truth"]) > 1e-4


def test_the_fold_truth_refuses_rules_it_cannot_reproduce() -> None:
    frame, _ = make_two_decision_rules(2_000, seed=7)
    result = _fit(frame)
    record = result.learned_regimen
    altered = np.array(record.fold_rules, copy=True)
    altered[0, 0, 0] = 1 - altered[0, 0, 0]
    mutated = dataclasses.replace(record, fold_rules=altered)

    class Mutated:
        learned_regimen = mutated
        data = result.data
        folds = result.folds
        estimator = result.estimator

    with pytest.raises(ValueError, match="differs from the fit's on 1 rows"):
        learned_regimen_truth(Mutated(), frame)

    # Another first-decision learner learns other rules, so the check catches it too.
    class OtherLearner(Mutated):
        learned_regimen = record
        estimator = SimpleNamespace(
            repeats=1, outcome_learner=study.outcome_learner(), pseudo_learner=LinearRegression()
        )

    with pytest.raises(ValueError, match="differs from the fit's"):
        learned_regimen_truth(OtherLearner(), frame)

    class Forest(Mutated):
        learned_regimen = record
        estimator = SimpleNamespace(
            repeats=1,
            outcome_learner=RandomForestClassifier(),
            pseudo_learner=study.declared_node1(),
        )

    with pytest.raises(ValueError, match="must be a LogisticRegression"):
        learned_regimen_truth(Forest(), frame)
    other, _ = make_two_decision_rules(2_000, seed=8)
    with pytest.raises(ValueError, match="frame must be the frame"):
        learned_regimen_truth(result, other)


def test_the_protocol_names_the_learned_strategy() -> None:
    protocol = program.two_decision_rules_protocol()
    assert len(protocol.treatment_strategies) == len(protocol.treatment_versions) == 1
    assert "learned" in protocol.treatment_strategies[0]
    assert "day 30" in protocol.outcome


def test_the_fold_truth_refuses_a_fit_without_a_learned_regimen() -> None:
    frame, _ = make_two_decision_rules(500, seed=3)

    class Plain:
        learned_regimen = None

    with pytest.raises(ValueError, match="LearnedRegimen"):
        learned_regimen_truth(Plain(), frame)

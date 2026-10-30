"""X82 part F4: the frozen-rule replay of a learned-rule fit.

A refit of a learned-rule fit relearns the rule inside each training fold, so a perturbed
draw would estimate the value of other rules.  The replay keeps the fit's own folds and the
realized rule of every row.  Each cell then evaluates the value of these fixed rules
fold-wise, with the ``1/V`` weights of the learned-rule record.  These tests pin the
reproduction at zero strength, the frozen regimes, the fold weights, and two mutation
controls that relearn the rule or redraw the folds.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pytest

from cleverly import CausalStudy, LearnedRuleValue, PointTreatment
from cleverly._summary.vocabulary import NOTE_TEXTS
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions.learned import FrozenLearnedRule, freeze_learned_rule
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_request as request_module
from tests.unit import _learned_rule_support as support
from tests.unit._confounding_support import replacement as _replacement

_SEED = 3
_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.2))
#: 403 rows in 5 folds give test folds of 81, 81, 81, 80 and 80 rows, so the ``1/V``
#: weights of the record differ from row weights.
_N = 403


@cache
def _fit() -> Any:
    study = CausalStudy(
        support.law_frame(n=_N),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    return study.identify(LearnedRuleValue()).estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
    )


@cache
def _surface() -> Any:
    return simulated_confounding(_fit(), estimand=support.NAME, grid=_GRID, random_state=_SEED)


def _frozen_refit(result: Any, data: Any) -> Any:
    """The manual replay: the fit's folds and the realized rule of every row."""
    replay = copy.copy(result.estimator)
    replay.learned_rule = FrozenLearnedRule(result.estimator.learned_rule.name, _treat(result))
    return replay.refit(data, random_state=_SEED, frozen_folds=[result.nuisance.folds])


def _treat(result: Any) -> np.ndarray:
    return np.asarray(result.nuisance.regimes.values[:, 1, 0]) > 0.5


def _check_cells_equal_frozen_refits(result: Any, surface: Any) -> None:
    for cell in surface.cells[1:]:
        data = _replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
        manual = _frozen_refit(result, data)
        assert cell.estimate == pytest.approx(manual[support.NAME].psi, abs=1e-12)


def test_zero_strength_reproduces_the_fit_exactly() -> None:
    result, surface = _fit(), _surface()
    assert result.sensitivity.capability("simulated_confounding").available
    assert surface.complete
    assert surface.cells[0].estimate == result[support.NAME].psi
    assert surface.cells[0].displacement == 0.0
    reproduced = _frozen_refit(result, result.data)
    assert reproduced[support.NAME].psi == result[support.NAME].psi


def test_every_cell_is_the_frozen_rule_refit_of_its_perturbed_table() -> None:
    result, surface = _fit(), _surface()
    _check_cells_equal_frozen_refits(result, surface)
    assert max(abs(cell.displacement) for cell in surface.cells) > 5e-3


def test_a_perturbed_refit_keeps_the_realized_rules_and_the_folds() -> None:
    result, surface = _fit(), _surface()
    cell = surface.cells[-1]
    data = _replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
    frozen = _frozen_refit(result, data)
    assert np.array_equal(_treat(frozen), _treat(result))
    assert [test.tolist() for _, test in frozen.nuisance.folds] == [
        test.tolist() for _, test in result.nuisance.folds
    ]
    # Witness: a relearning refit on the same folds and table changes the rule of some rows.
    relearned = result.estimator.refit(
        data, random_state=_SEED, frozen_folds=[result.nuisance.folds]
    )
    assert np.any(_treat(relearned) != _treat(result))


def test_a_cell_averages_the_fold_values_with_equal_weights() -> None:
    result, surface = _fit(), _surface()
    cell = surface.cells[-1]
    data = _replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
    record = _frozen_refit(result, data).extra["learned_rule"]
    assert record.fold_sizes == (81, 81, 81, 80, 80)
    folds = np.asarray(record.fold_estimates)
    sizes = np.asarray(record.fold_sizes, dtype=float)
    assert cell.estimate == pytest.approx(folds.mean(), abs=1e-12)
    # Row weights give another number, so the check separates the two averages.
    assert abs(float(np.average(folds, weights=sizes)) - folds.mean()) > 1e-6


@pytest.mark.parametrize("mutation", ["relearns_the_rule", "redraws_the_folds"])
def test_a_replay_without_the_frozen_rule_or_folds_fails_the_cell_check(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Mutation control: each replay that drops one frozen component moves the cells."""
    result = _fit()
    validate = request_module.validate_learned_rule_replay

    def mutated(fit: Any, estimand: str, key: Any) -> Any:
        replay, folds = validate(fit, estimand, key)
        if mutation == "relearns_the_rule":
            return fit.estimator, folds
        return replay, None

    monkeypatch.setattr(request_module, "validate_learned_rule_replay", mutated)
    surface = simulated_confounding(result, estimand=support.NAME, grid=_GRID, random_state=_SEED)
    assert surface.complete
    with pytest.raises(AssertionError):
        _check_cells_equal_frozen_refits(result, surface)


def test_the_summary_states_the_fixed_rule_reading() -> None:
    summary = _surface().summary()
    assert summary.note("learned_rule_reading").text == NOTE_TEXTS["learned_rule_reading"]
    settings = _surface().summary(detail="full").section("Settings").lines
    assert any("not of the learned-rule procedure" in line for line in settings)


def test_inconsistent_fold_metadata_is_refused() -> None:
    result = _fit()
    original = result.extra["learned_rule"]
    split = replace(original.splits[0], fold_sizes=(80, 81, 81, 81, 80))
    record = replace(original, splits=(split,))
    tampered = replace(result, extra={**result.extra, "learned_rule": record})
    with pytest.raises(CapabilityError, match="inconsistent learned-rule parameter metadata"):
        simulated_confounding(tampered, estimand=support.NAME, grid=_GRID)


def test_the_frozen_rule_checks_its_rows() -> None:
    result = _fit()
    rule = freeze_learned_rule(result)
    assert rule.name == result.estimator.learned_rule.name
    subset = rule.rows(np.array([2, 0, 0]))
    assert subset.treat.tolist() == [rule.treat[2], rule.treat[0], rule.treat[0]]
    with pytest.raises(DataError, match="one boolean per row"):
        FrozenLearnedRule("learned rule", np.zeros(3))
    replay = copy.copy(result.estimator)
    replay.learned_rule = subset
    with pytest.raises(DataError, match="holds 3 rows"):
        replay.refit(result.data, random_state=_SEED, frozen_folds=[result.nuisance.folds])

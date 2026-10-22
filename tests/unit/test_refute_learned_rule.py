"""X82 part F4: refutations of a learned-rule fit that hold the realized rules fixed.

``random_common_cause``, ``subset``, ``bootstrap_stability`` and
``bootstrap_measurement_error`` refit with the fit's own fold rules: one realized rule per
source unit.  A subset or a resample carries each kept unit's rule, a duplicated unit twice.
The measurement-error draw perturbs the covariates after the resample, so each unit keeps
the rule at its true covariates, which is the original target.  Each test below recomputes
the refits by hand or records them, and a mutation control that relearns the rule fails it.
"""

from __future__ import annotations

import copy
import importlib
from functools import cache
from typing import Any

import numpy as np
import pytest

from cleverly import CausalStudy, LearnedRuleValue, PointTreatment
from cleverly.interventions import learned
from cleverly.interventions.learned import FrozenLearnedRule
from cleverly.validation import refute
from cleverly.validation.refute import BootstrapMeasurementError, EmpiricalInclusionRule
from tests.unit import _learned_rule_support as support

# ``cleverly.validation.refute`` names the function, so read the module by its path.
refute_module = importlib.import_module("cleverly.validation.refute")

_SEED = 5


@cache
def _fit() -> Any:
    study = CausalStudy(
        support.law_frame(n=403),
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


def _treat(result: Any) -> np.ndarray:
    return np.asarray(result.nuisance.regimes.values[:, 1, 0]) > 0.5


def _frozen(result: Any, rows: np.ndarray | None = None) -> Any:
    """The manual replay estimator: the realized rule of each original row, by hand."""
    treat = _treat(result)
    replay = copy.copy(result.estimator)
    replay.learned_rule = FrozenLearnedRule(
        result.estimator.learned_rule.name, treat if rows is None else treat[rows]
    )
    return replay


def _relearn(estimator: Any, rule: Any, rows: Any = None) -> Any:
    """Mutation: the replay that relearns the rule on the refit data."""
    return estimator


def test_no_refit_test_meets_a_learned_rule_stop() -> None:
    names = ("random_common_cause", "subset", "bootstrap_stability")
    assert refute_module.refute_refusal(_fit(), estimand=support.NAME, tests=names) is None


def test_random_common_cause_refits_hold_the_rules_and_the_folds() -> None:
    result = _fit()
    report = refute(result, estimand=support.NAME, tests=("random_common_cause",), n_replicates=2)
    (test,) = report.tests
    rng = np.random.default_rng(report.random_state)
    seeds = [repeat.seed for repeat in result.repeats]
    manual = []
    for index in range(2):
        noisy = refute_module._noise_data(result.data, rng, index, False)
        refit = _frozen(result).refit(noisy, random_state=seeds[0], frozen_seeds=seeds)
        assert np.array_equal(_treat(refit), _treat(result))
        assert [t.tolist() for _, t in refit.nuisance.folds] == [
            t.tolist() for _, t in result.nuisance.folds
        ]
        manual.append(refit[support.NAME].psi)
    assert list(test.values) == pytest.approx(manual, abs=1e-12)


def test_a_relearning_noise_refit_fails_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: a noise refit that relearns the rule leaves the hand refits."""
    result = _fit()
    held = refute(result, estimand=support.NAME, tests=("random_common_cause",), n_replicates=2)
    monkeypatch.setattr(learned, "frozen_rule_replay", _relearn)
    relearned = refute(
        result, estimand=support.NAME, tests=("random_common_cause",), n_replicates=2
    )
    assert relearned.tests[0].values != pytest.approx(held.tests[0].values, abs=1e-6)


def test_subset_refits_carry_each_kept_rows_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fit()
    report = refute(result, estimand=support.NAME, tests=("subset",), n_replicates=3)
    (test,) = report.tests
    rng = np.random.default_rng(report.random_state)
    manual = []
    for _ in range(3):
        index, _, _ = refute_module._subset_draw(result.data, rng, 0.7)
        refit = _frozen(result, index).refit(
            result.data.subset(index),
            random_state=report.random_state,
            frozen_folds=refute_module._sliced_folds(result, index),
        )
        assert np.array_equal(_treat(refit), _treat(result)[index])
        manual.append(refit[support.NAME].psi)
    assert list(test.values) == pytest.approx(manual, abs=1e-12)
    # Mutation control: a subset refit that relearns the rule gives other values.
    monkeypatch.setattr(learned, "frozen_rule_replay", _relearn)
    relearned = refute(result, estimand=support.NAME, tests=("subset",), n_replicates=3)
    assert relearned.tests[0].values != pytest.approx(manual, abs=1e-6)


#: The measurement-error options of a four-draw run: the registered rule at the smallest
#: floor its alpha admits, and noise on the covariate the rule reads.
_MEASUREMENT: dict[str, Any] = {
    "bootstrap_measurement_error": BootstrapMeasurementError(variables=("W1",)),
    "measurement_error_rule": EmpiricalInclusionRule(alpha=0.5, minimum_draws=4),
}


@pytest.mark.parametrize(
    ("name", "options"),
    [("bootstrap_stability", {}), ("bootstrap_measurement_error", _MEASUREMENT)],
)
def test_bootstrap_refits_carry_a_duplicated_units_rule_twice(
    monkeypatch: pytest.MonkeyPatch, name: str, options: dict[str, Any]
) -> None:
    result = _fit()
    seen: list[tuple[np.ndarray, Any]] = []
    replay = learned.frozen_rule_replay

    def recording(estimator: Any, rule: Any, rows: Any = None) -> Any:
        built = replay(estimator, rule, rows)
        seen.append((np.asarray(rows), built.learned_rule))
        return built

    monkeypatch.setattr(learned, "frozen_rule_replay", recording)
    report = refute(result, estimand=support.NAME, tests=(name,), n_replicates=4, **options)
    (test,) = report.tests
    assert len(test.values) == 4 and len(seen) == 4
    treat = _treat(result)
    for rows, rule in seen:
        assert np.unique(rows).size < rows.size
        assert np.array_equal(rule.treat, treat[rows])
    # Mutation control: a resample refit that relearns the rule gives other values.
    monkeypatch.setattr(learned, "frozen_rule_replay", _relearn)
    relearned = refute(result, estimand=support.NAME, tests=(name,), n_replicates=4, **options)
    assert relearned.tests[0].values != pytest.approx(test.values, abs=1e-6)

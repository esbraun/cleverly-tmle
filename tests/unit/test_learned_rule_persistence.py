"""A learned-rule fit keeps its rule, its record and its estimate.

The realized rule of each row, the per-fold summaries, the folds and the seeds survive
``save`` and ``cleverly.load``.  ``retarget`` reads the saved rule rather than learning
it again, so it reproduces the fit.  The summary states the data-adaptive target through
:meth:`~cleverly.interventions.LearnedRuleRecord.describe`, and a study fit reports the
same estimate under the typed estimand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly import CausalStudy, LearnedRuleValue, PointTreatment, load
from cleverly.estimators.serialize import dumps, loads
from cleverly.interventions import LearnedRuleRecord
from tests.unit import _learned_rule_support as support


@pytest.fixture(scope="module")
def result() -> Any:
    return support.fit()


def test_the_record_describes_the_fit(result: Any) -> None:
    record = result.extra["learned_rule"]
    assert isinstance(record, LearnedRuleRecord)
    tests = [np.asarray(test) for _, test in result.nuisance.folds]
    assert record.fold_sizes == tuple(test.size for test in tests)
    assert record.fold_weights == (0.2,) * 5
    assert np.mean(record.fold_estimates) == pytest.approx(
        result.estimates[support.NAME].psi, abs=1e-12
    )
    rule = result.nuisance.regimes.values[:, 1, 0]
    np.testing.assert_allclose(record.treated_shares, [np.mean(rule[test]) for test in tests])
    assert record.target == "fold-average data-adaptive value"


def test_the_blip_quantiles_are_those_of_each_complement_fit(result: Any) -> None:
    """A longhand refit on each training complement gives the recorded quantiles."""
    record = result.extra["learned_rule"]
    data = result.data
    design = np.column_stack([data.treatment, data.covariates])
    outcome = np.asarray(data.outcome, dtype=float)
    for (train, test), recorded in zip(result.nuisance.folds, record.blip_quantiles, strict=True):
        model = support.outcome_learner().fit(design[train], outcome[train])
        blip = support.blip(model, data.covariates[test])
        np.testing.assert_allclose(
            recorded, np.quantile(blip, record.quantile_levels), rtol=0, atol=1e-9
        )
    assert record.quantile_levels == (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def test_the_summary_states_the_data_adaptive_target(result: Any) -> None:
    line = result.extra["learned_rule"].describe()
    assert line in result.summary()
    assert "data-adaptive" in line and "average over 5 training-fold rules" in line


@pytest.mark.parametrize("route", ["save", "dumps"])
def test_a_round_trip_keeps_the_rule_the_record_and_the_estimate(
    result: Any, route: str, tmp_path: Any
) -> None:
    restored = (
        load(result.save(tmp_path / "learned.joblib")) if route == "save" else loads(dumps(result))
    )
    assert restored.extra["learned_rule"] == result.extra["learned_rule"]
    np.testing.assert_array_equal(restored.nuisance.regimes.values, result.nuisance.regimes.values)
    estimate, saved = result.estimates[support.NAME], restored.estimates[support.NAME]
    assert saved.psi == estimate.psi and saved.variance == estimate.variance
    np.testing.assert_array_equal(saved.influence_curve, estimate.influence_curve)
    assert restored.config.parameter_axis == "learned_rule"
    # The seeds: the estimator's, and the one that drew the split.
    assert restored.config.random_state == result.config.random_state == 0
    origins = restored.split_plan.provenance
    assert origins == result.split_plan.provenance
    assert [origin.seed for origin in origins] == [0]
    for (train, test), (saved_train, saved_test) in zip(
        result.nuisance.folds, restored.nuisance.folds, strict=True
    ):
        np.testing.assert_array_equal(train, saved_train)
        np.testing.assert_array_equal(test, saved_test)


def test_retarget_reproduces_the_fit(result: Any) -> None:
    estimates, _ = result.estimator.retarget(
        result.data, result.nuisance, estimands=("ey_learned_rule",)
    )
    assert estimates[support.NAME].psi == pytest.approx(
        result.estimates[support.NAME].psi, abs=1e-12
    )
    np.testing.assert_allclose(
        estimates[support.NAME].influence_curve,
        result.estimates[support.NAME].influence_curve,
        atol=1e-12,
        rtol=0,
    )


def test_a_study_fit_reports_the_engine_estimate(result: Any) -> None:
    study = CausalStudy(
        support.law_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    effect = study.identify(LearnedRuleValue())
    assert effect.functional.axis == "learned_rule"
    assert "learned on the training rows of outer fold v" in effect.functional.expression
    assert any(item.startswith("C3, a limiting rule") for item in effect.identification.assumptions)
    fitted = effect.estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
    )
    assert fitted.estimates[support.NAME].psi == pytest.approx(
        result.estimates[support.NAME].psi, abs=1e-12
    )
    key = fitted.parameter_keys[support.NAME]
    assert (key.estimand, key.axis, key.value) == (
        "ey_learned_rule",
        "learned_rule",
        "learned rule",
    )

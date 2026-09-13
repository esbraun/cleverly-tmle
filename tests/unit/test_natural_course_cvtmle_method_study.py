"""Focused contracts for the stacked MAR natural-course comparator study."""

from __future__ import annotations

import numpy as np

from tests.studies import canonical_mar_natural_course_cvtmle as study


def test_the_reference_payload_serializes_the_fitted_predictions_and_folds() -> None:
    frame, _ = study.draw_scenario(study.SCENARIO, 200, 0)
    result = study.fit_cleverly(frame)
    sample = study.reference_sample(frame, result, scenario=study.SCENARIO, replicate=0)

    expected_q = result.nuisance.scaler.unscale_levels(result.nuisance.outcome.observed)
    expected_pi = result.nuisance.missingness_at_realised_arm(result.data.treatment)
    np.testing.assert_array_equal(sample["qn"], expected_q)
    np.testing.assert_array_equal(sample["pin"], expected_pi)
    np.testing.assert_array_equal(sample["fold"], result.nuisance.folds.assignment)
    assert set(sample["A"]) == {0.0, 1.0}
    assert set(sample["fold"]) == set(range(10))


def test_the_r_adapter_selects_the_population_mean_with_supplied_predictions() -> None:
    runner = study.STUDY.artifacts / "run_study.R"
    source = runner.read_text(encoding="utf-8")

    assert "Y = frame$Y" in source
    assert "A = rep(1, n)" in source
    assert "W = data.frame(A_original = frame$A, W = frame$W)" in source
    assert "Delta = frame$Delta" in source
    assert "Q = cbind(frame$qn, frame$qn)" in source
    assert "g1W = rep(1, n)" in source
    assert "pDelta1 = cbind(frame$pin, frame$pin)" in source
    assert 'family = "binomial"' in source
    assert 'fluctuation = "logistic"' in source
    assert "Qbounds = c(0, 1)" in source
    assert "gbound = c(0.01, 1)" in source
    assert "alpha = 0.9995" in source
    assert "cvQinit = FALSE" in source
    assert "prescreenW.g = FALSE" in source
    assert "target.gwt = FALSE" in source
    assert "B = 1" in source
    assert "evalATT = FALSE" in source
    assert "fit$estimates$EY1$var.psi" in source
    assert "fit$estimates$EY1$CI" in source
    assert "A = frame$A" not in source

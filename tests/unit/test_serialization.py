"""Whole-result joblib persistence."""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import FunctionTransformer

from cleverly import (
    ATE,
    CausalStudy,
    LongitudinalTreatment,
    PointTreatment,
    RegimeContrast,
    load,
)
from cleverly.datasets import make_longitudinal, make_nonlinear_ate
from cleverly.estimators.serialize import dumps, loads, save
from cleverly.validation import EmpiricalInclusionRule, GaussianIndependentOutcome


@pytest.fixture
def point_result():  # type: ignore[no-untyped-def]
    frame, _ = make_nonlinear_ate(n=300, seed=7)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
        ),
    )
    return study.estimate(
        ATE(),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=7,
        simultaneous=False,
    )


def test_file_round_trip_retains_the_complete_point_result(point_result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = point_result.save(tmp_path / "analysis.joblib")
    restored = load(path)

    assert restored["ate"].psi == pytest.approx(point_result["ate"].psi)
    assert np.array_equal(restored["ate"].influence_curve, point_result["ate"].influence_curve)
    assert isinstance(restored.estimator.outcome_learner, LinearRegression)
    assert isinstance(restored.estimator.treatment_learner, LogisticRegression)
    assert restored.replayability.refit_nuisances


def test_loaded_result_can_refit_nuisances(point_result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    restored = load(point_result.save(tmp_path / "refit.joblib"))
    # The fixture's own seed, so the gate is a fixed draw rather than one sample of a null.
    # One placebo replicate is 3 sigma of a null draw, which fails for 1 of the seeds 0-299;
    # raising ``n_replicates`` buys no margin, because the gate is 3 sigma at every k below 5.
    report = restored.diagnostics.refute(n_replicates=1, tests=["placebo"], random_state=7)
    assert report.passed


def test_generated_outcome_cache_and_records_survive_round_trip(point_result) -> None:  # type: ignore[no-untyped-def]
    kwargs = {
        "tests": ("dummy_outcome",),
        "dummy_outcome": GaussianIndependentOutcome(),
        "n_replicates": 4,
        # The rule refuses minimum_draws * alpha < 2, so a four-draw budget needs alpha=0.5.
        "outcome_rule": EmpiricalInclusionRule(alpha=0.5, minimum_draws=4),
        "random_state": 17,
    }
    report = point_result.diagnostics.refute(**kwargs)
    assert point_result.diagnostics.refute(**kwargs) is report

    restored = loads(dumps(point_result))
    cached = restored.diagnostics.refute(**kwargs)
    assert cached == report
    assert cached["dummy_outcome"].child_seeds == report["dummy_outcome"].child_seeds


def test_byte_round_trip_matches_file_round_trip(point_result) -> None:  # type: ignore[no-untyped-def]
    restored = loads(dumps(point_result))
    assert restored["ate"].psi == pytest.approx(point_result["ate"].psi)
    assert restored.parameter_keys == point_result.parameter_keys


def test_longitudinal_result_retains_method_and_refit_capability(tmp_path: Path) -> None:
    frame, _ = make_longitudinal(n=180, seed=11)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    )
    original = study.estimate(
        RegimeContrast({"always": 1, "never": 0}, reference="always"),
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=0,
        simultaneous=False,
    )
    restored = load(original.save(tmp_path / "longitudinal.joblib"))
    assert restored.replayability.refit_nuisances
    assert restored.parameter_keys == original.parameter_keys
    assert set(restored.estimates) == set(original.estimates)


def test_legacy_npz_is_refused_with_a_migration_message(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, __manifest__=np.array([1], dtype=np.uint8))
    with pytest.raises(ValueError, match=r"legacy cleverly \.npz"):
        load(path)


def test_legacy_npz_bytes_are_refused() -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, __manifest__=np.array([1], dtype=np.uint8))
    with pytest.raises(ValueError, match=r"legacy cleverly \.npz"):
        loads(buffer.getvalue())


def test_non_result_objects_are_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="fitted causal result"):
        save({"not": "a result"}, tmp_path / "bad.joblib")

    path = tmp_path / "bad-load.joblib"
    joblib.dump({"not": "a result"}, path)
    with pytest.raises(TypeError, match="fitted causal result"):
        load(path)


def test_non_picklable_components_fail_at_save(point_result) -> None:  # type: ignore[no-untyped-def]
    point_result.estimator.outcome_learner = FunctionTransformer(lambda values: values)
    with pytest.raises(TypeError, match="not joblib-serializable"):
        dumps(point_result)

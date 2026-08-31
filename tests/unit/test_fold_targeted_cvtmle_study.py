"""Structural controls for the fold-targeted CV-TMLE comparison study."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

from tests.canonical.zepid_cvtmle.run_zepid_cvtmle import assert_native_split_identity
from tests.studies import fold_targeted_cvtmle as study
from tests.studies.canonical_cvtmle import cv_fit


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return study.draw_from_seed("binary", 400, 731)[0]


@pytest.fixture(scope="module")
def fold_targeted(frame: pd.DataFrame) -> object:
    return study.fit_cleverly(frame)


@pytest.fixture(scope="module")
def pooled(frame: pd.DataFrame) -> object:
    assignment = frame["fold"].to_numpy(dtype=np.int64)
    return cv_fit(
        frame,
        binary=True,
        estimands=study.SUPPORTED,
        n_folds=study.N_FOLDS,
        targeting_scheme="pooled",
        cv_evaluation=True,
        estimator_factory=partial(study.FixedFoldTMLE, assignment),
    )


def test_the_stored_assignment_is_the_fitted_assignment(
    frame: pd.DataFrame, fold_targeted: object
) -> None:
    assignment = frame["fold"].to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(fold_targeted.nuisance.folds.assignment, assignment)  # type: ignore[attr-defined]
    assert np.bincount(assignment).tolist() == [200, 200]


def test_each_training_set_is_the_other_fold(fold_targeted: object) -> None:
    folds = fold_targeted.nuisance.folds  # type: ignore[attr-defined]
    for fold, (train, test) in enumerate(folds):
        np.testing.assert_array_equal(train, np.flatnonzero(folds.assignment != fold))
        np.testing.assert_array_equal(test, np.flatnonzero(folds.assignment == fold))
        assert set(train).isdisjoint(test)
        assert len(train) == len(test)


def test_each_fold_has_a_nonzero_solved_targeting_step(fold_targeted: object) -> None:
    fluctuation = fold_targeted.fluctuations["mean"]  # type: ignore[attr-defined]
    assert len(fluctuation.folds) == study.N_FOLDS
    assert all(np.max(np.abs(record.epsilon)) > 1e-8 for record in fluctuation.folds)
    assert all(np.max(np.abs(record.score)) < 1e-9 for record in fluctuation.folds)
    assert any(
        not np.allclose(record.epsilon, fluctuation.folds[0].epsilon)
        for record in fluctuation.folds[1:]
    )


def test_fold_targeting_is_displaced_from_pooled_targeting(
    fold_targeted: object, pooled: object
) -> None:
    assert fold_targeted.psi("ate") != pooled.psi("ate")  # type: ignore[attr-defined]
    assert fold_targeted.cv_targeting.fold_epsilon  # type: ignore[attr-defined]
    assert pooled.cv_targeting.fold_epsilon == {}  # type: ignore[attr-defined]


@pytest.mark.parametrize("estimand", ("rr", "or"))
def test_fold_evaluated_ratios_remain_refused(frame: pd.DataFrame, estimand: str) -> None:
    assignment = frame["fold"].to_numpy(dtype=np.int64)
    with pytest.raises(ValueError, match="nonlinear parameter"):
        cv_fit(
            frame,
            binary=True,
            estimands=(estimand,),
            n_folds=study.N_FOLDS,
            targeting_scheme="fold",
            cv_evaluation=True,
            estimator_factory=partial(study.FixedFoldTMLE, assignment),
        )


def test_the_registered_row_is_binary_ate_only() -> None:
    assert study.STUDY.scenarios == {"binary": ("ate",)}
    assert study.STUDY.estimands == {"ate"}
    assert study.STUDY.publication_policy == "gated"
    assert study.STUDY.resampling_seed == 20261026


def test_the_subject_row_retains_the_equal_fold_initial_plugin(frame: pd.DataFrame) -> None:
    result = study.fit_cleverly(frame)
    row = study.cleverly_rows(frame, {"ate": 0.0}, "binary", 0)[0]
    q_difference = result.nuisance.outcome.arms[1.0] - result.nuisance.outcome.arms[0.0]
    expected = np.mean(
        [
            result.nuisance.scaler.unscale_difference(float(np.mean(q_difference[test])))
            for _, test in result.nuisance.folds
        ]
    )
    assert row["initial_estimate"] == pytest.approx(expected, rel=1e-12)
    assert abs(float(row["estimate"]) - float(row["initial_estimate"])) > 1e-8


def test_a_changed_native_split_fails_before_it_can_return_rows(frame: pd.DataFrame) -> None:
    native = [frame.loc[frame["fold"] == fold].copy() for fold in range(study.N_FOLDS)]
    moved = native[0].iloc[[0]].copy()
    native[0] = native[0].iloc[1:].copy()
    native[1] = pd.concat([native[1], moved], ignore_index=True)

    with pytest.raises(RuntimeError, match="native split changed"):
        assert_native_split_identity(frame, native)

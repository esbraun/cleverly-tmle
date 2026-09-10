"""Public, reusable outer-fold assignments for point-treatment fits."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    Inference,
    LongitudinalTreatment,
    ModelSpec,
    PointTreatment,
    RegimeMean,
    Runtime,
    SplitPlan,
    TMLEMethod,
)
from cleverly.datasets import make_longitudinal, make_multi_arm, make_nonlinear_ate
from cleverly.estimators.serialize import dumps, loads
from cleverly.estimators.tmle import TMLE as TMLEEngine
from cleverly.exceptions import DataError, MethodConfigurationError

N_FOLDS = 3
SEED = 17


def _method(
    *,
    split_plan: SplitPlan | None = None,
    repeats: int = 1,
    n_jobs: int = 1,
    n_bootstrap: int = 0,
) -> TMLEMethod:
    return TMLEMethod(
        models=ModelSpec(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(
            n_folds=N_FOLDS,
            learner_folds=2,
            repeats=repeats,
            split_plan=split_plan,
        ),
        inference=Inference(simultaneous=False, n_bootstrap=n_bootstrap),
        runtime=Runtime(random_state=SEED, n_jobs=n_jobs),
    )


def _effect(frame: Any, *, cluster: str | None = None) -> Any:
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            cluster=cluster,
        ),
    ).identify(ATE())


def _assert_same_fit(generated: Any, supplied: Any) -> None:
    """Assert identity at every result-determining layer of the fit."""
    assert generated.split_plan == supplied.split_plan
    assert generated.split_plan.fingerprint == supplied.split_plan.fingerprint
    assert generated.provenance.fold_fingerprint == supplied.provenance.fold_fingerprint
    assert supplied.split_plan.fingerprint == supplied.provenance.fold_fingerprint

    assert len(generated.repeats) == len(supplied.repeats)
    for expected, actual in zip(generated.repeats, supplied.repeats, strict=True):
        np.testing.assert_array_equal(expected.folds.assignment, actual.folds.assignment)
        np.testing.assert_array_equal(
            expected.nuisance.propensity.values, actual.nuisance.propensity.values
        )
        np.testing.assert_array_equal(
            expected.nuisance.outcome.observed, actual.nuisance.outcome.observed
        )
        assert expected.nuisance.outcome.arms.keys() == actual.nuisance.outcome.arms.keys()
        for arm, values in expected.nuisance.outcome.arms.items():
            np.testing.assert_array_equal(values, actual.nuisance.outcome.arms[arm])
        assert expected.psi == actual.psi

    assert generated.estimates.keys() == supplied.estimates.keys()
    for name, expected in generated.estimates.items():
        actual = supplied.estimates[name]
        assert expected.psi == actual.psi
        assert expected.variance == actual.variance
        np.testing.assert_array_equal(expected.influence_curve, actual.influence_curve)

    # These are the fitted fold labels that diagnostics and fold-indexed artifacts read.
    for expected, actual in zip(generated.nuisances, supplied.nuisances, strict=True):
        np.testing.assert_array_equal(expected.folds.assignment, actual.folds.assignment)


class TestSplitPlanRecord:
    def test_sequences_are_normalized_to_an_immutable_value(self) -> None:
        source = [[0, 1, 0, 1], [1, 0, 1, 0]]
        plan = SplitPlan(source)

        assert plan.assignments == ((0, 1, 0, 1), (1, 0, 1, 0))
        assert plan.n == 4
        assert plan.n_folds == 2
        assert plan.n_repeats == 2
        assert len(plan.fingerprint) == 16
        source[0][0] = 1
        assert plan.assignments[0][0] == 0
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.assignments = ((0, 1),)  # type: ignore[misc]

    @pytest.mark.parametrize(
        "assignments",
        [
            [],
            [[]],
            [[0, 1], [0]],
            [[0, -1]],
            [[0, 2]],
            [[0.0, 1.5]],
        ],
        ids=(
            "no-repeats",
            "no-rows",
            "unequal-row-counts",
            "negative-label",
            "missing-label",
            "non-integral-label",
        ),
    )
    def test_malformed_assignments_are_refused(self, assignments: Any) -> None:
        with pytest.raises(DataError):
            SplitPlan(assignments)

    def test_materialize_returns_independent_internal_arrays(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]])
        first = plan.to_folds()
        second = plan.to_folds()

        assert len(first) == len(second) == plan.n_repeats
        for left, right, expected in zip(first, second, plan.assignments, strict=True):
            np.testing.assert_array_equal(left.assignment, expected)
            np.testing.assert_array_equal(right.assignment, expected)
            assert left.assignment is not right.assignment

    def test_one_fold_is_a_valid_record_for_an_in_sample_result(self) -> None:
        plan = SplitPlan([[0, 0, 0]])

        assert plan.n_folds == 1
        assert plan.to_folds()[0].is_single

    def test_fingerprint_is_canonical_and_sensitive_to_every_repeat(self) -> None:
        first = SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]])
        same = SplitPlan(np.asarray(first.assignments, dtype=np.int32))
        reordered = SplitPlan(tuple(reversed(first.assignments)))

        assert same == first
        assert hash(same) == hash(first)
        assert same.fingerprint == first.fingerprint
        assert reordered.fingerprint != first.fingerprint


class TestCrossFittingConfiguration:
    def test_flat_shortcut_sets_the_plan_on_the_normalized_configuration(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1]])

        method = TMLEMethod().with_overrides(n_folds=2, repeats=1, split_plan=plan)

        assert method.cross_fitting.split_plan is plan

    def test_fold_count_must_match_the_supplied_plan(self) -> None:
        plan = SplitPlan([[0, 1, 2, 0, 1, 2]])
        with pytest.raises(MethodConfigurationError, match="n_folds"):
            CrossFitting(n_folds=2, split_plan=plan)

    def test_repeat_count_must_match_the_supplied_plan(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]])
        with pytest.raises(MethodConfigurationError, match="repeats"):
            CrossFitting(n_folds=2, repeats=1, split_plan=plan)

    def test_a_plan_conflicts_with_disabled_cross_fitting(self) -> None:
        plan = SplitPlan([[0, 1, 0, 1]])
        with pytest.raises(MethodConfigurationError, match=r"enabled|cross.?fit"):
            CrossFitting(enabled=False, n_folds=2, split_plan=plan)

    def test_a_one_fold_plan_cannot_be_supplied_as_cross_fitting(self) -> None:
        plan = SplitPlan([[0, 0, 0]])
        with pytest.raises(MethodConfigurationError, match=r"n_folds|cross.?fit"):
            CrossFitting(enabled=True, n_folds=1, split_plan=plan)

    def test_only_a_split_plan_is_accepted(self) -> None:
        with pytest.raises(MethodConfigurationError, match="split_plan"):
            CrossFitting(n_folds=2, split_plan=[[0, 1]])  # type: ignore[arg-type]


class TestDataBoundValidationPrecedesNuisanceFitting:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(9)
        cells = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (12, 1))
        return pd.DataFrame(
            {
                "Y": cells[:, 1].astype(float),
                "A": cells[:, 0].astype(float),
                "W1": rng.normal(size=len(cells)),
                "W2": rng.normal(size=len(cells)),
                "W3": rng.normal(size=len(cells)),
                "pid": np.repeat(np.arange(len(cells) // 2), 2),
            }
        )

    def _refuse_before_fit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        frame: pd.DataFrame,
        plan: SplitPlan,
        *,
        stratify_by: str = "treatment",
        cluster: str | None = None,
    ) -> None:
        def unexpected(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("nuisance fitting began before the supplied split was validated")

        monkeypatch.setattr(TMLEEngine, "_nuisances", unexpected)
        method = dataclasses.replace(
            _method(),
            cross_fitting=CrossFitting(
                n_folds=plan.n_folds,
                learner_folds=2,
                repeats=plan.n_repeats,
                stratify_by=stratify_by,  # type: ignore[arg-type]
                split_plan=plan,
            ),
        )
        with pytest.raises(DataError):
            _effect(frame, cluster=cluster).estimate(method=method)

    def test_row_count_is_checked_before_fit(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.arange(len(frame) - 1) % N_FOLDS
        self._refuse_before_fit(monkeypatch, frame, SplitPlan([assignment]))

    def test_each_training_complement_keeps_every_treatment_arm(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.ones(len(frame), dtype=int)
        assignment[frame["A"].to_numpy() == 1] = 0
        self._refuse_before_fit(monkeypatch, frame, SplitPlan([assignment]))

    def test_each_training_complement_keeps_every_requested_stratum(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.ones(len(frame), dtype=int)
        rare_cell = (frame["A"].to_numpy() == 1) & (frame["Y"].to_numpy() == 1)
        assignment[rare_cell] = 0
        # Keep both treatment arms in fold 0, so the treatment-support check alone passes.
        assignment[[0, 2]] = 0
        self._refuse_before_fit(
            monkeypatch,
            frame,
            SplitPlan([assignment]),
            stratify_by="treatment+outcome",
        )

    def test_a_declared_cluster_cannot_be_split(
        self, frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assignment = np.arange(len(frame)) % 2
        self._refuse_before_fit(monkeypatch, frame, SplitPlan([assignment]), cluster="pid")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_generated_and_supplied_binary_plans_are_exactly_identical(backend: str) -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=4, backend=backend)
    effect = _effect(frame)
    generated = effect.estimate(method=_method())
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan))

    _assert_same_fit(generated, supplied)
    assert supplied.data.backend == backend


def test_generated_and_supplied_multi_arm_plans_are_exactly_identical() -> None:
    frame, _ = make_multi_arm(n=180, seed=7)
    effect = _effect(frame)
    generated = effect.estimate(method=_method())
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan))

    _assert_same_fit(generated, supplied)
    assert generated.nuisance.propensity.values.shape[1] == 3


def test_generated_and_supplied_clustered_plans_are_exactly_identical() -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=11)
    frame = frame.copy()
    frame["pid"] = np.repeat(np.arange(60), 3)
    effect = _effect(frame, cluster="pid")
    generated = effect.estimate(method=_method())
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan))

    _assert_same_fit(generated, supplied)
    cluster = supplied.data.cluster
    assert cluster is not None
    for assignment in supplied.split_plan.assignments:
        labels = np.asarray(assignment)
        assert all(np.unique(labels[cluster == code]).size == 1 for code in np.unique(cluster))


def test_generated_and_supplied_repeated_plans_are_exactly_identical() -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=19)
    effect = _effect(frame)
    generated = effect.estimate(method=_method(repeats=3))
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan, repeats=3))

    _assert_same_fit(generated, supplied)
    assert supplied.split_plan.n_repeats == supplied.n_repeats == 3
    assert supplied.config.crossfit.scheme == "supplied"
    assert supplied.config.crossfit.stratify_by == ()


def test_the_realized_plan_and_fingerprint_survive_result_persistence() -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=23)
    effect = _effect(frame)
    generated = effect.estimate(method=_method(repeats=2))
    supplied = effect.estimate(method=_method(split_plan=generated.split_plan, repeats=2))
    restored = loads(dumps(supplied))

    assert restored.split_plan == supplied.split_plan == generated.split_plan
    assert restored.method.cross_fitting.split_plan == supplied.split_plan
    assert restored.split_plan.fingerprint == restored.provenance.fold_fingerprint
    for expected, actual in zip(supplied.repeats, restored.repeats, strict=True):
        np.testing.assert_array_equal(expected.folds.assignment, actual.folds.assignment)


def test_an_in_sample_result_exposes_the_realized_one_fold_plan() -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=27)
    method = dataclasses.replace(
        _method(),
        cross_fitting=CrossFitting(enabled=False, learner_folds=2),
    )
    result = _effect(frame).estimate(method=method)

    assert result.split_plan.n_folds == 1
    assert result.split_plan.n_repeats == 1
    assert result.split_plan.assignments == ((0,) * len(frame),)
    assert result.split_plan.fingerprint == result.provenance.fold_fingerprint


def test_a_supplied_plan_is_exact_under_serial_and_parallel_scheduling() -> None:
    frame, _ = make_nonlinear_ate(n=180, seed=29)
    effect = _effect(frame)
    generated = effect.estimate(method=_method())
    serial = effect.estimate(method=_method(split_plan=generated.split_plan, n_jobs=1))
    parallel = effect.estimate(method=_method(split_plan=generated.split_plan, n_jobs=2))

    _assert_same_fit(serial, parallel)


def test_a_split_plan_is_refused_for_a_longitudinal_fit_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = make_longitudinal(n=120, seed=31)
    effect = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    ).identify(RegimeMean({"always": 1}))
    plan = SplitPlan([np.arange(len(frame)) % N_FOLDS])

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the longitudinal engine was constructed before refusing split_plan")

    monkeypatch.setattr("cleverly.study.LTMLE", unexpected)
    with pytest.raises(MethodConfigurationError, match="split_plan"):
        effect.estimate(method=_method(split_plan=plan))


def test_a_split_plan_is_refused_for_bootstrap_before_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, _ = make_nonlinear_ate(n=120, seed=37)
    assignment = np.arange(len(frame)) % N_FOLDS
    plan = SplitPlan([assignment])

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("the point engine was constructed before refusing split_plan with bootstrap")

    monkeypatch.setattr("cleverly.study.TMLE", unexpected)
    with pytest.raises(MethodConfigurationError, match=r"split_plan|bootstrap"):
        _effect(frame).estimate(method=_method(split_plan=plan, n_bootstrap=4))

"""Simulated common-cause stress surfaces and their mutation controls."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_binary_outcome, make_linear_ate
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import (
    ConfounderStrengthGrid,
    SimulatedConfoundingResult,
    simulated_confounding,
)
from cleverly.sensitivity import simulated_confounding as public_function
from cleverly.sensitivity.simulated_confounding import (
    _binary_calibration,
    _child_seeds,
    _flip_binary,
    _flip_mask,
    _gaussian_outcome,
)


def _fit(*, family: str = "gaussian", backend: str = "pandas", seed: int = 7) -> Any:
    if family == "gaussian":
        frame, _ = make_linear_ate(n=120, seed=seed, backend=backend)
        covariates = ("W1", "W2", "W3", "W4")
        outcome_learner = LinearRegression()
    else:
        frame, _ = make_binary_outcome(n=120, seed=seed, backend=backend)
        covariates = ("W1", "W2", "W3")
        outcome_learner = LogisticRegression(max_iter=1000)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=covariates),
    )
    return study.identify(ATE()).estimate(
        outcome_learner=outcome_learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        simultaneous=False,
    )


@pytest.fixture(scope="module")
def gaussian_result() -> Any:
    return _fit()


@pytest.fixture(scope="module")
def binomial_result() -> Any:
    return _fit(family="binomial")


def _record_refits(
    result: Any, monkeypatch: pytest.MonkeyPatch, *, fail_call: int | None = None
) -> list[tuple[Any, int | None]]:
    calls: list[tuple[Any, int | None]] = []

    def refit(
        data: Any,
        *,
        intermediate_value: float | None = None,
        random_state: int | None = None,
    ) -> Any:
        del intermediate_value
        calls.append((data, random_state))
        if fail_call is not None and len(calls) == fail_call:
            raise RuntimeError("deliberate refit failure")
        treated = data.outcome[data.treatment == 1.0]
        control = data.outcome[data.treatment == 0.0]
        psi = float(np.mean(treated) - np.mean(control))
        estimate = replace(result["ate"], psi=psi)
        return replace(result, data=data, estimates={"ate": estimate})

    monkeypatch.setattr(result.estimator, "refit", refit)
    return calls


def _grid() -> ConfounderStrengthGrid:
    return ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))


def test_zero_anchor_common_randomness_and_original_data_per_cell(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = gaussian_result
    original_treatment = result.data.treatment.copy()
    original_outcome = result.data.outcome.copy()
    calls = _record_refits(result, monkeypatch)

    surface = simulated_confounding(result, grid=_grid(), random_state=19)

    assert surface.cells[0].estimate == result["ate"].psi
    assert surface.cells[0].displacement == 0.0
    assert len(calls) == 3
    assert {seed for _, seed in calls} == {surface.refit_seed}
    latent = np.random.default_rng(surface.latent_seed).normal(size=result.data.n)
    expected_treatment = _flip_binary(original_treatment, _flip_mask(latent, 0.1))
    expected_outcome = _gaussian_outcome(original_outcome, latent, 0.2)
    assert np.array_equal(calls[0][0].treatment, original_treatment)
    assert np.array_equal(calls[0][0].outcome, expected_outcome)
    assert np.array_equal(calls[1][0].treatment, expected_treatment)
    assert np.array_equal(calls[1][0].outcome, original_outcome)
    assert np.array_equal(calls[2][0].treatment, expected_treatment)
    assert np.array_equal(calls[2][0].outcome, expected_outcome)
    assert np.array_equal(result.data.treatment, original_treatment)
    assert np.array_equal(result.data.outcome, original_outcome)


def test_gaussian_sign_and_each_flip_mask_are_active_mutation_controls() -> None:
    latent = np.array([-2.0, -0.1, 0.4, 2.0])
    values = np.array([0.0, 1.0, 0.0, 1.0])
    mask = _flip_mask(latent, 0.25)
    assert np.array_equal(mask, np.array([False, False, False, True]))
    assert np.array_equal(_flip_binary(values, mask), np.array([0.0, 1.0, 0.0, 0.0]))

    gaussian = _gaussian_outcome(values, latent, 0.5)
    sign_reversed = values + 0.5 * latent
    disabled_mask = _flip_binary(values, np.zeros_like(mask))
    assert np.array_equal(gaussian, values - 0.5 * latent)
    assert not np.array_equal(gaussian, sign_reversed)
    assert not np.array_equal(_flip_binary(values, mask), disabled_mask)


def test_binary_treatment_and_outcome_share_the_exact_latent_masks(
    binomial_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_refits(binomial_result, monkeypatch)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0, 0.3))
    surface = simulated_confounding(binomial_result, grid=grid, random_state=31)
    latent = np.random.default_rng(surface.latent_seed).normal(size=binomial_result.data.n)

    assert np.array_equal(
        calls[1][0].treatment,
        _flip_binary(binomial_result.data.treatment, _flip_mask(latent, 0.15)),
    )
    assert np.array_equal(
        calls[0][0].outcome,
        _flip_binary(binomial_result.data.outcome, _flip_mask(latent, 0.3)),
    )
    assert np.array_equal(calls[2][0].treatment, calls[1][0].treatment)
    assert np.array_equal(calls[2][0].outcome, calls[0][0].outcome)


def test_seed_replay_different_seed_response_and_unseeded_recording(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_refits(gaussian_result, monkeypatch)
    first = public_function(gaussian_result, grid=_grid(), random_state=41)
    first_arrays = [(data.treatment.copy(), data.outcome.copy()) for data, _ in calls]
    calls.clear()
    replay = public_function(gaussian_result, grid=_grid(), random_state=41)
    replay_arrays = [(data.treatment.copy(), data.outcome.copy()) for data, _ in calls]
    calls.clear()
    different = public_function(gaussian_result, grid=_grid(), random_state=42)
    different_arrays = [(data.treatment.copy(), data.outcome.copy()) for data, _ in calls]

    assert replay.root_seed == first.root_seed == 41
    assert replay.latent_seed == first.latent_seed
    assert all(
        np.array_equal(left_a, right_a) and np.array_equal(left_y, right_y)
        for (left_a, left_y), (right_a, right_y) in zip(first_arrays, replay_arrays, strict=True)
    )
    assert different.latent_seed != first.latent_seed
    assert any(
        not np.array_equal(first_a, different_a) or not np.array_equal(first_y, different_y)
        for (first_a, first_y), (different_a, different_y) in zip(
            first_arrays, different_arrays, strict=True
        )
    )

    unseeded = replace(gaussian_result, estimator=replace_estimator_seed(gaussian_result, None))
    _record_refits(unseeded, monkeypatch)
    recorded = simulated_confounding(
        unseeded,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
    )
    assert isinstance(recorded.root_seed, int)
    assert recorded.root_seed >= 0


def replace_estimator_seed(result: Any, seed: int | None) -> Any:
    """Copy a fitted estimator and change only its assessment fallback seed."""
    import copy

    estimator = copy.copy(result.estimator)
    estimator.random_state = seed
    return estimator


def test_failed_refits_and_arm_loss_remain_visible(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_refits(gaussian_result, monkeypatch, fail_call=1)
    failed = simulated_confounding(gaussian_result, grid=_grid(), random_state=23)
    assert len(calls) == 3
    assert not failed.complete
    assert failed.failures[0].error_type == "RuntimeError"
    assert "deliberate refit failure" in failed.failures[0].message
    assert failed.cells[1].estimate is None
    assert failed.failures[0].seed == failed.root_seed == 23

    replay_calls = _record_refits(gaussian_result, monkeypatch, fail_call=1)
    replay = simulated_confounding(
        gaussian_result,
        grid=_grid(),
        random_state=failed.failures[0].seed,
    )
    assert len(replay_calls) == 3
    assert replay.cells[1].failure == failed.cells[1].failure

    latent_seed, _ = _child_seeds(53)
    latent = np.random.default_rng(latent_seed).normal(size=gaussian_result.data.n)
    treatment = _flip_mask(latent, 0.5).astype(float)
    arm_loss_data = gaussian_result.data.with_treatment(treatment)
    arm_loss_result = replace(gaussian_result, data=arm_loss_data)
    _record_refits(arm_loss_result, monkeypatch)
    arm_loss = simulated_confounding(
        arm_loss_result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.5), outcome=(0.0,)),
        random_state=53,
    )
    assert arm_loss.failures[0].error_type == "DataError"
    assert "must keep every arm" in arm_loss.failures[0].message
    assert arm_loss.failures[0].seed == arm_loss.root_seed == 53

    replay_arm_loss = simulated_confounding(
        arm_loss_result,
        grid=arm_loss.grid,
        random_state=arm_loss.failures[0].seed,
    )
    assert replay_arm_loss.cells[1].failure == arm_loss.cells[1].failure


def test_numeric_calibration_matches_the_declared_formulas(
    gaussian_result: Any,
) -> None:
    surface = simulated_confounding(
        gaussian_result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("W1",),
        random_state=7,
    )
    rows = {row.role: row for row in surface.calibrations}
    index = gaussian_result.data.covariate_names.index("W1")
    design = StandardScaler().fit_transform(gaussian_result.data.covariates)
    expected_treatment = _binary_calibration(design, gaussian_result.data.treatment, index)
    expected_outcome = float(
        np.corrcoef(gaussian_result.data.covariates[:, index], gaussian_result.data.outcome)[0, 1]
        * np.std(gaussian_result.data.outcome)
    )
    assert rows["treatment"].strength == expected_treatment
    assert rows["outcome"].strength == pytest.approx(expected_outcome, rel=1e-12)
    assert "marginal" in rows["outcome"].method


def test_frames_follow_pandas_and_polars_backends(gaussian_result: Any) -> None:
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    pandas_surface = simulated_confounding(gaussian_result, grid=grid)
    assert isinstance(pandas_surface.to_frame(), pd.DataFrame)
    assert isinstance(pandas_surface.calibration_frame(), pd.DataFrame)

    pytest.importorskip("polars")
    polars_result = _fit(backend="polars", seed=8)
    polars_surface = simulated_confounding(polars_result, grid=grid)
    assert type(polars_surface.to_frame()).__module__.startswith("polars")
    assert type(polars_surface.calibration_frame()).__module__.startswith("polars")


def test_grid_validation_and_binomial_boundary(binomial_result: Any) -> None:
    with pytest.raises(ValueError, match="contain zero"):
        ConfounderStrengthGrid(treatment=(0.1,), outcome=(0.0,))
    with pytest.raises(ValueError, match=r"between 0 and 0\.5"):
        ConfounderStrengthGrid(treatment=(0.0, 0.6), outcome=(0.0,))
    with pytest.raises(ValueError, match="duplicates"):
        ConfounderStrengthGrid(treatment=(0.0, -0.0), outcome=(0.0,))
    with pytest.raises(ValueError, match="binomial outcome strengths"):
        simulated_confounding(
            binomial_result,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, 0.6)),
        )


@pytest.mark.parametrize("estimand", ["att", "atc", "ey1", "rr", "or", "ate_regime", "msm"])
def test_unsupported_estimands_are_refused_before_refit(
    gaussian_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    estimand: str,
) -> None:
    key = replace(gaussian_result.parameter_keys["ate"], estimand=estimand)
    altered = replace(
        gaussian_result,
        estimates={estimand: gaussian_result["ate"]},
        parameter_keys={estimand: replace(key, alias=estimand)},
    )
    calls = _record_refits(altered, monkeypatch)
    with pytest.raises(CapabilityError, match="only a marginal ATE"):
        simulated_confounding(altered, estimand=estimand, grid=_grid())
    assert calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("multi-arm", "binary treatment"),
        ("continuous", "binary treatment"),
        ("missing", "missing-outcome"),
        ("intermediate", "controlled-direct-effect"),
        ("weights", "observation-weighted"),
        ("cluster", "clustered"),
        ("repeats", "repeated cross-fitting"),
        ("restored", "replayable"),
        ("conditional", "marginal ATE"),
        ("regime", "marginal arm-indexed ATE"),
    ],
)
def test_unsupported_compositions_are_refused_before_refit(
    gaussian_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    message: str,
) -> None:
    result = gaussian_result
    data = result.data
    if change == "multi-arm":
        data = replace(data, treatment_levels=(0, 1, 2))
    elif change == "continuous":
        data = replace(data, treatment_kind="continuous", treatment_levels=())
    elif change == "missing":
        observed = data.observed.copy()
        observed[0] = False
        data = replace(data, observed=observed)
    elif change == "intermediate":
        data = replace(data, intermediate=np.zeros(data.n), intermediate_name="Z")
    elif change == "weights":
        data = replace(data, weights_name="weight")
    elif change == "cluster":
        data = replace(data, cluster=np.arange(data.n), cluster_name="id")
    elif change == "repeats":
        result = replace(result, repeats=result.repeats * 2)
    elif change == "restored":
        result = replace(result, estimator=None)
    elif change == "conditional":
        key = replace(result.parameter_keys["ate"], stratum=(0,))
        result = replace(result, parameter_keys={"ate": key})
    elif change == "regime":
        functional = replace(result.identified_effect.functional, axis="regime")
        identified = replace(result.identified_effect, functional=functional)
        result = replace(result, identified_effect=identified)
    result = replace(result, data=data)
    calls = [] if result.estimator is None else _record_refits(result, monkeypatch)
    with pytest.raises(CapabilityError, match=message):
        simulated_confounding(result, grid=_grid())
    assert calls == []


def test_categorical_calibration_is_refused_before_refit(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cleverly.data.causal_data import CategoricalEncoding

    data = replace(
        gaussian_result.data,
        encodings=(CategoricalEncoding("G", ("a", "b"), "a", ("G_b",)),),
    )
    result = replace(gaussian_result, data=data)
    calls = _record_refits(result, monkeypatch)
    with pytest.raises(CapabilityError, match="categorical covariate"):
        simulated_confounding(result, grid=_grid(), benchmark_covariates=("G",))
    assert calls == []


def test_non_result_and_ambiguous_alias_are_refused(gaussian_result: Any) -> None:
    with pytest.raises(CapabilityError, match="TMLEResult"):
        simulated_confounding(SimpleNamespace(), grid=_grid())
    with pytest.raises(ValueError, match="unavailable"):
        simulated_confounding(gaussian_result, estimand="unknown", grid=_grid())


def test_result_contract_has_no_inferential_or_verdict_fields(gaussian_result: Any) -> None:
    surface = simulated_confounding(
        gaussian_result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
    )
    assert isinstance(surface, SimulatedConfoundingResult)
    for prohibited in (
        "passed",
        "pvalue",
        "confidence_interval",
        "corrected_estimate",
        "bound",
        "robustness_value",
    ):
        assert not hasattr(surface, prohibited)
    assert "qualitative" in surface.summary()

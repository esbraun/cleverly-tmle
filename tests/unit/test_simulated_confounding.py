"""Simulated common-cause stress surfaces and their mutation controls."""

from __future__ import annotations

import importlib
from copy import copy
from dataclasses import replace
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from cleverly import (
    ATE,
    ATT,
    AssessmentStatus,
    CausalStudy,
    CollaborativeTMLEMethod,
    CounterfactualMean,
    DRTMLEMethod,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    OddsRatio,
    PointTreatment,
    RiskRatio,
)
from cleverly.datasets import make_binary_outcome, make_linear_ate, make_shift_dose
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Shift
from cleverly.sensitivity import (
    ConfounderStrengthGrid,
    SimulatedConfoundingResult,
    simulated_confounding,
)
from cleverly.sensitivity import simulated_confounding as public_function
from cleverly.sensitivity.simulated_confounding import (
    _binary_calibration,
    _continuous_calibration,
    _flip_binary,
    _flip_mask,
    _gaussian_outcome,
    _latent_child_seed,
    _linear_treatment,
    _perturb_treatment,
    _treatment_association,
)


def _fit(
    *,
    family: str = "gaussian",
    backend: str = "pandas",
    seed: int = 7,
    method: str = "tmle",
    repeats: int = 1,
) -> Any:
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
    configured_method: Any = method
    if method == "collaborative_tmle":
        configured_method = CollaborativeTMLEMethod(
            selection_folds=2,
            selection_inner_folds=2,
        )
    return study.identify(ATE()).estimate(
        method=configured_method,
        outcome_learner=outcome_learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        repeats=repeats,
        simultaneous=False,
    )


def _fit_binary_mean(
    *,
    treatment: float | None,
    method: str = "tmle",
    seed: int = 7,
    family: str = "gaussian",
) -> Any:
    if family == "binomial":
        frame, _ = make_binary_outcome(n=120, seed=seed)
        covariates = ("W1", "W2", "W3")
        outcome_learner: Any = LogisticRegression(max_iter=1000)
    else:
        frame, _ = make_linear_ate(n=120, seed=seed)
        covariates = ("W1", "W2", "W3", "W4")
        outcome_learner = LinearRegression()
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=covariates,
        ),
    )
    configured_method: Any = method
    if method == "collaborative_tmle":
        configured_method = CollaborativeTMLEMethod(
            selection_folds=2,
            selection_inner_folds=2,
            selection_estimand=(
                "ey" if treatment is None else "ey1" if treatment == 1.0 else "ey0"
            ),
        )
    return study.identify(CounterfactualMean(treatment=treatment)).estimate(
        method=configured_method,
        outcome_learner=outcome_learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        simultaneous=False,
    )


def _fit_ratio(
    *,
    target: str,
    method: str = "tmle",
    seed: int = 7,
    reference: float | None = None,
    repeats: int = 1,
) -> Any:
    frame, _ = make_binary_outcome(n=120, seed=seed)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
        ),
    )
    configured_method: Any = method
    if method == "collaborative_tmle":
        configured_method = CollaborativeTMLEMethod(
            selection_folds=2,
            selection_inner_folds=2,
            selection_estimand=target,
        )
    elif method == "drtmle":
        configured_method = DRTMLEMethod(
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
        )
    estimand = RiskRatio(reference=reference) if target == "rr" else OddsRatio(reference=reference)
    return study.identify(estimand).estimate(
        method=configured_method,
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        repeats=repeats,
        simultaneous=False,
    )


def _fit_att(*, seed: int = 7) -> Any:
    frame, _ = make_linear_ate(n=120, seed=seed)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
        ),
    )
    return study.identify(ATT()).estimate(
        outcome_learner=LinearRegression(),
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


@pytest.fixture(scope="module")
def binary_mean_result() -> Any:
    return _fit_binary_mean(treatment=1.0)


@pytest.fixture(scope="module")
def binary_means_result() -> Any:
    return _fit_binary_mean(treatment=None)


@pytest.fixture(scope="module")
def risk_ratio_result() -> Any:
    return _fit_ratio(target="rr")


@pytest.fixture(scope="module")
def odds_ratio_result() -> Any:
    return _fit_ratio(target="or")


@pytest.fixture(scope="module")
def att_result() -> Any:
    return _fit_att()


_TWO_POLICIES = (
    Shift(0.0, cap=3.0, name="natural course"),
    Shift(0.5, cap=3.0, name="up half"),
)
_THREE_POLICIES = (*_TWO_POLICIES, Shift(1.0, cap=3.0, name="up one"))


def _fit_continuous(
    *,
    family: str = "gaussian",
    seed: int = 7,
    policies: tuple[Shift, ...] = _TWO_POLICIES,
    means: bool = False,
    repeats: int = 1,
) -> Any:
    frame, _ = make_shift_dose(n=120, seed=seed)
    if family == "binomial":
        frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(float)
        outcome_learner: Any = LogisticRegression(max_iter=1000)
    else:
        outcome_learner = LinearRegression()
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
        ),
    )
    estimand: Any = (
        ModifiedTreatmentPolicy(shifts=policies)
        if means
        else ModifiedTreatmentPolicyEffect(shifts=policies)
    )
    return study.identify(estimand).estimate(
        method="tmle",
        outcome_learner=outcome_learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        repeats=repeats,
        simultaneous=False,
    )


@pytest.fixture(scope="module")
def continuous_gaussian_result() -> Any:
    return _fit_continuous()


@pytest.fixture(scope="module")
def continuous_binomial_result() -> Any:
    return _fit_continuous(family="binomial")


@pytest.fixture(scope="module")
def three_policy_result() -> Any:
    return _fit_continuous(policies=_THREE_POLICIES)


@pytest.fixture(scope="module")
def continuous_means_result() -> Any:
    return _fit_continuous(means=True)


@pytest.fixture(scope="module")
def continuous_binomial_means_result() -> Any:
    return _fit_continuous(family="binomial", means=True)


def _shift_alias(result: Any) -> str:
    aliases = [name for name in result.estimates if name.startswith("ate_shift[")]
    assert len(aliases) == 1
    return aliases[0]


def _mean_alias(result: Any, policy: str = "up half") -> str:
    alias = f"ey_shift[{policy}]"
    assert alias in result.estimates
    return alias


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


def _record_ratio_refits(
    result: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: str,
    fail_call: int | None = None,
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
            raise RuntimeError("deliberate ratio refit failure")
        treated = float(np.mean(data.outcome[data.treatment == 1.0]))
        control = float(np.mean(data.outcome[data.treatment == 0.0]))
        ratio = (
            treated / control
            if target == "rr"
            else (treated / (1.0 - treated)) / (control / (1.0 - control))
        )
        estimate = replace(result[target], psi=ratio, log_psi=float(np.log(ratio)))
        return replace(result, data=data, estimates={target: estimate})

    monkeypatch.setattr(result.estimator, "refit", refit)
    return calls


def _record_continuous_refits(
    result: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_call: int | None = None,
    alias: str | None = None,
) -> list[tuple[Any, int | None]]:
    calls: list[tuple[Any, int | None]] = []
    alias = _shift_alias(result) if alias is None else alias

    def refit(
        data: Any,
        *,
        intermediate_value: float | None = None,
        random_state: int | None = None,
    ) -> Any:
        del intermediate_value
        calls.append((data, random_state))
        if fail_call is not None and len(calls) == fail_call:
            raise RuntimeError("deliberate continuous refit failure")
        psi = float(np.mean(data.treatment * data.outcome))
        estimate = replace(result[alias], psi=psi)
        return replace(result, data=data, estimates={alias: estimate})

    monkeypatch.setattr(result.estimator, "refit", refit)
    return calls


def _grid() -> ConfounderStrengthGrid:
    return ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))


def _with_functional(result: Any, **changes: Any) -> Any:
    functional = replace(result.identified_effect.functional, **changes)
    identified = replace(result.identified_effect, functional=functional)
    return replace(result, identified_effect=identified)


class _UnsupportedTMLE(TMLE):
    """A near-miss estimator used to pin the exact supported-type boundary."""


@pytest.mark.parametrize(
    ("method", "estimator_name"),
    [
        ("tmle", "TMLE"),
        ("collaborative_tmle", "CTMLE"),
        ("drtmle", "DRTMLE"),
    ],
)
def test_each_supported_estimator_runs_a_real_refit_surface(
    method: str, estimator_name: str
) -> None:
    result = _fit(method=method, repeats=3)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,)),
        random_state=29,
    )

    assert type(result.estimator).__name__ == estimator_name
    assert not result.data.has_missing_outcome
    assert surface.complete
    assert len(surface.cells) == 2
    assert surface.n_repeats == result.n_repeats == 3
    assert surface.repeat_aggregation == "coordinatewise_median"
    assert "3 draw(s), aggregation=coordinatewise_median" in surface.summary()

    # The un-monkeypatched path. The anchor is the original fit itself, so it carries the
    # fitted point estimate rather than a refit of unchanged data.
    assert surface.original_estimate == result["ate"].psi
    assert surface.cells[0].estimate == result["ate"].psi
    assert surface.cells[0].displacement == 0.0

    # A flip of the upper 10% latent tail moves this fixture by about -0.61 through -0.64
    # on all three estimators. The gate sits far below that and far above numerical noise,
    # and it is signed, so a perturbation that never reaches the refit fails it.
    displacement = surface.cells[1].displacement
    assert displacement is not None
    assert displacement < -0.3
    assert surface.cells[1].estimate == pytest.approx(result["ate"].psi + displacement)
    assert surface.successful_cells == surface.cells


def _manual_repeated_refit(result: Any, surface: Any, *, treatment: float, outcome: float) -> Any:
    latent = np.random.default_rng(surface.latent_seed).normal(size=result.data.n)
    perturbed_treatment = _perturb_treatment(
        result.data.treatment,
        latent,
        treatment,
        surface.treatment_family,
    )
    replacement = result.data.with_treatment(perturbed_treatment)
    perturbed_outcome = (
        _gaussian_outcome(result.data.outcome, latent, outcome)
        if result.data.family == "gaussian"
        else _flip_binary(result.data.outcome, _flip_mask(latent, outcome))
    )
    replacement = replacement.with_outcome(
        perturbed_outcome,
        family=result.data.family,
        name="simulated-confounding outcome",
    )
    return result.estimator.refit(replacement, random_state=surface.root_seed)


def test_binary_additive_repeat_surface_equals_the_estimator_median() -> None:
    result = _fit(repeats=3)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))
    surface = simulated_confounding(result, grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.0)
    cell = surface.cells[1]
    per_draw = [repeat.psi["ate"] for repeat in manual.repeats]

    assert cell.estimate == manual["ate"].psi == float(np.median(per_draw))
    assert cell.displacement == manual["ate"].inference_value - result["ate"].inference_value
    assert cell.estimate != per_draw[0]
    second_manual = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.0)
    assert (
        manual.estimator.crossfit_plan(manual.data).seeds()
        == second_manual.estimator.crossfit_plan(second_manual.data).seeds()
    )
    assert any(
        not np.array_equal(first.folds.assignment, second.folds.assignment)
        for first, second in zip(manual.repeats, second_manual.repeats, strict=True)
    )


def test_binary_ratio_repeat_surface_equals_the_estimator_log_median() -> None:
    result = _fit_ratio(target="rr", repeats=3)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))
    surface = simulated_confounding(result, estimand="rr", grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.0)
    cell = surface.cells[1]
    per_draw = [repeat.psi["rr"] for repeat in manual.repeats]

    assert cell.estimate == manual["rr"].psi == float(np.median(per_draw))
    assert cell.displacement == manual["rr"].inference_value - result["rr"].inference_value
    assert cell.estimate != per_draw[0]


def test_continuous_policy_repeat_surface_equals_the_estimator_median() -> None:
    result = _fit_continuous(repeats=3)
    alias = _shift_alias(result)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
    surface = simulated_confounding(result, estimand=alias, grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.0)
    cell = surface.cells[1]
    per_draw = [repeat.psi[alias] for repeat in manual.repeats]

    assert cell.estimate == manual[alias].psi == float(np.median(per_draw))
    assert cell.displacement == manual[alias].inference_value - result[alias].inference_value
    assert cell.estimate != per_draw[0]


def test_repeated_surface_metadata_cache_and_serialization_round_trip() -> None:
    result = _fit(repeats=3)
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,)),
        "random_state": 31,
    }
    surface = result.sensitivity.simulated_confounding(**kwargs)

    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    assert surface.n_repeats == 3
    assert surface.repeat_aggregation == "coordinatewise_median"

    restored = loads(dumps(result))
    replayed = restored.sensitivity.simulated_confounding(**kwargs)
    assert replayed == surface
    assert replayed.n_repeats == 3
    assert replayed.repeat_aggregation == "coordinatewise_median"


def test_repeated_surface_retains_a_complete_refit_failure(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    repeated = replace(gaussian_result, repeats=gaussian_result.repeats * 3)
    calls = _record_refits(repeated, monkeypatch, fail_call=1)
    surface = simulated_confounding(
        repeated,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,)),
        random_state=31,
    )

    assert len(calls) == 1
    assert surface.n_repeats == 3
    assert surface.cells[1].failure is not None
    assert surface.cells[1].failure.error_type == "RuntimeError"


@pytest.mark.parametrize(
    ("method", "estimator_name"),
    [("tmle", "TMLE"), ("collaborative_tmle", "CTMLE"), ("drtmle", "DRTMLE")],
)
@pytest.mark.parametrize("target", ["rr", "or"])
def test_each_ratio_runs_a_real_log_movement_surface(
    method: str,
    estimator_name: str,
    target: str,
) -> None:
    """Both ratios keep raw estimates while all three engines move on the log scale."""
    result = _fit_ratio(target=target, method=method)
    surface = simulated_confounding(
        result,
        estimand=target,
        grid=_grid(),
        random_state=7,
    )

    assert type(result.estimator).__name__ == estimator_name
    assert surface.complete
    assert surface.estimand == target
    assert surface.original_estimate == result[target].psi
    assert surface.movement_scale == "log_ratio"
    assert set(surface.to_frame()["movement_scale"]) == {"log_ratio"}
    assert "movement (log ratio)" in surface.summary()
    assert surface.cells[0].estimate == result[target].psi
    assert surface.cells[0].displacement == 0.0
    assert any(abs(cell.displacement or 0.0) > 0.01 for cell in surface.cells[1:])

    witness = surface.cells[1]
    assert witness.estimate is not None
    assert witness.displacement == pytest.approx(
        np.log(witness.estimate) - result[target].inference_value
    )
    assert witness.displacement != pytest.approx(witness.estimate - result[target].psi, abs=1e-3)


@pytest.mark.parametrize(
    ("target", "fixture_name"),
    [("rr", "risk_ratio_result"), ("or", "odds_ratio_result")],
)
def test_reversing_the_ratio_contrast_inverts_estimates_and_log_movements(
    request: pytest.FixtureRequest,
    target: str,
    fixture_name: str,
) -> None:
    """A direction mutation must invert the ratio and negate each log displacement."""
    forward = request.getfixturevalue(fixture_name)
    reverse = _fit_ratio(target=target, reference=1.0)
    forward_surface = simulated_confounding(
        forward,
        estimand=target,
        grid=_grid(),
        random_state=7,
    )
    reverse_surface = simulated_confounding(
        reverse,
        estimand=target,
        grid=_grid(),
        random_state=7,
    )

    assert reverse.parameter_keys[target].value == forward.parameter_keys[target].reference
    assert reverse.parameter_keys[target].reference == forward.parameter_keys[target].value
    assert reverse[target].psi == pytest.approx(1.0 / forward[target].psi)
    for forward_cell, reverse_cell in zip(
        forward_surface.cells,
        reverse_surface.cells,
        strict=True,
    ):
        assert forward_cell.estimate is not None
        assert reverse_cell.estimate == pytest.approx(1.0 / forward_cell.estimate)
        assert reverse_cell.displacement == pytest.approx(-forward_cell.displacement)


def test_ratio_facade_selects_its_sole_parameter_and_preserves_the_cache(
    risk_ratio_result: Any,
) -> None:
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    capability = risk_ratio_result.sensitivity.capability("simulated_confounding")

    surface = risk_ratio_result.sensitivity.simulated_confounding(grid=grid)

    assert capability.requires_arguments == ("grid",)
    assert surface.estimand == "rr"
    assert surface.movement_scale == "log_ratio"
    assert risk_ratio_result.sensitivity.simulated_confounding(grid=grid) is surface


def test_ratio_cells_reuse_original_data_common_randomness_and_retain_failures(
    risk_ratio_result: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_ratio_refits(
        risk_ratio_result,
        monkeypatch,
        target="rr",
        fail_call=2,
    )
    surface = simulated_confounding(
        risk_ratio_result,
        estimand="rr",
        grid=_grid(),
        random_state=19,
    )

    assert len(calls) == 3
    assert all(seed == surface.root_seed for _, seed in calls)
    assert not surface.complete
    assert surface.cells[2].failure is not None
    assert surface.cells[2].failure.error_type == "RuntimeError"
    assert surface.cells[2].failure.message == "deliberate ratio refit failure"
    latent = np.random.default_rng(surface.latent_seed).normal(size=risk_ratio_result.data.n)
    expected_outcome = _flip_binary(
        risk_ratio_result.data.outcome,
        _flip_mask(latent, 0.2),
    )
    expected_treatment = _flip_binary(
        risk_ratio_result.data.treatment,
        _flip_mask(latent, 0.1),
    )
    assert np.array_equal(calls[0][0].treatment, risk_ratio_result.data.treatment)
    assert np.array_equal(calls[0][0].outcome, expected_outcome)
    assert np.array_equal(calls[1][0].treatment, expected_treatment)
    assert np.array_equal(calls[1][0].outcome, risk_ratio_result.data.outcome)
    assert np.array_equal(calls[2][0].treatment, expected_treatment)
    assert np.array_equal(calls[2][0].outcome, expected_outcome)


@pytest.mark.parametrize(
    ("method", "expected_engine"),
    [("tmle", "TMLE"), ("collaborative_tmle", "CTMLE"), ("drtmle", "DRTMLE")],
)
@pytest.mark.parametrize(
    ("treatment", "expected_alias"),
    [(1.0, "ey1"), (0.0, "ey0"), (None, None)],
)
def test_each_binary_mean_alias_runs_the_supported_real_refit(
    method: str,
    expected_engine: str,
    treatment: float | None,
    expected_alias: str | None,
) -> None:
    """Exercise compatibility aliases and an explicit ``ey[...]`` on every engine."""
    result = _fit_binary_mean(treatment=treatment, method=method)
    if expected_alias is None:
        alias = next(name for name, key in result.parameter_keys.items() if key.value == 1.0)
        assert alias.startswith("ey[")
    else:
        alias = expected_alias
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0, 0.2)),
        random_state=7,
    )

    assert type(result.estimator).__name__ == expected_engine
    assert surface.complete
    assert surface.estimand == alias
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.cells[0].displacement == 0.0
    assert any(abs(cell.displacement or 0.0) > 1e-5 for cell in surface.cells[1:])


def test_a_binomial_binary_mean_runs_a_nonzero_real_refit() -> None:
    result = _fit_binary_mean(treatment=1.0, family="binomial")
    surface = simulated_confounding(
        result,
        estimand="ey1",
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0, 0.2)),
        random_state=7,
    )

    assert surface.complete
    assert surface.outcome_family == "binomial"
    assert surface.cells[0].estimate == result["ey1"].psi
    assert any(abs(cell.displacement or 0.0) > 1e-5 for cell in surface.cells[1:])


def test_explicit_binary_means_keep_each_fixed_arm_and_move_nontrivially(
    binary_means_result: Any,
) -> None:
    """The same latent draw gives two distinct, nonzero fixed-arm witnesses."""
    result = binary_means_result
    aliases = {
        key.value: name for name, key in result.parameter_keys.items() if key.estimand == "ey"
    }
    assert set(aliases) == {0, 1}
    surfaces = {
        arm: simulated_confounding(
            result,
            estimand=alias,
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.25)),
            random_state=7,
        )
        for arm, alias in aliases.items()
    }

    for arm, surface in surfaces.items():
        alias = aliases[arm]
        assert surface.estimand == alias
        assert surface.cells[0].estimate == result[alias].psi
        assert surface.cells[0].displacement == 0.0
        assert any(abs(cell.displacement or 0.0) > 1e-5 for cell in surface.cells[1:])
        assert result.parameter_keys[alias].value == arm
    assert surfaces[0].cells[-1].estimate != pytest.approx(surfaces[1].cells[-1].estimate)


def test_zero_anchor_common_randomness_and_original_data_per_cell(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = gaussian_result
    original_treatment = result.data.treatment.copy()
    original_outcome = result.data.outcome.copy()
    calls = _record_refits(result, monkeypatch)

    surface = simulated_confounding(result, grid=_grid(), random_state=19)

    assert surface.original_estimate == result["ate"].psi
    assert surface.cells[0].estimate == result["ate"].psi
    assert surface.cells[0].displacement == 0.0
    assert len(calls) == 3
    assert {seed for _, seed in calls} == {surface.refit_seed}
    latent = np.random.default_rng(surface.latent_seed).normal(size=result.data.n)
    expected_treatment = _flip_binary(original_treatment, _flip_mask(latent, 0.1))
    expected_outcome = _gaussian_outcome(original_outcome, latent, 0.2)

    # Non-vacuity. The expectations above are built from the same helpers the surface
    # calls, so a helper that returned its input unchanged would satisfy them all.
    assert not np.array_equal(expected_treatment, original_treatment)
    assert not np.array_equal(expected_outcome, original_outcome)
    assert int(_flip_mask(latent, 0.1).sum()) == 13

    assert np.array_equal(calls[0][0].treatment, original_treatment)
    assert np.array_equal(calls[0][0].outcome, expected_outcome)
    assert np.array_equal(calls[1][0].treatment, expected_treatment)
    assert np.array_equal(calls[1][0].outcome, original_outcome)
    assert np.array_equal(calls[2][0].treatment, expected_treatment)
    assert np.array_equal(calls[2][0].outcome, expected_outcome)
    assert np.array_equal(result.data.treatment, original_treatment)
    assert np.array_equal(result.data.outcome, original_outcome)

    # The spy's psi is a plain arm-mean contrast on the data it receives, so every cell's
    # estimate and displacement are computable here without calling the surface again.
    # Displacement is signed and asymmetric across the three cells, so an inverted
    # subtraction, a dropped ``original_estimate``, or a dropped estimate fails.
    for index, (data, _) in enumerate(calls, start=1):
        treated = data.outcome[data.treatment == 1.0]
        control = data.outcome[data.treatment == 0.0]
        expected_psi = float(np.mean(treated) - np.mean(control))
        cell = surface.cells[index]
        assert cell.estimate == pytest.approx(expected_psi)
        assert cell.displacement == pytest.approx(expected_psi - result["ate"].psi)
    # The outcome-only cell moves by about +0.213 on this fixture. The witness is signed
    # and its magnitude is far above numerical noise, so ``original - estimate`` fails it.
    assert surface.cells[1].displacement > 0.1
    assert len({cell.displacement for cell in surface.cells}) == 4


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


def test_continuous_treatment_sign_and_dispatch_are_active_mutation_controls() -> None:
    treatment = np.array([-1.0, 0.0, 2.0, 4.0])
    latent = np.array([-2.0, -0.25, 0.5, 3.0])
    shifted = _linear_treatment(treatment, latent, 0.4)

    assert np.array_equal(shifted, treatment + 0.4 * latent)
    assert not np.array_equal(shifted, treatment - 0.4 * latent)
    assert np.array_equal(_perturb_treatment(treatment, latent, 0.4, "continuous"), shifted)

    binary = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.array_equal(
        _perturb_treatment(binary, latent, 0.25, "binary"),
        _flip_binary(binary, _flip_mask(latent, 0.25)),
    )


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
def test_continuous_surface_runs_a_real_ordinary_tmle_refit(
    family: str,
    continuous_gaussian_result: Any,
    continuous_binomial_result: Any,
) -> None:
    result = continuous_gaussian_result if family == "gaussian" else continuous_binomial_result
    alias = _shift_alias(result)
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.2)),
        random_state=29,
    )

    assert type(result.estimator) is TMLE
    assert surface.complete
    assert surface.estimand == alias
    assert surface.treatment_family == "continuous"
    assert surface.outcome_family == family
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.cells[0].displacement == 0.0
    assert any(abs(cell.displacement or 0.0) > 1e-4 for cell in surface.cells[1:])
    assert "adds signed strength" in surface.treatment_law
    summary = surface.summary()
    assert "what the continuous linear perturbation achieved" in summary
    # The reading guard. A zero outcome strength leaves the latent vector out of the
    # outcome, so that column carries no confounding path whatever its association.
    assert "no confounding path" in summary
    assert "dose perturbation alone" in summary
    # The symmetric half. A zero treatment strength leaves the latent vector out of the
    # treatment, so that column reports the outcome law alone.
    assert "A cell whose treatment strength is zero also has no confounding path" in summary
    if family == "gaussian":
        assert "a contrast removes most of it" in summary
    else:
        assert "attenuates the fitted outcome regression" in summary


def test_continuous_real_refit_moves_with_the_signed_dose_and_outcome_laws(
    continuous_gaussian_result: Any,
) -> None:
    """Pin the direction of both axes on the real estimator rather than on a spy.

    Every other signed continuous check replaces ``TMLE.refit`` with a spy, so the sign
    and the axis assignment of the real path rest on this test alone. The gates below are
    signed, and their margins sit far above numerical noise, so a swap of the two axes, a
    reversed ``A' = A + k_A U`` sign, or a reversed ``Y' = Y - k_Y U`` sign fails one of
    them. ``random_state`` equals the seed of the fit, so the anchor and every cell share
    the cross-fitting folds and no movement here is a fold artifact.
    """
    result = continuous_gaussian_result
    alias = _shift_alias(result)
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 1.0), outcome=(0.0, 0.5)),
        random_state=7,
    )
    anchor, outcome_only, treatment_only, joint = surface.cells

    assert surface.complete
    assert (anchor.treatment_strength, anchor.outcome_strength) == (0.0, 0.0)
    assert (outcome_only.treatment_strength, outcome_only.outcome_strength) == (0.0, 0.5)
    assert (treatment_only.treatment_strength, treatment_only.outcome_strength) == (1.0, 0.0)
    assert (joint.treatment_strength, joint.outcome_strength) == (1.0, 0.5)

    # ``A' = A + k_A U`` raises the latent-treatment correlation of this fixture from
    # about +0.06 at the anchor to about +0.70 at ``k_A = 1``. A reversed sign gives about
    # -0.70, and an untouched treatment stays at the anchor level.
    assert anchor.induced_treatment_association is not None
    assert abs(anchor.induced_treatment_association) < 0.2
    assert treatment_only.induced_treatment_association is not None
    assert treatment_only.induced_treatment_association > 0.4
    assert joint.induced_treatment_association == treatment_only.induced_treatment_association

    # The dose axis alone moves this fit by about -0.26 and the outcome axis alone by
    # about -0.012, so a surface that read the two strengths in the wrong order fails
    # both of these gates.
    assert treatment_only.displacement is not None
    assert treatment_only.displacement < -0.1
    assert outcome_only.displacement is not None
    assert abs(outcome_only.displacement) < 0.05

    # ``Y' = Y - k_Y U`` adds about -0.081 on top of the dose movement. A reversed outcome
    # sign carries the joint cell back toward the anchor instead.
    assert joint.displacement is not None
    assert joint.displacement < treatment_only.displacement - 0.02


def test_continuous_zero_anchor_common_randomness_and_original_data_per_cell(
    continuous_gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = continuous_gaussian_result
    alias = _shift_alias(result)
    original_treatment = result.data.treatment.copy()
    original_outcome = result.data.outcome.copy()
    calls = _record_continuous_refits(result, monkeypatch)
    grid = ConfounderStrengthGrid(
        treatment=(0.0, -0.3),
        outcome=(0.0, 0.4),
    )

    surface = simulated_confounding(result, estimand=alias, grid=grid, random_state=19)

    assert surface.cells[0].estimate == result[alias].psi
    assert surface.cells[0].displacement == 0.0
    assert len(calls) == 3
    assert {seed for _, seed in calls} == {surface.root_seed}
    latent = np.random.default_rng(surface.latent_seed).normal(size=result.data.n)
    expected_treatment = original_treatment - 0.3 * latent
    expected_outcome = original_outcome - 0.4 * latent

    assert not np.array_equal(expected_treatment, original_treatment)
    assert not np.array_equal(expected_outcome, original_outcome)
    assert np.array_equal(calls[0][0].treatment, original_treatment)
    assert np.array_equal(calls[0][0].outcome, expected_outcome)
    assert np.array_equal(calls[1][0].treatment, expected_treatment)
    assert np.array_equal(calls[1][0].outcome, original_outcome)
    assert np.array_equal(calls[2][0].treatment, expected_treatment)
    assert np.array_equal(calls[2][0].outcome, expected_outcome)
    assert np.array_equal(result.data.treatment, original_treatment)
    assert np.array_equal(result.data.outcome, original_outcome)

    for index, (data, _) in enumerate(calls, start=1):
        expected = float(np.mean(data.treatment * data.outcome))
        assert surface.cells[index].estimate == pytest.approx(expected)
        assert surface.cells[index].displacement == pytest.approx(expected - result[alias].psi)

    # Each sign and the joint path have a distinct nonzero witness. A no-op treatment,
    # reversed outcome sign, or cumulative mutation across cells breaks these values.
    assert surface.cells[1].displacement is not None
    assert surface.cells[2].displacement is not None
    assert surface.cells[3].displacement is not None
    assert len({round(cell.displacement or 0.0, 8) for cell in surface.cells}) == 4

    expected_association = float(np.corrcoef(latent, expected_treatment)[0, 1])
    assert surface.cells[2].induced_treatment_association == pytest.approx(
        expected_association, rel=1e-12
    )
    assert surface.cells[3].induced_treatment_association == pytest.approx(
        expected_association, rel=1e-12
    )


def test_continuous_binomial_outcome_uses_the_existing_tail_mask(
    continuous_binomial_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = continuous_binomial_result
    alias = _shift_alias(result)
    calls = _record_continuous_refits(result, monkeypatch)
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.3)),
        random_state=31,
    )
    latent = np.random.default_rng(surface.latent_seed).normal(size=result.data.n)
    expected_treatment = result.data.treatment + 0.2 * latent
    expected_outcome = _flip_binary(result.data.outcome, _flip_mask(latent, 0.3))

    assert np.array_equal(calls[0][0].outcome, expected_outcome)
    assert np.array_equal(calls[1][0].treatment, expected_treatment)
    assert np.array_equal(calls[2][0].treatment, expected_treatment)
    assert np.array_equal(calls[2][0].outcome, expected_outcome)
    assert int(_flip_mask(latent, 0.3).sum()) > 0


def test_continuous_real_refit_recomputes_the_active_cap_on_perturbed_dose(
    continuous_gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = continuous_gaussian_result
    alias = _shift_alias(result)
    policies = tuple(result.identified_effect.functional.interventions)
    original_refit = result.estimator.refit
    original_shifts = result.nuisance.shifts
    assert original_shifts is not None
    assert int(original_shifts.capped[:, 1].sum()) > 0
    witnessed: list[Any] = []

    def checked_refit(
        data: Any,
        *,
        intermediate_value: float | None = None,
        random_state: int | None = None,
    ) -> Any:
        refitted = original_refit(
            data,
            intermediate_value=intermediate_value,
            random_state=random_state,
        )
        shifts = refitted.nuisance.shifts
        assert shifts is not None
        expected_shifted = np.column_stack([policy.apply(data.treatment)[0] for policy in policies])
        expected_capped = np.column_stack([policy.apply(data.treatment)[1] for policy in policies])
        assert np.array_equal(shifts.shifted, expected_shifted)
        assert np.array_equal(shifts.capped, expected_capped)
        assert int(expected_capped[:, 1].sum()) > 0
        witnessed.append(data)
        return refitted

    monkeypatch.setattr(result.estimator, "refit", checked_refit)
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 1.0), outcome=(0.0,)),
        random_state=29,
    )

    assert surface.complete
    assert len(witnessed) == 1
    assert not np.array_equal(witnessed[0].treatment, result.data.treatment)
    assert not np.array_equal(
        policies[1].apply(witnessed[0].treatment)[1], original_shifts.capped[:, 1]
    )


def test_continuous_calibration_uses_the_signed_standardized_coefficient(
    continuous_gaussian_result: Any,
) -> None:
    result = continuous_gaussian_result
    alias = _shift_alias(result)
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("W1",),
    )
    rows = {row.role: row for row in surface.calibrations}
    index = result.data.covariate_names.index("W1")

    assert rows["treatment"].family == "gaussian"
    assert rows["treatment"].method == "signed standardized marginal coefficient"
    assert rows["treatment"].strength == pytest.approx(
        _continuous_calibration(result.data.covariates[:, index], result.data.treatment),
        rel=1e-12,
    )
    assert rows["outcome"].strength == pytest.approx(
        _continuous_calibration(result.data.covariates[:, index], result.data.outcome),
        rel=1e-12,
    )


def test_tiny_positive_flip_strength_uses_the_stable_upper_tail() -> None:
    """Bracket the tail cut so a wrong but positive threshold fails.

    ``-NormalDist().inv_cdf(1e-20)`` is 9.26234. Latent values on both sides of that
    number pin the cut. A single value of 10 passes for any threshold in ``(0, 10]``.
    """
    expected = -NormalDist().inv_cdf(1e-20)
    assert expected == pytest.approx(9.262340089798405)
    latent = np.array([0.0, 9.0, 9.5, 10.0])
    assert np.array_equal(_flip_mask(latent, 1e-20), np.array([False, False, True, True]))
    assert np.array_equal(_flip_mask(latent, 0.0), np.zeros(4, dtype=bool))


def test_a_cell_that_flips_no_row_reproduces_the_anchor_exactly(
    gaussian_result: Any,
) -> None:
    """A byte-identical cell must not move, so no displacement is a fold artifact.

    The zero-strength anchor is the original fit. Every other cell refits. A refit
    under a seed other than the seed of the original fit redraws the cross-fitting
    folds, which moves the estimate on unchanged data. This control makes that
    movement visible: at strength 1e-8 the flip mask selects no row.
    """
    grid = ConfounderStrengthGrid(treatment=(0.0, 1e-8), outcome=(0.0,))
    surface = simulated_confounding(gaussian_result, grid=grid, random_state=7)
    latent = np.random.default_rng(surface.latent_seed).normal(size=gaussian_result.data.n)

    assert gaussian_result.estimator.random_state == 7
    assert surface.root_seed == 7
    assert int(_flip_mask(latent, 1e-8).sum()) == 0
    assert surface.complete
    assert surface.cells[1].estimate == gaussian_result["ate"].psi
    assert surface.cells[1].displacement == 0.0
    assert surface.refit_seed == surface.root_seed


def test_latent_seed_is_tagged_and_aliases_no_other_child_stream(
    gaussian_result: Any,
) -> None:
    from cleverly.validation.refute import _generated_child_seeds

    for root in (0, 7, 19, 41, 2**31 - 1):
        latent = _latent_child_seed(root)
        bootstrap = tuple(
            int(child.generate_state(1)[0]) for child in np.random.SeedSequence(root).spawn(2)
        )
        assert latent != root
        assert latent not in bootstrap
        assert latent != int(np.random.SeedSequence(root).generate_state(1)[0])
        assert latent not in _generated_child_seeds(root, "dummy_outcome", 4)
        assert latent not in _generated_child_seeds(root, "simulated_outcome", 4)

    surface = simulated_confounding(
        gaussian_result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        random_state=19,
    )
    assert surface.root_seed == 19
    assert surface.latent_seed == _latent_child_seed(19)
    assert surface.latent_seed != surface.root_seed
    assert surface.refit_seed == surface.root_seed


def test_real_refit_surface_repeats_every_estimate_under_one_seed(
    gaussian_result: Any,
) -> None:
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.5))
    first = simulated_confounding(gaussian_result, grid=grid, random_state=7)
    second = simulated_confounding(gaussian_result, grid=grid, random_state=7)

    assert first.complete and second.complete
    assert [cell.estimate for cell in first.cells] == [cell.estimate for cell in second.cells]
    assert any(cell.displacement != 0.0 for cell in first.cells[1:])


def test_binary_treatment_and_outcome_share_the_exact_latent_masks(
    binomial_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_refits(binomial_result, monkeypatch)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0, 0.3))
    surface = simulated_confounding(binomial_result, grid=grid, random_state=31)
    latent = np.random.default_rng(surface.latent_seed).normal(size=binomial_result.data.n)
    expected_treatment = _flip_binary(binomial_result.data.treatment, _flip_mask(latent, 0.15))
    expected_outcome = _flip_binary(binomial_result.data.outcome, _flip_mask(latent, 0.3))

    # Non-vacuity. Both expectations come from the helpers under test, so a no-op mask or
    # a no-op flip would satisfy them against unchanged data.
    assert int(_flip_mask(latent, 0.15).sum()) > 0
    assert int(_flip_mask(latent, 0.3).sum()) > int(_flip_mask(latent, 0.15).sum())
    assert not np.array_equal(expected_treatment, binomial_result.data.treatment)
    assert not np.array_equal(expected_outcome, binomial_result.data.outcome)

    assert np.array_equal(calls[1][0].treatment, expected_treatment)
    assert np.array_equal(calls[0][0].outcome, expected_outcome)
    assert np.array_equal(calls[2][0].treatment, calls[1][0].treatment)
    assert np.array_equal(calls[2][0].outcome, calls[0][0].outcome)
    assert surface.outcome_family == "binomial"
    assert "flipped" in surface.outcome_law


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

    # An unseeded fit draws a root seed. The recorded seed is only useful if replaying it
    # reproduces the draw, and only honest if two unseeded calls do not share one seed.
    anchor = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    unseeded = replace(gaussian_result, estimator=replace_estimator_seed(gaussian_result, None))
    _record_refits(unseeded, monkeypatch)
    recorded = simulated_confounding(unseeded, grid=anchor)
    second = simulated_confounding(unseeded, grid=anchor)
    replayed = simulated_confounding(unseeded, grid=anchor, random_state=recorded.root_seed)

    assert isinstance(recorded.root_seed, int)
    assert recorded.root_seed >= 0
    assert replayed.root_seed == recorded.root_seed
    assert replayed.latent_seed == recorded.latent_seed == _latent_child_seed(recorded.root_seed)
    assert second.root_seed != recorded.root_seed
    assert second.latent_seed != recorded.latent_seed


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
    # The surface built this cell's treatment before the refit raised, so the cell keeps
    # its association. Cell 1 carries treatment strength zero, so it reports the baseline.
    failed_latent = np.random.default_rng(failed.latent_seed).normal(size=gaussian_result.data.n)
    assert failed.cells[1].induced_treatment_association == pytest.approx(
        _treatment_association(failed_latent, gaussian_result.data.treatment)
    )
    assert failed.cells[1].induced_treatment_association is not None

    replay_calls = _record_refits(gaussian_result, monkeypatch, fail_call=1)
    replay = simulated_confounding(
        gaussian_result,
        grid=_grid(),
        random_state=failed.failures[0].seed,
    )
    assert len(replay_calls) == 3
    assert replay.cells[1].failure == failed.cells[1].failure

    latent_seed = _latent_child_seed(53)
    latent = np.random.default_rng(latent_seed).normal(size=gaussian_result.data.n)
    treatment = _flip_mask(latent, 0.5).astype(float)
    arm_loss_data = gaussian_result.data.with_treatment(treatment)
    arm_loss_result = replace(gaussian_result, data=arm_loss_data)
    arm_loss_calls = _record_refits(arm_loss_result, monkeypatch)
    arm_loss = simulated_confounding(
        arm_loss_result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.5), outcome=(0.0,)),
        random_state=53,
    )
    # ``with_treatment`` rejects the single-arm replacement before any refit runs, so the
    # only cell that could call the estimator never reaches it.
    assert arm_loss_calls == []
    assert arm_loss.failures[0].error_type == "DataError"
    assert "must keep every arm" in arm_loss.failures[0].message
    assert arm_loss.failures[0].seed == arm_loss.root_seed == 53
    # The arm-loss cell holds a constant treatment, whose correlation is undefined, so the
    # zero-variance guard reports ``None``. The anchor of the same surface reports the
    # strong association this constructed treatment has with the latent vector.
    assert float(np.std(_flip_binary(treatment, _flip_mask(latent, 0.5)))) == 0.0
    assert arm_loss.cells[1].induced_treatment_association is None
    assert (arm_loss.cells[0].induced_treatment_association or 0.0) > 0.5

    replay_arm_loss = simulated_confounding(
        arm_loss_result,
        grid=arm_loss.grid,
        random_state=arm_loss.failures[0].seed,
    )
    assert replay_arm_loss.cells[1].failure == arm_loss.cells[1].failure


def _prediction_change_fraction(design: Any, target: Any, index: int) -> float:
    """Restate the class-prediction-change rule without the function under test.

    Fit a logistic model on the scaled design, zero one column, and report the fraction
    of rows whose predicted class changes.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(design, target)
    baseline = model.predict(design)
    removed = np.array(design, copy=True)
    removed[:, index] = 0.0
    return float(np.mean(model.predict(removed) != baseline))


@pytest.mark.parametrize(("covariate", "strength"), [("W1", 0.21666667), ("W2", 0.26666667)])
def test_numeric_calibration_matches_the_declared_formulas(
    gaussian_result: Any, covariate: str, strength: float
) -> None:
    surface = simulated_confounding(
        gaussian_result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=(covariate,),
        random_state=7,
    )
    rows = {row.role: row for row in surface.calibrations}
    index = gaussian_result.data.covariate_names.index(covariate)
    design = StandardScaler().fit_transform(gaussian_result.data.covariates)
    expected_treatment = _prediction_change_fraction(design, gaussian_result.data.treatment, index)
    expected_outcome = float(
        np.corrcoef(gaussian_result.data.covariates[:, index], gaussian_result.data.outcome)[0, 1]
        * np.std(gaussian_result.data.outcome)
    )

    # Non-vacuity. A calibration that always reported zero would satisfy an equality
    # against a formula it also defined.
    assert rows["treatment"].strength > 0.0
    assert rows["treatment"].strength == pytest.approx(strength)
    assert rows["treatment"].strength == pytest.approx(expected_treatment, rel=1e-12)
    assert rows["treatment"].strength == pytest.approx(
        _binary_calibration(design, gaussian_result.data.treatment, index), rel=1e-12
    )
    assert rows["treatment"].family == "binomial"
    assert rows["treatment"].method == "logistic class-prediction change fraction"
    assert rows["outcome"].strength == pytest.approx(expected_outcome, rel=1e-12)
    assert abs(rows["outcome"].strength) > 0.0
    assert rows["outcome"].family == "gaussian"
    assert rows["outcome"].method == "signed standardized marginal coefficient"
    assert rows["treatment"].covariate == rows["outcome"].covariate == covariate


def test_binomial_outcome_calibration_uses_the_class_prediction_rule(
    binomial_result: Any,
) -> None:
    """A binomial outcome calibrates by prediction change, not by a marginal coefficient.

    The Gaussian branch reports a signed standardized coefficient. Only a binomial-family
    fit reaches the second logistic call, so no Gaussian fixture covers it.
    """
    surface = simulated_confounding(
        binomial_result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("W1",),
        random_state=7,
    )
    rows = {row.role: row for row in surface.calibrations}
    index = binomial_result.data.covariate_names.index("W1")
    design = StandardScaler().fit_transform(binomial_result.data.covariates)
    expected_outcome = _prediction_change_fraction(design, binomial_result.data.outcome, index)

    assert surface.outcome_family == "binomial"
    assert rows["outcome"].family == "binomial"
    assert rows["outcome"].method == "logistic class-prediction change fraction"
    assert rows["outcome"].method != "signed standardized marginal coefficient"
    assert rows["outcome"].strength == pytest.approx(expected_outcome, rel=1e-12)
    assert rows["outcome"].strength == pytest.approx(0.15)
    assert 0.0 < rows["outcome"].strength <= 1.0
    assert rows["treatment"].strength == pytest.approx(
        _prediction_change_fraction(design, binomial_result.data.treatment, index), rel=1e-12
    )
    assert rows["treatment"].strength > 0.0


def test_pandas_frames_carry_estimates_displacements_and_failure_detail(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Populate every column before checking the frames.

    An all-zero grid gives one anchor cell with no failure, and no benchmark covariate
    gives an empty calibration frame. Both frames then agree with a build that dropped
    the estimate, the displacement, or the retained failure.
    """
    calls = _record_refits(gaussian_result, monkeypatch, fail_call=2)
    surface = simulated_confounding(
        gaussian_result,
        grid=_grid(),
        benchmark_covariates=("W1", "W2"),
        random_state=19,
    )
    frame = surface.to_frame()

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "treatment_strength",
        "outcome_strength",
        "movement_scale",
        "estimate",
        "displacement",
        "induced_treatment_association",
        "error_type",
        "message",
    ]
    assert len(frame) == 4
    assert list(frame["treatment_strength"]) == [0.0, 0.0, 0.1, 0.1]
    assert list(frame["outcome_strength"]) == [0.0, 0.2, 0.0, 0.2]
    assert list(frame["movement_scale"]) == ["estimate_difference"] * 4

    # Row 0 is the anchor and row 2 is the deliberate failure. Row 1 refit successfully,
    # so its estimate and displacement must carry real numbers.
    assert frame["estimate"][0] == gaussian_result["ate"].psi
    assert frame["estimate"][1] == pytest.approx(surface.cells[1].estimate)
    assert frame["displacement"][1] == pytest.approx(
        frame["estimate"][1] - gaussian_result["ate"].psi
    )
    assert frame["displacement"][1] > 0.1
    assert pd.isna(frame["error_type"][1])
    assert pd.isna(frame["message"][1])

    assert len(calls) == 3
    assert pd.isna(frame["estimate"][2])
    assert pd.isna(frame["displacement"][2])
    assert frame["error_type"][2] == "RuntimeError"
    assert "deliberate refit failure" in frame["message"][2]
    assert [value if isinstance(value, str) else None for value in frame["error_type"]] == [
        None,
        None,
        "RuntimeError",
        None,
    ]

    calibrations = surface.calibration_frame()
    assert isinstance(calibrations, pd.DataFrame)
    assert list(calibrations.columns) == [
        "covariate",
        "role",
        "family",
        "strength",
        "method",
    ]
    assert list(calibrations["covariate"]) == ["W1", "W1", "W2", "W2"]
    assert list(calibrations["role"]) == ["treatment", "outcome", "treatment", "outcome"]
    assert all(strength != 0.0 for strength in calibrations["strength"])


def test_polars_frames_carry_estimates_displacements_and_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the polars half in its own test.

    A mid-test ``importorskip`` would report the pandas checks as skipped whenever polars
    is absent, and a skipped correctness check reads exactly like a passing one.
    """
    polars = pytest.importorskip("polars")
    result = _fit(backend="polars", seed=8)
    _record_refits(result, monkeypatch, fail_call=2)
    surface = simulated_confounding(
        result,
        grid=_grid(),
        benchmark_covariates=("W1",),
        random_state=19,
    )
    frame = surface.to_frame()
    calibrations = surface.calibration_frame()

    assert isinstance(frame, polars.DataFrame)
    assert isinstance(calibrations, polars.DataFrame)
    assert frame.columns == [
        "treatment_strength",
        "outcome_strength",
        "movement_scale",
        "estimate",
        "displacement",
        "induced_treatment_association",
        "error_type",
        "message",
    ]
    assert frame.height == 4
    assert frame["estimate"][0] == result["ate"].psi
    assert frame["displacement"][1] == pytest.approx(surface.cells[1].displacement)
    assert frame["displacement"][1] != 0.0
    assert frame["estimate"][2] is None
    assert frame["error_type"][2] == "RuntimeError"
    assert "deliberate refit failure" in frame["message"][2]
    assert calibrations.columns == ["covariate", "role", "family", "strength", "method"]
    assert calibrations.height == 2


def test_grid_validation_and_binomial_boundary(
    binomial_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="contain zero"):
        ConfounderStrengthGrid(treatment=(0.1,), outcome=(0.0,))
    with pytest.raises(ValueError, match="duplicates"):
        ConfounderStrengthGrid(treatment=(0.0, -0.0), outcome=(0.0,))
    calls = _record_refits(binomial_result, monkeypatch)
    with pytest.raises(ValueError, match="binary treatment strengths"):
        simulated_confounding(
            binomial_result,
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.6), outcome=(0.0,)),
        )
    # The lower bound needs its own witness.  Without it a negative binary strength
    # reaches ``_flip_mask``, where ``NormalDist().inv_cdf`` raises inside the per-cell
    # try block.  The surface would then hand back a partially complete result with a
    # retained failure, rather than refuse before the latent draw and the refit.
    with pytest.raises(ValueError, match="binary treatment strengths"):
        simulated_confounding(
            binomial_result,
            grid=ConfounderStrengthGrid(treatment=(0.0, -0.1), outcome=(0.0,)),
        )
    with pytest.raises(ValueError, match="binomial outcome strengths"):
        simulated_confounding(
            binomial_result,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, 0.6)),
        )
    assert calls == []


@pytest.mark.parametrize("estimand", ["att", "atc", "ate_regime", "msm"])
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
    with pytest.raises(CapabilityError, match="only a marginal ATE, counterfactual arm mean"):
        simulated_confounding(altered, estimand=estimand, grid=_grid())
    assert calls == []


@pytest.mark.parametrize(
    ("target", "fixture_name"),
    [("rr", "risk_ratio_result"), ("or", "odds_ratio_result")],
)
@pytest.mark.parametrize(
    ("layer", "message"),
    [
        ("key-alias", "inconsistent registered binary parameter metadata"),
        ("key-direction", "inconsistent registered binary parameter metadata"),
        ("functional", "inconsistent registered binary parameter metadata"),
        ("typed", "inconsistent registered binary parameter metadata"),
        ("estimator", "inconsistent registered binary parameter metadata"),
        ("evidence", "inconsistent registered binary parameter metadata"),
        ("registry", "inconsistent registered ratio target metadata"),
        ("outcome-family", "risk ratio or odds ratio for a binary"),
        ("estimate-name", "inconsistent ratio-scale estimate metadata"),
        ("estimate-scale", "inconsistent ratio-scale estimate metadata"),
        ("missing-log", "needs the stored log-scale estimate for a ratio contrast"),
        ("degenerate-ratio", "finite positive ratio"),
        ("overflowed-ratio", "finite positive ratio"),
        ("inconsistent-log", "inconsistent reported and log-scale ratio estimates"),
    ],
)
def test_ratio_checks_every_structured_layer_before_the_latent_draw(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    fixture_name: str,
    layer: str,
    message: str,
) -> None:
    """A ratio request must agree across identification, fitting, and reporting state.

    Each layer pins its own refusal, so one guard cannot stand in for another. Both
    ratios run every layer, because the typed-estimand gate reads the requested target.
    """
    result = request.getfixturevalue(fixture_name)
    other = "or" if target == "rr" else "rr"
    if layer in {"key-alias", "key-direction"}:
        key = result.parameter_keys[target]
        key = (
            replace(key, alias=other)
            if layer == "key-alias"
            else replace(key, value=key.reference, reference=key.value)
        )
        result = replace(result, parameter_keys={target: key})
    elif layer == "functional":
        result = _with_functional(result, target="ate")
    elif layer == "typed":
        result = replace(
            result,
            identified_effect=replace(
                result.identified_effect,
                estimand=OddsRatio() if target == "rr" else RiskRatio(),
            ),
        )
    elif layer == "estimator":
        estimator = copy(result.estimator)
        estimator.estimands = ("ate",)
        result = replace(result, estimator=estimator)
    elif layer == "evidence":
        identification = replace(
            result.identified_effect.identification,
            references=("unregistered identification",),
        )
        result = replace(
            result,
            identified_effect=replace(result.identified_effect, identification=identification),
        )
    elif layer == "registry":
        from cleverly.targets import TARGETS

        monkeypatch.setitem(TARGETS, target, replace(TARGETS[target], scale="difference"))
    elif layer == "outcome-family":
        result = replace(result, data=replace(result.data, family="gaussian"))
    else:
        estimate = result[target]
        if layer == "estimate-name":
            estimate = replace(estimate, name=other)
        elif layer == "estimate-scale":
            estimate = replace(estimate, scale="difference")
        elif layer == "missing-log":
            estimate = replace(estimate, log_psi=None)
        elif layer == "degenerate-ratio":
            # The shape a boundary counterfactual risk produces: a zero ratio, whose log
            # is not finite. The finiteness guard has to fire, because ``exp(-inf)`` is
            # exactly ``0.0`` and the equality check below would accept the pair.
            estimate = replace(estimate, psi=0.0, log_psi=float("-inf"))
        elif layer == "overflowed-ratio":
            # A finite log whose exponential overflows. Of the three clauses in that guard
            # this is the only one a nonfinite log cannot also trip, so it needs its own
            # witness. Without it the ``psi`` finiteness test could be deleted unnoticed.
            estimate = replace(estimate, psi=float("inf"), log_psi=1000.0)
        else:
            assert estimate.log_psi is not None
            estimate = replace(estimate, log_psi=estimate.log_psi + 0.25)
        result = replace(result, estimates={target: estimate})

    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(
        module,
        "_latent_child_seed",
        lambda _: pytest.fail("refused before the latent draw"),
    )

    with pytest.raises(CapabilityError, match=message):
        simulated_confounding(
            result,
            estimand=target,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )


@pytest.mark.parametrize(
    "layer",
    ["key-alias", "key-arm", "key-reference", "functional", "typed", "estimator", "evidence"],
)
def test_binary_mean_checks_every_parameter_state_layer_before_the_latent_draw(
    binary_mean_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    result = binary_mean_result
    alias = "ey1"
    if layer in {"key-alias", "key-arm", "key-reference"}:
        key = result.parameter_keys[alias]
        if layer == "key-alias":
            key = replace(key, alias="ey0")
        elif layer == "key-arm":
            key = replace(key, value=0)
        else:
            key = replace(key, reference=0)
        result = replace(result, parameter_keys={alias: key})
    elif layer == "functional":
        result = _with_functional(result, target="ate")
    elif layer == "typed":
        result = replace(
            result,
            identified_effect=replace(
                result.identified_effect,
                estimand=CounterfactualMean(treatment=0),
            ),
        )
    elif layer == "estimator":
        estimator = copy(result.estimator)
        estimator.estimands = ("ate",)
        result = replace(result, estimator=estimator)
    else:
        identification = replace(
            result.identified_effect.identification,
            references=("unregistered identification",),
        )
        result = replace(
            result,
            identified_effect=replace(result.identified_effect, identification=identification),
        )

    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(
        module,
        "_latent_child_seed",
        lambda _: pytest.fail("refused before the latent draw"),
    )
    with pytest.raises(CapabilityError, match="registered binary parameter metadata"):
        simulated_confounding(
            result,
            estimand=alias,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )


@pytest.mark.parametrize("layer", ["key-direction", "estimator-reference"])
def test_binary_ate_checks_the_fitted_contrast_before_the_latent_draw(
    gaussian_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    result = gaussian_result
    if layer == "key-direction":
        key = result.parameter_keys["ate"]
        result = replace(
            result,
            parameter_keys={"ate": replace(key, value=key.reference, reference=key.value)},
        )
    else:
        estimator = copy(result.estimator)
        estimator.reference = result.data.arm_label(1.0)
        result = replace(result, estimator=estimator)

    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(
        module,
        "_latent_child_seed",
        lambda _: pytest.fail("refused before the latent draw"),
    )
    with pytest.raises(CapabilityError, match="registered binary parameter metadata"):
        simulated_confounding(
            result,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )


def test_explicit_ey_alias_refuses_a_swapped_structured_arm_before_refit(
    binary_means_result: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = binary_means_result
    aliases = {key.value: name for name, key in result.parameter_keys.items()}
    alias = aliases[1]
    key = replace(result.parameter_keys[alias], value=0)
    altered = replace(result, parameter_keys={**result.parameter_keys, alias: key})
    monkeypatch.setattr(
        altered.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )
    with pytest.raises(CapabilityError, match="registered binary parameter metadata"):
        simulated_confounding(
            altered,
            estimand=alias,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("multi-arm", "category-valued perturbation law"),
        ("missing", "missing-outcome"),
        ("intermediate", "controlled-direct-effect"),
        ("weights", "observation-weighted"),
        ("cluster", "clustered"),
        ("restored", "replayable"),
        ("estimator", "supports ordinary TMLE"),
        ("outcome-family", "outcome family"),
        ("identification", "identification metadata"),
        ("functional-type", "backdoor-identified marginal parameter"),
        ("provider", "explicit-adjustment backdoor provenance"),
        ("key", "structured parameter key"),
        ("provenance", "registered binary parameter metadata"),
        ("declared-provenance", "registered binary parameter metadata"),
        ("target-provenance", "registered binary parameter metadata"),
        ("conditional", "counterfactual arm mean"),
        ("stochastic", "marginal arm-indexed parameter"),
        ("incremental", "marginal arm-indexed parameter"),
        ("modified", "marginal arm-indexed parameter"),
        ("msm", "marginal arm-indexed parameter"),
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
    elif change == "restored":
        result = replace(result, estimator=None)
    elif change == "estimator":
        result = replace(result, estimator=_UnsupportedTMLE())
    elif change == "outcome-family":
        data = replace(data, family="poisson")
    elif change == "identification":
        result = replace(result, identified_effect=None)
    elif change == "functional-type":
        identified = replace(result.identified_effect, functional=SimpleNamespace())
        result = replace(result, identified_effect=identified)
    elif change == "provider":
        identified = replace(result.identified_effect, provider=SimpleNamespace())
        result = replace(result, identified_effect=identified)
    elif change == "key":
        result = replace(result, parameter_keys={"ate": SimpleNamespace()})
    elif change == "provenance":
        identification = replace(
            result.identified_effect.identification,
            references=("unregistered identification",),
        )
        identified = replace(result.identified_effect, identification=identification)
        result = replace(result, identified_effect=identified)
    elif change == "declared-provenance":
        identified = replace(result.identified_effect, estimand=ATT())
        result = replace(result, identified_effect=identified)
    elif change == "target-provenance":
        result = _with_functional(result, target="att")
    elif change == "conditional":
        key = replace(result.parameter_keys["ate"], stratum=(0,))
        result = replace(result, parameter_keys={"ate": key})
    elif change == "stochastic":
        result = _with_functional(
            result,
            axis="regime",
            interventions=(object(),),
        )
    elif change == "incremental":
        result = _with_functional(
            result,
            axis="ipsi",
            interventions=(object(),),
        )
    elif change == "modified":
        result = _with_functional(
            result,
            axis="shift",
            interventions=(object(),),
        )
    elif change == "msm":
        result = _with_functional(result, axis="msm", msm=object())
    result = replace(result, data=data)
    calls = [] if result.estimator is None else _record_refits(result, monkeypatch)
    with pytest.raises(CapabilityError, match=message):
        simulated_confounding(result, grid=_grid())
    assert calls == []


@pytest.mark.parametrize(
    ("change", "benchmark_covariates", "error_type", "message"),
    [
        ("categorical", ("G",), CapabilityError, "categorical covariate"),
        ("constant", ("W1",), CapabilityError, "constant covariate"),
        ("non-string", ("W1", 1), TypeError, "only column names"),
        ("duplicate", ("W1", "W1"), ValueError, "contains duplicates"),
        ("unknown", ("unknown",), ValueError, "is unavailable"),
    ],
)
def test_invalid_calibrations_are_refused_before_refit(
    gaussian_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    benchmark_covariates: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    from cleverly.data.causal_data import CategoricalEncoding

    data = gaussian_result.data
    if change == "categorical":
        data = replace(
            data,
            encodings=(CategoricalEncoding("G", ("a", "b"), "a", ("G_b",)),),
        )
    elif change == "constant":
        covariates = data.covariates.copy()
        covariates[:, data.covariate_names.index("W1")] = 1.0
        data = replace(data, covariates=covariates)
    result = replace(gaussian_result, data=data)
    calls = _record_refits(result, monkeypatch)
    with pytest.raises(error_type, match=message):
        simulated_confounding(
            result,
            grid=_grid(),
            benchmark_covariates=benchmark_covariates,
        )
    assert calls == []


def test_non_result_and_ambiguous_alias_are_refused(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(CapabilityError, match="TMLEResult"):
        simulated_confounding(SimpleNamespace(), grid=_grid())
    calls = _record_refits(gaussian_result, monkeypatch)
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(
        module,
        "_zero_delta_policy_means",
        lambda _: pytest.fail("a binary fit declares no shift policy to filter"),
    )
    with pytest.raises(ValueError, match="unavailable") as refusal:
        simulated_confounding(gaussian_result, estimand="unknown", grid=_grid())
    assert str(refusal.value) == (
        f"estimand 'unknown' is unavailable; choose one of {list(gaussian_result.estimates)}"
    )
    assert calls == []


def test_continuous_requires_an_explicit_policy_parameter_alias_before_refit(
    continuous_gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mutation control on the alias filter, run against a fabricated alias set.

    No real fit reports ``ate_shift`` and ``ey_shift`` together. This test fabricates
    the mixed set to hold the filter's two accepted prefixes.
    """
    calls = _record_continuous_refits(continuous_gaussian_result, monkeypatch)
    alias = _shift_alias(continuous_gaussian_result)
    with_level = replace(
        continuous_gaussian_result,
        estimates={
            **continuous_gaussian_result.estimates,
            "ey_shift[up half]": continuous_gaussian_result[alias],
        },
    )
    assert "ey_shift[up half]" in with_level.estimates

    with pytest.raises(ValueError, match=r"explicit ey_shift\[\.\.\.\] policy mean") as refusal:
        simulated_confounding(
            with_level,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )
    assert alias in str(refusal.value)
    assert "ey_shift[up half]" in str(refusal.value)
    assert calls == []


def test_a_means_only_fit_requires_selection_and_names_the_available_means(
    continuous_means_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default cannot select one mean, and it advertises only an admissible one.

    The fit reports two policy means. The surface refuses the zero-delta one, so naming
    it here would offer the reader a parameter that the next call rejects.
    """
    result = continuous_means_result
    assert sorted(result.estimates) == ["ey_shift[natural course]", "ey_shift[up half]"]
    assert not [name for name in result.estimates if name.startswith("ate_shift[")]
    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )

    with pytest.raises(ValueError, match=r"explicit ey_shift\[\.\.\.\] policy mean") as refusal:
        simulated_confounding(
            result,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )

    message = str(refusal.value)
    assert "ey_shift[up half]" in message
    assert "ey_shift[natural course]" not in message


def test_an_unavailable_alias_never_advertises_the_refused_natural_course_mean(
    continuous_means_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped alias must not be answered with a mean that the next call refuses.

    ``ey_shift[natural course]`` names the zero-delta policy, and the surface refuses its
    mean. The availability message filters that alias for the same reason the selection
    message does.
    """
    result = continuous_means_result
    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )

    with pytest.raises(ValueError, match="is unavailable") as refusal:
        simulated_confounding(
            result,
            estimand="ey_shif[up half]",
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )

    message = str(refusal.value)
    assert "choose one of ['ey_shift[up half]']" in message
    assert "ey_shift[natural course]" not in message


def test_an_unavailable_alias_reports_none_when_the_filter_empties_the_list(
    continuous_means_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty list must read as an absence, not as a choice.

    This fabricated fit reports the zero-delta mean alone. The filter removes it, so the
    message states that the fit reports no admissible alias.
    """
    result = continuous_means_result
    natural = _mean_alias(result, policy="natural course")
    fabricated = replace(result, estimates={natural: result[natural]})
    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )

    with pytest.raises(ValueError, match="this fit reports none") as refusal:
        simulated_confounding(
            fabricated,
            estimand=_mean_alias(result),
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )

    assert "choose one of" not in str(refusal.value)


def test_a_zero_delta_policy_mean_is_refused_before_the_latent_draw(
    continuous_means_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-delta shift is the identity map, so its mean carries no treatment path.

    ``d_delta(a, w)`` returns ``a`` on both branches at ``delta == 0``, whatever the cap.
    The mean of that policy is ``E[Y]``, and the treatment axis of the surface is
    identically zero. Without this refusal an analyst reads the outcome level shift of
    that column as robustness evidence for a parameter no common cause can move.
    """
    result = continuous_means_result
    alias = _mean_alias(result, policy="natural course")
    natural = result.identified_effect.functional.interventions[0]
    assert natural.name == "natural course"
    assert natural.delta == 0.0
    # The plug-in reaches ``E[Y]`` through the targeting arithmetic rather than through
    # ``np.mean``, so the two agree to rounding rather than bit for bit.  The observed
    # gap is one unit in the last place, and it varies with the platform BLAS.
    assert result[alias].psi == pytest.approx(float(np.mean(result.data.outcome)), rel=1e-12)
    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(
        module,
        "_latent_child_seed",
        lambda _: pytest.fail("refused before the latent draw"),
    )

    with pytest.raises(CapabilityError, match="natural course") as refusal:
        simulated_confounding(
            result,
            estimand=alias,
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0, 0.2)),
        )

    message = str(refusal.value)
    assert alias in message
    assert "E[Y]" in message
    assert "no counterfactual treatment dependence" in message


def test_a_contrast_against_the_natural_course_stays_accepted(
    continuous_gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberate-mutation control that stops the zero-delta refusal over-firing.

    The shipped continuous composition contrasts a shifted policy against the natural
    course, so the refused zero-delta policy is its reference. That contrast keeps
    counterfactual treatment dependence, and the surface has to accept it.
    """
    result = continuous_gaussian_result
    alias = _shift_alias(result)
    assert alias == "ate_shift[up half vs natural course]"
    reference = result.identified_effect.functional.interventions[0]
    assert reference.name == "natural course"
    assert reference.delta == 0.0
    calls = _record_continuous_refits(result, monkeypatch, alias=alias)

    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0, 0.2)),
        random_state=19,
    )

    assert surface.complete
    assert surface.estimand == alias
    assert len(calls) == 3


def test_continuous_policy_mean_reuses_the_grid_and_refit_paths(
    continuous_means_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin selection, the exact anchor, common randomness, and a nonzero mean witness."""
    result = continuous_means_result
    alias = _mean_alias(result)
    calls = _record_continuous_refits(result, monkeypatch, alias=alias)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0, 0.2))
    surface = simulated_confounding(result, estimand=alias, grid=grid, random_state=19)

    assert surface.estimand == alias
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.cells[0].displacement == 0.0
    assert len(calls) == 3
    outcome_only, treatment_only, both = (call[0] for call in calls)
    np.testing.assert_array_equal(outcome_only.treatment, result.data.treatment)
    np.testing.assert_array_equal(treatment_only.outcome, result.data.outcome)
    np.testing.assert_array_equal(both.treatment, treatment_only.treatment)
    np.testing.assert_array_equal(both.outcome, outcome_only.outcome)
    np.testing.assert_allclose(
        (treatment_only.treatment - result.data.treatment) / 0.3,
        -(outcome_only.outcome - result.data.outcome) / 0.2,
    )
    for cell, (data, _) in zip(surface.cells[1:], calls, strict=True):
        expected = float(np.mean(data.treatment * data.outcome))
        assert cell.estimate == pytest.approx(expected)
        assert cell.displacement == pytest.approx(expected - result[alias].psi)
    assert any(abs(cell.displacement or 0.0) > 1e-6 for cell in surface.cells[1:])


def test_continuous_policy_mean_runs_a_real_ordinary_tmle_refit(
    continuous_means_result: Any,
) -> None:
    """Anchor the policy-mean composition on the real estimator rather than on a spy.

    Every other ``ey_shift`` check replaces ``TMLE.refit`` with a spy that returns
    ``mean(A * Y)``. That fake answers the treatment axis by construction, so it would
    pass on a mean the real estimator never moves. The gated cell below carries a nonzero
    treatment strength and a nonzero outcome strength, because those are the two axes a
    confounding path needs. ``random_state`` equals the seed of the fit, so the anchor and
    every cell share the cross-fitting folds and no movement here is a fold artifact.
    """
    result = continuous_means_result
    alias = _mean_alias(result)
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 1.0), outcome=(0.0, 0.5)),
        random_state=7,
    )
    anchor, outcome_only, treatment_only, joint = surface.cells

    assert type(result.estimator) is TMLE
    assert surface.complete
    assert surface.estimand == alias
    assert surface.treatment_family == "continuous"
    assert anchor.estimate == result[alias].psi
    assert anchor.displacement == 0.0

    # ``A' = A + k_A U`` raises the latent-treatment correlation from about +0.063 at the
    # anchor to about +0.698 at ``k_A = 1``, so the treatment axis reached this fit.
    assert anchor.induced_treatment_association is not None
    assert abs(anchor.induced_treatment_association) < 0.2
    assert treatment_only.induced_treatment_association is not None
    assert treatment_only.induced_treatment_association > 0.4

    # The confounded cell moves this mean by about -0.383, which is far above the
    # numerical noise of a repeated fit under one seed.
    assert (joint.treatment_strength, joint.outcome_strength) == (1.0, 0.5)
    assert joint.displacement is not None
    assert abs(joint.displacement) > 0.1

    # The zero treatment-strength cell moves by about -0.050 under the level shift alone,
    # and the confounded cell has to sit below it.
    assert outcome_only.displacement is not None
    assert treatment_only.displacement is not None
    assert joint.displacement < outcome_only.displacement
    assert joint.displacement < treatment_only.displacement

    # The symmetric reading guard. The level shift does not cancel in a mean, so the
    # summary has to say that the zero treatment-strength column is an artifact.
    summary = surface.summary()
    assert "A cell whose treatment strength is zero also has no confounding path" in summary
    assert "A policy mean keeps that level shift, and a contrast removes most of it" in summary


@pytest.mark.parametrize(
    ("fixture", "family", "shape"),
    [
        ("continuous_means_result", "gaussian", "mean"),
        ("continuous_gaussian_result", "gaussian", "contrast"),
        ("continuous_binomial_means_result", "binomial", "mean"),
        ("continuous_binomial_result", "binomial", "contrast"),
    ],
)
def test_the_reading_guard_follows_the_outcome_family_of_the_fit(
    request: pytest.FixtureRequest, fixture: str, family: str, shape: str
) -> None:
    """The outcome axis of the guard reads the outcome family, not the estimand alone.

    The binomial law flips ``Y`` in the upper latent tail, which maps
    ``E[Y'|a,w] = p + (1 - 2p) E[Y|a,w]``. That map attenuates the fitted outcome
    regression. It is not a level shift, and a contrast does not cancel it. A guard that
    branched on the estimand alone printed the Gaussian sentence on a binomial fit, and
    that sentence contradicted the ``outcome_law`` line of its own summary.
    """
    result = request.getfixturevalue(fixture)
    alias = _mean_alias(result) if shape == "mean" else _shift_alias(result)
    surface = simulated_confounding(
        result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        random_state=7,
    )
    summary = surface.summary()

    assert surface.outcome_family == family
    assert surface.outcome_law in summary
    assert "A cell whose treatment strength is zero also has no confounding path" in summary
    if family == "gaussian":
        assert "subtracts signed strength" in surface.outcome_law
        assert "Its movement reports the outcome level shift alone" in summary
        assert "A policy mean keeps that level shift, and a contrast removes most of it" in summary
        assert "A small residual stays" in summary
    else:
        assert "flipped in the declared upper latent-normal tail" in surface.outcome_law
        assert "Its movement reports the outcome perturbation alone" in summary
        assert "attenuates the fitted outcome regression" in summary
        assert "level shift" not in summary
        assert "cancel" not in summary


def test_continuous_policy_mean_retains_a_refit_failure(
    continuous_means_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = _mean_alias(continuous_means_result)
    _record_continuous_refits(continuous_means_result, monkeypatch, alias=alias, fail_call=1)
    surface = simulated_confounding(
        continuous_means_result,
        estimand=alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,)),
        random_state=23,
    )
    assert surface.cells[1].failure is not None
    assert surface.cells[1].failure.error_type == "RuntimeError"


@pytest.mark.parametrize("layer", ["key", "functional", "typed", "estimator", "fitted"])
def test_policy_mean_checks_every_policy_state_layer_before_the_latent_draw(
    continuous_means_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    result = continuous_means_result
    alias = _mean_alias(result)
    if layer == "key":
        key = replace(result.parameter_keys[alias], value="wrong")
        result = replace(result, parameter_keys={**result.parameter_keys, alias: key})
    elif layer == "functional":
        policies = list(result.identified_effect.functional.interventions)
        policies[1] = replace(policies[1], delta=0.75)
        result = _with_functional(result, interventions=tuple(policies))
    elif layer == "typed":
        typed = result.identified_effect.estimand
        policies = list(typed.shifts)
        policies[1] = replace(policies[1], delta=0.75)
        result = replace(
            result,
            identified_effect=replace(
                result.identified_effect, estimand=replace(typed, shifts=tuple(policies))
            ),
        )
    elif layer == "estimator":
        estimator = copy(result.estimator)
        policies = list(estimator.shifts)
        policies[1] = replace(policies[1], delta=0.75)
        estimator.shifts = tuple(policies)
        result = replace(result, estimator=estimator)
    else:
        shifts = result.nuisance.shifts
        assert shifts is not None
        shifts = replace(shifts, deltas=(0.0, 0.75))
        repeat = replace(result.repeats[0], nuisance=replace(result.nuisance, shifts=shifts))
        result = replace(result, repeats=(repeat,))

    monkeypatch.setattr(
        result.estimator,
        "refit",
        lambda *args, **kwargs: pytest.fail("refused before any refit"),
    )
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(
        module,
        "_latent_child_seed",
        lambda _: pytest.fail("refused before the latent draw"),
    )
    with pytest.raises(CapabilityError, match="structured shift metadata"):
        simulated_confounding(
            result,
            estimand=alias,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )


def test_the_combined_report_reads_grammatically_for_one_and_for_two_arguments(
    gaussian_result: Any,
    binary_mean_result: Any,
    binary_means_result: Any,
    continuous_gaussian_result: Any,
) -> None:
    """Pin the rendered refusal, which ``requires_arguments`` alone does not cover.

    ``simulated_confounding`` on a continuous fit is the first row in the package that
    declares two arguments, and a bare ``", ".join`` rendered "an explicit grid, estimand
    argument". That is singular for two names, and it reads as one argument called
    "grid, estimand". The single-argument wording is unchanged, so this test pins both.
    """

    def rendered(result: Any) -> Any:
        report = result.sensitivity.run_all(include_refits=True)
        return next(item for item in report.items if item.name == "simulated_confounding")

    binary = rendered(gaussian_result)
    assert binary.status is AssessmentStatus.UNAVAILABLE
    assert binary.detail == (
        "needs an explicit grid argument, which a combined report has no basis to choose"
    )
    assert binary.next_steps == (
        "call result.sensitivity.simulated_confounding() directly with grid",
    )

    sole_mean = rendered(binary_mean_result)
    assert sole_mean.detail == binary.detail
    assert sole_mean.next_steps == binary.next_steps

    several_means = rendered(binary_means_result)
    assert several_means.detail == (
        "needs explicit grid and estimand arguments, which a combined report has no basis to choose"
    )
    assert several_means.next_steps == (
        "call result.sensitivity.simulated_confounding() directly with grid and estimand",
    )

    continuous = rendered(continuous_gaussian_result)
    assert continuous.status is AssessmentStatus.UNAVAILABLE
    assert continuous.detail == (
        "needs explicit grid and estimand arguments, which a combined report has no basis to choose"
    )
    assert continuous.next_steps == (
        "call result.sensitivity.simulated_confounding() directly with grid and estimand",
    )


def test_three_policy_fit_accepts_the_reference_contrast_and_refuses_any_other_base(
    three_policy_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A contrast against a fitted non-reference policy has no reachable alias.

    ``_difference_against_reference`` names every ``ate_shift`` alias against the fitted
    reference, so a three-policy fit reports no ``ate_shift[up one vs up half]`` and a
    caller who asks for one meets the ``estimand is unavailable`` refusal first. The
    ``key.reference != fitted_reference`` guard therefore needs fabricated metadata to
    reach, which the second half of this test supplies. The first half pins the
    three-policy accept path, which a caller can reach and no other test covers.
    """
    result = three_policy_result
    aliases = sorted(name for name in result.estimates if name.startswith("ate_shift["))
    assert aliases == [
        "ate_shift[up half vs natural course]",
        "ate_shift[up one vs natural course]",
    ]
    fitted = result.nuisance.shifts
    assert fitted is not None
    assert fitted.names == ("natural course", "up half", "up one")
    assert fitted.names[int(fitted.reference)] == "natural course"

    calls = _record_continuous_refits(result, monkeypatch, alias=aliases[1])
    surface = simulated_confounding(
        result,
        estimand=aliases[1],
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0,)),
        random_state=19,
    )
    assert surface.complete
    assert surface.estimand == aliases[1]
    assert len(calls) == 1

    # The unreachable branch, reached with fabricated metadata. The alias, its value, and
    # the name the surface rebuilds from them all agree here, so the reference clause is
    # the only one that can refuse this request.
    contrast = "ate_shift[up one vs up half]"
    key = replace(result.parameter_keys[aliases[1]], alias=contrast, reference="up half")
    assert key.value == "up one"
    fabricated = replace(
        result,
        estimates={contrast: result[aliases[1]]},
        parameter_keys={contrast: key},
    )
    with pytest.raises(CapabilityError, match="structured shift metadata"):
        simulated_confounding(
            fabricated,
            estimand=contrast,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )
    assert len(calls) == 1


def test_the_capability_row_declares_the_fit_dependent_estimand_requirement(
    gaussian_result: Any,
    binary_mean_result: Any,
    binary_means_result: Any,
    continuous_gaussian_result: Any,
) -> None:
    """``run_all`` learns the requirement from the row, and the row is fit-dependent.

    A continuous fit refuses the bare ``ate`` default, so its caller must pass ``grid``
    and ``estimand`` both. A binary fit needs ``grid`` alone.
    """

    def rows(result: Any) -> dict[str, Any]:
        return {row.operation: row for row in result.sensitivity.capabilities}

    binary = rows(gaussian_result)
    sole_mean = rows(binary_mean_result)
    several_means = rows(binary_means_result)
    continuous = rows(continuous_gaussian_result)

    assert binary["simulated_confounding"].requires_arguments == ("grid",)
    assert sole_mean["simulated_confounding"].requires_arguments == ("grid",)
    assert several_means["simulated_confounding"].requires_arguments == ("grid", "estimand")
    assert continuous["simulated_confounding"].requires_arguments == ("grid", "estimand")
    assert binary["benchmark"].requires_arguments == continuous["benchmark"].requires_arguments


def test_the_facade_substitutes_only_a_sole_binary_mean(
    binary_mean_result: Any,
    binary_means_result: Any,
) -> None:
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    sole = binary_mean_result.sensitivity.simulated_confounding(grid=grid)
    assert sole.estimand == "ey1"
    assert sole.cells[0].estimate == binary_mean_result["ey1"].psi

    with pytest.raises(ValueError, match="estimand 'ate' is unavailable"):
        binary_means_result.sensitivity.simulated_confounding(grid=grid)


def test_the_facade_names_the_source_boundary_for_a_sole_att(att_result: Any) -> None:
    """An ``att``-only fit reads the boundary refusal, not a missing-``ate`` message.

    The ratio-aware default names no candidate here, so the linear parameter set supplies
    ``"att"``. Without that fall-through the facade keeps the ``"ate"`` default, and the
    caller is told to pass ``"att"``, which then raises a different refusal.
    """
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    assert list(att_result.estimates) == ["att"]

    with pytest.raises(CapabilityError) as excinfo:
        att_result.sensitivity.simulated_confounding(grid=grid)

    assert "outside its source boundary" in str(excinfo.value)
    assert "is unavailable" not in str(excinfo.value)


def test_continuous_strengths_are_signed_and_outcome_bounds_follow_the_family(
    continuous_gaussian_result: Any,
    continuous_binomial_result: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gaussian_calls = _record_continuous_refits(continuous_gaussian_result, monkeypatch)
    gaussian = simulated_confounding(
        continuous_gaussian_result,
        estimand=_shift_alias(continuous_gaussian_result),
        grid=ConfounderStrengthGrid(treatment=(0.0, -0.75, 0.8), outcome=(0.0, -0.6)),
    )
    assert gaussian.complete
    assert len(gaussian_calls) == 5

    binomial_calls = _record_continuous_refits(continuous_binomial_result, monkeypatch)
    with pytest.raises(ValueError, match="binomial outcome strengths"):
        simulated_confounding(
            continuous_binomial_result,
            estimand=_shift_alias(continuous_binomial_result),
            grid=ConfounderStrengthGrid(treatment=(0.0, -0.75, 0.8), outcome=(0.0, -0.1)),
        )
    assert binomial_calls == []


def test_continuous_retains_refit_failure_and_replays_seed(
    continuous_gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = continuous_gaussian_result
    alias = _shift_alias(result)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
    calls = _record_continuous_refits(result, monkeypatch, fail_call=1)
    failed = simulated_confounding(result, estimand=alias, grid=grid, random_state=23)

    assert len(calls) == 1
    assert not failed.complete
    assert failed.cells[1].failure is not None
    assert failed.cells[1].failure.error_type == "RuntimeError"
    assert failed.cells[1].failure.seed == failed.root_seed
    assert failed.cells[1].induced_treatment_association is not None

    replay_calls = _record_continuous_refits(result, monkeypatch)
    replayed = simulated_confounding(
        result, estimand=alias, grid=grid, random_state=failed.root_seed
    )
    assert len(replay_calls) == 1
    assert replayed.latent_seed == failed.latent_seed
    assert np.array_equal(replay_calls[0][0].treatment, calls[0][0].treatment)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("collaborative", "exact ordinary TMLE"),
        ("key-estimand", "only a marginal ey_shift policy mean or ate_shift contrast"),
        ("key-axis", "only a marginal ey_shift policy mean or ate_shift contrast"),
        ("conditional", "only a marginal ey_shift policy mean or ate_shift contrast"),
        ("key-alias", "structured shift metadata"),
        ("key-value", "structured shift metadata"),
        ("fitted-names", "structured shift metadata"),
        ("fitted-deltas", "structured shift metadata"),
        ("fitted-reference", "structured shift metadata"),
        ("fitted-shifted", "structured shift metadata"),
        ("fitted-capped", "structured shift metadata"),
        ("functional-name", "structured shift metadata"),
        ("functional-delta", "structured shift metadata"),
        ("functional-cap", "structured shift metadata"),
        ("functional-reference", "structured shift metadata"),
        ("typed-delta", "structured shift metadata"),
        ("typed-cap", "structured shift metadata"),
        ("typed-reference", "structured shift metadata"),
        ("estimator-delta", "structured shift metadata"),
        ("estimator-cap", "structured shift metadata"),
        ("estimator-reference", "structured shift metadata"),
        ("provenance", "registered modified-policy identification provenance"),
        ("declared-provenance", "registered modified-policy identification provenance"),
        ("target-provenance", "registered modified-policy identification provenance"),
        ("arm-axis", "modified-treatment-policy parameter"),
    ],
)
def test_continuous_alias_metadata_and_provenance_are_refused_before_refit(
    continuous_gaussian_result: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    message: str,
) -> None:
    result = continuous_gaussian_result
    alias = _shift_alias(result)
    if change == "collaborative":
        result = replace(result, estimator=_UnsupportedTMLE())
    elif change in {"key-estimand", "key-axis", "conditional", "key-alias", "key-value"}:
        key = result.parameter_keys[alias]
        if change == "key-estimand":
            key = replace(key, estimand="ate")
        elif change == "key-axis":
            key = replace(key, axis="arm")
        elif change == "conditional":
            key = replace(key, stratum=(0,))
        elif change == "key-alias":
            key = replace(key, alias="ate_shift[wrong vs natural course]")
        else:
            key = replace(key, value="natural course")
        result = replace(result, parameter_keys={alias: key})
    elif change in {
        "fitted-names",
        "fitted-deltas",
        "fitted-reference",
        "fitted-shifted",
        "fitted-capped",
    }:
        shifts = result.nuisance.shifts
        assert shifts is not None
        if change == "fitted-names":
            shifts = replace(shifts, names=("natural course", "wrong"))
        elif change == "fitted-deltas":
            shifts = replace(shifts, deltas=(0.0, 0.75))
        elif change == "fitted-reference":
            shifts = replace(shifts, reference=1.0)
        elif change == "fitted-shifted":
            shifted = shifts.shifted.copy()
            shifted[:, 1] += 0.25
            shifts = replace(shifts, shifted=shifted)
        else:
            capped = shifts.capped.copy()
            capped[:, 1] = ~capped[:, 1]
            shifts = replace(shifts, capped=capped)
        repeat = replace(result.repeats[0], nuisance=replace(result.nuisance, shifts=shifts))
        result = replace(result, repeats=(repeat,))
    elif change in {"functional-name", "functional-delta", "functional-cap"}:
        interventions = list(result.identified_effect.functional.interventions)
        if change == "functional-name":
            interventions[1] = replace(interventions[1], name="wrong")
        elif change == "functional-delta":
            interventions[1] = replace(interventions[1], delta=0.75)
        else:
            interventions[1] = replace(interventions[1], cap=2.0)
        result = _with_functional(result, interventions=tuple(interventions))
    elif change == "functional-reference":
        result = _with_functional(result, reference="up half")
    elif change in {"typed-delta", "typed-cap", "typed-reference"}:
        typed = result.identified_effect.estimand
        if change == "typed-reference":
            typed = replace(typed, reference="up half")
        else:
            policies = list(typed.shifts)
            policies[1] = replace(
                policies[1],
                **({"delta": 0.75} if change == "typed-delta" else {"cap": 2.0}),
            )
            typed = replace(typed, shifts=tuple(policies))
        result = replace(
            result,
            identified_effect=replace(result.identified_effect, estimand=typed),
        )
    elif change in {"estimator-delta", "estimator-cap", "estimator-reference"}:
        estimator = copy(result.estimator)
        if change == "estimator-reference":
            estimator.reference = "up half"
        else:
            policies = list(estimator.shifts)
            policies[1] = replace(
                policies[1],
                **({"delta": 0.75} if change == "estimator-delta" else {"cap": 2.0}),
            )
            estimator.shifts = tuple(policies)
        result = replace(result, estimator=estimator)
    elif change == "provenance":
        identification = replace(
            result.identified_effect.identification,
            references=("unregistered identification",),
        )
        result = replace(
            result,
            identified_effect=replace(result.identified_effect, identification=identification),
        )
    elif change == "declared-provenance":
        result = replace(
            result,
            identified_effect=replace(result.identified_effect, estimand=ATE()),
        )
    elif change == "target-provenance":
        result = _with_functional(result, target="ate")
    else:
        result = _with_functional(result, axis="arm")

    calls = _record_continuous_refits(result, monkeypatch)
    with pytest.raises(CapabilityError, match=message):
        simulated_confounding(
            result,
            estimand=alias,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )
    assert calls == []


def test_multi_arm_refusal_precedes_the_latent_draw(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = replace(
        gaussian_result,
        data=replace(gaussian_result.data, treatment_levels=(0, 1, 2)),
    )
    calls = _record_refits(result, monkeypatch)
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")

    def forbidden_seed(_: int) -> int:
        raise AssertionError("latent seed requested before multi-arm refusal")

    monkeypatch.setattr(module, "_latent_child_seed", forbidden_seed)
    with pytest.raises(CapabilityError, match="category-valued perturbation law"):
        simulated_confounding(result, grid=_grid())
    assert calls == []


def test_summary_and_successful_cells_report_a_retained_failure(
    gaussian_result: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover the failure row of ``summary`` and the ``successful_cells`` property.

    Every other test builds a complete surface, so the failure row and the filtered
    property never run.
    """
    _record_refits(gaussian_result, monkeypatch, fail_call=2)
    surface = simulated_confounding(gaussian_result, grid=_grid(), random_state=19)
    text = surface.summary()
    lines = text.splitlines()

    assert not surface.complete
    assert len(surface.failures) == 1
    assert surface.successful_cells == (surface.cells[0], surface.cells[1], surface.cells[3])
    assert len(surface.successful_cells) == 3
    assert all(cell.estimate is not None for cell in surface.successful_cells)
    assert surface.cells[2] not in surface.successful_cells

    assert "failed" in text
    assert "RuntimeError" in text
    failure_row = next(line for line in lines if "failed" in line)
    assert "RuntimeError" in failure_row
    assert "0.1" in failure_row
    # The successful rows carry a signed movement and no failure marker.
    assert sum("failed" in line for line in lines) == 1
    assert f"original estimate: {gaussian_result['ate'].psi:.5g}" in text
    assert f"root={surface.root_seed}" in text
    assert f"latent={surface.latent_seed}" in text
    assert f"refit={surface.refit_seed}" in text
    assert "+" in "".join(line for line in lines if "failed" not in line and "0.2" in line)


@pytest.mark.parametrize(
    ("role", "values", "error_type", "message"),
    [
        ("treatment", 0.25, TypeError, "non-empty sequence"),
        ("outcome", 0.25, TypeError, "non-empty sequence"),
        ("treatment", (), ValueError, "must not be empty"),
        ("outcome", (), ValueError, "must not be empty"),
        ("treatment", (0.0, "0.1"), TypeError, "only numeric values"),
        ("outcome", (0.0, None), TypeError, "only numeric values"),
        ("treatment", (0.0, True), TypeError, "only numeric values"),
        ("outcome", (0.0, np.True_), TypeError, "only numeric values"),
        ("treatment", (0.0, float("nan")), ValueError, "must be finite"),
        ("outcome", (0.0, float("inf")), ValueError, "must be finite"),
        ("treatment", (0.0, 0.1, 0.1), ValueError, "must not contain duplicates"),
        ("outcome", (0.0, 0.2, 0.2), ValueError, "must not contain duplicates"),
    ],
)
def test_strength_declarations_reject_every_malformed_form(
    role: str, values: Any, error_type: type[Exception], message: str
) -> None:
    """Pin each ``_numeric_strengths`` branch, and pin the role named in each message.

    ``bool`` is a ``Real``, so a boolean strength would otherwise pass as 0.0 or 1.0.
    """
    declaration = {"treatment": (0.0,), "outcome": (0.0,), role: values}
    with pytest.raises(error_type, match=f"{role} strengths .*{message}"):
        ConfounderStrengthGrid(**declaration)


def test_the_grid_declaration_must_be_the_exact_public_type(gaussian_result: Any) -> None:
    """A subclass is refused, so no near-miss declaration reaches a refit."""

    class _NearMissGrid(ConfounderStrengthGrid):
        pass

    with pytest.raises(TypeError, match="exact ConfounderStrengthGrid declaration"):
        simulated_confounding(gaussian_result, grid=_NearMissGrid(treatment=(0.0,), outcome=(0.0,)))
    with pytest.raises(TypeError, match="exact ConfounderStrengthGrid declaration"):
        simulated_confounding(
            gaussian_result,
            grid=SimpleNamespace(treatment=(0.0,), outcome=(0.0,)),  # type: ignore[arg-type]
        )


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


def _treated_fraction_fit(fraction: float, *, n: int = 800, seed: int = 5) -> Any:
    """Fit one design whose treated fraction is chosen, not inherited from a generator.

    ``make_linear_ate`` draws a near-balanced treatment, and the induced association is a
    function of the treated fraction, so the contrast needs an arm the shipped generators
    do not produce. The size is 800 rows because a balanced design reports a sample
    correlation around zero, and its noise floor at 120 rows overlaps the imbalanced
    signal.
    """
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    treatment = (rng.random(n) < fraction).astype(float)
    outcome = 0.5 * covariate + treatment + rng.normal(size=n)
    frame = pd.DataFrame({"W1": covariate, "A": treatment, "Y": outcome})
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1",)),
    )
    return study.identify(ATE()).estimate(
        method="tmle",
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        simultaneous=False,
    )


@pytest.fixture(scope="module")
def balanced_result() -> Any:
    return _treated_fraction_fit(0.5)


@pytest.fixture(scope="module")
def low_treated_result() -> Any:
    return _treated_fraction_fit(0.2)


@pytest.fixture(scope="module")
def high_treated_result() -> Any:
    return _treated_fraction_fit(0.8)


def _association_grid() -> ConfounderStrengthGrid:
    return ConfounderStrengthGrid(treatment=(0.0, 0.1, 0.3), outcome=(0.0,))


def test_a_balanced_design_reports_no_induced_treatment_association(
    balanced_result: Any, low_treated_result: Any
) -> None:
    """Contrast a balanced arm against an imbalanced arm on one latent draw.

    The treatment flip is non-differential misclassification. It flips a treated row and
    an untreated row in the same latent tail, so the association it induces depends on
    the treated fraction. A balanced design therefore reports no association, and its
    movement along the treatment axis is misclassification and not confounding.
    """
    grid = _association_grid()
    balanced = simulated_confounding(balanced_result, grid=grid, random_state=21)
    imbalanced = simulated_confounding(low_treated_result, grid=grid, random_state=21)

    assert balanced_result.data.treated_fraction == pytest.approx(0.49375)
    assert low_treated_result.data.treated_fraction == pytest.approx(0.2025)
    assert balanced.complete and imbalanced.complete
    assert balanced.latent_seed == imbalanced.latent_seed

    # Measured values, not a direction. At a treated fraction of 0.2 the population
    # correlation is +0.240 at strength 0.1 and +0.430 at strength 0.3, which the
    # technical reference derives. A balanced design has a population correlation of
    # zero, so what it reports here is the sample noise floor of 800 rows.
    assert [cell.induced_treatment_association for cell in balanced.cells] == [
        pytest.approx(0.0421, abs=5e-4),
        pytest.approx(0.0734, abs=5e-4),
        pytest.approx(0.0267, abs=5e-4),
    ]
    assert [cell.induced_treatment_association for cell in imbalanced.cells] == [
        pytest.approx(0.0486, abs=5e-4),
        pytest.approx(0.3243, abs=5e-4),
        pytest.approx(0.4489, abs=5e-4),
    ]

    # The honest bounds around those numbers. Over 200 seeds a balanced design of 800
    # rows never passed 0.093, and the imbalanced value at strength 0.3 never fell
    # below 0.30.
    for cell in balanced.cells[1:]:
        assert cell.induced_treatment_association is not None
        assert abs(cell.induced_treatment_association) < 0.12
    assert abs(imbalanced.cells[1].induced_treatment_association or 0.0) > 0.18
    assert abs(imbalanced.cells[2].induced_treatment_association or 0.0) > 0.30

    # Both designs move along the treatment axis. The association is what separates
    # misclassification from confounding, and the movement alone does not.
    assert abs(balanced.cells[2].displacement or 0.0) > 0.02
    assert abs(imbalanced.cells[2].displacement or 0.0) > 0.02


def test_the_anchor_cell_reports_the_unperturbed_baseline(low_treated_result: Any) -> None:
    """The anchor reports ``corr(latent, A)`` on the original treatment.

    The anchor cell perturbs nothing, so its value is the null level of the same data
    under the same latent draw. A reader compares every other cell against it.
    """
    surface = simulated_confounding(low_treated_result, grid=_association_grid(), random_state=21)
    latent = np.random.default_rng(surface.latent_seed).normal(size=low_treated_result.data.n)
    expected = float(np.corrcoef(latent, low_treated_result.data.treatment)[0, 1])

    assert surface.cells[0].treatment_strength == 0.0
    assert surface.cells[0].induced_treatment_association == pytest.approx(expected, rel=1e-12)
    assert _treatment_association(latent, low_treated_result.data.treatment) == expected

    # Non-vacuity. The treatment is drawn independently of the latent vector, so the
    # baseline is near zero and every perturbed cell sits far above it.
    assert abs(expected) < 0.12
    for cell in surface.cells[1:]:
        assert abs(cell.induced_treatment_association or 0.0) > abs(expected) + 0.2


def test_the_induced_association_reverses_sign_across_a_treated_fraction_of_one_half(
    low_treated_result: Any, high_treated_result: Any
) -> None:
    """Above a treated fraction of one half the flip induces the opposite association.

    The same tail moves more treated rows to control than control rows to treated once
    the treated arm is the larger arm. The magnitude stays near symmetric around one
    half.
    """
    grid = _association_grid()
    low = simulated_confounding(low_treated_result, grid=grid, random_state=21)
    high = simulated_confounding(high_treated_result, grid=grid, random_state=21)

    assert low_treated_result.data.treated_fraction == pytest.approx(0.2025)
    assert high_treated_result.data.treated_fraction == pytest.approx(0.79875)
    assert low.cells[1].induced_treatment_association == pytest.approx(0.3243, abs=5e-4)
    assert high.cells[1].induced_treatment_association == pytest.approx(-0.2593, abs=5e-4)
    assert low.cells[2].induced_treatment_association == pytest.approx(0.4489, abs=5e-4)
    assert high.cells[2].induced_treatment_association == pytest.approx(-0.4288, abs=5e-4)

    for lower, upper in zip(low.cells[1:], high.cells[1:], strict=True):
        assert (lower.induced_treatment_association or 0.0) > 0.15
        assert (upper.induced_treatment_association or 0.0) < -0.15

    # Both anchors carry the same near-zero baseline, so the reversal belongs to the
    # perturbation and not to the two data sets.
    assert abs(low.cells[0].induced_treatment_association or 0.0) < 0.12
    assert abs(high.cells[0].induced_treatment_association or 0.0) < 0.12


def test_the_frame_and_the_summary_carry_the_induced_association(
    low_treated_result: Any,
) -> None:
    surface = simulated_confounding(low_treated_result, grid=_association_grid(), random_state=21)
    frame = surface.to_frame()
    text = surface.summary()

    assert list(frame.columns) == [
        "treatment_strength",
        "outcome_strength",
        "movement_scale",
        "estimate",
        "displacement",
        "induced_treatment_association",
        "error_type",
        "message",
    ]
    assert list(frame["induced_treatment_association"]) == [
        pytest.approx(cell.induced_treatment_association) for cell in surface.cells
    ]
    assert frame["induced_treatment_association"][2] == pytest.approx(0.4489, abs=5e-4)

    assert "induced association" in text
    assert surface.movement_scale == "estimate_difference"
    assert "movement (estimate difference)" in text
    assert "+0.4489" in text
    assert all(f"{cell.induced_treatment_association:+.4f}" in text for cell in surface.cells)
    assert "misclassification and not by confounding" in text
    assert "qualitative" in text

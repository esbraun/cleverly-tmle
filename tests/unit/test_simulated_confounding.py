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
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from cleverly import (
    ATE,
    ATT,
    AssessmentStatus,
    CausalStudy,
    CounterfactualMean,
    DRTMLEMethod,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    OddsRatio,
    PointTreatment,
    RiskRatio,
)
from cleverly.datasets import make_binary_outcome, make_linear_ate, make_shift_dose
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.ctmle import _LOSS_EPS, _Selector
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Shift
from cleverly.learners.crossfit import make_folds
from cleverly.sensitivity import (
    ConfounderStrengthGrid,
    SimulatedConfoundingResult,
    simulated_confounding,
)
from cleverly.sensitivity import simulated_confounding as public_function
from cleverly.sensitivity._simulated_confounding_request import (
    _BINARY_PARAMETER_TARGETS,
    _TMLE_ONLY_TARGETS,
)
from cleverly.sensitivity.simulated_confounding import (
    _binary_calibration,
    _continuous_calibration,
    _flip_binary,
    _flip_mask,
    _gaussian_outcome,
    _latent_child_seed,
    _linear_treatment,
    _perturb_treatment,
    _weighted_correlation,
    _weighted_std,
)
from cleverly.utils import resolve_g_bounds
from tests.conftest import assert_scale_normalizes_away, mean_one_weights, unweight
from tests.unit._confounding_support import (
    _collaborative_method,
    alias_for,
)
from tests.unit._confounding_support import (
    with_functional as _with_functional,
)


def _fit(
    *,
    family: str = "gaussian",
    backend: str = "pandas",
    seed: int = 7,
    method: str = "tmle",
    repeats: int = 1,
    weight_scale: float | None = None,
    weights_estimated: bool = False,
    constant_weights: bool = False,
    weight_spread: tuple[float, float] = (0.5, 1.5),
    collaborative_kwargs: dict[str, Any] | None = None,
) -> Any:
    if family == "gaussian":
        frame, _ = make_linear_ate(n=120, seed=seed, backend=backend)
        covariates = ("W1", "W2", "W3", "W4")
        outcome_learner = LinearRegression()
    else:
        frame, _ = make_binary_outcome(n=120, seed=seed, backend=backend)
        covariates = ("W1", "W2", "W3")
        outcome_learner = LogisticRegression(max_iter=1000)
    weight_name = None
    if weight_scale is not None:
        weight_name = "weight"
        relative_weights = (
            np.ones(len(frame)) if constant_weights else mean_one_weights(len(frame), weight_spread)
        )
        frame[weight_name] = weight_scale * relative_weights
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=covariates,
            weights=weight_name,
            weights_estimated=weights_estimated,
        ),
    )
    configured_method: Any = method
    if method == "collaborative_tmle":
        configured_method = _collaborative_method(overrides=collaborative_kwargs)
    elif method == "drtmle":
        configured_method = DRTMLEMethod(
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
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
    weight_scale: float | None = None,
    collaborative_kwargs: dict[str, Any] | None = None,
) -> Any:
    if family == "binomial":
        frame, _ = make_binary_outcome(n=120, seed=seed)
        covariates = ("W1", "W2", "W3")
        outcome_learner: Any = LogisticRegression(max_iter=1000)
    else:
        frame, _ = make_linear_ate(n=120, seed=seed)
        covariates = ("W1", "W2", "W3", "W4")
        outcome_learner = LinearRegression()
    weight_name = None
    if weight_scale is not None:
        weight_name = "weight"
        frame[weight_name] = weight_scale * mean_one_weights(len(frame))
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=covariates,
            weights=weight_name,
        ),
    )
    configured_method: Any = method
    if method == "collaborative_tmle":
        configured_method = _collaborative_method(
            selection_estimand=(
                "ey" if treatment is None else "ey1" if treatment == 1.0 else "ey0"
            ),
            overrides=collaborative_kwargs,
        )
    elif method == "drtmle":
        configured_method = DRTMLEMethod(
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
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
    weight_scale: float | None = None,
    collaborative_kwargs: dict[str, Any] | None = None,
) -> Any:
    frame, _ = make_binary_outcome(n=120, seed=seed)
    weight_name = None
    if weight_scale is not None:
        weight_name = "weight"
        frame[weight_name] = weight_scale * mean_one_weights(len(frame))
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            weights=weight_name,
        ),
    )
    configured_method: Any = method
    if method == "collaborative_tmle":
        configured_method = _collaborative_method(
            selection_estimand=target,
            overrides=collaborative_kwargs,
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
    weight_scale: float | None = None,
) -> Any:
    frame, _ = make_shift_dose(n=120, seed=seed)
    if family == "binomial":
        frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(float)
        outcome_learner: Any = LogisticRegression(max_iter=1000)
    else:
        outcome_learner = LinearRegression()
    weight_name = None
    if weight_scale is not None:
        weight_name = "weight"
        frame[weight_name] = weight_scale * mean_one_weights(len(frame))
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
            weights=weight_name,
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
    return alias_for(result, estimate_prefix="ate_shift[")


def _mean_alias(result: Any, policy: str = "up half") -> str:
    return alias_for(result, "ey_shift", value=policy)


def _arm_means(data: Any) -> tuple[float, float]:
    """``(treated, control)`` outcome means of a replacement sample."""
    return (
        float(np.mean(data.outcome[data.treatment == 1.0])),
        float(np.mean(data.outcome[data.treatment == 0.0])),
    )


def _odds_ratio(data: Any) -> float:
    treated, control = _arm_means(data)
    return (treated / (1.0 - treated)) / (control / (1.0 - control))


#: The stub ``psi`` a recorded refit reports, keyed by parameter shape. Each is a closed
#: form of the replacement sample alone, so a cell's number identifies the sample it was
#: refitted on and no learner runs.
_STUB_PSI: dict[str, Any] = {
    "difference": lambda data: float(np.subtract(*_arm_means(data))),
    "rr": lambda data: float(np.divide(*_arm_means(data))),
    "or": _odds_ratio,
    "shift": lambda data: float(np.mean(data.treatment * data.outcome)),
}


def _record_refits(
    result: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    psi: str = "difference",
    alias: str | None = None,
    fail_call: int | None = None,
) -> list[tuple[Any, int | None]]:
    """Replace ``estimator.refit`` with a stub, and record the sample each cell asked for.

    The surface's own bookkeeping is what these tests are about: which replacement data a
    cell built, which seed it refitted under, and what a raised failure does to the
    surface. A real refit answers none of that and costs a fit per cell, so the stub reads
    ``psi`` straight off the replacement sample.

    ``psi`` names the closed form to report, and it supplies the default ``alias`` as well.
    A ratio form is its own alias, ``"shift"`` resolves the fit's single shift alias, and
    ``"difference"`` is ``"ate"``. A ratio estimate also carries ``log_psi``, which is the
    field the surface reads on the log scale, so the stub keeps the two consistent.
    """
    form = _STUB_PSI[psi]
    resolved = alias
    if resolved is None:
        resolved = _shift_alias(result) if psi == "shift" else "ate" if psi == "difference" else psi
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
        original = result[resolved]
        value = form(data)
        changes: dict[str, Any] = {"psi": value}
        if original.scale == "ratio":
            changes["log_psi"] = float(np.log(value))
        return replace(result, data=data, estimates={resolved: replace(original, **changes)})

    monkeypatch.setattr(result.estimator, "refit", refit)
    return calls


def _grid() -> ConfounderStrengthGrid:
    return ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))


#: One cell, the anchor. ``ConfounderStrengthGrid`` asks only that ``0.0`` appears in each
#: tuple, and the anchor cell reuses the fitted estimate rather than running a refit. A test
#: that reads no cell and refits by hand therefore pays for the fit and for its own refits
#: only.
_ANCHOR_GRID = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))


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
    assert "cross-fitting: 3 draws, aggregation=coordinatewise_median" in surface.summary()

    # The un-monkeypatched path. The anchor is the original fit itself, so it carries the
    # fitted point estimate rather than a refit of unchanged data.
    assert surface.original_estimate == result["ate"].psi
    assert surface.cells[0].estimate == result["ate"].psi
    assert surface.cells[0].displacement == 0.0

    # A flip of the upper 10% latent tail moves this fixture by -0.7528 on tmle, -0.6835 on
    # collaborative_tmle, and -0.6211 on drtmle, all at repeats=3. The surface seed differs
    # from the seed of the fit, so -0.1050, -0.0140, and -0.0481 of those three are fold
    # noise rather than the flip. The gate sits far below the remainder and far above
    # numerical noise, and it is signed, so a perturbation that never reaches the refit
    # fails it.
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
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))
    surface = simulated_confounding(result, grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.0)
    # ``product(treatment, outcome)`` orders the cells (0, 0), (0, 0.2), (0.1, 0), (0.1, 0.2).
    cell = surface.cells[2]
    per_draw = [repeat.psi["ate"] for repeat in manual.repeats]

    assert cell.estimate == manual["ate"].psi == float(np.median(per_draw))
    assert cell.displacement == manual["ate"].inference_value - result["ate"].inference_value
    # The surface reports the middle of three distinct draws. Stating the rank keeps the
    # witness true by construction: ``cell.estimate != per_draw[0]`` holds only when draw
    # zero is not the middle one, which is a property of the seed and not of the surface.
    assert len(set(per_draw)) == 3
    assert sorted(per_draw).index(cell.estimate) == 1

    # The outcome axis runs ``Y' = Y - k_Y U``. Without a nonzero outcome strength every
    # repeat test would exercise ``_gaussian_outcome`` as an identity transform only.
    outcome_manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.2)
    outcome_cell = surface.cells[3]
    outcome_per_draw = [repeat.psi["ate"] for repeat in outcome_manual.repeats]

    assert outcome_cell.outcome_strength == 0.2
    assert outcome_cell.estimate == outcome_manual["ate"].psi == float(np.median(outcome_per_draw))
    assert (
        outcome_cell.displacement
        == outcome_manual["ate"].inference_value - result["ate"].inference_value
    )
    assert len(set(outcome_per_draw)) == 3
    assert sorted(outcome_per_draw).index(outcome_cell.estimate) == 1
    assert outcome_cell.estimate != cell.estimate

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
    # The surface reports the middle of three distinct draws, which stays true whatever
    # rank draw zero holds.
    assert len(set(per_draw)) == 3
    assert sorted(per_draw).index(cell.estimate) == 1


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
    # The surface reports the middle of three distinct draws, which stays true whatever
    # rank draw zero holds.
    assert len(set(per_draw)) == 3
    assert sorted(per_draw).index(cell.estimate) == 1


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


def test_fixed_weight_surface_repeats_cache_and_serialization_round_trip() -> None:
    result = _fit(repeats=3, weight_scale=4.0)
    kwargs = {
        # A nonzero outcome strength runs the weighted outcome replacement under repeats,
        # which the treatment axis alone never reaches.
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
        "benchmark_covariates": ("W1",),
        "random_state": 31,
    }
    surface = result.sensitivity.simulated_confounding(**kwargs)

    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    assert surface.target_measure == "fixed_empirical_tilt"
    assert surface.weight_report == result.data.weight_report()
    assert surface.weight_report.name == "weight"
    assert surface.n_repeats == 3

    # Without these the round trip would agree even if the restored fit refit no cell and
    # calibrated no covariate.
    assert surface.complete
    assert surface.repeat_aggregation == "coordinatewise_median"
    assert len(surface.cells) == 4
    assert all(
        cell.displacement is not None and abs(cell.displacement) > 1e-6
        for cell in surface.cells[1:]
    )
    # `product` places the outcome-only cell at index 1, so the grid does reach the
    # outcome axis under repeats. A displacement cannot witness that the replacement
    # changed the outcome, because this refit seed displaces every cell by fold noise on
    # its own. `test_fixed_weights_preserve_every_replacement_and_equal_a_manual_refit`
    # carries that witness, on the replacement the surface hands to the refit.
    assert surface.cells[1].treatment_strength == 0.0
    assert surface.cells[1].outcome_strength == 0.2
    calibrated = {row.role: row for row in surface.calibrations if row.covariate == "W1"}
    assert set(calibrated) == {"treatment", "outcome"}
    assert all(np.isfinite(row.strength) for row in calibrated.values())
    assert abs(calibrated["outcome"].strength) > 0.0

    restored = loads(dumps(result))
    replayed = restored.sensitivity.simulated_confounding(**kwargs)
    assert replayed == surface
    assert replayed.weight_report == restored.data.weight_report()
    assert replayed.target_measure == "fixed_empirical_tilt"


def test_fixed_weights_preserve_every_replacement_and_equal_a_manual_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit(weight_scale=2.0)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))
    surface = simulated_confounding(result, grid=grid, random_state=31)

    # Every refitted cell, not one of them. A single cell would leave the other two
    # replacements free to drop the weights, the strengths, or both.
    assert surface.complete
    assert len(surface.cells) == 4
    for cell in surface.cells[1:]:
        manual = _manual_repeated_refit(
            result,
            surface,
            treatment=cell.treatment_strength,
            outcome=cell.outcome_strength,
        )
        assert cell.estimate == manual["ate"].psi
        assert cell.displacement == (manual["ate"].inference_value - result["ate"].inference_value)
        # Non-vacuity. A manual refit that matched a surface which never moved would
        # agree at the original estimate.
        assert cell.displacement != 0.0

    calls = _record_refits(result, monkeypatch, fail_call=2)
    retained = simulated_confounding(result, grid=grid, random_state=31)
    assert len(calls) == 3
    assert not retained.complete
    assert retained.cells[2].failure is not None
    assert retained.cells[2].induced_treatment_association is not None
    for replacement, seed in calls:
        assert seed == retained.root_seed
        assert np.array_equal(replacement.weights, result.data.weights)
        assert replacement.weight_spec is result.data.weight_spec
        assert replacement.weights_name == result.data.weights_name == "weight"

    # Each axis reached the replacement the surface handed to the refit. `product` orders
    # the non-anchor cells outcome-only, treatment-only, then both, so each call below
    # must move its own variable and leave the other one alone. A displacement cannot
    # witness this, because the refit seed moves every cell by fold noise on its own.
    outcome_only, treatment_only, both = (replacement for replacement, _ in calls)
    assert not np.array_equal(outcome_only.outcome, result.data.outcome)
    assert np.array_equal(outcome_only.treatment, result.data.treatment)
    assert np.array_equal(treatment_only.outcome, result.data.outcome)
    assert not np.array_equal(treatment_only.treatment, result.data.treatment)
    assert not np.array_equal(both.outcome, result.data.outcome)
    assert not np.array_equal(both.treatment, result.data.treatment)


def test_a_fixed_weight_ratio_cell_equals_a_manual_refit_on_the_log_scale() -> None:
    """A second composition, so the manual-refit evidence is not one estimand and scale.

    A risk ratio reports its movement on the log scale. This pins the weighted cell
    against a manual refit and against the log rule the module applies.
    """
    result = _fit_ratio(target="rr", weight_scale=1.0)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))
    surface = simulated_confounding(result, estimand="rr", grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.0)
    cell = surface.cells[1]

    assert surface.target_measure == "fixed_empirical_tilt"
    assert surface.movement_scale == "log_ratio"
    assert cell.estimate == manual["rr"].psi
    assert cell.displacement == manual["rr"].inference_value - result["rr"].inference_value
    assert cell.displacement == pytest.approx(
        float(np.log(manual["rr"].psi) - np.log(result["rr"].psi)), rel=1e-12
    )
    # Non-vacuity. A cell that never refit would report the original ratio and no movement.
    assert cell.displacement != 0.0


def test_constant_declared_weights_preserve_unweighted_surface_arithmetic() -> None:
    unweighted = _fit()
    # This scale normalizes to values just above one on common NumPy builds. It
    # exercises the same tolerance CausalData uses to classify constant weights.
    constant = _fit(weight_scale=0.3, constant_weights=True)
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
        "benchmark_covariates": ("W1", "W2"),
        "random_state": 7,
    }
    plain_surface = simulated_confounding(unweighted, **kwargs)
    fixed_surface = simulated_confounding(constant, **kwargs)

    assert unweighted["ate"].psi == pytest.approx(constant["ate"].psi, abs=1e-12)
    assert unweighted["ate"].inference_value == pytest.approx(
        constant["ate"].inference_value, abs=1e-12
    )
    for fixed_cell, plain_cell in zip(fixed_surface.cells, plain_surface.cells, strict=True):
        assert fixed_cell.estimate == pytest.approx(plain_cell.estimate, abs=1e-12)
        assert fixed_cell.displacement == pytest.approx(plain_cell.displacement, abs=1e-12)
        assert fixed_cell.induced_treatment_association == plain_cell.induced_treatment_association
        assert fixed_cell.failure == plain_cell.failure
    assert fixed_surface.calibrations == plain_surface.calibrations
    assert plain_surface.target_measure == "unweighted"
    assert fixed_surface.target_measure == "fixed_empirical_tilt"
    assert not fixed_surface.weight_report.is_weighted
    assert "target measure: fixed_empirical_tilt" in fixed_surface.summary()


def test_fixed_weight_surface_is_invariant_to_a_common_weight_scale() -> None:
    """The scale is reported and normalised away, and the surface then repeats itself.

    :func:`assert_scale_normalizes_away` states what this comparison can and cannot show.
    The two fits store one array, so everything below runs on one set of numbers. What is
    left is that the surface carries the scale into its own report and that no cell reads
    the raw column.
    """
    unit_scale, larger_scale = assert_scale_normalizes_away(lambda scale: _fit(weight_scale=scale))
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
        "benchmark_covariates": ("W1",),
        "random_state": 7,
    }
    left = simulated_confounding(unit_scale, **kwargs)
    right = simulated_confounding(larger_scale, **kwargs)

    assert left.target_measure == right.target_measure == "fixed_empirical_tilt"
    assert left.weight_report.scale == pytest.approx(1.0)
    assert right.weight_report.scale == pytest.approx(13.0)
    assert left.weight_report.effective_n == pytest.approx(right.weight_report.effective_n)
    for left_cell, right_cell in zip(left.cells, right.cells, strict=True):
        assert left_cell.estimate == pytest.approx(right_cell.estimate, abs=1e-12)
        assert left_cell.displacement == pytest.approx(right_cell.displacement, abs=1e-12)
        assert left_cell.induced_treatment_association == pytest.approx(
            right_cell.induced_treatment_association, abs=1e-12
        )
    for left_row, right_row in zip(left.calibrations, right.calibrations, strict=True):
        assert left_row.strength == pytest.approx(right_row.strength, abs=1e-12)


def test_nonuniform_weights_change_a_weight_dependent_surface() -> None:
    plain = simulated_confounding(
        _fit(),
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
        random_state=7,
    )
    tilted = simulated_confounding(
        _fit(weight_scale=1.0),
        grid=plain.grid,
        random_state=7,
    )

    assert tilted.original_estimate != pytest.approx(plain.original_estimate, abs=1e-5)
    # Cell zero is the anchor. It runs no refit and reports the original estimate the line
    # above already separates, so it cannot witness that the weights reached a refit. Only
    # the refitted cells carry that evidence, and all three of them move.
    assert any(
        weighted.estimate != pytest.approx(unweighted.estimate, abs=1e-5)
        for unweighted, weighted in zip(plain.cells[1:], tilted.cells[1:], strict=True)
    )


def test_fixed_weight_tmle_control_detects_dropped_nuisance_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refitted ordinary-TMLE nuisances must read the row mass.

    The two tests above separate a weighted surface from an unweighted one by fitting both
    and comparing them, which is evidence that the weights reached *something*. It is not
    evidence about where. Two fits differ in their data as well as in their arithmetic, so
    a fit that read the mass in one place and dropped it in another passes that comparison.
    The C-TMLE and DR-TMLE surfaces each carry a mutation control for this reason, and this
    is the ordinary-TMLE one: it holds the fit fixed and unweights
    :meth:`TMLE._nuisances`, which is where the outcome regression and the mechanism are
    fitted, leaving the targeting step and the plug-in weighted.

    The gate is relative, as the C-TMLE selector control's gate is. The measured movement on
    this fixture is 0.04285 on a psi of 1.15248, which is 3.7 percent.
    """
    result = _fit(weight_scale=1.0)
    surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=31)
    baseline = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)

    unweight(monkeypatch, TMLE, "_nuisances")
    dropped = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)

    assert type(result.estimator) is TMLE
    assert not np.allclose(result.data.weights, 1.0)
    assert abs(baseline["ate"].psi - dropped["ate"].psi) > 1e-3 * abs(baseline["ate"].psi)


def test_unweighted_estimated_weight_flag_does_not_create_a_weight_refusal() -> None:
    result = _fit(weights_estimated=True)
    surface = simulated_confounding(result, grid=_grid(), random_state=7)

    assert result.data.weights_name is None
    assert surface.target_measure == "unweighted"
    assert surface.complete


def test_fixed_weights_run_every_supported_ordinary_tmle_parameter_surface() -> None:
    multiple_means = _fit_binary_mean(treatment=None, weight_scale=1.0)
    cases = [
        (_fit(weight_scale=1.0), ("ate",)),
        (_fit_binary_mean(treatment=1.0, weight_scale=1.0), ("ey1",)),
        (_fit_binary_mean(treatment=0.0, weight_scale=1.0), ("ey0",)),
        (multiple_means, tuple(multiple_means.estimates)),
        (_fit_ratio(target="rr", weight_scale=1.0), ("rr",)),
        (_fit_ratio(target="or", weight_scale=1.0), ("or",)),
    ]
    continuous_contrast = _fit_continuous(weight_scale=1.0)
    continuous_mean = _fit_continuous(weight_scale=1.0, means=True)
    cases.extend(
        [
            (continuous_contrast, (_shift_alias(continuous_contrast),)),
            (continuous_mean, (_mean_alias(continuous_mean),)),
        ]
    )

    exercised: set[str] = set()
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))
    for result, aliases in cases:
        for alias in aliases:
            surface = simulated_confounding(result, estimand=alias, grid=grid, random_state=7)
            assert surface.complete
            assert surface.target_measure == "fixed_empirical_tilt"
            assert surface.weight_report == result.data.weight_report()
            assert surface.cells[0].estimate == result[alias].psi
            assert surface.cells[1].estimate is not None
            exercised.add(result.parameter_keys[alias].estimand)

    assert exercised == {"ate", "ey", "ey1", "ey0", "rr", "or", "ey_shift", "ate_shift"}


def test_fixed_weights_run_every_supported_binary_drtmle_parameter_surface() -> None:
    """Exercise each admitted parameter through a real weighted DR-TMLE refit."""
    multiple_means = _fit_binary_mean(
        treatment=None,
        method="drtmle",
        weight_scale=1.0,
    )
    cases = [
        (_fit(method="drtmle", weight_scale=1.0), ("ate",)),
        (
            _fit_binary_mean(treatment=1.0, method="drtmle", weight_scale=1.0),
            ("ey1",),
        ),
        (
            _fit_binary_mean(treatment=0.0, method="drtmle", weight_scale=1.0),
            ("ey0",),
        ),
        (multiple_means, tuple(multiple_means.estimates)),
        (_fit_ratio(target="rr", method="drtmle", weight_scale=1.0), ("rr",)),
        (_fit_ratio(target="or", method="drtmle", weight_scale=1.0), ("or",)),
    ]
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0, 0.2))
    exercised: set[str] = set()

    for result, aliases in cases:
        assert type(result.estimator).__name__ == "DRTMLE"
        for alias in aliases:
            surface = simulated_confounding(result, estimand=alias, grid=grid, random_state=7)
            assert surface.complete
            assert surface.target_measure == "fixed_empirical_tilt"
            assert surface.weight_report == result.data.weight_report()
            assert surface.cells[0].estimate == result[alias].psi
            assert any(abs(cell.displacement or 0.0) > 1e-6 for cell in surface.cells[1:])
            exercised.add(result.parameter_keys[alias].estimand)

    assert exercised == set(_BINARY_PARAMETER_TARGETS) - _TMLE_ONLY_TARGETS


def test_fixed_weight_drtmle_additive_cell_equals_a_manual_complete_refit() -> None:
    result = _fit(method="drtmle", weight_scale=2.0)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))
    surface = simulated_confounding(result, grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.2)
    cell = surface.cells[3]

    assert surface.target_measure == "fixed_empirical_tilt"
    assert cell.estimate == manual["ate"].psi
    assert cell.displacement == manual["ate"].inference_value - result["ate"].inference_value
    assert cell.displacement != 0.0


def test_fixed_weight_drtmle_ratio_cell_equals_a_manual_log_refit() -> None:
    result = _fit_ratio(target="rr", method="drtmle", weight_scale=2.0)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))
    surface = simulated_confounding(result, estimand="rr", grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.2)
    cell = surface.cells[3]

    assert surface.movement_scale == "log_ratio"
    assert cell.estimate == manual["rr"].psi
    assert cell.displacement == manual["rr"].inference_value - result["rr"].inference_value
    assert cell.displacement == pytest.approx(
        float(np.log(manual["rr"].psi) - np.log(result["rr"].psi)),
        rel=1e-12,
    )
    assert cell.displacement != pytest.approx(cell.estimate - result["rr"].psi, abs=1e-3)


def test_fixed_weight_drtmle_preserves_weight_provenance_on_every_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit(method="drtmle", weight_scale=2.0)
    calls = _record_refits(result, monkeypatch)
    surface = simulated_confounding(result, grid=_grid(), random_state=31)

    assert surface.complete
    assert len(calls) == 3
    for replacement, seed in calls:
        assert seed == surface.root_seed
        assert np.array_equal(replacement.weights, result.data.weights)
        assert replacement.weight_spec is result.data.weight_spec
        assert replacement.weights_name == result.data.weights_name == "weight"

    outcome_only, treatment_only, both = (replacement for replacement, _ in calls)
    assert not np.array_equal(outcome_only.outcome, result.data.outcome)
    assert np.array_equal(outcome_only.treatment, result.data.treatment)
    assert np.array_equal(treatment_only.outcome, result.data.outcome)
    assert not np.array_equal(treatment_only.treatment, result.data.treatment)
    assert not np.array_equal(both.outcome, result.data.outcome)
    assert not np.array_equal(both.treatment, result.data.treatment)


def test_fixed_weight_drtmle_repeat_median_cache_and_serialization_round_trip() -> None:
    result = _fit(method="drtmle", repeats=3, weight_scale=4.0)
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,)),
        "random_state": 31,
    }
    surface = result.sensitivity.simulated_confounding(**kwargs)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.0)
    per_draw = [repeat.psi["ate"] for repeat in manual.repeats]

    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    assert surface.complete
    assert surface.n_repeats == 3
    assert surface.repeat_aggregation == "coordinatewise_median"
    assert surface.cells[1].estimate == manual["ate"].psi == float(np.median(per_draw))
    assert len(set(per_draw)) == 3
    assert sorted(per_draw).index(surface.cells[1].estimate) == 1

    # The facade memoizes the surface into ``assessment_cache``, and ``dumps`` carries that
    # cache with it.  Reading it back through the facade would compare the restored object
    # with itself and refit nothing, so the free function runs the replay instead.
    restored = loads(dumps(result))
    assert np.array_equal(restored.data.weights, result.data.weights)
    replayed = simulated_confounding(restored, **kwargs)
    assert replayed == surface
    assert replayed.weight_report == restored.data.weight_report()


def test_fixed_weight_drtmle_surface_is_invariant_to_a_common_weight_scale() -> None:
    """The DR-TMLE reading of the invariance the ordinary-TMLE test above states."""
    unit_scale, larger_scale = assert_scale_normalizes_away(
        lambda scale: _fit(method="drtmle", weight_scale=scale)
    )
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
        "random_state": 7,
    }
    left = simulated_confounding(unit_scale, **kwargs)
    right = simulated_confounding(larger_scale, **kwargs)

    assert left.weight_report.scale == pytest.approx(1.0)
    assert right.weight_report.scale == pytest.approx(13.0)
    for left_cell, right_cell in zip(left.cells, right.cells, strict=True):
        assert left_cell.estimate == pytest.approx(right_cell.estimate, abs=1e-12)
        assert left_cell.displacement == pytest.approx(right_cell.displacement, abs=1e-12)
        assert left_cell.induced_treatment_association == pytest.approx(
            right_cell.induced_treatment_association,
            abs=1e-12,
        )


def test_fixed_weight_drtmle_controls_detect_dropped_weights_and_tmle_fallback() -> None:
    """Separate the admitted fit from both tempting but incorrect implementations.

    Both gates are relative, as the C-TMLE controls' gates are, because an absolute one
    hides how large the movement is next to the estimate it is a movement in. The measured
    movements on this fixture are 0.04867 for the dropped weights and 0.24506 for the
    ordinary-TMLE fallback, on a witness estimate of 0.90742. That is 5.4 percent and 27
    percent, both about two orders above the gate.
    """
    weighted_drtmle = _fit(method="drtmle", weight_scale=1.0)
    surface = simulated_confounding(
        weighted_drtmle,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.3)),
        random_state=31,
    )
    witness = surface.cells[3]
    dropped_weights = _manual_repeated_refit(
        _fit(method="drtmle"),
        surface,
        treatment=0.2,
        outcome=0.3,
    )
    ordinary_tmle = _manual_repeated_refit(
        _fit(weight_scale=1.0),
        surface,
        treatment=0.2,
        outcome=0.3,
    )

    assert witness.estimate is not None
    gate = 1e-3 * abs(witness.estimate)
    assert abs(witness.estimate - dropped_weights["ate"].psi) > gate
    assert abs(witness.estimate - ordinary_tmle["ate"].psi) > gate


def test_fixed_weight_drtmle_witnesses_the_weight_on_the_reduced_regressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove the weight from the reductions alone, and the cell estimate has to move.

    The dropped-weight control removes the weight from every learner at once, so the
    primary outcome regression and mechanism dominate the separation it reports. The
    transport claim this surface rests on is narrower. It is a claim about which
    conditional expectations :math:`Q_r`, :math:`g_{r,1}` and :math:`g_{r,2}` are, and a
    fit that weighted its primary nuisances while fitting the reductions at the sampling
    law would satisfy every other control here.

    The gate is relative, as the control above it is. The measured movement on this fixture
    is 0.01292 on a witness estimate of 0.90742, which is 1.4 percent.
    """
    result = _fit(method="drtmle", weight_scale=1.0)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.3))
    surface = simulated_confounding(result, grid=grid, random_state=31)
    witness = surface.cells[3]

    # ``surface`` is already materialized, so patching the class now reaches the manual
    # refit below and nothing else.  Reusing ``result`` rather than a second fit is what
    # keeps the comparison exact: a strict inequality against an independent fit would
    # pass vacuously if the two fits ever diverged.
    unweight(monkeypatch, DRTMLE, "_fit_reduced")
    mutated = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)

    assert witness.estimate is not None
    assert abs(witness.estimate - mutated["ate"].psi) > 1e-3 * abs(witness.estimate)


def test_fixed_weights_run_every_supported_binary_ctmle_parameter_surface() -> None:
    """Exercise each admitted parameter through a real weighted C-TMLE refit."""
    multiple_means = _fit_binary_mean(
        treatment=None,
        method="collaborative_tmle",
        weight_scale=1.0,
    )
    cases = [
        (_fit(method="collaborative_tmle", weight_scale=1.0), ("ate",)),
        (
            _fit_binary_mean(
                treatment=1.0,
                method="collaborative_tmle",
                weight_scale=1.0,
            ),
            ("ey1",),
        ),
        (
            _fit_binary_mean(
                treatment=0.0,
                method="collaborative_tmle",
                weight_scale=1.0,
            ),
            ("ey0",),
        ),
        (multiple_means, tuple(multiple_means.estimates)),
        (
            _fit_ratio(target="rr", method="collaborative_tmle", weight_scale=1.0),
            ("rr",),
        ),
        (
            _fit_ratio(target="or", method="collaborative_tmle", weight_scale=1.0),
            ("or",),
        ),
    ]
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0, 0.2))
    exercised: set[str] = set()

    for result, aliases in cases:
        assert type(result.estimator) is CTMLE
        for alias in aliases:
            surface = simulated_confounding(result, estimand=alias, grid=grid, random_state=7)
            assert surface.complete
            assert surface.target_measure == "fixed_empirical_tilt"
            assert surface.weight_report == result.data.weight_report()
            assert surface.cells[0].estimate == result[alias].psi
            assert any(abs(cell.displacement or 0.0) > 1e-6 for cell in surface.cells[1:])
            exercised.add(result.parameter_keys[alias].estimand)

    assert exercised == set(_BINARY_PARAMETER_TARGETS) - _TMLE_ONLY_TARGETS


@pytest.mark.parametrize(
    ("collaborative_kwargs", "strategy", "preorder"),
    [
        ({"strategy": "greedy"}, "greedy", None),
        ({"strategy": "ordered", "preorder": "logistic"}, "ordered", "logistic"),
        (
            {"strategy": "ordered", "preorder": "partial_correlation"},
            "ordered",
            "partial_correlation",
        ),
        (
            {"strategy": "ordered", "ordering": ("W4", "W3", "W2", "W1")},
            "ordered",
            "custom",
        ),
        (
            {
                "strategy": "discrete",
                "candidates": ((), ("W1",), ("W1", "W2", "W3", "W4")),
            },
            "discrete",
            None,
        ),
        ({"strategy": "oat"}, "oat", None),
    ],
)
def test_fixed_weights_run_every_binary_ctmle_strategy_surface(
    collaborative_kwargs: dict[str, Any],
    strategy: str,
    preorder: str | None,
) -> None:
    result = _fit(
        method="collaborative_tmle",
        weight_scale=1.0,
        collaborative_kwargs=collaborative_kwargs,
    )
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.15), outcome=(0.0,)),
        random_state=17,
    )
    diagnostic = result.extra["ctmle"]

    assert surface.complete
    assert surface.target_measure == "fixed_empirical_tilt"
    assert surface.cells[1].estimate is not None
    assert result.method.strategy == result.estimator.strategy == strategy
    assert diagnostic.strategy == strategy
    if strategy != "oat":
        assert diagnostic.preorder == preorder
    if collaborative_kwargs.get("ordering") is not None:
        assert result.estimator.ordering == collaborative_kwargs["ordering"]
    if collaborative_kwargs.get("candidates") is not None:
        assert result.estimator.candidates == collaborative_kwargs["candidates"]


def test_fixed_weight_ctmle_additive_cell_equals_a_manual_complete_refit() -> None:
    result = _fit(method="collaborative_tmle", weight_scale=2.0)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))
    surface = simulated_confounding(result, grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.2)
    cell = surface.cells[3]

    assert surface.target_measure == "fixed_empirical_tilt"
    assert cell.estimate == manual["ate"].psi
    assert cell.displacement == manual["ate"].inference_value - result["ate"].inference_value
    assert cell.displacement != 0.0


def test_fixed_weight_ctmle_ratio_cell_equals_a_manual_log_refit() -> None:
    result = _fit_ratio(target="rr", method="collaborative_tmle", weight_scale=2.0)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2))
    surface = simulated_confounding(result, estimand="rr", grid=grid, random_state=31)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.2)
    cell = surface.cells[3]

    assert surface.movement_scale == "log_ratio"
    assert cell.estimate == manual["rr"].psi
    assert cell.displacement == manual["rr"].inference_value - result["rr"].inference_value
    assert cell.displacement == pytest.approx(
        float(np.log(manual["rr"].psi) - np.log(result["rr"].psi)),
        rel=1e-12,
    )
    assert cell.displacement != pytest.approx(cell.estimate - result["rr"].psi, abs=1e-3)


#: Unweighting the whole selector, from construction onward. Every selector method reads
#: ``self.data``, so replacing the mass in the constructor unweights the search, its
#: propensity fits, its targeting and its scoring together. Use it where the claim is about
#: the search's outcome rather than about which component produced it.
_EVERY_SELECTOR_WEIGHT = "__init__"


def _manual_ctmle_loss(
    selector: _Selector,
    targeted: Any,
    rows: Any,
    kind: str,
    *,
    weights: Any | None = None,
) -> float:
    """One loss of a targeted fit, without calling the selector's scoring methods.

    Both admitted kinds sum one per-row kernel over the eligible rows at the row mass, so
    the eligibility rule and the mass are written once and ``kind`` selects the kernel.
    ``"squared"`` is the gaussian branch of :meth:`_Selector.loss` and ``"loglik"`` is the
    binomial one, which a weighted ``rr`` or ``or`` surface runs.
    """
    eligible = np.asarray(rows)[selector.data.observed[rows]]
    mass = selector.data.weights[eligible] if weights is None else np.asarray(weights)[eligible]
    if kind == "squared":
        residual = selector.scaled[eligible] - targeted.observed[eligible]
        return float(np.sum(mass * residual**2))
    y = selector.scaled[eligible]
    q = np.clip(targeted.observed[eligible], _LOSS_EPS, 1.0 - _LOSS_EPS)
    return float(-np.sum(mass * (y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))


def _manual_ctmle_arm_mass(selector: _Selector, rows: Any, weights: Any | None) -> Any:
    """The row mass a manual curve carries, declared once for every manual curve."""
    index = np.asarray(rows)
    return selector.data.weights[index] if weights is None else np.asarray(weights)[index]


def _manual_ctmle_arm_curves(
    selector: _Selector,
    candidate: Any,
    rows: Any,
    *,
    weights: Any | None = None,
) -> dict[float, Any]:
    """Per-arm mean curves from the displayed formula, independent of ``influence``.

    Mirrors ``counterfactual_means`` in ``cleverly.inference.influence``: each arm gets
    ``mass * (clever * residual + prediction - psi)`` at its own weighted mean prediction.
    """
    index = np.asarray(rows)
    mass = _manual_ctmle_arm_mass(selector, rows, weights)
    residual = selector.scaled[index] - candidate.targeted.observed[index]
    curves: dict[float, Any] = {}
    for arm in selector.data.arm_codes:
        prediction = candidate.targeted.arms[arm][index]
        psi = float(np.average(prediction, weights=mass))
        clever = candidate.submodel.column_for(float(arm))[index]
        curves[float(arm)] = mass * (clever * residual + prediction - psi)
    return curves


def _manual_ctmle_ate_curve(
    selector: _Selector,
    candidate: Any,
    rows: Any,
    *,
    weights: Any | None = None,
) -> Any:
    """Binary ATE curve: the treated arm curve less the control arm curve."""
    curves = _manual_ctmle_arm_curves(selector, candidate, rows, weights=weights)
    return curves[1.0] - curves[0.0]


def _manual_ctmle_log_ratio_curve(
    selector: _Selector,
    candidate: Any,
    rows: Any,
    *,
    weights: Any | None = None,
) -> Any:
    """Log risk-ratio curve, mirroring ``log_ratio_influence`` in ``cleverly.inference.delta``.

    Divides each arm curve by that arm's weighted mean prediction, and contrasts the treated
    arm against the selector's own reference arm rather than against a hardcoded ``0.0``.
    """
    index = np.asarray(rows)
    mass = _manual_ctmle_arm_mass(selector, rows, weights)
    curves = _manual_ctmle_arm_curves(selector, candidate, rows, weights=weights)
    reference = float(selector.reference)
    psi_one = float(np.average(candidate.targeted.arms[1.0][index], weights=mass))
    psi_reference = float(np.average(candidate.targeted.arms[reference][index], weights=mass))
    return curves[1.0] / psi_one - curves[reference] / psi_reference


def _manual_curve_penalty(curve: Any) -> float:
    """Published variance-plus-bias penalty, without calling ``_penalty_of``."""
    matrix = np.asarray(curve, dtype=float).reshape(len(curve), -1)
    return float(
        np.sum(np.var(matrix, axis=0, ddof=1))
        + matrix.shape[0] * np.sum(np.mean(matrix, axis=0) ** 2)
    )


def _manual_ctmle_nested_risk(
    selector: _Selector,
    path: Any,
    folds: Any,
) -> Any:
    """Rebuild nested CV scoring without calling ``cross_validate`` or ``score``."""
    loss = np.zeros(len(path))
    curves = np.zeros((len(path), selector.data.n))
    for fold, (train, test) in enumerate(folds):
        train_folds, train_mask = selector._nested_folds(train)
        fold_base = selector._selection_base(train_folds, train_mask)
        fold_data = selector.data.subset(train)
        fold_bounds = resolve_g_bounds(
            selector.est.g_bounds,
            selector.est._bounds_n(fold_data),
            for_att=False,
        )
        fold_selector = _Selector(
            selector.est,
            selector.data,
            fold_base,
            fold_bounds,
            selector.intermediate_value,
            seed=selector.seed,
            train_folds=train_folds,
            train_mask=train_mask,
        )
        fold_path = fold_selector.build_path(train=train, tag=f"manual-cv{fold}")
        for position, candidate in enumerate(fold_path[: len(path)]):
            loss[position] += _manual_ctmle_loss(
                fold_selector,
                candidate.targeted,
                test,
                "squared",
            )
            curves[position, test] = _manual_ctmle_ate_curve(
                fold_selector,
                candidate,
                test,
            )
    return loss + np.array([_manual_curve_penalty(curve) for curve in curves])


def test_fixed_weight_ctmle_selector_components_recompute_from_the_refit() -> None:
    """Independently rebuild weighted loss, penalty, folds, and nested CV risk."""
    result = _fit(
        method="collaborative_tmle",
        weight_scale=1.0,
        collaborative_kwargs={"strategy": "discrete", "candidates": ((), ("W1",), ("W1", "W2"))},
    )
    surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=31)
    refit = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    selection = refit.extra["ctmle"]
    selector = _Selector(
        refit.estimator,
        refit.data,
        refit.nuisance,
        refit.config.g_bounds,
        refit.intermediate_value,
        seed=surface.root_seed,
    )
    rows = np.arange(refit.data.n)
    targeted = refit.nuisance.targeting_outcome
    assert targeted is not None
    submodel = selector.submodel(refit.nuisance.propensity)
    path = selector.build_path(train=None, tag="stored-refit")
    chosen = SimpleNamespace(targeted=targeted, submodel=submodel)
    manual_loss = _manual_ctmle_loss(selector, targeted, rows, "squared")
    manual_curve = _manual_ctmle_ate_curve(selector, chosen, rows)
    manual_penalty = _manual_curve_penalty(manual_curve)
    rebuilt_folds = make_folds(
        refit.data.n,
        refit.estimator.selection_folds,
        stratify=refit.estimator._fold_strata(refit.data),
        cluster=refit.data.cluster,
        random_state=surface.root_seed,
    )
    manual_cv_risk = _manual_ctmle_nested_risk(selector, path, rebuilt_folds)

    assert not np.allclose(refit.data.weights, 1.0)
    assert manual_loss == pytest.approx(selection.train_loss[selection.selected], rel=1e-12)
    assert manual_penalty == pytest.approx(selection.penalty[selection.selected], rel=1e-12)
    np.testing.assert_allclose(manual_cv_risk, selection.cv_risk, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(rebuilt_folds.assignment, selection.folds.assignment)


def test_fixed_weight_ctmle_binomial_loglik_components_recompute_from_the_refit() -> None:
    """Independently rebuild the weighted binomial loss and the weighted log-ratio penalty.

    The ``loss_kind == "loglik"`` branch of :meth:`_Selector.loss` is the branch a weighted
    ``rr`` or ``or`` surface runs, and the log-ratio curve is the penalty that branch is
    scored with. The family and the loss-kind assertions pin that this fixture reaches that
    branch, so a later fixture change that moves it back to the squared loss fails here
    rather than quietly retiring the coverage. The nested cross-validated rebuild stays on
    the gaussian fixture, which already reconstructs it.
    """
    result = _fit_ratio(
        target="rr",
        method="collaborative_tmle",
        weight_scale=1.0,
        collaborative_kwargs={"strategy": "discrete", "candidates": ((), ("W1",), ("W1", "W2"))},
    )
    surface = simulated_confounding(result, estimand="rr", grid=_ANCHOR_GRID, random_state=31)
    refit = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    selection = refit.extra["ctmle"]
    selector = _Selector(
        refit.estimator,
        refit.data,
        refit.nuisance,
        refit.config.g_bounds,
        refit.intermediate_value,
        seed=surface.root_seed,
    )
    rows = np.arange(refit.data.n)
    targeted = refit.nuisance.targeting_outcome
    assert targeted is not None
    chosen = SimpleNamespace(
        targeted=targeted,
        submodel=selector.submodel(refit.nuisance.propensity),
    )
    manual_loss = _manual_ctmle_loss(selector, targeted, rows, "loglik")
    manual_curve = _manual_ctmle_log_ratio_curve(selector, chosen, rows)
    manual_penalty = _manual_curve_penalty(manual_curve)

    assert refit.data.family == "binomial"
    assert selection.loss == "loglik"
    assert not np.allclose(refit.data.weights, 1.0)
    assert manual_loss == pytest.approx(selection.train_loss[selection.selected], rel=1e-12)
    assert manual_penalty == pytest.approx(selection.penalty[selection.selected], rel=1e-12)


def test_fixed_weight_ctmle_vector_target_penalty_recomputes_from_the_refit() -> None:
    """Independently rebuild the weighted penalty of a two-column vector target.

    A fit that reports both counterfactual means selects on ``ey``, whose influence curve is
    a matrix. That is the branch of ``_penalty_of`` no other reconstruction feeds, because
    every other admitted parameter reduces to one column.
    """
    result = _fit_binary_mean(
        treatment=None,
        method="collaborative_tmle",
        weight_scale=1.0,
        collaborative_kwargs={"strategy": "discrete", "candidates": ((), ("W1",))},
    )
    alias = next(iter(result.estimates))
    surface = simulated_confounding(result, estimand=alias, grid=_ANCHOR_GRID, random_state=31)
    refit = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    selection = refit.extra["ctmle"]
    selector = _Selector(
        refit.estimator,
        refit.data,
        refit.nuisance,
        refit.config.g_bounds,
        refit.intermediate_value,
        seed=surface.root_seed,
    )
    rows = np.arange(refit.data.n)
    targeted = refit.nuisance.targeting_outcome
    assert targeted is not None
    chosen = SimpleNamespace(
        targeted=targeted,
        submodel=selector.submodel(refit.nuisance.propensity),
    )
    curves = _manual_ctmle_arm_curves(selector, chosen, rows)
    matrix = np.column_stack([curves[float(arm)] for arm in selector.data.arm_codes])
    manual_loss = _manual_ctmle_loss(selector, targeted, rows, "squared")
    manual_penalty = _manual_curve_penalty(matrix)

    assert result.method.selection_estimand == "ey"
    assert len(selector.target_names) == 2
    assert len(selection.target_names) == 2
    assert selection.loss == "squared"
    assert matrix.shape == (refit.data.n, 2)
    assert not np.allclose(refit.data.weights, 1.0)
    assert manual_loss == pytest.approx(selection.train_loss[selection.selected], rel=1e-12)
    assert manual_penalty == pytest.approx(selection.penalty[selection.selected], rel=1e-12)


@pytest.fixture(scope="module")
def weighted_discrete_ctmle() -> Any:
    """A weighted discrete C-TMLE fit, its anchor surface, and its unmutated manual refit.

    The weight profile runs from 0.1 to 1.9 rather than the file's usual 0.5 to 1.5. Both
    have mean one, so neither rescales anything, and the steeper one gives the mutation
    cases below a wider margin over their gate.
    """
    result = _fit(
        method="collaborative_tmle",
        weight_scale=1.0,
        weight_spread=(0.1, 1.9),
        collaborative_kwargs={
            "strategy": "discrete",
            "candidates": ((), ("W1",), ("W1", "W2")),
        },
    )
    surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=31)
    baseline = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    return SimpleNamespace(result=result, surface=surface, baseline=baseline)


@pytest.mark.parametrize(
    "component",
    [
        "loss",
        "influence",
        "target",
        "_selection_base",
        "_intercept_propensity",
        "_fit_propensity_with",
    ],
)
def test_fixed_weight_ctmle_component_mutations_move_nested_risk(
    component: str,
    weighted_discrete_ctmle: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every selector component that reads the row mass must move the CV criterion.

    Three of these six -- ``target``, ``_selection_base`` and ``_intercept_propensity`` --
    are exactly the components the fixed-weight C-TMLE claim rests on, and the suite this
    replaces passed with the weights stripped from all three. Each case swaps unit weights
    into one production method and leaves the others weighted. :func:`unweight`
    names the one pair where that swap nests instead: ``_fit_propensity_with`` delegates to
    ``_intercept_propensity``, so the first case unweights the second one too.

    ``cv_risk`` is the assertion target because it is the one array every component reaches.
    ``_selection_base`` feeds ``cross_validate`` alone, so ``train_loss`` cannot witness it.
    The measured relative movements on this fixture run from 1.0 percent for
    ``_intercept_propensity`` to 8.8 percent for ``influence``.

    The gate is one part in ten thousand of the criterion's own scale, which is two orders
    below every movement above. It is not one part in a thousand, because the movement a
    component produces is a property of the sample rather than of the code. Rebuilding this
    fixture on the seeds 1 to 8 puts ``_intercept_propensity`` between 0.051 and 1.0
    percent, so a gate at one part in a thousand would fail on a fixture or learner change
    that left every component reading the mass correctly.
    """
    fixture = weighted_discrete_ctmle
    unweight(monkeypatch, _Selector, component)
    mutated = _manual_repeated_refit(
        fixture.result,
        fixture.surface,
        treatment=0.2,
        outcome=0.3,
    )
    weighted_risk = fixture.baseline.extra["ctmle"].cv_risk
    mutated_risk = mutated.extra["ctmle"].cv_risk
    scale = float(np.max(np.abs(weighted_risk)))

    assert not np.allclose(fixture.result.data.weights, 1.0)
    assert np.max(np.abs(weighted_risk - mutated_risk)) > 1e-4 * scale


@pytest.mark.parametrize(
    ("case", "family", "collaborative_kwargs", "patched"),
    [
        ("greedy-loglik", "binomial", {"strategy": "greedy"}, "loss"),
        (
            "ordered-logistic",
            "gaussian",
            {"strategy": "ordered", "preorder": "logistic"},
            "loss",
        ),
    ],
)
def test_fixed_weight_ctmle_search_order_follows_the_weighted_risk(
    case: str,
    family: str,
    collaborative_kwargs: dict[str, Any],
    patched: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The risk a search is scored on follows the observation weights.

    ``candidate.risk`` is ``loss + penalty``, and it is what the greedy search minimizes and
    what a preorder ranks against. Every other C-TMLE fixture here declares
    ``strategy="discrete"``, whose path order is the declared candidate list and ignores
    ``risk`` by construction, so nothing else pins the weighting of a search decision.

    The two cases cover the loglik greedy search and the logistic preorder, both of which
    score with ``loss``. The partial-correlation preorder does not reach ``loss`` at all,
    and ``test_fixed_weight_ctmle_partial_correlation_preorder_follows_the_row_mass``
    covers it on a fixture built for it. There is no explicit-ordering case: a user-declared
    ``ordering=`` fixes the order outright, so that path has no weight-driven choice to
    witness, and its scoring is already pinned by the ``loss`` case of the component family
    above.

    Both cases run on the file's own fixture seed, and the assertion is the risk alone.
    ``train_risk`` is what they can carry: rebuilding each fixture on the seeds 1 to 40
    clears the gate at 40 of 40 for both. The searched *sequence* is not, because it moves
    at only 5 of 40 seeds for the greedy case, so an assertion on the sequence would be an
    assertion about the one seed that was picked to satisfy it.
    ``test_fixed_weight_ctmle_selection_follows_the_weighted_risk`` witnesses the selected
    model on a fixture built to move it, rather than on a seed found to move it.
    """
    if family == "binomial":
        result = _fit_ratio(
            target="rr",
            method="collaborative_tmle",
            weight_scale=1.0,
            collaborative_kwargs=collaborative_kwargs,
        )
        surface = simulated_confounding(result, estimand="rr", grid=_ANCHOR_GRID, random_state=17)
    else:
        result = _fit(
            method="collaborative_tmle",
            weight_scale=1.0,
            collaborative_kwargs=collaborative_kwargs,
        )
        surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=17)
    baseline = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    unweight(monkeypatch, _Selector, patched)
    mutated = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    weighted = baseline.extra["ctmle"]
    unweighted = mutated.extra["ctmle"]
    scale = float(np.max(np.abs(weighted.train_risk)))

    assert result.estimator.strategy == collaborative_kwargs["strategy"]
    assert weighted.loss == ("loglik" if family == "binomial" else "squared")
    assert weighted.preorder == collaborative_kwargs.get("preorder")
    assert np.max(np.abs(weighted.train_risk - unweighted.train_risk)) > 1e-3 * scale


#: The discrete search ``_fit_split_mass_ctmle`` runs by default: two single-covariate
#: candidates and no empty model, because the intercept-only mechanism otherwise wins both
#: searches and neither one has a choice left to make.
_SPLIT_MASS_DISCRETE: dict[str, Any] = {
    "strategy": "discrete",
    "candidates": (("W_upper",), ("W_lower",)),
    "selection_folds": 3,
}


def _fit_split_mass_ctmle(*, seed: int = 7, overrides: dict[str, Any] | None = None) -> Any:
    """A weighted C-TMLE whose two candidate treatment models serve opposite weight blocks.

    The sample is two blocks of a hundred rows. The upper block carries weight 1.8 and the
    lower carries 0.2, so nine tenths of the mass sits in the upper block. ``W_upper``
    predicts treatment inside the upper block and is noise in the lower one, and ``W_lower``
    does the reverse. The lower block's confounding is the stronger of the two, so an
    unweighted search and a weighted search rank the same two candidates differently.

    ``DummyRegressor`` fits an intercept, so the initial ``Qbar`` explains none of either
    block and the fluctuation on ``H(g)`` is the only thing that can lower the loss. That
    also makes the outcome residual the preorder ranks against the centred outcome itself.

    ``overrides`` replaces the search settings, so one engineered sample serves both the
    selection claim and the preorder claim. The two blocks are what make either search
    weight-sensitive, and neither claim depends on which search reads them.
    """
    n = 200
    rng = np.random.default_rng(seed)
    upper = np.arange(n) >= n // 2
    upper_latent = rng.normal(size=n)
    lower_latent = rng.normal(size=n)
    covariate_upper = np.where(upper, upper_latent, rng.normal(size=n))
    covariate_lower = np.where(upper, rng.normal(size=n), lower_latent)
    signal = np.where(upper, 0.8 * upper_latent, 2.2 * lower_latent)
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-1.2 * signal))).astype(float)
    frame = pd.DataFrame(
        {
            "Y": treatment + 1.2 * signal + 0.5 * rng.normal(size=n),
            "A": treatment,
            "W_upper": covariate_upper,
            "W_lower": covariate_lower,
            "weight": np.where(upper, 1.8, 0.2),
        }
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W_upper", "W_lower"),
            weights="weight",
        ),
    )
    return study.identify(ATE()).estimate(
        method=_collaborative_method(overrides=dict(overrides or _SPLIT_MASS_DISCRETE)),
        outcome_learner=DummyRegressor(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        simultaneous=False,
    )


def test_fixed_weight_ctmle_selection_follows_the_weighted_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected treatment model, and not only the sequence, follows the row mass.

    ``path`` is the order the search visited, and a fit reads one position of it. A moved
    ``path`` that leaves ``selected_covariates`` alone leaves the fitted mechanism identical,
    so the selected set is what a selection claim has to assert. This fixture makes that
    distinction explicit: ``strategy="discrete"`` fixes the path outright, the assertion
    below pins it equal across the mutation, and the selection still moves.

    ``_fit_split_mass_ctmle`` is built to move the selection rather than found to. Its two
    candidates serve opposite weight blocks, and the two blocks carry mass in a ratio of
    nine to one, so the weighted search and the unweighted search rank the pair in opposite
    orders. The margin is the gap between the two ``cv_risk`` entries: 34 percent of the
    smaller entry weighted, and 22 percent unweighted, against the 6.5e-5 that separated
    the candidates on the seed the sequence assertion this replaces was pinned to.
    Rebuilding the fixture on the seeds 1 to 30 reverses the selection at 29 of them, with a
    smallest weighted margin of 16 percent, so the fixture seed is the file's own and is not
    chosen for its outcome.

    Both refits run at zero strength, so the replacement sample carries the treatment and
    the outcome the fit already held. The first two assertions pin that, and the engineered
    structure therefore reaches the selector intact. The refit still draws its folds from
    the surface's own root seed, so it is not a copy of the original fit.

    ``_ANCHOR_GRID`` is a seed source here and nothing more. Its one cell is the anchor,
    which copies the fitted estimate and refits nothing, so no assertion below reads a cell.
    """
    result = _fit_split_mass_ctmle()
    surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=17)
    baseline = _manual_repeated_refit(result, surface, treatment=0.0, outcome=0.0)
    unweight(monkeypatch, _Selector, _EVERY_SELECTOR_WEIGHT)
    mutated = _manual_repeated_refit(result, surface, treatment=0.0, outcome=0.0)
    weighted = baseline.extra["ctmle"]
    unweighted = mutated.extra["ctmle"]

    np.testing.assert_array_equal(baseline.data.treatment, result.data.treatment)
    np.testing.assert_array_equal(baseline.data.outcome, result.data.outcome)
    assert not np.allclose(result.data.weights, 1.0)
    assert weighted.path == unweighted.path == (("W_upper",), ("W_lower",))
    assert weighted.selected_covariates == ("W_lower",)
    assert unweighted.selected_covariates == ("W_upper",)


def test_fixed_weight_ctmle_partial_correlation_preorder_follows_the_row_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partial-correlation preorder ranks the covariates at the weighted law.

    ``_ordering`` is the one search decision that never reaches ``loss``. It scores each
    covariate by its weighted partial correlation with the outcome residual given treatment,
    so ``test_fixed_weight_ctmle_search_order_follows_the_weighted_risk`` says nothing about
    it and the assertion here is the produced order rather than a risk.

    The order is what this case mutates, so the order is what it asserts. Scoring the same
    ranking twice under two masses either reverses it or leaves everything downstream
    bit-identical, and a risk gate on the second outcome reports no movement at all. On the
    file's usual gaussian fixture that happened at 9 of the seeds 1 to 40.

    ``_fit_split_mass_ctmle`` removes the coin flip. ``W_upper`` carries the residual inside
    the block holding nine tenths of the mass and ``W_lower`` carries it in the other, so
    the weighted ranking leads with ``W_upper`` and the unweighted one leads with
    ``W_lower``. Rebuilding the fixture on the seeds 1 to 40 reverses the order at 40 of
    them. ``strategy="ordered"`` then makes the path the nested prefixes of that order, so
    the reversal is visible in the reported path.
    """
    result = _fit_split_mass_ctmle(
        overrides={"strategy": "ordered", "preorder": "partial_correlation"}
    )
    surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=17)
    baseline = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    unweight(monkeypatch, _Selector, "_ordering")
    mutated = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)
    weighted = baseline.extra["ctmle"]
    unweighted = mutated.extra["ctmle"]

    assert result.estimator.strategy == "ordered"
    assert weighted.preorder == "partial_correlation"
    assert not np.allclose(result.data.weights, 1.0)
    assert weighted.path == ((), ("W_upper",), ("W_upper", "W_lower"))
    assert unweighted.path == ((), ("W_lower",), ("W_lower", "W_upper"))


def test_fixed_weight_ctmle_preserves_provenance_repeat_cache_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit(
        method="collaborative_tmle",
        repeats=3,
        weight_scale=4.0,
        collaborative_kwargs={"strategy": "ordered", "preorder": "partial_correlation"},
    )
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
        "random_state": 31,
    }
    surface = result.sensitivity.simulated_confounding(**kwargs)
    manual = _manual_repeated_refit(result, surface, treatment=0.1, outcome=0.2)
    per_draw = [repeat.psi["ate"] for repeat in manual.repeats]

    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    assert surface.complete
    assert surface.root_seed == 31
    assert surface.n_repeats == 3
    assert surface.repeat_aggregation == "coordinatewise_median"
    assert surface.cells[3].estimate == manual["ate"].psi == float(np.median(per_draw))
    assert result.estimator.strategy == "ordered"
    assert result.estimator.preorder == "partial_correlation"

    restored = loads(dumps(result))
    replayed = simulated_confounding(restored, **kwargs)
    assert replayed == surface
    assert np.array_equal(restored.data.weights, result.data.weights)
    assert restored.data.weight_spec == result.data.weight_spec
    assert restored.data.weights_name == result.data.weights_name == "weight"
    assert restored.estimator.strategy == "ordered"
    assert restored.estimator.preorder == "partial_correlation"

    calls = _record_refits(result, monkeypatch)
    retained = simulated_confounding(result, grid=_grid(), random_state=37)
    assert retained.complete
    assert len(calls) == 3
    for replacement, seed in calls:
        assert seed == retained.root_seed == 37
        assert np.array_equal(replacement.weights, result.data.weights)
        assert replacement.weight_spec is result.data.weight_spec
        assert replacement.weights_name == result.data.weights_name == "weight"


@pytest.mark.parametrize("strategy", ["greedy", "oat"])
def test_fixed_weight_ctmle_surface_is_invariant_to_a_common_weight_scale(
    strategy: str,
) -> None:
    """The C-TMLE reading of the invariance the ordinary-TMLE test states.

    Both strategies run because a search is the one thing here that could read a scale:
    ``greedy`` scores candidates and ``oat`` fits a mechanism on the arm predictions.
    """
    method = {"strategy": strategy}
    unit_scale, larger_scale = assert_scale_normalizes_away(
        lambda scale: _fit(
            method="collaborative_tmle",
            weight_scale=scale,
            collaborative_kwargs=method,
        )
    )
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.2)),
        "random_state": 7,
    }
    left = simulated_confounding(unit_scale, **kwargs)
    right = simulated_confounding(larger_scale, **kwargs)

    assert left.weight_report.scale == pytest.approx(1.0)
    assert right.weight_report.scale == pytest.approx(13.0)
    for left_cell, right_cell in zip(left.cells, right.cells, strict=True):
        assert left_cell.estimate == pytest.approx(right_cell.estimate, abs=1e-12)
        assert left_cell.displacement == pytest.approx(right_cell.displacement, abs=1e-12)
        assert left_cell.induced_treatment_association == pytest.approx(
            right_cell.induced_treatment_association,
            abs=1e-12,
        )


def test_fixed_weight_ctmle_control_detects_dropped_selector_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selector built on unit weights must move the refitted point estimate.

    The baseline is a manual refit taken before the mutation rather than a grid cell, and
    ``test_fixed_weight_ctmle_additive_cell_equals_a_manual_complete_refit`` already pins
    that the two agree. The gate is relative, because an absolute one hides how large the
    movement is next to the estimate. The measured movement on this fixture is 0.01455 on a
    psi of 1.0056, which is 1.4 percent.
    """
    result = _fit(
        method="collaborative_tmle",
        weight_scale=1.0,
        collaborative_kwargs={
            "strategy": "discrete",
            "candidates": ((), ("W1",), ("W2",), ("W1", "W2", "W3", "W4")),
        },
    )
    surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=31)
    baseline = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)

    unweight(monkeypatch, _Selector, _EVERY_SELECTOR_WEIGHT)
    dropped = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)

    assert type(result.estimator) is CTMLE
    assert abs(baseline["ate"].psi - dropped["ate"].psi) > 1e-3 * abs(baseline["ate"].psi)


def test_fixed_weight_oat_control_detects_dropped_mechanism_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outcome-adaptive treatment model must read the row mass.

    The gate is relative for the same reason the selector control's gate is. The measured
    movement on this fixture is 0.00324 on a psi of 1.0615, which is 0.31 percent.
    """
    result = _fit(
        method="collaborative_tmle",
        weight_scale=1.0,
        collaborative_kwargs={"strategy": "oat"},
    )
    surface = simulated_confounding(result, grid=_ANCHOR_GRID, random_state=31)
    baseline = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)

    unweight(monkeypatch, CTMLE, "_outcome_adaptive_nuisances")
    dropped = _manual_repeated_refit(result, surface, treatment=0.2, outcome=0.3)

    assert abs(baseline["ate"].psi - dropped["ate"].psi) > 1e-3 * abs(baseline["ate"].psi)


def _manual_weighted_correlation(left: Any, right: Any, weights: Any) -> float:
    left_centered = left - np.average(left, weights=weights)
    right_centered = right - np.average(right, weights=weights)
    covariance = np.average(left_centered * right_centered, weights=weights)
    return float(
        covariance
        / np.sqrt(
            np.average(left_centered**2, weights=weights)
            * np.average(right_centered**2, weights=weights)
        )
    )


def test_fixed_weight_association_uses_the_tilted_empirical_law() -> None:
    result = _fit(weight_scale=1.0)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,)),
        random_state=19,
    )
    latent = np.random.default_rng(surface.latent_seed).normal(size=result.data.n)
    perturbed = _perturb_treatment(result.data.treatment, latent, 0.2, "binary")

    expected_anchor = _manual_weighted_correlation(
        latent, result.data.treatment, result.data.weights
    )
    expected_perturbed = _manual_weighted_correlation(latent, perturbed, result.data.weights)
    assert surface.cells[0].induced_treatment_association == pytest.approx(expected_anchor)
    assert surface.cells[1].induced_treatment_association == pytest.approx(expected_perturbed)
    assert surface.cells[1].induced_treatment_association != pytest.approx(
        float(np.corrcoef(latent, perturbed)[0, 1]), abs=1e-5
    )


def _manual_weighted_binary_calibration(result: Any, role: str, index: int) -> float:
    scaler = StandardScaler().fit(result.data.covariates, sample_weight=result.data.weights)
    design = scaler.transform(result.data.covariates)
    target = result.data.treatment if role == "treatment" else result.data.outcome
    model = LogisticRegression(max_iter=1000).fit(design, target, sample_weight=result.data.weights)
    baseline = model.predict(design)
    removed = design.copy()
    removed[:, index] = 0.0
    return float(np.average(model.predict(removed) != baseline, weights=result.data.weights))


def _manual_weighted_continuous_calibration(result: Any, role: str, index: int) -> float:
    covariate = result.data.covariates[:, index]
    target = result.data.treatment if role == "treatment" else result.data.outcome
    correlation = _manual_weighted_correlation(covariate, target, result.data.weights)
    target_centered = target - np.average(target, weights=result.data.weights)
    target_sd = np.sqrt(np.average(target_centered**2, weights=result.data.weights))
    return float(correlation * target_sd)


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
def test_fixed_weight_binary_calibration_uses_weighted_fit_scaling_and_fraction(
    family: str,
) -> None:
    result = _fit(family=family, weight_scale=1.0)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("W1",),
        random_state=7,
    )
    rows = {row.role: row for row in surface.calibrations}
    index = result.data.covariate_names.index("W1")

    assert rows["treatment"].strength == pytest.approx(
        _manual_weighted_binary_calibration(result, "treatment", index)
    )
    if family == "binomial":
        assert rows["outcome"].strength == pytest.approx(
            _manual_weighted_binary_calibration(result, "outcome", index)
        )
    else:
        assert rows["outcome"].strength == pytest.approx(
            _manual_weighted_continuous_calibration(result, "outcome", index)
        )


def test_fixed_weight_continuous_calibration_uses_weighted_correlation_and_scales() -> None:
    result = _fit_continuous(weight_scale=1.0)
    surface = simulated_confounding(
        result,
        estimand=_shift_alias(result),
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("W1",),
        random_state=7,
    )
    rows = {row.role: row for row in surface.calibrations}
    index = result.data.covariate_names.index("W1")

    for role in ("treatment", "outcome"):
        assert rows[role].strength == pytest.approx(
            _manual_weighted_continuous_calibration(result, role, index)
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("estimated", "cannot replay estimated observation weights"),
        ("drtmle-estimated", "cannot replay estimated observation weights"),
        ("collaborative-estimated", "cannot replay estimated observation weights"),
        ("cluster", "clustered fits"),
        ("kind", "fixed probability weights only"),
        ("name", "column and WeightSpec names disagree"),
        ("unnamed", "without a declared weight column"),
    ],
)
def test_weight_refusals_and_provenance_tampering_precede_draws_and_refits(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    method = {
        "collaborative-estimated": "collaborative_tmle",
        "drtmle-estimated": "drtmle",
    }.get(case, "tmle")
    result = _fit(
        method=method,
        weight_scale=1.0,
        weights_estimated=case
        in {
            "estimated",
            "drtmle-estimated",
            "collaborative-estimated",
        },
    )
    if case == "cluster":
        result = replace(
            result,
            data=replace(result.data, cluster=np.arange(result.data.n), cluster_name="id"),
        )
    elif case == "kind":
        result = replace(
            result,
            data=replace(
                result.data,
                weight_spec=replace(result.data.weight_spec, kind="frequency"),  # type: ignore[arg-type]
            ),
        )
    elif case == "name":
        result = replace(
            result,
            data=replace(result.data, weight_spec=replace(result.data.weight_spec, name="wrong")),
        )
    elif case == "unnamed":
        result = replace(
            result,
            data=replace(
                result.data,
                weights_name=None,
                weight_spec=replace(result.data.weight_spec, name=None),
            ),
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
    with pytest.raises(CapabilityError, match=message):
        simulated_confounding(result, grid=_grid())


def _fit_with_a_support_constant_covariate(*, seed: int = 7) -> Any:
    """Fit weighted data whose ``W4`` varies on zero-weight rows alone.

    ``check_weights`` allows a zero weight, so such a row carries no mass. ``W4`` is
    therefore degenerate under the weighted law and calibrates nothing.
    """
    frame, _ = make_linear_ate(n=120, seed=seed)
    weights = mean_one_weights(len(frame))
    unsupported = np.zeros(len(frame), dtype=bool)
    unsupported[::20] = True
    weights[unsupported] = 0.0
    frame["weight"] = weights
    # The supported value is not a binary fraction, so the weighted mean of the column
    # carries a rounding residual and its weighted standard deviation is not exactly zero.
    degenerate = np.full(len(frame), 3.14)
    degenerate[unsupported] = 1.0 + np.arange(int(unsupported.sum()))
    frame["W4"] = degenerate
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
            weights="weight",
        ),
    )
    return study.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        simultaneous=False,
    )


def test_a_covariate_constant_on_the_positive_weight_support_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constant-covariate guard reads the support, not a weighted standard deviation.

    A weighted standard deviation leaves a floating-point residual on this column, so a
    ``== 0.0`` test on it never fires and the surface reports a meaningless strength near
    zero. The exact test on the positive-weight support refuses the column instead.
    """
    result = _fit_with_a_support_constant_covariate()
    column = result.data.covariates[:, result.data.covariate_names.index("W4")]
    weights = result.data.weights

    # The deliberate-mutation control for the guard. Restore the old test and this test
    # fails, because the residual below is not zero.
    assert 0.0 < _weighted_std(column, weights) < 1e-8
    assert np.unique(column[weights > 0.0]).size == 1
    assert np.unique(column).size > 1

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
    with pytest.raises(CapabilityError, match="constant covariate 'W4'"):
        simulated_confounding(result, grid=_grid(), benchmark_covariates=("W4",))


def test_repeated_surface_retains_a_complete_refit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated = _fit(repeats=3)
    calls = _record_refits(repeated, monkeypatch, fail_call=1)
    surface = simulated_confounding(
        repeated,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,)),
        random_state=31,
    )

    # ``_record_refits`` replaces ``estimator.refit`` wholesale, so the estimator's own
    # repeat loop never runs. One recorded call therefore proves that the surface calls the
    # estimator once per cell. It does not prove that the estimator loops over the draws.
    assert len(calls) == 1
    assert surface.n_repeats == 3
    assert not surface.complete
    assert surface.cells[1].failure is not None
    assert surface.cells[1].failure.error_type == "RuntimeError"
    assert "deliberate refit failure" in surface.cells[1].failure.message
    assert surface.failures == (surface.cells[1].failure,)
    assert surface.failures[0].seed == surface.root_seed == 31
    assert surface.cells[1].estimate is None
    assert surface.cells[1].displacement is None


@pytest.mark.parametrize("repeats", [1, 3])
def test_a_median_dropped_estimand_reports_the_median_rule_and_not_a_missing_request(
    monkeypatch: pytest.MonkeyPatch, repeats: int
) -> None:
    """``median_estimates`` omits a name absent from one draw, and the cell says so."""
    result = _fit(repeats=repeats)

    def refit(data: Any, **kwargs: Any) -> Any:
        del kwargs
        # ``inference.influence.median_estimates`` drops a name missing from any draw, so a
        # refitted result can report no ``ate`` even though this cell requested one.
        return replace(result, data=data, estimates={})

    monkeypatch.setattr(result.estimator, "refit", refit)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,)),
        random_state=31,
    )

    failure = surface.cells[1].failure
    assert failure is not None
    if repeats == 1:
        # One draw runs no aggregation, so the accessor's own message stays correct.
        assert failure.error_type == "KeyError"
        assert "was not requested" in failure.message
    else:
        assert failure.error_type == "ValueError"
        assert "missing from at least one of the 3 cross-fitting draws" in failure.message
        assert "median report omits it" in failure.message
        assert "was not requested" not in failure.message


@pytest.mark.parametrize("layer", ["stored", "config", "estimator"])
def test_repeat_provenance_checks_each_count_before_the_latent_draw(
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
) -> None:
    result = _fit(repeats=3)
    if layer == "stored":
        result = replace(result, repeats=result.repeats[:1])
    elif layer == "config":
        result = replace(
            result,
            config=replace(
                result.config,
                crossfit=replace(result.config.crossfit, repeats=1),
            ),
        )
    else:
        estimator = copy(result.estimator)
        estimator.repeats = 1
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

    with pytest.raises(CapabilityError, match="consistent repeated-cross-fitting provenance"):
        simulated_confounding(
            result,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        )


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
    calls = _record_refits(risk_ratio_result, monkeypatch, psi="rr", fail_call=2)
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
    assert surface.cells[2].failure.message == "deliberate refit failure"
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
    calls = _record_refits(result, monkeypatch, psi="shift")
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
    calls = _record_refits(result, monkeypatch, psi="shift")
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
        _continuous_calibration(
            result.data.covariates[:, index], result.data.treatment, result.data.weights
        ),
        rel=1e-12,
    )
    assert rows["outcome"].strength == pytest.approx(
        _continuous_calibration(
            result.data.covariates[:, index], result.data.outcome, result.data.weights
        ),
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
        _weighted_correlation(
            failed_latent, gaussian_result.data.treatment, gaussian_result.data.weights
        )
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
        _binary_calibration(
            design, gaussian_result.data.treatment, index, gaussian_result.data.weights
        ),
        rel=1e-12,
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
        "association_population",
        "target_population_fraction",
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
        "calibration_population",
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
        "association_population",
        "target_population_fraction",
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
    assert calibrations.columns == [
        "covariate",
        "role",
        "family",
        "strength",
        "method",
        "calibration_population",
    ]
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
def test_relabeling_an_ate_cannot_substitute_another_estimand(
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
    message = (
        "registered binary parameter metadata"
        if estimand in {"att", "atc"}
        else (
            "fixed-policy parameter metadata"
            if estimand == "ate_regime"
            else "identity-link arm-based MSM only"
        )
    )
    with pytest.raises(CapabilityError, match=message):
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
        ("cluster", "clustered"),
        ("restored", "replayable"),
        ("estimator", "supports ordinary TMLE"),
        ("outcome-family", "outcome family"),
        ("identification", "identification metadata"),
        ("functional-type", "backdoor-identified parameter"),
        ("provider", "explicit-adjustment backdoor provenance"),
        ("key", "structured parameter key"),
        ("provenance", "registered binary parameter metadata"),
        ("declared-provenance", "registered binary parameter metadata"),
        ("target-provenance", "registered binary parameter metadata"),
        ("conditional", "inconsistent baseline-stratum metadata"),
        ("stochastic", "arm-indexed parameter"),
        ("incremental", "arm-indexed parameter"),
        ("modified", "arm-indexed parameter"),
        ("msm", "arm-indexed parameter"),
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
    module = importlib.import_module("cleverly.sensitivity._simulated_confounding_request")
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
    calls = _record_refits(continuous_gaussian_result, monkeypatch, psi="shift")
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
    calls = _record_refits(result, monkeypatch, psi="shift", alias=alias)

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
    calls = _record_refits(result, monkeypatch, psi="shift", alias=alias)
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
    _record_refits(continuous_means_result, monkeypatch, psi="shift", alias=alias, fail_call=1)
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

    calls = _record_refits(result, monkeypatch, psi="shift", alias=aliases[1])
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


def test_the_facade_selects_a_sole_att(att_result: Any) -> None:
    """The facade resolves a sole ATT and preserves its exact zero anchor."""
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    assert list(att_result.estimates) == ["att"]
    surface = att_result.sensitivity.simulated_confounding(grid=grid)
    assert surface.estimand == "att"
    assert surface.cells[0].estimate == att_result["att"].psi
    assert surface.cells[0].displacement == 0.0
    assert surface.population == "perturbed_treatment_group"


def test_continuous_strengths_are_signed_and_outcome_bounds_follow_the_family(
    continuous_gaussian_result: Any,
    continuous_binomial_result: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gaussian_calls = _record_refits(continuous_gaussian_result, monkeypatch, psi="shift")
    gaussian = simulated_confounding(
        continuous_gaussian_result,
        estimand=_shift_alias(continuous_gaussian_result),
        grid=ConfounderStrengthGrid(treatment=(0.0, -0.75, 0.8), outcome=(0.0, -0.6)),
    )
    assert gaussian.complete
    assert len(gaussian_calls) == 5

    binomial_calls = _record_refits(continuous_binomial_result, monkeypatch, psi="shift")
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
    calls = _record_refits(result, monkeypatch, psi="shift", fail_call=1)
    failed = simulated_confounding(result, estimand=alias, grid=grid, random_state=23)

    assert len(calls) == 1
    assert not failed.complete
    assert failed.cells[1].failure is not None
    assert failed.cells[1].failure.error_type == "RuntimeError"
    assert failed.cells[1].failure.seed == failed.root_seed
    assert failed.cells[1].induced_treatment_association is not None

    replay_calls = _record_refits(result, monkeypatch, psi="shift")
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
        ("key-estimand", "only an ey_shift policy mean or ate_shift contrast"),
        ("key-axis", "only an ey_shift policy mean or ate_shift contrast"),
        ("conditional", "inconsistent baseline-stratum metadata"),
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

    calls = _record_refits(result, monkeypatch, psi="shift")
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
    assert (
        _weighted_correlation(
            latent, low_treated_result.data.treatment, low_treated_result.data.weights
        )
        == expected
    )

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
        "association_population",
        "target_population_fraction",
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

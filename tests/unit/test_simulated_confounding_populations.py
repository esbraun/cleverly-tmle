"""Population contracts for simulated common causes, with independent refit witnesses."""

from __future__ import annotations

import importlib
from copy import copy
from dataclasses import replace
from functools import cache
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import (
    ATC,
    ATE,
    ATT,
    CausalStudy,
    CounterfactualMean,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    OddsRatio,
    PointTreatment,
    RiskRatio,
)
from cleverly.estimators import CTMLE, DRTMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Shift
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from tests.unit.test_simulated_confounding import _collaborative_method


@cache
def _fit_population(
    target: str = "att",
    *,
    method: str = "tmle",
    repeats: int = 1,
    reference: int = 0,
    strata: bool = True,
) -> Any:
    rng = np.random.default_rng(912)
    n = 180
    w = rng.normal(size=n)
    v = np.where(np.arange(n) % 3 == 0, "small", "large")
    a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * w + 0.9 * (v == "small"))))
    y = 0.4 + (1.2 + 1.8 * w) * a + 0.5 * w + rng.normal(scale=0.3, size=n)
    continuous = target in {"ey_shift", "ate_shift"}
    binary = target in {"rr", "or", "ey", "ey1", "ey0"}
    if continuous:
        a = 0.5 * w + rng.normal(size=n)
        y = 0.4 + (1.2 + 0.8 * w) * a + 0.5 * w + rng.normal(scale=0.3, size=n)
    elif binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-0.3 - 0.5 * a - 0.4 * w)))
    frame = pd.DataFrame({"W": w, "V": v, "A": a, "Y": y})
    frame["weight"] = np.where(v == "small", 3.1, 0.7) * np.where(w > 0, 1.8, 0.6)
    policies = (Shift(0.0, cap=10.0, name="natural"), Shift(0.4, cap=10.0, name="up"))
    targets = {
        "ate": ATE(reference=reference),
        "att": ATT(reference=reference),
        "atc": ATC(reference=reference),
        "rr": RiskRatio(reference=reference),
        "or": OddsRatio(reference=reference),
        "ey": CounterfactualMean(),
        "ey1": CounterfactualMean(treatment=1),
        "ey0": CounterfactualMean(treatment=0),
        "ey_shift": ModifiedTreatmentPolicy(shifts=policies),
        "ate_shift": ModifiedTreatmentPolicyEffect(shifts=policies),
    }
    configured: Any = method
    if method in {"greedy", "ordered", "discrete", "oat"}:
        settings: dict[str, Any] = {"strategy": method}
        if method == "ordered":
            settings["preorder"] = "logistic"
        elif method == "discrete":
            settings["candidates"] = ((), ("W",))
        configured = _collaborative_method(overrides=settings)
    return (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W", "V"),
                strata=("V",) if strata else (),
                weights="weight",
                treatment_kind="continuous" if continuous else "discrete",
            ),
        )
        .identify(targets[target])
        .estimate(
            method=configured,
            outcome_learner=(
                LogisticRegression(max_iter=1000)
                if binary
                else make_pipeline(PolynomialFeatures(2), LinearRegression())
            ),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=12,
            repeats=repeats,
            simultaneous=False,
        )
    )


def _alias(result: Any, target: str, stratum: tuple[str, ...] | None) -> str:
    matches = [
        alias
        for alias, key in result.parameter_keys.items()
        if key.estimand == target
        and key.stratum == stratum
        and (target != "ey_shift" or key.value == "up")
    ]
    # The all-arm counterfactual target reports one alias per arm.
    return sorted(matches)[0]


def _mask(result: Any, stratum: tuple[str, ...] | None) -> np.ndarray:
    if stratum is None:
        return np.ones(result.data.n, dtype=bool)
    code = result.data.strata_levels.index(stratum)
    return result.data.strata == code


def _replacement(result: Any, surface: Any, treatment: float, outcome: float) -> Any:
    """Reconstruct the published law without the production perturbation helpers."""
    data = result.data
    latent = np.random.default_rng(surface.latent_seed).normal(size=data.n)
    a = data.treatment.copy()
    if data.is_continuous_treatment:
        a += treatment * latent
    elif treatment:
        changed = latent >= NormalDist().inv_cdf(1 - treatment)
        a[changed] = 1 - a[changed]
    y = data.outcome.copy()
    if data.family == "gaussian":
        y -= outcome * latent
    elif outcome:
        changed = latent >= NormalDist().inv_cdf(1 - outcome)
        y[changed] = 1 - y[changed]
    return data.with_treatment(a).with_outcome(
        y, family=data.family, name="simulated-confounding outcome"
    )


def _correlation(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    return float(
        np.average(x * y, weights=w)
        / np.sqrt(np.average(x * x, weights=w) * np.average(y * y, weights=w))
    )


@pytest.mark.parametrize("target", ["ate", "att", "atc", "ey", "ey1", "ey0", "rr", "or"])
def test_binary_stratum_cells_equal_complete_weighted_refits(target: str) -> None:
    result = _fit_population(target)
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17)),
        random_state=31,
    )
    cell = surface.cells[-1]
    replacement = _replacement(result, surface, 0.22, 0.17)
    manual = result.estimator.refit(replacement, random_state=surface.root_seed)
    assert cell.failure is None
    assert cell.estimate == manual[alias].psi
    assert cell.displacement == manual[alias].inference_value - result[alias].inference_value
    assert abs(cell.displacement) > 1e-5
    assert surface.stratum == ("small",)
    assert surface.strata_names == ("V",)
    assert surface.refit_population == "full_fitted_population"
    np.testing.assert_array_equal(replacement.strata, result.data.strata)
    np.testing.assert_array_equal(replacement.weights, result.data.weights)
    assert replacement.n == result.data.n


@pytest.mark.parametrize("target", ["att", "atc"])
@pytest.mark.parametrize("reference", [0, 1])
@pytest.mark.parametrize("stratum", [None, ("small",)])
def test_conditional_population_moves_with_perturbed_treatment(
    target: str,
    reference: int,
    stratum: tuple[str, ...] | None,
) -> None:
    result = _fit_population(target, reference=reference)
    alias = _alias(result, target, stratum)
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
        random_state=31,
    )
    replacement = _replacement(result, surface, 0.32, 0.0)
    manual = result.estimator.refit(replacement, random_state=surface.root_seed)
    baseline = _mask(result, stratum)
    conditioning = 1 - reference if target == "att" else reference
    old = baseline & (result.data.treatment == conditioning)
    moved = baseline & (replacement.treatment == conditioning)
    assert np.count_nonzero(old != moved) > 10
    # The targeted outcome contrast is held fixed here: only membership can change.
    targeted = manual.fluctuations[target].targeted
    contrast = targeted.arms[float(1 - reference)] - targeted.arms[float(reference)]
    current = np.average(contrast[moved], weights=replacement.weights[moved])
    frozen = np.average(contrast[old], weights=replacement.weights[old])
    assert abs(current - frozen) > 1e-3
    assert manual.nuisance.scaler.unscale_difference(current) == pytest.approx(manual[alias].psi)
    assert abs(manual.nuisance.scaler.unscale_difference(frozen) - manual[alias].psi) > 1e-3
    assert surface.population == "perturbed_treatment_group"
    assert surface.conditioning_arm == conditioning
    assert surface.cells[-1].target_population_fraction == pytest.approx(
        np.average(
            replacement.treatment[baseline] == conditioning, weights=replacement.weights[baseline]
        )
    )
    assert surface.cells[-1].estimate == manual[alias].psi
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.cells[0].displacement == 0.0


@pytest.mark.parametrize("method", ["greedy", "ordered", "discrete", "oat"])
def test_collaborative_strata_keep_full_weighted_selection_refit(method: str) -> None:
    result = _fit_population("ate", method=method)
    alias = _alias(result, "ate", ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17)),
        random_state=31,
    )
    manual = result.estimator.refit(_replacement(result, surface, 0.22, 0.17), random_state=31)
    assert surface.cells[-1].estimate == manual[alias].psi
    assert abs(surface.cells[-1].displacement) > 1e-4
    assert surface.population == "baseline"
    assert surface.cells[-1].target_population_fraction == 1.0


@pytest.mark.parametrize("target", ["ey_shift", "ate_shift"])
def test_continuous_policy_strata_equal_complete_refits(target: str) -> None:
    result = _fit_population(target)
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.4), outcome=(0.0, 0.17)),
        random_state=31,
    )
    manual = result.estimator.refit(_replacement(result, surface, 0.4, 0.17), random_state=31)
    assert surface.cells[-1].failure is None
    assert surface.cells[-1].estimate == manual[alias].psi
    assert abs(surface.cells[-1].displacement) > 1e-5
    assert surface.stratum == ("small",)


def test_strata_share_full_latent_draw_and_global_calibration_but_local_association() -> None:
    result = _fit_population("att")
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
        "benchmark_covariates": ("W",),
        "random_state": 31,
    }
    surfaces = [
        simulated_confounding(result, _alias(result, "att", stratum), **kwargs)
        for stratum in [None, ("small",), ("large",)]
    ]
    pooled, small, large = surfaces
    assert pooled.latent_seed == small.latent_seed == large.latent_seed
    assert pooled.calibrations == small.calibrations == large.calibrations
    assert small.calibration_population == "full_fitted_population"
    latent = np.random.default_rng(small.latent_seed).normal(size=result.data.n)
    data = _replacement(result, small, 0.32, 0.0)
    for surface in surfaces:
        mask = _mask(result, surface.stratum)
        expected = _correlation(data.treatment[mask], latent[mask], data.weights[mask])
        assert surface.cells[-1].induced_treatment_association == pytest.approx(expected)
    assert (
        abs(
            small.cells[-1].induced_treatment_association
            - pooled.cells[-1].induced_treatment_association
        )
        > 0.03
    )
    assert small.association_population == "selected_baseline_stratum"
    assert pooled.association_population == "full_fitted_population"


def test_weighted_repeated_population_surface_cache_and_persistence() -> None:
    result = _fit_population("att", repeats=3)
    alias = _alias(result, "att", ("small",))
    kwargs = {
        "estimand": alias,
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0,)),
        "benchmark_covariates": ("W",),
        "random_state": 32,
    }
    surface = result.sensitivity.simulated_confounding(**kwargs)
    manual = result.estimator.refit(_replacement(result, surface, 0.22, 0.0), random_state=32)
    assert surface.n_repeats == 3
    assert surface.cells[-1].estimate == manual[alias].psi
    assert surface.cells[-1].estimate == float(
        np.median([draw.psi[alias] for draw in manual.repeats])
    )
    assert abs(surface.cells[-1].estimate - manual.repeats[0].psi[alias]) > 1e-5
    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    assert loads(dumps(result)).sensitivity.simulated_confounding(**kwargs) == surface
    frame = surface.to_frame()
    assert list(frame["target_population_fraction"]) == [
        cell.target_population_fraction for cell in surface.cells
    ]
    assert set(frame["association_population"]) == {"selected_baseline_stratum"}
    assert set(surface.calibration_frame()["calibration_population"]) == {"full_fitted_population"}
    assert "population" in surface.summary().lower()


@pytest.mark.parametrize("change", ["key", "levels", "names", "typed", "functional", "replay"])
def test_population_provenance_tampering_precedes_draws_and_refits(
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_population("att")
    alias = _alias(result, "att", ("small",))
    if change == "key":
        keys = dict(result.parameter_keys)
        keys[alias] = replace(keys[alias], stratum=("absent",))
        result = replace(result, parameter_keys=keys)
    elif change == "levels":
        result = replace(result, data=replace(result.data, strata_levels=(("absent",), ("large",))))
    elif change == "names":
        result = replace(result, data=replace(result.data, strata_names=("wrong",)))
    elif change == "typed":
        result = replace(
            result, identified_effect=replace(result.identified_effect, estimand=ATC())
        )
    elif change == "functional":
        identified = result.identified_effect
        result = replace(
            result,
            identified_effect=replace(
                identified, functional=replace(identified.functional, target="atc")
            ),
        )
    else:
        estimator = copy(result.estimator)
        estimator.estimands = ("atc",)
        result = replace(result, estimator=estimator)
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(module, "_latent_child_seed", lambda *_: pytest.fail("drew before refusal"))
    monkeypatch.setattr(
        result.estimator, "refit", lambda *_args, **_kwargs: pytest.fail("refit before refusal")
    )
    with pytest.raises(CapabilityError):
        simulated_confounding(
            result, alias, grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        )


@pytest.mark.parametrize("target", ["att", "atc"])
def test_empty_perturbed_conditioning_population_retains_a_failed_cell(target: str) -> None:
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    n = 160
    latent = np.random.default_rng(module._latent_child_seed(31)).normal(size=n)
    tail = latent >= NormalDist().inv_cdf(0.68)
    small = np.arange(n) % 2 == 0
    a = tail.astype(int) if target == "att" else (~tail).astype(int)
    # Outside the selected stratum both arms survive the same published flip.
    a[~small] = np.arange(n)[~small] % 3 == 0
    rng = np.random.default_rng(932)
    w = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "A": a,
            "W": w,
            "V": np.where(small, "small", "large"),
            "Y": a * (1 + w) + rng.normal(size=n),
        }
    )
    result = (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W", "V"),
                strata=("V",),
            ),
        )
        .identify(ATT() if target == "att" else ATC())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            learner_folds=2,
            simultaneous=False,
            random_state=12,
        )
    )
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
        random_state=31,
    )
    assert surface.cells[0].failure is None
    cell = surface.cells[-1]
    assert cell.target_population_fraction == 0.0
    assert cell.failure is not None
    assert cell.estimate is None and cell.displacement is None
    assert not surface.complete


@pytest.mark.parametrize("method", ["greedy", "oat", "drtmle"])
@pytest.mark.parametrize("target", ["att", "atc"])
def test_unsupported_conditional_population_estimators_refuse_before_draws(
    target: str,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_population(target, strata=False)
    estimator = DRTMLE() if method == "drtmle" else CTMLE(strategy=method)
    result = replace(result, estimator=estimator)
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(module, "_latent_child_seed", lambda *_: pytest.fail("drew before refusal"))
    monkeypatch.setattr(
        estimator, "refit", lambda *_args, **_kwargs: pytest.fail("refit before refusal")
    )
    with pytest.raises(CapabilityError, match="exact ordinary TMLE"):
        simulated_confounding(
            result, target, grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        )


def test_stratified_drtmle_refuses_before_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fit_population("ate")
    estimator = DRTMLE()
    result = replace(result, estimator=estimator)
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(module, "_latent_child_seed", lambda *_: pytest.fail("drew before refusal"))
    monkeypatch.setattr(
        estimator, "refit", lambda *_args, **_kwargs: pytest.fail("refit before refusal")
    )
    with pytest.raises(CapabilityError, match="stratified reduced-regression targeting"):
        simulated_confounding(
            result,
            _alias(result, "ate", ("small",)),
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,)),
        )


def test_unavailable_policy_alias_excludes_conditional_natural_course_means() -> None:
    result = _fit_population("ey_shift")
    supported = [alias for alias, key in result.parameter_keys.items() if key.value == "up"]
    natural = [alias for alias, key in result.parameter_keys.items() if key.value == "natural"]
    assert len(natural) == 3 and len(supported) == 3
    with pytest.raises(ValueError, match="unavailable") as refusal:
        simulated_confounding(
            result, "unknown", grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )
    assert all(alias in str(refusal.value) for alias in supported)
    assert all(alias not in str(refusal.value) for alias in natural)

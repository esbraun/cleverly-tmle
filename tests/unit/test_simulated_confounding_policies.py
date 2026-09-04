"""Fixed-policy replay witnesses for the simulated common-cause surface."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import (
    ATE,
    AssessmentStatus,
    CausalStudy,
    CounterfactualMean,
    IncrementalMean,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
)
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Rule, Static, Stochastic
from cleverly.msm import MSM
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_fixed as fixed_replay
from tests.unit.test_simulated_confounding_populations import (
    _forbid_draw_and_refit,
    _replacement,
)

_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17))


def _stochastic_density(w: Any) -> np.ndarray:
    p = 0.3 + 0.4 * (np.asarray(w["W"]) > 0)
    return np.column_stack((1 - p, p))


@dataclass
class _MSMDesign:
    treated: Any = 1
    saturated: bool = False

    def __call__(self, a: Any, w: Any) -> np.ndarray:
        columns = [np.ones(len(w)), np.full(len(w), a == self.treated)]
        if not self.saturated:
            columns.append(np.asarray(w["W"]))
        return np.column_stack(columns)


@dataclass
class _MSMWeight:
    treated: Any = 1

    def __call__(self, a: Any, w: Any) -> np.ndarray:
        return (0.5 + 2.0 * (np.asarray(w["W"]) > 0)) * (1.8 if a == self.treated else 0.6)


def _policy(kind: str, *, labels: bool = False) -> Any:
    """All declarations see baseline covariates and preserve the arm-label contract."""
    control, treated = ("control", "treated") if labels else (0, 1)
    if kind == "static":
        return Static(treated, name="policy")
    if kind == "rule":
        return Rule(lambda w: np.where(np.asarray(w["W"]) > 0, treated, control), name="policy")
    if kind == "stochastic":
        return Stochastic(_stochastic_density, name="policy")
    raise AssertionError(kind)


def _model(*, saturated: bool = False, labels: bool = False, link: str = "identity") -> MSM:
    treated = "treated" if labels else 1
    if saturated:
        return MSM(
            design=_MSMDesign(treated, True),
            terms=("intercept", "treatment"),
            link=link,
        )
    return MSM(
        design=_MSMDesign(treated),
        terms=("intercept", "treatment", "baseline"),
        weights=_MSMWeight(treated),
        link=link,
    )


def _study(
    *,
    backend: str = "pandas",
    labels: bool = False,
    weighted: bool = False,
    strata: bool = False,
    binary: bool = False,
) -> CausalStudy:
    rng = np.random.default_rng(317)
    n = 180
    w = rng.normal(size=n)
    v = np.where(np.arange(n) % 3 == 0, "small", "large")
    a = rng.binomial(1, 1 / (1 + np.exp(-0.7 * w)))
    linear = 0.4 + 0.8 * w + a * (0.9 + 1.6 * w) + 0.3 * w**2
    y = (
        rng.binomial(1, 1 / (1 + np.exp(-linear)))
        if binary
        else linear + rng.normal(scale=0.35, size=n)
    )
    frame = pd.DataFrame(
        {"W": w, "V": v, "A": np.where(a, "treated", "control") if labels else a, "Y": y}
    )
    frame["weight"] = np.where(w > 0, 2.8, 0.5)
    if backend == "polars":
        import polars as pl

        frame = pl.from_pandas(frame)
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W", "V"),
            strata=("V",) if strata else (),
            weights="weight" if weighted else None,
        ),
    )


def _estimate(study: CausalStudy, target: Any, *, repeats: int = 1, binary: bool = False) -> Any:
    return study.identify(target).estimate(
        outcome_learner=LogisticRegression(max_iter=1000)
        if binary
        else make_pipeline(PolynomialFeatures(2), LinearRegression()),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=12,
        simultaneous=False,
        repeats=repeats,
    )


@cache
def _fit_policy(
    kind: str = "stochastic",
    *,
    contrast: bool = True,
    backend: str = "pandas",
    labels: bool = False,
    weighted: bool = False,
    strata: bool = False,
    repeats: int = 1,
    binary: bool = False,
) -> Any:
    control = "control" if labels else 0
    interventions = (Static(control, name="reference"), _policy(kind, labels=labels))
    target = (
        RegimeContrast(interventions, reference="reference")
        if contrast
        else RegimeMean((interventions[1],))
    )
    return _estimate(
        _study(backend=backend, labels=labels, weighted=weighted, strata=strata, binary=binary),
        target,
        repeats=repeats,
        binary=binary,
    )


@cache
def _fit_msm(
    *,
    saturated: bool = False,
    weighted: bool = False,
    labels: bool = False,
    backend: str = "pandas",
    repeats: int = 1,
    binary: bool = False,
    link: str = "identity",
    strata: bool = False,
) -> Any:
    return _estimate(
        _study(weighted=weighted, labels=labels, backend=backend, binary=binary, strata=strata),
        MSMProjection(_model(saturated=saturated, labels=labels, link=link)),
        repeats=repeats,
        binary=binary,
    )


def _alias(
    result: Any, *, stratum: tuple[str, ...] | None = None, coefficient: str = "treatment"
) -> str:
    return next(
        name
        for name, key in result.parameter_keys.items()
        if key.stratum == stratum
        and ((key.term == coefficient) if key.axis == "msm" else key.value == "policy")
    )


@pytest.mark.parametrize(
    "kind,contrast,backend,labels,weighted,strata,repeats,binary",
    [
        ("static", False, "pandas", False, False, False, 1, False),
        ("rule", True, "polars", True, True, False, 1, False),
        ("stochastic", False, "pandas", False, True, True, 3, False),
        ("stochastic", True, "polars", False, False, False, 1, True),
    ],
)
def test_policy_cells_equal_independent_complete_refits(
    kind: str,
    contrast: bool,
    backend: str,
    labels: bool,
    weighted: bool,
    strata: bool,
    repeats: int,
    binary: bool,
) -> None:
    result = _fit_policy(
        kind,
        contrast=contrast,
        backend=backend,
        labels=labels,
        weighted=weighted,
        strata=strata,
        repeats=repeats,
        binary=binary,
    )
    alias = _alias(result, stratum=("small",) if strata else None)
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    assert len(surface.cells) == 4
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.movement_scale == "estimate_difference"
    for cell in surface.cells[1:]:
        manual = result.estimator.refit(
            _replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
            random_state=31,
        )
        assert cell.estimate == pytest.approx(manual[alias].psi, abs=1e-12)
        assert cell.displacement == pytest.approx(manual[alias].psi - result[alias].psi, abs=1e-12)
    assert surface.n_repeats == repeats
    assert surface.association_population == (
        "selected_baseline_stratum" if strata else "full_fitted_population"
    )


@pytest.mark.parametrize("contrast", [False, True])
def test_static_policy_reduces_to_arm_surface(contrast: bool) -> None:
    study = _study()
    regime = _fit_policy("static", contrast=contrast)
    arm = _estimate(study, ATE() if contrast else CounterfactualMean(treatment=1))
    regime_surface = simulated_confounding(
        regime, estimand=_alias(regime), grid=_GRID, random_state=31
    )
    arm_surface = simulated_confounding(
        arm, estimand="ate" if contrast else "ey1", grid=_GRID, random_state=31
    )
    assert [cell.estimate for cell in regime_surface.cells] == pytest.approx(
        [cell.estimate for cell in arm_surface.cells], abs=1e-10
    )
    assert [cell.displacement for cell in regime_surface.cells] == pytest.approx(
        [cell.displacement for cell in arm_surface.cells], abs=1e-10
    )


@pytest.mark.parametrize(
    "labels,backend,weighted,repeats,strata,binary",
    [
        (False, "pandas", False, 1, False, False),
        (True, "polars", True, 3, True, False),
        (False, "pandas", False, 1, False, True),
    ],
)
def test_msm_coefficient_surface_recomputes_full_projection(
    labels: bool, backend: str, weighted: bool, repeats: int, strata: bool, binary: bool
) -> None:
    result = _fit_msm(
        labels=labels,
        backend=backend,
        weighted=weighted,
        repeats=repeats,
        strata=strata,
        binary=binary,
    )
    alias = _alias(result, stratum=("small",) if strata else None)
    surface = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    for cell in surface.cells[1:]:
        manual = result.estimator.refit(
            _replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
            random_state=31,
        )
        assert cell.estimate == pytest.approx(manual[alias].psi, abs=1e-12)
        assert cell.displacement == pytest.approx(manual[alias].psi - result[alias].psi, abs=1e-12)
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-3
    assert surface.stratum == (("small",) if strata else None)
    if strata:
        assert (
            simulated_confounding(loads(dumps(result)), estimand=alias, grid=_GRID, random_state=31)
            == surface
        )


def test_saturated_msm_slope_reduces_to_ate_surface() -> None:
    msm = _fit_msm(saturated=True)
    ate = _estimate(_study(), ATE())
    msm_surface = simulated_confounding(msm, estimand=_alias(msm), grid=_GRID, random_state=31)
    ate_surface = simulated_confounding(ate, grid=_GRID, random_state=31)
    assert [cell.estimate for cell in msm_surface.cells] == pytest.approx(
        [cell.estimate for cell in ate_surface.cells], abs=1e-9
    )


def test_policy_surface_cache_persistence_and_assessment() -> None:
    result = _fit_policy(contrast=False)
    kwargs = {"grid": _GRID, "random_state": 31}
    surface = result.sensitivity.simulated_confounding(**kwargs)
    assert surface.estimand == _alias(result)
    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    assert loads(dumps(result)).sensitivity.simulated_confounding(**kwargs) == surface
    battery = result.assess(include_refits=True, arguments={"simulated_confounding": kwargs})
    item = battery.sensitivity["simulated_confounding"]
    assert item.status is AssessmentStatus.COMPLETED
    assert item._report is surface


def test_msm_surface_requires_unambiguous_coefficient() -> None:
    result = _fit_msm()
    with pytest.raises(ValueError, match=r"choose|explicit|unavailable"):
        result.sensitivity.simulated_confounding(grid=_GRID, random_state=31)


def _corrupt_policy(result: Any, alias: str, field: str) -> Any:
    """Change one provenance layer, preserving the others as independent witnesses."""
    result = replace(result, estimator=copy(result.estimator))
    identified = result.identified_effect
    if field.startswith("key-"):
        name = field.removeprefix("key-")
        changes = {
            name: {
                "axis": "shift",
                "value": "absent",
                "term": "absent",
                "reference": "absent",
                "alias": "wrong",
                "horizon": 3,
            }[name]
        }
        return replace(
            result,
            parameter_keys={
                **result.parameter_keys,
                alias: replace(result.parameter_keys[alias], **changes),
            },
        )
    if field.startswith("config-"):
        name = field.removeprefix("config-")
        return replace(
            result,
            config=replace(result.config, **{name: "arm" if name == "parameter_axis" else 1.0}),
        )
    if field == "functional-target":
        return replace(
            result,
            identified_effect=replace(
                identified, functional=replace(identified.functional, target="ate")
            ),
        )
    if field == "functional-reference":
        return replace(
            result,
            identified_effect=replace(
                identified, functional=replace(identified.functional, reference="policy")
            ),
        )
    if field == "typed-reference":
        return replace(
            result,
            identified_effect=replace(
                identified, estimand=replace(identified.estimand, reference="policy")
            ),
        )
    if field == "estimator-reference":
        result.estimator.reference = "policy"
        return result
    if field == "estimator-declaration":
        result.estimator.interventions = tuple(reversed(result.estimator.interventions))
        return result
    if field == "typed-model":
        return replace(
            result,
            identified_effect=replace(
                identified, estimand=replace(identified.estimand, model=_model(saturated=True))
            ),
        )
    if field == "estimator-model":
        result.estimator.msm = _model(saturated=True)
        return result
    if field == "estimate-name":
        return replace(
            result, estimates={**result.estimates, alias: replace(result[alias], name="wrong")}
        )
    # A later draw must be checked even when the first cache agrees with the declaration.
    draw = result.repeats[-1]
    nuisance = draw.nuisance
    if field.startswith("regime-"):
        state = nuisance.regimes
        change = field.removeprefix("regime-")
        if change == "values":
            state = replace(state, values=state.values[:, ::-1, :].copy())
        elif change == "names":
            state = replace(state, names=tuple(reversed(state.names)))
        elif change == "reference":
            state = replace(state, reference=1.0)
        elif change == "nan":
            values = state.values.copy()
            values[0, 0, 0] = np.nan
            state = replace(state, values=values)
        else:
            raise AssertionError(field)
        nuisance = replace(nuisance, regimes=state)
    elif field.startswith("msm-"):
        state = nuisance.msm
        change = field.removeprefix("msm-")
        if change == "design":
            design = state.design.copy()
            design[:, :, 1] *= -1
            state = replace(state, design=design)
        elif change == "weights":
            state = replace(state, weights=np.ones_like(state.weights))
        elif change == "terms":
            state = replace(state, terms=tuple(reversed(state.terms)))
        elif change == "arms":
            state = replace(state, arms=tuple(reversed(state.arms)))
        elif change == "link":
            state = replace(state, link="logit")
        else:
            raise AssertionError(field)
        nuisance = replace(nuisance, msm=state)
    else:
        raise AssertionError(field)
    return replace(result, repeats=(*result.repeats[:-1], replace(draw, nuisance=nuisance)))


@pytest.mark.parametrize(
    "field",
    [
        "key-axis",
        "key-value",
        "key-reference",
        "key-alias",
        "key-horizon",
        "config-parameter_axis",
        "config-reference_arm",
        "functional-target",
        "functional-reference",
        "typed-reference",
        "estimator-reference",
        "estimator-declaration",
        "estimate-name",
        "regime-values",
        "regime-names",
        "regime-reference",
        "regime-nan",
    ],
)
def test_regime_provenance_layers_refuse_before_draw_or_refit(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit_policy(repeats=3)
    alias = _alias(result)
    corrupted = _corrupt_policy(result, alias, field)
    _forbid_draw_and_refit(monkeypatch, corrupted.estimator)
    with pytest.raises(CapabilityError, match="fixed-policy parameter metadata"):
        simulated_confounding(corrupted, estimand=alias, grid=_GRID, random_state=31)


@pytest.mark.parametrize(
    "field",
    [
        "key-term",
        "key-value",
        "key-reference",
        "key-alias",
        "config-parameter_axis",
        "functional-target",
        "typed-model",
        "estimator-model",
        "msm-design",
        "msm-weights",
        "msm-terms",
        "msm-arms",
        "msm-link",
    ],
)
def test_msm_provenance_layers_refuse_before_draw_or_refit(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit_msm(repeats=3)
    alias = _alias(result)
    corrupted = _corrupt_policy(result, alias, field)
    _forbid_draw_and_refit(monkeypatch, corrupted.estimator)
    with pytest.raises(CapabilityError, match="fixed-policy parameter metadata"):
        simulated_confounding(corrupted, estimand=alias, grid=_GRID, random_state=31)


@dataclass
class _CountingDensity:
    calls: int = 0
    limit: int | None = None
    flipped: bool = False

    def __call__(self, w: Any) -> np.ndarray:
        self.calls += 1
        if self.limit is not None and self.calls > self.limit:
            raise AssertionError("policy callback was replayed after validation")
        values = _stochastic_density(w)
        return values[:, ::-1] if self.flipped else values


def test_regime_callback_is_checked_once_then_frozen_for_every_cell() -> None:
    density = _CountingDensity()
    result = _estimate(_study(), RegimeMean((Stochastic(density, name="policy"),)), repeats=3)
    original = result.estimator.interventions
    density.calls, density.limit = 0, 1
    surface = simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)
    assert density.calls == 1
    assert all(cell.failure is None for cell in surface.cells)
    assert result.estimator.interventions is original


def test_changed_callback_cannot_silently_change_the_assessed_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    density = _CountingDensity()
    result = _estimate(_study(), RegimeMean((Stochastic(density, name="policy"),)))
    density.flipped = True
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="declared regime densities disagree"):
        simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)


def test_wrong_frozen_density_moves_a_nonzero_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fit_policy(contrast=False)
    alias = _alias(result)
    baseline = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    original = fixed_replay._FrozenRegime.density

    def uniform(self: Any, data: Any) -> Any:
        values = original(self, data)
        return np.full_like(values, 0.5)

    monkeypatch.setattr(fixed_replay._FrozenRegime, "density", uniform)
    mutated = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    assert mutated.cells[0].estimate == baseline.cells[0].estimate
    assert abs(mutated.cells[-1].estimate - baseline.cells[-1].estimate) > 0.05


@pytest.mark.parametrize("component", ["design", "weights"])
def test_wrong_frozen_projection_moves_a_nonzero_coefficient(
    component: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit_msm()
    alias = _alias(result)
    baseline = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    original = fixed_replay._FrozenArmFunction.__call__

    def changed(self: Any, arm: Any, frame: Any) -> Any:
        values = original(self, arm, frame)
        if values.ndim == 1 and component == "weights":
            return np.ones_like(values)
        if values.ndim == 2 and component == "design":
            values[:, 1] *= -1
        return values

    monkeypatch.setattr(fixed_replay._FrozenArmFunction, "__call__", changed)
    mutated = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=31)
    assert mutated.cells[-1].failure is None
    assert mutated.cells[0].estimate == baseline.cells[0].estimate
    assert abs(mutated.cells[-1].estimate - baseline.cells[-1].estimate) > 0.05


def test_msm_persistence_replays_a_new_surface() -> None:
    result = _fit_msm(repeats=3)
    alias = _alias(result)
    loaded = loads(dumps(result))
    expected = simulated_confounding(result, estimand=alias, grid=_GRID, random_state=43)
    assert simulated_confounding(loaded, estimand=alias, grid=_GRID, random_state=43) == expected


@pytest.mark.parametrize("link", ["log", "logit"])
def test_nonlinear_msm_refuses_before_draws(link: str, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fit_msm(binary=True, saturated=True, link=link)
    result = replace(result, estimator=copy(result.estimator))
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="identity-link arm-based MSM only"):
        simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)


@dataclass
class _CountingArmFunction:
    function: Any
    calls: int = 0
    limit: int | None = None

    def __call__(self, arm: Any, frame: Any) -> np.ndarray:
        self.calls += 1
        if self.limit is not None and self.calls > self.limit:
            raise AssertionError("MSM callback was replayed after validation")
        return self.function(arm, frame)


def test_msm_callbacks_are_checked_once_per_arm_then_frozen() -> None:
    model = _model()
    design = _CountingArmFunction(model.design)
    weights = _CountingArmFunction(model.weights)
    declared = replace(model, design=design, weights=weights)
    result = _estimate(_study(weighted=True), MSMProjection(declared), repeats=3)
    design.calls = weights.calls = 0
    design.limit = weights.limit = 2
    surface = simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)
    assert all(cell.failure is None for cell in surface.cells)
    assert design.calls == weights.calls == 2
    assert result.estimator.msm is declared


def test_a_retained_cell_failure_does_not_stop_fixed_policy_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_policy()
    original = TMLE.refit
    calls = 0

    def refit(self: Any, data: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("deliberate fixed-policy cell failure")
        return original(self, data, **kwargs)

    monkeypatch.setattr(TMLE, "refit", refit)
    surface = simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)
    assert calls == 3
    assert surface.cells[1].estimate is None
    assert "deliberate fixed-policy cell failure" in surface.cells[1].failure.message
    assert surface.cells[-1].failure is None


class _CustomStochastic(Stochastic):
    """A protocol-compatible extension lacks the exact fixed-policy replay contract."""


def test_custom_intervention_refuses_before_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _estimate(
        _study(), RegimeMean((_CustomStochastic(_stochastic_density, name="policy"),))
    )
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="exact Static, Rule, and Stochastic regimes only"):
        simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)


def test_incremental_policy_remains_a_distinct_refused_functional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _estimate(
        _study(binary=True), IncrementalMean((Incremental(2.0, name="up"),)), binary=True
    )
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="arm-indexed parameter"):
        simulated_confounding(
            result, estimand=next(iter(result.estimates)), grid=_GRID, random_state=31
        )


def test_continuous_dose_msm_remains_outside_binary_policy_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(11)
    w = rng.normal(size=180)
    a = 0.4 * w + rng.normal(size=180)
    study = CausalStudy(
        pd.DataFrame({"Y": 1.0 + 2.0 * a + 0.3 * w, "A": a, "W": w}),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
    )
    result = study.estimate(
        MSMProjection(MSM.linear(doses=np.linspace(-1.5, 1.5, 9))),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        density_bins=8,
        simultaneous=False,
        random_state=3,
    )
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="identity-link arm-based MSM only"):
        simulated_confounding(result, estimand="msm[a]", grid=_GRID, random_state=31)

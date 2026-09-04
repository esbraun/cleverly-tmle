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
from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, Rule, Static, Stochastic
from cleverly.msm import MSM, MSMSet
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_fixed as fixed_replay
from tests.unit._confounding_support import (
    Counter,
    alias_for,
    with_estimator,
    with_functional,
    with_key,
    with_last_nuisance,
    with_typed,
)
from tests.unit._confounding_support import (
    confounding_estimate as _estimate,
)
from tests.unit._confounding_support import (
    confounding_study as _study,
)
from tests.unit._confounding_support import (
    forbid_draw_and_refit as _forbid_draw_and_refit,
)
from tests.unit._confounding_support import (
    replacement as _replacement,
)

_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17))


def _stochastic_density(w: Any) -> np.ndarray:
    p = 0.3 + 0.4 * (np.asarray(w["W"]) > 0)
    return np.column_stack((1 - p, p))


@dataclass(frozen=True)
class _MSMDesign:
    treated: Any = 1
    saturated: bool = False

    def __call__(self, a: Any, w: Any) -> np.ndarray:
        columns = [np.ones(len(w)), np.full(len(w), a == self.treated)]
        if not self.saturated:
            columns.append(np.asarray(w["W"]))
        return np.column_stack(columns)


@dataclass(frozen=True)
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
    return alias_for(
        result,
        stratum=stratum,
        coefficient=coefficient if result.config.parameter_axis == "msm" else None,
        value=None if result.config.parameter_axis == "msm" else "policy",
    )


@pytest.mark.parametrize(
    "kind,contrast,backend,labels,weighted,strata,repeats,binary",
    [
        ("static", False, "pandas", False, False, False, 1, False),
        ("rule", True, "polars", True, True, False, 1, False),
        ("rule", False, "pandas", False, False, False, 1, False),
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
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-3
    assert surface.population == "baseline"
    assert surface.conditioning_arm is None
    assert all(cell.target_population_fraction == 1.0 for cell in surface.cells)
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
    assert max(abs(cell.displacement) for cell in regime_surface.cells) > 1e-3
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
    assert surface.population == "baseline"
    assert surface.conditioning_arm is None
    assert all(cell.target_population_fraction == 1.0 for cell in surface.cells)
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
    assert max(abs(cell.displacement) for cell in msm_surface.cells) > 1e-3
    assert [cell.estimate for cell in msm_surface.cells] == pytest.approx(
        [cell.estimate for cell in ate_surface.cells], abs=1e-9
    )


@pytest.mark.parametrize("msm", [False, True])
def test_policy_surface_cache_persistence_and_assessment(msm: bool) -> None:
    result = _fit_msm() if msm else _fit_policy(contrast=False)
    kwargs = {"grid": _GRID, "random_state": 31}
    if msm:
        kwargs["estimand"] = _alias(result)
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
        return with_key(result, alias, **changes)
    if field.startswith("config-"):
        name = field.removeprefix("config-")
        return replace(
            result,
            config=replace(result.config, **{name: "arm" if name == "parameter_axis" else 1.0}),
        )
    if field == "functional-target":
        return with_functional(result, target="ate")
    if field == "functional-reference":
        return with_functional(result, reference="policy")
    if field == "typed-reference":
        return with_typed(result, reference="policy")
    if field == "estimator-reference":
        return with_estimator(result, reference="policy")
    if field == "estimator-declaration":
        return with_estimator(result, interventions=tuple(reversed(result.estimator.interventions)))
    if field == "typed-model":
        return with_typed(result, model=_model(saturated=True))
    if field == "estimator-model":
        return with_estimator(result, msm=_model(saturated=True))
    if field == "estimate-name":
        return replace(
            result, estimates={**result.estimates, alias: replace(result[alias], name="wrong")}
        )
    # A later draw must be checked even when the first cache agrees with the declaration.
    nuisance = result.repeats[-1].nuisance
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
        return with_last_nuisance(result, regimes=state)
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
        return with_last_nuisance(result, msm=state)
    else:
        raise AssertionError(field)


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
    with pytest.raises(
        CapabilityError,
        match={
            "key-horizon": "cannot replay this fixed-policy composition",
            "typed-reference": "regime declarations disagree",
            "estimator-declaration": "regime declarations disagree",
            "config-reference_arm": "regime labels or reference disagree",
            "key-reference": "regime labels or reference disagree",
            "regime-values": "declared regime densities disagree",
            "regime-nan": "non-finite probability",
        }.get(field, "fixed-policy parameter metadata"),
    ):
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
    with pytest.raises(
        CapabilityError,
        match={
            "typed-model": "identity-link arm-based MSM declarations disagree",
            "estimator-model": "identity-link arm-based MSM declarations disagree",
            "msm-design": "declared MSM arrays disagree",
        }.get(field, "fixed-policy parameter metadata"),
    ):
        simulated_confounding(corrupted, estimand=alias, grid=_GRID, random_state=31)


def test_regime_callback_is_checked_once_then_frozen_for_every_cell() -> None:
    density = Counter(_stochastic_density)
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
    density = Counter(_stochastic_density)
    result = _estimate(_study(), RegimeMean((Stochastic(density, name="policy"),)))
    density.function = lambda w: _stochastic_density(w)[:, ::-1]
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


def test_msm_callbacks_are_checked_once_per_arm_then_frozen() -> None:
    model = _model()
    design = Counter(model.design)
    weights = Counter(model.weights)
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


@pytest.mark.parametrize("regimens", [(1, 0), (Static(1), 0)])
def test_bare_regime_levels_replay_and_assess(regimens: tuple[Any, ...]) -> None:
    result = _estimate(_study(), RegimeContrast(regimens))
    explicit = _estimate(_study(), RegimeContrast((Static(1), Static(0))))
    alias = next(iter(result.estimates))
    kwargs = {"grid": _GRID, "random_state": 31}
    expected = simulated_confounding(explicit, estimand=alias, **kwargs)
    surface = result.sensitivity.simulated_confounding(**kwargs)
    assert surface == expected
    battery = result.assess(include_refits=True, arguments={"simulated_confounding": kwargs})
    assert battery.sensitivity["simulated_confounding"].status is AssessmentStatus.COMPLETED


@pytest.mark.parametrize("target", ["ey_regime", "ate_regime", "msm"])
def test_fixed_replay_requires_exact_ordinary_tmle(target: str) -> None:
    assert fixed_replay.fixed_replay_refusal(DRTMLE(), target) == (
        "simulated_confounding supports fixed regimes and MSMs under exact ordinary TMLE only"
    )


@pytest.mark.parametrize("component", ["density", "design", "weights"])
def test_user_callback_failure_keeps_its_original_error(component: str) -> None:
    broken = False

    def callback(*args: Any) -> Any:
        if broken:
            raise KeyError("W2")
        return (
            _stochastic_density(*args)
            if component == "density"
            else getattr(_model(), component)(*args)
        )

    target = (
        RegimeMean((Stochastic(callback, name="policy"),))
        if component == "density"
        else MSMProjection(replace(_model(), **{component: callback}))
    )
    result = _estimate(_study(), target)
    broken = True
    with pytest.raises(KeyError, match="W2"):
        simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)


def test_frozen_msm_refuses_a_missing_treatment_arm() -> None:
    result = _fit_msm(saturated=True)
    alias = _alias(result)
    replay = fixed_replay.validate_fixed_replay(result, alias, result.parameter_keys[alias])
    data = copy(result.data)
    object.__setattr__(data, "treatment_levels", (0,))
    with pytest.raises(CapabilityError, match="original baseline rows and arms"):
        MSMSet.evaluate(replay.msm, data)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_stochastic_density_refuses_nonfinite_values_on_original_fit(value: float) -> None:
    def density(w: Any) -> np.ndarray:
        values = _stochastic_density(w)
        values[0, 0] = value
        return values

    with pytest.raises(DataError, match="non-finite probability"):
        _estimate(_study(), RegimeMean((Stochastic(density, name="policy"),)))


def test_binary_fit_with_msm_doses_refuses_before_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _estimate(_study(), MSMProjection(replace(_model(), doses=(0.0, 0.5, 1.0))))
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="identity-link arm-based MSM only"):
        simulated_confounding(result, estimand=_alias(result), grid=_GRID, random_state=31)


@pytest.mark.parametrize("slot", ["shifts", "incremental"])
def test_arm_replay_refuses_other_declared_counterfactual_slots(
    slot: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _estimate(_study(), ATE())
    result = replace(result, estimator=copy(result.estimator))
    setattr(result.estimator, slot, (object(),))
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="supports an arm-indexed parameter"):
        simulated_confounding(result, grid=_GRID, random_state=31)

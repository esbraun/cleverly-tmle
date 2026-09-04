"""Nonlinear and continuous MSM replay on a fixed projection measure."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import AssessmentStatus, CausalStudy, MSMProjection, PointTreatment
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError
from cleverly.msm import LINKS, MSM, MSMSet
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_fixed as replay_module
from tests.unit._confounding_support import (
    forbid_draw_and_refit,
    replacement,
    with_estimator,
    with_last_nuisance,
)
from tests.unit.test_simulated_confounding_policies import _alias, _fit_msm

_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0, 0.02))
_DOSES = (-1.4, -0.8, -0.15, 0.35, 1.1, 1.6)


@dataclass(frozen=True)
class _DoseWeight:
    def __call__(self, dose: Any, frame: Any) -> np.ndarray:
        return 1.0 + 0.6 * (np.asarray(dose) + 0.4 * np.asarray(frame["W"])) ** 2


@cache
def _fit_continuous(
    link: str = "identity",
    *,
    repeats: int = 1,
    strata: bool = False,
    backend: str = "pandas",
    binary: bool = False,
) -> Any:
    rng = np.random.default_rng(81)
    n = 180
    w = rng.normal(size=n)
    a = 0.35 * w + rng.normal(size=n)
    y = 0.15 + 0.65 * expit(0.1 + 0.55 * a + 0.45 * w + 0.3 * a * w)
    y += 0.015 * rng.normal(size=n)
    if binary:
        y = rng.binomial(1, y)
    frame = pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "W": w,
            "S": np.where(w > 0, "high", "low"),
            "weight": np.where(w > 0, 1.8, 0.6),
        }
    )
    if backend == "polars":
        import polars as pl

        frame = pl.from_pandas(frame)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W", "S") if strata else ("W",),
            strata=("S",) if strata else (),
            weights="weight",
            treatment_kind="continuous",
        ),
    )
    return study.estimate(
        MSMProjection(
            MSM.linear(
                modifiers=("W",), interaction=False, weights=_DoseWeight(), link=link, doses=_DOSES
            )
        ),
        outcome_learner=LogisticRegression(max_iter=1000)
        if binary
        else make_pipeline(PolynomialFeatures(2), LinearRegression()),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        density_bins=6,
        simultaneous=False,
        repeats=repeats,
        random_state=12,
    )


@pytest.mark.parametrize("link", ["log", "logit"])
def test_nonlinear_binary_msm_replays_complete_unsaturated_projection(link: str) -> None:
    result = _fit_msm(binary=True, weighted=True, repeats=3, link=link)
    alias = _alias(result)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17))
    surface = simulated_confounding(result, estimand=alias, grid=grid, random_state=31)
    for cell in surface.cells[1:]:
        manual = result.estimator.refit(
            replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
            random_state=surface.refit_seed,
        )
        assert cell.failure is None
        assert cell.estimate == pytest.approx(manual[alias].psi, abs=1e-12)
        assert cell.displacement == pytest.approx(manual[alias].psi - result[alias].psi)
    assert surface.movement_scale == "estimate_difference"
    assert max(abs(cell.displacement) for cell in surface.cells) > 0.01
    assert (
        simulated_confounding(loads(dumps(result)), estimand=alias, grid=grid, random_state=31)
        == surface
    )


@pytest.mark.parametrize("link", ["identity", "log", "logit"])
def test_continuous_msm_cells_equal_complete_refits_and_rebuild_observed_doses(link: str) -> None:
    result = _fit_continuous(link)
    surface = simulated_confounding(result, estimand="msm[a]", grid=_GRID, random_state=31)
    replay = replay_module.validate_fixed_replay(result, "msm[a]", result.parameter_keys["msm[a]"])
    baseline = result.nuisances[0].msm
    for cell in surface.cells[1:]:
        data = replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
        manual = result.estimator.refit(data, random_state=surface.refit_seed)
        assert cell.failure is None
        assert cell.estimate == pytest.approx(manual["msm[a]"].psi, abs=1e-12)
        state = MSMSet.evaluate(replay.msm, data)
        for field in ("design", "weights", "clever_weights"):
            np.testing.assert_array_equal(getattr(state, field), getattr(baseline, field))
        assert state.dose_values == _DOSES
        # Independent formulas expose the observed-dose branch and both endpoints.
        observed_design = np.column_stack((np.ones(data.n), data.treatment, data.covariates[:, 0]))
        raw_h = 1 + 0.6 * (data.treatment + 0.4 * data.covariates[:, 0]) ** 2
        mask = (data.treatment >= _DOSES[0]) & (data.treatment <= _DOSES[-1])
        np.testing.assert_allclose(state.observed_design, observed_design)
        np.testing.assert_allclose(state.observed_weights, raw_h * mask)
        if cell.treatment_strength:
            assert not np.array_equal(state.observed_design, baseline.observed_design)
            assert not np.array_equal(state.observed_weights, baseline.observed_weights)
            assert np.any(mask != (baseline.observed_weights > 0))
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-4
    assert surface.movement_scale == "estimate_difference"


def test_continuous_grid_uses_raw_weights_once_in_trapezoid_measure() -> None:
    result = _fit_continuous()
    replay = replay_module.validate_fixed_replay(result, "msm[a]", result.parameter_keys["msm[a]"])
    state = MSMSet.evaluate(replay.msm, result.data)
    a = np.asarray(_DOSES)
    quadrature = np.r_[np.diff(a)[0] / 2, (a[2:] - a[:-2]) / 2, np.diff(a)[-1] / 2]
    raw_h = 1 + 0.6 * (a[None, :] + 0.4 * result.data.covariates[:, :1]) ** 2
    np.testing.assert_allclose(state.clever_weights, raw_h)
    np.testing.assert_allclose(state.weights, raw_h * quadrature)
    assert not np.allclose(state.weights, state.clever_weights)


@pytest.mark.parametrize("link", ["identity", "log", "logit"])
def test_continuous_msm_strata_refuse_before_nuisance_fitting(
    link: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TMLE, "_scaler", lambda *_: pytest.fail("fit reached nuisance preparation"))
    with pytest.raises(
        NotImplementedError, match="continuous MSMs do not yet support baseline strata"
    ):
        _fit_continuous(link, strata=True)


@pytest.mark.parametrize(
    "backend,binary,link",
    [("polars", True, "logit"), ("pandas", True, "log"), ("polars", False, "identity")],
)
def test_continuous_backend_and_outcome_families_replay(
    backend: str, binary: bool, link: str
) -> None:
    result = _fit_continuous(link, backend=backend, binary=binary)
    surface = simulated_confounding(result, estimand="msm[a]", grid=_GRID, random_state=31)
    for cell in surface.cells[1:]:
        manual = result.estimator.refit(
            replacement(result, surface, cell.treatment_strength, cell.outcome_strength),
            random_state=surface.refit_seed,
        )
        assert cell.failure is None
        assert cell.estimate == pytest.approx(manual["msm[a]"].psi, abs=1e-12)
    assert max(abs(cell.displacement) for cell in surface.cells) > 1e-4


def test_continuous_replay_freezes_grid_calls_and_copies_callback_owned_weights() -> None:
    result = _fit_continuous()
    replay = replay_module.validate_fixed_replay(result, "msm[a]", result.parameter_keys["msm[a]"])
    raw = np.ones(result.data.n)
    calls = []

    def observed_only(dose: Any, frame: Any) -> Any:
        assert np.asarray(dose).ndim == 1
        calls.append(np.asarray(dose).copy())
        return raw

    replay.msm = replace(replay.msm, weights=replace(replay.msm.weights, function=observed_only))
    data = result.data.with_treatment(result.data.treatment + 0.2)
    state = MSMSet.evaluate(replay.msm, data)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], data.treatment)
    np.testing.assert_array_equal(raw, np.ones(data.n))
    assert np.any(state.observed_weights == 0)
    np.testing.assert_array_equal(state.clever_weights, result.nuisances[0].msm.clever_weights)


@pytest.mark.parametrize("weights,value", [(False, np.nan), (True, np.inf), (True, -1.0)])
def test_invalid_observed_callback_is_refused(weights: bool, value: float) -> None:
    result = _fit_continuous()
    replay = replay_module.validate_fixed_replay(result, "msm[a]", result.parameter_keys["msm[a]"])
    field = "weights" if weights else "design"
    callback = getattr(replay.msm, field)
    replacement_callback = replace(callback, function=lambda dose, frame: np.full(len(dose), value))
    replay.msm = replace(replay.msm, **{field: replacement_callback})
    with pytest.raises(DataError, match="finite design and nonnegative finite weights"):
        MSMSet.evaluate(replay.msm, result.data)


def test_continuous_replay_requires_unchanged_baseline_rows() -> None:
    result = _fit_continuous()
    replay = replay_module.validate_fixed_replay(result, "msm[a]", result.parameter_keys["msm[a]"])
    data = result.data.with_covariates(result.data.covariates[::-1])
    with pytest.raises(CapabilityError, match="original baseline rows"):
        MSMSet.evaluate(replay.msm, data)


@pytest.mark.parametrize(
    "component", ["quadrature", "observed_design", "observed_weights", "mask", "link"]
)
def test_wrong_continuous_replay_changes_a_nonzero_cell(
    component: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit_continuous("logit")
    expected = simulated_confounding(result, estimand="msm[a]", grid=_GRID, random_state=31)
    original = replay_module._FrozenDoseFunction.__call__
    old_a = result.data.treatment.copy()

    def mutated(self: Any, dose: Any, frame: Any) -> Any:
        if np.asarray(dose).ndim == 0:
            values = original(self, dose, frame)
            if component == "quadrature" and self.weights:
                j = _DOSES.index(float(dose))
                return values * (0.3 if j == 0 else 1.7)
            return values
        if (component == "observed_weights" and self.weights) or (
            component == "observed_design" and not self.weights
        ):
            return original(self, old_a, frame)
        return original(self, dose, frame)

    monkeypatch.setattr(replay_module._FrozenDoseFunction, "__call__", mutated)
    if component == "link":
        freeze = replay_module._freeze_msm

        def identity_projection(*args: Any) -> Any:
            replay, alias = freeze(*args)
            replay.msm = replace(replay.msm, link="identity")
            return replay, alias

        monkeypatch.setattr(replay_module, "_freeze_msm", identity_projection)
    if component == "mask":
        evaluate = MSMSet.evaluate.__func__

        def unmasked(cls: Any, msm: Any, data: Any) -> Any:
            state = evaluate(cls, msm, data)
            if isinstance(msm.weights, replay_module._FrozenDoseFunction):
                w = data.covariates[:, 0]
                return replace(state, observed_weights=1 + 0.6 * (data.treatment + 0.4 * w) ** 2)
            return state

        monkeypatch.setattr(MSMSet, "evaluate", classmethod(unmasked))
    wrong = simulated_confounding(result, estimand="msm[a]", grid=_GRID, random_state=31)
    assert wrong.cells[-1].failure is None
    assert abs(expected.cells[-1].displacement) > 1e-4
    assert abs(wrong.cells[-1].estimate - expected.cells[-1].estimate) > 1e-5


@pytest.mark.parametrize(
    "field", ["dose_values", "clever_weights", "observed_design", "observed_weights"]
)
def test_continuous_msm_checks_every_cached_draw(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit_continuous(repeats=3)
    state = result.nuisances[-1].msm
    value = getattr(state, field)
    value = tuple(x + 0.01 for x in value) if field == "dose_values" else value + 0.01
    result = with_last_nuisance(result, msm=replace(state, **{field: value}))
    result = with_estimator(result)
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="declared MSM arrays disagree"):
        simulated_confounding(result, estimand="msm[a]", grid=_GRID)


def test_continuous_msm_persistence_and_native_assessment() -> None:
    result = _fit_continuous("logit", repeats=3)
    expected = simulated_confounding(result, estimand="msm[a]", grid=_GRID, random_state=31)
    loaded = loads(dumps(result))
    assert simulated_confounding(loaded, estimand="msm[a]", grid=_GRID, random_state=31) == expected
    kwargs = {"estimand": "msm[a]", "grid": _GRID, "random_state": 31}
    assert loaded.sensitivity.simulated_confounding(**kwargs) == expected
    report = loaded.sensitivity.run_all(
        include_refits=True, arguments={"simulated_confounding": kwargs}
    )
    assert report["simulated_confounding"].status is AssessmentStatus.COMPLETED
    battery = loaded.assess(include_refits=True, arguments={"simulated_confounding": kwargs})
    assert battery.sensitivity["simulated_confounding"].status is AssessmentStatus.COMPLETED


def test_custom_link_refuses_before_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(LINKS, "custom", replace(LINKS["identity"], name="custom"))
    result = with_estimator(
        _fit_continuous(), msm=replace(_fit_continuous().estimator.msm, link="custom")
    )
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="identity, log, or logit"):
        simulated_confounding(result, estimand="msm[a]", grid=_GRID)


@pytest.mark.parametrize("link", ["log", "logit"])
def test_link_support_failure_is_retained_per_cell(link: str) -> None:
    result = _fit_continuous(link)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0, 0.8))
    surface = simulated_confounding(result, estimand="msm[a]", grid=grid, random_state=31)
    assert surface.cells[1].failure is not None
    assert surface.cells[2].failure is None
    assert surface.cells[2].estimate is not None
    assert surface.cells[-1].failure is not None

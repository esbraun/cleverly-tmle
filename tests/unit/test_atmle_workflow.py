"""The study workflow of adaptive TMLE for a trial with external data (X90 T2).

A fit through ``CausalStudy`` reproduces the engine on the same arrays, carries the
``working_model_projection`` status, records the controls-only and few-treated cases, keeps
its estimate through ``save`` and ``load``, and every public sensitivity and validation entry
point refuses it before any point-treatment code runs.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
import cleverly.sensitivity
import cleverly.validation
from cleverly import (
    AdaptiveTMLEMethod,
    AdaptiveTMLEResult,
    AssessmentStatus,
    CapabilityError,
    CausalStudy,
    CrossFitting,
    ModelSpec,
    PositivityWarning,
    TrialATE,
    TrialWithExternalData,
)
from cleverly._trial_family import AUGMENTED_TRIAL_FAMILY
from cleverly.estimators.atmle import fit_adaptive_tmle

#: The text every adaptive TMLE refusal of a point-treatment analysis carries.
REFUSED = ("adaptive working-model fit", "adaptive fit's own state")


def _frame(
    n: int = 300,
    *,
    seed: int = 0,
    external_treated: float = 0.3,
    missing: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    a = np.where(s == 1, rng.binomial(1, 0.5, n), rng.binomial(1, external_treated, n))
    y = 1.0 + a + w + 0.5 * (1 - s) * (1 + a) + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W": w, "S": s})
    if missing:
        observed = rng.binomial(1, 0.85, n)
        frame["R"] = observed
        frame.loc[observed == 0, "Y"] = np.nan
    return frame


def _method(form: str = "cv", **models: Any) -> AdaptiveTMLEMethod:
    return AdaptiveTMLEMethod(
        form=form,  # type: ignore[arg-type]
        models=ModelSpec(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            **models,
        ),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
    ).with_overrides(random_state=7)


def _fit(frame: pd.DataFrame, population: str = "trial", **kwargs: Any) -> AdaptiveTMLEResult:
    design = TrialWithExternalData(
        outcome="Y",
        treatment="A",
        adjustment=("W",),
        source="S",
        missingness=kwargs.pop("missingness", None),
    )
    effect = CausalStudy(frame, design=design).identify(TrialATE(population=population))  # type: ignore[arg-type]
    result = effect.estimate(method=_method(**kwargs))
    assert isinstance(result, AdaptiveTMLEResult)
    return result


@pytest.fixture(scope="module")
def result() -> AdaptiveTMLEResult:
    return _fit(_frame())


@pytest.mark.parametrize("form", ["cv", "pooled"])
@pytest.mark.parametrize("population", ["trial", "pooled"])
def test_the_study_fit_reproduces_the_engine(form: str, population: str) -> None:
    """The wrapper hands the engine the declared arrays, learners, bounds and split."""
    frame = _frame(seed=1)
    fitted = _fit(frame, population, form=form)
    direct = fit_adaptive_tmle(
        frame["Y"].to_numpy(),
        frame["A"].to_numpy(dtype=float),
        frame["S"].to_numpy(dtype=float),
        frame[["W"]].to_numpy(),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        source_learner=LogisticRegression(),
        g_bounds=fitted.g_bounds,
        form=form,  # type: ignore[arg-type]
        population=population,  # type: ignore[arg-type]
        folds=fitted.folds,
        inner_folds=3,
        random_state=7,
    )
    estimate = fitted.estimate
    assert estimate.psi == pytest.approx(direct.estimate, abs=1e-12)
    assert estimate.std_error == pytest.approx(direct.std_error, rel=1e-12)
    assert fitted.inference_status == "working_model_projection"
    assert estimate.support.label == "experimental"
    assert fitted.fit.population == population and fitted.fit.form == form
    low, high = estimate.ci
    assert low < estimate.psi < high


def test_the_contrast_is_treated_against_control(result: AdaptiveTMLEResult) -> None:
    key = result.parameter_keys["ate"]
    assert (key.value, key.reference) == (1, 0)
    assert result.to_frame().loc[0, "estimand"] == "ate"
    with pytest.raises(CapabilityError, match="Only the difference scale"):
        result.ratio("ate", "ate")


def test_controls_only_external_data_take_the_controls_only_path() -> None:
    fitted = _fit(_frame(external_treated=0.0))
    support = fitted.diagnostics.support()
    assert support.controls_only and support.n_external_treated == 0
    assert "controls only" in str(fitted.summary())


def test_a_few_external_treated_rows_warn_and_keep_the_both_arms_path() -> None:
    frame = _frame(seed=2)
    external_treated = np.flatnonzero((frame["S"] == 0) & (frame["A"] == 1))
    frame.loc[external_treated[5:], "A"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = _fit(frame)
    assert fitted.diagnostics.support().n_external_treated == 5
    assert any(
        record.category == PositivityWarning.__name__ and "only 5" in record.message
        for record in fitted.fit_warnings
    )


def test_missing_outcomes_fit_a_missingness_model() -> None:
    fitted = _fit(_frame(missing=True), missingness="R", missingness_learner=LogisticRegression())
    assert "missingness" in fitted.learners
    assert np.isfinite(fitted.psi())


def test_the_result_round_trips_through_save_and_load(
    result: AdaptiveTMLEResult, tmp_path: Path
) -> None:
    path = result.save(tmp_path / "atmle.joblib")
    loaded = cleverly.load(path)
    assert isinstance(loaded, AdaptiveTMLEResult)
    assert loaded.psi() == result.psi()
    assert loaded.estimate.std_error == result.estimate.std_error
    assert loaded.inference_status == result.inference_status


def test_the_assessment_reads_the_fit_and_refuses_every_bound(result: AdaptiveTMLEResult) -> None:
    rows = {row.operation: row for row in result.diagnostics.capabilities}
    assert {name for name, row in rows.items() if row.available} == {
        "support",
        "nuisance_models",
        "score_equations",
        "corrections",
    }
    assert result.diagnostics.score_equations().passed
    assert result.diagnostics.corrections().converged
    assert len(result.diagnostics.nuisance_models().folds) == 3
    report = result.assess()
    ran = {item.name: item for item in report.diagnostics.items}
    assert ran["support"].status is AssessmentStatus.PASSED
    assert all(not row.available for row in result.sensitivity.capabilities)
    for row in result.sensitivity.capabilities:
        assert row.reason is not None and "adaptive working-model fit" in row.reason
    replay = result.replayability
    assert replay.rerun_configuration and not replay.retarget_cached_nuisances


# ----------------------------------------------------------------------- the entry points

#: Placeholder values for a required argument the refusal must precede.
_ARGUMENTS: dict[str, Any] = {"covariates": ("W",), "grid": object(), "tolerance": 1e-3}


def _entry_points() -> list[tuple[str, Any]]:
    """Every public sensitivity and validation function that takes a fitted result first."""
    found = []
    for module in (cleverly.sensitivity, cleverly.validation):
        for name in module.__all__:
            function = getattr(module, name)
            if not inspect.isfunction(function):
                continue
            parameters = list(inspect.signature(function).parameters)
            if parameters and parameters[0] == "result":
                found.append((f"{module.__name__}.{name}", function))
    return found


ENTRY_POINTS = _entry_points()


def _refusals(result: Any) -> dict[str, str | None]:
    """The adaptive TMLE refusal each entry point raised, or ``None`` when it raised another."""
    outcomes: dict[str, str | None] = {}
    for name, function in ENTRY_POINTS:
        required = [
            parameter.name
            for parameter in list(inspect.signature(function).parameters.values())[1:]
            if parameter.default is inspect.Parameter.empty
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                function(result, **{argument: _ARGUMENTS[argument] for argument in required})
        except CapabilityError as error:
            text = str(error)
            outcomes[name] = text if any(phrase in text for phrase in REFUSED) else None
        except Exception:  # point code reached and failed some other way
            outcomes[name] = None
        else:
            outcomes[name] = None
    return outcomes


def test_the_namespace_holds_the_entry_points() -> None:
    names = {name.rsplit(".", 1)[1] for name, _ in ENTRY_POINTS}
    assert {"evalue", "omitted_variable_bounds", "simulated_confounding", "refute"} <= names
    assert len(names) >= 16


def test_every_public_entry_point_refuses_the_family(result: AdaptiveTMLEResult) -> None:
    refused = _refusals(result)
    assert all(refused.values()), [name for name, text in refused.items() if not text]


def test_the_refusal_is_keyed_on_the_declared_family(
    result: AdaptiveTMLEResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberate mutation: a result that declares ``"point"`` reaches point code."""
    assert AdaptiveTMLEResult.assessment_family == AUGMENTED_TRIAL_FAMILY
    monkeypatch.setattr(AdaptiveTMLEResult, "assessment_family", "point")
    mutated = dataclasses.replace(result)
    refused = _refusals(mutated)
    assert not any(refused.values()), [name for name, text in refused.items() if text]

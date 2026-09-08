"""Exact fitted score-load diagnostics for the three intervention axes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import load
from cleverly.datasets import make_missing_outcome, make_nonlinear_ate, make_shift_dose
from cleverly.estimators import TMLE
from cleverly.interventions import Incremental, Rule, Shift, Static


def _screen_on_risk(frame: Any) -> np.ndarray:
    """A pickle-compatible dynamic rule for persistence coverage."""
    return (np.asarray(frame["W1"]) > 0.0).astype(float)


def _fit(frame: Any, **settings: Any) -> Any:
    weighted = bool(settings.pop("weighted", True))
    if weighted:
        weights = np.where(np.asarray(frame["W1"]) > 0.0, 4.0, 0.25)
        frame = frame.assign(w=weights)
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=3,
            learner_folds=2,
            random_state=settings.pop("random_state"),
            simultaneous=False,
            **settings,
        )
        .fit(frame, outcome="Y", treatment="A", weights="w" if weighted else None)
        .single()
    )


@pytest.fixture(scope="module")
def intervention_results() -> dict[str, Any]:
    point, _ = make_nonlinear_ate(n=500, seed=31)
    dose, _ = make_shift_dose(n=500, seed=32)
    return {
        "regime": _fit(
            point,
            random_state=31,
            interventions=(
                Static(0, name="none"),
                Rule(_screen_on_risk, name="screen"),
            ),
        ),
        "mtp": _fit(
            dose,
            random_state=32,
            shifts=(
                Shift(0.0, cap=None, name="current"),
                Shift(0.5, cap=5.0, name="up"),
            ),
            density_bins=20,
        ),
        "ipsi": _fit(
            point,
            random_state=31,
            incremental=(Incremental(0.5, name="down"), Incremental(2.0, name="up")),
        ),
    }


def _top_share(load: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(load.size * fraction)))
    return float(np.sort(load)[-count:].sum() / load.sum())


def _rows(result: Any, group: str) -> Mapping[str, Any]:
    report = result.diagnostics.support()
    return report.regimes if group == "regime" else report


def _without_load(item: Any) -> dict[str, Any]:
    return {
        field.name: getattr(item, field.name)
        for field in dataclasses.fields(item)
        if field.name not in {"score_load", "score_load_omission"}
    }


def _with_artifact(result: Any, group: str, artifact: Any) -> Any:
    repeat = result.repeats[0]
    fluctuations = dict(repeat.fluctuations)
    fluctuations[group] = dataclasses.replace(fluctuations[group], absolute_score_weights=artifact)
    return dataclasses.replace(
        result,
        repeats=(dataclasses.replace(repeat, fluctuations=fluctuations),),
    )


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_each_intervention_row_reads_its_exact_fitted_score_column(
    intervention_results: dict[str, Any], group: str
) -> None:
    """The hand calculation binds declaration order, masks, weights, and zeros at once."""
    result = intervention_results[group]
    fluctuation = result.fluctuations[group]
    artifact = fluctuation.absolute_score_weights
    assert artifact is not None
    rows = _rows(result, group)
    assert len(rows) == artifact.shape[1] == len(fluctuation.names)

    for index, (label, support) in enumerate(rows.items()):
        load = np.asarray(artifact[:, index], dtype=float)
        effective = float(np.square(load.sum()) / np.square(load).sum())
        reported = support.score_load
        assert reported is not None, support.score_load_omission
        assert reported == {
            "equation": fluctuation.names[index],
            "reported_repeat": 1,
            "n_repeats": 1,
            "n_total": float(result.data.n),
            "n_targeted": float(load.size),
            "effective": effective,
            "targeted_ratio": effective / float(load.size),
            "total_ratio": effective / float(result.data.n),
            "top_1pct": _top_share(load, 0.01),
            "top_5pct": _top_share(load, 0.05),
            "max_load": float(load.max()),
            "zero_load": float(np.count_nonzero(load == 0.0)),
        }, label


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_the_old_intervention_ratio_cannot_stand_in_for_the_fitted_score_load(
    intervention_results: dict[str, Any], group: str
) -> None:
    """Unequal observation weights make the old ratio-only summary a nonzero mutation."""
    rows = _rows(intervention_results[group], group)
    assert any(
        support.score_load is not None
        and not np.isclose(support.score_load["effective"], support.effective_sample_size)
        for support in rows.values()
    )


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_each_axis_reads_the_matching_column_and_keeps_old_support_fields(
    intervention_results: dict[str, Any], group: str
) -> None:
    result = intervention_results[group]
    artifact = result.fluctuations[group].absolute_score_weights
    assert artifact is not None
    moved = np.asarray(artifact, dtype=float).copy()
    moved[:, 0] *= np.linspace(0.25, 3.0, moved.shape[0])
    mutated = _with_artifact(result, group, moved)
    before = _rows(result, group)
    after = _rows(mutated, group)
    labels = list(before)

    assert after[labels[0]].score_load != before[labels[0]].score_load
    assert after[labels[1]].score_load == before[labels[1]].score_load
    assert {name: _without_load(item) for name, item in after.items()} == {
        name: _without_load(item) for name, item in before.items()
    }


def test_a_regime_structural_zero_stays_in_the_score_mask_denominator(
    intervention_results: dict[str, Any],
) -> None:
    result = intervention_results["regime"]
    screen = result.diagnostics.support().regimes["screen"].score_load
    assert screen is not None
    assert screen["zero_load"] > 0.0
    assert screen["n_targeted"] == float(result.data.n)


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_complete_data_uses_every_fitted_score_mask_row(
    intervention_results: dict[str, Any], group: str
) -> None:
    result = intervention_results[group]
    for row in _rows(result, group).values():
        assert row.score_load is not None
        assert row.score_load["n_targeted"] == row.score_load["n_total"] == float(result.data.n)


def test_the_combined_row_reports_load_without_grading_or_pooling_it(
    intervention_results: dict[str, Any],
) -> None:
    result = intervention_results["ipsi"]
    direct = result.diagnostics.support()
    combined = result.diagnostics.run_all()["support"]
    assert combined.report is direct
    assert combined.status.value == "completed"
    assert "intervention load:" in combined.detail
    assert "not estimator ESS" in combined.detail
    assert "minimum effective-sample-size ratio" in combined.detail


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_an_older_fluctuation_keeps_support_and_records_the_load_omission(
    intervention_results: dict[str, Any], group: str
) -> None:
    result = intervention_results[group]
    repeat = result.repeats[0]
    fluctuations = dict(repeat.fluctuations)
    fluctuations[group] = dataclasses.replace(fluctuations[group], absolute_score_weights=None)
    legacy = dataclasses.replace(
        result,
        repeats=(dataclasses.replace(repeat, fluctuations=fluctuations),),
    )
    rows = _rows(legacy, group)
    assert rows
    assert all(support.score_load is None for support in rows.values())
    assert {support.score_load_omission for support in rows.values()} == {
        "the fitted artifact has no exact absolute score weights"
    }


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_a_report_pickled_before_the_new_fields_restores_explicit_defaults(
    intervention_results: dict[str, Any], group: str
) -> None:
    item = next(iter(_rows(intervention_results[group], group).values()))
    state = dict(item.__dict__)
    state.pop("score_load")
    state.pop("score_load_omission")
    restored = object.__new__(type(item))
    restored.__setstate__(state)
    assert restored.score_load is None
    assert restored.score_load_omission == "the report predates fitted score-load diagnostics"


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_cached_intervention_loads_replay_after_persistence(
    intervention_results: dict[str, Any], tmp_path: Any, group: str
) -> None:
    result = intervention_results[group]
    result.diagnostics.support()
    detail = result.diagnostics.run_all()["support"].detail
    restored = load(result.save(tmp_path / f"{group}-loads.joblib"))
    assert {name: row.score_load for name, row in _rows(restored, group).items()} == {
        name: row.score_load for name, row in _rows(result, group).items()
    }
    assert restored.diagnostics.run_all()["support"].detail == detail
    restored.assessment_cache.clear()
    assert {name: row.score_load for name, row in _rows(restored, group).items()} == {
        name: row.score_load for name, row in _rows(result, group).items()
    }


def test_a_repeated_fit_reports_the_first_draws_matching_support_artifact(
    intervention_results: dict[str, Any],
) -> None:
    original = intervention_results["ipsi"]
    artifact = original.fluctuations["ipsi"].absolute_score_weights
    assert artifact is not None
    second = np.asarray(artifact, dtype=float).copy()
    second[:, 0] *= np.linspace(0.25, 3.0, second.shape[0])
    changed_second = _with_artifact(original, "ipsi", second).repeats[0]
    repeated = dataclasses.replace(original, repeats=(original.repeats[0], changed_second))
    reported = {name: row.score_load for name, row in repeated.diagnostics.support().items()}
    first = {name: row.score_load for name, row in original.diagnostics.support().items()}
    second_rows = {
        name: row.score_load
        for name, row in _with_artifact(original, "ipsi", second).diagnostics.support().items()
    }
    assert reported == {
        name: {**values, "n_repeats": 2} for name, values in first.items() if values is not None
    }
    assert reported != second_rows


def test_missing_outcomes_use_the_fitted_score_mask_denominator() -> None:
    frame, _ = make_missing_outcome(n=400, seed=61)
    result = (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            missingness_learner=LogisticRegression(max_iter=1000),
            interventions=(Static(0, name="none"), Static(1, name="all")),
            n_folds=3,
            learner_folds=2,
            random_state=61,
            simultaneous=False,
        )
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )
    load = result.diagnostics.support().regimes["all"].score_load
    assert load is not None
    assert load["n_targeted"] == float(result.data.observed.sum())
    assert load["n_targeted"] < load["n_total"]
    assert load["total_ratio"] < load["targeted_ratio"]


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_an_impossible_score_mask_size_records_an_omission(
    intervention_results: dict[str, Any], group: str
) -> None:
    result = intervention_results[group]
    artifact = result.fluctuations[group].absolute_score_weights
    assert artifact is not None
    malformed = np.vstack([artifact, artifact[:1]])
    rows = _rows(_with_artifact(result, group, malformed), group)
    assert all(row.score_load is None for row in rows.values())
    assert {row.score_load_omission for row in rows.values()} == {
        "the fitted score mask size is outside the fitted data"
    }


@pytest.fixture(scope="module")
def backend_results() -> dict[str, tuple[Any, Any]]:
    point_pandas, _ = make_nonlinear_ate(n=300, seed=31, backend="pandas")
    point_polars, _ = make_nonlinear_ate(n=300, seed=31, backend="polars")
    dose_pandas, _ = make_shift_dose(n=300, seed=32, backend="pandas")
    dose_polars, _ = make_shift_dose(n=300, seed=32, backend="polars")

    regime_settings = {
        "random_state": 31,
        "weighted": False,
        "interventions": (Static(0, name="none"), Static(1, name="all")),
    }
    shift_settings = {
        "random_state": 32,
        "weighted": False,
        "shifts": (
            Shift(0.0, cap=None, name="current"),
            Shift(0.5, cap=5.0, name="up"),
        ),
        "density_bins": 20,
    }
    incremental_settings = {
        "random_state": 31,
        "weighted": False,
        "incremental": (
            Incremental(0.5, name="down"),
            Incremental(2.0, name="up"),
        ),
    }
    return {
        "regime": (
            _fit(point_pandas, **regime_settings),
            _fit(point_polars, **regime_settings),
        ),
        "mtp": (
            _fit(dose_pandas, **shift_settings),
            _fit(dose_polars, **shift_settings),
        ),
        "ipsi": (
            _fit(point_pandas, **incremental_settings),
            _fit(point_polars, **incremental_settings),
        ),
    }


@pytest.mark.parametrize("group", ["regime", "mtp", "ipsi"])
def test_intervention_loads_are_identical_for_pandas_and_polars(
    backend_results: dict[str, tuple[Any, Any]], group: str
) -> None:
    from_pandas, from_polars = backend_results[group]
    pandas_rows = _rows(from_pandas, group)
    polars_rows = _rows(from_polars, group)
    assert {name: row.score_load for name, row in pandas_rows.items()} == {
        name: row.score_load for name, row in polars_rows.items()
    }

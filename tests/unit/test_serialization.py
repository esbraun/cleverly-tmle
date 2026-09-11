"""Whole-result joblib persistence."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import io
import math
import pkgutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import FunctionTransformer

import cleverly
from cleverly import (
    ATE,
    CapabilityError,
    CausalStudy,
    CounterfactualMean,
    LongitudinalTreatment,
    ModifiedTreatmentPolicyEffect,
    PointTreatment,
    RegimeContrast,
    RiskRatio,
    load,
)
from cleverly.datasets import (
    make_binary_outcome,
    make_longitudinal,
    make_nonlinear_ate,
    make_shift_dose,
)
from cleverly.estimators.serialize import dumps, loads, save
from cleverly.interventions import Shift
from cleverly.sensitivity import ConfounderStrengthGrid
from cleverly.validation import (
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    GaussianIndependentOutcome,
    RelativeGaussianNoise,
)
from cleverly.validation.score import DEFAULT_TOLERANCE


def _assert_same_graph(left: Any, right: Any, *, path: str = "result") -> None:
    """Compare every stored field without relying on array-hostile dataclass equality.

    Walk :func:`dataclasses.fields` rather than call ``==``, which ignores every
    ``compare=False`` field. ``AssessmentItem._report`` is one of those, so a
    payload-blind ``assert cached == report`` cannot see a diagnostic payload that the
    round trip dropped.
    """
    assert type(left) is type(right), path
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path, strict=True)
        return
    if isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True, obj=path)
        return
    # A Series and an Index reach the final ``==`` as an elementwise comparison, whose
    # truth value raises even when the two are equal.
    if isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, check_exact=True, obj=path)
        return
    if isinstance(left, pd.Index):
        pd.testing.assert_index_equal(left, right, exact=True, obj=path)
        return
    if isinstance(left, BaseEstimator):
        _assert_same_graph(
            left.get_params(deep=False), right.get_params(deep=False), path=f"{path}.parameters"
        )
        _assert_same_graph(vars(left), vars(right), path=f"{path}.state")
        return
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        for field in dataclasses.fields(left):
            _assert_same_graph(
                getattr(left, field.name),
                getattr(right, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(left, Mapping):
        assert tuple(left) == tuple(right), path
        for key in left:
            _assert_same_graph(left[key], right[key], path=f"{path}[{key!r}]")
        return
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_same_graph(left_item, right_item, path=f"{path}[{index}]")
        return
    if type(left).__module__.startswith("cleverly") and hasattr(left, "__dict__"):
        _assert_same_graph(vars(left), vars(right), path=path)
        return
    # ``np.float64`` subclasses ``float`` and ``np.float32`` does not, so the plain
    # ``float`` guard would send a NaN of the narrower width to ``NaN == NaN``.
    if isinstance(left, (float, np.floating)) and math.isnan(left):
        assert math.isnan(right), path
        return
    assert left == right, path


def _capability_refusal(result: Any, operation: str) -> str:
    with pytest.raises(CapabilityError) as refusal:
        getattr(result.sensitivity, operation)()
    return str(refusal.value)


@pytest.fixture
def point_result():  # type: ignore[no-untyped-def]
    frame, _ = make_nonlinear_ate(n=300, seed=7)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
        ),
    )
    return study.estimate(
        ATE(),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=7,
        simultaneous=False,
    )


@pytest.fixture
def binary_mean_result():  # type: ignore[no-untyped-def]
    frame, _ = make_nonlinear_ate(n=300, seed=7)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
        ),
    )
    return study.estimate(
        CounterfactualMean(treatment=1),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=7,
        simultaneous=False,
    )


@pytest.fixture
def risk_ratio_result():  # type: ignore[no-untyped-def]
    frame, _ = make_binary_outcome(n=300, seed=7)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
        ),
    )
    return study.estimate(
        RiskRatio(),
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=7,
        simultaneous=False,
    )


@pytest.fixture
def categorical_point_result():  # type: ignore[no-untyped-def]
    """A fit whose adjustment set carries a three-level categorical variable.

    ``bootstrap_measurement_error`` perturbs a numeric variable and a categorical block by
    two different mechanisms, so a round-trip check needs one of each to exercise both.
    """
    frame, _ = make_nonlinear_ate(n=300, seed=7)
    frame = frame.assign(G=np.resize(["a", "b", "c"], len(frame)))
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4", "G"),
        ),
    )
    return study.estimate(
        ATE(),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=7,
        simultaneous=False,
    )


@pytest.fixture
def continuous_result():  # type: ignore[no-untyped-def]
    frame, _ = make_shift_dose(n=120, seed=7)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
        ),
    )
    return study.identify(
        ModifiedTreatmentPolicyEffect(
            shifts=(
                Shift(0.0, cap=10.0, name="natural course"),
                Shift(0.5, cap=10.0, name="up half"),
            )
        )
    ).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=7,
        simultaneous=False,
    )


def test_file_round_trip_retains_the_complete_point_result(point_result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = point_result.save(tmp_path / "analysis.joblib")
    restored = load(path)

    # Every stored field, so the name holds as the result grows one. A point fit stores
    # its learners and its solved corrections here, which psi and one influence curve
    # leave unread.
    for field in dataclasses.fields(point_result):
        _assert_same_graph(
            getattr(point_result, field.name), getattr(restored, field.name), path=field.name
        )

    # The learner classes the fixture asked for, which the field walk only checks agree.
    assert isinstance(restored.estimator.outcome_learner, LinearRegression)
    assert isinstance(restored.estimator.treatment_learner, LogisticRegression)
    assert restored.replayability.refit_nuisances


def test_loaded_result_can_refit_nuisances(point_result, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    restored = load(point_result.save(tmp_path / "refit.joblib"))
    # The fixture's own seed, so the gate is a fixed draw rather than one sample of a null.
    # One placebo replicate is 3 sigma of a null draw, which fails for 1 of the seeds 0-299;
    # raising ``n_replicates`` buys no margin, because the gate is 3 sigma at every k below 5.
    report = restored.diagnostics.refute(n_replicates=1, tests=["placebo"], random_state=7)
    assert report.passed


def test_generated_outcome_cache_and_records_survive_round_trip(point_result) -> None:  # type: ignore[no-untyped-def]
    kwargs = {
        "tests": ("dummy_outcome",),
        "dummy_outcome": GaussianIndependentOutcome(),
        "n_replicates": 4,
        # The rule refuses minimum_draws * alpha < 2, so a four-draw budget needs alpha=0.5.
        "outcome_rule": EmpiricalInclusionRule(alpha=0.5, minimum_draws=4),
        "random_state": 17,
    }
    report = point_result.diagnostics.refute(**kwargs)
    assert point_result.diagnostics.refute(**kwargs) is report

    restored = loads(dumps(point_result))
    cached = restored.diagnostics.refute(**kwargs)
    assert cached == report
    assert cached["dummy_outcome"].child_seeds == report["dummy_outcome"].child_seeds


def test_simulated_confounding_cache_survives_round_trip(point_result) -> None:  # type: ignore[no-untyped-def]
    # Nonzero on both perturbation paths, and one benchmark covariate. An all-zero grid
    # is one anchor cell that never refits, and no benchmark covariate calibrates
    # nothing, so the round trip would agree even if the restored fit refit no cell and
    # calibrated no covariate.
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.3))
    kwargs = {"grid": grid, "benchmark_covariates": ("W1",), "random_state": 17}
    report = point_result.sensitivity.simulated_confounding(**kwargs)
    assert point_result.sensitivity.simulated_confounding(**kwargs) is report

    # The witness that the recorded cells carry a refit and a calibration at all.
    assert report.complete
    assert len(report.cells) == 4
    assert report.cells[0].estimate == point_result["ate"].psi
    assert report.cells[0].displacement == 0.0
    assert any(abs(cell.displacement) > 0.05 for cell in report.cells[1:])
    assert len(report.calibrations) == 2
    assert all(row.strength != 0.0 for row in report.calibrations)

    restored = loads(dumps(point_result))
    replayed = restored.sensitivity.simulated_confounding(**kwargs)
    assert replayed == report
    assert replayed.cells == report.cells
    assert replayed.calibrations == report.calibrations
    assert replayed.latent_seed == report.latent_seed
    assert replayed.refit_seed == report.refit_seed == report.root_seed


def test_binary_mean_simulated_confounding_cache_survives_round_trip(
    binary_mean_result,
) -> None:  # type: ignore[no-untyped-def]
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.3)),
        "random_state": 17,
    }
    report = binary_mean_result.sensitivity.simulated_confounding(**kwargs)
    assert binary_mean_result.sensitivity.simulated_confounding(**kwargs) is report
    assert report.complete
    assert report.estimand == "ey1"
    assert any(abs(cell.displacement or 0.0) > 0.01 for cell in report.cells[1:])

    restored = loads(dumps(binary_mean_result))
    replayed = restored.sensitivity.simulated_confounding(**kwargs)
    assert replayed == report
    assert replayed.cells == report.cells
    assert replayed.latent_seed == report.latent_seed


def test_ratio_simulated_confounding_cache_survives_round_trip(
    risk_ratio_result,
) -> None:  # type: ignore[no-untyped-def]
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.3)),
        "random_state": 17,
    }
    report = risk_ratio_result.sensitivity.simulated_confounding(**kwargs)
    assert risk_ratio_result.sensitivity.simulated_confounding(**kwargs) is report
    assert report.complete
    assert report.estimand == "rr"
    assert report.movement_scale == "log_ratio"
    assert all(report.to_frame()["movement_scale"] == "log_ratio")
    assert any(abs(cell.displacement or 0.0) > 0.01 for cell in report.cells[1:])

    restored = loads(dumps(risk_ratio_result))
    replayed = restored.sensitivity.simulated_confounding(**kwargs)
    assert replayed == report
    assert replayed.cells == report.cells
    assert replayed.movement_scale == report.movement_scale
    assert replayed.latent_seed == report.latent_seed


def test_continuous_simulated_confounding_cache_survives_round_trip(
    continuous_result,
) -> None:  # type: ignore[no-untyped-def]
    alias = next(name for name in continuous_result.estimates if name.startswith("ate_shift["))
    kwargs = {
        "estimand": alias,
        "grid": ConfounderStrengthGrid(treatment=(0.0, -0.2), outcome=(0.0, 0.3)),
        "benchmark_covariates": ("W1",),
        "random_state": 17,
    }
    report = continuous_result.sensitivity.simulated_confounding(**kwargs)
    assert continuous_result.sensitivity.simulated_confounding(**kwargs) is report
    assert report.complete
    assert report.treatment_family == "continuous"
    assert len(report.cells) == 4
    assert any(abs(cell.displacement or 0.0) > 1e-4 for cell in report.cells[1:])
    assert report.calibrations[0].family == "gaussian"

    restored = loads(dumps(continuous_result))
    replayed = restored.sensitivity.simulated_confounding(**kwargs)
    assert replayed == report
    assert replayed.cells == report.cells
    assert replayed.calibrations == report.calibrations
    assert replayed.latent_seed == report.latent_seed


def test_measurement_error_cache_and_records_survive_round_trip(categorical_point_result) -> None:  # type: ignore[no-untyped-def]
    # Nonzero on both perturbation paths. A declaration with zero noise and no categorical
    # variable perturbs nothing, so the round trip would agree even if the restored fit
    # applied no measurement error at all.
    active = BootstrapMeasurementError(
        ("W1", "G"),
        numeric_noise=RelativeGaussianNoise(0.5),
        categorical_change_probability=0.4,
    )
    kwargs = {
        "tests": ("bootstrap_measurement_error",),
        "bootstrap_measurement_error": active,
        "n_replicates": 4,
        "measurement_error_rule": EmpiricalInclusionRule(alpha=0.5, minimum_draws=4),
        "random_state": 17,
    }
    report = categorical_point_result.diagnostics.refute(**kwargs)
    assert categorical_point_result.diagnostics.refute(**kwargs) is report
    assert categorical_point_result["ate"].psi == report["bootstrap_measurement_error"].original

    # The witness that the recorded draws carry a perturbation at all.
    unperturbed = categorical_point_result.diagnostics.refute(
        **{
            **kwargs,
            "bootstrap_measurement_error": BootstrapMeasurementError(
                ("W1", "G"),
                numeric_noise=RelativeGaussianNoise(0.0),
                categorical_change_probability=0.0,
            ),
        }
    )["bootstrap_measurement_error"]
    assert report["bootstrap_measurement_error"].values != unperturbed.values

    restored = loads(dumps(categorical_point_result))
    cached = restored.diagnostics.refute(**kwargs)
    assert cached == report


def test_byte_round_trip_matches_file_round_trip(point_result) -> None:  # type: ignore[no-untyped-def]
    restored = loads(dumps(point_result))
    assert restored["ate"].psi == pytest.approx(point_result["ate"].psi)
    assert restored.parameter_keys == point_result.parameter_keys


def test_longitudinal_result_retains_the_complete_fitted_graph_and_assessment(
    tmp_path: Path,
) -> None:
    frame, _ = make_longitudinal(n=180, seed=11)
    frame = frame.assign(w=np.linspace(0.5, 2.0, len(frame)))
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
            weights="w",
        ),
    )
    original = study.estimate(
        # A third regimen, written as one arm per treatment node, so the fit reports two
        # contrasts. Two regimens report one, and one parameter leaves ``simultaneous``
        # None, which is the band this test needs to round-trip. The contrast estimand
        # also gives the only ``ParameterKey.reference`` that is not None.
        RegimeContrast({"always": 1, "never": 0, "late": (0, 1)}, reference="always"),
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=0,
        simultaneous=True,
    )

    validation = original.validate()
    diagnostics = original.diagnostics.run_all()
    assessment = original.assess()
    refusal = _capability_refusal(original, "omitted_confounding")

    cache_operations = {key.split(":", 1)[0] for key in original.assessment_cache}
    assert {"validate", "diagnostics.run_all"} <= cache_operations
    cached_run_all = next(
        value
        for key, value in original.assessment_cache.items()
        if key.startswith("diagnostics.run_all:")
    )
    packed_payloads: list[Any] = [
        item._report for item in cached_run_all.items if dataclasses.is_dataclass(item._report)
    ]
    assert {
        "LongitudinalDiagnostics",
        "LongitudinalNuisanceDiagnostics",
        "LongitudinalScoreDiagnostics",
    } <= {type(payload).__name__ for payload in packed_payloads}
    # No ``hasattr`` guard: a renamed ``rows`` must raise here rather than empty the
    # selection and leave ``all([])`` reading as a pass.
    assert all(payload.rows for payload in packed_payloads)
    assert original.folds.n_folds == 3
    assert np.unique(original.folds.assignment).size == 3
    assert original.simultaneous is not None
    assert not np.all(original.data.uncensored)
    assert not np.allclose(original.data.weights, 1.0)
    assert any(
        np.any(np.abs(step.fluctuation.epsilon) > 1e-8)
        for fit in original.fits.values()
        for step in fit.steps
    )
    assert any(
        np.any(step.trained_on != step.at_risk)
        for fit in original.fits.values()
        for step in fit.steps
    )

    restored = load(original.save(tmp_path / "longitudinal.joblib"))

    result_fields = {field.name for field in dataclasses.fields(original)}
    assert result_fields == {
        "assessment_cache",
        "config",
        "data",
        "estimates",
        "fitted_method",
        "fits",
        "folds",
        "identified_effect",
        "mechanism",
        "method",
        "msm",
        "msm_fits",
        "parameter_index",
        "parameter_keys",
        "provenance",
        "replay_recipe",
        "scaler",
        "simultaneous",
    }
    for field in result_fields - {"assessment_cache"}:
        _assert_same_graph(getattr(original, field), getattr(restored, field), path=field)

    _assert_same_graph(
        original.assessment_cache, restored.assessment_cache, path="assessment_cache"
    )
    # A restored result answers from its own cache rather than recomputing, so the key the
    # pickle carried still resolves against the operation that wrote it.
    cached_validation = next(
        value for key, value in restored.assessment_cache.items() if key.startswith("validate:")
    )
    assert restored.validate() is cached_validation

    # ``restored.validate()``, ``restored.diagnostics.run_all()`` and ``restored.assess()``
    # are cache hits, so they return the unpickled objects the comparison above already
    # covered. ``dataclasses.replace`` copies the restored graph with an empty cache,
    # because ``assessment_cache`` is ``init=False``, and that copy has to recompute each
    # operation from the restored nuisances and targeting steps. Do not move these calls
    # back onto ``restored``, which would delete the only recomputation this test does.
    recomputed = dataclasses.replace(restored)
    assert recomputed.assessment_cache == {}
    _assert_same_graph(validation, recomputed.validate(), path="validation")
    _assert_same_graph(diagnostics, recomputed.diagnostics.run_all(), path="diagnostics")
    _assert_same_graph(assessment, recomputed.assess(), path="assessment")
    _assert_same_graph(original.replayability, restored.replayability, path="replayability")
    assert restored.replayability.refit_nuisances
    assert _capability_refusal(restored, "omitted_confounding") == refusal


def test_legacy_npz_is_refused_with_a_migration_message(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, __manifest__=np.array([1], dtype=np.uint8))
    with pytest.raises(ValueError, match=r"legacy cleverly \.npz"):
        load(path)


def test_legacy_npz_bytes_are_refused() -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, __manifest__=np.array([1], dtype=np.uint8))
    with pytest.raises(ValueError, match=r"legacy cleverly \.npz"):
        loads(buffer.getvalue())


def test_non_result_objects_are_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="fitted causal result"):
        save({"not": "a result"}, tmp_path / "bad.joblib")

    path = tmp_path / "bad-load.joblib"
    joblib.dump({"not": "a result"}, path)
    with pytest.raises(TypeError, match="fitted causal result"):
        load(path)


def test_non_picklable_components_fail_at_save(point_result) -> None:  # type: ignore[no-untyped-def]
    point_result.estimator.outcome_learner = FunctionTransformer(lambda values: values)
    with pytest.raises(TypeError, match="not joblib-serializable"):
        dumps(point_result)


# ------------------------------------------------------------ derived values


#: Every class in the package that owns a ``cached_property`` and that no artifact can
#: reach, with the reason it cannot.  A class named here needs no memo filter. A class
#: that becomes reachable needs the filter rather than an entry.
_UNREACHABLE_MEMO_OWNERS = {
    "cleverly.targets.base.TargetContext": (
        "built inside the estimator's per-group estimate loop, read by the target "
        "builders, and dropped when that loop ends. Each estimate keeps the arrays it "
        "computed and not the context that computed them, which "
        "`test_no_saved_result_carries_a_value_it_derived` witnesses on a real fit."
    ),
}


def _memo_names(owner: type) -> tuple[str, ...]:
    """The attribute names a ``cached_property`` declared on ``owner`` writes."""
    return tuple(
        name
        for name, attribute in vars(owner).items()
        if isinstance(attribute, functools.cached_property)
    )


def _memo_owners() -> dict[str, type]:
    """Every shipped class that declares a ``cached_property``, keyed by import path.

    Walking the package rather than naming the classes, because a list of names passes
    the day a new one is added, which is the day the artifact written by that version is
    already wrong.
    """
    modules = [cleverly]
    for found in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
        modules.append(importlib.import_module(found.name))
    owners: dict[str, type] = {}
    for module in modules:
        for obj in vars(module).values():
            if not inspect.isclass(obj) or not obj.__module__.startswith("cleverly"):
                continue
            if _memo_names(obj):
                owners[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return owners


def _memos_in_graph(root: Any) -> list[str]:
    """Every ``cached_property`` memo a shipped object still holds under ``root``.

    Reads the descriptors on each visited object's own class, so the answer stays true
    for a memo added after this walk was written. Only a class this package ships can
    report one, because a dependency's memo is that dependency's business and pandas
    declares several.
    """
    found: list[str] = []
    seen: set[int] = set()
    stack: list[tuple[Any, str]] = [(root, "result")]
    while stack:
        value, path = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value is None or isinstance(value, (str, bytes, bytearray, int, float, complex)):
            continue
        # Neither carries a fitted result, and both are deep enough to dominate the walk.
        if type(value).__module__.split(".")[0] in {"numpy", "pandas", "polars"}:
            continue
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            for name, item in state.items():
                if type(value).__module__.startswith("cleverly") and isinstance(
                    getattr(type(value), name, None), functools.cached_property
                ):
                    found.append(f"{path}.{name}")
                stack.append((item, f"{path}.{name}"))
        if isinstance(value, Mapping):
            for key, item in value.items():
                stack.append((item, f"{path}[{key!r}]"))
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                stack.append((item, f"{path}[{index}]"))
    return found


def test_every_memo_owner_either_drops_its_memos_or_reaches_no_artifact() -> None:
    """A ``cached_property`` on a persisted class is a stale verdict waiting to be written.

    ``save`` pickles the result whole and the default protocol writes the instance
    dictionary out verbatim, so every memo a caller warmed becomes part of the file. The
    sweep is the point: naming the three properties that were wrong would pass on the day
    a fourth is added, and the artifact written that day already carries it.

    A class reaches this assertion unless ``_UNREACHABLE_MEMO_OWNERS`` says why no
    artifact can hold it. Each exemption is checked against the walk, so a renamed class
    cannot leave a silent one behind.
    """
    owners = _memo_owners()

    assert "cleverly.estimators.base.TMLEResult" in owners, "check the package walk"
    assert set(_UNREACHABLE_MEMO_OWNERS) <= set(owners), (
        "an exemption names no shipped memo owner; rename it with the class, or delete it"
    )

    for path, owner in sorted(owners.items()):
        if path in _UNREACHABLE_MEMO_OWNERS:
            continue
        names = _memo_names(owner)
        poisoned = dict.fromkeys(names, "a verdict from a version that is not this one")

        written = owner.__new__(owner)
        written.__dict__.update(poisoned)
        assert not set(names) & set(written.__getstate__()), (
            f"{path} writes a memo into every artifact it is saved in"
        )

        read = owner.__new__(owner)
        read.__setstate__(poisoned)  # type: ignore[attr-defined]
        assert not set(names) & set(read.__dict__), (
            f"{path} restores whatever memo an older artifact already carries"
        )


def test_no_saved_result_carries_a_value_it_derived(point_result) -> None:  # type: ignore[no-untyped-def]
    """The whole result graph, warmed as a caller warms it, then round-tripped.

    ``fit``, ``summary()``, ``save()`` is the ordinary order, and it wrote
    ``score_verdict`` into the file. The facades warmed here reach the memos ff5559e
    filtered, and the walk reaches every other object the artifact holds, including the
    one ``_UNREACHABLE_MEMO_OWNERS`` claims no artifact can reach.

    The warmed count is asserted first, because a walk that finds nothing before the
    round trip would find nothing after it for the wrong reason.
    """
    point_result.summary()
    point_result.sensitivity.run_all()
    assert point_result.diagnostics.capabilities

    warmed = _memos_in_graph(point_result)
    assert "result.score_verdict" in warmed
    assert "result.sensitivity" in warmed
    assert "result.diagnostics" in warmed
    assert any(name.endswith("._capability_map") for name in warmed)

    assert _memos_in_graph(loads(dumps(point_result))) == []


def test_a_loaded_result_recomputes_the_score_verdict_an_artifact_carried(point_result) -> None:  # type: ignore[no-untyped-def]
    """A stale verdict inside the file is the migration case, and it is not diagnosable.

    Nothing invalidates a memo. There is no generation counter to compare, because a memo
    records no question, so whatever the artifact carries is restored and reported. A
    ``ScoreCheck`` saved at one tolerance answers at that tolerance for ever, and an
    object of the wrong type surfaces as an ``AttributeError`` from ``summary()`` rather
    than as anything a reader can act on.

    A same-version round trip pins the write side. The revived states below pin the read
    side, and need no committed binary artifact.
    """
    fresh = point_result.score_verdict
    assert fresh.tolerance == pytest.approx(DEFAULT_TOLERANCE)
    assert "score_verdict" in point_result.__dict__

    state = point_result.__getstate__()
    assert "score_verdict" not in state
    assert state["estimates"] is point_result.estimates

    restored = loads(dumps(point_result))
    assert "score_verdict" not in restored.__dict__
    assert restored.score_verdict.tolerance == pytest.approx(DEFAULT_TOLERANCE)
    assert restored.score_verdict.rows == fresh.rows

    # A verdict this version does not reach, at a tolerance no caller asked for.
    stale = dict(state)
    stale["score_verdict"] = dataclasses.replace(
        fresh, tolerance=1e-12, corrected=not fresh.corrected
    )
    revived = type(point_result).__new__(type(point_result))
    revived.__setstate__(stale)
    assert "score_verdict" not in revived.__dict__
    assert revived.score_verdict.tolerance == pytest.approx(DEFAULT_TOLERANCE)
    assert revived.score_verdict.corrected == fresh.corrected

    # Not merely stale: an artifact restores whatever it holds, of whatever type.
    poisoned = dict(state)
    poisoned["score_verdict"] = "a verdict from a version that is not this one"
    healed = type(point_result).__new__(type(point_result))
    healed.__setstate__(poisoned)
    assert healed.score_verdict.passed == fresh.passed
    assert healed.summary() == point_result.summary()


def test_a_loaded_result_rebuilds_the_facades_it_was_saved_with(point_result) -> None:  # type: ignore[no-untyped-def]
    """The two facades are memos as well, and a saved one is this version's object graph.

    Each facade already drops its own verdicts, so what an artifact pinned here is a
    derived object rather than a stale answer. It is dropped for the same reason and at
    the same cost: one constructor call, against a class whose shape the file cannot
    check.
    """
    assert point_result.sensitivity is point_result.sensitivity
    assert point_result.diagnostics is point_result.diagnostics

    state = point_result.__getstate__()
    assert "sensitivity" not in state
    assert "diagnostics" not in state

    restored = loads(dumps(point_result))
    assert "sensitivity" not in restored.__dict__
    assert "diagnostics" not in restored.__dict__
    assert restored.sensitivity._result is restored
    assert restored.diagnostics._result is restored
    assert restored.sensitivity.capabilities == point_result.sensitivity.capabilities

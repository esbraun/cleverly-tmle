"""Bootstrap measurement-error validation and its deliberate mutation controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import CapabilityError, DataError
from cleverly.inference import run_bootstrap
from cleverly.validation import (
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    GeneratedOutcomeRecord,
    RelativeGaussianNoise,
    refute,
)


def _rule(draws: int) -> EmpiricalInclusionRule:
    return EmpiricalInclusionRule(alpha=2.0 / draws, minimum_draws=draws)


def _child_seeds(seed: int, draws: int) -> tuple[int, ...]:
    return tuple(
        int(sequence.generate_state(1)[0]) for sequence in np.random.SeedSequence(seed).spawn(draws)
    )


def _categorical_perturbation(
    data: CausalData,
    declaration: BootstrapMeasurementError,
    seed: int,
    *,
    deterministic_alternatives: bool = False,
    reuse_change_mask: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    expected = data.covariates.copy()
    first_mask: np.ndarray | None = None
    for encoding in data.encodings:
        if encoding.column not in declaration.variables:
            continue
        columns = [data.covariate_names.index(name) for name in encoding.generated]
        block = data.covariates[:, columns]
        total = np.sum(block, axis=1)
        codes = np.where(total == 0.0, 0, np.argmax(block, axis=1) + 1)
        drawn_mask = rng.random(data.n) < declaration.categorical_change_probability
        change = first_mask if reuse_change_mask and first_mask is not None else drawn_mask
        first_mask = change
        if deterministic_alternatives:
            codes[change] = (codes[change] + 1) % len(encoding.levels)
        else:
            alternatives = rng.integers(0, len(encoding.levels) - 1, size=int(np.sum(change)))
            current = codes[change]
            codes[change] = alternatives + (alternatives >= current)
        expected[:, columns] = np.column_stack(
            [np.asarray(codes == code, dtype=float) for code in range(1, len(encoding.levels))]
        )
    return expected


@dataclass
class _RecordingEstimator:
    random_state: int | None = 0
    fail_calls: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.calls: list[tuple[int, CausalData]] = []

    def refit(
        self,
        data: CausalData,
        *,
        intermediate_value: float | None,
        random_state: int,
    ) -> Any:
        del intermediate_value
        call = len(self.calls)
        self.calls.append((random_state, data))
        if call in self.fail_calls:
            raise RuntimeError(f"failed draw {call}")
        estimate = SimpleNamespace(psi=float(np.mean(data.covariates[:, 0])), std_error=1.0)
        return _RefitResult(data, estimate)


class _RefitResult:
    def __init__(self, data: CausalData, estimate: Any) -> None:
        self.data = data
        self.estimate = estimate

    def __getitem__(self, name: str) -> Any:
        assert name == "ate"
        return self.estimate


class _Result(SimpleNamespace):
    def __getitem__(self, name: str) -> Any:
        return self.estimates[name]


def _data(*, backend: str = "pandas", clusters: bool = False, strata: bool = False) -> CausalData:
    n = 60
    payload: dict[str, Any] = {
        "Y": np.linspace(-1.0, 1.0, n),
        "A": np.tile([0, 1], n // 2),
        "numeric": np.arange(n, dtype=float),
        "flag": np.tile([False, True, True], n // 3),
        "group": np.tile(["a", "b", "c"], n // 3),
    }
    if clusters:
        payload["id"] = np.repeat(np.arange(20), 3)
    frame = pl.DataFrame(payload) if backend == "polars" else pd.DataFrame(payload)
    return CausalData.from_frame(
        frame,
        outcome="Y",
        treatment="A",
        covariates=("numeric", "flag", "group"),
        id="id" if clusters else None,
        strata=("group",) if strata else None,
    )


def _result(data: CausalData, estimator: _RecordingEstimator | None = None) -> _Result:
    return _Result(
        estimator=estimator or _RecordingEstimator(),
        estimates={
            "ate": SimpleNamespace(psi=float(np.mean(data.covariates[:, 0])), std_error=1.0)
        },
        data=data,
        intermediate_value=None,
        assessment_cache={},
    )


def _run(
    data: CausalData,
    declaration: BootstrapMeasurementError,
    *,
    draws: int = 4,
    estimator: _RecordingEstimator | None = None,
    seed: int = 17,
) -> tuple[Any, _RecordingEstimator]:
    fitted = estimator or _RecordingEstimator()
    report = refute(
        _result(data, fitted),
        tests=("bootstrap_measurement_error",),
        n_replicates=draws,
        bootstrap_measurement_error=declaration,
        measurement_error_rule=_rule(draws),
        random_state=seed,
    )
    return report, fitted


class TestDeclarationsAndReplacement:
    def test_declarations_are_immutable_and_validate_values(self) -> None:
        declaration = BootstrapMeasurementError(("numeric",))
        normalized = BootstrapMeasurementError(["numeric"])  # type: ignore[arg-type]
        assert normalized.variables == ("numeric",)
        with pytest.raises(FrozenInstanceError):
            declaration.resampling = "iid"  # type: ignore[misc]
        with pytest.raises(ValueError, match="must not be empty"):
            BootstrapMeasurementError(())
        with pytest.raises(ValueError, match="must not repeat"):
            BootstrapMeasurementError(("numeric", "numeric"))
        with pytest.raises(ValueError, match="between zero and one"):
            BootstrapMeasurementError(("numeric",), categorical_change_probability=1.1)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            RelativeGaussianNoise(-0.1)
        with pytest.raises(ValueError, match="resampling"):
            BootstrapMeasurementError(("numeric",), resampling="rows")  # type: ignore[arg-type]

    def test_complete_covariate_replacement_validates_shape_and_finiteness(self) -> None:
        data = _data()
        with pytest.raises(DataError, match="shape"):
            data.with_covariates(data.covariates[:, :-1])
        invalid = data.covariates.copy()
        invalid[0, 0] = np.nan
        with pytest.raises(DataError, match="non-finite"):
            data.with_covariates(invalid)
        replaced = data.with_covariates(data.covariates.copy())
        assert replaced.encodings == data.encodings
        assert replaced.treatment is data.treatment


class TestBootstrapAndPerturbation:
    def test_zero_noise_uses_exact_plain_bootstrap_samples(self) -> None:
        data = _data()
        plain: list[np.ndarray] = []
        run_bootstrap(
            data,
            lambda sample: plain.append(sample.covariates.copy()) or {"ate": 0.0},
            n_replicates=4,
            random_state=17,
        )
        declaration = BootstrapMeasurementError(
            ("numeric", "flag", "group"),
            numeric_noise=RelativeGaussianNoise(0.0),
            categorical_change_probability=0.0,
        )
        _, estimator = _run(data, declaration)
        assert len(plain) == len(estimator.calls)
        for expected, (_, actual) in zip(plain, estimator.calls, strict=True):
            assert np.array_equal(actual.covariates, expected)

    def test_numeric_noise_uses_each_sample_scale_and_recorded_child_seed(self) -> None:
        data = _data()
        active = BootstrapMeasurementError(("numeric",), numeric_noise=RelativeGaussianNoise(0.5))
        report, active_estimator = _run(data, active)
        zero = BootstrapMeasurementError(("numeric",), numeric_noise=RelativeGaussianNoise(0.0))
        _, zero_estimator = _run(data, zero)
        test = report["bootstrap_measurement_error"]
        assert test.child_seeds == _child_seeds(17, 4)
        constant_seed = test.child_seeds[0]
        original_scale = float(np.std(data.covariates[:, 0], ddof=0))
        for child_seed, (_, active_data), (_, sample) in zip(
            test.child_seeds, active_estimator.calls, zero_estimator.calls, strict=True
        ):
            values = sample.covariates[:, 0]
            sample_scale = float(np.std(values, ddof=0))
            expected = np.random.default_rng(child_seed).normal(
                0.0,
                0.5 * sample_scale,
                size=sample.n,
            )
            assert np.array_equal(active_data.covariates[:, 0], values + expected)
            original_scale_mutation = np.random.default_rng(child_seed).normal(
                0.0, 0.5 * original_scale, size=sample.n
            )
            assert not np.array_equal(
                active_data.covariates[:, 0], values + original_scale_mutation
            )
            if child_seed != constant_seed:
                constant_seed_mutation = np.random.default_rng(constant_seed).normal(
                    0.0, 0.5 * sample_scale, size=sample.n
                )
                assert not np.array_equal(
                    active_data.covariates[:, 0], values + constant_seed_mutation
                )

    def test_boolean_and_multilevel_changes_rebuild_valid_indicator_blocks(self) -> None:
        data = _data()
        declaration = BootstrapMeasurementError(
            ("flag", "group"), categorical_change_probability=1.0
        )
        _, estimator = _run(data, declaration)
        zero = BootstrapMeasurementError(("flag", "group"), categorical_change_probability=0.0)
        _, zero_estimator = _run(data, zero)
        for (_, sample), (_, unperturbed) in zip(
            estimator.calls, zero_estimator.calls, strict=True
        ):
            flag = sample.covariates[:, sample.covariate_names.index("flag")]
            columns = [
                sample.covariate_names.index("group__b"),
                sample.covariate_names.index("group__c"),
            ]
            group = sample.covariates[:, columns]
            original_flag = unperturbed.covariates[:, unperturbed.covariate_names.index("flag")]
            original_group = unperturbed.covariates[:, columns]
            assert np.all(np.isin(flag, (0.0, 1.0)))
            assert np.all(np.isin(group, (0.0, 1.0)))
            assert np.all(np.sum(group, axis=1) <= 1.0)
            assert np.all(flag != original_flag)
            group_codes = np.where(np.sum(group, axis=1) == 0.0, 0, np.argmax(group, axis=1) + 1)
            original_codes = np.where(
                np.sum(original_group, axis=1) == 0.0,
                0,
                np.argmax(original_group, axis=1) + 1,
            )
            assert np.all(group_codes != original_codes)

    def test_intermediate_categorical_changes_use_seeded_masks_and_alternatives(self) -> None:
        data = _data()
        probability = 0.35
        active = BootstrapMeasurementError(
            ("flag", "group"), categorical_change_probability=probability
        )
        report, active_estimator = _run(data, active)
        zero = BootstrapMeasurementError(("flag", "group"), categorical_change_probability=0.0)
        _, zero_estimator = _run(data, zero)

        test = report["bootstrap_measurement_error"]
        for child_seed, (_, active_data), (_, sample) in zip(
            test.child_seeds, active_estimator.calls, zero_estimator.calls, strict=True
        ):
            expected = _categorical_perturbation(sample, active, child_seed)
            assert np.array_equal(active_data.covariates, expected)
            deterministic_mutation = _categorical_perturbation(
                sample, active, child_seed, deterministic_alternatives=True
            )
            assert not np.array_equal(active_data.covariates, deterministic_mutation)
            reused_mask_mutation = _categorical_perturbation(
                sample, active, child_seed, reuse_change_mask=True
            )
            assert not np.array_equal(active_data.covariates, reused_mask_mutation)

    def test_cluster_samples_keep_whole_clusters_and_report_resolved_mode(self) -> None:
        data = _data(clusters=True)
        declaration = BootstrapMeasurementError(
            ("numeric",), numeric_noise=RelativeGaussianNoise(0.0), resampling="auto"
        )
        report, estimator = _run(data, declaration)
        assert report["bootstrap_measurement_error"].resampling == "cluster"
        for _, sample in estimator.calls:
            values, counts = np.unique(sample.covariates[:, 0] // 3, return_counts=True)
            del values
            assert np.all(counts % 3 == 0)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_seed_replay_child_separation_and_backend(self, backend: str) -> None:
        declaration = BootstrapMeasurementError(("numeric",))
        first, _ = _run(_data(backend=backend), declaration)
        second, _ = _run(_data(backend=backend), declaration)
        test = first["bootstrap_measurement_error"]
        assert first == second
        assert len(set(test.child_seeds)) == test.requested_draws
        assert all(type(record) is GeneratedOutcomeRecord for record in test.records)
        assert type(first.draws_frame("bootstrap_measurement_error")).__module__.startswith(backend)

    def test_failed_refits_are_retained_and_fail_the_rule(self) -> None:
        report, _ = _run(
            _data(),
            BootstrapMeasurementError(("numeric",)),
            estimator=_RecordingEstimator(fail_calls=(1,)),
        )
        test = report["bootstrap_measurement_error"]
        assert not test.passed
        assert test.n_failed == 1
        assert test.failures[0].replicate == 1

    def test_report_summary_and_repr_describe_measurement_error(self) -> None:
        report, _ = _run(
            _data(),
            BootstrapMeasurementError(("numeric", "group"), resampling="iid"),
        )

        summary = report.summary()
        assert "bootstrap_measurement_error: variables=numeric, group" in summary
        assert "resampling=iid" in summary
        assert "original=29.5" in summary
        assert repr(report) == summary


class TestPreflightRefusals:
    @pytest.mark.parametrize("variables", [("missing",), ("group__b",)])
    def test_unknown_and_generated_names_refuse_before_refit(
        self, variables: tuple[str, ...]
    ) -> None:
        estimator = _RecordingEstimator()
        with pytest.raises(CapabilityError):
            _run(_data(), BootstrapMeasurementError(variables), estimator=estimator)
        assert estimator.calls == []

    def test_cluster_and_strata_refuse_before_refit(self) -> None:
        estimator = _RecordingEstimator()
        with pytest.raises(CapabilityError, match="cluster ids"):
            _run(
                _data(),
                BootstrapMeasurementError(("numeric",), resampling="cluster"),
                estimator=estimator,
            )
        assert estimator.calls == []
        with pytest.raises(CapabilityError, match="strata"):
            _run(
                _data(strata=True),
                BootstrapMeasurementError(("group",)),
                estimator=estimator,
            )
        assert estimator.calls == []

    def test_incomplete_categorical_block_refuses_before_refit(self) -> None:
        group = np.tile(["a", "b", "c"], 20)
        frame = pd.DataFrame(
            {
                "Y": np.linspace(-1.0, 1.0, 60),
                "A": np.tile([0, 1], 30),
                "duplicate_b": group == "b",
                "group": group,
            }
        )
        with pytest.warns(UserWarning, match="group__b"):
            data = CausalData.from_frame(
                frame,
                outcome="Y",
                treatment="A",
                covariates=("duplicate_b", "group"),
            )
        assert "group__b" not in data.covariate_names
        assert "group__c" in data.covariate_names
        estimator = _RecordingEstimator()

        with pytest.raises(CapabilityError, match=r"group.*incomplete.*group__b"):
            _run(data, BootstrapMeasurementError(("group",)), estimator=estimator)

        assert estimator.calls == []

    def test_invalid_budget_and_missing_declaration_refuse_before_refit(self) -> None:
        result = _result(_data())
        with pytest.raises(CapabilityError, match="declaration"):
            refute(result, tests=("bootstrap_measurement_error",), n_replicates=4)
        assert result.estimator.calls == []

    def test_unsupported_result_family_is_named_before_refit(self) -> None:
        estimator = _RecordingEstimator()
        result = _Result(
            estimator=estimator,
            estimates={"ate": SimpleNamespace(psi=0.0, std_error=1.0)},
            data=SimpleNamespace(backend=None),
            intermediate_value=None,
        )
        with pytest.raises(CapabilityError, match=r"_Result.*SimpleNamespace"):
            refute(
                result,
                tests=("bootstrap_measurement_error",),
                n_replicates=4,
                bootstrap_measurement_error=BootstrapMeasurementError(("numeric",)),
                measurement_error_rule=_rule(4),
            )
        assert estimator.calls == []

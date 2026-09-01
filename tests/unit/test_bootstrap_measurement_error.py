"""Bootstrap measurement-error validation and its deliberate mutation controls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import CapabilityError, DataError
from cleverly.inference import run_bootstrap
from cleverly.validation import (
    DEFAULT_OUTCOME_REPLICATES,
    DEFAULT_TESTS,
    BootstrapMeasurementError,
    EmpiricalInclusionRule,
    GaussianNoise,
    GeneratedOutcomeRecord,
    RelativeGaussianNoise,
    ReplicationFailure,
    refute,
)
from cleverly.validation.refute import _run_empirical_refits


def _rule(draws: int) -> EmpiricalInclusionRule:
    return EmpiricalInclusionRule(alpha=2.0 / draws, minimum_draws=draws)


def _child_seeds(seed: int, draws: int) -> tuple[int, ...]:
    return tuple(
        int(sequence.generate_state(1)[0]) for sequence in np.random.SeedSequence(seed).spawn(draws)
    )


def _column_mean(covariates: np.ndarray) -> float:
    """Read the first adjustment column's mean, which mean-zero noise leaves alone."""
    return float(np.mean(covariates[:, 0]))


def _column_spread(covariates: np.ndarray) -> float:
    """Read the first adjustment column's dispersion, which added noise always inflates."""
    return float(np.std(covariates[:, 0], ddof=0))


def _two_sided_rank(values: Sequence[float], truth: float) -> float:
    """Return the two-sided empirical rank probability with inclusive half-ties."""
    below = sum(1 for value in values if value < truth)
    above = sum(1 for value in values if value > truth)
    ties = len(values) - below - above
    return min(1.0, 2.0 * min(below + 0.5 * ties, above + 0.5 * ties) / len(values))


def _cells(frame: Any, name: str) -> list[Any]:
    """Return one column's values, with every backend's missing marker as ``None``."""
    missing = frame[name].is_null().to_list()
    return [
        None if absent else value
        for absent, value in zip(missing, frame[name].to_list(), strict=True)
    ]


def _reference_perturbation(
    data: CausalData,
    declaration: BootstrapMeasurementError,
    seed: int,
    *,
    deterministic_alternatives: bool = False,
    reuse_change_mask: bool = False,
) -> np.ndarray:
    """Model the perturbation independently, in the order the declaration names.

    The refuter draws every variable's noise from one seeded stream, so the model has to
    consume that stream in the same order the caller declared. Iterating ``data.encodings``
    instead would agree only while a declaration happens to be in encoding order.
    """
    rng = np.random.default_rng(seed)
    encodings = {encoding.column: encoding for encoding in data.encodings}
    expected = data.covariates.copy()
    first_mask: np.ndarray | None = None
    for name in declaration.variables:
        encoding = encodings.get(name)
        if encoding is None:
            column = data.covariate_names.index(name)
            values = expected[:, column]
            scale = declaration.numeric_noise.standard_deviation * float(np.std(values, ddof=0))
            expected[:, column] = values + rng.normal(0.0, scale, size=data.n)
            continue
        columns = [data.covariate_names.index(item) for item in encoding.generated]
        block = expected[:, columns]
        total = np.sum(block, axis=1)
        codes = np.where(total == 0.0, 0, np.argmax(block, axis=1) + 1)
        drawn_mask = rng.random(data.n) < declaration.categorical_change_probability
        change = first_mask if reuse_change_mask and first_mask is not None else drawn_mask
        first_mask = change
        if np.any(change):
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
    #: Statistic every refit reports. The default is unbiased under both the bootstrap and
    #: mean-zero measurement error, so it is the file's stable pipeline.
    statistic: Callable[[np.ndarray], float] = field(default=_column_mean)

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
        estimate = SimpleNamespace(psi=self.statistic(data.covariates), std_error=1.0)
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


def _data(
    *,
    backend: str = "pandas",
    clusters: bool = False,
    strata: bool = False,
    extra: bool = False,
) -> CausalData:
    """Build the reference study data.

    ``extra`` adds a second numeric variable and a two-level string categorical, whose
    ``rng.integers(0, 1)`` alternative draw is degenerate.
    """
    n = 60
    payload: dict[str, Any] = {
        "Y": np.linspace(-1.0, 1.0, n),
        "A": np.tile([0, 1], n // 2),
        "numeric": np.arange(n, dtype=float),
        "flag": np.tile([False, True, True], n // 3),
        "group": np.tile(["a", "b", "c"], n // 3),
    }
    names = ["numeric", "flag", "group"]
    if extra:
        payload["scaled"] = np.linspace(0.0, 10.0, n)
        payload["pair"] = np.tile(["p", "q"], n // 2)
        names = ["numeric", "scaled", "flag", "group", "pair"]
    if clusters:
        payload["id"] = np.repeat(np.arange(20), 3)
    frame = pl.DataFrame(payload) if backend == "polars" else pd.DataFrame(payload)
    return CausalData.from_frame(
        frame,
        outcome="Y",
        treatment="A",
        covariates=tuple(names),
        id="id" if clusters else None,
        strata=("group",) if strata else None,
    )


def _thin_cluster_data() -> CausalData:
    """One large cluster and eight singletons, so a cluster draw can hold too few rows."""
    n = 60
    frame = pd.DataFrame(
        {
            "Y": np.linspace(-1.0, 1.0, n),
            "A": np.tile([0, 1], n // 2),
            "numeric": np.arange(n, dtype=float),
            "id": np.concatenate([np.zeros(52, dtype=int), np.arange(1, 9)]),
        }
    )
    return CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=("numeric",), id="id"
    )


def _result(data: CausalData, estimator: _RecordingEstimator | None = None) -> _Result:
    fitted = estimator or _RecordingEstimator()
    return _Result(
        estimator=fitted,
        estimates={"ate": SimpleNamespace(psi=fitted.statistic(data.covariates), std_error=1.0)},
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


def _refuse_before_any_refit(
    data: Any,
    declaration: Any,
    match: str,
    *,
    draws: int = 4,
    rule: Any = None,
    estimator: _RecordingEstimator | None = None,
) -> None:
    """Assert a mixed call refuses the measurement request before the placebo refits.

    ``placebo`` runs first in the requested order, so a refusal that escaped the
    ``refute()`` preflight and fired inside the measurement test would leave four placebo
    refits behind. ``estimator.calls == []`` is a witness only in a mixed call.
    """
    fitted = estimator or _RecordingEstimator()
    result = _result(data, fitted) if type(data) is CausalData else data
    with pytest.raises(CapabilityError, match=match):
        refute(
            result,
            tests=("placebo", "bootstrap_measurement_error"),
            n_replicates=draws,
            bootstrap_measurement_error=declaration,
            measurement_error_rule=_rule(4) if rule is None else rule,
            random_state=17,
        )
    assert result.estimator.calls == []


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

    def test_a_bare_string_is_not_a_variable_sequence(self) -> None:
        # ``tuple("numeric")`` is seven distinct one-character names, so it clears every
        # later guard. Only the string check refuses it, and only this message says so.
        with pytest.raises(ValueError, match="must be a sequence of names"):
            BootstrapMeasurementError("numeric")  # type: ignore[arg-type]

    def test_a_non_sequence_is_refused_as_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="must be a sequence of names"):
            BootstrapMeasurementError(5)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "noise",
        [
            GaussianNoise(),
            0.1,
            type("_NoiseSubclass", (RelativeGaussianNoise,), {})(),
        ],
    )
    def test_only_the_registered_relative_noise_law_is_accepted(self, noise: Any) -> None:
        with pytest.raises(ValueError, match="numeric_noise must be a RelativeGaussianNoise"):
            BootstrapMeasurementError(("numeric",), numeric_noise=noise)

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
        sequences = np.random.SeedSequence(17).spawn(4)
        indices = tuple(
            np.random.default_rng(sequence).integers(0, data.n, size=data.n, dtype=np.int64)
            for sequence in sequences
        )
        expected = tuple(data.covariates[index] for index in indices)
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

        # Reconstruct each iid draw without a production sampling helper. Equality
        # between the two callers alone is tautological because they share the sampler.
        assert len(estimator.calls) == 4
        assert len(plain) == 4
        for index, expected_sample, plain_sample, (_, sample) in zip(
            indices, expected, plain, estimator.calls, strict=True
        ):
            # This explicitly witnesses sampling with replacement. A no-resampling
            # mutation returns the original rows and cannot satisfy this index oracle.
            assert len(np.unique(index)) < data.n
            assert sample.n == data.n
            assert np.array_equal(plain_sample, expected_sample)
            assert np.array_equal(sample.covariates, expected_sample)

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

    def test_two_numeric_variables_each_draw_their_own_noise(self) -> None:
        data = _data(extra=True)
        declaration = BootstrapMeasurementError(
            ("numeric", "scaled"), numeric_noise=RelativeGaussianNoise(0.5)
        )
        report, estimator = _run(data, declaration)
        zero = BootstrapMeasurementError(
            ("numeric", "scaled"), numeric_noise=RelativeGaussianNoise(0.0)
        )
        _, zero_estimator = _run(data, zero)
        test = report["bootstrap_measurement_error"]
        for child_seed, (_, perturbed), (_, sample) in zip(
            test.child_seeds, estimator.calls, zero_estimator.calls, strict=True
        ):
            assert np.array_equal(
                perturbed.covariates, _reference_perturbation(sample, declaration, child_seed)
            )
            # Both variables moved, and the second did not reuse the first draw.
            first = perturbed.covariates[:, 0] - sample.covariates[:, 0]
            second = perturbed.covariates[:, 1] - sample.covariates[:, 1]
            assert np.all(first != 0.0)
            assert np.all(second != 0.0)
            assert not np.array_equal(first, second)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_boolean_and_multilevel_changes_rebuild_valid_indicator_blocks(
        self, backend: str
    ) -> None:
        data = _data(backend=backend)
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

    def test_a_two_level_string_categorical_flips_on_a_degenerate_draw(self) -> None:
        # ``rng.integers(0, 1)`` is degenerate at two levels: every alternative is code 0,
        # and only the ``alternatives >= current`` shift turns that into the other level.
        data = _data(extra=True)
        declaration = BootstrapMeasurementError(("pair",), categorical_change_probability=1.0)
        report, estimator = _run(data, declaration)
        zero = BootstrapMeasurementError(("pair",), categorical_change_probability=0.0)
        _, zero_estimator = _run(data, zero)
        encoding = next(item for item in data.encodings if item.column == "pair")
        assert encoding.generated == ("pair__q",)
        column = data.covariate_names.index("pair__q")
        test = report["bootstrap_measurement_error"]
        for child_seed, (_, perturbed), (_, sample) in zip(
            test.child_seeds, estimator.calls, zero_estimator.calls, strict=True
        ):
            values = perturbed.covariates[:, column]
            assert np.all(np.isin(values, (0.0, 1.0)))
            assert np.all(values == 1.0 - sample.covariates[:, column])
            assert np.array_equal(
                perturbed.covariates, _reference_perturbation(sample, declaration, child_seed)
            )

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
            expected = _reference_perturbation(sample, active, child_seed)
            assert np.array_equal(active_data.covariates, expected)
            deterministic_mutation = _reference_perturbation(
                sample, active, child_seed, deterministic_alternatives=True
            )
            assert not np.array_equal(active_data.covariates, deterministic_mutation)
            reused_mask_mutation = _reference_perturbation(
                sample, active, child_seed, reuse_change_mask=True
            )
            assert not np.array_equal(active_data.covariates, reused_mask_mutation)

    def test_the_declaration_order_drives_the_noise_stream(self) -> None:
        # One seeded stream serves every selected variable, so the order the caller
        # declares is part of the recorded perturbation. Iterating the recorded encodings
        # instead would silently give a different perturbation for the same seed.
        data = _data(extra=True)
        forward = BootstrapMeasurementError(
            ("group", "numeric", "pair", "scaled"),
            numeric_noise=RelativeGaussianNoise(0.3),
            categorical_change_probability=0.4,
        )
        reverse = BootstrapMeasurementError(
            ("scaled", "pair", "numeric", "group"),
            numeric_noise=RelativeGaussianNoise(0.3),
            categorical_change_probability=0.4,
        )
        forward_report, forward_estimator = _run(data, forward)
        reverse_report, reverse_estimator = _run(data, reverse)
        zero = BootstrapMeasurementError(
            ("group", "numeric", "pair", "scaled"),
            numeric_noise=RelativeGaussianNoise(0.0),
            categorical_change_probability=0.0,
        )
        _, zero_estimator = _run(data, zero)

        assert forward_report["bootstrap_measurement_error"].child_seeds == (
            reverse_report["bootstrap_measurement_error"].child_seeds
        )
        for (_, first), (_, second) in zip(
            forward_estimator.calls, reverse_estimator.calls, strict=True
        ):
            assert not np.array_equal(first.covariates, second.covariates)
        for declaration, estimator in (
            (forward, forward_estimator),
            (reverse, reverse_estimator),
        ):
            seeds = forward_report["bootstrap_measurement_error"].child_seeds
            for child_seed, (_, perturbed), (_, sample) in zip(
                seeds, estimator.calls, zero_estimator.calls, strict=True
            ):
                assert np.array_equal(
                    perturbed.covariates,
                    _reference_perturbation(sample, declaration, child_seed),
                )

    def test_cluster_samples_keep_whole_clusters_and_report_resolved_mode(self) -> None:
        data = _data(clusters=True)
        declaration = BootstrapMeasurementError(
            ("numeric",), numeric_noise=RelativeGaussianNoise(0.0), resampling="auto"
        )
        report, estimator = _run(data, declaration)
        assert report["bootstrap_measurement_error"].resampling == "cluster"
        repeated_source = False
        for _, sample in estimator.calls:
            assert sample.cluster is not None
            source_codes = (sample.covariates[:, 0] // 3).astype(int)
            repeated_source |= np.unique(source_codes).size < 20
            np.testing.assert_array_equal(np.unique(sample.cluster), np.arange(20))
            np.testing.assert_array_equal(np.bincount(sample.cluster), np.full(20, 3))
            for occurrence in range(20):
                rows = sample.covariates[sample.cluster == occurrence, 0]
                sources = np.unique(source_codes[sample.cluster == occurrence])
                assert sources.size == 1
                source = int(sources[0])
                np.testing.assert_array_equal(rows, np.arange(3 * source, 3 * source + 3))
        # This fixed seed actively exercises the defect: at least one source cluster is
        # selected more than once, and its copies still have separate occurrence codes.
        assert repeated_source

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


class TestRuleVerdict:
    def test_a_stable_pipeline_passes_and_records_every_statistic(self) -> None:
        data = _data()
        declaration = BootstrapMeasurementError(
            ("numeric",), numeric_noise=RelativeGaussianNoise(0.1)
        )
        report, estimator = _run(data, declaration, draws=20)
        test = report["bootstrap_measurement_error"]

        assert test.passed
        assert report.passed
        assert bool(report)
        assert test.rule == _rule(20)
        assert test.declaration == declaration
        assert test.original == _column_mean(data.covariates)
        assert test.declared_effect == _column_mean(data.covariates)
        assert test.expectation == "includes 29.5"
        assert test.resampling == "iid"
        assert test.requested_draws == 20
        assert test.n_replicates == 20
        assert test.n_failed == 0
        assert test.family == data.family
        assert test.standard_errors == (1.0,) * 20
        # Recomputed from the data each refit was handed, not from the report's records.
        assert test.values == tuple(
            _column_mean(sample.covariates) for _, sample in estimator.calls
        )
        assert test.mean == pytest.approx(float(np.mean(test.values)))
        assert test.spread == pytest.approx(float(np.std(test.values, ddof=1)))
        assert test.empirical_pvalue == pytest.approx(_two_sided_rank(test.values, test.original))
        assert test.empirical_pvalue > test.rule.alpha
        assert "two-sided empirical p=" in test.detail

    def test_measurement_error_that_moves_every_draw_fails_the_rule(self) -> None:
        # Independent noise adds variance, so a pipeline that reads the dispersion of the
        # perturbed variable is biased upward on every draw. The whole refit sample then
        # lies on one side of the declared effect, which is what the rule rejects.
        data = _data()
        declaration = BootstrapMeasurementError(
            ("numeric",), numeric_noise=RelativeGaussianNoise(1.0)
        )
        report, _ = _run(
            data,
            declaration,
            draws=20,
            estimator=_RecordingEstimator(statistic=_column_spread),
        )
        test = report["bootstrap_measurement_error"]

        assert not test.passed
        assert not report.passed
        assert test.n_failed == 0
        assert len(test.records) == 20
        assert test.original == _column_spread(data.covariates)
        assert min(test.values) > test.original
        assert test.empirical_pvalue == 0.0
        assert test.empirical_pvalue == pytest.approx(_two_sided_rank(test.values, test.original))
        assert test.detail == ("two-sided empirical p=0 with inclusive half-ties (alpha=0.1)")

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
        assert test.detail == "1 refit(s) failed; failure_policy='fail' keeps every failure"

    def test_inconsistent_refit_families_fail_the_test(self) -> None:
        # A deliberate-mutation control for a defensive branch. ``_run_empirical_refits``
        # refuses any refit whose family differs from the fitted one, so the only way to
        # reach the disagreement branch is a fitted family that compares equal to both.
        class _AnyFamily:
            def __eq__(self, other: object) -> bool:
                return True

            def __hash__(self) -> int:
                return 0

        class _DriftingEstimator(_RecordingEstimator):
            def refit(self, data: CausalData, **kwargs: Any) -> Any:
                call = len(self.calls)
                refitted = super().refit(data, **kwargs)
                reported = "binomial" if call % 2 else "gaussian"
                return _RefitResult(SimpleNamespace(family=reported), refitted.estimate)

        data = replace(_data(), family=_AnyFamily())
        report, _ = _run(
            data,
            BootstrapMeasurementError(("numeric",)),
            estimator=_DriftingEstimator(),
        )
        test = report["bootstrap_measurement_error"]
        assert not test.passed
        assert test.family is None
        assert test.detail == (
            "successful refits reported inconsistent outcome families ['binomial', 'gaussian']"
        )

    def test_the_default_budget_is_the_empirical_one_not_the_legacy_one(self) -> None:
        # The default rule needs 40 successful draws. The legacy five-replicate budget
        # could never satisfy it, so falling back to it would turn every unspecified
        # measurement request into a "too few draws" failure.
        result = _result(_data())
        report = refute(
            result,
            tests=("bootstrap_measurement_error",),
            bootstrap_measurement_error=BootstrapMeasurementError(("numeric",)),
            random_state=17,
        )
        test = report["bootstrap_measurement_error"]
        assert DEFAULT_OUTCOME_REPLICATES == 100
        assert test.requested_draws == DEFAULT_OUTCOME_REPLICATES
        assert test.n_replicates == DEFAULT_OUTCOME_REPLICATES
        assert len(result.estimator.calls) == DEFAULT_OUTCOME_REPLICATES
        assert test.rule == EmpiricalInclusionRule()
        assert test.passed

    def test_the_measurement_test_is_not_a_default_test(self) -> None:
        # It needs a declaration only the analyst can supply, so a default run must not
        # request it. If it were in DEFAULT_TESTS this call would refuse instead.
        assert "bootstrap_measurement_error" not in DEFAULT_TESTS
        report = refute(_result(_data()), random_state=3)
        assert [test.name for test in report.tests] == list(DEFAULT_TESTS)


class TestReporting:
    def test_report_summary_describes_measurement_error(self) -> None:
        report, _ = _run(
            _data(),
            BootstrapMeasurementError(("numeric", "group"), resampling="iid"),
        )

        summary = report.summary()
        assert (
            "bootstrap_measurement_error: variables=numeric, group; resampling=iid; "
            "original=29.5; successful=4; failed=0" in summary
        )
        assert "rule: two-sided empirical p=" in summary
        assert "includes 29.5" in summary

    def test_a_failed_measurement_test_is_rendered_as_a_failure(self) -> None:
        report, _ = _run(
            _data(),
            BootstrapMeasurementError(("numeric",), numeric_noise=RelativeGaussianNoise(1.0)),
            draws=20,
            estimator=_RecordingEstimator(statistic=_column_spread),
        )
        summary = report.summary()
        assert (
            "bootstrap_measurement_error: variables=numeric; resampling=iid; "
            "original=17.318; successful=20; failed=0" in summary
        )
        assert "rule: two-sided empirical p=0 with inclusive half-ties (alpha=0.1)" in summary
        assert "VERDICT" not in summary

    def test_the_report_table_carries_the_resolved_resampling_mode(self) -> None:
        data = _data(clusters=True)
        result = _result(data)
        report = refute(
            result,
            tests=("placebo", "bootstrap_measurement_error"),
            n_replicates=4,
            bootstrap_measurement_error=BootstrapMeasurementError(("numeric",)),
            measurement_error_rule=_rule(4),
            random_state=17,
        )
        frame = nw.from_native(report.to_frame(), eager_only=True)
        assert frame["test"].to_list() == ["placebo", "bootstrap_measurement_error"]
        # ``placebo`` resamples nothing, so it names no bootstrap mode at all.
        assert _cells(frame, "resampling") == [None, "cluster"]
        assert _cells(frame, "requested_draws") == [None, 4]
        assert _cells(frame, "passed") == [False, True]

    def test_draw_rows_carry_every_record_and_retained_failure(self) -> None:
        report, _ = _run(
            _data(),
            BootstrapMeasurementError(("numeric",)),
            estimator=_RecordingEstimator(fail_calls=(1,)),
        )
        test = report["bootstrap_measurement_error"]
        frame = nw.from_native(report.draws_frame("bootstrap_measurement_error"), eager_only=True)
        assert list(frame.columns) == [
            "test",
            "replicate",
            "seed",
            "estimate",
            "std_error",
            "family",
            "error_type",
            "message",
        ]
        assert frame["test"].to_list() == ["bootstrap_measurement_error"] * 4
        assert frame["replicate"].to_list() == [0, 1, 2, 3]
        assert frame["seed"].to_list() == list(test.child_seeds)
        assert _cells(frame, "estimate") == [
            test.records[0].estimate,
            None,
            test.records[1].estimate,
            test.records[2].estimate,
        ]
        assert _cells(frame, "std_error") == [1.0, None, 1.0, 1.0]
        assert _cells(frame, "family") == ["gaussian", None, "gaussian", "gaussian"]
        assert _cells(frame, "error_type") == [None, "RuntimeError", None, None]
        assert _cells(frame, "message") == [None, "failed draw 1", None, None]

    def test_draws_frame_refuses_a_test_without_draw_records(self) -> None:
        report = refute(_result(_data()), tests=("placebo",), n_replicates=2, random_state=3)
        with pytest.raises(ValueError, match="no empirical draw records"):
            report.draws_frame("placebo")


class TestPreparationFailureContract:
    def test_a_thin_cluster_draw_is_retained_rather_than_aborting_the_report(self) -> None:
        # A cluster bootstrap that misses the one large cluster leaves nine rows, which
        # ``CausalData.subset`` refuses. That is a property of the draw, not of the
        # request, so the operation keeps the draw as a failure and reports the rest.
        data = _thin_cluster_data()
        report, estimator = _run(
            data,
            BootstrapMeasurementError(("numeric",), resampling="cluster"),
            draws=20,
            seed=3,
        )
        test = report["bootstrap_measurement_error"]
        assert test.resampling == "cluster"
        assert test.n_failed > 0
        assert len(test.records) > 0
        assert len(test.records) + len(test.failures) == 20
        assert len(test.child_seeds) == 20
        assert len(estimator.calls) == len(test.records)
        assert {failure.error_type for failure in test.failures} == {"DataError"}
        assert all("need at least 10" in failure.message for failure in test.failures)
        assert all(isinstance(failure, ReplicationFailure) for failure in test.failures)
        assert not test.passed

    def test_the_perturbed_design_is_validated_before_the_refit(self, monkeypatch: Any) -> None:
        # The perturbation hands its design to ``with_covariates``, which refuses a
        # non-finite one. Building the replacement without that validation would send an
        # infinite design to the estimator instead.
        monkeypatch.setattr(
            RelativeGaussianNoise,
            "draw",
            lambda self, rng, values: np.full(np.asarray(values).size, np.inf),
        )
        report, estimator = _run(_data(), BootstrapMeasurementError(("numeric",)))
        test = report["bootstrap_measurement_error"]
        assert estimator.calls == []
        assert not test.passed
        assert [failure.error_type for failure in test.failures] == ["DataError"] * 4
        assert all(
            "measurement-error covariates contains non-finite values" in failure.message
            for failure in test.failures
        )

    def test_a_broken_indicator_block_is_refused_inside_the_draw(self) -> None:
        # A deliberate-mutation control on the container: the encoded block is corrupted
        # past what any constructor allows, so the perturbation's own validity guard is
        # the only thing that can catch it.
        data = _data()
        corrupt = data.covariates.copy()
        corrupt[:, data.covariate_names.index("group__b")] = 1.0
        corrupt[:, data.covariate_names.index("group__c")] = 1.0
        report, estimator = _run(
            replace(data, covariates=corrupt), BootstrapMeasurementError(("group",))
        )
        test = report["bootstrap_measurement_error"]
        assert estimator.calls == []
        assert [failure.error_type for failure in test.failures] == ["RuntimeError"] * 4
        assert all(
            "encoded block for 'group' is not a valid drop-first indicator block" in failure.message
            for failure in test.failures
        )

    @pytest.mark.parametrize(
        ("retain", "expected"),
        [(True, "retained"), (False, "raised")],
    )
    def test_preparation_failures_are_retained_only_where_declared(
        self, retain: bool, expected: str
    ) -> None:
        # The measurement-error path asks for retention because a bootstrap draw fails per
        # sample. The generated-outcome path does not, because a deterministic replacement
        # fails the same way on every draw and the caller must see it at once.
        result = _result(_data())

        def explode(replicate: int, seed: int) -> Any:
            del seed
            raise DataError(f"cannot prepare draw {replicate}")

        if expected == "raised":
            with pytest.raises(DataError, match="cannot prepare draw 0"):
                _run_empirical_refits(
                    result,
                    estimand="ate",
                    draws=((0, 11), (1, 12)),
                    replacement=explode,
                    expected_family="gaussian",
                    retain_preparation_failures=retain,
                )
            assert result.estimator.calls == []
            return

        records, failures = _run_empirical_refits(
            result,
            estimand="ate",
            draws=((0, 11), (1, 12)),
            replacement=explode,
            expected_family="gaussian",
            retain_preparation_failures=retain,
        )
        assert records == ()
        assert [failure.replicate for failure in failures] == [0, 1]
        assert [failure.seed for failure in failures] == [11, 12]
        assert [failure.error_type for failure in failures] == ["DataError", "DataError"]
        assert failures[0].message == "cannot prepare draw 0"
        assert result.estimator.calls == []

    def test_the_generated_outcome_path_does_not_ask_for_retention(self) -> None:
        # The default is the raising contract, so a caller that forgets the keyword gets
        # it. Only ``bootstrap_measurement_error`` passes ``True``.
        result = _result(_data())

        def explode(replicate: int, seed: int) -> Any:
            del replicate, seed
            raise DataError("deterministic preparation failure")

        with pytest.raises(DataError, match="deterministic preparation failure"):
            _run_empirical_refits(
                result,
                estimand="ate",
                draws=((0, 11),),
                replacement=explode,
                expected_family="gaussian",
            )


class TestPreflightRefusals:
    @pytest.mark.parametrize(
        ("variables", "match"),
        [
            (("missing",), r"unknown adjustment variables \['missing'\]; choose from"),
            (
                ("group__b",),
                r"'group__b' is a generated indicator for original categorical variable "
                r"'group'; select the original variable",
            ),
        ],
    )
    def test_unknown_and_generated_names_refuse_before_refit(
        self, variables: tuple[str, ...], match: str
    ) -> None:
        _refuse_before_any_refit(_data(), BootstrapMeasurementError(variables), match)

    def test_cluster_and_strata_refuse_before_refit(self) -> None:
        _refuse_before_any_refit(
            _data(),
            BootstrapMeasurementError(("numeric",), resampling="cluster"),
            r"resampling='cluster' requires the data to carry cluster ids",
        )
        _refuse_before_any_refit(
            _data(strata=True),
            BootstrapMeasurementError(("group",)),
            r"cannot perturb selected strata variables .*\['group'\]",
        )

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

        _refuse_before_any_refit(
            data,
            BootstrapMeasurementError(("group",)),
            r"group.*incomplete.*group__b",
        )

    def test_an_undeclared_binary_column_refuses_before_refit(self) -> None:
        # ``from_arrays`` records no encoding, so a 0/1 column reaches the numeric path,
        # where continuous noise would make it real-valued.
        covariates = np.column_stack([np.arange(60, dtype=float), np.tile([0.0, 1.0], 30)])
        data = CausalData.from_arrays(
            np.linspace(-1.0, 1.0, 60),
            np.tile([0, 1], 30),
            covariates,
            covariate_names=("numeric", "binary"),
        )
        assert data.encodings == ()
        _refuse_before_any_refit(
            data,
            BootstrapMeasurementError(("binary",)),
            r"cannot perturb 'binary': its values are all 0 or 1 but it carries no "
            r"CategoricalEncoding",
        )

    def test_a_constant_numeric_column_refuses_before_refit(self) -> None:
        # A constant column is dropped at construction, so the zero-variance guard is
        # reachable only through a replacement design.
        data = _data()
        constant = data.covariates.copy()
        constant[:, data.covariate_names.index("numeric")] = 7.0
        _refuse_before_any_refit(
            data.with_covariates(constant),
            BootstrapMeasurementError(("numeric",)),
            r"cannot perturb 'numeric': the variable is constant",
        )

    def test_a_missing_declaration_refuses_before_refit(self) -> None:
        result = _result(_data())
        with pytest.raises(
            CapabilityError,
            match=r"requires the exact registered BootstrapMeasurementError declaration",
        ):
            refute(
                result,
                tests=("placebo", "bootstrap_measurement_error"),
                n_replicates=4,
                measurement_error_rule=_rule(4),
            )
        assert result.estimator.calls == []

    def test_an_invalid_draw_budget_refuses_before_refit(self) -> None:
        result = _result(_data())
        with pytest.raises(
            CapabilityError,
            match=r"bootstrap_measurement_error was asked for 3 draw\(s\) under a rule that "
            r"requires 4; raise n_replicates",
        ):
            refute(
                result,
                tests=("placebo", "bootstrap_measurement_error"),
                n_replicates=3,
                bootstrap_measurement_error=BootstrapMeasurementError(("numeric",)),
                measurement_error_rule=_rule(4),
                random_state=17,
            )
        assert result.estimator.calls == []

    def test_declaration_subclasses_are_not_registered(self) -> None:
        class _UnregisteredDeclaration(BootstrapMeasurementError):
            pass

        _refuse_before_any_refit(
            _data(),
            _UnregisteredDeclaration(("numeric",)),
            r"requires the exact registered BootstrapMeasurementError declaration",
        )

    def test_rule_subclasses_are_not_registered(self) -> None:
        class _UnregisteredRule(EmpiricalInclusionRule):
            pass

        _refuse_before_any_refit(
            _data(),
            BootstrapMeasurementError(("numeric",)),
            r"requires the exact registered EmpiricalInclusionRule declaration",
            rule=_UnregisteredRule(alpha=0.5, minimum_draws=4),
        )

    def test_unsupported_result_family_is_named_before_refit(self) -> None:
        estimator = _RecordingEstimator()
        result = _Result(
            estimator=estimator,
            estimates={"ate": SimpleNamespace(psi=0.0, std_error=1.0)},
            data=SimpleNamespace(backend=None),
            intermediate_value=None,
        )
        _refuse_before_any_refit(
            result,
            BootstrapMeasurementError(("numeric",)),
            r"_Result.*SimpleNamespace",
        )

"""Generated-outcome refutations and their deliberate negative controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from types import SimpleNamespace
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment, SuperLearner
from cleverly.data import CausalData
from cleverly.datasets import make_binary_outcome
from cleverly.exceptions import CapabilityError, DataError
from cleverly.study import BackdoorMeanContrast, ExplicitAdjustmentProvider, ParameterKey
from cleverly.targets import TARGETS
from cleverly.validation import (
    EmpiricalInclusionRule,
    GaussianAdjustmentOutcome,
    GaussianNoise,
    ReplicationFailure,
    refute,
)


@dataclass
class _MeanDifferenceEstimator:
    """Small deterministic refit seam that isolates generated-outcome behavior."""

    random_state: int | None = 0
    fail_calls: tuple[int, ...] = ()
    family: str = "auto"
    outcome_learner: Any = None

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
        treated = data.outcome[data.treatment == 1.0]
        control = data.outcome[data.treatment == 0.0]
        effect = float(np.mean(treated) - np.mean(control))
        standard_error = float(
            np.sqrt(np.var(treated, ddof=1) / treated.size + np.var(control, ddof=1) / control.size)
        )
        estimate = SimpleNamespace(psi=effect, std_error=standard_error)
        return _RefitResult(data, estimate)


class _RefitResult:
    def __init__(self, data: CausalData, estimate: Any) -> None:
        self.data = data
        self._estimate = estimate

    def __getitem__(self, name: str) -> Any:
        assert name == "ate"
        return self._estimate


class _Result(SimpleNamespace):
    def __getitem__(self, name: str) -> Any:
        return self.estimates[name]


def _eligible_result(*, estimator: Any = None, backend: str | None = None) -> Any:
    rng = np.random.default_rng(4)
    covariate_pairs = rng.normal(size=(120, 2))
    covariates = np.repeat(covariate_pairs, 2, axis=0)
    treatment = np.tile([0.0, 1.0], covariate_pairs.shape[0])
    if backend is None:
        data = CausalData.from_arrays(
            rng.normal(size=treatment.size),
            treatment,
            covariates,
            covariate_names=("W1", "W2"),
        )
    else:
        payload = {
            "Y": rng.normal(size=treatment.size),
            "A": treatment,
            "W1": covariates[:, 0],
            "W2": covariates[:, 1],
        }
        frame = pl.DataFrame(payload) if backend == "polars" else pd.DataFrame(payload)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=("W1", "W2"))
    functional = BackdoorMeanContrast(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2"),
        target="ate",
    )
    return _Result(
        estimator=estimator or _MeanDifferenceEstimator(),
        estimates={"ate": SimpleNamespace(psi=2.0, std_error=0.1)},
        data=data,
        identified_effect=SimpleNamespace(
            estimand=ATE(),
            functional=functional,
            identification=TARGETS["ate"].identification,
            provider=ExplicitAdjustmentProvider(),
        ),
        parameter_keys={"ate": ParameterKey("ate", "ate", value=1, reference=0)},
        intermediate_value=None,
        assessment_cache={},
    )


class TestOutcomeReplacement:
    def test_validates_length_finiteness_and_family_support(self) -> None:
        data = _eligible_result().data
        with pytest.raises(DataError, match="length"):
            data.with_outcome(np.zeros(data.n - 1))
        with pytest.raises(DataError, match="flagged as observed"):
            data.with_outcome(np.full(data.n, np.nan))
        with pytest.raises(DataError, match="requires 0/1"):
            data.with_outcome(np.arange(data.n), family="binomial")
        with pytest.raises(DataError, match="family"):
            data.with_outcome(np.zeros(data.n), family="poisson")

    def test_preserves_roles_and_resolves_the_replacement_family(self) -> None:
        data = _eligible_result().data
        replaced = data.with_outcome(np.linspace(-1.0, 1.0, data.n))
        assert replaced.family == "gaussian"
        assert replaced.treatment is data.treatment
        assert replaced.covariates is data.covariates


class TestGeneratedProcesses:
    def test_declarations_are_immutable_and_replay_their_noise(self) -> None:
        process = GaussianAdjustmentOutcome(noise=GaussianNoise(0.0, 0.5))
        data = _eligible_result().data
        assert process.draw(data, seed=8) == pytest.approx(process.draw(data, seed=8))
        assert not np.array_equal(process.draw(data, seed=8), process.draw(data, seed=9))
        with pytest.raises(FrozenInstanceError):
            process.effect = 4.0  # type: ignore[misc]

    def test_dummy_and_nonzero_processes_pass(self) -> None:
        rule = EmpiricalInclusionRule(minimum_draws=20)
        report = refute(
            _eligible_result(),
            tests=("dummy_outcome", "simulated_outcome"),
            n_replicates=40,
            outcome_rule=rule,
            random_state=12,
        )
        assert report.passed
        assert report["dummy_outcome"].declared_effect == 0.0
        assert report["simulated_outcome"].declared_effect == 1.0
        assert report["simulated_outcome"].standard_errors
        assert report["simulated_outcome"].family == "gaussian"

    @pytest.mark.parametrize(
        "mutation",
        ["treatment-removed", "sign-reversed", "arms-reversed"],
        ids=("treatment-removed", "sign-reversed", "arms-reversed"),
    )
    def test_nonzero_mutations_fail(self, mutation: str, monkeypatch: pytest.MonkeyPatch) -> None:
        def mutated_draw(process: GaussianAdjustmentOutcome, data: Any, *, seed: int) -> np.ndarray:
            adjustment = process.adjustment(data.covariates)
            noise = process.noise.draw(np.random.default_rng(seed), data.n)
            if mutation == "treatment-removed":
                treatment_term = 0.0
            elif mutation == "sign-reversed":
                treatment_term = -process.effect * data.treatment
            else:
                treatment_term = process.effect * (1.0 - data.treatment)
            return np.asarray(adjustment + treatment_term + noise, dtype=float)

        # Deliberately mutate the exact registered implementation. Unsupported subclasses
        # are refused before their draw method can run and therefore are not valid controls.
        monkeypatch.setattr(GaussianAdjustmentOutcome, "draw", mutated_draw)
        report = refute(
            _eligible_result(),
            tests=("simulated_outcome",),
            simulated_outcome=GaussianAdjustmentOutcome(),
            n_replicates=40,
            outcome_rule=EmpiricalInclusionRule(minimum_draws=20),
            random_state=12,
        )
        assert not report.passed
        assert report["simulated_outcome"].empirical_pvalue == 0.0

    def test_recorded_child_seed_replays_outcome_and_refit(self) -> None:
        result = _eligible_result()
        process = GaussianAdjustmentOutcome()
        report = refute(
            result,
            tests=("simulated_outcome",),
            simulated_outcome=process,
            n_replicates=4,
            outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            random_state=31,
        )
        record = report["simulated_outcome"].records[2]
        replacement = result.data.with_outcome(
            process.draw(result.data, seed=record.seed), family=process.family
        )
        replay = _MeanDifferenceEstimator().refit(
            replacement, intermediate_value=None, random_state=record.seed
        )["ate"]
        assert replay.psi == record.estimate
        assert replay.std_error == record.std_error

    def test_refit_failures_are_retained_and_fail_the_rule(self) -> None:
        result = _eligible_result(estimator=_MeanDifferenceEstimator(fail_calls=(1, 3)))
        report = refute(
            result,
            tests=("dummy_outcome",),
            n_replicates=5,
            outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            random_state=5,
        )
        test = report["dummy_outcome"]
        assert not test.passed
        assert [failure.replicate for failure in test.failures] == [1, 3]
        assert all(isinstance(item, ReplicationFailure) for item in test.failures)
        assert len(nw.from_native(report.draws_frame("dummy_outcome"), eager_only=True)) == 5

    def test_nonfinite_refit_is_retained_and_later_draws_continue(self) -> None:
        class NonfiniteEstimator(_MeanDifferenceEstimator):
            def refit(self, *args: Any, **kwargs: Any) -> Any:
                refitted = super().refit(*args, **kwargs)
                if len(self.calls) == 2:
                    refitted._estimate.psi = np.nan
                return refitted

        estimator = NonfiniteEstimator()
        test = refute(
            _eligible_result(estimator=estimator),
            tests=("dummy_outcome",),
            n_replicates=3,
            outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            random_state=5,
        )["dummy_outcome"]
        assert len(estimator.calls) == 3
        assert [record.replicate for record in test.records] == [0, 2]
        assert [failure.replicate for failure in test.failures] == [1]
        assert test.failures[0].error_type == "ValueError"
        assert not test.passed

    def test_all_failure_behavior_keeps_every_seed(self) -> None:
        result = _eligible_result(estimator=_MeanDifferenceEstimator(fail_calls=(0, 1, 2)))
        test = refute(
            result,
            tests=("dummy_outcome",),
            n_replicates=3,
            outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            random_state=9,
        )["dummy_outcome"]
        assert not test.passed
        assert not test.records
        assert len(test.failures) == 3
        assert len(test.child_seeds) == 3
        assert np.isnan(test.mean)

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_report_tables_use_the_input_backend(self, backend: str) -> None:
        result = _eligible_result(backend=backend)
        report = refute(
            result,
            tests=("dummy_outcome",),
            n_replicates=2,
            outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
        )
        assert type(report.to_frame()).__module__.startswith(backend)
        assert type(report.draws_frame("dummy_outcome")).__module__.startswith(backend)


class TestGeneratedOutcomeRefusals:
    def test_legacy_fit_is_refused_before_a_refit(self) -> None:
        result = _eligible_result()
        result.identified_effect = None
        with pytest.raises(CapabilityError, match="legacy fit"):
            refute(result, tests=("placebo", "dummy_outcome"), n_replicates=1)
        assert result.estimator.calls == []

    def test_process_subclasses_are_not_registered(self) -> None:
        class UnregisteredProcess(GaussianAdjustmentOutcome):
            pass

        result = _eligible_result()
        with pytest.raises(CapabilityError, match="exact registered GaussianAdjustmentOutcome"):
            refute(
                result,
                tests=("simulated_outcome",),
                simulated_outcome=UnregisteredProcess(),
                n_replicates=1,
            )
        assert result.estimator.calls == []

    def test_rule_subclasses_are_refused_before_mixed_operation_refits(self) -> None:
        class UnregisteredRule(EmpiricalInclusionRule):
            pass

        result = _eligible_result()
        with pytest.raises(CapabilityError, match="exact registered EmpiricalInclusionRule"):
            refute(
                result,
                tests=("placebo", "dummy_outcome"),
                outcome_rule=UnregisteredRule(),
                n_replicates=1,
            )
        assert result.estimator.calls == []

    def test_arbitrary_provider_name_is_not_backdoor_provenance(self) -> None:
        result = _eligible_result()
        result.identified_effect.provider = SimpleNamespace(name="explicit-adjustment")
        with pytest.raises(CapabilityError, match="registered backdoor provider provenance"):
            refute(result, tests=("dummy_outcome",), n_replicates=1)
        assert result.estimator.calls == []

    def test_inconsistent_target_artifacts_are_refused(self) -> None:
        result = _eligible_result()
        result.identified_effect.functional = replace(
            result.identified_effect.functional, target="att"
        )
        with pytest.raises(CapabilityError, match="inconsistent registered target provenance"):
            refute(result, tests=("dummy_outcome",), n_replicates=1)
        assert result.estimator.calls == []

    def test_incompatible_configured_family_is_refused(self) -> None:
        result = _eligible_result()
        result.estimator.family = "binomial"
        with pytest.raises(CapabilityError, match="configured with family='binomial'"):
            refute(result, tests=("dummy_outcome",), n_replicates=1)
        assert result.estimator.calls == []

    @pytest.mark.parametrize(
        "outcome_learner",
        [
            LogisticRegression(),
            SuperLearner(
                [
                    LogisticRegression(max_iter=200),
                    RandomForestClassifier(n_estimators=5, random_state=0),
                ]
            ),
            SuperLearner(
                [
                    ("logistic", LogisticRegression(max_iter=200)),
                    ("forest", RandomForestClassifier(n_estimators=5, random_state=0)),
                ]
            ),
            SuperLearner([LinearRegression()], task="classification"),
        ],
        ids=(
            "direct-classifier",
            "unnamed-all-classifier-library",
            "named-all-classifier-library",
            "explicit-classification-task",
        ),
    )
    def test_classification_only_saved_learner_is_refused_pre_refit(
        self, outcome_learner: Any
    ) -> None:
        result = _eligible_result()
        result.estimator.outcome_learner = outcome_learner
        with pytest.raises(CapabilityError, match="regression-capable saved outcome learner"):
            refute(result, tests=("dummy_outcome",), n_replicates=1)
        assert result.estimator.calls == []

    def test_mixed_ensemble_with_regression_candidate_is_eligible(self) -> None:
        estimator = _MeanDifferenceEstimator(
            outcome_learner=SuperLearner(
                [
                    LogisticRegression(max_iter=200),
                    RandomForestClassifier(n_estimators=5, random_state=0),
                    LinearRegression(),
                ]
            )
        )
        report = refute(
            _eligible_result(estimator=estimator),
            tests=("dummy_outcome",),
            n_replicates=2,
            outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
        )
        assert len(report["dummy_outcome"].records) == 2
        assert len(estimator.calls) == 2

    def test_real_binomial_fit_is_refused_before_refit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame, _ = make_binary_outcome(n=100, seed=23)
        result = (
            CausalStudy(
                frame,
                design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
            )
            .identify(ATE())
            .estimate(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                n_folds=2,
                simultaneous=False,
                random_state=0,
            )
        )

        def unexpected_refit(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("family preflight must run before estimator.refit")

        monkeypatch.setattr(result.estimator, "refit", unexpected_refit)
        with pytest.raises(CapabilityError, match="original outcome family='binomial'"):
            refute(result, tests=("dummy_outcome",), n_replicates=1)


class TestGeneratedOutcomeFailureBoundary:
    def test_generation_errors_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_draw(process: GaussianAdjustmentOutcome, data: Any, *, seed: int) -> np.ndarray:
            del process, data, seed
            raise DataError("broken generation")

        monkeypatch.setattr(GaussianAdjustmentOutcome, "draw", broken_draw)
        with pytest.raises(DataError, match="broken generation"):
            refute(
                _eligible_result(),
                tests=("simulated_outcome",),
                n_replicates=1,
                outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            )

    def test_replacement_errors_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_replacement(*args: Any, **kwargs: Any) -> Any:
            raise DataError("broken replacement")

        monkeypatch.setattr(CausalData, "with_outcome", broken_replacement)
        with pytest.raises(DataError, match="broken replacement"):
            refute(
                _eligible_result(),
                tests=("dummy_outcome",),
                n_replicates=1,
                outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            )

    def test_authoritative_refit_family_must_match_process(self) -> None:
        class WrongFamilyEstimator(_MeanDifferenceEstimator):
            def refit(self, *args: Any, **kwargs: Any) -> Any:
                refitted = super().refit(*args, **kwargs)
                refitted.data = SimpleNamespace(family="binomial")
                return refitted

        with pytest.raises(RuntimeError, match="authoritative family='binomial'"):
            refute(
                _eligible_result(estimator=WrongFamilyEstimator()),
                tests=("dummy_outcome",),
                n_replicates=1,
                outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            )

    def test_malformed_refit_result_propagates(self) -> None:
        class MalformedResultEstimator(_MeanDifferenceEstimator):
            def refit(self, *args: Any, **kwargs: Any) -> Any:
                del args, kwargs
                return object()

        with pytest.raises(AttributeError):
            refute(
                _eligible_result(estimator=MalformedResultEstimator()),
                tests=("dummy_outcome",),
                n_replicates=1,
                outcome_rule=EmpiricalInclusionRule(minimum_draws=1),
            )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ("nonbinary", "non-binary"),
            ("missing", "missing-outcome"),
            ("axis", "intervention-indexed"),
            ("msm", "MSM"),
            ("ratio", "ratio"),
        ],
    )
    def test_unsupported_compositions_are_refused_pre_fit(self, change: str, message: str) -> None:
        result = _eligible_result()
        # Keep the structured records intact while isolating each routing guard.
        result.data = SimpleNamespace(**vars(result.data))
        result.data.n = len(result.data.outcome)
        result.data.is_binary_treatment = len(result.data.treatment_levels) == 2
        result.data.has_missing_outcome = not bool(np.all(result.data.observed))
        result.data.has_intermediate = False
        if change == "nonbinary":
            result.data.treatment_levels = (0, 1, 2)
        elif change == "missing":
            result.data.observed = np.zeros(result.data.n, dtype=bool)
        elif change == "axis":
            result.identified_effect.functional = replace(
                result.identified_effect.functional, axis="regime"
            )
        elif change == "msm":
            result.identified_effect.functional = replace(
                result.identified_effect.functional, msm=object()
            )
        else:
            result.parameter_keys["ate"] = replace(result.parameter_keys["ate"], estimand="rr")
        result.data.is_binary_treatment = len(result.data.treatment_levels) == 2
        result.data.has_missing_outcome = not bool(np.all(result.data.observed))
        with pytest.raises(CapabilityError, match=message):
            refute(result, tests=("dummy_outcome",), n_replicates=1)
        assert result.estimator.calls == []


def test_empirical_rule_uses_inclusive_half_ties_and_failure_policy() -> None:
    rule = EmpiricalInclusionRule(alpha=0.2, minimum_draws=4)
    assert rule.pvalue((-1.0, 0.0, 0.0, 1.0), 0.0) == 1.0
    assert rule.evaluate((-1.0, 0.0, 0.0, 1.0), 0.0, ())[0]
    failure = ReplicationFailure(0, 1, "RuntimeError", "no")
    assert not rule.evaluate((-1.0, 0.0, 0.0, 1.0), 0.0, (failure,))[0]


def test_empirical_rule_rejects_probability_equal_to_alpha() -> None:
    rule = EmpiricalInclusionRule(alpha=0.2, minimum_draws=10)
    values = (-1.0, *([1.0] * 9))
    assert rule.pvalue(values, 0.0) == pytest.approx(rule.alpha)
    assert not rule.evaluate(values, 0.0, ())[0]

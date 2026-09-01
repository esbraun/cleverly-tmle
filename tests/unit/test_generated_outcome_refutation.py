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

from cleverly import ATC, ATE, ATT, CausalStudy, PointTreatment, SuperLearner
from cleverly.data import CausalData
from cleverly.datasets import make_binary_outcome, make_linear_ate
from cleverly.exceptions import CapabilityError, DataError
from cleverly.study import BackdoorMeanContrast, ExplicitAdjustmentProvider, ParameterKey
from cleverly.targets import TARGETS
from cleverly.validation import (
    DEFAULT_OUTCOME_REPLICATES,
    EmpiricalInclusionRule,
    GaussianAdjustmentOutcome,
    GaussianNoise,
    ReplicationFailure,
    refute,
)


def _small_rule(draws: int) -> EmpiricalInclusionRule:
    """Return the widest-alpha legal rule at ``draws`` successful draws.

    ``EmpiricalInclusionRule`` refuses ``minimum_draws * alpha < 2``, so a test that wants
    a small draw budget has to buy it with a wide alpha rather than with a small floor.
    """
    return EmpiricalInclusionRule(alpha=2.0 / draws, minimum_draws=draws)


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
    return _result_for(data, estimator or _MeanDifferenceEstimator())


def _result_for(data: CausalData, estimator: Any) -> Any:
    """Wrap prepared rows in the structured provenance a generated outcome requires."""
    functional = BackdoorMeanContrast(
        outcome="Y",
        treatment="A",
        adjustment=data.covariate_names,
        target="ate",
    )
    return _Result(
        estimator=estimator,
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


def _confounded_result(n_covariates: int, *, estimator: Any = None, seed: int = 11) -> Any:
    """Rows whose treatment depends on the same standardized covariate sum the process uses.

    A pipeline that ignores the covariates is biased here, and the bias does not depend on
    how many covariates carry the signal.
    """
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(400, n_covariates))
    signal = covariates.sum(axis=1) / np.sqrt(n_covariates)
    treatment = (
        rng.uniform(size=covariates.shape[0]) < 1.0 / (1.0 + np.exp(-1.5 * signal))
    ).astype(float)
    data = CausalData.from_arrays(
        rng.normal(size=treatment.size),
        treatment,
        covariates,
        covariate_names=tuple(f"W{index + 1}" for index in range(n_covariates)),
    )
    return _result_for(data, estimator or _MeanDifferenceEstimator())


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

    def test_a_refusal_names_the_argument_the_caller_passed(self) -> None:
        data = _eligible_result().data
        with pytest.raises(DataError, match="negative_control_outcome has length"):
            data.with_outcome(np.zeros(data.n - 1), name="negative_control_outcome")

    def test_the_negative_control_branch_names_its_own_argument(self) -> None:
        result = _eligible_result()
        with pytest.raises(DataError, match="negative_control_outcome has length"):
            refute(
                result,
                tests=("negative_control_outcome",),
                negative_control_outcome=np.zeros(result.data.n - 1),
            )

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
        rule = _small_rule(20)
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
            outcome_rule=_small_rule(20),
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
            outcome_rule=_small_rule(4),
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
            outcome_rule=_small_rule(5),
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
            outcome_rule=_small_rule(3),
            random_state=5,
        )["dummy_outcome"]
        assert len(estimator.calls) == 3
        assert [record.replicate for record in test.records] == [0, 2]
        assert [failure.replicate for failure in test.failures] == [1]
        assert test.failures[0].error_type == "ValueError"
        assert not test.passed

    def test_all_failure_behavior_keeps_every_seed(self) -> None:
        result = _eligible_result(estimator=_MeanDifferenceEstimator(fail_calls=(0, 1, 2)))
        report = refute(
            result,
            tests=("dummy_outcome",),
            n_replicates=3,
            outcome_rule=_small_rule(3),
            random_state=9,
        )
        test = report["dummy_outcome"]
        assert not test.passed
        assert not test.records
        assert len(test.failures) == 3
        assert len(test.child_seeds) == 3
        assert np.isnan(test.mean)
        # No draw succeeded, so there is no rank information. A reported zero would read
        # as the strongest possible rejection instead.
        assert test.empirical_pvalue is None
        frame = nw.from_native(report.to_frame(), eager_only=True)
        assert frame["empirical_pvalue"].is_null().all()

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_report_tables_use_the_input_backend(self, backend: str) -> None:
        result = _eligible_result(backend=backend)
        report = refute(
            result,
            tests=("dummy_outcome",),
            n_replicates=3,
            outcome_rule=_small_rule(3),
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
            n_replicates=3,
            outcome_rule=_small_rule(3),
        )
        assert len(report["dummy_outcome"].records) == 3
        assert len(estimator.calls) == 3

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
                n_replicates=3,
                outcome_rule=_small_rule(3),
            )

    def test_replacement_errors_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_replacement(*args: Any, **kwargs: Any) -> Any:
            raise DataError("broken replacement")

        monkeypatch.setattr(CausalData, "with_outcome", broken_replacement)
        with pytest.raises(DataError, match="broken replacement"):
            refute(
                _eligible_result(),
                tests=("dummy_outcome",),
                n_replicates=3,
                outcome_rule=_small_rule(3),
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
                n_replicates=3,
                outcome_rule=_small_rule(3),
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
                n_replicates=3,
                outcome_rule=_small_rule(3),
            )

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ("nonbinary", "non-binary"),
            ("missing", "missing-outcome"),
            ("axis", "intervention-indexed"),
            ("msm", "MSM"),
            ("ratio", "ratio"),
            ("longitudinal", "longitudinal"),
            ("intermediate-data", "controlled-direct-effect"),
            ("intermediate-functional", "controlled-direct-effect"),
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
        elif change == "longitudinal":
            result.identified_effect.functional = replace(
                result.identified_effect.functional, longitudinal=True
            )
        elif change == "intermediate-data":
            result.data.has_intermediate = True
        elif change == "intermediate-functional":
            result.identified_effect.functional = replace(
                result.identified_effect.functional, intermediate=0.0
            )
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
    rule = EmpiricalInclusionRule(alpha=0.5, minimum_draws=4)
    assert rule.pvalue((-1.0, 0.0, 0.0, 1.0), 0.0) == 1.0
    assert rule.evaluate((-1.0, 0.0, 0.0, 1.0), 0.0, ())[0]
    failure = ReplicationFailure(0, 1, "RuntimeError", "no")
    assert not rule.evaluate((-1.0, 0.0, 0.0, 1.0), 0.0, (failure,))[0]


def test_empirical_rule_rejects_probability_equal_to_alpha() -> None:
    rule = EmpiricalInclusionRule(alpha=0.2, minimum_draws=10)
    values = (-1.0, *([1.0] * 9))
    assert rule.pvalue(values, 0.0) == pytest.approx(rule.alpha)
    assert not rule.evaluate(values, 0.0, ())[0]


def _linear_ate_fit(estimand: Any, *, labels: bool = False, n: int = 300, seed: int = 17) -> Any:
    """Fit a real TMLE on ``make_linear_ate`` with explicit parametric learners.

    The learners are passed rather than defaulted so the fit stays inside the fast tier;
    the default learner library costs 30 to 120 seconds per fit.
    """
    frame, _ = make_linear_ate(n=n, seed=seed)
    if labels:
        frame = frame.assign(A=np.where(np.asarray(frame["A"]) == 1.0, "treated", "control"))
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
    )
    return study.identify(estimand).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        simultaneous=False,
        random_state=seed,
    )


class TestContrastDirection:
    """The declared effect follows the parameter key, not the process alone."""

    def test_a_non_default_reference_declares_the_negated_effect(self) -> None:
        process = GaussianAdjustmentOutcome(effect=1.0)
        rule = _small_rule(10)
        forward = refute(
            _linear_ate_fit(ATE()),
            tests=("simulated_outcome",),
            simulated_outcome=process,
            n_replicates=20,
            outcome_rule=rule,
            random_state=5,
        )["simulated_outcome"]
        # ATE(reference=1) reports code zero against code one, so the same process
        # declares the negated effect and the refits centre there.
        reversed_key = refute(
            _linear_ate_fit(ATE(reference=1)),
            tests=("simulated_outcome",),
            simulated_outcome=process,
            n_replicates=20,
            outcome_rule=rule,
            random_state=5,
        )["simulated_outcome"]
        assert forward.declared_effect == 1.0
        assert reversed_key.declared_effect == -1.0
        assert reversed_key.declared_effect == -forward.declared_effect
        assert forward.passed and reversed_key.passed
        assert forward.mean == pytest.approx(1.0, abs=0.3)
        assert reversed_key.mean == pytest.approx(-1.0, abs=0.3)
        assert "includes -1" in reversed_key.expectation

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_a_non_default_reference_negates_the_effect_for_att_and_atc(
        self, estimand: str
    ) -> None:
        # ate, att, and atc build their keys at one site, so the reversed reference has to
        # negate the declared effect for all three rather than for ate alone.
        target = {"att": ATT, "atc": ATC}[estimand]
        test = refute(
            _linear_ate_fit(target(reference=1)),
            estimand=estimand,
            tests=("simulated_outcome",),
            simulated_outcome=GaussianAdjustmentOutcome(effect=1.0),
            n_replicates=20,
            outcome_rule=_small_rule(10),
            random_state=5,
        )["simulated_outcome"]
        assert test.declared_effect == -1.0
        assert test.passed
        assert test.mean == pytest.approx(-1.0, abs=0.3)

    def test_a_string_labelled_reference_resolves_to_its_arm_code(self) -> None:
        # "control" sorts before "treated", so reference="treated" is the non-default arm.
        test = refute(
            _linear_ate_fit(ATE(reference="treated"), labels=True),
            tests=("simulated_outcome",),
            simulated_outcome=GaussianAdjustmentOutcome(effect=1.0),
            n_replicates=20,
            outcome_rule=_small_rule(10),
            random_state=5,
        )["simulated_outcome"]
        assert test.declared_effect == -1.0
        assert test.passed

    def test_an_unresolvable_key_endpoint_is_refused_pre_refit(self) -> None:
        result = _eligible_result()
        result.parameter_keys["ate"] = replace(result.parameter_keys["ate"], value="unknown-arm")
        with pytest.raises(CapabilityError, match="the parameter key value="):
            refute(result, tests=("simulated_outcome",), n_replicates=40)
        assert result.estimator.calls == []

    def test_a_one_arm_key_is_refused_pre_refit(self) -> None:
        result = _eligible_result()
        result.parameter_keys["ate"] = replace(result.parameter_keys["ate"], value=0, reference=0)
        with pytest.raises(CapabilityError, match="value and reference are the same arm"):
            refute(result, tests=("simulated_outcome",), n_replicates=40)
        assert result.estimator.calls == []

    def test_an_absent_key_endpoint_falls_back_to_the_declared_arms(self) -> None:
        result = _eligible_result()
        # A key with neither endpoint named contrasts the second arm against the first.
        result.parameter_keys["ate"] = replace(
            result.parameter_keys["ate"], value=None, reference=None
        )
        test = refute(
            result,
            tests=("simulated_outcome",),
            n_replicates=40,
            outcome_rule=_small_rule(20),
            random_state=6,
        )["simulated_outcome"]
        assert test.declared_effect == 1.0
        assert test.passed


class TestRealFitAcceptance:
    """A genuine study, fitted and refitted, recovers the declared generated effect."""

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
    def test_real_fits_recover_the_declared_generated_effect(self, estimand: str) -> None:
        target = {"ate": ATE(), "att": ATT(), "atc": ATC()}[estimand]
        process = GaussianAdjustmentOutcome(effect=0.75, adjustment_scale=1.0)
        report = refute(
            _linear_ate_fit(target),
            estimand=estimand,
            tests=("simulated_outcome",),
            simulated_outcome=process,
            n_replicates=20,
            outcome_rule=_small_rule(10),
            random_state=13,
        )
        test = report["simulated_outcome"]
        assert test.declared_effect == 0.75
        assert test.passed
        assert not test.failures
        assert len(test.records) == 20
        # The generated effect is constant, so ate, att, and atc all equal it.
        assert test.mean == pytest.approx(0.75, abs=0.25)

    def test_a_real_fit_passes_under_the_shipped_default_rule(self) -> None:
        # Every other real-fit case widens alpha to keep its draw budget small, so the
        # shipped alpha 0.05 and 40-draw minimum would otherwise run on no real fit.
        report = refute(
            _linear_ate_fit(ATE()),
            tests=("simulated_outcome",),
            simulated_outcome=GaussianAdjustmentOutcome(effect=0.75),
            n_replicates=40,
            random_state=13,
        )
        test = report["simulated_outcome"]
        assert test.rule == EmpiricalInclusionRule()
        assert test.rule.alpha == 0.05
        assert test.declared_effect == 0.75
        assert test.passed
        assert len(test.records) == 40

    def test_the_refute_cache_key_is_value_based(self) -> None:
        result = _linear_ate_fit(ATE())
        options: dict[str, Any] = {
            "tests": ("simulated_outcome",),
            "n_replicates": 4,
            "random_state": 7,
        }
        first = result.diagnostics.refute(
            simulated_outcome=GaussianAdjustmentOutcome(effect=0.5),
            outcome_rule=_small_rule(4),
            **options,
        )
        # Fresh declarations, equal by value and distinct by identity.
        second = result.diagnostics.refute(
            simulated_outcome=GaussianAdjustmentOutcome(effect=0.5),
            outcome_rule=_small_rule(4),
            **options,
        )
        assert first is second
        assert len([key for key in result.assessment_cache if "refute" in key]) == 1


class TestAdjustmentSignalScale:
    """The confounding signal keeps its scale as the adjustment set widens."""

    @pytest.mark.parametrize("n_covariates", [1, 2, 5, 20, 50])
    def test_the_adjustment_standard_deviation_is_invariant_to_the_column_count(
        self, n_covariates: int
    ) -> None:
        rng = np.random.default_rng(2)
        covariates = rng.normal(size=(4000, n_covariates))
        process = GaussianAdjustmentOutcome(adjustment_scale=1.0)
        values = process.adjustment(covariates)
        # Under a mean reduction this would be about 1/sqrt(n_covariates), so the signal
        # would vanish against the fixed noise standard deviation of one.
        assert float(np.std(values)) == pytest.approx(1.0, abs=0.1)

    def test_constant_and_partly_constant_covariates_keep_the_active_count(self) -> None:
        rng = np.random.default_rng(3)
        active = rng.normal(size=(4000, 4))
        padded = np.hstack([active, np.ones((4000, 6))])
        process = GaussianAdjustmentOutcome(adjustment_scale=1.0)
        assert process.adjustment(padded) == pytest.approx(process.adjustment(active))
        assert process.adjustment(np.ones((10, 3))) == pytest.approx(np.zeros(10))

    @pytest.mark.parametrize("n_covariates", [2, 50])
    def test_an_unadjusted_estimator_is_detected_on_a_wide_adjustment_set(
        self, n_covariates: int
    ) -> None:
        # The stub takes a raw difference in means and never reads the covariates, while
        # treatment here depends on them. That bias must be visible at any column count.
        result = _confounded_result(n_covariates)
        test = refute(
            result,
            tests=("simulated_outcome",),
            simulated_outcome=GaussianAdjustmentOutcome(effect=1.0),
            n_replicates=40,
            outcome_rule=_small_rule(20),
            random_state=4,
        )["simulated_outcome"]
        assert not test.passed
        assert test.empirical_pvalue == 0.0
        assert abs(test.mean - 1.0) > 3.0 * test.spread

    def test_a_dummy_outcome_still_passes_under_the_same_confounding(self) -> None:
        # The control for the test above: independent noise carries no covariate signal,
        # so the same unadjusted stub is unbiased for it.
        test = refute(
            _confounded_result(50),
            tests=("dummy_outcome",),
            n_replicates=40,
            outcome_rule=_small_rule(20),
            random_state=4,
        )["dummy_outcome"]
        assert test.passed


class TestDrawBudget:
    def test_a_budget_below_the_rule_is_refused_before_any_refit(self) -> None:
        result = _eligible_result()
        with pytest.raises(CapabilityError, match="asked for 10 draw"):
            refute(result, tests=("dummy_outcome",), n_replicates=10)
        assert result.estimator.calls == []

    def test_a_mixed_call_refuses_the_budget_before_a_placebo_refit(self) -> None:
        result = _eligible_result()
        with pytest.raises(CapabilityError, match="requires 40"):
            refute(result, tests=("placebo", "dummy_outcome"), n_replicates=10)
        assert result.estimator.calls == []

    def test_the_declared_defaults_request_a_workable_budget(self) -> None:
        rule = EmpiricalInclusionRule()
        assert rule.alpha == 0.05
        assert rule.minimum_draws == 40
        assert rule.failure_policy == "fail"
        assert DEFAULT_OUTCOME_REPLICATES == 100
        assert rule.minimum_draws <= DEFAULT_OUTCOME_REPLICATES
        result = _eligible_result()
        test = refute(result, tests=("dummy_outcome",), random_state=2)["dummy_outcome"]
        assert len(result.estimator.calls) == DEFAULT_OUTCOME_REPLICATES
        assert test.n_replicates == DEFAULT_OUTCOME_REPLICATES
        assert test.rule == rule


class TestDeclarationValidation:
    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (lambda: GaussianNoise(mean=np.inf), "mean must be finite"),
            (lambda: GaussianNoise(mean=np.nan), "mean must be finite"),
            (lambda: GaussianNoise(standard_deviation=0.0), "finite and positive"),
            (lambda: GaussianNoise(standard_deviation=-1.0), "finite and positive"),
            (lambda: GaussianNoise(standard_deviation=np.nan), "finite and positive"),
            (lambda: GaussianAdjustmentOutcome(effect=np.nan), "effect must be finite"),
            (
                lambda: GaussianAdjustmentOutcome(adjustment_scale=np.inf),
                "adjustment_scale must be finite",
            ),
            (lambda: EmpiricalInclusionRule(alpha=0.0), "between zero and one"),
            (lambda: EmpiricalInclusionRule(alpha=1.0), "between zero and one"),
            (
                lambda: EmpiricalInclusionRule(alpha=0.05, minimum_draws=0),
                "minimum_draws must be positive",
            ),
            (
                lambda: EmpiricalInclusionRule(alpha=0.05, minimum_draws=2),
                r"minimum_draws >= 40 at alpha=0\.05",
            ),
            (
                lambda: EmpiricalInclusionRule(alpha=0.1, minimum_draws=19),
                r"minimum_draws >= 20 at alpha=0\.1",
            ),
            (
                lambda: EmpiricalInclusionRule(failure_policy="keep"),  # type: ignore[arg-type]
                "failure_policy=",
            ),
        ],
    )
    def test_unusable_declarations_are_refused(self, build: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            build()

    def test_the_shipped_default_rule_constructs(self) -> None:
        rule = EmpiricalInclusionRule()
        assert rule.minimum_draws * rule.alpha >= 2.0
        assert rule.minimum_draw_count == 40

    def test_a_nonpositive_replicate_count_is_refused(self) -> None:
        result = _eligible_result()
        with pytest.raises(ValueError, match="n_replicates must be positive"):
            refute(result, tests=("dummy_outcome",), n_replicates=0)
        assert result.estimator.calls == []


class TestDummyOutcomeNegativeControl:
    def test_a_manufactured_effect_on_a_null_outcome_fails(self) -> None:
        class LeakyEstimator(_MeanDifferenceEstimator):
            """Adds a fixed effect the null outcome cannot support."""

            def refit(self, *args: Any, **kwargs: Any) -> Any:
                refitted = super().refit(*args, **kwargs)
                refitted._estimate.psi += 0.5
                return refitted

        test = refute(
            _eligible_result(estimator=LeakyEstimator()),
            tests=("dummy_outcome",),
            n_replicates=40,
            outcome_rule=_small_rule(20),
            random_state=8,
        )["dummy_outcome"]
        assert test.declared_effect == 0.0
        assert not test.passed
        assert test.empirical_pvalue == 0.0
        # Not the degenerate all-ties case the rank rule scores as p=1: the draws vary.
        assert test.spread > 0.0
        assert len(set(test.values)) == len(test.values)

    def test_a_constant_estimator_is_the_degenerate_case_the_rank_rule_passes(self) -> None:
        class ConstantEstimator(_MeanDifferenceEstimator):
            """Returns the declared effect exactly, so every draw ties with it."""

            def refit(self, *args: Any, **kwargs: Any) -> Any:
                refitted = super().refit(*args, **kwargs)
                refitted._estimate.psi = 0.0
                return refitted

        test = refute(
            _eligible_result(estimator=ConstantEstimator()),
            tests=("dummy_outcome",),
            n_replicates=40,
            outcome_rule=_small_rule(20),
            random_state=8,
        )["dummy_outcome"]
        assert test.empirical_pvalue == 1.0
        assert test.passed
        assert test.spread == 0.0

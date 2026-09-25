"""A saved cross-fitted result on an undeclared outcome scale loads without an interval.

Roadmap row RM33. Releases 0.1.0 and 0.1.1 set ``q_bounds=None`` by default, so a
continuous outcome took its scale from every observed outcome, held-out rows included.
This version refuses that scale under cross-fitting, because no shipped result covers it.
RM31 withholds the interval of each saved cross-fitted result of a discrete treatment,
because those releases stratified its split. A continuous dose drew an unstratified split,
so RM31 leaves it. A result on the undeclared scale now loads under
``"undeclared_scale_plugin"``: its point estimates stand, and ``ci``, ``pvalue`` and
``std_error`` refuse.

The witness fits the roadmap probe with the old entry rule restored in the test only, by
a patch of the raise wrapper ``TMLE._refuse_unbounded_cross_fitted_scale``. The patch
never touches the predicate ``TMLE._outcome_scale_refusal``, which the status reads. Each
mutation must fail the check that its witness or control passes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from cleverly import variable_importance
from cleverly.assessment import POINT_REPLAY_REFIT_CONFIGURATION, replayability
from cleverly.data import CausalData
from cleverly.datasets import make_clustered, make_linear_ate
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.inference.cluster import cluster_inference_status
from cleverly.targets.base import parameter_stem
from tests.conftest import linear_drtmle, linear_in_sample
from tests.unit._capability_sweep_support import (
    as_saved_by_v011,
    dose_frame,
    fit_shift,
    outcome_bounds,
    reconfigured,
)
from tests.unit._declaration_support import assert_replay_agrees
from tests.unit._inference_status_support import (
    ROUTES,
    assert_assessment_note,
    assert_evalue_unavailable,
    assert_fold_reports_restamped,
    assert_keeps_inference,
    assert_restamped,
    assert_variable_importance_refuses,
    assert_withholds,
    legacy_copy,
    restore,
)
from tests.unit._natural_course_support import NeverFit, never_fit_learners

pytestmark = pytest.mark.xdist_group("saved_scale_status")

STATUS = "undeclared_scale_plugin"

#: The roadmap probe of RM33: ``ey_shift[+0.5]`` and its interval as release 0.1.1 saved
#: them, which the patched fit of this version reproduces.
PROBE_PSI = 3.798198
PROBE_CI = (3.3419, 4.2545)


@contextmanager
def the_old_entry_rule() -> Iterator[None]:
    """The scale refusal of releases 0.1.0 and 0.1.1, which was none, for one fit.

    Only the raise wrapper is patched. The predicate that the status reads stays whole, so
    the live fit already stamps the status, and :func:`legacy_copy` rewrites each saved
    estimate to ``"influence_curve"`` as those releases saved it.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TMLE, "_refuse_unbounded_cross_fitted_scale", lambda self, data: None)
        yield


def saved_by_v011(result: Any) -> Any:
    """``result`` as release 0.1.1 saved it: the default policy, and no ``split_plan``."""
    return as_saved_by_v011(reconfigured(result, stratify_folds="treatment"))


def copied_ate(engine: type[TMLE], **settings: Any) -> Any:
    """A cross-fitted ATE fit with a declared scale, restored with ``q_bounds=None``.

    The copied-estimator shape: a discrete treatment under ``"none"``, which no release
    saved. ``make_linear_ate(400, 2)`` has a continuous outcome.
    """
    frame, _ = make_linear_ate(n=400, seed=2)
    bounds = outcome_bounds(frame)
    if engine is DRTMLE:
        estimator: TMLE = DRTMLE(**linear_drtmle(n_folds=2, q_bounds=bounds, **settings))
    else:
        estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2, q_bounds=bounds, **settings))
    result = estimator.fit(frame, outcome="Y", treatment="A").single()
    return reconfigured(result, q_bounds=None)


@pytest.fixture(scope="module")
def saved_dose() -> Any:
    """The roadmap probe: a cross-fitted dose of a continuous outcome with ``q_bounds=None``."""
    with the_old_entry_rule():
        result = fit_shift(cross_fit=True, n_folds=2)
    return saved_by_v011(result)


@pytest.fixture(scope="module")
def copied_drtmle() -> Any:
    """A copied ``DRTMLE`` estimator, which reaches the rule through ``super()``."""
    return copied_ate(DRTMLE, estimands=("ate",))


@pytest.fixture(scope="module")
def fold_evaluated() -> Any:
    """A copied ``TMLE`` ATE fit with ``cv_evaluation=True``, which adds two fold reports."""
    return copied_ate(TMLE, estimands=("ate",), cv_evaluation=True)


@pytest.fixture(scope="module")
def saved_discrete() -> Any:
    """A cross-fitted discrete-treatment fit on the undeclared scale, as release 0.1.1 saved it."""
    return reconfigured(copied_ate(TMLE, estimands=("ate",)), stratify_folds="treatment")


@pytest.fixture(scope="module")
def few_clustered() -> Any:
    """A copied cross-fitted fit of 39 equal clusters, which a cluster status also reaches.

    ``make_clustered(n=390, cluster_size=10, seed=7)`` is the few-cluster frame of
    ``tests/unit/test_cluster_status.py``.
    """
    frame = make_clustered(n=390, cluster_size=10, seed=7)[0]
    estimator = TMLE(**linear_in_sample(cross_fit=True, n_folds=2, q_bounds=(-15.0, 15.0)))
    result = estimator.fit(frame, outcome="Y", treatment="A", id="cluster").single()
    return reconfigured(result, q_bounds=None)


@pytest.fixture(scope="module")
def declared() -> Any:
    """The probe dose with a declared ``q_bounds``, a fit that this version runs."""
    frame = dose_frame()
    return saved_by_v011(
        fit_shift(frame, cross_fit=True, n_folds=2, q_bounds=outcome_bounds(frame))
    )


@pytest.fixture(scope="module")
def in_sample_dose() -> Any:
    """The probe dose fitted in sample, with ``q_bounds=None``. It draws no split."""
    return saved_by_v011(fit_shift())


@pytest.fixture(scope="module")
def binary_dose() -> Any:
    """The probe dose of a binary outcome, cross-fitted with ``q_bounds=None``."""
    frame = dose_frame()
    frame = frame.assign(Y=(frame["Y"] > frame["Y"].median()).astype(int))
    result = saved_by_v011(fit_shift(frame, cross_fit=True, n_folds=2))
    assert result.data.family == "binomial"
    return result


class TestASavedUndeclaredScaleResult:
    """RM33: a saved cross-fitted result on an undeclared scale withholds its interval."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_restored_artifact_takes_the_status(self, saved_dose: Any, route: str) -> None:
        # The nonzero witness: the saved copy publishes the interval of the roadmap probe.
        legacy = legacy_copy(saved_dose)
        assert legacy.estimator.stratify_folds == "treatment"
        assert legacy.estimator.q_bounds is None
        assert legacy.data.family == "gaussian"
        assert legacy.estimator.crossfit_plan(legacy.data).stratify_by == ()
        (estimate,) = legacy.estimates.values()
        assert round(estimate.psi, 6) == PROBE_PSI
        assert tuple(round(bound, 4) for bound in estimate.ci) == PROBE_CI
        restored = assert_restamped(saved_dose, STATUS, route)
        assert_withholds(restored, STATUS)
        assert_assessment_note(restored, STATUS)
        (kept,) = restored.estimates.values()
        assert kept.psi == estimate.psi
        assert tuple(round(bound, 4) for bound in kept.plugin_interval) == PROBE_CI

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_copied_estimator_takes_the_status(self, copied_drtmle: Any, route: str) -> None:
        restored = assert_restamped(copied_drtmle, STATUS, route)
        assert_withholds(restored, STATUS)
        assert_evalue_unavailable(restored, STATUS)

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_fold_level_reports_take_the_status(self, fold_evaluated: Any, route: str) -> None:
        restored = assert_restamped(fold_evaluated, STATUS, route)
        assert_fold_reports_restamped(restored, fold_evaluated, STATUS)

    def test_the_replay_slots_agree_and_a_retarget_takes_the_status(self, saved_dose: Any) -> None:
        restored = restore(legacy_copy(saved_dose), "pickle")
        estimands = tuple(dict.fromkeys(parameter_stem(name) for name in restored.estimates))
        assert_replay_agrees(restored, estimands)
        replay = replayability(restored)
        assert replay.retarget_cached_nuisances
        assert not replay.refit_nuisances
        assert replay.unreconstructible == (POINT_REPLAY_REFIT_CONFIGURATION,)
        estimates, _ = restored.estimator.retarget(
            restored.data, restored.nuisance, estimands=estimands
        )
        assert [estimate.inference for estimate in estimates.values()] == [STATUS]
        curve = restored.diagnostics.truncation_curve(bounds=[0.05])
        assert len(curve) > 0


class TestThePrecedence:
    """RM33 is order 7: after the saved stratified split, before the cluster statuses."""

    def test_a_saved_stratified_split_comes_first(self, saved_discrete: Any) -> None:
        # The nonzero witness: the same estimator meets the scale rule on its own.
        assert saved_discrete.estimator._saved_scale_status(saved_discrete.data) == STATUS
        restored = assert_restamped(saved_discrete, "stratified_fold_plugin", "pickle")
        assert_withholds(restored, "stratified_fold_plugin")

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_status_comes_before_a_cluster_status(self, few_clustered: Any, route: str) -> None:
        # The nonzero witness: the same labels take the few-cluster status on their own.
        data = few_clustered.data
        assert cluster_inference_status(data.cluster, cross_fit=True) == "few_cluster_plugin"
        restored = assert_restamped(few_clustered, STATUS, route)
        assert_withholds(restored, STATUS)


class TestTheSupportedArtifactsKeepTheirIntervals:
    """The controls: a declared scale, no split, or a binary outcome."""

    def test_a_declared_scale(self, declared: Any) -> None:
        assert declared.estimator.q_bounds is not None
        assert_keeps_inference(restore(declared, "pickle"))

    def test_an_in_sample_result(self, in_sample_dose: Any) -> None:
        assert not in_sample_dose.estimator.cross_fit
        assert in_sample_dose.estimator.q_bounds is None
        assert_keeps_inference(restore(in_sample_dose, "pickle"))

    def test_a_binary_outcome(self, binary_dose: Any) -> None:
        assert binary_dose.estimator.cross_fit
        assert binary_dose.estimator.q_bounds is None
        assert_keeps_inference(restore(binary_dose, "pickle"))

    def test_a_live_fit_with_a_declared_scale(self) -> None:
        """No live fit changes: every live fit on an undeclared scale is refused first."""
        frame = dose_frame()
        result = fit_shift(frame, cross_fit=True, n_folds=2, q_bounds=outcome_bounds(frame))
        assert_keeps_inference(result)
        with pytest.raises(CapabilityError, match="needs a declared q_bounds"):
            fit_shift(frame, cross_fit=True, n_folds=2)


def _without_cross_fitting(self: TMLE, data: Any) -> str | None:
    """The mutant predicate that forgets the cross-fitting condition."""
    return "mutant" if data.family != "binomial" and self.q_bounds is None else None


def _without_the_family(self: TMLE, data: Any) -> str | None:
    """The mutant predicate that forgets the outcome family."""
    return "mutant" if self.cross_fit and self.q_bounds is None else None


def _without_the_declaration(self: TMLE, data: Any) -> str | None:
    """The mutant predicate that forgets the declared ``q_bounds``."""
    return "mutant" if self.cross_fit and data.family != "binomial" else None


class TestTheMutationsFailTheWitness:
    """Each mutation of the rule fails the witness or the control it names."""

    def test_skipping_the_decision_fails_the_witness(
        self, saved_dose: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_saved_scale_status", lambda self, data: "influence_curve")
        with pytest.raises(AssertionError):
            assert_restamped(saved_dose, STATUS, "pickle")

    def test_dropping_the_cross_fitting_condition_fails_the_in_sample_control(
        self, in_sample_dose: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_outcome_scale_refusal", _without_cross_fitting)
        with pytest.raises(AssertionError):
            assert_keeps_inference(restore(in_sample_dose, "pickle"))

    def test_dropping_the_family_condition_fails_the_binary_control(
        self, binary_dose: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_outcome_scale_refusal", _without_the_family)
        with pytest.raises(AssertionError):
            assert_keeps_inference(restore(binary_dose, "pickle"))

    def test_dropping_the_declaration_condition_fails_the_declared_control(
        self, declared: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_outcome_scale_refusal", _without_the_declaration)
        with pytest.raises(AssertionError):
            assert_keeps_inference(restore(declared, "pickle"))


# ------------------------------------------------------------------ variable importance


def undeclared_estimator() -> TMLE:
    """A cross-fitted estimator with ``q_bounds=None`` and learners that count each fit."""
    return TMLE(**linear_in_sample(cross_fit=True, n_folds=2, **never_fit_learners()))


class TestVariableImportance:
    """``variable_importance`` raises the scale refusal before the status, and before a fit."""

    def test_a_run_under_an_undeclared_scale_meets_the_scale_refusal(self) -> None:
        """The refusal names the remedy, and it arrives before any learner."""
        frame, _ = make_linear_ate(n=400, seed=2)
        estimator = undeclared_estimator()
        prepared = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=("W1", "W2"))
        with pytest.raises(CapabilityError) as raised:
            variable_importance(
                frame, outcome="Y", candidates=["A"], covariates=["W1", "W2"], estimator=estimator
            )
        assert str(raised.value) == estimator._outcome_scale_refusal(prepared)
        assert "Declare the known outcome support" in str(raised.value)
        assert NeverFit.calls == 0

    def test_without_the_scale_check_the_status_reason_answers_instead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control for the order: the status hook alone gives a reason with no remedy."""
        monkeypatch.setattr(TMLE, "_refuse_unbounded_cross_fitted_scale", lambda self, data: None)
        frame, _ = make_linear_ate(n=400, seed=2)
        assert_variable_importance_refuses(
            STATUS, frame, estimator=undeclared_estimator(), covariates=["W1", "W2"]
        )

"""A saved cross-fitted result whose split read the treatment loads without an interval.

Roadmap row RM31. Releases 0.1.0 and 0.1.1 drew every cross-fitted split of a discrete
treatment under ``stratify_folds="treatment"``, their default, and this version refuses
that policy because no shipped result covers it. A result that either release saved loads
under ``"stratified_fold_plugin"``: its point estimates stand, and ``ci``, ``pvalue`` and
``std_error`` refuse. ``"treatment+outcome"`` takes the same status. Both releases also
stratified the outer ``LTMLE`` split on the first treatment node, so a restored
longitudinal result whose split records no :attr:`~cleverly.learners.Folds.origin` takes
it too.

Each point-treatment witness fits with the old entry rule restored in the test only, by
the patch of ``tests/unit/test_fold_policy_rules.py``, and then loads a legacy copy. The
longitudinal witness fits the study seam that draws the retired split. Each mutation must
fail the check that its witness or control passes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._inference_status import NO_SIMULTANEOUS_BANDS
from cleverly.assessment import POINT_REPLAY_REFIT_CONFIGURATION, replayability
from cleverly.datasets import make_longitudinal, make_shift_dose
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Shift
from cleverly.learners import crossfit
from cleverly.longitudinal import LTMLE
from tests import discrete_law
from tests.conftest import linear_drtmle, linear_in_sample
from tests.studies.ltmle_crossfit_properties import FirstNodeStratifiedLTMLE
from tests.unit._capability_sweep_support import as_saved_by_v011, ctmle_stratified, reconfigured
from tests.unit._declaration_support import assert_replay_agrees
from tests.unit._inference_status_support import (
    ROUTES,
    assert_assessment_note,
    assert_evalue_unavailable,
    assert_fold_reports_restamped,
    assert_keeps_inference,
    assert_restamped,
    assert_withholds,
    legacy_copy,
    restore,
)

pytestmark = pytest.mark.xdist_group("saved_fold_policy_status")

#: The estimator module, whose ``_saved_split_status`` and ``_refit_bound`` the mutations
#: patch. The package re-exports a function named ``ltmle``, so the module is imported by
#: its path.
longitudinal_estimator = importlib.import_module("cleverly.longitudinal.estimator")

STATUS = "stratified_fold_plugin"
SAVED_GROUPED = "cross_fitted_longitudinal_plugin"
POLICIES = ("treatment", "treatment+outcome")

#: One builder per point-treatment engine. ``DRTMLE`` reaches the rule through ``super()``.
ENGINES: dict[str, Callable[..., Any]] = {
    "tmle": lambda **settings: TMLE(**linear_in_sample(**settings)),
    "drtmle": lambda **settings: DRTMLE(**linear_drtmle(estimands=("ate",), **settings)),
}


def fit_discrete(engine: str, policy: str, **settings: Any) -> Any:
    """A two-fold fit of the discrete law under ``policy``, as release 0.1.1 drew it.

    Releases 0.1.0 and 0.1.1 accepted the policy. The patch restores that entry rule for
    this fit only, so current public fits still refuse the policy before any learner.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(crossfit, "fold_strata_refusal", lambda *args, **kwargs: None)
        estimator = ENGINES[engine](cross_fit=True, n_folds=2, stratify_folds=policy, **settings)
        return estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def stratified() -> dict[tuple[str, str], Any]:
    """One stratified cross-fitted fit per engine and policy."""
    return {
        (engine, policy): fit_discrete(engine, policy) for engine in ENGINES for policy in POLICIES
    }


@pytest.fixture(scope="module")
def in_sample() -> Any:
    """An in-sample fit that carries the policy. It draws no split, so ``__init__`` admits it."""
    estimator = TMLE(**linear_in_sample(stratify_folds="treatment"))
    return estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()


def keeps_the_dose_interval() -> Any:
    """A cross-fitted continuous-dose fit, restored with the policy that release 0.1.1 wrote.

    Release 0.1.1 drew no strata for a dose, so the split read no treatment. The scale is
    declared, so the fit is one that this version runs. RM33 holds the undeclared scale.
    """
    frame, _ = make_shift_dose(n=300, seed=0)
    bounds = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
    estimator = TMLE(
        **linear_in_sample(shifts=[Shift(0.5, cap=5.0)], cross_fit=True, n_folds=2, q_bounds=bounds)
    )
    result = estimator.fit(frame, outcome="Y", treatment="A").single()
    return reconfigured(result, stratify_folds="treatment")


class TestASavedStratifiedResult:
    """RM31: a saved stratified cross-fitted point result withholds its interval."""

    @pytest.mark.parametrize("route", ROUTES)
    @pytest.mark.parametrize("policy", POLICIES)
    @pytest.mark.parametrize("engine", list(ENGINES))
    def test_the_restored_artifact_takes_the_status(
        self, stratified: dict[tuple[str, str], Any], engine: str, policy: str, route: str
    ) -> None:
        result = stratified[engine, policy]
        restored = assert_restamped(result, STATUS, route)
        assert_withholds(restored, STATUS)
        assert_assessment_note(restored, STATUS)
        assert_evalue_unavailable(restored, STATUS)

    @pytest.mark.parametrize("policy", POLICIES)
    def test_the_shape_that_release_0_1_1_saved_takes_the_status(
        self, stratified: dict[tuple[str, str], Any], policy: str
    ) -> None:
        """Release 0.1.1 saved no ``split_plan``, so one witness covers both releases."""
        restored = as_saved_by_v011(legacy_copy(stratified["tmle", policy]))
        assert restored.estimator.stratify_folds == policy
        assert_withholds(restored, STATUS)

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_fold_level_reports_take_the_status(self, route: str) -> None:
        result = fit_discrete("tmle", "treatment", cv_evaluation=True, estimands=("ate",))
        restored = assert_restamped(result, STATUS, route)
        assert_fold_reports_restamped(restored, result, STATUS)

    def test_the_replay_slots_agree_and_a_retarget_takes_the_status(
        self, stratified: dict[tuple[str, str], Any]
    ) -> None:
        restored = restore(legacy_copy(stratified["tmle", "treatment"]), "pickle")
        assert_replay_agrees(restored, ("ate",))
        replay = replayability(restored)
        assert replay.retarget_cached_nuisances
        assert not replay.refit_nuisances
        assert replay.unreconstructible == (POINT_REPLAY_REFIT_CONFIGURATION,)
        estimates, _ = restored.estimator.retarget(
            restored.data, restored.nuisance, estimands=("ate",)
        )
        assert [estimate.inference for estimate in estimates.values()] == [STATUS]
        curve = restored.diagnostics.truncation_curve(bounds=[0.05])
        assert len(curve) > 0


class TestTheSupportedArtifactsKeepTheirIntervals:
    """The controls: no split that read the data, or a status that comes first."""

    def test_an_unstratified_cross_fitted_result(self) -> None:
        result = fit_discrete("tmle", "none")
        assert_keeps_inference(restore(result, "pickle"))

    def test_an_in_sample_result_that_carries_the_policy(self, in_sample: Any) -> None:
        assert in_sample.estimator.stratify_folds == "treatment"
        assert_keeps_inference(restore(in_sample, "pickle"))

    def test_a_cross_fitted_continuous_dose_result(self) -> None:
        restored = keeps_the_dose_interval()
        assert restored.estimator.stratify_folds == "treatment"
        assert restored.estimator.cross_fit
        assert_keeps_inference(restored)

    def test_a_collaborative_result_keeps_its_own_status(self) -> None:
        restored = ctmle_stratified()
        assert restored.estimator.stratify_folds == "treatment"
        assert restored.inference_status == "working_mechanism_plugin"


def _keyed_on_treatment(self: TMLE, data: Any) -> str:
    """The mutant that keys on ``"treatment"`` alone and forgets the crossed policy."""
    keyed = self.cross_fit and self.stratify_folds == "treatment"
    return STATUS if keyed else "influence_curve"


def _without_cross_fitting(self: TMLE, data: Any) -> str:
    """The mutant that drops the cross-fitting condition and reads the policy alone."""
    return STATUS if self.stratify_folds != "none" else "influence_curve"


class TestTheMutationsFailTheWitness:
    """Each mutation of the point rule fails the witness or the control it names."""

    def test_skipping_the_decision_fails_the_witness(
        self, stratified: dict[tuple[str, str], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_saved_fold_policy_status", lambda self, data: "influence_curve")
        with pytest.raises(AssertionError):
            assert_restamped(stratified["tmle", "treatment"], STATUS, "pickle")

    def test_keying_on_treatment_alone_fails_the_crossed_witness(
        self, stratified: dict[tuple[str, str], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_saved_fold_policy_status", _keyed_on_treatment)
        # The mutant still answers the policy it keys on, so it is not a no-op.
        assert_restamped(stratified["tmle", "treatment"], STATUS, "pickle")
        with pytest.raises(AssertionError):
            assert_restamped(stratified["tmle", "treatment+outcome"], STATUS, "pickle")

    def test_dropping_the_cross_fitting_condition_fails_the_in_sample_control(
        self, in_sample: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TMLE, "_saved_fold_policy_status", _without_cross_fitting)
        with pytest.raises(AssertionError):
            assert_keeps_inference(restore(in_sample, "pickle"))


# ------------------------------------------------------------------------ longitudinal

N = 400

#: The node declaration of :func:`~cleverly.datasets.make_longitudinal`.
NODES: dict[str, Any] = {
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def fit_longitudinal(estimator: type[LTMLE], *, n_folds: int = 5, **roles: Any) -> Any:
    """The RM29 probe fit of ``make_longitudinal(400, 0)``, with bands, by ``estimator``."""
    frame, _ = make_longitudinal(n=N, seed=0)
    if "id" in roles:
        frame = frame.assign(cluster=np.arange(N) * 40 // N)
    return estimator(
        {"always": 1, "never": 0},
        reference="never",
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=n_folds,
        random_state=0,
        simultaneous=True,
    ).fit(frame, outcome="Y", **NODES, **roles)


@pytest.fixture(scope="module")
def first_node() -> Any:
    """A five-fold fit whose outer split is stratified on the first treatment node.

    :class:`~tests.studies.ltmle_crossfit_properties.FirstNodeStratifiedLTMLE` draws the
    split that releases 0.1.0 and 0.1.1 drew. The live fit keeps its interval, because the
    rule reads a restored result only.
    """
    result = fit_longitudinal(FirstNodeStratifiedLTMLE)
    assert result.folds.origin is None
    assert result.inference_status == "influence_curve"
    return result


class TestASavedFirstNodeStratifiedLongitudinalResult:
    """RM31 on the longitudinal path: a restored split with no origin withholds inference."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_the_restored_artifact_takes_the_status(self, first_node: Any, route: str) -> None:
        restored = assert_restamped(first_node, STATUS, route)
        assert_withholds(restored, STATUS)
        assert_assessment_note(restored, STATUS)
        assert NO_SIMULTANEOUS_BANDS in restored.summary()
        # The replay reads the status that the restored fit carries, so it matches the fit.
        replay = restored.diagnostics.truncation_curve(bounds=[0.05])
        assert len(replay) == len(restored.estimates)

    def test_skipping_the_decision_fails_the_witness(
        self, first_node: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            longitudinal_estimator, "_saved_split_status", lambda folds: "influence_curve"
        )
        with pytest.raises(AssertionError):
            assert_restamped(first_node, STATUS, "pickle")

    def test_dropping_the_fold_count_condition_fails_the_in_sample_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = fit_longitudinal(LTMLE, n_folds=1)
        assert result.folds.origin is None
        monkeypatch.setattr(
            longitudinal_estimator,
            "_saved_split_status",
            lambda folds: STATUS if folds.origin is None else "influence_curve",
        )
        with pytest.raises(AssertionError):
            assert_keeps_inference(restore(result, "pickle"))

    def test_a_replay_that_drops_the_restored_status_is_refused(
        self, first_node: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The replay control: without the restored status, the replay and the fit differ."""
        restored = restore(legacy_copy(first_node), "pickle")
        refit = longitudinal_estimator._refit_bound

        def unstamped(*args: Any, **kwargs: Any) -> Any:
            replay = refit(*args, **kwargs)
            estimates = {
                name: replace(estimate, inference="influence_curve")
                for name, estimate in replay.estimates.items()
            }
            return replace(replay, estimates=estimates)

        monkeypatch.setattr(longitudinal_estimator, "_refit_bound", unstamped)
        with pytest.raises(CapabilityError, match="longitudinal_replay_fitted_bound_mismatch"):
            restored.diagnostics.truncation_curve(bounds=[0.05])

    def test_the_supported_artifacts_keep_their_intervals(self) -> None:
        current = fit_longitudinal(LTMLE)
        # The nonzero witness: this version records how it drew the split.
        assert current.folds.origin is not None
        assert_keeps_inference(restore(current, "pickle"))
        in_sample = fit_longitudinal(LTMLE, n_folds=1)
        assert_keeps_inference(restore(in_sample, "pickle"))

    @pytest.mark.parametrize("route", ROUTES)
    def test_a_clustered_result_takes_the_grouped_status_first(
        self, monkeypatch: pytest.MonkeyPatch, route: str
    ) -> None:
        """A saved grouped split takes the RM29 status, which comes first in the table."""
        monkeypatch.setattr(LTMLE, "_refuse_cross_fitted_design", lambda self, data: None)
        result = fit_longitudinal(FirstNodeStratifiedLTMLE, id="cluster")
        assert result.folds.origin is None
        restored = assert_restamped(result, SAVED_GROUPED, route)
        assert_withholds(restored, SAVED_GROUPED)

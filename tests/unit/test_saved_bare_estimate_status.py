"""A saved estimate that records no inference status loads under the unrecorded status.

Roadmap row RM34. Releases 0.1.0 and 0.1.1 wrote no ``inference`` field on a
``ParameterEstimate``. The class default reads ``"influence_curve"``, so an estimate that
a caller saved apart from its result answered ``ci``. Its result re-stamps it, and an
estimate alone holds no estimator, data or folds to re-stamp from. Such an estimate now
loads under ``"unrecorded_status_plugin"``: its point estimate and plug-in diagnostic
stand, and ``ci``, ``pvalue`` and ``std_error`` refuse.

Pickle builds each estimate before the result that holds it, so a whole result without
the key sees the unrecorded status on every estimate. The result re-stamps each one from
its own configuration, in both directions. The controls load in-sample ``TMLE``, ``LTMLE``
and variable-importance results in the released shape, and each keeps its interval, its
bands, its saved answers and its adjusted p-values. Each mutation must fail the witness or
the control it names.
"""

from __future__ import annotations

import importlib
import io
import pickle
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from cleverly import variable_importance
from cleverly._inference_status import UNRECORDED_STATUS, supplies_inference
from cleverly.datasets import make_longitudinal
from cleverly.estimators import TMLE, CVTargeting
from cleverly.estimators import base as base_module
from cleverly.exceptions import CapabilityError, inference_refusal
from cleverly.inference.influence import ParameterEstimate, carries_status
from cleverly.longitudinal import LTMLE, LongitudinalResult
from cleverly.variable_importance import VariableImportanceEntry, VariableImportanceResult
from tests import discrete_law
from tests.conftest import linear_in_sample
from tests.pickles import legacy_without
from tests.unit._inference_status_support import (
    LONGITUDINAL_NODES,
    ROUTES,
    assert_refused_by,
    assert_restamped,
    legacy_copy,
    longitudinal_estimator,
    longitudinal_learners,
    restore,
)

pytestmark = pytest.mark.xdist_group("saved_bare_estimate_status")

#: The module, which the package shadows with the function of the same name.
variable_importance_module = importlib.import_module("cleverly.variable_importance")

#: The two ways a bare object is saved: a pickle, and joblib, whose unpickler is the
#: pure-Python one.
BARE_ROUTES = ("pickle", "joblib")

#: What :func:`legacy_copy` writes in place of the bands and the saved answers.
SAVED_BANDS = "bands built before the status"
SAVED_ANSWERS = {"sensitivity.evalue": "an answer read off .ci"}


# ----------------------------------------------------------------------------- fits


@pytest.fixture(scope="module")
def in_sample() -> Any:
    """An in-sample ``TMLE`` fit of the discrete law, which supplies inference."""
    return TMLE(**linear_in_sample()).fit(discrete_law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def fold_evaluated() -> Any:
    """A cross-fitted ``cv_evaluation=True`` fit, which adds both fold-level reports."""
    estimator = TMLE(
        **linear_in_sample(cross_fit=True, n_folds=2, cv_evaluation=True, estimands=("ate",))
    )
    return estimator.fit(discrete_law.frame(), outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def longitudinal() -> Any:
    """An in-sample ``LTMLE`` fit of two regimens, which supplies inference."""
    frame, _ = make_longitudinal(n=400, seed=0)
    estimator = LTMLE({"always": 1, "never": 0}, reference="never", **longitudinal_learners())
    return estimator.fit(frame, outcome="Y", **LONGITUDINAL_NODES)


@pytest.fixture(scope="module")
def importance() -> Any:
    """An in-sample run over three candidates, so each adjustment moves a p-value."""
    rng = np.random.default_rng(34)
    n = 300
    exposures = {name: rng.binomial(1, 0.5, size=n) for name in ("X1", "X2", "X3")}
    w = rng.normal(size=n)
    y = 0.35 * exposures["X1"] + 0.2 * exposures["X2"] + 0.1 * exposures["X3"] + 0.2 * w
    frame = pd.DataFrame({"Y": y + rng.normal(scale=1.0, size=n), "W": w, **exposures})
    return variable_importance(
        frame,
        outcome="Y",
        candidates=("X1", "X2", "X3"),
        covariates=("W",),
        estimator=TMLE(**linear_in_sample()),
    )


# -------------------------------------------------------------------------- helpers


def _estimates_of(record: Any) -> list[ParameterEstimate]:
    """Every estimate that a bare estimate, entry or fold-level report holds."""
    if isinstance(record, ParameterEstimate):
        return [record]
    if isinstance(record, VariableImportanceEntry):
        return [record.estimate]
    if isinstance(record, CVTargeting):
        return [*record.pooled.values(), *record.canonical.values()]
    if isinstance(record, VariableImportanceResult):
        return [
            *(entry.estimate for entry in record.entries),
            *(estimate for fit in record.fits.values() for estimate in fit.estimates.values()),
        ]
    raise TypeError(f"no estimates known for {type(record).__name__}")


def _load(record: Any, route: str) -> Any:
    """``record`` saved and loaded by ``route``, one of :data:`BARE_ROUTES`."""
    if route == "pickle":
        return pickle.loads(pickle.dumps(record))
    buffer = io.BytesIO()
    joblib.dump(record, buffer)
    buffer.seek(0)
    return joblib.load(buffer)


def saved_without_status(record: Any, route: str) -> Any:
    """``record`` as release 0.1.1 wrote it, with no ``inference`` key, loaded by ``route``."""
    bare = pickle.loads(pickle.dumps(record))
    for estimate in _estimates_of(bare):
        # An entry and its fit hold one estimate object, so the key can already be gone.
        estimate.__dict__.pop("inference", None)
    # The nonzero witness: the copy is the released shape, and it reads the class default.
    assert all("inference" not in estimate.__dict__ for estimate in _estimates_of(bare))
    assert all(estimate.inference == "influence_curve" for estimate in _estimates_of(bare))
    return _load(bare, route)


def assert_unrecorded(estimate: ParameterEstimate, live: ParameterEstimate) -> None:
    """The estimate withholds its inference by the RM34 reason, and keeps every number."""
    assert estimate.inference == UNRECORDED_STATUS
    assert not estimate.supplies_inference
    for accessor in ("ci", "pvalue", "std_error"):
        with pytest.raises(CapabilityError) as raised:
            getattr(estimate, accessor)
        assert str(raised.value) == inference_refusal(f".{accessor}", UNRECORDED_STATUS)
    assert estimate.psi == live.psi
    assert estimate.plugin_std_error == live.plugin_std_error
    assert estimate.plugin_interval == live.plugin_interval
    assert estimate.to_dict()["inference"] == UNRECORDED_STATUS


def check_estimate_witness(result: Any, route: str) -> None:
    """The ``ate`` estimate saved alone loads under the unrecorded status."""
    live = result["ate"]
    # The nonzero witness: the live estimate answers every accessor that must refuse.
    assert live.ci[0] < live.ci[1]
    assert_unrecorded(saved_without_status(live, route), live)


def check_entry_witness(importance: Any, route: str) -> None:
    """An entry saved alone reads no adjusted p-value, where the live entry reads one."""
    live = importance.entries[0]
    assert isinstance(live.adjusted_pvalue, float)
    entry = saved_without_status(live, route)
    assert_unrecorded(entry.estimate, live.estimate)
    assert entry.adjusted_pvalue is None


def check_keeps_everything(restored: Any, live: Any) -> None:
    """A result saved without the key keeps its status, intervals, bands and answers.

    The saved bands are a marker string, which ``to_frame`` cannot read, so the check
    reads each estimate rather than the frame.
    """
    assert restored.inference_status == "influence_curve"
    for name, estimate in restored.estimates.items():
        assert estimate.inference == "influence_curve"
        assert estimate.ci == live.estimates[name].ci
        assert estimate.pvalue == live.estimates[name].pvalue
        assert estimate.std_error == live.estimates[name].std_error
    assert restored.simultaneous == SAVED_BANDS
    assert restored.assessment_cache == SAVED_ANSWERS


def check_in_sample_control(result: Any, route: str) -> None:
    """The released shape of an in-sample result loads with everything it saved."""
    check_keeps_everything(restore(legacy_copy(result, recorded=False), route), result)


def check_importance_control(importance: Any, route: str) -> None:
    """The released shape of an in-sample run keeps each adjusted p-value exactly."""
    # The nonzero witness: the adjustment moves at least one p-value.
    assert any(entry.adjusted_pvalue != entry.estimate.pvalue for entry in importance.entries)
    restored = saved_without_status(importance, route)
    assert len(restored.entries) == len(importance.entries)
    for entry, live in zip(restored.entries, importance.entries, strict=True):
        assert entry.estimate.inference == "influence_curve"
        assert entry.adjusted_pvalue == live.adjusted_pvalue
    assert "p_value_adjusted" in restored.to_frame().columns


def check_recorded_status_control(result: Any) -> None:
    """A current result keeps a recorded status that its hook does not give."""
    current = pickle.loads(pickle.dumps(result))
    current.__dict__["estimates"] = {
        name: replace(estimate, inference="stratified_fold_plugin")
        for name, estimate in result.estimates.items()
    }
    restored = restore(current, "pickle")
    assert restored.inference_status == "stratified_fold_plugin"


# ------------------------------------------------------------------------ witnesses


class TestTheWitness:
    """An estimate, an entry, or a fold-level report saved alone withholds its inference."""

    def test_the_shared_helper_loads_the_estimate_under_the_status(self, in_sample: Any) -> None:
        """``legacy_without`` reaches ``__setstate__`` the way an old pickle does."""
        live = in_sample["ate"]
        assert_unrecorded(legacy_without(live, "inference"), live)

    @pytest.mark.parametrize("route", BARE_ROUTES)
    def test_an_estimate_saved_alone(self, in_sample: Any, route: str) -> None:
        check_estimate_witness(in_sample, route)

    @pytest.mark.parametrize("route", BARE_ROUTES)
    def test_an_entry_saved_alone(self, importance: Any, route: str) -> None:
        check_entry_witness(importance, route)

    @pytest.mark.parametrize("route", BARE_ROUTES)
    def test_a_fold_level_report_saved_alone(self, fold_evaluated: Any, route: str) -> None:
        live = fold_evaluated.cv_targeting
        # The nonzero witness: the live report answers ``std_error``.
        assert live.std_error == live.plugin_std_error
        report = saved_without_status(live, route)
        assert report.inference == UNRECORDED_STATUS
        with pytest.raises(CapabilityError) as raised:
            _ = report.std_error
        assert_refused_by(UNRECORDED_STATUS, raised)
        assert report.plugin_std_error == live.plugin_std_error


# ------------------------------------------------------------------------- controls


class TestResultsWithoutTheKey:
    """A whole result in the released shape gives each estimate its status again."""

    @pytest.mark.parametrize("route", ROUTES)
    def test_an_in_sample_tmle_result(self, in_sample: Any, route: str) -> None:
        check_in_sample_control(in_sample, route)

    @pytest.mark.parametrize("route", ROUTES)
    def test_an_in_sample_ltmle_result(self, longitudinal: Any, route: str) -> None:
        check_in_sample_control(longitudinal, route)

    @pytest.mark.parametrize("route", ROUTES)
    def test_both_fold_level_reports(self, fold_evaluated: Any, route: str) -> None:
        restored = restore(legacy_copy(fold_evaluated, recorded=False), route)
        check_keeps_everything(restored, fold_evaluated)
        detail = restored.cv_targeting
        assert detail.inference == "influence_curve"
        for report in (detail.pooled, detail.canonical):
            assert {estimate.inference for estimate in report.values()} == {"influence_curve"}
        assert detail.std_error == fold_evaluated.cv_targeting.std_error

    @pytest.mark.parametrize("route", BARE_ROUTES)
    def test_an_in_sample_variable_importance_result(self, importance: Any, route: str) -> None:
        check_importance_control(importance, route)


class TestCurrentObjectsLoadAsSaved:
    """An object that records its status loads with that status."""

    def test_a_current_estimate_keeps_its_interval(self, in_sample: Any) -> None:
        live = in_sample["ate"]
        assert "inference" in live.__dict__
        restored = pickle.loads(pickle.dumps(live))
        assert restored.inference == "influence_curve"
        assert restored.ci == live.ci

    def test_a_current_non_inferential_estimate_keeps_its_status(self, in_sample: Any) -> None:
        marked = replace(in_sample["ate"], inference="stratified_fold_plugin")
        assert pickle.loads(pickle.dumps(marked)).inference == "stratified_fold_plugin"

    def test_a_current_result_keeps_a_recorded_status(self, in_sample: Any) -> None:
        check_recorded_status_control(in_sample)

    def test_a_result_forced_to_the_status_drops_its_bands(
        self, in_sample: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both saved shapes load under a hook that gives the unrecorded status."""
        monkeypatch.setattr(TMLE, "_inference_status", lambda self, data: UNRECORDED_STATUS)
        assert_restamped(in_sample, UNRECORDED_STATUS, "pickle")


# ------------------------------------------------------------------------ mutations


def _old_stamp(estimates: Any, status: str) -> dict[str, ParameterEstimate]:
    """The stamp before RM34, which returns every estimate as given under inference."""
    if supplies_inference(status):
        return dict(estimates)
    return {name: replace(estimate, inference=status) for name, estimate in estimates.items()}


def _stamp_everything(estimates: Any, status: str) -> dict[str, ParameterEstimate]:
    """A stamp that returns every estimate to ``status``, a recorded status included."""
    return {name: replace(estimate, inference=status) for name, estimate in estimates.items()}


def _tmle_status(result: Any) -> str:
    return result.estimator._inference_status(result.__dict__["data"])


def _longitudinal_status(result: Any) -> str:
    state = result.__dict__
    return longitudinal_estimator.precedent_status(
        [
            longitudinal_estimator._inference_status(
                state["data"], state["folds"], state["config"].regimens, state.get("msm")
            ),
            longitudinal_estimator._saved_split_status(state["folds"]),
        ]
    )


def _old_return(
    original: Callable[[Any], None], status: Callable[[Any], str]
) -> Callable[[Any], None]:
    """The re-stamp before RM34, which returns early on every inferential status."""

    def restamp(self: Any) -> None:
        if supplies_inference(status(self)):
            return
        original(self)

    return restamp


def _drops_on_every_status(original: Callable[[Any], None]) -> Callable[[Any], None]:
    """A re-stamp that drops the bands and the answers after every stamp."""

    def restamp(self: Any) -> None:
        estimates = self.__dict__.get("estimates")
        original(self)
        if self.__dict__.get("estimates") is not estimates:
            self.__dict__["simultaneous"] = None
            self.__dict__["assessment_cache"] = {}

    return restamp


@pytest.fixture
def old_stamp(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``stamp_inference`` without its RM34 branch, where both results read it."""
    monkeypatch.setattr(base_module, "stamp_inference", _old_stamp)
    monkeypatch.setattr(longitudinal_estimator, "stamp_inference", _old_stamp)
    yield


class TestTheMutationsFail:
    """Each mutation fails the witness or the control it names."""

    @pytest.mark.parametrize("route", BARE_ROUTES)
    def test_a_missing_estimate_setstate_fails_the_witness(
        self, in_sample: Any, route: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delattr(ParameterEstimate, "__setstate__")
        with pytest.raises(AssertionError):
            check_estimate_witness(in_sample, route)

    @pytest.mark.parametrize("route", BARE_ROUTES)
    def test_a_missing_entry_setstate_fails_the_entry_witness(
        self, importance: Any, route: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delattr(VariableImportanceEntry, "__setstate__")
        with pytest.raises(AssertionError):
            check_entry_witness(importance, route)

    def test_the_old_tmle_return_fails_the_tmle_control(
        self, in_sample: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = base_module.TMLEResult._restamp_inference_status
        monkeypatch.setattr(
            base_module.TMLEResult, "_restamp_inference_status", _old_return(original, _tmle_status)
        )
        with pytest.raises(AssertionError):
            check_in_sample_control(in_sample, "pickle")

    def test_the_old_ltmle_return_fails_the_ltmle_control(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = LongitudinalResult._restamp_inference_status
        monkeypatch.setattr(
            LongitudinalResult,
            "_restamp_inference_status",
            _old_return(original, _longitudinal_status),
        )
        with pytest.raises(AssertionError):
            check_in_sample_control(longitudinal, "pickle")

    @pytest.mark.usefixtures("old_stamp")
    def test_the_old_stamp_fails_the_tmle_control(self, in_sample: Any) -> None:
        with pytest.raises(AssertionError):
            check_in_sample_control(in_sample, "pickle")

    @pytest.mark.usefixtures("old_stamp")
    def test_the_old_stamp_fails_the_ltmle_control(self, longitudinal: Any) -> None:
        with pytest.raises(AssertionError):
            check_in_sample_control(longitudinal, "pickle")

    def test_a_stamp_of_every_estimate_fails_the_recorded_status_control(
        self, in_sample: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(base_module, "stamp_inference", _stamp_everything)
        with pytest.raises(AssertionError):
            check_recorded_status_control(in_sample)

    def test_dropped_bands_fail_the_tmle_control(
        self, in_sample: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = base_module.TMLEResult._restamp_inference_status
        monkeypatch.setattr(
            base_module.TMLEResult, "_restamp_inference_status", _drops_on_every_status(original)
        )
        with pytest.raises(AssertionError):
            check_in_sample_control(in_sample, "pickle")

    def test_dropped_bands_fail_the_ltmle_control(
        self, longitudinal: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = LongitudinalResult._restamp_inference_status
        monkeypatch.setattr(
            LongitudinalResult, "_restamp_inference_status", _drops_on_every_status(original)
        )
        with pytest.raises(AssertionError):
            check_in_sample_control(longitudinal, "pickle")

    def test_no_second_adjustment_fails_the_variable_importance_control(
        self, importance: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(variable_importance_module, "_readjusted", lambda entries: entries)
        with pytest.raises(AssertionError):
            check_importance_control(importance, "pickle")

    def test_a_settled_unrecorded_status_keeps_the_bands_of_a_forced_fit(
        self, in_sample: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The early return without its unrecorded clause fails the forced-status control."""

        def settled(reports: Any, status: Any) -> bool:
            return all(
                estimate.inference == status for report in reports for estimate in report.values()
            )

        assert carries_status([in_sample.estimates], "influence_curve")
        monkeypatch.setattr(base_module, "carries_status", settled)
        monkeypatch.setattr(TMLE, "_inference_status", lambda self, data: UNRECORDED_STATUS)
        with pytest.raises(AssertionError):
            assert_restamped(in_sample, UNRECORDED_STATUS, "pickle")

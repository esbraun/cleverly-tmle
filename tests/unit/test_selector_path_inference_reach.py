"""How far the selector-path inference refusal reaches past the estimate accessors.

RM12 refuses ``ci``, ``pvalue`` and ``std_error`` on the greedy, ordered and discrete
collaborative paths.  Two surfaces reached those accessors from the inside, so each one
ran the expensive part of its work and then raised a message naming an accessor the caller
had never written.

*   :func:`~cleverly.variable_importance` fitted one complete model per candidate and then
    read ``pvalue`` off every one of them, inside the Benjamini--Hochberg adjustment.  It
    now refuses before the first fit.  The nonzero control is a learner that raises when
    it is fitted: the refusal has to arrive without that learner being touched.
*   :func:`~cleverly.sensitivity.missingness_tilt` advertised ``available=True`` and then
    read ``std_error``.  It now reports the retained diagnostic under the names
    :func:`~cleverly.sensitivity.positivity.truncation_curve` already uses, and its
    point-estimate curve is unchanged.  The control is the same tilt on an ordinary TMLE
    fit of the same frame, whose column names must not move.

The round-trip case is here rather than beside the accessors because ``inference`` is a
dataclass field with a class-level default.  A result restored from a pickle written
before that field existed arrives without it, and its estimates read the default,
``"influence_curve"``.  ``TMLEResult.__setstate__`` re-stamps them from the estimator the
artifact carries, and :class:`TestALegacySelectorArtifactIsReStamped` pins that on a
simulated pre-field artifact.

The later classes follow the refusal into every other surface that publishes a spread of
the reported curve: the bootstrap limits, the repeat-spread report, the refutation frame,
the coverage study, the argument-aware capability rows, and the combined report.  Each
reads its names through :func:`~cleverly.inference.influence.spread_name`, so a test here
reads them the same way, and one literal table in ``tests/unit/test_inference.py`` pins
the names themselves.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from cleverly import variable_importance
from cleverly._inference_status import NON_INFERENTIAL
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_binary_outcome, make_instrument, make_missing_outcome
from cleverly.estimators import CTMLE, TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import (
    WORKING_MECHANISM_ASSESSMENT_NOTE,
    WORKING_MECHANISM_NOT_INFERENTIAL,
    CapabilityError,
    inference_refusal,
)
from cleverly.inference.influence import spread_name
from cleverly.sensitivity import missingness_tilt, tipping_gamma
from cleverly.validation import CoverageStudy, refute
from cleverly.validation.score import score_check
from tests.conftest import SELECTOR_CONFIGS, linear_ctmle, linear_in_sample

#: The status every selector path stamps, and the three spread columns an ordinary fit's
#: tilt carries and a selector-path fit renames.  Read through ``spread_name`` rather than
#: spelt again, because the two tests below are one claim read from opposite sides.
DIAGNOSTIC = "working_mechanism_plugin"
INFERENTIAL_COLUMNS = frozenset({"std_err", "ci_lower", "ci_upper"})
DIAGNOSTIC_COLUMNS = frozenset(spread_name(name, DIAGNOSTIC) for name in INFERENTIAL_COLUMNS)

#: What the combined report writes when an operation it ran raised ``CapabilityError``.
DECLINED = "the operation declined this request"


class Unfittable(BaseEstimator):
    """A learner that fails when it is fitted, which is how "no fit ran" is measured.

    A refusal placed after the loop would fit this learner and raise ``AssertionError``
    here instead of ``CapabilityError`` at the entry point.  The test then fails on the
    exception type rather than on a count nobody maintains.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Unfittable:
        raise AssertionError("variable_importance fitted a model before it refused")

    def predict(self, X: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("variable_importance predicted before it refused")


@pytest.fixture(scope="module")
def missing_frame() -> Any:
    return make_missing_outcome(n=400, seed=7)[0]


@pytest.fixture(scope="module")
def selector_fit(missing_frame: Any) -> Any:
    """A greedy C-TMLE that models response, which is what reaches the tilt."""
    return (
        linear_ctmle("greedy", selection_folds=3, estimands=("ate",))
        .fit(missing_frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


@pytest.fixture(scope="module")
def ordinary_fit(missing_frame: Any) -> Any:
    """The same frame under ordinary TMLE, which must keep the inferential columns."""
    return (
        TMLE(**linear_in_sample(estimands=("ate",)))
        .fit(missing_frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


class TestVariableImportanceRefusesBeforeItFits:
    """The multiplicity adjustment has no diagnostic form, so the entry point refuses."""

    @staticmethod
    def _call(estimator: Any) -> Any:
        frame, _ = make_instrument(n=300, seed=3)
        return variable_importance(
            frame,
            outcome="Y",
            candidates=["A"],
            covariates=["W1", "W2"],
            estimator=estimator,
        )

    @pytest.mark.parametrize("strategy", sorted(SELECTOR_CONFIGS))
    def test_a_selector_path_is_refused_by_the_name_the_caller_wrote(self, strategy: str) -> None:
        with pytest.raises(CapabilityError) as raised:
            self._call(linear_ctmle(strategy, estimands=("ate",), **SELECTOR_CONFIGS[strategy]))
        message = str(raised.value)
        assert message.startswith("variable_importance() is not defined here.")
        assert WORKING_MECHANISM_NOT_INFERENTIAL in message
        # The old message named an accessor no caller had written.
        assert ".pvalue" not in message

    def test_the_refusal_arrives_before_the_first_learner_is_fitted(self) -> None:
        estimator = linear_ctmle(
            "greedy",
            selection_folds=3,
            estimands=("ate",),
            outcome_learner=Unfittable(),
            treatment_learner=Unfittable(),
        )
        with pytest.raises(CapabilityError):
            self._call(estimator)

    def test_the_outcome_adaptive_path_is_refused_by_its_own_reason(self) -> None:
        """``oat`` refuses at the entry point too, with the reason of its own status.

        It was the control against a refusal broadened to every collaborative fit until
        RM20 gave it a status. The ordinary estimator below is that control now.
        """
        with pytest.raises(CapabilityError) as raised:
            self._call(linear_ctmle("oat", estimands=("ate",)))
        message = str(raised.value)
        assert message.startswith("variable_importance() is not defined here.")
        assert NON_INFERENTIAL["generated_design_plugin"].reason in message
        assert WORKING_MECHANISM_NOT_INFERENTIAL not in message

    def test_an_ordinary_estimator_is_untouched(self) -> None:
        """The control against a refusal broadened to every estimator."""
        result = self._call(TMLE(**linear_in_sample(estimands=("ate",))))
        frame = result.to_frame()
        assert {"std_err", "ci_lower", "ci_upper", "p_value"} <= set(frame.columns)
        assert np.isfinite(result[0].estimate.pvalue)
        assert np.isfinite(result[0].adjusted_pvalue)


class TestTheMissingnessTiltAgreesWithItsCapabilityRow:
    """The row says the tilt runs, so the tilt has to run and say what it reports."""

    @pytest.mark.parametrize("operation", ["missingness", "tipping_gamma"])
    def test_the_row_is_available_and_the_operation_answers(
        self, selector_fit: Any, operation: str
    ) -> None:
        capability = selector_fit.sensitivity.capability(operation)
        assert capability.available is True
        assert capability.reason is None
        # The defect was exactly this: an available row whose operation raised. A search
        # that detects no crossing returns ``None``, which is one of its two answers.
        getattr(selector_fit.sensitivity, operation)()

    def test_the_tilt_reports_the_diagnostic_columns(self, selector_fit: Any) -> None:
        curve = missingness_tilt(selector_fit, [-1.0, 0.0, 1.0])
        columns = set(curve.columns)
        assert columns >= DIAGNOSTIC_COLUMNS
        assert not (INFERENTIAL_COLUMNS & columns)

    def test_the_diagnostic_column_holds_the_retained_plug_in_error(
        self, selector_fit: Any
    ) -> None:
        curve = missingness_tilt(selector_fit, [0.0])
        assert float(curve["plugin_std_err"][0]) == selector_fit["ate"].plugin_std_error

    def test_the_curve_still_passes_through_the_reported_estimate(self, selector_fit: Any) -> None:
        """The point-estimate sweep is what the swap keeps, so it is checked here."""
        curve = missingness_tilt(selector_fit, [0.0])
        assert float(curve["psi"][0]) == pytest.approx(selector_fit["ate"].psi, abs=1e-12)

    def test_an_ordinary_fit_keeps_the_inferential_columns(self, ordinary_fit: Any) -> None:
        """The control: the swap must not reach a fit that has an interval."""
        curve = missingness_tilt(ordinary_fit, [-1.0, 0.0, 1.0])
        columns = set(curve.columns)
        assert columns >= INFERENTIAL_COLUMNS
        assert not (DIAGNOSTIC_COLUMNS & columns)
        assert float(curve["std_err"][1]) == ordinary_fit["ate"].std_error


class TestTippingGammaSearchesThePointEstimateOnly:
    """One float carries no column name, so only the point-estimate search survives."""

    def test_the_point_search_finds_a_declared_crossing(self, selector_fit: Any) -> None:
        """A null placed between two swept estimates, so the crossing must exist."""
        curve = missingness_tilt(selector_fit, [0.0, 2.0])
        low, high = sorted(float(value) for value in curve["psi"])
        assert low < high, "the tilt moved the estimate by nothing, so no null separates it"
        null = 0.5 * (low + high)

        found = tipping_gamma(selector_fit, "ate", null_hypothesis=null, search=(0.0, 2.0))
        assert found is not None
        assert 0.0 < float(found) <= 2.0

    def test_the_interval_search_is_refused_by_name(self, selector_fit: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            tipping_gamma(selector_fit, "ate", use_ci=True)
        message = str(raised.value)
        assert message.startswith("tipping_gamma(use_ci=True) is not defined here.")
        assert WORKING_MECHANISM_NOT_INFERENTIAL in message
        assert ".std_error" not in message

    def test_an_ordinary_fit_still_searches_its_interval(self, ordinary_fit: Any) -> None:
        found = tipping_gamma(ordinary_fit, "ate", use_ci=True)
        assert found is None or np.isfinite(float(found))


class TestARestoredSelectorFitStillRefuses:
    """``inference`` is a field with a class-level default, so its survival is pinned.

    A restored result that lost the field would read ``"influence_curve"``, publish a
    confidence interval for the selected working mechanism, and look exactly like a fit
    the package supports.
    """

    @pytest.fixture
    def restored(self, selector_fit: Any) -> Any:
        """A fresh round trip per test, because the tilt below writes the result's cache."""
        return pickle.loads(pickle.dumps(selector_fit))

    def test_the_restored_result_reports_the_diagnostic_status(self, restored: Any) -> None:
        assert restored["ate"].inference == "working_mechanism_plugin"

    def test_the_restored_result_still_refuses_its_interval(self, restored: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            _ = restored["ate"].ci
        assert WORKING_MECHANISM_NOT_INFERENTIAL in str(raised.value)

    def test_the_retained_diagnostic_survives_the_round_trip_bit_for_bit(
        self, restored: Any, selector_fit: Any
    ) -> None:
        assert restored["ate"].plugin_std_error == selector_fit["ate"].plugin_std_error

    def test_the_restored_tilt_reports_the_diagnostic_columns(self, restored: Any) -> None:
        """The two fixes meet here: a restored fit reaches the renamed columns too."""
        columns = set(missingness_tilt(restored, [0.0]).columns)
        assert columns >= DIAGNOSTIC_COLUMNS
        assert not (INFERENTIAL_COLUMNS & columns)


def _legacy(result: Any) -> Any:
    """A copy of ``result`` shaped as an artifact written before ``inference`` existed.

    Each estimate loses the field from its instance state, so it reads the class-level
    default. The result also carries the two things such an artifact could hold that the
    re-stamp has to drop: a band object and a saved assessment answer.
    """
    legacy = pickle.loads(pickle.dumps(result))
    for estimate in legacy.estimates.values():
        estimate.__dict__.pop("inference")
        assert estimate.inference == "influence_curve"
    legacy.__dict__["simultaneous"] = "bands built before the refusal"
    legacy.__dict__["assessment_cache"] = {"sensitivity.evalue": "an answer read off .ci"}
    return legacy


class TestALegacySelectorArtifactIsReStamped:
    """A selector fit saved before the field existed must not load as inferential.

    Without the re-stamp every estimate of such an artifact reads ``"influence_curve"``,
    so ``.ci`` answers and the fit publishes the interval this version refuses.
    """

    @pytest.fixture(scope="class", params=["serialize", "pickle"])
    def restored(self, request: Any, selector_fit: Any) -> Any:
        legacy = _legacy(selector_fit)
        if request.param == "serialize":
            return loads(dumps(legacy))
        return pickle.loads(pickle.dumps(legacy))

    def test_the_estimates_carry_the_estimator_status_again(self, restored: Any) -> None:
        assert {estimate.inference for estimate in restored.estimates.values()} == {DIAGNOSTIC}
        assert restored.inference_status == DIAGNOSTIC

    def test_the_interval_is_refused_again(self, restored: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            _ = restored["ate"].ci
        assert WORKING_MECHANISM_NOT_INFERENTIAL in str(raised.value)

    def test_the_diagnostic_is_bit_identical(self, restored: Any, selector_fit: Any) -> None:
        assert restored["ate"].plugin_std_error == selector_fit["ate"].plugin_std_error
        assert restored["ate"].plugin_interval == selector_fit["ate"].plugin_interval
        assert restored["ate"].psi == selector_fit["ate"].psi

    def test_what_was_derived_under_the_old_status_is_dropped(self, restored: Any) -> None:
        assert restored.simultaneous is None
        assert restored.assessment_cache == {}

    def test_an_ordinary_legacy_artifact_loads_as_it_was_saved(self, ordinary_fit: Any) -> None:
        """The control: a fit whose estimator supplies inference is not touched."""
        restored = loads(dumps(_legacy(ordinary_fit)))
        assert restored.inference_status == "influence_curve"
        assert restored.simultaneous == "bands built before the refusal"
        assert restored.assessment_cache == {"sensitivity.evalue": "an answer read off .ci"}
        assert restored["ate"].ci == ordinary_fit["ate"].ci


def _pre_strategy(estimator: Any) -> Any:
    """A copy of ``estimator`` shaped as one pickled before ``strategy`` replaced ``search``.

    Commit d429d90 renamed the attribute. The old one took ``"greedy"``, ``"ordered"`` or
    ``"discrete"``, so every such artifact holds a selector path.
    """
    legacy = pickle.loads(pickle.dumps(estimator))
    legacy.__dict__["search"] = legacy.__dict__.pop("strategy")
    return legacy


class TestAPreStrategyArtifactLoadsAndRefuses:
    """An estimator that stores ``search`` must load, and must re-stamp as a selector.

    ``TMLEResult.__setstate__`` asks the estimator for its status while the result loads.
    A reader of ``strategy`` alone raised ``AttributeError`` there, on an artifact that
    loaded before the re-stamp existed. Mapping ``search`` to ``"oat"`` would load it as
    inferential, so each test checks the refusal as well as the load.
    """

    @pytest.fixture(scope="class", params=["serialize", "pickle"])
    def restored(self, request: Any, selector_fit: Any) -> Any:
        legacy = _legacy(selector_fit)
        legacy.__dict__["estimator"] = _pre_strategy(selector_fit.estimator)
        if request.param == "serialize":
            return loads(dumps(legacy))
        return pickle.loads(pickle.dumps(legacy))

    def test_the_estimator_carries_the_strategy_under_its_current_name(self, restored: Any) -> None:
        assert restored.estimator.strategy == "greedy"
        assert "search" not in restored.estimator.__dict__

    def test_the_estimates_are_re_stamped_as_a_selector_path(self, restored: Any) -> None:
        assert restored.inference_status == DIAGNOSTIC
        assert restored.simultaneous is None
        assert restored.assessment_cache == {}

    def test_the_interval_is_refused(self, restored: Any) -> None:
        with pytest.raises(CapabilityError) as raised:
            _ = restored["ate"].ci
        assert WORKING_MECHANISM_NOT_INFERENTIAL in str(raised.value)

    @pytest.mark.parametrize("strategy", sorted(SELECTOR_CONFIGS))
    def test_variable_importance_refuses_the_restored_estimator(self, strategy: str) -> None:
        estimator = linear_ctmle(strategy, estimands=("ate",), **SELECTOR_CONFIGS[strategy])
        restored = pickle.loads(pickle.dumps(_pre_strategy(estimator)))
        assert restored.strategy == strategy
        with pytest.raises(CapabilityError) as raised:
            TestVariableImportanceRefusesBeforeItFits._call(restored)
        assert str(raised.value).startswith("variable_importance() is not defined here.")


@pytest.fixture(scope="module")
def binary_frame() -> Any:
    return make_binary_outcome(n=400, seed=17)[0]


def _binary_selector(**extra: Any) -> CTMLE:
    return CTMLE(
        strategy="greedy",
        selection_folds=3,
        estimands=("ate",),
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        simultaneous=False,
        random_state=0,
        **extra,
    )


@pytest.fixture(scope="module")
def repeated_selector_fit(binary_frame: Any) -> Any:
    """A greedy fit over two cross-fitting draws, which is what reports a repeat spread."""
    return (
        _binary_selector(repeats=2, n_folds=2)
        .fit(binary_frame, outcome="Y", treatment="A")
        .single()
    )


class TestEverySpreadIsNamedByItsStatus:
    """Each frame and label that publishes the plug-in error names it as a diagnostic.

    The numbers are unchanged. Only the name moves, and it moves through ``spread_name``,
    so the control on each surface is an ordinary fit that keeps the inferential name.
    """

    def test_the_fit_declares_one_status(
        self, repeated_selector_fit: Any, ordinary_fit: Any
    ) -> None:
        assert repeated_selector_fit.inference_status == DIAGNOSTIC
        assert ordinary_fit.inference_status == "influence_curve"
        assert repeated_selector_fit["ate"].supplies_inference is False
        assert ordinary_fit["ate"].supplies_inference is True

    def test_the_bootstrap_limits_are_a_range(self, binary_frame: Any) -> None:
        refused = (
            _binary_selector(n_bootstrap=4, cross_fit=False)
            .fit(binary_frame, outcome="Y", treatment="A")
            .single()
        )
        row = refused["ate"].to_dict()
        low, high = refused["ate"].bootstrap.ci
        assert row[spread_name("bootstrap_ci_lower", DIAGNOSTIC)] == low
        assert row[spread_name("bootstrap_ci_upper", DIAGNOSTIC)] == high
        assert row["bootstrap_std_err"] == refused["ate"].bootstrap.std_error
        assert not {"bootstrap_ci_lower", "bootstrap_ci_upper"} & set(refused.to_frame().columns)
        assert "percentile range" in refused.summary()

    def test_the_repeat_spread_frame_names_the_diagnostic(self, repeated_selector_fit: Any) -> None:
        report = repeated_selector_fit.diagnostics.nuisance_models()
        columns = list(report.repeat_spread_frame().columns)
        assert columns[-2:] == [
            spread_name("reported_standard_error", DIAGNOSTIC),
            spread_name("ratio_to_standard_error", DIAGNOSTIC),
        ]
        assert "reported_standard_error" not in columns
        summary = report.summary()
        assert spread_name("sd/se", DIAGNOSTIC) in summary
        assert "reported se" not in summary

    def test_the_split_noise_share_names_the_diagnostic(self, repeated_selector_fit: Any) -> None:
        summary = repeated_selector_fit.summary()
        label = spread_name("std_err", DIAGNOSTIC)
        assert f"of {label})" in summary
        assert "of std_err)" not in summary

    def test_the_nuisance_note_keeps_its_capitals(self, repeated_selector_fit: Any) -> None:
        """``str.capitalize`` printed "f18". Only the first letter may change."""
        report = repeated_selector_fit.diagnostics.nuisance_models()
        assert report.inference_note == WORKING_MECHANISM_ASSESSMENT_NOTE
        note = WORKING_MECHANISM_ASSESSMENT_NOTE
        assert note[0].upper() + note[1:] + "." in report.summary()

    def test_an_ordinary_report_has_no_note(self, ordinary_fit: Any) -> None:
        assert ordinary_fit.diagnostics.nuisance_models().inference_note is None

    def test_the_refutation_frame_names_the_diagnostic(self, repeated_selector_fit: Any) -> None:
        report = refute(
            repeated_selector_fit, tests=("random_common_cause",), n_replicates=2, random_state=0
        )
        test = report.tests[0]
        assert test.inference == DIAGNOSTIC
        columns = set(test.to_frame().columns)
        assert spread_name("std_error", DIAGNOSTIC) in columns
        assert "std_error" not in columns
        # The detail counts the shift in the same unit the frame names.
        assert f" {spread_name('standard errors', DIAGNOSTIC)}); " in test.detail

    def test_an_ordinary_refutation_detail_counts_standard_errors(self, ordinary_fit: Any) -> None:
        report = refute(
            ordinary_fit, tests=("random_common_cause",), n_replicates=2, random_state=0
        )
        detail = report.tests[0].detail
        assert " standard errors); " in detail
        assert spread_name("standard errors", DIAGNOSTIC) not in detail

    def test_a_failing_score_verdict_names_the_diagnostic(self, repeated_selector_fit: Any) -> None:
        """A zero tolerance fails every nonzero score, which is what reaches the verdict."""
        check = score_check(repeated_selector_fit, tolerance=0.0)
        assert not check.passed
        assert check.inference == DIAGNOSTIC
        unit = spread_name("standard errors", DIAGNOSTIC)
        assert f"The {unit} above" in check.one_line()
        assert f"the {unit} this fit reports" in check.summary()
        assert "influence-curve standard errors" not in check.summary()

    def test_an_ordinary_failing_score_verdict_is_unchanged(self, ordinary_fit: Any) -> None:
        check = score_check(ordinary_fit, tolerance=0.0)
        assert not check.passed
        assert check.inference == "influence_curve"
        assert "  The standard errors above are read off" in check.one_line()
        assert "the influence-curve standard errors this fit reports" in check.summary()

    def test_the_coverage_study_labels_what_it_measured(self) -> None:
        study = CoverageStudy(
            dgp=lambda n, seed: make_binary_outcome(n=n, seed=seed),
            estimator=lambda: _binary_selector(cross_fit=False),
            n=300,
            n_replicates=3,
            seed=1,
            fit_kwargs={"outcome": "Y", "treatment": "A"},
        ).run()
        assert study.n_failed == 0
        assert study["ate"].inference == DIAGNOSTIC
        frame = study.to_frame()
        assert list(frame["inference"]) == [DIAGNOSTIC]
        assert spread_name("mean_std_error", DIAGNOSTIC) in frame.columns
        assert "mean_std_error" not in frame.columns
        verdict = study.verdict()
        assert "coverage and bias are consistent" not in verdict
        assert "certifies no confidence interval" in verdict


class TestTheArgumentAwareRowsAgreeWithTheCall:
    """A row that depends on the request is resolved for the request, before the call."""

    def test_the_interval_tipping_row_is_unavailable_with_the_raise_sentence(
        self, selector_fit: Any
    ) -> None:
        report = selector_fit.sensitivity.run_all(
            include_retargets=True, arguments={"tipping_gamma": {"use_ci": True}}
        )
        item = report["tipping_gamma"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert inference_refusal("tipping_gamma(use_ci=True)", DIAGNOSTIC) in item.detail
        assert DECLINED not in item.detail

    def test_the_point_tipping_row_stays_available(self, selector_fit: Any) -> None:
        report = selector_fit.sensitivity.run_all(
            include_retargets=True, arguments={"tipping_gamma": {"use_ci": False}}
        )
        assert report["tipping_gamma"].status is not AssessmentStatus.UNAVAILABLE

    def test_an_ordinary_fit_keeps_its_interval_tipping_row(self, ordinary_fit: Any) -> None:
        report = ordinary_fit.sensitivity.run_all(
            include_retargets=True, arguments={"tipping_gamma": {"use_ci": True}}
        )
        assert report["tipping_gamma"].status is not AssessmentStatus.UNAVAILABLE

    def test_a_level_is_not_applicable_before_the_fit_is_refused(self, missing_frame: Any) -> None:
        """``ey1`` has no E-value on any fit, so the request says that, not the refusal."""
        fit = (
            linear_ctmle("greedy", selection_folds=3, estimands=("ate", "ey1", "ey0"))
            .fit(missing_frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        level = fit.sensitivity.run_all(arguments={"evalue": {"estimand": "ey1"}})["evalue"]
        assert level.status is AssessmentStatus.NOT_APPLICABLE
        contrast = fit.sensitivity.run_all(arguments={"evalue": {"estimand": "ate"}})["evalue"]
        assert contrast.status is AssessmentStatus.UNAVAILABLE
        assert WORKING_MECHANISM_NOT_INFERENTIAL in contrast.detail


class TestNoAvailableRowDeclinesOnASelectorPath:
    """The combined report never runs a row that then raises ``CapabilityError``.

    Such a row reads ``unavailable`` with "the operation declined this request", which
    is the state RM12 removes: the declared row said available and the call refused.
    A future reader that raises on a selector fit would be demoted to that row silently,
    so this contract runs every row, refits and retargets included, on all three paths.
    """

    @pytest.mark.parametrize("strategy", sorted(SELECTOR_CONFIGS))
    def test_every_row_that_ran_answered(self, missing_frame: Any, strategy: str) -> None:
        fit = (
            linear_ctmle(strategy, estimands=("ate", "ey1", "ey0"), **SELECTOR_CONFIGS[strategy])
            .fit(missing_frame, outcome="Y", treatment="A", delta="Delta")
            .single()
        )
        assert fit.inference_status == DIAGNOSTIC
        report = fit.assess(
            include_refits=True,
            include_retargets=True,
            arguments={"tipping_gamma": {"use_ci": True}},
            random_state=0,
        )
        items = (*report.diagnostics.items, *report.sensitivity.items)
        declined = [item.name for item in items if DECLINED in (item.detail or "")]
        assert declined == []

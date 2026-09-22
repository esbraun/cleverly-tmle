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
before that field existed arrives without it, so the refusal on a restored selector fit is
carried by the field's persistence rather than by any rule this module can see.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import variable_importance
from cleverly.datasets import make_instrument, make_missing_outcome
from cleverly.estimators import CTMLE, TMLE
from cleverly.exceptions import WORKING_MECHANISM_NOT_INFERENTIAL, CapabilityError
from cleverly.sensitivity import missingness_tilt, tipping_gamma

#: Learners fast enough that a fixture costs less than a second, and explicit, so no
#: default library is constructed.
FAST: dict[str, Any] = {
    "outcome_learner": LinearRegression(),
    "treatment_learner": LogisticRegression(max_iter=1000),
    "cross_fit": False,
    "simultaneous": False,
    "random_state": 0,
}

#: The three spread columns an ordinary fit's tilt carries, and the three a selector-path
#: fit receives instead.  Named once, because the two tests below are one claim read from
#: opposite sides.
INFERENTIAL_COLUMNS = frozenset({"std_err", "ci_lower", "ci_upper"})
DIAGNOSTIC_COLUMNS = frozenset({"plugin_std_err", "plugin_interval_lower", "plugin_interval_upper"})


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
        CTMLE(strategy="greedy", selection_folds=3, estimands=("ate",), **FAST)
        .fit(missing_frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


@pytest.fixture(scope="module")
def ordinary_fit(missing_frame: Any) -> Any:
    """The same frame under ordinary TMLE, which must keep the inferential columns."""
    return (
        TMLE(estimands=("ate",), **FAST)
        .fit(missing_frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


class TestVariableImportanceRefusesBeforeItFits:
    """The multiplicity adjustment has no diagnostic form, so the entry point refuses."""

    @staticmethod
    def _call(strategy: str, **extra: Any) -> Any:
        frame, _ = make_instrument(n=300, seed=3)
        return variable_importance(
            frame,
            outcome="Y",
            candidates=["A"],
            covariates=["W1", "W2"],
            estimator=CTMLE(strategy=strategy, estimands=("ate",), **FAST, **extra),
        )

    @pytest.mark.parametrize(
        "strategy, extra",
        [
            ("greedy", {"selection_folds": 3}),
            ("ordered", {"selection_folds": 3, "ordering": ("W1", "W2")}),
            ("discrete", {"selection_folds": 3, "candidates": ((), ("W1",))}),
        ],
    )
    def test_a_selector_path_is_refused_by_the_name_the_caller_wrote(
        self, strategy: str, extra: dict[str, Any]
    ) -> None:
        with pytest.raises(CapabilityError) as raised:
            self._call(strategy, **extra)
        message = str(raised.value)
        assert message.startswith("variable_importance() is not defined here.")
        assert WORKING_MECHANISM_NOT_INFERENTIAL in message
        # The old message named an accessor no caller had written.
        assert ".pvalue" not in message

    def test_the_refusal_arrives_before_the_first_learner_is_fitted(self) -> None:
        frame, _ = make_instrument(n=300, seed=3)
        estimator = CTMLE(
            strategy="greedy",
            selection_folds=3,
            estimands=("ate",),
            outcome_learner=Unfittable(),
            treatment_learner=Unfittable(),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        with pytest.raises(CapabilityError):
            variable_importance(
                frame,
                outcome="Y",
                candidates=["A"],
                covariates=["W1", "W2"],
                estimator=estimator,
            )

    def test_the_outcome_adaptive_path_still_reports_its_adjusted_p_values(self) -> None:
        """The control against a refusal broadened to every collaborative fit."""
        result = self._call("oat")
        frame = result.to_frame()
        assert {"std_err", "ci_lower", "ci_upper", "p_value"} <= set(frame.columns)
        assert np.isfinite(result[0].adjusted_pvalue)

    def test_an_ordinary_estimator_is_untouched(self) -> None:
        frame, _ = make_instrument(n=300, seed=3)
        result = variable_importance(
            frame,
            outcome="Y",
            candidates=["A"],
            covariates=["W1", "W2"],
            estimator=TMLE(estimands=("ate",), **FAST),
        )
        assert np.isfinite(result[0].estimate.pvalue)


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

    def test_the_restored_result_reports_the_diagnostic_status(self, selector_fit: Any) -> None:
        restored = pickle.loads(pickle.dumps(selector_fit))
        assert restored["ate"].inference == "working_mechanism_plugin"

    def test_the_restored_result_still_refuses_its_interval(self, selector_fit: Any) -> None:
        restored = pickle.loads(pickle.dumps(selector_fit))
        with pytest.raises(CapabilityError) as raised:
            _ = restored["ate"].ci
        assert WORKING_MECHANISM_NOT_INFERENTIAL in str(raised.value)

    def test_the_retained_diagnostic_survives_the_round_trip_bit_for_bit(
        self, selector_fit: Any
    ) -> None:
        restored = pickle.loads(pickle.dumps(selector_fit))
        assert restored["ate"].plugin_std_error == selector_fit["ate"].plugin_std_error

    def test_the_restored_tilt_reports_the_diagnostic_columns(self, selector_fit: Any) -> None:
        """The two fixes meet here: a restored fit reaches the renamed columns too."""
        restored = pickle.loads(pickle.dumps(selector_fit))
        columns = set(missingness_tilt(restored, [0.0]).columns)
        assert columns >= DIAGNOSTIC_COLUMNS
        assert not (INFERENTIAL_COLUMNS & columns)

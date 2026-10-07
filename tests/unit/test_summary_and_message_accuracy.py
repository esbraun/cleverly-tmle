"""RM16: summaries, messages and argument checks that state what a fit records or a call needs.

RM16 in ``docs/roadmap.md`` at ``4ce96cda`` lists surfaces that misstated a recorded fact or what a call
accepts. This module holds the witnesses of the rows that RM16a corrects:

* the missing-outcome ``DataError`` and its two sensitivity siblings name ``missingness=`` on
  ``PointTreatment`` beside ``delta=`` on ``fit()`` or ``CausalData``;
* the in-sample C-TMLE selection-fold refusal names the selection fold, or the inner fold of
  the nested split, and a remedy that runs as written.  The nuisance-fold backstop names the
  same remedy, and a cross-fitted ``TMLE`` keeps the in-sample remedy;
* the continuous-treatment ``DataError`` names F21 and the in-sample fit on a cross-fitted
  ``TMLE`` with missing outcomes, and the in-sample fit it names runs.  A cross-fitted ``CTMLE``
  or ``DRTMLE`` refuses the dose without shift advice;
* ``benchmark`` refuses an empty request and an indicator column, accepts the logical name of
  an encoded covariate, drops its whole block, and accepts a boolean covariate;
* a facade reads a one-shot iterator once, on a direct call, in a combined report and in the
  cache key;
* the guarded DR-TMLE truncation curve refuses a refused reduction setting, or a construction
  other than the fitted one, through the row, the facade and the module call alike.

The backstop witness reaches the outcome regression of ``_selection_base``, which trains on
the respondents.  The other two selection calls of ``cross_fit_predictions``, the propensity of
``_fit_propensity_with`` and the missingness regression of ``_selection_base``, pass the same
constant and need a training set with no rows at all, which no preflight-free frame here
builds.

It also holds the witnesses of the summary displays that RM16b corrects:

* a longitudinal summary prints the reference only beside a contrast;
* a longitudinal identification summary prints the history at each treatment node;
* a DR-TMLE summary names the guard and the reduction, or the empty guard;
* the regime support table names the mechanism that each column reads;
* ``protocol="fingerprint"`` prints one protocol line, and no summary prints the fingerprint
  twice;
* the split-spread fact and the bootstrap row name their spread as the status allows, and the
  bootstrap heading names the full-refit bootstrap.

The mutation controls are hand mutations, recorded in the RM16a and RM16b pull requests: each
reverts one correction and names the test here that fails.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    LongitudinalTreatment,
    PointTreatment,
    RegimeContrast,
)
from cleverly._inference_status import status_record
from cleverly.assessment import INTERPRETERS
from cleverly.data.validate import MISSING_OUTCOME_DECLARATION
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_longitudinal,
    make_missing_outcome_binary,
    navigation_protocol,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE, TMLEResultSet
from cleverly.estimators.targeting import ONE_STEP_NESTED_REFUSAL
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Shift, Static
from cleverly.sensitivity import benchmark
from cleverly.sensitivity.positivity import truncation_curve
from cleverly.validation import RepeatSpreadRow
from tests.conftest import linear_drtmle, linear_in_sample
from tests.unit import _capability_sweep_support as sweep
from tests.unit import test_fold_policy_rules as fold_rules
from tests.unit.test_intervention_load_diagnostics import SUPPORT_SUMMARY_CAVEAT

# ------------------------------------------------------------ row 6: missing outcomes


def _missing_frame() -> pd.DataFrame:
    """``make_missing_outcome_binary(200, 4)`` with each unobserved outcome set to NaN."""
    frame, _ = make_missing_outcome_binary(n=200, seed=4)
    frame.loc[frame["Delta"] == 0, "Y"] = np.nan
    return frame


def _design() -> PointTreatment:
    return PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))


def _complete_study_fit() -> Any:
    """A study fit of the same law with every outcome observed."""
    return (
        CausalStudy(_missing_frame().fillna({"Y": 0.0}), design=_design())
        .identify(ATE())
        .estimate(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
    )


class TestTheMissingOutcomeRemedyNamesBothSpellings:
    """Row 6. One remedy names ``missingness=`` and ``delta=``, at three sites."""

    def test_the_declaration_names_the_study_and_the_direct_spelling(self) -> None:
        assert MISSING_OUTCOME_DECLARATION == (
            "missingness=<column> on PointTreatment, or delta=<column> on fit() or CausalData"
        )

    def test_a_study_without_missingness_reads_the_shared_sentence(self) -> None:
        with pytest.raises(DataError) as raised:
            CausalStudy(_missing_frame(), design=_design())
        assert str(raised.value) == (
            "Y has 48 missing value(s) but no missingness indicator was supplied. Declare one "
            f"(1 = outcome observed) with {MISSING_OUTCOME_DECLARATION}, so the missingness mechanism is "
            "estimated and enters the clever covariate."
        )

    def test_a_direct_fit_without_delta_reads_the_same_sentence(self) -> None:
        with pytest.raises(DataError) as study:
            CausalStudy(_missing_frame(), design=_design())
        with pytest.raises(DataError) as direct:
            TMLE(**linear_in_sample()).fit(
                _missing_frame().drop(columns=["Delta"]), outcome="Y", treatment="A"
            )
        assert str(direct.value) == str(study.value)

    def test_the_named_study_declaration_runs(self) -> None:
        """The remedy's first spelling fits the frame the refusal came from."""
        design = PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), missingness="Delta"
        )
        result = (
            CausalStudy(_missing_frame(), design=design)
            .identify(ATE())
            .estimate(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
            )
        )
        assert result.data.has_missing_outcome

    def test_the_missingness_tilt_refusal_names_both_spellings(self) -> None:
        with pytest.raises(CapabilityError) as raised:
            _complete_study_fit().sensitivity.missingness([1.0, 2.0])
        assert str(raised.value).endswith(
            "missingness_tilt requires a fit with missing outcomes. Declare the indicator "
            f"with {MISSING_OUTCOME_DECLARATION}, so the missingness mechanism is estimated."
        )

    def test_the_mechanism_axis_refusal_names_both_spellings(self) -> None:
        with pytest.raises(CapabilityError) as raised:
            _complete_study_fit().diagnostics.truncation_curve([0.05], mechanism=True)
        assert str(raised.value).endswith(
            f"Declare the indicator with {MISSING_OUTCOME_DECLARATION}, or the intermediate variable with "
            "intermediate=<column> on PointTreatment or fit()."
        )


# --------------------------------------------------- row 9: the selection-split remedy


def _selection_fit(seed: int) -> CTMLE:
    """The in-sample greedy search of the RM16 probe, on ``respondents_in_one_fold``."""
    return CTMLE(
        strategy="greedy",
        cross_fit=False,
        selection_folds=6,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        q_bounds=(0.0, 1.0),
        estimands=["ate"],
        simultaneous=False,
        random_state=seed,
    )


_SELECTION_COLUMNS = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"], "delta": "D"}


class TestTheSelectionSplitRefusalOffersAnInSampleFitNothing:
    """Row 9. The in-sample fit is already in sample, so the remedy names no ``cross_fit``."""

    REMEDY = (
        "fit strategy='oat' with every selector setting (selection_folds, "
        "selection_inner_folds, loss, penalty, ctmle_estimand) at its default, or the "
        "ordinary TMLE in sample, TMLE(cross_fit=False), neither of which draws a "
        "selection split"
    )

    def test_the_preflight_names_the_selection_fold_and_its_remedy(self) -> None:
        frame, seed = fold_rules.respondents_in_one_fold()
        with pytest.raises(DataError) as raised:
            _selection_fit(seed).fit(frame, **_SELECTION_COLUMNS)
        message = str(raised.value)
        assert message.startswith(
            "C-TMLE selection cannot fit its nuisances because selection fold 2's training "
            "complement contains no row with an observed outcome."
        )
        assert f"Either {self.REMEDY}, or collect more observations" in message
        assert "fit in sample with cross_fit=False on the engine" not in message

    def test_each_named_remedy_fits_the_refused_frame_as_written(self) -> None:
        """The refused fit with ``strategy='oat'`` and its selector setting at the default.

        The ordinary remedy is ``TMLE(cross_fit=False)``, given only the explicit learners
        a fast test needs in place of the default library.
        """
        frame, seed = fold_rules.respondents_in_one_fold()
        refused = _selection_fit(seed)
        oat = CTMLE(
            strategy="oat",
            cross_fit=refused.cross_fit,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            q_bounds=refused.q_bounds,
            estimands=["ate"],
            simultaneous=False,
            random_state=seed,
        )
        ordinary = TMLE(
            cross_fit=False,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
        )
        assert "ate" in oat.fit(frame, **_SELECTION_COLUMNS).single().estimates
        assert "ate" in ordinary.fit(frame, **_SELECTION_COLUMNS).single().estimates

    def test_the_nested_split_names_its_inner_fold(self) -> None:
        """Seed 0 passes the selection split and strands the respondents of a nested split."""
        frame, _ = fold_rules.respondents_in_one_fold()
        with pytest.raises(DataError) as raised:
            _selection_fit(0).fit(frame, **_SELECTION_COLUMNS)
        message = str(raised.value)
        assert message.startswith(
            "the nested split of C-TMLE selection fold 5 cannot fit its nuisances because "
            "inner fold 1's training complement contains no row with an observed outcome."
        )
        assert f"Either {self.REMEDY}, or collect more observations" in message

    def test_the_backstop_names_the_selection_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TMLE, "_check_training_support", lambda *a, **k: None)
        frame, seed = fold_rules.respondents_in_one_fold()
        with pytest.raises(ValueError) as raised:
            _selection_fit(seed).fit(frame, **_SELECTION_COLUMNS)
        message = str(raised.value)
        assert message.startswith("a cross-fitting fold has no trainable rows")
        assert f"Either {self.REMEDY}, or collect more observations" in message
        assert "fit in sample with cross_fit=False on the engine" not in message

    def test_a_cross_fitted_tmle_keeps_the_in_sample_remedy(self) -> None:
        """The control: a cross-fitted outer split can be fitted in sample instead."""
        frame, seed = fold_rules.TestAStrandedArmIsRefusedBeforeAnyLearner.stranded_frame()
        with pytest.raises(DataError) as raised:
            fold_rules.bounded_fit(n_folds=10, random_state=seed).fit(
                frame, outcome="Y", treatment="A", covariates=["W1", "W2"]
            )
        message = str(raised.value)
        assert "because repeat 0, fold " in message
        assert "Either fit in sample with cross_fit=False on the engine" in message


# -------------------------------------------- row 12: the suggested shift and F21


def _dose_frame() -> pd.DataFrame:
    """``make_missing_outcome_binary(400, 4)`` with a continuous dose ``D = A + N(0, 1)``."""
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    frame["D"] = frame["A"] + np.random.default_rng(0).normal(size=len(frame))
    return frame


def _dose_fit(frame: pd.DataFrame, *, cross_fit: bool, missing: bool, **settings: Any) -> Any:
    estimator = TMLE(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LinearRegression(),
        missingness_learner=LogisticRegression(max_iter=1000),
        cross_fit=cross_fit,
        simultaneous=False,
        random_state=0,
        **settings,
    )
    columns: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "D",
        "covariates": ("W1", "W2", "W3"),
        "treatment_kind": "continuous",
    }
    if missing:
        columns["delta"] = "Delta"
    else:
        frame = frame.drop(columns=["Delta"]).fillna({"Y": 0.0})
    return estimator.fit(frame, **columns)


_F21_SENTENCE = (
    " A cross-fitted fit with missing outcomes, declared with missingness=<column> on "
    "PointTreatment, or delta=<column> on fit() or CausalData, refuses a policy target (F21 "
    "in docs/roadmap.md). To estimate the natural course, fit in sample with cross_fit=False "
    "on the engine (CrossFitting(enabled=False))."
)


class TestTheSuggestedShiftNamesTheFitThatRunsIt:
    """Row 12. The natural course is suggested with the fit that can estimate it."""

    def test_a_cross_fitted_fit_with_delta_names_f21_and_the_in_sample_fit(self) -> None:
        with pytest.raises(DataError) as raised:
            _dose_fit(_dose_frame(), cross_fit=True, missing=True)
        assert str(raised.value).endswith(_F21_SENTENCE)

    @pytest.mark.parametrize(
        ("engine", "settings", "refusal"),
        [
            (CTMLE, {"strategy": "greedy"}, "CTMLE strategies require a discrete treatment."),
            (CTMLE, {"strategy": "oat"}, "CTMLE strategies require a discrete treatment."),
            (
                DRTMLE,
                {"randomized": True},
                "the reduced-dimension regressions read a per-arm mechanism g(a | W)",
            ),
        ],
        ids=["ctmle_greedy", "ctmle_oat", "drtmle"],
    )
    def test_an_engine_that_refuses_a_dose_gives_no_shift_advice(
        self, engine: type[TMLE], settings: dict[str, Any], refusal: str
    ) -> None:
        """The controls: a cross-fitted CTMLE or DRTMLE with ``delta=`` refuses the dose.

        Neither engine fits a shift, in sample or cross-fitted, so neither message names a
        shift, F21 or the in-sample fit.
        """
        estimator = engine(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LinearRegression(),
            missingness_learner=LogisticRegression(max_iter=1000),
            cross_fit=True,
            simultaneous=False,
            random_state=0,
            **settings,
        )
        with pytest.raises(CapabilityError) as raised:
            estimator.fit(
                _dose_frame(),
                outcome="Y",
                treatment="D",
                delta="Delta",
                covariates=("W1", "W2", "W3"),
                treatment_kind="continuous",
            )
        message = str(raised.value)
        assert message.startswith(refusal)
        for advice in ("policies=", "F21", "cross_fit=False"):
            assert advice not in message

    def test_the_named_in_sample_fit_reports_the_natural_course(self) -> None:
        result = _dose_fit(
            _dose_frame(), cross_fit=False, missing=True, policies=[Shift(0.0, cap=None)]
        ).single()
        assert "ey_policy[natural course]" in result.estimates

    @pytest.mark.parametrize(
        ("cross_fit", "missing"), [(False, True), (True, False)], ids=["in_sample", "complete"]
    )
    def test_a_fit_the_shift_refusal_does_not_reach_names_no_f21(
        self, cross_fit: bool, missing: bool
    ) -> None:
        """The controls: an in-sample fit, and a cross-fitted fit with complete outcomes."""
        with pytest.raises(DataError) as raised:
            _dose_fit(_dose_frame(), cross_fit=cross_fit, missing=missing)
        message = str(raised.value)
        assert message.startswith("D was declared continuous")
        assert "F21" not in message


# ----------------------------------------------------- rows 13 and 14: benchmark names


class TestAnEmptyBenchmarkRequestIsMalformed:
    """Row 13. ``covariates=[]`` drops nothing, so it is refused before the refit."""

    EMPTY = (
        "benchmark needs at least one covariate to drop; covariates= names none, and a "
        "refit that drops nothing measures nothing"
    )

    @pytest.mark.parametrize("surface", ["module", "facade"])
    def test_the_empty_request_raises_before_any_refit(
        self, surface: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = sweep.fit_ordinary()

        def refit(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the empty request reached the refit")

        monkeypatch.setattr(result.estimator, "refit", refit)
        with pytest.raises(DataError) as raised:
            if surface == "module":
                benchmark(result, [])
            else:
                result.sensitivity.benchmark(covariates=[])
        assert str(raised.value) == self.EMPTY


def _encoded_fit() -> Any:
    """``make_linear_ate(400, 2)`` with a three-level covariate ``V``, fitted in sample."""
    frame, _ = make_linear_ate(n=400, seed=2)
    frame["V"] = np.random.default_rng(5).choice(["a", "b", "c"], size=len(frame))
    covariates = ("W1", "W2", "W3", "W4", "V")
    return (
        TMLE(**linear_in_sample())
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def _short_psi(result: Any, columns: tuple[str, ...], seed: int) -> float:
    """The estimate of a refit that drops ``columns``, under the benchmark's seed."""
    short = result.estimator.refit(
        result.data.without_covariates(columns),
        intermediate_value=result.intermediate_value,
        random_state=seed,
    )
    return float(short["ate"].psi)


class TestBenchmarkNamesAreLogicalCovariates:
    """Row 14. A benchmark names the covariate the fit adjusts for, not an indicator."""

    def test_the_logical_name_drops_the_whole_encoded_block(self) -> None:
        result = _encoded_fit()
        assert result.data.covariate_names == ("W1", "W2", "W3", "W4", "V__b", "V__c")
        report = benchmark(result, ["V"])
        assert report.covariates == ("V",)
        block = _short_psi(result, ("V__b", "V__c"), report.random_state)
        assert report.psi_short == block
        assert report.psi_short != _short_psi(result, ("V__b",), report.random_state)

    def test_the_facade_reads_the_logical_name(self) -> None:
        result = _encoded_fit()
        assert result.sensitivity.benchmark(covariates=["V"]).covariates == ("V",)

    def test_an_indicator_column_is_refused_by_name(self) -> None:
        with pytest.raises(DataError) as raised:
            benchmark(_encoded_fit(), ["V__b"])
        assert str(raised.value) == (
            "'V__b' is an indicator column of the encoded covariate 'V'. Name 'V' to drop "
            "its whole encoded block"
        )

    def test_an_unknown_name_lists_the_logical_covariates(self) -> None:
        with pytest.raises(DataError) as raised:
            benchmark(_encoded_fit(), ["Z"])
        assert str(raised.value) == (
            "unknown covariates ['Z']; this fit adjusts for ['W1', 'W2', 'W3', 'W4', 'V']"
        )

    def test_a_numeric_covariate_still_runs(self) -> None:
        """The control: a proper subset of numeric covariates."""
        assert benchmark(_encoded_fit(), ["W1"]).covariates == ("W1",)

    def test_a_boolean_covariate_runs_and_its_row_agrees(self) -> None:
        """A boolean covariate is one column under its own name, so its name is no indicator."""
        frame, _ = make_linear_ate(n=400, seed=2)
        frame["B"] = np.random.default_rng(5).random(len(frame)) < 0.5
        result = (
            TMLE(**linear_in_sample())
            .fit(frame, outcome="Y", treatment="A", covariates=("W1", "W2", "W3", "W4", "B"))
            .single()
        )
        assert result.data.covariate_names == ("W1", "W2", "W3", "W4", "B")
        row = result.sensitivity._capability_for_arguments("benchmark", {"covariates": ["B"]})
        assert row.available
        report = benchmark(result, ["B"])
        assert report.covariates == ("B",)
        assert report.psi_short == _short_psi(result, ("B",), report.random_state)


# --------------------------------------------------- row 13: one-shot iterator arguments


class TestAFacadeReadsAnIteratorOnce:
    """Row 13. A generator argument answers as the list of the same names does."""

    def test_a_benchmark_generator_equals_the_list(self) -> None:
        result = sweep.fit_ordinary()
        listed = result.sensitivity.benchmark(covariates=["W1"])
        generated = result.sensitivity.benchmark(covariates=(name for name in ["W1"]))
        assert generated.covariates == ("W1",)
        assert generated.delta_psi == listed.delta_psi

    def test_a_refute_generator_equals_the_list(self) -> None:
        result = sweep.fit_ordinary()
        listed = result.diagnostics.refute(tests=["placebo"], n_replicates=1, random_state=1)
        generated = result.diagnostics.refute(
            tests=(name for name in ["placebo"]), n_replicates=1, random_state=1
        )
        assert [test.name for test in generated.tests] == ["placebo"]
        pd.testing.assert_frame_equal(generated.to_frame(), listed.to_frame())

    def test_a_combined_report_runs_and_records_the_generated_names(self) -> None:
        """``assess()`` runs each operation on the generated names and records them."""
        result = sweep.fit_ordinary()
        report = result.assess(
            include_refits=True,
            random_state=1,
            arguments={
                "benchmark": {"covariates": (name for name in ["W1"])},
                "refute": {"tests": (name for name in ["placebo"]), "n_replicates": 1},
            },
        )
        benchmarked = next(item for item in report.sensitivity.items if item.name == "benchmark")
        refuted = next(item for item in report.diagnostics.items if item.name == "refute")
        assert benchmarked.report.covariates == ("W1",)
        assert benchmarked.arguments["covariates"] == ("W1",)
        assert [test.name for test in refuted.report.tests] == ["placebo"]
        assert refuted.arguments["tests"] == ("placebo",)

    def test_two_truncation_generators_give_their_own_curves(self) -> None:
        """A generator's ``repr`` names a memory address, which CPython reuses.

        Before the fix the cache key held that ``repr``, so a second generator with other
        bounds could read the first curve back.  The key must hold the bounds themselves.
        """
        result = sweep.fit_ordinary()
        result.diagnostics.truncation_curve(bound for bound in [0.05])
        second = result.diagnostics.truncation_curve(bound for bound in [0.2, 0.3])
        listed = result.diagnostics.truncation_curve([0.2, 0.3])
        pd.testing.assert_frame_equal(second, listed)
        assert not any("generator" in key for key in result.assessment_cache)

    def test_two_missingness_generators_give_their_own_tilts(self) -> None:
        """The same for an operation that the sensitivity facade dispatches."""
        result = sweep.fit_missing()
        result.sensitivity.missingness(gamma for gamma in [1.0])
        second = result.sensitivity.missingness(gamma for gamma in [2.0, 3.0])
        listed = result.sensitivity.missingness([2.0, 3.0])
        pd.testing.assert_frame_equal(second, listed)
        assert not any("generator" in key for key in result.assessment_cache)


# ------------------------ row 15: the guarded curve asks its reduction settings


class TestTheGuardedCurveReadsItsReductionSettings:
    """Row 15. The row, the facade call and the module call agree on a guarded fit.

    The curve retargets the cached primary nuisances, so a reconfigured primary split runs
    (``TestGuardedTruncationUsesCachedNuisanceReplay`` in
    ``tests/unit/test_capability_row_predicates.py``).  The alternation refits the reduced
    regressions under the live reduction settings, so a refused setting or a construction
    other than the fitted one refuses every surface.  Without the check, a changed
    ``reduction`` or ``reduced_crossfit`` and the one-step nested pair raise a plain
    ``ValueError`` inside the retarget, and ``guard=("Q",)`` on a ``("Q", "g")`` fit returns
    the curve of another construction: an ATE of 0.3235 where the fitted construction gives
    0.3168.
    """

    _PREFIX = (
        "a guarded DR-TMLE truncation curve refits the reduced regressions under this "
        "estimator's reduction settings, and it refuses those settings: "
    )

    @staticmethod
    def _refused_everywhere(result: Any) -> str:
        assert not result.diagnostics.capability("truncation_curve").available
        with pytest.raises(CapabilityError):
            result.diagnostics.truncation_curve([0.05])
        with pytest.raises(CapabilityError) as raised:
            truncation_curve(result, [0.05])
        return str(raised.value)

    def test_a_refused_reduction_setting_refuses_every_surface(self) -> None:
        # ``n_folds=3`` passes the nested fold-count check, so the one-step refusal answers.
        result = sweep.reconfigured(
            sweep.fit_drtmle(), targeting="one_step", reduced_crossfit="nested", n_folds=3
        )
        assert self._refused_everywhere(result) == self._PREFIX + ONE_STEP_NESTED_REFUSAL

    @pytest.mark.parametrize(
        ("configuration", "named"),
        [
            ({"guard": ("Q",)}, "guard=('Q',) (fitted ('Q', 'g'))"),
            ({"reduction": "bivariate"}, "reduction='bivariate' (fitted 'univariate')"),
            (
                {"reduced_crossfit": "nested", "n_folds": 3},
                "reduced_crossfit='nested' (fitted 'pooled')",
            ),
        ],
        ids=["guard", "reduction", "reduced_crossfit"],
    )
    def test_a_changed_construction_refuses_every_surface(
        self, configuration: dict[str, Any], named: str
    ) -> None:
        result = sweep.reconfigured(sweep.fit_drtmle(), **configuration)
        assert self._refused_everywhere(result) == (
            self._PREFIX
            + "the estimator's reduction construction differs from the fitted one: "
            + named
            + ". Fit again with these settings, or restore the fitted ones."
        )

    def test_a_result_without_its_construction_record_refuses(self) -> None:
        """``extra["drtmle"]`` is the only record of the fitted guard, so its loss refuses."""
        result = dataclasses.replace(sweep.fit_drtmle(), extra={})
        assert self._refused_everywhere(result) == (
            self._PREFIX
            + "the result records no fitted DR-TMLE construction in extra['drtmle'], so the "
            "guard of the reduced refit cannot be checked. Fit again."
        )

    def test_the_live_fit_answers_on_every_surface(self) -> None:
        """The control: the unmodified estimator keeps the row, and both calls agree."""
        result = sweep.fit_drtmle()
        assert result.diagnostics.capability("truncation_curve").available
        facade = result.diagnostics.truncation_curve([0.05])
        module = truncation_curve(result, [0.05])
        pd.testing.assert_frame_equal(facade, module)


# ============================================================== RM16b: summary displays


def _longitudinal_study(protocol: Any = None) -> CausalStudy:
    """The two-node design of :func:`tests.unit._capability_sweep_support.fit_ltmle`."""
    frame, _ = make_longitudinal(n=400, seed=12)
    return CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        ),
        protocol=protocol,
    )


def _longitudinal_fit(effect: Any) -> Any:
    """The in-sample learners and settings of ``fit_ltmle``."""
    return effect.estimate(
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=5,
        simultaneous=False,
        cross_fit=False,
    )


def _contrast() -> RegimeContrast:
    return RegimeContrast({"always": 1, "never": 0}, reference="never")


# --------------------------------------------- row 1: the reference line of a mean


class TestTheReferenceLineBelongsToAContrast:
    """Row 1. A longitudinal summary prints the reference only beside a contrast."""

    def test_a_regime_mean_summary_prints_no_reference(self) -> None:
        result = sweep.fit_ltmle()
        lines = result.summary().splitlines()
        assert not any(line.strip().startswith("reference:") for line in lines)
        # The replay reads the field, so the correction hides the line and keeps the field.
        assert result.config.reference == "always"

    def test_a_regime_contrast_summary_keeps_its_reference(self) -> None:
        """The nonzero witness: a result that holds a contrast prints its reference."""
        result = _longitudinal_fit(_longitudinal_study().identify(_contrast()))
        assert "  reference: never" in result.summary().splitlines()


# ------------------------------------------- row 2: the history at each treatment node


class TestTheIdentificationSummaryPrintsEachNodesHistory:
    """Row 2. A longitudinal identification summary gives the history at each node."""

    def test_each_node_lists_the_history_its_mechanism_reads(self) -> None:
        """The validated names, in design order: a constant ``K`` is dropped, not printed."""
        frame, _ = make_longitudinal(n=400, seed=12)
        design = LongitudinalTreatment(
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W1", "W2", "K"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
        study = CausalStudy(frame.assign(K=1.0), design=design)
        assert "K" in study.design.baseline
        lines = study.identify(_contrast()).summary().splitlines()
        assert "history at A1: ['W1', 'W2']" in lines
        assert "history at A2: ['W1', 'W2', 'L2', 'A1']" in lines
        assert not any(line.startswith("adjustment/history") for line in lines)

    def test_a_point_design_keeps_its_adjustment_line(self) -> None:
        study = CausalStudy(_missing_frame().fillna({"Y": 0.0}), design=_design())
        lines = study.identify(ATE()).summary()
        assert "adjustment/history: ['W1', 'W2', 'W3']" in lines.splitlines()
        assert "history at" not in lines


# --------------------------------------------------------- row 3: the DR-TMLE line


def _guarded(guard: tuple[str, ...]) -> Any:
    """:func:`tests.unit._capability_sweep_support.fit_drtmle` at the given guard."""
    frame, _ = make_binary_outcome(n=160, seed=11)
    estimator = DRTMLE(**linear_drtmle(n_folds=2, estimands=("ate",)), guard=guard)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


class TestTheDRTMLESummaryNamesItsGuardAndReduction:
    """Row 3. A DR-TMLE summary names the method, the guard and the reduction."""

    def test_the_default_guard(self) -> None:
        lines = sweep.fit_drtmle().summary().splitlines()
        assert "DR-TMLE: guard Q, g; univariate reduction" in lines

    def test_a_one_equation_guard(self) -> None:
        lines = _guarded(("Q",)).summary().splitlines()
        assert "DR-TMLE: guard Q; univariate reduction" in lines

    def test_an_empty_guard_names_the_ordinary_tmle(self) -> None:
        lines = _guarded(()).summary().splitlines()
        assert (
            "DR-TMLE: empty guard, so no reduced regression was fitted and the estimate is "
            "the ordinary TMLE"
        ) in lines

    def test_the_bivariate_reduction(self) -> None:
        frame, _ = make_binary_outcome(n=160, seed=11)
        estimator = DRTMLE(**linear_drtmle(n_folds=2, estimands=("ate",)), reduction="bivariate")
        lines = estimator.fit(frame, outcome="Y", treatment="A").single().summary().splitlines()
        assert "DR-TMLE: guard Q, g; bivariate reduction" in lines

    def test_the_missing_outcome_reduction(self) -> None:
        """The line names the construction that ran, which the constructor did not name."""
        estimator = DRTMLE(
            **linear_drtmle(
                estimands=("ate",),
                cross_fit=False,
                randomized=True,
                missingness_learner=LogisticRegression(max_iter=1000),
            )
        )
        assert estimator.reduction == "univariate"
        result = estimator.fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        lines = result.single().summary().splitlines()
        assert "DR-TMLE: guard Q, g; missing_outcome reduction" in lines

    def test_an_ordinary_fit_prints_no_dr_tmle_line(self) -> None:
        lines = sweep.fit_ordinary().summary().splitlines()
        assert not any(line.startswith("DR-TMLE:") for line in lines)


# ------------------------------------------------------ row 4: the support columns' basis


def _kish(values: np.ndarray) -> float:
    return float(values.sum() ** 2 / (values**2).sum())


class TestTheSupportTableStatesEachColumnsBasis:
    """Row 4. The regime table says which mechanism each column reads.

    A bound that binds is the nonzero witness: at ``g_bounds=(0.4, 0.6)`` the two bases
    give different numbers. ``min g`` falls below the bound, ``ratio effective n`` equals
    the Kish count of the untruncated ratio, and ``score load`` equals the Kish count of
    the truncated one.
    """

    def test_a_binding_bound_separates_the_two_bases(self) -> None:
        frame, _ = make_linear_ate(n=400, seed=2)
        result = (
            TMLE(**linear_in_sample(interventions=(Static(1), Static(0)), g_bounds=(0.4, 0.6)))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        report = result.diagnostics.support()
        assert tuple(report.summary().splitlines()[-2:]) == SUPPORT_SUMMARY_CAVEAT

        g = np.asarray(result.nuisance.propensity.values, dtype=float)
        observed = np.asarray(result.data.treatment, dtype=float)
        for code, item in zip((1, 0), report.regimes.values(), strict=True):
            arm = observed == code
            raw = np.where(arm, 1.0 / g[:, code], 0.0)
            clipped = np.where(arm, 1.0 / np.clip(g[:, code], 0.4, 0.6), 0.0)
            assert item.min_support_propensity < 0.4
            assert item.effective_sample_size == pytest.approx(_kish(raw), rel=1e-9)
            assert item.max_ratio == pytest.approx(raw.max(), rel=1e-12)
            assert item.max_ratio > clipped.max()
            assert item.score_load is not None
            assert item.score_load["effective"] == pytest.approx(_kish(clipped), rel=1e-9)
            assert item.score_load["effective"] > item.effective_sample_size + 1.0


# ------------------------------------------------- row 5: the fingerprint-only protocol


def _record_lines() -> tuple[str, ...]:
    return navigation_protocol().summary_lines()


def _count(text: str, lines: tuple[str, ...]) -> list[int]:
    """How many times each of ``lines`` is a line of ``text``, indent removed."""
    printed = [line.strip() for line in text.splitlines()]
    return [printed.count(line) for line in lines]


class TestTheProtocolPrintsOnceOrAsItsFingerprint:
    """Row 5. ``protocol="fingerprint"`` prints one line; the default prints the record.

    The result summary no longer repeats the fingerprint as a provenance line.
    """

    @pytest.fixture(scope="class")
    def point(self) -> Any:
        effect = CausalStudy(
            _missing_frame().fillna({"Y": 0.0}),
            design=_design(),
            protocol=navigation_protocol(),
        ).identify(ATE())
        return effect, effect.estimate(**linear_in_sample())

    @pytest.fixture(scope="class")
    def longitudinal(self) -> Any:
        effect = _longitudinal_study(navigation_protocol()).identify(_contrast())
        return effect, _longitudinal_fit(effect)

    @staticmethod
    def _summaries(effect: Any, result: Any, **options: Any) -> list[str]:
        return [
            effect.summary(**options),
            "\n".join(effect.summary_lines(**options)),
            result.summary(**options),
        ]

    @pytest.mark.parametrize("kind", ["point", "longitudinal"])
    def test_the_fingerprint_option_prints_one_line(
        self, kind: str, request: pytest.FixtureRequest
    ) -> None:
        effect, result = request.getfixturevalue(kind)
        for text in self._summaries(effect, result, protocol="fingerprint"):
            assert _count(text, _record_lines()) == [1] + [0] * 10

    @pytest.mark.parametrize("kind", ["point", "longitudinal"])
    def test_the_default_prints_the_record_once_in_order(
        self, kind: str, request: pytest.FixtureRequest
    ) -> None:
        effect, result = request.getfixturevalue(kind)
        fingerprint = navigation_protocol().fingerprint
        for text in self._summaries(effect, result):
            printed = [line.strip() for line in text.splitlines()]
            start = printed.index(_record_lines()[0])
            assert tuple(printed[start : start + 11]) == _record_lines()
            assert text.count(fingerprint) == 1

    def test_an_unknown_option_is_refused(self, point: Any) -> None:
        effect, result = point
        for call in (effect.summary, effect.summary_lines, result.summary):
            with pytest.raises(ValueError, match="protocol must be 'full' or 'fingerprint'"):
                call(protocol="short")

    @pytest.mark.parametrize("levels", [(None,), (0.0, 1.0)], ids=["single", "by_level"])
    def test_a_result_set_forwards_the_option(
        self, levels: tuple[float | None, ...], point: Any
    ) -> None:
        _, result = point
        results = TMLEResultSet(dict.fromkeys(levels, result), intermediate_name="Z")
        text = results.summary(protocol="fingerprint")
        counts = _count(text, _record_lines())
        assert counts == [len(levels)] + [0] * 10

    def test_an_absent_protocol_prints_absent_in_both_modes(self) -> None:
        fitted = TMLE(**linear_in_sample()).fit(_linear_frame(), outcome="Y", treatment="A")
        for option in ("full", "fingerprint"):
            assert "causal study protocol: absent" in fitted.summary(protocol=option)
        with pytest.raises(ValueError, match="protocol must be 'full' or 'fingerprint'"):
            fitted.summary(protocol="short")


def _linear_frame() -> pd.DataFrame:
    frame, _ = make_linear_ate(n=200, seed=1)
    return frame


# -------------------------------------------------------- row 7: the split-spread name


def _binary_frame() -> pd.DataFrame:
    frame, _ = make_binary_outcome(n=400, seed=17)
    return frame


def _repeated(estimator: type, **settings: Any) -> Any:
    """Two cross-fitting draws of two folds, which is what reports a split spread."""
    fitted = estimator(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        estimands=("ate",),
        simultaneous=False,
        random_state=0,
        repeats=2,
        n_folds=2,
        **settings,
    )
    return fitted.fit(_binary_frame(), outcome="Y", treatment="A").single()


class TestTheSplitSpreadFactReadsTheStatusName:
    """Row 7. The ``nuisance_models`` row names the spread ratio as its report does."""

    def test_a_selector_fit_names_the_plugin_ratio(self) -> None:
        result = _repeated(CTMLE, strategy="greedy", selection_folds=3)
        report = result.diagnostics.nuisance_models()
        widest = max(report.repeat_spread, key=lambda row: row.ratio_to_standard_error)
        detail = result.diagnostics.run_all()["nuisance_models"].detail
        assert (
            f"largest sd/plugin se {widest.ratio_to_standard_error:.3g} for {widest.estimand}"
        ) in detail
        assert "largest sd/se" not in detail

    @pytest.mark.parametrize(
        ("status", "expected"),
        [("working_mechanism_plugin", "sd/plugin se"), ("influence_curve", "sd/se")],
    )
    def test_an_unavailable_ratio_reads_the_status_name(self, status: str, expected: str) -> None:
        """No row has a finite ratio, so the fact names the ratio as unavailable."""
        report = SimpleNamespace(
            findings=(),
            models=(),
            selection=None,
            selection_omission=None,
            repeat_spread=(RepeatSpreadRow("ate", 2, 0.0, 0.0, float("nan")),),
            repeat_spread_omission=None,
            n_repeats=2,
            reported_repeat=1,
            inference=status,
        )
        detail = INTERPRETERS["nuisance_models"](report, None).detail
        assert detail.endswith(f"across 2 draws; {expected} unavailable")

    def test_an_ordinary_fit_keeps_the_inferential_ratio(self) -> None:
        detail = _repeated(TMLE).diagnostics.run_all()["nuisance_models"].detail
        assert "largest sd/se" in detail
        assert "plugin" not in detail


# --------------------------------------------- row 11 and the name: the bootstrap row


class TestTheBootstrapPublishesUnderTheStatusName:
    """Row 11. A non-inferential fit publishes its bootstrap spread as a standard deviation.

    The heading names the procedure: each replicate refits the whole estimator.
    """

    @staticmethod
    def _fit(estimator: type, **settings: Any) -> Any:
        fitted = estimator(
            outcome_learner=LogisticRegression(max_iter=1000),
            treatment_learner=LogisticRegression(max_iter=1000),
            estimands=("ate",),
            simultaneous=False,
            random_state=0,
            cross_fit=False,
            n_bootstrap=4,
            **settings,
        )
        return fitted.fit(_binary_frame(), outcome="Y", treatment="A").single()

    @staticmethod
    def _heading(result: Any) -> str:
        usable = result.bootstrap.n_requested - result.bootstrap.n_failed
        return f"full-refit bootstrap (iid resampling, {usable} usable replicates):"

    def test_a_selector_fit_publishes_a_standard_deviation(self) -> None:
        result = self._fit(CTMLE, strategy="greedy", selection_folds=3)
        estimate = result["ate"]
        row = estimate.to_dict()
        assert row["bootstrap_sd"] == estimate.bootstrap.std_error
        assert "bootstrap_std_err" not in row
        low, high = estimate.bootstrap.ci
        lines = result.summary().splitlines()
        assert self._heading(result) in lines
        assert (
            f"  ate   bootstrap sd {estimate.bootstrap.std_error:.4g}  percentile range "
            f"[{low:.5g}, {high:.5g}] ({status_record(estimate.inference).bootstrap_note})"
        ) in lines

    def test_an_ordinary_fit_keeps_the_standard_error(self) -> None:
        result = self._fit(TMLE)
        estimate = result["ate"]
        row = estimate.to_dict()
        assert row["bootstrap_std_err"] == estimate.bootstrap.std_error
        assert "bootstrap_sd" not in row
        low, high = estimate.bootstrap.ci
        lines = result.summary().splitlines()
        assert self._heading(result) in lines
        assert (
            f"  ate   bootstrap se {estimate.bootstrap.std_error:.4g}  "
            f"percentile CI [{low:.5g}, {high:.5g}]"
        ) in lines

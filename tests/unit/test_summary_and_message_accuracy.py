"""RM16a: messages and argument checks that state what a call accepts or needs.

RM16 in ``docs/roadmap.md`` lists surfaces that misstated a recorded fact or what a call
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

The mutation controls are hand mutations, recorded in the RM16a pull request: each reverts one
correction and names the test here that fails.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.data.validate import MISSING_OUTCOME_DECLARATION
from cleverly.datasets import make_linear_ate, make_missing_outcome_binary
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.targeting import ONE_STEP_NESTED_REFUSAL
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Shift
from cleverly.sensitivity import benchmark
from cleverly.sensitivity.positivity import truncation_curve
from tests.conftest import linear_in_sample
from tests.unit import _capability_sweep_support as sweep
from tests.unit import test_fold_policy_rules as fold_rules

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
    "PointTreatment, or delta=<column> on fit() or CausalData, refuses a shift target (F21 "
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
        for advice in ("shifts=", "F21", "cross_fit=False"):
            assert advice not in message

    def test_the_named_in_sample_fit_reports_the_natural_course(self) -> None:
        result = _dose_fit(
            _dose_frame(), cross_fit=False, missing=True, shifts=[Shift(0.0, cap=None)]
        ).single()
        assert "ey_shift[natural course]" in result.estimates

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

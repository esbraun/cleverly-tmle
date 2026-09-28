"""Every learned-rule composition outside the RM30 contract refuses, in order, before a learner.

The table in ``docs/roadmap.md`` (RM30, "The refusals, their order and the shared text")
lists fourteen rows.  Each row raises :class:`~cleverly.exceptions.CapabilityError` with
its item before any learner is fitted, at the engine and at ``CausalStudy``.  A refusal
that no setting repairs comes before a refusal whose remedy is a setting, which the RM24
order rule requires, so the pair witnesses below meet the earlier row.

No remedy may send the caller to a call that is refused.  Each remedy is therefore run as
written, and each shared refusal that a learned-rule fit can meet is checked for the
in-sample remedy it must not name.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import (
    CausalStudy,
    CrossFitting,
    LearnedRuleValue,
    PointTreatment,
    RegimeMean,
    TMLEMethod,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, LearnedRule, Shift, Static
from cleverly.interventions.learned import (
    LEARNED_RULE_REFIT_REFUSAL,
    LEARNED_RULE_REMEDY,
    LEARNED_RULE_SCALE_REMEDY,
    LEARNED_RULE_SENSITIVITY_REFUSAL,
)
from cleverly.learners.crossfit import _POST_DRAW_REMEDY
from cleverly.methods import ModelSpec
from cleverly.msm import MSM
from cleverly.sensitivity import evalue
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.sensitivity.missingness import fit_wide_tilt_refusal
from cleverly.sensitivity.omitted_variable import fit_wide_bound_refusal
from cleverly.validation.refute import refute
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The engine module, whose name the package's ``tmle`` function shadows.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

#: The two routes a learned-rule message must not offer: an in-sample fit, which X11 (a)
#: refuses, and a shift, which F17 refuses beside ``learned_rule=``.
FORBIDDEN = ("cross_fit=False", "CrossFitting(enabled=False)", "shifts=[")


def _engine(**overrides: Any) -> TMLE:
    """A learned-rule estimator whose learners fail if a refusal comes too late."""
    return TMLE(
        **{
            "learned_rule": LearnedRule(),
            **never_fit_learners(),
            **support.FOLD_EVALUATED,
            "n_folds": 5,
            "simultaneous": False,
            "random_state": 0,
            **overrides,
        }
    )


def _frame(kind: str = "binary") -> Any:
    """The support law, reshaped to the data declaration a row refuses."""
    frame = support.law_frame(120, 3)
    if kind == "continuous":
        frame["A"] = frame["A"] + frame["W1"]
    if kind == "multi_arm":
        frame["A"] = np.arange(len(frame)) % 3
    if kind == "delta":
        frame["Delta"] = (np.arange(len(frame)) % 5 != 0).astype(float)
    if kind == "intermediate":
        frame["Z"] = (np.arange(len(frame)) % 2).astype(float)
    if kind == "weights":
        frame["w"] = 1.0 + 0.5 * frame["W2"]
    if kind == "id":
        frame["cluster"] = np.arange(len(frame)) // 4
    return frame


#: The fit arguments of each data declaration, beside the frame of :func:`_frame`.
_FIT_ARGUMENTS: dict[str, dict[str, Any]] = {
    "binary": {},
    "continuous": {"treatment_kind": "continuous"},
    "multi_arm": {},
    "delta": {"delta": "Delta"},
    "intermediate": {"intermediate": "Z"},
    "weights": {"weights": "w"},
    "id": {"id": "cluster"},
    "strata": {"strata": ("W2",)},
}


def _fit(estimator: TMLE, kind: str = "binary") -> Any:
    frame = _frame(kind)
    return estimator.fit(
        frame, outcome="Y", treatment="A", covariates=support.COVARIATES, **_FIT_ARGUMENTS[kind]
    )


def _refused(call: Any, item: str) -> str:
    """Run ``call``, which must refuse with ``item`` and fit no learner; return the message."""
    NeverFit.calls = 0
    with pytest.raises(CapabilityError) as caught:
        call()
    assert NeverFit.calls == 0
    message = str(caught.value)
    assert item in message, message
    return message


#: Rows 3 to 9: the data declarations that no setting repairs.
DATA_ROWS: dict[str, str] = {
    "continuous": "F27",
    "multi_arm": "X11 (d)",
    "delta": "F21",
    "intermediate": "F27",
    "weights": "F27",
    "id": "F27",
    "strata": "F27",
}

#: Rows 10 to 14: the settings, each with its item.
SCHEME_ROWS: dict[str, tuple[dict[str, Any], str]] = {
    "in_sample": ({"cross_fit": False, "cv_evaluation": False}, "X11 (a)"),
    "stacked": ({"cv_evaluation": False}, "X11 (e)"),
    "fold_targeting": ({"targeting_scheme": "fold"}, "X11 (h)"),
    "repeats": ({"repeats": 2}, "F27"),
    "bootstrap": ({"n_bootstrap": 10}, "F27"),
}


class TestEachRowRefusesBeforeAnyLearner:
    """Rows 1 to 14 at the engine."""

    @pytest.mark.parametrize("variant", [CTMLE, DRTMLE])
    @pytest.mark.parametrize("fold_evaluation", [False, True])
    def test_row_1_the_variants(self, variant: type[TMLE], fold_evaluation: bool) -> None:
        """Before the variant's own refusal of fold evaluation, whose remedy meets this one."""
        _refused(
            lambda: variant(
                learned_rule=LearnedRule(), cross_fit=True, cv_evaluation=fold_evaluation
            ),
            "F27",
        )

    @pytest.mark.parametrize(
        ("overrides", "item"),
        [
            ({"interventions": [Static(1)]}, "X11 (c)"),
            ({"reference": "learned rule"}, "X11 (c)"),
            ({"estimands": ("ate",)}, "X11 (c)"),
            ({"shifts": [Shift(0.5, cap=None)]}, "F17"),
            ({"incremental": [Incremental(2.0)]}, "F17"),
            ({"msm": MSM.linear()}, "F17"),
        ],
    )
    def test_row_2_another_axis_or_a_contrast(self, overrides: dict[str, Any], item: str) -> None:
        _refused(lambda: _engine(**overrides), item)

    @pytest.mark.parametrize("kind", list(DATA_ROWS))
    def test_rows_3_to_9_the_data(self, kind: str) -> None:
        _refused(lambda: _fit(_engine(), kind), DATA_ROWS[kind])

    @pytest.mark.parametrize("row", list(SCHEME_ROWS))
    def test_rows_10_to_14_the_settings(self, row: str) -> None:
        overrides, item = SCHEME_ROWS[row]
        message = _refused(lambda: _fit(_engine(**overrides)), item)
        assert "cv_evaluation=True" in message and "fold_evaluation=True" in message

    def test_row_10_one_fold_refuses_at_construction(self) -> None:
        """The shared fold-policy sentence offers an in-sample fit, and this one does not."""
        message = _refused(lambda: _engine(n_folds=1), "X11 (a)")
        assert not any(route in message for route in FORBIDDEN)

    def test_a_copied_estimator_meets_the_same_rows(self) -> None:
        """The preflight asks rows 1 and 2 again, for a configuration no constructor saw."""
        estimator = _engine()
        estimator.interventions = (Static(1),)
        _refused(lambda: _fit(estimator), "X11 (c)")

    def test_a_learned_rule_that_is_not_a_learned_rule(self) -> None:
        with pytest.raises(DataError, match="LearnedRule"):
            TMLE(learned_rule="learned rule")  # type: ignore[arg-type]


class TestTheOrder:
    """A request that meets two rows raises the earlier one (the RM24 order rule)."""

    @pytest.mark.parametrize("kind", list(DATA_ROWS))
    @pytest.mark.parametrize("row", list(SCHEME_ROWS))
    def test_a_data_row_comes_before_every_setting_row(self, kind: str, row: str) -> None:
        overrides, _ = SCHEME_ROWS[row]
        message = _refused(lambda: _fit(_engine(**overrides), kind), DATA_ROWS[kind])
        assert "fold_evaluation=True" not in message

    def test_a_continuous_treatment_is_not_told_to_declare_a_shift(self) -> None:
        """``_check_shifts`` would suggest ``shifts=``, which row 2 then refuses."""
        message = _refused(lambda: _fit(_engine(), "continuous"), "F27")
        assert not any(route in message for route in FORBIDDEN)

    def test_the_contrast_row_comes_before_the_data_rows(self) -> None:
        estimator = _engine()
        estimator.estimands = ("ate",)
        _refused(lambda: _fit(estimator, "delta"), "X11 (c)")


class TestEachRemedyRunsAsWritten:
    """Every remedy that a learned-rule refusal names fits."""

    def test_the_setting_remedy_fits_at_the_engine(self) -> None:
        result = support.fit(
            cross_fit=True,
            cv_evaluation=True,
            n_folds=5,
            repeats=1,
            targeting_scheme="pooled",
            n_bootstrap=0,
        )
        assert set(result.estimates) == {support.NAME}

    def test_the_setting_remedy_fits_on_the_method(self) -> None:
        study = CausalStudy(
            support.law_frame(),
            design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
        )
        method = TMLEMethod(
            models=ModelSpec(
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
            ),
            cross_fitting=CrossFitting(enabled=True, n_folds=5, fold_evaluation=True),
        )
        result = study.identify(LearnedRuleValue()).estimate(method, simultaneous=False)
        assert set(result.estimates) == {support.NAME}

    def test_the_contrast_remedy_fits_the_value_alone(self) -> None:
        result = support.fit(estimands=None)
        assert set(result.estimates) == {support.NAME}


class TestTheSharedRefusalsNameTheLearnedRuleRemedy:
    """No shared refusal sends a learned-rule caller to an in-sample fit or a shift."""

    def test_the_arm_minimum(self) -> None:
        frame = support.law_frame(120, 3)
        frame.loc[:, "A"] = 0.0
        frame.loc[0, "A"] = 1.0
        NeverFit.calls = 0
        with pytest.raises(DataError) as caught:
            _fit_frame(_engine(), frame)
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert LEARNED_RULE_REMEDY in message
        assert not any(route in message for route in FORBIDDEN)
        # The remedy, as written: the same fit on a larger sample.
        assert np.isfinite(support.fit(support.law_frame(400, 3)).estimates[support.NAME].psi)

    def test_the_drawn_complement(self) -> None:
        """A draw whose training complement lacks an arm, found by seed, refuses after the draw."""
        frame = support.law_frame(40, 5)
        frame.loc[:, "A"] = 0.0
        frame.loc[[0, 1], "A"] = 1.0
        for seed in range(200):
            NeverFit.calls = 0
            try:
                _fit_frame(_engine(n_folds=10, random_state=seed), frame)
            except DataError as error:
                message = str(error)
                if "training complement" in message:
                    break
            except AssertionError:
                # This draw passed the preflight, and its first learner refused to fit.
                continue
        else:  # pragma: no cover - the search is deterministic
            pytest.fail("no seed drew a complement without an arm")
        assert NeverFit.calls == 0
        assert _POST_DRAW_REMEDY.format(remedy=LEARNED_RULE_REMEDY) in message
        assert not any(route in message for route in FORBIDDEN)

    def test_the_fold_backstop_receives_the_learned_rule_remedy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The backstop fires only on a fold with no trainable rows, which the preflight
        prevents, so the witness reads the remedy the fit hands to it."""
        seen: list[str] = []
        original = tmle_module.fit_nuisances

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get("remedy", ""))
            return original(*args, **kwargs)

        monkeypatch.setattr(tmle_module, "fit_nuisances", spy)
        support.fit()
        assert seen == [_POST_DRAW_REMEDY.format(remedy=LEARNED_RULE_REMEDY)]
        assert not any(route in seen[0] for route in FORBIDDEN)

    def test_the_outcome_scale(self) -> None:
        frame = support.law_frame(120, 3)
        frame["Y"] = frame["Y"] + frame["W1"]
        NeverFit.calls = 0
        with pytest.raises(CapabilityError) as caught:
            _fit_frame(_engine(), frame)
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert message.endswith(LEARNED_RULE_SCALE_REMEDY)
        assert not any(route in message for route in FORBIDDEN)
        # The remedy, as written: a declared support.
        bounds = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
        regression = make_pipeline(
            PolynomialFeatures(2, interaction_only=True, include_bias=False), LinearRegression()
        )
        result = support.fit(frame, q_bounds=bounds, outcome_learner=regression)
        assert np.isfinite(result.estimates[support.NAME].psi)


def _fit_frame(estimator: TMLE, frame: Any) -> Any:
    return estimator.fit(frame, outcome="Y", treatment="A", covariates=support.COVARIATES)


class TestTheStudyRefusesTheSameRows:
    """``CausalStudy.identify`` refuses rows 3 to 9, and ``estimate`` rows 10 to 14."""

    @staticmethod
    def _study(kind: str) -> CausalStudy:
        frame = _frame(kind)
        design: dict[str, Any] = {
            "outcome": "Y",
            "treatment": "A",
            "adjustment": support.COVARIATES,
        }
        design.update(
            {
                "continuous": {"treatment_kind": "continuous"},
                "delta": {"missingness": "Delta"},
                "intermediate": {"intermediate": "Z"},
                "weights": {"weights": "w"},
                "id": {"cluster": "cluster"},
                "strata": {"strata": ("W2",)},
            }.get(kind, {})
        )
        return CausalStudy(frame, design=PointTreatment(**design))

    @pytest.mark.parametrize("kind", list(DATA_ROWS))
    def test_identify_refuses_the_data_rows(self, kind: str) -> None:
        _refused(lambda: self._study(kind).identify(LearnedRuleValue()), DATA_ROWS[kind])

    def test_the_default_method_refuses_with_the_fold_evaluated_remedy(self) -> None:
        effect = self._study("binary").identify(LearnedRuleValue())
        message = _refused(
            lambda: effect.estimate(**never_fit_learners(), simultaneous=False), "X11 (e)"
        )
        assert "CrossFitting(enabled=True, fold_evaluation=True)" in message

    def test_the_variants_are_unavailable_and_cite_f27(self) -> None:
        effect = self._study("binary").identify(LearnedRuleValue())
        rows = {item.name: item for item in effect.available_methods()}
        assert rows["tmle"].available
        for name in ("collaborative_tmle", "drtmle"):
            assert not rows[name].available
            assert "F27" in (rows[name].reason or "")


class TestTheAssessmentRows:
    """A refit relearns the rules, so no refutation or sensitivity analysis runs.

    Each row reads unavailable, and each call refuses with the same sentence.  The
    summarize rows and ``truncation_curve`` run; the capability sweep
    (``tests/unit/test_capability_row_sweep.py``, kind ``learned_rule``) pins that.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return support.fit()

    def test_refute_reads_unavailable_and_refuses(self, result: Any) -> None:
        row = result.diagnostics.capability("refute")
        assert not row.available and row.reason == LEARNED_RULE_REFIT_REFUSAL
        with pytest.raises(CapabilityError, match="relearns the rule"):
            refute(result, estimand=support.NAME, tests=("random_common_cause",))

    @pytest.mark.parametrize(
        "operation",
        [
            "omitted_confounding",
            "robustness_value",
            "elements",
            "contour",
            "benchmark",
            "evalue",
            "simulated_confounding",
            "missingness",
            "tipping_gamma",
        ],
    )
    def test_each_sensitivity_row_reads_unavailable_and_cites_f27(
        self, result: Any, operation: str
    ) -> None:
        row = result.sensitivity.capability(operation)
        assert not row.available
        assert "F27" in (row.reason or "")

    def test_each_call_refuses_with_the_same_sentence(self, result: Any) -> None:
        for reason in (
            fit_wide_bound_refusal(result),
            fit_wide_tilt_refusal(result),
            simulated_confounding_refusal(result, support.NAME),
        ):
            assert reason is not None and LEARNED_RULE_SENSITIVITY_REFUSAL in reason
        with pytest.raises(CapabilityError, match="F27"):
            evalue(result, support.NAME)


class TestALearnedRuleIsNotARegime:
    """A ``LearnedRule`` in a set of regimes, shifts or tilts refuses by name.

    Without the check, ``interventions=`` would wrap it as a ``Static`` treatment level.
    """

    @pytest.mark.parametrize("keyword", ["interventions", "shifts", "incremental"])
    def test_the_engine_names_the_learned_rule_route(self, keyword: str) -> None:
        message = _refused(lambda: TMLE(**{keyword: [LearnedRule()]}), "LearnedRuleValue")
        assert "TMLE(learned_rule=..." in message

    def test_a_regime_mean_names_the_learned_rule_route(self) -> None:
        study = CausalStudy(
            support.law_frame(120, 3),
            design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
        )
        _refused(lambda: study.identify(RegimeMean(regimens=(LearnedRule(),))), "LearnedRuleValue")

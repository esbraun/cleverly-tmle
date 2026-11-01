"""Every learned-rule composition outside the RM30 contract refuses, in order, before a learner.

The refusal table of the learned-rule contract in
``docs/technical-reference/point-treatment-tmle.md`` lists the rows in order.  Each row raises
:class:`~cleverly.exceptions.CapabilityError` with its item before any learner is fitted, at the
engine and at ``CausalStudy``.  ``cross_fit=False`` is no row: it fits the value of the one rule
learned on every row (X11 (a), ``tests/unit/test_learned_rule_in_sample.py``).  ``n_folds=1``
under cross-fitting meets the shared fold-policy refusal, a
:class:`~cleverly.exceptions.MethodConfigurationError`.  A refusal that no setting repairs comes
before a refusal whose remedy is a setting, which the RM24 order rule requires, so the pair
witnesses below meet the earlier row.

No remedy may send the caller to a call that is refused, or to another target without saying
so.  Each remedy that a learned-rule caller reads is therefore run as written: the witness finds
the remedy's text in the message and evaluates that text.  On the engine, each shared refusal
that a learned-rule fit can meet offers the in-sample fit only as "the value of the full-sample
rule ..., a different target".  On the method, the bare offer stays for other estimands, and the
witnesses check the clause that follows it.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import (
    CausalStudy,
    CrossFitting,
    Inference,
    LearnedRuleValue,
    LongitudinalTreatment,
    PointTreatment,
    RegimeMean,
    TMLEMethod,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import (
    CapabilityError,
    ConvergenceWarning,
    DataError,
    MethodConfigurationError,
    PositivityWarning,
    RefusalReason,
)
from cleverly.interventions import Incremental, LearnedRule, Rule, Shift, Static
from cleverly.interventions.learned import (
    LEARNED_RULE_ARM_REMEDY,
    LEARNED_RULE_BENCHMARK_REFUSAL,
    LEARNED_RULE_CONFIGURATION,
    LEARNED_RULE_EVALUE_REFUSAL,
    LEARNED_RULE_IN_SAMPLE_BENCHMARK_REFUSAL,
    LEARNED_RULE_REMEDY,
    LEARNED_RULE_TILT_REFUSAL,
)
from cleverly.learners.crossfit import (
    _LEARNED_RULE_METHOD_CLAUSE,
    _POST_DRAW_REMEDY,
    LEARNED_RULE_IN_SAMPLE,
)
from cleverly.methods import ModelSpec
from cleverly.msm import MSM
from cleverly.sensitivity import evalue
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.sensitivity.missingness import fit_wide_tilt_refusal
from cleverly.sensitivity.omitted_variable import (
    benchmark,
    benchmark_refusal,
    fit_wide_bound_refusal,
)
from cleverly.validation.refute import refute_refusal
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The in-sample remedies these witnesses run use the support module's Pipeline learners, so
#: each fit warns that its interval needs condition C5a (``test_learned_rule_in_sample.py``).
pytestmark = pytest.mark.filterwarnings("ignore::cleverly.exceptions.InSampleLearnerWarning")

#: The engine settings of :data:`LEARNED_RULE_CONFIGURATION`, each as the message prints it.
ENGINE_REMEDY = (
    "cross_fit=True",
    "n_bootstrap=0",
)

#: The method remedy that ends each method-layer fold-policy sentence.  Either report fits,
#: so the requirement names no ``fold_evaluation``.
METHOD_REMEDY = "CrossFitting(enabled=True)"

#: The clause both remedies end with: the default is the stacked report, which no
#: registered study covers, so each remedy names the fold-evaluated report.
REGISTERED_PATH = "gives the fold-evaluated report that the registered studies cover"

#: The route a learned-rule message must not offer: a shift, which X11 (c) holds beside
#: ``learned_rule=``.
FORBIDDEN = ("policies=[",)

#: The spellings of an in-sample fit.  On a learned-rule fit it estimates the value of the
#: full-sample rule (X11 (a)), so a learned-rule message that offers it names that target.
IN_SAMPLE_ROUTES = ("cross_fit=False", "CrossFitting(enabled=False)")
TARGET_CHANGE = "a different target"


def _offers_no_silent_target_change(message: str) -> None:
    """Assert that ``message`` offers no shift, and no in-sample fit without its target."""
    assert not any(route in message for route in FORBIDDEN), message
    if any(route in message for route in IN_SAMPLE_ROUTES):
        assert TARGET_CHANGE in message, message
    assert "fit in sample with cross_fit=False" not in message, message


def _engine(**overrides: Any) -> TMLE:
    """A learned-rule estimator whose learners fail if a refusal comes too late."""
    return support.learned_tmle(**{**never_fit_learners(), **overrides})


def _frame(kind: str = "binary") -> Any:
    """The support law, reshaped to the data declaration a row refuses."""
    frame = support.law_frame(120, 3)
    if kind == "continuous":
        frame["A"] = frame["A"] + frame["W1"]
    if kind.startswith("multi_arm"):
        frame["A"] = np.arange(len(frame)) % 3
    if kind == "delta":
        frame["Delta"] = (np.arange(len(frame)) % 5 != 0).astype(float)
    if kind == "intermediate":
        frame["Z"] = (np.arange(len(frame)) % 2).astype(float)
    if kind == "estimated_weights":
        frame["w"] = 1.0 + 0.5 * frame["W2"]
    if kind in ("clusters", "strata_clusters"):
        frame["c"] = np.arange(len(frame)) % 20
    return frame


#: The fit arguments of each data declaration, beside the frame of :func:`_frame`.
_FIT_ARGUMENTS: dict[str, dict[str, Any]] = {
    "binary": {},
    "continuous": {"treatment_kind": "continuous"},
    "multi_arm_blip": {},
    "multi_arm_classification": {},
    "delta": {"delta": "Delta"},
    "intermediate": {"intermediate": "Z"},
    "estimated_weights": {"weights": "w", "weights_estimated": True},
    "strata_clusters": {"strata": ("W2",), "id": "c"},
    "clusters": {"id": "c"},
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


#: The data rows: the declarations that no setting repairs.
DATA_ROWS: dict[str, str] = {
    "continuous": "F27",
    "multi_arm_blip": "X11 (d2)",
    "multi_arm_classification": "X11 (d2)",
    "delta": "F27",
    "intermediate": "X11 (j)",
    "estimated_weights": "X11 (k2)",
    "strata_clusters": "X28",
}

#: The declared rule of each data row whose refusal reads the rule: a rule learner on more
#: than two arms (X11 (d2)).  Every other row uses the plug-in rule, which more than two arms
#: admit (X11 (d)).
DATA_RULES: dict[str, LearnedRule] = {
    "multi_arm_blip": LearnedRule(loss="blip", learner=LinearRegression()),
    "multi_arm_classification": LearnedRule(
        loss="weighted_classification", learner=LogisticRegression()
    ),
}


def _data_engine(kind: str, **overrides: Any) -> TMLE:
    """The engine of a data row, with the rule that row reads."""
    if kind in DATA_RULES:
        overrides = {"learned_rule": DATA_RULES[kind], **overrides}
    return _engine(**overrides)


#: The fold-policy setting rows, each with its item.  Each names the configuration that fits.
SCHEME_ROWS: dict[str, tuple[dict[str, Any], str]] = {
    "bootstrap": ({"n_bootstrap": 10}, "F27"),
}


class TestEachRowRefusesBeforeAnyLearner:
    """Every row at the engine."""

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
            ({"reference": "learned rule"}, "interventions=[Static(1)]"),
            ({"estimands": ("ate",)}, "interventions=[Static(1)]"),
            ({"estimands": ("ate_learned_rule",)}, "interventions=[Static(1)]"),
            ({"interventions": [Static(1)], "estimands": ("ey_regime",)}, "ey_known_regime"),
            ({"interventions": [Static(1)], "estimands": ("ate_regime",)}, "ate_learned_rule"),
            ({"interventions": [Static(1)], "reference": "always 1"}, "a regime contrast"),
            ({"interventions": [Static(1)], "reference": "nope"}, "reference='learned rule'"),
            ({"interventions": [Static(1, name="learned rule")]}, "LearnedRule(name=...)"),
            ({"policies": [Shift(0.5, cap=None)]}, "X11 (c2)"),
            ({"incremental": [Incremental(2.0)]}, "X11 (c2)"),
            ({"msm": MSM.linear()}, "X11 (c2)"),
            ({"interventions": [Static(1)], "msm": MSM.linear()}, "X11 (c2)"),
        ],
    )
    def test_row_2_another_axis_or_a_contrast(self, overrides: dict[str, Any], item: str) -> None:
        """Known regimes beside the rule run (X11 (c)); another axis and arm estimands refuse."""
        _refused(lambda: _engine(**overrides), item)

    def test_row_2_admits_known_regimes(self) -> None:
        """``interventions=`` and ``reference=`` naming the rule pass the configuration rows."""
        _engine(interventions=[Static(1)])
        _engine(interventions=[Static(1), Static(0)], reference="learned rule")

    @pytest.mark.parametrize("kind", list(DATA_ROWS))
    def test_the_data_rows(self, kind: str) -> None:
        _refused(lambda: _fit(_data_engine(kind), kind), DATA_ROWS[kind])

    @pytest.mark.parametrize("row", list(SCHEME_ROWS))
    def test_the_setting_rows(self, row: str) -> None:
        overrides, item = SCHEME_ROWS[row]
        message = _refused(lambda: _fit(_engine(**overrides)), item)
        assert LEARNED_RULE_CONFIGURATION in message and METHOD_REMEDY in message

    def test_the_stacked_report_reads_its_row(self) -> None:
        """``cv_evaluation=False`` fits the stacked report (X11 (e)), keyed and registered."""
        result = support.fit(cv_evaluation=False)
        estimate = result.estimates[support.NAME]
        assert result.extra["learned_rule"].scheme == "stacked"
        assert result.config.support_key.splitting == ("stacked",)
        assert estimate.support.label == "supported"
        assert estimate.support.row == "T42.rows"

    def test_the_in_sample_fit_is_no_row(self) -> None:
        """``cross_fit=False`` fits the full-sample rule's value (X11 (a)), keyed and experimental.

        The helper's outcome learner is a ``Pipeline``, which keys ``adaptive``, so T43, the
        row of parametric in-sample learners, does not cover it.
        """
        result = support.fit(cross_fit=False, cv_evaluation=False)
        assert result.extra["learned_rule"].scheme == "in_sample"
        assert result.config.support_key.splitting == ("in_sample",)
        assert dict(result.config.support_key.options)["outcome_learner_family"] == "adaptive"
        assert result.estimates[support.NAME].support.label == "experimental"

    def test_the_in_sample_bootstrap_keeps_the_in_sample_target(self) -> None:
        """The bootstrap row stays in sample, and its remedy keeps the caller's target."""
        message = _refused(
            lambda: _fit(_engine(cross_fit=False, cv_evaluation=False, n_bootstrap=10)), "F27"
        )
        assert "Fit the value of the full-sample rule with n_bootstrap=0" in message
        assert LEARNED_RULE_CONFIGURATION not in message
        settings = _as_written(message, "n_bootstrap=0")
        result = support.fit(cross_fit=False, cv_evaluation=False, **settings)
        assert result.extra["learned_rule"].scheme == "in_sample"

    def test_strata_with_clusters_refuse_in_sample(self) -> None:
        """Row 8 holds for the in-sample fit too."""
        _refused(
            lambda: _fit(_engine(cross_fit=False, cv_evaluation=False), "strata_clusters"), "X28"
        )

    def test_a_copied_estimator_meets_the_same_rows(self) -> None:
        """The preflight asks the first two rows again, for a configuration no constructor saw."""
        estimator = _engine()
        estimator.incremental = (Incremental(2.0),)
        _refused(lambda: _fit(estimator), "X11 (c2)")

    def test_a_learned_rule_that_is_not_a_learned_rule(self) -> None:
        with pytest.raises(DataError, match="LearnedRule"):
            TMLE(learned_rule="learned rule")  # type: ignore[arg-type]


class TestTheOrder:
    """A request that meets two rows raises the earlier one (the RM24 order rule)."""

    @pytest.mark.parametrize("kind", list(DATA_ROWS))
    @pytest.mark.parametrize("row", list(SCHEME_ROWS))
    def test_a_data_row_comes_before_every_setting_row(self, kind: str, row: str) -> None:
        overrides, _ = SCHEME_ROWS[row]
        message = _refused(lambda: _fit(_data_engine(kind, **overrides), kind), DATA_ROWS[kind])
        assert LEARNED_RULE_CONFIGURATION not in message

    def test_a_continuous_treatment_is_not_told_to_declare_a_shift(self) -> None:
        """``_check_shifts`` would suggest ``policies=``, which the contrast row then refuses."""
        message = _refused(lambda: _fit(_engine(), "continuous"), "F27")
        # ``policies=[`` appears once, as the route of a separate fit without ``learned_rule=``.
        assert not any(route in message for route in IN_SAMPLE_ROUTES)
        assert "A fit without learned_rule= can use TMLE(policies=[" in message

    def test_the_contrast_row_comes_before_the_data_rows(self) -> None:
        estimator = _engine()
        estimator.estimands = ("ate",)
        _refused(lambda: _fit(estimator, "delta"), "interventions=[Static(1)]")


class TestEachRemedyRunsAsWritten:
    """Every remedy that a learned-rule refusal names fits."""

    def test_the_setting_remedy_fits_at_the_engine(self) -> None:
        result = support.fit(
            cross_fit=True,
            cv_evaluation=True,
            n_folds=5,
            repeats=1,
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

    def test_the_static_arm_remedy_fits_the_contrast(self) -> None:
        """An arm estimand is told to declare the arm as ``Static``, and that call fits.

        The call runs with the constructor's defaults, ``simultaneous=True`` among them,
        because the remedy prints no other setting.
        """
        message = _refused(lambda: _engine(estimands=("ate",)), "interventions=[Static(1)]")
        assert "with estimands=None" in message
        settings = eval("dict(interventions=[Static(1)])", {"Static": Static})
        result = _default_fit(estimands=None, **settings)
        assert "ate_learned_rule[learned rule vs always 1]" in result.estimates
        assert result.simultaneous is None

    @pytest.mark.parametrize("asked", ["ey_regime", "ate_regime"])
    def test_the_regime_redirection_fits_as_written(self, asked: str) -> None:
        """``ey_regime`` and ``ate_regime`` name this fit's targets, and both spellings fit."""
        with pytest.raises(CapabilityError) as caught:
            _engine(interventions=[Static(1)], estimands=(asked,))
        assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION
        message = _refused(
            lambda: _engine(interventions=[Static(1)], estimands=(asked,)), "ey_known_regime"
        )
        assert "or estimands=None for all three." in message
        names = ["ey_learned_rule", "ey_known_regime", "ate_learned_rule"]
        assert f"estimands={names}" in message
        for estimands in (names, None):
            result = _default_fit(interventions=[Static(1)], estimands=estimands)
            assert list(result.estimates) == [
                support.NAME,
                "ey_known_regime[always 1]",
                "ate_learned_rule[learned rule vs always 1]",
            ]

    @pytest.mark.parametrize(
        ("overrides", "ending"),
        [
            (
                {"reference": "learned rule"},
                "or drop reference= to fit the learned-rule value alone.",
            ),
            (
                {"policies": [Shift(0.5, cap=None)]},
                "Fit the learned-rule value without policies=, and fit the other target in a "
                "fit of its own.",
            ),
            (
                {"interventions": [Static(1)], "reference": "always 1"},
                "TMLE(interventions=[...], reference=...), a fit of its own.",
            ),
        ],
        ids=["lone_reference", "another_axis", "known_reference"],
    )
    def test_the_contrast_remedy_names_what_was_declared(
        self, overrides: dict[str, Any], ending: str
    ) -> None:
        """Each remedy names the keyword to drop or to change, and what it then fits."""
        message = _refused(lambda: _engine(**overrides), "")
        assert message.endswith(ending), message


def _default_fit(**overrides: Any) -> Any:
    """Fit with the constructor's defaults, except the explicit learners and the folds."""
    estimator = TMLE(
        learned_rule=LearnedRule(),
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        random_state=0,
        **support.FOLD_EVALUATED,
        **overrides,
    )
    assert estimator.simultaneous
    return estimator.fit(
        support.law_frame(), outcome="Y", treatment="A", covariates=support.COVARIATES
    ).single()


def _as_written(message: str, *keywords: str) -> dict[str, Any]:
    """Return the keyword arguments that ``message`` prints, each found there verbatim."""
    settings: dict[str, Any] = {}
    for keyword in keywords:
        assert keyword in message, (keyword, message)
        settings.update(eval(f"dict({keyword})", {"LearnedRule": LearnedRule}))
    return settings


def _method_as_written(message: str) -> CrossFitting:
    """Return :data:`METHOD_REMEDY`, found verbatim at the end of ``message``, evaluated."""
    assert message.endswith(f"{_LEARNED_RULE_METHOD_CLAUSE}."), message
    assert METHOD_REMEDY in _LEARNED_RULE_METHOD_CLAUSE
    assert _LEARNED_RULE_METHOD_CLAUSE.endswith(REGISTERED_PATH)
    return eval(METHOD_REMEDY, {"CrossFitting": CrossFitting})


def _study_fit(cross_fitting: CrossFitting) -> Any:
    """Fit ``LearnedRuleValue`` in a study, under ``cross_fitting``."""
    study = CausalStudy(
        support.law_frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    method = TMLEMethod(
        models=ModelSpec(
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
        ),
        cross_fitting=cross_fitting,
    )
    return study.identify(LearnedRuleValue()).estimate(method, simultaneous=False)


def test_each_remedy_names_the_registered_path_and_it_fits_the_fold_evaluated_report() -> None:
    """The default of both layers is the stacked report, so each remedy ends by naming the
    fold-evaluated report.  The keyword each one prints, run as written, fits that report."""
    assert LEARNED_RULE_CONFIGURATION.endswith(REGISTERED_PATH)
    assert _LEARNED_RULE_METHOD_CLAUSE.endswith(REGISTERED_PATH)
    engine = _as_written(LEARNED_RULE_CONFIGURATION, "cross_fit=True", "cv_evaluation=True")
    fit = support.fit(**engine)
    assert fit.extra["learned_rule"].scheme == "fold_evaluated"
    clause = "CrossFitting(fold_evaluation=True)"
    for text in (LEARNED_RULE_CONFIGURATION, _LEARNED_RULE_METHOD_CLAUSE):
        assert clause in text
    method = eval(clause, {"CrossFitting": CrossFitting})
    assert _study_fit(method).extra["learned_rule"].scheme == "fold_evaluated"
    assert (
        _study_fit(eval(METHOD_REMEDY, {"CrossFitting": CrossFitting})).extra["learned_rule"].scheme
        == "stacked"
    )


class TestEachSharedRefusalRemedyRunsAsWritten:
    """Every shared refusal that a learned-rule caller can meet names a remedy that fits.

    One witness per case.  The engine knows the estimand, so it answers with a row of
    the RM30 table or with a shared sentence whose in-sample offer names the change of
    target.  ``CrossFitting`` refuses before it sees the estimand, so its sentence ends with
    the learned-rule clause, and the witness fits the clause's remedy.
    """

    @pytest.mark.parametrize(
        ("overrides", "cross_fitted", "in_sample"),
        [
            (
                {"cross_fit": False, "cv_evaluation": False, "repeats": 2},
                ("Enable cross-fitting", {"cross_fit": True}),
                ("repeats=1", "cross_fit=False"),
            ),
            ({"n_folds": 1}, ("Set n_folds to at least 2", {"n_folds": 2}), ("cross_fit=False",)),
        ],
        ids=["in_sample_with_repeats", "one_fold"],
    )
    def test_the_engine_shared_sentence_names_the_target_change(
        self,
        overrides: dict[str, Any],
        cross_fitted: tuple[str, dict[str, Any]],
        in_sample: tuple[str, ...],
    ) -> None:
        """The shared sentences offer a cross-fitted fit and an in-sample fit, and the in-sample
        offer names its target, the value of the full-sample rule.  Both run as written."""
        with pytest.raises(MethodConfigurationError) as caught:
            _engine(**overrides)
        message = str(caught.value)
        assert LEARNED_RULE_IN_SAMPLE in message
        _offers_no_silent_target_change(message)
        phrase, settings = cross_fitted
        assert phrase in message
        result = support.fit(**{**overrides, **settings})
        assert result.extra["learned_rule"].scheme != "in_sample"
        settings = _as_written(message, *in_sample)
        result = support.fit(**{**overrides, **settings, "cv_evaluation": False})
        assert result.extra["learned_rule"].scheme == "in_sample"

    def test_the_engine_stratified_folds_name_the_target_change(self) -> None:
        with pytest.raises(ValueError, match="stratify_folds='treatment'") as caught:
            _engine(stratify_folds="treatment")
        message = str(caught.value)
        _offers_no_silent_target_change(message)
        assert f"Otherwise {LEARNED_RULE_IN_SAMPLE}" in message
        settings = _as_written(message, "stratify_folds='none'")
        assert np.isfinite(support.fit(**settings).estimates[support.NAME].psi)
        settings = _as_written(message, "cross_fit=False")
        result = support.fit(stratify_folds="treatment", cv_evaluation=False, **settings)
        assert result.extra["learned_rule"].scheme == "in_sample"

    @pytest.mark.parametrize(
        "declared",
        [
            {"n_folds": 1, "fold_evaluation": True},
            {"enabled": False, "repeats": 2},
            {"fold_evaluation": True, "stratify_by": "treatment"},
        ],
        ids=["one_fold", "in_sample_with_repeats", "stratified"],
    )
    def test_the_method_sentence_ends_with_the_learned_rule_remedy(
        self, declared: dict[str, Any]
    ) -> None:
        with pytest.raises(MethodConfigurationError) as caught:
            CrossFitting(**declared)
        result = _study_fit(_method_as_written(str(caught.value)))
        assert np.isfinite(result.estimates[support.NAME].psi)

    def test_the_method_stratified_folds_field_remedy_fits(self) -> None:
        """The sentence also names the field to change, ``CrossFitting(stratify_by='none')``."""
        with pytest.raises(MethodConfigurationError) as caught:
            CrossFitting(fold_evaluation=True, stratify_by="treatment")
        assert "CrossFitting(stratify_by='none')" in str(caught.value)
        result = _study_fit(CrossFitting(fold_evaluation=True, stratify_by="none"))
        assert np.isfinite(result.estimates[support.NAME].psi)

    def test_a_learned_rule_target_without_learned_rule_names_it(self) -> None:
        estimator = TMLE(
            estimands=("ey_learned_rule",),
            **never_fit_learners(),
            n_folds=5,
            simultaneous=False,
            random_state=0,
        )
        NeverFit.calls = 0
        with pytest.raises(ValueError, match="ey_learned_rule") as caught:
            _fit(estimator)
        assert NeverFit.calls == 0
        settings = _as_written(
            str(caught.value), "learned_rule=LearnedRule()", "cross_fit=True", "cv_evaluation=True"
        )
        result = TMLE(
            estimands=("ey_learned_rule",),
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
            n_folds=5,
            simultaneous=False,
            random_state=0,
            **settings,
        ).fit(support.law_frame(), outcome="Y", treatment="A", covariates=support.COVARIATES)
        assert np.isfinite(result.single().estimates[support.NAME].psi)


class TestRetargetKeepsTheLearnedRuleAxis:
    """``retarget`` refuses a target of another axis, so a learned rule is never reported
    as a known regime or as an arm."""

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return support.fit()

    @pytest.mark.parametrize("estimands", [("ey_regime",), ("ate",)])
    def test_another_axis_refuses(self, result: Any, estimands: tuple[str, ...]) -> None:
        with pytest.raises(CapabilityError, match="rule learned inside each training fold"):
            result.estimator.retarget(result.data, result.nuisance, estimands=estimands)

    @pytest.mark.parametrize("asked", ["ey_known_regime", "ate_learned_rule"])
    def test_a_known_regime_target_without_known_regimes_refuses(
        self, result: Any, asked: str
    ) -> None:
        """As the constructor refuses it, rather than returning no estimate."""
        with pytest.raises(CapabilityError) as caught:
            result.estimator.retarget(result.data, result.nuisance, estimands=(asked,))
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "interventions=[Static(1)]" in str(caught.value)
        with pytest.raises(CapabilityError) as constructed:
            support.learned_tmle(estimands=(asked,))
        assert constructed.value.refusal == caught.value.refusal


def test_a_longitudinal_design_cites_x11_b() -> None:
    rng = np.random.default_rng(8080)
    frame = pd.DataFrame(
        {
            "L0": rng.normal(size=60),
            "A0": rng.binomial(1, 0.5, 60).astype(float),
            "L1": rng.normal(size=60),
            "A1": rng.binomial(1, 0.5, 60).astype(float),
            "Y": rng.binomial(1, 0.5, 60).astype(float),
        }
    )
    design = LongitudinalTreatment(
        outcome="Y", treatment=("A0", "A1"), baseline=("L0",), time_varying=((), ("L1",))
    )
    _refused(lambda: CausalStudy(frame, design=design).identify(LearnedRuleValue()), "X11 (b)")


class TestTheSharedRefusalsNameTheLearnedRuleRemedy:
    """No shared refusal sends a learned-rule caller to a shift, or to an in-sample fit without
    naming its target, the value of the full-sample rule."""

    def test_the_arm_minimum(self) -> None:
        frame = support.law_frame(120, 3)
        frame.loc[:, "A"] = 0.0
        frame.loc[0, "A"] = 1.0
        NeverFit.calls = 0
        with pytest.raises(DataError) as caught:
            support.fit(frame, **never_fit_learners())
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert message.endswith(LEARNED_RULE_ARM_REMEDY)
        # The way out that keeps the cross-fitted target comes first.
        assert "collect more observations at the rare arm, or " in message
        _offers_no_silent_target_change(message)
        # The in-sample remedy, as written. This is a reachability check, not an estimate:
        # a one-unit arm gives no sound fit under either scheme, so the fit clips every
        # unit's propensity and its fluctuation has no finite solution.
        settings = _as_written(message, "cross_fit=False")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = support.fit(frame, cv_evaluation=False, **settings)
        assert result.extra["learned_rule"].scheme == "in_sample"

    def test_the_drawn_complement(self) -> None:
        """A draw whose training complement lacks an arm, found by seed, refuses after the draw."""
        frame = support.law_frame(40, 5)
        frame.loc[:, "A"] = 0.0
        frame.loc[[0, 1], "A"] = 1.0
        for seed in range(200):
            NeverFit.calls = 0
            try:
                support.fit(frame, **never_fit_learners(), n_folds=10, random_state=seed)
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
        _offers_no_silent_target_change(message)
        # The in-sample remedy, as written. This is a reachability check, not an estimate:
        # two treated units in 40 rows clip every unit's propensity in sample too.
        settings = _as_written(message, "cross_fit=False")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = support.fit(frame, cv_evaluation=False, **settings)
        assert result.extra["learned_rule"].scheme == "in_sample"

    def test_the_outcome_scale(self) -> None:
        frame = support.law_frame(120, 3)
        frame["Y"] = frame["Y"] + frame["W1"]
        NeverFit.calls = 0
        with pytest.raises(CapabilityError) as caught:
            support.fit(frame, **never_fit_learners())
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert message.endswith(
            "Declare the known outcome support (Targeting(q_bounds=(lower, upper))). Without a "
            f"known finite support, {LEARNED_RULE_IN_SAMPLE}."
        )
        _offers_no_silent_target_change(message)
        # The in-sample remedy, as written: no support is needed in sample.
        regression = make_pipeline(
            PolynomialFeatures(2, interaction_only=True, include_bias=False), LinearRegression()
        )
        settings = _as_written(message, "cross_fit=False")
        result = support.fit(frame, cv_evaluation=False, outcome_learner=regression, **settings)
        assert result.extra["learned_rule"].scheme == "in_sample"
        # The remedy, as written: a declared support.
        bounds = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
        regression = make_pipeline(
            PolynomialFeatures(2, interaction_only=True, include_bias=False), LinearRegression()
        )
        result = support.fit(frame, q_bounds=bounds, outcome_learner=regression)
        assert np.isfinite(result.estimates[support.NAME].psi)
        assert result.extra["learned_rule"].scheme == "fold_evaluated"


class TestTheStudyRefusesTheSameRows:
    """``CausalStudy.identify`` refuses the data rows, and ``estimate`` the setting rows.

    One fold refuses earlier, in ``CrossFitting`` itself.  ``CrossFitting(enabled=False)``
    fits the full-sample rule's value (X11 (a)).
    """

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
                "estimated_weights": {"weights": "w", "weights_estimated": True},
                "strata_clusters": {"strata": ("W2",), "cluster": "c"},
            }.get(kind, {})
        )
        return CausalStudy(frame, design=PointTreatment(**design))

    @pytest.mark.parametrize("kind", list(DATA_ROWS))
    def test_identify_refuses_the_data_rows(self, kind: str) -> None:
        rule = DATA_RULES.get(kind, LearnedRule())
        _refused(lambda: self._study(kind).identify(LearnedRuleValue(rule=rule)), DATA_ROWS[kind])

    def test_the_default_method_fits_the_stacked_report(self) -> None:
        """The default ``CrossFitting()`` is the stacked report, which X11 (e) admits."""
        effect = self._study("binary").identify(LearnedRuleValue())
        result = effect.estimate(
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
            simultaneous=False,
        )
        assert result.extra["learned_rule"].scheme == "stacked"
        assert np.isfinite(result.estimates[support.NAME].psi)

    def test_the_in_sample_method_fits_the_full_sample_rule(self) -> None:
        """``CrossFitting(enabled=False)`` is no row: it fits the full-sample rule's value."""
        result = _study_fit(CrossFitting(enabled=False))
        assert result.extra["learned_rule"].scheme == "in_sample"
        assert result.estimates[support.NAME].support.label == "experimental"

    #: The setting rows on the method, each with its exception and item.  One fold refuses in
    #: ``CrossFitting`` itself, before the estimand is known, so it raises the configuration
    #: error and ends with the learned-rule clause, which names the in-sample target.
    METHOD_ROWS: ClassVar[dict[str, tuple[dict[str, Any], dict[str, Any], type, str]]] = {
        "one_fold": (
            {"n_folds": 1, "fold_evaluation": True},
            {},
            MethodConfigurationError,
            TARGET_CHANGE,
        ),
        "bootstrap": ({"fold_evaluation": True}, {"n_bootstrap": 10}, CapabilityError, "F27"),
    }

    @pytest.mark.parametrize("row", list(METHOD_ROWS))
    def test_estimate_refuses_the_setting_rows(self, row: str) -> None:
        cross_fitting, inference, error, item = self.METHOD_ROWS[row]
        effect = self._study("binary").identify(LearnedRuleValue())
        NeverFit.calls = 0
        with pytest.raises(error) as caught:
            effect.estimate(
                TMLEMethod(
                    models=ModelSpec(**never_fit_learners()),
                    cross_fitting=CrossFitting(**cross_fitting),
                    inference=Inference(**inference),
                ),
                simultaneous=False,
            )
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert item in message and METHOD_REMEDY in message, message

    def test_the_variants_are_unavailable_and_cite_f27(self) -> None:
        effect = self._study("binary").identify(LearnedRuleValue())
        rows = {item.name: item for item in effect.available_methods()}
        assert rows["tmle"].available
        for name in ("collaborative_tmle", "drtmle"):
            assert not rows[name].available
            assert "F27" in (rows[name].reason or "")


class TestTheAssessmentRows:
    """The rows of the sensitivity analyses that a learned-rule fit reads.

    The omitted-variable point bound and its robustness value, elements and contour run on the
    fit's frozen fold rules (X11 part (n)). The benchmark, the E-value and the missingness
    analyses read unavailable, and each call refuses with the same sentence. Simulated
    confounding and the other refutations run, because their replay holds the realized fold
    rules (X82 part F4).  The summarize rows and ``truncation_curve`` run; the capability sweep
    (``tests/unit/test_capability_row_sweep.py``, kind ``learned_rule``) pins that.
    """

    @pytest.fixture(scope="class")
    def result(self) -> Any:
        return support.fit()

    def test_no_refit_test_meets_a_learned_rule_stop(self, result: Any) -> None:
        """X82 part F4 holds the realized rules on every refit of the fit."""
        tests = ("placebo", "random_common_cause", "subset", "bootstrap_stability")
        assert refute_refusal(result, estimand=support.NAME, tests=tests) is None

    @pytest.mark.parametrize(
        ("operation", "item"),
        [
            ("benchmark", "X11 (n2)"),
            ("evalue", "F27"),
            ("missingness", "F27"),
            ("tipping_gamma", "F27"),
        ],
    )
    def test_each_sensitivity_row_reads_unavailable_and_cites_its_row(
        self, result: Any, operation: str, item: str
    ) -> None:
        row = result.sensitivity.capability(operation)
        assert not row.available
        assert f"{item} in docs/roadmap.md" in (row.reason or "")

    @pytest.mark.parametrize(
        "operation", ["omitted_confounding", "robustness_value", "elements", "contour"]
    )
    def test_each_point_bound_row_reads_available(self, result: Any, operation: str) -> None:
        assert result.sensitivity.capability(operation).available

    def test_simulated_confounding_has_no_learned_rule_stop(self, result: Any) -> None:
        """X82 part F4 replays the fit with its fold rules frozen.

        This direct fit records no identification, so the replay stops there instead.
        ``test_simulated_confounding_learned_rule.py`` runs the replay on a study fit.
        """
        refusal = simulated_confounding_refusal(result, support.NAME)
        assert refusal is not None
        assert "identification metadata" in refusal.message()
        assert "relearn" not in refusal.message()

    def test_each_call_refuses_with_its_row_sentence(self, result: Any) -> None:
        assert fit_wide_bound_refusal(result) is None
        assert benchmark_refusal(result, ["W2"]) is LEARNED_RULE_BENCHMARK_REFUSAL
        with pytest.raises(CapabilityError) as caught:
            benchmark(result, ["W2"], estimand=support.NAME)
        assert LEARNED_RULE_BENCHMARK_REFUSAL.condition in str(caught.value)
        reason = fit_wide_tilt_refusal(result)
        assert reason is not None and LEARNED_RULE_TILT_REFUSAL in reason.message()
        with pytest.raises(CapabilityError) as caught:
            evalue(result, support.NAME)
        assert LEARNED_RULE_EVALUE_REFUSAL in str(caught.value)

    def test_the_in_sample_benchmark_names_its_one_rule(self) -> None:
        """An in-sample fit has one rule, so the stated reason is the missing build."""
        result = support.fit(cross_fit=False, cv_evaluation=False)
        assert benchmark_refusal(result, ["W2"]) is LEARNED_RULE_IN_SAMPLE_BENCHMARK_REFUSAL
        with pytest.raises(CapabilityError) as caught:
            benchmark(result, ["W2"], estimand=support.NAME)
        message = str(caught.value)
        assert LEARNED_RULE_IN_SAMPLE_BENCHMARK_REFUSAL.condition in message
        assert "training fold" not in message
        assert "X11 (n2)" in message
        row = result.sensitivity.capability("benchmark")
        assert not row.available
        assert "training fold" not in (row.reason or "")
        # The shared target sentence of the E-value and the tilt holds for either scheme.
        for text in (LEARNED_RULE_EVALUE_REFUSAL, LEARNED_RULE_TILT_REFUSAL):
            assert "training fold" not in text


class TestALearnedRuleIsNotARegime:
    """A ``LearnedRule`` in a set of regimes, shifts or tilts refuses by name.

    Without the check, ``interventions=`` would wrap it as a ``Static`` treatment level.
    """

    @pytest.mark.parametrize("keyword", ["interventions", "policies", "incremental"])
    def test_the_engine_names_the_learned_rule_route(self, keyword: str) -> None:
        message = _refused(lambda: TMLE(**{keyword: [LearnedRule()]}), "LearnedRuleValue")
        assert "TMLE(learned_rule=..." in message

    def test_a_regime_mean_names_the_learned_rule_route(self) -> None:
        study = CausalStudy(
            support.law_frame(120, 3),
            design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
        )
        _refused(lambda: study.identify(RegimeMean(regimens=(LearnedRule(),))), "LearnedRuleValue")


class TestTheCompositionRemediesRunAsWritten:
    """The remedies of the refusals that the X11 compositions reach fit as written."""

    def _weighted(self) -> pd.DataFrame:
        return support.law_frame(200, 4).assign(w=lambda f: 1.0 + 0.5 * f["W2"])

    def test_the_estimated_weights_remedies_fit(self) -> None:
        frame = self._weighted()
        fit = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}
        message = _refused(
            lambda: _engine().fit(frame, weights="w", weights_estimated=True, **fit), "X11 (k2)"
        )
        assert "weights_estimated=False" in message
        assert "TMLE(interventions=[Rule(..., rule_kind='known')])" in message
        known = support.learned_tmle().fit(frame, weights="w", weights_estimated=False, **fit)
        assert np.isfinite(known.single().estimates[support.NAME].psi)
        rule = Rule(lambda w: (w["W1"] > 0).astype(int), name="w1", rule_kind="known")
        fixed = TMLE(
            interventions=[rule],
            outcome_learner=support.outcome_learner(),
            treatment_learner=support.treatment_learner(),
        ).fit(frame, weights="w", weights_estimated=True, **fit)
        assert np.isfinite(fixed.single().estimates["ey_regime[w1]"].psi)

    def test_one_cluster_in_a_fold_offers_the_fold_count_or_the_stacked_report(self) -> None:
        """A learned rule is offered the stacked report, as arm fits are (X11 (e)), and both fit."""
        frame = support.law_frame(120, 3).assign(c=np.arange(120) % 6)
        fit = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}
        with pytest.raises(CapabilityError) as caught:
            support.learned_tmle().fit(frame, id="c", **fit)
        message = str(caught.value)
        assert "Request at most 3 folds" in message
        assert "or use the stacked report (cv_evaluation=False" in message
        _offers_no_silent_target_change(message)
        for settings in ({"n_folds": 3}, {"cv_evaluation": False}):
            result = support.learned_tmle(**settings).fit(frame, id="c", **fit).single()
            assert np.isfinite(result.estimates[support.NAME].psi)

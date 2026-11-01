"""Every learned-rule composition outside the RM30 contract refuses, in order, before a learner.

The refusal table of the learned-rule contract in
``docs/technical-reference/point-treatment-tmle.md`` lists the rows in order.  Each row raises
:class:`~cleverly.exceptions.CapabilityError` with its item before any learner is fitted, at the engine and at ``CausalStudy``, with one
exception: ``CrossFitting(n_folds=1)`` refuses at construction, before it sees the
estimand, with a :class:`~cleverly.exceptions.MethodConfigurationError` that ends with the
learned-rule clause.  A refusal that no setting repairs comes before a refusal whose remedy
is a setting, which the RM24 order rule requires, so the pair witnesses below meet the
earlier row.

No remedy may send the caller to a call that is refused.  Each remedy that a learned-rule
caller reads is therefore run as written: the witness finds the remedy's text in the
message and evaluates that text.  On the engine, each shared refusal that a learned-rule
fit can meet is also checked for the in-sample remedy it must not name.  On the method,
that remedy stays for other estimands, and the witnesses check the clause that follows it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
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
from cleverly.exceptions import CapabilityError, DataError, MethodConfigurationError
from cleverly.interventions import Incremental, LearnedRule, Rule, Shift, Static
from cleverly.interventions.learned import (
    LEARNED_RULE_BENCHMARK_REFUSAL,
    LEARNED_RULE_CONFIGURATION,
    LEARNED_RULE_EVALUE_REFUSAL,
    LEARNED_RULE_REMEDY,
    LEARNED_RULE_TILT_REFUSAL,
)
from cleverly.learners.crossfit import _LEARNED_RULE_METHOD_CLAUSE, _POST_DRAW_REMEDY
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

#: The engine settings of :data:`LEARNED_RULE_CONFIGURATION`, each as the message prints it.
ENGINE_REMEDY = (
    "cross_fit=True",
    "cv_evaluation=True",
    "n_bootstrap=0",
)

#: The method remedy that ends each method-layer fold-policy sentence.
METHOD_REMEDY = "CrossFitting(enabled=True, fold_evaluation=True)"

#: The two routes a learned-rule message must not offer: an in-sample fit, which X11 (a)
#: refuses, and a shift, which X11 (c) holds beside ``learned_rule=``.
FORBIDDEN = ("cross_fit=False", "CrossFitting(enabled=False)", "policies=[")


def _engine(**overrides: Any) -> TMLE:
    """A learned-rule estimator whose learners fail if a refusal comes too late."""
    return support.learned_tmle(**{**never_fit_learners(), **overrides})


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
    if kind == "estimated_weights":
        frame["w"] = 1.0 + 0.5 * frame["W2"]
    if kind in ("clusters", "strata_clusters"):
        frame["c"] = np.arange(len(frame)) % 20
    return frame


#: The fit arguments of each data declaration, beside the frame of :func:`_frame`.
_FIT_ARGUMENTS: dict[str, dict[str, Any]] = {
    "binary": {},
    "continuous": {"treatment_kind": "continuous"},
    "multi_arm": {},
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
    "multi_arm": "X11 (d)",
    "delta": "F27",
    "intermediate": "X11 (j)",
    "estimated_weights": "X11 (k2)",
    "strata_clusters": "X28",
}

#: The fold-policy setting rows, each with its item.  Each names the configuration that fits.
SCHEME_ROWS: dict[str, tuple[dict[str, Any], str]] = {
    "in_sample": ({"cross_fit": False, "cv_evaluation": False}, "X11 (a)"),
    "stacked": ({"cv_evaluation": False}, "X11 (e)"),
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
            ({"interventions": [Static(1)]}, "X11 (c)"),
            ({"reference": "learned rule"}, "X11 (c)"),
            ({"estimands": ("ate",)}, "X11 (c)"),
            ({"policies": [Shift(0.5, cap=None)]}, "X11 (c)"),
            ({"incremental": [Incremental(2.0)]}, "X11 (c)"),
            ({"msm": MSM.linear()}, "X11 (c)"),
        ],
    )
    def test_row_2_another_axis_or_a_contrast(self, overrides: dict[str, Any], item: str) -> None:
        _refused(lambda: _engine(**overrides), item)

    @pytest.mark.parametrize("kind", list(DATA_ROWS))
    def test_the_data_rows(self, kind: str) -> None:
        _refused(lambda: _fit(_engine(), kind), DATA_ROWS[kind])

    @pytest.mark.parametrize("row", list(SCHEME_ROWS))
    def test_the_setting_rows(self, row: str) -> None:
        overrides, item = SCHEME_ROWS[row]
        message = _refused(lambda: _fit(_engine(**overrides)), item)
        assert "cv_evaluation=True" in message and "fold_evaluation=True" in message

    def test_one_fold_refuses_at_construction(self) -> None:
        """The shared fold-policy sentence offers an in-sample fit, and this one does not."""
        message = _refused(lambda: _engine(n_folds=1), "X11 (a)")
        assert not any(route in message for route in FORBIDDEN)

    def test_a_copied_estimator_meets_the_same_rows(self) -> None:
        """The preflight asks the first two rows again, for a configuration no constructor saw."""
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
        """``_check_shifts`` would suggest ``policies=``, which the contrast row then refuses."""
        message = _refused(lambda: _fit(_engine(), "continuous"), "F27")
        # ``policies=[`` appears once, as the route of a separate fit without ``learned_rule=``.
        assert not any(route in message for route in FORBIDDEN if route != "policies=[")
        assert "A fit without learned_rule= can use TMLE(policies=[" in message

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

    @pytest.mark.parametrize(
        ("overrides", "ending"),
        [
            ({"reference": "learned rule"}, "alone, without reference="),
            (
                {"estimands": ("ate",)},
                "alone, with estimands=None, and fit the other target in a fit of its own",
            ),
            (
                {"interventions": [Static(1)], "reference": "learned rule"},
                "alone, without interventions= or reference=, and fit the other target in a fit of its own",
            ),
        ],
        ids=["lone_reference", "arm_estimand", "regime_and_reference"],
    )
    def test_the_contrast_remedy_names_what_was_declared(
        self, overrides: dict[str, Any], ending: str
    ) -> None:
        """A lone ``reference=`` names no other target, so its remedy drops the keyword."""
        message = _refused(lambda: _engine(**overrides), "X11 (c)")
        assert message.endswith(f"Fit the learned-rule value {ending}."), message


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


class TestEachSharedRefusalRemedyRunsAsWritten:
    """Every shared refusal that a learned-rule caller can meet names a remedy that fits.

    One witness per case.  The engine knows the estimand, so it answers with a row of
    the RM30 table or with a shared sentence that offers no in-sample fit.
    ``CrossFitting`` refuses before it sees the estimand, so its sentence ends with the
    learned-rule clause, and the witness fits the clause's remedy.
    """

    @pytest.mark.parametrize(
        "overrides",
        [{"cross_fit": False, "cv_evaluation": False, "repeats": 2}, {"n_folds": 1}],
        ids=["in_sample_with_repeats", "one_fold"],
    )
    def test_the_engine_meets_the_one_fold_row_before_the_shared_sentence(
        self, overrides: dict[str, Any]
    ) -> None:
        """The shared sentences offer "set repeats=1" and an in-sample fit, which the one-fold
        row refuses, and "enable cross-fitting", which the stacked row refuses."""
        message = _refused(lambda: _engine(**overrides), "X11 (a)")
        assert LEARNED_RULE_CONFIGURATION in message
        assert not any(route in message for route in FORBIDDEN)
        settings = _as_written(message, *ENGINE_REMEDY)
        assert np.isfinite(support.fit(**settings).estimates[support.NAME].psi)

    def test_the_engine_stratified_folds_offer_no_in_sample_fit(self) -> None:
        with pytest.raises(ValueError, match="stratify_folds='treatment'") as caught:
            _engine(stratify_folds="treatment")
        message = str(caught.value)
        assert not any(route in message for route in FORBIDDEN)
        assert "in sample" not in message
        settings = _as_written(message, "stratify_folds='none'")
        assert np.isfinite(support.fit(**settings).estimates[support.NAME].psi)

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
    """No shared refusal sends a learned-rule caller to an in-sample fit or a shift."""

    def test_the_arm_minimum(self) -> None:
        frame = support.law_frame(120, 3)
        frame.loc[:, "A"] = 0.0
        frame.loc[0, "A"] = 1.0
        NeverFit.calls = 0
        with pytest.raises(DataError) as caught:
            support.fit(frame, **never_fit_learners())
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
        assert not any(route in message for route in FORBIDDEN)

    def test_the_outcome_scale(self) -> None:
        frame = support.law_frame(120, 3)
        frame["Y"] = frame["Y"] + frame["W1"]
        NeverFit.calls = 0
        with pytest.raises(CapabilityError) as caught:
            support.fit(frame, **never_fit_learners())
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert message.endswith(
            "Declare the known outcome support (Targeting(q_bounds=(lower, upper)))."
        )
        assert not any(route in message for route in FORBIDDEN)
        # The remedy, as written: a declared support.
        bounds = (float(frame["Y"].min()) - 1.0, float(frame["Y"].max()) + 1.0)
        regression = make_pipeline(
            PolynomialFeatures(2, interaction_only=True, include_bias=False), LinearRegression()
        )
        result = support.fit(frame, q_bounds=bounds, outcome_learner=regression)
        assert np.isfinite(result.estimates[support.NAME].psi)


class TestTheStudyRefusesTheSameRows:
    """``CausalStudy.identify`` refuses the data rows, and ``estimate`` the setting rows.

    One fold refuses earlier, in ``CrossFitting`` itself.
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
        _refused(lambda: self._study(kind).identify(LearnedRuleValue()), DATA_ROWS[kind])

    def test_the_default_method_refuses_with_the_fold_evaluated_remedy(self) -> None:
        effect = self._study("binary").identify(LearnedRuleValue())
        message = _refused(
            lambda: effect.estimate(**never_fit_learners(), simultaneous=False), "X11 (e)"
        )
        assert "CrossFitting(enabled=True, fold_evaluation=True)" in message

    #: The setting rows on the method, each with its exception and item.  One fold refuses in
    #: ``CrossFitting`` itself, before the estimand is known, so it raises the configuration
    #: error and ends with the learned-rule clause.
    METHOD_ROWS: ClassVar[dict[str, tuple[dict[str, Any], dict[str, Any], type, str]]] = {
        "in_sample": ({"enabled": False}, {}, CapabilityError, "X11 (a)"),
        "one_fold": (
            {"n_folds": 1, "fold_evaluation": True},
            {},
            MethodConfigurationError,
            "X11 (a)",
        ),
        "stacked": ({}, {}, CapabilityError, "X11 (e)"),
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

    def test_one_cluster_in_a_fold_offers_the_fold_count_alone(self) -> None:
        """The stacked report that arm fits are offered is refused for a learned rule."""
        frame = support.law_frame(120, 3).assign(c=np.arange(120) % 6)
        fit = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}
        with pytest.raises(CapabilityError) as caught:
            support.learned_tmle().fit(frame, id="c", **fit)
        message = str(caught.value)
        assert "Request at most 3 folds" in message
        assert "cv_evaluation=False" not in message
        assert "stacked" not in message
        assert not any(route in message for route in FORBIDDEN)
        result = support.learned_tmle(n_folds=3).fit(frame, id="c", **fit).single()
        assert np.isfinite(result.estimates[support.NAME].psi)

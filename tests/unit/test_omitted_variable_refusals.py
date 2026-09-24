r"""Where the omitted-variable bound stops, and what it says it is missing.

Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026) derive the bound under a
consistently estimated treatment mechanism and a complete outcome.  Several shipped fits
fall outside what this package implements, and each one reported a robustness value with
nothing in the report to say so: a DR-TMLE fit, a collaborative TMLE fit, a fit with a
response mechanism, a fit with an intermediate variable, and a median-combined repeated
fit whose capability rows said available.  This module pins the refusals that replaced
those numbers, and the arithmetic witnesses that say what the numbers were.

The instruments differ because the claims differ.

*   The refusals are pinned against :func:`fit_wide_bound_refusal` itself rather than
    against a respelt sentence, and every entry point is asked for the same fit, because
    the defect was one surface refusing while three others computed.
*   Witness 4 is an *inequality* on one instrument law.  It computes the number the
    C-TMLE refusal now blocks, through :func:`_elements_for`, and shows it is smaller than
    the plain fit's.  Delete the ``collaborative_tmle`` rule and the public call stops
    raising, which is the nonzero control: the test fails on the ``pytest.raises`` rather
    than on a tolerance.
*   Witness 5 and the conditional-effect witness are *exact*.  :mod:`tests.discrete_law`
    has a known :math:`\nu_0^2`, and a mechanism pinned to known wrong values makes every
    term of the Riesz identity a closed form of the law.  Each wrong mechanism moves the
    doubly robust estimate away from the plug-in, so a score that squared the fitted
    representer twice, or read the fitted mechanism where the identity needs the observed
    arm, fails by a visible margin rather than by a tolerance.
"""

from __future__ import annotations

import copy
import pickle
import re
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from cleverly import CausalStudy, NaturalCourseMean, PointTreatment
from cleverly._inference_status import NON_INFERENTIAL
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import (
    make_binary_outcome,
    make_instrument,
    make_linear_ate,
    make_missing_outcome,
    make_nonlinear_ate,
    make_shift_dose,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Shift
from cleverly.msm import MSM
from cleverly.sensitivity.omitted_variable import (
    _FIT_WIDE_BOUND_RULES,
    _PLUGIN_LIMITS_REFUSAL,
    _UNRECORDED_ESTIMATOR_REFUSAL,
    NU2_ESTIMATORS,
    OMITTED_VARIABLE_OPERATIONS,
    SensitivityBounds,
    _elements_for,
    benchmark,
    contour_data,
    fit_wide_bound_refusal,
    omitted_variable_bounds,
    resolve_parameter,
    robustness_value,
    sensitivity_elements,
)
from tests import discrete_law as law
from tests import discrete_law_mar
from tests.conftest import (
    FAST_KWARGS,
    IN_SAMPLE,
    ConstantProbability,
    OracleMissingness,
    OracleOutcome,
    OracleTreatment,
    fast_tmle,
    linear_in_sample,
)
from tests.pickles import legacy_without
from tests.unit._direct_effect_support import cde_frame, fit_cde
from tests.unit._exact_sensitivity_support import binary_oracle_fit

# --------------------------------------------------------------------------- the fits


class PinnedMechanism:
    """A wrong mechanism that varies with ``W``, duck-typed for :class:`OracleTreatment`.

    A constant mechanism makes the fitted representer ``+-2`` on every row, so the doubly
    robust estimate and the plug-in agree and a witness cannot tell them apart.  Three
    distinct wrong values, none of them the law's, break that tie.
    """

    def __init__(self, g: tuple[float, ...]) -> None:
        self.g = np.asarray(g, dtype=float)

    def propensity(self, covariates: Any) -> Any:
        return self.g[np.rint(np.asarray(covariates, dtype=float).reshape(-1)).astype(int)]


#: ``g_hat(w)`` for the three levels of :data:`tests.discrete_law.P_W`.  The law's own
#: values are ``(0.4, 0.6, 0.25)``.  These are the values of the reviewer's probe.
WRONG_MECHANISM: tuple[float, ...] = (0.5, 0.3, 0.6)


def _blocked(result: Any, nu2_estimator: str = "auto") -> Any:
    """The ``ate`` elements a refusal blocks: :func:`sensitivity_elements` minus its rule table.

    Every entry point reaches :func:`_elements_for` once :func:`fit_wide_bound_refusal`
    returns ``None``.  Called directly, it yields the number the refusal withholds, which
    is what each witness below measures.
    """
    return _elements_for(result, result.repeats[0], resolve_parameter(result, "ate"), nu2_estimator)


@pytest.fixture(scope="module")
def plain_fit() -> Any:
    """A complete-outcome arm-indexed fit, which the bound answers for."""
    frame, _ = make_linear_ate(n=400, seed=90)
    return (
        fast_tmle(**IN_SAMPLE, estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
    )


@pytest.fixture(scope="module")
def drtmle_fit() -> Any:
    """A DR-TMLE fit of the exact law at the pinned :data:`WRONG_MECHANISM`.

    One fixture answers two questions.  It is a ``drtmle`` fit, so it pins that refusal;
    and its mechanism is wrong by a known amount, so the number the refusal blocks is the
    closed form witness 5 asserts.
    """
    return (
        DRTMLE(
            outcome_learner=OracleOutcome(law.DiscreteLaw()),
            treatment_learner=OracleTreatment(PinnedMechanism(WRONG_MECHANISM)),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LinearRegression(),
            estimands=("ate",),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


@pytest.fixture(scope="module")
def instrument_pair() -> tuple[Any, Any]:
    """The plain and collaborative fits of one instrument law, in that order."""
    frame, _ = make_instrument(n=2000, seed=44)
    plain = (
        fast_tmle(**IN_SAMPLE, estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
    )
    collaborative = (
        CTMLE(**FAST_KWARGS, **IN_SAMPLE, estimands=("ate",), selection_folds=3)
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )
    return plain, collaborative


@pytest.fixture(scope="module")
def collaborative_fit(instrument_pair: tuple[Any, Any]) -> Any:
    return instrument_pair[1]


@pytest.fixture(scope="module")
def response_fit() -> Any:
    """An arm-indexed fit that models response, which the bound does not implement."""
    frame, _ = make_missing_outcome(n=500, seed=91)
    return (
        fast_tmle(**IN_SAMPLE, estimands=("ate",))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


@pytest.fixture(scope="module")
def intermediate_fit() -> Any:
    """The controlled direct effect at ``Z = 0``, which the bound does not implement."""
    return fit_cde(cde_frame(), ("ate",))[0.0]


@pytest.fixture(scope="module")
def repeated_fit() -> Any:
    """A median over two cross-fitting draws, the one refusal a refit with one split lifts."""
    frame, _ = make_binary_outcome(n=400, seed=17)
    return fast_tmle(repeats=2, estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()


@pytest.fixture(scope="module")
def ipsi_fit() -> Any:
    """A real incremental fit, whose estimand contains the mechanism."""
    frame, _ = make_nonlinear_ate(n=300, seed=0)
    return (
        TMLE(**{**FAST_KWARGS, **IN_SAMPLE}, incremental=[Incremental(1.0), Incremental(2.0)])
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@pytest.fixture(scope="module")
def shift_fit() -> Any:
    """A real modified-treatment-policy fit on a continuous dose."""
    frame, _ = make_shift_dose(n=300, seed=0)
    return (
        TMLE(**linear_in_sample(shifts=[Shift(0.0, cap=None), Shift(0.5, cap=5.0)]))
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


@pytest.fixture(scope="module")
def msm_fit() -> Any:
    """A real point-treatment working-model fit."""
    frame, _ = make_linear_ate(n=300, seed=0)
    return (
        TMLE(**{**FAST_KWARGS, **IN_SAMPLE}, msm=MSM.linear())
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@pytest.fixture(scope="module")
def natural_course_fit() -> Any:
    """The missing-outcome natural-course mean, whose tilt rows are unavailable too."""
    mar = discrete_law_mar.DiscreteLaw()
    study = CausalStudy(
        discrete_law_mar.frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"),
    )
    return study.identify(NaturalCourseMean()).estimate(
        outcome_learner=OracleOutcome(mar),
        treatment_learner=OracleTreatment(mar),
        missingness_learner=OracleMissingness(mar),
        cross_fit=False,
        simultaneous=False,
    )


#: The refused fits, with the covariate :func:`benchmark` would have dropped.  The
#: covariate is carried beside the fit because ``benchmark`` is the one entry point that
#: takes an argument the fit's own columns decide.  Every one is a real fit of its kind,
#: so each rule is reached through the fields a fit sets rather than through a tampered
#: configuration.
REFUSED_FITS: tuple[tuple[str, str], ...] = (
    ("drtmle_fit", "W"),
    ("collaborative_fit", "W1"),
    ("response_fit", "W1"),
    ("intermediate_fit", "W1"),
    ("ipsi_fit", "W1"),
    ("shift_fit", "W1"),
    ("msm_fit", "W1"),
    ("repeated_fit", "W1"),
)

#: Every public entry point that reaches :func:`sensitivity_elements`, called the way a
#: user would.  The defect was that each of the five computed its own way in; a rule that
#: only ``elements`` consulted would leave the other four reporting numbers.
ENTRY_POINTS: dict[str, Any] = {
    "elements": lambda result, covariate: sensitivity_elements(result, "ate"),
    "omitted_variable_bounds": lambda result, covariate: omitted_variable_bounds(result, "ate"),
    "robustness_value": lambda result, covariate: robustness_value(result, "ate"),
    "contour_data": lambda result, covariate: contour_data(result, "ate", grid_size=3),
    "benchmark": lambda result, covariate: benchmark(result, [covariate], estimand="ate"),
}


class TestOneRefusalReachesEveryEntryPoint:
    """The defect: five entry points, each with its own way into the elements.

    ``benchmark`` matters most here. It refits the whole model, so a refusal it reached
    only through its own second call would have paid for a fit before saying no.
    """

    @pytest.mark.parametrize(("fixture", "covariate"), REFUSED_FITS)
    @pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
    def test_every_entry_point_raises_the_rule_tables_own_reason(
        self, request: pytest.FixtureRequest, fixture: str, covariate: str, entry: str
    ) -> None:
        result = request.getfixturevalue(fixture)
        with pytest.raises(CapabilityError) as refusal:
            ENTRY_POINTS[entry](result, covariate)
        assert str(refusal.value) == fit_wide_bound_refusal(result)

    @pytest.mark.parametrize(("fixture", "covariate"), REFUSED_FITS)
    def test_the_five_declared_rows_carry_that_same_reason(
        self, request: pytest.FixtureRequest, fixture: str, covariate: str
    ) -> None:
        result = request.getfixturevalue(fixture)
        expected = fit_wide_bound_refusal(result)
        for operation in OMITTED_VARIABLE_OPERATIONS:
            row = result.sensitivity.capability(operation)
            assert not row.available, operation
            assert row.status is AssessmentStatus.UNAVAILABLE, operation
            assert row.reason == expected, operation

    def test_an_unrefused_fit_still_answers_every_entry_point(self, plain_fit: Any) -> None:
        """The nonzero control. A rule table that refused everything would pass above."""
        assert fit_wide_bound_refusal(plain_fit) is None
        assert sensitivity_elements(plain_fit, "ate").nu2 > 0
        assert robustness_value(plain_fit, "ate")["rv"] > 0
        assert omitted_variable_bounds(plain_fit, "ate").max_bias > 0
        assert len(contour_data(plain_fit, "ate", grid_size=3)) == 9
        for operation in OMITTED_VARIABLE_OPERATIONS:
            row = plain_fit.sensitivity.capability(operation)
            assert row.available and row.reason is None, operation


class TestEachReasonNamesTheMissingResult:
    """Correction 2. "Not implemented" alone tells a reader nothing about the science."""

    @pytest.mark.parametrize(
        ("fixture", "phrase"),
        [
            ("drtmle_fit", "no nu^2 estimate for a 'drtmle' fit"),
            ("collaborative_fit", "no nu^2 estimate for a 'collaborative_tmle' fit"),
        ],
    )
    def test_the_reason_names_the_result_the_package_does_not_have(
        self, request: pytest.FixtureRequest, fixture: str, phrase: str
    ) -> None:
        reason = fit_wide_bound_refusal(request.getfixturevalue(fixture))
        assert reason is not None
        assert phrase in reason
        assert "not implemented" not in reason

    @pytest.mark.parametrize(
        ("fixture", "missing"),
        [
            (
                "response_fit",
                (
                    "omits the response indicator Delta",
                    "E[Delta (Y - Qbar)^2]",
                    "joint strength over the treatment and response mechanisms",
                ),
            ),
            (
                "intermediate_fit",
                (
                    "1{Z = z} / P(Z = z | A, W)",
                    "joint strength over the treatment and intermediate mechanisms",
                ),
            ),
        ],
    )
    def test_a_well_posed_mechanism_refusal_names_the_theorem_and_what_is_missing(
        self, request: pytest.FixtureRequest, fixture: str, missing: tuple[str, ...]
    ) -> None:
        """The bound exists for these fits, so the reason is an implementation gap.

        Each functional is linear in an outcome regression, which is the hypothesis of
        Theorem 2. What the package lacks is named piece by piece.
        """
        reason = fit_wide_bound_refusal(request.getfixturevalue(fixture))
        assert reason is not None
        assert "is not implemented for a fit with" in reason
        assert "Theorem 2" in reason
        assert "well posed" in reason
        assert "treatment-side strength" in reason
        for piece in missing:
            assert piece in reason, piece

    def test_the_collaborative_reason_names_the_conditioning_set(
        self, collaborative_fit: Any
    ) -> None:
        """``E[alpha_W | A]`` holds only for an empty selection; the set is part of it."""
        reason = fit_wide_bound_refusal(collaborative_fit)
        assert reason is not None
        assert "E[alpha_W | A, V]" in reason
        assert "W_S" in reason
        # The representer is that average only where the working mechanism is P(A | V).
        assert "Where the working mechanism is P(A | V) in the limit" in reason

    @pytest.mark.parametrize(
        ("fixture", "axis", "phrase"),
        [
            ("ipsi_fit", "ipsi", "part of the estimand"),
            ("shift_fit", "shift", "A modified-policy mean"),
            ("msm_fit", "msm", "A point-treatment MSM coefficient"),
        ],
    )
    def test_a_real_fit_on_each_non_arm_axis_hears_its_own_reason(
        self, request: pytest.FixtureRequest, fixture: str, axis: str, phrase: str
    ) -> None:
        """The axis is the one the fit records, not one written into its configuration."""
        result = request.getfixturevalue(fixture)
        assert result.config.parameter_axis == axis
        reason = fit_wide_bound_refusal(result)
        assert reason is not None
        assert f"indexed by {axis!r}" in reason
        assert phrase in reason
        assert ("well posed" in reason) is (axis != "ipsi")

    @pytest.mark.parametrize("fixture", ["drtmle_fit", "collaborative_fit"])
    def test_a_mechanism_refusal_names_the_assumption_it_lacks(
        self, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        reason = fit_wide_bound_refusal(request.getfixturevalue(fixture))
        assert reason is not None
        assert "does not assume a consistent treatment mechanism" in reason
        assert "standard error" in reason

    def test_the_response_refusal_points_at_the_tilt_that_does_answer(
        self, response_fit: Any
    ) -> None:
        reason = fit_wide_bound_refusal(response_fit)
        assert reason is not None
        assert "sensitivity.missingness()" in reason
        assert "sensitivity.tipping_gamma()" in reason
        # The pointer is only worth printing if the row it names is open.
        for operation in ("missingness", "tipping_gamma"):
            assert response_fit.sensitivity.capability(operation).available, operation

    def test_a_natural_course_fit_is_not_sent_to_a_second_refusal(
        self, natural_course_fit: Any
    ) -> None:
        """The pointer is gated, because both tilt rows are unavailable here.

        A natural-course fit has a response mechanism, so it hears the response refusal.
        It has no arm-specific tilt, so ``missingness`` and ``tipping_gamma`` are refused
        in turn. Printing the pointer would answer a refusal with a refusal.
        """
        reason = fit_wide_bound_refusal(natural_course_fit)
        assert reason is not None
        assert "response mechanism" in reason
        assert "sensitivity.missingness()" not in reason
        for operation in ("missingness", "tipping_gamma"):
            assert not natural_course_fit.sensitivity.capability(operation).available, operation


class TestTheRuleTableIsOrderedAndEveryRuleIsReachable:
    """The names are the introspection contract, so the order is pinned without a message.

    ``longitudinal`` first is not a preference. ``LongitudinalData`` declares no
    ``has_missing_outcome``, so the response rule would raise ``AttributeError`` on a
    longitudinal result instead of refusing it, and the message a user reads would be a
    traceback.
    """

    def test_the_declared_order_is_the_one_the_rules_assume(self) -> None:
        assert [name for name, _ in _FIT_WIDE_BOUND_RULES] == [
            "longitudinal",
            "drtmle",
            "collaborative_tmle",
            "response_mechanism",
            "intermediate",
            "parameter_axis",
            "repeats",
        ]

    def test_a_fit_refused_twice_hears_the_rule_a_refit_cannot_lift(self, drtmle_fit: Any) -> None:
        """Why ``repeats`` is last.

        A repeated DR-TMLE fit breaks two rules. Told to fit one split, its reader would
        refit and meet the DR-TMLE refusal next, so the table reports that one first.
        """
        repeated = replace(drtmle_fit, repeats=drtmle_fit.repeats * 2)
        rules = dict(_FIT_WIDE_BOUND_RULES)
        assert rules["repeats"](repeated) is not None
        assert fit_wide_bound_refusal(repeated) == rules["drtmle"](repeated)

    def test_the_longitudinal_reason_is_the_literal_the_facade_published(self) -> None:
        rules = dict(_FIT_WIDE_BOUND_RULES)

        class Elsewhere:
            assessment_family = "longitudinal"

        assert rules["longitudinal"](Elsewhere()) == (
            "no longitudinal sensitivity derivation is registered"
        )

    def test_a_response_rule_placed_after_longitudinal_would_have_raised(self) -> None:
        """Why the order is a contract rather than a preference.

        A longitudinal result answers no ``has_missing_outcome``, so this is what the
        second rule would meet if the first were moved.
        """
        rules = dict(_FIT_WIDE_BOUND_RULES)

        class Elsewhere:
            assessment_family = "longitudinal"
            data = object()

        with pytest.raises(AttributeError):
            rules["response_mechanism"](Elsewhere())
        assert fit_wide_bound_refusal(Elsewhere()) is not None

    @pytest.mark.parametrize(
        ("rule", "fixture", "tamper"),
        [
            ("drtmle", "plain_fit", {"fitted_method": "drtmle"}),
            ("collaborative_tmle", "plain_fit", {"fitted_method": "collaborative_tmle"}),
            ("response_mechanism", "response_fit", {}),
            ("intermediate", "intermediate_fit", {}),
            ("parameter_axis", "plain_fit", {"parameter_axis": "regime"}),
            ("repeats", "repeated_fit", {}),
        ],
    )
    def test_each_rule_is_the_one_a_matching_fit_reaches(
        self,
        request: pytest.FixtureRequest,
        rule: str,
        fixture: str,
        tamper: dict[str, str],
    ) -> None:
        result = request.getfixturevalue(fixture)
        if "parameter_axis" in tamper:
            result = replace(result, config=replace(result.config, **tamper))
        elif tamper:
            result = replace(result, **tamper)
        own = dict(_FIT_WIDE_BOUND_RULES)[rule](result)
        assert own is not None
        assert fit_wide_bound_refusal(result) == own


class TestAnArmIndexedFitWithNoLinearParameter:
    """Conflict (a): the old message contradicted itself on a fit that has arms.

    A fit reporting only ratios is indexed by arms. It heard that "a fit whose
    counterfactuals are not arms does not have" a Riesz representer, which is a statement
    about a different fit.
    """

    @pytest.fixture(scope="class")
    def ratio_only_fit(self) -> Any:
        frame, _ = make_binary_outcome(n=400, seed=92)
        return (
            fast_tmle(**IN_SAMPLE, estimands=("rr", "or"))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    def test_it_is_refused_as_arm_indexed_and_reporting_no_linear_estimand(
        self, ratio_only_fit: Any
    ) -> None:
        assert ratio_only_fit.config.parameter_axis == "arm"
        with pytest.raises(CapabilityError) as refusal:
            sensitivity_elements(ratio_only_fit, "ate")
        message = str(refusal.value)
        assert "arm-indexed fit reported none of them" in message
        assert "counterfactuals are not arms" not in message

    def test_a_ratio_request_keeps_the_pointer_to_the_evalue(self, ratio_only_fit: Any) -> None:
        with pytest.raises(CapabilityError, match=r"sensitivity\.evalue\(\)"):
            sensitivity_elements(ratio_only_fit, "rr")

    def test_an_attributable_fraction_is_refused_without_that_pointer(self, plain_fit: Any) -> None:
        """Correction 7. ``evalue`` has no attributable-fraction branch to send anyone to.

        The pointer belongs to the two ratio scales and to nothing else, so offering it
        here sent the reader from one refusal to a second one. The refusal reads the
        estimand name alone, so the plain fit serves. The public facade and the resolver
        must give the same sentence.
        """
        with pytest.raises(CapabilityError) as refusal:
            resolve_parameter(plain_fit, "paf")
        message = str(refusal.value)
        assert "linear functional of the outcome regression" in message
        assert "evalue" not in message
        with pytest.raises(CapabilityError) as public:
            plain_fit.sensitivity.omitted_confounding("paf")
        assert str(public.value) == message


class TestANonpositiveRieszSecondMoment:
    """Correction 4. The old code substituted the plug-in value and said nothing.

    ``# pragma: no cover`` marked the branch unreachable, so nothing failed when a
    mechanism this wrong reached it. The substitution is not a smaller estimate of the
    same thing: the plug-in squares the same fitted representer, so it reports the second
    moment of the wrong function under the name of the right one.
    """

    @pytest.fixture(scope="class")
    def hopeless_mechanism_fit(self) -> Any:
        frame, _ = make_linear_ate(n=400, seed=94)
        return (
            # P(A = 1 | W) = 0.01 on every row. nu^2 is nu_0^2 minus the squared error of
            # the fitted representer, so a mechanism this wrong drives it below zero with no
            # monkeypatching. The g_bounds are wide enough that truncation does not rescue it.
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=ConstantProbability(0.01),
                g_bounds=(0.001, 0.999),
                cross_fit=False,
                estimands=("ate",),
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    def test_the_doubly_robust_estimator_refuses_rather_than_substituting(
        self, hopeless_mechanism_fit: Any
    ) -> None:
        with pytest.raises(CapabilityError) as refusal:
            sensitivity_elements(hopeless_mechanism_fit, "ate")
        message = str(refusal.value)
        assert "'doubly_robust' estimator of nu^2" in message
        assert "nu2_estimator='auto' resolved to 'doubly_robust'" in message

    def test_the_plugin_value_it_no_longer_returns_was_there_all_along(
        self, hopeless_mechanism_fit: Any
    ) -> None:
        """The control that makes the refusal a choice rather than a failure.

        The plug-in estimate exists and is positive on this fit, which is exactly what the
        old branch fell back to. Asserting it here says the refusal declines a value it
        could have returned.
        """
        fallback = sensitivity_elements(hopeless_mechanism_fit, "ate", nu2_estimator="plugin")
        assert fallback.nu2 > 0
        assert fallback.nu2_estimator == "plugin"

    def test_it_is_a_capability_error_so_a_battery_keeps_running(
        self, hopeless_mechanism_fit: Any
    ) -> None:
        """``run_all`` catches ``CapabilityError`` alone. A ``ValueError`` aborts it."""
        report = hopeless_mechanism_fit.sensitivity.run_all()
        assert report["robustness_value"].status == "unavailable"
        assert "declined this request" in report["robustness_value"].detail
        # The rows after it still ran, which is what a ``ValueError`` would have cost.
        assert {item.name for item in report.items} == {
            row.operation for row in hopeless_mechanism_fit.sensitivity.capabilities
        }
        hopeless_mechanism_fit.assessment_cache.clear()

    def test_an_unknown_name_is_still_a_plain_value_error(self, plain_fit: Any) -> None:
        with pytest.raises(ValueError, match="nu2_estimator must be one of"):
            sensitivity_elements(plain_fit, "ate", nu2_estimator="riesz")

    @pytest.mark.parametrize("fixture", ["drtmle_fit", "repeated_fit"])
    def test_an_unknown_name_is_reported_before_any_refusal(
        self, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        """The argument is checked first, so a refused fit does not hide a typo.

        ``CapabilityError`` subclasses ``ValueError``, so the type is compared exactly.
        """
        with pytest.raises(ValueError, match="nu2_estimator must be one of") as raised:
            sensitivity_elements(request.getfixturevalue(fixture), "ate", nu2_estimator="riesz")
        assert type(raised.value) is ValueError


class TestTheDocumentedEstimatorNames:
    """Correction 8. The docstrings listed ``"analytic"`` and ``"riesz"``, which raise."""

    #: Every site that documents the argument.  Two of them are unrendered helpers, and
    #: they are checked anyway: the defect was three sites drifting apart, and a site that
    #: Sphinx never reads drifts as easily as one it does.
    DOCUMENTED = (
        sensitivity_elements,
        omitted_variable_bounds,
        benchmark,
        robustness_value,
        contour_data,
    )

    @pytest.mark.parametrize("name", NU2_ESTIMATORS)
    def test_every_documented_value_is_accepted(self, plain_fit: Any, name: str) -> None:
        elements = sensitivity_elements(plain_fit, "ate", nu2_estimator=name)
        assert elements.nu2 > 0
        assert elements.nu2_estimator == ("doubly_robust" if name == "auto" else name)

    def test_every_docstring_lists_exactly_the_accepted_values(self) -> None:
        pattern = re.compile(r"nu2_estimator : \{([^}]*)\}")
        for function in self.DOCUMENTED:
            match = pattern.search(function.__doc__ or "")
            assert match is not None, function.__name__
            listed = tuple(part.strip().strip('"') for part in match.group(1).split(","))
            assert listed == NU2_ESTIMATORS, function.__name__


class TestTheCollaborativeSecondMomentIsSmallerByConstruction:
    r"""Witness 4. The number the C-TMLE refusal blocks, and why it is optimistic.

    The selected working mechanism is near-constant here, so its representer is close to
    :math:`E[\alpha_W \mid A]`. By Jensen's inequality the second moment of a conditional
    mean is no larger than the second moment of what it averages, so :math:`\nu^2` falls
    and the robustness value rises. Nothing about the estimand changed.

    The comparison needs :func:`_elements_for` because the public call now refuses, which
    is the whole point: the number exists, it is computable, and it is not a bound.
    """

    def test_the_public_call_refuses_and_the_blocked_number_is_the_smaller_one(
        self, instrument_pair: tuple[Any, Any]
    ) -> None:
        plain, collaborative = instrument_pair
        with pytest.raises(CapabilityError):
            sensitivity_elements(collaborative, "ate")
        blocked = _blocked(collaborative)
        full = sensitivity_elements(plain, "ate")
        assert blocked.nu2 < full.nu2
        # The residual outcome variance is the same regression's on both fits, so the
        # whole difference is the representer's -- which is the claim.
        assert blocked.sigma2 == pytest.approx(full.sigma2, rel=0.1)

    def test_the_smaller_moment_is_the_one_that_reads_as_more_robust(
        self, instrument_pair: tuple[Any, Any]
    ) -> None:
        """The consequence a reader would have acted on, stated as a number."""
        plain, collaborative = instrument_pair
        blocked = _blocked(collaborative)
        assert blocked.max_bias < sensitivity_elements(plain, "ate").max_bias


#: ``P(W = w)``, ``g_0(w)`` and the pinned ``g_hat(w)`` as the realised sample has them,
#: for the longhand below.
_P_W = law.PROBS.sum(axis=(1, 2))
_G0 = law.G_EXACT
_G_HAT = np.asarray(WRONG_MECHANISM)


class TestTheDoublyRobustShortfallOnAKnownLaw:
    r"""Witness 5, exact. :math:`\nu_0^2` is a closed form of twelve cell probabilities.

    With the pinned :math:`\hat g` the fitted ATE representer is
    :math:`\hat\alpha = A / \hat g - (1 - A) / (1 - \hat g)`, a known wrong function of
    the truth :math:`\alpha_0 = A / g_0 - (1 - A) / (1 - g_0)`. The Riesz identity says
    the doubly robust estimator equals :math:`\nu_0^2 - E[(\hat\alpha - \alpha_0)^2]`.
    Both terms are written here from the law's constants, with no library code, and the
    plug-in :math:`E[\hat\alpha^2]` is a third, different closed form.
    """

    #: :math:`\nu_0^2 = E[1/g_0 + 1/(1 - g_0)]`, written out from the law's own constants.
    NU2_TRUTH = float(np.sum(_P_W * (1.0 / _G0 + 1.0 / (1.0 - _G0))))

    #: :math:`E[(\hat\alpha - \alpha_0)^2]`, one term per arm, since the indicators are
    #: disjoint.
    SHORTFALL = float(
        np.sum(
            _P_W
            * (
                _G0 * (1.0 / _G_HAT - 1.0 / _G0) ** 2
                + (1.0 - _G0) * (1.0 / (1.0 - _G_HAT) - 1.0 / (1.0 - _G0)) ** 2
            )
        )
    )

    #: :math:`E[\hat\alpha^2] = E[g_0 / \hat g^2 + (1 - g_0) / (1 - \hat g)^2]`.
    PLUGIN = float(np.sum(_P_W * (_G0 / _G_HAT**2 + (1.0 - _G0) / (1.0 - _G_HAT) ** 2)))

    def test_the_law_supplies_the_truth_this_witness_needs(self) -> None:
        """The reviewer's probe read 3.202522675737 and 5.321286848073 off this law."""
        truth, shortfall, plugin = self.NU2_TRUTH, self.SHORTFALL, self.PLUGIN
        assert truth == pytest.approx(4.4, abs=1e-12)
        assert truth - shortfall == pytest.approx(3.202522675737, abs=1e-11)
        assert plugin == pytest.approx(5.321286848073, abs=1e-11)

    def test_the_blocked_estimate_falls_short_of_the_truth_by_that_amount(
        self, drtmle_fit: Any
    ) -> None:
        with pytest.raises(CapabilityError):
            sensitivity_elements(drtmle_fit, "ate")
        blocked = _blocked(drtmle_fit)
        assert blocked.nu2_estimator == "doubly_robust"
        assert blocked.nu2 < self.NU2_TRUTH
        assert blocked.nu2 == pytest.approx(self.NU2_TRUTH - self.SHORTFALL, abs=1e-12)

    def test_the_plugin_estimate_is_a_different_wrong_number(self, drtmle_fit: Any) -> None:
        """Which is why the old fallback was not a repair, and why this witness is not blind.

        The plug-in squares the same fitted ``alpha``. Here it overshoots the truth, 5.32
        against 4.4, while the doubly robust value falls short. A doubly robust branch
        that returned the plug-in would fail the test above by more than two.
        """
        plugin = _blocked(drtmle_fit, "plugin")
        robust = _blocked(drtmle_fit)
        assert plugin.nu2 == pytest.approx(self.PLUGIN, abs=1e-12)
        assert plugin.nu2 - robust.nu2 > 2.0


class TestTheConditionalEffectScoreReadsTheObservedArm:
    r"""The ATT and ATC doubly robust :math:`\nu^2`, exact on the known law.

    The ATT's own score weights the contrast by :math:`1\{A = c\} / P(A = c)`. With
    :math:`\hat g = 1/2` the fitted representer is :math:`(1\{A = c\} - 1\{A \ne c\}) / s`,
    with :math:`s = P(A = c)`, so the estimator is :math:`4/s - 1/s^2` and the plug-in is
    :math:`1/s^2`. The Riesz identity gives the same number as
    :math:`\nu_0^2 - E[(\hat\alpha - \alpha_0)^2]`, and both are written here from the
    law.

    The control: a score that read the fitted :math:`\hat g_c / s` in place of the
    observed arm returned :math:`1/s^2`. For the ATT that is 5.4083 against
    :math:`\nu_0^2 = 4.5971`, above the truth, which the identity forbids. For the ATC it
    is 3.0779 against the closed form 3.9397. Both differ from the closed form by more
    than 0.8.
    """

    @pytest.fixture(scope="class")
    def half_mechanism_fit(self) -> Any:
        return (
            TMLE(
                outcome_learner=OracleOutcome(law.DiscreteLaw()),
                # 1/2 on every row, which the law's three propensities never equal, so the
                # fitted representer is a known wrong function of the truth.
                treatment_learner=ConstantProbability(0.5),
                estimands=("att", "atc"),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
            )
            .fit(law.frame(), outcome="Y", treatment="A", covariates=["W"])
            .single()
        )

    @staticmethod
    def _longhand(estimand: str) -> tuple[float, float, float]:
        """``(s, nu_0^2, E[(alpha_hat - alpha_0)^2])`` for the arm ``estimand`` conditions on."""
        g_c = _G0 if estimand == "att" else 1.0 - _G0
        share = float(np.sum(_P_W * g_c))
        odds = g_c / (1.0 - g_c)
        # alpha_0 is 1/s on the conditioning arm and -odds/s on the other; alpha_hat is
        # 1/s and -1/s. Squares and differences only, so the sign convention drops out.
        # The truth is the law's own oracle, which the witness of the curve differentiates.
        truth = float(law.riesz_second_moment(law.PROBS, estimand))
        error = float(np.sum(_P_W * (1.0 - g_c) * (odds - 1.0) ** 2)) / share**2
        return share, truth, error

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_the_estimate_is_the_riesz_identity_evaluated_longhand(
        self, half_mechanism_fit: Any, estimand: str
    ) -> None:
        share, truth, error = self._longhand(estimand)
        elements = sensitivity_elements(half_mechanism_fit, estimand)
        assert elements.nu2_estimator == "doubly_robust"
        assert elements.nu2 == pytest.approx(truth - error, abs=1e-12)
        assert elements.nu2 == pytest.approx(4.0 / share - 1.0 / share**2, abs=1e-12)
        assert elements.nu2 <= truth

    @pytest.mark.parametrize("estimand", ["att", "atc"])
    def test_the_plugin_is_the_value_the_fitted_score_returned(
        self, half_mechanism_fit: Any, estimand: str
    ) -> None:
        share, _, _ = self._longhand(estimand)
        plugin = sensitivity_elements(half_mechanism_fit, estimand, nu2_estimator="plugin")
        robust = sensitivity_elements(half_mechanism_fit, estimand)
        assert plugin.nu2 == pytest.approx(1.0 / share**2, abs=1e-12)
        assert abs(plugin.nu2 - robust.nu2) > 0.8


class TestTheIntermediateRuleIsNotVacuous:
    """The number the intermediate rule blocks, and why ``cf_d`` would misread it.

    Without the rule the bound returns a value. Its representer is zero on every row at
    the other level of ``Z`` and carries the intermediate weight on the rest, so the
    confounder strength it scales is not a treatment-side strength.
    """

    def test_the_representer_vanishes_off_the_level_and_not_on_it(
        self, intermediate_fit: Any
    ) -> None:
        with pytest.raises(CapabilityError):
            sensitivity_elements(intermediate_fit, "ate")
        blocked = _blocked(intermediate_fit)
        level = intermediate_fit.data.intermediate == intermediate_fit.intermediate_value
        assert 0 < int(level.sum()) < level.size
        assert np.all(blocked.riesz_representer[~level] == 0.0)
        assert np.all(blocked.riesz_representer[level] != 0.0)
        assert np.isfinite(blocked.nu2) and blocked.nu2 > 0


class TestTheResponseRepresenterOmitsTheIndicator:
    """The response reason says the implemented representer omits ``Delta``. It does."""

    def test_the_representer_is_nonzero_on_rows_with_no_outcome(self, response_fit: Any) -> None:
        blocked = _blocked(response_fit, "plugin")
        unobserved = ~response_fit.data.observed
        assert unobserved.any()
        assert np.count_nonzero(blocked.riesz_representer[unobserved]) > 0


# ------------------------------------------------------- the plug-in limits (RM22, F26)

#: The six accessors that read the bound's curve, which refuse under the plug-in.
LIMIT_ACCESSORS: tuple[str, ...] = (
    "ci_lower",
    "ci_upper",
    "robustness_value_ci",
    "plugin_interval_lower",
    "plugin_interval_upper",
    "robustness_value_plugin_interval",
)

#: The strength of the RM22 contract, at which each bound is read.
STRONG = {"cf_y": 0.5, "cf_d": 0.3, "rho": 1.0}


def _refusal_text(accessor: str, estimator: str, reason: str) -> str:
    """The exact message a guarded accessor raises."""
    return (
        f"SensitivityBounds.{accessor} is not defined under nu2_estimator={estimator!r}: "
        + reason
        + "."
    )


def _assert_the_limits_refuse(bound: Any, estimator: str, reason: str) -> None:
    for accessor in LIMIT_ACCESSORS:
        with pytest.raises(CapabilityError) as refused:
            getattr(bound, accessor)
        assert str(refused.value) == _refusal_text(accessor, estimator, reason)


class TestThePluginLimitsRefuse:
    """No derivation gives the standard error of a bound built on the plug-in nu^2.

    The plug-in keeps its point quantities: the bounds, ``max_bias`` and ``rv``.  The six
    accessors that read the curve refuse with the F26 reason, the mapping and the summary
    say so, and the elements carry no curve.  The doubly robust bound on the same fit
    reports finite limits, which is what makes each refusal withhold a number.
    """

    @pytest.fixture(scope="class")
    def exact_fit(self) -> Any:
        return binary_oracle_fit()[0]

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
    def test_each_limit_accessor_refuses(self, exact_fit: Any, estimand: str) -> None:
        bound = omitted_variable_bounds(exact_fit, estimand, nu2_estimator="plugin", **STRONG)
        assert bound.nu2_estimator == "plugin"
        _assert_the_limits_refuse(bound, "plugin", _PLUGIN_LIMITS_REFUSAL)

    def test_a_fitted_mechanism_refuses_too(self, plain_fit: Any) -> None:
        """The exact law's plug-in is exact; a learned mechanism is the case F26 is about."""
        bound = omitted_variable_bounds(plain_fit, "ate", nu2_estimator="plugin")
        _assert_the_limits_refuse(bound, "plugin", _PLUGIN_LIMITS_REFUSAL)

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
    def test_the_doubly_robust_bound_reports_finite_limits(
        self, exact_fit: Any, estimand: str
    ) -> None:
        """The nonzero witness: the refusal withholds a number the other estimator gives."""
        bound = omitted_variable_bounds(exact_fit, estimand, **STRONG)
        assert bound.nu2_estimator == "doubly_robust"
        assert np.isfinite(bound.ci_lower) and bound.ci_lower < bound.lower
        assert np.isfinite(bound.ci_upper) and bound.ci_upper > bound.upper
        assert 0.0 <= bound.robustness_value_ci <= bound.robustness_value
        assert bound.plugin_interval_lower == bound.ci_lower

    @pytest.mark.parametrize("estimand", ["ate", "att", "atc"])
    def test_the_point_quantities_remain(self, exact_fit: Any, estimand: str) -> None:
        """The plug-in bounds are the formula at the plug-in nu^2, written here by hand."""
        elements = sensitivity_elements(exact_fit, estimand, nu2_estimator="plugin")
        bound = omitted_variable_bounds(exact_fit, estimand, nu2_estimator="plugin", **STRONG)
        strength = np.sqrt(STRONG["cf_y"] * STRONG["cf_d"] / (1.0 - STRONG["cf_d"]))
        max_bias = np.sqrt(elements.sigma2 * elements.nu2)
        psi = exact_fit.psi(estimand)
        assert bound.max_bias == pytest.approx(max_bias, abs=1e-12)
        assert bound.lower == pytest.approx(psi - strength * max_bias, abs=1e-12)
        assert bound.upper == pytest.approx(psi + strength * max_bias, abs=1e-12)
        at_rv = omitted_variable_bounds(
            exact_fit,
            estimand,
            cf_y=bound.robustness_value,
            cf_d=bound.robustness_value,
            nu2_estimator="plugin",
        )
        assert min(abs(at_rv.lower), abs(at_rv.upper)) < 1e-4

    def test_the_mapping_names_the_estimator_and_omits_the_limits(self, exact_fit: Any) -> None:
        plugin = omitted_variable_bounds(exact_fit, "att", nu2_estimator="plugin").to_dict()
        robust = omitted_variable_bounds(exact_fit, "att").to_dict()
        assert plugin["nu2_estimator"] == "plugin"
        assert not set(plugin) & set(LIMIT_ACCESSORS)
        assert list(plugin) == [
            "estimand",
            "nu2_estimator",
            "psi",
            "cf_y",
            "cf_d",
            "rho",
            "max_bias",
            "bias",
            "lower",
            "upper",
            "robustness_value",
        ]
        # The doubly robust mapping keeps its keys and their order.
        assert list(robust) == [
            "estimand",
            "psi",
            "cf_y",
            "cf_d",
            "rho",
            "max_bias",
            "bias",
            "lower",
            "upper",
            "ci_lower",
            "ci_upper",
            "robustness_value",
            "robustness_value_ci",
        ]

    def test_the_summary_says_why(self, exact_fit: Any) -> None:
        plugin = omitted_variable_bounds(exact_fit, "att", nu2_estimator="plugin").summary()
        robust = omitted_variable_bounds(exact_fit, "att").summary()
        assert "F26" in plugin and "plug-in nu^2" in plugin
        assert "one-sided CIs" not in plugin and "RVa  =" not in plugin
        assert "robustness value RV   =" in plugin
        assert "one-sided CIs" in robust and "RVa  =" in robust and "F26" not in robust

    def test_the_robustness_value_reports_no_limit_value(self, exact_fit: Any) -> None:
        plugin = robustness_value(exact_fit, "att", nu2_estimator="plugin")
        robust = robustness_value(exact_fit, "att")
        assert list(plugin) == ["nu2_estimator", "rv", "max_bias"]
        assert plugin["nu2_estimator"] == "plugin"
        assert not any("rva" in key for key in plugin)
        assert list(robust) == ["rv", "rva", "max_bias"]
        assert plugin["rv"] == pytest.approx(robust["rv"], abs=1e-6)

    def test_the_elements_carry_no_curve(self, exact_fit: Any) -> None:
        plugin = sensitivity_elements(exact_fit, "att", nu2_estimator="plugin")
        robust = sensitivity_elements(exact_fit, "att")
        assert plugin.psi_nu2 is None and plugin.psi_max_bias is None
        assert np.asarray(plugin.psi_sigma2).shape == (law.N,)
        assert robust.psi_nu2 is not None and robust.psi_max_bias is not None

    def test_the_status_refuses_first_and_the_estimator_second(self, exact_fit: Any) -> None:
        """A non-inferential status answers the inferential names; F26 answers the others."""
        bound = copy.copy(omitted_variable_bounds(exact_fit, "att", nu2_estimator="plugin"))
        status = next(iter(NON_INFERENTIAL))
        object.__setattr__(bound, "inference", status)
        with pytest.raises(CapabilityError) as by_status:
            _ = bound.ci_lower
        assert "F26" not in str(by_status.value)
        with pytest.raises(CapabilityError) as by_estimator:
            _ = bound.plugin_interval_lower
        assert str(by_estimator.value) == _refusal_text(
            "plugin_interval_lower", "plugin", _PLUGIN_LIMITS_REFUSAL
        )
        assert bound.to_dict()["inference"] == status

    def test_a_bound_saved_before_rm22_refuses_its_limits(self, exact_fit: Any) -> None:
        """An older pickle may hold plug-in limits or ATT limits without the share term."""
        current = omitted_variable_bounds(exact_fit, "att", **STRONG)
        legacy = legacy_without(current, "nu2_estimator")
        assert legacy.nu2_estimator == "unrecorded"
        _assert_the_limits_refuse(legacy, "unrecorded", _UNRECORDED_ESTIMATOR_REFUSAL)
        assert legacy.to_dict()["nu2_estimator"] == "unrecorded"
        assert "saved before" in legacy.summary()
        assert legacy.lower == current.lower
        restored = pickle.loads(pickle.dumps(current))
        assert restored.nu2_estimator == "doubly_robust"
        assert restored.ci_lower == current.ci_lower

    def test_a_guard_that_refuses_nothing_fails_the_witness(
        self, exact_fit: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation control: without the guard the plug-in limits read as numbers."""
        bound = omitted_variable_bounds(exact_fit, "att", nu2_estimator="plugin")
        monkeypatch.setattr(
            SensitivityBounds, "_limit_refusal", lambda self, operation, diagnostic=False: None
        )
        with pytest.raises(pytest.fail.Exception):
            _assert_the_limits_refuse(bound, "plugin", _PLUGIN_LIMITS_REFUSAL)

    def test_the_assessment_row_names_the_stop(self, exact_fit: Any) -> None:
        report = replace(exact_fit).sensitivity.run_all(
            arguments={"robustness_value": {"estimand": "att", "nu2_estimator": "plugin"}}
        )
        item = report["robustness_value"]
        assert item.status is AssessmentStatus.COMPLETED
        assert "no confidence-limit value under the plug-in nu^2" in item.detail
        assert "F26" in item.detail

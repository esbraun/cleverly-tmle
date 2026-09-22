r"""Where the omitted-variable bound stops, and what it says it is missing.

Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022) derive the bound under a
consistently estimated treatment mechanism and a complete outcome.  Three shipped fits
break one of those two premises, and each broke it silently: a DR-TMLE fit, a
collaborative TMLE fit, and a fit with a response mechanism all reported a robustness
value with nothing in the report to say which.  This module pins the refusals that
replaced those numbers, and the two arithmetic witnesses that say what the numbers were.

The instruments differ because the claims differ.

*   The refusals are pinned against :func:`fit_wide_bound_refusal` itself rather than
    against a respelt sentence, and every entry point is asked for the same fit, because
    the defect was one surface refusing while three others computed.
*   Witness 4 is an *inequality* on one instrument law.  It computes the number the
    C-TMLE refusal now blocks, through :func:`_elements_for`, and shows it is smaller than
    the plain fit's.  Delete the ``collaborative_tmle`` rule and the public call stops
    raising, which is the nonzero control: the test fails on the ``pytest.raises`` rather
    than on a tolerance.
*   Witness 5 is *exact*.  :mod:`tests.discrete_law` has a known :math:`\nu_0^2`, and a
    mechanism pinned to one half everywhere makes every quantity in the Riesz identity a
    closed-form rational number.  There is no sampling error to absorb a sign error.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_instrument, make_linear_ate, make_missing_outcome
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity.omitted_variable import (
    _FIT_WIDE_BOUND_RULES,
    NU2_ESTIMATORS,
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
from tests.conftest import FAST_KWARGS, IN_SAMPLE, OracleOutcome, fast_tmle

# --------------------------------------------------------------------------- the fits


class ConstantHalf(BaseEstimator):
    """``P(A = 1 | W) = 1/2`` on every row, which :mod:`tests.discrete_law` does not satisfy.

    The wrongness is the point.  The law's propensity takes three values, none of them a
    half, so the fitted representer is a *known* wrong function of the truth and every
    term of the Riesz identity is exact arithmetic rather than an estimate.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantHalf:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        half = np.full(np.asarray(X, dtype=float).shape[0], 0.5)
        return np.column_stack([half, half])


class ConstantHundredth(BaseEstimator):
    """``P(A = 1 | W) = 0.01`` on every row, far enough out to drive ``nu^2`` negative.

    ``E[2 m(alpha_hat) - alpha_hat^2]`` is ``nu_0^2`` minus the squared error of the
    fitted representer, so a mechanism this wrong sends it below zero without any
    monkeypatching of the estimator.  The fit declares ``g_bounds`` wide enough that the
    truncation does not rescue the value before the estimator sees it.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ConstantHundredth:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        p = np.full(np.asarray(X, dtype=float).shape[0], 0.01)
        return np.column_stack([1.0 - p, p])


@pytest.fixture(scope="module")
def plain_fit() -> Any:
    """A complete-outcome arm-indexed fit, which the bound answers for."""
    frame, _ = make_linear_ate(n=400, seed=90)
    return (
        fast_tmle(**IN_SAMPLE, estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
    )


@pytest.fixture(scope="module")
def drtmle_fit() -> Any:
    """A DR-TMLE fit of the exact law at a mechanism pinned to one half.

    One fixture answers two questions.  It is a ``drtmle`` fit, so it pins that refusal;
    and its mechanism is wrong by a known amount, so the number the refusal blocks is the
    closed form witness 5 asserts.
    """
    return (
        DRTMLE(
            outcome_learner=OracleOutcome(law.DiscreteLaw()),
            treatment_learner=ConstantHalf(),
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
    """An arm-indexed fit that models response, which the bound is not derived for."""
    frame, _ = make_missing_outcome(n=500, seed=91)
    return (
        fast_tmle(**IN_SAMPLE, estimands=("ate",))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


@pytest.fixture(scope="module")
def natural_course_fit() -> Any:
    """The missing-outcome natural-course mean, whose tilt rows are unavailable too."""
    from cleverly import CausalStudy, NaturalCourseMean, PointTreatment
    from tests import discrete_law_mar
    from tests.conftest import OracleMissingness, OracleTreatment

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


#: The three refused fits, with the covariate :func:`benchmark` would have dropped.  The
#: covariate is carried beside the fit because ``benchmark`` is the one entry point that
#: takes an argument the fit's own columns decide.
REFUSED_FITS: tuple[tuple[str, str], ...] = (
    ("drtmle_fit", "W"),
    ("collaborative_fit", "W1"),
    ("response_fit", "W1"),
)

#: Every public entry point that reaches :func:`sensitivity_elements`, called the way a
#: user would.  The defect was that each of the four computed its own way in; a rule that
#: only ``elements`` consulted would leave the other three reporting numbers.
ENTRY_POINTS: dict[str, Any] = {
    "elements": lambda result, covariate: sensitivity_elements(result, "ate"),
    "omitted_variable_bounds": lambda result, covariate: omitted_variable_bounds(result, "ate"),
    "robustness_value": lambda result, covariate: robustness_value(result, "ate"),
    "contour_data": lambda result, covariate: contour_data(result, "ate", grid_size=3),
    "benchmark": lambda result, covariate: benchmark(result, [covariate], estimand="ate"),
}

#: The four rows the bound's own refusal fills, plus the benchmark, which shares it on
#: every fit but a longitudinal one.
DECLARED_ROWS: tuple[str, ...] = (
    "omitted_confounding",
    "robustness_value",
    "elements",
    "contour",
    "benchmark",
)


class TestOneRefusalReachesEveryEntryPoint:
    """The defect: four entry points, each with its own way into the elements.

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
        for operation in DECLARED_ROWS:
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
        for operation in DECLARED_ROWS:
            row = plain_fit.sensitivity.capability(operation)
            assert row.available and row.reason is None, operation


class TestEachReasonNamesTheMissingResult:
    """Correction 2. "Not implemented" alone tells a reader nothing about the science."""

    @pytest.mark.parametrize(
        ("fixture", "phrase"),
        [
            ("drtmle_fit", "no nu^2 estimate for a 'drtmle' fit"),
            ("collaborative_fit", "no nu^2 estimate for a 'collaborative_tmle' fit"),
            ("response_fit", "is not derived for a fit with a response mechanism"),
        ],
    )
    def test_the_reason_names_the_result_the_package_does_not_have(
        self, request: pytest.FixtureRequest, fixture: str, phrase: str
    ) -> None:
        reason = fit_wide_bound_refusal(request.getfixturevalue(fixture))
        assert reason is not None
        assert phrase in reason
        assert "not implemented" not in reason

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
            "parameter_axis",
        ]

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
            ("parameter_axis", "plain_fit", {"parameter_axis": "regime"}),
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
        from cleverly.datasets import make_binary_outcome

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

    def test_an_attributable_fraction_is_refused_without_that_pointer(self) -> None:
        """Correction 7. ``evalue`` has no attributable-fraction branch to send anyone to."""
        frame, _ = make_linear_ate(n=200, seed=93)
        result = (
            fast_tmle(**IN_SAMPLE, estimands=("ate",))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        with pytest.raises(CapabilityError) as refusal:
            resolve_parameter(result, "paf")
        message = str(refusal.value)
        assert "linear functional of the outcome regression" in message
        assert "evalue" not in message


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
            TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=ConstantHundredth(),
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
        blocked = _elements_for(
            collaborative,
            collaborative.repeats[0],
            resolve_parameter(collaborative, "ate"),
            "auto",
        )
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
        blocked = _elements_for(
            collaborative,
            collaborative.repeats[0],
            resolve_parameter(collaborative, "ate"),
            "auto",
        )
        assert blocked.max_bias < sensitivity_elements(plain, "ate").max_bias


class TestTheDoublyRobustShortfallOnAKnownLaw:
    r"""Witness 5, exact. :math:`\nu_0^2` is a closed form of twelve cell probabilities.

    With :math:`\hat g = 1/2` everywhere the representer is :math:`\pm 2`, so
    :math:`E[\hat\alpha^2] = 4` and :math:`m(W, \hat\alpha) = 4`. The doubly robust
    estimator is therefore :math:`2 \times 4 - 4 = 4`, and the Riesz identity says that is
    :math:`\nu_0^2 - E[(\hat\alpha - \alpha_0)^2]`. Both sides are computed here from the
    law alone, and neither is a tolerance.
    """

    #: :math:`\nu_0^2 = E[1/g_0 + 1/(1 - g_0)]`, written out from the law's own constants.
    NU2_TRUTH = float(np.sum(law.P_W * (1.0 / law.G + 1.0 / (1.0 - law.G))))

    #: :math:`E[(\hat\alpha - \alpha_0)^2]` at :math:`\hat g = 1/2`, longhand. The cross
    #: term is ``E[2 + 2] = 4`` because the two indicator halves are disjoint.
    SHORTFALL = NU2_TRUTH + 4.0 - 2.0 * 4.0

    def test_the_law_supplies_the_truth_this_witness_needs(self) -> None:
        truth, shortfall = self.NU2_TRUTH, self.SHORTFALL
        assert truth == pytest.approx(4.4, abs=1e-12)
        assert shortfall == pytest.approx(0.4, abs=1e-12)

    def test_the_blocked_estimate_falls_short_of_the_truth_by_that_amount(
        self, drtmle_fit: Any
    ) -> None:
        with pytest.raises(CapabilityError):
            sensitivity_elements(drtmle_fit, "ate")
        blocked = _elements_for(
            drtmle_fit, drtmle_fit.repeats[0], resolve_parameter(drtmle_fit, "ate"), "auto"
        )
        assert blocked.nu2_estimator == "doubly_robust"
        assert blocked.nu2 == pytest.approx(4.0, abs=1e-12)
        assert blocked.nu2 < self.NU2_TRUTH
        assert self.NU2_TRUTH - blocked.nu2 == pytest.approx(self.SHORTFALL, abs=1e-12)

    def test_the_plugin_estimate_is_the_same_wrong_representers_second_moment(
        self, drtmle_fit: Any
    ) -> None:
        """Which is why the old fallback was not a repair.

        Both estimators square the same fitted ``alpha``. Here they agree exactly at 4,
        and the truth is 4.4: the plug-in carries the identical shortfall.
        """
        blocked = _elements_for(
            drtmle_fit, drtmle_fit.repeats[0], resolve_parameter(drtmle_fit, "ate"), "plugin"
        )
        assert blocked.nu2 == pytest.approx(4.0, abs=1e-12)

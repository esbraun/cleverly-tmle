"""The E-value configurations of X82 part G.

Part G lifts five E-value refusals.  Each class below pins one configuration with a nonzero
witness and, where a sign, a mask, or a square root can be wrong, a mutation that fails it.

*   :class:`TestThePerArmBound` checks the controlled-direct-effect bound on a finite law,
    without a fit.  :class:`TestAControlledDirectEffect` checks that every branch on a fit
    with an intermediate variable inverts the product of the two per-arm factors.
*   :class:`TestTheDerivedRatioOnEveryMethod` checks the retargeted risk ratio of C-TMLE and
    DR-TMLE against the same fit that requests the ratio.
*   :class:`TestAStratumContrast` checks that a stratum-conditional contrast reads its own rows.
*   :class:`TestTheMissingAtRandomScale` checks the plug-in standard deviation of an outcome
    that is missing at random.
*   :class:`TestATimeToEventRiskRatio` checks the risk ratio of two static regimens.
*   :class:`TestTheLimitReading` checks that every part G confidence limit asks the registry,
    and prints as descriptive until a registered row covers it.
"""

from __future__ import annotations

import importlib
import re
import typing
import warnings
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, PointTreatment, RegimeMean, RiskRatio, TimeToEvent
from cleverly._summary import number_text
from cleverly._summary.vocabulary import NOTE_TEXTS
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_point_survival, make_shift_dose
from cleverly.estimators import CTMLE, DRTMLE, TMLE, direct_effect
from cleverly.estimators.direct_effect import LEVELS, declares_intermediate
from cleverly.exceptions import CapabilityError, PositivityWarning
from cleverly.interventions import Shift
from cleverly.learners import ParametricMechanism, ParametricOutcome
from cleverly.sensitivity._outcome_moments import missing_at_random_sd
from cleverly.sensitivity.evalue import (
    _INCOMPLETE_INTERMEDIATE_REFUSAL,
    _PER_ARM_CONDITIONS,
    _SMD_TO_LOG_RR,
    _TIME_TO_EVENT_CONDITIONS,
    EVALUE_LIMIT_INSTRUMENT,
    _Branch,
    _EValueRefusal,
    evalue_from_rr,
)
from cleverly.validation import _interpretation as interpretation
from cleverly.validation._interpretation import (
    DESCRIPTIVE_SENTENCE,
    CalibrationScope,
    VerdictEvidence,
)
from tests.conftest import SELECTOR_CONFIGS, linear_drtmle, linear_in_sample
from tests.unit._direct_effect_support import (
    COVARIATES,
    binary_cde_frame,
    cde_frame,
    column_only,
    fit_cde,
    fit_without_intermediate,
    level_only,
)

pytestmark = pytest.mark.xdist_group("evalue_part_g")

#: Imported by name, because ``cleverly.sensitivity`` rebinds ``evalue`` to the function.
evalue_module = importlib.import_module("cleverly.sensitivity.evalue")
derived_module = importlib.import_module("cleverly.sensitivity._derived")


def bounding_factor(treatment: float, outcome: float) -> float:
    """Ding and VanderWeele's joint bounding factor of two risk-ratio associations."""
    return treatment * outcome / (treatment + outcome - 1.0)


class FiniteLaw(NamedTuple):
    """A binary latent ``U`` that confounds both arms at one level ``z`` of the intermediate.

    The outcome risk depends on ``U`` only, so every controlled direct effect is 1 and the
    observed ratio is pure bias.
    """

    prior: float
    treated: float
    control: float
    risk_high: float
    risk_low: float

    def arm_risk(self, share: float) -> float:
        return share * self.risk_high + (1.0 - share) * self.risk_low

    @property
    def bias(self) -> float:
        """The observed CDE ratio, which is all bias because the true one is 1."""
        return self.arm_risk(self.treated) / self.arm_risk(self.control)

    @property
    def associations(self) -> tuple[float, float, float]:
        """The cell ratio of each arm and the outcome ratio, as Proposition A.2 reads them.

        The treated arm's factor is observed over counterfactual risk, so its cell ratio is
        ``sup_u P(u | a, z) / P(u)``.  The control arm's factor is the inverse, so its cell
        ratio is ``sup_u P(u) / P(u | a', z)``.
        """
        treated = max(self.treated / self.prior, (1 - self.treated) / (1 - self.prior))
        control = max(self.prior / self.control, (1 - self.prior) / (1 - self.control))
        return treated, control, self.risk_high / self.risk_low

    @property
    def per_arm_factors(self) -> tuple[float, float]:
        treated, control, outcome = self.associations
        return bounding_factor(treated, outcome), bounding_factor(control, outcome)


#: The plan's witness law.  Each per-arm factor alone is below the bias, so a bound that
#: keeps one factor misses it.
SPLIT_LAW = FiniteLaw(prior=0.5, treated=0.9, control=0.1, risk_high=0.9, risk_low=0.1)
#: A law that explains its ratio away with every association at most 9.  The single-factor
#: E-value of its ratio is 13.6, so an inversion without the square root overstates robustness.
SKEWED_LAW = FiniteLaw(prior=0.18, treated=0.9, control=0.02, risk_high=0.18, risk_low=0.02)


class TestThePerArmBound:
    """The bias of a CDE ratio is a product of two per-arm factors, bounded one at a time."""

    @pytest.mark.parametrize("law", [SPLIT_LAW, SKEWED_LAW], ids=["split", "skewed"])
    def test_the_product_bounds_the_bias(self, law: FiniteLaw) -> None:
        treated, control = law.per_arm_factors
        assert law.bias > 1.0
        assert law.bias <= treated * control * (1 + 1e-12)

    def test_one_factor_is_not_enough(self) -> None:
        """Nonzero witness: each factor alone, or the treated one twice, is below the bias."""
        treated, control = SPLIT_LAW.per_arm_factors
        assert SPLIT_LAW.bias == pytest.approx(41 / 9, rel=1e-12)
        assert treated == pytest.approx(1.6531, abs=1e-4)
        assert control == pytest.approx(45 / 13, rel=1e-12)
        assert max(treated, control) < SPLIT_LAW.bias
        assert treated**2 < SPLIT_LAW.bias

    @pytest.mark.parametrize("ratio", [1.0, 1.3, 4.0, 41 / 9, 25.0])
    def test_equal_strengths_invert_the_square_root(self, ratio: float) -> None:
        strength = evalue_from_rr(float(np.sqrt(ratio)))
        assert bounding_factor(strength, strength) ** 2 == pytest.approx(ratio, rel=1e-12)

    def test_a_law_that_explains_the_ratio_away_has_an_association_at_the_evalue(self) -> None:
        """The per-arm E-value is consistent with the law, and one without the root is not."""
        strongest = max(SKEWED_LAW.associations)
        assert strongest == pytest.approx(9.0, rel=1e-12)
        assert evalue_from_rr(float(np.sqrt(SKEWED_LAW.bias))) <= strongest
        # Mutation: the single-factor inversion asks for more than the law needed.
        assert evalue_from_rr(SKEWED_LAW.bias) > strongest


#: Each law: its frame and the estimands its fit reports.  ``odds`` reports no ``rr``, so its
#: default request derives one.
LAWS: dict[str, tuple[Any, tuple[str, ...]]] = {
    "gaussian": (cde_frame, ("ate", "att", "atc", "ey1", "ey0")),
    "ratio": (binary_cde_frame, ("ate", "rr", "or", "ey1", "ey0")),
    "odds": (binary_cde_frame, ("ate", "or", "ey1", "ey0")),
}


def cv_evaluated(result: Any) -> Any:
    """The fit read as CV-evaluated, so the exact retarget refuses and the fallbacks answer.

    The edit refits nothing, so the result is a routing witness only.
    """
    return replace(result, config=replace(result.config, cv_evaluation=True))


class Request(NamedTuple):
    """One E-value request and the branch it takes."""

    name: str
    law: str
    edit: Any
    estimand: str | None
    branch: str


def as_fitted(result: Any) -> Any:
    return result


REQUESTS = (
    Request("gaussian-default", "gaussian", as_fitted, None, "gaussian_difference"),
    Request("gaussian-ate", "gaussian", as_fitted, "ate", "gaussian_difference"),
    Request("gaussian-att", "gaussian", as_fitted, "att", "gaussian_difference"),
    Request("gaussian-atc", "gaussian", as_fitted, "atc", "gaussian_difference"),
    Request("ratio-default", "ratio", as_fitted, None, "reported_rr"),
    Request("ratio-or", "ratio", as_fitted, "or", "reported_or"),
    Request("ratio-ate", "ratio", as_fitted, "ate", "derived_rr"),
    Request("odds-default", "odds", as_fitted, None, "derived_rr"),
    Request("odds-cv-default", "odds", cv_evaluated, None, "reported_or"),
    Request("odds-cv-ate", "odds", cv_evaluated, "ate", "fixed_baseline_ate"),
)


def ids(case: Request) -> str:
    return case.name


@pytest.fixture(scope="module")
def cde_fits() -> dict[str, Any]:
    """Each law fitted with ``intermediate="Z"``, keyed by law and then by level."""
    return {law: fit_cde(frame(), estimands) for law, (frame, estimands) in LAWS.items()}


@pytest.fixture(scope="module")
def plain_fits() -> dict[str, Any]:
    """Each law fitted without the intermediate variable."""
    return {
        law: fit_without_intermediate(frame(), estimands)
        for law, (frame, estimands) in LAWS.items()
    }


class TestAControlledDirectEffect:
    """Each branch on a CDE fit converts to the risk-ratio scale, then takes the square root."""

    def test_the_requests_reach_every_point_branch(self) -> None:
        point_branches = set(typing.get_args(_Branch)) - {"time_to_event_rr"}
        assert {case.branch for case in REQUESTS} == point_branches

    @pytest.mark.parametrize("case", REQUESTS, ids=ids)
    @pytest.mark.parametrize("level", LEVELS)
    def test_the_request_inverts_the_per_arm_product(
        self, cde_fits: dict[str, Any], level: float, case: Request
    ) -> None:
        result = case.edit(cde_fits[case.law][level])
        selection = evalue_module._select_evalue(result, case.estimand)
        assert selection.branch == case.branch
        assert selection.features[0] == "cde"
        row = replace(result).sensitivity.capability("evalue")
        if case.estimand is None:
            assert row.available
        report = evalue_module.evalue(replace(result), case.estimand)
        assert report.bias_factor == "per_arm_product"
        rr = report.risk_ratio
        assert report.point == pytest.approx(evalue_from_rr(float(np.sqrt(rr))), rel=1e-12)
        # Mutation: the single-factor inversion gives a larger number on every request.
        assert evalue_from_rr(rr) > report.point
        assert _PER_ARM_CONDITIONS in report.note
        assert ("per_arm", _PER_ARM_CONDITIONS) in report.note_parts
        summary = report.summary()
        assert (
            summary.note("sensitivity_assumption").text == NOTE_TEXTS["assumption_evalue_per_arm"]
        )
        assert summary.note("per_arm_conditions").text == NOTE_TEXTS["per_arm_conditions"]
        assumptions = report.summary(detail="full").section("Assumptions").lines
        assert _PER_ARM_CONDITIONS in assumptions
        # The reading names four associations, and the formula inverts the square root.
        point = number_text(report.point, 2)
        assert summary.reading.startswith(
            NOTE_TEXTS["verdict_evalue_per_arm"].split("{point}")[0] + point + " in each of four"
        )
        assert assumptions[0].startswith("E-value = sqrt(RR) + ")

    def test_the_limit_keeps_its_side_of_the_null(self, cde_fits: dict[str, Any]) -> None:
        report = evalue_module.evalue(cde_fits["ratio"][0.0])
        low, high = report.risk_ratio_ci
        limit = low if report.risk_ratio >= 1.0 else high
        expected = evalue_module._evalue_for_limit(
            float(np.sqrt(limit)), above_null=report.risk_ratio >= 1.0
        )
        assert report.limit == pytest.approx(expected, rel=1e-12)

    def test_an_odds_ratio_takes_two_square_roots(self, cde_fits: dict[str, Any]) -> None:
        """The common-outcome conversion, then the per-arm root: OR^(1/4)."""
        result = cde_fits["ratio"][0.0]
        report = evalue_module.evalue(result, "or")
        odds = result["or"].psi
        assert report.risk_ratio == pytest.approx(np.sqrt(odds), rel=1e-12)
        assert report.point == pytest.approx(evalue_from_rr(odds**0.25), rel=1e-12)
        # Mutation: one root, the conversion alone, is the single-factor number.
        assert evalue_from_rr(odds**0.5) > report.point

    def test_a_gaussian_contrast_converts_before_the_root(self, cde_fits: dict[str, Any]) -> None:
        result = cde_fits["gaussian"][0.0]
        report = evalue_module.evalue(result, "ate")
        sd = float(np.std(result.data.outcome, ddof=1))
        rr = np.exp(_SMD_TO_LOG_RR * result["ate"].psi / sd)
        assert report.risk_ratio == pytest.approx(rr, rel=1e-12)
        assert report.point == pytest.approx(evalue_from_rr(float(np.sqrt(rr))), rel=1e-12)

    @pytest.mark.parametrize("level", LEVELS)
    def test_the_derived_ratio_passes_the_fitted_level(
        self, cde_fits: dict[str, Any], level: float
    ) -> None:
        """The retarget of the ``odds`` fit equals the ratio the ``ratio`` fit reports."""
        derived = evalue_module.evalue(cde_fits["odds"][level])
        reported = cde_fits["ratio"][level]["rr"]
        assert derived.risk_ratio == pytest.approx(reported.psi, rel=1e-10)
        assert derived.risk_ratio_ci == pytest.approx(reported.ci, rel=1e-10)

    def test_the_two_levels_differ(self, cde_fits: dict[str, Any]) -> None:
        """Nonzero witness for the level: a retarget at the wrong level would move the ratio."""
        low, high = (evalue_module.evalue(cde_fits["odds"][level]) for level in LEVELS)
        assert abs(low.risk_ratio - high.risk_ratio) > 1e-3

    @pytest.mark.parametrize("case", REQUESTS, ids=ids)
    def test_a_fit_without_an_intermediate_keeps_one_factor(
        self, plain_fits: dict[str, Any], case: Request
    ) -> None:
        result = case.edit(plain_fits[case.law])
        assert not declares_intermediate(result)
        report = evalue_module.evalue(replace(result), case.estimand)
        assert report.bias_factor == "single"
        assert report.point == pytest.approx(evalue_from_rr(report.risk_ratio), rel=1e-12)
        assert report.limit_reading is None
        assert not report.limit_descriptive

    @pytest.mark.parametrize("edit", [level_only, column_only], ids=["level", "column"])
    def test_one_half_of_an_intermediate_refuses(
        self, plain_fits: dict[str, Any], edit: Any
    ) -> None:
        result = edit(plain_fits["ratio"])
        assert declares_intermediate(result)
        for estimand in (None, "ate"):
            with pytest.raises(CapabilityError) as caught:
                evalue_module.evalue(replace(result), estimand)
            assert str(caught.value) == _INCOMPLETE_INTERMEDIATE_REFUSAL.message()

    def test_a_continuous_dose_with_an_intermediate_is_not_applicable(self) -> None:
        frame, _ = make_shift_dose(n=400, seed=3)
        frame["Z"] = np.random.default_rng(0).integers(0, 2, len(frame))
        with warnings.catch_warnings():
            # The uncapped shift extrapolates on a few rows. The refusal reads no mechanism.
            warnings.simplefilter("ignore", PositivityWarning)
            result = fit_cde(frame, policies=[Shift(0.5, cap=None)])[0.0]
        row = result.sensitivity.capability("evalue")
        assert row.status is AssessmentStatus.NOT_APPLICABLE

    def test_the_assessment_reports_the_per_arm_reading(self, cde_fits: dict[str, Any]) -> None:
        item = replace(cde_fits["ratio"][0.0]).assess().sensitivity["evalue"]
        assert item.status is AssessmentStatus.COMPLETED
        assert "per arm" in item.detail
        assert "descriptive" in item.detail

    @pytest.mark.parametrize(
        "module",
        [
            "cleverly.sensitivity.evalue",
            "cleverly.sensitivity._derived",
            "cleverly.sensitivity.omitted_variable",
            "cleverly.sensitivity._simulated_confounding_request",
        ],
    )
    def test_every_sensitivity_surface_reads_one_predicate(self, module: str) -> None:
        surface = importlib.import_module(module)
        assert surface.declares_intermediate is direct_effect.declares_intermediate


def binary_frame() -> Any:
    """The CDE law without its intermediate column, binary outcome."""
    return binary_cde_frame().drop(columns=["Z"])


def logistic_in_sample(**overrides: Any) -> dict[str, Any]:
    return linear_in_sample(outcome_learner=LogisticRegression(max_iter=1000), **overrides)


#: Every C-TMLE strategy, and DR-TMLE, as a factory of estimators for given estimands.
METHODS: dict[str, Any] = {
    **{
        f"ctmle-{strategy}": (
            lambda estimands, strategy=strategy, config=config: CTMLE(
                strategy=strategy, estimands=estimands, **logistic_in_sample(**config)
            )
        )
        for strategy, config in SELECTOR_CONFIGS.items()
    },
    "ctmle-derivative": lambda estimands: CTMLE(
        strategy="derivative", estimands=estimands, **logistic_in_sample()
    ),
    "ctmle-oat": lambda estimands: CTMLE(
        strategy="oat", estimands=estimands, **logistic_in_sample()
    ),
    "ctmle-oat-per-arm-parametric": lambda estimands: CTMLE(
        strategy="oat",
        oat_design="per_arm",
        estimands=estimands,
        **linear_in_sample(
            outcome_learner=ParametricOutcome(), treatment_learner=ParametricMechanism()
        ),
    ),
    "drtmle": lambda estimands: DRTMLE(
        estimands=estimands,
        **linear_drtmle(n_folds=2, outcome_learner=LogisticRegression(max_iter=1000)),
        reduced_crossfit="pooled",
    ),
}
#: The fits whose ratio carries an interval.  The selector paths and the shared outcome-
#: adaptive design report a point only (F18, F19), so their E-value refuses on inference.
WITH_INFERENCE = ("ctmle-derivative", "ctmle-oat-per-arm-parametric", "drtmle")


def method_fits(method: str) -> tuple[Any, Any]:
    """The fit that reports only the ATE, and the same fit that also requests ``rr``."""
    make = METHODS[method]
    frame = binary_frame()
    fits = (
        make(estimands).fit(frame, outcome="Y", treatment="A", covariates=COVARIATES).single()
        for estimands in (("ate",), ("ate", "rr"))
    )
    return next(fits), next(fits)


class TestTheDerivedRatioOnEveryMethod:
    """The retargeted ratio equals the ratio the same fit reports when it requests one."""

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_the_derived_point_equals_the_requested_one(self, method: str) -> None:
        source, reference = method_fits(method)
        assert source.fitted_method in derived_module.DERIVED_METHODS
        derived = derived_module._derived_risk_ratio(source, "ate")
        assert derived.psi == pytest.approx(reference["rr"].psi, rel=1e-10)
        assert derived.inference == reference["rr"].inference

    @pytest.mark.parametrize("method", WITH_INFERENCE)
    def test_the_evalue_reads_the_derived_interval(self, method: str) -> None:
        source, reference = method_fits(method)
        assert reference["rr"].supplies_inference
        selection = evalue_module._select_evalue(source, None)
        assert selection.branch == "derived_rr"
        assert selection.features == (f"derived_{source.fitted_method}",)
        report = evalue_module.evalue(source)
        assert report.risk_ratio == pytest.approx(reference["rr"].psi, rel=1e-10)
        assert report.risk_ratio_ci == pytest.approx(reference["rr"].ci, rel=1e-10)
        assert report.limit_descriptive
        assert report.bias_factor == "single"
        row = source.sensitivity.capability("evalue")
        assert row.available
        assert row.execution == "retarget"

    @pytest.mark.parametrize("method", sorted(set(METHODS) - set(WITH_INFERENCE)))
    def test_a_point_only_path_refuses_on_inference(self, method: str) -> None:
        source, _ = method_fits(method)
        with pytest.raises(_EValueRefusal, match="is built from the reported estimate"):
            evalue_module.evalue(source)

    def test_the_method_gate_admits_the_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation control: the gate before part G refuses the DR-TMLE retarget."""
        source, _ = method_fits("drtmle")
        assert derived_module._risk_ratio_refusal(source, "ate") is None
        monkeypatch.setattr(derived_module, "DERIVED_METHODS", frozenset({"tmle"}))
        assert derived_module._risk_ratio_refusal(source, "ate") is not None

    def test_a_drtmle_fit_names_its_linear_remainder(self) -> None:
        source, _ = method_fits("drtmle")
        report = evalue_module.evalue(source)
        assert "linear in the mechanism error" in report.limit_reading.sentence


def stratified_frame(binary: bool) -> Any:
    """Two baseline strata whose effects and outcome scales differ."""
    rng = np.random.default_rng(18)
    n = 600
    stratum = np.repeat(["low", "high"], n // 2)
    high = stratum == "high"
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.3 * w + 0.4 * high)))
    if binary:
        logit = -0.5 + (0.6 + 0.5 * high) * a + 0.3 * w
        y = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)
    else:
        y = (0.5 + 0.5 * high) * a + w + rng.normal(size=n) * np.where(high, 3.0, 1.0)
    return pd.DataFrame({"Y": y, "A": a, "W": w, "V": stratum})


def stratified_fit(binary: bool, target: Any) -> Any:
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W", "V"), strata=("V",))
    outcome = LogisticRegression(max_iter=1000) if binary else LinearRegression()
    return (
        CausalStudy(stratified_frame(binary), design=design)
        .identify(target)
        .estimate(
            outcome_learner=outcome,
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=2,
            simultaneous=False,
        )
    )


def stratum_names(result: Any, estimand: str) -> list[str]:
    return [
        name
        for name, key in result.parameter_keys.items()
        if key.estimand == estimand and key.stratum is not None
    ]


class TestAStratumContrast:
    """A stratum's E-value reads the stratum's rows, as the stratum's own study would."""

    def test_a_reported_stratum_ratio_inverts_its_own_ratio(self) -> None:
        result = stratified_fit(True, RiskRatio())
        names = stratum_names(result, "rr")
        assert len(names) == 2
        points = []
        for name in names:
            report = evalue_module.evalue(result, name)
            assert report.risk_ratio == result[name].psi
            assert report.point == pytest.approx(evalue_from_rr(result[name].psi), rel=1e-12)
            assert evalue_module._select_evalue(result, name).features == ("stratum",)
            assert report.limit_descriptive
            points.append(report.point)
        assert abs(points[0] - points[1]) > 1e-3

    def test_a_derived_stratum_ratio_equals_the_reported_one(self) -> None:
        from cleverly import ATE

        derived_fit = stratified_fit(True, ATE())
        reported_fit = stratified_fit(True, RiskRatio())
        reported = {
            reported_fit.parameter_keys[name].stratum: reported_fit[name]
            for name in stratum_names(reported_fit, "rr")
        }
        for name in stratum_names(derived_fit, "ate"):
            report = evalue_module.evalue(derived_fit, name)
            target = reported[derived_fit.parameter_keys[name].stratum]
            assert report.estimand == target.name
            assert report.risk_ratio == pytest.approx(target.psi, rel=1e-10)

    def test_a_gaussian_stratum_divides_by_its_own_scale(self) -> None:
        from cleverly import ATE

        result = stratified_fit(False, ATE())
        data = result.data
        for name in stratum_names(result, "ate"):
            code = list(data.strata_levels).index(tuple(result.parameter_keys[name].stratum))
            rows = np.asarray(data.strata == code)
            sd = float(np.std(data.outcome[rows], ddof=1))
            report = evalue_module.evalue(result, name)
            rr = np.exp(_SMD_TO_LOG_RR * result[name].psi / sd)
            assert report.risk_ratio == pytest.approx(rr, rel=1e-12)
            assert "within the stratum" in report.note
            # Mutation: the whole-sample scale, which the strata's 1:3 noise moves.
            overall = float(np.std(data.outcome, ddof=1))
            assert abs(np.exp(_SMD_TO_LOG_RR * result[name].psi / overall) - rr) > 1e-3


def missing_frame(seed: int = 1, n: int = 4000) -> tuple[Any, float, float]:
    """A law where ``E[Y^2 | A, W]`` is linear and high ``W`` raises nonresponse.

    Returns the frame, the full-sample standard deviation before masking, and the
    respondents' standard deviation.
    """
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, 0.5, size=n).astype(float)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w)))
    y = 2.0 * w + rng.normal(size=n)
    responded = rng.binomial(1, 1 / (1 + np.exp(-(1.5 - 3.0 * w))))
    frame = pd.DataFrame({"Y": np.where(responded == 1, y, np.nan), "A": a, "W": w})
    frame["D"] = responded
    return frame, float(y.std()), float(y[responded == 1].std())


def missing_fit(frame: Any) -> Any:
    return (
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            missingness_learner=LogisticRegression(),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"], delta="D")
        .single()
    )


class TestTheMissingAtRandomScale:
    """The standardized E-value divides by the population sd(Y), not the respondents'."""

    def test_every_outcome_observed_gives_the_complete_outcome_scale(self) -> None:
        """Exact witness: with no missing row both branches divide by one standard deviation.

        The stratum mask reaches the plug-in too: on the rows with ``W = 1`` it equals the
        complete-outcome scale of those rows, which differs from the whole sample's.
        """
        frame, _, _ = missing_frame()
        frame = frame.assign(Y=frame["Y"].fillna(0.0), D=1)
        result = missing_fit(frame)
        assert missing_at_random_sd(result) == pytest.approx(
            evalue_module._standardising_sd(result)[0], rel=1e-12
        )
        # Mutation: the population form without the reliability correction.
        assert abs(missing_at_random_sd(result) - float(np.std(frame["Y"]))) > 1e-6
        rows = np.asarray(frame["W"] == 1.0)
        within = missing_at_random_sd(result, rows)
        assert within == pytest.approx(evalue_module._standardising_sd(result, rows)[0], rel=1e-12)
        assert abs(within - missing_at_random_sd(result)) > 0.01

    def test_the_plugin_recovers_the_population_scale(self) -> None:
        frame, population, respondents = missing_frame()
        result = missing_fit(frame)
        assert result.data.has_missing_outcome
        plugin = missing_at_random_sd(result)
        # Nonzero witness: the respondents' scale misses by more than 0.1.
        assert abs(respondents - population) > 0.1
        assert abs(plugin - population) < 0.03
        report = evalue_module.evalue(result, "ate")
        assert "missing-at-random plug-in" in report.note
        rr = np.exp(_SMD_TO_LOG_RR * result["ate"].psi / plugin)
        assert report.risk_ratio == pytest.approx(rr, rel=1e-12)
        assert evalue_module._select_evalue(result, "ate").features == ("response_sd",)

    def test_the_capability_row_defers_the_refit(self) -> None:
        frame, _, _ = missing_frame(n=600)
        result = missing_fit(frame)
        row = result.sensitivity.capability("evalue")
        assert row.available
        assert row.execution == "refit"
        assert row.cost == "moderate"
        item = result.assess().sensitivity["evalue"]
        assert item.status is AssessmentStatus.DEFERRED


#: The two risks at the second horizon.
NUMERATOR, DENOMINATOR = "risk_regimen[a1 @ t=2]", "risk_regimen[a0 @ t=2]"


@pytest.fixture(scope="module")
def survival_fit() -> Any:
    frame, _ = make_point_survival(n=600, seed=3, n_times=4)
    design = TimeToEvent(time="time", event="event", treatment="A", baseline=["W1", "W2"])
    learners = {
        name: LogisticRegression(max_iter=1000)
        for name in ("outcome_learner", "treatment_learner", "censoring_learner")
    }
    return (
        CausalStudy(frame, design=design)
        .identify(RegimeMean(regimens={"a1": 1, "a0": 0}))
        .estimate(
            **learners,
            random_state=5,
            simultaneous=False,
            cross_fit=False,
            survival_model="hazard",
        )
    )


class TestATimeToEventRiskRatio:
    """The E-value of the risk ratio of two static regimens at one horizon."""

    def test_the_ratio_is_the_fit_s_own(self, survival_fit: Any) -> None:
        report = evalue_module.evalue(survival_fit, (NUMERATOR, DENOMINATOR))
        ratio = survival_fit.ratio(NUMERATOR, DENOMINATOR, kind="rr")
        assert report.estimand == ratio.name
        assert report.risk_ratio == ratio.psi
        assert report.risk_ratio_ci == ratio.ci
        assert report.point == pytest.approx(evalue_from_rr(ratio.psi), rel=1e-12)
        assert report.note == _TIME_TO_EVENT_CONDITIONS
        assert report.limit_descriptive
        # The ratio is exact, so no conversion note fires, yet the censoring conditions
        # the bound needs still print in plain words, and formally at full detail.
        assert not report.approximate
        summary = report.summary()
        assert "approximate_scale" not in summary.note_codes
        assert summary.note("censoring_conditions").text == NOTE_TEXTS["censoring_conditions"]
        full = report.summary(detail="full").section("Assumptions").lines
        assert _TIME_TO_EVENT_CONDITIONS in full
        # Nonzero witness for the order: the inverse ratio has the same E-value and
        # the reciprocal risk ratio.
        swapped = evalue_module.evalue(survival_fit, (DENOMINATOR, NUMERATOR))
        assert swapped.risk_ratio == pytest.approx(1 / ratio.psi, rel=1e-12)
        assert swapped.point == pytest.approx(report.point, rel=1e-12)

    def test_the_facade_defers_to_an_explicit_pair(self, survival_fit: Any) -> None:
        row = survival_fit.sensitivity.capability("evalue")
        assert row.status is AssessmentStatus.DEFERRED
        assert row.requires_arguments == ("estimand",)
        assert NUMERATOR in row.reason
        point = survival_fit.sensitivity.evalue((NUMERATOR, DENOMINATOR)).point
        assert point == evalue_module.evalue(survival_fit, (NUMERATOR, DENOMINATOR)).point

    @pytest.mark.parametrize(
        "estimand,message",
        [
            (NUMERATOR, "reads the risk ratio of two reported risks"),
            ((NUMERATOR,), "reads the risk ratio of two reported risks"),
            ((NUMERATOR, NUMERATOR), "are not the risks of two regimens"),
            ((NUMERATOR, "risk_regimen[a0 @ t=3]"), "are not the risks of two regimens"),
            ((NUMERATOR, "rr_regimen[a1 vs a0 @ t=2]"), "are not risks this fit reports"),
        ],
    )
    def test_a_malformed_request_names_the_accepted_form(
        self, survival_fit: Any, estimand: Any, message: str
    ) -> None:
        with pytest.raises(_EValueRefusal) as caught:
            evalue_module._select_evalue(survival_fit, estimand)
        assert message in str(caught.value)
        assert "kind='rr'; an odds ratio has no conversion here" in str(caught.value)

    def test_a_dynamic_regimen_refuses(self, survival_fit: Any) -> None:
        """Mutation control for the static check: a regimen of another class refuses."""

        class Dynamic:
            label = "a1"

        regimens = (Dynamic(), *survival_fit.config.regimens[1:])
        edited = replace(survival_fit, config=replace(survival_fit.config, regimens=regimens))
        with pytest.raises(_EValueRefusal) as caught:
            evalue_module._select_evalue(edited, (NUMERATOR, DENOMINATOR))
        assert caught.value.refusal.roadmap == "F34"


def every_part_g_configuration(cde_fits: dict[str, Any], survival_fit: Any) -> dict[str, Any]:
    """One request per part G configuration, keyed by its registry rule."""
    frame, _, _ = missing_frame(n=600)
    drtmle = (
        METHODS["drtmle"](("ate",))
        .fit(binary_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
        .single()
    )
    ctmle = (
        METHODS["ctmle-derivative"](("ate",))
        .fit(binary_frame(), outcome="Y", treatment="A", covariates=COVARIATES)
        .single()
    )
    stratified = stratified_fit(True, RiskRatio())
    return {
        "cde": lambda: evalue_module.evalue(replace(cde_fits["ratio"][0.0])),
        "stratum": lambda: evalue_module.evalue(stratified, stratum_names(stratified, "rr")[0]),
        "response_sd": lambda: evalue_module.evalue(missing_fit(frame), "ate"),
        "derived_drtmle": lambda: evalue_module.evalue(drtmle),
        "derived_collaborative_tmle": lambda: evalue_module.evalue(ctmle),
        "time_to_event": lambda: evalue_module.evalue(survival_fit, (NUMERATOR, DENOMINATOR)),
    }


def covering_row(rule: str) -> VerdictEvidence:
    """A registry row that licenses one part G configuration on every fit."""
    return VerdictEvidence(
        instrument=EVALUE_LIMIT_INSTRUMENT,
        rule=rule,
        rung="L2",
        theory="test row",
        scope=CalibrationScope.unrestricted(),
        sentence="covered by the test row",
        exact=True,
    )


def _evidence(summary: Any) -> str:
    """The limit's evidence sentence, as the full summary's settings print it."""
    prefix = "Evidence for the limit: "
    (line,) = [line for line in summary.section("Settings").lines if line.startswith(prefix)]
    return line.removeprefix(prefix)


class TestTheLimitReading:
    """Every part G confidence limit resolves through the registry, one rule each."""

    def test_every_test_fit_is_descriptive_on_the_shipped_registry(
        self, cde_fits: dict[str, Any], survival_fit: Any
    ) -> None:
        # The shipped rows need n >= 2,000 and the studies' learners; these fits have neither.
        for rule, call in every_part_g_configuration(cde_fits, survival_fit).items():
            report = call()
            assert report.limit_reading is not None, rule
            assert not report.limit_reading.licensed, rule
            assert report.limit_reading.sentence.startswith(DESCRIPTIVE_SENTENCE), rule
            assert report.limit_descriptive, rule
            assert report.to_dict()["limit_licensed"] is False
            summary = report.summary(detail="full")
            assert summary.note("limits_descriptive").text == NOTE_TEXTS[
                "limits_descriptive"
            ].format(what="E-value of the limit")
            assert _evidence(summary).startswith(DESCRIPTIVE_SENTENCE), rule

    def test_a_registered_row_licenses_exactly_its_configuration(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cde_fits: dict[str, Any],
        survival_fit: Any,
    ) -> None:
        """Mutation control: each row flips its own rule and leaves the others descriptive."""
        calls = every_part_g_configuration(cde_fits, survival_fit)
        for rule in calls:
            monkeypatch.setattr(interpretation, "VERDICT_EVIDENCE", (covering_row(rule),))
            readings = {name: call() for name, call in calls.items()}
            licensed = readings.pop(rule)
            assert licensed.limit_reading.licensed, rule
            assert not licensed.limit_descriptive
            summary = licensed.summary(detail="full")
            assert "limits_descriptive" not in summary.note_codes
            assert _evidence(summary) == "covered by the test row"
            assert all(report.limit_descriptive for report in readings.values()), rule

    def test_a_cde_stratum_reads_the_joined_rule(self) -> None:
        """Two features join with ``+``, so a row for one of them does not cover both."""
        frame = stratified_frame(True)
        frame["Z"] = np.random.default_rng(0).integers(0, 2, len(frame))
        design = PointTreatment(
            outcome="Y", treatment="A", adjustment=("W", "V"), strata=("V",), intermediate="Z"
        )
        from cleverly import ControlledDirectEffect

        result = (
            CausalStudy(frame, design=design)
            .identify(ControlledDirectEffect(intermediate=0.0))
            .estimate(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                cross_fit=False,
                random_state=2,
                simultaneous=False,
            )
        )
        name = next(
            name
            for name, key in result.parameter_keys.items()
            if key.stratum is not None and key.reference is not None
        )
        selection = evalue_module._select_evalue(result, name)
        assert selection.features[:2] == ("cde", "stratum")
        assert re.fullmatch(r"cde\+stratum(\+\w+)?", "+".join(selection.features))


def test_a_missing_treatment_with_a_missing_outcome_refuses_the_standardized_path() -> None:
    """The plug-in reads ``E[Y^k | A, W]`` on a row whose outcome is missing.

    A row whose treatment is also unrecorded has no ``A`` to read, so the path refuses
    before any regression runs. Control: the same request with every treatment recorded
    answers (``TestTheMissingAtRandomScale``).
    """
    from tests.unit._capability_sweep_support import fit_tmle_missing_treatment_weights

    result = fit_tmle_missing_treatment_weights()
    assert result.data.has_missing_outcome and result.data.has_missing_treatment
    row = result.sensitivity.capability("evalue")
    assert row.status is AssessmentStatus.UNAVAILABLE
    assert "treatment is not recorded" in row.reason
    with pytest.raises(CapabilityError, match="treatment is not recorded"):
        evalue_module.evalue(result, "ate")

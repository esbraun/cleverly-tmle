"""Witnesses for the result-level post-fit assessment battery."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    AssessmentReport,
    AssessmentStatus,
    CapabilityError,
    CausalStudy,
    CollaborativeTMLEMethod,
    OddsRatio,
    ParameterKey,
    PointTreatment,
    RiskRatio,
    load,
)
from cleverly.assessment import (
    ASSESSMENT_CAPABILITIES,
    INTERPRETERS,
    SENSITIVITY_ROUTES,
    VALIDATION_OPERATIONS,
    AssessmentItem,
)
from cleverly.datasets import make_binary_outcome, make_multi_arm
from cleverly.estimators import TMLE
from cleverly.sensitivity._derived import _derived_risk_ratio
from cleverly.sensitivity.evalue import _select_evalue, _standardising_sd, evalue_from_rr


def _study(*, strata: bool = False) -> CausalStudy:
    frame, _ = make_binary_outcome(n=260, seed=17)
    if strata:
        frame["V"] = np.where(frame["W1"] >= frame["W1"].median(), "high", "low")
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", *(("V",) if strata else ())),
            strata=("V",) if strata else (),
        ),
    )


def _learners(**overrides: object) -> dict[str, object]:
    """The cheap explicit-learner block every raw fit in this file shares.

    A fresh pair of learners per call: scikit-learn estimators are stateful, so a shared
    instance would carry one test's fit into another's.  Only the settings that were
    literally the same at every call site are defaulted here.  ``learner_folds`` is not,
    because the fits that pass it and the fits that do not are different fits.
    """
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "random_state": 3,
        "simultaneous": False,
        **overrides,
    }


def _raw(engine: type = TMLE, **overrides: object):
    """An unfitted estimator built directly, bypassing ``CausalStudy``."""
    return engine(**_learners(**overrides))


def _fit(study: CausalStudy, estimand: object):
    return study.identify(estimand).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=23,
        simultaneous=False,
    )


@pytest.mark.parametrize("strata", [False, True])
def test_cached_nuisance_risk_ratio_equals_a_typed_fit(strata: bool) -> None:
    study = _study(strata=strata)
    ate = _fit(study, ATE())
    reported = _fit(study, RiskRatio())["rr"]
    derived = _derived_risk_ratio(ate, "ate")

    assert derived.psi == pytest.approx(reported.psi, abs=1e-12)
    assert derived.variance == pytest.approx(reported.variance, abs=1e-12)
    np.testing.assert_allclose(derived.influence_curve, reported.influence_curve, atol=1e-12)

    direct = _fit(study, RiskRatio()).sensitivity.evalue("rr")
    assert direct.scale == "risk ratio"
    assert not direct.approximate


def test_cached_nuisance_risk_ratio_reads_the_propensity_it_claims_to_retarget() -> None:
    result = _fit(_study(), ATE())
    before = _derived_risk_ratio(result, "ate").psi
    propensity = result.repeats[0].nuisance.propensity.values
    propensity[:, 1] = np.clip(0.7 * propensity[:, 1], 0.02, 0.98)
    propensity[:, 0] = 1.0 - propensity[:, 1]
    result.assessment_cache.clear()

    assert _derived_risk_ratio(result, "ate").psi != pytest.approx(before)


def test_default_odds_ratio_request_derives_exact_rr_but_explicit_or_is_approximate() -> None:
    result = _fit(_study(), OddsRatio())

    exact = result.sensitivity.evalue()
    assert exact.estimand == "rr"
    assert exact.scale == "risk ratio"
    assert not exact.approximate
    assert "source contrast 'or'" in exact.note

    approximate = result.sensitivity.evalue("or")
    assert approximate.estimand == "or"
    assert approximate.scale == "odds ratio"
    assert approximate.approximate


def test_assess_retains_reports_arguments_and_omissions(tmp_path) -> None:
    result = _fit(_study(), ATE())
    battery = result.assess(include_retargets=True)

    assert isinstance(battery, AssessmentReport)
    assert battery.report("evalue").estimand == "rr"
    assert battery.diagnostics["corrections"].status is AssessmentStatus.NOT_APPLICABLE
    assert battery.omissions
    assert "surface" in battery.to_frame().columns

    restored = load(result.save(tmp_path / "assessment.joblib"))
    replayed = restored.assess(include_retargets=True)
    assert replayed.report("evalue") == battery.report("evalue")


def test_bare_assess_runs_only_the_cheap_cached_nuisance_retarget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit(_study(), ATE())
    calls = []
    retarget = result.estimator.retarget

    def tracked(*args, **kwargs):
        calls.append("retarget")
        return retarget(*args, **kwargs)

    monkeypatch.setattr(result.estimator, "retarget", tracked)
    monkeypatch.setattr(
        result.estimator, "refit", lambda *_args, **_kwargs: pytest.fail("unexpected refit")
    )
    bare = result.assess()
    assert calls == ["retarget"]
    assert bare.sensitivity["evalue"].status is AssessmentStatus.COMPLETED
    assert bare.sensitivity["evalue"].arguments == {"estimand": "ate"}
    assert bare.diagnostics["truncation_curve"].status is AssessmentStatus.UNAVAILABLE
    assert result.assess().report("evalue") == bare.report("evalue")
    assert calls == ["retarget"]
    opted_in = result.assess(include_retargets=True)
    assert calls
    assert opted_in.report("evalue").estimand == "rr"


@pytest.fixture
def typed_multi_arm_result():
    frame, _ = make_multi_arm(n=300, seed=17, family="binomial")
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    return _fit(study, ATE(reference="low"))


def test_multi_arm_evalue_refuses_an_ambiguous_default(typed_multi_arm_result) -> None:
    result = typed_multi_arm_result
    aliases = tuple(result.estimates)
    assert len(aliases) == 2
    with pytest.raises(CapabilityError, match="choose an explicit estimand") as caught:
        result.sensitivity.evalue()
    assert all(alias in str(caught.value) for alias in aliases)
    row = result.sensitivity.run_all(include_retargets=True)["evalue"]
    assert row.status is AssessmentStatus.UNAVAILABLE
    assert all(alias in row.detail for alias in aliases)


def test_explicit_multi_arm_evalue_uses_its_request_for_availability_and_cost(
    typed_multi_arm_result,
) -> None:
    result = typed_multi_arm_result
    alias = tuple(result.estimates)[1]
    requested = {"evalue": {"estimand": alias}}
    default = result.sensitivity.run_all(arguments=requested)["evalue"]
    assert default.status is AssessmentStatus.COMPLETED
    combined = result.sensitivity.run_all(include_retargets=True, arguments=requested)
    direct = result.sensitivity.evalue(alias)
    assert combined.report("evalue") == direct
    assert alias in direct.note


def test_explicit_collaborative_or_evalue_is_not_blocked_by_default_derivation() -> None:
    result = (
        _study()
        .identify(OddsRatio())
        .estimate(
            method=CollaborativeTMLEMethod(selection_estimand="or"),
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=3,
            learner_folds=2,
            random_state=23,
            simultaneous=False,
        )
    )
    assert result.sensitivity.capability("evalue").available
    assert result.sensitivity.evalue().approximate
    direct = result.sensitivity.evalue("or")
    combined = result.sensitivity.run_all(arguments={"evalue": {"estimand": "or"}})
    assert combined.report("evalue") == direct
    assert direct.scale == "odds ratio"
    assert "common outcomes" in direct.note
    assert "understates" not in direct.note


def test_warnings_preserve_score_and_support_measurements() -> None:
    support = SimpleNamespace(
        truncated={"fraction": 0.12},
        effective_sample_size={"treated": {"ratio": 0.16}},
    )
    support_row = INTERPRETERS["support"](support, None)
    assert support_row.status is AssessmentStatus.WARNING
    assert "truncated fraction 12.0%" in support_row.detail
    assert "minimum effective-sample-size ratio 16.0%" in support_row.detail
    score = SimpleNamespace(rows=(SimpleNamespace(ratio=0.25),), passed=True)
    fitted = SimpleNamespace(
        repeats=(
            SimpleNamespace(
                fluctuations={
                    "mean": SimpleNamespace(reduction=SimpleNamespace(ill_conditioned=1, rounds=4))
                }
            ),
        )
    )
    score_row = INTERPRETERS["score_equations"](score, fitted)
    assert score_row.status is AssessmentStatus.WARNING
    assert "1 score row(s)" in score_row.detail
    assert "worst abs(score) / threshold = 0.25" in score_row.detail
    assert "ill-conditioned" in score_row.detail


def test_assessment_item_equality_does_not_compare_numpy_arguments() -> None:
    first = AssessmentItem(
        "refute",
        AssessmentStatus.COMPLETED,
        "done",
        arguments={"negative_control_outcome": np.array([0.0, 1.0])},
    )
    second = replace(
        first,
        arguments={"negative_control_outcome": np.array([1.0, 0.0])},
    )
    assert first == second


def test_interpreters_and_capabilities_cover_each_other() -> None:
    """Every declared operation has an interpreter, and every interpreter has a row.

    Both sides are derived. A hardcoded list of the sensitivity names stood here and made
    the test one-way: a new key in ``SENSITIVITY_ROUTES`` with no interpreter left it
    green, and ``_run_all`` then raised ``KeyError`` at ``INTERPRETERS[operation]`` on
    the first combined report that reached the row.
    """
    diagnostic = {row.operation for row in ASSESSMENT_CAPABILITIES}
    sensitivity = set(SENSITIVITY_ROUTES)
    assert set(INTERPRETERS) == diagnostic | sensitivity


def test_a_route_without_an_interpreter_is_what_the_two_way_check_catches() -> None:
    """The deliberate-mutation control for the check above.

    The equality is blind if the two registries happen to agree for another reason. Add
    one route and the check must fail, which is what tells a reader that it reads
    ``SENSITIVITY_ROUTES`` rather than a copy of its keys.
    """
    routes = set(SENSITIVITY_ROUTES) | {"invented_analysis"}
    assert set(INTERPRETERS) != {row.operation for row in ASSESSMENT_CAPABILITIES} | routes


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"fitted_method": "collaborative_tmle"}, "collaborative_tmle"),
        ({"fitted_method": "drtmle"}, "drtmle"),
        ({"intermediate_value": 0.0}, "controlled direct effects"),
    ],
)
def test_derived_ratio_preserves_method_and_estimand_refusal_boundaries(
    change: dict[str, object], reason: str
) -> None:
    result = _fit(_study(), ATE())
    changed = replace(result, **change)
    capability = changed.sensitivity.capability("evalue")
    assert not capability.available
    assert reason in capability.reason
    with pytest.raises(CapabilityError, match=reason):
        changed.sensitivity.evalue("ate")


def test_derived_ratio_preserves_cv_legacy_axis_and_conditional_refusals() -> None:
    result = _fit(_study(), ATE())
    key = result.parameter_keys["ate"]
    cases = (
        (replace(result, config=replace(result.config, cv_evaluation=True)), "CV-evaluated"),
        (
            replace(result, parameter_keys={"ate": replace(key, axis="shift")}),
            "arm contrast",
        ),
        (
            replace(result, parameter_keys={"ate": replace(key, estimand="att")}),
            "conditional baseline risk",
        ),
        (
            replace(result, parameter_keys={"ate": replace(key, estimand="atc")}),
            "conditional baseline risk",
        ),
    )
    for changed, reason in cases:
        changed.assessment_cache.clear()
        with pytest.raises(CapabilityError, match=reason):
            changed.sensitivity.evalue("ate")


def test_fixed_baseline_fallback_is_explicitly_approximate() -> None:
    result = _fit(_study(), ATE())
    baseline = replace(result["ate"], name="ey0", psi=0.5)
    key = result.parameter_keys["ate"]
    restored = replace(
        result,
        estimator=None,
        estimates={"ate": result["ate"], "ey0": baseline},
        parameter_keys={
            "ate": key,
            "ey0": ParameterKey("ey0", "ey0", value=key.reference),
        },
    )

    report = restored.sensitivity.evalue("ate")
    assert report.scale == "risk difference"
    assert report.approximate
    assert "holds that risk fixed" in report.note


def _simulated_interpreter_report(*, failures: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        successful_cells=(SimpleNamespace(displacement=0.1),),
        cells=(SimpleNamespace(induced_treatment_association=None if failures else 0.2),),
        failures=failures,
        movement_scale="estimate_difference",
        population="perturbed_treatment_group",
        stratum=("low",),
        conditioning_arm=1,
        association_population="selected_baseline_stratum",
        calibration_population="full_fitted_population",
    )


def test_descriptive_interpreters_complete_without_inventing_a_verdict() -> None:
    frame = pd.DataFrame
    reports = {
        "truncation_curve": frame({"bound": [0.01, 0.1], "estimate": [1.0, 1.2]}),
        "robustness_value": {"rv": 0.2, "rva": 0.1},
        "elements": SimpleNamespace(sigma2=1.0, nu2=2.0, max_bias=0.3),
        "contour": frame({"cf_d": [0.0, 0.1], "cf_y": [0.0, 0.1], "value": [1.0, 0.9]}),
        "benchmark": SimpleNamespace(
            covariates=("W1",), cf_y=0.1, cf_d=0.2, rho=0.3, delta_psi=0.4
        ),
        "simulated_confounding": _simulated_interpreter_report(),
        "evalue": SimpleNamespace(point=2.0, limit=1.5, scale="risk ratio", approximate=False),
        "missingness": frame({"gamma": [0.5, 1.5], "psi": [0.9, 1.1]}),
        "tipping_gamma": None,
        "stagewise": SimpleNamespace(rows=()),
    }

    for operation, report in reports.items():
        assert INTERPRETERS[operation](report, None).status is AssessmentStatus.COMPLETED

    item = INTERPRETERS["simulated_confounding"](reports["simulated_confounding"], None)
    assert "movement scale estimate_difference" in item.detail
    assert "target population perturbed_treatment_group" in item.detail
    assert "baseline stratum ('low',); conditioning arm 1" in item.detail
    assert "association population selected_baseline_stratum" in item.detail
    assert "calibration population full_fitted_population" in item.detail


def test_interpreters_reserve_failed_and_warning_for_evidence_backed_rules() -> None:
    failed_score = SimpleNamespace(rows=(SimpleNamespace(ratio=2.0),), passed=False)
    failed_correction = SimpleNamespace(
        rows=(SimpleNamespace(residual=0.1, reported=0.2),),
        contract="identity",
        passed=False,
    )
    failed_refutation = SimpleNamespace(tests=(SimpleNamespace(name="placebo", passed=False),))
    spanning_bound = SimpleNamespace(
        lower=-0.1,
        upper=0.2,
        null_hypothesis=0.0,
        cf_y=0.1,
        cf_d=0.2,
        rho=0.3,
    )
    failed_surface = _simulated_interpreter_report(failures=("cell refused",))
    nuisance_warning = SimpleNamespace(findings=("calibration slope outside its rule",))

    assert INTERPRETERS["score_equations"](failed_score, None).status is AssessmentStatus.FAILED
    assert INTERPRETERS["corrections"](failed_correction, None).status is AssessmentStatus.FAILED
    assert INTERPRETERS["refute"](failed_refutation, None).status is AssessmentStatus.FAILED
    assert (
        INTERPRETERS["omitted_confounding"](spanning_bound, None).status is AssessmentStatus.WARNING
    )
    assert (
        INTERPRETERS["simulated_confounding"](failed_surface, None).status
        is AssessmentStatus.WARNING
    )
    assert (
        INTERPRETERS["nuisance_models"](nuisance_warning, None).status is AssessmentStatus.WARNING
    )


@pytest.mark.parametrize("engine_name", ["tmle", "drtmle", "ctmle", "cv"])
def test_reported_baseline_fallback_is_identical_live_saved_and_detached(
    engine_name, tmp_path, monkeypatch
):
    from cleverly.estimators import CTMLE, DRTMLE

    engine = {"tmle": TMLE, "drtmle": DRTMLE, "ctmle": CTMLE, "cv": TMLE}[engine_name]
    frame, _ = make_binary_outcome(n=160, seed=3)
    options = {"cv_evaluation": True} if engine_name == "cv" else {}
    raw = (
        _raw(engine, estimands=("ate", "ey0"), learner_folds=2, **options)
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )
    expected_method = {"ctmle": "collaborative_tmle", "cv": "tmle"}.get(engine_name, engine_name)
    assert raw.fitted_method == expected_method
    assert replace(raw, extra={}).fitted_method == expected_method
    restored = load(raw.save(tmp_path / f"{engine_name}.joblib"))
    restored.assessment_cache.clear()
    before = raw.sensitivity.evalue("ate")
    after = restored.sensitivity.evalue("ate")
    assert before == after
    if engine_name != "tmle":
        assert before.approximate
        assert before.risk_ratio == pytest.approx(1 + raw["ate"].psi / raw["ey0"].psi)
        assert before.risk_ratio_ci == pytest.approx(
            tuple(1 + x / raw["ey0"].psi for x in raw["ate"].ci)
        )
        monkeypatch.setattr(
            raw.estimator, "retarget", lambda *a, **k: pytest.fail("variant retarget")
        )
        raw.assessment_cache.clear()
        assert raw.sensitivity.evalue("ate") == before
        detached = replace(raw, estimator=None)
        assert detached.sensitivity.evalue("ate") == before


@pytest.mark.parametrize("target", ["ate", "att", "atc"])
@pytest.mark.parametrize("typed", [False, True])
def test_gaussian_differences_use_the_documented_nonzero_conversion(target, typed):
    from cleverly import ATC, ATT
    from cleverly.datasets import make_linear_ate

    frame, _ = make_linear_ate(n=180, seed=3)
    options = _learners()
    if typed:
        result = (
            CausalStudy(
                frame,
                design=PointTreatment(
                    outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")
                ),
            )
            .identify({"ate": ATE, "att": ATT, "atc": ATC}[target]())
            .estimate(**options)
        )
    else:
        result = (
            TMLE(estimands=(target,), **options).fit(frame, outcome="Y", treatment="A").single()
        )
    report = result.sensitivity.evalue(target)
    sd = np.std(frame["Y"], ddof=1)
    expected = np.exp((1.81 / 2) * result[target].psi / sd)
    assert abs(expected - 1) > 0.1
    assert report.approximate
    assert report.risk_ratio == pytest.approx(expected, rel=1e-12)
    assert report.risk_ratio_ci == pytest.approx(
        np.exp((1.81 / 2) * np.asarray(result[target].ci) / sd)
    )


def test_a_weighted_gaussian_conversion_standardises_by_the_weighted_sd():
    """The witness for the standardising sd reading the observation weights.

    ``psi`` targets the population the observation weights describe, so the standard
    deviation that standardises it must describe that population too. The unweighted
    sample deviation puts the numerator and the denominator on two scales, and it moves
    the reported risk ratio with no user-visible sign. The weights here depend on ``W1``
    alone and move ``sd(Y)`` by more than a tenth, so the two answers cannot coincide.
    """
    from cleverly.datasets import make_linear_ate

    frame, _ = make_linear_ate(n=200, seed=3)
    frame = frame.assign(obs_weight=np.where(frame["W1"] > 0, 4.0, 0.25))
    result = (
        _raw(estimands=("ate",))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W3", "W4"],
            weights="obs_weight",
        )
        .single()
    )
    assert result.data.is_weighted

    outcome = np.asarray(result.data.outcome)
    weights = np.asarray(result.data.weights)
    total = weights.sum()
    correction = total - (weights**2).sum() / total
    mean = np.average(outcome, weights=weights)
    weighted_sd = float(np.sqrt((weights * (outcome - mean) ** 2).sum() / correction))
    unweighted_sd = float(np.std(outcome, ddof=1))
    # The nonzero witness: the two scales differ, so a dropped weight is visible here.
    assert abs(weighted_sd / unweighted_sd - 1) > 0.1

    assert _standardising_sd(result) == (pytest.approx(weighted_sd, rel=1e-12), True)
    report = result.sensitivity.evalue("ate")
    expected = float(np.exp((1.81 / 2) * result["ate"].psi / weighted_sd))
    assert report.risk_ratio == pytest.approx(expected, rel=1e-12)
    assert report.risk_ratio != pytest.approx(
        np.exp((1.81 / 2) * result["ate"].psi / unweighted_sd), rel=1e-6
    )
    assert report.point == pytest.approx(evalue_from_rr(expected), rel=1e-12)
    assert f"weighted sd(Y) = {weighted_sd:.4g}" in report.note


def test_an_unweighted_gaussian_conversion_keeps_the_plain_sample_deviation():
    """The reliability-weight correction is ``n - 1`` when every weight is one."""
    from cleverly.datasets import make_linear_ate

    frame, _ = make_linear_ate(n=180, seed=3)
    result = _raw(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
    sd, weighted = _standardising_sd(result)
    assert not weighted
    assert sd == pytest.approx(float(np.std(np.asarray(result.data.outcome), ddof=1)), rel=1e-12)
    assert "weighted" not in result.sensitivity.evalue("ate").note


def test_a_single_observed_row_is_refused_by_the_cause_that_produced_it():
    """The reliability correction is ``w - w^2 / w = 0`` on one row, whatever Y is.

    That is a row count and not a variance of zero. The refusal used to say the outcome
    had no variance, which sends a reader to look for constant values that are not there.
    """
    from cleverly.sensitivity.evalue import _standardising_sd

    # The nonzero witness: two distinct outcome values, so the variance is not the cause.
    def fitted(observed):
        return SimpleNamespace(
            data=SimpleNamespace(
                outcome=np.array([1.0, 4.0]),
                observed=np.array(observed),
                weights=np.array([1.0, 1.0]),
                is_weighted=False,
            )
        )

    assert _standardising_sd(fitted([True, True]))[0] == pytest.approx(
        float(np.std([1.0, 4.0], ddof=1))
    )
    with pytest.raises(CapabilityError, match="at least two observed rows of positive weight"):
        _standardising_sd(fitted([True, False]))


def test_default_or_records_a_source_without_changing_replay_semantics():
    result = _fit(_study(), OddsRatio())
    battery = result.assess()
    row = battery.sensitivity["evalue"]
    assert row.arguments == {"estimand": None}
    assert row.report.source_estimand == "or"
    assert not result.sensitivity.evalue(**row.arguments).approximate


@pytest.mark.parametrize("target", [RiskRatio(), OddsRatio()])
def test_unstamped_artifact_keeps_reported_ratio_conversions(target):
    result = replace(_fit(_study(), target), fitted_method="unknown")
    alias = next(iter(result.estimates))
    assert result.sensitivity.evalue() == result.sensitivity.evalue(alias)


def test_evalue_selector_is_resolved_once_per_request(monkeypatch):
    import importlib

    module = importlib.import_module("cleverly.sensitivity.evalue")
    result = _fit(_study(), ATE())
    calls = []
    original = module._select_evalue

    def tracked(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(module, "_select_evalue", tracked)
    result.assess()
    assert calls == [None]


def test_missing_or_invalid_baseline_cannot_enable_approximation():
    result = _fit(_study(), ATE())
    for value in (0.0, -0.1, float("nan"), float("inf")):
        baseline = replace(result["ate"], name="ey0", psi=value)
        variant = replace(
            result,
            fitted_method="drtmle",
            estimates={**result.estimates, "ey0": baseline},
            parameter_keys={},
        )
        with pytest.raises(CapabilityError, match="finite positive reported reference-arm mean"):
            variant.sensitivity.evalue("ate")


def _rare_outcome_drtmle_fit(
    seed: int = 10, n: int = 240, risk: float = 0.05, treated_risk: float | None = None
):
    """A DR-TMLE fit whose ATE interval reaches below the negative of its baseline risk.

    DR-TMLE has no cached-nuisance risk-ratio retarget, so ``ate`` takes the
    fixed-baseline conversion. The outcome is rare, so the difference interval is wide
    against a small reference risk. That is the geometry the affine conversion sends
    outside the risk-ratio parameter space.

    ``treated_risk=None`` gives a null effect, and the point ratio then sits above the
    null. A ``treated_risk`` below ``risk`` gives a protective effect, and the point ratio
    then sits below the null. The two sides truncate the same bound and read different
    ones, so both are witnessed.
    """
    from cleverly.estimators import DRTMLE

    rng = np.random.default_rng(seed)
    # The draw order is load-bearing: both witnesses below are facts about one realised
    # fit, so a reordered stream would silently retire the geometry they assert.
    covariates = {name: rng.normal(size=n) for name in ("W1", "W2", "W3")}
    treatment = rng.binomial(1, 0.5, size=n)
    frame = pd.DataFrame(
        {
            **covariates,
            "A": treatment,
            "Y": rng.binomial(
                1,
                np.full(n, risk)
                if treated_risk is None
                else np.where(treatment == 1, treated_risk, risk),
            ),
        }
    )
    return (
        _raw(DRTMLE, estimands=("ate", "ey0"), learner_folds=2, random_state=seed)
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )


def test_a_difference_bound_below_the_negative_baseline_reports_a_limit_of_one():
    """The witness for the null test running before the positivity test.

    The conversion ``x -> (baseline + x) / baseline`` is affine, so a difference bound
    below ``-baseline`` becomes a negative ratio bound. Such an interval covers the null,
    so no confounding is needed to move it there and the confidence-limit E-value is 1.
    An implementation that screens for a positive ratio first reports ``nan``, which reads
    as "no answer" for the one case where the answer is exact.
    """
    result = _rare_outcome_drtmle_fit()
    baseline, difference = result["ey0"], result["ate"]
    # The nonzero witness: the geometry is real in this fit, not an artefact of a
    # constructed estimate. The point ratio stays inside the parameter space.
    assert difference.psi > 0
    assert difference.ci[0] < -baseline.psi < 0
    assert baseline.psi > baseline.std_error

    report = result.sensitivity.evalue("ate")
    assert report.risk_ratio == pytest.approx(1 + difference.psi / baseline.psi)
    assert report.point == pytest.approx(evalue_from_rr(report.risk_ratio))
    assert report.point > 1.0
    assert report.limit == 1.0
    assert report.risk_ratio_ci[0] == 0.0
    assert report.truncated_bound == pytest.approx(1 + difference.ci[0] / baseline.psi)
    assert report.risk_ratio_ci[1] == pytest.approx(1 + difference.ci[1] / baseline.psi)
    assert "outside the risk-ratio parameter space" in report.note
    assert "truncates it at 0" in report.note
    # Above the null the claim in the note is the one the ``limit`` field reports.
    assert report.risk_ratio_ci[0] <= 1.0 <= report.risk_ratio_ci[1]
    assert "the interval covers the null" in report.note


def test_a_protective_truncated_interval_keeps_its_confidence_limit_evalue():
    """The below-null half of the same truncation, where the interval excludes the null.

    A strong protective effect against a small reference risk sends the *lower* converted
    bound below zero while the upper bound stays under 1. The ratio interval is then
    ``[<0, <1]``, which excludes the null, and the confidence-limit E-value comes from the
    untruncated upper bound. A note that claims the interval covers the null either way
    contradicts the ``limit`` the same object reports.
    """
    result = _rare_outcome_drtmle_fit(seed=1, n=260, risk=0.12, treated_risk=0.02)
    baseline, difference = result["ey0"], result["ate"]
    # The nonzero witness: a real fit, on the protective side, with a truncated bound.
    assert difference.psi < 0
    assert difference.ci[0] < -baseline.psi < 0
    assert baseline.psi > baseline.std_error

    report = result.sensitivity.evalue("ate")
    assert report.risk_ratio == pytest.approx(1 + difference.psi / baseline.psi)
    assert report.risk_ratio < 1.0
    assert report.risk_ratio_ci[0] == 0.0
    assert report.truncated_bound == pytest.approx(1 + difference.ci[0] / baseline.psi)
    assert report.truncated_bound < 0.0
    # The upper bound is never truncated: ``normal_ci`` gives ``high >= psi`` and the
    # refusal above forces ``baseline.psi + psi > 0`` with ``baseline.psi > 0``.
    assert report.risk_ratio_ci[1] == pytest.approx(1 + difference.ci[1] / baseline.psi)
    assert 0.0 < report.risk_ratio_ci[1] < 1.0

    # The interval excludes the null, so the limit is a real number and not 1.
    assert report.limit == pytest.approx(evalue_from_rr(report.risk_ratio_ci[1]))
    assert report.limit > 1.0
    assert "the interval covers the null" not in report.note
    assert "reads the upper bound" in report.note


def test_the_truncated_bound_reaches_every_surface_that_reports_it():
    """``to_dict`` and the battery row must not present a fabricated 0 as a limit.

    ``summary()`` disclosed the truncation and the other two surfaces did not, so a caller
    who read only the mapping or only the report row saw ``0.0`` as a converted confidence
    limit.
    """
    result = _rare_outcome_drtmle_fit()
    report = result.sensitivity.evalue("ate")
    assert report.truncated_bound is not None

    payload = report.to_dict()
    assert payload["rr_ci_lower"] == 0.0
    assert payload["truncated_bound"] == pytest.approx(report.truncated_bound)
    assert payload["note"] == report.note
    assert payload["source_estimand"] == "ate"

    row = INTERPRETERS["evalue"](report, result, {})
    assert "truncated at 0" in row.detail

    # The control: a conversion that truncates nothing says nothing about truncation.
    plain = _fit(_study(), ATE()).sensitivity.evalue()
    assert plain.truncated_bound is None
    assert plain.to_dict()["truncated_bound"] is None
    assert "truncated" not in INTERPRETERS["evalue"](plain, None, {}).detail


@pytest.mark.parametrize(
    "baseline_psi,baseline_variance,difference_psi,expected",
    [
        (1e-12, None, 0.2643, "no stable denominator"),
        (0.02, 1e-6, -0.15, "nonpositive risk in the contrast arm"),
    ],
)
def test_a_fixed_baseline_outside_the_parameter_space_is_refused_by_name(
    baseline_psi, baseline_variance, difference_psi, expected
):
    """A positive sign is not a parameter-space guard.

    A baseline of ``1e-12`` divides into any ratio the caller cares to name, and a
    difference below ``-baseline`` implies a nonpositive risk in the contrast arm. Both
    passed the ``psi > 0`` test and were reported as ordinary approximate E-values.
    """
    result = _fit(_study(), ATE())
    baseline = replace(result["ate"], name="ey0", psi=baseline_psi)
    if baseline_variance is not None:
        baseline = replace(baseline, variance=baseline_variance)
    variant = replace(
        result,
        fitted_method="drtmle",
        estimates={
            **result.estimates,
            "ate": replace(result["ate"], psi=difference_psi),
            "ey0": baseline,
        },
        parameter_keys={},
    )
    with pytest.raises(CapabilityError, match=expected):
        variant.sensitivity.evalue("ate")
    # Refused while the branch is selected, which is what the battery reads to decide
    # availability, rather than raised after the row claims the analysis can run.
    with pytest.raises(CapabilityError, match=expected):
        _select_evalue(variant, "ate")


@pytest.mark.parametrize("guard", [(), ("Q", "g")])
def test_correction_participation_is_stamped_independently_of_extra(guard):
    from cleverly.estimators import DRTMLE

    frame, _ = make_binary_outcome(n=160, seed=3)
    result = (
        _raw(DRTMLE, guard=guard, estimands=("ate",), learner_folds=2)
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )
    changed = replace(result, extra={})
    assert changed.fitted_method == "drtmle"
    assert changed.solved_corrections == bool(guard)
    assert changed.diagnostics.capability("corrections").available == bool(guard)


@pytest.mark.parametrize(
    "method, options",
    [
        ("drtmle", {}),
        (CollaborativeTMLEMethod(), {}),
        ("tmle", {"cv_evaluation": True}),
    ],
)
def test_typed_ate_without_a_reported_baseline_refuses_variant_conversion(method, options):
    result = (
        _study()
        .identify(ATE())
        .estimate(
            method=method,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=3,
            simultaneous=False,
            **options,
        )
    )
    assert tuple(result.estimates) == ("ate",)
    with pytest.raises(CapabilityError, match="reported reference-arm mean"):
        result.sensitivity.evalue()


def test_repeated_exact_ratio_preserves_combination_and_retargets_each_repeat(monkeypatch):
    study = _study()
    options = {
        "repeats": 2,
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "learner_folds": 2,
        "random_state": 3,
        "simultaneous": False,
    }
    source = study.identify(ATE()).estimate(**options)
    expected = study.identify(RiskRatio()).estimate(**options)["rr"]
    calls = []
    original = source.estimator.retarget

    def tracked(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(source.estimator, "retarget", tracked)
    report = source.assess().report("evalue")
    derived = _derived_risk_ratio(source, "ate")
    assert len(calls) == 2
    assert derived.psi == pytest.approx(expected.psi, abs=1e-12)
    assert derived.variance == pytest.approx(expected.variance, abs=1e-12)
    np.testing.assert_allclose(derived.influence_curve, expected.influence_curve, atol=1e-12)
    assert report.risk_ratio_ci == pytest.approx(expected.ci)


def test_raw_multi_arm_labels_with_delimiters_route_forward():
    from cleverly.targets import parameter_name

    frame, _ = make_multi_arm(n=220, seed=3, family="binomial")
    labels = {"high": "a vs b", "low": "m[reference]", "medium": "z[vs]"}
    frame["A"] = frame["A"].map(labels)
    result = (
        _raw(estimands=("ate", "rr", "ey"), reference=labels["low"])
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )
    for value in (labels["high"], labels["medium"]):
        alias = parameter_name("rr", arm=value, versus=labels["low"])
        source = parameter_name("ate", arm=value, versus=labels["low"])
        report = result.sensitivity.evalue(source)
        assert report.estimand == alias
        assert report.risk_ratio == pytest.approx(result[alias].psi)


def test_real_tipping_search_retains_a_completed_none(tmp_path):
    from cleverly.datasets import make_missing_outcome

    frame, _ = make_missing_outcome(n=180, seed=3)
    result = (
        _raw(
            estimands=("ate",),
            missingness_learner=LogisticRegression(max_iter=1000),
        )
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )
    arguments = {"tipping_gamma": {"null_hypothesis": 10000.0, "search": (-0.01, 0.01)}}
    battery = result.assess(include_retargets=True, arguments=arguments)
    assert battery.sensitivity["tipping_gamma"].status is AssessmentStatus.COMPLETED
    assert battery.report("tipping_gamma") is None
    assert battery.sensitivity.reports()["tipping_gamma"] is None
    restored = load(result.save(tmp_path / "none-tip.joblib"))
    assert (
        restored.assess(include_retargets=True, arguments=arguments).report("tipping_gamma") is None
    )


def test_cached_evalue_refusals_store_data_without_exception_tracebacks(tmp_path):
    import joblib

    result = replace(_fit(_study(), ATE()), intermediate_value=0.0)
    facade = result.sensitivity
    capability = facade.capability("evalue")
    assert not capability.available
    assert facade._evalue_selections[None] == ("unavailable", capability.reason)
    path = tmp_path / "refused-facade.joblib"
    joblib.dump(facade, path)
    restored = joblib.load(path)
    assert restored.capability("evalue") == capability
    for candidate in (facade, restored):
        with pytest.raises(CapabilityError, match="controlled direct effects"):
            candidate.evalue()
        with pytest.raises(CapabilityError, match="controlled direct effects"):
            candidate.evalue("ate")
        assert all(isinstance(value, tuple) for value in candidate._evalue_selections.values())


@pytest.mark.parametrize("operation", ["evalue", "elements"])
def test_unavailable_evalue_capability_does_not_break_result_persistence(operation, tmp_path):
    result = replace(_fit(_study(), ATE()), estimator=None)
    capability = result.sensitivity.capability(operation)
    restored = load(result.save(tmp_path / f"capability-{operation}.joblib"))
    assert restored.sensitivity.capability(operation) == capability
    with pytest.raises(CapabilityError, match="reported reference-arm mean"):
        restored.sensitivity.evalue()
    restored_again = load(restored.save(tmp_path / f"refused-{operation}.joblib"))
    assert not restored_again.sensitivity.capability("evalue").available


@pytest.mark.parametrize("method", ["tmle", CollaborativeTMLEMethod()])
def test_a_method_outside_the_correction_system_refuses_corrections_by_name(method):
    """Both non-DR-TMLE methods refuse ``corrections()``, and say which system they are in.

    Only the ordinary-TMLE half was pinned. Collaborative TMLE stamps its own
    ``fitted_method``, reaches the same declaration by a different route, and would fail
    silently open if the row ever gained ``collaborative_tmle``.
    """
    result = (
        _study()
        .identify(ATE())
        .estimate(
            method=method,
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=3,
            simultaneous=False,
        )
    )
    expected = "collaborative_tmle" if isinstance(method, CollaborativeTMLEMethod) else "tmle"
    assert result.fitted_method == expected
    assert not result.solved_corrections

    capability = result.diagnostics.capability("corrections")
    assert not capability.available
    assert capability.status is AssessmentStatus.NOT_APPLICABLE
    with pytest.raises(CapabilityError, match="does not use the correction system"):
        result.diagnostics.corrections()

    # The refusal reaches the combined report rather than stopping it.
    assert result.assess().diagnostics["corrections"].status is AssessmentStatus.NOT_APPLICABLE


def test_the_battery_summary_prints_every_surface_and_both_verdict_lists():
    """``summary`` is the reader-facing surface of the whole battery, and had no test.

    A row that reaches ``attention`` or ``omissions`` must also be visible in its own
    section, and the section must carry the operation, the status, the detail, and the
    next step it recorded.
    """
    result = _fit(_study(), ATE())
    battery = result.assess()
    text = battery.summary()

    for surface in ("Validation", "Diagnostics", "Sensitivity"):
        assert surface in text
    for column in ("operation", "status", "detail", "next step"):
        assert column in text

    presented = [item for _, item in battery._presented()]
    assert presented
    for item in presented:
        assert item.name in text
        assert item.status.value in text

    # The two summary lists agree with the properties that build them, and the omission
    # list is not empty here, so "none" would be a wrong answer rather than an untested
    # one.
    assert battery.omissions
    assert f"Omissions: {', '.join(item.name for item in battery.omissions)}" in text
    attention = ", ".join(item.name for item in battery.attention) or "none"
    assert f"Attention: {attention}" in text


def test_next_steps_are_de_duplicated_and_keep_presentation_order():
    """``next_steps`` is what a reader acts on, and neither report class had a test.

    The battery reads three surfaces that repeat one another's advice, so the contract is
    ``dict.fromkeys`` order: every step appears once, in the order it was first presented.
    """
    result = _fit(_study(), ATE())
    battery = result.assess()

    presented = [step for _, item in battery._presented() for step in item.next_steps]
    assert presented, "the battery recorded no next step to de-duplicate"
    assert battery.next_steps() == tuple(dict.fromkeys(presented))

    diagnostics = result.diagnostics.run_all()
    flat = [step for item in diagnostics.items for step in item.next_steps]
    assert diagnostics.next_steps() == tuple(dict.fromkeys(flat))
    for step in diagnostics.next_steps():
        assert step in battery.next_steps()

    # The nonzero witness. This fit happens to record no step twice, so the equality above
    # holds for a report whose de-duplication never fires. Build one where it must: three
    # surfaces that repeat one another, which is the case the property exists for.
    from cleverly.assessment import DiagnosticReport, ValidationReport

    def item(name: str, *steps: str) -> AssessmentItem:
        return AssessmentItem(name, AssessmentStatus.WARNING, "detail", steps)

    repeated = AssessmentReport(
        validation=ValidationReport((item("support", "read support", "read the fit"),)),
        diagnostics=DiagnosticReport((item("truncation_curve", "read the fit", "read bounds"),)),
        sensitivity=DiagnosticReport((item("evalue", "read support", "read strength"),)),
    )
    flat_repeated = [step for _, row in repeated._presented() for step in row.next_steps]
    assert len(flat_repeated) > len(set(flat_repeated))
    assert repeated.next_steps() == (
        "read support",
        "read the fit",
        "read bounds",
        "read strength",
    )
    assert repeated.diagnostics.next_steps() == ("read the fit", "read bounds")


def test_a_stagewise_row_reports_its_two_metrics_on_the_support_scale():
    """``stagewise`` and ``support`` answer the same two questions, so they read alike.

    ``stagewise`` interpolated both numbers raw and printed ``0.8888888888888887`` beside
    a sibling row reading ``88.9%``, and ``None`` where the sibling prints nothing.
    """
    from cleverly.assessment import (
        LongitudinalDiagnostics,
        LongitudinalStageRow,
        _stagewise_item,
        _support_item,
    )

    def row(share: float, effective: float) -> LongitudinalStageRow:
        return LongitudinalStageRow(
            regimen="always",
            cause=None,
            horizon=None,
            time=0,
            n_followed=90,
            assignment=1.0,
            max_weight=2.0,
            effective_n=effective,
            share_truncated=share,
            epsilon=(),
            converged=True,
        )

    def diagnostics(*rows: LongitudinalStageRow) -> LongitudinalDiagnostics:
        return LongitudinalDiagnostics(rows, (), False, False, False)

    detail = _stagewise_item(diagnostics(row(0.0, 80.0), row(0.125, 72.0)), None).detail

    assert detail == (
        "2 stage row(s); maximum truncated fraction 12.5%; "
        "minimum effective-sample-size ratio 80.0%"
    )
    # The paired witness: the sibling row formats the same two numbers the same way, so
    # the two details cannot drift apart again without one of these strings changing.
    support = _support_item(
        SimpleNamespace(
            truncated={"fraction": 0.125},
            effective_sample_size={"ate": {"ratio": 0.8}},
            mechanisms={},
        ),
        None,
    )
    assert support.detail.startswith(
        "maximum truncated fraction 12.5%; minimum effective-sample-size ratio 80.0%"
    )
    # And an empty report says nothing rather than "None".
    assert _stagewise_item(diagnostics(), None).detail == "0 stage row(s)"


def test_an_unstamped_artifact_says_it_records_no_method():
    """The refusal must name the real cause, which is a missing stamp.

    ``the fitted method 'unknown' does not support this operation`` sent a reader looking
    for a method called ``unknown``. No such method exists. The artifact records none.
    """
    result = replace(_fit(_study(), ATE()), fitted_method="unknown")
    capability = result.diagnostics.capability("refute")

    assert not capability.available
    assert capability.reason == (
        "this artifact records no fitted method, so its support for this operation "
        "cannot be established"
    )

    # The paired witness: a stamped method that genuinely lacks the operation still names
    # itself, so the sentence above is about the missing stamp rather than about refusals.
    unknown_method = replace(_fit(_study(), ATE()), fitted_method="a_method_we_never_shipped")
    assert unknown_method.diagnostics.capability("refute").reason == (
        "the fitted method 'a_method_we_never_shipped' does not support this operation"
    )


def test_a_derived_result_starts_with_an_empty_cache_it_owns():
    """``dataclasses.replace`` must not hand a derived result the original's verdicts.

    The cache key records the operation and its arguments and nothing about the result
    that answered them, so an aliased mapping served a stale report under a new method
    stamp. ``attach_bootstrap`` is public and changes ``estimates``, which every
    sensitivity analysis reads.
    """
    result = _fit(_study(), ATE())
    before = result.validate()
    assert result.assessment_cache

    clone = replace(result, fitted_method="drtmle", solved_corrections=True)
    assert clone.assessment_cache is not result.assessment_cache
    assert clone.assessment_cache == {}
    assert clone.validate() is not before

    # Writing through the derived result does not reach the original.
    keys = set(result.assessment_cache)
    clone.sensitivity.run_all()
    assert set(result.assessment_cache) == keys
    assert set(clone.assessment_cache) - keys

    # The constructor refuses the argument outright rather than accepting and ignoring it.
    # ``replace`` raises ``ValueError`` before Python 3.13 and ``TypeError`` from 3.13 on,
    # so this pins the refusal rather than the class the interpreter chose for it.
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(result, assessment_cache={})


def test_persistence_still_restores_a_cache_the_constructor_would_not_accept(tmp_path):
    """The round trip does not go through ``__init__``, so a saved cache survives.

    The pair for the check above: making the cache constructor-free must not silently
    empty a restored result, which is the failure that would make every replay refit.
    """
    result = _fit(_study(), ATE())
    result.assess()
    assert len(result.assessment_cache) > 1

    restored = load(result.save(tmp_path / "cache.joblib"))
    assert set(restored.assessment_cache) == set(result.assessment_cache)


def test_a_validation_owned_argument_is_refused_rather_than_dropped():
    """The caller's own question must not be answered and then hidden.

    ``assess`` presents the validation row for the three names the validation battery
    owns. Forwarding ``arguments`` ran the caller's tolerance on the diagnostics side and
    then showed the argument-free answer, so a ``failed`` check never reached
    ``attention``.
    """
    result = _fit(_study(), ATE())

    # An empty mapping applies nothing, so nothing can be hidden and nothing is refused.
    for name in VALIDATION_OPERATIONS:
        assert result.assess(arguments={name: {}}).validation[name].name == name

    # ``support`` and ``nuisance_models`` take no parameter, so the remedy the refusal
    # would name is a call nobody can make. Signature binding says so precisely instead.
    for name in ("support", "nuisance_models"):
        assert not inspect.signature(getattr(type(result.diagnostics), name)).parameters.keys() - {
            "self"
        }
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            result.assess(arguments={name: {"tolerance": 1e-30}})

    strict = {"score_equations": {"tolerance": 1e-30}}
    with pytest.raises(CapabilityError, match="run_all"):
        result.assess(arguments=strict)

    # The nonzero witness: the refused tolerance changes the verdict, so the row that used
    # to be discarded was a failure rather than a second copy of the same answer.
    assert not result.diagnostics.score_equations(tolerance=1e-30).passed
    assert result.diagnostics.score_equations().passed

    # The named alternative answers the question the battery refused.
    assert result.diagnostics.run_all(arguments=strict)["score_equations"].status is (
        AssessmentStatus.FAILED
    )

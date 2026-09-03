"""Witnesses for the result-level post-fit assessment battery."""

from __future__ import annotations

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
from cleverly.assessment import ASSESSMENT_CAPABILITIES, INTERPRETERS, AssessmentItem
from cleverly.datasets import make_binary_outcome, make_multi_arm
from cleverly.sensitivity._derived import _derived_risk_ratio


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
    diagnostic = {row.operation for row in ASSESSMENT_CAPABILITIES}
    sensitivity = {
        "omitted_confounding",
        "robustness_value",
        "elements",
        "contour",
        "benchmark",
        "simulated_confounding",
        "evalue",
        "missingness",
        "tipping_gamma",
    }
    assert set(INTERPRETERS) == diagnostic | sensitivity


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
    changed = replace(result, assessment_cache={}, **change)
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
        assessment_cache={},
    )

    report = restored.sensitivity.evalue("ate")
    assert report.scale == "risk difference"
    assert report.approximate
    assert "holds that risk fixed" in report.note


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
        "simulated_confounding": SimpleNamespace(
            successful_cells=(SimpleNamespace(displacement=0.1),),
            cells=(SimpleNamespace(induced_treatment_association=0.2),),
            failures=(),
        ),
        "evalue": SimpleNamespace(point=2.0, limit=1.5, scale="risk ratio", approximate=False),
        "missingness": frame({"gamma": [0.5, 1.5], "psi": [0.9, 1.1]}),
        "tipping_gamma": None,
        "stagewise": SimpleNamespace(rows=()),
    }

    for operation, report in reports.items():
        assert INTERPRETERS[operation](report, None).status is AssessmentStatus.COMPLETED


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
    failed_surface = SimpleNamespace(
        successful_cells=(SimpleNamespace(displacement=0.1),),
        cells=(SimpleNamespace(induced_treatment_association=None),),
        failures=("cell refused",),
    )
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
    from cleverly.estimators import CTMLE, DRTMLE, TMLE

    engine = {"tmle": TMLE, "drtmle": DRTMLE, "ctmle": CTMLE, "cv": TMLE}[engine_name]
    frame, _ = make_binary_outcome(n=160, seed=3)
    options = {"cv_evaluation": True} if engine_name == "cv" else {}
    raw = (
        engine(
            estimands=("ate", "ey0"),
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=3,
            simultaneous=False,
            **options,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
        .single()
    )
    expected_method = {"ctmle": "collaborative_tmle", "cv": "tmle"}.get(engine_name, engine_name)
    assert raw.assessment_method == expected_method
    assert replace(raw, extra={}).assessment_method == expected_method
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
        detached = replace(raw, estimator=None, assessment_cache={})
        assert detached.sensitivity.evalue("ate") == before


@pytest.mark.parametrize("target", ["ate", "att", "atc"])
@pytest.mark.parametrize("typed", [False, True])
def test_gaussian_differences_use_the_documented_nonzero_conversion(target, typed):
    from cleverly import ATC, ATT
    from cleverly.datasets import make_linear_ate
    from cleverly.estimators import TMLE

    frame, _ = make_linear_ate(n=180, seed=3)
    options = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "random_state": 3,
        "simultaneous": False,
    }
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


def test_default_or_records_a_source_without_changing_replay_semantics():
    result = _fit(_study(), OddsRatio())
    battery = result.assess()
    row = battery.sensitivity["evalue"]
    assert row.arguments == {"estimand": None}
    assert row.report.source_estimand == "or"
    assert not result.sensitivity.evalue(**row.arguments).approximate


@pytest.mark.parametrize("target", [RiskRatio(), OddsRatio()])
def test_unstamped_artifact_keeps_reported_ratio_conversions(target):
    result = replace(_fit(_study(), target), fitted_method="unknown", assessment_cache={})
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
            assessment_cache={},
        )
        with pytest.raises(CapabilityError, match="finite positive reported reference-arm mean"):
            variant.sensitivity.evalue("ate")


@pytest.mark.parametrize("guard", [(), ("Q", "g")])
def test_correction_participation_is_stamped_independently_of_extra(guard):
    from cleverly.estimators import DRTMLE

    frame, _ = make_binary_outcome(n=160, seed=3)
    result = (
        DRTMLE(
            guard=guard,
            estimands=("ate",),
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=3,
            simultaneous=False,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )
    changed = replace(result, extra={}, assessment_cache={})
    assert changed.assessment_method == "drtmle"
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
    from cleverly.estimators import TMLE
    from cleverly.targets import parameter_name

    frame, _ = make_multi_arm(n=220, seed=3, family="binomial")
    labels = {"high": "a vs b", "low": "m[reference]", "medium": "z[vs]"}
    frame["A"] = frame["A"].map(labels)
    result = (
        TMLE(
            estimands=("ate", "rr", "ey"),
            reference=labels["low"],
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            random_state=3,
            simultaneous=False,
        )
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
    from cleverly.estimators import TMLE

    frame, _ = make_missing_outcome(n=180, seed=3)
    result = (
        TMLE(
            estimands=("ate",),
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            missingness_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            random_state=3,
            simultaneous=False,
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

    result = replace(_fit(_study(), ATE()), intermediate_value=0.0, assessment_cache={})
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
    result = replace(_fit(_study(), ATE()), estimator=None, assessment_cache={})
    capability = result.sensitivity.capability(operation)
    restored = load(result.save(tmp_path / f"capability-{operation}.joblib"))
    assert restored.sensitivity.capability(operation) == capability
    with pytest.raises(CapabilityError, match="reported reference-arm mean"):
        restored.sensitivity.evalue()
    restored_again = load(restored.save(tmp_path / f"refused-{operation}.joblib"))
    assert not restored_again.sensitivity.capability("evalue").available

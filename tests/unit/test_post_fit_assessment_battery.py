"""Witnesses for the result-level post-fit assessment battery."""

from __future__ import annotations

import copy
import inspect
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.sensitivity.positivity as positivity_module
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
    DiagnosticReport,
    ValidationReport,
)
from cleverly.datasets import make_binary_outcome, make_multi_arm
from cleverly.estimators import DRTMLE, TMLE
from cleverly.sensitivity._derived import _derived_risk_ratio
from cleverly.sensitivity.evalue import _select_evalue, _standardising_sd, evalue_from_rr
from cleverly.sensitivity.positivity import PositivityReport
from cleverly.validation import RepeatSpreadRow
from cleverly.validation.nuisance import SPREAD_SINGLE_DRAW


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
    assert bare.diagnostics["truncation_curve"].status is AssessmentStatus.DEFERRED
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


def test_multi_arm_evalue_defers_an_ambiguous_default(typed_multi_arm_result) -> None:
    result = typed_multi_arm_result
    aliases = tuple(result.estimates)
    assert len(aliases) == 2
    capability = result.sensitivity.capability("evalue")
    # The row says what the caller will meet. It cannot run as asked, it says why, and it
    # names the argument that makes it run. Advertising ``available=True`` beside the
    # refusal sentence invited the bare call the next line shows raising.
    assert not capability.available
    assert capability.status is AssessmentStatus.DEFERRED
    assert capability.reason and "choose an explicit estimand" in capability.reason
    assert capability.requires_arguments == ("estimand",)
    with pytest.raises(CapabilityError, match="choose an explicit estimand") as caught:
        result.sensitivity.evalue()
    assert all(alias in str(caught.value) for alias in aliases)
    row = result.sensitivity.run_all(include_retargets=True)["evalue"]
    assert row.status is AssessmentStatus.DEFERRED
    assert row.arguments == {}
    assert row.next_steps == ("call result.sensitivity.evalue() directly with estimand",)
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


def _positivity(fraction: float, ratio: float, **overrides: object) -> PositivityReport:
    """A real report carrying the two quantities the support row is judged on.

    A ``SimpleNamespace`` stood here, and it is why the aggregate/detail disagreement
    survived a green suite: the stub answered every attribute the interpreter happened to
    reach, so a test could pass an effective-sample-size ratio the production code never
    read and still assert a warning. The tier now comes from
    :meth:`~cleverly.PositivityReport.severity`, so the object under test has to be the one
    that owns it.
    """
    fields: dict[str, object] = {
        "propensity_quantiles": {"overall": {0.5: 0.5}},
        "tail_mass": {0.01: {"below": 0.0, "above": 0.0}},
        "effective_sample_size": {
            "treated": {"n": 100.0, "effective": 100.0 * ratio, "ratio": ratio}
        },
        "weight_share": {"treated": {"top_1pct": 0.0, "top_5pct": 0.0}},
        "truncated": {"count": 0.0, "fraction": fraction, "most_extreme": 0.01},
        "clever_covariate_max": {"mean": 1.0},
        "bounds": (0.01, 0.99),
        "n": 100,
    }
    fields.update(overrides)
    return PositivityReport(**fields)  # type: ignore[arg-type]


def test_warnings_preserve_score_and_support_measurements() -> None:
    support = _positivity(0.12, 0.16)
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


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize(
    "status", list(AssessmentStatus), ids=[row.value for row in AssessmentStatus]
)
def test_every_status_reaches_both_combined_frame_backends(
    backend: str, status: AssessmentStatus
) -> None:
    """Both frame backends render the machine-readable status, for every member.

    Written for ``DEFERRED`` alone, which said nothing about the other six and needed a
    second edit for the next one. The taxonomy is the parametrize axis, so a new member is
    covered on the day it is added. The second axis was ``module``, which never differed
    from ``backend``, so the backend name is read once.
    """
    from cleverly.assessment import DiagnosticReport

    report = DiagnosticReport(
        (AssessmentItem("benchmark", status, "a detail this test does not read"),),
        backend=backend,
    )
    frame = report.to_frame()

    assert type(frame).__module__.startswith(backend)
    assert list(frame["status"]) == [status.value]


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


def _simulated_cell(
    treatment_strength: float, fraction: float, *, association: float | None
) -> SimpleNamespace:
    return SimpleNamespace(
        treatment_strength=treatment_strength,
        outcome_strength=0.0,
        displacement=0.1,
        induced_treatment_association=association,
        target_population_fraction=fraction,
    )


def _simulated_interpreter_report(
    *,
    failures: tuple[str, ...] = (),
    perturbed_fraction: float = 0.3,
    population: str = "perturbed_treatment_group",
) -> SimpleNamespace:
    association = None if failures else 0.2
    cells = (
        _simulated_cell(0.0, 0.34, association=association),
        _simulated_cell(0.2, perturbed_fraction, association=association),
    )
    return SimpleNamespace(
        successful_cells=cells,
        cells=cells,
        failures=failures,
        movement_scale="estimate_difference",
        population=population,
        stratum=("low",),
        conditioning_arm=1,
        association_population="selected_baseline_stratum",
        calibration_population="full_fitted_population",
        population_lines=lambda: (
            f"target population: {population}",
            "conditioning arm: 1",
            "baseline stratum: ('low',)",
            "strata columns: ('V',)",
            "association population: selected_baseline_stratum",
            "calibration population: full_fitted_population",
            "refit population: full_fitted_population",
        ),
    )


def test_a_baseline_surface_reports_no_population_collapse() -> None:
    """The collapse rule reads ``population`` and not the fractions alone.

    A real baseline surface reports a fraction of one in every cell, so ``0.05`` against
    an anchor of ``0.34`` is unreachable there by construction. That is why the
    ``population`` conjunct is unreachable insurance, and it is also why no fit can
    witness it. This stand-in carries the collapsing fractions on a ``"baseline"`` report,
    so it is the one shape that separates the conjunct from its absence.
    """
    report = _simulated_interpreter_report(perturbed_fraction=0.05, population="baseline")
    item = INTERPRETERS["simulated_confounding"](report, None)

    assert item.status is AssessmentStatus.COMPLETED
    assert item.next_steps == ()
    # The fractions still reach the detail line, so the row hides nothing; only the
    # verdict is withheld.
    assert "minimum target population fraction 0.05 against anchor 0.34" in item.detail
    assert "target population: baseline" in item.detail


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
    # The population fields are the surface's own ``population_lines()``, joined.
    assert "target population: perturbed_treatment_group" in item.detail
    assert "conditioning arm: 1; baseline stratum: ('low',)" in item.detail
    assert "association population: selected_baseline_stratum" in item.detail
    assert "calibration population: full_fitted_population" in item.detail
    assert "refit population: full_fitted_population" in item.detail
    assert "minimum target population fraction 0.3 against anchor 0.34" in item.detail


def test_the_truncation_row_summarizes_each_parameter_against_its_fitted_result() -> None:
    frame = pd.DataFrame(
        {
            "bound": [0.01, 0.1, 0.01, 0.1],
            "upper_bound": [0.99, 0.9, 0.99, 0.9],
            "estimand": ["msm[(intercept)]", "msm[(intercept)]", "msm[dose]", "msm[dose]"],
            "psi": [1.0, 1.0, 2.5, 1.5],
            "fitted_lower_bound": [0.025] * 4,
            "fitted_upper_bound": [0.975] * 4,
            "fitted_psi": [1.0, 1.0, 2.0, 2.0],
            "delta_from_fitted": [0.0, 0.0, 0.5, -0.5],
            "is_fitted_bound": [False] * 4,
        }
    )

    item = INTERPRETERS["truncation_curve"](frame, None)

    assert item.status is AssessmentStatus.COMPLETED
    # The intercept did not move and the slope moved by half a unit. Pooling the two
    # parameters would give both of them the slope's range, so the two ranges differ here.
    assert item.detail == (
        "2 parameter(s) over evaluated lower bounds [0.01, 0.1]; signed movement from the "
        "fitted estimate: msm[(intercept)] [0, 0], msm[dose] [-0.5, 0.5]; "
        "fitted pair not evaluated for 2 of 2"
    )


def test_the_truncation_row_counts_only_the_parameters_that_skipped_their_fitted_pair() -> None:
    """The clause is a count, so one evaluated parameter has to change it."""
    frame = pd.DataFrame(
        {
            "bound": [0.025, 0.1, 0.025, 0.1],
            "estimand": ["ate", "ate", "ey1", "ey1"],
            "psi": [1.0, 1.2, 2.0, 2.1],
            "delta_from_fitted": [0.0, 0.2, 0.0, 0.1],
            "is_fitted_bound": [True, False, False, False],
        }
    )

    item = INTERPRETERS["truncation_curve"](frame, None)

    assert "fitted pair not evaluated for 1 of 2" in item.detail

    evaluated = frame.assign(is_fitted_bound=[True, False, True, False])
    assert "not evaluated" not in INTERPRETERS["truncation_curve"](evaluated, None).detail


def test_the_truncation_row_counts_a_skipped_fitted_pair_without_the_deltas() -> None:
    """The markers decide the clause, and the deltas are a separate column.

    ``truncation_curve`` writes both columns today, so this frame shape is latent. The
    rule the row states is about the markers alone, so a frame that carries them and no
    ``delta_from_fitted`` still gets the count.
    """
    frame = pd.DataFrame(
        {
            "bound": [0.025, 0.1, 0.025, 0.1],
            "estimand": ["ate", "ate", "ey1", "ey1"],
            "psi": [1.0, 1.2, 2.0, 2.1],
            "is_fitted_bound": [True, False, False, False],
        }
    )

    item = INTERPRETERS["truncation_curve"](frame, None)

    assert item.detail == (
        "2 parameter(s) over evaluated lower bounds [0.025, 0.1]; estimate range: "
        "ate [1, 1.2], ey1 [2, 2.1]; fitted pair not evaluated for 1 of 2"
    )
    # The control: readable markers that name no skipped parameter write no clause.
    evaluated = frame.assign(is_fitted_bound=[True, False, True, False])
    assert "not evaluated" not in INTERPRETERS["truncation_curve"](evaluated, None).detail


# The characters the compressed row can spend before it names a parameter. The four
# pieces are the count and its label (42 plus the digits), the bound range (26 at the
# widest ``%.4g`` pair), the movement label (44), and the skipped-pair clause (42).
TRUNCATION_FIXED_BUDGET = 160
# What one more parameter can add: ``, alias [low, high]`` at the same widest endpoints.
TRUNCATION_PARAMETER_BUDGET = 30


def _truncation_budget(aliases: list[str]) -> int:
    """Characters the truncation row may spend on a curve over these parameters."""
    return TRUNCATION_FIXED_BUDGET + sum(
        len(alias) + TRUNCATION_PARAMETER_BUDGET for alias in dict.fromkeys(aliases)
    )


@pytest.mark.parametrize(
    ("aliases", "measured"),
    [
        (["ate", "ey1", "ey0"], 158),
        (
            [
                f"{estimand}{suffix}"
                for estimand in ("ate", "att", "ey_obs", "par")
                for suffix in ("", "[V='low']", "[V='high']")
            ],
            427,
        ),
    ],
    ids=["three-parameters", "twelve-parameters"],
)
def test_the_truncation_row_costs_a_bounded_number_of_characters_per_parameter(
    aliases: list[str], measured: int
) -> None:
    """One long detail sets the width of the whole section, because nothing wraps it.

    ``format_table`` sizes each column by its widest cell, so the combined report is as
    wide as this row plus about 106 columns of name, status, and next step. What is
    bounded is the per-parameter cost and not the row itself, because a stratified fit
    reports one parameter for each estimand and stratum and the row names every one of
    them. ``measured`` is the length each of these two frames produces: 158 characters
    over three parameters, and 427 over twelve.

    A real fit reaches the same size. A ``TMLE`` fit with
    ``estimands=("ate", "att", "ey_obs", "par")`` over one two-level stratum reports 12
    parameters. On data where the default bound grid moves the estimates, its detail
    measured 438 characters and its ``run_all`` summary line measured 544 columns. The
    per-parameter sentences this format replaced measured 1725 characters on that same
    curve.

    The compressed row drops the identity of a parameter that skipped its fitted pair. It
    states a count alone, so a reader who needs the names reads the retained curve.
    """
    values = [0.0008045, 2.821e-07, -0.0008042]
    frame = pd.DataFrame(
        {
            "bound": [bound for bound in (0.001, 0.2) for _ in aliases],
            "estimand": aliases * 2,
            "psi": [1.44 + index for index in range(2 * len(aliases))],
            "delta_from_fitted": [0.0] * len(aliases)
            + [values[index % 3] for index in range(len(aliases))],
            "is_fitted_bound": [True] * len(aliases) + [False] * len(aliases),
        }
    )

    detail = INTERPRETERS["truncation_curve"](frame, None).detail

    assert len(detail) == measured
    assert len(detail) <= _truncation_budget(aliases)


def test_the_missingness_row_measures_each_parameter_from_its_own_mar_estimate() -> None:
    """The tilt frame holds one row per ``(gamma, estimand)`` pair.

    ``ate`` falls to 0.2 and ``ey1`` rises to 2.0, so a minimum and a maximum taken over
    every row belong to two different parameters and describe neither.
    """
    frame = pd.DataFrame(
        {
            "gamma": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            "estimand": ["ate", "ey1"] * 3,
            "psi": [0.2, 2.0, 0.5, 1.5, 0.8, 1.0],
            "is_mar": [False, False, True, True, False, False],
        }
    )

    item = INTERPRETERS["missingness"](frame, None)

    assert item.status is AssessmentStatus.COMPLETED
    assert item.detail == (
        "gamma range [-1, 1]; 2 parameter(s); signed movement from the MAR estimate: "
        "ate [-0.3, 0.3], ey1 [-0.5, 0.5]"
    )
    # The pooled estimate range, which no parameter covers and the row no longer reports.
    assert "[0.2, 2]" not in item.detail


def test_the_missingness_row_reports_a_level_when_no_mar_estimate_is_retained() -> None:
    """An explicit gamma grid need not contain zero, so the baseline can be absent.

    The markers are readable here, so the row can say the baseline is not among them.
    """
    frame = pd.DataFrame(
        {
            "gamma": [0.5, 0.5, 1.5, 1.5],
            "estimand": ["ate", "ey1", "ate", "ey1"],
            "psi": [0.4, 1.6, 0.3, 1.9],
            "is_mar": [False, False, False, False],
        }
    )

    item = INTERPRETERS["missingness"](frame, None)

    assert item.detail == (
        "gamma range [0.5, 1.5]; 2 parameter(s); no MAR estimate retained; "
        "estimate range: ate [0.3, 0.4], ey1 [1.6, 1.9]"
    )


def test_the_missingness_row_claims_no_baseline_when_the_markers_are_absent() -> None:
    """A missing ``is_mar`` column is not evidence that the baseline is missing.

    This frame retains its ``gamma == 0`` rows and carries no marker column. Reading the
    absent column as an absent baseline would deny those rows, so the row states the
    level it can read and names no baseline at all.
    """
    frame = pd.DataFrame(
        {
            "gamma": [0.0, 0.0, 1.0, 1.0],
            "estimand": ["ate", "ey1", "ate", "ey1"],
            "psi": [0.5, 1.5, 0.8, 1.0],
        }
    )

    item = INTERPRETERS["missingness"](frame, None)

    assert item.detail == (
        "gamma range [0, 1]; 2 parameter(s); estimate range: ate [0.5, 0.8], ey1 [1, 1.5]"
    )
    assert "no MAR estimate retained" not in item.detail


def test_the_missingness_row_keeps_the_tilted_estimates_when_the_baseline_is_not_finite() -> None:
    """A marked row carrying no finite estimate is no more usable than an absent one.

    Every delta from a non-finite baseline is non-finite, so a movement clause would read
    ``no finite values`` while two finite tilted estimates sit in the frame. The row
    reports those estimates and says the baseline is not retained.
    """
    frame = pd.DataFrame(
        {
            "gamma": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            "estimand": ["ate"] * 3 + ["ey1"] * 3,
            "psi": [0.2, float("nan"), 0.8, 1.4, 1.5, 1.6],
            "is_mar": [False, True, False, False, True, False],
        }
    )

    item = INTERPRETERS["missingness"](frame, None)

    assert item.detail == (
        "gamma range [-1, 1]; 2 parameter(s); signed movement from the MAR estimate: "
        "ey1 [-0.1, 0.1]; no MAR estimate retained; estimate range: ate [0.2, 0.8]"
    )
    # The defect this replaces: the row said the sweep produced nothing at all.
    assert "ate no finite values" not in item.detail


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


def _nuisance_stub(**overrides: object) -> SimpleNamespace:
    """A nuisance report holding exactly the attributes ``_nuisance_item`` reads.

    A stub rather than a fitted report, because these cases are about the arithmetic the
    interpreter does over rows it is handed rather than about producing the rows.

    Every attribute the interpreter reads is present, including ``reported_repeat``. A
    stub that omits it reads as passing only while ``selection`` is ``None``, which is the
    one case that never reaches the draw suffix.
    """
    fields: dict[str, object] = {
        "findings": (),
        "models": (),
        "selection": None,
        "selection_omission": None,
        "repeat_spread": (),
        "repeat_spread_omission": None,
        "n_repeats": 3,
        "reported_repeat": 1,
    }
    return SimpleNamespace(**{**fields, **overrides})


def _real_selection() -> object:
    """A genuine :class:`~cleverly.estimators.CTMLESelection` over a two-candidate path.

    Constructed rather than stubbed so the rendered sentence comes from the shipped
    ``describe``. A stub with the same fields renders whatever the test writes down, which
    is how the two call sites drifted apart before ``describe`` existed.
    """
    from cleverly.estimators.ctmle import CTMLESelection
    from cleverly.learners.crossfit import Folds

    risks = np.asarray([1.0, 0.5])
    return CTMLESelection(
        strategy="greedy",
        preorder=None,
        estimand="ate",
        target_names=("ate",),
        loss="loglik",
        penalized=False,
        path=((), ("W1",)),
        n_steps=(1, 1),
        train_risk=risks,
        train_loss=risks,
        penalty=np.zeros(2),
        treatment_risk=risks,
        cv_risk=risks,
        selected=1,
        covariates=("W1",),
        folds=Folds(np.asarray([0, 1, 0, 1]), 2),
    )


def test_method_facts_do_not_invent_a_warning_threshold() -> None:
    """Facts, a draw, and a status: a descriptive ratio never becomes a warning."""
    selection = _real_selection()
    report = _nuisance_stub(
        models=(object(), object()),
        selection=selection,
        repeat_spread=(RepeatSpreadRow("ate", 3, 0.2, 0.1, 2.0),),
    )

    item = INTERPRETERS["nuisance_models"](report, None)
    assert item.status is AssessmentStatus.COMPLETED
    assert selection.describe() in item.detail
    assert "C-TMLE greedy selected candidate 2 of 2 for ate on draw 1 of 3" in item.detail
    assert "largest sd/se 2 for ate" in item.detail


@pytest.mark.parametrize(
    ("rows", "expected", "absent"),
    [
        pytest.param(
            (
                RepeatSpreadRow("ey0", 3, 0.0, 0.0, float("nan")),
                RepeatSpreadRow("ate", 3, 0.2, 0.1, 2.0),
            ),
            "largest sd/se 2 for ate",
            "largest sd/se nan",
            id="one row has no usable ratio",
        ),
        pytest.param(
            (
                RepeatSpreadRow("ey0", 3, 0.0, 0.0, float("nan")),
                RepeatSpreadRow("ey1", 3, 0.0, 0.0, float("nan")),
            ),
            "sd/se unavailable",
            "largest sd/se",
            id="no row has one",
        ),
    ],
)
def test_method_facts_choose_only_finite_split_ratios(
    rows: tuple[RepeatSpreadRow, ...], expected: str, absent: str
) -> None:
    detail = INTERPRETERS["nuisance_models"](_nuisance_stub(repeat_spread=rows), None).detail
    assert expected in detail
    assert absent not in detail
    assert f"split spread for {len(rows)} parameter(s) across 3 draws" in detail


def test_method_facts_carry_the_reason_no_split_spread_is_available() -> None:
    """The row states the cause rather than leaving the reader an empty tuple."""
    detail = INTERPRETERS["nuisance_models"](
        _nuisance_stub(repeat_spread_omission="the reason it is absent"), None
    ).detail
    assert "split spread unavailable: the reason it is absent" in detail

    # And an ordinary one-draw fit never prints its ordinary reason.
    ordinary = INTERPRETERS["nuisance_models"](
        _nuisance_stub(n_repeats=1, repeat_spread_omission=SPREAD_SINGLE_DRAW), None
    ).detail
    assert "split spread unavailable" not in ordinary


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


def test_the_battery_summary_expands_results_before_compact_checks_and_omissions():
    """Every status is visible, while only completed analyses expand to full text.

    The duplicate ``shared`` name is a nonzero witness for surface qualification. The
    long warning detail is a witness for compactness: it remains in the ledger but does
    not push returned results below review prose.
    """

    def item(name: str, status: AssessmentStatus, detail: str) -> AssessmentItem:
        return AssessmentItem(name, status, detail, (f"act on {name}",))

    battery = AssessmentReport(
        validation=ValidationReport(
            (
                item("shared", AssessmentStatus.FAILED, "failed detail"),
                item("score", AssessmentStatus.PASSED, "passed detail"),
            )
        ),
        diagnostics=DiagnosticReport(
            (
                item("warning", AssessmentStatus.WARNING, "warning detail " * 20),
                item("choose", AssessmentStatus.DEFERRED, "deferred detail " * 20),
                item("missing", AssessmentStatus.UNAVAILABLE, "unavailable detail"),
            )
        ),
        sensitivity=DiagnosticReport(
            (
                item("irrelevant", AssessmentStatus.NOT_APPLICABLE, "not applicable detail"),
                item("estimate", AssessmentStatus.COMPLETED, "completed detail"),
                item("shared", AssessmentStatus.UNAVAILABLE, "second unavailable detail"),
            )
        ),
    )

    text = battery.summary()
    results, remainder = text.split("\n\nChecks\n", maxsplit=1)
    checks, omissions = remainder.split("\n\nNot run\n", maxsplit=1)
    assert text.startswith("Returned results\n----------------\n")
    assert "surface" in results and "operation" in results and "result" in results
    assert "sensitivity" in results and "completed detail" in results
    assert "passed detail" not in text
    assert "warning detail" not in text
    assert "deferred detail" not in text

    expected_checks = {
        AssessmentStatus.FAILED: (1, "validation.shared"),
        AssessmentStatus.WARNING: (1, "diagnostics.warning"),
        AssessmentStatus.PASSED: (1, "validation.score"),
    }
    expected_omissions = {
        AssessmentStatus.DEFERRED: (1, ("diagnostics.choose",)),
        AssessmentStatus.UNAVAILABLE: (2, ("diagnostics.missing", "sensitivity.shared")),
        AssessmentStatus.NOT_APPLICABLE: (1, ("sensitivity.irrelevant",)),
    }
    for status, (_, qualified_name) in expected_checks.items():
        assert status.value in checks
        assert qualified_name in checks
    for status, (_, qualified_names) in expected_omissions.items():
        assert status.value in omissions
        for qualified_name in qualified_names:
            assert qualified_name in omissions
    assert "Full ledger: call to_frame(). Next steps: call next_steps()." in omissions
    assert "Retained payloads: call report(...)." in omissions
    expected_statuses = {status.value for status in (*expected_checks, *expected_omissions)}
    inventory_counts = {
        fields[0]: int(fields[1])
        for line in (checks + omissions).splitlines()
        if len(fields := line.split(maxsplit=2)) == 3 and fields[0] in expected_statuses
    }
    expected_counts = {
        status.value: count
        for status, (count, _) in {**expected_checks, **expected_omissions}.items()
    }
    assert inventory_counts == expected_counts
    assert 1 + sum(inventory_counts.values()) == len(battery._presented())

    frame = battery.to_frame()
    warning = frame.loc[frame["check"] == "warning"].iloc[0]
    assert warning["detail"] == "warning detail " * 20
    assert warning["next_steps"] == "act on warning"
    assert battery.omissions == (
        battery.diagnostics["choose"],
        battery.diagnostics["missing"],
        battery.sensitivity["irrelevant"],
        battery.sensitivity["shared"],
    )


def test_the_battery_summary_names_empty_result_check_and_omission_sections():
    no_review = AssessmentReport(
        ValidationReport(
            (AssessmentItem("analysis", AssessmentStatus.COMPLETED, "returned result"),)
        ),
        DiagnosticReport(()),
        DiagnosticReport(()),
    )
    no_results = AssessmentReport(
        ValidationReport((AssessmentItem("score", AssessmentStatus.FAILED, "outside tolerance"),)),
        DiagnosticReport(()),
        DiagnosticReport(()),
    )

    assert "\nChecks\n------\nnone\n" in no_review.summary()
    assert "\nNot run\n-------\nnone\n" in no_review.summary()
    assert no_results.summary().startswith("Returned results\n----------------\nnone\n")


def test_the_documented_seed_fit_puts_results_before_its_compact_review_inventory():
    battery = _fit(_study(), ATE()).assess()
    text = battery.summary()

    assert len(battery.to_frame()) == 16
    assert text.index("bias-adjusted interval") < text.index("validation.nuisance_models")
    assert "poorly calibrated" not in text
    assert "not run by default because it retargets the fit" not in text
    review = text.split("\n\nChecks\n", maxsplit=1)[1]
    assert max(len(line) for line in review.splitlines()) < 80


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
    support = _support_item(_positivity(0.125, 0.8), None)
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


def test_the_support_row_never_claims_a_pass() -> None:
    """Support has no pass criterion, so it must not report one.

    The effective-sample-size ratio is reported and never graded: a Kish ratio has no
    derived cutoff, so a threshold on it would present a house convention as a finding.
    What is left to grade is truncation, and this fit truncates almost nothing. The row
    therefore states the numbers and stops, exactly as the nuisance row does.
    """
    row = INTERPRETERS["support"](_positivity(0.005, 0.25), None)
    assert row.status is AssessmentStatus.COMPLETED
    assert row.status is not AssessmentStatus.PASSED
    assert "minimum effective-sample-size ratio 25.0%" in row.detail
    assert row.next_steps == ()


def test_the_effective_sample_size_ratio_is_reported_and_never_graded() -> None:
    """The witness for the rule itself, not for one point on it.

    Two reports that differ only in effective sample size, by two orders of magnitude,
    must reach the same status. Any threshold anyone reinstates on that ratio separates
    this pair and fails here. Both details still carry their own number, because
    reporting it is the whole obligation.
    """
    ample = INTERPRETERS["support"](_positivity(0.0, 0.99), None)
    threadbare = INTERPRETERS["support"](_positivity(0.0, 0.007), None)

    assert ample.status is threadbare.status is AssessmentStatus.COMPLETED
    assert "99.0%" in ample.detail
    assert "0.7%" in threadbare.detail


@pytest.mark.parametrize(
    ("fraction", "severity", "status"),
    [
        (0.0099, "adequate", AssessmentStatus.COMPLETED),
        (0.0101, "strain", AssessmentStatus.WARNING),
        (0.0499, "strain", AssessmentStatus.WARNING),
        (0.0501, "serious", AssessmentStatus.WARNING),
    ],
)
def test_the_support_row_grades_truncation_at_the_reports_boundaries(
    fraction: float, severity: str, status: AssessmentStatus
) -> None:
    """Truncation is graded, and the pairs straddle each boundary.

    Truncation is not a judgement about the data. It counts rows the fit clipped at a
    bound the caller configured, and a clipped row contributes extrapolation rather than
    data. The values sit either side of 1% and 5%, so a drifted threshold fails rather
    than passing on a midpoint.
    """
    report = _positivity(fraction, 0.9)
    assert report.severity == severity
    row = INTERPRETERS["support"](report, None)
    assert row.status is status
    if status is AssessmentStatus.WARNING:
        assert row.next_steps == ("inspect result.diagnostics.support()",)


def test_the_support_row_and_the_retained_verdict_cannot_disagree() -> None:
    """A deliberate-mutation control on the single tiering decision.

    Both consumers must move together. If the combined row ever re-derives its own
    thresholds again, it will keep answering ``completed`` while the patched report says
    otherwise, and this fails.
    """
    report = _positivity(0.0, 1.0)
    assert INTERPRETERS["support"](report, None).status is AssessmentStatus.COMPLETED

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            PositivityReport,
            "_verdict_parts",
            lambda _self: ("serious", "VERDICT: truncation is carrying this estimate. Mutated."),
        )
        mutated = INTERPRETERS["support"](report, None)
    assert mutated.status is AssessmentStatus.WARNING


def test_the_verdict_prose_states_the_ratio_in_every_tier() -> None:
    """The reader-facing half of the contract: ungraded is not unreported."""
    for fraction in (0.0, 0.02, 0.06):
        verdict = _positivity(fraction, 0.25).verdict()
        assert "Kish-equivalent weight count of 25%" in verdict
        assert "no threshold is applied because none is derived" in verdict


def _load(ratio: float, **overrides: float | str) -> dict[str, float | str]:
    """One exact-equation absolute-load concentration row."""
    return {
        "equation": "mean[1]",
        "n_total": 120.0,
        "n_targeted": 100.0,
        "effective": 100.0 * ratio,
        "targeted_ratio": ratio,
        "total_ratio": 100.0 * ratio / 120.0,
        "top_1pct": 0.3,
        "top_5pct": 0.5,
        "max_load": 9.0,
        "zero_load": 0.0,
        "lower_bound": 0.05,
        "upper_bound": 0.95,
        "clipped_count": 0.0,
        "clipped_fraction": 0.0,
        **overrides,
    }


#: One fitted factor row, so a report can carry a mechanism ratio beside its arm ratio.
_MECHANISM = {
    "min": 0.2,
    "q01": 0.3,
    "q05": 0.4,
    "median": 0.6,
    "ess_ratio": 0.7,
    "top_1pct": 0.1,
    "top_5pct": 0.2,
    "clipped": 0.0,
    "clipped_fraction": 0.0,
}


def test_the_support_row_does_not_pool_group_concentration_with_mechanism_ess() -> None:
    """Unlike units stay separate: score-load concentration is not mechanism ESS."""
    groups = {"mean": _load(0.44), "att": _load(0.2)}
    pooled = _positivity(
        0.0,
        0.9,
        clever_covariate_max={"mean": 1.0, "att": 4.0},
        group_leverage=groups,
        mechanisms={"P(Delta=1|A,W)": _MECHANISM},
    )
    without = replace(pooled, group_leverage={})
    fact = "minimum effective-sample-size ratio"

    detail = INTERPRETERS["support"](pooled, None).detail
    assert f"{fact} 70.0%" in detail
    assert "group load: att:mean[1] 20.0/100 Kish-equivalent" in detail
    assert f"{fact} 70.0%" in INTERPRETERS["support"](without, None).detail


def test_a_low_group_ratio_is_reported_beside_the_arm_share_and_never_graded() -> None:
    """The group share is a second sentence, not a second threshold.

    The pair differs by two orders of magnitude in the group ratio and by nothing else,
    so any cutoff anyone puts on that share separates them and fails here. Both verdicts
    still carry the arm sentence word for word, because the two are different weightings
    and the reader needs both.
    """
    ample = _positivity(0.0, 0.9, group_leverage={"mean": _load(0.99)})
    threadbare = _positivity(0.0, 0.9, group_leverage={"mean": _load(0.04)})

    assert ample.severity == threadbare.severity == "adequate"
    assert INTERPRETERS["support"](threadbare, None).status is AssessmentStatus.COMPLETED
    assert "Absolute-load concentration is greatest" in ample.verdict()
    assert "99.0 Kish-equivalent rows out of 100 score-mask rows (99%" in ample.verdict()
    assert "4.0 Kish-equivalent rows out of 100 score-mask rows (4%" in threadbare.verdict()
    for report in (ample, threadbare):
        assert "Kish-equivalent weight count of 90%" in report.verdict()
        assert "not estimator effective sample size" in report.verdict()


def test_the_group_table_renders_when_a_row_is_not_finite() -> None:
    """A group with no targeted rows stores ``nan``, and the summary still has to print.

    Built by hand rather than fitted, because a fit that targets a group and then weights
    none of its rows is not a case the fast tier can reach cheaply. The verdict drops the
    non-finite share instead of reporting it, which is the same rule the arm sentence
    follows.
    """
    empty = _load(float("nan"), equation="empty")
    report = _positivity(
        0.0, 0.9, clever_covariate_max={"mean": float("nan")}, group_leverage={"mean": empty}
    )
    summary = report.summary()
    header = next(line for line in summary.splitlines() if line.startswith("group "))

    assert header.index("Kish-equivalent rows") < header.index("max |w h|")
    assert "empty" in summary
    assert "Absolute-load concentration" not in report.verdict()


def test_a_report_built_without_the_group_table_still_lists_the_maxima() -> None:
    """The field is defaulted, so a hand-built report constructs and still shows the maxima.

    This is the *constructor* path and nothing more: ``default_factory`` fires normally
    here, so the attribute is present before any reader looks at it. It does not exercise
    backward compatibility, which is a different mechanism and is covered by
    ``test_a_report_pickled_before_the_group_table_existed_still_reads``. What it does
    check is that a report with no group table presents a reader with the maxima rather
    than with nothing where they used to be.
    """
    report = _positivity(0.0, 0.9, clever_covariate_max={"mean": 1.0, "att": 4.0})

    assert report.group_leverage == {}
    assert "max |clever covariate| (mean): 1" in report.summary()
    assert "max |clever covariate| (att): 4" in report.summary()
    assert "Kish-equivalent rows" not in report.summary()
    assert "Absolute-load concentration" not in report.verdict()


def _older_state(report: PositivityReport) -> dict[str, object]:
    """The instance dictionary a report pickled before ``group_leverage`` existed carries.

    An older pickle stores the fields the older class had, and no key of that name. This
    is that dictionary, taken from a real report so that every other field is a real
    value rather than a stand-in.
    """
    state = dict(report.__dict__)
    del state["group_leverage"]
    del state["group_leverage_omissions"]
    return state


def test_a_report_pickled_before_the_group_table_existed_still_reads() -> None:
    """Backward compatibility, through ``pickle`` rather than through the constructor.

    ``dataclasses`` **deletes** the class attribute for a ``default_factory`` field, so
    the default cannot stand behind an old pickle: the instance arrives with no
    ``group_leverage`` key and nothing on the class to fall back to, and every reader of
    the attribute raises ``AttributeError``. The first two assertions are that fact,
    stated so that a future change which gives the class a real attribute fails here and
    tells someone the mechanism moved. ``__setstate__`` is what closes the gap, and it is
    driven by ``dataclasses.fields``, so ``n_repeats`` and ``backend`` come back filled by
    the same pass.
    """
    report = _positivity(
        0.0,
        0.9,
        clever_covariate_max={"mean": 1.0, "att": 4.0},
        group_leverage={"mean": _load(0.5), "att": _load(0.3)},
    )
    state = _older_state(report)

    assert getattr(type(report), "group_leverage", None) is None
    raw = PositivityReport.__new__(PositivityReport)
    raw.__dict__.update(state)
    with pytest.raises(AttributeError, match="group_leverage"):
        getattr(raw, "group_leverage")  # noqa: B009 -- the raise is the point

    older = copy.copy(report)
    older.__dict__.pop("group_leverage")
    restored = pickle.loads(pickle.dumps(older))

    assert "group_leverage" not in state
    assert restored.group_leverage == {}
    assert restored.group_leverage_omissions == {}
    assert restored.n_repeats == 1 and restored.backend is None
    assert restored.severity == "adequate"
    assert "Absolute-load concentration" not in restored.verdict()
    # And the summary takes the fallback branch: the table is gone, the maxima are not.
    summary = restored.summary()
    assert "max |clever covariate| (mean): 1" in summary
    assert "max |clever covariate| (att): 4" in summary
    assert "Kish-equivalent rows" not in summary


def test_a_result_saved_before_the_group_table_existed_still_assesses(tmp_path) -> None:
    """The same restore over the real artifact, which is where an old report comes from.

    ``diagnostics.support()`` files its answer in ``assessment_cache``, and ``save()``
    joblib-pickles the whole result with that cache inside it. So a stored artifact
    written before this table existed carries exactly the report the test above builds by
    hand, and ``cleverly.load`` is the reader that meets it. The cached report is edited
    to drop the key rather than the class being rolled back, because the older class is
    not importable from here.
    """
    result = _fit(_study(), ATE())
    del result.diagnostics.support().__dict__["group_leverage"]
    del result.diagnostics.support().__dict__["group_leverage_omissions"]
    key = next(name for name in result.assessment_cache if "support" in name)
    assert "group_leverage" not in result.assessment_cache[key].__dict__

    restored = load(result.save(tmp_path / "older-result.joblib"))
    report = restored.diagnostics.support()

    assert report.group_leverage == {}
    assert report.severity in {"adequate", "strain", "serious"}
    assert "max |clever covariate| (mean): " in report.summary()
    assert "max |h|" not in report.summary()
    # The combined battery reads `severity` off the same object, which is the caller the
    # missing attribute broke.
    assert restored.assess().report("support").group_leverage == {}


def test_the_truncation_verdict_keeps_the_requested_estimand() -> None:
    """A binding bound changes the procedure, not the parameter it estimates."""
    verdict = _positivity(0.06, 0.9).verdict()

    assert "sensitive to this finite-sample regularisation" in verdict
    explanations = (
        positivity_module.__doc__,
        positivity_module.truncation_curve.__doc__,
        verdict,
    )
    for explanation in explanations:
        assert explanation is not None
        assert "does not change the requested estimand" in " ".join(explanation.split())
    assert "describing the region of overlap only" not in verdict


def _drtmle(**overrides: object):
    """A DR-TMLE fit at cheap explicit learners, guarded unless a caller says otherwise."""
    frame, _ = make_binary_outcome(n=260, seed=17)
    settings: dict[str, object] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(max_iter=1000),
        "estimands": ("ey1", "ey0", "ate"),
        "n_folds": 2,
        "random_state": 3,
        "simultaneous": False,
    }
    settings.update(overrides)
    return DRTMLE(**settings).fit(frame, outcome="Y", treatment="A").single()  # type: ignore[arg-type]


def test_a_guarded_drtmle_truncation_curve_is_declared_a_refit() -> None:
    """The cost label has to follow the work the method actually does.

    ``truncation_curve`` calls ``estimator.retarget`` once per bound. On a guarded
    DR-TMLE that reaches ``_solve_reduction``, and ``DRTMLE._reduction`` hands the
    alternation a closure that refits the reduced regressions at the swept bounds. One
    shared row called that a moderate retarget, so a caller who declined refits was given
    fit-cost work under a retarget permission.

    The unguarded fit is the other half of the witness. Without it, a branch applied to
    every DR-TMLE result would pass.
    """
    guarded = _drtmle()
    unguarded = _drtmle(guard=())
    assert guarded.solved_corrections
    assert not unguarded.solved_corrections

    row = guarded.diagnostics.capability("truncation_curve")
    assert (row.execution, row.cost) == ("refit", "expensive")
    assert row.requires_replay == "refit_nuisances"

    plain = unguarded.diagnostics.capability("truncation_curve")
    assert (plain.execution, plain.cost) == ("retarget", "moderate")
    assert plain.requires_replay == "retarget_cached_nuisances"


def test_a_guarded_drtmle_truncation_curve_asks_for_the_refit_flag() -> None:
    """The gate reads the execution class, so the corrected label must reroute the caller."""
    guarded = _drtmle().diagnostics.run_all(include_retargets=True)["truncation_curve"]
    assert guarded.status is AssessmentStatus.DEFERRED
    assert "refits nuisance models" in guarded.detail
    assert "include_refits=True" in guarded.detail

    unguarded = _drtmle(guard=()).diagnostics.run_all(include_retargets=True)["truncation_curve"]
    assert unguarded.status is AssessmentStatus.COMPLETED

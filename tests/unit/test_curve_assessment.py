"""The assessment of a fitted effect curve (P1 plan 3.5): rows, fold locality and readings."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly._curve_assessment import STABILITY_WARNING, CurveDiagnostics
from cleverly.assessment import CURVE_OPERATIONS, CURVE_REFUTE_REFUSAL
from cleverly.bases import Linear, Modifiers
from cleverly.exceptions import CapabilityError

MODIFIERS = Modifiers(["W1", "W2"])


def frame(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, (n, 2))
    a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w[:, 0])))
    y = rng.binomial(1, 0.3 + 0.1 * w[:, 1] + a * (0.15 + 0.1 * w[:, 0]))
    return pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})


def curve(data: pd.DataFrame, scale: str = "difference", basis: Any = None, **settings: Any) -> Any:
    study = CausalStudy(
        data, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    contrast = ConditionalContrast(
        MODIFIERS,
        scale=scale,  # type: ignore[arg-type]
        model=Linear(["W1", "W2"]) if basis is None else basis,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            study.identify(contrast)
            .estimate(
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(),
                n_folds=2,
                random_state=0,
                **settings,
            )
            .effect_curve()
        )


def ep_curve(sieve: Any) -> Any:
    """The EP curve of ``W1`` on a declared sieve."""
    from cleverly.bases import Modifiers
    from cleverly.methods import EPMethod

    study = CausalStudy(
        frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    method = EPMethod(sieve=sieve).with_overrides(
        outcome_learner=LogisticRegression(),
        treatment_learner=LogisticRegression(),
        random_state=0,
    )
    contrast = ConditionalContrast(Modifiers(["W1"], domains={"W1": (-1, 1)}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(contrast).estimate(method)


@pytest.fixture(scope="module")
def fitted() -> Any:
    return curve(frame())


def test_the_rows_are_the_curve_family(fitted: Any) -> None:
    rows = {row.operation: row for row in fitted.diagnostics.capabilities}
    assert set(CURVE_OPERATIONS) <= set(rows)
    assert all(rows[name].available for name in (*CURVE_OPERATIONS, "support"))
    assert not rows["refute"].available
    assert rows["refute"].refusal.alternative == (
        "Run refute on the fit that produced the curve, naming a coefficient such as "
        "'cate_projection[W2]'."
    )
    assert rows["refute"].refusal.reason is CURVE_REFUTE_REFUSAL.reason
    assert rows["calibration"].execution == "refit" and rows["calibration"].accepts_random_state


def _refute_messages(curve: Any) -> list[str]:
    """What the curve's ``refute`` row says, and what its refute and preview calls raise."""
    messages = [
        str(_raised(curve.diagnostics.refute)),
        str(_raised(curve.diagnostics.planned_refits)),
    ]
    refusal = curve.diagnostics.capability("refute").refusal
    return [*messages, f"{refusal.condition} {refusal.alternative}"]


@pytest.mark.parametrize(
    ("scale", "coefficient"),
    [("difference", "cate_projection[W2]"), ("log_ratio", "log_crr_projection[W2]")],
)
def test_a_curve_s_refute_row_names_one_of_its_own_coefficients(
    scale: str, coefficient: str
) -> None:
    """Nonzero witness: the named coefficient is this curve's, at its scale and basis.

    A constant example name (the record once named 'cate_projection[W1]' for every curve)
    fails both cases, since neither curve has that coefficient.
    """
    fitted = curve(frame(), scale=scale, basis=Linear(["W2"]))
    for message in _refute_messages(fitted):
        assert f"'{coefficient}'" in message
        assert "[W1]" not in message


def test_an_ep_curve_s_refute_row_redirects_to_the_same_basis_projection(
    sieve_curve: Any,
) -> None:
    """An EP fit has no coefficients, so the row names the projection on its basis."""
    for message in _refute_messages(sieve_curve):
        assert "model=curve.model.basis" in message
        assert "method='tmle'" in message
        assert "projection[" not in message


def test_a_held_out_fold_never_sees_its_own_outcomes(fitted: Any) -> None:
    """Flip the outcomes of fold 0: its out-of-fold predictions do not move.

    Mutation control: predicting fold 0 from a refit on every row moves them.
    """
    diagnostics = fitted.diagnostics
    predictions, folds = diagnostics._out_of_fold(2, 0)
    fit = fitted.fit
    held = folds == 0
    flipped = fit.data
    outcome = np.asarray(flipped.outcome, dtype=float).copy()
    outcome[held] = 1.0 - outcome[held]
    moved = replace(fitted, fit=_with_data(fit, replace(flipped, outcome=outcome)))
    again, _ = CurveDiagnostics(moved)._out_of_fold(2, 0)
    np.testing.assert_allclose(again[held], predictions[held], atol=1e-12)
    assert np.max(np.abs(again[~held] - predictions[~held])) > 1e-3
    whole = fit.estimator.refit(replace(flipped, outcome=outcome), random_state=0)
    frame_held = pd.DataFrame(np.asarray(flipped.covariates)[held], columns=flipped.covariate_names)
    leaked = whole.effect_curve().predict(frame_held)
    assert np.max(np.abs(leaked - predictions[held])) > 1e-3


class _Fit:
    """The producing fit, with its data replaced and everything else read through."""

    def __init__(self, fit: Any, data: Any) -> None:
        self._fit, self.data = fit, data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fit, name)


def _with_data(fit: Any, data: Any) -> Any:
    return _Fit(fit, data)


def test_the_calibration_is_the_least_squares_line_of_the_pseudo_outcome(fitted: Any) -> None:
    diagnostics = fitted.diagnostics
    report = diagnostics.calibration(n_folds=2, random_state=0)
    predictions, _ = diagnostics._out_of_fold(2, 0)
    _, _, d_t, d_r = diagnostics._pieces()[3]
    line = np.polyfit(predictions, d_t - d_r, 1)
    assert report.slope == pytest.approx(line[0], rel=1e-10)
    assert report.intercept == pytest.approx(line[1], abs=1e-12)
    table = report.to_frame()
    assert table["rows"].sum() == len(predictions) and len(table) == 5


def test_the_risk_comparison_names_its_comparators(fitted: Any) -> None:
    report = fitted.diagnostics.risk_comparison(n_folds=2, random_state=0)
    assert set(report.risks) == {"curve", "constant", "t_learner", "dr_learner"}
    assert report.omitted == ""
    ratio = curve(frame(), "log_ratio").diagnostics.risk_comparison(n_folds=2, random_state=0)
    assert set(ratio.risks) == {"curve", "constant", "t_learner"}
    assert "No DR-learner" in ratio.omitted


def test_stability_and_its_threshold(fitted: Any) -> None:
    report = fitted.diagnostics.stability(n_seeds=2, random_state=0)
    assert report.warning == (report.relative_sd > STABILITY_WARNING)
    assert report.alternative.startswith("none: a Linear basis")
    assert report.alternative_relative_sd is None


def test_modifier_stability_drops_each_modifier(fitted: Any) -> None:
    table = fitted.diagnostics.modifier_stability(n_folds=2, random_state=0).to_frame()
    assert list(table["dropped"]) == ["W1", "W2"]
    one = curve(frame(), basis=Linear(["W1"]))
    with pytest.raises(CapabilityError, match="fewer than two modifiers"):
        one.diagnostics.modifier_stability()


def test_a_curve_without_its_fit_refuses(fitted: Any) -> None:
    with pytest.raises(CapabilityError, match="keeps no producing fit"):
        replace(fitted, fit=None).diagnostics.calibration()


def test_every_reading_becomes_one_report_row(fitted: Any) -> None:
    """Each curve operation's report has an interpreter, and a warning reaches its status.

    The mutation control is the stability row: the same report with ``warning`` flipped
    must change status, so the row reads the flag rather than always completing.
    """
    from cleverly.assessment import INTERPRETERS, AssessmentStatus

    diagnostics = fitted.diagnostics
    reports = {
        "calibration": diagnostics.calibration(n_folds=2, random_state=0),
        "risk_comparison": diagnostics.risk_comparison(n_folds=2, random_state=0),
        "stability": diagnostics.stability(n_seeds=2, random_state=0),
        "modifier_stability": diagnostics.modifier_stability(n_folds=2, random_state=0),
        "targeting_curve": diagnostics.targeting_curve(n_half_samples=20, random_state=0),
    }
    assert set(reports) == set(CURVE_OPERATIONS)
    for name, report in reports.items():
        item = INTERPRETERS[name](report, fitted, {})
        assert item.name == name and item.detail
    stability = reports["stability"]
    calm = INTERPRETERS["stability"](replace(stability, warning=False), fitted, {})
    loud = INTERPRETERS["stability"](replace(stability, warning=True), fitted, {})
    assert calm.status is AssessmentStatus.COMPLETED
    assert loud.status is AssessmentStatus.WARNING and loud.next_steps


def test_the_tables_come_back_in_the_fit_backend() -> None:
    """A polars fit's curve readings emit polars, as every other result table does."""
    pl = pytest.importorskip("polars")
    polars_curve = curve(pl.from_pandas(frame()))
    assert polars_curve.backend == "polars"
    diagnostics = polars_curve.diagnostics
    tables = (
        diagnostics.calibration(n_folds=2, random_state=0).to_frame(),
        diagnostics.risk_comparison(n_folds=2, random_state=0).to_frame(),
        diagnostics.modifier_stability(n_folds=2, random_state=0).to_frame(),
    )
    assert all(isinstance(table, pl.DataFrame) for table in tables)


# ------------------------------------------------------------------ the facade surface

#: The curve operations' cheapest arguments.
CHEAP = {
    "calibration": {"n_folds": 2},
    "risk_comparison": {"n_folds": 2},
    "stability": {"n_seeds": 2},
    "modifier_stability": {"n_folds": 2},
    "targeting_curve": {"n_half_samples": 20},
}
SUMMARIES = ("support", "nuisance_models", "score_equations")


@pytest.fixture(scope="module")
def sieve_curve() -> Any:
    """The EP curve on a declared cosine sieve of size 2, whose B-spline match holds fewer."""
    from cleverly.bases import Cosine, Sieve

    return ep_curve(Sieve(Cosine(["W1"], size=3), size=2))


def _raised(call: Any, **arguments: Any) -> CapabilityError:
    with pytest.raises(CapabilityError) as caught:
        call(**arguments)
    return caught.value


def test_the_curve_is_a_diagnostics_facade_with_the_shared_gates(fitted: Any) -> None:
    from cleverly.assessment import DiagnosticsFacade, assessment_capabilities

    diagnostics = fitted.diagnostics
    assert isinstance(diagnostics, DiagnosticsFacade)
    assert diagnostics.capabilities == assessment_capabilities(fitted)
    rows = {row.operation: row for row in diagnostics.capabilities}
    assert {rows[name].requires_replay for name in SUMMARIES} == {"summarize_existing_artifacts"}
    assert {rows[name].requires_replay for name in CURVE_OPERATIONS} == {"refit_nuisances"}


def test_every_sensitivity_route_refuses_a_curve_by_row_and_by_call(
    fitted: Any, sieve_curve: Any
) -> None:
    """Every route refuses on both curve kinds, and each call raises the row's record."""
    from cleverly.assessment import SENSITIVITY_ROUTES

    for result in (fitted, sieve_curve):
        sensitivity = result.sensitivity
        assert {row.operation for row in sensitivity.capabilities} == set(SENSITIVITY_ROUTES)
        for operation in SENSITIVITY_ROUTES:
            row = sensitivity.capability(operation)
            assert not row.available
            assert row.refusal.roadmap != "P1"
            for arguments in ({}, {"estimand": "ate"}):
                error = _raised(getattr(sensitivity, operation), **arguments)
                assert row.refusal.condition in str(error)
                assert error.refusal.alternative == row.refusal.alternative


def _sensitivity_record(result: Any, operation: str) -> Any:
    return result.sensitivity.capability(operation).refusal


def test_a_projection_curve_redirects_where_its_fit_runs_and_carries_its_record_otherwise(
    fitted: Any,
) -> None:
    """A redirection names a call that runs; a refused fit row is carried, code and owner."""
    from cleverly.assessment import projection_sensitivity_refusal
    from cleverly.exceptions import RefusalReason

    assert fitted.fit.sensitivity._gated_map["omitted_confounding"].available
    redirected = _sensitivity_record(fitted, "omitted_confounding")
    assert redirected.reason is RefusalReason.API_REDIRECTION
    assert redirected.roadmap is None
    assert redirected.condition == (
        "A curve is computed from a fit and reports no estimand of its own. Run "
        "omitted_confounding on the fit that produced the curve."
    )
    assert redirected.alternative == (
        "Call it on the producing fit, naming a coefficient such as 'cate_projection[W2]'."
    )
    # The call it names runs on the producing fit.
    assert fitted.fit.sensitivity.omitted_confounding(estimand="cate_projection[W2]")
    carried = _sensitivity_record(fitted, "missingness")
    assert carried == fitted.fit.sensitivity._gated_map["missingness"].refusal
    assert carried == projection_sensitivity_refusal("missingness")
    assert (carried.reason, carried.roadmap) == (RefusalReason.IMPLEMENTATION_GAP, "X82")


def test_a_log_ratio_projection_curve_carries_its_fit_s_bound_refusal() -> None:
    """Nonzero witness for a bound row: the log-ratio fit refuses the bound, so no redirect."""
    from cleverly.exceptions import RefusalReason

    ratio = curve(frame(), scale="log_ratio")
    target = ratio.fit.sensitivity._gated_map["omitted_confounding"]
    assert not target.available
    record = _sensitivity_record(ratio, "omitted_confounding")
    assert record == target.refusal
    assert (record.reason, record.roadmap) == (RefusalReason.IMPLEMENTATION_GAP, "X82")


def test_an_ep_curve_redirects_to_the_projection_on_its_basis_or_carries_its_record(
    sieve_curve: Any,
) -> None:
    """The EP versions of both outcomes: the same-basis projection decides each row."""
    from cleverly.assessment import projection_sensitivity_refusal
    from cleverly.exceptions import RefusalReason

    redirected = _sensitivity_record(sieve_curve, "omitted_confounding")
    assert redirected.reason is RefusalReason.API_REDIRECTION
    assert redirected.roadmap is None
    assert redirected.condition == (
        "An EP curve has no influence curve, and omitted_confounding is defined for a fit that "
        "has one. The projection on the same basis, curve.model.basis, the basis the curve "
        "used at its size (3 terms), is the related estimand it is defined for."
    )
    assert redirected.alternative == (
        "Fit ConditionalContrast(..., model=curve.model.basis) with method='tmle' and run "
        "omitted_confounding on that fit."
    )
    carried = _sensitivity_record(sieve_curve, "missingness")
    r9 = projection_sensitivity_refusal("missingness")
    assert (carried.reason, carried.roadmap, carried.alternative) == (
        r9.reason,
        "X82",
        r9.alternative,
    )
    assert carried.condition == (
        "An EP curve has no influence curve. On the projection on the same basis, " + r9.condition
    )
    truncation = sieve_curve.diagnostics.capability("truncation_curve").refusal
    assert truncation.reason is RefusalReason.API_REDIRECTION
    assert truncation.condition.startswith(
        "The truncation curve reports an estimate and its interval at each bound, and an EP "
        "curve has no interval."
    )


def test_the_redirection_of_a_cv_sized_ep_curve_names_a_basis_tmle_fits() -> None:
    """A CV-sized curve's alternative names its fixed basis, so the call it names runs.

    Nonzero witness: the sieve the curve was declared with is refused by ``tmle`` (R4), and
    the basis the record names is not.
    """
    from cleverly.bases import Modifiers

    cv = ep_curve(None)
    assert cv.sieve.mode == "cv"
    record = _sensitivity_record(cv, "omitted_confounding")
    assert "model=curve.model.basis" in record.alternative
    assert f"({len(cv.model.terms)} terms)" in record.condition
    study = CausalStudy(
        frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    )
    modifiers = Modifiers(["W1"], domains={"W1": (-1, 1)})
    declared = study.identify(ConditionalContrast(modifiers)).available_methods()
    assert not {m.name: m for m in declared}["tmle"].available
    effect = study.identify(ConditionalContrast(modifiers, model=cv.model.basis))
    assert {m.name: m for m in effect.available_methods()}["tmle"].available
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        projection = effect.estimate(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            n_folds=2,
            random_state=0,
        )
    slope = list(projection.estimates)[-1]
    assert projection.sensitivity.omitted_confounding(estimand=slope) is not None
    assert projection.diagnostics.capability("truncation_curve").available


@pytest.mark.parametrize(
    "edits",
    [
        [("ep = _is_ep_curve(curve)", "ep = not _is_ep_curve(curve)")],
        [("if not target.available:", "if False:")],
    ],
    ids=["inverted_ep_predicate", "always_redirect"],
)
def test_each_resolver_mutation_fails_a_witness(
    monkeypatch: pytest.MonkeyPatch, fitted: Any, sieve_curve: Any, edits: Any
) -> None:
    """Mutation controls: each broken resolver changes a record the witnesses pin."""
    import cleverly.assessment as assessment
    from tests.unit._source_mutation import mutate

    expected = {
        (id(result), operation): _sensitivity_record(result, operation)
        for result in (fitted, sieve_curve)
        for operation in ("omitted_confounding", "missingness")
    }
    mutate(monkeypatch, assessment, assessment, "curve_sensitivity_record", edits)
    moved = [
        key
        for key, record in expected.items()
        for result in (fitted, sieve_curve)
        if key[0] == id(result)
        and assessment.curve_sensitivity_rows(result)[
            [row.operation for row in assessment._STRUCTURAL_SENSITIVITY].index(key[1])
        ].refusal
        != record
    ]
    assert moved


def test_assess_composes_the_three_surfaces(fitted: Any) -> None:
    from cleverly.assessment import AssessmentStatus

    report = fitted.assess(include_refits=True, arguments=CHEAP, random_state=0)
    assert [item.name for item in report.validation.items] == [
        "score_equations",
        "support",
        "nuisance_models",
    ]
    ran = {item.name: item for item in report.diagnostics.items}
    assert all(ran[name].status is not AssessmentStatus.UNAVAILABLE for name in CURVE_OPERATIONS)
    assert ran["calibration"].report.n_folds == 2
    assert all(item.status is AssessmentStatus.UNAVAILABLE for item in report.sensitivity.items)
    # A direct call reads the answer the battery filed.
    assert fitted.diagnostics.calibration(n_folds=2, random_state=0) is ran["calibration"].report
    skipped = {item.name: item for item in fitted.assess().diagnostics.items}
    assert all(skipped[name].status is AssessmentStatus.DEFERRED for name in CURVE_OPERATIONS)


def test_an_ep_curve_score_row_completes_with_its_debiasing_scores(sieve_curve: Any) -> None:
    """The EP debiasing scores have no derived tolerance, so the row completes.

    The detail carries the stored largest score after the regression; the generic score
    path would read ``failed`` with no score rows.
    """
    from cleverly.assessment import AssessmentStatus

    scores = sieve_curve.diagnostics.score_equations()
    item = sieve_curve.validate()["score_equations"]
    assert item.status is AssessmentStatus.COMPLETED
    assert f"{scores.max_after:.3g}" in item.detail
    assert f"{len(scores.features)} debiasing feature(s)" in item.detail


def test_a_curve_without_its_fit_refuses_every_reading_by_its_replay_record(fitted: Any) -> None:
    """With no producing fit every reading row refuses, and each call raises that record.

    The nonzero witness is the kept fit: the same rows are available and a summary runs.
    A fit whose estimator is removed keeps its summaries and refuses its refits.
    """
    from cleverly.assessment import CURVE_REPLAY_NO_ESTIMATOR, CURVE_REPLAY_NO_FIT

    kept = fitted.diagnostics
    assert all(kept.capability(name).available for name in (*SUMMARIES, *CURVE_OPERATIONS))
    assert kept.support() is not None
    bare = replace(fitted, fit=None)
    assert bare.replayability.unreconstructible == (CURVE_REPLAY_NO_FIT,)
    for name in (*SUMMARIES, *CURVE_OPERATIONS):
        row = bare.diagnostics.capability(name)
        assert not row.available, name
        assert CURVE_REPLAY_NO_FIT in row.refusal.condition
        assert "effect_curve()" in row.refusal.alternative
        error = _raised(getattr(bare.diagnostics, name), **CHEAP.get(name, {}))
        assert row.refusal.condition in str(error) and error.refusal.alternative == (
            row.refusal.alternative
        )
    unfitted = replace(fitted, fit=replace(fitted.fit, estimator=None))
    assert unfitted.replayability.unreconstructible == (CURVE_REPLAY_NO_ESTIMATOR,)
    diagnostics = unfitted.diagnostics
    assert all(diagnostics.capability(name).available for name in SUMMARIES)
    assert diagnostics.score_equations().passed
    for name in CURVE_OPERATIONS:
        row = diagnostics.capability(name)
        assert not row.available and CURVE_REPLAY_NO_ESTIMATOR in row.refusal.condition
        error = _raised(getattr(diagnostics, name), **CHEAP[name])
        assert row.refusal.condition in str(error)


def test_a_projection_refit_preflight_reaches_the_curve_rows(
    fitted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producing fit whose estimator refit() refuses refuses the curve's refits too."""
    from cleverly.assessment import POINT_REPLAY_REFIT_CONFIGURATION

    estimator = type(fitted.fit.estimator)
    monkeypatch.setattr(
        estimator, "_refit_configuration_refusal", lambda self, data: "refused", raising=False
    )
    fresh = replace(fitted)
    assert fresh.replayability.unreconstructible == (POINT_REPLAY_REFIT_CONFIGURATION,)
    rows = {row.operation: row for row in fresh.diagnostics.capabilities}
    assert all(rows[name].available for name in SUMMARIES)
    assert not any(rows[name].available for name in CURVE_OPERATIONS)


def _rule_witness(result: Any, operation: str, arguments: dict[str, Any]) -> None:
    row = result.diagnostics.capability(operation)
    assert not row.available, operation
    error = _raised(getattr(result.diagnostics, operation), **arguments)
    assert error.refusal == row.refusal, operation


def test_each_fit_wide_rule_refuses_the_row_and_the_call_with_one_record(
    fitted: Any, sieve_curve: Any
) -> None:
    """Each rule's witness curve: the row and the call refuse with the call's own record.

    The control is the two-modifier, non-constant, single-split projection, where each row
    is available.
    """
    from cleverly.targeting_curves import TARGETING_REPEATS_REFUSAL

    rows = {row.operation: row for row in fitted.diagnostics.capabilities}
    assert all(rows[name].available for name in CURVE_OPERATIONS)
    one = curve(frame(), basis=Linear(["W1"]))
    _rule_witness(one, "modifier_stability", CHEAP["modifier_stability"])
    assert "fewer than two modifiers" in one.diagnostics.capability("modifier_stability").reason
    constant = replace(fitted, in_sample=np.full(fitted.in_sample.size, 0.1))
    _rule_witness(constant, "stability", CHEAP["stability"])
    assert "spread is zero" in constant.diagnostics.capability("stability").reason
    _rule_witness(sieve_curve, "stability", CHEAP["stability"])
    assert "sieve declares 2 terms" in sieve_curve.diagnostics.capability("stability").reason
    _rule_witness(sieve_curve, "modifier_stability", CHEAP["modifier_stability"])
    repeated = curve(frame(), repeats=2)
    _rule_witness(repeated, "targeting_curve", CHEAP["targeting_curve"])
    assert repeated.diagnostics.capability("targeting_curve").refusal is TARGETING_REPEATS_REFUSAL
    assert repeated.diagnostics.capability("calibration").available


def test_an_ep_curve_whose_matched_family_holds_its_size_keeps_stability() -> None:
    """The sieve rule's control: a size the matched B-spline family holds keeps the row."""
    from cleverly.bases import Cosine, Sieve

    small = ep_curve(Sieve(Cosine(["W1"], size=4), size=1))
    assert small.diagnostics.capability("stability").available
    assert small.diagnostics.stability(n_seeds=2, random_state=0).alternative.startswith("BSpline")


def test_the_request_gates_follow_the_calls(fitted: Any) -> None:
    """``n_folds``, ``split`` and ``compare`` resolve the row as the call raises."""
    from cleverly.targeting_curves import TARGETING_SPLIT_REFUSAL

    diagnostics = fitted.diagnostics
    many = diagnostics._capability_for_arguments("calibration", {"n_folds": 10_000})
    assert not many.available and "cannot fill 10000 folds" in many.reason
    assert diagnostics._capability_for_arguments("calibration", {"n_folds": 2}).available
    with pytest.raises(CapabilityError, match="cannot fill 10000 folds"):
        diagnostics.calibration(n_folds=10_000)
    fold_average = diagnostics._capability_for_arguments("targeting_curve", {"split": 1})
    assert fold_average.refusal is TARGETING_SPLIT_REFUSAL
    assert _raised(diagnostics.targeting_curve, split=1).refusal is TARGETING_SPLIT_REFUSAL
    tiny = diagnostics._capability_for_arguments("targeting_curve", {"split": 0.001})
    assert not tiny.available and "units to train on" in tiny.reason
    engine = replace(fitted, identified_effect=None).diagnostics
    compared = engine._capability_for_arguments("targeting_curve", {"compare": "W2"})
    assert not compared.available and "fitted by the engine" in compared.reason
    assert engine.capability("targeting_curve").available


def test_influence_refuses_on_a_curve(fitted: Any, sieve_curve: Any) -> None:
    from cleverly.exceptions import RefusalReason

    projection = _raised(fitted.diagnostics.influence)
    assert projection.refusal.reason is RefusalReason.API_REDIRECTION
    assert "curve.fit.diagnostics.influence()" in projection.refusal.alternative
    ep = _raised(sieve_curve.diagnostics.influence)
    assert ep.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_without_rebuilds_a_tensor_and_an_interaction_basis() -> None:
    """Dropping a modifier from a product or interaction basis keeps the rest of the model.

    Before the fix, both bases went through ``dataclasses.replace(basis, columns=...)`` and
    raised ``TypeError``, while the ``modifier_stability`` row read available.
    """
    from cleverly._curve_assessment import _without, _without_refusal
    from cleverly.bases import Additive, Cosine, Interactions, Tensor

    product = Tensor(Linear(["W1"]), Cosine(["W2"], 2))
    assert _without(product, "W1") == Cosine(["W2"], 2)
    assert _without(product, "W2") == Linear(["W1"])
    wide = Tensor(Linear(["W1", "W2"]), Cosine(["W1"], 2))
    assert _without(wide, "W1") == Linear(["W2"])
    assert _without(wide, "W2") == Tensor(Linear(["W1"]), Cosine(["W1"], 2))
    three = Interactions(Additive(Linear(["W1"]), Linear(["W2"]), Linear(["W3"])), 2)
    assert _without(three, "W1") == Interactions(Additive(Linear(["W2"]), Linear(["W3"])), 2)
    two = Interactions(Additive(Linear(["W1"]), Linear(["W2"])), 2)
    reduced = _without(two, "W1")
    assert isinstance(reduced, Additive) and reduced.columns == ("W2",)
    # A product whose two factors read the modifier alone keeps nothing without it.
    alone = Tensor(Linear(["W1"]), Cosine(["W1"], 2))
    assert "both factors" in _without_refusal(alone, "W1").condition
    with pytest.raises(CapabilityError, match="both factors"):
        _without(alone, "W1")


def test_modifier_stability_runs_on_a_tensor_basis_and_refuses_a_user_basis() -> None:
    """The row follows the call on both: a product runs, a written function refuses.

    The tensor curve is the nonzero witness that the fix reaches the refit: each dropped
    modifier gives a finite out-of-fold risk.
    """
    from cleverly.bases import Tensor, UserBasis

    product = curve(frame(), basis=Tensor(Linear(["W1"]), Linear(["W2"])))
    assert product.diagnostics.capability("modifier_stability").available
    table = product.diagnostics.modifier_stability(n_folds=2, random_state=0).to_frame()
    assert list(table["dropped"]) == ["W1", "W2"]
    assert np.isfinite(table.select_dtypes("number").to_numpy()).all()
    written = UserBasis(
        lambda rows: np.column_stack(
            [np.ones_like(np.asarray(rows["W1"], float)), np.asarray(rows["W1"] * rows["W2"])]
        ),
        terms=("(intercept)", "W1*W2"),
        columns=("W1", "W2"),
        basis_kind="known",
    )
    user = curve(frame(), basis=written)
    _rule_witness(user, "modifier_stability", CHEAP["modifier_stability"])
    assert (
        "UserBasis evaluates one function"
        in user.diagnostics.capability("modifier_stability").reason
    )


GROUPS = ("home", "alone", "carer")


def text_group_frame(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """A group column stored as text, whose levels carry effects 0.05, 0.35 and -0.15."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, n)
    group = rng.choice(np.array(GROUPS, dtype=object), size=n)
    effect = pd.Series(group).map({"home": 0.05, "alone": 0.35, "carer": -0.15}).to_numpy()
    a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * w)))
    y = rng.binomial(1, 0.4 + 0.1 * w + a * effect)
    return pd.DataFrame({"Y": y, "A": a, "W1": w, "G": group})


def text_group_curve(data: pd.DataFrame) -> Any:
    from cleverly.bases import Additive, Indicators

    study = CausalStudy(
        data, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "G"))
    )
    contrast = ConditionalContrast(
        Modifiers(["G", "W1"], levels={"G": GROUPS}),
        model=Additive(Indicators("G"), Linear(["W1"])),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = study.identify(contrast).estimate(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            n_folds=2,
            random_state=0,
        )
    return fit.effect_curve()


@pytest.fixture(scope="module")
def text_group() -> tuple[pd.DataFrame, Any]:
    data = text_group_frame()
    return data, text_group_curve(data)


def _held_out_matches_the_raw_rows(data: pd.DataFrame, fitted: Any) -> bool:
    """Whether each out-of-fold prediction is the fold refit's curve at the caller's own row.

    The expected values read the text column the caller passed, not the indicators the
    data layer made of it, so a frame that rebuilds the levels wrongly disagrees.
    """
    diagnostics = fitted.diagnostics
    predictions, folds = diagnostics._out_of_fold(2, 0)
    fit = fitted.fit
    expected = np.empty(len(predictions))
    for k in range(2):
        test = folds == k
        refit = fit.estimator.refit(fit.data.subset(np.flatnonzero(~test)), random_state=k)
        expected[test] = refit.effect_curve().predict(data.loc[test, ["G", "W1"]])
    return bool(np.allclose(predictions, expected, rtol=0.0, atol=1e-12))


def test_the_held_out_checks_run_on_a_group_column_stored_as_text(
    text_group: tuple[pd.DataFrame, Any],
) -> None:
    """Every refit reading of a text-coded ``Indicators`` curve runs, where it used to raise.

    The covariate matrix holds the group as indicators, so a frame built from it alone
    lacks the column the basis reads, and each call raised ``DataError``.
    """
    from cleverly.assessment import AssessmentStatus

    _, fitted = text_group
    diagnostics = fitted.diagnostics
    calibration = diagnostics.calibration(n_folds=2, random_state=0)
    risks = diagnostics.risk_comparison(n_folds=2, random_state=0).risks
    stability = diagnostics.stability(n_seeds=2, random_state=0)
    dropped = diagnostics.modifier_stability(n_folds=2, random_state=0).to_frame()
    targeting = diagnostics.targeting_curve(n_half_samples=20, random_state=0)
    assert np.isfinite([calibration.slope, calibration.intercept, *risks.values()]).all()
    assert np.isfinite(stability.relative_sd)
    assert list(dropped["dropped"]) == ["G", "W1"]
    assert targeting.n_eval == 300
    report = fitted.assess(include_refits=True, arguments=CHEAP, random_state=0)
    rows = {item.name: item for item in report.diagnostics.items}
    for name in CURVE_OPERATIONS:
        assert rows[name].status is not AssessmentStatus.UNAVAILABLE, name


def test_the_held_out_curve_reads_the_caller_s_levels(
    text_group: tuple[pd.DataFrame, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nonzero witness: the held-out predictions differ by group and match the raw rows.

    Mutation control: rebuild the group column with its levels rotated by one row, and
    the predictions no longer match the caller's rows.
    """
    import cleverly.effect_modification as effect_modification

    data, fitted = text_group
    predictions, _ = fitted.diagnostics._out_of_fold(2, 0)
    by_group = pd.Series(predictions).groupby(data["G"].to_numpy()).mean()
    assert by_group["alone"] - by_group["carer"] > 0.2
    assert _held_out_matches_the_raw_rows(data, fitted)

    original = effect_modification.modifier_frame

    def rotated(rows: Any, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
        values = original(rows, columns)
        return {**values, "G": np.roll(values["G"], 1)}

    monkeypatch.setattr(effect_modification, "modifier_frame", rotated)
    assert not _held_out_matches_the_raw_rows(data, fitted)

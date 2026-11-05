"""The study workflow of adaptive TMLE for a trial with external data (X90 T2).

A fit through ``CausalStudy`` reproduces the engine on the same arrays, carries the
``working_model_projection`` status, records the controls-only and few-treated cases, keeps
its estimate through ``save`` and ``load``, and every public sensitivity and validation entry
point refuses it before any point-treatment code runs.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
import cleverly.sensitivity
import cleverly.validation
from cleverly import (
    AdaptiveTMLEMethod,
    AdaptiveTMLEResult,
    AssessmentStatus,
    CapabilityError,
    CausalStudy,
    CrossFitting,
    ModelSpec,
    PositivityWarning,
    SplitPlan,
    TrialATE,
    TrialWithExternalData,
)
from cleverly._refusal import RefusalReason
from cleverly._trial_family import AUGMENTED_TRIAL_FAMILY
from cleverly.estimators.atmle import fit_adaptive_tmle
from cleverly.learners.crossfit import Folds, random_partition

#: The text every adaptive TMLE refusal of a point-treatment analysis carries.
REFUSED = ("adaptive working-model fit", "adaptive fit's own state")


def _frame(
    n: int = 300,
    *,
    seed: int = 0,
    external_treated: float = 0.3,
    missing: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    a = np.where(s == 1, rng.binomial(1, 0.5, n), rng.binomial(1, external_treated, n))
    y = 1.0 + a + w + 0.5 * (1 - s) * (1 + a) + rng.normal(size=n)
    frame = pd.DataFrame({"Y": y, "A": a, "W": w, "S": s})
    if missing:
        observed = rng.binomial(1, 0.85, n)
        frame["R"] = observed
        frame.loc[observed == 0, "Y"] = np.nan
    return frame


def _method(form: str = "cv", **models: Any) -> AdaptiveTMLEMethod:
    return AdaptiveTMLEMethod(
        form=form,  # type: ignore[arg-type]
        models=ModelSpec(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            **models,
        ),
        cross_fitting=CrossFitting(n_folds=3, learner_folds=3),
    ).with_overrides(random_state=7)


def _fit(frame: pd.DataFrame, population: str = "trial", **kwargs: Any) -> AdaptiveTMLEResult:
    design = TrialWithExternalData(
        outcome="Y",
        treatment="A",
        adjustment=("W",),
        source="S",
        missingness=kwargs.pop("missingness", None),
    )
    effect = CausalStudy(frame, design=design).identify(TrialATE(population=population))  # type: ignore[arg-type]
    result = effect.estimate(method=_method(**kwargs))
    assert isinstance(result, AdaptiveTMLEResult)
    return result


@pytest.fixture(scope="module")
def result() -> AdaptiveTMLEResult:
    return _fit(_frame())


@pytest.mark.parametrize("form", ["cv", "pooled"])
@pytest.mark.parametrize("population", ["trial", "pooled"])
def test_the_study_fit_reproduces_the_engine(form: str, population: str) -> None:
    """The wrapper hands the engine the declared arrays, learners, bounds and split."""
    frame = _frame(seed=1)
    fitted = _fit(frame, population, form=form)
    direct = fit_adaptive_tmle(
        frame["Y"].to_numpy(),
        frame["A"].to_numpy(dtype=float),
        frame["S"].to_numpy(dtype=float),
        frame[["W"]].to_numpy(),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        source_learner=LogisticRegression(),
        g_bounds=fitted.g_bounds,
        form=form,  # type: ignore[arg-type]
        population=population,  # type: ignore[arg-type]
        folds=fitted.folds,
        inner_folds=3,
        random_state=7,
    )
    estimate = fitted.estimate
    assert estimate.psi == pytest.approx(direct.estimate, abs=1e-12)
    assert estimate.std_error == pytest.approx(direct.std_error, rel=1e-12)
    assert fitted.inference_status == "working_model_projection"
    assert estimate.support.label == "experimental"
    assert fitted.fit.population == population and fitted.fit.form == form
    low, high = estimate.ci
    assert low < estimate.psi < high


def test_the_pooled_form_fits_with_cross_fitting_off() -> None:
    """One fold trains on every row (the engine's one-fold case), not a raw ValueError."""
    frame = _frame(seed=2)
    method = dataclasses.replace(
        _method("pooled"), cross_fitting=CrossFitting(enabled=False, learner_folds=3)
    )
    effect = CausalStudy(
        frame,
        design=TrialWithExternalData(outcome="Y", treatment="A", adjustment=("W",), source="S"),
    ).identify(TrialATE(population="trial"))
    fitted = effect.estimate(method=method)
    assert isinstance(fitted, AdaptiveTMLEResult)
    assert fitted.folds.n_folds == 1 and fitted.folds.is_single
    direct = fit_adaptive_tmle(
        frame["Y"].to_numpy(),
        frame["A"].to_numpy(dtype=float),
        frame["S"].to_numpy(dtype=float),
        frame[["W"]].to_numpy(),
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        source_learner=LogisticRegression(),
        g_bounds=fitted.g_bounds,
        form="pooled",
        population="trial",
        folds=Folds.single(len(frame)),
        inner_folds=3,
        random_state=7,
    )
    assert fitted.estimate.psi == pytest.approx(direct.estimate, abs=1e-12)


def test_a_supplied_split_plan_is_the_outer_split() -> None:
    frame = _frame(seed=3)
    plan = SplitPlan.from_folds([random_partition(len(frame), 4, seed=11)])
    method = dataclasses.replace(
        _method("cv"), cross_fitting=CrossFitting(n_folds=4, learner_folds=3, split_plan=plan)
    )
    effect = CausalStudy(
        frame,
        design=TrialWithExternalData(outcome="Y", treatment="A", adjustment=("W",), source="S"),
    ).identify(TrialATE(population="trial"))
    fitted = effect.estimate(method=method)
    assert isinstance(fitted, AdaptiveTMLEResult)
    assert fitted.folds.assignment.tolist() == list(plan.assignments[0])
    # A plan that records no generator is refused by the group itself.
    with pytest.raises(cleverly.MethodConfigurationError, match="no generator record"):
        CrossFitting(n_folds=4, split_plan=SplitPlan([list(plan.assignments[0])]))


def test_the_contrast_is_treated_against_control(result: AdaptiveTMLEResult) -> None:
    key = result.parameter_keys["ate"]
    assert (key.value, key.reference) == (1, 0)
    assert result.to_frame().loc[0, "estimand"] == "ate"
    with pytest.raises(CapabilityError, match="Only the difference scale"):
        result.ratio("ate", "ate")


def test_controls_only_external_data_take_the_controls_only_path() -> None:
    fitted = _fit(_frame(external_treated=0.0))
    support = fitted.diagnostics.support()
    assert support.controls_only and support.n_external_treated == 0
    assert "controls only" in str(fitted.summary())


def test_a_few_external_treated_rows_warn_and_keep_the_both_arms_path() -> None:
    frame = _frame(seed=2)
    external_treated = np.flatnonzero((frame["S"] == 0) & (frame["A"] == 1))
    frame.loc[external_treated[5:], "A"] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = _fit(frame)
    assert fitted.diagnostics.support().n_external_treated == 5
    assert any(
        record.category == PositivityWarning.__name__ and "only 5" in record.message
        for record in fitted.fit_warnings
    )


def test_missing_outcomes_fit_a_missingness_model() -> None:
    fitted = _fit(_frame(missing=True), missingness="R", missingness_learner=LogisticRegression())
    assert "missingness" in fitted.learners
    assert np.isfinite(fitted.psi())


def test_the_result_round_trips_through_save_and_load(
    result: AdaptiveTMLEResult, tmp_path: Path
) -> None:
    path = result.save(tmp_path / "atmle.joblib")
    loaded = cleverly.load(path)
    assert isinstance(loaded, AdaptiveTMLEResult)
    assert loaded.psi() == result.psi()
    assert loaded.estimate.std_error == result.estimate.std_error
    assert loaded.inference_status == result.inference_status


def test_the_assessment_reads_the_fit_and_refuses_every_bound(result: AdaptiveTMLEResult) -> None:
    rows = {row.operation: row for row in result.diagnostics.capabilities}
    assert {name for name, row in rows.items() if row.available} == {
        "support",
        "nuisance_models",
        "score_equations",
        "corrections",
    }
    assert result.diagnostics.score_equations().passed
    assert result.diagnostics.corrections().converged
    assert len(result.diagnostics.nuisance_models().folds) == 3
    report = result.assess()
    ran = {item.name: item for item in report.diagnostics.items}
    assert ran["support"].status is AssessmentStatus.PASSED
    assert all(not row.available for row in result.sensitivity.capabilities)
    for row in result.sensitivity.capabilities:
        assert row.reason is not None and "adaptive working-model fit" in row.reason
    replay = result.replayability
    assert replay.rerun_configuration and not replay.retarget_cached_nuisances


# ----------------------------------------------------------------------- the entry points

#: Placeholder values for a required argument the refusal must precede.
_ARGUMENTS: dict[str, Any] = {"covariates": ("W",), "grid": object(), "tolerance": 1e-3}


def _entry_points() -> list[tuple[str, Any]]:
    """Every public sensitivity and validation function that takes a fitted result first."""
    found = []
    for module in (cleverly.sensitivity, cleverly.validation):
        for name in module.__all__:
            function = getattr(module, name)
            if not inspect.isfunction(function):
                continue
            parameters = list(inspect.signature(function).parameters)
            if parameters and parameters[0] == "result":
                found.append((f"{module.__name__}.{name}", function))
    return found


ENTRY_POINTS = _entry_points()


def _refusals(result: Any) -> dict[str, str | None]:
    """The adaptive TMLE refusal each entry point raised, or ``None`` when it raised another."""
    outcomes: dict[str, str | None] = {}
    for name, function in ENTRY_POINTS:
        required = [
            parameter.name
            for parameter in list(inspect.signature(function).parameters.values())[1:]
            if parameter.default is inspect.Parameter.empty
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                function(result, **{argument: _ARGUMENTS[argument] for argument in required})
        except CapabilityError as error:
            text = str(error)
            outcomes[name] = text if any(phrase in text for phrase in REFUSED) else None
        except Exception:  # point code reached and failed some other way
            outcomes[name] = None
        else:
            outcomes[name] = None
    return outcomes


def test_the_namespace_holds_the_entry_points() -> None:
    names = {name.rsplit(".", 1)[1] for name, _ in ENTRY_POINTS}
    assert {"evalue", "omitted_variable_bounds", "simulated_confounding", "refute"} <= names
    assert len(names) >= 16


def test_every_public_entry_point_refuses_the_family(result: AdaptiveTMLEResult) -> None:
    refused = _refusals(result)
    assert all(refused.values()), [name for name, text in refused.items() if not text]


def test_the_refusal_is_keyed_on_the_declared_family(
    result: AdaptiveTMLEResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberate mutation: a result that declares ``"point"`` reaches point code."""
    assert AdaptiveTMLEResult.assessment_family == AUGMENTED_TRIAL_FAMILY
    monkeypatch.setattr(AdaptiveTMLEResult, "assessment_family", "point")
    mutated = dataclasses.replace(result)
    refused = _refusals(mutated)
    assert not any(refused.values()), [name for name, text in refused.items() if text]


def _uncovered_frame() -> pd.DataFrame:
    """External rows sit far from every trial row in W, so P(S = 1 | W) is near zero there."""
    frame = _frame(n=600, seed=4)
    frame.loc[frame["S"] == 0, "W"] += 4.0
    return frame


def test_pooled_positivity_is_reported_only_where_the_effect_needs_it() -> None:
    with pytest.warns(PositivityWarning, match="below 0.01"):
        pooled = _fit(_uncovered_frame(), "pooled", form="pooled")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PositivityWarning)
        trial = _fit(_uncovered_frame(), "trial", form="pooled")
    pooled_support = pooled.diagnostics.support()
    trial_support = trial.diagnostics.support()
    assert not pooled_support.adequate
    assert trial_support.external_source_below_floor > 0
    assert trial_support.adequate
    assert pooled.fit_warnings and any("below 0.01" in w.message for w in pooled.fit_warnings)
    assert not any("below 0.01" in w.message for w in trial.fit_warnings)
    # The summary note and the combined report follow the population.
    assert "limit the estimate" not in str(pooled.summary())
    trial_text = str(trial.summary())
    assert "does not limit the estimate" in trial_text
    ran = {item.name: item for item in trial.assess().diagnostics.items}
    assert ran["support"].status is AssessmentStatus.PASSED
    ran = {item.name: item for item in pooled.assess().diagnostics.items}
    assert ran["support"].status is AssessmentStatus.WARNING


def test_the_positivity_warning_points_at_the_user_call() -> None:
    frame = _uncovered_frame()
    with pytest.warns(PositivityWarning) as caught:
        _fit(frame, "pooled", form="pooled")
    assert all(Path(item.filename) == Path(__file__) for item in caught)


def test_score_equations_refuses_a_tolerance_it_would_ignore(
    result: AdaptiveTMLEResult,
) -> None:
    with pytest.raises(CapabilityError, match=r"sqrt\(n\) log n") as error:
        result.diagnostics.score_equations(tolerance=1e-3)
    assert error.value.refusal is not None
    assert error.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    report = result.diagnostics.score_equations()
    assert report.passed and report.tolerance > 0


def test_a_shipped_family_keeps_the_default_score_tolerance() -> None:
    from cleverly import ATE, PointTreatment
    from cleverly.validation.score import DEFAULT_TOLERANCE

    frame = _frame(n=150, seed=5)
    point = CausalStudy(
        frame.drop(columns="S"),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)),
    )
    fitted = point.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        cross_fit=False,
        random_state=0,
    )
    default = fitted.diagnostics.score_equations()
    explicit = fitted.diagnostics.score_equations(tolerance=DEFAULT_TOLERANCE)
    assert default.tolerance == explicit.tolerance == DEFAULT_TOLERANCE


@pytest.mark.parametrize("population", ["trial", "pooled"])
@pytest.mark.parametrize("missing", [False, True])
def test_the_assumptions_follow_the_population_and_the_missing_outcomes(
    population: str, missing: bool
) -> None:
    """A3 belongs to the combined-rows effect and the missing-outcome item to its design."""
    fitted = _fit(
        _frame(missing=missing),
        population,
        form="pooled",
        missingness="R" if missing else None,
        **({"missingness_learner": LogisticRegression()} if missing else {}),
    )
    recorded = " | ".join(fitted.identified_effect.identification.assumptions)
    shown = str(fitted.summary(detail="full"))
    for text in (recorded, shown):
        assert ("trial support" in text or "trial covers the external rows" in text) == (
            population == "pooled"
        )
        assert "delta=" not in text
        assert ("missing at random in the trial" in text) == missing
    if missing:
        assert "nothing is assumed about why external outcomes are missing" in recorded
        assert "given trial membership" not in shown and "given trial membership" not in recorded
        assert "missingness= declared" in recorded


@pytest.mark.parametrize("form", ["cv", "pooled"])
def test_the_covariance_is_the_variance_the_interval_uses(form: str) -> None:
    """Per-fold means are not zero under the pooled epsilon, so a centered covariance of the
    stacked curve differs from the fold-variance rule the standard error follows."""
    fitted = _fit(_frame(seed=6), form=form)
    cov = fitted.covariance()
    assert cov.shape == (1, 1)
    assert cov[0, 0] == pytest.approx(fitted.estimate.std_error**2, rel=1e-12)
    assert fitted.covariance(["ate"])[0, 0] == cov[0, 0]


def test_the_data_fingerprint_reads_trial_membership_only_when_it_is_declared() -> None:
    from cleverly.data.causal_data import CausalData
    from cleverly.provenance import data_fingerprint, fingerprint_array

    frame = _frame(n=80, seed=7)

    def build(table: pd.DataFrame, **roles: Any) -> CausalData:
        return CausalData.from_frame(table, outcome="Y", treatment="A", covariates=("W",), **roles)

    declared = build(frame, source="S")
    flipped_frame = frame.copy()
    flipped_frame.loc[0, "S"] = 1 - flipped_frame.loc[0, "S"]
    # Moving one row between the trial and the external data changes the digest.
    assert data_fingerprint(build(flipped_frame, source="S")) != data_fingerprint(declared)
    # Without a declared source, the digest is the one it was before the field existed.
    plain = build(frame)
    assert plain.source is None
    assert data_fingerprint(plain) == fingerprint_array(
        plain.outcome, plain.treatment, plain.covariates
    )
    assert data_fingerprint(plain) != data_fingerprint(declared)


def test_a_point_treatment_fit_is_unchanged_by_an_unused_membership_column() -> None:
    """Shipped paths replay as they did: the extra column is no role, so nothing moves."""
    from cleverly import ATE, PointTreatment

    frame = _frame(n=150, seed=8)

    def fit(table: pd.DataFrame) -> Any:
        study = CausalStudy(
            table, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        return study.identify(ATE()).estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        )

    with_column, without = fit(frame), fit(frame.drop(columns="S"))
    assert with_column.provenance.data_fingerprint == without.provenance.data_fingerprint
    assert with_column.psi() == without.psi()

"""Every refusal of adaptive TMLE for a trial with external data, before any fit (X90 T2).

One test per row of the plan's section-1 refusal table, and the field allowlist of
:class:`~cleverly.AdaptiveTMLEMethod`. A spy replaces the engine, so a refusal that let the
fit start fails here rather than passing after the work.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    ATT,
    AdaptiveTMLEMethod,
    CapabilityError,
    CausalStudy,
    CrossFitting,
    DataError,
    MethodConfigurationError,
    OddsRatio,
    PointTreatment,
    RiskRatio,
    TrialATE,
    TrialWithExternalData,
)
from cleverly._refusal import RefusalReason
from cleverly.data.causal_data import CausalData
from cleverly.methods import ATMLE_FIELDS

GROUPS = ("models", "cross_fitting", "targeting", "inference", "runtime")


def _frame(n: int = 120, *, seed: int = 0, arms: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.5, n)
    w = rng.normal(size=n)
    a = rng.integers(0, arms, n)
    y = a + w + 0.5 * (1 - s) + rng.normal(size=n)
    return pd.DataFrame({"Y": y, "A": a, "W": w, "S": s, "R": rng.binomial(1, 0.9, n)})


def _design(**changes: Any) -> TrialWithExternalData:
    fields: dict[str, Any] = {"outcome": "Y", "treatment": "A", "adjustment": ("W",), "source": "S"}
    fields.update(changes)
    return TrialWithExternalData(**fields)


@pytest.fixture
def no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test whose request reaches the adaptive TMLE engine."""

    def spy(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the adaptive TMLE engine was reached")

    monkeypatch.setattr("cleverly.estimators.atmle_result.fit_adaptive_tmle", spy)


def _code(error: pytest.ExceptionInfo[CapabilityError]) -> RefusalReason:
    assert error.value.refusal is not None
    return error.value.refusal.reason


@pytest.mark.parametrize("estimand", [ATE(), ATT()])
def test_a_population_estimand_is_a_different_estimand(no_engine: None, estimand: Any) -> None:
    study = CausalStudy(_frame(), design=_design())
    with pytest.raises(CapabilityError, match="Request TrialATE") as error:
        study.identify(estimand)
    assert _code(error) is RefusalReason.DIFFERENT_ESTIMAND


@pytest.mark.parametrize("estimand", [RiskRatio(), OddsRatio()])
def test_a_ratio_scale_is_not_derived(no_engine: None, estimand: Any) -> None:
    study = CausalStudy(_frame(), design=_design())
    with pytest.raises(CapabilityError, match="Only the difference scale") as error:
        study.identify(estimand)
    assert _code(error) is RefusalReason.IMPLEMENTATION_GAP
    assert error.value.refusal is not None and error.value.refusal.roadmap == "F40"


def test_trial_ate_needs_the_trial_membership_column(no_engine: None) -> None:
    study = CausalStudy(
        _frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    with pytest.raises(CapabilityError, match="TrialWithExternalData") as error:
        study.identify(TrialATE(population="trial"))
    assert _code(error) is RefusalReason.INVALID_DECLARATION


def test_a_treatment_with_more_than_two_levels_is_refused(no_engine: None) -> None:
    with pytest.raises(CapabilityError, match="binary treatment") as error:
        CausalStudy(_frame(arms=3), design=_design())
    assert _code(error) is RefusalReason.IMPLEMENTATION_GAP


def test_a_vector_treatment_is_refused_at_declaration() -> None:
    with pytest.raises(CapabilityError, match="binary treatment"):
        _design(treatment=("A", "W"))


@pytest.mark.parametrize(
    "field", ["weights", "cluster", "strata", "intermediate", "treatment_missingness"]
)
def test_the_design_takes_no_unsupported_role(field: str) -> None:
    with pytest.raises(TypeError):
        _design(**{field: "W"})


def test_repeated_splits_are_refused() -> None:
    with pytest.raises(MethodConfigurationError, match="repeated splits") as error:
        AdaptiveTMLEMethod(cross_fitting=CrossFitting(n_folds=5, repeats=2))
    assert _code(error) is RefusalReason.INFERENCE_GAP


def test_the_cv_form_needs_three_folds() -> None:
    with pytest.raises(MethodConfigurationError, match="at least three folds") as error:
        AdaptiveTMLEMethod(cross_fitting=CrossFitting(n_folds=2))
    assert _code(error) is RefusalReason.INVALID_DECLARATION
    # The single-sample form selects on every row and takes any split.
    cross = CrossFitting(n_folds=2)
    assert AdaptiveTMLEMethod(form="pooled", cross_fitting=cross).form == "pooled"


def test_the_cv_form_needs_cross_fitting_on() -> None:
    with pytest.raises(MethodConfigurationError, match="needs folds") as error:
        AdaptiveTMLEMethod(cross_fitting=CrossFitting(enabled=False))
    assert _code(error) is RefusalReason.INVALID_DECLARATION
    assert error.value.refusal is not None and "form='pooled'" in str(
        error.value.refusal.alternative
    )
    assert AdaptiveTMLEMethod(form="pooled", cross_fitting=CrossFitting(enabled=False)).form


def test_stratified_folds_take_the_package_refusal() -> None:
    with pytest.raises(MethodConfigurationError, match="stratify_folds='treatment'") as error:
        AdaptiveTMLEMethod(cross_fitting=CrossFitting(n_folds=5, stratify_by="treatment"))
    assert _code(error) is RefusalReason.IMPLEMENTATION_GAP


def test_a_trial_with_one_arm_is_a_data_error() -> None:
    frame = _frame()
    frame.loc[frame["S"] == 1, "A"] = 1
    with pytest.raises(DataError, match="one arm"):
        CausalStudy(frame, design=_design())


def test_a_frame_without_external_rows_is_a_data_error() -> None:
    frame = _frame()
    frame["S"] = 1
    with pytest.raises(DataError, match="external"):
        CausalStudy(frame, design=_design())


@pytest.mark.parametrize("dtype", ["float", "object", "boolean"])
def test_a_null_trial_membership_label_is_a_data_error(dtype: str) -> None:
    """A row with no label is not an external row."""
    frame = _frame()
    column: Any = frame["S"].astype(float)
    column.iloc[3] = np.nan
    if dtype == "object":
        column = column.astype(object)
    elif dtype == "boolean":
        column = column.astype("boolean")
    frame["S"] = column
    with pytest.raises(DataError, match="column 'S' contains missing"):
        CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=("W",), source="S")
    with pytest.raises(DataError, match="missing"):
        CausalStudy(frame, design=_design())


def test_a_point_design_refuses_a_container_that_marks_trial_membership() -> None:
    data = CausalData.from_frame(
        _frame(), outcome="Y", treatment="A", covariates=("W",), source="S"
    )
    with pytest.raises(DataError, match="TrialWithExternalData"):
        PointTreatment(outcome="Y", treatment="A", adjustment=("W",)).prepare(data)


def test_every_other_method_is_refused_on_the_design(no_engine: None) -> None:
    effect = CausalStudy(_frame(), design=_design()).identify(TrialATE(population="pooled"))
    rows = {row.name: row for row in effect.available_methods()}
    assert [name for name, row in rows.items() if row.available] == ["atmle"]
    for name, row in rows.items():
        if name == "atmle":
            continue
        assert row.refusal is not None and row.refusal.reason is RefusalReason.DIFFERENT_ESTIMAND
        with pytest.raises(CapabilityError):
            effect.estimate(method=name)


def test_adaptive_tmle_needs_the_trial_design(no_engine: None) -> None:
    study = CausalStudy(
        _frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    effect = study.identify(ATE())
    row = {row.name: row for row in effect.available_methods()}["atmle"]
    assert not row.available and row.refusal is not None
    assert row.refusal.reason is RefusalReason.INVALID_DECLARATION
    with pytest.raises(CapabilityError, match="TrialWithExternalData"):
        effect.estimate(method="atmle")


def test_a_fluctuation_setting_is_refused_through_the_shortcuts(no_engine: None) -> None:
    effect = CausalStudy(_frame(), design=_design()).identify(TrialATE(population="trial"))
    with pytest.raises(MethodConfigurationError, match=r"targeting\.fluctuation does not apply"):
        effect.estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(),
            fluctuation="linear",
        )


# ----------------------------------------------------------------------- the allowlist

#: A value other than the default for each field that a group's own checks accept. A field
#: missing here fails the coverage test, so a new option is either allowlisted or refused.
_OTHER: dict[tuple[str, str], Any] = {
    ("models", "intermediate_learner"): LinearRegression(),
    ("models", "pseudo_learner"): LinearRegression(),
    ("models", "censoring_learner"): LogisticRegression(),
    ("models", "competing_learner"): LogisticRegression(),
    ("models", "survival_model"): "hazard",
    ("models", "density_bins"): 8,
    ("models", "min_retain"): 2,
    ("models", "hazard_pooling"): "pooled",
    ("targeting", "q_bounds"): (0.01, 0.99),
    ("targeting", "step_bound"): 0.5,
    ("targeting", "ratio_trim"): 0.99,
    ("targeting", "node_bounds"): {1: (0.05, 0.95)},
    ("inference", "n_multiplier"): 1000,
    ("inference", "simultaneous"): True,
}


def _other_value(group: Any, item: dataclasses.Field[Any]) -> Any:
    default = getattr(group, item.name)
    hint = typing.get_type_hints(type(group))[item.name]
    literals = [arg for arg in typing.get_args(hint) if isinstance(arg, str)]
    if typing.get_origin(hint) is typing.Literal or literals:
        choices = typing.get_args(hint) if typing.get_origin(hint) is typing.Literal else literals
        return next(choice for choice in choices if choice != default)
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 1
    if isinstance(default, float):
        return default / 2
    raise KeyError(item.name)


def _refused_fields() -> list[tuple[str, str]]:
    method = AdaptiveTMLEMethod()
    return [
        (group, item.name)
        for group in GROUPS
        for item in dataclasses.fields(getattr(method, group))
        if (group, item.name) not in ATMLE_FIELDS
    ]


@pytest.mark.parametrize(("group", "name"), _refused_fields())
def test_every_field_off_the_allowlist_is_refused(group: str, name: str) -> None:
    """Enumerated from ``dataclasses.fields``, so a new option cannot slip through."""
    method = AdaptiveTMLEMethod()
    value = getattr(method, group)
    item = next(item for item in dataclasses.fields(value) if item.name == name)
    other = _OTHER.get((group, name))
    if other is None:
        other = _other_value(value, item)
    changes = {name: other}
    if (group, name) == ("targeting", "step_search"):
        # The group itself ties step_search to a declared step_bound.
        changes["step_bound"] = 0.5
    with pytest.raises(MethodConfigurationError) as error:
        # CrossFitting refuses a stratified split itself, with the package's refusal.
        AdaptiveTMLEMethod(**{group: dataclasses.replace(value, **changes)})
    if (group, name) == ("cross_fitting", "repeats"):
        assert "repeated splits" in str(error.value)
    elif (group, name) == ("cross_fitting", "stratify_by"):
        assert "stratify_folds" in str(error.value)
    elif (group, name) == ("targeting", "step_search"):
        assert "targeting.step_bound does not apply" in str(error.value)
    else:
        assert f"{group}.{name} does not apply" in str(error.value)
        if (group, name) in {("runtime", "n_jobs"), ("inference", "n_bootstrap")}:
            # The reason names the real cause, not an unrelated fluctuation.
            assert "no outcome fluctuation" not in str(error.value)


def test_the_allowlist_keeps_what_the_engine_reads() -> None:
    assert ("runtime", "n_jobs") not in ATMLE_FIELDS
    method = AdaptiveTMLEMethod(
        cross_fitting=CrossFitting(n_folds=4, learner_folds=3),
    ).with_overrides(alpha=0.1, random_state=3, run_id="x", g_bounds=(0.05, 0.95))
    kwargs = method.estimator_kwargs()
    assert (kwargs["n_folds"], kwargs["inner_folds"], kwargs["random_state"]) == (4, 3, 3)
    assert kwargs["g_bounds"] == (0.05, 0.95)
    assert method.inference.alpha == 0.1


def test_a_custom_method_named_atmle_is_refused_by_message_not_by_assert(
    no_engine: None,
) -> None:
    from cleverly import TMLEMethod

    impostor = dataclasses.replace(TMLEMethod(), name="atmle")
    effect = CausalStudy(_frame(), design=_design()).identify(TrialATE(population="trial"))
    with pytest.raises(CapabilityError, match="not AdaptiveTMLEMethod") as error:
        effect.estimate(method=impostor)
    assert _code(error) is RefusalReason.INVALID_DECLARATION
    assert error.value.refusal is not None and "AdaptiveTMLEMethod" in str(
        error.value.refusal.alternative
    )

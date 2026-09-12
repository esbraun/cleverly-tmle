"""Typed study, identification, capability, and configuration contracts."""

import dataclasses
import inspect
import pickle
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model

import cleverly
from cleverly import (
    ATC,
    ATE,
    ATT,
    CapabilityError,
    CausalStudy,
    CleverlyError,
    CollaborativeTMLEMethod,
    ControlledDirectEffect,
    CounterfactualMean,
    CrossFitting,
    DataError,
    DRTMLEMethod,
    LongitudinalTreatment,
    MethodConfigurationError,
    ModelSpec,
    ModifiedTreatmentPolicy,
    MSMProjection,
    NaturalCourseMean,
    OddsRatio,
    PointTreatment,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
    RegimeContrast,
    RegimeMean,
    RiskRatio,
    Runtime,
    TMLEMethod,
)
from cleverly.data import CausalData, validate
from cleverly.data.validate import RANDOMIZED_INTERCEPT
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_longitudinal,
    make_multi_arm,
    make_shift_dose,
)
from cleverly.estimators import TMLE
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal.estimator import DEFAULT_LTMLE_G_BOUNDS
from cleverly.methods import (
    DEFAULT_LONGITUDINAL_G_BOUNDS,
    DEFAULT_LONGITUDINAL_MULTIPLIER,
    DEFAULT_POINT_MULTIPLIER,
    SHORTCUTS,
)
from cleverly.msm import MSM
from cleverly.study import (
    _ARM_STATEMENTS,
    _LONGITUDINAL_IDENTIFICATION,
    BackdoorMeanContrast,
    _arm_scope,
    _matches_registered_point_identification,
    _point_identification,
)
from cleverly.targets import TARGETS
from cleverly.targets.base import (
    INTERMEDIATE_MECHANISM,
    MISSINGNESS_MECHANISM,
    POINT_NUISANCES,
)
from cleverly.targets.builtin import (
    BUILTIN_TARGETS,
    DR_DESIGN_CONDITIONAL_CLAUSE,
    DR_MECHANISM_HEAD,
    INTERMEDIATE_CAVEAT_PREFIX,
    MISSINGNESS_CAVEAT_PREFIX,
    TWO_ARM_POSITIVITY_PREFIX,
)
from cleverly.targets.population_intervention import (
    POPULATION_INTERVENTION_TARGETS,
    population_intervention_refusal,
)
from tests import discrete_law_cde, discrete_law_mar
from tests.conftest import (
    FAST_KWARGS,
    OracleDirectOutcome,
    OracleIntermediate,
    OracleMissingness,
    OracleOutcome,
    OracleTreatment,
)
from tests.pickles import PRE_SCHEMA_1_FIELDS, _LegacyPickle, legacy_without


def _study() -> CausalStudy:
    frame, _ = make_linear_ate(n=120, seed=19)
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
        ),
    )


def _fitted_nuisance_names(result: Any) -> tuple[str, ...]:
    """Map fitted point-treatment artifacts to the public identification vocabulary."""
    nuisance = result.nuisance
    names = ["outcome_regression", "treatment_mechanism"]
    if nuisance.intermediate is not None:
        names.append("intermediate_mechanism")
    if nuisance.missingness is not None:
        names.append("missingness_mechanism")
    return tuple(names)


def test_the_design_is_immutable_and_requires_an_identification_declaration() -> None:
    design = _study().design
    assert design.adjustment == ("W1", "W2", "W3", "W4")
    with pytest.raises(FrozenInstanceError):
        design.outcome = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-empty adjustment set"):
        PointTreatment(outcome="Y", treatment="A")


def test_the_study_cannot_be_pointed_at_a_design_it_did_not_prepare() -> None:
    """Freezing the design is not enough while the study can be handed a different one.

    ``_data`` is prepared once, at construction, from the design's column roles; ``identify``
    is the only reader of ``design``.  So a study whose ``design`` had been rebound would
    report an outcome column and adjustment set that no fitted number came from, and nothing
    downstream compares the two.
    """
    study = _study()
    other = PointTreatment(outcome="W1", treatment="A", adjustment=("W2", "W3"))
    with pytest.raises(AttributeError):
        study.design = other  # type: ignore[misc]
    assert study.design.outcome == "Y"


class TestAPreparedContainerIsReconciledWithTheDesign:
    """Handing the study a built container used to adopt it without a single check.

    ``prepare`` returned any ``CausalData``/``LongitudinalData`` unchanged, so every role on
    the design became a claim nothing had verified. The design is what
    ``IdentifiedEffect.functional`` records and ``summary()`` prints, so a design naming an
    adjustment set the container never adjusted for reported that set as identification --
    and this PR then persisted it. It is the data-construction half of the hazard
    ``test_the_study_cannot_be_pointed_at_a_design_it_did_not_prepare`` argues for rebinding.
    """

    @staticmethod
    def _prepared():  # type: ignore[no-untyped-def]
        frame, _ = make_linear_ate(n=120, seed=19)
        return frame, CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=("W1", "W2", "W3")
        )

    def test_a_matching_container_is_accepted_and_reported_truthfully(self) -> None:
        _, data = self._prepared()
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"))
        effect = CausalStudy(data, design=design).identify(ATE())
        assert effect.functional.adjustment == ("W1", "W2", "W3")
        assert "W1" in effect.summary()

    @pytest.mark.parametrize(
        ("design", "reason"),
        [
            (
                PointTreatment(outcome="Y", treatment="A", adjustment=("Z1",)),
                "adjusts for",
            ),
            (
                PointTreatment(outcome="W1", treatment="A", adjustment=("W1", "W2", "W3")),
                "outcome=",
            ),
            (
                PointTreatment(outcome="Y", treatment="W1", adjustment=("W1", "W2", "W3")),
                "treatment=",
            ),
            (
                PointTreatment(
                    outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), cluster="G"
                ),
                "cluster=",
            ),
            (
                PointTreatment(
                    outcome="Y",
                    treatment="A",
                    adjustment=("W1", "W2", "W3"),
                    treatment_kind="continuous",
                ),
                "treatment_kind=",
            ),
            (
                PointTreatment(
                    outcome="Y",
                    treatment="A",
                    adjustment=("W1", "W2", "W3"),
                    outcome_family="binomial",
                ),
                "outcome_family=",
            ),
        ],
    )
    def test_every_role_the_design_states_is_checked(self, design, reason) -> None:  # type: ignore[no-untyped-def]
        _, data = self._prepared()
        with pytest.raises(DataError, match=reason):
            CausalStudy(data, design=design)

    def test_an_adjustment_set_survives_encoding_and_dropping(self) -> None:
        """``covariate_names`` is post-encoding, so the comparison uses the named columns.

        A categorical adjustment variable becomes several generated columns and a degenerate
        one is dropped outright, so comparing the stored names to the declaration directly
        would reject the very containers this check exists to accept.
        """
        frame, _ = make_linear_ate(n=120, seed=23)
        frame = frame.assign(G=pd.Categorical(np.where(frame["W1"] > 0, "high", "low")), C=1.0)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=("W1", "G", "C"))
        assert data.dropped_covariates == ("C",)
        assert set(data.covariate_names) != {"W1", "G", "C"}
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "G", "C"))
        assert CausalStudy(data, design=design).data is data

    def test_a_longitudinal_container_is_reconciled_node_by_node(self) -> None:
        frame, _ = make_longitudinal(n=100, seed=11)
        columns = {
            "outcome": "Y",
            "treatment": ("A1", "A2"),
            "baseline": ("W1", "W2"),
            "time_varying": ((), ("L2",)),
            "censoring": ("C1", "C2"),
        }
        data = LongitudinalData.from_frame(frame, **columns)
        assert CausalStudy(data, design=LongitudinalTreatment(**columns)).data is data
        for role, changed in (
            ("baseline", {"baseline": ("W1",)}),
            ("outcome event nodes", {"outcome": ("Y1", "Y2")}),
            ("censoring", {"censoring": None}),
            ("time_varying", {"time_varying": (("L2",), ())}),
        ):
            with pytest.raises(DataError, match=role):
                CausalStudy(data, design=LongitudinalTreatment(**{**columns, **changed}))


def test_the_reserved_intercept_column_survives_the_constant_sweep() -> None:
    """One name, two packages, and only one thing standing between them and a bug.

    ``PointTreatment(randomized=True, adjustment=())`` is a claim of *no* adjustment, and the
    reserved constant column is what keeps the design well formed for learners that fit their
    own intercept. ``check_covariates`` drops constant columns, so the exemption keyed on that
    name is the only reason the column survives -- and the name was written out as a bare
    literal at both ends, where renaming one would have left the fit with no covariates.
    """
    written_out = [
        path
        for path in Path(cleverly.__file__).parent.rglob("*.py")
        if f'"{RANDOMIZED_INTERCEPT}"' in path.read_text(encoding="utf-8")
    ]
    assert written_out == [Path(validate.__file__)], (
        "the reserved name is spelled out somewhere other than its definition; import "
        "RANDOMIZED_INTERCEPT instead, so a rename cannot reach one end and not the other"
    )
    frame, _ = make_linear_ate(n=120, seed=29)
    study = CausalStudy(
        frame[["Y", "A"]], design=PointTreatment(outcome="Y", treatment="A", randomized=True)
    )
    assert study.data.covariate_names == (RANDOMIZED_INTERCEPT,)
    assert study.data.dropped_covariates == ()
    result = study.estimate(
        ATE(),
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
    )
    assert np.isfinite(result.psi("ate"))
    # The design still says what it claimed: no adjustment variables at all.
    assert result.identified_effect.functional.adjustment == ()


def test_identification_is_inspectable_before_estimation() -> None:
    effect = _study().identify(ATE())
    assert effect.functional.adjustment == ("W1", "W2", "W3", "W4")
    assert effect.functional.treatment_levels == (0, 1)
    assert effect.identification.required_nuisances == (
        "outcome_regression",
        "treatment_mechanism",
    )
    summary = effect.summary()
    assert "E_W[E(Y | A=a, W)" in summary
    assert "no unmeasured confounding" in summary
    # The design-bound sentence, not the registry's binary template.  ``"0" in summary``
    # held at the old truth too -- ``positivity: 0 < P(A = 1 | W) < 1`` contains both
    # digits -- so it could not tell the two apart.
    assert (
        "positivity: P(A = a | W) > 0 almost surely for every supported treatment "
        "level a in [0, 1]" in summary
    )
    assert "0 < P(A = 1 | W) < 1" not in summary
    assert "required nuisances" in summary

    result = effect.estimate(**FAST_KWARGS)
    assert _fitted_nuisance_names(result) == effect.identification.required_nuisances


def test_multi_arm_identification_names_every_level_and_fitted_mechanism() -> None:
    frame, _ = make_multi_arm(n=180, seed=17)
    effect = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    ).identify(ATE(reference="low"))

    assert effect.functional.treatment_levels == ("high", "low", "medium")
    summary = effect.summary()
    for level in effect.functional.treatment_levels:
        assert repr(level) in summary
    assert "P(A = 1 | W)" not in summary
    assert "both counterfactual means" not in summary
    assert "treatment_mechanism" in summary

    result = effect.estimate(**FAST_KWARGS)
    assert result.nuisance.propensity.values.shape == (len(frame), 3)
    assert _fitted_nuisance_names(result) == effect.identification.required_nuisances


def test_missing_outcome_identification_matches_the_observed_data_fit() -> None:
    law = discrete_law_mar.DiscreteLaw()
    effect = CausalStudy(
        discrete_law_mar.frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"),
    ).identify(ATE())

    assert effect.functional.missingness == "Delta"
    assert "E_W[E(Y | A=a, Delta=1, W)]" in effect.functional.expression
    assert any("missingness at random" in item for item in effect.identification.assumptions)
    assert any("response positivity" in item for item in effect.identification.assumptions)
    assert effect.identification.required_nuisances == (
        "outcome_regression",
        "treatment_mechanism",
        "missingness_mechanism",
    )
    assert "missingness_mechanism" in effect.summary()

    result = effect.estimate(
        outcome_learner=OracleOutcome(law),
        treatment_learner=OracleTreatment(law),
        missingness_learner=OracleMissingness(law),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    assert result.nuisance.missingness is not None
    assert result.nuisance.missingness.shape == (result.data.n, 2)
    assert _fitted_nuisance_names(result) == effect.identification.required_nuisances


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_cde_identification_matches_each_fitted_score_factor(level: float) -> None:
    law = discrete_law_cde.DiscreteLaw()
    effect = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    ).identify(ControlledDirectEffect(intermediate=level))

    assert effect.functional.intermediate == level
    assert effect.functional.intermediate_name == "Z"
    assert f"Z={level:g}" in effect.functional.expression
    assert "Delta=1" in effect.functional.expression
    assumptions = effect.identification.assumptions
    assert any("unmeasured treatment confounding" in item for item in assumptions)
    assert any("unmeasured intermediate confounding" in item for item in assumptions)
    assert any(f"intermediate positivity at Z = {level:g}" in item for item in assumptions)
    assert any("missingness at random" in item for item in assumptions)
    assert any("response positivity" in item for item in assumptions)
    assert any("observation-mechanism restriction" in item for item in assumptions)
    assert effect.identification.required_nuisances == (
        "outcome_regression",
        "treatment_mechanism",
        "intermediate_mechanism",
        "missingness_mechanism",
    )

    result = effect.estimate(
        outcome_learner=OracleDirectOutcome(law),
        treatment_learner=OracleTreatment(law),
        missingness_learner=OracleMissingness(law),
        intermediate_learner=OracleIntermediate(law),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
    )
    assert result.nuisance.intermediate_density(level, 0.0) is not None
    assert _fitted_nuisance_names(result) == effect.identification.required_nuisances


def test_identification_functional_metadata_survives_pickle_and_backfills() -> None:
    effect = _study().identify(ATE())
    restored = pickle.loads(pickle.dumps(effect.functional))
    assert restored == effect.functional
    assert restored.expression == effect.functional.expression

    legacy = legacy_without(effect.functional, *BackdoorMeanContrast._SCHEMA_1_FIELDS)
    assert legacy.missingness is None
    assert legacy.intermediate_name is None
    assert legacy.treatment_levels == ()
    assert legacy.treatment_value is None
    assert legacy.schema_version == 0


def test_design_bound_identification_provenance_requires_the_complete_record(tmp_path) -> None:
    effect = _study().identify(ATE())
    assert effect._study is not None
    registered = TARGETS["ate"].identification
    assert _matches_registered_point_identification(
        effect.identification, registered, effect, effect._study.data, effect.functional.axis
    )
    # A fresh functional declares its support, so swapping its dynamic record for the
    # generic registry template is not a legacy compatibility case.
    assert not _matches_registered_point_identification(
        registered, registered, effect, effect._study.data, effect.functional.axis
    )

    state = {
        name: value
        for name, value in effect.functional.__dict__.items()
        if name not in set(BackdoorMeanContrast._SCHEMA_1_FIELDS)
    }
    legacy_effect = dataclasses.replace(
        effect,
        functional=_LegacyPickle(type(effect.functional), state),
        identification=registered,
    )
    result = effect.estimate(**FAST_KWARGS)
    legacy_result = dataclasses.replace(result, identified_effect=legacy_effect)
    restored = cleverly.load(legacy_result.save(tmp_path / "legacy-identification.joblib"))
    restored_effect = restored.identified_effect
    assert restored_effect.functional.schema_version == 0
    assert restored_effect.functional.treatment_value is None
    assert "identified by explicit-adjustment" in restored_effect.summary()
    assert _matches_registered_point_identification(
        registered,
        registered,
        restored_effect,
        effect._study.data,
        effect.functional.axis,
    )

    altered_assumptions = dataclasses.replace(
        effect.identification, assumptions=("same citations, different claim",)
    )
    altered_nuisances = dataclasses.replace(
        effect.identification, required_nuisances=("outcome_regression",)
    )
    for altered in (altered_assumptions, altered_nuisances):
        assert altered.references == registered.references
        assert not _matches_registered_point_identification(
            altered, registered, effect, effect._study.data, effect.functional.axis
        )


def test_transitional_design_bound_pickle_remains_valid_but_stale_cde_mar_does_not() -> None:
    effect = _study().identify(ATE())
    assert effect._study is not None
    transitional_state = {
        name: value
        for name, value in effect.functional.__dict__.items()
        if name not in set(BackdoorMeanContrast._TRANSITIONAL_FIELDS)
    }
    restored = pickle.loads(
        pickle.dumps(
            dataclasses.replace(
                effect,
                functional=_LegacyPickle(type(effect.functional), transitional_state),
            )
        )
    )
    registered = TARGETS["ate"].identification
    assert restored.functional.schema_version == 0
    assert restored.functional.treatment_levels == (0, 1)
    assert _matches_registered_point_identification(
        restored.identification,
        registered,
        restored,
        effect._study.data,
        effect.functional.axis,
    )

    cde = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    ).identify(ControlledDirectEffect(intermediate=0.0))
    assert cde._study is not None
    stale_assumptions = tuple(
        item.replace("given (A, Z, W)", "given (A, W)")
        if item.startswith("missingness at random")
        else item
        for item in cde.identification.assumptions
    )
    stale = dataclasses.replace(
        cde,
        functional=legacy_without(cde.functional, *BackdoorMeanContrast._TRANSITIONAL_FIELDS),
        identification=dataclasses.replace(cde.identification, assumptions=stale_assumptions),
    )
    cde_registered = TARGETS["ate"].identification
    assert not _matches_registered_point_identification(
        stale.identification,
        cde_registered,
        stale,
        cde._study.data,
        stale.functional.axis,
    )


#: One row per arm-indexed target: the estimand that selects it, the functional the record
#: prints, and the positivity sentence the design binds to it.  ``None`` in the positivity
#: column means the target states no positivity assumption at all.
#:
#: The rows are the completeness gate as well as the assertions.
#: ``test_the_estimand_table_covers_every_registered_arm_target`` compares the keys with
#: the registry, so registering an arm target without deciding what it prints fails here
#: rather than printing the counterfactual mean's functional under the new name.
ARM_ESTIMAND_TABLE: tuple[tuple[str, Any, str, str | None], ...] = (
    (
        "ate",
        ATE(),
        "E_W[E(Y | A=a, W)] - E_W[E(Y | A=0, W)] for a in [1]",
        "positivity: P(A = a | W) > 0 almost surely for every supported treatment level a in [0, 1]",
    ),
    (
        "att",
        ATT(),
        "E_{W | A=a}[E(Y | A=a, W) - E(Y | A=0, W)] for a in [1]",
        "positivity for the reference arm within each comparison population: "
        "P(A = 0 | W) > 0 wherever P(A = a | W) > 0, for every comparison arm a",
    ),
    (
        "atc",
        ATC(),
        "E_{W | A=0}[E(Y | A=a, W) - E(Y | A=0, W)] for a in [1]",
        "positivity for each comparison arm within the reference population: "
        "P(A = a | W) > 0 wherever P(A = 0 | W) > 0, for every comparison arm a",
    ),
    (
        "ey",
        CounterfactualMean(),
        "E_W[E(Y | A=a, W)] for a in [0, 1]",
        "positivity: P(A = a | W) > 0 almost surely for every supported treatment level a in [0, 1]",
    ),
    (
        "ey1",
        CounterfactualMean(treatment=1),
        "E_W[E(Y | A=1, W)]",
        "positivity for the evaluated arm: P(A = 1 | W) > 0 almost surely",
    ),
    (
        "ey0",
        CounterfactualMean(treatment=0),
        "E_W[E(Y | A=0, W)]",
        "positivity for the evaluated arm: P(A = 0 | W) > 0 almost surely",
    ),
    (
        "ey_obs",
        NaturalCourseMean(),
        "E(Y) under the observed treatment course",
        None,
    ),
    (
        "par",
        PopulationAttributableRisk(),
        "E(Y) - E_W[E(Y | A=0, W)]",
        "positivity for the reference intervention: P(A = 0 | W) > 0 almost surely",
    ),
    (
        "paf",
        PopulationAttributableFraction(),
        "1 - E_W[E(Y | A=0, W)] / E(Y)",
        "positivity for the reference intervention: P(A = 0 | W) > 0 almost surely",
    ),
    (
        "rr",
        RiskRatio(),
        "E_W[E(Y | A=a, W)] / E_W[E(Y | A=0, W)] for a in [1]",
        "positivity: P(A = a | W) > 0 almost surely for every supported treatment level a in [0, 1]",
    ),
    (
        "or",
        OddsRatio(),
        "(E_W[E(Y | A=a, W)] / (1 - E_W[E(Y | A=a, W)])) / "
        "(E_W[E(Y | A=0, W)] / (1 - E_W[E(Y | A=0, W)])) for a in [1]",
        "positivity: P(A = a | W) > 0 almost surely for every supported treatment level a in [0, 1]",
    ),
)


@pytest.mark.parametrize(
    ("estimand", "expression", "positivity"),
    [row[1:] for row in ARM_ESTIMAND_TABLE],
    ids=[row[0] for row in ARM_ESTIMAND_TABLE],
)
def test_arm_functionals_and_positivity_are_target_specific(
    estimand: Any, expression: str, positivity: str | None
) -> None:
    """Each arm-indexed target prints its own functional and its own treatment support."""
    effect = _study().identify(estimand)
    assert effect.functional.expression == expression
    assumptions = effect.identification.assumptions
    stated = [item for item in assumptions if "positivity" in item]
    if positivity is None:
        assert stated == []
    else:
        assert positivity in stated


def test_the_estimand_table_covers_every_registered_arm_target() -> None:
    """A newly registered arm target fails here rather than borrowing another's prose."""
    assert {row[0] for row in ARM_ESTIMAND_TABLE} == {
        target.name for target in TARGETS.values() if target.parameter_axis == "arm"
    }
    identified = {name: _study().identify(estimand) for name, estimand, _, _ in ARM_ESTIMAND_TABLE}
    assert {name: effect.functional.target for name, effect in identified.items()} == {
        name: name for name in identified
    }


def test_conditional_effects_do_not_claim_support_at_every_arm() -> None:
    study = _study()
    for estimand in (ATT(), ATC()):
        effect = study.identify(estimand)
        assert not any(
            "every supported treatment level" in item for item in effect.identification.assumptions
        )


def test_population_intervention_records_state_their_definedness_and_nuisances() -> None:
    study = _study()
    natural = study.identify(NaturalCourseMean())
    assert natural.identification.required_nuisances == ()
    # The record explains the empty nuisance list and the adjustment set summary() prints.
    assert any(
        "functional of the observed law alone" in item
        for item in natural.identification.assumptions
    )
    for estimand in (PopulationAttributableRisk(), PopulationAttributableFraction()):
        effect = study.identify(estimand)
        assert effect.identification.required_nuisances == (
            "outcome_regression",
            "treatment_mechanism",
        )
    assert any(
        "natural-course risk E(Y) is strictly positive" in item
        for item in study.identify(PopulationAttributableFraction()).identification.assumptions
    )


@pytest.mark.parametrize(
    ("estimand", "condition"),
    [
        (RiskRatio(), "both counterfactual risks are strictly positive"),
        (OddsRatio(), "both counterfactual risks lie strictly inside (0, 1)"),
        (PopulationAttributableFraction(), "natural-course risk E(Y) is strictly positive"),
    ],
    ids=["rr", "or", "paf"],
)
def test_every_boundary_undefined_estimand_states_its_definedness_condition(
    estimand: Any, condition: str
) -> None:
    """``cleverly.inference.delta`` raises on each of these; the record states all three.

    The attributable fraction stated its condition and the two ratios did not, which is
    an inconsistent treatment of one class of condition rather than a difference between
    the estimands.
    """
    effect = _study().identify(estimand)
    assert any(condition in item for item in effect.identification.assumptions)


def test_multi_arm_att_and_atc_name_their_distinct_target_populations() -> None:
    frame, _ = make_multi_arm(n=180, seed=41)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    att = study.identify(ATT(reference="low"))
    atc = study.identify(ATC(reference="low"))

    assert "E_{W | A=a}" in att.functional.expression
    assert att.functional.expression.endswith("for a in ['high', 'medium']")
    assert "A='low'" in att.functional.expression
    assert "E_{W | A='low'}" in atc.functional.expression
    assert atc.functional.expression.endswith("for a in ['high', 'medium']")
    assert "A=a" in atc.functional.expression
    assert "P(A = 'low' | W) > 0 wherever P(A = a | W) > 0" in " ".join(
        att.identification.assumptions
    )
    assert "P(A = a) > 0 for every comparison arm a" in " ".join(att.identification.assumptions)
    assert "P(A = a | W) > 0 wherever P(A = 'low' | W) > 0" in " ".join(
        atc.identification.assumptions
    )
    assert "P(A = 'low') > 0" in " ".join(atc.identification.assumptions)

    selected = study.identify(CounterfactualMean(treatment="medium"))
    assert selected.functional.treatment_value == "medium"
    assert selected.functional.expression == "E_W[E(Y | A='medium', W)]"
    assert "for a in" not in selected.functional.expression
    assert "P(A = 'medium' | W) > 0" in " ".join(selected.identification.assumptions)
    assert not any(
        "every supported treatment level" in item for item in selected.identification.assumptions
    )


@pytest.mark.parametrize(
    ("contrast", "population"),
    [(ATT(), "P(A = a | W) > 0"), (ATC(), "P(A = 0 | W) > 0")],
)
def test_cde_conditional_mechanism_positivity_names_both_arms_and_population(
    contrast: Any,
    population: str,
) -> None:
    effect = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    ).identify(ControlledDirectEffect(intermediate=0.0, contrast=contrast))
    intermediate = next(
        item
        for item in effect.identification.assumptions
        if item.startswith("intermediate positivity")
    )
    response = next(
        item for item in effect.identification.assumptions if item.startswith("response positivity")
    )
    for item in (intermediate, response):
        assert "b in {a, 0}" in item
        assert population in item


@pytest.mark.parametrize(
    ("estimand", "roadmap"),
    [
        (PopulationAttributableRisk(), "RM8"),
        (PopulationAttributableFraction(), "RM8"),
    ],
)
def test_missing_population_interventions_refuse_at_identification_boundary(
    estimand: Any,
    roadmap: str,
) -> None:
    with pytest.raises(CapabilityError, match=rf"score equation.*{roadmap}"):
        CausalStudy(
            discrete_law_mar.frame(),
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W",),
                missingness="Delta",
            ),
        ).identify(estimand)


def test_missing_natural_course_identification_names_only_its_two_nuisances() -> None:
    effect = CausalStudy(
        discrete_law_mar.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
        ),
    ).identify(NaturalCourseMean())
    assert effect.functional.expression == "E_{A,W}[E(Y | Delta=1, A, W)]"
    assert effect.identification.required_nuisances == (
        "outcome_regression",
        "missingness_mechanism",
    )
    assumptions = " ".join(effect.identification.assumptions)
    assert "missingness at random" in assumptions
    assert "response positivity" in assumptions
    assert "no treatment mechanism" in effect.identification.dr_condition


def test_point_treatment_targets_state_no_interference_separately() -> None:
    """Every potential-outcome target exposes the SUTVA component the API omitted."""
    by_name = {target.name: target for target in BUILTIN_TARGETS}
    checked = 0
    for target in BUILTIN_TARGETS:
        assumptions = target.identification.assumptions
        if not any(item.startswith("consistency:") for item in assumptions):
            continue
        checked += 1
        assert sum("no interference" in item for item in assumptions) == 1, target.name
    assert checked >= 10
    assert sum("no interference" in item for item in _LONGITUDINAL_IDENTIFICATION.assumptions) == 1
    assert (
        sum(item.startswith("consistency:") for item in _LONGITUDINAL_IDENTIFICATION.assumptions)
        == 1
    )
    for name in ("ey_ipsi", "ate_ipsi"):
        assert any(
            "no positivity assumption" in item for item in by_name[name].identification.assumptions
        )
    for name in ("ey_regime", "ate_regime"):
        assert any(
            "positivity *for the regime*" in item
            for item in by_name[name].identification.assumptions
        )
    for name in ("ey_shift", "ate_shift"):
        assert any(
            "positivity *for the shifted dose*" in item
            for item in by_name[name].identification.assumptions
        )


def test_an_unknown_reference_is_refused_during_identification() -> None:
    with pytest.raises(ValueError, match="is not a treatment level"):
        _study().identify(ATE(reference="not-an-arm"))


def test_stratified_parameter_keys_are_structured_before_the_alias_is_displayed() -> None:
    frame, _ = make_linear_ate(n=120, seed=20)
    frame = frame.assign(S=(frame["W1"] > 0).astype(int))
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4", "S"),
            strata=("S",),
        ),
    )
    result = study.identify(ATE()).estimate(**FAST_KWARGS)
    assert result.parameter_keys["ate[S=0]"].stratum == (0,)
    assert result.parameter_keys["ate[S=1]"].stratum == (1,)


def test_method_availability_is_structured_and_refuses_before_fitting(monkeypatch) -> None:
    effect = _study().identify(ATE())
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["tmle"].available
    assert not methods["riesz_tmle"].available
    assert "representer" in (methods["riesz_tmle"].reason or "")

    class MustNotConstruct:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("nuisance fitting path was reached")

    monkeypatch.setattr("cleverly.study.TMLE", MustNotConstruct)
    with pytest.raises(CapabilityError, match="direct-Riesz engine"):
        effect.estimate(method="riesz_tmle")


def test_a_controlled_direct_effect_refuses_the_variants_before_fitting(monkeypatch) -> None:
    """The variant check has to read the intermediate, not only the functional's target.

    A ``ControlledDirectEffect``'s ``functional.target`` is its *contrast's* name -- ``ate`` --
    so a check reading the target alone declared C-TMLE and DR-TMLE available for it, and both
    engines then refused partway through a fit.  Refusing after nuisance fitting has started is
    the thing ``docs/architecture-invariants.md`` puts at the identification boundary.
    """
    frame, _ = make_linear_ate(n=120, seed=21)
    study = CausalStudy(
        frame.assign(Z=(frame["W1"] > 0).astype(int)),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2"), intermediate="Z"
        ),
    )
    effect = study.identify(ControlledDirectEffect(intermediate=1.0))
    methods = {record.name: record for record in effect.available_methods()}
    assert methods["tmle"].available
    for name in ("collaborative_tmle", "drtmle"):
        assert not methods[name].available
        assert "controlled direct effect" in (methods[name].reason or "")

    monkeypatch.setattr("cleverly.study.CTMLE", _MustNotConstruct)
    monkeypatch.setattr("cleverly.study.DRTMLE", _MustNotConstruct)
    for name in ("collaborative_tmle", "drtmle"):
        with pytest.raises(CapabilityError, match="controlled direct effect"):
            effect.estimate(method=name)


class _MustNotConstruct:
    """An engine stand-in proving a refusal happened before nuisance fitting."""

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("nuisance fitting path was reached")


@pytest.mark.parametrize(
    ("estimand", "reason"),
    [
        ("ate", "not a typed causal estimand"),
        (None, "not a typed causal estimand"),
        (ATE, "not a typed causal estimand"),
    ],
)
def test_an_untyped_estimand_is_refused_by_name(estimand, reason, monkeypatch) -> None:
    """``identify("ate")`` used to die on ``'str' object has no attribute 'name'``.

    Every legacy call site spelled its estimands as strings, so that is the first thing a
    migrating reader tries.  The provider dereferences ``estimand.name`` as its opening move,
    so the failure surfaced from inside identification with nothing pointing at the typed
    object to pass instead.  ``ATE`` the *class* is here too: it has a ``name`` attribute, so
    it got further than a string did and failed later and less legibly.
    """
    monkeypatch.setattr("cleverly.study.TMLE", _MustNotConstruct)
    with pytest.raises(CapabilityError, match=reason) as raised:
        _study().identify(estimand)
    if estimand == "ate":
        assert "ATE()" in str(raised.value)


def test_a_string_contrast_on_a_controlled_direct_effect_is_refused_at_construction() -> None:
    with pytest.raises(DataError, match="typed arm contrast"):
        ControlledDirectEffect(intermediate=1.0, contrast="ate")


@pytest.mark.parametrize(
    ("estimand", "reason"),
    [
        (RegimeMean(regimens=(), horizons=(1, 2)), "one time point"),
        (RegimeContrast(regimens=(), horizons=(1,)), "one time point"),
        (MSMProjection(MSM.linear(), horizons=(1,)), "one time point"),
        (MSMProjection(MSM.linear(), regimens={"always": 1}), "longitudinal regimen cells"),
    ],
)
def test_a_sequential_declaration_on_a_point_design_is_refused(
    estimand, reason, monkeypatch
) -> None:
    """A declaration that cannot take effect is refused, not dropped.

    ``horizons=`` and ``MSMProjection(regimens=...)`` are read only on the longitudinal path.
    On a point design they were silently discarded, so the fit answered a different question
    from the one written down and reported it under the name of the one asked for.
    """
    monkeypatch.setattr("cleverly.study.TMLE", _MustNotConstruct)
    with pytest.raises(CapabilityError, match=reason):
        _study().identify(estimand)


def test_a_continuous_dose_refuses_by_axis_and_admits_a_working_model() -> None:
    """The dose rule is about the parameter axis, and ``msm`` is not an arm axis.

    The refusal named targets rather than axes, so a continuous-dose ``MSMProjection`` was
    turned away with the message "is arm-indexed" -- which ``TARGETS["msm"].parameter_axis``
    contradicts, and which the engine contradicts too: ``tests/unit/test_continuous_msm.py``
    fits exactly this composition.  The two genuinely arm-shaped axes stay refused, and now
    say which axis they are.
    """
    rng = np.random.default_rng(11)
    n = 180
    w = rng.normal(size=n)
    a = 0.4 * w + rng.normal(size=n)
    frame = pd.DataFrame({"Y": 1.0 + 2.0 * a + 0.3 * w, "A": a, "W": w})
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous"
        ),
    )

    result = study.estimate(
        MSMProjection(MSM.linear(doses=np.linspace(-1.5, 1.5, 9))),
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        cross_fit=False,
        density_bins=8,
        simultaneous=False,
        random_state=3,
    )
    # The same slope and score the engine-level test pins, reached through the typed API.
    assert result["msm[a]"].psi == pytest.approx(2.0, abs=2e-6)
    assert abs(result["msm[a]"].score) < 1e-10
    assert result.parameter_keys["msm[a]"].term == "a"

    for refused, axis in ((ATE(), "arm"), (RegimeMean(regimens=()), "regime")):
        with pytest.raises(CapabilityError, match=f"indexed by {axis}"):
            study.identify(refused)


def test_keyword_shortcuts_normalize_to_the_same_typed_method() -> None:
    normalized = TMLEMethod().with_overrides(
        outcome_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        n_folds=4,
        random_state=7,
    )
    declared = TMLEMethod(
        models=ModelSpec(
            outcome_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
        ),
        cross_fitting=CrossFitting(n_folds=4),
        runtime=Runtime(random_state=7),
    )
    assert dataclasses.replace(normalized, models=ModelSpec()) == dataclasses.replace(
        declared, models=ModelSpec()
    )
    assert type(normalized.models.outcome_learner) is type(declared.models.outcome_learner)
    assert (
        normalized.models.outcome_learner.get_params()
        == declared.models.outcome_learner.get_params()
    )
    assert type(normalized.models.treatment_learner) is type(declared.models.treatment_learner)
    assert (
        normalized.models.treatment_learner.get_params()
        == declared.models.treatment_learner.get_params()
    )


def test_a_shortcut_named_like_a_field_sets_that_field() -> None:
    """The rule that stops a keyword from silently moving a different knob.

    ``alpha=`` mapped to ``Targeting.submodel_alpha`` -- the 0.9995 logistic-submodel bound --
    while the field named ``alpha`` was ``Inference.alpha``, the interval level.  Both are
    floats near zero-to-one, so nothing raised and nothing looked wrong; the interval simply
    stayed at 95% while the shrink bound moved.  This checks the general property rather than
    that one pair, so the next shortcut added cannot reintroduce it.
    """
    fields = {
        group: {f.name for f in dataclasses.fields(getattr(TMLEMethod(), group))}
        for group in SHORTCUTS
    }
    for group, mapping in SHORTCUTS.items():
        for shortcut, attribute in mapping.items():
            assert attribute in fields[group], f"{group}.{attribute} is not a field"
            owners = [other for other, names in fields.items() if shortcut in names]
            assert owners in ([], [group]), (
                f"shortcut {shortcut!r} sets {group}.{attribute} but is also the name of a "
                f"field on {owners}; a caller will reasonably expect it to set that one"
            )


def test_auto_resolves_to_each_engines_own_default_and_says_so_when_they_move() -> None:
    """``Inference`` serves two engines whose defaults are not the same number.

    ``n_multiplier`` was a single literal 1000, which matches ``TMLE`` and silently halved
    every study-driven longitudinal fit: ``LTMLE`` draws 2000. Nothing reported the change,
    and the parity suite could not see it because it turns simultaneous bands off. ``"auto"``
    now defers to the engine, exactly as ``g_bounds`` already did.

    The resolved values are restated in ``cleverly.methods`` rather than imported, so this
    reads both engine signatures and fails when a restatement stops matching -- otherwise the
    duplication would be free to drift back apart.
    """
    point = inspect.signature(TMLE).parameters
    sequential = inspect.signature(LTMLE).parameters
    assert point["n_multiplier"].default == DEFAULT_POINT_MULTIPLIER
    assert sequential["n_multiplier"].default == DEFAULT_LONGITUDINAL_MULTIPLIER
    assert DEFAULT_POINT_MULTIPLIER != DEFAULT_LONGITUDINAL_MULTIPLIER, (
        "if the engines ever agree, delete the sentinel rather than keeping a split that "
        "no longer splits anything"
    )
    assert DEFAULT_LONGITUDINAL_G_BOUNDS == DEFAULT_LTMLE_G_BOUNDS

    method = TMLEMethod()
    assert method.inference.n_multiplier == "auto"
    assert method.estimator_kwargs()["n_multiplier"] == DEFAULT_POINT_MULTIPLIER
    assert (
        method.estimator_kwargs(longitudinal=True)["n_multiplier"]
        == DEFAULT_LONGITUDINAL_MULTIPLIER
    )
    # An explicit request still wins on both paths.
    asked = TMLEMethod().with_overrides(n_multiplier=400)
    assert asked.estimator_kwargs()["n_multiplier"] == 400
    assert asked.estimator_kwargs(longitudinal=True)["n_multiplier"] == 400


def test_the_interval_level_and_the_submodel_bound_are_reachable_separately() -> None:
    method = TMLEMethod().with_overrides(alpha=0.10, submodel_alpha=0.99)
    assert method.inference.alpha == 0.10
    assert method.targeting.submodel_alpha == 0.99
    # And they land on the engine under the names it uses, which are the other way around.
    kwargs = method.estimator_kwargs()
    assert kwargs["alpha_sig"] == 0.10
    assert kwargs["alpha"] == 0.99
    with pytest.raises(MethodConfigurationError, match="alpha_sig") as raised:
        TMLEMethod().with_overrides(alpha_sig=0.10)
    assert isinstance(raised.value, CleverlyError)


@pytest.mark.parametrize(
    "option",
    [
        {"missingness_learner": sklearn.linear_model.LogisticRegression(max_iter=1000)},
        {"intermediate_learner": sklearn.linear_model.LogisticRegression(max_iter=1000)},
        {"density_bins": 12},
        {"screen_treatment": True},
        {"screen_threshold": 0.2},
        {"min_retain": 2},
        {"repeats": 2},
        {"stratify_folds": "outcome"},
        {"targeting_scheme": "foldwise"},
        {"cv_evaluation": True},
        {"fluctuation": "linear"},
        {"targeting": "one_step"},
        {"nuisance_bound": 0.02},
        {"target_weights": True},
        {"step_size": 0.002},
        {"n_bootstrap": 500},
        {"bootstrap_resampling": "iid"},
    ],
    ids=lambda option: next(iter(option)),
)
def test_every_point_only_option_is_refused_by_longitudinal_translation(
    option: dict[str, object],
) -> None:
    """A normalized declaration must reach the engine or fail before construction.

    These are the 18 point-only fields in the shared configuration. Sixteen used to be
    accepted and omitted from the longitudinal kwargs; ``repeats`` alone had a bespoke
    refusal. Pinning the whole list prevents a future field from disappearing just because
    the two engine signatures differ.
    """
    name = next(iter(option))
    method = TMLEMethod().with_overrides(**option)
    with pytest.raises(MethodConfigurationError, match=name) as raised:
        method.estimator_kwargs(longitudinal=True)
    assert isinstance(raised.value, CleverlyError)


def test_cross_fit_false_keeps_its_supported_longitudinal_meaning() -> None:
    method = TMLEMethod().with_overrides(cross_fit=False)
    assert method.estimator_kwargs(longitudinal=True)["n_folds"] == 1


@pytest.mark.parametrize("method", [CollaborativeTMLEMethod(), DRTMLEMethod()])
def test_variant_longitudinal_refusals_use_the_library_error_hierarchy(method) -> None:
    with pytest.raises(MethodConfigurationError) as raised:
        method.estimator_kwargs(longitudinal=True)
    assert isinstance(raised.value, CleverlyError)


def test_a_longitudinal_option_refuses_before_engine_construction(monkeypatch) -> None:
    frame, _ = make_longitudinal(n=100, seed=23)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
    )
    effect = study.identify(RegimeMean({"always": 1}))
    monkeypatch.setattr("cleverly.study.LTMLE", _MustNotConstruct)
    with pytest.raises(MethodConfigurationError, match="n_bootstrap"):
        effect.estimate(n_bootstrap=500)


def test_estimation_options_cannot_reassign_study_roles() -> None:
    effect = _study().identify(ATE())
    with pytest.raises(
        MethodConfigurationError, match="Study-design roles belong on PointTreatment"
    ):
        effect.estimate(covariates=["W1"])


def test_an_invalid_method_declaration_uses_the_library_error_hierarchy() -> None:
    effect = _study().identify(ATE())
    with pytest.raises(MethodConfigurationError) as raised:
        effect.estimate(method=object())  # type: ignore[arg-type]
    assert isinstance(raised.value, CleverlyError)


# ------------------------------------------------------------- the missing-outcome boundary


def _complete_indicator_study() -> CausalStudy:
    """A design that declares a response indicator which is identically one.

    The outcome is binary so that ``paf`` -- which needs counterfactual risks -- is one of
    the three estimands the row below reaches.
    """
    frame, _ = make_binary_outcome(n=200, seed=19)
    frame = frame.copy()
    frame["Delta"] = 1
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2"),
            missingness="Delta",
        ),
    )


@pytest.mark.parametrize(
    "estimand",
    [NaturalCourseMean(), PopulationAttributableRisk(), PopulationAttributableFraction()],
    ids=["ey_obs", "par", "paf"],
)
def test_a_declared_but_complete_response_indicator_identifies_the_observed_law(
    estimand: Any,
) -> None:
    """The refusal keys on the outcome being missing, not on the declaration.

    ``TargetContext.observed_mean`` -- the engine guard this refusal fronts -- keys on the
    observation mask, and docs/roadmap.md RM8 states the stop as "refused when outcomes
    are missing".  A response indicator that is identically one leaves no missing outcome,
    so E[Y] is exactly the empirical mean and the estimator fits it.  Keying the refusal
    on ``missingness=`` refused a composition the engine performs.
    """
    study = _complete_indicator_study()
    assert not study.data.has_missing_outcome
    effect = study.identify(estimand)
    assert effect.functional.target in {"ey_obs", "par", "paf"}
    result = effect.estimate(**FAST_KWARGS)
    assert np.isfinite(result.psi())
    # No response mechanism is fitted, which is what the empty/short nuisance list claims.
    assert result.nuisance.missingness is None


def test_the_natural_course_mean_matches_the_empirical_mean_exactly() -> None:
    """A nonzero witness for the row above: the fit is the empirical mean, not a surrogate."""
    study = _complete_indicator_study()
    result = study.identify(NaturalCourseMean()).estimate(**FAST_KWARGS)
    outcome = np.asarray(study.data.outcome, dtype=float)
    assert result.psi("ey_obs") == pytest.approx(float(np.mean(outcome)), rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("estimand", "roadmap"),
    [
        (PopulationAttributableRisk(), "RM8"),
        (PopulationAttributableFraction(), "RM8"),
    ],
    ids=["par", "paf"],
)
def test_a_genuinely_missing_outcome_still_refuses_the_attributable_interventions(
    estimand: Any, roadmap: str
) -> None:
    frame, _ = make_linear_ate(n=120, seed=19)
    frame = frame.copy()
    frame["Delta"] = 1
    frame.loc[frame.index[:20], "Delta"] = 0
    frame.loc[frame.index[:20], "Y"] = np.nan
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
            missingness="Delta",
        ),
    )
    assert study.data.has_missing_outcome
    with pytest.raises(CapabilityError, match=rf"score equation.*{roadmap}"):
        study.identify(estimand)


def test_one_refusal_sentence_and_one_exception_type_serve_all_three_call_sites() -> None:
    """The identification, engine and target-context refusals are one implementation.

    Three near-identical sentences, three copies of the estimand set and three exception
    types -- ``ValueError``, ``NotImplementedError`` and ``CapabilityError`` -- had drifted
    apart, and docs/architecture-invariants.md requires a method-configuration failure to
    derive from ``CleverlyError``.
    """
    assert frozenset({"par", "paf"}) == POPULATION_INTERVENTION_TARGETS
    frame, _ = make_linear_ate(n=120, seed=19)
    frame = frame.copy()
    frame["Delta"] = 1
    frame.loc[frame.index[:20], "Delta"] = 0
    frame.loc[frame.index[:20], "Y"] = np.nan
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
            missingness="Delta",
        ),
    )
    with pytest.raises(CapabilityError) as identified:
        study.identify(PopulationAttributableRisk())
    with pytest.raises(CapabilityError) as fitted:
        TMLE(estimands=("par",), **FAST_KWARGS).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W1", "W2", "W3", "W4"),
            delta="Delta",
        )
    for raised in (identified, fitted):
        assert isinstance(raised.value, CleverlyError)
        assert "outcome/missingness score equation" in str(raised.value)
        assert "docs/roadmap.md RM" in str(raised.value)
    # The message is built once, so the shared clause is byte-identical at both sites.
    shared = "under missingness at random the natural-course mean E[Y] needs an additional"
    assert shared in str(identified.value)
    assert shared in str(fitted.value)


def test_the_sibling_intermediate_refusal_is_a_cleverly_error_too() -> None:
    """The fourth site, three lines below the third, and the one type it did not share.

    ``ey_obs``, ``par`` and ``paf`` refuse ``delta=`` and ``intermediate=`` for two
    different reasons, so the two sentences stay separate.  The exception hierarchy is not
    a reason: docs/architecture-invariants.md says a caller never has to catch an
    implementation-language exception beside a library one, and this refusal raised
    ``NotImplementedError`` while its neighbour raised ``CapabilityError``.
    """
    frame, _ = make_linear_ate(n=120, seed=21)
    frame = frame.assign(Z=(frame["W1"] > 0).astype(int))
    with pytest.raises(CapabilityError) as raised:
        TMLE(estimands=("par",), **FAST_KWARGS).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W1", "W2"),
            intermediate="Z",
        )
    assert isinstance(raised.value, CleverlyError)
    assert "do not yet support intermediate=" in str(raised.value)


def test_the_target_context_refusal_is_a_cleverly_error_and_still_a_value_error() -> None:
    """A fold dropping ``paf`` catches ``ValueError``; the hierarchy has to keep that."""
    error = population_intervention_refusal(POPULATION_INTERVENTION_TARGETS, declaration="delta=")
    assert isinstance(error, CleverlyError)
    assert isinstance(error, ValueError)
    assert isinstance(error, CapabilityError)
    assert "par and paf do not yet support delta=" in str(error)


# ------------------------------------------------------------------ no silent fall-through


def test_an_unregistered_arm_target_refuses_instead_of_printing_another_functional() -> None:
    """A registered arm target with no entry is refused, not given ``ey``'s functional.

    The fall-through returned ``E_W[E(Y | A=a, W)]`` -- the counterfactual mean's
    functional -- under whatever name the new target carried, in the record ``summary()``
    prints and ``save()`` persists.
    """
    functional = _study().identify(ATE()).functional
    invented = dataclasses.replace(functional, target="ate_but_different")
    with pytest.raises(CapabilityError, match="has no identified expression"):
        _ = invented.expression


def test_an_unregistered_arm_target_refuses_instead_of_borrowing_positivity_prose() -> None:
    study = _study()
    design = study.design
    assert isinstance(design, PointTreatment)
    functional = study.identify(ATE()).functional
    invented = dataclasses.replace(functional, target="ate_but_different")
    with pytest.raises(CapabilityError, match="states no design-bound support of its own"):
        _arm_scope(design, invented)


#: The one registered arm target with no row in ``_ARM_STATEMENTS``, and why it needs
#: none. Its record keeps only the complete-outcome assumption, so the two-arm positivity
#: sentence whose branch resolves the arms is dropped before that branch runs, and its
#: functional asks nothing of the treatment mechanism.
_NO_ARM_SUPPORT_TO_STATE = "ey_obs"


def test_the_arm_statement_rows_cover_every_target_that_can_reach_them() -> None:
    """The completeness claim, read off ``_ARM_STATEMENTS`` rather than through a table.

    ``ARM_ESTIMAND_TABLE`` covered the same ground transitively, which left the rows
    themselves referenced by no test at all: a dead row and a deleted one both passed.

    A target reaches ``_arm_scope`` when its registered assumptions carry the two-arm
    positivity sentence, because the branch that resolves the arms is the branch that
    rewrites that sentence.  ``ey_obs`` carries it and still never arrives: its own filter
    keeps the complete-outcome assumption alone, and drops every other registered sentence
    before that branch.
    """
    declared = {
        name
        for name, target in TARGETS.items()
        if any(
            item.startswith(TWO_ARM_POSITIVITY_PREFIX) for item in target.identification.assumptions
        )
    }
    assert _NO_ARM_SUPPORT_TO_STATE in declared
    assert _NO_ARM_SUPPORT_TO_STATE not in _ARM_STATEMENTS
    assert set(_ARM_STATEMENTS) | {_NO_ARM_SUPPORT_TO_STATE} == declared


def test_the_natural_course_mean_resolves_no_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exclusion above is a fact about the code, not a note beside it.

    ``ey_obs`` carried a row returning an empty positivity sentence. The row never ran,
    and the caller dropped that empty sentence in silence, so a target that did reach it
    would have lost its statement rather than been refused.
    """

    def _must_not_resolve(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ey_obs reached _arm_scope")

    monkeypatch.setattr("cleverly.study._arm_scope", _must_not_resolve)
    effect = _study().identify(NaturalCourseMean())
    assert effect.functional.target == _NO_ARM_SUPPORT_TO_STATE
    assert not any("positivity" in item for item in effect.identification.assumptions)


def test_a_row_stating_no_treatment_support_is_refused_rather_than_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty sentence reads as "this target needs none" and prints as nothing."""
    monkeypatch.setitem(_ARM_STATEMENTS, "ate", lambda scope: ("", "at every arm"))
    with pytest.raises(CapabilityError, match="states no treatment support"):
        _study().identify(ATE())


def test_a_conditioning_event_names_the_population_it_conditions_on() -> None:
    study = _study()
    design = study.design
    assert isinstance(design, PointTreatment)
    functional = study.identify(ATT()).functional
    invented = dataclasses.replace(functional, target="ate")
    with pytest.raises(CapabilityError, match="names no population it conditions on"):
        _point_identification(TARGETS["att"].identification, design, invented)


def test_the_printed_expression_and_positivity_resolve_one_reference_arm() -> None:
    """Two implementations of the reference-arm default could print different arms."""
    frame, _ = make_multi_arm(n=180, seed=17)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    design = study.design
    assert isinstance(design, PointTreatment)
    for estimand in (ATE(), ATE(reference="medium"), ATT(reference="high")):
        effect = study.identify(estimand)
        scope, _, _ = _arm_scope(design, effect.functional)
        assert scope.reference == effect.functional.reference_arm
        assert repr(scope.reference) in effect.functional.expression


# ------------------------------------------------------------------- the unsupported arm


def test_an_unavailable_treatment_is_refused_at_identification() -> None:
    """Deleting the guard left the whole unit suite green and printed ``E(Y | A=7, W)``."""
    with pytest.raises(DataError, match=r"treatment 7 is not available; choose from"):
        _study().identify(CounterfactualMean(treatment=7))


def test_an_unavailable_treatment_is_refused_when_the_narrowing_runs() -> None:
    """The second copy of the guard, reached only when a record names a missing arm."""
    frame, _ = make_multi_arm(n=180, seed=17)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    effect = study.identify(CounterfactualMean(treatment="medium"))
    assert effect.functional.target == "ey"
    result = effect.estimate(**FAST_KWARGS)
    tampered = dataclasses.replace(effect, estimand=CounterfactualMean(treatment="nowhere"))
    with pytest.raises(DataError, match=r"treatment 'nowhere' is not available; choose from"):
        tampered._select_point_parameters(result, result.method)


# --------------------------------------------------- the design-bound double-robustness


def test_a_missing_outcome_design_binds_the_double_robustness_condition() -> None:
    """``g`` alone is not sufficient once a response mechanism enters the product."""
    effect = CausalStudy(
        discrete_law_mar.frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"),
    ).identify(ATE())
    condition = effect.identification.dr_condition
    assert "g * P(Delta = 1 | A, W)" in condition
    assert "either Qbar(A, W) or g(W) is consistent" not in condition
    complete = _study().identify(ATE()).identification.dr_condition
    assert "either Qbar(A, W) or g(W) is consistent" in complete


#: What ``att`` and ``atc`` say about their own remainder, and what ``msm`` says about
#: its own.  Neither is a fact a design decides, so a narrowed record keeps both.  ``ate``
#: is the fourth registered case and has no such tail: its sentence writes the mechanism
#: half as a condition, which a design resolves rather than adds to.
_CONDITIONING_TAIL = "randomness of the conditioning event"
_PROJECTION_TAIL = "beta is defined as a projection"

_MAR_DESIGN = PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta")
_CDE_MAR_DESIGN = PointTreatment(
    outcome="Y", treatment="A", adjustment=("W",), missingness="Delta", intermediate="Z"
)


@pytest.mark.parametrize(
    ("design", "product", "rows"),
    [
        (
            _MAR_DESIGN,
            "g * P(Delta = 1 | A, W)",
            (
                ("att", ATT(), _CONDITIONING_TAIL),
                ("atc", ATC(), _CONDITIONING_TAIL),
                ("msm", MSMProjection(MSM.linear()), _PROJECTION_TAIL),
            ),
        ),
        # A design declaring intermediate= identifies a ControlledDirectEffect and nothing
        # else, and that estimand takes an arm contrast, so ``msm`` has no row here.
        (
            _CDE_MAR_DESIGN,
            "g * q_z * pi",
            (
                (
                    "att",
                    ControlledDirectEffect(intermediate=0.0, contrast=ATT()),
                    _CONDITIONING_TAIL,
                ),
                (
                    "atc",
                    ControlledDirectEffect(intermediate=0.0, contrast=ATC()),
                    _CONDITIONING_TAIL,
                ),
            ),
        ),
    ],
    ids=["mar", "cde-mar"],
)
def test_a_narrowed_remainder_keeps_what_the_target_says_about_its_own(
    design: PointTreatment, product: str, rows: tuple[tuple[str, Any, str], ...]
) -> None:
    """The design writes the mechanism half, and the target keeps the rest of its sentence.

    Replacing the whole registered sentence deleted the conditioning-event term and the
    projection note.  No test read either one: both appeared nowhere outside the registry,
    so every suite stayed green while the record lost half of what it says.
    """
    frame = discrete_law_cde.frame() if design.intermediate else discrete_law_mar.frame()
    study = CausalStudy(frame, design=design)
    for target, estimand, phrase in rows:
        condition = study.identify(estimand).identification.dr_condition
        registered = TARGETS[target].identification.dr_condition
        # The tail is read off the registry, so rewording it below the head stays free.
        # The phrase is pinned too, so deleting the tail from the registry fails here
        # rather than leaving the comparison to hold vacuously.
        kept = registered.removeprefix(DR_MECHANISM_HEAD)
        assert phrase in kept
        assert kept != registered
        assert condition.endswith(kept)
        # ... and the design-bound mechanism half stands where the registered head was.
        assert product in condition
        assert DR_MECHANISM_HEAD not in condition


def test_a_narrowed_mean_remainder_resolves_its_condition_rather_than_extending_it() -> None:
    """``ate`` states the mechanism half as a condition, so a design answers it.

    The fourth row of the case above: the registered sentence for the counterfactual mean
    ends with the clause a study resolves, and nothing else follows it, so the design-bound
    statement stands alone.
    """
    registered = TARGETS["ate"].identification.dr_condition
    assert registered.endswith(DR_DESIGN_CONDITIONAL_CLAUSE)
    condition = (
        CausalStudy(
            discrete_law_mar.frame(),
            design=PointTreatment(
                outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"
            ),
        )
        .identify(ATE())
        .identification.dr_condition
    )
    assert "with delta=" not in condition
    assert condition.startswith("consistent if either Qbar(A, W) is consistent")


def test_a_conditional_caveat_survives_when_its_replacement_does_not_apply() -> None:
    """The shift target's ``with delta=:`` paragraph is dropped only when it is replaced."""
    frame, _ = make_shift_dose(n=200, seed=11)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2"), treatment_kind="continuous"
        ),
    )
    effect = study.identify(ModifiedTreatmentPolicy(shifts=(0.0, 0.5)))
    assumptions = effect.identification.assumptions
    assert any(item.startswith(MISSINGNESS_CAVEAT_PREFIX) for item in assumptions)
    assert any(item.startswith(INTERMEDIATE_CAVEAT_PREFIX) for item in assumptions)


# ------------------------------------------------------------ the legacy provenance guard


def _legacy_shaped(effect: Any) -> Any:
    """Return ``effect`` with a functional written before the design-bound fields existed."""
    return dataclasses.replace(
        effect,
        functional=legacy_without(effect.functional, *BackdoorMeanContrast._SCHEMA_1_FIELDS),
    )


def test_every_field_of_the_functional_is_classified_for_provenance() -> None:
    """A new field on the record forces a decision rather than shipping unchecked.

    ``_SCHEMA_1_FIELDS`` is written by hand, and ``_tamperings`` refuses a matrix that
    does not cover it.  Neither notices a field the tuple never names: ``_restored_without``
    raises on a *renamed* field only, so a field added to the record and not to the tuple
    keeps every refusal suite green while nothing forges it and the legacy reconstruction
    leaves it in place.
    """
    declared = {field.name for field in dataclasses.fields(BackdoorMeanContrast)}
    classified = set(BackdoorMeanContrast._SCHEMA_1_FIELDS) | set(PRE_SCHEMA_1_FIELDS)
    assert declared == classified, (
        f"unclassified fields {sorted(declared - classified)}; stale names "
        f"{sorted(classified - declared)}. Decide for each new field: if a design writes "
        "it, add it to BackdoorMeanContrast._SCHEMA_1_FIELDS and a forged value to "
        "tests.pickles._FORGED_FUNCTIONAL_VALUES, so the provenance matcher reconstructs "
        "a record without it and every tampering surface gets a row for it; if every "
        "record ever written carries it, add it to tests.pickles.PRE_SCHEMA_1_FIELDS. A "
        "renamed field needs renaming in whichever tuple names it."
    )
    assert BackdoorMeanContrast._SCHEMA_1_FIELDS[
        -len(BackdoorMeanContrast._TRANSITIONAL_FIELDS) :
    ] == (BackdoorMeanContrast._TRANSITIONAL_FIELDS)


def test_a_legacy_shaped_record_with_an_altered_identification_is_refused() -> None:
    """The witness the legacy branch never had.

    Deleting ``and actual == registered`` from the static-legacy disjunct left 510 tests
    across the four provenance files green, because nothing ever paired a legacy-shaped
    functional with an identification that was not the registry's own.
    """
    effect = _study().identify(ATE())
    assert effect._study is not None
    registered = TARGETS["ate"].identification
    legacy = _legacy_shaped(effect)
    assert _matches_registered_point_identification(
        registered, registered, legacy, effect._study.data, effect.functional.axis
    )
    for altered in (
        dataclasses.replace(registered, assumptions=("same citations, different claim",)),
        dataclasses.replace(registered, required_nuisances=("outcome_regression",)),
        dataclasses.replace(registered, dr_condition="whatever the caller wants"),
    ):
        assert altered.references == registered.references
        assert not _matches_registered_point_identification(
            altered, registered, legacy, effect._study.data, effect.functional.axis
        )


def test_an_untouched_legacy_record_still_replays() -> None:
    """Back-compat stays witnessed: a plain design's legacy record is still accepted."""
    study = _study()
    for estimand in (ATE(), ATT(), ATC()):
        effect = study.identify(estimand)
        assert effect._study is not None
        registered = TARGETS[effect.functional.target].identification
        assert _matches_registered_point_identification(
            registered,
            registered,
            _legacy_shaped(effect),
            effect._study.data,
            effect.functional.axis,
        )


def _intermediate_only_frame() -> pd.DataFrame:
    """A complete-outcome sample carrying an intermediate column.

    ``discrete_law_cde`` cannot serve this row: its ``Y`` is ``NaN`` wherever ``Delta`` is
    zero, so a design that declares no ``missingness=`` refuses it before identification.
    """
    frame, _ = make_linear_ate(n=120, seed=21)
    return frame.assign(Z=(frame["W1"] > 0).astype(int))


@pytest.mark.parametrize(
    ("design", "frame"),
    [
        (
            PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"),
            discrete_law_mar.frame,
        ),
        (
            PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W",),
                missingness="Delta",
                intermediate="Z",
            ),
            discrete_law_cde.frame,
        ),
        # The third row is the witness the guard's second half never had. Both rows above
        # declare missingness=, so ``design.intermediate is None`` could be deleted from
        # the admissibility test with 214 selected tests still passing. A design that
        # declares intermediate= alone is supported, and a schema-0 record cannot have
        # come from one either.
        (
            PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"), intermediate="Z"),
            _intermediate_only_frame,
        ),
    ],
    ids=["mar", "cde-mar", "cde"],
)
def test_a_legacy_record_is_refused_when_the_design_declares_a_further_mechanism(
    design: PointTreatment, frame: Callable[[], pd.DataFrame]
) -> None:
    """The version discriminator was a downgrade switch, and this closes it.

    A record predating the five design-bound fields carries the registry's generic
    identification, which states neither missingness at random nor response positivity nor
    intermediate positivity.  A design declaring ``missingness=`` or ``intermediate=``
    cannot have produced one -- both compositions post-date the schema -- so accepting the
    pairing would replay against assumptions the record does not state.  It was reachable
    only because ``simulated_confounding`` and ``refute`` refuse missing-outcome results
    earlier, and nothing pinned that ordering.
    """
    estimand: Any = ControlledDirectEffect(intermediate=0.0) if design.intermediate else ATE()
    effect = CausalStudy(frame(), design=design).identify(estimand)
    assert effect._study is not None
    registered = TARGETS["ate"].identification
    assert not _matches_registered_point_identification(
        registered,
        registered,
        _legacy_shaped(effect),
        effect._study.data,
        effect.functional.axis,
    )
    # The complete design-bound record is still accepted, so the refusal is about the
    # legacy shape rather than about the design.
    assert _matches_registered_point_identification(
        effect.identification,
        registered,
        effect,
        effect._study.data,
        effect.functional.axis,
    )


def test_the_matcher_refuses_a_record_whose_study_did_not_come_back_from_disk() -> None:
    """``IdentifiedEffect`` says restored metadata may omit ``_study``; joblib retains it.

    Pinned rather than left to chance, because the matcher's answer for a record without a
    study is "refuse", and a caller reading the docstring could expect the opposite.
    """
    effect = _study().identify(ATE())
    assert effect._study is not None
    data = effect._study.data
    detached = dataclasses.replace(effect, _study=None)
    assert detached._study is None
    assert not _matches_registered_point_identification(
        effect.identification,
        TARGETS["ate"].identification,
        detached,
        data,
        effect.functional.axis,
    )


def test_a_tampered_intermediate_refuses_rather_than_raising_out_of_the_matcher() -> None:
    """``float("x")`` used to escape the guard, because the narrowing sat outside the try."""
    effect = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    ).identify(ControlledDirectEffect(intermediate=0.0))
    assert effect._study is not None
    tampered = dataclasses.replace(
        effect, estimand=ControlledDirectEffect.__new__(ControlledDirectEffect)
    )
    object.__setattr__(tampered.estimand, "intermediate", "x")
    object.__setattr__(tampered.estimand, "contrast", ATE())
    object.__setattr__(tampered.estimand, "name", "controlled_direct_effect")
    assert not _matches_registered_point_identification(
        effect.identification,
        TARGETS["ate"].identification,
        tampered,
        effect._study.data,
        effect.functional.axis,
    )


def test_the_study_declares_no_nuisance_name_the_registry_does_not_own() -> None:
    """The four names are declared once, beside ``Identification``."""
    effect = CausalStudy(
        discrete_law_cde.frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            missingness="Delta",
            intermediate="Z",
        ),
    ).identify(ControlledDirectEffect(intermediate=0.0))
    assert set(effect.identification.required_nuisances) <= set(POINT_NUISANCES)
    assert {INTERMEDIATE_MECHANISM, MISSINGNESS_MECHANISM} <= set(
        effect.identification.required_nuisances
    )

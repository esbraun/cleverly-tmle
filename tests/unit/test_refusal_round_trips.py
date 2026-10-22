"""A saved result keeps what a refusal and a declaration are made of.

``result.save()`` pickles the whole fitted graph, so a field that the graph holds survives a
load. These tests name the fields the other refusal tests read, so that a later refactor
cannot drop one silently: the refusal record inside an assessment item, the policy and
regimen declarations, the fold assignment, the identification assumptions, the inference
status of every estimate and the summary text.

Every comparison has a nonzero witness. A record with the default code would compare equal
to a record rebuilt with the default code, so the records here carry a code other than
``invalid_declaration`` and an alternative, and a check on the original run first states
that the value to preserve is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (
    CausalStudy,
    LongitudinalTreatment,
    ModifiedTreatmentPolicyEffect,
    PointTreatment,
    RegimeContrast,
)
from cleverly._inference_status import STATUS_RECORDS
from cleverly.assessment import AssessmentItem, AssessmentStatus
from cleverly.datasets import make_binary_outcome, make_longitudinal, make_shift_dose
from cleverly.estimators import TMLE
from cleverly.exceptions import (
    CapabilityError,
    RefusalReason,
    inference_refusal,
)
from cleverly.interventions import ModifiedPolicy, Piece, Shift
from cleverly.longitudinal import DynamicRegimen
from tests import discrete_law_mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment, linear_in_sample
from tests.unit._simulated_confounding_support import _fit_policy

INF = float("inf")


def _shift_up(dose: Any, history: Any) -> Any:
    return np.asarray(dose, dtype=float) + 0.5


def _shift_down(dose: Any, history: Any) -> Any:
    return np.asarray(dose, dtype=float) - 0.5


def _unit_derivative(dose: Any, history: Any) -> Any:
    return 1.0


def _treat_when_l2_is_high(history: Any) -> Any:
    return (np.asarray(history["L2"]) > 0).astype(float)


def _round_trip(result: Any, tmp_path: Path) -> Any:
    path = tmp_path / "result.joblib"
    result.save(path)
    return cleverly.load(path)


def _statuses(result: Any) -> dict[str, str]:
    return {name: estimate.inference for name, estimate in result.estimates.items()}


# --------------------------------------------------------------------------------------
# (a) A refusal record inside an assessment item.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def natural_course_result() -> Any:
    law = discrete_law_mar.DiscreteLaw()
    study = CausalStudy(
        discrete_law_mar.frame(),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"),
    )
    return study.identify(cleverly.NaturalCourseMean()).estimate(
        outcome_learner=OracleOutcome(law),
        treatment_learner=OracleTreatment(law),
        missingness_learner=OracleMissingness(law),
        cross_fit=False,
        simultaneous=False,
    )


def _unavailable_items(report: Any) -> dict[str, AssessmentItem]:
    return {item.name: item for item in report.items if item.status is AssessmentStatus.UNAVAILABLE}


#: The interval search of ``tipping_gamma``, which the natural-course tilt refuses.
_INTERVAL = {"tipping_gamma": {"use_ci": True}}


def test_an_unavailable_item_keeps_its_refusal_record(
    natural_course_result: Any, tmp_path: Path
) -> None:
    original = _unavailable_items(natural_course_result.sensitivity.run_all(arguments=_INTERVAL))
    # The witness: the items carry records with a non-default code, an alternative and, for
    # the omitted-variable bound, a roadmap id, so equality can fail on a dropped field.
    assert {"omitted_confounding", "tipping_gamma"} <= set(original)
    bound = original["omitted_confounding"].refusal
    assert bound is not None
    assert bound.reason is RefusalReason.IMPLEMENTATION_GAP
    assert bound.roadmap is not None
    assert bound.alternative
    tipping = original["tipping_gamma"].refusal
    assert tipping is not None
    assert tipping.reason is RefusalReason.IMPLEMENTATION_GAP
    assert tipping.alternative and "point search" in tipping.alternative
    assert bound != tipping

    restored = _round_trip(natural_course_result, tmp_path)
    cached = next(
        value
        for key, value in restored.assessment_cache.items()
        if key.startswith("sensitivity.run_all:")
    )
    after = _unavailable_items(cached)
    assert set(after) == set(original)
    for name, item in original.items():
        assert after[name].refusal == item.refusal, name
        assert after[name].refusal is not None
        assert after[name].refusal.reason is item.refusal.reason  # type: ignore[union-attr]
    # A new call on the loaded result answers from the same record.
    again = _unavailable_items(restored.sensitivity.run_all(arguments=_INTERVAL))
    assert {name: item.refusal for name, item in again.items()} == {
        name: item.refusal for name, item in original.items()
    }


def test_the_capability_row_of_a_loaded_result_keeps_its_refusal(
    natural_course_result: Any, tmp_path: Path
) -> None:
    declared = natural_course_result.sensitivity.capability("omitted_confounding")
    assert not declared.available
    assert declared.refusal is not None
    restored = _round_trip(natural_course_result, tmp_path)
    loaded = restored.sensitivity.capability("omitted_confounding")
    assert loaded.refusal == declared.refusal
    assert loaded.reason == declared.reason
    with pytest.raises(CapabilityError) as raised:
        restored.sensitivity.omitted_confounding()
    assert raised.value.refusal.reason is declared.refusal.reason
    assert declared.refusal.condition in raised.value.refusal.condition


def test_a_withheld_interval_refuses_with_the_same_record_after_a_load(tmp_path: Path) -> None:
    frame = make_binary_outcome(n=500, seed=3)[0]
    frame["cluster"] = np.arange(len(frame)) % 2
    status = "no_cluster_variance_plugin"
    result = (
        TMLE(**linear_in_sample(estimands=("ate", "ey0")))
        .fit(frame, outcome="Y", treatment="A", id="cluster")
        .single()
    )
    assert _statuses(result) == dict.fromkeys(result.estimates, status)
    restored = _round_trip(result, tmp_path)
    assert _statuses(restored) == _statuses(result)
    for name, estimate in restored.estimates.items():
        with pytest.raises(CapabilityError) as raised:
            _ = estimate.std_error
        expected = inference_refusal(".std_error", status)
        assert raised.value.refusal == expected
        assert expected.reason is RefusalReason.INFERENCE_GAP
        assert STATUS_RECORDS[status].reason in str(raised.value)
        assert name in result.estimates


# --------------------------------------------------------------------------------------
# (b) A point fit with a ModifiedPolicy, and a stratified fit with a known regime.
# --------------------------------------------------------------------------------------


def _fit_modified_policy() -> Any:
    frame, _ = make_shift_dose(n=120, seed=7)
    frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(float)
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
        ),
    )
    up = ModifiedPolicy(
        "up half",
        pieces=(Piece(-INF, INF, _shift_up, _shift_down, _unit_derivative),),
        policy_kind="known",
    )
    return study.identify(
        ModifiedTreatmentPolicyEffect(policies=(Shift(0.0, cap=None, name="natural"), up))
    ).estimate(
        outcome_learner=LogisticRegression(max_iter=1000),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=7,
        simultaneous=False,
    )


def _fit_stratified_known_regime() -> Any:
    return _fit_policy("stochastic", strata=True)


def _assert_same_fit(restored: Any, original: Any) -> None:
    assert _statuses(restored) == _statuses(original)
    assert tuple(restored.estimates) == tuple(original.estimates)
    for name, estimate in original.estimates.items():
        assert restored.estimates[name].psi == estimate.psi, name
    assert restored.inference_status == original.inference_status
    assert restored.identified_effect.identification.assumptions == (
        original.identified_effect.identification.assumptions
    )
    assert len(original.repeats) == len(restored.repeats)
    for before, after in zip(original.repeats, restored.repeats, strict=True):
        np.testing.assert_array_equal(after.folds.assignment, before.folds.assignment)
        assert after.folds.n_folds == before.folds.n_folds
    assert restored.summary() == original.summary()


@pytest.mark.filterwarnings("ignore::cleverly.exceptions.PositivityWarning")
def test_a_modified_policy_fit_keeps_its_declarations_and_folds(tmp_path: Path) -> None:
    original = _fit_modified_policy()
    # The witnesses: a real cross-fit with two folds, a policy the fit stored, and
    # assumptions that the identification wrote.
    assert original.repeats[0].folds.n_folds == 2
    assert len(np.unique(original.repeats[0].folds.assignment)) == 2
    assert any(isinstance(policy, ModifiedPolicy) for policy in original.estimator.policies)
    assert original.identified_effect.identification.assumptions

    restored = _round_trip(original, tmp_path)
    _assert_same_fit(restored, original)
    assert restored.estimator.policies == original.estimator.policies
    loaded = next(p for p in restored.estimator.policies if isinstance(p, ModifiedPolicy))
    assert loaded.name == "up half"
    assert loaded.policy_kind == "known"
    # The policy behaves as it did: the stored maps are the same functions on the same data.
    assert isinstance(loaded.pieces, tuple)
    piece = loaded.pieces[0]
    assert piece.map is not None and piece.inverse is not None
    doses = np.linspace(-1.0, 3.0, 7)
    np.testing.assert_array_equal(piece.map(doses, None), doses + 0.5)
    np.testing.assert_array_equal(piece.inverse(piece.map(doses, None), None), doses)


def test_a_stratified_known_regime_fit_keeps_its_declarations_and_strata(
    tmp_path: Path,
) -> None:
    original = _fit_stratified_known_regime()
    assert original.data.has_strata
    assert original.data.n_strata > 1
    assert original.estimator.interventions
    restored = _round_trip(original, tmp_path)
    _assert_same_fit(restored, original)
    np.testing.assert_array_equal(restored.data.strata, original.data.strata)
    assert restored.data.n_strata == original.data.n_strata
    assert restored.estimator.interventions == original.estimator.interventions
    assert [i.name for i in restored.estimator.interventions] == [
        i.name for i in original.estimator.interventions
    ]
    kinds = {getattr(i, "density_kind", None) for i in restored.estimator.interventions}
    assert "known" in kinds


# --------------------------------------------------------------------------------------
# (c) A longitudinal fit with a DynamicRegimen rule and cross-fitting.
# --------------------------------------------------------------------------------------


def _fit_dynamic_regimen() -> Any:
    frame, _ = make_longitudinal(n=180, seed=11)
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
    dynamic = DynamicRegimen("dynamic", (1, _treat_when_l2_is_high), rule_kind="known")
    return study.estimate(
        RegimeContrast({"always": 1, "never": 0, "dynamic": dynamic}, reference="never"),
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        learner_folds=2,
        random_state=0,
        simultaneous=False,
    )


def test_a_dynamic_regimen_fit_keeps_its_rule_folds_and_assumptions(tmp_path: Path) -> None:
    original = _fit_dynamic_regimen()
    regimens = {regimen.label: regimen for regimen in original.config.regimens}
    assert set(regimens) == {"always", "never", "dynamic"}
    assert original.folds.n_folds == 3
    assert len(np.unique(original.folds.assignment)) == 3
    assert original.identified_effect.identification.assumptions

    restored = _round_trip(original, tmp_path)
    assert _statuses(restored) == _statuses(original)
    assert tuple(restored.estimates) == tuple(original.estimates)
    for name, estimate in original.estimates.items():
        assert restored.estimates[name].psi == estimate.psi, name
    np.testing.assert_array_equal(restored.folds.assignment, original.folds.assignment)
    assert restored.folds.n_folds == original.folds.n_folds
    assert restored.identified_effect.identification.assumptions == (
        original.identified_effect.identification.assumptions
    )
    assert restored.summary() == original.summary()

    loaded = {regimen.label: regimen for regimen in restored.config.regimens}
    assert set(loaded) == set(regimens)
    assert loaded["dynamic"] == regimens["dynamic"]
    # The rule behaves as it did on the same history, and it is not the always-treat rule.
    history = {"L2": np.array([-1.0, 0.5, 2.0, -0.2])}
    rule = loaded["dynamic"].plan[1]
    assert loaded["dynamic"].rule_kind == "known"
    np.testing.assert_array_equal(rule(history), [0.0, 1.0, 1.0, 0.0])


def test_a_loaded_longitudinal_fit_keeps_its_capability_refusals(tmp_path: Path) -> None:
    original = _fit_dynamic_regimen()
    declared = {
        operation: original.sensitivity.capability(operation)
        for operation in ("omitted_confounding", "evalue", "simulated_confounding")
    }
    # The witness: each row refuses with a code that is not the default and a roadmap id.
    for operation, row in declared.items():
        assert not row.available, operation
        assert row.refusal is not None
        assert row.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert row.refusal.roadmap is not None
    assert len({row.refusal for row in declared.values()}) == 3

    restored = _round_trip(original, tmp_path)
    for operation, row in declared.items():
        assert restored.sensitivity.capability(operation).refusal == row.refusal, operation
        with pytest.raises(CapabilityError) as raised:
            getattr(restored.sensitivity, operation)()
        assert raised.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert raised.value.refusal.roadmap == row.refusal.roadmap
        assert row.refusal.condition in raised.value.refusal.condition

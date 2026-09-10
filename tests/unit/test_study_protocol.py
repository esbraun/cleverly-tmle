"""The causal study protocol record and its result-persistence boundary."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from joblib import hash as joblib_hash
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    ExplicitAdjustmentProvider,
    LongitudinalTreatment,
    PointTreatment,
    Provenance,
    RegimeMean,
    StudyProtocol,
    load,
)
from cleverly.datasets import make_linear_ate, make_longitudinal
from tests.conftest import FAST_KWARGS
from tests.pickles import _LegacyPickle


def _protocol(**overrides: Any) -> StudyProtocol:
    values: dict[str, Any] = {
        "target_population": "Adults eligible for transition navigation",
        "eligibility": ["Discharged alive", "Lives in the service area"],
        "time_zero": "Hospital discharge",
        "treatment_strategies": ["Offer navigation", "Usual care"],
        "treatment_versions": ["Calls at discharge and day seven", "No navigation call"],
        "outcome": "Top-box transition score",
        "horizon": "30 days after discharge",
        "intercurrent_event_handling": ["Use the score regardless of readmission"],
        "interference_unit": "Patient",
        "assumption_rationale": [
            "Recorded baseline variables cover the common causes used for adjustment"
        ],
    }
    values.update(overrides)
    return StudyProtocol(**values)


def test_protocol_normalizes_sequences_and_has_stable_canonical_json() -> None:
    protocol = _protocol()

    assert isinstance(protocol.eligibility, tuple)
    assert isinstance(protocol.treatment_strategies, tuple)
    assert isinstance(protocol.treatment_versions, tuple)
    assert isinstance(protocol.intercurrent_event_handling, tuple)
    assert isinstance(protocol.assumption_rationale, tuple)
    assert protocol.schema_version == 1
    assert protocol.to_dict()["eligibility"] == [
        "Discharged alive",
        "Lives in the service area",
    ]
    assert StudyProtocol.from_dict(protocol.to_dict()) == protocol

    expected = json.dumps(
        protocol.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert protocol.canonical_json == expected
    assert (
        protocol.fingerprint == hashlib.blake2b(expected.encode("utf-8"), digest_size=8).hexdigest()
    )
    assert _protocol(eligibility=tuple(protocol.eligibility)).fingerprint == protocol.fingerprint
    unicode_protocol = _protocol(target_population="Niños eligible for navigation")
    assert "Niños" in unicode_protocol.canonical_json
    assert "\\u00f1" not in unicode_protocol.canonical_json


def test_protocol_is_frozen_and_its_schema_version_cannot_be_supplied() -> None:
    protocol = _protocol()
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.horizon = "90 days"  # type: ignore[misc]
    with pytest.raises(TypeError, match="schema_version"):
        _protocol(schema_version=2)
    with pytest.raises(ValueError, match="schema_version must be 1"):
        StudyProtocol.from_dict({**protocol.to_dict(), "schema_version": 2})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_population", "  "),
        ("time_zero", ""),
        ("outcome", "\t"),
        ("horizon", "\n"),
        ("interference_unit", " "),
        ("eligibility", ["valid", " "]),
        ("treatment_strategies", []),
        ("treatment_versions", ["valid", ""]),
        ("intercurrent_event_handling", []),
        ("assumption_rationale", [" "]),
    ],
)
def test_protocol_rejects_blank_required_text(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        _protocol(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "eligibility",
        "treatment_strategies",
        "treatment_versions",
        "intercurrent_event_handling",
        "assumption_rationale",
    ],
)
def test_protocol_rejects_a_scalar_string_for_every_sequence(field: str) -> None:
    with pytest.raises(TypeError, match="sequence of text entries"):
        _protocol(**{field: "one undivided sentence"})


@pytest.mark.parametrize(
    "value",
    [{"entry"}, {"entry": "value"}, iter(["entry"])],
    ids=["set", "mapping", "generator"],
)
def test_protocol_rejects_unordered_or_one_shot_iterables(value: Any) -> None:
    with pytest.raises(TypeError, match="ordered sequence of text entries"):
        _protocol(eligibility=value)


def test_protocol_rejects_unmatched_strategy_and_version_counts() -> None:
    with pytest.raises(ValueError, match="same number"):
        _protocol(treatment_versions=["One version"])


class _ConflictingProvider:
    name = "conflicting test provider"

    def __init__(self, protocol: StudyProtocol | None) -> None:
        self.protocol = protocol

    def identify(self, study: CausalStudy, estimand: Any) -> Any:
        effect = ExplicitAdjustmentProvider().identify(study, estimand)
        return replace(effect, protocol=self.protocol, _study=None)


@pytest.mark.parametrize("returned", [None, _protocol(horizon="90 days after discharge")])
def test_study_overwrites_a_custom_provider_protocol(returned: StudyProtocol | None) -> None:
    frame, _ = make_linear_ate(n=80, seed=11)
    protocol = _protocol()
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
        protocol=protocol,
    )

    effect = study.identify(ATE(), provider=_ConflictingProvider(returned))

    assert effect.protocol is protocol
    assert effect._study is study


def test_treatment_version_changes_only_protocol_metadata_across_point_fits() -> None:
    frame, _ = make_linear_ate(n=180, seed=12)
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    first_protocol = _protocol()
    second_protocol = _protocol(
        treatment_versions=["One call within two days", "No navigation call"]
    )

    first = CausalStudy(frame, design=design, protocol=first_protocol).estimate(
        ATE(), **FAST_KWARGS
    )
    second = CausalStudy(frame, design=design, protocol=second_protocol).estimate(
        ATE(), **FAST_KWARGS
    )

    assert first.identified_effect.functional == second.identified_effect.functional
    assert joblib_hash(first.config) == joblib_hash(second.config)
    assert joblib_hash(first.method) == joblib_hash(second.method)
    assert joblib_hash(first.repeats) == joblib_hash(second.repeats)
    assert first.provenance.data_fingerprint == second.provenance.data_fingerprint
    assert first.provenance.fold_fingerprint == second.provenance.fold_fingerprint
    assert first.provenance.protocol_fingerprint == first_protocol.fingerprint
    assert second.provenance.protocol_fingerprint == second_protocol.fingerprint
    assert first.provenance.protocol_fingerprint != second.provenance.protocol_fingerprint
    np.testing.assert_array_equal(first.covariance(), second.covariance())
    assert first["ate"].psi == second["ate"].psi
    assert first["ate"].ci == second["ate"].ci
    np.testing.assert_array_equal(first["ate"].influence_curve, second["ate"].influence_curve)


@pytest.fixture(scope="module")
def protocol_point_result() -> Any:
    frame, _ = make_linear_ate(n=140, seed=13)
    return CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
        protocol=_protocol(),
    ).estimate(ATE(), **FAST_KWARGS)


@pytest.fixture(scope="module")
def protocol_longitudinal_result() -> Any:
    frame, _ = make_longitudinal(n=240, seed=33)
    return CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y",
            treatment=("A1", "A2"),
            baseline=("W1", "W2"),
            time_varying=((), ("L2",)),
            censoring=("C1", "C2"),
        ),
        protocol=_protocol(
            treatment_strategies=["Always treat", "Never treat"],
            treatment_versions=["Treatment at both nodes", "No treatment at either node"],
        ),
    ).estimate(
        RegimeMean({"always": 1, "never": 0}, reference="always"),
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=0,
        simultaneous=False,
    )


@pytest.mark.parametrize("fixture_name", ["protocol_point_result", "protocol_longitudinal_result"])
def test_complete_protocol_and_digest_survive_result_round_trip(
    fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    result = request.getfixturevalue(fixture_name)

    restored = load(result.save(tmp_path / f"{fixture_name}.joblib"))

    assert not hasattr(restored, "protocol")
    assert restored.identified_effect.protocol == result.identified_effect.protocol
    assert restored.provenance.protocol_fingerprint == result.provenance.protocol_fingerprint
    assert restored.identified_effect.protocol is restored.identified_effect._study.protocol
    for line in restored.identified_effect.protocol.summary_lines():
        assert line in restored.identified_effect.summary()
        assert line in restored.summary()


def _write_pre_protocol_artifact(result: Any, path: Path) -> None:
    study = result.identified_effect._study
    study_state = {name: value for name, value in vars(study).items() if name != "_protocol"}
    effect_state = {
        name: value
        for name, value in result.identified_effect.__dict__.items()
        if name != "protocol"
    }
    effect_state["_study"] = _LegacyPickle(type(study), study_state)
    provenance_state = {
        name: value
        for name, value in result.provenance.__dict__.items()
        if name != "protocol_fingerprint"
    }
    result_state = dict(result.__getstate__() if hasattr(result, "__getstate__") else vars(result))
    result_state["identified_effect"] = _LegacyPickle(type(result.identified_effect), effect_state)
    result_state["provenance"] = _LegacyPickle(type(result.provenance), provenance_state)
    joblib.dump(_LegacyPickle(type(result), result_state), path)


@pytest.mark.parametrize("fixture_name", ["protocol_point_result", "protocol_longitudinal_result"])
def test_public_load_backfills_pre_protocol_point_and_longitudinal_artifacts(
    fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    current = request.getfixturevalue(fixture_name)
    without_protocol = replace(
        current,
        identified_effect=replace(current.identified_effect, protocol=None),
        provenance=replace(current.provenance, protocol_fingerprint=None),
    )
    path = tmp_path / f"legacy-{fixture_name}.joblib"
    _write_pre_protocol_artifact(without_protocol, path)

    restored = load(path)

    assert restored.identified_effect.protocol is None
    assert restored.identified_effect._study.protocol is None
    assert restored.provenance.protocol_fingerprint is None
    assert "causal study protocol: absent" in restored.identified_effect.summary()
    assert "causal study protocol: absent" in restored.summary()


def test_provenance_from_dict_accepts_a_record_without_protocol_digest() -> None:
    provenance = Provenance(
        cleverly_version="0.1",
        python_version="3.13",
        platform="test",
        created_utc="2026-09-10T00:00:00+00:00",
        n=10,
        n_covariates=2,
        n_clusters=None,
        data_fingerprint="data",
        fold_fingerprint="folds",
    )
    payload = provenance.to_dict()
    payload.pop("protocol_fingerprint")

    assert Provenance.from_dict(payload).protocol_fingerprint is None


@pytest.mark.parametrize("fixture_name", ["protocol_point_result", "protocol_longitudinal_result"])
def test_result_without_identification_metadata_reports_protocol_absence(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    result = request.getfixturevalue(fixture_name)

    assert "causal study protocol: absent" in replace(result, identified_effect=None).summary()

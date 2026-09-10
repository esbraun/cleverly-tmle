"""The causal study protocol record and its result-persistence boundary."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unicodedata
from collections.abc import Callable
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
    DataError,
    ExplicitAdjustmentProvider,
    LongitudinalTreatment,
    PointTreatment,
    Provenance,
    RegimeMean,
    StudyProtocol,
    load,
)
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.protocol import _TEXT, _TEXT_SEQUENCE, protocol_lines
from tests.conftest import FAST_KWARGS
from tests.pickles import _LegacyPickle, legacy_state, legacy_without


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


def _provenance(**overrides: Any) -> Provenance:
    """Build one provenance record without running a fit."""
    values: dict[str, Any] = {
        "cleverly_version": "0.1",
        "python_version": "3.13",
        "platform": "test",
        "created_utc": "2026-09-10T00:00:00+00:00",
        "n": 10,
        "n_covariates": 2,
        "n_clusters": None,
        "data_fingerprint": "data",
        "fold_fingerprint": "folds",
    }
    values.update(overrides)
    return Provenance(**values)


_FIELD_NAMES: tuple[str, ...] = tuple(spec.name for spec in dataclasses.fields(StudyProtocol))


def _revised(protocol: StudyProtocol, name: str) -> StudyProtocol:
    """Return ``protocol`` with one field changed to a different, still valid, value.

    Parameters
    ----------
    protocol : StudyProtocol
        The record to change.
    name : str
        The one field to change.

    Returns
    -------
    StudyProtocol
        A record that differs from ``protocol`` in that field alone.

    Notes
    -----
    A sequence field is rewritten entry by entry rather than extended, because
    ``treatment_strategies`` and ``treatment_versions`` have to keep the same number of
    entries. So one change per field is valid for every field, which is what lets the
    witness below be driven by :func:`dataclasses.fields`.
    """
    value = getattr(protocol, name)
    changed: Any = (
        tuple(f"revised {entry}" for entry in value)
        if isinstance(value, tuple)
        else f"revised {value}"
    )
    return replace(protocol, **{name: changed})


# -------------------------------------------------------------- the canonical form, pinned

#: The canonical JSON text and digest of the one fixed record below, committed rather than
#: recomputed. Recomputing ``json.dumps`` and ``blake2b`` in the test restates the
#: implementation, so it reports agreement with whatever the implementation now does and
#: cannot notice the canonical form moving. If either value below has to change, the
#: canonical form changed and every digest this package has ever reported changed with it:
#: that is a deliberate schema revision, which raises ``_SCHEMA_VERSION`` and says so. It
#: is never the side effect of a refactor.
#:
#: The canonical form is the serialized text *and* the normalization that produces the
#: values in it. Stripping surrounding whitespace is therefore part of it, because it
#: decides which input reaches the digest. That half was settled in this change, before any
#: release and before any committed artifact stored a protocol digest, so schema 1 is the
#: first schema to ship either half and nothing needed a revision to reach it. A later
#: change to the strip is the change that raises ``_SCHEMA_VERSION``.
#:
#: The record carries "Niños" so that
#: ``ensure_ascii=False`` is pinned here too, and a 30-day horizon so that the JSON
#: carries a bare number inside text.
_PINNED_CANONICAL_JSON = (
    '{"assumption_rationale":["Adjustment covers measured common causes"],'
    '"eligibility":["Discharged alive"],'
    '"horizon":"30 days",'
    '"intercurrent_event_handling":["Score regardless of readmission"],'
    '"interference_unit":"Patient",'
    '"outcome":"Transition score",'
    '"schema_version":1,'
    '"target_population":"Niños discharged alive",'
    '"time_zero":"Hospital discharge",'
    '"treatment_strategies":["Navigation"],'
    '"treatment_versions":["Two calls"]}'
)
_PINNED_FINGERPRINT = "9d588871aa5fb415"


def _pinned_protocol() -> StudyProtocol:
    """Build the one record whose canonical form and digest are committed above."""
    return StudyProtocol(
        target_population="Niños discharged alive",
        eligibility=["Discharged alive"],
        time_zero="Hospital discharge",
        treatment_strategies=["Navigation"],
        treatment_versions=["Two calls"],
        outcome="Transition score",
        horizon="30 days",
        intercurrent_event_handling=["Score regardless of readmission"],
        interference_unit="Patient",
        assumption_rationale=["Adjustment covers measured common causes"],
    )


def test_one_fixed_record_keeps_its_committed_canonical_form_and_digest() -> None:
    """The committed form, which a recomputed expectation cannot check."""
    protocol = _pinned_protocol()

    assert protocol.canonical_json == _PINNED_CANONICAL_JSON
    assert protocol.fingerprint == _PINNED_FINGERPRINT
    assert len(protocol.fingerprint) == 16


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


# ----------------------------------------------- every field is serialized, digested, read


def test_every_protocol_field_is_serialized_labelled_and_digested() -> None:
    """A field added to the record forces a decision rather than shipping unreported.

    Two per-field lists can fall behind the fields themselves. ``_LABELS`` is read by
    ``summary_lines``, and a field with no entry there is left out of every summary in
    silence. The ``"kind"`` metadata is read by ``__post_init__``, and a field with no
    entry there is stored with no strip and no refusal, which is the failure the two
    hand-written name tuples before it had. ``to_dict`` walks
    :func:`dataclasses.fields` and so cannot fall behind, and it is checked here because
    the digest is what two fits compare.
    """
    declared = set(_FIELD_NAMES)
    serialized = set(_protocol().to_dict())
    labelled = set(StudyProtocol._LABELS)
    kinds = {spec.name: spec.metadata.get("kind") for spec in dataclasses.fields(StudyProtocol)}

    assert "schema_version" not in declared, (
        "schema_version is a property over a module constant rather than a field, so that "
        "a record cannot be constructed claiming a schema this package does not implement"
    )
    assert serialized == declared | {"schema_version"}, (
        f"to_dict() does not describe the record: unserialized {sorted(declared - serialized)}, "
        f"stale keys {sorted(serialized - declared - {'schema_version'})}. Every field enters "
        "the canonical JSON, because the digest is what two fits compare."
    )
    assert labelled == declared, (
        f"unlabelled fields {sorted(declared - labelled)}; stale labels "
        f"{sorted(labelled - declared)}. Add a StudyProtocol._LABELS entry for each new "
        "field, or summary_lines() omits it from every summary without saying so."
    )
    unvalidated = sorted(
        name for name, kind in kinds.items() if kind not in {_TEXT, _TEXT_SEQUENCE}
    )
    assert not unvalidated, (
        f"fields that name no validated kind: {unvalidated}. Declare each field with "
        "field(metadata={'kind': _TEXT}) or _TEXT_SEQUENCE, or __post_init__ reaches it "
        "through neither branch and stores the value with no strip and no refusal."
    )


def test_a_field_that_names_no_validated_kind_is_refused_at_construction() -> None:
    """The sweep above is what a developer sees; this is what protects the record.

    A record built from a class whose field names no kind has an unvalidated value in its
    canonical JSON and in its summary. So the walk refuses the field rather than skipping
    it, and the omission cannot reach a stored record by another route.
    """

    @dataclasses.dataclass(frozen=True)
    class _WithUnkindedField(StudyProtocol):
        sensitivity_plan: str = "  "

    with pytest.raises(RuntimeError, match="sensitivity_plan names no validated kind"):
        _WithUnkindedField(**dataclasses.asdict(_pinned_protocol()))


def test_summary_lines_report_every_protocol_field_exactly_once() -> None:
    """The docstring promises the complete record, one fact per line."""
    protocol = _protocol()

    lines = protocol.summary_lines()

    assert len(lines) == 1 + len(_FIELD_NAMES)
    for name in _FIELD_NAMES:
        value = getattr(protocol, name)
        rendered = ", ".join(f'"{entry}"' for entry in value) if isinstance(value, tuple) else value
        expected = f"{StudyProtocol._LABELS[name]}: {rendered}"
        assert lines.count(expected) == 1, f"{name} is not reported as {expected!r}: {lines}"
    # Each entry of a sequence field is quoted, so the line says where one entry ends.
    assert lines[1:] == (
        "target population: Adults eligible for transition navigation",
        'eligibility: "Discharged alive", "Lives in the service area"',
        "time zero: Hospital discharge",
        'treatment strategies: "Offer navigation", "Usual care"',
        'treatment versions: "Calls at discharge and day seven", "No navigation call"',
        "outcome: Top-box transition score",
        "horizon: 30 days after discharge",
        'intercurrent-event handling: "Use the score regardless of readmission"',
        "interference unit: Patient",
        'assumption rationale: "Recorded baseline variables cover the common causes used '
        'for adjustment"',
    )


def test_two_records_that_group_the_same_text_differently_render_different_lines() -> None:
    """A witness for the rendering, which a digest-only check cannot supply.

    Two entries and one entry carrying a comma are different records with different
    digests. A line that joined entries on ", " alone reported both as the same fact,
    which is the property an interior newline is refused for.
    """
    split = _protocol(eligibility=["Discharged alive", "Lives in the service area"])
    joined = _protocol(eligibility=["Discharged alive, Lives in the service area"])
    quoted = _protocol(eligibility=['Discharged alive", "Lives in the service area'])

    assert split.fingerprint != joined.fingerprint != quoted.fingerprint
    assert split.summary_lines() != joined.summary_lines()
    # The escape, so a quote inside an entry cannot forge an entry boundary either.
    assert split.summary_lines() != quoted.summary_lines()
    assert 'eligibility: "Discharged alive", "Lives in the service area"' in split.summary_lines()
    assert 'eligibility: "Discharged alive, Lives in the service area"' in joined.summary_lines()


@pytest.mark.parametrize("name", _FIELD_NAMES)
def test_changing_any_single_field_moves_the_protocol_digest(name: str) -> None:
    """One witness per field, so no field can drop out of the digest unnoticed.

    Only ``treatment_versions`` was witnessed before, by the fit-neutrality test. A
    ``to_dict`` that emitted a constant for any other field left the whole file green,
    which means the digest answered "was this the same study protocol?" for one field and
    reported agreement for the rest.
    """
    protocol = _protocol()

    changed = _revised(protocol, name)

    assert getattr(changed, name) != getattr(protocol, name)
    for other in _FIELD_NAMES:
        if other != name:
            assert getattr(changed, other) == getattr(protocol, other)
    assert changed.to_dict()[name] != protocol.to_dict()[name], (
        f"to_dict() reports the same value for {name} before and after it changed, so no "
        "digest can witness that field"
    )
    assert changed.canonical_json != protocol.canonical_json
    assert changed.fingerprint != protocol.fingerprint, (
        f"two records that differ in {name} share a digest, so a fit under one of them "
        "reports provenance agreement with a fit under the other"
    )
    assert changed != protocol


# ------------------------------------------------------------- the two pinned report lines


def test_the_protocol_record_reports_its_schema_and_never_its_own_digest() -> None:
    """The digest is rendered on the provenance line, and there only.

    Every other digest this package reports is on that line. Printing this one twice, in
    two formats, was the redundancy the first summary line dropped, and only a pinned
    string keeps it dropped.
    """
    protocol = _protocol()

    lines = protocol.summary_lines()

    assert lines[0] == "causal study protocol: schema 1"
    assert protocol.fingerprint not in "\n".join(lines)
    assert "digest" not in "\n".join(lines)


def test_the_provenance_line_names_the_protocol_digest() -> None:
    """The one place the digest is rendered."""
    protocol = _protocol()

    described = _provenance(protocol_fingerprint=protocol.fingerprint).describe()

    assert f"protocol digest {protocol.fingerprint}" in described
    assert not any("protocol" in line for line in _provenance().describe())


def test_protocol_lines_distinguish_absence_from_a_record_that_was_not_retained() -> None:
    """Three states, because a stored digest with no record is not absence."""
    protocol = _protocol()

    assert protocol_lines(protocol) == protocol.summary_lines()
    assert protocol_lines(None) == ("causal study protocol: absent",)
    assert protocol_lines(None, None) == ("causal study protocol: absent",)
    assert protocol_lines(None, "0123456789abcdef") == (
        "causal study protocol: record not retained; the provenance line names its digest",
    )
    # The digest is read for the distinction between the second state and the third, and
    # rendered on neither line. The provenance line is where this package reports a digest.
    assert "0123456789abcdef" not in "\n".join(protocol_lines(None, "0123456789abcdef"))


# --------------------------------------------------------------- normalization and refusal


def test_surrounding_whitespace_is_normalized_away_before_the_digest() -> None:
    """A deliberate witness: the strip is digest-affecting, so it needs a nonzero case.

    Two records whose text differs only in surrounding whitespace are the same record and
    carry the same digest. Dropping the strip splits them, and every fit run from a form
    that pads its fields reports disagreement with the same protocol typed without padding.
    """
    plain = _protocol()

    padded = _protocol(
        target_population="  Adults eligible for transition navigation\t",
        eligibility=[" Discharged alive", "Lives in the service area\n"],
    )

    assert padded.target_population == "Adults eligible for transition navigation"
    assert padded.eligibility == ("Discharged alive", "Lives in the service area")
    assert padded.canonical_json == plain.canonical_json
    assert padded.fingerprint == plain.fingerprint
    assert padded == plain
    # Interior whitespace is text rather than padding, so it stays and it still counts.
    inner = _protocol(target_population="Adults  eligible for transition navigation")
    assert inner.fingerprint != plain.fingerprint


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
    with pytest.raises(DataError, match=field):
        _protocol(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("target_population", "Adults \ud800 discharged", "unpaired"),
        ("eligibility", ["Discharged alive", "Lives in the \ud800 area"], "unpaired"),
        ("outcome", "Top-box\x00score", "control or line-separator"),
        ("horizon", "30 days\x1b", "control or line-separator"),
        ("eligibility", ["Discharged\nalive"], "control or line-separator"),
        ("outcome", "Top-box\u2028score", "control or line-separator"),
        ("eligibility", ["Discharged\u2029alive"], "control or line-separator"),
    ],
    ids=[
        "surrogate-text",
        "surrogate-entry",
        "nul",
        "escape",
        "interior-newline",
        "line-separator",
        "paragraph-separator",
    ],
)
def test_protocol_refuses_text_that_has_no_canonical_form(
    field: str, value: Any, expected: str
) -> None:
    """Both refusals run at construction, not when somebody asks for the digest.

    A record that raises only on ``fingerprint`` is a record whose digest is not a
    property of the record, and by then the study that stored it has finished.
    """
    with pytest.raises(DataError, match=expected):
        _protocol(**{field: value})


def test_the_refused_separators_are_the_ones_that_split_a_rendered_line() -> None:
    """Why U+2028 and U+2029 are refused, though neither is a control character.

    A guard over the Unicode category "Cc" alone accepted both. A summary line carrying
    either still splits under :meth:`str.splitlines`, so the refusal said it prevented a
    split that it let through. The split is checked here rather than asserted.
    """
    for separator in ("\u2028", "\u2029"):
        assert unicodedata.category(separator) != "Cc"
        assert len(f"horizon: 30 days{separator}after discharge".splitlines()) == 2
        with pytest.raises(DataError, match="control or line-separator"):
            _protocol(horizon=f"30 days{separator}after discharge")


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


def test_protocol_names_the_array_types_it_refuses() -> None:
    """The refusal is deliberate, so the message has to say what to pass instead."""
    with pytest.raises(TypeError, match="numpy array") as refusal:
        _protocol(eligibility=np.array(["Discharged alive"]))

    assert "pandas" in str(refusal.value)
    assert "list or a tuple" in str(refusal.value)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("time_zero", 7, "time_zero must be text, not int"),
        ("interference_unit", None, "interference_unit must be text, not NoneType"),
        ("eligibility", [7], r"eligibility\[0\] must be text, not int"),
        ("eligibility", ["Discharged alive", None], r"eligibility\[1\] must be text, not NoneType"),
        ("assumption_rationale", b"bytes", "not one text value"),
    ],
    ids=["scalar-int", "scalar-none", "entry-int", "entry-none", "bytes"],
)
def test_protocol_refuses_a_wrong_type_with_a_type_error(
    field: str, value: Any, expected: str
) -> None:
    with pytest.raises(TypeError, match=expected):
        _protocol(**{field: value})


def test_the_two_refusal_classes_stay_distinguishable() -> None:
    """A caller who passed the wrong kind of object is not a caller with unstatable text.

    ``DataError`` subclasses ``ValueError``, so a content refusal is catchable as either.
    A wrong type is neither, so a ``except ValueError`` around a construction cannot
    swallow it and report a data problem.
    """
    assert issubclass(DataError, ValueError)
    assert not issubclass(TypeError, ValueError)
    with pytest.raises(DataError):
        _protocol(target_population=" ")
    with pytest.raises(TypeError):
        _protocol(target_population=7)


def test_protocol_rejects_unmatched_strategy_and_version_counts() -> None:
    # A count mismatch is content the schema cannot state rather than a wrong type, so it
    # is a DataError, and DataError subclasses ValueError so the older catch still holds.
    with pytest.raises(DataError, match="same number"):
        _protocol(treatment_versions=["One version"])
    with pytest.raises(ValueError, match="same number"):
        _protocol(treatment_versions=["One version"])


# --------------------------------------------------------------- the study that stamps one


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


_POINT_PROTOCOL = _protocol()
_LONGITUDINAL_PROTOCOL = _protocol(
    treatment_strategies=["Always treat", "Never treat"],
    treatment_versions=["Treatment at both nodes", "No treatment at either node"],
)


def _point_fit(protocol: StudyProtocol) -> Any:
    """Fit the point-treatment path under one study protocol."""
    frame, _ = make_linear_ate(n=140, seed=13)
    return CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4")),
        protocol=protocol,
    ).estimate(ATE(), **FAST_KWARGS)


def _longitudinal_fit(protocol: StudyProtocol) -> Any:
    """Fit the longitudinal path under one study protocol."""
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
        protocol=protocol,
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


#: One row per estimate path, because each has its own stamping site and its own
#: provenance builder. Before this matrix only the point path was witnessed, so deleting
#: the longitudinal stamp left the whole file green. The third entry names the settings
#: records a protocol must not reach. ``config``, ``method`` and ``simultaneous`` are on
#: both result families, and ``repeats`` is on the point result alone.
_FITS: dict[str, tuple[Callable[[StudyProtocol], Any], StudyProtocol, tuple[str, ...]]] = {
    "point": (_point_fit, _POINT_PROTOCOL, ("config", "method", "repeats")),
    "longitudinal": (
        _longitudinal_fit,
        _LONGITUDINAL_PROTOCOL,
        ("config", "method", "simultaneous"),
    ),
}


@pytest.fixture(scope="module")
def protocol_point_result() -> Any:
    return _point_fit(_POINT_PROTOCOL)


@pytest.fixture(scope="module")
def protocol_longitudinal_result() -> Any:
    return _longitudinal_fit(_LONGITUDINAL_PROTOCOL)


@pytest.mark.parametrize("path", list(_FITS))
def test_a_treatment_version_change_moves_only_protocol_metadata(path: str) -> None:
    """The record is descriptive, so it reaches the digest and nothing else.

    Both estimate paths run this, because each stamps the digest itself.
    """
    fit, base, settings = _FITS[path]
    revised = _revised(base, "treatment_versions")

    first = fit(base)
    second = fit(revised)

    assert first.identified_effect.functional == second.identified_effect.functional
    for name in settings:
        assert joblib_hash(getattr(first, name)) == joblib_hash(getattr(second, name)), (
            f"the study protocol reached {name}, which is a setting of the fit"
        )
    assert first.provenance.data_fingerprint == second.provenance.data_fingerprint
    assert first.provenance.fold_fingerprint == second.provenance.fold_fingerprint
    assert first.provenance.protocol_fingerprint == base.fingerprint
    assert second.provenance.protocol_fingerprint == revised.fingerprint
    assert first.provenance.protocol_fingerprint != second.provenance.protocol_fingerprint
    np.testing.assert_array_equal(first.covariance(), second.covariance())
    assert set(first.estimates) == set(second.estimates)
    for name in first.estimates:
        assert first[name].psi == second[name].psi
        assert first[name].ci == second[name].ci
        np.testing.assert_array_equal(first[name].influence_curve, second[name].influence_curve)


@pytest.mark.parametrize("fixture_name", ["protocol_point_result", "protocol_longitudinal_result"])
def test_a_result_summary_renders_the_protocol_digest_once(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """The complete record and the digest, each reported in one place.

    The record's own first line carries no digest, so the only digest in the report is the
    one on the provenance line. A second rendering would return the redundancy that line
    was written to remove, and nothing else in the suite would notice.
    """
    result = request.getfixturevalue(fixture_name)
    digest = result.provenance.protocol_fingerprint

    summary = result.summary()

    assert digest is not None
    assert summary.count(digest) == 1, f"the protocol digest is rendered {summary.count(digest)}x"
    assert f"protocol digest {digest}" in summary
    assert "causal study protocol: schema 1" in summary
    assert digest not in result.identified_effect.summary()

    # The same count on the degraded path, where no field is reported and the line that
    # replaces them used to name the digest a second time.
    degraded = replace(result, identified_effect=None).summary()

    assert "causal study protocol: record not retained" in degraded
    assert degraded.count(digest) == 1, (
        f"the protocol digest is rendered {degraded.count(digest)}x on the path that "
        "retained no record"
    )
    assert f"protocol digest {digest}" in degraded


# --------------------------------------------------------------------------- persistence


@pytest.mark.parametrize("fixture_name", ["protocol_point_result", "protocol_longitudinal_result"])
def test_complete_protocol_and_digest_survive_result_round_trip(
    fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    result = request.getfixturevalue(fixture_name)

    restored = load(result.save(tmp_path / f"{fixture_name}.joblib"))

    assert not hasattr(restored, "protocol")
    assert restored.identified_effect.protocol == result.identified_effect.protocol
    # `is not None` is the load-bearing half. Comparing the two records' digests alone
    # compares `None` with `None` on a path that never stamped one, so deleting a stamping
    # site left this assertion passing.
    stamped = restored.provenance.protocol_fingerprint
    assert stamped is not None, "the fit stamped no protocol digest onto its provenance"
    assert stamped == restored.identified_effect.protocol.fingerprint
    assert stamped == result.provenance.protocol_fingerprint
    assert restored.identified_effect.protocol is restored.identified_effect._study.protocol
    for line in restored.identified_effect.protocol.summary_lines():
        assert line in restored.identified_effect.summary()
        assert line in restored.summary()


def _write_pre_protocol_artifact(result: Any, path: Path) -> None:
    """Write ``result`` as a joblib artifact from before the protocol fields existed.

    Every level goes through :func:`tests.pickles.legacy_state`, which refuses a name the
    live record does not carry. Filtering ``__dict__`` here instead dropped nothing once a
    field was renamed, and the one test that reads this artifact then replayed the current
    shape as if it were the old one.
    """
    study = result.identified_effect._study
    effect_state = legacy_state(result.identified_effect, "protocol")
    effect_state["_study"] = _LegacyPickle(type(study), legacy_state(study, "_protocol"))
    result_state = dict(result.__getstate__() if hasattr(result, "__getstate__") else vars(result))
    result_state["identified_effect"] = _LegacyPickle(type(result.identified_effect), effect_state)
    result_state["provenance"] = _LegacyPickle(
        type(result.provenance), legacy_state(result.provenance, "protocol_fingerprint")
    )
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
    payload = _provenance().to_dict()
    payload.pop("protocol_fingerprint")

    assert Provenance.from_dict(payload).protocol_fingerprint is None


def test_a_provenance_pickled_before_package_versions_restores_the_empty_default() -> None:
    """The one field for which ``_DefaultingUnpickle`` is load-bearing on this record.

    ``@dataclass`` keeps the class attribute behind a plain ``default=``, so a pickle that
    predates ``protocol_fingerprint`` reads ``None`` off the class with no mixin in sight.
    It *deletes* the class attribute for a ``default_factory`` field, and
    ``package_versions`` is the only such field here. So this is the only case that fails
    when the mixin goes, and without it nothing witnesses the mixin on either record class.
    """
    # The asymmetry the paragraph above rests on, checked rather than asserted.
    assert "package_versions" not in vars(Provenance)
    assert Provenance.protocol_fingerprint is None
    provenance = _provenance(
        package_versions={"numpy": "2.0.0"}, protocol_fingerprint=_protocol().fingerprint
    )

    restored = legacy_without(provenance, "package_versions")
    without_digest = legacy_without(provenance, "protocol_fingerprint")

    assert hasattr(restored, "package_versions"), (
        "Provenance no longer restores a pickle written before package_versions existed; "
        "every reader of that attribute raises AttributeError instead"
    )
    assert restored.package_versions == {}
    assert restored == replace(provenance, package_versions={})
    assert without_digest.protocol_fingerprint is None
    assert without_digest.describe() == _provenance().describe()


def test_legacy_state_refuses_a_name_the_record_does_not_carry() -> None:
    """A renamed field has to fail the helper rather than drop nothing."""
    with pytest.raises(KeyError, match="carries no such attribute: renamed_away"):
        legacy_state(_provenance(), "renamed_away")
    with pytest.raises(KeyError, match="package_versions_renamed"):
        legacy_without(_provenance(), "package_versions", "package_versions_renamed")


@pytest.mark.parametrize("fixture_name", ["protocol_point_result", "protocol_longitudinal_result"])
def test_result_without_identification_metadata_reports_an_unretained_record(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """A stored digest and no record is not absence, and the summary must not say it is."""
    result = request.getfixturevalue(fixture_name)

    summary = replace(result, identified_effect=None).summary()

    assert (
        "causal study protocol: record not retained; the provenance line names its digest"
    ) in summary
    assert "causal study protocol: absent" not in summary

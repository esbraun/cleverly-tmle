"""Structured records for the scientific protocol behind a causal study."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from .exceptions import DataError
from .provenance import fingerprint_payload
from .utils.records import _DefaultingUnpickle

__all__ = ["StudyProtocol", "protocol_lines"]

#: Version of the normalized record schema.  A module constant rather than a field, so a
#: record cannot be constructed claiming a schema this package does not implement.
_SCHEMA_VERSION = 1

#: Field-metadata values naming the two validated kinds.  ``__post_init__``,
#: :meth:`StudyProtocol.to_dict` and :meth:`StudyProtocol.summary_lines` all walk
#: :func:`dataclasses.fields`, so declaring a field is what validates, serializes and
#: reports it.  The two hand-written name tuples this replaces went stale silently: a
#: field left out of them was stored unvalidated.  The metadata is a third per-field list
#: and could go stale the same way, so ``__post_init__`` refuses a field that names no
#: kind here rather than skipping it.
_TEXT = "text"
_TEXT_SEQUENCE = "text_sequence"

#: Unicode general categories of every character that ends a line under
#: :meth:`str.splitlines`, which is what a refusal over control characters alone missed.
_LINE_ENDING_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})

#: The one line a summary prints when no study protocol was recorded.
_ABSENT = "causal study protocol: absent"


def _text(name: str, value: object) -> str:
    """Return one text field stripped of surrounding whitespace, or refuse it."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text, not {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise DataError(f"{name} must be non-blank text")
    # Both refusals below run at construction rather than at fingerprint time.  A record
    # that raises only when somebody asks for its digest is a record whose digest is not a
    # property of the record, and the study that stored it has already finished.
    try:
        stripped.encode("utf-8")
    except UnicodeEncodeError as error:
        raise DataError(
            f"{name} must be text that encodes as UTF-8; the value carries an unpaired "
            "surrogate, which has no canonical JSON form and so no digest"
        ) from error
    # Every category that ends a line, rather than "Cc" alone: `str.splitlines` also
    # splits on U+2028 LINE SEPARATOR (Zl) and U+2029 PARAGRAPH SEPARATOR (Zp), so a guard
    # over control characters alone accepted the two characters whose whole purpose is the
    # split this refusal exists to prevent.
    splitter = next(
        (item for item in stripped if unicodedata.category(item) in _LINE_ENDING_CATEGORIES),
        None,
    )
    if splitter is not None:
        raise DataError(
            f"{name} must not carry {splitter!r}; that is a control or line-separator "
            "character, and one summary line has to render as one fact"
        )
    return stripped


def _entries(values: tuple[str, ...]) -> str:
    """Return one sequence field as a line only that one tuple can produce."""
    # Each entry quoted, rather than joined bare: a comma is both the separator and a
    # character an entry may carry, so ["a", "b"] and ["a, b"] rendered the same line while
    # their digests differ.  A summary line that two different records produce is the same
    # failure an interior newline is refused for.  `json.dumps` ends a literal at its first
    # unescaped quote and escapes any quote inside one, so the entries stay recoverable, and
    # `ensure_ascii=False` keeps the text readable.
    return ", ".join(json.dumps(entry, ensure_ascii=False) for entry in values)


def _text_tuple(name: str, values: object) -> tuple[str, ...]:
    """Return one sequence field as a tuple of normalized entries, or refuse it."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of text entries, not one text value")
    if not isinstance(values, Sequence):
        # A numpy array and a pandas Index are both refused here, and deliberately: the
        # record states a declared order of text, and an array also carries a dtype that
        # can turn an entry into a truncated or numeric value on the way in.
        raise TypeError(
            f"{name} must be an ordered sequence of text entries, such as a list or a "
            "tuple; a set, a mapping, a one-shot iterator, a numpy array and a pandas "
            f"Index are all refused; got {type(values).__name__}"
        )
    normalized = tuple(_text(f"{name}[{index}]", value) for index, value in enumerate(values))
    if not normalized:
        raise DataError(f"{name} must contain at least one entry")
    return normalized


@dataclass(frozen=True)
class StudyProtocol(_DefaultingUnpickle):
    """Record the scientific protocol that gives a causal question its context.

    Parameters
    ----------
    target_population : str
        Population to which the causal question applies.
    eligibility : sequence of str
        Inclusion and exclusion criteria.
    time_zero : str
        Event that aligns eligibility, treatment assignment, and follow-up.
    treatment_strategies : sequence of str
        Treatment strategies under comparison.
    treatment_versions : sequence of str
        Version attached to each treatment strategy, in the same order.
    outcome : str
        Outcome definition and measurement.
    horizon : str
        Follow-up horizon for the outcome.
    intercurrent_event_handling : sequence of str
        Rules for events after time zero that affect outcome interpretation.
    interference_unit : str
        Unit within which interference may occur.
    assumption_rationale : sequence of str
        Study-specific rationale for the identification assumptions.

    Attributes
    ----------
    schema_version : int

    See Also
    --------
    cleverly.CausalStudy : Attach a protocol to observed study data.
    cleverly.IdentifiedEffect : Persist the record with an identified question.
    cleverly.Provenance : Store the protocol digest on a fitted result.

    Notes
    -----
    This hybrid record uses target-trial and ICH estimand vocabulary. It is not a
    complete target-trial protocol or a complete ICH estimand. The typed estimand stores
    the contrast, and the method stores the analysis configuration.

    Every text field is stripped of surrounding whitespace and then validated. The class
    refuses a wrong type with :class:`TypeError`, and refused content with
    :class:`~cleverly.DataError`. The split is the difference between a caller who passed
    the wrong kind of object and a caller who passed a record the schema cannot state.

    Examples
    --------
    >>> from cleverly import StudyProtocol
    >>> protocol = StudyProtocol(
    ...     target_population="Adults eligible for navigation",
    ...     eligibility=["Discharged alive"],
    ...     time_zero="Hospital discharge",
    ...     treatment_strategies=["Navigation", "Usual care"],
    ...     treatment_versions=["Two calls", "No call"],
    ...     outcome="Thirty-day transition score",
    ...     horizon="30 days",
    ...     intercurrent_event_handling=["Use outcome regardless of readmission"],
    ...     interference_unit="Patient",
    ...     assumption_rationale=["Adjustment covers measured common causes"],
    ... )
    >>> len(protocol.fingerprint)
    16
    """

    target_population: str = field(metadata={"kind": _TEXT})
    eligibility: Sequence[str] = field(metadata={"kind": _TEXT_SEQUENCE})
    time_zero: str = field(metadata={"kind": _TEXT})
    treatment_strategies: Sequence[str] = field(metadata={"kind": _TEXT_SEQUENCE})
    treatment_versions: Sequence[str] = field(metadata={"kind": _TEXT_SEQUENCE})
    outcome: str = field(metadata={"kind": _TEXT})
    horizon: str = field(metadata={"kind": _TEXT})
    intercurrent_event_handling: Sequence[str] = field(metadata={"kind": _TEXT_SEQUENCE})
    interference_unit: str = field(metadata={"kind": _TEXT})
    assumption_rationale: Sequence[str] = field(metadata={"kind": _TEXT_SEQUENCE})

    #: Summary label for each reported field, in report order.  Written out rather than
    #: derived from the field name, because ``name.replace("_", " ")`` cannot produce
    #: "intercurrent-event handling" and these lines are read by a person.
    _LABELS: ClassVar[dict[str, str]] = {
        "target_population": "target population",
        "eligibility": "eligibility",
        "time_zero": "time zero",
        "treatment_strategies": "treatment strategies",
        "treatment_versions": "treatment versions",
        "outcome": "outcome",
        "horizon": "horizon",
        "intercurrent_event_handling": "intercurrent-event handling",
        "interference_unit": "interference unit",
        "assumption_rationale": "assumption rationale",
    }

    def __post_init__(self) -> None:
        for spec in fields(self):
            kind = spec.metadata.get("kind")
            value = getattr(self, spec.name)
            if kind == _TEXT:
                object.__setattr__(self, spec.name, _text(spec.name, value))
            elif kind == _TEXT_SEQUENCE:
                object.__setattr__(self, spec.name, _text_tuple(spec.name, value))
            else:
                # The metadata is a third per-field list, and a list nothing checks goes
                # stale the way the two name tuples it replaced did: a field declared with
                # no kind is stored with no strip and no refusal, and it still reaches the
                # digest and the summary.  Refused here rather than skipped, so the
                # omission cannot ship.  Not a caller's error, so neither TypeError nor
                # DataError: the record is declared wrong, not built wrong.
                raise RuntimeError(
                    f"StudyProtocol.{spec.name} names no validated kind, so no code would "
                    f"strip or refuse its value; declare the field with "
                    f"metadata={{'kind': ...}} as {_TEXT!r} or {_TEXT_SEQUENCE!r}"
                )
        if len(self.treatment_strategies) != len(self.treatment_versions):
            raise DataError(
                "treatment_strategies and treatment_versions must contain the same number "
                "of entries"
            )

    @property
    def schema_version(self) -> int:
        """Return the version of this normalized record schema."""
        return _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible normalized form.

        Returns
        -------
        dict
            A mapping with sequences represented as JSON arrays.
        """
        payload: dict[str, Any] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            payload[spec.name] = list(value) if isinstance(value, tuple) else value
        # The schema version is a property rather than a field, so the walk cannot reach
        # it, and the serialized form has to carry it for `from_dict` to check.
        payload["schema_version"] = self.schema_version
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StudyProtocol:
        """Construct a protocol from its normalized form.

        Parameters
        ----------
        payload : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        StudyProtocol
            Validated immutable protocol record.
        """
        values = dict(payload)
        schema_version = values.pop("schema_version", _SCHEMA_VERSION)
        if schema_version != _SCHEMA_VERSION:
            raise DataError(f"schema_version must be {_SCHEMA_VERSION}; got {schema_version}")
        return cls(**values)

    @property
    def canonical_json(self) -> str:
        """Return canonical UTF-8 JSON text used by :attr:`fingerprint`."""
        return fingerprint_payload(self.to_dict())[0]

    @property
    def fingerprint(self) -> str:
        """Return the BLAKE2b digest of the canonical UTF-8 JSON record."""
        return fingerprint_payload(self.to_dict())[1]

    def summary_lines(self) -> tuple[str, ...]:
        """Return every protocol field as readable summary lines.

        Returns
        -------
        tuple of str
            The complete protocol record, one fact per line.

        Notes
        -----
        The digest is not repeated here. Every digest a provenance record reports is
        rendered on the provenance line, and :meth:`cleverly.Provenance.describe` reports
        this one there.
        """
        lines = [f"causal study protocol: schema {self.schema_version}"]
        for name, label in self._LABELS.items():
            value = getattr(self, name)
            lines.append(f"{label}: {_entries(value) if isinstance(value, tuple) else value}")
        return tuple(lines)


def protocol_lines(protocol: StudyProtocol | None, digest: str | None = None) -> tuple[str, ...]:
    """Return the study-protocol lines of one summary.

    Parameters
    ----------
    protocol : StudyProtocol or None
        Record to report. ``None`` means the caller holds no record.
    digest : str or None
        Digest the fit recorded. Read only when ``protocol`` is ``None``, and read only
        for its presence, because the provenance line renders the digest itself.

    Returns
    -------
    tuple of str
        Every protocol field as one line, or the single line that says why no field is
        reported.

    Notes
    -----
    A digest with no record is a third state rather than absence. It says the fit ran
    under a protocol and that the descriptive record did not survive to this summary, so
    reporting absence there would contradict the digest on the provenance line.

    The not-retained line does not carry the digest. Printing it there and on the
    provenance line is the same redundancy :meth:`StudyProtocol.summary_lines` dropped,
    on the one path where no field is reported.
    """
    if protocol is not None:
        return protocol.summary_lines()
    if digest is None:
        return (_ABSENT,)
    return ("causal study protocol: record not retained; the provenance line names its digest",)

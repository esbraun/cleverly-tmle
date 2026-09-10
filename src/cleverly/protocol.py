"""Structured records for the scientific protocol behind a causal study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

__all__ = ["StudyProtocol"]

_DIGEST_BYTES = 8


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-blank text")


def _text_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of text entries, not one text value")
    if not isinstance(values, Sequence):
        raise TypeError(f"{name} must be an ordered sequence of text entries")
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{name} must contain at least one entry")
    for index, value in enumerate(normalized):
        _text(f"{name}[{index}]", value)
    return normalized


@dataclass(frozen=True)
class StudyProtocol:  # numpydoc ignore=PR02
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
    schema_version : int
        Version of this normalized record schema.

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

    target_population: str
    eligibility: Sequence[str]
    time_zero: str
    treatment_strategies: Sequence[str]
    treatment_versions: Sequence[str]
    outcome: str
    horizon: str
    intercurrent_event_handling: Sequence[str]
    interference_unit: str
    assumption_rationale: Sequence[str]
    schema_version: int = field(default=1, init=False)
    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        for name in ("target_population", "time_zero", "outcome", "horizon", "interference_unit"):
            _text(name, getattr(self, name))
        for name in (
            "eligibility",
            "treatment_strategies",
            "treatment_versions",
            "intercurrent_event_handling",
            "assumption_rationale",
        ):
            object.__setattr__(self, name, _text_tuple(name, getattr(self, name)))
        if len(self.treatment_strategies) != len(self.treatment_versions):
            raise ValueError(
                "treatment_strategies and treatment_versions must contain the same number "
                "of entries"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible normalized form.

        Returns
        -------
        dict
            A mapping with sequences represented as JSON arrays.
        """
        return {
            "target_population": self.target_population,
            "eligibility": list(self.eligibility),
            "time_zero": self.time_zero,
            "treatment_strategies": list(self.treatment_strategies),
            "treatment_versions": list(self.treatment_versions),
            "outcome": self.outcome,
            "horizon": self.horizon,
            "intercurrent_event_handling": list(self.intercurrent_event_handling),
            "interference_unit": self.interference_unit,
            "assumption_rationale": list(self.assumption_rationale),
            "schema_version": self.schema_version,
        }

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
        schema_version = values.pop("schema_version", cls.CURRENT_SCHEMA_VERSION)
        if schema_version != cls.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {cls.CURRENT_SCHEMA_VERSION}; got {schema_version}"
            )
        return cls(**values)

    @property
    def canonical_json(self) -> str:
        """Return canonical UTF-8 JSON text used by :attr:`fingerprint`."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        """Return the BLAKE2b digest of the canonical UTF-8 JSON record."""
        return hashlib.blake2b(
            self.canonical_json.encode("utf-8"), digest_size=_DIGEST_BYTES
        ).hexdigest()

    def summary_lines(self) -> tuple[str, ...]:
        """Return every protocol field as readable summary lines.

        Returns
        -------
        tuple of str
            Full protocol record and fingerprint, one fact per line.
        """
        return (
            f"causal study protocol: schema {self.schema_version}; {self.fingerprint}",
            f"target population: {self.target_population}",
            f"eligibility: {list(self.eligibility)}",
            f"time zero: {self.time_zero}",
            f"treatment strategies: {list(self.treatment_strategies)}",
            f"treatment versions: {list(self.treatment_versions)}",
            f"outcome: {self.outcome}",
            f"horizon: {self.horizon}",
            f"intercurrent-event handling: {list(self.intercurrent_event_handling)}",
            f"interference unit: {self.interference_unit}",
            f"assumption rationale: {list(self.assumption_rationale)}",
        )

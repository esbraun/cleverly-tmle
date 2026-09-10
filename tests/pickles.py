"""Small pickle reducers shared by backward-persistence tests."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from typing import Any

from cleverly.study import BackdoorMeanContrast


def _blank(record_class: type) -> Any:
    """Build an uninitialised record for the unpickler to fill from a state."""
    return object.__new__(record_class)


class _LegacyPickle:
    """Pickles as ``record_class`` carrying ``state`` and nothing else.

    The point is to reach ``__setstate__`` the way an old pickle reaches it, rather than by
    calling it. ``__reduce__`` returns a builder, its arguments, and a state, and it is the
    *unpickler* that applies that state -- through ``__setstate__`` when the class defines
    one. A test that calls ``__setstate__`` by hand proves the method works and proves
    nothing about whether unpickling routes through it, so deleting the method outright
    still passed.
    """

    def __init__(self, record_class: type, state: dict[str, Any]) -> None:
        self._record_class = record_class
        self._state = state

    def __reduce__(self) -> tuple[Any, ...]:
        return (_blank, (self._record_class,), self._state)


def legacy_state(record: Any, *dropped: str) -> dict[str, Any]:
    """Return ``record``'s pickle state as if ``dropped`` did not exist when it was written.

    Parameters
    ----------
    record : Any
        A live record whose instance dictionary describes the current shape.
    *dropped : str
        Names the older pickle did not carry. Each one has to be an attribute the live
        record does carry.

    Returns
    -------
    dict of str to Any
        The instance dictionary with those names removed.

    Raises
    ------
    KeyError
        When a name in ``dropped`` is not an attribute of ``record``. The filter below
        would drop nothing for such a name, so the helper would hand back a state of the
        *current* shape and the caller's back-compatibility assertion would keep passing
        against it. ``_DefaultingUnpickle.check_pickle_backfill`` refuses a stale
        ``_PICKLE_BACKFILL`` key for that reason, and a rename reaches this helper the
        same way: ``legacy_state(record, "treatment_value")`` still builds a valid state
        after ``treatment_value`` is renamed, and the test that pins the legacy branch
        then replays the current shape as the old one.

    Notes
    -----
    A nested artifact needs the state rather than a restored record, because the old
    pickle it stands in for carried one legacy state per level. So this is the guard and
    :func:`legacy_without` is the one-level convenience over it. Filtering
    ``vars(record)`` at the call site instead is what this function exists to stop: that
    form carries no guard, and three call sites wrote it out before.
    """
    absent = sorted(set(dropped) - set(vars(record)))
    if absent:
        raise KeyError(
            f"{type(record).__name__} carries no such attribute: {', '.join(absent)}; "
            "a renamed field needs its caller renamed with it, or this helper drops "
            "nothing and the test replays the current shape as if it were the old one"
        )
    return {name: value for name, value in vars(record).items() if name not in dropped}


def legacy_without(record: Any, *dropped: str) -> Any:
    """Unpickle ``record`` as if ``dropped`` did not exist when it was written.

    Parameters
    ----------
    record : Any
        A live record to rewrite as an older pickle of the same class.
    *dropped : str
        Names the older pickle did not carry. Each one has to be an attribute the live
        record does carry.

    Returns
    -------
    Any
        The record restored from a state with those names removed.
    """
    return pickle.loads(pickle.dumps(_LegacyPickle(type(record), legacy_state(record, *dropped))))


#: The fields :class:`~cleverly.study.BackdoorMeanContrast` carried before the
#: design-bound revision. Every record ever written carries these, so none of them
#: distinguishes a schema-0 record from a current one and none belongs in
#: ``_SCHEMA_1_FIELDS``. Named here so that the two tuples together account for the whole
#: dataclass: ``test_every_field_of_the_functional_is_classified_for_provenance`` compares
#: their union with ``dataclasses.fields``, and a new field that is in neither fails it.
#: ``_SCHEMA_1_FIELDS`` alone could not, because omitting a field from a hand-written
#: tuple removes a tampering row in silence.
PRE_SCHEMA_1_FIELDS: tuple[str, ...] = (
    "outcome",
    "treatment",
    "adjustment",
    "target",
    "axis",
    "reference",
    "interventions",
    "horizons",
    "msm",
    "intermediate",
    "longitudinal",
)


#: One forged value per field the design-bound revision added to
#: :class:`~cleverly.study.BackdoorMeanContrast`. Each value has to differ from what a
#: reconstruction of the record produces, so that pairing it with an otherwise valid fit
#: is a record no design could have written.
_FORGED_FUNCTIONAL_VALUES: Mapping[str, Any] = {
    "missingness": "forged_delta",
    "intermediate_name": "forged_z",
    "treatment_levels": ("forged",),
    "treatment_value": 1,
    "schema_version": 2,
}


def _tamperings(
    schema_fields: Sequence[str], forged: Mapping[str, Any]
) -> tuple[tuple[str, Any], ...]:
    """Pair every field in ``schema_fields`` with the value that must break provenance.

    Parameters
    ----------
    schema_fields : sequence of str
        The fields the provenance matcher reconstructs and compares.
    forged : mapping of str to Any
        One tampered value per field.

    Returns
    -------
    tuple of tuple of (str, Any)
        The parametrisation matrix, in the field order of the class.

    Raises
    ------
    ValueError
        When the two do not describe the same set of fields. A matrix that is missing a
        row is the bug this guard exists for: it under-covers in silence, so a field
        added to the record ships with no tampering case and every surface that reads it
        still reports a green refusal suite.
    """
    missing = [name for name in schema_fields if name not in forged]
    extra = [name for name in forged if name not in schema_fields]
    if missing or extra:
        raise ValueError(
            "the functional tampering matrix does not cover the design-bound fields: "
            f"missing {sorted(missing)}, unknown {sorted(extra)}; every field the "
            "provenance matcher compares needs a row, or the matrix passes while a "
            "forged value for that field goes unchecked"
        )
    return tuple((name, forged[name]) for name in schema_fields)


#: The one tampering matrix for a :class:`~cleverly.study.BackdoorMeanContrast` record.
#: Every surface that refuses a forged functional parametrises over it, so a new field on
#: the record reaches each of those surfaces at once. It is derived from the class rather
#: than written out, and :func:`_tamperings` refuses a derivation that does not cover the
#: class, so the matrix cannot fall behind the record it tampers with.
FUNCTIONAL_TAMPERINGS: tuple[tuple[str, Any], ...] = _tamperings(
    BackdoorMeanContrast._SCHEMA_1_FIELDS, _FORGED_FUNCTIONAL_VALUES
)

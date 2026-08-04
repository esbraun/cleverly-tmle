"""Equality for the report rows whose ``nan`` means "this column does not apply".

Several report classes are frozen dataclasses with a ``nan`` default on the columns that
only some of their rows carry -- :class:`~cleverly.validation.ScoreCheckRow`'s
``score_initial`` and ``hessian_condition``, :class:`~cleverly.validation.CorrectionRow`'s
``clip_bias`` and ``margin``.  That ``nan`` is a **sentinel for an inapplicable column**
rather than an unknown quantity, so two rows both carrying it are the same row, and a
reloaded fit's rows should equal the ones it was saved from.

The generated ``__eq__`` does not say that, and until Python 3.13 it *appeared* to.  Up to
3.12 the dataclass machinery emitted one tuple comparison per class::

    (self.name, ..., self.hessian_condition) == (other.name, ..., other.hessian_condition)

and tuple comparison shortcuts on identity before it compares, so two rows that both took
the class's default ``nan`` -- the *same* float object -- compared equal.  3.13 emits a
short-circuiting chain of per-field ``==`` instead, which has no identity shortcut, so
``nan != nan`` decides it and every such comparison became ``False``.  Five lines reproduce
it with no numpy in sight::

    @dataclass(frozen=True)
    class R:
        a: int
        b: float = float("nan")

    R(1) == R(1)     # True on 3.12, False on 3.13

That the older behaviour was right by accident is the part worth keeping in mind: it held
only while both rows took the *default*, so a row built with an explicit ``float("nan")``
at the call site compared unequal on every interpreter that has ever run this package.
:func:`sentinel_equality` states the intended semantics instead of inheriting whichever one
the interpreter happens to generate, which is why the test that pins it constructs its two
``nan``\\ s separately rather than round-tripping a fit.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any, TypeVar

__all__ = ["sentinel_equality"]

T = TypeVar("T", bound=type)


class _NotApplicable:
    """The canonical stand-in every sentinel ``nan`` maps to, so they compare equal."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<not applicable>"


_NOT_APPLICABLE = _NotApplicable()


def _key(record: Any) -> tuple[Any, ...]:
    """The record's compared fields, with every sentinel ``nan`` canonicalised."""
    values = (getattr(record, field.name) for field in fields(record) if field.compare)
    return tuple(
        _NOT_APPLICABLE if isinstance(value, float) and math.isnan(value) else value
        for value in values
    )


def sentinel_equality(cls: T) -> T:
    """Compare a report row field by field, treating two sentinel ``nan``\\ s as equal.

    Apply *below* ``@dataclass`` (so it runs after it) on a frozen record class whose
    optional columns default to ``nan``::

        @sentinel_equality
        @dataclass(frozen=True)
        class ScoreCheckRow:
            ...

    ``__hash__`` is replaced alongside ``__eq__`` and over the same key, because Python
    unsets a class's hash the moment ``__eq__`` is assigned and these rows live inside
    frozen containers that are themselves hashable.  Hashing over the canonicalised key
    rather than the raw fields is not merely bookkeeping: ``hash(nan)`` is ``0`` for every
    ``nan``, so the two agree on which rows collide either way -- but the key is what makes
    "equal implies equal hashes" true by construction rather than by coincidence.

    Only ``float`` fields are canonicalised, and only when they are ``nan``.  A numpy array
    field is left alone and will raise from ``bool()`` on an ambiguous comparison exactly as
    it does today; no class this decorates has one.
    """

    def __eq__(self: Any, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return _key(self) == _key(other)

    def __hash__(self: Any) -> int:
        return hash(_key(self))

    cls.__eq__ = __eq__  # type: ignore[method-assign, assignment]
    cls.__hash__ = __hash__  # type: ignore[method-assign, assignment]
    return cls

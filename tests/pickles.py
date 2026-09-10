"""Small pickle reducers shared by backward-persistence tests."""

from __future__ import annotations

import pickle
from typing import Any


def _blank(record_class: type) -> Any:
    """Build an uninitialised record for the unpickler to fill from a state."""
    return object.__new__(record_class)


class _LegacyPickle:
    """Pickle as ``record_class`` carrying ``state`` and no newer fields."""

    def __init__(self, record_class: type, state: dict[str, Any]) -> None:
        self._record_class = record_class
        self._state = state

    def __reduce__(self) -> tuple[Any, ...]:
        return (_blank, (self._record_class,), self._state)


def legacy_without(record: Any, *dropped: str) -> Any:
    """Unpickle ``record`` as if ``dropped`` did not exist when it was written."""
    state = {name: value for name, value in record.__dict__.items() if name not in dropped}
    return pickle.loads(pickle.dumps(_LegacyPickle(type(record), state)))

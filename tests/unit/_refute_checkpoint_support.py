"""Fits and a bitwise field walk for the refutation checkpoint witnesses (X106 part G3).

The learner here lives at module level, so a fresh process builds the same fit. Its repr holds
a memory address, so a key that read the repr would change in every process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
import sys
import warnings
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


class AddressedLinear(LinearRegression):
    """A linear learner whose repr is the default ``object`` repr, which holds an address."""

    def __repr__(self) -> str:
        return object.__repr__(self)


def frame(n: int = 300, seed: int = 0) -> Any:
    from tests.unit.test_planned_refits import _point_frame

    return _point_frame(n, seed)


def in_sample(
    *, seed: int = 0, random_state: int | None = 0, learner: Any = None, n: int = 300
) -> Any:
    """An in-sample point fit with no outcome bounds, which every generated test takes."""
    from cleverly import ATE, CausalStudy, PointTreatment

    study = CausalStudy(
        frame(n, seed),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.identify(ATE()).estimate(
            outcome_learner=LinearRegression() if learner is None else learner,
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=random_state,
        )


def walk(value: Any) -> Any:
    """A canonical form compared exactly: floats by their bytes, so ``nan`` equals ``nan``
    and ``-0.0`` differs from ``0.0``."""
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (float, np.floating)):
        return ("float", struct.pack("<d", float(value)))
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (field.name, walk(getattr(value, field.name)))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, dict):
        return ("dict", tuple((walk(key), walk(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(walk(item) for item in value))
    return ("repr", type(value).__qualname__, repr(value))


def digest(value: Any) -> str:
    return hashlib.sha256(repr(walk(value)).encode()).hexdigest()


class Stop(Exception):
    """Raised by a progress callback to stop a call."""


def stopper(after: int, seen: list[Any] | None = None) -> Any:
    """A progress callback that raises once ``after`` draws have finished."""

    def progress(record: Any) -> None:
        if seen is not None:
            seen.append(record)
        if record.completed == after:
            raise Stop

    return progress


def _subprocess_main(mode: str, directory: str, stop_after: int) -> None:
    """Run the dummy-outcome test of the addressed fit in a fresh process."""
    from cleverly.validation import RefutationCheckpoint, refute

    result = in_sample(learner=AddressedLinear())
    arguments: dict[str, Any] = {"tests": ["dummy_outcome"], "n_replicates": 10}
    checkpoint = RefutationCheckpoint(directory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "stop":
            try:
                refute(result, **arguments, checkpoint=checkpoint, progress=stopper(stop_after))
            except Stop:
                return
            raise AssertionError("the call did not stop")
        print(digest(refute(result, **arguments, checkpoint=checkpoint)))


if __name__ == "__main__":
    _subprocess_main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

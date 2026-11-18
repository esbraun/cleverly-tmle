"""Shared random-seed conventions for post-fit assessment operations.

This module also holds the markers that the assessment cache reads to tell whether an answer can
be recomputed exactly. A site that draws operating-system entropy, fits a nuisance learner or
enters a parallel section calls a marker, and the mark goes to the innermost assessment computation
that is running in this context (:func:`cleverly._assessment_cache._cached`). Outside such a
computation a marker does nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

__all__ = [
    "note_fresh_entropy",
    "note_learner_fit",
    "note_parallel",
    "recording_marks",
    "resolve_assessment_seed",
]

#: The open assessment computations of this context, innermost last. Each one collects the kinds
#: of mark its own code made; a nested computation has its own set.
_FRAMES: ContextVar[tuple[set[str], ...]] = ContextVar("assessment_marks", default=())


def _mark(kind: str) -> None:
    frames = _FRAMES.get()
    if frames:
        frames[-1].add(kind)


def note_fresh_entropy() -> None:
    """Record that operating-system entropy was drawn, so a recomputation would differ."""
    _mark("entropy")


def note_learner_fit() -> None:
    """Record that a nuisance learner was fitted, whose randomness the package may not control."""
    _mark("learner_fit")


def note_parallel() -> None:
    """Record entry to a parallel section, whose workers cannot record marks of their own."""
    _mark("parallel")


@contextmanager
def recording_marks() -> Iterator[set[str]]:
    """Collect the marks made by the enclosed code, and by no nested recording.

    Yields
    ------
    set of str
        The kinds of mark the enclosed code made, filled as it runs.
    """
    frame: set[str] = set()
    token = _FRAMES.set((*_FRAMES.get(), frame))
    try:
        yield frame
    finally:
        _FRAMES.reset(token)


def resolve_assessment_seed(result: Any, random_state: int | None) -> int:
    """Resolve the reproducible root seed for an assessment that refits.

    An explicit seed wins. Otherwise, the fitted estimator's seed wins. An unseeded
    fit draws one seed so the returned assessment can record and replay it.
    """
    estimator = result.estimator
    if random_state is not None:
        return int(random_state)
    if estimator.random_state is not None:
        return int(estimator.random_state)
    note_fresh_entropy()
    return int(np.random.SeedSequence().generate_state(1)[0])

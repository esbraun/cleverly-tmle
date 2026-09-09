"""Shared random-seed conventions for post-fit assessment operations."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["resolve_assessment_seed"]


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
    return int(np.random.SeedSequence().generate_state(1)[0])

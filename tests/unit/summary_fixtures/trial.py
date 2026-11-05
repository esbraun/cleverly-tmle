"""Adaptive TMLE fits of a trial with external data, whose summaries the contract test checks.

A fit with external rows from both arms and one with external controls only, each with
explicit fast learners (X90 T2).
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

__all__ = ["FIXTURES"]


@functools.cache
def _fit(external_treated: float) -> Any:
    from tests.unit.test_atmle_workflow import _fit as fit
    from tests.unit.test_atmle_workflow import _frame

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(_frame(external_treated=external_treated))


def atmle() -> Any:
    """An adaptive CV-TMLE fit with external rows from both arms."""
    return _fit(0.3)


def atmle_controls_only() -> Any:
    """The same with external controls only."""
    return _fit(0.0)


FIXTURES: dict[str, Any] = {
    "atmle": atmle,
    "atmle_controls_only": atmle_controls_only,
}

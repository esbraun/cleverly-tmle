"""Fitted objects whose printed summaries the summary contract test checks.

Each module here exports ``FIXTURES``, a non-empty mapping from a fixture name to a function
that returns one fitted object with a ``summary()``. A seam thread that migrates a summary
adds its own module. ``tests/unit/test_summary_contract.py`` collects every module and runs
each contract item over every fixture.

Fixtures are compact and use explicit learners, so the whole contract runs in the fast tier.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

__all__ = ["collect"]


def collect() -> dict[str, Callable[[], Any]]:
    """Every fixture of every module in this package, keyed ``module:name``.

    Returns
    -------
    dict of str to callable
        The fixtures.

    Raises
    ------
    AssertionError
        When a module exports no ``FIXTURES``, or an empty one.
    """
    fixtures: dict[str, Callable[[], Any]] = {}
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        found = getattr(module, "FIXTURES", None)
        assert found, f"{module.__name__} exports no FIXTURES"
        for name, build in found.items():
            fixtures[f"{info.name}:{name}"] = build
    return fixtures

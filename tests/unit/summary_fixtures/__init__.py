"""Fitted objects whose printed summaries the summary contract test checks.

Each module here exports ``FIXTURES``, a non-empty mapping from a fixture name to a function
that returns one fitted object with a ``summary()``, or to a :class:`Case` that wraps such a
function with what the contract test cannot read off the object itself. A seam thread that
migrates a summary adds its own module. ``tests/unit/test_summary_contract.py`` collects every
module and runs each contract item over every fixture.

Fixtures are compact and use explicit learners, so the whole contract runs in the fast tier.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["Case", "collect"]


@dataclass(frozen=True)
class Case:
    """One fixture, with what the contract test needs beyond the object.

    A ``TMLEResult`` carries its data and its estimates frame, so its fixture is a plain
    function. A check or a report carries neither, so its fixture names the user-supplied
    strings it put into the object and a function that reads each printed number back.

    Parameters
    ----------
    build : callable
        Returns the object.
    names : tuple of str
        The user-supplied names (columns, labels) the object prints, which the escaping and
        house-style checks treat as the user's own text.
    fidelity : callable or None
        ``fidelity(obj, text)`` parses ``text`` (the short summary) and asserts that every
        printed number equals the value read independently from ``obj``. Required for an
        object without an estimates table.
    """

    build: Callable[[], Any]
    names: tuple[str, ...] = ()
    fidelity: Callable[[Any, str], None] | None = None

    def __call__(self) -> Any:
        return self.build()


def collect() -> dict[str, Case]:
    """Every fixture of every module in this package, keyed ``module:name``.

    Returns
    -------
    dict of str to Case
        The fixtures; a plain function becomes a :class:`Case` with no names and no
        fidelity function.

    Raises
    ------
    AssertionError
        When a module exports no ``FIXTURES``, or an empty one.
    """
    fixtures: dict[str, Case] = {}
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        found = getattr(module, "FIXTURES", None)
        assert found, f"{module.__name__} exports no FIXTURES"
        for name, build in found.items():
            case = build if isinstance(build, Case) else Case(build)
            fixtures[f"{info.name}:{name}"] = case
    return fixtures

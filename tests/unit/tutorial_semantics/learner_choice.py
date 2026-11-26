"""The reviewed semantic callback for ``docs/examples/learner-choice``."""

from __future__ import annotations

from typing import Any

from tests.unit.tutorial_semantics import EXAMPLES

#: The refit cells the rule derives from this notebook today, in notebook order.
REFIT_CELLS = ("refutation",)

NOTEBOOK = EXAMPLES / "learner-choice.ipynb"


def check(namespace: dict[str, Any]) -> None:
    """Check the page's claims."""


def check_refits(namespace: dict[str, Any]) -> None:
    """Check the refutation cell."""

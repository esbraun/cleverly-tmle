"""A number cell whose missing value stays comparable.

A cell that holds ``nan`` never equals another cell, so two builds of one summary would
differ. A number that is missing or not finite is stored as ``None`` and prints the missing
mark (or ``inf``), as :func:`number_text` prints it.
"""

from __future__ import annotations

import math
from typing import Any

from . import Cell
from ._render import number_text

__all__ = ["number_cell"]


def number_cell(value: Any, decimals: int) -> Cell:
    """A cell for ``value`` at fixed decimals; ``None`` as its value when it is not finite.

    Parameters
    ----------
    value : float or None
        The number.
    decimals : int
        Decimals to print.

    Returns
    -------
    Cell
        The cell.
    """
    number = None if value is None else float(value)
    kept = number if number is not None and math.isfinite(number) else None
    return Cell(kept, number_text(number, decimals))

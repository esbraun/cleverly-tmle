"""Plain-text rendering for the report objects.

Every ``summary()`` in the package is a title, a rule, one or more tables and a verdict,
rendered without pulling in a dataframe dependency -- a summary has to work whether or not
the caller installed pandas.

These lived in :mod:`cleverly.estimators.base` and were imported *upward* from there by
:mod:`cleverly.validation`, :mod:`cleverly.sensitivity` and :mod:`cleverly.longitudinal`,
which is backwards: a fixed-width table knows nothing about an estimator.  One consequence
was visible -- :mod:`cleverly.interventions` hand-rolled its own column widths rather than
import from ``estimators``, and drifted, since a hand-aligned table does not widen when a
regime's label does.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["format_pvalue", "format_table"]


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a fixed-width table without pulling in a dataframe dependency."""
    columns: list[Sequence[str]] = (
        list(zip(*([list(headers)] + [list(row) for row in rows]), strict=True))
        if rows
        else [[header] for header in headers]
    )
    widths = [max(len(str(cell)) for cell in column) for column in columns]
    lines = [
        "  ".join(str(header).ljust(width) for header, width in zip(headers, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(str(cell).ljust(width) for cell, width in zip(row, widths, strict=True))
        )
    return "\n".join(lines)


def format_pvalue(pvalue: float) -> str:
    """A p-value at the precision a report shows, with a floor rather than ``0.0000``."""
    if not np.isfinite(pvalue):
        return "nan"
    if pvalue < 1e-4:
        return "<1e-4"
    return f"{pvalue:.4f}"

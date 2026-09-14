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

__all__ = ["format_draw", "format_negligible", "format_pvalue", "format_table"]


def format_negligible(
    value: float,
    reference: float,
    *,
    digits: int,
    fraction: float = 1e-5,
) -> str:
    """Format a diagnostic value, displaying negligible floating residue as zero.

    The raw report frames retain ``value``. Summaries suppress only values at least five
    orders below the tolerance that gives them meaning. This keeps executed documentation
    stable across equivalent BLAS and interpreter arithmetic without hiding a diagnostic
    movement near its decision boundary.
    """
    if not np.isfinite(value):
        return "-"
    displayed = value
    if np.isfinite(reference) and reference > 0 and abs(value) < reference * fraction:
        displayed = 0.0
    return f"{displayed:.{digits}e}"


def format_draw(reported: int, total: int) -> str:
    """Name the cross-fitting draw a report describes.

    A repeated fit retains method-specific artifacts for one draw only, so several reports
    have to say which one.  The sentence was written out by hand in
    :mod:`cleverly.validation.nuisance`, :mod:`cleverly.assessment` and
    :mod:`cleverly.sensitivity.positivity`, which is how one of them starts saying
    something the others do not.  It lives here rather than beside
    :data:`~cleverly.data.weighting.REPORTED_DRAW` because it renders text and reads no
    weight: all three callers already import this module, and none of them would import
    ``data.weighting`` for a string.

    The zero-padded spelling in :func:`~cleverly.data.weighting.format_score_load` is
    deliberately not this one.  That function serves the score-load rows, whose ``cell``
    style is a fixed-width table column, and it pads every one of its three styles so a
    row and the sentence describing it agree.  These callers have no table cell to align,
    so they read as the prose they are.

    Parameters
    ----------
    reported : int
        One-based draw the report describes.
    total : int
        Draws the fit combined.

    Returns
    -------
    str
        The phrase ``"draw R of N"``, ready to embed in a sentence.
    """
    return f"draw {reported} of {total}"


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

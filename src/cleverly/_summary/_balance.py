"""The printed summary of a :class:`~cleverly.sensitivity.CovariateBalance` (X92 V11)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import Cell, Summary, SummarySection, SummaryTable, note, vocabulary
from ._render import count_text, number_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..sensitivity.balance import CovariateBalance

__all__ = ["balance_summary"]

#: Decimals of a difference.
_DECIMALS = 3


def _size(value: float) -> bool:
    return math.isfinite(value)


def balance_summary(balance: CovariateBalance, detail: str) -> Summary:
    """Build the summary of a covariate balance.

    Parameters
    ----------
    balance : CovariateBalance
        The balance.
    detail : {"short", "full"}
        ``"full"`` adds the group means and the spread each difference divides by.

    Returns
    -------
    Summary
        The report.
    """
    threshold = balance.threshold
    covariates = list(dict.fromkeys(row.covariate for row in balance.rows))
    before = sum(
        1
        for row in balance.rows
        if _size(row.smd_unweighted) and abs(row.smd_unweighted) > threshold
    )
    after = sum(
        1 for row in balance.rows if _size(row.smd_weighted) and abs(row.smd_weighted) > threshold
    )
    finite = [row for row in balance.rows if _size(row.smd_weighted)]
    worst = max(finite, key=lambda row: abs(row.smd_weighted)) if finite else None
    header = [
        (vocabulary.BALANCE_TREATMENT, balance.treatment),
        (vocabulary.BALANCE_COVARIATES, count_text(len(covariates))),
        (vocabulary.BALANCE_HELD_AGAINST, f"{threshold:g}"),
        (vocabulary.ROWS, count_text(balance.n)),
    ]
    if worst is not None:
        header.append(
            (
                vocabulary.BALANCE_LARGEST,
                f"{worst.covariate} ({number_text(abs(worst.smd_weighted), _DECIMALS)})",
            )
        )
    columns = vocabulary.BALANCE_COLUMNS
    tables = []
    for contrast in balance.contrasts:
        rows = sorted(
            balance.rows_of(contrast),
            key=lambda row: -abs(row.smd_unweighted) if _size(row.smd_unweighted) else 0.0,
        )
        cells = [
            (
                Cell(row.covariate, row.covariate),
                Cell(row.smd_unweighted, number_text(row.smd_unweighted, _DECIMALS)),
                Cell(row.smd_weighted, number_text(row.smd_weighted, _DECIMALS)),
            )
            for row in rows
        ]
        tables.append(
            SummaryTable(
                columns, tuple(cells), title=contrast if len(balance.contrasts) > 1 else None
            )
        )
    readings = vocabulary.BALANCE_READINGS
    if worst is not None and abs(worst.smd_weighted) > threshold:
        reading = readings["residual"].format(
            before=before,
            after=after,
            total=len(balance.rows),
            threshold=f"{threshold:g}",
            worst=worst.covariate,
            size=number_text(abs(worst.smd_weighted), _DECIMALS),
        )
    else:
        reading = readings["balanced"].format(
            before=before, after=after, total=len(balance.rows), threshold=f"{threshold:g}"
        )
    limits = (
        ""
        if balance.limits is None
        else f", kept within [{balance.limits[0]:g}, {balance.limits[1]:g}]"
    )
    notes = [
        note("balance_descriptive", vocabulary.NOTE_TEXTS["balance_descriptive"]),
        note(
            "balance_weights",
            vocabulary.NOTE_TEXTS["balance_weights"].format(
                limits=limits,
                origin=(
                    "the out-of-fold predictions"
                    if balance.n_repeats == 1
                    else f"the first of {balance.n_repeats} repeats' out-of-fold predictions"
                ),
            ),
        ),
        note("balance_unlisted", vocabulary.NOTE_TEXTS["balance_unlisted"]),
    ]
    sections: tuple[SummarySection, ...] = ()
    if detail == "full":
        sections = tuple(
            SummarySection(
                contrast if len(balance.contrasts) > 1 else vocabulary.BALANCE_MEANS_TITLE,
                tuple(
                    vocabulary.BALANCE_MEANS_LINE.format(
                        covariate=row.covariate,
                        group=number_text(row.mean_group, 4),
                        rest=number_text(row.mean_rest, 4),
                        group_weighted=number_text(row.mean_group_weighted, 4),
                        rest_weighted=number_text(row.mean_rest_weighted, 4),
                        spread=number_text(row.spread, 4),
                    )
                    for row in balance.rows_of(contrast)
                ),
            )
            for contrast in balance.contrasts
        )
    return Summary.build(
        title=vocabulary.BALANCE_TITLE,
        header=header,
        tables=tables,
        notes=notes,
        reading=reading,
        sections=sections,
    )

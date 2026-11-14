"""The printed summary of a :class:`~cleverly.sensitivity.BalanceOverTime` (X103)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from . import Cell, Summary, SummarySection, SummaryTable, note, vocabulary
from ._render import count_text, number_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..sensitivity.balance_time import BalanceOverTime

__all__ = ["balance_time_summary", "summary_rows"]

#: Decimals of a difference.
_DECIMALS = 3


def summary_rows(balance: BalanceOverTime, diagnostic: str) -> list[dict[str, Any]]:
    """The rows one diagnostic's table prints: the averages over earlier treatments.

    Parameters
    ----------
    balance : BalanceOverTime
        The balance.
    diagnostic : str
        The diagnostic.

    Returns
    -------
    list of dict
        One average per exposure, comparison and covariate, in report order.
    """
    from ..sensitivity.balance_time import averages

    return averages(balance.rows, over="history", diagnostic=diagnostic)


def _exposure_words(group: dict[str, Any], diagnostic: str, many: bool) -> str:
    if diagnostic == "joint":
        return str(group["exposure"])
    if diagnostic == "censoring" or many:
        return str(group["contrast"])
    return str(group["exposure"])


def balance_time_summary(balance: BalanceOverTime, detail: str) -> Summary:
    """Build the summary of a balance over time.

    Parameters
    ----------
    balance : BalanceOverTime
        The balance.
    detail : {"short", "full"}
        ``"full"`` adds the strata with no comparison.

    Returns
    -------
    Summary
        The report.
    """
    threshold = balance.threshold
    header = [
        (vocabulary.BALANCE_TIME_TREATMENTS, ", ".join(balance.treatments)),
        (vocabulary.BALANCE_TIME_DIAGNOSTICS, ", ".join(balance.diagnostics)),
        (vocabulary.BALANCE_HELD_AGAINST, f"{threshold:g}"),
        (vocabulary.ROWS, count_text(balance.n)),
    ]
    tables = []
    every: list[dict[str, Any]] = []
    for diagnostic in balance.diagnostics:
        groups = summary_rows(balance, diagnostic)
        every.extend(groups)
        many = len({group["contrast"] for group in groups}) > len(
            {group["exposure"] for group in groups}
        )
        cells = [
            (
                Cell(group["exposure"], _exposure_words(group, diagnostic, many)),
                Cell(group["covariate"], group["covariate"]),
                Cell(group["smd_unweighted"], number_text(group["smd_unweighted"], _DECIMALS)),
                Cell(group["smd_weighted"], number_text(group["smd_weighted"], _DECIMALS)),
            )
            for group in groups
        ]
        tables.append(
            SummaryTable(
                vocabulary.BALANCE_TIME_COLUMNS,
                tuple(cells),
                title=vocabulary.BALANCE_TIME_DIAGNOSTIC_TITLES[diagnostic],
            )
        )

    def large(value: float) -> bool:
        return math.isfinite(value) and abs(value) > threshold

    before = sum(1 for group in every if large(group["smd_unweighted"]))
    after = sum(1 for group in every if large(group["smd_weighted"]))
    finite = [group for group in every if math.isfinite(group["smd_weighted"])]
    worst = max(finite, key=lambda group: abs(group["smd_weighted"])) if finite else None
    readings = vocabulary.BALANCE_TIME_READINGS
    if worst is not None and abs(worst["smd_weighted"]) > threshold:
        reading = readings["residual"].format(
            before=before,
            after=after,
            total=len(every),
            threshold=f"{threshold:g}",
            worst=worst["covariate"],
            exposure=worst["exposure"],
            size=number_text(abs(worst["smd_weighted"]), _DECIMALS),
        )
    else:
        reading = readings["balanced"].format(
            before=before, after=after, total=len(every), threshold=f"{threshold:g}"
        )
    texts = vocabulary.NOTE_TEXTS
    notes = [
        note("balance_time_descriptive", texts["balance_time_descriptive"]),
        note("balance_time_averaged", texts["balance_time_averaged"]),
    ]
    if balance.limits is not None:
        notes.append(
            note(
                "balance_time_bounds",
                texts["balance_time_bounds"].format(
                    lower=balance.limits[0],
                    upper=balance.limits[1],
                    share=f"{balance.bounded_share:.1%}",
                ),
            )
        )
    for diagnostic, reason in balance.omitted:
        notes.append(
            note(
                "balance_time_omitted",
                texts["balance_time_omitted"].format(diagnostic=diagnostic, reason=reason),
            )
        )
    for text in balance.notes:
        notes.append(
            note(
                "balance_time_collaborative", texts["balance_time_collaborative"].format(text=text)
            )
        )
    sections: tuple[SummarySection, ...] = ()
    if detail == "full":
        sections = (
            SummarySection(
                vocabulary.BALANCE_TIME_DROPPED_TITLE,
                tuple(
                    vocabulary.BALANCE_TIME_DROPPED_LINE.format(
                        diagnostic=item.diagnostic,
                        exposure=item.exposure,
                        strata=item.strata,
                        units=item.units,
                    )
                    for item in balance.dropped
                )
                or ("none",),
            ),
        )
    return Summary.build(
        title=vocabulary.BALANCE_TIME_TITLE,
        header=header,
        tables=tables,
        notes=notes,
        reading=reading,
        sections=sections,
    )

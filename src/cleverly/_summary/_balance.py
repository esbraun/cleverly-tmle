"""The printed summary of a :class:`~cleverly.sensitivity.CovariateBalance` (X92 V11).

The sentences that say what a difference divides by, how a unit counts after weighting and
where the probabilities come from are built here once; the love plot prints the same ones.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import Cell, Summary, SummarySection, SummaryTable, note, vocabulary
from ._render import count_text, number_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..sensitivity.balance import CovariateBalance

__all__ = [
    "balance_summary",
    "descriptive_text",
    "rest_text",
    "source_text",
    "weights_text",
]

#: Decimals of a difference.
_DECIMALS = 3


def _size(value: float) -> bool:
    return math.isfinite(value)


def descriptive_text(balance: CovariateBalance) -> str:
    """That the differences are descriptive, and what each one is divided by.

    Parameters
    ----------
    balance : CovariateBalance
        The balance.

    Returns
    -------
    str
        The sentence.
    """
    spreads = vocabulary.BALANCE_SPREAD_TEXTS
    if balance.estimand in ("att", "atc"):
        spread = spreads[balance.estimand]
    elif balance.n_arms > 2:
        spread = spreads["pooled_arms"].format(arms=balance.n_arms)
    else:
        spread = spreads["pooled_two"]
    return vocabulary.NOTE_TEXTS["balance_descriptive"].format(spread=spread)


def source_text(balance: CovariateBalance) -> str:
    """Where the propensities come from: in-sample, out-of-fold, or the first repeat.

    Parameters
    ----------
    balance : CovariateBalance
        The balance.

    Returns
    -------
    str
        The sentence.
    """
    sources = vocabulary.BALANCE_SOURCE_TEXTS
    if not balance.cross_fitted:
        return sources["in_sample"]
    if balance.n_repeats > 1:
        return sources["repeats"].format(repeats=balance.n_repeats)
    return sources["out_of_fold"]


def weights_text(balance: CovariateBalance) -> str:
    """How a unit counts after weighting, within which limits, and before weighting.

    Parameters
    ----------
    balance : CovariateBalance
        The balance.

    Returns
    -------
    str
        The sentences.
    """
    limits = (
        ""
        if balance.limits is None
        else f", kept within [{balance.limits[0]:.3g}, {balance.limits[1]:.3g}]"
    )
    text = vocabulary.BALANCE_WEIGHT_TEXTS[(balance.estimand, balance.weighted_data)]
    return text.format(limits=limits)


def rest_text(balance: CovariateBalance, part: str) -> str | None:
    """That each comparison is one arm against the rest, or ``None`` when none is.

    Parameters
    ----------
    balance : CovariateBalance
        The balance.
    part : {"table", "panel"}
        What holds one comparison.

    Returns
    -------
    str or None
        The sentence.
    """
    if balance.estimand != "ate" or balance.n_arms <= 2:
        return None
    return vocabulary.NOTE_TEXTS["balance_rest"].format(part=part)


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
        (vocabulary.BALANCE_WEIGHTS_FOR, vocabulary.BALANCE_ESTIMAND_WORDS[balance.estimand]),
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
    notes = [
        note("balance_descriptive", descriptive_text(balance)),
        note(
            "balance_weights",
            vocabulary.NOTE_TEXTS["balance_weights"].format(
                weights=weights_text(balance), source=source_text(balance)
            ),
        ),
    ]
    rest = rest_text(balance, "table")
    if rest is not None:
        notes.append(note("balance_rest", rest))
    notes.append(note("balance_unlisted", vocabulary.NOTE_TEXTS["balance_unlisted"]))
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

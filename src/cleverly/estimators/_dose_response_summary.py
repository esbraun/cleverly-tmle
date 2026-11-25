"""The readable summary of a kernel-smoothed dose-response fit.

``DoseResponseResult.summary()`` builds here, on the shared :mod:`cleverly._summary` layer.
Every fact is read from the result: the declaration, the pairs' estimates and band, the stored
outcome-regression record, and, on a fit through :class:`~cleverly.CausalStudy`, the identified
question and the study protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._summary import Detail, Summary, SummarySection, count_text, note, vocabulary
from .._summary._estimates import estimates_table, plain_support_reason
from .._summary._setup import _protocol_section

if TYPE_CHECKING:
    from .dose_response import DoseResponseResult


def _numbers(values: tuple[float, ...]) -> str:
    return ", ".join(f"{value:g}" for value in values)


def _header(result: DoseResponseResult) -> tuple[list[tuple[str, str]], int]:
    curve = result.curve
    record = result.nuisance_record
    low, high = curve.support
    folds = int(record.folds.max()) + 1 if record.folds.size else 1
    left: list[tuple[str, str]] = [
        (vocabulary.QUESTION, vocabulary.DOSE_RESPONSE_QUESTION),
        (vocabulary.DOSE_RANGE, f"[{low:g}, {high:g}]"),
        (vocabulary.DOSE_KERNEL, str(curve.kernel)),
    ]
    right: list[tuple[str, str]] = [
        (vocabulary.ROWS, count_text(int(record.outcome.size))),
        (vocabulary.DOSE_COUNT, count_text(len(curve.doses))),
        (vocabulary.DOSE_BANDWIDTHS, count_text(len(curve.bandwidths))),
        (
            vocabulary.OUT_OF_FOLD,
            vocabulary.DOSE_FOLDS_WORD.format(k=folds) if record.cross_fitted else "no",
        ),
    ]
    return [*left, *right], len(left)


def _assumptions(result: DoseResponseResult) -> SummarySection:
    effect = result.identified_effect
    items = (
        tuple(effect.identification.assumptions)
        if effect is not None
        else vocabulary.DOSE_RESPONSE_ASSUMPTIONS
    )
    lines = [vocabulary.ASSUMPTIONS_LEAD_GENERIC]
    lines.extend(f"- {item}" for item in items)
    return SummarySection(vocabulary.SECTION_ASSUMPTIONS, tuple(lines))


def _settings(result: DoseResponseResult) -> SummarySection:
    curve = result.curve
    low, high = curve.support
    lines = [
        vocabulary.DOSE_SETTINGS_DECLARATION.format(
            doses=_numbers(curve.doses),
            bandwidths=_numbers(curve.bandwidths),
            low=f"{low:g}",
            high=f"{high:g}",
            kernel=curve.kernel,
        )
    ]
    band = result.band.support
    if band is not None:
        lines.append(vocabulary.DOSE_SETTINGS_BAND.format(label=band.label))
        if band.label == vocabulary.EXPERIMENTAL and band.reason:
            lines.append(
                vocabulary.DOSE_SETTINGS_BAND_REASON.format(
                    reason=plain_support_reason(band.reason)
                )
            )
    reasons = dict.fromkeys(
        estimate.support.reason
        for estimate in result.estimates.values()
        if estimate.support.label == vocabulary.EXPERIMENTAL and estimate.support.reason
    )
    for reason in reasons:
        lines.append(
            vocabulary.DOSE_SETTINGS_ROWS_REASON.format(reason=plain_support_reason(reason))
        )
    return SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines))


def dose_response_summary(result: DoseResponseResult, detail: Detail) -> Summary:
    """Build the summary of a dose-response fit.

    Parameters
    ----------
    result : DoseResponseResult
        The fitted curve.
    detail : {"short", "full"}
        ``"full"`` adds the Assumptions, Settings and Study description sections.

    Returns
    -------
    Summary
        The summary.
    """
    header, split = _header(result)
    level = 1.0 - result.alpha
    table, notes = estimates_table(result.estimates, level=level)
    if len(result.estimates) > 1:
        notes.append(
            note(
                "simultaneous_bands",
                vocabulary.NOTE_TEXTS["simultaneous_bands"].format(
                    count=count_text(len(result.estimates)), level=f"{level:.0%}"
                ),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        effect = result.identified_effect
        sections.append(_assumptions(result))
        sections.append(_settings(result))
        sections.append(_protocol_section(None if effect is None else effect.protocol))
    return Summary.build(
        title=vocabulary.DOSE_RESPONSE_TITLE,
        header=header,
        header_split=split,
        tables=[table],
        notes=notes,
        sections=sections,
    )

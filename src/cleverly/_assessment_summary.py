"""The printed summaries of the combined reports of checks (X89 T3).

``result.validate()``, ``result.diagnostics.run_all()``, ``result.sensitivity.run_all()`` and
``result.assess()`` print like every other check: a header of counts, a table with a row per
check (its plain name, its result in words and one finding), a reading, and notes for the
suggested next steps and how to reach each row's full report. A row's finding is the reading
of the report it retained, computed when the summary prints; a row that did not run prints
the first sentence of why. The full detail lists every row with its whole reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ._summary import Cell, Detail, Note, Summary, SummarySection, SummaryTable, note, vocabulary
from ._summary import short_repr as _short_repr
from ._summary._estimates import without_roadmap
from ._summary._render import count_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .assessment import (
        AssessmentCapability,
        AssessmentItem,
        AssessmentReport,
        DiagnosticReport,
        Replayability,
        ValidationReport,
    )

__all__ = [
    "assessment_repr",
    "assessment_summary",
    "capability_cost",
    "capability_repr",
    "check_name",
    "diagnostic_repr",
    "diagnostic_summary",
    "facade_repr",
    "facade_summary",
    "item_finding",
    "item_repr",
    "quoted_reasons",
    "replay_part",
    "replay_repr",
    "replay_summary",
    "validation_repr",
    "validation_summary",
]

_SENTENCE_END = re.compile(r"(?<=[.;])\s")
_READING_END = re.compile(r"(?<=\.)\s")


def check_name(name: str) -> str:
    """The plain name of an operation.

    Parameters
    ----------
    name : str
        The operation, such as ``"nuisance_models"``.

    Returns
    -------
    str
        Such as ``"outcome and treatment models"``. An unknown operation prints with
        spaces for its underscores.
    """
    return vocabulary.CHECK_NAMES.get(name, name.replace("_", " "))


def _joined(items: Sequence[str], conjunction: str = "and") -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {conjunction} " + items[-1]


def _first_sentence(text: str) -> str:
    """The first sentence or clause of a reason, without its roadmap ids."""
    plain = without_roadmap(text).strip()
    first = _SENTENCE_END.split(plain, maxsplit=1)[0]
    return first.rstrip(".;")


def _omitted(item: AssessmentItem) -> bool:
    from .assessment import _OMISSIONS

    return item.status in _OMISSIONS


def _retains(item: AssessmentItem) -> bool:
    from .assessment import _AbsentReport

    return item._report is not _AbsentReport.TOKEN


def item_finding(item: AssessmentItem) -> str:
    """One finding for a battery row.

    Parameters
    ----------
    item : AssessmentItem
        The row.

    Returns
    -------
    str
        The reading of the report the row retained; for a row whose report has no reading,
        where to find the report; for a row that did not run or retained no report, the
        first sentence of its detail.
    """
    if _omitted(item) or not _retains(item):
        return _first_sentence(item.detail)
    summary = getattr(item.report, "summary", None)
    reading = getattr(summary(), "reading", None) if callable(summary) else None
    return str(reading) if reading else vocabulary.FINDING_HELD.format(name=item.name)


def quoted_reasons(items: Sequence[AssessmentItem]) -> tuple[str, ...]:
    """The refusal reasons a battery quotes, whose words belong to the code that refused.

    Parameters
    ----------
    items : sequence of AssessmentItem
        The rows.

    Returns
    -------
    tuple of str
        Each refused row's whole reason and its first sentence, as printed.
    """
    reasons = []
    for item in items:
        if item.refusal is not None and _omitted(item):
            reasons.extend([without_roadmap(item.detail).strip(), _first_sentence(item.detail)])
    return tuple(reasons)


def _status(item: AssessmentItem) -> str:
    return vocabulary.STATUS_WORDS[item.status.value]


def _finding(item: AssessmentItem) -> Cell:
    """A table's finding: the first sentence of a reading, which "Rows in full" prints whole."""
    text = item_finding(item)
    return Cell(text, _READING_END.split(text, maxsplit=1)[0])


def _row(item: AssessmentItem, label: str) -> tuple[Cell, ...]:
    return (Cell(item.name, label), Cell(item.status.value, _status(item)), _finding(item))


def _labels(presented: Sequence[tuple[str, AssessmentItem]]) -> list[str]:
    """Each row's plain name, with its surface when two rows share a name."""
    names = [check_name(item.name) for _, item in presented]
    return [
        vocabulary.BATTERY_SURFACE.format(name=name, surface=surface)
        if names.count(name) > 1
        else name
        for name, (surface, _) in zip(names, presented, strict=True)
    ]


def _table(presented: Sequence[tuple[str, AssessmentItem]]) -> SummaryTable:
    if not presented:
        return SummaryTable(("checks",), ((Cell(None, "none"),),))
    rows = tuple(
        _row(item, label) for (_, item), label in zip(presented, _labels(presented), strict=True)
    )
    return SummaryTable(vocabulary.BATTERY_COLUMNS, rows)


def _counts(items: Sequence[AssessmentItem]) -> list[tuple[str, str]]:
    from .assessment import _ATTENTION

    omitted = sum(_omitted(item) for item in items)
    return [
        (vocabulary.CHECKS_RUN, count_text(len(items) - omitted)),
        (vocabulary.NEED_ATTENTION, count_text(sum(item.status in _ATTENTION for item in items))),
        (vocabulary.NOT_RUN, count_text(omitted)),
    ]


def _names(items: Sequence[AssessmentItem], statuses: Any) -> list[str]:
    return [check_name(item.name) for item in items if item.status in statuses]


def _inference_reasons(items: Sequence[AssessmentItem]) -> list[str]:
    """The plain reason of each labelled inference status a retained report names."""
    from ._inference_status import status_record

    reasons = []
    for item in items:
        report: Any = item.report if _retains(item) else None
        if getattr(report, "inference_note", None):
            reasons.append(status_record(report.inference).plain_reason)
    return list(dict.fromkeys(reasons))


def _notes(items: Sequence[AssessmentItem], steps: Sequence[str]) -> list[Note]:
    texts = vocabulary.NOTE_TEXTS
    notes = [
        note("inference_status", texts["inference_status"].format(reason=reason))
        for reason in _inference_reasons(items)
    ]
    if steps:
        plain = [step.rstrip(".") for step in steps]
        notes.append(note("next_steps", texts["next_steps"].format(steps="; ".join(plain))))
    notes.append(note("report_access", texts["report_access"]))
    return notes


def _rows_in_full(presented: Sequence[tuple[str, AssessmentItem]]) -> list[SummarySection]:
    lines = [
        vocabulary.BATTERY_ROW_LINE.format(
            name=label,
            status=_status(item),
            text=(
                without_roadmap(item.detail).strip()
                if _omitted(item) or not _retains(item)
                else item_finding(item)
            ),
        )
        for (_, item), label in zip(presented, _labels(presented), strict=True)
    ]
    if not lines:
        lines.append(vocabulary.BATTERY_NO_ROWS)
    return [SummarySection(vocabulary.BATTERY_ROWS_IN_FULL, tuple(lines))]


def _attention_reading(items: Sequence[AssessmentItem]) -> str | None:
    from .assessment import AssessmentStatus

    failed = _names(items, {AssessmentStatus.FAILED})
    if failed:
        return vocabulary.BATTERY_READINGS["failed"].format(names=_joined(failed))
    warned = _names(items, {AssessmentStatus.WARNING})
    if warned:
        return vocabulary.BATTERY_READINGS["attention"].format(
            names=_joined(warned), verb="needs" if len(warned) == 1 else "need"
        )
    return None


def _validation_state(report: ValidationReport) -> str:
    from .assessment import _OMISSIONS, AssessmentStatus

    statuses = {item.status for item in report.items}
    if AssessmentStatus.FAILED in statuses:
        return "failed"
    if AssessmentStatus.WARNING in statuses:
        return "attention"
    if statuses & (_OMISSIONS - {AssessmentStatus.NOT_APPLICABLE}):
        return "incomplete"
    return "passed"


def validation_summary(report: ValidationReport, detail: Detail) -> Summary:
    """The printed summary of ``result.validate()``.

    Parameters
    ----------
    report : ValidationReport
        The report.
    detail : {"short", "full"}
        ``"full"`` adds every row with its whole reason.

    Returns
    -------
    Summary
        The summary.
    """
    from .assessment import AssessmentStatus

    items = report.items
    state = _validation_state(report)
    reading = _attention_reading(items)
    if reading is None:
        if state == "incomplete":
            missing = _names(items, {AssessmentStatus.DEFERRED, AssessmentStatus.UNAVAILABLE})
            reading = vocabulary.BATTERY_READINGS["incomplete"].format(names=_joined(missing))
        else:
            reading = vocabulary.BATTERY_READINGS["passed"]
    presented = [("validation", item) for item in items]
    return Summary.build(
        title=vocabulary.BATTERY_TITLES["validation"],
        header=[
            (vocabulary.RESULT, vocabulary.BATTERY_RESULTS_WORDS[state]),
            *_counts(items),
        ],
        tables=[_table(presented)],
        notes=_notes(items, report.next_steps()),
        reading=reading,
        sections=_rows_in_full(presented) if detail == "full" else (),
    )


def diagnostic_summary(report: DiagnosticReport, detail: Detail) -> Summary:
    """The printed summary of ``run_all()`` on the diagnostics or sensitivity facade.

    Parameters
    ----------
    report : DiagnosticReport
        The report.
    detail : {"short", "full"}
        ``"full"`` adds every row with its whole reason.

    Returns
    -------
    Summary
        The summary.
    """
    items = report.items
    reading = _attention_reading(items) or vocabulary.BATTERY_READINGS["diagnostics_clear"]
    presented = [(report.surface, item) for item in items]
    return Summary.build(
        title=vocabulary.BATTERY_TITLES.get(
            report.surface, vocabulary.BATTERY_TITLES["diagnostics"]
        ),
        header=[
            *_counts(items),
            (vocabulary.REFITS_ALLOWED, vocabulary.YES_NO[report.include_refits]),
            (vocabulary.RETARGETS_ALLOWED, vocabulary.YES_NO[report.include_retargets]),
        ],
        tables=[_table(presented)],
        notes=_notes(items, report.next_steps()),
        reading=reading,
        sections=_rows_in_full(presented) if detail == "full" else (),
    )


def assessment_summary(report: AssessmentReport, detail: Detail) -> Summary:
    """The printed summary of ``result.assess()``.

    Parameters
    ----------
    report : AssessmentReport
        The report.
    detail : {"short", "full"}
        ``"full"`` adds every row with its whole reason.

    Returns
    -------
    Summary
        The summary.
    """
    from .assessment import (
        _ATTENTION,
        _SUMMARY_CHECK_ORDER,
        _SUMMARY_OMISSION_ORDER,
        AssessmentStatus,
    )

    presented = list(report._presented())
    items = [item for _, item in presented]
    labels = dict(zip((id(item) for item in items), _labels(presented), strict=True))
    checks = [
        (surface, item)
        for status in _SUMMARY_CHECK_ORDER
        for surface, item in presented
        if item.status is status
    ]
    results = [pair for pair in presented if pair[1].status is AssessmentStatus.COMPLETED]
    omitted = [
        (surface, item)
        for status in _SUMMARY_OMISSION_ORDER
        for surface, item in presented
        if item.status is status
    ]
    tables = []
    # Returned results lead: they carry the numbers a reader interprets. The header and
    # the reading still count and name every check that needs attention.
    if results:
        tables.append(
            SummaryTable(
                vocabulary.BATTERY_RESULT_COLUMNS,
                tuple((Cell(item.name, labels[id(item)]), _finding(item)) for _, item in results),
                title=vocabulary.BATTERY_RESULTS,
            )
        )
    if checks:
        tables.append(
            SummaryTable(
                vocabulary.BATTERY_COLUMNS,
                tuple(_row(item, labels[id(item)]) for _, item in checks),
                title=vocabulary.BATTERY_CHECKS,
            )
        )
    if omitted:
        tables.append(
            SummaryTable(
                vocabulary.BATTERY_NOT_RUN_COLUMNS,
                tuple(
                    (Cell(item.name, labels[id(item)]), Cell(item.detail, item_finding(item)))
                    for _, item in omitted
                ),
                title=vocabulary.BATTERY_NOT_RUN,
            )
        )
    if not tables:
        tables.append(SummaryTable(("checks",), ((Cell(None, "none"),),)))
    attention = [labels[id(item)] for item in items if item.status in _ATTENTION]
    if attention:
        reading = vocabulary.BATTERY_READINGS["assessment"].format(
            count=count_text(len(attention)),
            checks="check" if len(attention) == 1 else "checks",
            verb="needs" if len(attention) == 1 else "need",
            names=_joined(attention),
        )
    else:
        reading = vocabulary.BATTERY_READINGS["assessment_clear"]
    statuses = {item.status for item in items}
    if AssessmentStatus.FAILED in statuses:
        state = "failed"
    elif AssessmentStatus.WARNING in statuses:
        state = "attention"
    else:
        state = "clear"
    return Summary.build(
        title=vocabulary.BATTERY_TITLES["assessment"],
        header=[(vocabulary.RESULT, vocabulary.BATTERY_RESULTS_WORDS[state]), *_counts(items)],
        tables=tables,
        notes=_notes(items, report.next_steps()),
        reading=reading,
        sections=_rows_in_full(presented) if detail == "full" else (),
    )


def item_repr(item: AssessmentItem) -> str:
    """A one-line repr of a battery row.

    Parameters
    ----------
    item : AssessmentItem
        The row.

    Returns
    -------
    str
        Such as ``AssessmentItem(support, done (no pass or fail))``.
    """
    return _short_repr("AssessmentItem", [("", item.name), ("", _status(item))])


def _report_repr(name: str, items: Sequence[AssessmentItem], result: str) -> str:
    return _short_repr(name, [("", f"{len(items)} rows"), ("", result)])


def validation_repr(report: ValidationReport) -> str:
    """A one-line repr of ``result.validate()``.

    Parameters
    ----------
    report : ValidationReport
        The report.

    Returns
    -------
    str
        Such as ``ValidationReport(3 rows, passed)``.
    """
    words = vocabulary.BATTERY_RESULTS_WORDS[_validation_state(report)]
    return _report_repr("ValidationReport", report.items, words)


def diagnostic_repr(report: DiagnosticReport) -> str:
    """A one-line repr of a ``run_all()`` report.

    Parameters
    ----------
    report : DiagnosticReport
        The report.

    Returns
    -------
    str
        Such as ``DiagnosticReport(6 rows, 0 need attention)``.
    """
    from .assessment import _ATTENTION

    attention = sum(item.status in _ATTENTION for item in report.items)
    return _report_repr("DiagnosticReport", report.items, f"{attention} need attention")


def assessment_repr(report: AssessmentReport) -> str:
    """A one-line repr of ``result.assess()``.

    Parameters
    ----------
    report : AssessmentReport
        The report.

    Returns
    -------
    str
        Such as ``AssessmentReport(16 rows, 0 need attention)``.
    """
    return _report_repr(
        "AssessmentReport",
        [item for _, item in report._presented()],
        f"{len(report.attention)} need attention",
    )


# ----------------------------------------------------------------- capabilities and replay


def capability_cost(capability: AssessmentCapability) -> str:
    """What running a check costs, in words.

    Parameters
    ----------
    capability : AssessmentCapability
        The declared row.

    Returns
    -------
    str
        ``"cheap"``, ``"re-estimates"`` or ``"refits the models"``.
    """
    if capability.execution == "summarize" or capability.cost == "cheap":
        return vocabulary.CHECK_COSTS["cheap"]
    return vocabulary.CHECK_COSTS[capability.execution]


def _capability_state(capability: AssessmentCapability) -> str:
    if capability.available:
        return vocabulary.YES_NO[True]
    return vocabulary.STATUS_WORDS[capability.status.value]


def capability_repr(capability: AssessmentCapability) -> str:
    """A one-line repr of a declared check.

    Parameters
    ----------
    capability : AssessmentCapability
        The declared row.

    Returns
    -------
    str
        Such as ``AssessmentCapability(refute, available, refits the models)``, or
        ``AssessmentCapability(benchmark, on request)`` for a row that cannot run as declared.
    """
    if capability.available:
        parts = [("", capability.operation), ("", "available"), ("", capability_cost(capability))]
    else:
        parts = [("", capability.operation), ("", _capability_state(capability))]
    return _short_repr("AssessmentCapability", parts)


#: The longest repr of a facade before its list of checks is cut.
_FACADE_REPR_WIDTH = 100


def facade_repr(name: str, capabilities: Sequence[AssessmentCapability]) -> str:
    """A one-line repr of a facade: the checks it can run now.

    Parameters
    ----------
    name : str
        The facade's class name.
    capabilities : sequence of AssessmentCapability
        Its declared rows.

    Returns
    -------
    str
        Such as ``DiagnosticsFacade(support, nuisance_models, ...)``, at most 100 characters.
    """
    names = [row.operation for row in capabilities if row.available]
    text = f"{name}({', '.join(names)})"
    if len(text) <= _FACADE_REPR_WIDTH:
        return text
    kept: list[str] = []
    for operation in names:
        if len(f"{name}({', '.join([*kept, operation])}, ...)") > _FACADE_REPR_WIDTH:
            break
        kept.append(operation)
    return f"{name}({', '.join([*kept, '...'])})"


def facade_summary(surface: str, capabilities: Sequence[AssessmentCapability]) -> Summary:
    """The checks a facade offers on this fit, for its notebook display.

    Parameters
    ----------
    surface : {"diagnostics", "sensitivity"}
        The facade.
    capabilities : sequence of AssessmentCapability
        Its declared rows.

    Returns
    -------
    Summary
        A table of each check, the question it answers, whether it can run and its cost,
        and a note naming each check that cannot run and where to read why.
    """
    blocked = [
        f"{row.operation} ({_capability_state(row)})" for row in capabilities if not row.available
    ]
    rows = tuple(
        (
            Cell(row.operation, row.operation),
            Cell(row.operation, check_name(row.operation)),
            Cell(str(row.available), vocabulary.YES_NO[row.available]),
            Cell(row.execution, capability_cost(row)),
        )
        for row in capabilities
    )
    return Summary.build(
        title=vocabulary.FACADE_TITLES[surface],
        header=[
            (vocabulary.CHECKS_LISTED, count_text(len(capabilities))),
            (vocabulary.CHECKS_AVAILABLE, count_text(sum(row.available for row in capabilities))),
        ],
        tables=[SummaryTable(vocabulary.FACADE_COLUMNS, rows)],
        notes=[
            note(
                "why_not",
                vocabulary.NOTE_TEXTS["why_not"].format(surface=surface, names=_joined(blocked)),
            )
        ]
        if blocked
        else [],
        reading=vocabulary.FACADE_READING.format(surface=surface),
    )


def _replay_slots(replay: Replayability) -> list[tuple[str, bool]]:
    return [(slot, bool(getattr(replay, slot))) for slot in vocabulary.REPLAY_ACTIONS]


def replay_part(code: str) -> str:
    """A part a saved result lacks, in words.

    Parameters
    ----------
    code : str
        An entry of ``Replayability.unreconstructible``.

    Returns
    -------
    str
        Its plain words; an unknown code prints with spaces for its underscores.
    """
    return vocabulary.REPLAY_PARTS.get(code, code.replace("_", " "))


def replay_summary(replay: Replayability, detail: Detail) -> Summary:
    """The printed summary of what a saved result can redo.

    Parameters
    ----------
    replay : Replayability
        The record.
    detail : {"short", "full"}
        ``"full"`` adds what each action needs.

    Returns
    -------
    Summary
        The summary.
    """
    slots = _replay_slots(replay)
    actions = vocabulary.REPLAY_ACTIONS
    can = [actions[slot][0] for slot, possible in slots if possible]
    cannot = [actions[slot][0] for slot, possible in slots if not possible]
    if not cannot:
        reading = vocabulary.REPLAY_READINGS["all"]
    elif not can:
        reading = vocabulary.REPLAY_READINGS["none"]
    else:
        reading = vocabulary.REPLAY_READINGS["some"].format(cannot=_joined(cannot, "or"))
    notes = []
    if replay.unreconstructible:
        parts = list(dict.fromkeys(replay_part(code) for code in replay.unreconstructible))
        notes.append(
            note(
                "missing_parts", vocabulary.NOTE_TEXTS["missing_parts"].format(parts=_joined(parts))
            )
        )
    sections: tuple[SummarySection, ...] = ()
    if detail == "full":
        lines = tuple(
            vocabulary.REPLAY_NEED_LINE.format(action=actions[slot][0], needs=actions[slot][2])
            for slot, _ in slots
        )
        sections = (SummarySection(vocabulary.REPLAY_NEEDS, lines),)
    return Summary.build(
        title=vocabulary.REPLAY_TITLE,
        header=[(vocabulary.REPLAY_POSSIBLE, f"{len(can)} of {len(slots)}")],
        tables=[
            SummaryTable(
                vocabulary.REPLAY_COLUMNS,
                tuple(
                    (Cell(slot, actions[slot][0]), Cell(possible, vocabulary.YES_NO[possible]))
                    for slot, possible in slots
                ),
            )
        ],
        notes=notes,
        reading=reading,
        sections=sections,
    )


def replay_repr(replay: Replayability) -> str:
    """A one-line repr naming the actions a saved result can redo.

    Parameters
    ----------
    replay : Replayability
        The record.

    Returns
    -------
    str
        Such as ``Replayability(summarize, re-estimate, refit, rerun)``.
    """
    names = [
        vocabulary.REPLAY_ACTIONS[slot][1] for slot, possible in _replay_slots(replay) if possible
    ]
    return f"Replayability({', '.join(names) or 'nothing'})"

"""The reader that checks every row and count a combined report prints (X89 T3).

``result.validate()``, ``run_all()`` and ``result.assess()`` print a header of counts and a
table of rows. The reader parses the printed text and compares each count with the report's
rows, and each row's label, result and finding with what it reads off the row on its own: the
row's status, its reason, or the reading of the report it retained.
"""

from __future__ import annotations

import re
from typing import Any

from cleverly._summary import vocabulary
from cleverly._summary._estimates import without_roadmap
from cleverly.assessment import (
    AssessmentReport,
    AssessmentStatus,
    DiagnosticReport,
)

from ._check_fidelity import header_values, text_tables

__all__ = ["read_battery"]

_ATTENTION = {AssessmentStatus.FAILED, AssessmentStatus.WARNING}
_OMITTED = {
    AssessmentStatus.DEFERRED,
    AssessmentStatus.UNAVAILABLE,
    AssessmentStatus.NOT_APPLICABLE,
}
_CHECK_ORDER = (AssessmentStatus.FAILED, AssessmentStatus.WARNING, AssessmentStatus.PASSED)
_OMISSION_ORDER = (
    AssessmentStatus.DEFERRED,
    AssessmentStatus.UNAVAILABLE,
    AssessmentStatus.NOT_APPLICABLE,
)


def _rows(report: Any) -> list[tuple[str, Any]]:
    if isinstance(report, AssessmentReport):
        return list(report._presented())
    surface = getattr(report, "surface", "validation")
    return [(surface, item) for item in report.items]


def _label(name: str, surface: str, names: list[str]) -> str:
    plain = vocabulary.CHECK_NAMES.get(name, name.replace("_", " "))
    plains = [vocabulary.CHECK_NAMES.get(other, other.replace("_", " ")) for other in names]
    return f"{plain} ({surface})" if plains.count(plain) > 1 else plain


def _finding(item: Any) -> str:
    """The finding read off the row: its reason's first clause, or its report's reading."""
    if item.status in _OMITTED or item._report.__class__.__name__ == "_AbsentReport":
        return re.split(r"(?<=[.;])\s", without_roadmap(item.detail).strip())[0].rstrip(".;")
    summary = getattr(item.report, "summary", None)
    reading = summary().reading if callable(summary) else None
    if not reading:
        return f"see report.report('{item.name}')"
    return re.split(r"(?<=\.)\s(?=[A-Z])", reading)[0]


def _same(shown: str, expected: str) -> bool:
    """Equal up to wrapping, which may break a long word."""
    return "".join(shown.split()) == "".join(expected.split())


def read_battery(report: Any, text: str) -> None:
    """Assert that every count and row printed in ``text`` matches ``report``.

    Parameters
    ----------
    report : ValidationReport, DiagnosticReport or AssessmentReport
        The report.
    text : str
        Its short printed summary.
    """
    presented = _rows(report)
    items = [item for _, item in presented]
    names = [item.name for item in items]
    facts = header_values(text)
    omitted = sum(item.status in _OMITTED for item in items)
    assert facts[vocabulary.CHECKS_RUN] == str(len(items) - omitted)
    assert facts[vocabulary.NOT_RUN] == str(omitted)
    assert facts[vocabulary.NEED_ATTENTION] == str(sum(item.status in _ATTENTION for item in items))
    if isinstance(report, DiagnosticReport):
        words = {True: "yes", False: "no"}
        assert facts[vocabulary.REFITS_ALLOWED] == words[report.include_refits]
        assert facts[vocabulary.RETARGETS_ALLOWED] == words[report.include_retargets]

    if isinstance(report, AssessmentReport):
        expected = {
            "Results": [pair for pair in presented if pair[1].status is AssessmentStatus.COMPLETED],
            "Checks": [
                pair for status in _CHECK_ORDER for pair in presented if pair[1].status is status
            ],
            "Not run": [
                pair for status in _OMISSION_ORDER for pair in presented if pair[1].status is status
            ],
        }
        expected = {title: pairs for title, pairs in expected.items() if pairs}
    else:
        expected = {"": presented}
    tables = text_tables(text)
    assert [title for title, _, _ in tables] == list(expected), [t for t, _, _ in tables]
    for (title, labels, rows), pairs in zip(tables, expected.values(), strict=True):
        assert len(rows) == len(pairs), (title, rows)
        for cells, (surface, item) in zip(rows, pairs, strict=True):
            assert cells[0] == _label(item.name, surface, names), (cells, item.name)
            if labels[1] == "result":
                assert cells[1] == vocabulary.STATUS_WORDS[item.status.value], cells
            assert _same(cells[-1], _finding(item)), (cells[-1], _finding(item))

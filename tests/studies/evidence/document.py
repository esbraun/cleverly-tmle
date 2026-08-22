"""Refresh the measured values a study's document quotes, from the study's own results.

``tests/unit/test_method_evidence.py`` checks that every value in a section's measured table is
what its artefacts produce.  This is the other half: after a regeneration, the table is rewritten
from the artefacts rather than retyped, so the gate is something the numbers satisfy by
construction instead of something a person has to chase.

Run it explicitly after ``regenerate.py``::

    python -m tests.studies.evidence.document

Deliberately not called by the regeneration script.  Editing a reader-facing document is a change
a reviewer should see requested, not a side effect of producing a CSV.
"""

from __future__ import annotations

import argparse
import re

from tests.studies.evidence.claims import load, value
from tests.studies.evidence.manifest import write_lines
from tests.studies.evidence.registry import StudyRecord, registered

MEASURED_COLUMNS = ("quantity", "value", "source")

_ROW = re.compile(
    r"^\|\s*(?P<quantity>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|\s*(?P<source>[^|]*?)\s*\|$"
)


def render(computed: float) -> str:
    """How precisely to quote a value, given nothing but the value.

    Counts are counts.  Everything else gets four decimals, or six where four would round a real
    quantity to nothing -- a bound of 0.0002 printed as ``0.0000`` is a worse claim than a long one.

    The six-decimal rung has a floor of its own, and a study eventually reached it: paired
    agreement at solver precision put ``4.45e-08`` and ``4.72e-09`` in a table that printed both
    as ``0.000000``, which a literal zero also satisfies.  A figure that small is quoted in
    scientific notation instead, so the cell carries the magnitude rather than the absence of
    one.  ``claims.matches`` reads those to *significant* digits, which is the precision they
    were actually printed to.
    """
    if computed == int(computed):
        return str(int(computed))
    if abs(computed) < 1e-6:
        return f"{computed:.3e}"
    if abs(computed) < 0.001:
        return f"{computed:.6f}"
    return f"{computed:.4f}"


def _section(lines: list[str], anchor: str, document: object) -> tuple[int, int]:
    """The half-open line range of the level-two section ``anchor`` names."""

    def matches(line: str) -> bool:
        if not line.startswith("## "):
            return False
        heading = line[3:].strip()
        kept = "".join(
            character
            for character in heading.casefold()
            if character.isalnum() or character in " -_"
        )
        return heading.casefold() == anchor.casefold() or kept.strip().replace(" ", "-") == anchor

    start = next((index for index, line in enumerate(lines) if matches(line)), None)
    if start is None:
        raise LookupError(f"{document} has no level-two section {anchor!r}")
    stop = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start=start + 1)
            if line.startswith("## ")
        ),
        len(lines),
    )
    return start, stop


def fill(record: StudyRecord) -> list[str]:
    """Rewrite the measured table's value column in place.  Returns the rows that changed."""
    document = record.document_path
    lines = document.read_text(encoding="utf-8").splitlines(keepends=True)
    # Only this study's section.  The three studies share one document now, and searching
    # the whole file finds the first measured table whichever record asked -- so filling the
    # second study resolved the *first* study's quantity names against the second's
    # artefacts, which fails outright on a name the second does not report.  This is the
    # same rule ``tests.documents.pipe_table`` applies for the gate that reads these tables.
    start, stop = _section(lines, record.anchor, document)
    header = next(
        (
            index
            for index in range(start, stop)
            if [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            == list(MEASURED_COLUMNS)
        ),
        None,
    )
    if header is None:
        raise LookupError(f"{document} has no measured-values table under {record.anchor!r}")

    data = load(record)
    changed: list[str] = []
    for index in range(header + 2, stop):
        match = _ROW.match(lines[index].rstrip("\n"))
        if match is None:
            break
        quantity = match.group("quantity").strip("`")
        rendered = render(value(record, quantity, data))
        if rendered != match.group("value"):
            changed.append(f"{quantity}: {match.group('value')} -> {rendered}")
        lines[index] = f"| {match.group('quantity')} | {rendered} | {match.group('source')} |" + (
            "\n" if lines[index].endswith("\n") else ""
        )
    write_lines(document, "".join(lines))
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", help="only this study; default is every registered one")
    arguments = parser.parse_args()
    for record in registered():
        if arguments.slug and record.slug != arguments.slug:
            continue
        changed = fill(record)
        print(f"{record.slug}: {len(changed)} value(s) updated in {record.document}")
        for line in changed:
            print(f"  {line}")


if __name__ == "__main__":
    main()

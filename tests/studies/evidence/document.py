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
from tests.studies.evidence.registry import StudyRecord, registered

MEASURED_COLUMNS = ("quantity", "value", "source")

_ROW = re.compile(
    r"^\|\s*(?P<quantity>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|\s*(?P<source>[^|]*?)\s*\|$"
)


def render(computed: float) -> str:
    """How precisely to quote a value, given nothing but the value.

    Counts are counts.  Everything else gets four decimals, or six where four would round a real
    quantity to nothing -- a bound of 0.0002 printed as ``0.0000`` is a worse claim than a long one.
    """
    if computed == int(computed):
        return str(int(computed))
    if abs(computed) < 0.001:
        return f"{computed:.6f}"
    return f"{computed:.4f}"


def fill(record: StudyRecord) -> list[str]:
    """Rewrite the measured table's value column in place.  Returns the rows that changed."""
    document = record.document_path
    lines = document.read_text(encoding="utf-8").splitlines(keepends=True)
    header = next(
        (
            index
            for index, line in enumerate(lines)
            if [cell.strip() for cell in line.strip().strip("|").split("|")]
            == list(MEASURED_COLUMNS)
        ),
        None,
    )
    if header is None:
        raise LookupError(f"{document} has no measured-values table")

    data = load(record)
    changed: list[str] = []
    for index in range(header + 2, len(lines)):
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
    document.write_text("".join(lines), encoding="utf-8")
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

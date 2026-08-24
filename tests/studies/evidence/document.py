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
from collections.abc import Sequence
from typing import Any

import pandas as pd

from tests.studies.evidence import descriptions
from tests.studies.evidence.claims import load, value
from tests.studies.evidence.manifest import write_lines
from tests.studies.evidence.registry import StudyRecord, registered

MEASURED_COLUMNS = ("quantity", "value", "source")

#: Header cells of each generated table, in the order they are rendered.
ACCURACY_COLUMNS = (
    "law",
    "estimand",
    "what was tested",
    "implementation",
    "bias (99% interval)",
    "coverage",
    "SE ratio",
    "result",
)
AGREEMENT_COLUMNS = (
    "law",
    "estimand",
    "what was compared",
    "paired difference",
    "share of margin used",
    "RMSE ratio bound",
    "coverage difference",
    "result",
)
PROPERTY_COLUMNS = (
    "property",
    "cell",
    "role",
    "what was tested",
    "what must hold",
    "measured",
    "result",
)

#: Sentinel comments delimiting a generated block, so ``fill`` can replace it whole.
_OPEN = "<!-- generated: {name} -->"
_CLOSE = "<!-- /generated -->"

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


def _verdict(passed: object) -> str:
    """How a committed ``passed`` column reads in a published table."""
    return "pass" if bool(passed) else "**fail**"


def _interval(lower: float, upper: float) -> str:
    return f"{render(float(lower))} to {render(float(upper))}"


def _table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    return [header, rule, *("| " + " | ".join(row) + " |" for row in rows)]


def accuracy_table(record: StudyRecord, data: dict[str, pd.DataFrame]) -> list[str]:
    """One row per implementation-estimand test against the law's known truth."""
    frame = data["performance"].sort_values(["scenario", "estimand", "implementation"])
    rows = [
        (
            descriptions.scenario(str(row.scenario)),
            f"`{row.estimand}`",
            descriptions.estimand(str(row.estimand)),
            descriptions.implementation(str(row.implementation)),
            _interval(row.bias_ci_lower, row.bias_ci_upper),
            render(float(row.coverage)),
            render(float(row.se_ratio)),
            _verdict(row.passed),
        )
        for row in frame.itertuples()
    ]
    return _table(ACCURACY_COLUMNS, rows)


def agreement_table(record: StudyRecord, data: dict[str, pd.DataFrame]) -> list[str]:
    """One row per paired comparison with the canonical implementation."""
    frame = data["equivalence"].sort_values(["scenario", "estimand"])
    rows = [
        (
            descriptions.scenario(str(row.scenario)),
            f"`{row.estimand}`",
            descriptions.estimand(str(row.estimand)),
            render(float(row.mean_difference)),
            render(float(row.margin_utilization)),
            render(float(row.rmse_ratio_upper)),
            render(float(row.coverage_difference)),
            _verdict(row.passed),
        )
        for row in frame.itertuples()
    ]
    return _table(AGREEMENT_COLUMNS, rows)


def property_table(record: StudyRecord, data: dict[str, pd.DataFrame]) -> list[str]:
    """One row per property cell, with the endpoint its own verdict was read from.

    The measured column is deliberately the endpoint the cell is judged on rather than a fixed
    column: a bias-gated cell is judged on its bias interval and a calibration cell on its SE
    ratio, and printing the same column for both would show a number that decided nothing.
    """
    frame = data["properties"].sort_values(["property", "cell"])
    rows = []
    for row in frame.itertuples():
        tested, required = descriptions.cell(
            str(row.property),
            str(row.cell),
            exact_efficiency=_has_exact_efficiency(row),
            role=str(row.role),
        )
        rows.append(
            (
                f"`{row.property}`",
                f"`{row.cell}`",
                str(row.role),
                tested,
                required,
                _measured(row),
                _verdict(row.passed),
            )
        )
    return _table(PROPERTY_COLUMNS, rows)


def _measured(row: Any) -> str:
    """The endpoint a cell's own verdict was read from, named and valued."""
    family = str(row.property)
    if family in _BIAS_GATED:
        return f"bias {_interval(row.bias_ci_lower, row.bias_ci_upper)}, margin {render(float(row.bias_margin))}"
    if family == "root_n_rate":
        return f"slope {_interval(row.slope_ci_lower, row.slope_ci_upper)}"
    if family in {"type_i_error", "power"}:
        return (
            f"rejection {render(float(row.rejection_rate))}, "
            f"{_interval(row.rejection_ci_lower, row.rejection_ci_upper)}"
        )
    if family == "root_n_and_efficiency":
        return (
            f"bias {render(float(row.bias))}, coverage {_interval(row.coverage_ci_lower, row.coverage_ci_upper)}, "
            f"SE ratio {render(float(row.se_ratio))}"
        )
    if family == "interval_calibration":
        measured = (
            f"coverage {_interval(row.coverage_ci_lower, row.coverage_ci_upper)}, "
            f"SE ratio {_interval(row.se_ratio_ci_lower, row.se_ratio_ci_upper)}"
        )
        if _has_exact_efficiency(row):
            measured += (
                ", empirical efficiency ratio "
                f"{_interval(row.efficiency_empirical_ci_lower, row.efficiency_empirical_ci_upper)}, "
                "reported efficiency ratio "
                f"{_interval(row.efficiency_reported_ci_lower, row.efficiency_reported_ci_upper)}"
            )
        return measured
    return f"SE ratio {_interval(row.se_ratio_ci_lower, row.se_ratio_ci_upper)}"


def _has_exact_efficiency(row: Any) -> bool:
    """Whether this result compares against an independently computed efficiency bound.

    Read per row, while the gate in ``tests/unit/test_method_evidence.py`` is per study.  A
    study whose bound covered some calibration cells and not others would render mixed
    language and still pass that gate; ``efficiency_bounds`` is keyed by plan and so covers
    every cell of a plan or none of them, which is what keeps the two consistent.
    """
    return hasattr(row, "efficiency_empirical_ci_lower") and pd.notna(
        row.efficiency_empirical_ci_lower
    )


#: Families whose per-cell verdict is the bias claim.  Mirrors the classification the gate in
#: ``tests/unit/test_method_evidence.py`` applies; a family missing from both fails there.
_BIAS_GATED = frozenset(
    {
        "double_robustness",
        "robustness_contract",
        "selector_necessity",
        "survival_recursion_necessity",
        "targeting_necessity",
    }
)

#: The generated blocks a study section carries, and the renderer for each.
GENERATED = {
    "accuracy": accuracy_table,
    "agreement": agreement_table,
    "properties": property_table,
}


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


def _replace_block(
    lines: list[str], name: str, rendered: list[str], bounds: tuple[int, int]
) -> tuple[bool, tuple[int, int]]:
    """Swap one sentinel-delimited block for freshly rendered table lines."""
    start, stop = bounds
    opening = _OPEN.format(name=name)
    head = next((index for index in range(start, stop) if lines[index].strip() == opening), None)
    if head is None:
        return False, bounds
    tail = next((index for index in range(head + 1, stop) if lines[index].strip() == _CLOSE), None)
    if tail is None:
        raise LookupError(f"{opening} is never closed by {_CLOSE}")
    replacement = [line + "\n" for line in rendered]
    changed = lines[head + 1 : tail] != replacement
    lines[head + 1 : tail] = replacement
    return changed, (start, stop + len(replacement) - (tail - head - 1))


def fill(record: StudyRecord) -> list[str]:
    """Rewrite a study's generated tables and its measured values.  Returns what changed."""
    document = record.document_path
    lines = document.read_text(encoding="utf-8").splitlines(keepends=True)
    # Only this study's section.  The studies share one document, and searching the whole file
    # finds the first measured table whichever record asked -- so filling the second study
    # resolved the *first* study's quantity names against the second's artefacts, which fails
    # outright on a name the second does not report.  This is the same rule
    # ``tests.documents.pipe_table`` applies for the gate that reads these tables.
    bounds = _section(lines, record.anchor, document)
    data = load(record)
    changed: list[str] = []

    # Generated tables first.  Each replacement moves every later line, so the section bounds
    # are carried forward rather than recomputed against stale indices.
    for name, renderer in GENERATED.items():
        rendered = renderer(record, data)
        if name == "agreement" and record.reference is None:
            rendered = []
        moved, bounds = _replace_block(lines, name, rendered, bounds)
        if moved:
            changed.append(f"{name} table: {len(rendered)} line(s)")

    start, stop = bounds
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

    for index in range(header + 2, stop):
        match = _ROW.match(lines[index].rstrip("\n"))
        if match is None:
            break
        quantity = match.group("quantity").strip("`")
        rendered_value = render(value(record, quantity, data))
        if rendered_value != match.group("value"):
            changed.append(f"{quantity}: {match.group('value')} -> {rendered_value}")
        lines[index] = (
            f"| {match.group('quantity')} | {rendered_value} | {match.group('source')} |"
            + ("\n" if lines[index].endswith("\n") else "")
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

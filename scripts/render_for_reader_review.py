"""Render a documentation page for the reader review, and measure it.

``docs/development/writing-for-readers.md`` describes the reader review. This script is its
mechanical part. It reads a notebook or a Markdown page and writes two files:

``<stem>.reader.md``
    The reader's view. Every markdown cell, code cell, and stored output in order, with each
    cell id shown and each Details dropdown marked. The persona reviewer reads only this file.

``<stem>.report.md``
    The measures of the guide's mechanical report, with a pass or fail for the gated ones.

The exit status is 1 when a gated measure fails, and 0 otherwise. The script needs no kernel and
no network. It reads the stored outputs and never executes a cell.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The main-path word budget of the guide.
DEFAULT_BUDGET = 2000

#: Pages that carry two or three analyses get a larger budget.
BUDGETS = {
    "interventions": 2600,
    "sequential-offers": 2600,
    "sequential-pricing": 2600,
}

#: The words of the guide's "Library words" table, with the pattern that finds each one. The
#: report lists the first use of each in prose, so a reviewer can check for a plain definition.
TERMS: dict[str, str] = {
    "estimand": r"\bestimands?\b",
    "protocol": r"\bprotocols?\b|\bStudyProtocol\b",
    "fingerprint": r"\bfingerprints?\b",
    "design": r"\bdesigns?\b",
    "identify": r"\bidentif(?:y|ies|ied|ication|iable)\b",
    "nuisance": r"\bnuisances?\b",
    "targeting": r"\btarget(?:ing|ed) step\b|\btargeting\b",
    "fluctuation": r"\bfluctuations?\b",
    "clever covariate": r"\bclever covariates?\b",
    "influence curve": r"\binfluence (?:curves?|functions?)\b",
    "positivity": r"\bpositivity\b",
    "overlap": r"\boverlap\b",
    "support report": r"\bsupport reports?\b",
    "double robustness": r"\bdoubl[ey][ -]robust(?:ness)?\b",
    "cross-fitting": r"\bcross-fit(?:ting|ted)?\b",
    "CV-TMLE": r"\bCV-TMLE\b",
    "folds": r"\bfolds?\b",
    "truncation": r"\btruncat(?:ion|ed|e)\b",
    "q_bounds": r"\bq_bounds\b",
    "refute": r"\brefut(?:e|es|ed|ation|ations)\b",
    "placebo": r"\bplacebos?\b",
    "random_common_cause": r"\brandom_common_cause\b",
    "subset": r"\bsubset\b",
    "plasmode": r"\bplasmodes?\b",
    "generated outcome": r"\bgenerated outcomes?\b",
    "rung": r"\brungs?\b",
    "L0, L1, L2": r"\bL[012]\b",
    "verdict": r"\bverdicts?\b",
    "registry": r"\bregistry\b",
    "VERDICT_EVIDENCE": r"\bVERDICT_EVIDENCE\b",
    "Covered:": r"\bCovered:",
    "sensitivity analysis": r"\bsensitivity analys[ie]s\b",
    "robustness value": r"\brobustness values?\b",
    "cf_y, cf_d": r"\bcf_[yd]\b",
    "nu squared": r"\\nu\^?2|\\nu\^\{2\}",
    "Riesz representer": r"\bRiesz\b",
    "benchmark": r"\bbenchmark(?:s|ed|ing)?\b",
    "E-value": r"\bE-values?\b",
    "registered study": r"\bregistered (?:study|studies)\b",
    "review probe": r"\breview probes?\b",
    "method evidence": r"\bmethod evidence\b|\bmethod-evidence\b",
    "experimental": r"\bexperimental\b",
    "ATE": r"\bATE\b",
    "ATT": r"\bATT\b",
    "ATC": r"\bATC\b",
    "MSM": r"\bMSMs?\b",
    "LTMLE": r"\bLTMLE\b",
    "C-TMLE": r"\bC-TMLE\b",
    "OAT": r"\bOAT\b",
    "MTP": r"\bMTPs?\b",
    "IPSI": r"\bIPSI\b",
}

#: Acronyms and API tokens keep their case. Every other term matches in any case, so a word that
#: opens a sentence or a heading still counts as a use.
CASE_SENSITIVE = frozenset(
    {
        "StudyProtocol",
        "q_bounds",
        "random_common_cause",
        "L0, L1, L2",
        "VERDICT_EVIDENCE",
        "Covered:",
        "cf_y, cf_d",
        "nu squared",
        "CV-TMLE",
        "ATE",
        "ATT",
        "ATC",
        "MSM",
        "LTMLE",
        "C-TMLE",
        "OAT",
        "MTP",
        "IPSI",
    }
)

TERM_PATTERNS = {
    term: re.compile(pattern, 0 if term in CASE_SENSITIVE else re.IGNORECASE)
    for term, pattern in TERMS.items()
}

BOLD_OPENER = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?\*\*(?P<label>[^*\n]+?)\*\*")
BOILERPLATE = "what this output tells you"
STEP_NUMBER = re.compile(r"\bsteps?\s+\d+\b(?![.,]\d)", re.IGNORECASE)
CITED_STEP = re.compile(r"\b(?:Algorithm|Section)\s+[\d.]+,\s*$")
NUMBERED_HEADING = re.compile(r"^#{1,6}\s+\d+[.)]\s")
HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
GO_DEEPER = re.compile(r"\bgo deeper\b", re.IGNORECASE)
ON_THIS_DRAW = re.compile(r"\bon this draw\b", re.IGNORECASE)
NOT_CONTRAST = re.compile(r",\s+not\s+(?!only\b|just\b)", re.IGNORECASE)
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
DROPDOWN_OPEN = re.compile(r"^\s*(?P<fence>:{3,}|`{3,}|~{3,})\{dropdown\}\s*(?P<title>.*)$")
DIRECTIVE_FENCE = re.compile(r"^\s*(?P<fence>:{3,})(?:\{[\w-]+\}.*)?\s*$")
CODE_FENCE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'%.,-]*")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass
class Segment:
    """A run of lines of one kind inside one markdown cell."""

    kind: str  # "prose", "code", "math", or "dropdown"
    lines: list[str] = field(default_factory=list)
    title: str = ""


@dataclass
class Cell:
    """One notebook cell, or one heading section of a Markdown page."""

    cell_id: str
    cell_type: str  # "markdown" or "code"
    source: str
    outputs: list[str] = field(default_factory=list)


def _closes(line: str, fence: str) -> bool:
    """Whether ``line`` closes a fence opened with ``fence``."""
    stripped = line.strip()
    return (
        len(stripped) >= len(fence)
        and set(stripped) == {fence[0]}
        and (fence[0] != ":" or len(stripped) == len(fence))
    )


def segments(markdown: str) -> list[Segment]:
    """Split markdown into prose, code, display math, and dropdown segments.

    A dropdown holds everything up to its closing fence, including any code inside it. Other
    directive fences, such as ``:::{note}``, open and close prose; the fence lines themselves
    carry no prose.
    """
    out: list[Segment] = [Segment("prose")]
    state = "prose"
    fence = ""
    for line in markdown.splitlines():
        if state == "prose":
            dropdown = DROPDOWN_OPEN.match(line)
            if dropdown:
                fence = dropdown["fence"]
                out.append(Segment("dropdown", title=dropdown["title"].strip()))
                state = "dropdown"
                continue
            code = CODE_FENCE.match(line)
            if code:
                fence = code["fence"]
                out.append(Segment("code", [line]))
                state = "code"
                continue
            if line.strip().startswith("$$"):
                out.append(Segment("math", [line]))
                if "$$" in line.strip()[2:]:
                    out.append(Segment("prose"))  # the whole display sits on this line
                else:
                    state = "math"
                continue
            if DIRECTIVE_FENCE.match(line):
                continue
            out[-1].lines.append(line)
            continue
        if state == "dropdown":
            if _closes(line, fence):
                out.append(Segment("prose"))
                state = "prose"
            else:
                out[-1].lines.append(line)
            continue
        if state == "code":
            out[-1].lines.append(line)
            if _closes(line, fence):
                out.append(Segment("prose"))
                state = "prose"
            continue
        out[-1].lines.append(line)
        if "$$" in line:
            out.append(Segment("prose"))
            state = "prose"
    return [segment for segment in out if segment.lines or segment.kind == "dropdown"]


def prose_text(lines: list[str]) -> str:
    """The prose of ``lines``, with any code fences and display math removed."""
    kept: list[str] = []
    for segment in segments("\n".join(lines)):
        if segment.kind == "prose":
            kept.extend(segment.lines)
    return "\n".join(kept)


def word_count(text: str) -> int:
    """Words a reader reads. Markup characters and table rules do not count."""
    cleaned = re.sub(r"\]\([^)]*\)", "]", text)  # drop link targets
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return len(WORD.findall(cleaned))


def _output_text(output: dict[str, Any]) -> str:
    """The text a reader sees for one stored output."""
    kind = output.get("output_type")
    if kind == "stream":
        return ANSI.sub("", "".join(output.get("text", "")))
    if kind == "error":
        return f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
    data = output.get("data", {})
    parts: list[str] = []
    if any(key.startswith("image/") for key in data):
        parts.append("[figure]")
    if "text/plain" in data:
        parts.append(ANSI.sub("", "".join(data["text/plain"])))
    elif "text/html" in data:
        parts.append("[HTML output]")
    return "\n".join(parts)


def read_cells(path: Path) -> list[Cell]:
    """The cells of a notebook, or the heading sections of a Markdown page."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        cells: list[Cell] = []
        for number, raw in enumerate(json.loads(text)["cells"], start=1):
            source = "".join(raw.get("source", ""))
            cell_id = str(raw.get("id") or f"cell-{number}")
            outputs = [_output_text(output) for output in raw.get("outputs", [])]
            cell_type = "code" if raw["cell_type"] == "code" else "markdown"
            cells.append(Cell(cell_id, cell_type, source, [o for o in outputs if o.strip()]))
        return cells
    sections: list[Cell] = []
    current: list[str] = []
    start = 1
    in_fence = ""
    for number, line in enumerate(text.splitlines(), start=1):
        code = CODE_FENCE.match(line)
        if in_fence:
            if _closes(line, in_fence):
                in_fence = ""
        elif code:
            in_fence = code["fence"]
        elif HEADING.match(line) and current:
            sections.append(Cell(f"L{start}", "markdown", "\n".join(current)))
            current, start = [], number
        current.append(line)
    if current:
        sections.append(Cell(f"L{start}", "markdown", "\n".join(current)))
    return sections


def _fence_for(text: str) -> str:
    """A backtick fence longer than any run of backticks in ``text``."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def reader_view(path: Path, cells: list[Cell]) -> str:
    """The page as the persona reviewer reads it."""
    out = [f"# Reader's view of {path.name}", ""]
    for cell in cells:
        if cell.cell_type == "markdown":
            out.append(f"[cell {cell.cell_id}: text]")
            out.append("")
            for segment in segments(cell.source):
                if segment.kind == "dropdown":
                    out.append(f"[DETAILS, collapsed until the reader opens it: {segment.title}]")
                    out.extend(segment.lines)
                    out.append("[END OF DETAILS]")
                else:
                    out.extend(segment.lines)
            out.append("")
            continue
        fence = _fence_for(cell.source)
        out += [f"[cell {cell.cell_id}: code]", "", f"{fence}python", cell.source, fence, ""]
        for output in cell.outputs:
            fence = _fence_for(output)
            out += [
                f"[cell {cell.cell_id}: output]",
                "",
                f"{fence}text",
                output.rstrip(),
                fence,
                "",
            ]
    return "\n".join(out).rstrip() + "\n"


@dataclass
class Report:
    """The measures of the mechanical report."""

    main_words: int = 0
    budget: int = DEFAULT_BUDGET
    bold_openers: Counter[str] = field(default_factory=Counter)
    bold_cells: dict[str, list[str]] = field(default_factory=dict)
    step_numbers: list[tuple[str, str]] = field(default_factory=list)
    dropdowns: list[tuple[str, str, int]] = field(default_factory=list)
    first_terms: dict[str, tuple[str, str]] = field(default_factory=dict)
    tables: list[tuple[str, int, int]] = field(default_factory=list)
    on_this_draw: int = 0
    not_contrasts: int = 0

    @property
    def repeated_openers(self) -> list[str]:
        """Bold openers that fail: the boilerplate one, or any label used twice or more."""
        return sorted(
            label
            for label, count in self.bold_openers.items()
            if count > 1 or BOILERPLATE in label.lower()
        )

    @property
    def failures(self) -> list[str]:
        """The gated measures that fail."""
        failed: list[str] = []
        if self.repeated_openers:
            failed.append("repeated bold opener")
        if self.step_numbers:
            failed.append("step numbers")
        if self.main_words > self.budget:
            failed.append("main-path words")
        return failed


def _snippet(text: str, start: int, end: int) -> str:
    left = max(0, start - 40)
    return " ".join(text[left : end + 40].split())


def _tables(text: str) -> list[tuple[int, int]]:
    """``(data rows, columns)`` of each pipe table in ``text``."""
    lines = text.splitlines()
    found: list[tuple[int, int]] = []
    index = 0
    while index < len(lines) - 1:
        header, rule = lines[index], lines[index + 1]
        if "|" in header and "|" in rule and TABLE_SEPARATOR.match(rule):
            columns = len(header.strip().strip("|").split("|"))
            rows = 0
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows += 1
                index += 1
            found.append((rows, columns))
            continue
        index += 1
    return found


def _step_numbers(report: Report, cell_id: str, text: str) -> None:
    """Record each step number in ``text``, except a step that an algorithm citation names."""
    for match in STEP_NUMBER.finditer(text):
        if CITED_STEP.search(text[: match.start()]):
            continue
        report.step_numbers.append((cell_id, _snippet(text, *match.span())))


def measure(cells: list[Cell], budget: int) -> Report:
    """Measure the markdown of ``cells`` against the guide's mechanical report."""
    report = Report(budget=budget)
    main_path = True
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        for segment in segments(cell.source):
            if segment.kind == "dropdown":
                words = word_count(prose_text(segment.lines))
                report.dropdowns.append((cell.cell_id, segment.title, words))
                _step_numbers(report, cell.cell_id, "\n".join(segment.lines))
                continue
            if segment.kind != "prose":
                continue
            for line in segment.lines:
                heading = HEADING.match(line)
                if heading and GO_DEEPER.search(heading["title"]):
                    main_path = False
                if NUMBERED_HEADING.match(line):
                    report.step_numbers.append((cell.cell_id, line.strip()))
                if main_path:
                    report.main_words += word_count(line)
            text = "\n".join(segment.lines)
            # A heading line ends a paragraph even with no blank line after it.
            spaced = re.sub(r"^(#{1,6}\s.*)$", r"\1\n", text, flags=re.MULTILINE)
            for paragraph in re.split(r"\n\s*\n|\n(?=\s*(?:[-*+]|\d+[.)])\s)", spaced):
                opener = BOLD_OPENER.match(paragraph)
                if opener:
                    label = opener["label"].strip()
                    report.bold_openers[label] += 1
                    report.bold_cells.setdefault(label, []).append(cell.cell_id)
            _step_numbers(report, cell.cell_id, text)
            for term, pattern in TERM_PATTERNS.items():
                if term in report.first_terms:
                    continue
                found = pattern.search(text)
                if found:
                    report.first_terms[term] = (cell.cell_id, _snippet(text, *found.span()))
            for rows, columns in _tables(text):
                report.tables.append((cell.cell_id, rows, columns))
            report.on_this_draw += len(ON_THIS_DRAW.findall(text))
            report.not_contrasts += len(NOT_CONTRAST.findall(text))
    return report


def _cell(value: str) -> str:
    return value.replace("|", "\\|")


def report_text(path: Path, report: Report) -> str:
    """The report as Markdown."""
    status = "FAIL: " + ", ".join(report.failures) if report.failures else "PASS"
    out = [
        f"# Mechanical report for {path.name}",
        "",
        f"Gated measures: {status}.",
        "",
        "| measure | value | pass |",
        "| --- | --- | --- |",
        f"| repeated bold openers | {len(report.repeated_openers)} | "
        f"{'no' if report.repeated_openers else 'yes'} |",
        f"| step numbers | {len(report.step_numbers)} | {'no' if report.step_numbers else 'yes'} |",
        f"| main-path words | {report.main_words} of {report.budget} | "
        f"{'no' if report.main_words > report.budget else 'yes'} |",
        f"| dropdowns | {len(report.dropdowns)} | reviewed |",
        f"| tables with fewer than three rows | "
        f"{sum(rows < 3 for _, rows, _ in report.tables)} | needs a reason |",
        f'| "on this draw" | {report.on_this_draw} | judged as a tell |',
        f'| "X, not Y" | {report.not_contrasts} | judged as a tell |',
        "",
        "## Bold openers",
        "",
    ]
    if report.bold_openers:
        out += ["| label | uses | cells |", "| --- | --- | --- |"]
        for label, count in report.bold_openers.most_common():
            cells = ", ".join(report.bold_cells[label])
            out.append(f"| {_cell(label)} | {count} | {cells} |")
    else:
        out.append("None.")
    out += ["", "## Step numbers", ""]
    if report.step_numbers:
        out += [f"- {cell}: {_cell(text)}" for cell, text in report.step_numbers]
    else:
        out.append("None.")
    out += ["", "## Dropdowns", ""]
    if report.dropdowns:
        out += ["| cell | title | words |", "| --- | --- | --- |"]
        out += [f"| {c} | {_cell(t)} | {w} |" for c, t, w in report.dropdowns]
    else:
        out.append("None.")
    out += ["", "## First use of each library word", ""]
    if report.first_terms:
        out += ["| term | cell | context |", "| --- | --- | --- |"]
        for term, (cell, snippet) in report.first_terms.items():
            out.append(f"| {_cell(term)} | {cell} | {_cell(snippet)} |")
    else:
        out.append("None.")
    out += ["", "## Tables in prose", ""]
    if report.tables:
        out += ["| cell | rows | columns |", "| --- | --- | --- |"]
        out += [f"| {c} | {r} | {k} |" for c, r, k in report.tables]
    else:
        out.append("None.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the reader's view and the report, and return 1 when a gated measure fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("page", type=Path, help="a notebook or a Markdown page")
    parser.add_argument("--output", type=Path, required=True, help="the directory to write into")
    parser.add_argument("--budget", type=int, help="the main-path word budget")
    args = parser.parse_args(argv)
    page: Path = args.page
    budget = args.budget or BUDGETS.get(page.stem, DEFAULT_BUDGET)
    cells = read_cells(page)
    report = measure(cells, budget)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{page.stem}.reader.md").write_text(reader_view(page, cells), encoding="utf-8")
    (output / f"{page.stem}.report.md").write_text(report_text(page, report), encoding="utf-8")
    status = "FAIL: " + ", ".join(report.failures) if report.failures else "PASS"
    print(f"{page}: {status}; {report.main_words} main-path words of {report.budget}")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())

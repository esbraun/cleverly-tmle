"""The mechanical report of the reader review measures what the guide says it measures.

``docs/development/writing-for-readers.md`` gates three measures and reports the rest. Each test
pairs a page that should pass with one change that should fail, so a measure that silently stops
counting fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.render_for_reader_review import main, measure, read_cells, reader_view, segments

CLEAN = [
    (
        "glance",
        "markdown",
        "# Did the offer work?\n\n## At a glance\n\nThe offer raised the score.",
    ),
    ("fit", "code", "print('estimate 0.26')"),
    ("fit-reading", "markdown", "The estimate is 0.26, and its interval excludes zero."),
    (
        "details",
        "markdown",
        ":::{dropdown} Details\nThe efficient influence curve gives the variance.\n:::\n",
    ),
    ("deeper", "markdown", "## Go deeper\n\nThe nuisance models " + "word " * 50),
]


def _notebook(path: Path, cells: list[tuple[str, str, str]]) -> Path:
    raw = []
    for cell_id, cell_type, source in cells:
        cell: dict[str, object] = {
            "id": cell_id,
            "cell_type": cell_type,
            "metadata": {},
            "source": source,
        }
        if cell_type == "code":
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "0.26\n"}]
            cell["execution_count"] = 1
        raw.append(cell)
    path.write_text(json.dumps({"cells": raw, "metadata": {}, "nbformat": 4}), encoding="utf-8")
    return path


def test_a_clean_page_passes(tmp_path: Path) -> None:
    page = _notebook(tmp_path / "page.ipynb", CLEAN)
    report = measure(read_cells(page), budget=2000)
    assert report.failures == []
    assert main([str(page), "--output", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "page.reader.md").exists()
    assert (tmp_path / "out" / "page.report.md").exists()


def test_the_boilerplate_opener_fails_even_once(tmp_path: Path) -> None:
    cells = [*CLEAN, ("x", "markdown", "**What this output tells you.** The score rose.")]
    report = measure(read_cells(_notebook(tmp_path / "p.ipynb", cells)), budget=2000)
    assert report.failures == ["repeated bold opener"]


def test_a_bold_label_fails_only_when_repeated(tmp_path: Path) -> None:
    once = [*CLEAN, ("x", "markdown", "- **Placebo.** The estimate is near zero.")]
    assert measure(read_cells(_notebook(tmp_path / "a.ipynb", once)), 2000).failures == []
    twice = [*once, ("y", "markdown", "- **Placebo.** Again.")]
    report = measure(read_cells(_notebook(tmp_path / "b.ipynb", twice)), 2000)
    assert report.repeated_openers == ["Placebo."]


def test_step_numbers_fail_in_prose_and_in_numbered_headings(tmp_path: Path) -> None:
    for text in ("## Step 5: identify", "As Step 7 shows, the fit moves.", "## 1. Setup"):
        cells = [*CLEAN, ("x", "markdown", text)]
        report = measure(read_cells(_notebook(tmp_path / "p.ipynb", cells)), budget=2000)
        assert report.failures == ["step numbers"], text


def test_the_budget_counts_only_the_main_path(tmp_path: Path) -> None:
    page = _notebook(tmp_path / "p.ipynb", CLEAN)
    report = measure(read_cells(page), budget=2000)
    # The 50 words after "Go deeper" and the dropdown's words stay out of the main path.
    assert report.main_words < 30
    assert report.dropdowns == [("details", "Details", 7)]
    assert measure(read_cells(page), budget=report.main_words - 1).failures == ["main-path words"]
    assert main([str(page), "--output", str(tmp_path / "o"), "--budget", "5"]) == 1


def test_terms_report_their_first_use_outside_dropdowns(tmp_path: Path) -> None:
    report = measure(read_cells(_notebook(tmp_path / "p.ipynb", CLEAN)), budget=2000)
    # "influence curve" appears only in the dropdown, so it is not a prose use.
    assert "influence curve" not in report.first_terms
    assert report.first_terms["nuisance"][0] == "deeper"


def test_tables_report_their_rows_and_columns(tmp_path: Path) -> None:
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    cells = [*CLEAN, ("t", "markdown", table)]
    report = measure(read_cells(_notebook(tmp_path / "p.ipynb", cells)), budget=2000)
    assert report.tables == [("t", 1, 2)]


def test_the_reader_view_shows_ids_outputs_and_dropdowns(tmp_path: Path) -> None:
    page = _notebook(tmp_path / "p.ipynb", CLEAN)
    view = reader_view(page, read_cells(page))
    assert "[cell fit: code]" in view
    assert "[cell fit: output]" in view
    assert "[DETAILS, collapsed until the reader opens it: Details]" in view
    assert ":::{dropdown}" not in view


def test_code_fences_hold_no_prose() -> None:
    kinds = [s.kind for s in segments("Text.\n```python\nStep 3\n```\nMore text.")]
    assert kinds == ["prose", "code", "prose"]


def test_a_markdown_page_splits_at_headings(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text("# Title\n\nIntro.\n\n```bash\n# not a heading\n```\n\n## Next\n\nText.\n")
    assert [cell.cell_id for cell in read_cells(page)] == ["L1", "L9"]

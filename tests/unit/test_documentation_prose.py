r"""Every prose finding carries a recorded judgment.  Not: every finding is gone.

**Read the distinction before changing anything here.**  Nothing in this module asserts that the
documentation is clean.  It asserts that somebody looked.  A finding with
``accepted: <reason>`` beside it in ``tests/prose-report.md`` passes; the same finding with an
empty disposition fails; and the failure is cleared either by changing the sentence or by writing
down why it stays.

**Why it is built this way**, because the obvious version was tried and did damage.  A Vale rule
failed the build on an em dash, and the sweep that followed optimized for the only thing a build
error rewards.  It stripped dashes mechanically and shipped six sentences with no predicate, the
"Five conditions" enumeration split after its fourth member, a dropped item from the
not-established list, five altered technical claims and four deleted evidence clauses.  Every one
of those was introduced *by* satisfying the rule.  A report cannot be gamed the same way, because
there is nothing to make green: the only way to clear a row is to make a decision and sign it.

So the invariant this module protects is narrow and worth stating plainly: **a rule may never
make a sentence worse.**  If a finding is wrong, the reason column says so and the sentence stays.

:mod:`tests.prose` carries the rules, the rules that were rejected and why, and the reason
sentence length reports without gating.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests import prose
from tests.documents import READER_FACING, ROOT

#: A floor on the set, so a glob that stopped matching cannot report success over nothing.  Well
#: under the count at the time of writing, because this is a guard and not a census.
_EXPECTED_DOCUMENTS = 40


def test_the_scan_reaches_something() -> None:
    """Every assertion below passes on an empty set, so the set is checked first."""
    assert len(READER_FACING) >= _EXPECTED_DOCUMENTS, (
        f"only {len(READER_FACING)} reader-facing sources found; check documents.READER_FACING"
    )
    suffixes = {path.suffix for path in READER_FACING}
    assert suffixes == {".md", ".rst", ".ipynb"}, (
        f"the scan reaches {sorted(suffixes)}; a format dropped here is a format nothing reads"
    )


def test_the_scanner_sees_what_it_should_and_ignores_what_it_should_not() -> None:
    """A deliberate mutation, because a scanner that matches nothing reports a clean tree.

    This is the control the Vale rule never had: its em-dash token could not match, so it
    reported ``0 errors`` over the whole repository on a configuration that could not fail.
    """
    caught = prose.prose_lines("An em dash — here.\nA flag -- here.\n")
    assert [n for n, line in caught if any(d in line for d in prose.DASHES)] == [1, 2]

    exempt = prose.prose_lines("```bash\ngit diff main -- docs\n```\nA span `--` and ``a -- b``.\n")
    assert not [line for _, line in exempt if any(d in line for d in prose.DASHES)]

    assert prose.TRANSITION_RE.search("The identity holds. Thus, the sign is fixed.")
    assert not prose.TRANSITION_RE.search("The estimator is robust, efficient and consistent.")
    assert not prose.INTENSIFIER_RE.search(
        "row leverage, a robust and efficient estimator, a significant power gain"
    ), "the intensifier list must never reach a statistical term of art"


def test_the_notebook_reader_skips_code_and_output() -> None:
    """Markdown cells in, code cells and stored outputs out.

    A rendered dataframe draws its rules with hyphens, and ``----------  --  ---`` in the TWINS
    notebook contains the spaced double hyphen this scanner looks for.
    """
    notebooks = [path for path in READER_FACING if path.suffix == ".ipynb"]
    assert notebooks, "no notebook in the reader-facing set; check documents.READER_FACING"
    for notebook in notebooks:
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        assert any(cell.get("cell_type") == "code" for cell in cells), notebook
        assert any(cell.get("outputs") for cell in cells), notebook
        text = prose.markdown_cells(notebook)
        assert "import " not in text, "a code cell reached the markdown-only text"


def test_every_finding_carries_a_judgment() -> None:
    """The gate.  It fails on an unread finding, never on the prose itself."""
    recorded = prose.dispositions()
    unjudged = [
        item
        for item in prose.scan()
        if not recorded.get(item.id, "").strip().startswith("accepted:")
    ]
    assert not unjudged, "\n".join(
        [
            f"{len(unjudged)} finding(s) with no recorded judgment. Read each one, then either",
            "change the sentence or run `python -m tests.prose --update` and write",
            "`accepted: <reason>` beside it. Do not reword a sentence merely to clear a row.",
            *(f"  {item.file}:{item.line}  {item.rule}  {item.excerpt}" for item in unjudged),
        ]
    )


def test_the_report_holds_no_stale_rows() -> None:
    """A row whose finding is gone must go too.

    Removing a row is not a prose edit, so this adds no pressure to reword anything.  Left
    alone, the accumulated rows would read as review that happened when it did not.
    """
    live = {item.id for item in prose.scan()}
    stale = sorted(set(prose.dispositions()) - live)
    assert not stale, (
        f"{len(stale)} row(s) in {prose.LEDGER.relative_to(ROOT).as_posix()} no longer match a "
        f"finding: {stale}. Run `python -m tests.prose --update`."
    )


def test_a_disposition_survives_a_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ledger's one promise: a recorded judgment is never silently lost.

    A first version looked the rendered id up with its backticks still attached, so every
    disposition vanished on the next ``--update``.  Nothing failed, which is what made it worth
    a test of its own.
    """
    ledger = tmp_path / "prose-report.md"
    monkeypatch.setattr(prose, "LEDGER", ledger)
    found = prose.scan()
    assert found, "this test needs at least one finding to carry"

    ledger.write_text(prose.render(found, {found[0].id: "accepted: a reason"}), encoding="utf-8")
    assert prose.dispositions()[found[0].id] == "accepted: a reason"

    ledger.write_text(prose.render(found, prose.dispositions()), encoding="utf-8")
    assert prose.dispositions()[found[0].id] == "accepted: a reason"


def test_no_assertion_here_demands_clean_prose() -> None:
    """The rule this module exists to keep, checked against this module.

    A future edit asserting the scan is empty would restore exactly the gate that broke six
    sentences.  Cheap to state, and the failure it prevents is on record.

    The pattern is assembled at runtime rather than written as a literal, because a literal
    would match this module's own source and fail on the first run.
    """
    source = (ROOT / "tests" / "unit" / "test_documentation_prose.py").read_text(encoding="utf-8")
    banned = re.findall("assert" + r"\s+not\s+" + r"(?:prose\.)?scan\(\)", source)
    assert not banned, (
        "this module must never assert the documentation is free of findings; gate the "
        "recorded judgment instead"
    )


@pytest.mark.parametrize("rule", prose.GATING_RULES)
def test_each_gating_rule_is_reachable(rule: str) -> None:
    """A rule nothing can trigger is a rule that reports a clean tree forever."""
    samples = {
        "clause-dash": "A clause — and another.",
        "empty-transition": "The identity holds. Thus, the sign is fixed.",
        "empty-intensifier": "This is a crucial step.",
        "paragraph-length": " ".join(f"Sentence number {n} here." for n in range(1, 9)),
    }
    lines = prose.prose_lines(samples[rule])
    if rule == "paragraph-length":
        found = [len(prose.sentences(text)) for _, text in prose.paragraphs(lines)]
        assert found and max(found) > prose.PARAGRAPH_SENTENCES
    else:
        matched = any(
            (any(d in line for d in prose.DASHES) if rule == "clause-dash" else False)
            or (prose.TRANSITION_RE.search(line) if rule == "empty-transition" else False)
            or (prose.INTENSIFIER_RE.search(line) if rule == "empty-intensifier" else False)
            for _, line in lines
        )
        assert matched, f"{rule} matched nothing in its own sample"

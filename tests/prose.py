r"""The prose report: what to look at in a reader-facing document, and what not to.

**This reports; it does not enforce.**  Nothing here fails a build because of the prose.  What
fails is a finding nobody has looked at, and the way to pass is to record a judgment in
``tests/prose-report.md``.  "I read this and the dash stays, because it quotes a source" is a
*passing* outcome.

**The reason is not a preference.**  When the dash rule was a build error, the sweep that
followed optimized for green: it stripped dashes mechanically and left six sentences without a
predicate, split the "Five conditions" enumeration after its fourth member, dropped a
not-established list item, altered five technical claims and deleted four evidence clauses.  The
rule was right.  Making it a build error turned a writer into a rewriter.  A report cannot be
satisfied mechanically, because there is nothing to make green.

**One engine.**  This replaces Vale.  Vale needed a binary, a version pinned in two places, and
docutils' ``rst2html`` for a single ``.rst`` file, and it could not read a notebook at all --
which is how an em dash survived a whole sweep inside ``twins-causal-inference.ipynb``.  Markdown,
``.rst`` and notebooks are one scanner here.

Rules that gate
---------------

Each produces a row in ``tests/prose-report.md`` that needs a disposition.

``clause-dash``
    ``—`` and ``' -- '``.  Exact, no false positives, and the clearest slop tell there is.  Tables
    are in scope; code is not.

``empty-transition``
    A sentence opening with ``Thus,``, ``Moreover,``, ``Notably,`` and the rest, or with "it is
    worth noting".  Exact, and it catches a regression that actually happened: the sweep above
    introduced six ``Thus,`` where ``main`` had none, each one standing where a dash had been.
    Removing a dash and writing ``Thus,`` moves the tell rather than the problem.

``empty-intensifier``
    A word that adds no verifiable information, which ``CLAUDE.md`` asks to remove.

``paragraph-length``
    More than six sentences.

Rules deliberately rejected, and why
------------------------------------

Measured against this corpus rather than against a generic style guide.

*A generic slop word list.*  ``leverage`` appears three times here and means the leverage of a
row.  ``robust`` is doubly robust, ``efficient`` is the efficient influence curve, and
``significant``, ``consistent`` and ``power`` are all terms of art.  :data:`INTENSIFIERS` is
therefore tiny and every entry was checked to have no statistical meaning.

*Passive voice.*  False-positive heavy, and identification prose legitimately uses it: "the
parameter is identified under" is the standard construction, not a lapse.

*Hedges and weasel words.*  Thirty-nine uses, every one deliberate.  This repository writes
"supported, not shown", "estimated, unmeasured" and "measured, not bounded".  Flagging a hedge
pushes an author toward overclaiming, which is the opposite of what the evidence standard asks
for.

*Readability scores.*  Meaningless over a corpus of displayed formulae.

*The STE vocabulary restriction.*  ``CLAUDE.md`` exempts statistical terms of art, and there are
165 uses of them.  A dictionary check would report the documentation as mostly non-compliant and
be right about nothing.

*Noun-cluster depth.*  Not derivable without a parser, and a wrong answer here reads as a demand
to reword something that is already clear.

Sentence length reports but does not gate
-----------------------------------------

Over 25 words, ``CLAUDE.md``'s limit for a description.  It stands at 350 across the tree once
code, tables, citations and API listings are out of scope, and the long ones are enumerations
rather than slop.  Three hundred and fifty rubber-stamped rows would bury the four rules that
carry signal, and gating it would push an author to split an enumeration mechanically, which is
the failure this module exists to prevent.  So it prints as a per-file summary and does not
appear in the ledger.  The 25-word standard still stands in ``CLAUDE.md``; what deserves a
reader's attention is not the same set as what touches the standard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.documents import READER_FACING, ROOT

__all__ = [
    "GATING_RULES",
    "LEDGER",
    "LEDGER_COLUMNS",
    "Finding",
    "findings_for",
    "long_sentences",
    "prose_lines",
    "scan",
]

#: The committed ledger.  It sits outside ``docs/`` on purpose: this scanner *scans* ``docs/``,
#: and the ledger quotes the text it flags, so a ledger under ``docs/`` would report its own
#: excerpts and never converge.
LEDGER = ROOT / "tests" / "prose-report.md"

LEDGER_COLUMNS = ("file", "rule", "id", "excerpt", "disposition")

#: The rules that produce a ledger row.  ``sentence-length`` is deliberately absent; see the
#: module docstring.
GATING_RULES = ("clause-dash", "empty-transition", "empty-intensifier", "paragraph-length")

#: ``CLAUDE.md``'s limit for a description.  Reported, never gated.
SENTENCE_WORDS = 25

#: ``CLAUDE.md``'s paragraph limit.
PARAGRAPH_SENTENCES = 6

#: The two dash forms.  The double hyphen is spaced because a bare ``--`` is a command-line
#: flag, a YAML document marker and a markdown table rule.
DASHES = ("—", " -- ")

#: Openers that assert a relation the sentence then does not supply.  Where the relation is
#: real, write it: "That is why", "The decomposition is therefore".
TRANSITIONS = (
    "Thus",
    "Moreover",
    "Furthermore",
    "Additionally",
    "Importantly",
    "Notably",
    "Overall",
    "Indeed",
    "Essentially",
    "Basically",
    "Ultimately",
    "In conclusion",
)

#: Deliberately tiny.  Every entry was checked against this corpus for a statistical reading and
#: has none.  See the module docstring for the six words that are *not* here and why.
INTENSIFIERS = (
    "crucial",
    "vital",
    "seamless",
    "delve",
    "myriad",
    "plethora",
    "utilize",
    "cutting-edge",
    "state-of-the-art",
)

FENCE = re.compile(r"^\s*(```|~~~)")
CODE_SPAN = re.compile(r"``[^`]+``|`[^`]+`")
TRANSITION_RE = re.compile(
    r"(?:^|(?<=[.!?])\s)\s*(" + "|".join(TRANSITIONS) + r")\s*,",
)
NOTING_RE = re.compile(r"[Ii]t (?:is|'s) (?:worth|important) (?:noting|to note)")
INTENSIFIER_RE = re.compile(r"\b(" + "|".join(INTENSIFIERS) + r")\b", re.IGNORECASE)

#: A sentence boundary: terminal punctuation, whitespace, then something that can open a
#: sentence.  Requiring the opener stops ``1e-12.`` and ``§3.2`` from splitting mid-number.
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(*`\[$])")

#: ``- Author (2014), *Title*`` and friends.  A citation is a record, not a sentence, and
#: measuring its length says nothing about anyone's writing.
CITATION = re.compile(r"^\s*[-*]\s+.{0,80}\(\d{4}[a-z]?\)")

#: An ``.rst`` directive opener, whose indented body is a listing of API names rather than
#: prose.  ``docs/api/object-index.rst`` is one such body of 364 dotted names, which reads as a
#: single 364-word sentence to anything that does not skip it.
DIRECTIVE = re.compile(r"^(\s*)\.\.\s+\S+::")

#: An ``.rst`` section underline, and a field like ``:toctree:``.
RST_FURNITURE = re.compile(r"^\s*([=~^-]{3,}\s*$|:\w[\w-]*:)")

#: A list marker.  Each item is its own unit: a numbered procedure of six steps is six things to
#: read, not one thirteen-sentence paragraph.
LIST_ITEM = re.compile(r"^\s*(\d+\.|[-*+])\s+")

LINK = re.compile(r"\[([^\]]*)\]\([^)\s]*\)")


def strip_directives(text: str) -> str:
    """Blank the body of every ``.rst`` directive, keeping line numbers intact."""
    lines = text.splitlines()
    out, skip_to = list(lines), None
    for index, line in enumerate(lines):
        if skip_to is not None:
            if not line.strip() or len(line) - len(line.lstrip()) > skip_to:
                out[index] = ""
                continue
            skip_to = None
        opener = DIRECTIVE.match(line)
        if opener:
            skip_to = len(opener.group(1))
            out[index] = ""
    return "\n".join(out)


@dataclass(frozen=True)
class Finding:
    """One thing to look at, in one file."""

    file: str
    rule: str
    line: int
    text: str

    @property
    def id(self) -> str:
        """Identity that survives a reflow.

        Deliberately *not* the line number.  A line number churns whenever anything above it
        changes, which would both bloat the ledger's diff and make a finding that merely moved
        look new.  Hashing the normalized text instead means a finding keeps its row when the
        paragraph is rewrapped, and loses it when the sentence itself is edited, which is
        exactly when it deserves a second reading.

        The **file** is in the hash, and leaving it out was a bug a planted-dash run caught: two
        files carrying the same sentence collapsed to one row, so judging the occurrence in one
        file silently accepted the unread occurrence in the other.  Two occurrences of one
        sentence *within* a file do still share a row, which is correct -- that is one sentence
        and one judgment -- and a path does not churn when a paragraph is rewrapped, so the
        stability the line number lacked is kept.
        """
        normalized = " ".join(self.text.split()).casefold()
        payload = f"{self.file}\x00{self.rule}\x00{normalized}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def excerpt(self) -> str:
        """A readable, table-safe fragment of the finding."""
        flat = " ".join(self.text.split()).replace("|", r"\|")
        return flat if len(flat) <= 60 else flat[:57].rstrip() + "..."


def prose_lines(text: str) -> list[tuple[int, str]]:
    """``(line number, text)`` for each line of prose, with code blanked.

    Fenced blocks are dropped whole and inline spans are blanked in place rather than deleted,
    so a column still means something on the line the reader sees.  Code is exempt in all its
    forms because each is a real source of these characters: ``git diff main -- docs`` is a
    command and ``--`` is how ``CLAUDE.md`` names the thing it bans.
    """
    out: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        out.append((number, CODE_SPAN.sub(lambda match: " " * len(match.group()), line)))
    return out


def markdown_cells(notebook: Path) -> str:
    """Every markdown cell of ``notebook``, joined, with code cells and outputs dropped.

    Not tidiness: a stored output draws its table rules with hyphens, and
    ``----------  ------  --  ---`` in the TWINS notebook contains the spaced double hyphen.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "\n".join(
        "".join(cell["source"]) for cell in cells if cell.get("cell_type") == "markdown"
    )


def read(document: Path) -> str:
    """The reader-facing text of ``document``, whatever its format."""
    if document.suffix == ".ipynb":
        return markdown_cells(document)
    text = document.read_text(encoding="utf-8")
    return strip_directives(text) if document.suffix == ".rst" else text


def paragraphs(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """``(line number of the first line, text)`` for each prose paragraph.

    Tables, quotes, headings, citations and ``.rst`` furniture are dropped: none of them is a
    paragraph, and counting sentences in a table row measures the table.  A list item opens a
    new paragraph, so a six-step numbered procedure is six units rather than one long one.
    """
    out: list[tuple[int, str]] = []
    start: int = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if buffer:
            out.append((start, " ".join(buffer)))
            buffer = []

    for number, line in [*lines, (0, "")]:
        skip = line.lstrip().startswith(("|", ">", "#")) or RST_FURNITURE.match(line)
        if not line.strip() or skip:
            flush()
            continue
        if LIST_ITEM.match(line):
            flush()
        if not buffer:
            start = number
        buffer.append(line.strip())
    return [(number, text) for number, text in out if not CITATION.match(text)]


def sentences(paragraph: str) -> list[str]:
    """The sentences of ``paragraph``, with links reduced to their text."""
    flat = LINK.sub(r"\1", paragraph)
    flat = re.sub(r"^\s*(\d+\.|[-*])\s+", "", flat)
    return [part.strip() for part in SENTENCE.split(flat) if part.strip()]


def words(sentence: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", sentence))


def findings_for(document: Path) -> list[Finding]:
    """Every gating-rule finding in ``document``.  Sentence length is not among them."""
    name = document.relative_to(ROOT).as_posix()
    lines = prose_lines(read(document))
    found: list[Finding] = []

    for number, line in lines:
        for dash in DASHES:
            if dash in line:
                found.append(Finding(name, "clause-dash", number, line.strip()))
                break
        for match in TRANSITION_RE.finditer(line):
            found.append(Finding(name, "empty-transition", number, match.group(1)))
        noting = NOTING_RE.search(line)
        if noting:
            found.append(Finding(name, "empty-transition", number, noting.group()))
        for match in INTENSIFIER_RE.finditer(line):
            found.append(Finding(name, "empty-intensifier", number, match.group(1)))

    for number, paragraph in paragraphs(lines):
        count = len(sentences(paragraph))
        if count > PARAGRAPH_SENTENCES:
            found.append(
                Finding(name, "paragraph-length", number, f"{count} sentences: {paragraph[:80]}")
            )
    return found


def long_sentences(document: Path) -> list[tuple[int, int, str]]:
    """``(line, word count, sentence)`` over :data:`SENTENCE_WORDS`.  Reported, never gated."""
    out: list[tuple[int, int, str]] = []
    for number, paragraph in paragraphs(prose_lines(read(document))):
        for sentence in sentences(paragraph):
            count = words(sentence)
            if count > SENTENCE_WORDS:
                out.append((number, count, sentence))
    return out


def scan(documents: list[Path] | None = None) -> list[Finding]:
    """Every finding across ``documents``, defaulting to the whole reader-facing set."""
    found: list[Finding] = []
    for document in documents if documents is not None else READER_FACING:
        found.extend(findings_for(document))
    return sorted(found, key=lambda item: (item.file, item.rule, item.id))


def dispositions() -> dict[str, str]:
    """``id -> disposition`` from the committed ledger, empty if it does not exist yet.

    The backtick strip is load-bearing.  The ledger renders the id as code so it is readable,
    and a first version of this looked the raw cell up against a bare id: every disposition was
    silently dropped on the next ``--update``, which is the one thing the ledger must never do.
    """
    if not LEDGER.exists():
        return {}
    from tests.documents import pipe_table

    return {
        row["id"].strip().strip("`"): row["disposition"].strip()
        for row in pipe_table(LEDGER, LEDGER_COLUMNS)
    }


def render(found: list[Finding], existing: dict[str, str]) -> str:
    """The ledger, with every disposition that still has a finding carried over."""
    header = (
        "# Prose report\n"
        "\n"
        "Generated by `python -m tests.prose --update`. Do not edit the first four columns.\n"
        "\n"
        "Each row is something to look at, not something that is wrong. Read the sentence, then\n"
        "either change it or write `accepted: <reason>` in the last column. A row with no reason\n"
        "fails `tests/unit/test_documentation_prose.py`; a row with one passes. Fixing the prose\n"
        "removes the row on the next `--update`.\n"
        "\n"
        "Sentence length is not here on purpose. It is advisory output of `python -m tests.prose`,\n"
        "because splitting a long enumeration to clear a gate is the failure this report exists to\n"
        "prevent. `tests/prose.py` records which rules were rejected and why.\n"
        "\n"
        "| " + " | ".join(LEDGER_COLUMNS) + " |\n"
        "| " + " | ".join("---" for _ in LEDGER_COLUMNS) + " |\n"
    )
    rows = [
        f"| `{item.file}` | {item.rule} | `{item.id}` | {item.excerpt} | "
        f"{existing.get(item.id, '')} |"
        for item in found
    ]
    return header + "\n".join(rows) + ("\n" if rows else "")


def hook() -> int:
    """Report on the file a ``PostToolUse`` hook just saw edited.

    Reads Claude Code's hook payload on stdin and returns the findings as
    ``additionalContext``, so they reach the agent that made the edit while it still has the
    reason for it in mind.  That is the moment when judgment is cheap; at review time the
    author has to reconstruct why the sentence is the shape it is.

    Silent for anything outside the reader-facing set, and silent when there is nothing to say.
    """
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not raw:
        return 0
    try:
        document = Path(raw).resolve()
    except OSError:
        return 0
    if document not in set(READER_FACING):
        return 0

    found = findings_for(document)
    recorded = dispositions()
    unjudged = [
        item for item in found if not recorded.get(item.id, "").strip().startswith("accepted:")
    ]
    long = long_sentences(document)
    if not unjudged and not long:
        return 0

    name = document.relative_to(ROOT).as_posix()
    report = [f"Prose report for {name}. These are things to look at, not errors."]
    if unjudged:
        report.append(
            f"{len(unjudged)} finding(s) with no recorded judgment. Fix the sentence, or run "
            "`python -m tests.prose --update` and write `accepted: <reason>` in "
            "tests/prose-report.md. Do not reword a sentence merely to clear a row."
        )
        report += [f"  line {item.line}  {item.rule}  {item.excerpt}" for item in unjudged]
    if long:
        worst = max(long, key=lambda item: item[1])
        report.append(
            f"Advisory, not gated: {len(long)} sentence(s) over {SENTENCE_WORDS} words, longest "
            f"{worst[1]} at line {worst[0]}. Split one only where it carries more than one idea."
        )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "\n".join(report),
                }
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Report on reader-facing prose.")
    parser.add_argument("--hook", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--path", help="report on one file rather than the whole set")
    parser.add_argument("--update", action="store_true", help="rewrite the ledger")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="rewrite the ledger, marking every undecided row as pre-existing",
    )
    arguments = parser.parse_args()

    if arguments.hook:
        return hook()

    if arguments.path:
        document = Path(arguments.path).resolve()
        if document not in set(READER_FACING):
            return 0
        documents = [document]
    else:
        documents = list(READER_FACING)

    found = scan(documents)
    existing = dispositions()

    if arguments.update or arguments.seed:
        if arguments.path:
            parser.error("--update and --seed rewrite the whole ledger; drop --path")
        if arguments.seed:
            existing = {
                item.id: existing.get(item.id)
                or "accepted: pre-existing when this report was introduced"
                for item in found
            }
        whole = scan()
        LEDGER.write_text(render(whole, existing), encoding="utf-8", newline="\n")
        undecided = sum(1 for item in whole if not existing.get(item.id, "").strip())
        print(f"{LEDGER.relative_to(ROOT)}: {len(whole)} finding(s), {undecided} undecided")
        return 0

    unjudged = [item for item in found if not existing.get(item.id, "").strip()]
    for item in found:
        mark = " " if item.id in existing and existing[item.id].strip() else "!"
        print(f"{mark} {item.file}:{item.line}  {item.rule}  {item.excerpt}")
    if unjudged:
        print(
            f"\n{len(unjudged)} finding(s) with no recorded judgment. Read each sentence, then"
            f"\neither change it or run `python -m tests.prose --update` and write"
            f"\n`accepted: <reason>` in {LEDGER.relative_to(ROOT).as_posix()}."
        )

    measured = [(document, long_sentences(document)) for document in documents]
    measured = [(document, long) for document, long in measured if long]
    if measured:
        total = sum(len(long) for _, long in measured)
        print(f"\nAdvisory, not gated: {total} sentence(s) over {SENTENCE_WORDS} words.")
        for document, long in sorted(measured, key=lambda item: -len(item[1])):
            worst = max(long, key=lambda item: item[1])
            print(
                f"  {len(long):4d}  longest {worst[1]:3d}w at line {worst[0]:<5d} "
                f"{document.relative_to(ROOT).as_posix()}"
            )
        if arguments.path:
            for line, count, sentence in sorted(measured[0][1], key=lambda item: -item[1])[:5]:
                print(f"    line {line}, {count}w: {' '.join(sentence.split())[:90]}")
        print("  Split one only where it carries more than one idea.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

r"""Every relative link in the documentation resolves, file and anchor.

**This is the same kind of check as** ``tests/unit/test_evidence_manifest.py``: two things that
have to agree, kept in agreement by a test rather than by care.  The documentation is the argument
this repository makes -- ``docs/roadmap.md``'s standing-decisions table exists so that a decision
costs a table rather than seven write-ups, and every row of it links to its evidence -- so a link
that silently stops resolving turns a cited decision into an unsupported one.

**And it fails silently, which is why it needs a test.**  A wrong anchor renders as an ordinary
link: it is only on the click that the reader lands at the top of the page instead of at the
section, and the page is long enough that this reads as the section having been deleted.  Four
were found this way, and two of them had been wrong since the section they pointed at was
renamed.

Two kinds of mistake are checked, and a third is refused rather than checked:

* a relative **path** that does not exist -- a moved or renamed file;
* a **fragment** naming no heading in the destination -- a renamed heading, or a slug guessed
  rather than derived;
* and a heading whose slug this module cannot derive *exactly* is a test failure asking for the
  heading to be reworded, rather than a check that is skipped.  See :func:`slug`.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Every markdown file the documentation set is made of.  ``docs/`` is the bulk; the other three
#: are linked from it and link back into it, so leaving them out would check one direction of a
#: two-way relationship.
DOCUMENTS = sorted(
    {
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("evidence/**/*.md"),
    }
)

#: A fenced block, so a link *inside* an example is not read as a link.  Several documents quote
#: markdown, and a command line carrying ``](`` is not a cross-reference.
FENCE = re.compile(r"^\s*(```|~~~)")

#: ``[text](target)`` with no whitespace in the target, which is the only form used here.
LINK = re.compile(r"\]\(([^)\s]+)\)")

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def slug(heading: str) -> str:
    """GitHub's own anchor rule, for the subset of it this repository stays inside.

    Lowercase, drop inline markup, drop everything that is not alphanumeric, a space, a hyphen or
    an underscore, then hyphenate the spaces.

    **Underscores are kept**, and getting that wrong is what made a first version of this check
    report two correct links as broken: ``### Reporting `R_Q` and `R_g` separately`` anchors at
    ``reporting-r_q-and-r_g-separately``, because an underscore inside a word is not emphasis.
    Emphasis in this repository's headings is written with ``*``.
    """
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"`|\*\*|\*|~~", "", text)
    kept = "".join(c for c in text.lower() if c.isalnum() or c in " -_")
    return kept.strip().replace(" ", "-")


def ambiguous(heading: str) -> list[str]:
    """Characters whose slug this module cannot derive, so a heading must not carry them.

    **One class, and it is narrower than the obvious guess.**  Superscripts and subscripts are
    *not* here -- ``²`` and ``₀`` are category ``No``, and ``str.isalnum`` is true of both, so
    :func:`slug` keeps them exactly as GitHub does and ``O(T² n)`` needs no rewording.  What
    cannot be derived is a **combining mark**: ``D̂`` is ``D`` followed by ``U+0302``, which is not
    alphanumeric, so :func:`slug` drops it and GitHub keeps it.

    Rather than encode a guess, a heading carrying one fails this module and is reworded -- the
    notation belongs in the section's first line, where it can be written however it reads best
    and no anchor depends on it.  ``P₀D̂`` was exactly this case, and the one link to it had
    guessed ``#evaluating-pd-...``, which is what the guess looks like when it is wrong.
    """
    return [c for c in heading if unicodedata.combining(c) != 0]


def headings(text: str) -> list[str]:
    """Every heading outside a fenced block."""
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        found = HEADING.match(line)
        if found:
            out.append(found.group(2))
    return out


def links(text: str) -> list[str]:
    """Every link target outside a fenced block."""
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            out.extend(LINK.findall(line))
    return out


ANCHORS = {
    path: {slug(head) for head in headings(path.read_text(encoding="utf-8"))} for path in DOCUMENTS
}


def test_there_are_documents_to_check() -> None:
    """The negative control: a glob that stopped matching would make every test below vacuous."""
    assert len(DOCUMENTS) > 10
    assert ROOT / "docs" / "roadmap.md" in ANCHORS


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_heading_has_a_derivable_anchor(path: Path) -> None:
    """No heading carries a character whose anchor this module has to guess at.

    A policy rather than a limitation: the alternative is a check that skips the headings most
    likely to be linked wrongly, which is where the guessing happens.
    """
    offenders = {head: ambiguous(head) for head in headings(path.read_text(encoding="utf-8"))}
    carrying = {head: found for head, found in offenders.items() if found}

    assert not carrying, (
        f"{path.relative_to(ROOT)} has heading(s) whose GitHub anchor cannot be derived here: "
        f"{carrying}. Move the notation into the section's first line and give the heading a "
        "plain-text title"
    )


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(path: Path) -> None:
    """Both halves of a target: the file exists, and the fragment names one of its headings."""
    broken: list[str] = []
    for target in links(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#!")):
            continue
        relative, _, fragment = target.partition("#")
        destination = (path.parent / relative).resolve() if relative else path
        if relative and not destination.exists():
            broken.append(f"{target} -- no such file")
            continue
        if not fragment:
            continue
        known = ANCHORS.get(destination)
        if known is None:
            # A fragment into a file outside the documentation set -- source, or a directory.
            # Not this module's to adjudicate, and the path itself was checked above.
            continue
        if fragment not in known:
            broken.append(f"{target} -- no heading anchors at #{fragment}")

    assert not broken, f"{path.relative_to(ROOT)}: " + "; ".join(broken)

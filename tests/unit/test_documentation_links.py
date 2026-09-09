r"""Every relative link in the documentation resolves, file and anchor.

**This is the same kind of check as** :class:`tests.unit.test_registry.TestEvidenceManifest`: two things that
have to agree, kept in agreement by a test rather than by care.  The documentation is the argument
this repository makes -- ``docs/architecture-invariants.md`` states each standing decision beside
the evidence that supports it and the condition that would reopen it -- so a link that silently
stops resolving turns a cited decision into an unsupported one.

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

**Source files are checked too, and the reason is a link this module once missed.**  A
restructure deleted ``docs/roadmap.md``'s *Current limitations* section while a docstring in
:mod:`tests.unit.test_oracle_reductions` still linked to that anchor, and the whole sweep passed
green: the globs below used to cover ``*.md`` alone, so a markdown link inside a Python docstring
was invisible to exactly the check written to catch it.  :data:`SOURCES` closes that.  It reaches
only ``[text](target)`` forms -- prose that names ``docs/roadmap.md`` in double backticks is still
unchecked, and there is a lot of it -- so this is one hole closed rather than the class solved.
"""

from __future__ import annotations

import importlib
import re
import unicodedata
from pathlib import Path

import pytest

from tests.documents import DOCUMENTS, ROOT

#: Every Python file that may carry a cross-reference in a docstring or comment.  These are not
#: link *destinations* -- nothing anchors into a module -- but they are link *sources*, and a
#: reference that rots here rots as silently as one in a document.
SOURCES = sorted(
    {
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
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
    assert len(SOURCES) > 10
    assert ROOT / "docs" / "roadmap.md" in ANCHORS
    # The document this module's own rationale rests on: standing decisions live here, and a
    # rename that quietly emptied it would leave every cited decision unsupported.
    assert ROOT / "docs" / "architecture-invariants.md" in ANCHORS


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


def unresolved(path: Path, targets: list[str]) -> list[str]:
    """Both halves of each target: the file exists, and the fragment names one of its headings."""
    broken: list[str] = []
    for target in targets:
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
    return broken


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(path: Path) -> None:
    """Both halves of a target: the file exists, and the fragment names one of its headings."""
    broken = unresolved(path, links(path.read_text(encoding="utf-8")))
    assert not broken, f"{path.relative_to(ROOT)}: " + "; ".join(broken)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_documentation_link_in_source_resolves(path: Path) -> None:
    """The same two halves, for links written inside docstrings and comments.

    **Narrowed to targets naming a markdown file on purpose.**  ``]( `` is not rare in Python --
    ``handlers[name](arg)`` matches :data:`LINK` and means nothing -- so an unfiltered sweep would
    report call expressions as broken documentation.  The filter costs the ability to catch a link
    to a non-markdown path and buys a check that does not cry wolf.
    """
    targets = [target for target in links(path.read_text(encoding="utf-8")) if ".md" in target]
    broken = unresolved(path, targets)
    assert not broken, f"{path.relative_to(ROOT)}: " + "; ".join(broken)


#: A link into this repository's own source on GitHub, as the technical reference writes them.
#: The branch is part of the pattern rather than a capture: a link pinned to a tag or a commit is
#: a deliberate snapshot and is not claiming to track the tree this test runs against.
REPOSITORY_SOURCE = re.compile(
    r"^https://github\.com/esbraun/cleverly-tmle/(?:blob|tree)/main/(?P<path>[^#?]+)"
)


def repository_targets(text: str) -> list[str]:
    """The repository-relative path of every ``blob/main`` or ``tree/main`` link in ``text``."""
    return [
        found.group("path").rstrip("/")
        for found in (REPOSITORY_SOURCE.match(target) for target in links(text))
        if found is not None
    ]


def test_the_source_link_sweep_finds_something() -> None:
    """The negative control, and it is not hypothetical.

    The technical reference is built almost entirely out of these links, so a pattern that stopped
    matching would report every one of them as fine.
    """
    found = sum(len(repository_targets(path.read_text(encoding="utf-8"))) for path in DOCUMENTS)
    assert found > 20, f"only {found} repository source links matched; check REPOSITORY_SOURCE"


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_repository_source_link_resolves(path: Path) -> None:
    """A link naming a file in this repository names one that exists.

    **Why this is separate from :func:`test_every_relative_link_resolves`.**  That check skips
    anything beginning ``https://``, which is right for a DOI and wrong for a link into this tree:
    the technical reference cites its own modules and tests by absolute URL so the rendered site
    resolves them, and thirty-odd such links were invisible to the sweep written to catch exactly
    the failure they are prone to.  A renamed module leaves them pointing at a GitHub 404, which
    reads to a reader as the implementation having been deleted rather than moved -- the same
    silent rot the fragment check exists for, one level out.

    Only ``main`` is checked.  A link pinned to a tag or a commit is a snapshot of something that
    was true then, and asserting it against the working tree would be asserting the wrong thing.
    """
    broken = [
        target
        for target in repository_targets(path.read_text(encoding="utf-8"))
        if not (ROOT / target).exists()
    ]
    assert not broken, (
        f"{path.relative_to(ROOT)}: no such path in this repository: {broken}. "
        f"A blob/main link is a claim about the current tree; update it or pin it to a commit"
    )


#: The roadmap item id in a prose citation, which is the form the runtime refusal messages
#: use.  ``LINK`` cannot see these: a refusal is a sentence a user reads in an exception, not
#: a rendered link, so it names the file and the item rather than anchoring into it.
ROADMAP_CITATION = re.compile(r"docs/roadmap\.md ([A-Z]+\d+)")

#: ``### F13. Longitudinal simulated-confounding replay`` and its siblings, keyed by item id.
ROADMAP_ITEMS = {
    head.split(".", 1)[0]: head
    for head in headings((ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8"))
    if "." in head
}


def rule_texts(rule: object) -> list[str]:
    """Every string a fit-wide refusal rule can return, inline or by named constant.

    A rule returns either a literal, which is one of its code constants, or a module-level
    constant, which is a global it loads by name.  Reading both means a citation added in
    either form is covered without a hand-written list.
    """
    code = rule.__code__  # type: ignore[attr-defined]
    module = importlib.import_module(rule.__module__)  # type: ignore[attr-defined]
    found = [constant for constant in code.co_consts if isinstance(constant, str)]
    found.extend(
        value for name in code.co_names if isinstance(value := getattr(module, name, None), str)
    )
    return found


def test_every_roadmap_item_a_refusal_cites_exists() -> None:
    """A refusal that cites a roadmap item cites one the roadmap still declares.

    ``simulated_confounding`` refuses six compositions by naming the artifact each one needs
    and the roadmap item that tracks it.  The citation is prose, so
    :func:`test_every_documentation_link_in_source_resolves` does not see it, and renaming an
    item leaves the user an id that resolves to nothing.  This pull request renamed ``F14`` to
    ``X8``, which is the rot in question.

    Driven off :data:`~cleverly.sensitivity._simulated_confounding_request._FIT_WIDE_RULES`
    rather than a list, so a rule that cites a new item is checked the day it is added.
    """
    from cleverly.sensitivity._simulated_confounding_request import _FIT_WIDE_RULES

    cited = {
        item
        for _name, rule in _FIT_WIDE_RULES
        for text in rule_texts(rule)
        for item in ROADMAP_CITATION.findall(text)
    }

    # The negative control.  A rule table read through the wrong attribute, or a citation
    # reworded past ``ROADMAP_CITATION``, would make the assertion below vacuously true.
    assert len(cited) >= 6, f"only {sorted(cited)} matched; check ROADMAP_CITATION"
    missing = sorted(item for item in cited if item not in ROADMAP_ITEMS)
    assert not missing, (
        f"refusal messages cite roadmap item(s) docs/roadmap.md does not declare: {missing}. "
        f"Declared items are {sorted(ROADMAP_ITEMS)}"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_roadmap_item_cited_in_source_exists(path: Path) -> None:
    """The same check one level out, for any prose citation in any module or test.

    The rule table is where these citations live today.  The habit is not confined to it,
    and a citation written into a docstring rots exactly as quietly.
    """
    missing = sorted(
        {
            item
            for item in ROADMAP_CITATION.findall(path.read_text(encoding="utf-8"))
            if item not in ROADMAP_ITEMS
        }
    )
    assert not missing, (
        f"{path.relative_to(ROOT)} cites roadmap item(s) that do not exist: {missing}"
    )

"""The registry-decided lists of the method-evidence navigation pages.

Two lists on those pages follow from the registry alone, and a hand-kept copy of either drifts
the moment a branch registers a study: the index once named four off-grid studies when there
were five.  So both are generated blocks, filled from :func:`registered` the way
:mod:`tests.studies.evidence.red_cells` fills the red-cell ledger.

* ``off-grid-studies`` on the index: each study that declares neither property cells nor a
  reference, so it has a page and no row of the validation grid.
* ``pending-comparators`` on the reading guide: each record with a ``pending_reference``, the
  version its declared image pins, and what the study rests on until the comparator runs.

``tests/unit/test_method_evidence.py`` asserts that each published block is what :func:`rendered`
returns.  ``python -m tests.studies.evidence.document`` calls :func:`fill`, so registering a study
refreshes both blocks with no hand edit.  No study imports this module.

Run it on its own with::

    python -m tests.studies.evidence.navigation
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from tests.studies.evidence.document import _CLOSE, _OPEN, _table
from tests.studies.evidence.manifest import write_lines
from tests.studies.evidence.registry import ROOT, StudyRecord, registered

#: The method-evidence directory.
EVIDENCE = ROOT / "docs" / "technical-reference" / "method-evidence"

#: Each generated block and the page that carries it once.
PAGES: Mapping[str, Path] = {
    "off-grid-studies": EVIDENCE / "index.md",
    "pending-comparators": EVIDENCE / "how-to-read.md",
}

#: Columns of the pending-comparator table.  The page's typed table of comparators that no
#: record declares uses the same columns.
PENDING_COLUMNS = ("study", "comparator and pin", "owner", "what the study rests on meanwhile")

#: A record whose pending comparator ran in another registered study, mapped to that study's
#: slug.  ``adaptive-iptw`` keeps ``pending_reference="drtmle"`` by the decision recorded in
#: ``tests/studies/canonical_adaptive_iptw.py``; the pairing study registers ``drtmle`` as its
#: reference.  The test of this module checks that each target does.
PAIRED_ELSEWHERE: Mapping[str, str] = {"adaptive-iptw": "adaptive-iptw-pairing"}


def _markdown_link(text: str, target: str) -> str:
    # Joined from parts because ``test_documentation_links`` reads every bracket-parenthesis
    # pair in a source file as a link, and a template's placeholder names no file.
    return "".join(("[", text, "]", "(", target, ")"))


#: The owner a pending comparator reads when no roadmap item names one.
NONE_NAMED = "none named; see " + _markdown_link("Comparisons", "../../roadmap.md#comparisons")


def _link(record: StudyRecord) -> str:
    return _markdown_link(record.name, Path(record.document).name)


def on_the_grid(record: StudyRecord) -> bool:
    """Whether the validation grid lists the study: it declares property cells or a reference.

    Parameters
    ----------
    record : StudyRecord
        The study.

    Returns
    -------
    bool
        ``True`` when the study has a grid row.
    """
    return bool(record.property_cells) or record.reference is not None


def render_off_grid(records: Sequence[StudyRecord]) -> list[str]:
    """One line per off-grid study: its link, why it has no row, and its publication policy.

    Parameters
    ----------
    records : Sequence of StudyRecord
        The studies to read, in registry order.

    Returns
    -------
    list of str
        The block's lines.
    """
    lines = []
    for record in records:
        if on_the_grid(record):
            continue
        policy = f"published under the `{record.publication_policy}` policy"
        if record.publication_policy == "gated":
            policy += (
                ", for the reason `GATED_OFF_GRID` in `tests/unit/test_method_evidence.py` records"
            )
        lines.append(
            f"- {_link(record)}: no property cell and no comparator in its record; {policy}."
        )
    return lines


def pinned_version(record: StudyRecord) -> str | None:
    """The version of the pending comparator that the image the manifest declares installs.

    Parameters
    ----------
    record : StudyRecord
        A study with a ``pending_reference``.

    Returns
    -------
    str or None
        The version a declared ``Dockerfile`` asserts with ``packageVersion``, or ``None`` when
        no declared image asserts one.
    """
    manifest = json.loads(record.artifact("manifest.json").read_text(encoding="utf-8"))
    sources = manifest.get("comparator", {}).get("declared_sources_sha256", {})
    assertion = re.compile(
        r"packageVersion\('" + re.escape(str(record.pending_reference)) + r"'\)\)\s*==\s*'([^']+)'"
    )
    for source in sorted(sources):
        if Path(source).name != "Dockerfile":
            continue
        match = assertion.search((ROOT / source).read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return None


def render_pending(records: Sequence[StudyRecord]) -> list[str]:
    """One table row per record with a ``pending_reference``.

    Parameters
    ----------
    records : Sequence of StudyRecord
        The studies to read, in registry order.

    Returns
    -------
    list of str
        The block's lines: a table with :data:`PENDING_COLUMNS`.

    Raises
    ------
    LookupError
        If a :data:`PAIRED_ELSEWHERE` target is not a registered study.
    """
    by_slug = {record.slug: record for record in records}
    rows = []
    for record in records:
        if record.pending_reference is None:
            continue
        version = pinned_version(record)
        comparator = f"R `{record.pending_reference}`" + (
            f" {version}" if version else ", no version asserted by the declared image"
        )
        owner = NONE_NAMED
        if record.slug in PAIRED_ELSEWHERE:
            target = by_slug.get(PAIRED_ELSEWHERE[record.slug])
            if target is None:
                raise LookupError(
                    f"{record.slug} is paired in {PAIRED_ELSEWHERE[record.slug]}, "
                    "which is not registered"
                )
            owner = f"ran in {_markdown_link(target.slug, Path(target.document).name)}"
        cells = len(record.property_cells)
        rests_on = (
            f"its truth tests and {cells} property cell{'s' if cells != 1 else ''}, "
            f"published under the `{record.publication_policy}` policy"
        )
        rows.append((_link(record), comparator, owner, rests_on))
    return _table(PENDING_COLUMNS, rows)


def rendered(records: Sequence[StudyRecord] | None = None) -> dict[str, list[str]]:
    """Render both blocks from the registry.

    Parameters
    ----------
    records : Sequence of StudyRecord, optional
        The studies to read.  Defaults to every registered study.

    Returns
    -------
    dict
        Each block's lines, keyed by its name in :data:`PAGES`.
    """
    records = registered() if records is None else records
    return {
        "off-grid-studies": render_off_grid(records),
        "pending-comparators": render_pending(records),
    }


def _span(lines: Sequence[str], name: str, page: Path) -> tuple[int, int]:
    opening = _OPEN.format(name=name)
    heads = [index for index, line in enumerate(lines) if line.strip() == opening]
    if len(heads) != 1:
        raise LookupError(f"{page.name} carries {opening} {len(heads)} times, not once")
    head = heads[0]
    tail = next(
        (index for index in range(head + 1, len(lines)) if lines[index].strip() == _CLOSE),
        None,
    )
    if tail is None:
        raise LookupError(f"{opening} is never closed by {_CLOSE}")
    return head, tail


def published(name: str, page: Path | None = None) -> list[str]:
    """The lines of one generated block as its page carries them.

    Parameters
    ----------
    name : str
        The block, a key of :data:`PAGES`.
    page : Path, optional
        The page to read.  Defaults to the block's page in :data:`PAGES`.

    Returns
    -------
    list of str
        The lines between the block's sentinels.

    Raises
    ------
    LookupError
        If the page does not carry the block exactly once, or never closes it.
    """
    page = PAGES[name] if page is None else page
    lines = page.read_text(encoding="utf-8").splitlines()
    head, tail = _span(lines, name, page)
    return lines[head + 1 : tail]


def fill() -> list[str]:
    """Rewrite both generated blocks from the registry.

    Returns
    -------
    list of str
        One line per block whose content changed.

    Raises
    ------
    LookupError
        If a page does not carry its block exactly once.
    """
    changed: list[str] = []
    for name, block in rendered().items():
        page = PAGES[name]
        lines = page.read_text(encoding="utf-8").splitlines(keepends=True)
        head, tail = _span(lines, name, page)
        replacement = [line + "\n" for line in block]
        if lines[head + 1 : tail] != replacement:
            changed.append(f"{page.name} {name}: {len(block)} line(s)")
            lines[head + 1 : tail] = replacement
            write_lines(page, "".join(lines))
    return changed


def report() -> None:
    """Rewrite the blocks and print what changed."""
    changed = fill()
    print(f"method-evidence navigation: {len(changed)} block(s) updated")
    for line in changed:
        print(f"  {line}")


def main() -> None:
    """Parse no arguments, so ``--help`` describes the module, and rewrite the blocks."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    report()


if __name__ == "__main__":
    main()

"""Parse and select the executable sections embedded in the documentation."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = sorted({*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md")})

SECTION = re.compile(r"^<!-- doc-section: (?P<meta>.*?) -->[ \t]*$", re.MULTILINE)
BLOCK = re.compile(
    r"(?P<comments>(?:^<!--(?:(?!-->)[\s\S])*-->[ \t]*\n\s*)*)"
    r"^```python\n(?P<code>.*?)^```",
    re.MULTILINE | re.DOTALL,
)
COMMENT = re.compile(r"<!--\s*(.*?)\s*-->", re.DOTALL)
CATALOGUE = "catalogue:"


@dataclass(frozen=True)
class DocBlock:
    document: Path
    index: int
    block_id: str
    section_id: str
    code: str
    start_line: int
    tier: str = "fast"
    catalogue_reason: str | None = None


@dataclass(frozen=True)
class DocSection:
    document: Path
    section_id: str
    requires: tuple[str, ...]
    paths: tuple[str, ...]
    start_line: int
    end_line: int
    blocks: tuple[DocBlock, ...]


def _fields(meta: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in meta.split(";"):
        if not item.strip():
            continue
        name, separator, value = item.strip().partition("=")
        if not separator:
            raise ValueError(f"metadata field has no '=': {item!r}")
        fields[name.strip()] = value.strip()
    return fields


def _items(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_document(document: Path) -> tuple[DocSection, ...]:
    """Return explicitly declared executable sections from one Markdown document."""
    text = document.read_text(encoding="utf-8")
    declarations = list(SECTION.finditer(text))
    sections: list[DocSection] = []
    for position, declaration in enumerate(declarations):
        fields = _fields(declaration.group("meta"))
        unknown = set(fields) - {"id", "requires", "paths"}
        if unknown:
            raise ValueError(f"{document}: unknown doc-section fields: {sorted(unknown)}")
        section_id = fields.get("id", "")
        if not section_id:
            raise ValueError(f"{document}: doc-section has no id")
        start = declaration.end()
        end = declarations[position + 1].start() if position + 1 < len(declarations) else len(text)
        blocks: list[DocBlock] = []
        for index, match in enumerate(BLOCK.finditer(text)):
            fence_position = match.start("code")
            if not start <= fence_position < end:
                continue
            comments = [item.strip() for item in COMMENT.findall(match.group("comments"))]
            block_markers = [item for item in comments if item.startswith("doc-block:")]
            if len(block_markers) != 1:
                line = text.count("\n", 0, fence_position) + 1
                raise ValueError(
                    f"{document}:{line}: executable block needs exactly one doc-block marker"
                )
            block_fields = _fields(block_markers[0][len("doc-block:") :].strip())
            unknown_block = set(block_fields) - {"id", "tier"}
            if unknown_block:
                raise ValueError(f"{document}: unknown doc-block fields: {sorted(unknown_block)}")
            block_id = block_fields.get("id", "")
            if not block_id:
                raise ValueError(f"{document}: doc-block has no id")
            catalogue = next(
                (item[len(CATALOGUE) :].strip() for item in comments if item.startswith(CATALOGUE)),
                None,
            )
            blocks.append(
                DocBlock(
                    document=document,
                    index=index,
                    block_id=block_id,
                    section_id=section_id,
                    code=match.group("code"),
                    start_line=text.count("\n", 0, fence_position) + 1,
                    tier=block_fields.get("tier", "fast"),
                    catalogue_reason=catalogue,
                )
            )
        sections.append(
            DocSection(
                document=document,
                section_id=section_id,
                requires=_items(fields.get("requires", "")),
                paths=_items(fields.get("paths", "")),
                start_line=text.count("\n", 0, declaration.start()) + 1,
                end_line=text.count("\n", 0, end) + 1,
                blocks=tuple(blocks),
            )
        )
    return tuple(sections)


def all_sections() -> tuple[DocSection, ...]:
    return tuple(section for document in DOCUMENTS for section in parse_document(document))


def all_blocks() -> tuple[DocBlock, ...]:
    return tuple(block for section in all_sections() for block in section.blocks)


def validate(sections: tuple[DocSection, ...]) -> list[str]:
    """Return metadata errors without stopping at the first malformed relationship."""
    errors: list[str] = []
    section_ids = [section.section_id for section in sections]
    block_ids = [block.block_id for section in sections for block in section.blocks]
    for label, values in (("section", section_ids), ("block", block_ids)):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            errors.append(f"duplicate {label} ids: {duplicates}")
    blocks = {block.block_id: block for section in sections for block in section.blocks}
    order = {
        block.block_id: position
        for position, block in enumerate(block for section in sections for block in section.blocks)
    }
    for section in sections:
        if not section.blocks:
            errors.append(f"section {section.section_id!r} has no executable blocks")
        if not section.paths:
            errors.append(f"section {section.section_id!r} declares no affected paths")
        for pattern in section.paths:
            if not any(path.is_file() for path in ROOT.glob(pattern)):
                errors.append(
                    f"section {section.section_id!r} affected path matches no file: {pattern!r}"
                )
        for required in section.requires:
            dependency = blocks.get(required)
            if dependency is None:
                errors.append(f"section {section.section_id!r} requires missing block {required!r}")
                continue
            first = min((order[block.block_id] for block in section.blocks), default=len(order))
            if dependency.document != section.document or order[required] >= first:
                errors.append(
                    f"section {section.section_id!r} requires {required!r}, which is not an "
                    "earlier block in the same document"
                )
            owner = next(item for item in sections if item.section_id == dependency.section_id)
            runnable = tuple(block for block in owner.blocks if block.catalogue_reason is None)
            if runnable and dependency != runnable[0]:
                errors.append(
                    f"section {section.section_id!r} requires {required!r}, which is not its "
                    "owner section's self-contained setup block"
                )
            if dependency.tier == "slow" and any(block.tier == "fast" for block in section.blocks):
                errors.append(
                    f"fast section {section.section_id!r} requires slow block {required!r}"
                )
    for block in blocks.values():
        if block.tier not in {"fast", "slow"}:
            errors.append(f"block {block.block_id!r} has unknown tier {block.tier!r}")

    graph = {
        section.section_id: {
            blocks[required].section_id for required in section.requires if required in blocks
        }
        for section in sections
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            errors.append(f"cyclic documentation dependency through {node!r}")
            return
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, set()):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    return errors


def dependency_blocks(
    section: DocSection, sections: tuple[DocSection, ...]
) -> tuple[DocBlock, ...]:
    """Resolve the transitive prerequisite closure in document reading order."""
    by_block = {block.block_id: block for item in sections for block in item.blocks}
    by_section = {item.section_id: item for item in sections}
    selected: dict[str, DocBlock] = {}

    def add(item: DocSection) -> None:
        for required in item.requires:
            block = by_block[required]
            add(by_section[block.section_id])
            selected[block.block_id] = block

    add(section)
    order = {
        block.block_id: index
        for index, block in enumerate(block for item in sections for block in item.blocks)
    }
    return tuple(sorted(selected.values(), key=lambda block: order[block.block_id]))


GLOBAL_PATHS = (
    "src/cleverly/estimators/base.py",
    "src/cleverly/estimators/tmle.py",
    "src/cleverly/learners/**",
    "src/cleverly/utils/**",
    "tests/e2e/test_doc_snippets.py",
    "tests/doc_sections.py",
    "pyproject.toml",
)


def select_sections(
    changes: list[tuple[str, int | None]], sections: tuple[DocSection, ...]
) -> set[str]:
    """Select ordinary sections affected by ``(path, changed line)`` records."""
    selected: set[str] = set()
    fast_sections = {
        section.section_id
        for section in sections
        if any(block.tier == "fast" and block.catalogue_reason is None for block in section.blocks)
    }
    for path, line in changes:
        normalized = path.replace("\\", "/")
        if any(fnmatch.fnmatch(normalized, pattern) for pattern in GLOBAL_PATHS):
            selected.update(fast_sections)
            continue
        matched = False
        for section in sections:
            relative = section.document.relative_to(ROOT).as_posix()
            markdown_match = normalized == relative and (
                line is None or section.start_line <= line < section.end_line
            )
            source_match = any(fnmatch.fnmatch(normalized, pattern) for pattern in section.paths)
            if markdown_match or source_match:
                selected.add(section.section_id)
                matched = True
        if normalized.startswith(("src/", "tests/")) and not matched:
            selected.update(fast_sections)

    changed = True
    while changed:
        changed = False
        selected_blocks = {
            block.block_id
            for section in sections
            if section.section_id in selected
            for block in section.blocks
        }
        for section in sections:
            if set(section.requires) & selected_blocks and section.section_id not in selected:
                selected.add(section.section_id)
                changed = True
    return selected & fast_sections


def git_changes(base: str) -> list[tuple[str, int | None]]:
    """Read zero-context added-line hunks; ambiguous deletions conservatively have no line."""
    output = subprocess.run(
        ["git", "diff", "--unified=0", "--no-ext-diff", f"{base}...HEAD", "--"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    changes: list[tuple[str, int | None]] = []
    path: str | None = None
    for row in output.splitlines():
        if row.startswith("+++ b/"):
            path = row[6:]
        elif row.startswith("@@") and path is not None:
            match = re.search(r"\+(\d+)(?:,(\d+))?", row)
            if match is None:
                continue
            start, count = int(match.group(1)), int(match.group(2) or "1")
            if count == 0:
                changes.append((path, None))
            else:
                changes.extend((path, line) for line in range(start, start + count))
    return changes

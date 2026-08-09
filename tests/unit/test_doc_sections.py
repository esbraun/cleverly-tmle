"""The documentation dependency graph and pull-request selector fail closed."""

from pathlib import Path

from tests.doc_sections import (
    ROOT,
    DocBlock,
    DocSection,
    all_sections,
    dependency_blocks,
    select_sections,
    validate,
)


def _block(document: Path, index: int, name: str, section: str, tier: str = "fast") -> DocBlock:
    return DocBlock(document, index, name, section, "pass\n", index + 1, tier)


def _graph() -> tuple[DocSection, ...]:
    document = ROOT / "docs" / "user-guide.md"
    setup = _block(document, 0, "shared-fit", "shared")
    ctmle = _block(document, 1, "ctmle-fit-test", "ctmle-test")
    dependent = _block(document, 2, "dependent-fit", "dependent")
    slow = _block(document, 3, "coverage", "validation-test", "slow")
    return (
        DocSection(document, "shared", (), ("src/shared.py",), 1, 10, (setup,)),
        DocSection(
            document,
            "ctmle-test",
            (),
            ("src/cleverly/estimators/ctmle.py",),
            10,
            20,
            (ctmle,),
        ),
        DocSection(
            document, "dependent", ("shared-fit",), ("src/dependent.py",), 20, 30, (dependent,)
        ),
        DocSection(
            document,
            "validation-test",
            (),
            ("src/cleverly/validation/**",),
            30,
            40,
            (slow,),
        ),
    )


def test_a_source_change_selects_only_its_ordinary_section() -> None:
    assert select_sections([("src/cleverly/estimators/ctmle.py", None)], _graph()) == {"ctmle-test"}


def test_a_changed_setup_selects_its_reverse_dependents() -> None:
    assert select_sections([("src/shared.py", None)], _graph()) == {"shared", "dependent"}


def test_a_markdown_hunk_selects_its_enclosing_section() -> None:
    assert select_sections([("docs/user-guide.md", 15)], _graph()) == {"ctmle-test"}


def test_an_ambiguous_markdown_deletion_selects_the_whole_document_but_not_slow_only() -> None:
    assert select_sections([("docs/user-guide.md", None)], _graph()) == {
        "shared",
        "ctmle-test",
        "dependent",
    }


def test_the_real_ctmle_mapping_excludes_longitudinal_and_coverage_sections() -> None:
    selected = select_sections([("src/cleverly/estimators/ctmle.py", None)], all_sections())
    assert selected == {"collaborative-tmle"}


def test_unknown_implementation_and_global_changes_fail_closed() -> None:
    expected = {"shared", "ctmle-test", "dependent"}
    assert select_sections([("src/cleverly/new_core.py", None)], _graph()) == expected
    assert select_sections([("pyproject.toml", None)], _graph()) == expected


def test_dependency_closure_contains_only_named_setup_blocks() -> None:
    sections = _graph()
    assert [block.block_id for block in dependency_blocks(sections[2], sections)] == ["shared-fit"]


def test_metadata_rejects_a_fast_dependency_on_slow_evidence() -> None:
    sections = _graph()
    bad = DocSection(
        sections[0].document,
        "bad",
        ("coverage",),
        ("src/bad.py",),
        40,
        50,
        (_block(sections[0].document, 4, "bad-fit", "bad"),),
    )
    errors = validate((*sections, bad))
    assert any("requires slow block" in error for error in errors)


def test_metadata_rejects_duplicate_ids_missing_dependencies_and_unknown_tiers() -> None:
    sections = _graph()
    document = sections[0].document
    malformed = DocSection(
        document,
        "shared",
        ("does-not-exist",),
        ("src/shared.py",),
        40,
        50,
        (_block(document, 4, "shared-fit", "shared", "benchmark"),),
    )
    errors = validate((*sections, malformed))
    assert any("duplicate section ids" in error for error in errors)
    assert any("duplicate block ids" in error for error in errors)
    assert any("requires missing block" in error for error in errors)
    assert any("unknown tier" in error for error in errors)

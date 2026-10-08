"""The RM39 resilience functions exist once, in four byte-identical copies (M4).

``tests/canonical/study_harness.R`` defines them.  The three runners that stay off the harness
(``ctmle3_oat``, ``ctmle_selector``, ``drtmle``) carry a copy inline, because sourcing the harness
would change a gated reference source more than the copy does.  A fix that lands in one copy and
not the others is the failure the harness exists to prevent, so the copies are compared function
by function, and the marked block as a whole.  A runner-specific difference is an argument,
never an edit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.studies.evidence.registry import ROOT

CANONICAL = ROOT / "tests" / "canonical"
HARNESS = CANONICAL / "study_harness.R"
COPIES = (
    CANONICAL / "ctmle3_oat" / "run_ctmle3_oat.R",
    CANONICAL / "ctmle_selector" / "run_ctmle.R",
    CANONICAL / "drtmle" / "run_drtmle.R",
)
BEGIN = "# ---- RM39 shared resilience functions: begin ----"
END = "# ---- RM39 shared resilience functions: end ----"

#: The functions of plan section 3.1, plus calibration's driver (M16).  Each must be in the block.
SHARED = (
    "study_env",
    "study_memory",
    "study_rss_mb",
    "study_hwm_mb",
    "study_private_mb",
    "study_parent_pid",
    "study_fork",
    "study_worker_plan",
    "study_seeds",
    "study_seed_for",
    "study_checkpoint_paths",
    "study_fit_group",
    "study_calibrate_group",
    "study_calibrate",
    "study_map",
    "study_startup",
    "study_retry",
    "study_assemble",
)

_DEFINITION = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_.]*) <- ", re.M)


def _statement_end(source: str, start: int) -> int:
    """The end of the top-level statement at ``start``: brackets balanced, strings and comments
    skipped, at the first line feed outside any bracket."""
    depth = 0
    quote: str | None = None
    escaped = False
    comment = False
    for index in range(start, len(source)):
        character = source[index]
        if comment:
            if character == "\n":
                comment = False
                if depth == 0:
                    return index
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'", "`"}:
            quote = character
        elif character == "#":
            comment = True
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "\n" and depth == 0:
            return index
    return len(source)


def definitions(source: str) -> dict[str, str]:
    """Every top-level ``name <- ...`` statement, by name, as written."""
    found: dict[str, str] = {}
    for match in _DEFINITION.finditer(source):
        name = match.group("name")
        if name in found:
            raise AssertionError(f"{name} is defined twice")
        found[name] = source[match.start() : _statement_end(source, match.end())]
    return found


def block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.count(BEGIN) == 1 and text.count(END) == 1, f"{path.name} lacks the shared block"
    return text[text.index(BEGIN) : text.index(END) + len(END)]


def drift(reference: dict[str, str], copy: dict[str, str]) -> list[str]:
    """The names whose text differs, or that one side lacks."""
    return sorted(
        name for name in set(reference) | set(copy) if reference.get(name) != copy.get(name)
    )


def test_the_block_defines_every_shared_function() -> None:
    defined = definitions(block(HARNESS))
    missing = [name for name in SHARED if name not in defined]
    assert not missing, f"the shared block does not define {missing}"


@pytest.mark.parametrize("copy", COPIES, ids=lambda path: path.parent.name)
def test_each_copy_matches_the_harness_function_by_function(copy: Path) -> None:
    differing = drift(definitions(block(HARNESS)), definitions(block(copy)))
    assert not differing, (
        f"{copy.parent.name}/{copy.name} differs from study_harness.R in {differing}. Copy the "
        f"harness block into the runner; a runner-specific difference is an argument"
    )
    assert block(copy) == block(HARNESS), "the block differs outside its definitions"


@pytest.mark.parametrize("path", (HARNESS, *COPIES), ids=lambda path: path.name)
def test_no_shared_function_is_defined_again_outside_the_block(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    outside = text.replace(block(path), "")
    shared = set(definitions(block(HARNESS)))
    assert not shared & set(definitions(outside))


def test_a_one_character_edit_in_one_function_is_named() -> None:
    """The mutation control: an edit inside ``study_retry`` names ``study_retry`` alone."""
    original = block(HARNESS)
    mutated = original.replace("workers %/% 2L", "workers %/% 3L", 1)
    assert mutated != original
    assert drift(definitions(original), definitions(mutated)) == ["study_retry"]


def test_a_brace_inside_a_string_or_comment_does_not_end_a_definition() -> None:
    source = 'f <- function() {\n  x <- "}"  # }\n  x\n}\ng <- 1\n'
    found = definitions(source)
    assert found["f"].endswith("  x\n}")
    assert found["g"] == "g <- 1"

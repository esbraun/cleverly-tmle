r"""One reviewed semantic callback per tutorial, discovered by the tutorial's file stem.

**Why a package and not a dict.**  The callbacks used to live in one mapping inside
:mod:`tests.unit.test_documentation_runtime`.  Converting a tutorial from Markdown to a notebook
changes its registry key, so several authors converting several tutorials at once would all edit
the same lines of the same file.  A module per tutorial gives each author one file to own.

**The convention.**  A tutorial is ``docs/examples/<stem>.md`` or ``docs/examples/<stem>.ipynb``.
Its callback is ``tests/unit/tutorial_semantics/<module>.py``, where ``<module>`` is ``<stem>``
with each hyphen replaced by an underscore.  The module defines ``check(namespace)``, which
receives the namespace the tutorial's code built at its documented size.  A notebook's module may
also define ``UNPRINTED_DECIMALS``, a mapping from a decimal the prose writes to the reason no
stored output prints it.  :func:`narration_mismatches` reads that mapping.

**What is not a tutorial.**  ``index`` is navigation.  The TWINS notebook downloads its data, so
it cannot run in the offline fast tier.  :data:`NOT_TUTORIALS` names both, and the runtime module
asserts that every other example has exactly one callback and every callback has a tutorial.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from tests.documents import READER_FACING, ROOT
from tests.prose import markdown_cells

__all__ = [
    "EXAMPLES",
    "NOT_TUTORIALS",
    "PACKAGE",
    "callback",
    "callback_module",
    "callback_modules",
    "markdown_text",
    "module_name",
    "narrated_decimals",
    "narration_mismatches",
    "notebook_code",
    "stored_output",
    "stored_text",
    "tutorials",
]

EXAMPLES = ROOT / "docs" / "examples"

#: The package directory, whose modules :func:`callback_modules` lists.
PACKAGE = Path(__file__).resolve().parent

#: Stems under ``docs/examples`` that are not program tutorials.
NOT_TUTORIALS = frozenset({"index", "twins-causal-inference"})


def tutorials() -> list[Path]:
    """Every tutorial source under ``docs/examples``, sorted, Markdown or notebook."""
    return sorted(
        path
        for path in READER_FACING
        if path.parent == EXAMPLES
        and path.suffix in {".md", ".ipynb"}
        and path.stem not in NOT_TUTORIALS
    )


def module_name(stem: str) -> str:
    """The callback module name for a tutorial stem."""
    return stem.replace("-", "_")


def callback_modules() -> set[str]:
    """Every callback module present in this package."""
    return {path.stem for path in PACKAGE.glob("*.py") if path.stem != "__init__"}


def callback_module(stem: str) -> ModuleType:
    """Import the callback module for ``stem``."""
    return importlib.import_module(f"{__name__}.{module_name(stem)}")


def callback(stem: str) -> Callable[[dict[str, Any]], None]:
    """The ``check`` function a tutorial's module defines."""
    check: Callable[[dict[str, Any]], None] = callback_module(stem).check
    return check


def _notebook(path: Path) -> dict[str, Any]:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return notebook


def _joined(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def notebook_code(path: Path) -> list[tuple[str, str]]:
    """``(cell id, source)`` for each code cell of the notebook at ``path``, in order."""
    return [
        (str(cell.get("id")), _joined(cell["source"]))
        for cell in _notebook(path)["cells"]
        if cell["cell_type"] == "code"
    ]


def _output_text(output: Mapping[str, Any]) -> str:
    """The readable text of one stored output: its stream, plain text, and HTML payloads."""
    parts = [_joined(output.get("text", ""))]
    data = output.get("data", {})
    parts += [_joined(data[mime]) for mime in ("text/plain", "text/html") if mime in data]
    return "\n".join(part for part in parts if part)


def stored_output(path: Path, cell_id: str) -> str:
    """The stored readable output of one code cell, found by its cell id.

    A callback uses this to compare a committed output against the fresh run.  The lookup by id
    is why the authoring guide asks for stable ids on the cells a test reads.
    """
    cells = [cell for cell in _notebook(path)["cells"] if cell.get("id") == cell_id]
    assert len(cells) == 1, f"{path.name} has {len(cells)} cell(s) with id {cell_id!r}"
    return "\n".join(_output_text(output) for output in cells[0].get("outputs", ()))


def stored_text(path: Path) -> str:
    """The stored readable output of every code cell, joined."""
    return "\n".join(
        _output_text(output)
        for cell in _notebook(path)["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", ())
    )


def markdown_text(path: Path) -> str:
    """Every markdown cell of the notebook at ``path``, joined as the prose report reads them."""
    return markdown_cells(path)


#: Inline code, a fenced block, a link target, and a bare URL.  None of them is narration: a
#: learner argument such as ``alpha=0.05`` or a pinned version in a URL is not a reported number.
_NOT_NARRATION = re.compile(r"```.*?```|`[^`\n]*`|\]\([^)]*\)|https?://\S+", re.DOTALL)

#: A decimal written in prose.  The lookarounds refuse a version such as ``1.5.4`` and a digit run
#: glued to a word, and still accept a decimal that ends a sentence.
_DECIMAL = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?!\w|\.\d)")

#: Any number a stored output prints, including scientific notation.
_OUTPUT_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def narrated_decimals(markdown: str) -> list[str]:
    """Each decimal literal the prose of ``markdown`` writes, in order of appearance."""
    prose = _NOT_NARRATION.sub(" ", markdown)
    return [f"{whole}.{fraction}" for whole, fraction in _DECIMAL.findall(prose)]


def narration_mismatches(
    markdown: str, outputs: str, unprinted: Mapping[str, str] | None = None
) -> list[str]:
    """Decimals the prose writes that no stored output prints at that precision.

    A narrated ``0.26`` matches a stored ``0.2634`` or ``-0.2571``, because each lies within half
    a unit of the last written place once the sign is dropped.  It does not match ``0.2491``.
    The check is static: it reads the committed outputs, so a platform difference in a fresh run
    cannot move it.  The execution stamp ties those outputs to the code, and this ties the prose to them.

    A decimal the prose writes for another reason, such as a law's parameter, goes in the
    module's ``UNPRINTED_DECIMALS`` with its reason.  Printing the number is the better repair.
    """
    allowed = dict(unprinted or {})
    printed = []
    for token in _OUTPUT_NUMBER.findall(outputs):
        try:
            printed.append(abs(float(token)))
        except ValueError:  # pragma: no cover - the pattern admits only numeric text
            continue
    missing = []
    for literal in narrated_decimals(markdown):
        if literal in allowed:
            continue
        places = len(literal.split(".")[1])
        value = float(literal)
        tolerance = 0.5 * 10.0**-places + 1e-12
        if not any(abs(number - value) <= tolerance for number in printed):
            missing.append(literal)
    return missing

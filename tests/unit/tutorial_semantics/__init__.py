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

**What is checked statically only.**  A notebook in :data:`STATIC_ONLY` is not a tutorial, and
its prose is still compared against its stored outputs.  Its module defines no ``check``.  It may
define ``UNPRINTED_DECIMALS``, and ``check_stored()`` for the committed outputs its readings rely
on.  :func:`narrated_notebooks` lists every notebook the static narration check reads.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from tests.documents import READER_FACING, ROOT
from tests.notebooks import code_cells
from tests.prose import markdown_cells

__all__ = [
    "EXAMPLES",
    "NOT_TUTORIALS",
    "PACKAGE",
    "STATIC_ONLY",
    "assert_plugin_limits_refuse",
    "assert_protocol_recorded",
    "callback",
    "callback_module",
    "callback_modules",
    "changed_fields",
    "covers",
    "markdown_text",
    "module_name",
    "narrated_decimals",
    "narrated_notebooks",
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

#: Notebooks outside the tutorial program whose stored text the static checks still read.  Each
#: needs the network to execute, so none enters the offline semantic gate.
STATIC_ONLY = frozenset({"twins-causal-inference"})


def tutorials() -> list[Path]:
    """Every tutorial source under ``docs/examples``, sorted, Markdown or notebook."""
    return sorted(
        path
        for path in READER_FACING
        if path.parent == EXAMPLES
        and path.suffix in {".md", ".ipynb"}
        and path.stem not in NOT_TUTORIALS
    )


def narrated_notebooks() -> list[Path]:
    """Every notebook whose narrated decimals the static check reads, sorted.

    That is each tutorial notebook, and each notebook in :data:`STATIC_ONLY`.
    """
    return sorted(
        [
            *(path for path in tutorials() if path.suffix == ".ipynb"),
            *(EXAMPLES / f"{stem}.ipynb" for stem in STATIC_ONLY),
        ]
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


def changed_fields(protocol: Any, base: Any) -> set[str]:
    """The names of the dataclass fields in which ``protocol`` differs from ``base``.

    A tutorial that starts from :func:`cleverly.datasets.navigation_protocol` names the fields its
    question changes.  Asserting the exact set checks that sentence in both directions: a field
    the reading omits and a field the code no longer changes each fail.
    """
    return {
        field.name
        for field in dataclasses.fields(protocol)
        if getattr(protocol, field.name) != getattr(base, field.name)
    }


def covers(interval: Any, value: float) -> bool:
    """Whether the closed interval contains ``value``.

    ``interval`` is a ``(low, high)`` pair, or anything with a ``ci`` pair such as a
    :class:`cleverly.ParameterEstimate`.  A reading that says "the interval contains the true value"
    on one draw is a seeded relation, and every callback checks it the same way.
    """
    low, high = getattr(interval, "ci", interval)
    return bool(low <= value <= high)


def assert_protocol_recorded(path: Path, cell_id: str, protocol: Any, *results: Any) -> str:
    """Assert that a stored output prints the protocol fingerprint and each result carries it.

    The protocol step of every tutorial prints ``protocol.summary_lines()``, and each result fitted
    from that study stores the fingerprint in its provenance.  Returns the fingerprint, so a
    callback can look for it in a later output.
    """
    fingerprint: str = protocol.fingerprint
    assert fingerprint in stored_output(path, cell_id), (
        f"{path.name}: cell {cell_id!r} does not print the protocol fingerprint {fingerprint}"
    )
    for result in results:
        assert result.provenance.protocol_fingerprint == fingerprint
    return fingerprint


def assert_plugin_limits_refuse(result: Any, bound: Any, namespace: Mapping[str, Any]) -> None:
    """Assert that a tutorial's plug-in bound refused its limits, as RM22 decided.

    The two tutorials that read the plug-in ``nu^2`` print the refusal of ``ci_lower`` into
    ``limit_refusal`` and assert that ``robustness`` carries no ``rva``.  The bound itself
    refuses the other limits too.  The nonzero witness is that the plug-in elements carry no
    curve while their point ``nu^2`` is positive.

    Parameters
    ----------
    result : Any
        The fit the tutorial read the bound from.
    bound : Any
        The plug-in :class:`~cleverly.sensitivity.omitted_variable.SensitivityBounds`.
    namespace : Mapping
        The tutorial's namespace, holding ``limit_refusal`` and ``robustness``.
    """
    from cleverly import CapabilityError

    refusal = namespace["limit_refusal"]
    assert "F26" in refusal
    assert "plug-in nu^2" in refusal
    assert "nu2_estimator='plugin'" in refusal
    assert "rva" not in namespace["robustness"]
    assert namespace["robustness"]["nu2_estimator"] == "plugin"
    assert bound.nu2_estimator == "plugin"
    for accessor in ("ci_upper", "robustness_value_ci", "plugin_interval_lower"):
        try:
            getattr(bound, accessor)
        except CapabilityError as error:
            assert "F26" in str(error)
        else:
            raise AssertionError(f"the plug-in bound reported {accessor}")
    elements = result.sensitivity.elements(estimand="ate", nu2_estimator="plugin")
    assert elements.psi_nu2 is None and elements.psi_max_bias is None
    assert elements.nu2 > 0.0


def _notebook(path: Path) -> dict[str, Any]:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return notebook


def _joined(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def notebook_code(path: Path) -> list[tuple[str, str]]:
    """``(cell id, source)`` for each code cell of the notebook at ``path``, in order."""
    return [(str(cell.get("id")), _joined(cell["source"])) for cell in code_cells(_notebook(path))]


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
        for cell in code_cells(_notebook(path))
        for output in cell.get("outputs", ())
    )


def markdown_text(path: Path) -> str:
    """Every markdown cell of the notebook at ``path``, joined as the prose report reads them."""
    return markdown_cells(path)


#: Inline code, a fenced block, a link target, and a bare URL.  None of them is narration: a
#: learner argument such as ``alpha=0.05`` or a pinned version in a URL is not a reported number.
_NOT_NARRATION = re.compile(r"```.*?```|`[^`\n]*`|\]\([^)]*\)|https?://\S+", re.DOTALL)

#: A signed decimal written in prose. The lookarounds refuse a version such as ``1.5.4`` and a
#: digit run glued to a word, and still accept a decimal that ends a sentence. Unicode minus is
#: normalized because mathematical prose often uses it while stored Python output uses ASCII.
_UNICODE_MINUS = "\N{MINUS SIGN}"
_DECIMAL = re.compile(r"(?<![\w.])([+\-\N{MINUS SIGN}]?)(\d+)\.(\d+)(?!\w|\.\d)")

#: Any number a stored output prints, including scientific notation.
_OUTPUT_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def narrated_decimals(markdown: str) -> list[str]:
    """Each decimal literal the prose of ``markdown`` writes, in order of appearance."""
    prose = _NOT_NARRATION.sub(" ", markdown)
    return [
        f"{'-' if sign == _UNICODE_MINUS else sign}{whole}.{fraction}"
        for sign, whole, fraction in _DECIMAL.findall(prose)
    ]


def narration_mismatches(
    markdown: str, outputs: str, unprinted: Mapping[str, str] | None = None
) -> list[str]:
    """Decimals the prose writes that no stored output prints at that precision.

    A narrated ``-0.26`` matches a stored ``-0.2634`` because it lies within half a unit of the
    last written place. It does not match ``0.2634`` or ``-0.2491``.
    The check is static: it reads the committed outputs, so a platform difference in a fresh run
    cannot move it.  The execution stamp ties those outputs to the code, and this ties the prose to them.

    A decimal the prose writes for another reason, such as a law's parameter, goes in the
    module's ``UNPRINTED_DECIMALS`` with its reason.  Printing the number is the better repair.
    """
    allowed = dict(unprinted or {})
    printed = []
    for token in _OUTPUT_NUMBER.findall(outputs):
        try:
            printed.append(float(token))
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

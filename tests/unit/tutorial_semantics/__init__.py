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
stored output prints it.  :func:`narration_mismatches` reads that mapping.  A notebook's module
also defines ``REFIT_CELLS``, the tuple of its refit cells in notebook order, which the runtime
module's pin test compares with :func:`refit_cells`.  The check reads
fixed-point and scientific decimals in prose, keyed by their written form, such as ``0.26`` or
``5.114e-04``.  It does not read a decimal inside inline code.

**What is not a tutorial.**  ``index`` is navigation.  The TWINS notebook downloads its data, so
it cannot run in the offline fast tier.  :data:`NOT_TUTORIALS` names both, and the runtime module
asserts that every other example has exactly one callback and every callback has a tutorial.

**What is checked statically only.**  A notebook in :data:`STATIC_ONLY` is not a tutorial, and
its prose is still compared against its stored outputs.  Its module defines no ``check``.  It may
define ``UNPRINTED_DECIMALS``, and ``check_stored()`` for the committed outputs its readings rely
on.  :func:`narrated_notebooks` lists every notebook the static narration check reads.

**Refit cells run in the slow tier.**  A refutation or sensitivity cell refits the estimator
many times, and those cells were most of the gate's cost.  :func:`refit_cells` derives them from
the code: a cell that calls a name in :data:`REFIT_CALLS`, and any later cell that reads a name
one of them binds.  The fast gate executes the other cells, compiles the refit cells without
running them, and calls ``check(namespace)``.  The slow gate (``pytest -m slow``) executes every
cell and then calls ``check`` and the module's ``check_refits(namespace)``.  So ``check`` must
not read a refit-bound name, and a module defines ``check_refits`` exactly when its notebook has
a refit cell.  No cell is tagged: the rule reads only the code that the execution stamp covers.
"""

from __future__ import annotations

import ast
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
    "REFIT_CALLS",
    "STATIC_ONLY",
    "RefitNamespace",
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
    "refit_bound",
    "refit_cells",
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
    """Assert that a stored output prints the protocol, by its fingerprint or by every field,
    and that each result carries the protocol's fingerprint.

    The protocol step of every tutorial prints ``protocol.summary_lines()``, whose fields the
    check reads as text, and a later step that declares a record may print its fingerprint
    instead. Each result fitted from that study stores the fingerprint in its provenance.  Returns the fingerprint, so
    a callback can look for it in a later output.
    """
    fingerprint: str = protocol.fingerprint
    printed = stored_output(path, cell_id)
    fields = [
        protocol.target_population,
        protocol.time_zero,
        protocol.outcome,
        protocol.horizon,
        protocol.interference_unit,
        *protocol.eligibility,
        *protocol.treatment_strategies,
        *protocol.treatment_versions,
        *protocol.intercurrent_event_handling,
        *protocol.assumption_rationale,
    ]
    # A later step that declares a record prints its fingerprint and the fields it changed.
    if fingerprint not in printed:
        for text in fields:
            assert text in printed, (
                f"{path.name}: cell {cell_id!r} prints neither the protocol fingerprint "
                f"{fingerprint} nor {text!r}"
            )
    for result in results:
        assert result.provenance.protocol_fingerprint == fingerprint
    return fingerprint


def _notebook(path: Path) -> dict[str, Any]:
    notebook: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return notebook


def _joined(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def notebook_code(path: Path) -> list[tuple[str, str]]:
    """``(cell id, source)`` for each code cell of the notebook at ``path``, in order."""
    return [(str(cell.get("id")), _joined(cell["source"])) for cell in code_cells(_notebook(path))]


#: A call to any of these names, as a function or as a method, makes a cell a refit cell.  Each
#: one refits the estimator once per replicate, grid point, or dropped covariate.
REFIT_CALLS = frozenset({"benchmark", "refute", "simulated_confounding"})


def _called(tree: ast.AST) -> set[str]:
    """The function and method names that ``tree`` calls."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                names.add(function.id)
            elif isinstance(function, ast.Attribute):
                names.add(function.attr)
    return names


def _root(node: ast.AST) -> str | None:
    """The name at the base of ``a.b[c].d``, or ``None`` when the base is not a name."""
    while isinstance(node, ast.Attribute | ast.Subscript | ast.Starred):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


#: Nodes whose own names stay local, so a binding inside them is not the cell's binding.
_LOCAL_SCOPES = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def _cell_nodes(tree: ast.AST) -> list[ast.AST]:
    """Every node of a cell outside a comprehension, a lambda, and a function or class body."""
    nodes, pending = [], [tree]
    while pending:
        node = pending.pop()
        nodes.append(node)
        if not isinstance(node, _LOCAL_SCOPES):
            pending.extend(ast.iter_child_nodes(node))
    return nodes


def _bound(tree: ast.AST) -> set[str]:
    """The names a cell binds, or changes through an attribute or item assignment.

    A cell that sets ``fits["a"] = ...`` changes the object under ``fits``, so ``fits`` counts as
    bound.  A name bound inside a comprehension or a function body is local to it and does not
    count.  A change through a method call, such as ``rows.append(...)``, is not seen.
    """
    names = set()
    for node in ast.walk(tree):
        # An assignment expression in a comprehension binds in the cell, not in the comprehension.
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    for node in _cell_nodes(tree):
        if isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            names.add(node.id)
        elif isinstance(node, ast.Attribute | ast.Subscript) and isinstance(
            node.ctx, ast.Store | ast.Del
        ):
            root = _root(node)
            if root is not None:
                names.add(root)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update(
                alias.asname or alias.name.partition(".")[0]
                for alias in node.names
                if alias.name != "*"
            )
    return names


def _aliases(tree: ast.AST) -> set[str]:
    """The local names that a cell imports a refit call under, such as ``r`` in ``as r``."""
    return {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in REFIT_CALLS and alias.asname
    }


class _FreeReads(ast.NodeVisitor):
    """Collect the names a cell reads before the cell itself binds them.

    The visit follows evaluation order closely enough for notebook code: a value before its
    targets, a loop's iterable before its target and body, and a comprehension's iterables before
    its element.  So ``for name, fit in fits.items(): print(name)`` reads ``fits`` and not
    ``name``.  A function body is visited with its parameters bound and the cell's bindings
    so far, which is when a later call can run it at the earliest.
    """

    def __init__(self, bound: set[str] | None = None) -> None:
        self.bound = set() if bound is None else set(bound)
        self.reads: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.bound:
                self.reads.add(node.id)
        else:
            self.bound.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        root = _root(node.target)
        if root is not None and root not in self.bound:
            self.reads.add(root)
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    visit_AsyncFor = visit_For

    def _comprehension(self, node: Any, *elements: ast.AST) -> None:
        inner = _FreeReads(self.bound)
        for generator in node.generators:
            inner.visit(generator.iter)
            inner.visit(generator.target)
            for condition in generator.ifs:
                inner.visit(condition)
        for element in elements:
            inner.visit(element)
        self.reads |= inner.reads

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node, node.elt)

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node, node.key, node.value)

    def _function(self, node: Any) -> None:
        for decorator in getattr(node, "decorator_list", ()):
            self.visit(decorator)
        arguments = node.args
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)
        parameters = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *(argument for argument in (arguments.vararg, arguments.kwarg) if argument),
        ]
        if not isinstance(node, ast.Lambda):
            self.bound.add(node.name)
        inner = _FreeReads(self.bound | {parameter.arg for parameter in parameters})
        for statement in node.body if isinstance(node.body, list) else [node.body]:
            inner.visit(statement)
        self.reads |= inner.reads

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function
    visit_Lambda = _function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression)
        inner = _FreeReads(self.bound)
        for statement in node.body:
            inner.visit(statement)
        self.reads |= inner.reads
        self.bound.add(node.name)

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> None:
        self.bound.update(alias.asname or alias.name.partition(".")[0] for alias in node.names)

    visit_ImportFrom = visit_Import

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self.bound.add(node.name)
        for statement in node.body:
            self.visit(statement)


def _loaded(tree: ast.AST) -> set[str]:
    """The names a cell reads before it binds them itself."""
    visitor = _FreeReads()
    visitor.visit(tree)
    return visitor.reads


def _classify(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The refit cells of a notebook with their reasons, and the names they leave bound."""
    refits: dict[str, str] = {}
    bound_by: dict[str, str] = {}
    calls_of_refits = set(REFIT_CALLS)
    for cell_id, code in notebook_code(path):
        tree = ast.parse(code, filename=f"{path.name}:{cell_id}")
        calls_of_refits |= _aliases(tree)
        calls = sorted(_called(tree) & calls_of_refits)
        reads = sorted(_loaded(tree) & set(bound_by))
        if calls:
            refits[cell_id] = f"calls {', '.join(calls)}"
        elif reads:
            sources = ", ".join(f"{name!r} from {bound_by[name]!r}" for name in reads)
            refits[cell_id] = f"reads {sources}"
        for name in _bound(tree):
            if cell_id in refits:
                bound_by[name] = cell_id
            else:
                bound_by.pop(name, None)
    return refits, bound_by


def refit_cells(path: Path) -> dict[str, str]:
    """Each refit cell of the notebook at ``path``, mapped to the reason it is one.

    A cell is a refit cell when it calls a name in :data:`REFIT_CALLS`, directly, as a method, or
    under an imported alias.  A cell is also a refit cell when it reads a name that an earlier
    refit cell bound and no later ordinary cell rebound.  The rule runs in cell order, so a cell
    that only prints a refutation is a refit cell too.

    The rule reads names, not objects, and it reads each cell as straight-line code.  These
    cases escape it:

    - a refit cell that changes an earlier object through a method call, such as
      ``rows.append(...)``;
    - a name that a cell binds only inside an ``if`` branch and reads later in the same cell;
    - a function, defined in an ordinary cell, that reads a refit-bound global when a later
      ordinary cell calls it.

    In each case the fast gate either raises or hands ``check`` an earlier value of the name.
    The slow gate runs the notebook as written, so it still checks the published behavior.
    Move an assertion that fails this way to ``check_refits``.
    """
    return _classify(path)[0]


def refit_bound(path: Path) -> dict[str, str]:
    """Each name that refit cells leave bound, mapped to the last refit cell that binds it.

    The fast gate skips the refit cells, so these are the names its namespace lacks or holds
    from an earlier ordinary cell.
    """
    return _classify(path)[1]


class RefitNamespace(dict[str, Any]):
    """The fast gate's namespace, which refuses a name that only a skipped refit cell binds.

    ``check(namespace)`` receives this in the fast gate.  Reading a refit-bound name fails with
    the cell that binds it and the remedy, whether the fast run left the name unset or left an
    earlier ordinary value under it.  The guard covers ``namespace[name]`` and
    ``namespace.get(name)``, which are the reads a callback makes.  Iteration, ``in`` and
    ``dict(namespace)`` are not guarded.
    """

    def __init__(self, namespace: Mapping[str, Any], refit_names: Mapping[str, str]) -> None:
        super().__init__(namespace)
        self.refit_names = dict(refit_names)

    def __getitem__(self, name: str) -> Any:
        if name in self.refit_names:
            raise KeyError(
                f"{name!r} is bound by the refit cell {self.refit_names[name]!r}, which the fast "
                "gate does not run. Read it in check_refits(namespace), which the slow gate "
                "(pytest -m slow) calls"
            )
        return super().__getitem__(name)

    def get(self, name: str, default: Any = None) -> Any:
        return self[name] if name in self.refit_names else super().get(name, default)


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

#: A signed decimal written in prose, optionally in scientific notation such as ``5.114e-04``.
#: The lookarounds refuse a version such as ``1.5.4`` and a digit run glued to a word, and still
#: accept a decimal that ends a sentence. Unicode minus is normalized because mathematical prose
#: often uses it while stored Python output uses ASCII.  A literal without a decimal point, such
#: as the threshold ``1e-10``, is not read, in either form.
_UNICODE_MINUS = "\N{MINUS SIGN}"
_DECIMAL = re.compile(
    r"(?<![\w.])([+\-\N{MINUS SIGN}]?)(\d+)\.(\d+)"
    r"(?:([eE])([+\-\N{MINUS SIGN}]?\d+))?(?!\w|\.\d)"
)

#: Any number a stored output prints, including scientific notation.
_OUTPUT_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def narrated_decimals(markdown: str) -> list[str]:
    """Each decimal literal the prose of ``markdown`` writes, in order of appearance.

    A scientific literal keeps its written form, such as ``5.114e-04``, with a Unicode minus
    read as ASCII.  Inline code is not narration and is not read, so a decimal inside backticks
    is never checked.
    """
    prose = _NOT_NARRATION.sub(" ", markdown)
    return [
        f"{sign}{whole}.{fraction}{marker}{exponent}".replace(_UNICODE_MINUS, "-")
        for sign, whole, fraction, marker, exponent in _DECIMAL.findall(prose)
    ]


def _resolution(literal: str) -> float:
    """One unit in the last written place of ``literal``, scaled by its exponent if it has one."""
    mantissa, _, exponent = literal.lower().partition("e")
    places = len(mantissa.partition(".")[2])
    # ``float`` saturates to infinity or zero where ``10.0 ** n`` would raise on stray text.
    return float(f"1e{(int(exponent) if exponent else 0) - places}")


def narration_mismatches(
    markdown: str, outputs: str, unprinted: Mapping[str, str] | None = None
) -> list[str]:
    """Decimals the prose writes that no stored output prints at that precision.

    A narrated ``-0.26`` matches a stored ``-0.2634`` because it lies within half a unit of the
    last written place. It does not match ``0.2634`` or ``-0.2491``.
    The check is static: it reads the committed outputs, so a platform difference in a fresh run
    cannot move it.  The execution stamp ties those outputs to the code, and this ties the prose to them.

    A scientific literal such as ``5.114e-04`` has a tolerance of half a unit in the mantissa's
    last place, times ten to its exponent.  An output matches it only if it also shows at least
    that precision, in either notation: ``5.114e-04`` and ``0.0005114`` match, and ``0.0005``
    and ``5.1e-04`` do not.  A fixed-point literal keeps the absolute tolerance above and puts no
    precision condition on the output.

    A decimal the prose writes for another reason, such as a law's parameter, goes in the
    module's ``UNPRINTED_DECIMALS`` with its reason, keyed by its written form.  Printing the
    number is the better repair.
    """
    allowed = dict(unprinted or {})
    printed = []
    for token in _OUTPUT_NUMBER.findall(outputs):
        try:
            printed.append((float(token), _resolution(token)))
        except ValueError:  # pragma: no cover - the pattern admits only numeric text
            continue
    missing = []
    for literal in narrated_decimals(markdown):
        if literal in allowed:
            continue
        value = float(literal)
        unit = _resolution(literal)
        if "e" in literal.lower():
            slack = 1.0 + 1e-9
            matched = any(
                abs(number - value) <= 0.5 * unit * slack and resolution <= unit * slack
                for number, resolution in printed
            )
        else:
            tolerance = 0.5 * unit + 1e-12
            matched = any(abs(number - value) <= tolerance for number, _ in printed)
        if not matched:
            missing.append(literal)
    return missing

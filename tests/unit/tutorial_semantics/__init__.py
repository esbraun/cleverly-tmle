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
``5.114e-04``.  It does not read a decimal inside inline code.  A key of the form
``"<markdown cell>: <literal>"`` exempts the literal in that cell only, which the section-scoped
check (:func:`scoped_narration_mismatches`) enforces.  ``DERIVED_DECIMALS`` maps such a key to a
printed number it is computed from, as ``(code cell, printed text, kind)``: ``"percent"`` for a
share written as a percentage, ``"magnitude"`` for a signed number written as a size.  The check
then confirms that the code cell prints that text and that the derivation rounds to the literal.

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
from tests.notebook_freshness import FIGURE_REPRS
from tests.notebooks import code_cells
from tests.prose import markdown_cells

__all__ = [
    "EXAMPLES",
    "GO_DEEPER",
    "NOT_TUTORIALS",
    "PACKAGE",
    "REFIT_CALLS",
    "STATIC_ONLY",
    "RefitNamespace",
    "assert_protocol_recorded",
    "assert_weights_lesson",
    "callback",
    "callback_module",
    "callback_modules",
    "cap_moves",
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
#: one refits the estimator once per replicate, grid point, dropped covariate, or split seed
#: (``stability``, the conditional effect curve's refits under other cross-fitting splits).
REFIT_CALLS = frozenset({"benchmark", "refute", "simulated_confounding", "stability"})


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


#: The ``text/plain`` reprs of a matplotlib figure or axes.
_FIGURE_REPRS = FIGURE_REPRS


def _output_text(output: Mapping[str, Any]) -> str:
    """The readable text of one stored output: its stream, plain text, and HTML payloads.

    A ``text/plain`` figure repr such as ``<Figure size 800x400 with 1 Axes>`` is skipped when
    the same output carries an image, so a figure never becomes the last line of a cell.

    Parameters
    ----------
    output : Mapping[str, Any]
        One stored output of a code cell.

    Returns
    -------
    str
        The output's readable text.
    """
    parts = [_joined(output.get("text", ""))]
    data = output.get("data", {})
    has_image = any(mime.startswith("image/") for mime in data)
    for mime in ("text/plain", "text/html"):
        if mime not in data:
            continue
        text = _joined(data[mime])
        if mime == "text/plain" and has_image and text.lstrip().startswith(_FIGURE_REPRS):
            continue
        parts.append(text)
    return "\n".join(part for part in parts if part)


def figure_title(figure: Any) -> str:
    """The title of a figure that a notebook cell drew with :mod:`cleverly.plotting`.

    The runner closes every figure before a callback runs, and a closed figure still holds its
    text, so a callback asserts the question a figure asks instead of reading an image.

    Parameters
    ----------
    figure : Any
        A matplotlib figure bound to a name in the notebook namespace.

    Returns
    -------
    str
        The title as one line.
    """
    from cleverly.plotting._core import title_text

    return title_text(figure)


def stored_output(path: Path, cell_id: str) -> str:
    """The stored readable output of one code cell, found by its cell id.

    A callback uses this to compare a committed output against the fresh run.  The lookup by id
    is why the authoring guide asks for stable ids on the cells a test reads.
    """
    cells = [cell for cell in _notebook(path)["cells"] if cell.get("id") == cell_id]
    assert len(cells) == 1, f"{path.name} has {len(cells)} cell(s) with id {cell_id!r}"
    return "\n".join(_output_text(output) for output in cells[0].get("outputs", ()))


def adjustment_table(printed: str) -> dict[str, tuple[str, ...]]:
    """The "What each treatment time adjusts for" table of a printed summary, by treatment.

    Rows that wrap continue on indented lines; the names are split on commas after the join.
    """
    lines = printed.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("What each treatment time"))
    rows: dict[str, str] = {}
    key = ""
    for line in lines[start + 3 :]:
        if line.startswith("="):
            break
        if line[:1].isspace():
            rows[key] += " " + line.strip()
        else:
            key, _, rest = line.partition(" ")
            rows[key] = rest.strip()
    return {name: tuple(part.strip() for part in text.split(",")) for name, text in rows.items()}


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
    :func:`scoped_narration_mismatches` applies it per section, so a reading cannot borrow a
    number that only another section prints.

    A scientific literal such as ``5.114e-04`` has a tolerance of half a unit in the mantissa's
    last place, times ten to its exponent.  An output matches it only if it also shows at least
    that precision, in either notation: ``5.114e-04`` and ``0.0005114`` match, and ``0.0005``
    and ``5.1e-04`` do not.  A fixed-point literal keeps the absolute tolerance above and puts no
    precision condition on the output.

    A decimal the prose writes for another reason, such as a law's parameter, goes in the
    module's ``UNPRINTED_DECIMALS`` with its reason, keyed by its written form.  Printing the
    number is the better repair.
    """
    allowed = {key.rpartition(": ")[2] for key in (unprinted or {})}
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


#: The markdown cells that summarize the page, so their decimals may read any stored output of the
#: main path: the "At a glance" opening (``title``) and the conclusion (``interpret``), as the page
#: outline in ``docs/development/example-notebooks.md`` names them.  A summary that quotes a lesson
#: under "Go deeper" declares the quote in ``NARRATION_SOURCES``.
SUMMARY_CELLS = frozenset({"title", "interpret"})

_HEADING = re.compile(r"^(#{1,6})\s+\S")

#: The "Go deeper" heading that ends a tutorial's main path (any level; the word-budget test
#: requires exactly one, at level two).
GO_DEEPER = re.compile(r"^#{1,6}\s.*\bgo deeper\b", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclasses.dataclass(frozen=True)
class NarrationScope:
    """One stretch of a markdown cell and the code cells whose outputs it may quote.

    Parameters
    ----------
    cell : str
        The markdown cell id.
    text : str
        The stretch of the cell between two headings.
    sources : tuple[str, ...]
        The code cell ids in scope.
    """

    cell: str
    text: str
    sources: tuple[str, ...]


def narration_scopes(path: Path) -> list[NarrationScope]:
    """Split each markdown cell at its headings, and give each stretch its scope.

    A stretch belongs to the innermost section open where it stands, from a heading of any level
    to the next heading of the same or a higher level.  Its scope is the code cells that sit
    directly in that section, before or after it, together with the code cells that sit directly
    in each enclosing section (outside any of that section's subsections).  So each lesson under
    "Go deeper" (level three) is its own scope, and a reading cannot quote a number that only
    another lesson prints.  A summary cell (:data:`SUMMARY_CELLS`) reads every code cell of the
    main path, the cells before the :data:`GO_DEEPER` heading.
    """
    cells = _notebook(path)["cells"]
    # Each section: (level, parent index or None).  Section 0 is the page before any heading.
    sections: list[tuple[int, int | None]] = [(0, None)]
    stack = [0]
    direct: dict[int, list[str]] = {0: []}
    stretches: list[tuple[str, str, int]] = []
    main_path: list[str] = []
    deeper = False
    for cell in cells:
        if cell["cell_type"] == "code":
            direct[stack[-1]].append(cell["id"])
            if not deeper:
                main_path.append(cell["id"])
            continue
        if cell["cell_type"] != "markdown":
            continue
        lines: list[str] = []
        fenced = False
        for line in _joined(cell.get("source", "")).splitlines():
            if _FENCE.match(line):
                fenced = not fenced
            heading = None if fenced else _HEADING.match(line)
            if heading and GO_DEEPER.match(line):
                deeper = True
            if heading:
                if lines:
                    stretches.append((cell["id"], "\n".join(lines), stack[-1]))
                lines = []
                level = len(heading.group(1))
                while sections[stack[-1]][0] >= level:
                    stack.pop()
                sections.append((level, stack[-1]))
                stack.append(len(sections) - 1)
                direct[stack[-1]] = []
            lines.append(line)
        if lines:
            stretches.append((cell["id"], "\n".join(lines), stack[-1]))
    scopes = []
    for cell_id, text, section in stretches:
        if cell_id in SUMMARY_CELLS:
            scopes.append(NarrationScope(cell_id, text, tuple(main_path)))
            continue
        sources: list[str] = []
        index: int | None = section
        while index is not None:
            sources.extend(direct[index])
            index = sections[index][1]
        scopes.append(NarrationScope(cell_id, text, tuple(sources)))
    return scopes


#: How a ``DERIVED_DECIMALS`` entry computes its literal from the printed number.
DERIVATIONS = {"percent": lambda value: 100.0 * value, "magnitude": abs}


def split_unprinted(unprinted: Mapping[str, str] | None) -> tuple[set[str], dict[str, set[str]]]:
    """The page-wide literals of ``UNPRINTED_DECIMALS``, and the literals exempt per cell."""
    page: set[str] = set()
    by_cell: dict[str, set[str]] = {}
    for key in unprinted or {}:
        cell, separator, literal = key.rpartition(": ")
        if separator:
            by_cell.setdefault(cell, set()).add(literal)
        else:
            page.add(literal)
    return page, by_cell


def derived_mismatches(
    path: Path, derived: Mapping[str, tuple[str, str, str]] | None
) -> tuple[dict[str, set[str]], list[str]]:
    """The literals each cell derives from a printed number, and each entry that does not hold.

    An entry ``"cell: literal": (source, printed, kind)`` holds when the markdown cell writes the
    literal, the code cell ``source`` prints ``printed`` as one of its numbers, and
    ``DERIVATIONS[kind]`` of the printed number rounds to the literal.
    """
    cells = {cell["id"]: cell for cell in _notebook(path)["cells"]}
    by_cell: dict[str, set[str]] = {}
    problems = []
    for key, (source, printed, kind) in (derived or {}).items():
        cell, _, literal = key.rpartition(": ")
        markdown = cells.get(cell)
        code = cells.get(source)
        if markdown is None or markdown["cell_type"] != "markdown":
            problems.append(f"{key}: {cell} is not a markdown cell of the page")
            continue
        if literal not in narrated_decimals(_joined(markdown.get("source", ""))):
            problems.append(f"{key}: the cell no longer writes {literal}")
            continue
        output = "" if code is None else "\n".join(map(_output_text, code.get("outputs", ())))
        if printed not in _OUTPUT_NUMBER.findall(output):
            problems.append(f"{key}: {source} does not print {printed}")
            continue
        value = DERIVATIONS[kind](float(printed))
        if narration_mismatches(literal, repr(value)):
            problems.append(f"{key}: {kind} of {printed} is {value!r}, which does not round to it")
            continue
        by_cell.setdefault(cell, set()).add(literal)
    return by_cell, problems


def scoped_narration_mismatches(
    path: Path,
    unprinted: Mapping[str, str] | None = None,
    declared: Mapping[str, Mapping[str, str]] | None = None,
    derived: Mapping[str, tuple[str, str, str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Decimals that no output in their own scope prints, and stale declared cross-references.

    ``declared`` is a callback module's ``NARRATION_SOURCES``: for a markdown cell id, each
    literal the cell quotes from a code cell outside its scope, mapped to that code cell's id.
    An entry is stale when the markdown cell is gone, the literal is no longer in that cell, the
    named code cell does not print a match, or the named code cell is already in scope.  A
    cell-scoped ``UNPRINTED_DECIMALS`` key is stale when its cell no longer writes the literal, and
    a ``derived`` (``DERIVED_DECIMALS``) entry when :func:`derived_mismatches` refuses it.

    Returns
    -------
    tuple[list[str], list[str]]
        ``"cell: literal"`` for each unmatched decimal, and a description of each stale entry.
    """
    cells = _notebook(path)["cells"]
    outputs = {
        cell["id"]: "\n".join(_output_text(output) for output in cell.get("outputs", ()))
        for cell in cells
        if cell["cell_type"] == "code"
    }
    declared = {cell: dict(entries) for cell, entries in (declared or {}).items()}
    scopes = narration_scopes(path)
    page, by_cell = split_unprinted(unprinted)
    derived_by_cell, stale = derived_mismatches(path, derived)
    missing = []
    used: set[tuple[str, str]] = set()
    written: dict[str, set[str]] = {}
    for scope in scopes:
        written.setdefault(scope.cell, set()).update(narrated_decimals(scope.text))
        exempt = page | by_cell.get(scope.cell, set()) | derived_by_cell.get(scope.cell, set())
        printed = "\n".join(outputs[cell] for cell in scope.sources)
        for literal in narration_mismatches(scope.text, printed, dict.fromkeys(exempt, "")):
            source = declared.get(scope.cell, {}).get(literal)
            if (
                source is not None
                and source in outputs
                and source not in scope.sources
                and not narration_mismatches(literal, outputs[source])
            ):
                used.add((scope.cell, literal))
                continue
            missing.append(f"{scope.cell}: {literal}")
    for cell, literals in by_cell.items():
        stale.extend(
            f"UNPRINTED_DECIMALS {cell}: {literal} (the cell no longer writes it)"
            for literal in sorted(literals - written.get(cell, set()))
        )
    markdown = {scope.cell for scope in scopes}
    for cell, entries in declared.items():
        for literal, source in entries.items():
            if cell not in markdown:
                stale.append(f"{cell} is not a markdown cell of the page")
            elif (cell, literal) not in used:
                stale.append(
                    f"{cell}: {literal} from {source} (the literal is gone, the cell {source} does "
                    "not print it, or it is already in scope)"
                )
    return missing, stale


def cap_moves(namespace: Mapping[str, Any], column: str = "lower_bound") -> Any:
    """The "Weights and balance" lesson's cap curve: each estimate's move in standard errors.

    One row per estimand and one column per cap, as the lesson's ``caps`` cell prints it.
    """
    curve = namespace["cap_curve"]
    return curve.pivot(index="estimand", columns=column, values="move in SEs")


def assert_weights_lesson(namespace: Mapping[str, Any], *, feedback: bool | None) -> None:
    """Check the readings every "Weights and balance" lesson shares.

    ``feedback`` says whether the fit has feedback rows (``True``), has none because it is a
    survival fit (``False``), or has no balance table at all because its treatment is
    continuous (``None``).  Each lesson's own callback checks its page's numbers.
    """
    rows = namespace["weight_rows"]
    # "no weight hit the cap"
    assert (rows["share_truncated"] == 0).all()
    # "kish" counts fewer equally weighted rows than the fit holds.
    influence = namespace["influence_rows"].loc[namespace["final_estimates"]]
    assert (influence["kish"] < influence["n"]).all()
    # The replay at the fit's own cap reproduces the fit.
    curve = namespace["cap_curve"]
    fitted = curve.loc[curve["is_fitted_bound"].astype(bool), "move in SEs"]
    assert fitted.empty or fitted.abs().max() < 1e-9
    if feedback is None:
        return
    largest = namespace["balance_by_check"].groupby("diagnostic")[["before", "after"]].max()
    # "Weighting shrinks the largest gap" for the confounding and censoring checks.
    for check in ("confounding", "censoring"):
        assert largest.loc[check, "after"] < largest.loc[check, "before"]
    assert largest.loc["confounding", "after"] < 0.1
    if feedback:
        # The feedback rows "need not shrink": on these pages they grow, the code's own effect.
        assert largest.loc["feedback", "after"] > largest.loc["feedback", "before"]
    else:
        # "A survival fit has no feedback check".
        assert "feedback" not in largest.index

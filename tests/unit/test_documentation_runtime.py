r"""Every worked example in the reader-facing documentation actually runs.

**Compiling a block is not running it, and the gap is where the bugs were.**  Its sibling
:mod:`tests.unit.test_documentation_examples` compiles each ``python`` fence, which catches a
fence that is not Python at all.  It cannot catch a fence that is *valid* Python naming something
the package does not have, and that is the failure this module was added for.  Six shipped
examples were broken in exactly that way and passed every gate the repository had:

* ``estimate.se`` in the quickstart and ``point.se`` in the results guide -- the attribute is
  :attr:`~cleverly.inference.ParameterEstimate.std_error`, and both rendered as the first thing a
  new reader copies;
* ``diagnostics.support().summary()`` on a modified-treatment-policy result, where ``support()``
  returns a mapping of policy name to report rather than one report;
* the same call on a longitudinal result, where the three stage reports carry ``to_frame`` and no
  ``summary`` at all -- three consecutive lines, none of which could run;
* ``Stochastic(0.6, ...)``, which passes a float where a density *function* is required;
* a design stratifying on ``region`` without adjusting for it, which the data container refuses.

The last two are the argument for registering the *guide* pages and not only the self-contained
examples: both live on a page whose fences assume a ``study`` from the surrounding prose, so a
check limited to documents that build their own data would have run neither.

**What this checks and what it deliberately does not.**  The general gate is a smoke check on
*executability*: the assertion is that the block raises nothing.  A bounded gate runs every
tutorial under ``docs/examples`` at its documented size and replaces its smaller smoke pass.  A
tutorial is a Markdown page or an executed notebook, and a notebook's code cells run in order in
one namespace, offline, without a kernel.  Each tutorial's explicit callback lives in its own
module in :mod:`tests.unit.tutorial_semantics`, and checks identification metadata, display
identities, and the seeded relations that the page narrates.  A notebook tutorial also has its
narrated decimals compared against its stored outputs.  None of this turns one seeded example
into statistical evidence.
``docs/architecture-invariants.md`` keeps that rule, and method claims still need an ordinary fast
test or a registered study.

**The smoke blocks are shrunk, and only in two declared ways.**  :class:`Shrink` rewrites the
``n=`` argument of a ``make_*`` generator call and the ``density_bins=`` argument, and rewrites
nothing else.  Learners, fold counts, seeds, estimands and interventions run exactly as the reader
reads them, because those are what the example is *about* -- a rewrite that reached them would
leave this module checking a configuration nobody is shown.

**Documents are discovered, then executable Markdown is registered because it needs a prelude.**
A guide page picks up ``study`` or ``result`` from the surrounding prose rather than building it,
so :data:`PRELUDES` gives each one the names its fences assume. Every reader-facing Python fence
must have an entry. A new example therefore enters one of the two runtime gates through the
shared document set rather than through somebody remembering a directory or the current
notebook's name.

**A reader-facing notebook carries an execution stamp instead, and half of that stamp is
gated.** ``tests/notebooks.py`` says which half and why. The gated half covers the notebook's
own code cells and stored outputs, so this module asserts it equal. The recorded half names the
repository context, so this module asserts it present and well formed and never asserts it equal.
The stamp is not provenance-complete, and this module does not claim that it
is: it reaches no markdown cell, no cell metadata such as the ``tags`` ``myst-nb`` reads, and no
notebook metadata such as ``kernelspec``.
``python scripts/execute_notebook.py <path> --check`` covers what a digest comparison cannot: a
library change that moves a published number while every cell keeps its bytes.
"""

from __future__ import annotations

import ast
import importlib.util
import ipaddress
import re
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nbformat
import pytest

from tests.documents import NOTEBOOKS, READER_FACING, ROOT, python_blocks
from tests.notebooks import (
    GATED_DIGESTS,
    GENERATOR_MODULES,
    RECORDED_DIGESTS,
    RECORDED_IDENTITY,
    UNKNOWN,
    code_cells,
    notebook_execution_stamp,
)
from tests.unit.tutorial_semantics import (
    NOT_TUTORIALS,
    STATIC_ONLY,
    assert_protocol_recorded,
    callback,
    callback_module,
    callback_modules,
    changed_fields,
    covers,
    markdown_text,
    module_name,
    narrated_decimals,
    narrated_notebooks,
    narration_mismatches,
    notebook_code,
    stored_text,
    tutorials,
)

#: Small enough that the whole module is a fast-tier cost, large enough that a fit converges.
SMALL_N = 200

#: Conditional-density bins.  The documented value is tuned for a readable support report on a
#: few thousand rows; at :data:`SMALL_N` it is only a cost.
SMALL_BINS = 8

#: MyST forms that render Python but :func:`tests.documents.python_blocks` does not execute.
#: The runtime gate supports one house form, so a new alias must fail rather than bypass it.
ALTERNATE_PYTHON_BLOCK = re.compile(
    r"^(?:```(?:py|python3|ipython3)[ \t]*$|~~~(?:py|python)[ \t]*$|"
    r"```\{(?:code-block|code-cell)\}[ \t]+(?:python|ipython3)[ \t]*$|"
    r"\.\. code-block:: python[ \t]*$)",
    re.MULTILINE,
)


class Shrink(ast.NodeTransformer):
    """Replace two size arguments, and leave every other part of the example alone.

    ``n=`` is narrowed to calls named ``make_*`` on purpose: it is a common keyword, and a
    blanket rewrite would silently resize things like ``n_folds`` or a user's own helper.
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name is None and isinstance(node.func, ast.Name):
            name = node.func.id
        for keyword in node.keywords:
            if not isinstance(keyword.value, ast.Constant):
                continue
            if keyword.arg == "n" and name is not None and name.startswith("make_"):
                keyword.value = ast.copy_location(ast.Constant(SMALL_N), keyword.value)
            elif keyword.arg == "density_bins":
                keyword.value = ast.copy_location(ast.Constant(SMALL_BINS), keyword.value)
        return node


def shrunk(code: str, name: str) -> Any:
    """Compile one block with the two size rewrites applied."""
    tree = Shrink().visit(ast.parse(code))
    return compile(ast.fix_missing_locations(tree), name, "exec")


# ------------------------------------------------------------------------- the preludes

#: A point-treatment frame carrying every column the design guides name: a binary outcome so
#: ratio estimands are defined, plus the weight, cluster, stratum and continuous-dose roles.
_FRAME = """
import numpy as np
from cleverly.datasets import make_binary_outcome

frame, truth = make_binary_outcome(n=200, seed=101)
_rng = np.random.default_rng(101)
_n = len(frame)
frame = frame.assign(
    sampling_weight=_rng.uniform(0.5, 1.5, _n),
    household=_rng.integers(0, 40, _n),
    region=_rng.integers(0, 3, _n),
    dose=_rng.normal(2.0, 1.0, _n),
)
"""

_STUDY = (
    _FRAME
    + """
from cleverly import CausalStudy, PointTreatment

study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")
    ),
)
dose_study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y", treatment="dose", adjustment=("W1", "W2"), treatment_kind="continuous"
    ),
)
"""
)

_EFFECT = (
    _STUDY
    + """
from cleverly import ATE

effect = study.identify(ATE())
"""
)

#: ``CounterfactualMean`` rather than ``ATE`` so the result carries two parameters and the
#: guide's ``if len(names) >= 2`` contrast branch is reached rather than skipped.
_RESULT = (
    _STUDY
    + """
from cleverly import CounterfactualMean
from sklearn.linear_model import LinearRegression, LogisticRegression

result = study.estimate(
    CounterfactualMean(),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=3,
)
"""
)

#: ``docs/workflow.md`` narrates an applied study, so its columns are applied names rather than
#: a generator's.  Renaming here keeps the prose honest instead of rewriting the guide to say
#: ``W1``.
_WORKFLOW = (
    _FRAME
    + """
data = frame.rename(
    columns={"Y": "outcome", "A": "treatment", "W1": "age", "W2": "baseline_score", "W3": "site"}
)
"""
)

#: Document -> the code its fences assume was already run.  An empty string means the document
#: builds everything it uses.  Tutorials under ``docs/examples`` hold that standard without an
#: entry here: :data:`TUTORIALS` discovers them, so converting one to a notebook edits no
#: shared registry.
PRELUDES: dict[str, str] = {
    "README.md": "",
    "docs/getting-started/installation.md": "",
    "docs/getting-started/quickstart.md": "",
    "docs/technical-reference/dr-tmle/supported-estimands.md": "",
    "docs/technical-reference/validation-methods.md": "",
    "docs/user-guide/longitudinal.md": "",
    "docs/user-guide/data-design.md": _FRAME,
    "docs/user-guide/estimands.md": _STUDY,
    "docs/user-guide/methods-learners.md": _EFFECT,
    "docs/user-guide/results-assessment.md": _RESULT,
    "docs/workflow.md": _WORKFLOW,
}

TWINS_NOTEBOOK = ROOT / "docs/examples/twins-causal-inference.ipynb"


# ---------------------------------------------------------------- semantic tutorial assertions

#: Tutorial sources, Markdown or notebook.  Each has one reviewed callback module in
#: :mod:`tests.unit.tutorial_semantics`, and none needs a prelude.
TUTORIALS = tuple(tutorials())


def _tutorial_id(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_every_tutorial_has_one_semantic_assertion() -> None:
    """Every tutorial, and no index page, enters the documented-size gate through one module.

    The comparison runs in both directions.  A tutorial with no module never runs at its
    documented size, and a module with no tutorial is a callback that checks nothing.
    """
    stems = [path.stem for path in TUTORIALS]
    duplicated = sorted({stem for stem in stems if stems.count(stem) > 1})
    assert not duplicated, (
        f"tutorial(s) {duplicated} exist as both Markdown and a notebook. A converted tutorial "
        "replaces its Markdown source, so delete the .md file"
    )
    assert len(TUTORIALS) >= 9, "the tutorial discovery found fewer pages than the program has"
    expected = {module_name(stem) for stem in (*stems, *STATIC_ONLY)}
    present = callback_modules()
    assert present == expected, (
        f"tutorial(s) without a callback module: {sorted(expected - present)}; "
        f"callback module(s) without a tutorial: {sorted(present - expected)}"
    )
    for stem in stems:
        assert callable(getattr(callback_module(stem), "check", None)), (
            f"tests/unit/tutorial_semantics/{module_name(stem)}.py defines no check(namespace)"
        )
    # A static-only module is never executed, so a check there would read as a gate that runs.
    assert STATIC_ONLY <= NOT_TUTORIALS
    for stem in STATIC_ONLY:
        module = callback_module(stem)
        assert not hasattr(module, "check") and callable(module.check_stored), (
            f"tests/unit/tutorial_semantics/{module_name(stem)}.py must define check_stored() "
            "and no check(namespace), because no offline gate executes the notebook"
        )
    assert not {_tutorial_id(path) for path in TUTORIALS} & set(PRELUDES), (
        "a tutorial builds everything it uses, so it takes no PRELUDES entry"
    )
    # The exclusion is by name, so a renamed TWINS notebook must fail here rather than enter an
    # offline gate it cannot pass.
    assert TWINS_NOTEBOOK.is_file() and TWINS_NOTEBOOK.stem in NOT_TUTORIALS


def documented() -> set[str]:
    """Every reader-facing document that carries Python, as repository-relative posix paths."""
    return {path.relative_to(ROOT).as_posix() for path in READER_FACING if python_blocks(path)}


def test_every_documented_example_is_registered() -> None:
    """A new example is covered by discovery, not by being remembered.

    A notebook stores its Python in cells rather than in a fence, so it never enters
    :func:`documented` and needs no exemption here.  Its own gate is
    :func:`test_every_notebook_has_an_internally_consistent_execution_artifact`, parametrized over
    the same document set.  A tutorial is registered by discovery through :data:`TUTORIALS`.
    """
    unregistered = documented() - set(PRELUDES) - {_tutorial_id(path) for path in TUTORIALS}
    assert not unregistered, (
        f"reader-facing document(s) with Python and no runtime gate: {sorted(unregistered)}. "
        f"Add a PRELUDES entry for Markdown, or commit a stamped notebook artifact"
    )


def test_every_python_example_uses_the_executable_house_fence() -> None:
    """A MyST Python alias must not render while bypassing the runtime gate."""
    unsupported = [
        path.relative_to(ROOT).as_posix()
        for path in READER_FACING
        if path.suffix in {".md", ".rst"}
        and ALTERNATE_PYTHON_BLOCK.search(path.read_text(encoding="utf-8"))
    ]
    assert not unsupported, (
        f"Python blocks in {unsupported} bypass tests.documents.python_blocks; use ```python"
    )


@pytest.mark.parametrize(
    "opening",
    (
        "```py",
        "```python3",
        "```ipython3",
        "~~~python",
        "```{code-block} python",
        "```{code-cell} ipython3",
        ".. code-block:: python",
    ),
)
def test_each_rendered_python_alias_is_recognized_as_unsupported(opening: str) -> None:
    """The house-syntax guard needs a witness for every alias it refuses."""
    assert ALTERNATE_PYTHON_BLOCK.search(f"# Example\n{opening}\nprint(1)\n```")


def test_the_registry_names_real_documents() -> None:
    """The negative control: a rename would otherwise empty this module silently."""
    assert len(PRELUDES) >= 10
    assert all(path.is_file() for path in TUTORIALS)
    assert NOTEBOOKS, "the reader-facing notebook set is unexpectedly empty"
    for relative in PRELUDES:
        assert (ROOT / relative).is_file(), f"{relative} is registered but does not exist"


@pytest.mark.parametrize(
    "path",
    NOTEBOOKS,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_every_notebook_has_an_internally_consistent_execution_artifact(path: Path) -> None:
    """Every notebook's stamp still matches its successful stored code and output payload."""
    notebook = nbformat.read(path, as_version=4)
    code = code_cells(notebook)
    relative = path.relative_to(ROOT).as_posix()
    assert code, f"{relative} has no code cells"

    unexecuted = [cell["id"] for cell in code if cell.get("execution_count") is None]
    errors = [
        (cell["id"], output.get("ename"), output.get("evalue"))
        for cell in code
        for output in cell.get("outputs", ())
        if output.get("output_type") == "error"
    ]
    counts = [cell.get("execution_count") for cell in code]
    execution = notebook.get("metadata", {}).get("cleverly_execution", {})
    expected = notebook_execution_stamp(notebook, path)

    assert not unexecuted, f"unexecuted cell(s) in {relative}: {unexecuted}"
    assert not errors, f"error output(s) in {relative}: {errors}"
    assert counts == list(range(1, len(code) + 1)), (
        f"execution counts in {relative} are not contiguous: {counts}"
    )
    assert execution.get("schema_version") == expected["schema_version"], (
        f"{relative} carries stamp schema {execution.get('schema_version')!r} and this checkout "
        f"writes {expected['schema_version']}; restamp the notebook"
    )
    assert execution.get("command") == expected["command"], (
        f"{relative} records the command {execution.get('command')!r} and now sits at a path "
        f"whose command is {expected['command']!r}; restamp the notebook"
    )
    assert execution.get("gated") == expected["gated"], (
        f"{relative} changed after its execution stamp was written. Both digests here are "
        f"computed from the notebook alone, so this gate detects internal edits rather than "
        f"proving which process produced them. Run the command arguments {expected['command']}"
    )


@pytest.mark.parametrize(
    "path",
    NOTEBOOKS,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_every_notebook_records_its_repository_context(path: Path) -> None:
    """The recorded half fingerprints the run's repository context.

    Asserted present and well formed, and deliberately not asserted equal.  An equality gate
    over the shipped source tree and the dependency lock fails the default handoff gate on any
    one-character library edit, and the only repair is a networked re-execution that refits
    every estimator.  ``tests/studies/evidence/manifest.py`` takes the same position for a study
    manifest, and ``docs/development/method-benchmarking.md`` states the rule: re-execution is
    what keeps an artifact honest, and a hash comparison is not re-execution.

    These three digests fingerprint the run rather than the current working tree.  A stamp
    rewritten without a re-execution carries them forward unchanged.  Recomputing them would
    attach the current context to outputs that it did not produce.

    ``generator_files`` is the exception, and it is checked against the current definition
    rather than carried.  A digest is only recomputable by whoever still has the formula behind
    it, so a changed file set strands the value it produced: the stamp's first
    ``generator_sha256`` folded one file, the generator became two, and the stored digest then
    matched no checkout in history.  Nothing saw it, because a stranded digest and a live one
    are the same 64 characters.  Comparing the recorded set to :data:`GENERATOR_MODULES` turns
    that silence into an instruction.
    """
    notebook = nbformat.read(path, as_version=4)
    relative = path.relative_to(ROOT).as_posix()
    recorded = notebook.get("metadata", {}).get("cleverly_execution", {}).get("recorded", {})

    assert set(recorded) == set(RECORDED_DIGESTS) | set(RECORDED_IDENTITY), (
        f"{relative} records {sorted(recorded)} and the stamp defines "
        f"{sorted(set(RECORDED_DIGESTS) | set(RECORDED_IDENTITY))}; restamp the notebook"
    )
    malformed = sorted(
        field
        for field, digest in recorded.items()
        if field in RECORDED_DIGESTS and not re.fullmatch(r"[0-9a-f]{64}", str(digest))
    )
    assert not malformed, (
        f"{relative} records {malformed} as something other than a SHA-256 digest, so the "
        f"stamp no longer carries a valid repository-context fingerprint"
    )

    assert recorded["generator_files"] == list(GENERATOR_MODULES), (
        f"{relative} folded {recorded['generator_files']} into its generator digest and this "
        f"checkout defines {list(GENERATOR_MODULES)}. The current schema cannot interpret that "
        f"stored generator digest. Re-execute the notebook rather than carrying the value forward"
    )
    assert str(recorded["cleverly_commit"]) == UNKNOWN or re.fullmatch(
        r"[0-9a-f]{40}", str(recorded["cleverly_commit"])
    ), f"{relative} records a commit that is neither a revision nor {UNKNOWN!r}"
    assert isinstance(recorded["cleverly_version"], str) and recorded["cleverly_version"]
    if recorded["cleverly_commit"] == UNKNOWN:
        assert recorded["cleverly_worktree_clean"] is None
    else:
        assert isinstance(recorded["cleverly_worktree_clean"], bool)


def test_the_stamp_splits_into_a_gated_half_and_a_recorded_half() -> None:
    """The two halves are disjoint, and every digest belongs to exactly one of them.

    Without this, a digest moved from ``recorded`` to ``gated`` would keep passing:
    :func:`test_every_notebook_identifies_the_checkout_that_ran_it` would stop seeing it and
    report nothing missing.
    """
    stamp = notebook_execution_stamp(nbformat.read(TWINS_NOTEBOOK, as_version=4), TWINS_NOTEBOOK)

    assert not GATED_DIGESTS & RECORDED_DIGESTS
    assert not GATED_DIGESTS & RECORDED_IDENTITY
    assert not RECORDED_DIGESTS & RECORDED_IDENTITY
    assert set(stamp["gated"]) == set(GATED_DIGESTS)
    assert set(stamp["recorded"]) == set(RECORDED_DIGESTS) | set(RECORDED_IDENTITY)


@pytest.mark.parametrize("stem", sorted(STATIC_ONLY))
def test_every_static_only_notebook_keeps_its_pinned_outputs(stem: str) -> None:
    """A notebook the offline tier cannot execute keeps the stored outputs its readings rely on.

    The assertions live in the notebook's own module, so the author who owns the notebook owns
    them.  They read committed text only.
    """
    callback_module(stem).check_stored()


@pytest.mark.parametrize(
    "path",
    NOTEBOOKS,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_every_published_figure_has_a_non_image_companion(path: Path) -> None:
    """An image-only result cannot participate in the deterministic ``--check`` comparison."""
    notebook = nbformat.read(path, as_version=4)
    unsupported = []
    for cell in code_cells(notebook):
        has_image = any(
            mime.startswith("image/")
            for output in cell.get("outputs", ())
            for mime in output.get("data", {})
        )
        if not has_image:
            continue
        has_companion = any(
            str(output.get("text", "")).strip()
            or any(
                not mime.startswith("image/")
                and (
                    mime != "text/plain"
                    or not str(payload).lstrip().startswith(("<Figure", "Figure("))
                )
                for mime, payload in output.get("data", {}).items()
            )
            for output in cell.get("outputs", ())
        )
        if not has_companion:
            unsupported.append(cell["id"])

    assert not unsupported, (
        f"{path.relative_to(ROOT).as_posix()} has image-only cell(s) {unsupported}; publish the "
        "plotted values as text, HTML, or JSON so --check can compare them"
    )


def _run_document(
    relative: str,
    tmp_path: Path,
    monkeypatch: Any,
    *,
    transform: Callable[[str, str], Any],
    namespace_name: str,
) -> dict[str, Any]:
    """Run one document's prelude and fences, in order, in one returned namespace.

    The scratch directory matters: several examples end by calling ``result.save(...)``, and a
    check that littered the working tree would be its own kind of failure.

    Both gates use this harness. Their callers select disjoint documents and transforms: the
    smoke gate shrinks non-tutorial blocks, while the semantic gate compiles tutorials as written.
    ``namespace_name`` names the namespace each gate builds, so a traceback says which one ran.
    A tutorial has no :data:`PRELUDES` entry and runs with an empty prelude.
    """
    monkeypatch.chdir(tmp_path)
    document = ROOT / relative
    namespace: dict[str, Any] = {"__name__": namespace_name}
    exec(compile(PRELUDES.get(relative, ""), f"<prelude for {relative}>", "exec"), namespace)

    for line, code in python_blocks(document):
        name = f"{relative}:{line}"
        try:
            exec(transform(code, name), namespace)
        except Exception as error:  # pragma: no cover - the failure is the message
            pytest.fail(f"{name} raised {type(error).__name__}: {error}")
    return namespace


# ------------------------------------------------------------------- notebook tutorials offline

#: The name a notebook cell's last bare expression is routed through.  Dunder-wrapped so no
#: tutorial can collide with it.
DISPLAY_HOOK = "__cleverly_display__"


def _render(*values: Any) -> None:
    """Build the representations a kernel would publish, and discard them.

    A kernel renders a cell's last expression through ``repr`` and, where the object has one,
    ``_repr_html_``.  A library change that breaks either one breaks the published cell, so the
    offline gate calls both rather than only evaluating the expression.
    """
    for value in values:
        if value is None or isinstance(value, type):
            continue
        repr(value)
        html = getattr(value, "_repr_html_", None)
        if callable(html):
            html()


def displayed(code: str, name: str) -> Any:
    """Compile one notebook cell so its last bare expression is rendered, as a kernel does.

    A cell that is not plain Python raises :class:`SyntaxError` here.  IPython magics and shell
    escapes are the usual cause, and the offline gate has no kernel to run them.
    """
    tree = ast.parse(code, filename=name)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        hook = ast.Call(func=ast.Name(DISPLAY_HOOK, ast.Load()), args=[last.value], keywords=[])
        tree.body[-1] = ast.copy_location(ast.Expr(hook), last)
    return compile(ast.fix_missing_locations(tree), name, "exec")


def refuse_network(monkeypatch: Any) -> None:
    """Make any non-loopback socket connection raise for the rest of one test.

    Tutorials run in the offline fast tier, and a documentation build renders stored outputs
    without a network.  A tutorial that downloads data passes on a connected laptop and fails
    on a runner that blocks egress, so the gate refuses the connection everywhere.
    """
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _checked(sock: socket.socket, address: Any) -> None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return
        host = str(address[0])
        try:
            loopback = ipaddress.ip_address(host.split("%")[0]).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            raise OSError(f"the tutorial gate runs offline; a tutorial tried to reach {address!r}")

    def connect(sock: socket.socket, address: Any) -> None:
        _checked(sock, address)
        original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        _checked(sock, address)
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


def _headless_matplotlib(monkeypatch: Any) -> None:
    """Select the non-interactive backend, so ``plt.show()`` cannot open a window and block."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("Agg")


def _close_figures() -> None:
    """Release every figure a notebook opened, so the next test starts with none."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")


def _run_notebook(relative: str, tmp_path: Path, monkeypatch: Any) -> dict[str, Any]:
    """Run a notebook's code cells in order, in one namespace, without a kernel.

    Execution happens in a scratch directory with the Agg backend.  The namespace carries
    ``display`` because a kernel provides it as a builtin.
    """
    monkeypatch.chdir(tmp_path)
    _headless_matplotlib(monkeypatch)
    namespace: dict[str, Any] = {
        "__name__": "__doc_semantic_example__",
        DISPLAY_HOOK: _render,
        "display": _render,
    }
    try:
        for cell_id, code in notebook_code(ROOT / relative):
            name = f"{relative}:{cell_id}"
            try:
                compiled = displayed(code, name)
            except SyntaxError as error:
                pytest.fail(
                    f"{name} is not plain Python ({error}). The offline gate runs cells without "
                    "a kernel, so remove IPython magics and shell escapes"
                )
            try:
                exec(compiled, namespace)
            except Exception as error:  # pragma: no cover - the failure is the message
                pytest.fail(f"{name} raised {type(error).__name__}: {error}")
    finally:
        _close_figures()
    return namespace


@pytest.mark.parametrize("relative", sorted(PRELUDES), ids=lambda name: name)
def test_every_nonsemantic_example_runs(relative: str, tmp_path: Path, monkeypatch: Any) -> None:
    """Each registered non-tutorial document runs once at the shrunken size."""
    _run_document(
        relative,
        tmp_path,
        monkeypatch,
        transform=shrunk,
        namespace_name="__doc_example__",
    )


@pytest.mark.parametrize("path", TUTORIALS, ids=_tutorial_id)
def test_tutorial_semantics_at_documented_size(
    path: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    """Reviewed seeded tutorials satisfy their claims at the displayed sample size.

    A Markdown tutorial runs its fences and a notebook tutorial runs its code cells.  Both run
    offline, and both hand their namespace to the tutorial's own callback module.
    """
    relative = _tutorial_id(path)
    refuse_network(monkeypatch)
    if path.suffix == ".ipynb":
        namespace = _run_notebook(relative, tmp_path, monkeypatch)
    else:
        namespace = _run_document(
            relative,
            tmp_path,
            monkeypatch,
            transform=lambda code, name: compile(code, name, "exec"),
            namespace_name="__doc_semantic_example__",
        )
    callback(path.stem)(namespace)


@pytest.mark.parametrize("path", narrated_notebooks(), ids=_tutorial_id)
def test_every_narrated_decimal_matches_a_stored_output(path: Path) -> None:
    """Each decimal a notebook's prose writes is one of its stored outputs, rounded.

    The check covers each tutorial notebook and each :data:`STATIC_ONLY` notebook, because it
    needs no execution.

    The execution stamp ties the stored outputs to the code cells.  This ties the prose to the
    stored outputs, so a re-execution that moves a number fails here until the prose follows.
    It reads committed text only, so a platform difference in a fresh run cannot move it.
    """
    markdown = markdown_text(path)
    declared = getattr(callback_module(path.stem), "UNPRINTED_DECIMALS", {})
    unprinted: dict[str, str] = dict(declared)
    missing = narration_mismatches(markdown, stored_text(path), unprinted)
    assert not missing, (
        f"{_tutorial_id(path)} narrates {missing}, and no stored output prints a value that "
        "rounds to it. Print the number the prose reads, correct the prose, or give the reason "
        f"in UNPRINTED_DECIMALS in tests/unit/tutorial_semantics/{module_name(path.stem)}.py"
    )
    stale = sorted(set(unprinted) - set(narrated_decimals(markdown)))
    assert not stale, f"UNPRINTED_DECIMALS names {stale}, which the prose no longer writes"
    for literal, reason in unprinted.items():
        assert reason.strip(), f"UNPRINTED_DECIMALS gives no reason for {literal}"


def test_the_narration_check_reads_precision_and_refuses_a_moved_number() -> None:
    """The narration gate accepts a rounded value and refuses one a re-execution moved.

    Without the refusal, a helper that accepted everything would pass every notebook.
    """
    outputs = "estimate: -0.2634\nchange: +0.110\nCI=(-0.417, 0.417)\n"
    assert not narration_mismatches("The estimate is about -0.26. The bound is 0.42.", outputs)
    assert not narration_mismatches("The estimate is about \N{MINUS SIGN}0.26.", outputs)
    assert not narration_mismatches("The change is +0.11.", outputs)
    assert narration_mismatches("The estimate is about 0.26.", outputs) == ["0.26"]
    assert narration_mismatches("The change is -0.11.", outputs) == ["-0.11"]
    assert narration_mismatches("The estimate is about 0.25.", outputs) == ["0.25"]
    assert narration_mismatches("The estimate is -0.263.", outputs) == []
    assert narration_mismatches("The estimate is -0.264.", outputs) == ["-0.264"]
    prose = "Use `alpha=0.07`, R 4.5.2, and [a](https://example.org/v1.23/)."
    assert narrated_decimals(prose) == []
    assert not narration_mismatches("A law value 0.73.", outputs, {"0.73": "a law parameter"})


def test_the_shared_callback_helpers_refuse_what_they_must(tmp_path: Path) -> None:
    """Each helper every callback leans on fails on the case a callback relies on it to catch.

    A ``covers`` that returned ``True``, a ``changed_fields`` that returned an empty set, or a
    protocol check that read no output would pass every callback that uses it.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from cleverly.datasets import navigation_protocol

    estimate = SimpleNamespace(ci=(1.0, 2.0))
    assert covers(estimate, 1.0) and covers(estimate, 2.0) and covers((1.0, 2.0), 1.5)
    assert not covers(estimate, 2.5) and not covers((1.0, 2.0), 0.5)

    program = navigation_protocol()
    assert changed_fields(program, program) == set()
    moved = replace(program, horizon="60 days after discharge", interference_unit="Navigator team")
    assert changed_fields(moved, program) == {"horizon", "interference_unit"}

    notebook = tmp_path / "protocol.ipynb"
    notebook.write_text(
        '{"cells": [{"cell_type": "code", "id": "protocol", "source": "", "outputs": '
        f'[{{"output_type": "stream", "name": "stdout", "text": "{program.fingerprint}"}}]}}]}}',
        encoding="utf-8",
    )
    carried = SimpleNamespace(provenance=SimpleNamespace(protocol_fingerprint=program.fingerprint))
    assert assert_protocol_recorded(notebook, "protocol", program, carried) == program.fingerprint
    with pytest.raises(AssertionError, match="does not print the protocol fingerprint"):
        assert_protocol_recorded(notebook, "protocol", moved)
    stale = SimpleNamespace(provenance=SimpleNamespace(protocol_fingerprint=moved.fingerprint))
    with pytest.raises(AssertionError):
        assert_protocol_recorded(notebook, "protocol", program, carried, stale)


def test_the_offline_gate_renders_the_last_expression_and_refuses_the_network(
    monkeypatch: Any,
) -> None:
    """Both notebook-harness guards fail when their target behaviour appears."""

    class Broken:
        def _repr_html_(self) -> str:
            raise RuntimeError("the HTML representation broke")

    namespace: dict[str, Any] = {DISPLAY_HOOK: _render, "Broken": Broken}
    # A class is not rendered, and an assignment has no last expression to render.
    exec(displayed("value = Broken()\nBroken", "<witness>"), namespace)
    with pytest.raises(RuntimeError, match="HTML representation"):
        exec(displayed("value = Broken()\nvalue", "<witness>"), namespace)
    with pytest.raises(SyntaxError):
        displayed("%matplotlib inline\n", "<witness>")

    refuse_network(monkeypatch)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="runs offline"):
            probe.connect(("192.0.2.1", 80))
    finally:
        probe.close()

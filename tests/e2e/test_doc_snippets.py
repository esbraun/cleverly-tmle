r"""The documentation's examples, run.

``ruff format`` reaches inside the fenced ``python`` blocks of every markdown file in the
tree, so a snippet here is *syntactically* valid and formatted -- and that is the whole of
what anything checked about it until now.  :mod:`tests.unit.test_documentation_links` makes
the argument for why that is not enough, about anchors, and it transfers word for word: an
example that stopped working renders as an ordinary example.  It is only on running it that
the reader finds out, and by then they are debugging their own script against a page that
was wrong before they opened it.

**Separate tiers, because the halves cost three orders of magnitude apart.**

* :func:`test_every_python_block_parses` compiles every block and runs nothing.  It costs
  milliseconds, it is in the fast tier, and it catches the class of breakage that a rename
  in ``src`` cannot cause but a careless edit to the prose can;
* modular ``docs`` tests execute one affected H2/H3 section and its declared setup blocks
  on pull requests;
* ``slow`` blocks are statistical evidence and the complete ``docs_full`` reading-order
  transcript is a manual validation gate. These are real fits at the sizes the guide quotes.

The manual transcript still executes a document as a document: one namespace, blocks in
reading order. Modular tests deliberately use fresh namespaces, making every dependency
reviewable in the document's ``doc-section`` metadata.

What this does **not** do is check output.  A block showing a printed table would need its
numbers pinned, and those move with every learner and seed; the claim here is that the code
in the documentation runs, not that a number in the prose is still the number.  Pinning the
numbers is what the oracle laws are for, on quantities chosen to be exact.
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

from cleverly import LTMLE, TMLE
from cleverly.estimators.base import TMLEResult
from cleverly.fluctuation.submodel import SUBMODEL_BUILDERS
from cleverly.sensitivity.api import SensitivityAnalysis
from cleverly.targets import TARGETS
from cleverly.validation import CoverageStudy
from cleverly.validation.api import ValidationSuite
from tests.doc_sections import (
    CATALOGUE,
    DOCUMENTS,
    ROOT,
    DocBlock,
    DocSection,
    all_blocks,
    all_sections,
    dependency_blocks,
    validate,
)
from tests.parallel import available_cores

SECTIONS = all_sections()
BLOCKS = all_blocks()


def blocks_of(document: Path) -> list[str]:
    return [block.code for block in BLOCKS if block.document == document]


def catalogue_reason(document: Path, index: int) -> str | None:
    """The reason on a block's ``<!-- catalogue: ... -->`` marker, or ``None``."""
    return next(
        block.catalogue_reason
        for block in BLOCKS
        if block.document == document and block.index == index
    )


#: The documents that carry any Python at all, computed rather than listed so that a new
#: guide is covered by existing here rather than by being remembered.
WITH_CODE = [document for document in DOCUMENTS if blocks_of(document)]


def test_the_extractor_found_something() -> None:
    """The one way a check over a discovered set fails open."""
    assert WITH_CODE, "no markdown file in the tree has a ```python block; check BLOCK"
    assert sum(len(blocks_of(document)) for document in WITH_CODE) >= 20


def test_every_python_block_has_execution_metadata() -> None:
    """A new fence must not silently fall outside both modular and transcript execution."""
    written = sum(document.read_text(encoding="utf-8").count("```python") for document in DOCUMENTS)
    assert len(BLOCKS) == written


def test_section_metadata_is_well_formed() -> None:
    assert validate(SECTIONS) == []


def test_the_catalogue_marker_is_recognised_where_it_is_written() -> None:
    """Counts the markers in the *text* against the ones the parser attaches to a block.

    Without this the marker fails open twice over: a comment the regex does not match means
    ``catalogue_reason`` returns ``None``, so :func:`test_every_catalogue_block_names_methods_that_exist`
    iterates over nothing and passes, *and* the ``docs`` tier goes on to execute a block that
    was declared unexecutable.  Both halves look green.  That is what happened -- the marker
    was written with a blank line before its fence and the pattern required adjacency, and
    the failure surfaced eighteen minutes into a full-transcript run rather than here.
    """
    written = sum(
        document.read_text(encoding="utf-8").count(f"<!-- {CATALOGUE}") for document in WITH_CODE
    )
    attached = sum(
        1
        for document in WITH_CODE
        for index in range(len(blocks_of(document)))
        if catalogue_reason(document, index) is not None
    )
    assert attached == written, (
        f"{written} catalogue markers are written in the documentation and {attached} were "
        f"attached to a block. A marker the parser misses is not a skipped check, it is a "
        f"check that silently covers nothing while the block runs anyway"
    )
    # Equality alone is symmetric, so a marker misspelt in *both* counts reads as 0 == 0 and
    # passes -- verified by misspelling it and watching this test stay green. The floor is
    # what makes the check non-vacuous, exactly as in `test_the_extractor_found_something`.
    assert written >= 1, (
        "no catalogue marker is written anywhere. Either the convention has been removed -- "
        "in which case CATALOGUE and its checks should go with it -- or a marker has been "
        "misspelt and every block it exempted is now being executed"
    )


@pytest.mark.parametrize("document", WITH_CODE, ids=lambda path: str(path.relative_to(ROOT)))
def test_every_python_block_parses(document: Path) -> None:
    """Free, and in the fast tier, because a syntax error in a guide should not wait a day.

    ``ruff format`` already reformats these blocks, so in practice it catches the same
    thing -- but it catches it only when someone runs the formatter over the whole tree.
    This has been a live trap before: a formatter release
    began formatting markdown and turned CI red with no commit to blame.  This is the same
    guarantee under the test suite's own control.
    """
    for index, block in enumerate(blocks_of(document)):
        name = f"{document.relative_to(ROOT)}#{index}"
        try:
            compile(block, name, "exec")
        except SyntaxError as error:  # pragma: no cover - the failure is the message
            pytest.fail(f"{name} does not parse: {error}")


#: Blocks that do **not** run today, keyed by ``(document, index)``, each with the reason.
#: A registry rather than a skip, and bidirectional: a row whose block has started working
#: is a failure, because a dead exemption reads as load-bearing -- the same rule
#: :mod:`tests.unit.test_estimator_contract` applies to its non-participants.
#:
#: Each row is a **defect in the documentation**, not a limitation of this module.  A guide
#: is read as a script even when it is written as an anthology, so a block that cannot run
#: after the ones above it is a page that does not work for the reader who follows it.
#:
#: **It is empty, and that is the intended steady state.**  The first sweep of
#: ``docs/user-guide.md`` found eight broken blocks and they were fixed rather than listed:
#: three sensitivity calls naming parameters their fit had not requested, a ``delta=`` fit
#: whose ``Delta`` column the prose only *described*, a ``DRTMLE(...)`` whose literal
#: ``...`` was a placeholder, an ``rr`` on a gaussian fit, a survey fit whose weight and
#: PSU columns did not exist, and an MNAR tilt with no missingness.  One of them could not
#: be fixed in the guide at all -- ``MultiArmDGP`` had no correct binomial law to move the
#: E-value onto -- so that one became a change to ``src`` and a test beside it.
#:
#: A row here is therefore an admission, not a mechanism: it says a page is known wrong and
#: is staying wrong for now.  Prefer fixing the page.
KNOWN_BROKEN: dict[tuple[str, int], str] = {}


#: Receivers a catalogue block may name, and the class each one is.  A catalogue naming
#: something not in here fails rather than passing vacuously -- the check has to be able to
#: say *what* it verified, or it verifies nothing.
RECEIVERS: dict[str, type] = {
    "res.sensitivity": SensitivityAnalysis,
    "res.validation": ValidationSuite,
    "res": TMLEResult,
}


def _receiver_path(node: ast.AST) -> str | None:
    """``res.sensitivity`` from the ``res.sensitivity.evalue`` attribute node, or ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def named_attributes(block: str) -> list[tuple[str, str]]:
    """``(receiver, attribute)`` for every attribute a block reads off a known receiver."""
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(block)):
        if not isinstance(node, ast.Attribute):
            continue
        receiver = _receiver_path(node.value)
        if receiver in RECEIVERS:
            found.append((receiver, node.attr))
    return found


@pytest.mark.parametrize("document", WITH_CODE, ids=lambda path: str(path.relative_to(ROOT)))
def test_every_catalogue_block_names_methods_that_exist(document: Path) -> None:
    """A catalogue is not executed, so this is what stands in for executing it.

    Some blocks in the guides *enumerate* an API rather than work an example -- the
    Sensitivity section lists ``positivity()``, ``truncation_curve(mechanism=True)``,
    ``evalue()`` and ``missingness_tilt()`` together, and no single fit supports all four:
    one needs a missingness mechanism, one a binary outcome.  Giving that block four fits
    would turn a five-line list into a page of setup and would say nothing a reader wanted.

    So a ``<!-- catalogue: <reason> -->`` marker exempts the block from execution, and in
    exchange every method it names on a known receiver must exist on that class.  That
    catches the regression the block is actually exposed to -- a renamed or removed method
    -- without a fit, and it runs in the **fast** tier, where a catalogue's cost belongs.

    **What this deliberately does not check is liveness**, and the boundary is worth being
    exact about.  Every other registry added alongside this module fails when an exemption
    goes stale; here the equivalent would be "this block would now run", which can only be
    established by running it -- the cost the marker exists to avoid.  A marker is therefore
    a claim a reviewer makes and this test cannot audit. What it *can* audit, and does, is
    that the claim is not empty: a catalogue naming no known receiver fails.
    """
    for index, block in enumerate(blocks_of(document)):
        reason = catalogue_reason(document, index)
        if reason is None:
            continue
        name = f"{document.relative_to(ROOT)}#{index}"
        assert len(reason) > 20, f"{name} is marked catalogue with no reason: {reason!r}"
        named = named_attributes(block)
        assert named, (
            f"{name} is marked catalogue but names no attribute on any of "
            f"{sorted(RECEIVERS)}, so this check would pass without verifying anything. "
            f"Either it is not a catalogue, or RECEIVERS needs the receiver it uses"
        )
        missing = [
            f"{receiver}.{attribute}"
            for receiver, attribute in named
            if not hasattr(RECEIVERS[receiver], attribute)
        ]
        assert missing == [], f"{name} names methods that no longer exist: {missing}"


#: The ``n_jobs`` default this tier sets on each class for the duration of a run.  Every one
#: of these defaults to ``1`` and threads it into
#: :func:`cleverly.utils.parallel.map_parallel`; the guide's examples never pass it, because
#: a reader would not.
#:
#: **All three are raised, including the one that contains the others, and that is measured
#: rather than assumed.**  A :class:`~cleverly.validation.CoverageStudy` holds estimators --
#: the guide builds one as ``estimator=lambda: TMLE(...)`` -- so raising both nests, and
#: joblib does not collapse that: ``effective_n_jobs(16)`` inside a worker still reports 16.
#: The demand really does multiply.
#:
#: It is still the right setting here, and the alternative was tried:
#:
#: * **study parallel, folds nested** -- the whole tier ran in **17:33**, against a 40:48
#:   baseline with everything serial;
#: * **study serial, folds parallel** -- the "obviously safer" split -- left that one block
#:   running past **ten minutes on its own**, and it had been finishing inside the total
#:   above.  Two hundred replicates strictly in sequence, each paying loky's spawn-based
#:   pool start-up for ten folds, is a worse trade than nesting: on this platform a pool
#:   costs more to create than a fold costs to run.
#:
#: So the rule is not "never nest".  It is that the *outer* loop over independent replicates
#: is the one worth having, and the inner pools it spawns are short-lived enough that the
#: product is a peak rather than a sustained load.  Change this only with the pair of
#: measurements above retaken.
PARALLEL_CLASSES: tuple[type, ...] = (TMLE, LTMLE, CoverageStudy)


def _parallel_defaults(cores: int) -> tuple[tuple[type, str, int], ...]:
    return tuple((cls, "n_jobs", cores) for cls in PARALLEL_CLASSES)


@pytest.fixture
def cores_for_the_examples() -> Iterator[int]:
    r"""Raise the default ``n_jobs`` for the run, and put it back.

    A selected module or one document transcript is sequential. Xdist can balance multiple
    selected modules or documents but cannot split an individual example, so the useful
    budget for its nuisance folds remains inward.

    **Patched here rather than written into the guide**, and that is the point of doing it
    in a fixture: the examples have to keep showing what a reader actually runs, and a
    reader does not pass ``n_jobs``.  Only the *default* moves, so a block that passed one
    would keep it.

    Legitimate only because :mod:`tests.unit.test_parallel_invariance` pins that ``n_jobs``
    does not move a number -- point estimates, influence curves and nuisance predictions
    bit for bit.  If that ever goes red this fixture comes out; a faster wrong answer is
    not the trade being made.

    The budget comes from :func:`tests.parallel.available_cores`, which reads what the
    *process* may use rather than what the machine has -- a container's CFS quota is
    invisible to ``os.cpu_count()`` and this tier runs in one.
    """
    cores = available_cores()
    # `n_jobs` is keyword-only on all three, so the default lives in ``__kwdefaults__`` and
    # a copy of that mapping is the whole of what has to be restored.
    saved = [(cls, dict(cls.__init__.__kwdefaults__)) for cls in PARALLEL_CLASSES]
    try:
        for cls, name, value in _parallel_defaults(cores):
            cls.__init__.__kwdefaults__[name] = value
        yield cores
    finally:
        for cls, defaults in saved:
            cls.__init__.__kwdefaults__.update(defaults)


def _selected(section: DocSection, request: pytest.FixtureRequest) -> bool:
    requested = set(request.config.getoption("--doc-section") or ())
    active = request.config._doc_selection_active  # type: ignore[attr-defined]
    return not active or section.section_id in requested


def _run(blocks: tuple[DocBlock, ...], namespace: dict[str, object]) -> None:
    for block in blocks:
        if block.catalogue_reason is not None:
            continue  # checked statically above, in the fast tier
        relative = block.document.relative_to(ROOT)
        key = (relative.as_posix(), block.index)
        known = KNOWN_BROKEN.get(key)
        first_line = next(
            (line for line in block.code.splitlines() if line.strip()), "(empty)"
        ).strip()
        with warnings.catch_warnings():
            # The guides deliberately show configurations that warn -- weak overlap is
            # a worked example there. A warning is the documentation working, not the
            # documentation failing, and `filterwarnings = error::RuntimeWarning` in
            # pyproject would otherwise make it the latter.
            warnings.simplefilter("ignore")
            try:
                exec(compile(block.code, f"{relative}#{block.index}", "exec"), namespace)
            except Exception as error:
                if known is None:
                    pytest.fail(
                        f"{relative} block {block.index} ({block.block_id}) raised "
                        f"{type(error).__name__}: {error}\n  the block begins: {first_line}"
                    )
                continue
        assert known is None, (
            f"{relative} block {block.index} is listed in KNOWN_BROKEN and now runs. Delete "
            f"its row -- an exemption nobody removed reads as a standing decision. The "
            f"row said: {known}"
        )


class _IsolatedRun:
    @staticmethod
    def run(
        blocks: tuple[DocBlock, ...],
        name: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        targets, builders = dict(TARGETS), dict(SUBMODEL_BUILDERS)
        try:
            _run(blocks, {"__name__": f"doc:{name}"})
        finally:
            TARGETS.clear()
            TARGETS.update(targets)
            SUBMODEL_BUILDERS.clear()
            SUBMODEL_BUILDERS.update(builders)


FAST_SECTIONS = tuple(
    section
    for section in SECTIONS
    if any(block.tier == "fast" and block.catalogue_reason is None for block in section.blocks)
)
SLOW_BLOCKS = tuple(block for block in BLOCKS if block.tier == "slow")


@pytest.mark.docs
@pytest.mark.parametrize("section", FAST_SECTIONS, ids=lambda section: section.section_id)
def test_documentation_section_runs(
    section: DocSection,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cores_for_the_examples: int,
) -> None:
    """One selected section runs with only its declared setup closure."""
    if not _selected(section, request):
        pytest.skip("section was not selected by --doc-section")
    prerequisites = dependency_blocks(section, SECTIONS)
    own = tuple(
        block for block in section.blocks if block.tier == "fast" and block.catalogue_reason is None
    )
    _IsolatedRun.run(prerequisites + own, section.section_id, tmp_path, monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize("slow_block", SLOW_BLOCKS, ids=lambda block: block.block_id)
def test_statistical_documentation_block_runs(
    slow_block: DocBlock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cores_for_the_examples: int,
) -> None:
    section = next(item for item in SECTIONS if item.section_id == slow_block.section_id)
    prerequisites = dependency_blocks(section, SECTIONS)
    earlier = tuple(
        block
        for block in section.blocks
        if block.index < slow_block.index and block.catalogue_reason is None
    )
    _IsolatedRun.run(
        prerequisites + earlier + (slow_block,), slow_block.block_id, tmp_path, monkeypatch
    )


@pytest.mark.docs_full
@pytest.mark.parametrize("document", WITH_CODE, ids=lambda path: str(path.relative_to(ROOT)))
def test_document_runs_as_a_complete_transcript(
    document: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cores_for_the_examples: int,
) -> None:
    """Manual gate: every ordinary and statistical block in literal reading order."""
    blocks = tuple(block for block in BLOCKS if block.document == document)
    _IsolatedRun.run(blocks, str(document.relative_to(ROOT)), tmp_path, monkeypatch)

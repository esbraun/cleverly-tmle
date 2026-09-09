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

**What this checks and what it deliberately does not.**  This is a smoke check on
*executability*: the assertion is that the block raises nothing.  Nothing here asserts an
estimate, an interval, or a diagnostic verdict, and nothing here is statistical evidence --
``docs/architecture-invariants.md`` keeps that rule, and behaviour shown in a guide still has to
be covered by an ordinary fast test or a registered study. A documented example that runs is a
much weaker claim than a documented example that is right, and only the weaker one is made here.

**The blocks are shrunk, and only in two declared ways.**  :class:`Shrink` rewrites the ``n=``
argument of a ``make_*`` generator call and the ``density_bins=`` argument, and rewrites nothing
else.  Learners, fold counts, seeds, estimands and interventions run exactly as the reader reads
them, because those are what the example is *about* -- a rewrite that reached them would leave
this module checking a configuration nobody is shown.

**Documents are discovered, then executable Markdown is registered because it needs a prelude.**
A guide page picks up ``study`` or ``result`` from the surrounding prose rather than building it,
so :data:`PRELUDES` gives each one the names its fences assume. Every reader-facing Python fence
must have an entry. A new example therefore enters one of the two runtime gates through the
shared document set rather than through somebody remembering a directory or the current
notebook's name.

**A reader-facing notebook carries an execution stamp instead, and half of that stamp is
gated.** ``tests/notebooks.py`` says which half and why. The gated half covers the notebook's
own code cells and stored outputs, so this module asserts it equal. The recorded half names the
checkout that ran the notebook, so this module asserts it present and well formed and never
asserts it equal. The stamp is not provenance-complete, and this module does not claim that it
is: it reaches no markdown cell, no cell metadata such as the ``tags`` ``myst-nb`` reads, and no
notebook metadata such as ``kernelspec``.
``python scripts/execute_notebook.py <path> --check`` covers what a digest comparison cannot: a
library change that moves a published number while every cell keeps its bytes.
"""

from __future__ import annotations

import ast
import re
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

#: Small enough that the whole module is a fast-tier cost, large enough that a fit converges.
SMALL_N = 200

#: Conditional-density bins.  The documented value is tuned for a readable support report on a
#: few thousand rows; at :data:`SMALL_N` it is only a cost.
SMALL_BINS = 8


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
#: builds everything it uses, which is the standard the examples section is held to.
PRELUDES: dict[str, str] = {
    "README.md": "",
    "docs/examples/point-treatment-tmle.md": "",
    "docs/examples/cross-fitting.md": "",
    "docs/examples/collaborative-tmle.md": "",
    "docs/examples/dr-tmle.md": "",
    "docs/examples/interventions.md": "",
    "docs/examples/survey-nonresponse.md": "",
    "docs/examples/longitudinal-tmle.md": "",
    "docs/examples/longitudinal-survival.md": "",
    "docs/examples/msm-projections.md": "",
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


def documented() -> set[str]:
    """Every reader-facing document that carries Python, as repository-relative posix paths."""
    return {path.relative_to(ROOT).as_posix() for path in READER_FACING if python_blocks(path)}


def test_every_documented_example_is_registered() -> None:
    """A new example is covered by discovery, not by being remembered.

    A notebook stores its Python in cells rather than in a fence, so it never enters
    :func:`documented` and needs no exemption here.  Its own gate is
    :func:`test_every_notebook_is_a_current_successfully_executed_artifact`, parametrized over
    the same document set.
    """
    unregistered = documented() - set(PRELUDES)
    assert not unregistered, (
        f"reader-facing document(s) with Python and no runtime gate: {sorted(unregistered)}. "
        f"Add a PRELUDES entry for Markdown, or commit a stamped notebook artifact"
    )


def test_the_registry_names_real_documents() -> None:
    """The negative control: a rename would otherwise empty this module silently."""
    assert len(PRELUDES) >= 10
    assert NOTEBOOKS, "the reader-facing notebook set is unexpectedly empty"
    for relative in PRELUDES:
        assert (ROOT / relative).is_file(), f"{relative} is registered but does not exist"


@pytest.mark.parametrize(
    "path",
    NOTEBOOKS,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_every_notebook_is_a_current_successfully_executed_artifact(path: Path) -> None:
    """Every published notebook stores successful outputs from its own code cells."""
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
        f"{relative} stores outputs its own code cells did not produce. Both digests here are "
        f"computed from the notebook alone, so no library or lockfile edit can move them. "
        f"Run {expected['command']}"
    )


@pytest.mark.parametrize(
    "path",
    NOTEBOOKS,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_every_notebook_identifies_the_checkout_that_ran_it(path: Path) -> None:
    """The recorded half names the run, and no library edit can fail this.

    Asserted present and well formed, and deliberately not asserted equal.  An equality gate
    over the shipped source tree and the dependency lock fails the default handoff gate on any
    one-character library edit, and the only repair is a networked re-execution that refits
    every estimator.  ``tests/studies/evidence/manifest.py`` takes the same position for a study
    manifest, and ``docs/development/method-benchmarking.md`` states the rule: re-execution is
    what keeps an artifact honest, and a hash comparison is not re-execution.

    These three digests name a run rather than a working tree, so a stamp rewritten without a
    re-execution carries them forward unchanged.  Recomputing them would make the notebook
    claim that a checkout which never ran it produced its outputs.

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
        f"stamp no longer identifies the checkout that ran the notebook"
    )

    assert recorded["generator_files"] == list(GENERATOR_MODULES), (
        f"{relative} folded {recorded['generator_files']} into its generator digest and this "
        f"checkout defines {list(GENERATOR_MODULES)}. The stored digest was produced by a "
        f"formula that no longer exists, so it names no checkout. Re-execute the notebook "
        f"rather than carrying the value forward"
    )
    assert str(recorded["cleverly_commit"]) == UNKNOWN or re.fullmatch(
        r"[0-9a-f]{40}", str(recorded["cleverly_commit"])
    ), f"{relative} records a commit that is neither a revision nor {UNKNOWN!r}"
    assert isinstance(recorded["cleverly_version"], str) and recorded["cleverly_version"]


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


def test_the_twins_notebook_retains_its_specific_evidence_outputs() -> None:
    """The TWINS artifact retains the figures and ordinary TMLE interval its prose interprets."""
    notebook = nbformat.read(TWINS_NOTEBOOK, as_version=4)
    code = code_cells(notebook)
    figures = [
        output
        for cell in code
        for output in cell.get("outputs", ())
        if "image/png" in output.get("data", {})
    ]
    comparison_cell = next(cell for cell in code if cell["id"] == "comparison-figure")
    comparison_text = "".join(
        text
        for output in comparison_cell.get("outputs", ())
        for text in output.get("data", {}).get("text/plain", ())
    )
    ordinary_tmle_row = next(
        line for line in comparison_text.splitlines() if "ordinary package TMLE" in line
    )

    assert len(figures) >= 3, "the TWINS notebook lost one or more evidence figures"
    assert "NaN" not in ordinary_tmle_row, (
        "the ordinary package TMLE lost its confidence interval in the comparison figure"
    )


@pytest.mark.parametrize("relative", sorted(PRELUDES), ids=lambda name: name)
def test_every_example_runs(relative: str, tmp_path: Path, monkeypatch: Any) -> None:
    """Each document's fences, in order, in one namespace, from a scratch directory.

    The scratch directory matters: several examples end by calling ``result.save(...)``, and a
    check that littered the working tree would be its own kind of failure.
    """
    monkeypatch.chdir(tmp_path)
    document = ROOT / relative
    namespace: dict[str, Any] = {"__name__": "__doc_example__"}
    exec(compile(PRELUDES[relative], f"<prelude for {relative}>", "exec"), namespace)

    for line, code in python_blocks(document):
        name = f"{relative}:{line}"
        try:
            exec(shrunk(code, name), namespace)
        except Exception as error:  # pragma: no cover - the failure is the message
            pytest.fail(f"{name} raised {type(error).__name__}: {error}")

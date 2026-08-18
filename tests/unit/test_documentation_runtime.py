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
be covered by an ordinary fast test or a named slow study.  A documented example that runs is a
much weaker claim than a documented example that is right, and only the weaker one is made here.

**The blocks are shrunk, and only in two declared ways.**  :class:`Shrink` rewrites the ``n=``
argument of a ``make_*`` generator call and the ``density_bins=`` argument, and rewrites nothing
else.  Learners, fold counts, seeds, estimands and interventions run exactly as the reader reads
them, because those are what the example is *about* -- a rewrite that reached them would leave
this module checking a configuration nobody is shown.

**Documents are registered rather than discovered, because most of them need a prelude.**  A
guide page picks up ``study`` or ``result`` from the surrounding prose rather than building it,
so :data:`PRELUDES` gives each one the names its fences assume.  Registering is not optional:
:func:`test_every_documented_example_is_registered` fails when a page with Python appears in the
reader-facing set and is not listed, since the alternative is a new guide quietly escaping the
check -- which is the same failure :mod:`tests.documents` exists to prevent.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from tests.documents import ROOT, python_blocks

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
    "docs/examples/point-treatment.md": "",
    "docs/examples/interventions.md": "",
    "docs/examples/longitudinal.md": "",
    "docs/examples/assessment.md": "",
    "docs/getting-started/installation.md": "",
    "docs/getting-started/quickstart.md": "",
    "docs/user-guide/longitudinal.md": "",
    "docs/user-guide/data-design.md": _FRAME,
    "docs/user-guide/estimands.md": _STUDY,
    "docs/user-guide/methods-learners.md": _EFFECT,
    "docs/user-guide/results-assessment.md": _RESULT,
    "docs/workflow.md": _WORKFLOW,
}

#: The reader-facing set this module is responsible for.  ``docs/development/`` and the design
#: documents are excluded by not being here: they argue about work that is proposed or historical,
#: and several show an API on purpose that no longer exists.
REACHED = ("docs/examples", "docs/getting-started", "docs/user-guide", "docs/workflow.md")

#: Declared, not silent.  ``docs/user-guide.md`` is the legacy recipe compendium that the site
#: routes into the user guide.  It is one worked recipe per capability and would need a prelude
#: per section; it is left out here rather than half-covered, and that is a gap in this module
#: rather than a property of the document.
EXCLUDED = frozenset({"docs/user-guide.md"})


def documented() -> set[str]:
    """Every reader-facing document that carries Python, as repository-relative posix paths."""
    found = set()
    for reached in REACHED:
        target = ROOT / reached
        paths = [target] if target.is_file() else sorted(target.glob("*.md"))
        for path in paths:
            if python_blocks(path):
                found.add(path.relative_to(ROOT).as_posix())
    return found


def test_every_documented_example_is_registered() -> None:
    """A new guide is covered by existing, not by being remembered."""
    unregistered = documented() - set(PRELUDES) - EXCLUDED
    assert not unregistered, (
        f"reader-facing document(s) with python fences and no entry in PRELUDES: "
        f"{sorted(unregistered)}. Add a prelude (or an empty one, if the document builds "
        f"everything it uses), or list it in EXCLUDED with the reason"
    )


def test_the_registry_names_real_documents() -> None:
    """The negative control: a rename would otherwise empty this module silently."""
    assert len(PRELUDES) >= 10
    for relative in [*PRELUDES, *EXCLUDED]:
        assert (ROOT / relative).is_file(), f"{relative} is registered but does not exist"


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

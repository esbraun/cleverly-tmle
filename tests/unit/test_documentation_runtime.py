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
*executability*: the assertion is that the block raises nothing.  A second, bounded gate runs four
reviewed tutorials at their documented sizes.  Its explicit callbacks check identification
metadata, display identities, and the seeded relations that those pages narrate.  This does not
turn one seeded example into statistical evidence. ``docs/architecture-invariants.md`` keeps that
rule, and method claims still need an ordinary fast test or a registered study.

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
repository context, so this module asserts it present and well formed and never asserts it equal.
The stamp is not provenance-complete, and this module does not claim that it
is: it reaches no markdown cell, no cell metadata such as the ``tags`` ``myst-nb`` reads, and no
notebook metadata such as ``kernelspec``.
``python scripts/execute_notebook.py <path> --check`` covers what a digest comparison cannot: a
library change that moves a published number while every cell keeps its bytes.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
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


# ---------------------------------------------------------------- semantic tutorial assertions


def _missing_outcome_semantics(namespace: dict[str, Any]) -> None:
    """The survey question reports the observation factor its fitted score uses.

    The same documented-size run also witnesses the tutorial's interval and sensitivity claims.
    """
    effect = namespace["effect"]
    fitted = namespace["full"]
    expression = effect.functional.expression
    assumptions = " ".join(effect.identification.assumptions).lower()
    nuisances = effect.identification.required_nuisances
    summary = effect.summary()

    assert "responded=1" in expression
    assert "missingness at random" in assumptions
    assert "response positivity" in assumptions
    assert "missingness_mechanism" in nuisances
    assert fitted.nuisance.missingness is not None
    assert expression in summary
    assert "missingness_mechanism" in summary
    assert "E_W[E(Y | A=a, W)] and the declared smooth contrast" not in summary

    truth = namespace["truth"]["ate"]
    complete_case = namespace["complete_case"]["ate"]
    full = fitted["ate"]
    assert complete_case.psi > truth
    assert not complete_case.ci[0] <= truth <= complete_case.ci[1]
    assert full.ci[0] <= truth <= full.ci[1]

    mild_truth = namespace["mild_truth"]["ate"]
    mild = namespace["mild"]["ate"]
    assert mild.ci[0] <= mild_truth <= mild.ci[1]

    box_study = namespace["box_study"]
    box_method = namespace["box_method"]
    risk_ratio = box_study.identify(namespace["RiskRatio"](reference=0)).estimate(
        method=box_method
    )["rr"]
    odds_ratio = box_study.identify(namespace["OddsRatio"](reference=0)).estimate(
        method=box_method
    )["or"]
    assert odds_ratio.psi > risk_ratio.psi > 1.0
    assert namespace["tipping_gamma"] == pytest.approx(1.30, abs=0.02)


def _multi_arm_identification_semantics(namespace: dict[str, Any]) -> None:
    """The arm-mean question names its actual support instead of a binary surrogate."""
    effect = namespace["arms"]
    fitted = namespace["arm_result"]
    levels = tuple(namespace["study"].data.treatment_levels)
    positivity = " ".join(
        assumption
        for assumption in effect.identification.assumptions
        if "positivity" in assumption.lower()
    )
    summary = effect.summary()

    assert len(levels) == 3, "the tutorial no longer witnesses multi-arm identification"
    assert fitted.nuisance.propensity.values.shape[1] == len(levels)
    for level in levels:
        assert str(level) in positivity
    assert "treatment_mechanism" in summary
    assert "P(A = 1 | W)" not in summary
    assert "both counterfactual means" not in summary

    population = namespace["population"]
    assert population[0] < population[1] < population[2]
    assert population[2] - population[1] > population[1] - population[0]

    projection = namespace["projection"]
    trend = namespace["trend_result"]
    for name, target in zip(
        ("msm[(intercept)]", "msm[assigned contacts]"), projection, strict=True
    ):
        assert trend[name].ci[0] <= target <= trend[name].ci[1]

    saturated = namespace["saturated_result"]
    mappings = {
        "msm[(intercept)]": namespace["arm_result"]["ey[low]"],
        "msm[medium vs low]": namespace["arm_contrasts"]["ate[medium vs low]"],
        "msm[high vs low]": namespace["arm_contrasts"]["ate[high vs low]"],
    }
    for name, counterpart in mappings.items():
        coefficient = saturated[name]
        assert coefficient.psi == pytest.approx(counterpart.psi, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(
            coefficient.influence_curve,
            counterpart.influence_curve,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(coefficient.ci, counterpart.ci, rtol=1e-12, atol=1e-12)


def _survival_output_semantics(namespace: dict[str, Any]) -> None:
    """The survival view and incidence total keep their public numerical meanings."""
    result = namespace["exit_result"]
    risk = result.curve(scale="risk")
    survival = namespace["survival_curve"]
    keys = set(result.estimates)
    assert len(risk) == len(survival) > 0

    # Three columns, three separate jobs, so all three are pinned.  ``view`` records the
    # view the caller asked for, ``scale`` keeps the ``to_frame()`` vocabulary that tells a
    # level row from a contrast row, and ``parameter`` carries the key the row came from.
    # One column carrying two of those meanings is what this frame stopped doing.
    assert set(risk["view"]) == {"risk"}
    assert set(survival["view"]) == {"survival"}
    assert set(risk["scale"]) == {"level"}
    assert set(survival["scale"]) == {"level"}
    assert list(risk["parameter"]) == list(survival["parameter"])
    assert set(survival["parameter"]) <= keys
    # The risk view reports what the fit estimated, so its ``estimand`` is its own key.
    assert list(risk["estimand"]) == list(risk["parameter"])
    # The survival view reports a quantity the fit never estimated under that name, which
    # is the reason ``parameter`` exists: ``result[row.parameter]`` is defined here and
    # ``result[row.estimand]`` is not.
    assert not set(survival["estimand"]) & keys
    assert all(str(name).startswith("risk_regimen[") for name in risk["estimand"])
    assert all(str(name).startswith("survival_regimen[") for name in survival["estimand"])
    assert not any(str(name).startswith("risk_regimen[") for name in survival["estimand"])
    np.testing.assert_allclose(survival["psi"], 1.0 - risk["psi"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(survival["ci_lower"], 1.0 - risk["ci_upper"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(survival["ci_upper"], 1.0 - risk["ci_lower"], rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(survival["std_err"], risk["std_err"])

    contrast = namespace["exit_contrast"]
    contrast_keys = set(contrast.estimates)
    risk_difference = contrast.curve(scale="risk")
    survival_difference = contrast.curve(scale="survival")
    assert set(risk_difference["view"]) == {"risk"}
    assert set(survival_difference["view"]) == {"survival"}
    assert set(risk_difference["scale"]) == {"difference"}
    assert set(survival_difference["scale"]) == {"difference"}
    # A contrast is the same parameter up to a sign under either view, so here the
    # ``estimand`` is the key and the two columns agree.  That is the half of the rule the
    # level rows above break, and pinning both halves is what makes the rule visible.
    assert list(survival_difference["estimand"]) == list(survival_difference["parameter"])
    assert set(survival_difference["parameter"]) <= contrast_keys
    assert all(str(name).startswith("ate_regimen[") for name in survival_difference["estimand"])
    assert not any(
        str(name).startswith(("risk_regimen[", "survival_regimen["))
        for name in survival_difference["estimand"]
    )
    np.testing.assert_allclose(
        survival_difference["psi"], -risk_difference["psi"], rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        survival_difference["ci_lower"], -risk_difference["ci_upper"], rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        survival_difference["ci_upper"], -risk_difference["ci_lower"], rtol=0.0, atol=1e-14
    )
    np.testing.assert_array_equal(survival_difference["std_err"], risk_difference["std_err"])

    retention = survival.set_index(["regimen", "time"])
    assert retention.loc[("always", 2), "psi"] > retention.loc[("never", 2), "psi"]
    exit_t1 = contrast["ate_regimen[always vs never @ t=1]"]
    exit_t2 = contrast["ate_regimen[always vs never @ t=2]"]
    assert exit_t1.psi < 0.0 and exit_t2.psi < 0.0
    assert abs(exit_t2.psi) > abs(exit_t1.psi)

    events = namespace["event_result"]
    relapse_t1 = events["ate_regimen[always vs never, relapse @ t=1]"]
    relapse_t2 = events["ate_regimen[always vs never, relapse @ t=2]"]
    assert relapse_t1.psi < 0.0 and relapse_t2.psi < 0.0
    assert abs(relapse_t2.psi) < abs(relapse_t1.psi)
    assert relapse_t2.ci[0] <= 0.0 <= relapse_t2.ci[1]

    event_truth = namespace["event_truth"]
    assert event_truth["ate_regimen[always vs never, relapse @ t=1]"] < 0.0
    assert event_truth["ate_regimen[always vs never, relapse @ t=2]"] > 0.0

    death_t1 = events["ate_regimen[always vs never, death @ t=1]"]
    death_t2 = events["ate_regimen[always vs never, death @ t=2]"]
    assert death_t1.psi < 0.0 and death_t2.psi < 0.0
    assert abs(death_t2.psi) > abs(death_t1.psi)

    exit_support = namespace["exit_diagnostics"].report("support").to_frame()
    event_support = namespace["event_diagnostics"].report("support").to_frame()
    assert len(exit_support) == 6
    assert len(event_support) == 12

    competing = namespace["event_levels"]
    # The page says a fit that declares two or more causes refuses the survival view, and
    # this fit declares two.  A one-cause fit is the other side of that rule and is pinned
    # in ``tests/e2e/test_ltmle.py``; here the declaration is what the assertion reads.
    assert len(competing.config.causes) == 2
    with pytest.raises(ValueError, match="not all-cause survival"):
        competing.curve(scale="survival")

    # The reported standard error is derived here from the influence curves alone, and not
    # from ``influence_covariance``, which is the helper the reported line itself calls: a
    # recomputation through that helper would assert it against itself and would pass with
    # the helper wrong.  This fit declares no cluster column, so the independent formula is
    # the iid one, and the assertion below states that rather than reading it off the fit.
    # ``tests/e2e/test_ltmle.py`` carries the clustered derivation and its own witness that
    # the two formulas separate, so repeating it here would only copy that test.
    totals = namespace["incidence_totals"]
    index = competing.parameter_index or {}
    assert competing.data.cluster is None
    for row in totals.itertuples(index=False):
        names = [
            name
            for name, (regimen, _cause, horizon) in index.items()
            if regimen == row.regimen and horizon == row.time
        ]
        assert len(names) == len(competing.config.causes)
        curve = np.sum(np.column_stack([competing[name].influence_curve for name in names]), axis=1)
        expected = float(np.sqrt(np.var(curve, ddof=1) / competing.data.n))
        # The nonzero control: the reported value divided by another root n is what the
        # ``/n`` bug produced, and it fails this line if it comes back.
        old_extra_scaling = expected / np.sqrt(competing.data.n)
        assert row.std_err == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert row.std_err != pytest.approx(old_extra_scaling, rel=1e-6, abs=0.0)


def _collaborative_summary_semantics(namespace: dict[str, Any]) -> None:
    """A selector report describes its loss without assigning causal roles to omissions."""
    selection = namespace["selection"]
    assert selection.selected_covariates == ()

    summary = selection.summary().lower()
    assert "cross-validated" in summary and "loss" in summary
    assert "does not determine why" in summary
    # Whole words, because a substring test reads a footer that says "covariance" as a
    # footer that says "variance", and the report would then fail for a word it never used.
    for unsupported in ("bias", "variance", "confounder", "instrument"):
        assert not re.search(rf"\b{unsupported}\b", summary)

    weak_selection = namespace["weak_selection"]
    assert "baseline_readiness" in weak_selection.selected_covariates
    assert "queue_lottery_position" not in weak_selection.selected_covariates
    assert namespace["weak_collaborative"]["ate"].std_error < (
        0.5 * namespace["weak_plain"]["ate"].std_error
    )
    assert namespace["nuisance"].treatment_role == "collaborative_working_model"


#: These tutorials make seeded output claims that need more than the shrunken smoke gate. The
#: narrower map makes each semantic assertion an explicit review decision instead of a prose
#: heuristic.
TUTORIAL_SEMANTIC_ASSERTIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "docs/examples/collaborative-tmle.md": _collaborative_summary_semantics,
    "docs/examples/longitudinal-survival.md": _survival_output_semantics,
    "docs/examples/msm-projections.md": _multi_arm_identification_semantics,
    "docs/examples/survey-nonresponse.md": _missing_outcome_semantics,
}


def test_every_tutorial_semantic_assertion_names_one_reviewed_runtime_example() -> None:
    """The bounded semantic gate names only documents the smoke gate already registers.

    A copied list of the registry's own keys asserts nothing, so this checks the one
    relation the registry does not state about itself: every semantic callback names a
    document :data:`PRELUDES` knows how to run.  Which tutorials belong here is a review
    decision, and ``docs/architecture-invariants.md`` records it.
    """
    assert set(TUTORIAL_SEMANTIC_ASSERTIONS) <= set(PRELUDES)


def documented() -> set[str]:
    """Every reader-facing document that carries Python, as repository-relative posix paths."""
    return {path.relative_to(ROOT).as_posix() for path in READER_FACING if python_blocks(path)}


def test_every_documented_example_is_registered() -> None:
    """A new example is covered by discovery, not by being remembered.

    A notebook stores its Python in cells rather than in a fence, so it never enters
    :func:`documented` and needs no exemption here.  Its own gate is
    :func:`test_every_notebook_has_an_internally_consistent_execution_artifact`, parametrized over
    the same document set.
    """
    unregistered = documented() - set(PRELUDES)
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
    module_name: str,
) -> dict[str, Any]:
    """Run one document's prelude and fences, in order, in one returned namespace.

    The scratch directory matters: several examples end by calling ``result.save(...)``, and a
    check that littered the working tree would be its own kind of failure.

    The two gates below differ in ``transform`` alone, which is the whole difference between
    them: the smoke gate shrinks each block and the semantic gate compiles it as written.
    ``module_name`` names the namespace each gate builds, so a traceback says which one ran.
    """
    monkeypatch.chdir(tmp_path)
    document = ROOT / relative
    namespace: dict[str, Any] = {"__name__": module_name}
    exec(compile(PRELUDES[relative], f"<prelude for {relative}>", "exec"), namespace)

    for line, code in python_blocks(document):
        name = f"{relative}:{line}"
        try:
            exec(transform(code, name), namespace)
        except Exception as error:  # pragma: no cover - the failure is the message
            pytest.fail(f"{name} raised {type(error).__name__}: {error}")
    return namespace


@pytest.mark.parametrize("relative", sorted(PRELUDES), ids=lambda name: name)
def test_every_example_runs(relative: str, tmp_path: Path, monkeypatch: Any) -> None:
    """Every registered document's fences run at the shrunken size and raise nothing."""
    _run_document(
        relative,
        tmp_path,
        monkeypatch,
        transform=shrunk,
        module_name="__doc_example__",
    )


@pytest.mark.parametrize("relative", sorted(TUTORIAL_SEMANTIC_ASSERTIONS), ids=lambda name: name)
def test_tutorial_semantics_at_documented_size(
    relative: str, tmp_path: Path, monkeypatch: Any
) -> None:
    """Reviewed seeded tutorials satisfy their claims at the displayed sample size."""
    namespace = _run_document(
        relative,
        tmp_path,
        monkeypatch,
        transform=lambda code, name: compile(code, name, "exec"),
        module_name="__doc_semantic_example__",
    )
    TUTORIAL_SEMANTIC_ASSERTIONS[relative](namespace)

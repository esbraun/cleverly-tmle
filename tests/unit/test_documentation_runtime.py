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
*executability*: the assertion is that the block raises nothing.  A second, bounded gate runs nine
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

    # "About a quarter of patients never return the 30-day survey."
    assert 0.70 < float(namespace["frame"]["responded"].mean()) < 0.80

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
    # "near gamma=1.3", and "at most about 0.29 of the score range" is expit(gamma) - 0.5.
    assert namespace["tipping_gamma"] == pytest.approx(1.30, abs=0.03)
    # The refusal is the attributable fraction's missing-outcome refusal, not another error.
    assert "does not yet support PointTreatment(missingness=...)" in namespace["refusal"]

    # The curve's gamma=0 row is the MAR estimate, and its interval reuses the MAR standard error.
    curve = namespace["missingness_curve"]
    mar = curve[curve["gamma"] == 0.0]
    assert float(mar["psi"].iloc[0]) == pytest.approx(full.psi, rel=1e-9)
    assert (curve["std_err"] == full.std_error).all()
    # Positive gamma lowers the navigation-arm mean, so the ATE falls along the grid.
    assert curve["psi"].is_monotonic_decreasing


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
    # The page refuses text cadence labels. Only the MSM.linear refusal may satisfy it.
    assert "reads the treatment level" in namespace["refusal"]

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

    line = namespace["design"] @ projection
    residual = line - population
    assert abs(residual[1]) > 0.15, "the contact mapping no longer witnesses misspecification"
    assert 0.17 < abs(residual[1]) < 0.21, "the prose says the line misses medium by about 0.19"
    # "Well below" the medium interval: the gap is measured at about 3.5 standard errors.
    medium = namespace["arm_result"]["ey[medium]"]
    gap = medium.ci[0] - namespace["estimated_line"][1]
    assert gap > 2.0 * medium.std_error, (
        "on this draw the estimated line no longer falls well below the medium interval"
    )


def _survival_output_semantics(namespace: dict[str, Any]) -> None:
    """The survival view, delta-method differences, and competing-risk narrative hold."""
    result = namespace["exit_result"]
    risk = result.curve(scale="risk")
    survival = namespace["survival_curve"]
    keys = set(result.estimates)
    assert len(risk) == len(survival) == 4

    # ``estimand`` names the survival quantity and ``parameter`` names the risk behind it.
    assert set(risk["view"]) == {"risk"} and set(survival["view"]) == {"survival"}
    assert set(risk["scale"]) == {"level"} and set(survival["scale"]) == {"level"}
    assert list(risk["parameter"]) == list(survival["parameter"])
    assert set(survival["parameter"]) <= keys
    assert not set(survival["estimand"]) & keys
    assert all(str(name).startswith("survival_regimen[") for name in survival["estimand"])
    np.testing.assert_allclose(survival["psi"], 1.0 - risk["psi"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(survival["ci_lower"], 1.0 - risk["ci_upper"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(survival["ci_upper"], 1.0 - risk["ci_lower"], rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(survival["std_err"], risk["std_err"])

    # "both risk differences are negative, and the 60-day difference is larger" (ratio 1.37).
    exit_differences = namespace["exit_differences"]
    exit_t1, exit_t2 = exit_differences[1], exit_differences[2]
    assert exit_t1.psi < -0.05
    assert exit_t2.psi < 1.15 * exit_t1.psi
    retention = survival.set_index(["regimen", "time"])
    assert retention.loc[("always", 2), "psi"] > retention.loc[("never", 2), "psi"] + 0.05
    exit_truth = namespace["exit_truth"]
    assert (
        exit_truth["ate_regimen[always vs never @ t=2]"]
        < exit_truth["ate_regimen[always vs never @ t=1]"]
        < 0.0
    )
    # "26% ... within 30 days and 46% within 60 days under no navigation" are exact truths.
    assert exit_truth["risk_regimen[never @ t=1]"] == pytest.approx(0.26, abs=0.005)
    assert exit_truth["risk_regimen[never @ t=2]"] == pytest.approx(0.46, abs=0.006)

    # "the same estimate and standard error as a separate RegimeContrast fit". The refit reuses
    # the page's clustered study and method, and costs one more backward pass.
    from cleverly import RegimeContrast

    contrast = (
        namespace["exit_study"]
        .identify(RegimeContrast({"always": 1, "never": 0}, reference="never", horizons=(1, 2)))
        .estimate(method=namespace["sequential"])
    )
    assert result.data.cluster is not None
    for horizon, estimate in exit_differences.items():
        fitted = contrast[f"ate_regimen[always vs never @ t={horizon}]"]
        assert estimate.psi == pytest.approx(fitted.psi, abs=1e-10)
        assert estimate.std_error == pytest.approx(fitted.std_error, rel=1e-8)

    # The renamed censoring columns gate the event in their own period.
    frame = namespace["exit_frame"]
    assert frame.loc[frame["tracked_p1"] == 0, "plan_exit_p1"].isna().all()
    assert frame.loc[frame["tracked_p1"] == 1, "plan_exit_p1"].notna().all()

    event_truth = namespace["event_truth"]
    # "60-day death risk is 21% under no navigation and 7% under navigation at both periods."
    assert event_truth["cif_regimen[never, death @ t=2]"] == pytest.approx(0.21, abs=0.006)
    assert event_truth["cif_regimen[always, death @ t=2]"] == pytest.approx(0.07, abs=0.006)
    events = namespace["event_differences"]
    readmission_t1, readmission_t2 = events["readmission", 1], events["readmission", 2]
    # "negative at 30 days and shrinks toward zero by 60 days" (measured ratio 0.21).
    assert readmission_t1.psi < -0.015
    assert abs(readmission_t2.psi) < 0.5 * abs(readmission_t1.psi)
    assert event_truth["ate_regimen[always vs never, relapse @ t=1]"] < 0.0
    assert event_truth["ate_regimen[always vs never, relapse @ t=2]"] > 0.0
    # "negative at both horizons and larger at 60 days" (measured ratio 1.55).
    death_t1, death_t2 = events["death", 1], events["death", 2]
    assert death_t1.psi < -0.03
    assert death_t2.psi < 1.25 * death_t1.psi

    # Death coded as censoring gives a larger reduction than the total effect, and it raises
    # never-plan readmission risk more than always-plan risk.
    eliminated = namespace["eliminated"]
    assert eliminated.data.censoring_names == ("alive_p1", "alive_p2")
    assert namespace["eliminated_t2"].psi < readmission_t2.psi - 0.01
    levels = namespace["event_levels"]
    raised_never = (
        eliminated["risk_regimen[never @ t=2]"].psi
        - levels["cif_regimen[never, readmission @ t=2]"].psi
    )
    raised_always = (
        eliminated["risk_regimen[always @ t=2]"].psi
        - levels["cif_regimen[always, readmission @ t=2]"].psi
    )
    assert raised_never > raised_always + 0.01 and raised_always > 0.0

    for assessment in (namespace["exit_assessment"], namespace["event_assessment"]):
        sensitivity = assessment.sensitivity.items
        assert sensitivity and all(item.status.value == "unavailable" for item in sensitivity)
        assert assessment.diagnostics["refute"].status.value == "unavailable"
    event_support = namespace["event_assessment"].report("support").to_frame()
    assert {"cause", "horizon"} <= set(event_support.columns)

    assert len(levels.config.causes) == 2
    with pytest.raises(ValueError, match="not all-cause survival"):
        levels.curve(scale="survival")

    # The reported total standard error is derived from the influence curves alone. This fit
    # declares no cluster, so the independent formula is the iid one.
    totals = namespace["incidence_totals"]
    index = levels.parameter_index or {}
    assert levels.data.cluster is None
    for row in totals.itertuples(index=False):
        names = [
            name
            for name, (regimen, _cause, horizon) in index.items()
            if regimen == row.regimen and horizon == row.time
        ]
        assert len(names) == len(levels.config.causes)
        curve = np.sum(np.column_stack([levels[name].influence_curve for name in names]), axis=1)
        expected = float(np.sqrt(np.var(curve, ddof=1) / levels.data.n))
        old_extra_scaling = expected / np.sqrt(levels.data.n)
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

    # The rename is load-bearing: a stale column name would make the exclusion vacuous.
    covariates = set(namespace["frame"].columns)
    assert {"baseline_readiness", "queue_lottery_draw", "social_support"} <= covariates
    weak_selection = namespace["weak_selection"]
    assert "baseline_readiness" in weak_selection.selected_covariates
    assert "queue_lottery_draw" not in weak_selection.selected_covariates
    assert namespace["weak_collaborative"]["ate"].std_error < (
        0.5 * namespace["weak_plain"]["ate"].std_error
    )
    assert namespace["nuisance"].treatment_role == "collaborative_working_model"

    # "More than 5% ... below 0.1, and more than 5% ... above 0.9" for the plain fit.
    overall = namespace["plain"].diagnostics.support().propensity_quantiles["overall"]
    assert overall[0.05] < 0.1 and overall[0.95] > 0.9
    # "its AUC is about 0.5" for the intercept-only working model. The calibration slope of a
    # constant model is undefined noise, so the page makes no claim about it.
    assert 0.45 < namespace["nuisance"]["propensity"].metrics["auc"] < 0.55


def _cross_fitting_semantics(namespace: dict[str, Any]) -> None:
    """The seeded relations docs/examples/cross-fitting.md narrates, at its documented size."""
    cross_fitted = namespace["cross_fitted"]["ate"]
    in_sample = namespace["in_sample"]["ate"]

    assert "stacked CV-TMLE" in namespace["cross_fitted"].summary()
    # "the two point estimates are close"; "less than a third" (measured ratio about 0.28).
    assert abs(in_sample.psi - cross_fitted.psi) < 0.1
    assert in_sample.std_error < cross_fitted.std_error / 3

    # "nearly identical" points and "about 1.7 times larger" standard error.
    ignoring = namespace["ignoring"]["ate"]
    clustered = namespace["clustered"]["ate"]
    assert abs(ignoring.psi - clustered.psi) < 0.02
    assert 1.5 < clustered.std_error / ignoring.std_error < 1.9
    assert namespace["clustered"].data.n_clusters == 200

    # "The new seed alone draws new folds."
    redrawn = namespace["redrawn"]
    assert (
        redrawn.provenance.fold_fingerprint != namespace["cross_fitted"].provenance.fold_fingerprint
    )

    # "warns that the out-of-fold propensity model is poorly calibrated" (slope about 0.49,
    # warning below 0.7); "about 1%" truncated; "about 25%"; in-sample "above 85%".
    diagnostics = namespace["diagnostics"]
    assert diagnostics["nuisance_models"].status.value == "warning"
    nuisance = diagnostics.report("nuisance_models")
    assert nuisance["propensity"].metrics["calibration_slope"] < 0.6
    support = diagnostics.report("support")
    assert 0.005 < support.truncated["fraction"] < 0.02
    ratios = [ess["ratio"] for ess in support.effective_sample_size.values()]
    assert 0.2 < min(ratios) < 0.3
    in_sample_support = namespace["in_sample"].diagnostics.support()
    in_sample_ratios = [ess["ratio"] for ess in in_sample_support.effective_sample_size.values()]
    assert min(in_sample_ratios) > 0.85

    spread = namespace["repeated_nuisance"].repeat_spread
    assert [row.n_repeats for row in spread] == [3]


def _drtmle_semantics(namespace: dict[str, Any]) -> None:
    """The DR-TMLE tutorial's seeded claims hold at the documented size.

    The page claims an exact reduction to the ordinary estimator, a close estimate and standard
    error on this draw, passing score and correction reports with no active truncation, and a
    reduced-regression table whose g_r1 fits all favor the spline candidate.
    """
    ordinary = namespace["ordinary"]["ate"]
    empty_guard = namespace["empty_guard"]["ate"]
    guarded = namespace["guarded"]["ate"]
    assert ordinary.psi == empty_guard.psi

    # "moves by less than one standard error, and the standard errors are similar". The lower
    # bound witnesses "the guarded estimate moves": about 0.4 SE on this draw.
    shift = abs(guarded.psi - ordinary.psi) / ordinary.std_error
    assert 0.1 < shift < 0.9
    assert 0.8 < guarded.std_error / ordinary.std_error < 1.25

    assessment = namespace["assessment"]
    corrections = assessment.report("corrections")
    assert corrections.passed
    assert corrections.contract == "theorem"  # "no truncation was active"
    assert assessment.report("score_equations").passed
    summary = assessment.summary()
    assert "omitted_confounding" in summary

    nuisance = assessment.report("nuisance_models")
    assert "look reasonable" in nuisance.summary()

    reduced = namespace["reduced"]
    assert set(reduced) == {"qr", "gr1", "gr2"}
    # Fails on a revert to plain linear reducers, which leave these diagnostics empty.
    assert reduced["gr1"] and all(fit.best == "spline" for fit in reduced["gr1"])


def _longitudinal_semantics(namespace: dict[str, Any]) -> None:
    """The two-decision tutorial keeps its named roles and its seeded narrative."""
    frame = namespace["frame"]
    # The page renames L2 by the signs of the law. Engagement must rise with discharge
    # navigation and with day-seven navigation among tracked patients, or the name is wrong.
    tracked = frame[frame["tracked_day7"] == 1]
    by_first = tracked.groupby("navigation_discharge")["engagement_day7"].mean()
    assert by_first[1.0] - by_first[0.0] > 0.5
    by_second = tracked.groupby("navigation_day7")["engagement_day7"].mean()
    assert by_second[1.0] > by_second[0.0]

    summary = namespace["effect"].summary()
    assert "sequential exchangeability" in summary and "sequential positivity" in summary
    assert "The protocol scores death before day 30 as not top box" in summary

    # "42% with no navigation and 78% with both offers" are quadrature truths, not draws.
    truth = namespace["truth"]
    assert truth["ey_regimen[never]"] == pytest.approx(0.42, abs=0.005)
    assert truth["ey_regimen[always]"] == pytest.approx(0.78, abs=0.005)

    target = namespace["target"]
    adjusted, baseline_only = namespace["adjusted"], namespace["baseline_only"]
    sequential = namespace["result"]["ate_regimen[always vs never]"]
    # "more than 0.1 below" (measured 0.137), "about 0.05 above" (measured 0.051), and
    # "within one standard error" (measured 0.43 standard errors).
    assert adjusted.psi < target - 0.1
    assert 0.03 < baseline_only.psi - target < 0.08
    assert abs(sequential.psi - target) < sequential.std_error

    rule = namespace["rule_result"]
    assert set(rule.estimates) == {"ate_regimen[continue if engaged vs never]"}
    shares = rule.diagnostics.support().to_frame().set_index(["regimen", "time"])
    assert shares.loc[("continue if engaged", 1), "share_assigned_1"] == 1.0
    assert 0.0 < shares.loc[("continue if engaged", 2), "share_assigned_1"] < 1.0

    assessment = namespace["assessment"]
    sensitivity = assessment.sensitivity.items
    assert sensitivity and all(item.status.value == "unavailable" for item in sensitivity)
    assert assessment.diagnostics["refute"].status.value == "unavailable"
    assert assessment.diagnostics["corrections"].status.value == "not_applicable"
    support = namespace["support"]
    # "the bound replaces no row": the largest weight (measured 33) is well under the cap of 100.
    assert (support["share_truncated"] == 0.0).all()
    assert (support["max_weight"] < 60.0).all()
    curve = namespace["curve"]
    assert list(curve["lower_bound"]) == [0.01, 0.05, 0.1]
    assert curve["truncated_score_cells"].iloc[0] == 0
    assert curve["truncated_score_cells"].iloc[-1] > 0
    # "less than 0.005, a small fraction of its standard error" (measured 0.0019 and 0.11 SE).
    movement = float(curve["delta_from_fitted"].abs().max())
    assert 0.0 < movement < 0.005
    assert movement < 0.3 * sequential.std_error


def _point_treatment_semantics(namespace: dict[str, Any]) -> None:
    """The ATE tutorial's protocol, failure mode, warnings, and sensitivity claims hold."""
    summary = namespace["effect"].summary()
    assert "Discharge-home order" in summary
    assert "scores death before day 30 as the worst transition score" in summary
    assert "stacked CV-TMLE" in namespace["result"].summary()
    spread_truth = namespace["spread_truth"]
    assert spread_truth["att"] > spread_truth["ate"] > spread_truth["atc"]
    assert spread_truth["atc"] < 0.25 * spread_truth["att"]
    # "misses by several times more than either fit with one flexible learner"; about 6x here.
    truth = namespace["truth"]["ate"]
    errors = {label: abs(point.psi - truth) for label, point in namespace["dr_points"].items()}
    one_flexible = max(errors["flexible Q, linear g"], errors["linear Q, flexible g"])
    assert errors["both linear"] > 3.0 * one_flexible
    assessment = namespace["assessment"]
    assert "nuisance_models" in {item.name for item in assessment.attention}
    ledger = assessment.to_frame()
    rows = ledger[ledger["check"] == "nuisance_models"]
    assert rows["detail"].str.contains("propensity is poorly calibrated").any()
    assert 0.0 < namespace["support"].truncated["fraction"] < 0.05
    # "about 0.26", at the worst-case alignment rho = 1 the prose names.
    assert 0.2 < namespace["robustness"]["rv"] < 0.33
    benchmark = namespace["benchmark"]
    assert benchmark.covariates == ("discharge_risk",)
    bounds = namespace["bounds"]
    assert (bounds.cf_y, bounds.cf_d, bounds.rho) == (benchmark.cf_y, benchmark.cf_d, 1.0)
    assert bounds.ci_lower > 0.0


def _intervention_axes_semantics(namespace: dict[str, Any]) -> None:
    """The three-axis tutorial's truths, support figures, and draw-specific claims hold."""
    offer_all, screen = namespace["truth"]["ate"], namespace["screen_truth"]
    assert 0.5 * offer_all < screen < offer_all
    assert "positivity *for the shifted dose*" in namespace["shift_effect"].summary()
    assert "*no positivity assumption*" in namespace["incremental_effect"].summary()
    warned = namespace["positivity_warnings"]
    assert len(warned) == 2
    assert "'+0.5 uncapped'" in warned[0] and "'+1.0 uncapped'" in warned[1]
    support = namespace["shift_assessment"].report("support")
    assert support["current practice"].ess_ratio == pytest.approx(1.0)
    capped = support["+0.5 capped at 5"]
    assert 0.015 < capped.capped_fraction < 0.035
    assert 0.35 < capped.ess_ratio < 0.55
    # "more than three times as wide as either +0.5 interval"; about 4.5x here.
    frame = namespace["shift_result"].to_frame().set_index("estimand")
    width = frame["ci_upper"] - frame["ci_lower"]
    wide = width["ate_shift[+1.0 uncapped vs current practice]"]
    assert wide > 3.0 * width["ate_shift[+0.5 capped at 5 vs current practice]"]
    assert wide > 3.0 * width["ate_shift[+0.5 uncapped vs current practice]"]
    # "a few percent", far below the exp(-1) a true normal density ratio would keep.
    assert 0.01 < support["+1.0 uncapped"].ess_ratio < 0.1
    assert support["+1.0 uncapped"].ess_ratio < 0.5 * np.exp(-1.0)
    dose = np.asarray(namespace["dose_frame"]["assigned_navigation_intensity"], dtype=float)
    # "a handful of rows" above the observed maximum: 2 and 3 rows on this draw.
    beyond = {delta: int(np.sum(dose + delta > dose.max())) for delta in (0.5, 1.0)}
    assert 1 <= beyond[0.5] <= 10 and 2 <= beyond[1.0] <= 10
    incremental = namespace["incremental_assessment"]
    assert "nuisance_models" in {item.name for item in incremental.attention}
    ledger = incremental.to_frame()
    rows = ledger[ledger["check"] == "nuisance_models"]
    assert rows["detail"].str.contains("propensity is poorly calibrated").any()


#: These tutorials make seeded output claims that need more than the shrunken smoke gate. The
#: narrower map makes each semantic assertion an explicit review decision instead of a prose
#: heuristic.
TUTORIAL_SEMANTIC_ASSERTIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "docs/examples/collaborative-tmle.md": _collaborative_summary_semantics,
    "docs/examples/cross-fitting.md": _cross_fitting_semantics,
    "docs/examples/dr-tmle.md": _drtmle_semantics,
    "docs/examples/interventions.md": _intervention_axes_semantics,
    "docs/examples/longitudinal-survival.md": _survival_output_semantics,
    "docs/examples/longitudinal-tmle.md": _longitudinal_semantics,
    "docs/examples/msm-projections.md": _multi_arm_identification_semantics,
    "docs/examples/point-treatment-tmle.md": _point_treatment_semantics,
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

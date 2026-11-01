r"""Registered evidence for the cross-fitted learned rule over three arms (X11 (d)).

The subject is ``TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto")`` on a three-arm treatment, with every other setting
at the constructor's default. Each fold's rule is the plug-in argmax of its outcome model over
the three arms, a tie going to the lowest arm code, and the fit reports the fold-evaluated value
``ey_learned_rule[learned rule]``, whose target is the data-adaptive
:math:`\tilde\psi_{0n} = V^{-1}\sum_v \Psi_{d_{nv}}(P_0)`.

The study has one law (``three_arms``, :mod:`tests.studies._learned_rule_arms_law`) and one
size, n = 2,000. It fits three outcome learners on it:

``declared``
    A correct logistic model. Its limit rule is the optimal rule, which gives every arm a share
    of at least 0.175 and a top-two gap below 0.01 on 1.6% of the population. The primary rows
    and the ``interval_calibration`` family read it (condition C4's first branch: both nuisances
    consistent), with the SE x 0.8 and noise controls.
``missing_quadratic``
    The same model without arm 2's quadratic term, with the correct multinomial treatment model
    fitted by maximum likelihood. Its limit rule differs from the optimal rule on 7.0% of the
    population, gives every arm a share of at least 0.245 and has a top-two gap below 0.01 on
    1.8%. The fit's curve omits the treatment model's projection term, which Corollary 2 of van
    der Laan and Luedtke (2015) makes conservative for an MLE of :math:`g` in a correct model,
    extended to cross-validated TMLE by analogy only. The predicted SE ratio of that term on this
    law is 1.0004, so the ``conservative_coverage`` cells measure coverage under a misspecified
    outcome model, not conservativeness: the bias leg and a coverage interval at or above the
    calibration floor, with no SE calibration.
``drop_w1``
    The outcome model without ``W1`` (``targeting_necessity``): the targeted value against the
    untargeted plug-in, whose limit is biased by the confounding through ``W1``.

Both law cells gate under the declared margin condition (:data:`MIN_ARM_SHARE`,
:data:`MAX_NEAR_TIE`), which their learners' population limits meet by quadrature
(``test_learned_rule_arms_design``). The ``rule_margin_diagnostic`` family publishes, for each,
how far the fitted rules sit from that limit, with no verdict.

The X11 wave-2 plan (``X11-wave2-plan.md`` rev 2.1 in the plans repository, section 5), its
review (finding 9), the implementation review ``reviews/x11/X11-H-impl-review.md`` (its eight
study-design answers) and the study design record ``reviews/x11/X11-H-study-design.md`` declare
every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any, TypedDict

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_arms_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The primary replications: the ``declared`` fit of ``three_arms``. The primary gate reads its
#: bias leg (the 99% interval of the mean error inside 0.25 error SDs), coverage floor and SE
#: sanity band. PRIMARY_REPLICATES is the smallest multiple of 400 of at least the study's floor
#: of 1,000 whose predictive pass probability of the bias leg, on the 4,000-draw pilot, is at
#: least 0.95 (``test_learned_rule_arms_design``).
PRIMARY_REPLICATES = 1_200
PRIMARY_N = 2_000
SEED = 20268111
RESAMPLING_SEED = 20268112

PRIMARY_LAW = law.LAW
SCENARIOS = {PRIMARY_LAW: (law.ESTIMAND,)}

#: The declared margin condition of a gated cell (impl review, study-design answer 1): at the
#: cell's learner limit every arm's population share is at least ``MIN_ARM_SHARE`` and the
#: top-two gap of the limit is below :data:`~tests.studies._learned_rule_arms_law.NEAR_TIE` on a
#: population share of at most ``MAX_NEAR_TIE``. A cell whose limit misses either reports.
MIN_ARM_SHARE = 0.10
MAX_NEAR_TIE = 0.02


#: The learner limit of each law cell, by quadrature on the RM30 grid
#: (``test_learned_rule_arms_design`` recomputes each): every arm's population share in arm-code
#: order, ``P(gap < c)`` for ``c`` in 0.005, 0.01 and 0.02, and the share of the population on
#: which the limit rule differs from the optimal rule.
class Limit(TypedDict):
    """One law cell's learner limit (see :data:`LIMITS`)."""

    shares: tuple[float, float, float]
    near_tie: dict[float, float]
    disagreement: float


LIMITS: dict[str, Limit] = {
    "declared": {
        "shares": (0.175000, 0.416625, 0.408375),
        "near_tie": {0.005: 0.008125, 0.01: 0.015875, 0.02: 0.032125},
        "disagreement": 0.0,
    },
    "missing_quadratic": {
        "shares": (0.245500, 0.416625, 0.337875),
        "near_tie": {0.005: 0.009125, 0.01: 0.018250, 0.02: 0.036750},
        "disagreement": 0.070500,
    },
}


def gated(terms: str) -> bool:
    """Whether a law cell's learner limit meets the declared margin condition."""
    limit = LIMITS[terms]
    return min(limit["shares"]) >= MIN_ARM_SHARE and limit["near_tie"][law.NEAR_TIE] <= MAX_NEAR_TIE


#: The SD of ``sqrt(n)`` times the estimate at the optimal rule (the ``declared`` learner's limit
#: rule) on ``three_arms``, by quadrature (:func:`~tests.studies._learned_rule_arms_law.rule_value`).
#: It sizes the noise control only, at one unit, and no verdict reads an efficiency ratio.
EFFICIENCY_BOUND = 0.670220

#: The positive arm's cell suffix.
POSITIVE = "declared"

#: The calibration label: the ``declared`` learner on ``three_arms``.
CALIBRATION_LABELS = (PRIMARY_LAW,)
EFFICIENCY_BOUNDS = {PRIMARY_LAW: EFFICIENCY_BOUND}

#: The ``missing_quadratic`` family: the bias leg and the coverage interval's lower bound at or
#: above the calibration floor 0.92, with no SE ratio gate (Corollary 3's route, the module
#: docstring). The control is the same estimate with its SE times :data:`COVERAGE_SHRINK`, whose
#: coverage interval must fall wholly below 0.92.
COVERAGE_FAMILY = "conservative_coverage"
COVERAGE_LABEL = "missing_quadratic"
COVERAGE_CONTROL = "shrunken_coverage_control"

#: The shrink of the coverage family's control, declared before the pilot (G's rule, D5 of its
#: design record). The claim admits any over-coverage, so the control must fail at every SE ratio
#: the claim admits: at a true ratio of 1.30 the shrunken interval covers about 0.87
#: (``2 Phi(1.96 * 0.6 * 1.3) - 1``), below the floor by more than the 99% half-width at the
#: declared count.
COVERAGE_SHRINK = 0.60

#: The bias control (plan section 5): the ``drop_w1`` learner on ``three_arms``. The positive arm
#: is the targeted value; the control is the same fit's untargeted fold average of the initial
#: plug-ins. The learner's limit rule is arm 1 everywhere, and its plug-in limit is 0.129 above
#: that rule's value, 9.4 SEs at n = 2,000 (``test_learned_rule_arms_design``).
NECESSITY_FAMILY = "targeting_necessity"
NECESSITY_LABEL = "drop_w1"
NECESSITY_CELLS = (f"{NECESSITY_LABEL}__targeted", f"{NECESSITY_LABEL}__untargeted")

#: The margin diagnostic of each law cell (plan section 5, "weak-margin diagnostics for both
#: cells"): role ``diagnostic``, no verdict. Each row is the cell's positive estimate with three
#: columns of its fit: the share of rows whose realized rule differs from the cell's learner-limit
#: rule, the share whose own fold's top-two gap is below 0.01, and the smallest realized arm
#: share. The summary publishes their means.
MARGIN_FAMILY = "rule_margin_diagnostic"
MARGIN = "rule_margin"
MARGIN_LABELS = (PRIMARY_LAW, COVERAGE_LABEL)

#: The learner of each law cell's label.
LABEL_LEARNERS = {PRIMARY_LAW: "declared", COVERAGE_LABEL: "missing_quadratic"}

PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in (POSITIVE, "shrunken_se_control", "noise_control")
    ),
    COVERAGE_FAMILY: (f"{COVERAGE_LABEL}__{POSITIVE}", f"{COVERAGE_LABEL}__{COVERAGE_CONTROL}"),
    NECESSITY_FAMILY: NECESSITY_CELLS,
    MARGIN_FAMILY: tuple(f"{label}__{MARGIN}" for label in MARGIN_LABELS),
}

#: The support row the registering change adds, declared before any run: the P11 row of the
#: fold-evaluated learned rule with ``treatment=("arms3",)``. A fit with four or more arms
#: (``arms4`` and above) is a different key and stays experimental, and so do weights, clusters,
#: strata, fold targeting, repeats, the stacked report and contrasts on three arms (plan
#: section 5, impl review answer 5).
#:
#: The row is added if, and only if, these verdicts pass as declared: the primary performance
#: tests (bias leg, coverage floor, SE sanity); every ``interval_calibration`` verdict, positive
#: and both controls; both ``conservative_coverage`` verdicts; and the ``targeting_necessity``
#: verdicts.
#:
#: A red primary bias leg, a red positive calibration or coverage cell, or a control that does
#: not fail is diagnosed before any route is chosen. With no defect found, the cell is published
#: red under ``reporting`` with an owner row, and the row is not added.
SUPPORT_ROWS = {
    "fold_evaluated": {"splitting": ("fold_evaluated",), "treatment": ("arms3",)},
}

#: The truth-harness artifact: one row per primary replication.
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned treatment rules over three arms",
    slug="learned-rule-arms",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_arms",
    document="docs/technical-reference/method-evidence/learned-rule-arms.md",
    anchor="learned-treatment-rules-over-three-arms",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-arms",
    reference=None,
    failure_bound={"cleverly-learned-rule-arms": 0.01},
    modules=(
        "tests/studies/learned_rule_arms.py",
        "tests/studies/learned_rule_arms_properties.py",
        "tests/studies/_learned_rule_arms_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/estimators/tmle.py",
        "src/cleverly/estimators/_nuisance.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/interventions/base.py",
        "src/cleverly/targets/builtin.py",
        "src/cleverly/learners/crossfit.py",
        "src/cleverly/learners/_fitting.py",
        "src/cleverly/_support.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_arms",
    properties_module="tests.studies.learned_rule_arms_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, "
        "targeting_scheme='pooled', g_bounds='auto', random_state=<fold seed>) at n = 2000 on a "
        "three-arm treatment; every other setting the constructor's default; repeats=1"
    ),
    "rule": "argmax_a Qbar_v(a, w) over the arms 0, 1, 2 of each fold's outcome model; ties to the "
    "lowest arm code",
    "outcome_learners": {
        terms: f"ArmLogistic({terms!r}): LogisticRegression(C=inf, max_iter=5000, tol=1e-10) on "
        f"{', '.join(law.TERMS[terms])} of the design (A1, A2, W1, W2)"
        for terms in law.TERMS
    },
    "treatment_learner": "LogisticRegression(C=inf, max_iter=5000) on (W1, W2): the multinomial "
    "MLE, a correct model",
    "law": (
        "W1 ~ U(-1, 1), W2 ~ Bernoulli(0.5); g0(a | W) proportional to exp(eta_a), eta_0 = 0, "
        "eta_1 = 0.7 W1 + 0.2 W2, eta_2 = -0.4 W1 + 0.2 W2; logit Qbar0(a, W) = 0.2 + 0.5 W1 - "
        "0.3 W2 + b_a(W), b_0 = 0, b_1 = -1 + 6 W1, b_2 = -1.75 - 6 W1 - W2 - 4 (W1^2 - 1/3)"
    ),
    "cells": {
        "declared": "the primary and interval_calibration (SE x 0.80 and noise controls)",
        "missing_quadratic": (
            "conservative_coverage: bias leg and coverage interval lower bound >= 0.92; control "
            "SE x 0.60, coverage interval upper bound < 0.92 (Corollary 3's route, by the "
            "paper's analogy; predicted SE ratio 1.0004, so not a test of conservativeness)"
        ),
        "drop_w1": "targeting_necessity: targeted value against the untargeted plug-in",
    },
    "margin_condition": (
        f"gate a law cell when every arm's limit share >= {MIN_ARM_SHARE} and the limit's "
        f"P(top-two gap < {law.NEAR_TIE}) <= {MAX_NEAR_TIE}; both law cells meet it"
    ),
    "variance": "V^-2 sum_v n_v^-2 sum_{i in v} D_i^2 of the stored curve",
    "truth": (
        "per replication: refit each fold's outcome learner on the fit's training rows with "
        "the package's fit_on_rows, take the argmax at the three arms, require it to equal the "
        "fit's rule on every validation row and the stored fold rule on every row (rule L4), "
        "and integrate Qbar0(d_v(W), W) by the trapezoid rule on 4001 points of W1 for each W2; "
        "weight 1/V"
    ),
    "oracle_se": "sqrt(V^-2 sum_v Var D*(d_v) / n_v) at the fold rules; descriptive",
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def harness_columns(fitted: law.Measured) -> dict[str, Any]:
    """The harness columns of one row of a fit."""
    return {
        "oracle_se": fitted.truth.oracle_se,
        "rule_rows_checked": fitted.truth.rows_checked,
        "solver_warnings": fitted.solver_warnings,
    }


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, law.outcome_learner("declared"), fold_seed)
    estimate = fitted.result.estimates[law.ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    truth = fitted.truth.truth
    return [
        {
            "implementation": record.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": law.ESTIMAND,
            "truth": truth,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= truth <= high),
            "initial_estimate": law.untargeted_estimate(fitted.result),
            **harness_columns(fitted),
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.draw(size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``, with its harness columns.

    ``record`` is the declared one, or the throwaway record of a smoke run (rule L9).
    """
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, n)
            for scenario in record.scenarios
            for replicate in range(replicates)
        ],
        partial(_primary, record),
        draw=partial(_sample, record),
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    rows = pd.DataFrame([row for fitted in outcomes.kept for row in fitted])
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *law.HARNESS_COLUMNS]])


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication."""
    columns = ["scenario", "replicate", "truth", *law.HARNESS_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "replicate"], ignore_index=True
        )
    }

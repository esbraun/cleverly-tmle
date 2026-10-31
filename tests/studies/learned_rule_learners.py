"""Registered evidence for the learned-rule value with a blip or weighted-classification rule learner.

The subject is ``TMLE(learned_rule=LearnedRule(loss=..., learner=...), n_folds=10,
cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")``, which X11 part (g) lifts. Each
fold's rule comes from a rule learner fitted on the fold's training rows to the doubly robust
blip pseudo-outcome, whose outcome and treatment nuisances are cross-fitted in five inner folds
of those rows. The outer fluctuation, estimate and variance are the plug-in rule's. The scenarios
are the two losses on the RM30 ``misspecified_limit`` law with the main-terms outcome learner,
rows only: no weights, no clusters, pooled targeting and one split. The primary phase fits
n = 2,000; the calibration cells fit n = 4,000, where the pilot's error-variance excess of a
rule-learner fit over the fixed-rule variance is within about 3% of zero
(:mod:`tests.studies.learned_rule_learners_properties`). A diagnostic cell per loss at n = 1,000
reports the smaller size, and no verdict reads it.

``blip``
    ``LearnedRule(loss="blip")`` with ``PolynomialFeatures(2)`` and ``LinearRegression()``.
``weighted_classification``
    ``LearnedRule(loss="weighted_classification")`` with ``PolynomialFeatures(2)`` and
    ``LogisticRegression(C=1e6, max_iter=5000)``.

The target of each replication is data-adaptive, so each row carries its own truth: the
fold-average value of the rules the fit learned. :mod:`tests.studies._learned_rule_learners_law`
holds the fits, the limits and the controls. The X11 plan (``X11-plan.md`` rev 2.1 in the plans
repository, sections 2.3 and 3) and its study design record ``reviews/x11/X11-C-study-design.md``
declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_learners_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 2_000
PRIMARY_N = 2_000
SEED = 20263031
RESAMPLING_SEED = 20263032

SCENARIOS = dict.fromkeys(law.LOSSES, (law.ESTIMAND,))

#: The asymptotic SD of ``sqrt(n)`` times the estimate at each loss's limit rule: the SD of the
#: row curve at the rule of the rule learner's population projection
#: (:func:`~tests.studies._learned_rule_learners_law.limit_rule`), by the RM30 grid quadrature.
#: They size the noise controls only, at one unit each, and no verdict reads an efficiency
#: ratio (``calibration_efficiency_ratio=False``).
EFFICIENCY_BOUNDS = {
    "blip": 0.663184,
    "weighted_classification": 0.662914,
}

#: The calibration kinds of each loss, in the order the rows are written.
CALIBRATION_KINDS = (
    "declared",
    "shrunken_se_control",
    "noise_control",
    "validation_rule_control",
)

#: The diagnostic family: one cell per loss at n = 1,000, role ``diagnostic``, no verdict.
DIAGNOSTIC_FAMILY = "small_sample_diagnostic"
DIAGNOSTIC_KIND = "n1000"

PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{loss}__{kind}" for loss in law.LOSSES for kind in CALIBRATION_KINDS
    ),
    DIAGNOSTIC_FAMILY: tuple(f"{loss}__{DIAGNOSTIC_KIND}" for loss in law.LOSSES),
}

HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned-rule value, rule learners",
    slug="learned-rule-learners",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_learners",
    document="docs/technical-reference/method-evidence/learned-rule-learners.md",
    anchor="learned-rule-value-rule-learners",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-learners",
    reference=None,
    failure_bound={"cleverly-learned-rule-learners": 0.01},
    modules=(
        "tests/studies/learned_rule_learners.py",
        "tests/studies/learned_rule_learners_properties.py",
        "tests/studies/_learned_rule_learners_law.py",
        "tests/studies/_learned_rule_law.py",
        "src/cleverly/interventions/_rule_learners.py",
        "src/cleverly/interventions/learned.py",
        "src/cleverly/estimators/tmle.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_learners",
    properties_module="tests.studies.learned_rule_learners_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds=EFFICIENCY_BOUNDS,
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(loss=<scenario>, learner=<rule learner>), n_folds=10, "
        "cv_evaluation=True, targeting_scheme='pooled', g_bounds='auto', "
        "random_state=<fold seed>); primary n = 2000, interval_calibration n = 4000, small_sample_diagnostic n = 1000; rows "
        "only (no weights, no clusters), repeats=1"
    ),
    "rule_learners": {
        "blip": "Pipeline(PolynomialFeatures(2), LinearRegression()) on (W1, W2)",
        "weighted_classification": (
            "Pipeline(PolynomialFeatures(2), LogisticRegression(C=1e6, max_iter=5000)) on "
            "(W1, W2), sample weights |D|"
        ),
    },
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "law": "RM30 misspecified_limit",
    "truth": (
        "per replication: refit each fold's rule through fit_fold_rule with the recorded inner "
        "seed and the fit's g_bounds, require the fit's rule on every validation row, and "
        "average the fold rules' values E Qbar0(d, W) by the trapezoid rule on 4001 points of "
        "W1 for each W2"
    ),
    "oracle_se": "the row-curve oracle SE of the fit's fold rules; descriptive",
    "value_gain": (
        "plug_in_estimate, plug_in_std_error and plug_in_truth: the plug-in rule's fit of the "
        "same sample with the same fold seed, its reported standard error and its harness truth; "
        "descriptive"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate); "
    "the inner seeds derive from it by SeedSequence([split_seed, v])",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return base.draw(law.LAW, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    fitted = law.measured(frame, scenario, fold_seed)
    estimate = fitted.result.estimates[law.ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    return [
        {
            "implementation": record.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": law.ESTIMAND,
            "truth": fitted.truth.truth,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= fitted.truth.truth <= high),
            "initial_estimate": base.untargeted_estimate(fitted.result),
            "oracle_se": fitted.truth.oracle_se,
            "rule_rows_checked": fitted.truth.rows_checked,
            "solver_warnings": fitted.warnings,
            **law.value_gain_columns(fitted),
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return base.draw(law.LAW, size, replicate_seed(record, scenario, replicate))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``, with its harness columns."""
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
    return replication.result(
        outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *base.HARNESS_COLUMNS, *law.VALUE_GAIN_COLUMNS]]
    )


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness and value-gain columns of every primary replication.

    ``replicates.parquet`` keeps the shared schema, so the paired plug-in fit's estimate,
    standard error and truth of each primary replication are published here beside the harness
    columns.
    """
    columns = ["scenario", "replicate", "truth", *base.HARNESS_COLUMNS, *law.VALUE_GAIN_COLUMNS]
    return {
        HARNESS_ARTIFACT: rows.loc[:, columns].sort_values(
            ["scenario", "replicate"], ignore_index=True
        )
    }

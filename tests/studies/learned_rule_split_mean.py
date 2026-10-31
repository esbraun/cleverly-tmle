"""Registered evidence for the learned-rule value as the mean over a fixed number of splits (X11).

The subject is ``TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True,
g_bounds="auto", repeats=5)``, which X11 part (i) lifts. A learned-rule fit with ``repeats=R``
above 1 draws ``R`` independent ten-fold splits, learns each split's fold rules on its training
rows, and reports the mean of the ``R`` split estimates with the variance
``n^-2 sum_i phibar_i^2`` of the mean of the split curves. The scenarios are the RM30 laws
``non_exceptional`` and ``misspecified_limit`` with rows only: no weights and no clusters.

The target of each replication is data-adaptive, so each row carries its own truth: the mean
over the ``R`` splits of each split's fold-average value of the rules it learned.
:mod:`tests.studies._learned_rule_split_mean_law` holds the fit, the harness and the controls.
The X11 plan (``X11-plan.md`` rev 2.1 in the plans repository, sections 2.2 and 3) and its
study design record ``reviews/x11/X11-B-study-design.md`` declare every quantity here before
any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_law as base
from tests.studies import _learned_rule_split_mean_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 2_000
PRIMARY_N = 2_000
SEED = 20263021
RESAMPLING_SEED = 20263022

SCENARIOS = dict.fromkeys(law.LAWS, (law.ESTIMAND,))

#: The asymptotic SD of ``sqrt(n)`` times the estimate at each law's limit rule: the SD of the
#: row curve at the rule of the declared learner's population projection
#: (:func:`~tests.studies._learned_rule_split_mean_law.population_rule`), by the RM30 grid
#: quadrature. Under C3 every split's rules share that limit, so the mean curve has the SD of
#: one split's. They size the noise controls only, at one unit each, and no verdict reads an
#: efficiency ratio (``calibration_efficiency_ratio=False``).
EFFICIENCY_BOUNDS = {
    "non_exceptional": 0.664438,
    "misspecified_limit": 0.666347,
}

#: The calibration kinds of each law, in the order the rows are written.
CALIBRATION_KINDS = (
    "declared",
    "shrunken_se_control",
    "noise_control",
    "independent_splits_control",
)

PROPERTY_CELLS = {
    "interval_calibration": tuple(
        f"{label}__{kind}" for label in law.LAWS for kind in CALIBRATION_KINDS
    ),
}

HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned-rule value, mean over splits",
    slug="learned-rule-split-mean",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_split_mean",
    document="docs/technical-reference/method-evidence/learned-rule-split-mean.md",
    anchor="learned-rule-value-mean-over-splits",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-split-mean",
    reference=None,
    failure_bound={"cleverly-learned-rule-split-mean": 0.01},
    modules=(
        "tests/studies/learned_rule_split_mean.py",
        "tests/studies/learned_rule_split_mean_properties.py",
        "tests/studies/_learned_rule_split_mean_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_split_mean",
    properties_module="tests.studies.learned_rule_split_mean_properties",
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
        "targeting_scheme='pooled', g_bounds='auto', repeats=5, random_state=<fold seed>) "
        "at n = 2000; rows only (no weights, no clusters)"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "laws": {
        "non_exceptional": "RM30 non_exceptional",
        "misspecified_limit": "RM30 misspecified_limit",
    },
    "truth": (
        "per replication: for each of the 5 splits, refit each fold's outcome learner on the "
        "split's training rows, require the split's rule on every validation row, and average "
        "the fold rules' values E Qbar0(d, W) by the trapezoid rule on 4001 points of W1 for "
        "each W2; the truth is the mean of the 5 split values"
    ),
    "oracle_se": "the mean over the splits of each split's row-curve oracle SE; descriptive",
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate); "
    "the engine draws the 5 split seeds from it",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return base.draw(scenario, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _primary(
    record: StudyRecord, key: tuple[str, int, int], frame: pd.DataFrame
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    result, truth, solver_warnings = law.measured(
        frame, scenario, base.outcome_learner(), fold_seed
    )
    estimate = result.estimates[law.ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    return [
        {
            "implementation": record.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": law.ESTIMAND,
            "truth": truth.truth,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= truth.truth <= high),
            "initial_estimate": law.untargeted_estimate(result),
            "oracle_se": truth.oracle_se,
            "rule_rows_checked": truth.rows_checked,
            "solver_warnings": solver_warnings,
            **law.first_split_columns(result, truth),
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return base.draw(scenario, size, replicate_seed(record, scenario, replicate))


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
        outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *base.HARNESS_COLUMNS, *law.FIRST_SPLIT_COLUMNS]]
    )


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication."""
    return {HARNESS_ARTIFACT: base.harness_artifact(rows)}

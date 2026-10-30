"""Registered evidence for the learned-rule value with fold targeting, weights and clusters (X11).

The subject is ``TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True,
g_bounds="auto")`` under the three settings that X11 parts (h), (k) and (l) lift:

``fold_targeted``
    ``targeting_scheme="fold"`` on the RM30 ``non_exceptional`` law.
``weighted``
    ``weights=`` with the known weight ``1 + 1.5 * 1{W1 > 0}`` on the ``misspecified_limit``
    law, where the weighted rule has its own limit. The target is the fold-average value in the
    weighted population.
``clustered``
    ``id=`` with 200 clusters of 10 rows and a shared logit-scale effect of SD 1.5 on the
    ``non_exceptional`` linear predictor.

The target of each replication is data-adaptive, so each row carries its own truth.
:mod:`tests.studies._learned_rule_compositions_law` holds the laws, the fits and the harness.
The X11 plan (``X11-plan.md`` rev 2.1 in the plans repository, section 3) and its study design
record declare every quantity here before any run.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_compositions_law as law
from tests.studies import _learned_rule_law as base
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 4_000
PRIMARY_N = 2_000
SEED = 20263011
RESAMPLING_SEED = 20263012

SCENARIOS = dict.fromkeys(law.PRIMARY_SCENARIOS, (law.ESTIMAND,))

#: The asymptotic SD of ``sqrt(n)`` times the estimate at each scenario's limit rule, by the grid
#: quadrature of :mod:`tests.studies._learned_rule_compositions_law`. The limit rule is the rule
#: of the declared learner's population projection (``population_rule``). For ``fold_targeted``
#: and ``weighted`` it is the SD of the row curve; for ``clustered`` it adds the within-cluster
#: covariance, ``sqrt(Var D + (m - 1) Cov)`` with ``m = 10`` (``cluster_covariance``; the
#: correlation is 0.144). They size the noise controls only, at one unit each.
EFFICIENCY_BOUNDS = {
    "fold_targeted": 0.664438,
    "weighted": 0.715236,
    "clustered": 1.021125,
}

PROPERTY_CELLS = {
    "interval_calibration": (
        *tuple(
            f"{label}__{kind}"
            for label in ("fold_targeted", "weighted", "clustered")
            for kind in ("declared", "shrunken_se_control", "noise_control")
        ),
        "clustered__row_variance_control",
        "clustered_few__declared",
        "clustered_few__shrunken_se_control",
        "clustered_few__noise_control",
    ),
    "targeting_necessity": ("fold_targeted__targeted", "fold_targeted__untargeted"),
}

HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="Learned-rule value with fold targeting, weights and clusters",
    slug="learned-rule-compositions",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_compositions",
    document="docs/technical-reference/method-evidence/learned-rule-compositions.md",
    anchor="learned-rule-value-with-fold-targeting-weights-and-clusters",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-compositions",
    reference=None,
    failure_bound={"cleverly-learned-rule-compositions": 0.01},
    modules=(
        "tests/studies/learned_rule_compositions.py",
        "tests/studies/learned_rule_compositions_properties.py",
        "tests/studies/_learned_rule_compositions_law.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_compositions",
    properties_module="tests.studies.learned_rule_compositions_properties",
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
        "targeting_scheme='fold' (fold_targeted) or 'pooled', g_bounds='auto', "
        "random_state=<fold seed>) at n = 2000; weights='w' (weighted); id='cluster' (clustered)"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "laws": {
        "fold_targeted": "RM30 non_exceptional",
        "weighted": "RM30 misspecified_limit, weight w = 1 + 1.5 * 1{W1 > 0}",
        "clustered": (
            "RM30 non_exceptional plus u ~ N(0, 1.5^2) shared by 10 rows per cluster; "
            "Qbar0 marginal over u by 64-node Gauss-Hermite quadrature"
        ),
    },
    "truth": (
        "per replication: refit each fold's outcome learner on the fit's own training rows "
        "with its weights, require the fit's rule on every validation row, and average the "
        "fold rules' values E[w Qbar0(d, W)] / E[w] by the trapezoid rule on 4001 points of "
        "W1 for each W2"
    ),
    "oracle_se": "the row-level curve's SD at each fold rule, by the same quadrature; "
    "descriptive, and row-level for clustered",
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed; the truth depends on the fit, so none here."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(scenario, n, seed), {}


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
            "initial_estimate": base.untargeted_estimate(result),
            "oracle_se": truth.oracle_se,
            "rule_rows_checked": truth.rows_checked,
            "solver_warnings": solver_warnings,
        }
    ]


def _sample(record: StudyRecord, key: tuple[str, int, int]) -> pd.DataFrame:
    scenario, replicate, size = key
    return law.draw(scenario, size, replicate_seed(record, scenario, replicate))


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
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *base.HARNESS_COLUMNS]])


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication."""
    return {HARNESS_ARTIFACT: base.harness_artifact(rows)}

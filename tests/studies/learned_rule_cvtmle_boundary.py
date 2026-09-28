"""Registered evidence for the learned-rule value CV-TMLE at its boundary (RM30, reporting).

The estimator, the learners, the size, the truth harness and the oracle SE are those of the
gated study :mod:`tests.studies.learned_rule_cvtmle`.  Only the laws differ.
``exceptional`` has no treatment effect anywhere, so the fold rules have no fixed limit and
condition C3 of the RM30 contract fails.  ``weak_blip`` has a small blip ``0.15 W1``.  It is
within C3, and a red cell there is a finite-sample limit at n = 2,000.

The study publishes under ``publication_policy="reporting"``, and it has no property cell.  Each
red cell enters the red-cell ledger with the owner F27.  The reading table of the RM30
declaration reads each law's 99% Clopper-Pearson coverage interval from the top: an upper end
below 0.95 reads "under-covers at the <law> law", a lower end at or above 0.90 reads "no
under-coverage resolved at the declared budget", and anything else reads "unresolved".
:func:`reading` writes it as ``reading.csv``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_law as law
from tests.studies.evidence.inference import Interval, clopper_pearson
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 6_000
PRIMARY_N = 2_000
SEED = 20263003
RESAMPLING_SEED = 20263004

SCENARIOS = {
    "exceptional": (law.ESTIMAND,),
    "weak_blip": (law.ESTIMAND,),
}

#: The artefacts beside the six the framework writes.
HARNESS_ARTIFACT = "harness.csv.gz"
READING_ARTIFACT = "reading.csv"

#: The label of every reading a smoke run writes (rule L9).
SMOKE = "smoke run, not the declared budget"

STUDY = StudyRecord(
    name="CV-TMLE of the fold-local learned-rule value at exceptional and weak-blip laws",
    slug="learned-rule-cvtmle-boundary",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_cvtmle_boundary",
    document="docs/technical-reference/method-evidence/learned-rule-cvtmle-boundary.md",
    anchor="cv-tmle-of-the-fold-local-learned-rule-value-at-exceptional-and-weak-blip-laws",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-cvtmle",
    reference=None,
    modules=(
        "tests/studies/learned_rule_cvtmle_boundary.py",
        "tests/studies/learned_rule_cvtmle_boundary_properties.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_cvtmle_boundary",
    properties_module="tests.studies.learned_rule_cvtmle_boundary_properties",
    property_cells={},
    publication_policy="reporting",
    extra_artifacts=(HARNESS_ARTIFACT, READING_ARTIFACT),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": (
        "TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True, "
        "targeting_scheme='pooled', g_bounds='auto', random_state=<fold seed>) at n = 2000"
    ),
    "outcome_learner": (
        "Pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), "
        "LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)) on (A, W1, W2)"
    ),
    "treatment_learner": "LogisticRegression(C=1e6, max_iter=1000) on (W1, W2)",
    "law": (
        "W1 ~ U(-1, 1); W2 ~ Bernoulli(0.5); logit g0 = 0.3 W1 - 0.2 W2; "
        "logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + A b(W)"
    ),
    "blips": {"exceptional": "0", "weak_blip": "0.15 W1"},
    "truth": (
        "per replication: refit each fold's outcome learner on the fit's own training rows, "
        "require the fit's rule on every validation row, and average the fold rules' values "
        "by the trapezoid rule on 4001 points of W1 for each W2, with the known Qbar0"
    ),
    "oracle_se": (
        "sqrt(V^-2 sum_v Var_P0 D*(d_v, Qbar0, g0) / n_v), by the same quadrature; "
        "descriptive, in harness.csv.gz"
    ),
    "fold_seed": "stream_seed(record, 'fold_partition', 'primary', scenario, replicate)",
    "solver_warnings": (
        "scikit-learn and cleverly ConvergenceWarnings of the fit and its truth refits, "
        "counted in harness.csv.gz; no rule reads the count"
    ),
    "reading": (
        "from the top, on each law's 99% Clopper-Pearson coverage interval: upper end below "
        "0.95 -> under-covers at the <law> law; lower end at or above 0.90 -> no "
        "under-coverage resolved at the declared budget; otherwise unresolved"
    ),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed, for the published-seed audit.

    The truth of a replication depends on the rules its fit learns, so the sample alone
    carries none, and the mapping is empty.
    """
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return law.draw(scenario, n, seed), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario``, from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> pd.DataFrame:
    """Draw and fit every primary replication of ``record``, with its harness columns.

    ``record`` is the declared one, or the throwaway record of a smoke run (rule L9).
    """
    return law.draw_and_fit(record, replicates=replicates, n=n, n_jobs=n_jobs)


def read(interval: Interval, scenario: str) -> str:
    """The first row of the declared reading table whose condition holds (rule L5)."""
    if interval.high < 0.95:
        return f"under-covers at the {scenario} law"
    if interval.low >= 0.90:
        return "no under-coverage resolved at the declared budget"
    return "unresolved"


def reading(rows: pd.DataFrame) -> pd.DataFrame:
    """The declared reading of each law, read from the top of the table (rule L5).

    A run whose replication count is not the declared budget writes :data:`SMOKE` in place
    of every reading (rule L9).
    """
    level = STUDY.margins.confidence_level
    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        group = rows.loc[rows["scenario"] == scenario]
        replicates = len(group)
        interval = clopper_pearson(int(group["covered"].sum()), replicates, confidence_level=level)
        text = SMOKE if replicates != PRIMARY_REPLICATES else read(interval, scenario)
        records.append(
            {
                "scenario": scenario,
                "replicates": replicates,
                "confidence_level": level,
                "coverage": float(group["covered"].mean()),
                "coverage_ci_lower": interval.low,
                "coverage_ci_upper": interval.high,
                "reading": text,
            }
        )
    return pd.DataFrame.from_records(records)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication, and the declared reading."""
    return {HARNESS_ARTIFACT: law.harness_artifact(rows), READING_ARTIFACT: reading(rows)}

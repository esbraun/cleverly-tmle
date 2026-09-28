"""Registered evidence for the fold-local learned-rule value CV-TMLE (RM30, gated study).

The subject is ``TMLE(learned_rule=LearnedRule(), n_folds=10, cv_evaluation=True,
targeting_scheme="pooled", g_bounds="auto")``: the fold-evaluated CV-TMLE of the average, over
the ten outer folds, of the value of the plug-in rule learned on each fold's training rows.  Van
der Laan and Luedtke (2015), Section 7 and Appendix B, define the target and its inference.

The study states only the published conditions.  ``non_exceptional`` is a correctly specified
law whose limit rule is the optimal rule.  ``misspecified_limit`` fits a working outcome model
that is wrong, beside a correct treatment model, so its limit rule differs from the optimal one
and the cell rests on Corollary 3 of that article.  The exceptional and weak-blip laws are the
reporting study :mod:`tests.studies.learned_rule_cvtmle_boundary`.

The RM30 section of ``docs/roadmap.md`` declares every quantity here before any run.
:mod:`tests.studies._learned_rule_law` holds the laws, the fit and the truth harness that both
studies share.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _learned_rule_law as law
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 6_000
PRIMARY_N = 2_000
SEED = 20263001
RESAMPLING_SEED = 20263002

SCENARIOS = {
    "non_exceptional": (law.ESTIMAND,),
    "misspecified_limit": (law.ESTIMAND,),
}

#: ``sqrt(Var_P0 D*(d0, Qbar0, g0))`` on ``non_exceptional``, by quadrature.  It sizes the
#: noise control only, and no gate reads an efficiency band.
EFFICIENCY_BOUND = 0.664444

PROPERTY_CELLS = {
    "interval_calibration": (
        "non_exceptional__correctly_specified",
        "non_exceptional__shrunken_se_control",
        "non_exceptional__noise_control",
    ),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "targeting_necessity": ("non_exceptional__targeted", "non_exceptional__untargeted"),
    "fold_locality": ("non_exceptional__fold_local", "non_exceptional__validation_rows"),
}

#: The artefact that carries the harness columns of the primary rows (rule L4).
HARNESS_ARTIFACT = "harness.csv.gz"

STUDY = StudyRecord(
    name="CV-TMLE of the fold-local learned-rule value",
    slug="learned-rule-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "learned_rule_cvtmle",
    document="docs/technical-reference/method-evidence/learned-rule-cvtmle.md",
    anchor="cv-tmle-of-the-fold-local-learned-rule-value",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-learned-rule-cvtmle",
    reference=None,
    modules=(
        "tests/studies/learned_rule_cvtmle.py",
        "tests/studies/learned_rule_cvtmle_properties.py",
        "tests/studies/_learned_rule_law.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.learned_rule_cvtmle",
    properties_module="tests.studies.learned_rule_cvtmle_properties",
    property_cells=PROPERTY_CELLS,
    efficiency_bounds={"non_exceptional": EFFICIENCY_BOUND},
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(HARNESS_ARTIFACT,),
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
    "blips": {
        "non_exceptional": "0.1 + W1",
        "misspecified_limit": "0.8 W1 + W1^2 - 0.3",
    },
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


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    """The harness columns of every primary replication."""
    return {HARNESS_ARTIFACT: law.harness_artifact(rows)}

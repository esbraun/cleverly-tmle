"""Registered evidence for clustered point-treatment CV-TMLE at unequal cluster sizes.

The subject is the stacked cross-fitted TMLE with ``id=``: five grouped folds drawn by
``random_partition`` from the cluster labels and ``random_state=0``, an unpenalized
main-terms logistic outcome regression, and the exact propensity. The law is the informative
law of :mod:`tests.studies.clustered_unequal_laws`: cluster sizes uniform on 2 to 18, with the
size in the outcome. Its parameters were declared before any run; that module's docstring
records the pilot and the orchestrator decision that set them.

Three scenarios read two sample streams:

* ``unequal_noninformative``: the size does not enter the outcome, so the row-weighted mean
  :math:`\\mu_I` equals the cluster-average mean :math:`\\mu_C`.
* ``unequal_informative``: the size enters the outcome, and the truth is :math:`\\mu_I`.
* ``unequal_informative_cluster_average``: the same samples as ``unequal_informative``, fitted
  with ``weights`` equal to one over the cluster size, and the truth is :math:`\\mu_C`.

Each draw holds 200 clusters of random size, about 2,000 rows. The rows publish the nominal
``n = 2,000`` because the schema requires every row to carry the record's ``n``;
:data:`CONFIGURATION` states the realized size. The study has no comparator: of the pinned
implementations only R ``ltmle`` aggregates by cluster sums, and it does not cross-fit
(``docs/development/method-benchmarking.md``, the cluster survey).

Publication policy is ``reporting``, by the red-cell route the plan declared before any run. The
first full run (2026-10-03) read two red cells. The DR-TMLE pair carried a study defect: its
fits used the exact-propensity bounds ``(1e-9, 1 - 1e-9)`` with a fitted reduced treatment
regression, and two of 2,400 curves reached entries of 4.7e5. The DR-TMLE fits now use the
canonical DR-TMLE study's bound convention for a fitted treatment regression, ``(0.01, 0.99)``.
The true propensity reaches 0.01 or 0.99 only at about 6.8 SD of its logit, so the bounds do not
change the estimand. The fold-evaluated pair on the cluster-level covariate law read
coverage 0.9325 with lower endpoint 0.918 against the 0.92 calibration floor, with its SE ratio
inside its band. No package defect was found: at 40 clusters the declared rule keeps the normal
reference, and that variance has J - V = 30 degrees of freedom. The record switched to
``reporting`` with owner F28 before the one re-run. No margin, budget, law or learner changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.learners.crossfit import check_integrity
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_unequal_laws as laws
from tests.studies.canonical_cvtmle import fitted_rows
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import initial_estimates, primary_rows

PRIMARY_REPLICATES = 800
#: The nominal row count. Each draw holds ``PRIMARY_N // 10`` clusters of mean size 10.
PRIMARY_N = 2_000
PROPERTY_REPLICATES = 2_400
N_FOLDS = 5
SEED = 20262401
RESAMPLING_SEED = 20262402
ESTIMANDS = ("ey0", "ey1", "ate")
G_BOUNDS = (1e-9, 1.0 - 1e-9)
STRATIFY_FOLDS = "none"
RANDOM_STATE = 0

NONINFORMATIVE = "unequal_noninformative"
INFORMATIVE = "unequal_informative"
CLUSTER_AVERAGE = "unequal_informative_cluster_average"
SCENARIOS = {NONINFORMATIVE: ESTIMANDS, INFORMATIVE: ESTIMANDS, CLUSTER_AVERAGE: ESTIMANDS}

PROPERTY_CELLS = {
    "clustered_inference": (
        "cluster_robust",
        "iid_control",
        "cluster_robust_fold_evaluated",
        "iid_control_fold_evaluated",
        "cluster_robust_drtmle",
        "iid_control_drtmle",
        "cluster_robust_fold_evaluated_cluster_covariate",
        "iid_control_fold_evaluated_cluster_covariate",
    ),
    "estimand_weighting": (
        "individual_average",
        "cluster_average_truth",
        "cluster_average_weights",
        "individual_average_truth",
    ),
    "cluster_aggregation_rule": ("cluster_sum", "cluster_mean"),
}

STUDY = StudyRecord(
    name="clustered point-treatment CV-TMLE at unequal cluster sizes",
    slug="clustered-unequal-cvtmle",
    artifacts=ROOT / "tests" / "canonical" / "clustered_unequal_cvtmle",
    document="docs/technical-reference/method-evidence/clustered-unequal-cv-tmle.md",
    anchor="clustered-point-treatment-cv-tmle-at-unequal-cluster-sizes",
    scenarios=SCENARIOS,
    scenario_seed_owners={CLUSTER_AVERAGE: INFORMATIVE},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-clustered-unequal-cvtmle",
    reference=None,
    modules=(
        "tests/studies/clustered_unequal_cvtmle.py",
        "tests/studies/clustered_unequal_properties.py",
        "tests/studies/clustered_unequal_laws.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.clustered_unequal_cvtmle",
    properties_module="tests.studies.clustered_unequal_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "clusters_per_draw": PRIMARY_N // 10,
    "cluster_size": "uniform on 2 to 18, mean 10, drawn per cluster",
    "realized_rows": (
        "random, mean 2,000; every row publishes the nominal n = 2,000, which the schema requires"
    ),
    "law": (
        f"informative law, delta = {laws.DELTA}, gamma = {laws.GAMMA}, effect modifier "
        f"{laws.EFFECT_MODIFIER}; the non-informative scenario sets delta = gamma = 0"
    ),
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "folds": (
        "grouped random_partition folds from the cluster labels and random_state=0; the "
        "labels differ by draw, so the partition does too"
    ),
    "targeting_scheme": "pooled",
    "outcome_learner": "unpenalized logistic regression (C=1e6) in W1 and W2",
    "treatment_mechanism": "exact",
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "stratify_folds": STRATIFY_FOLDS,
    "cluster_average_weights": "one over the cluster size, on the cluster-average scenario",
    "q_bounds": "none: the outcome is binary",
}


class ExactPropensity:
    """The informative law's propensity, as ``OracleTreatment`` reads it."""

    @staticmethod
    def propensity(covariates: Any) -> Any:
        x = np.asarray(covariates, dtype=float)
        return laws.informative_propensity(x[:, 0], x[:, 1])


def law_parameters(scenario: str) -> tuple[float, float]:
    """``(delta, gamma)`` of ``scenario``."""
    if scenario == NONINFORMATIVE:
        return 0.0, 0.0
    if scenario in (INFORMATIVE, CLUSTER_AVERAGE):
        return laws.DELTA, laws.GAMMA
    raise KeyError(scenario)


def scenario_truth(scenario: str) -> dict[str, float]:
    """The exact truth of every estimand ``scenario`` reports."""
    delta, gamma = law_parameters(scenario)
    truth = laws.informative_truth(delta, gamma)
    suffix = "_cluster" if scenario == CLUSTER_AVERAGE else ""
    return {name: float(truth[f"{name}{suffix}"]) for name in ESTIMANDS}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``n // 10`` clusters from an explicit seed."""
    delta, gamma = law_parameters(scenario)
    frame = laws.draw_informative(n // 10, np.random.default_rng(seed), delta=delta, gamma=gamma)
    return frame, scenario_truth(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of ``scenario`` from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def outcome_learner() -> LogisticRegression:
    """The unpenalized main-terms logistic outcome regression."""
    return LogisticRegression(C=1e6, max_iter=1000)


def tmle_settings(**overrides: Any) -> dict[str, Any]:
    """The registered stacked CV-TMLE configuration."""
    return {
        "outcome_learner": outcome_learner(),
        "treatment_learner": OracleTreatment(ExactPropensity()),
        "cross_fit": True,
        "n_folds": N_FOLDS,
        "targeting_scheme": "pooled",
        "stratify_folds": STRATIFY_FOLDS,
        "estimands": ESTIMANDS,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": RANDOM_STATE,
        **overrides,
    }


def fit_cleverly(frame: pd.DataFrame, scenario: str, **overrides: Any) -> Any:
    """The registered fit of ``scenario``: weighted by one over the size on the cluster average."""
    roles: dict[str, Any] = {}
    data = frame
    if scenario == CLUSTER_AVERAGE:
        data = frame.assign(w=1.0 / frame["size"])
        roles["weights"] = "w"
    result = (
        TMLE(**tmle_settings(**overrides))
        .fit(data, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster", **roles)
        .single()
    )
    check_integrity(result.nuisance.folds, cluster=np.asarray(frame["cluster"]))
    return result


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Fit one replication and return its rows at the nominal ``n``."""
    result = fit_cleverly(frame, scenario)
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
        n=PRIMARY_N,
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )

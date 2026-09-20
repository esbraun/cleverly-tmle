"""Registered evidence study for clustered point-treatment CV-TMLE."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import clustered_dgp
from cleverly.estimators import TMLE
from cleverly.learners import random_partition
from cleverly.learners.crossfit import check_integrity
from cleverly.utils.parallel import map_parallel
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import initial_estimates, primary_rows

LMTP_COMMIT = "f04a2b47f46debc515ce4ae778e05ebfde922c44"
IFE_VERSION = "0.2.3"
IFE_SHA256 = "b6be1e9ba514db95118e425d2f78deabb2c9f745f44f35a301ff9b5f266d5ed2"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
PROPERTY_REPLICATES = 2_400
N_FOLDS = 5
CLUSTER_SIZE = 10
SEED = 20260929
RESAMPLING_SEED = 20260930
SCENARIO = "clustered_binary"
ESTIMANDS = ("ey0", "ey1", "ate")
G_BOUNDS = (1e-9, 1.0 - 1e-9)
#: The fold policy this row declares.  The package reads nothing but the cluster identifier
#: when it partitions, so the split is grouped and balanced on nothing.
STRATIFY_FOLDS = "none"
#: Every replication partitions from this one integer, and every draw carries the same
#: cluster labels, so all of them share one grouped partition.  See
#: :func:`assert_the_declared_fixed_partition`.
RANDOM_STATE = 0

STUDY = StudyRecord(
    name="clustered point-treatment CV-TMLE",
    slug="clustered-tmle",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_clustered_tmle",
    document="docs/technical-reference/method-evidence/clustered-point-treatment-cv-tmle.md",
    anchor="clustered-point-treatment-cv-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-clustered-cvtmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_clustered_tmle.py",
        "tests/studies/clustered_tmle_properties.py",
        "tests/studies/missing_outcome_study_helpers.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_clustered_tmle",
    properties_module="tests.studies.clustered_tmle_properties",
    property_cells={
        "clustered_inference": ("cluster_robust", "iid_control"),
    },
)

REFERENCE_METADATA = {
    "ife_sha256": IFE_SHA256,
    "ife_version": IFE_VERSION,
    "lmtp_commit": LMTP_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "five-fold deterministic point-treatment TMLE with cluster identifiers",
}

CONFIGURATION = {
    "cluster_size": CLUSTER_SIZE,
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "folds": (
        "one grouped five-fold partition, drawn by cleverly.random_partition from the "
        f"cluster identifier and the fixed seed {RANDOM_STATE}, and supplied unchanged to "
        "both implementations. Every replication uses that same partition"
    ),
    "targeting_scheme": "pooled",
    "outcome_type": "binary, declared to the R comparator as outcome_type='binomial'",
    "outcome_learner": "unpenalized logistic regression (C=1e6), matching SL.glm",
    "treatment_mechanism": "exact",
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "stratify_folds": STRATIFY_FOLDS,
    "q_bounds": "none: the law is binary, so the outcome scaler is already the identity",
}


def law() -> Any:
    """Return the declared clustered law."""
    return clustered_dgp(cluster_size=CLUSTER_SIZE, family="binomial")


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one clustered sample from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = law().sample(n, seed=seed, backend="pandas")
    return frame, {name: float(truth[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def outcome_learner() -> LogisticRegression:
    """The unpenalized main-effects logistic regression this row fits ``Q`` with.

    ``C=1e6`` rather than a penalty, because the R comparator fits ``SL.glm``, which is an
    unpenalized generalized linear model.  A default-penalized fit would make the two
    implementations disagree about the working model before targeting ever ran.

    Returns
    -------
    LogisticRegression
        An unpenalized main-effects logistic regression.
    """
    return LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")


@functools.cache
def _declared_partition(n: int) -> tuple[int, ...]:
    """The one grouped partition every replication of size ``n`` is fitted under.

    Rebuilt from the generator rather than read off a fit, so the check below compares the
    realized assignment against a statement of what the declared split *is*.

    Parameters
    ----------
    n : int
        Rows in the replication.

    Returns
    -------
    tuple of int
        The fold index of every row.
    """
    labels = np.repeat(np.arange(n // CLUSTER_SIZE), CLUSTER_SIZE)
    folds = random_partition(n, N_FOLDS, cluster=labels, seed=RANDOM_STATE)
    return tuple(int(value) for value in folds.assignment)


def assert_the_declared_fixed_partition(result: Any, frame: pd.DataFrame) -> None:
    """Refuse a fit that did not run under this row's one declared partition.

    Three facts make the published coverage conditional on a single split, and this states
    all three.  ``random_state`` is a fixed integer for every replication.  Every draw emits
    the same cluster labels, in the same order, because the law numbers clusters by row
    block.  The partition therefore depends on nothing the draw varies, so all 800 primary
    and 2,400 property replications share one byte-identical grouped assignment.

    The evidence page says so, and without this the claim would rest on reading the code.

    Parameters
    ----------
    result : Any
        A fitted single-parameter result.
    frame : pandas.DataFrame
        The replication the fit read, whose ``cluster`` column the split is grouped on.

    Raises
    ------
    RuntimeError
        When the fold policy, the scheme, the recorded seed, or the realized assignment is
        not the declared one.
    """
    plan = result.config.crossfit
    origin = result.nuisance.folds.origin
    if plan.scheme != "grouped" or plan.stratify_by != ():
        raise RuntimeError(
            f"this study declares a grouped split balanced on nothing, and this fit "
            f"recorded scheme={plan.scheme!r} balanced on {plan.stratify_by!r}"
        )
    if origin is None or origin.seed != RANDOM_STATE or origin.scheme != "grouped":
        raise RuntimeError(f"the realized folds record the origin {origin!r}")
    cluster = np.asarray(frame["cluster"])
    expected = np.repeat(np.arange(len(cluster) // CLUSTER_SIZE), CLUSTER_SIZE)
    if not np.array_equal(cluster, expected):
        raise RuntimeError("this draw's cluster labels are not the declared row blocks")
    assignment = np.asarray(result.nuisance.folds.assignment)
    if not np.array_equal(assignment, np.asarray(_declared_partition(len(assignment)))):
        raise RuntimeError("this replication was fitted under a different grouped partition")


def fit_cleverly(frame: pd.DataFrame) -> Any:
    """Fit the grouped five-fold estimator with the exact treatment mechanism."""
    dgp = law()
    result = (
        TMLE(
            outcome_learner=outcome_learner(),
            treatment_learner=OracleTreatment(dgp),
            cross_fit=True,
            n_folds=N_FOLDS,
            targeting_scheme="pooled",
            stratify_folds=STRATIFY_FOLDS,
            estimands=ESTIMANDS,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=RANDOM_STATE,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            id="cluster",
        )
        .single()
    )
    check_integrity(result.nuisance.folds, cluster=np.asarray(frame["cluster"]))
    if not np.array_equal(result.data.cluster, np.asarray(frame["cluster"])):
        raise AssertionError("the clustered fit did not retain the supplied identifier")
    assert_the_declared_fixed_partition(result, frame)
    return result


def rows_from_result(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Convert one fit to the shared primary-replication schema."""
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Fit and return one replication's subject rows."""
    return rows_from_result(fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    sample = frame.copy()
    sample.insert(0, "fold", result.nuisance.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, rows_from_result(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, fit, and retain the exact grouped folds for the R comparator."""
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

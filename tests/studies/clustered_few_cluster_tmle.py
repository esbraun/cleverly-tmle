"""Registered evidence for clustered intervals on a Student t reference, at 10 to 39 clusters.

Below 40 clusters with positive weight mass a clustered estimate reports its interval and
p-value on a Student t reference with ``J - 2`` degrees of freedom (Nugent et al. (2024),
Section 2.2). This study measures that rule. Its publication policy is ``reporting``: every
cell may read red, and F28 in the roadmap owns any red cell. The policy and every budget were
declared before any run. Its smallest cluster count, 10, is the interval floor of the package:
below it a fit takes ``"few_cluster_plugin"``, because no registered study measures an interval
there.

**Primary.** In-sample ``LTMLE`` on one treatment node with ``id=``, on the informative law of
:mod:`tests.studies.clustered_unequal_laws` at 20 clusters, 1,000 replications. The reference is
R ``ltmle`` 1.3-0 with ``id=``, ``variance.method = "ic"``, the exact propensity as a numeric
``gform`` and the main-terms formula ``Q.kplus1 ~ W1 + W2``, stratified on the arm. The point
and the household standard error pair exactly. The intervals differ by a declared convention:
``ltmle`` uses t with ``J - 1`` degrees of freedom and this package ``J - 2``, so this
package's coverage is never lower. Each draw holds about 200 rows; the rows publish the nominal
``n = 200``, as the schema requires.

**Property ``few_cluster_reference``.** Five fits share each draw: cross-fitted ``TMLE``,
fold-evaluated ``TMLE``, in-sample ``TMLE``, cross-fitted ``DRTMLE`` and in-sample ``LTMLE``.
They cross 10, 20 and 30 clusters with equal clusters of 10 rows (``clustered_dgp``) and
with the informative law. ``tests.studies.clustered_few_cluster_properties`` states the cells.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_unequal_laws as laws
from tests.studies.canonical_ltmle import (
    LTMLE_SOURCE_COMMIT,
    LTMLE_TARBALL_SHA256,
    LTMLE_VERSION,
    R_BASE_IMAGE,
    regimen_initials,
    regimen_rows,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM

PRIMARY_REPLICATES = 1_000
#: The nominal row count: 20 clusters of mean size 10.
PRIMARY_N = 200
PROPERTY_REPLICATES = 4_000
SEED = 20262411
RESAMPLING_SEED = 20262412
SCENARIO = "few_unequal_informative_j20"
G_BOUNDS = (1e-8, 1.0)
REGIMENS = {"never": 0, "always": 1}
REFERENCE = "never"
ESTIMANDS = (
    "ey_regimen[never]",
    "ey_regimen[always]",
    "ate_regimen[always vs never]",
)
#: The reported names and the law's truth keys they read.
TRUTH_KEYS = {
    "ey_regimen[never]": "ey0",
    "ey_regimen[always]": "ey1",
    "ate_regimen[always vs never]": "ate",
}


#: The property grid: five fits, three cluster counts, two size laws, and the arms of
#: :func:`arms`.
FITS = (
    "tmle_crossfit",
    "tmle_cv_evaluation",
    "tmle_in_sample",
    "drtmle_crossfit",
    "ltmle_in_sample",
)
#: The grid starts at 10, which sets the package floor. A pre-run probe of 400 draws
#: at J = 4 found 0.25% to 9% of fits impossible on the draw itself: the outcome does not vary
#: among one plan's followers, or a training complement holds no outcome of one class. The
#: framework refuses a cell that lost a replication, and no learner setting repairs a draw.
#: At J = 10 the probe found none in 3,000 fits. F28 owns 4 to 9 clusters.
CLUSTER_COUNTS = (10, 20, 30)
SIZE_LAWS = ("equal10", "unequal_informative")
#: Each arm and its role. ``normal_reference`` is reported only.
ARMS = (
    ("t_reference", "positive"),
    ("iid_t_control", "control"),
    ("normal_reference", "diagnostic"),
)
#: The reported arm of the fold-evaluated fit that keeps t with J - 2 degrees of freedom,
#: beside the min(J - 2, J - V) the fit reports, so the run measures what the fold rule buys.
FOLD_ARM = ("t_j_minus_2_reference", "diagnostic")
FAMILY = "few_cluster_reference"
#: The validation folds of every cross-fitted fit; each grid count holds 2 clusters per fold.
N_FOLDS = 5


def arms(fit: str) -> tuple[tuple[str, str], ...]:
    """The arms one fit publishes, each with its role."""
    return (*ARMS, FOLD_ARM) if fit == "tmle_cv_evaluation" else ARMS


def expected_reference_df(fit: str, clusters: int) -> int:
    """The df a fit must report: J - 2, or min(J - 2, J - V) for the fold-evaluated fit."""
    if fit == "tmle_cv_evaluation":
        return min(clusters - 2, clusters - N_FOLDS)
    return clusters - 2


def cell_name(fit: str, sizes: str, clusters: int, arm: str) -> str:
    """The published name of one property cell."""
    return f"{fit}__{sizes}__j{clusters}__{arm}"


PUBLISHED_CELLS = tuple(
    cell_name(fit, sizes, clusters, arm)
    for fit in FITS
    for sizes in SIZE_LAWS
    for clusters in CLUSTER_COUNTS
    for arm, _ in arms(fit)
)

STUDY = StudyRecord(
    name="clustered TMLE intervals on a t reference at few clusters",
    slug="clustered-few-cluster-tmle",
    artifacts=ROOT / "tests" / "canonical" / "ltmle_few_cluster",
    document="docs/technical-reference/method-evidence/clustered-few-cluster-tmle.md",
    anchor="clustered-tmle-intervals-on-a-t-reference-at-few-clusters",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="ltmle",
    modules=(
        "tests/studies/clustered_few_cluster_tmle.py",
        "tests/studies/clustered_few_cluster_properties.py",
        "tests/studies/clustered_unequal_laws.py",
        "tests/studies/clustered_unequal_cvtmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/fractional_glm.py",
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
    runner_module="tests.studies.clustered_few_cluster_tmle",
    properties_module="tests.studies.clustered_few_cluster_properties",
    property_cells={FAMILY: PUBLISHED_CELLS},
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "ltmle_version": LTMLE_VERSION,
    "ltmle_source_commit": LTMLE_SOURCE_COMMIT,
    "ltmle_tarball_sha256": LTMLE_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "clusters_per_draw": PRIMARY_N // 10,
    "cluster_size": "uniform on 2 to 18, mean 10, drawn per cluster",
    "realized_rows": "random, mean 200; every row publishes the nominal n = 200",
    "law": (
        f"informative law, delta = {laws.DELTA}, gamma = {laws.GAMMA}, effect modifier "
        f"{laws.EFFECT_MODIFIER}"
    ),
    "cross_fit": False,
    "treatment_nodes": 1,
    "outcome_learner": "quasibinomial GLM, Q.kplus1 ~ W1 + W2 among the followers of each plan",
    "treatment_mechanism": "exact",
    "variance_method": "ic, summed within clusters",
    "reference_distribution": (
        "Student t with J - 2 degrees of freedom here; R ltmle uses J - 1 below 100 clusters"
    ),
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "regimens": list(REGIMENS),
}


class ExactLongitudinalPropensity(BaseEstimator):
    """The informative law's propensity at the one treatment node, from ``[W1, W2]``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ExactLongitudinalPropensity:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != 2:  # pragma: no cover - a changed design is a contract failure
            raise ValueError(f"unexpected treatment design {matrix.shape}")
        probability = laws.informative_propensity(matrix[:, 0], matrix[:, 1])
        return np.column_stack([1.0 - probability, probability])


def law_truth() -> dict[str, float]:
    """The row-weighted truth of every reported name."""
    truth = laws.informative_truth()
    return {name: float(truth[key]) for name, key in TRUTH_KEYS.items()}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of ``n // 10`` clusters of the informative law."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame = laws.draw_informative(n // 10, np.random.default_rng(seed))
    return frame, law_truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame, **overrides: Any) -> Any:
    """In-sample LTMLE on one node, with the exact propensity and ``id=``."""
    settings: dict[str, Any] = {
        "reference": REFERENCE,
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": QuasiBinomialGLM(),
        "treatment_learner": ExactLongitudinalPropensity(),
        "n_folds": 1,
        "g_bounds": G_BOUNDS,
        "simultaneous": False,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
        **overrides,
    }
    return LTMLE(REGIMENS, **settings).fit(
        frame, outcome="Y", treatment=["A"], baseline=["W1", "W2"], id="cluster"
    )


def rows_from_result(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    return regimen_rows(
        STUDY,
        result,
        truth,
        regimen_initials(result, REGIMENS, ESTIMANDS[2:]),
        ESTIMANDS,
        scenario,
        replicate,
        n=PRIMARY_N,
    )


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    return rows_from_result(fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    rows = cleverly_rows(frame, truth, scenario, replicate)
    sample = frame.copy()
    sample.insert(0, "nominal_n", n)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, fit, and keep each sample for the R comparator."""
    outcomes = map_parallel(
        _replicate, [((SCENARIO, replicate, n),) for replicate in range(replicates)], n_jobs=n_jobs
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

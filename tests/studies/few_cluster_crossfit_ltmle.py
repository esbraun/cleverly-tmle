"""Registered evidence for the cross-fitted clustered ``LTMLE`` on a t reference at 20 and 30 clusters.

Below 40 clusters with positive weight mass a clustered estimate reports its interval on a
Student t reference with ``J - 2`` degrees of freedom. ``clustered-few-cluster-tmle`` measures
that rule for in-sample ``LTMLE`` from 10 clusters. This study measures it for the cross-fitted
``LTMLE`` with five whole-cluster folds, whose targeting pools every follower, so the fold count
does not enter the degrees of freedom. The grid starts at 20, so a cross-fitted fit below 20
clusters takes ``"few_cluster_plugin"``
(``cleverly._inference_status.MINIMUM_CROSS_FITTED_LONGITUDINAL_CLUSTERS``).

The subject is the ``clustered-cross-fitted-ltmle`` subject. The law is the end-of-study law
of :mod:`tests.studies.clustered_longitudinal_laws` with the **scaled** latent
:math:`u = s v`, :math:`v` uniform on :math:`[0.5, 1]`: with a Rademacher latent every
training cluster of a few-cluster fit can share one sign, and the fit then has no outcome
variation to fit (about 1 to 2% of draws at 10 clusters). The registered harness refuses a cell
that lost a replication.

**Primary.** 20 clusters of 40 rows, 1,000 replications, no comparator. No pinned
implementation cross-fits a clustered longitudinal fit at few clusters with a t reference.

**Property ``few_cluster_reference``.** The family and the verdict rule of
``clustered-few-cluster-tmle``: a ``t_reference`` arm (99% exact coverage lower bound at least
0.90, bias within 0.25 empirical SD), an ``iid_t_control`` arm (SE-ratio upper bound below
0.80), and a reported ``normal_reference`` arm.

Publication policy is ``reporting``, declared before any run. Every red cell is published and
F28 owns it. No margin, budget, law, latent or learner changes after a result is seen.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_longitudinal_laws as law
from tests.studies.canonical_ltmle import regimen_initials, regimen_rows
from tests.studies.canonical_ltmle_crossfit import CONTRAST_NAMES, ESTIMANDS, REGIMENS
from tests.studies.clustered_crossfit_ltmle import CLUSTER_SIZE, ID, NODES, subject
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

LATENT = "scaled"
PRIMARY_REPLICATES = 1_000
PRIMARY_CLUSTERS = 20
PRIMARY_N = PRIMARY_CLUSTERS * CLUSTER_SIZE
PROPERTY_REPLICATES = 4_000
SEED = 20262521
RESAMPLING_SEED = 20262522
SCENARIO = "few_equal_j20"
FAMILY = "few_cluster_reference"
TARGET = CONTRAST_NAMES[0]
#: The declared grid. A cross-fitted fit below 20 clusters takes ``"few_cluster_plugin"``,
#: because no cross-fitted cell measures 4 to 19 clusters. A grid cell with any
#: failure in the declared failure-only probe is dropped before the run and becomes an F28 limit.
#: The probe of 2,000 draws per cell found 3 failures at ``equal40``, 10 clusters, and 8 at
#: ``unequal40``, 10 clusters, and none at 20 or 30, so both 10-cluster cells are dropped. No
#: cross-fitted cell measures 4 to 19 clusters, and F28 records that.
CLUSTER_COUNTS = (20, 30)
#: The failure-only probe record: failed fits per declared cell, recorded before any run. The
#: first probe read the first :data:`FAILURE_PROBE_DRAWS` streams of each cell. A second probe
#: read streams 2,000 to 3,999 of each kept cell, so every declared property stream is probed,
#: and found no failure. If a failure still occurs in the run, that cell is dropped to F28 and
#: the run repeats without it, with no other change.
FAILURE_PROBE = {
    "equal40/J10": 3,
    "equal40/J20": 0,
    "equal40/J30": 0,
    "unequal40/J10": 8,
    "unequal40/J20": 0,
    "unequal40/J30": 0,
}
#: The second probe: failed fits on streams 2,000 to 3,999 of each kept cell.
FAILURE_PROBE_UPPER_STREAMS = {
    "equal40/J20": 0,
    "equal40/J30": 0,
    "unequal40/J20": 0,
    "unequal40/J30": 0,
}
#: Failed fits on all 1,000 declared primary streams, recorded before any run.
PRIMARY_FAILURE_PROBE = 0
#: The control design record, RM36's rule: on a design-labelled stream
#: (``stream_seed(STUDY, "design", "<sizes>/J<count>", i)``, 300 draws per cell), the mean
#: reported SE over the empirical SD of the target, for the cluster-robust and the IID SE.
CONTROL_DESIGN = {
    "equal40/J20": (1.0252, 0.6769),
    "equal40/J30": (1.0222, 0.6733),
    "unequal40/J20": (0.9429, 0.5953),
    "unequal40/J30": (0.9597, 0.5945),
}
SIZE_LAWS = ("equal40", "unequal40")
#: Each arm and its role. ``normal_reference`` is reported only.
ARMS = (
    ("t_reference", "positive"),
    ("iid_t_control", "control"),
    ("normal_reference", "diagnostic"),
)
#: The failure-only probe the declaration rests on: draws per cell and the failures seen.
FAILURE_PROBE_DRAWS = 2_000


def cell_name(sizes: str, clusters: int, arm: str) -> str:
    """The published name of one property cell."""
    return f"ltmle_crossfit__{sizes}__j{clusters}__{arm}"


PUBLISHED_CELLS = tuple(
    cell_name(sizes, clusters, arm)
    for sizes in SIZE_LAWS
    for clusters in CLUSTER_COUNTS
    for arm, _ in ARMS
)

STUDY = StudyRecord(
    name="cross-fitted clustered longitudinal TMLE on a t reference at few clusters",
    slug="few-cluster-cross-fitted-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_crossfit_ltmle",
    document="docs/technical-reference/method-evidence/few-cluster-cross-fitted-longitudinal-tmle.md",
    anchor="cross-fitted-clustered-longitudinal-tmle-on-a-t-reference-at-few-clusters",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-few-cluster-cross-fitted-ltmle",
    reference=None,
    modules=(
        "tests/studies/few_cluster_crossfit_ltmle.py",
        "tests/studies/few_cluster_crossfit_properties.py",
        "tests/studies/clustered_crossfit_ltmle.py",
        "tests/studies/clustered_longitudinal_laws.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_clustered_tmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.few_cluster_crossfit_ltmle",
    properties_module="tests.studies.few_cluster_crossfit_properties",
    property_cells={FAMILY: PUBLISHED_CELLS},
    publication_policy="reporting",
)

CONFIGURATION = {
    "clusters_per_draw": PRIMARY_CLUSTERS,
    "cluster_size": CLUSTER_SIZE,
    "law": (
        f"make_longitudinal with a mean-zero cluster component on the final outcome, "
        f"delta = {law.DELTA}, latent u = s v with s a sign and v uniform on [0.5, 1]"
    ),
    "cross_fit": True,
    "outer_folds": 5,
    "folds": "grouped random_partition folds from the cluster labels and random_state=0",
    "targeting": "pooled over every follower",
    "outcome_learner": "quasibinomial GLM, the canonical-ltmle-crossfit node regressions",
    "treatment_mechanism": "exact, the law's own",
    "variance_method": "ic, summed within clusters",
    "reference_distribution": "Student t with J - 2 degrees of freedom",
    "simultaneous_intervals": False,
    "regimens": list(REGIMENS),
}


def draw(sizes: str, clusters: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of ``clusters`` clusters of ``sizes`` with the scaled latent."""
    return law.draw_end_of_study(clusters, sizes, seed, latent=LATENT)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw of ``n // 40`` clusters of 40 rows."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = draw(f"equal{CLUSTER_SIZE}", n // CLUSTER_SIZE, seed)
    return frame, {name: float(truth[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(frame: pd.DataFrame) -> Any:
    """The ``clustered-cross-fitted-ltmle`` subject."""
    return subject().fit(frame, outcome="Y", id=ID, **NODES)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    result = fit_cleverly(frame)
    return regimen_rows(
        STUDY,
        result,
        truth,
        regimen_initials(result, REGIMENS, CONTRAST_NAMES),
        ESTIMANDS,
        scenario,
        replicate,
        n=result.n,
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    rows = cleverly_rows(frame, truth, scenario, replicate)
    sample = frame.copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

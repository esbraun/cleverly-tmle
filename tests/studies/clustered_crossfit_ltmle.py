"""Registered evidence for the cross-fitted clustered end-of-study longitudinal TMLE.

The subject is the ``canonical-ltmle-crossfit`` subject with ``id=``: quasi-binomial node
regressions, the law's own treatment and censoring mechanism, five whole-cluster folds drawn
by ``random_partition`` from the cluster labels and ``random_state=0``, and the pooled
targeting. The law is the end-of-study law of
:mod:`tests.studies.clustered_longitudinal_laws` with the Rademacher latent, 100 clusters of 40
rows. Its regimen truths are the ``make_longitudinal`` quadrature truths exactly.

**Primary.** 1,600 replications, paired with R ``lmtp`` 1.5.4 and ``ife`` 0.2.3 through
``lmtp_tmle_with_folds`` with ``id``, the realized grouped fold column, and the exact per-node
density ratios. At equal sizes the ``ife`` cluster-mean rule equals the cluster-sum rule, so the
paired standard errors measure one quantity. ``lmtp`` fluctuates on the training rows and this
package pools, so the constructions differ as in ``canonical-ltmle-crossfit``.

**Properties.** ``tests.studies.clustered_crossfit_ltmle_properties`` declares four
``clustered_inference`` pairs and one ``simultaneous_coverage`` band cell.

Publication policy was declared ``gated``. The red-cell route was declared before any run: a red cell is
diagnosed for a defect first. A defect found is fixed and the study regenerated once, with the
revision declared in ``tests/canonical/provenance-revisions.md``. With none found, the record
switches to ``reporting`` with owner F28 before one re-run. No margin, budget, law, latent,
size or learner changes after a result is seen.

**The route was taken.** The first full run passed every ``clustered_inference`` cell and every
paired comparison, and read one red cell: ``clustered_regimens__simultaneous_band`` covered
0.9333, 99% interval 0.919 to 0.946, against the calibration band 0.92 to 0.98. The diagnosis
found no defect. The band draws one multiplier per cluster from the fit's cluster sums, as the
fast tier pins. Its mean critical value is 2.387 against the design oracle 2.409, and the max-t
statistic itself has a 95% quantile of 2.50, because the five pointwise SE ratios read 0.976 to
0.997 at 100 clusters. With the oracle critical value the band would cover 0.939. The
point-treatment clustered band of ``default-simultaneous-bands`` reads 0.938 at 200 clusters,
the same finite-sample shortfall of a cluster multiplier band. So the record is ``reporting``,
F28 owns the cell, and the declared run is repeated once with nothing else changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_longitudinal_laws as law
from tests.studies.canonical_clustered_tmle import IFE_SHA256, IFE_VERSION
from tests.studies.canonical_ltmle import (
    KnownLongitudinalMechanism,
    QuasiBinomialGLM,
    regimen_initials,
    regimen_rows,
)
from tests.studies.canonical_ltmle_crossfit import (
    CONTRAST_NAMES,
    ESTIMANDS,
    G_BOUNDS,
    REFERENCE,
    REGIMENS,
)
from tests.studies.canonical_ltmle_crossfit import REFERENCE_METADATA as LMTP_METADATA
from tests.studies.evidence.constructions import (
    POOLED_LONGITUDINAL_CROSS_FIT,
    TRAINING_FOLD_FLUCTUATION,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_600
#: Rows per cluster and clusters per draw of every scenario but the unequal one.
CLUSTER_SIZE = 40
CLUSTERS = 100
PRIMARY_N = CLUSTERS * CLUSTER_SIZE
SEED = 20262511
RESAMPLING_SEED = 20262512
SCENARIO = "clustered_end_of_study"
N_FOLDS = 5
LEARNER_FOLDS = 2
RANDOM_STATE = 0
ID = "id"

NODES: dict[str, Any] = {
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}

#: The ``clustered_inference`` pairs, as (positive cell, control cell).
CLUSTERED_PAIRS = (
    ("cluster_robust_static", "iid_control_static"),
    ("cluster_robust_dynamic", "iid_control_dynamic"),
    ("cluster_robust_unequal", "iid_control_unequal"),
    ("cluster_robust_survival", "iid_control_survival"),
)
#: The joint label of the band cell.
BAND_LABEL = "clustered_regimens"

PROPERTY_CELLS = {
    "clustered_inference": tuple(cell for pair in CLUSTERED_PAIRS for cell in pair),
    "simultaneous_coverage": (
        f"{BAND_LABEL}__simultaneous_band",
        f"{BAND_LABEL}__pointwise_joint_control",
    ),
}

STUDY = StudyRecord(
    name="clustered cross-fitted end-of-study longitudinal TMLE",
    slug="clustered-cross-fitted-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_clustered_ltmle",
    document="docs/technical-reference/method-evidence/clustered-cross-fitted-longitudinal-tmle.md",
    anchor="clustered-cross-fitted-end-of-study-longitudinal-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-clustered-cross-fitted-ltmle",
    reference="lmtp",
    modules=(
        "tests/studies/clustered_crossfit_ltmle.py",
        "tests/studies/clustered_crossfit_ltmle_properties.py",
        "tests/studies/clustered_longitudinal_laws.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle_survival_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_clustered_tmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_clustered_ltmle/run_study.R",
    ),
    runner_module="tests.studies.clustered_crossfit_ltmle",
    properties_module="tests.studies.clustered_crossfit_ltmle_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": POOLED_LONGITUDINAL_CROSS_FIT,
    "reference_construction": TRAINING_FOLD_FLUCTUATION,
    "outcome_kind": "end_of_study",
    "horizon_mode": "terminal_only",
    "r_survival_outcome": False,
    "cross_fit": True,
    "outer_folds": N_FOLDS,
    "folds": (
        "grouped random_partition folds from the cluster labels and random_state=0; R lmtp "
        "receives the realized assignment"
    ),
    "learner_folds": LEARNER_FOLDS,
    "clusters_per_draw": CLUSTERS,
    "cluster_size": CLUSTER_SIZE,
    "law": (
        f"make_longitudinal with a mean-zero cluster component on the final outcome, "
        f"delta = {law.DELTA}, a Rademacher latent per cluster and first-node arm"
    ),
    "simultaneous_intervals": False,
    "variance_method": "ic, summed within clusters",
    "reference_variance": "ife cluster means, equal to cluster sums at equal sizes",
    "g_bounds": list(G_BOUNDS),
    "regimens": list(REGIMENS),
    "outcome_designs": [["W1", "W2"], ["W1", "W2", "L2"]],
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node",
}

#: Provenance of the comparator, for the manifest's ``generated_with.reference`` block.
REFERENCE_METADATA = {**LMTP_METADATA, "ife_version": IFE_VERSION, "ife_sha256": IFE_SHA256}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of ``n // 40`` clusters of 40 rows."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = law.draw_end_of_study(n // CLUSTER_SIZE, f"equal{CLUSTER_SIZE}", seed)
    return frame, {name: float(truth[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def subject(regimens: Mapping[str, Any] = REGIMENS, *, simultaneous: bool = False) -> LTMLE:
    """The ``canonical-ltmle-crossfit`` subject; the fit passes ``id=``."""
    return LTMLE(
        dict(regimens),
        reference=REFERENCE,
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        simultaneous=simultaneous,
        max_iter=100,
        tol=1e-10,
        random_state=RANDOM_STATE,
    )


def fit_cleverly(frame: pd.DataFrame, *, simultaneous: bool = False) -> Any:
    """The cross-fitted clustered fit of the end-of-study law."""
    return subject(simultaneous=simultaneous).fit(frame, outcome="Y", id=ID, **NODES)


def _rows_from_result(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
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


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return _rows_from_result(fit_cleverly(frame), truth, scenario, replicate)


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    sample = frame.copy()
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, _rows_from_result(result, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, fit, and keep each sample and its realized grouped folds for R ``lmtp``."""
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

r"""Declared evidence for the clustered interval floor of the cross-fitted LTMLE regimen means
(X54 part (k), S-k2).

``few-cluster-floor-cross-fitted-ltmle`` certifies the cross-fitted ``LTMLE`` contrast from 4
clusters (``CLUSTER_INTERVAL_FLOORS["ltmle_cross_fitted"]``); its regimen means read the
longitudinal default, 20. This study measures the two static regimen means,
``ey_regimen[always]`` and ``ey_regimen[never]``, of the same fit at every count of
:data:`LADDER`, so that :func:`tests.studies.few_cluster_floor.arm_mean_floor` can set their
floor below the default or above it.

**Law and fit.** Those of ``few-cluster-floor-cross-fitted-ltmle``
(:func:`tests.studies.few_cluster_floor_ltmle.draw` and
:func:`~tests.studies.few_cluster_floor_ltmle.subject`): the end-of-study law with the scaled
latent at ``equal40`` and ``unequal40``, :math:`V = \min(5, J)` whole-cluster folds, the law's
own mechanisms, the study quasibinomial GLM. No arm-mean variant of the law is declared: the
150-draw probe of plan 5.3 (``diag/k2probe``) reads each regimen mean's paired IID-to-cluster
ratio at 0.64 to 0.81, close to the contrast's 0.66 to 0.77, so both means carry the cluster
dependence the contrast does and clause 1 is gated on both laws.

**Cells.** One fit per ``(law, J, replicate)`` writes the ``few-cluster-floor-cross-fitted-ltmle``
arms of each of three estimands (:func:`cell_name` puts the estimand's tag after the fit): the
two regimen means gated (``t_reference`` positive, ``iid_t_control`` control) and the
always-versus-never contrast reported.

**Primary.** ``ey_regimen[always]`` at 10 clusters of ``equal40``, 1,000 replications, no
comparator; the pilot's G1 pass probability there at 1,000 replications is recorded
(:data:`PILOT_PRIMARY_PASS`). The policy is ``reporting``, declared before any run.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import pandas as pd

from cleverly._inference_status import DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS
from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_arm_means as point
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_ltmle as source
from tests.studies.canonical_cvtmle import fitted_rows
from tests.studies.canonical_ltmle import regimen_initials, regimen_rows
from tests.studies.clustered_crossfit_ltmle import CLUSTER_SIZE, ID, NODES
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate, stream_seed

SEED: Final[int] = 20265461
RESAMPLING_SEED: Final[int] = 20265462
FAMILY: Final[str] = floor.FAMILY
FIT: Final[str] = source.FIT
FLOOR_FAMILY: Final[str] = "ltmle_cross_fitted"
RED_OWNER: Final[str] = point.RED_OWNER
SIZE_LAWS: Final[tuple[str, ...]] = source.SIZE_LAWS
#: The laws whose control clause 1 gates: both, since each regimen mean carries the latent.
CLAUSE_ONE_LAWS: Final[tuple[str, ...]] = SIZE_LAWS
#: The ladder (plan 5.3): the contrast floor 4 to the default 20, and 30.
LADDER: Final[tuple[int, ...]] = (*range(4, 21), 30)
ANCHOR: Final[int] = DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS
CONTRAST: Final[str] = source.TARGET
ARM_MEANS: Final[tuple[str, ...]] = ("ey_regimen[never]", "ey_regimen[always]")
ESTIMANDS: Final[tuple[str, ...]] = (*ARM_MEANS, CONTRAST)
TAGS: Final[dict[str, str]] = {name: point.TAGS[name] for name in ESTIMANDS}
ESTIMAND_OF_TAG: Final[dict[str, str]] = {tag: name for name, tag in TAGS.items()}
#: Replications per cell before the pilot: 4,000, raised off the committed pilot.
PROPERTY_REPLICATES: Final[int] = 4_000
#: The pilot's predicted reds and their readings; empty until the pilot.
PILOT_PREDICTED_REDS: Final[dict[str, str]] = {}
#: Whether the plug-in family figure stays under the target at the cap (M5); empty until the pilot.
PILOT_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: The pilot's predicted arm-mean floor; ``None`` until the pilot.
PILOT_PREDICTED_FLOOR: Final[int | None] = None
#: The pilot's G1 pass probability of the primary cell at 1,000 replications; ``None`` until
#: the pilot.
PILOT_PRIMARY_PASS: Final[float | None] = None
#: The declared budget of the property phase (plan 5.3), in CPU hours.
DECLARED_CPU_HOURS: Final[float] = 10.0

PRIMARY_REPLICATES: Final[int] = 1_000
PRIMARY_LAW: Final[str] = "equal40"
PRIMARY_ESTIMAND: Final[str] = "ey_regimen[always]"
PRIMARY_CLUSTERS: Final[int] = 10
PRIMARY_N: Final[int] = PRIMARY_CLUSTERS * CLUSTER_SIZE
SCENARIO: Final[str] = f"arm_means_{PRIMARY_LAW}_j{PRIMARY_CLUSTERS}"


def arms(estimand: str) -> tuple[tuple[str, str], ...]:
    """The arms of one estimand with their roles: those of ``few-cluster-floor-cross-fitted-ltmle``
    for a regimen mean, every arm reported for the contrast."""
    published = source.arms(0)
    if estimand in ARM_MEANS:
        return published
    return tuple((arm, floor.DIAGNOSTIC) for arm, _ in published)


def cell_name(estimand: str, sizes: str, clusters: int, arm: str) -> str:
    """The published name of one property cell: the estimand's tag follows the fit."""
    return f"{FIT}-{TAGS[estimand]}__{sizes}__j{clusters}__{arm}"


def parse(cell: str) -> tuple[str, str, int, str]:
    """``(estimand, law, J, arm)`` of a published cell name."""
    token, sizes, clusters, arm = floor.parse(cell)
    _, tag = token.split("-", 1)
    return ESTIMAND_OF_TAG[tag], sizes, clusters, arm


def grid() -> tuple[tuple[str, int], ...]:
    """Every ``(law, J)`` of the property grid."""
    return tuple((sizes, clusters) for sizes in SIZE_LAWS for clusters in LADDER)


def requested(budget: int | None = None) -> int:
    """A cell's requested replications, capped at ``budget``."""
    return PROPERTY_REPLICATES if budget is None else min(PROPERTY_REPLICATES, budget)


PUBLISHED_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(estimand, sizes, clusters, arm)
    for sizes, clusters in grid()
    for estimand in ESTIMANDS
    for arm, _ in arms(estimand)
)

STUDY = StudyRecord(
    name="the clustered interval floor of the cross-fitted LTMLE regimen means",
    slug="few-cluster-arm-means-cross-fitted-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_arm_means_ltmle",
    document="docs/technical-reference/method-evidence/few-cluster-arm-means-cross-fitted-ltmle.md",
    anchor="the-clustered-interval-floor-of-the-cross-fitted-ltmle-regimen-means",
    scenarios={SCENARIO: source.ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    failure_bound={"cleverly": floor.PRIMARY_FAILURE_BOUND},
    modules=(
        "tests/studies/few_cluster_arm_means_ltmle.py",
        "tests/studies/few_cluster_arm_means_ltmle_properties.py",
        "tests/studies/few_cluster_arm_means.py",
        "tests/studies/few_cluster_arm_means_properties.py",
        "tests/studies/few_cluster_floor_ltmle.py",
        "tests/studies/few_cluster_floor.py",
        "tests/studies/few_cluster_floor_pilot.py",
        "tests/studies/clustered_longitudinal_laws.py",
        "tests/studies/clustered_crossfit_ltmle.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.few_cluster_arm_means_ltmle",
    properties_module="tests.studies.few_cluster_arm_means_ltmle_properties",
    property_cells={FAMILY: PUBLISHED_CELLS},
    publication_policy="reporting",
)

CONFIGURATION = {
    "law": f"the law of {source.STUDY.slug} (scaled latent)",
    "size_laws": list(SIZE_LAWS),
    "ladder": list(LADDER),
    "fit": f"the {source.STUDY.slug} subject: min(5, J) whole-cluster folds, known mechanisms",
    "estimands": list(ESTIMANDS),
    "clause_one_laws": list(CLAUSE_ONE_LAWS),
    "primary": f"{SCENARIO}: {PRIMARY_ESTIMAND}, {PRIMARY_CLUSTERS} clusters",
}


def draw_seed(sizes: str, clusters: int, replicate: int, stream: str = "property") -> int:
    """The seed of one draw of a ``(law, J)`` cell on a named stream (``property`` or ``pilot``)."""
    return int(stream_seed(STUDY, stream, f"{sizes}/J{clusters}", replicate))


def draw(sizes: str, clusters: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw and its truths (:func:`tests.studies.few_cluster_floor_ltmle.draw`)."""
    return source.draw(sizes, clusters, seed)


def fit_estimates(frame: pd.DataFrame, clusters: int) -> dict[str, Any]:
    """Each reported estimate of one draw, its ``reference_df`` checked where it supplies one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = source.subject(clusters).fit(frame, outcome="Y", id=ID, **NODES)
    out = {}
    for name in ESTIMANDS:
        estimate = result[name]
        if estimate.supplies_inference and estimate.reference_df != clusters - 2:
            raise AssertionError(
                f"{name} at J = {clusters}: reference_df {estimate.reference_df}, not J - 2"
            )
        out[name] = estimate
    return out


def fit_rows(
    sizes: str,
    clusters: int,
    replicate: int,
    drawn: tuple[pd.DataFrame, Mapping[str, float]],
    *,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """Every row of one draw: each estimand's arms. Only the fit raising loses the key."""
    frame, truths = drawn
    rows = []
    for name, estimate in fit_estimates(frame, clusters).items():
        published = arms(name)
        readings = floor.reference_readings(
            estimate,
            clusters=clusters,
            covariates=source.BASELINE_COVARIATES,
            arms=[arm for arm, _ in published],
        )
        rows += [
            floor.arm_row(
                readings[arm],
                cell=cell_name(name, sizes, clusters, arm),
                role=role,
                replicate=replicate,
                n=CLUSTER_SIZE * clusters,
                requested=requested(budget),
                truth=float(truths[name]),
                clusters=clusters,
            )
            for arm, role in published
        ]
    return rows


# ------------------------------------------------------------------------------------ primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw of ``n // 40`` clusters of 40 rows."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = draw(PRIMARY_LAW, n // CLUSTER_SIZE, seed)
    return frame, {name: float(truth[name]) for name in source.ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: every estimand of the subject at 10 clusters."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = source.subject(PRIMARY_CLUSTERS).fit(frame, outcome="Y", id=ID, **NODES)
    return regimen_rows(
        STUDY,
        result,
        truth,
        regimen_initials(result, source.REGIMENS, source.CONTRAST_NAMES),
        source.ESTIMANDS,
        scenario,
        replicate,
        n=result.n,
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )


__all__: Sequence[str] = (
    "LADDER",
    "STUDY",
    "arms",
    "draw",
    "draw_and_fit",
    "fit_rows",
    "grid",
)

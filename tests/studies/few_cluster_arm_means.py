r"""Declared evidence for the clustered interval floor of the arm means (X54 part (k)).

A family's interval floor (``CLUSTER_INTERVAL_FLOORS``) certifies the contrast its study
measured; every other estimate reads the default floor (10 for a point fit, 20 for ``LTMLE``).
This study measures the two arm means of every point floor family and of in-sample ``LTMLE``
(:math:`E[Y^1]` and :math:`E[Y^0]`, the always-treat and never-treat regimen means) at every
count of a ladder that runs from the family's contrast floor past the default, so that a floor
for the arm means can be set by :func:`tests.studies.few_cluster_floor.arm_mean_floor`, below
the default or above it.

**Laws** (:func:`law_parameters`). The two ATE laws of ``few-cluster-floor-tmle``, ``equal10``
and ``unequal_informative``, and two arm-mean laws, ``equal10_arms`` and ``unequal_arms``: the
same laws with the main latent coefficient :math:`c` (:data:`ARM_LATENT`) set so that both arms
carry the cluster latent. On the ATE laws the latent enters the logit as :math:`0.6 w_3 + m a w_3`
(:math:`m = 8` on ``equal10``, 0 on ``unequal_informative``), so the untreated arm carries almost
no cluster dependence and its IID control has nothing to detect (X54 plan rev 2.4, 5.2, F1).
``equal10_arms`` sets :math:`c = -4` with :math:`m = 8` (arm coefficients -4 and +4);
``unequal_arms`` sets :math:`c = 3` with :math:`m = 0` and the size terms of
``unequal_informative``. The 150-draw probe of the plan (``diag/review2``) reads each arm's paired
IID-to-cluster ratio at 0.51 to 0.58 from 4 to 30 clusters, under the declared 0.70. Truths are
exact (:func:`tests.studies.clustered_unequal_laws.informative_truth`).

**Fits** (:data:`FITS`, the fits of ``few-cluster-floor-tmle`` with three estimands each):

=========================  ===================================================  ================
family                     configurations                                       J
=========================  ===================================================  ================
``tmle_in_sample``         known, estimated (``"auto"``)                        6 to 10, 15 to 39
``tmle_stacked``           known, estimated; d5 known and estimated (unequal)   4 to 10, 15 to 39
``tmle_fold_evaluated``    known, estimated; equal-size laws only (Fix-4)       8 to 10, 15 to 39
``drtmle_known``           declared mechanism, ``DRTMLE_G_BOUNDS``              8 to 10, 15 to 39
``ltmle_in_sample``        the S-a construction (exact propensity, study GLM)   6 to 20, 30, 39
=========================  ===================================================  ================

An estimated mechanism takes ``g_bounds="auto"`` (:data:`MECHANISM_BOUNDS`); a known one keeps
the exact bounds; DR-TMLE keeps its declared ``(0.01, 0.99)`` on the reduced regression.

**Cells.** Every fit of a ``(law, J, replicate)`` reads one draw, and each fit writes one group
of arms per estimand (:func:`cell_name` puts the estimand's tag, :data:`TAGS`, after the fit).
The arm means are gated on all four laws: ``t_reference`` (positive) and ``iid_t_control``
(control), with the reported arms of ``few-cluster-floor-tmle``. Clause 1 of the control gates
on the arm-mean laws only (:data:`ARM_MEAN_LAWS`, ``clause_one_laws``); on the ATE laws it is
reported and its red reads ``control unpowered by law``. The contrast of each fit is reported,
except on the ATE laws for the estimated configurations at ``"auto"``, where it is gated as the
transfer check of the shipped contrast floors (plan 1.4 and 5.4).

**Primary.** In-sample known TMLE on ``equal10_arms`` (:func:`primary_choice`): the first of
``ey1`` at 10, 20 and 39 clusters whose pilot G1 pass probability at 1,000 replications reaches
0.90, else the ATE at 10. No comparator. The policy is ``reporting``, declared before any run.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from cleverly._inference_status import (
    DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS,
    DEFAULT_POINT_INTERVAL_CLUSTERS,
)
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_unequal_laws as laws
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_tmle as source
from tests.studies.canonical_cvtmle import fitted_rows, rows_from_result
from tests.studies.clustered_few_cluster_properties import size_law_parameters
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate, stream_seed

SEED: Final[int] = 20265441
RESAMPLING_SEED: Final[int] = 20265442
FAMILY: Final[str] = floor.FAMILY
#: The owner of the study's reds (X54 plan rev 2.4, 5.5), in the roadmap's owner table.
RED_OWNER: Final[str] = "X54-estimand-floor"

#: The ATE laws of ``few-cluster-floor-tmle`` and the two arm-mean laws.
ATE_LAWS: Final[tuple[str, ...]] = ("equal10", "unequal_informative")
ARM_MEAN_LAWS: Final[tuple[str, ...]] = ("equal10_arms", "unequal_arms")
LAWS: Final[tuple[str, ...]] = (*ATE_LAWS, *ARM_MEAN_LAWS)
EQUAL_LAWS: Final[tuple[str, ...]] = ("equal10", "equal10_arms")
UNEQUAL_LAWS: Final[tuple[str, ...]] = ("unequal_informative", "unequal_arms")
#: Each arm-mean law's ATE law, whose size terms and effect modifier it keeps.
ARM_LAW_BASE: Final[dict[str, str]] = {
    "equal10_arms": "equal10",
    "unequal_arms": "unequal_informative",
}
#: The main latent coefficient of each arm-mean law (declared before the pilot, plan 5.2).
ARM_LATENT: Final[dict[str, float]] = {"equal10_arms": -4.0, "unequal_arms": 3.0}

#: Each fit and its floor family: the fits of ``few-cluster-floor-tmle``.
FITS: Final[dict[str, str]] = dict(source.FITS)
LTMLE_FAMILY: Final[str] = "ltmle_in_sample"
#: The ladder of each family (plan 5.2): from its contrast floor (4 for stacked, whose floor
#: ``few-cluster-floor-stacked`` sets) past the default to 39.
LADDERS: Final[dict[str, tuple[int, ...]]] = {
    "tmle_in_sample": (6, 7, 8, 9, 10, 15, 20, 30, 39),
    "tmle_stacked": (4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 39),
    "tmle_fold_evaluated": (8, 9, 10, 15, 20, 30, 39),
    "drtmle_known": (8, 9, 10, 15, 20, 30, 39),
    LTMLE_FAMILY: (*range(6, 21), 30, 39),
}
#: The default floor of each family's arm means: the anchor ``arm_mean_floor`` reads against.
ANCHORS: Final[dict[str, int]] = {
    family: (
        DEFAULT_LONGITUDINAL_INTERVAL_CLUSTERS
        if family == LTMLE_FAMILY
        else DEFAULT_POINT_INTERVAL_CLUSTERS
    )
    for family in LADDERS
}
POINT_ESTIMANDS: Final[tuple[str, ...]] = ("ey0", "ey1", "ate")
LTMLE_ESTIMANDS: Final[tuple[str, ...]] = (
    "ey_regimen[never]",
    "ey_regimen[always]",
    source.LTMLE_TARGET,
)
#: The contrast each fit reports beside its arm means.
CONTRASTS: Final[frozenset[str]] = frozenset({"ate", source.LTMLE_TARGET})
#: The tag of each estimand in a cell name.
TAGS: Final[dict[str, str]] = {
    "ey0": "ey0",
    "ey1": "ey1",
    "ate": "ate",
    "ey_regimen[never]": "never",
    "ey_regimen[always]": "always",
    source.LTMLE_TARGET: "contrast",
}
ESTIMAND_OF_TAG: Final[dict[str, str]] = {tag: name for name, tag in TAGS.items()}
#: The key of :func:`~tests.studies.clustered_unequal_laws.informative_truth` of each estimand.
TRUTH_KEYS: Final[dict[str, str]] = {
    "ey0": "ey0",
    "ey1": "ey1",
    "ate": "ate",
    "ey_regimen[never]": "ey0",
    "ey_regimen[always]": "ey1",
    source.LTMLE_TARGET: "ate",
}
#: The propensity bounds of each mechanism kind (plan 5.2): ``"auto"`` for an estimated
#: mechanism, the exact bounds for a known one. DR-TMLE and ``LTMLE`` keep their own.
MECHANISM_BOUNDS: Final[dict[str, Any]] = {"known": source.G_BOUNDS, "estimated": "auto"}
#: Replications per cell before the pilot: 4,000 per family, raised off the committed pilot.
PROPERTY_REPLICATES: Final[dict[str, int]] = dict.fromkeys(LADDERS, 4_000)
#: The pilot's predicted reds and their readings (``pilot-sizing.csv``); empty until the pilot.
PILOT_PREDICTED_REDS: Final[dict[str, str]] = {}
#: Families whose plug-in pass probability stays under the target at the cap (M5).
PILOT_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: The pilot's predicted arm-mean floor of each family; empty until the pilot.
PILOT_PREDICTED_FLOORS: Final[dict[str, int]] = {}
#: S-a's known hard cell (review F3) and its pilot G1 and G2 pass probabilities at the declared
#: R; ``None`` until the pilot.
HARD_CELL: Final[tuple[str, str, str, int]] = (
    "ltmle_in_sample",
    "ey_regimen[never]",
    "unequal_informative",
    10,
)
PILOT_HARD_CELL: Final[tuple[float, float] | None] = None
#: The declared budget of the property phase (plan 5.2), in CPU hours.
DECLARED_CPU_HOURS: Final[float] = 80.0

PRIMARY_REPLICATES: Final[int] = 1_000
PRIMARY_FIT: Final[str] = "tmle_in_sample_known"
PRIMARY_LAW: Final[str] = "equal10_arms"
#: The primary's candidates in the declared order, then its fallback (plan 5.2).
PRIMARY_CANDIDATES: Final[tuple[tuple[str, int], ...]] = (("ey1", 10), ("ey1", 20), ("ey1", 39))
PRIMARY_FALLBACK: Final[tuple[str, int]] = ("ate", 10)
#: The primary's G1 pass probability at 1,000 replications that a candidate must reach.
PRIMARY_PASS: Final[float] = 0.90
#: The declared primary cell. Before the pilot it is the first candidate; the declaration after
#: the pilot sets it to :func:`primary_choice` of the committed pilot.
PRIMARY_ESTIMAND: Final[str] = PRIMARY_CANDIDATES[0][0]
PRIMARY_CLUSTERS: Final[int] = PRIMARY_CANDIDATES[0][1]
PRIMARY_N: Final[int] = 10 * PRIMARY_CLUSTERS
SCENARIO: Final[str] = f"arm_means_{PRIMARY_LAW}_j{PRIMARY_CLUSTERS}"


def law_parameters(law: str) -> dict[str, float]:
    """The keyword arguments of :func:`~tests.studies.clustered_unequal_laws.draw_informative`
    and :func:`~tests.studies.clustered_unequal_laws.informative_truth` of ``law``."""
    if law in ATE_LAWS:
        return {**size_law_parameters(law), "latent": laws.LATENT}
    return {**size_law_parameters(ARM_LAW_BASE[law]), "latent": ARM_LATENT[law]}


def law_truth(law: str) -> dict[str, float]:
    """The exact truth of every reported name on ``law``."""
    truth = laws.informative_truth(**law_parameters(law))
    return {name: float(truth[key]) for name, key in TRUTH_KEYS.items()}


def draw(law: str, clusters: int, seed: int) -> pd.DataFrame:
    """One draw of ``clusters`` clusters of ``law``, with the exact propensity columns."""
    fixed = np.full(clusters, 10) if law in EQUAL_LAWS else None
    frame = laws.draw_informative(
        clusters, np.random.default_rng(seed), sizes=fixed, **law_parameters(law)
    )
    p = laws.informative_propensity(frame["W1"].to_numpy(), frame["W2"].to_numpy())
    return frame.assign(g1=p, g0=1.0 - p)


def family_of(fit: str) -> str:
    """The floor family of a fit, or of a fit-and-estimand token of a cell name."""
    return FITS[fit.split("-", 1)[0]]


def ladder(fit: str) -> tuple[int, ...]:
    """The cluster counts a fit runs at."""
    return LADDERS[FITS[fit]]


def laws_of(fit: str) -> tuple[str, ...]:
    """The laws a fit runs on: fold-evaluated at equal sizes, d5 at unequal ones."""
    if FITS[fit] == "tmle_fold_evaluated":
        return EQUAL_LAWS
    if fit in source.D5_FITS:
        return UNEQUAL_LAWS
    return LAWS


def estimands_of(fit: str) -> tuple[str, ...]:
    """The estimands a fit reports: its two arm means, then its contrast."""
    return LTMLE_ESTIMANDS if FITS[fit] == LTMLE_FAMILY else POINT_ESTIMANDS


def arm_means_of(fit: str) -> tuple[str, ...]:
    """The two gated arm means of a fit."""
    return tuple(name for name in estimands_of(fit) if name not in CONTRASTS)


def mechanism(fit: str) -> str:
    """``"known"`` (a declared mechanism) or ``"estimated"``."""
    return "estimated" if fit.endswith("_estimated") else "known"


def transfer_check(fit: str, law: str, estimand: str) -> bool:
    """Whether the cell is a gated transfer check: an estimated configuration's contrast at
    ``"auto"`` on an ATE law (plan 1.4)."""
    return estimand in CONTRASTS and law in ATE_LAWS and mechanism(fit) == "estimated"


def gated(fit: str, law: str, estimand: str) -> bool:
    """Whether the cell's positive and control carry verdicts."""
    return estimand not in CONTRASTS or transfer_check(fit, law, estimand)


def arms(fit: str, law: str, clusters: int, estimand: str) -> tuple[tuple[str, str], ...]:
    """The ``few_cluster_reference`` arms of one estimand of a fit, with their roles: those of
    ``few-cluster-floor-tmle``, every arm reported where the estimand is not gated."""
    published = source.arms(fit, clusters)
    if gated(fit, law, estimand):
        return published
    return tuple((arm, floor.DIAGNOSTIC) for arm, _ in published)


def cell_name(fit: str, estimand: str, law: str, clusters: int, arm: str) -> str:
    """The published name of one property cell: the estimand's tag follows the fit."""
    return f"{fit}-{TAGS[estimand]}__{law}__j{clusters}__{arm}"


def parse(cell: str) -> tuple[str, str, str, int, str]:
    """``(fit, estimand, law, J, arm)`` of a published cell name."""
    token, law, clusters, arm = floor.parse(cell)
    fit, tag = token.split("-", 1)
    return fit, ESTIMAND_OF_TAG[tag], law, clusters, arm


def grid() -> tuple[tuple[str, str, int], ...]:
    """Every ``(fit, law, J)`` of the property grid."""
    return tuple(
        (fit, law, clusters) for fit in FITS for law in laws_of(fit) for clusters in ladder(fit)
    )


def requested(fit: str, budget: int | None = None) -> int:
    """A cell's requested replications, capped at ``budget``."""
    replicates = PROPERTY_REPLICATES[FITS[fit]]
    return replicates if budget is None else min(replicates, budget)


def bounds(fit: str) -> Any:
    """The ``g_bounds`` a point TMLE fit passes, read from :data:`MECHANISM_BOUNDS` at call time."""
    return MECHANISM_BOUNDS[mechanism(fit)]


PUBLISHED_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(fit, estimand, law, clusters, arm)
    for fit, law, clusters in grid()
    for estimand in estimands_of(fit)
    for arm, _ in arms(fit, law, clusters, estimand)
)

STUDY = StudyRecord(
    name="the clustered interval floor of the arm means",
    slug="few-cluster-arm-means",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_arm_means",
    document="docs/technical-reference/method-evidence/few-cluster-arm-means.md",
    anchor="the-clustered-interval-floor-of-the-arm-means",
    scenarios={SCENARIO: POINT_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    failure_bound={"cleverly": floor.PRIMARY_FAILURE_BOUND},
    modules=(
        "tests/studies/few_cluster_arm_means.py",
        "tests/studies/few_cluster_arm_means_properties.py",
        "tests/studies/few_cluster_floor_tmle.py",
        "tests/studies/few_cluster_floor.py",
        "tests/studies/few_cluster_floor_pilot.py",
        "tests/studies/clustered_unequal_laws.py",
        "tests/studies/clustered_few_cluster_properties.py",
        "tests/studies/clustered_few_cluster_tmle.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_drtmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.few_cluster_arm_means",
    properties_module="tests.studies.few_cluster_arm_means_properties",
    property_cells={FAMILY: PUBLISHED_CELLS},
    publication_policy="reporting",
)

CONFIGURATION = {
    "laws": {law: law_parameters(law) for law in LAWS},
    "ladders": {family: list(counts) for family, counts in LADDERS.items()},
    "fits": dict(FITS),
    "estimands": {"point": list(POINT_ESTIMANDS), "ltmle": list(LTMLE_ESTIMANDS)},
    "clause_one_laws": list(ARM_MEAN_LAWS),
    "mechanism_bounds": {key: str(value) for key, value in MECHANISM_BOUNDS.items()},
    "drtmle_g_bounds": list(source.DRTMLE_G_BOUNDS),
    "primary": f"{SCENARIO}: {PRIMARY_FIT}, {PRIMARY_ESTIMAND}, {PRIMARY_CLUSTERS} clusters",
}


# ------------------------------------------------------------------------------- draws, fits


def draw_seed(law: str, clusters: int, replicate: int, stream: str = "property") -> int:
    """The seed of one draw of a ``(law, J)`` cell on a named stream (``property`` or ``pilot``)."""
    return int(stream_seed(STUDY, stream, f"{law}/J{clusters}", replicate))


def frame_of(fit: str, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The frame a fit reads: the draw, with the noise columns for a d5 configuration."""
    return source.with_noise(frame, seed) if fit in source.D5_FITS else frame


def fit_estimates(fit: str, frame: pd.DataFrame, clusters: int) -> dict[str, Any]:
    """Every reported estimate of one fit of one draw, each ``reference_df`` checked."""
    if FITS[fit] == LTMLE_FAMILY:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = source.fit_ltmle(frame, **source.ltmle_settings())
    else:
        overrides: dict[str, Any] = {"estimands": POINT_ESTIMANDS}
        if FITS[fit] != "drtmle_known":
            overrides["g_bounds"] = bounds(fit)
        result = source.point_result(fit, frame, clusters, **overrides)
    return {
        name: source.checked_reference_df(fit, clusters, result[name]) for name in estimands_of(fit)
    }


def fit_rows(
    fit: str,
    law: str,
    clusters: int,
    replicate: int,
    frame: pd.DataFrame,
    *,
    budget: int | None = None,
    stream: str = "property",
) -> list[dict[str, Any]]:
    """Every row of one fit of one draw: each estimand's arms. Only the fit raising loses the key."""
    truth = law_truth(law)
    frame = frame_of(fit, frame, draw_seed(law, clusters, replicate, stream))
    estimates = fit_estimates(fit, frame, clusters)
    rows = []
    for name, estimate in estimates.items():
        published = arms(fit, law, clusters, name)
        readings = floor.reference_readings(
            estimate,
            clusters=clusters,
            covariates=len(source.covariates(fit)),
            arms=[arm for arm, _ in published],
        )
        rows += [
            floor.arm_row(
                readings[arm],
                cell=cell_name(fit, name, law, clusters, arm),
                role=role,
                replicate=replicate,
                n=10 * clusters,
                requested=requested(fit, budget),
                truth=truth[name],
                clusters=clusters,
            )
            for arm, role in published
        ]
    return rows


# ------------------------------------------------------------------------------------ primary


def primary_choice(coverage: Mapping[tuple[str, int], float]) -> tuple[str, int]:
    """The declared primary cell (plan 5.2) given the pilot's all-draw coverage of each candidate.

    The first of :data:`PRIMARY_CANDIDATES` whose plug-in G1 pass probability at
    :data:`PRIMARY_REPLICATES` reaches :data:`PRIMARY_PASS`, else :data:`PRIMARY_FALLBACK`.
    """
    for candidate in PRIMARY_CANDIDATES:
        if floor.g1_pass_probability(coverage[candidate], PRIMARY_REPLICATES) >= PRIMARY_PASS:
            return candidate
    return PRIMARY_FALLBACK


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw: ``n // 10`` clusters of the primary law."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    truth = law_truth(PRIMARY_LAW)
    return draw(PRIMARY_LAW, n // 10, seed), {name: truth[name] for name in POINT_ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: the in-sample known TMLE's three estimands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = source.point_result(
            PRIMARY_FIT,
            frame,
            PRIMARY_CLUSTERS,
            estimands=POINT_ESTIMANDS,
            g_bounds=bounds(PRIMARY_FIT),
        )
    return rows_from_result(STUDY, result, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )


__all__: Sequence[str] = (
    "FITS",
    "LAWS",
    "STUDY",
    "arms",
    "draw",
    "draw_and_fit",
    "fit_estimates",
    "fit_rows",
    "grid",
    "law_parameters",
    "primary_choice",
)

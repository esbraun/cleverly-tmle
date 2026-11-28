r"""Declared evidence for the clustered interval floor of stacked cross-fitted TMLE (X54 (j)).

``few-cluster-floor-tmle`` fitted every estimated mechanism with the exact-propensity bounds
``(1e-9, 1 - 1e-9)``. On a near-separated training fold the fitted propensity of a GLM reaches
that bound, the clever covariate reaches :math:`10^9`, and the cluster influence-curve variance
carries the tail that turned the five-covariate estimated IID control red at 10 clusters (X54
plan rev 2.4, 1.1). That pairing is a study defect, the one ``clustered-unequal-cvtmle`` records
in its own docstring. This study redeclares the stacked family with the bounds that ship.

**Laws.** ``equal10`` and ``unequal_informative`` (:mod:`tests.studies.clustered_unequal_laws`,
truths exact). The five-covariate configurations add three N(0, 1) columns
(:func:`tests.studies.few_cluster_floor_tmle.with_noise`) on ``unequal_informative`` only, from a
child stream of the draw's own seed.

**Configurations** (:data:`SOURCES`, one floor family ``tmle_stacked``): ``stacked_known`` and
``stacked_estimated`` on both laws, ``stacked_d5_known`` and ``stacked_d5_estimated`` on
``unequal_informative``. Each is the fit of ``few-cluster-floor-tmle`` it names, at
:math:`V = \min(5, J)` whole-cluster folds, with the bounds of :data:`MECHANISM_BOUNDS`: an
estimated mechanism (a binomial main-terms ``GLM``) takes ``g_bounds="auto"``, the shipped
default; a known mechanism keeps the exact bounds, which move no known value of these laws
(the propensity lies in [0.062, 0.951]). The d5 configurations are inside the floor rule.

**Fixed-bounds arm** (:data:`FIXED_BOUNDS_FITS`): each estimated configuration refitted on the
same draws at :data:`FIXED_G_BOUNDS`, ``(0.01, 0.99)``. Its ``t_reference`` row is gated on G1
and G2; clause 2 does not apply, so its other arms are reported.

**Ladder.** :data:`LADDER`, 4 to 10, 15, 20, 30, 39. 10 is the anchor.

**Arms.** ``t_reference`` (positive), ``iid_t_control`` (control), ``normal_reference`` and
``t_j_minus_1_reference`` (reported); on the default-bounds configurations at :math:`J \ge 10`,
where every fold holds 2 clusters, (C) ``cluster_jackknife`` on the first 2,000 draws in its own
key, paired with a copy of (A)'s row (``few_cluster_candidate``). No (B).

**Tail readings** (reported, :mod:`tests.studies.few_cluster_floor_stacked_properties`). Each
``t_reference`` row records ``bound_reached``: whether any cross-fitted propensity of the fit
reached a bound (``result.diagnostics.support().truncated``).

**Primary.** ``stacked_d5_estimated`` at 10 clusters on ``unequal_informative``, 1,000
replications, the ATE, no comparator: none of the pinned R packages cross-fits with cluster sums.
The policy is ``reporting``, declared before any run.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final
from unittest import mock

import pandas as pd

from cleverly.inference import cluster_jackknife
from tests.parallel import STUDY_JOBS
from tests.studies import few_cluster_floor as floor
from tests.studies import few_cluster_floor_tmle as source
from tests.studies.canonical_cvtmle import fitted_rows, rows_from_result
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate, stream_seed

PRIMARY_REPLICATES: Final[int] = 1_000
PRIMARY_CLUSTERS: Final[int] = 10
#: The nominal rows of the primary: 10 clusters of mean size 10.
PRIMARY_N: Final[int] = 10 * PRIMARY_CLUSTERS
SEED: Final[int] = 20265431
RESAMPLING_SEED: Final[int] = 20265432
PRIMARY_FIT: Final[str] = "stacked_d5_estimated"
PRIMARY_LAW: Final[str] = "unequal_informative"
SCENARIO: Final[str] = "floor_stacked_d5_estimated_unequal_informative_j10"
FAMILY: Final[str] = floor.FAMILY
#: The package's floor family this study measures.
FLOOR_FAMILY: Final[str] = "tmle_stacked"
SIZE_LAWS: Final[tuple[str, ...]] = ("equal10", "unequal_informative")
LADDER: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 39)
#: The point anchor: the default floor the family returns to when a count at or above it fails.
ANCHOR: Final[int] = 10
TARGET: Final[str] = source.TARGET
#: The fit of ``few-cluster-floor-tmle`` each configuration refits.
SOURCES: Final[dict[str, str]] = {
    "stacked_known": "tmle_stacked_known",
    "stacked_estimated": "tmle_stacked_estimated",
    "stacked_d5_known": "tmle_stacked_d5_known",
    "stacked_d5_estimated": "tmle_stacked_d5_estimated",
    "stacked_estimated_fixed_bounds": "tmle_stacked_estimated",
    "stacked_d5_estimated_fixed_bounds": "tmle_stacked_d5_estimated",
}
#: The fixed-bounds arm: each estimated configuration it refits on the same draws.
FIXED_BOUNDS_FITS: Final[dict[str, str]] = {
    "stacked_estimated_fixed_bounds": "stacked_estimated",
    "stacked_d5_estimated_fixed_bounds": "stacked_d5_estimated",
}
#: The default-bounds configurations, whose positive and control pairs set the floor.
CONFIGURATIONS: Final[tuple[str, ...]] = tuple(
    fit for fit in SOURCES if fit not in FIXED_BOUNDS_FITS
)
#: Each estimated configuration and the known one of the same covariates (the paired slack).
KNOWN_OF: Final[dict[str, str]] = {
    "stacked_estimated": "stacked_known",
    "stacked_d5_estimated": "stacked_d5_known",
    "stacked_estimated_fixed_bounds": "stacked_known",
    "stacked_d5_estimated_fixed_bounds": "stacked_d5_known",
}
#: The propensity bounds of each mechanism kind (plan 5.1): an estimated mechanism takes the
#: shipped default ``"auto"`` (:func:`cleverly.utils.bounds.resolve_g_bounds`,
#: :math:`5/(\sqrt n \log n)` at the fit's rows); a known one keeps the exact bounds, which move
#: no known value of these laws, as the package requires of a declared mechanism.
MECHANISM_BOUNDS: Final[dict[str, Any]] = {"known": source.G_BOUNDS, "estimated": "auto"}
#: The fixed-bounds arm's bounds, the convention of ``clustered-unequal-cvtmle``.
FIXED_G_BOUNDS: Final[tuple[float, float]] = (0.01, 0.99)
#: The per-draw tail column a ``t_reference`` row carries.
BOUND_COLUMN: Final[str] = "bound_reached"
#: Replications per cell before the pilot (plan 5): 4,000, raised by the family-wise rule (M5)
#: off the committed pilot; ``tests/unit/test_few_cluster_floor_stacked_design.py`` recomputes it.
PROPERTY_REPLICATES: Final[dict[str, int]] = {FLOOR_FAMILY: 4_000}
#: The pilot's predicted reds and their readings (``pilot-sizing.csv``); empty until the pilot.
PILOT_PREDICTED_REDS: Final[dict[str, str]] = {}
#: Whether the family's plug-in pass probability stays under the target at the cap (M5).
PILOT_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: Whether the pilot measures (C) short of admissibility (``pilot-candidates.csv``).
JACKKNIFE_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: The declared budget of the property phase (plan 5.1), in CPU hours.
DECLARED_CPU_HOURS: Final[float] = 25.0


def mechanism(fit: str) -> str:
    """``"known"``, ``"estimated"`` or ``"fixed_bounds"``."""
    if fit in FIXED_BOUNDS_FITS:
        return "fixed_bounds"
    return "known" if fit.endswith("_known") else "estimated"


def bounds(fit: str) -> Any:
    """The ``g_bounds`` a fit passes, read from :data:`MECHANISM_BOUNDS` at call time."""
    kind = mechanism(fit)
    return FIXED_G_BOUNDS if kind == "fixed_bounds" else MECHANISM_BOUNDS[kind]


def laws_of(fit: str) -> tuple[str, ...]:
    """The size laws a fit runs on: the d5 configurations on ``unequal_informative`` only."""
    return ("unequal_informative",) if SOURCES[fit] in source.D5_FITS else SIZE_LAWS


def folds(clusters: int) -> int:
    """The outer folds at ``clusters``: five, capped at :math:`J`."""
    return min(source.MAX_FOLDS, clusters)


def covariates(fit: str) -> tuple[str, ...]:
    """The fit's baseline covariates."""
    return source.covariates(SOURCES[fit])


def jackknife_counts(fit: str) -> tuple[int, ...]:
    """The counts where (C) runs: a default-bounds configuration, every fold holding 2 clusters."""
    if fit in FIXED_BOUNDS_FITS:
        return ()
    return tuple(c for c in LADDER if c >= 2 * folds(c))


def arms(fit: str, clusters: int) -> tuple[tuple[str, str], ...]:
    """The ``few_cluster_reference`` arms of one fit at ``clusters``, with their roles.

    A fixed-bounds fit gates its interval only (G1 and G2): its control is reported.
    """
    del clusters
    control = floor.DIAGNOSTIC if fit in FIXED_BOUNDS_FITS else floor.CONTROL
    return (
        (floor.REFERENCE_ARM, floor.POSITIVE),
        ("iid_t_control", control),
        ("normal_reference", floor.DIAGNOSTIC),
        ("t_j_minus_1_reference", floor.DIAGNOSTIC),
    )


def candidates(fit: str, clusters: int) -> tuple[str, ...]:
    """The candidates with a ``few_cluster_candidate`` pair at ``clusters``: (C) or none."""
    return (floor.JACKKNIFE_ARM,) if clusters in jackknife_counts(fit) else ()


def candidate_arms(fit: str, clusters: int) -> tuple[tuple[str, str], ...]:
    """The ``few_cluster_candidate`` arms at ``clusters``: (C), then (A)'s copy."""
    return tuple(
        arm
        for candidate in candidates(fit, clusters)
        for arm in ((candidate, floor.POSITIVE), (floor.reference_for(candidate), floor.CONTROL))
    )


def cell_name(fit: str, law: str, clusters: int, arm: str) -> str:
    """The published name of one property cell."""
    return f"{fit}__{law}__j{clusters}__{arm}"


def grid() -> tuple[tuple[str, str, int], ...]:
    """Every ``(fit, law, J)`` of the property grid."""
    return tuple(
        (fit, law, clusters) for fit in SOURCES for law in laws_of(fit) for clusters in LADDER
    )


def requested(budget: int | None = None, *, candidate: bool = False) -> int:
    """A cell's requested replications, capped at ``budget``; a (C) cell reads the first
    :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS`."""
    replicates = PROPERTY_REPLICATES[FLOOR_FAMILY]
    if budget is not None:
        replicates = min(replicates, budget)
    return min(replicates, floor.CANDIDATE_DRAWS) if candidate else replicates


PUBLISHED_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(fit, law, clusters, arm)
    for fit, law, clusters in grid()
    for arm, _ in arms(fit, clusters)
)
CANDIDATE_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(fit, law, clusters, arm)
    for fit, law, clusters in grid()
    for arm, _ in candidate_arms(fit, clusters)
)
POOLED_CELLS: Final[tuple[str, ...]] = tuple(
    floor.pooled_name(fit, floor.JACKKNIFE_ARM) for fit in CONFIGURATIONS
)

STUDY = StudyRecord(
    name="the clustered interval floor of stacked cross-fitted TMLE",
    slug="few-cluster-floor-stacked",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_floor_stacked",
    document="docs/technical-reference/method-evidence/few-cluster-floor-stacked.md",
    anchor="the-clustered-interval-floor-of-stacked-cross-fitted-tmle",
    scenarios={SCENARIO: (TARGET,)},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    failure_bound={"cleverly": floor.PRIMARY_FAILURE_BOUND},
    modules=(
        "tests/studies/few_cluster_floor_stacked.py",
        "tests/studies/few_cluster_floor_stacked_properties.py",
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
    runner_module="tests.studies.few_cluster_floor_stacked",
    properties_module="tests.studies.few_cluster_floor_stacked_properties",
    property_cells={
        FAMILY: PUBLISHED_CELLS,
        floor.CANDIDATE_FAMILY: CANDIDATE_CELLS,
        floor.POOLED_FAMILY: POOLED_CELLS,
    },
    publication_policy="reporting",
)

CONFIGURATION = {
    "laws": {
        "equal10": "clustered_dgp(10, 'binomial'): 10 rows per cluster, arm-by-latent 8",
        "unequal_informative": "sizes uniform on 2 to 18, informative (clustered_unequal_laws)",
    },
    "ladder": list(LADDER),
    "fits": dict(SOURCES),
    "outcome_learner": "unpenalized logistic regression (C=1e6) in the fit's covariates",
    "estimated_treatment": "binomial main-terms GLM, g_bounds='auto'",
    "known_treatment": "treatment_probabilities= the law's exact propensity",
    "mechanism_bounds": {key: str(value) for key, value in MECHANISM_BOUNDS.items()},
    "fixed_bounds": list(FIXED_G_BOUNDS),
    "noise_columns": list(source.NOISE_COLUMNS),
    "candidate_draws": floor.CANDIDATE_DRAWS,
    "references": {
        "A": "cluster IC variance on t(J-2)",
        "C": "cluster_variance='jackknife' on t(J-2), J >= 10",
    },
    "primary": f"{SCENARIO}: {PRIMARY_FIT}, 10 clusters, {PRIMARY_LAW}",
}


# ------------------------------------------------------------------------------- draws, fits


def draw_seed(law: str, clusters: int, replicate: int, stream: str = "property") -> int:
    """The seed of one draw of a ``(law, J)`` cell on a named stream (``property`` or ``pilot``)."""
    return int(stream_seed(STUDY, stream, f"{law}/J{clusters}", replicate))


def draw(law: str, clusters: int, seed: int) -> pd.DataFrame:
    """One draw of ``clusters`` clusters of ``law``, with the exact propensity columns."""
    return source.draw(law, clusters, seed)


def frame_of(fit: str, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The frame a fit reads: the draw, with the noise columns for a d5 configuration."""
    return source.with_noise(frame, seed) if SOURCES[fit] in source.D5_FITS else frame


def fit_result(fit: str, frame: pd.DataFrame, clusters: int, **overrides: Any) -> Any:
    """The fitted result of one configuration at its declared bounds; ``overrides`` win."""
    return source.point_result(
        SOURCES[fit], frame, clusters, **{"g_bounds": bounds(fit), **overrides}
    )


def bound_reached(result: Any) -> float:
    """1 when any cross-fitted propensity of the fit reached a bound, else 0; ``nan`` when the
    fit publishes no support report (a reported column, never raised)."""
    try:
        truncated = result.diagnostics.support().truncated
    except Exception:  # a reported column: its failure never costs the row
        return math.nan
    return float(float(truncated.get("count", 0.0)) > 0.0)


def fit_estimate(fit: str, frame: pd.DataFrame, clusters: int, **overrides: Any) -> Any:
    """The ATE of one configuration, its ``reference_df`` checked where it is shown."""
    estimate = fit_result(fit, frame, clusters, **overrides)[TARGET]
    return source.checked_reference_df(SOURCES[fit], clusters, estimate)


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
    """Every row of one fit of one draw but (C)'s; ``budget`` caps the requested replications.

    The ``few_cluster_reference`` arms, the positive row carrying :data:`BOUND_COLUMN`, then on
    the first :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS` draws (A)'s copy for each
    (C) pair. Only the main fit raising loses the key.
    """
    truth = source.law_truth(law)
    frame = frame_of(fit, frame, draw_seed(law, clusters, replicate, stream))
    result = fit_result(fit, frame, clusters)
    estimate = source.checked_reference_df(SOURCES[fit], clusters, result[TARGET])
    published = arms(fit, clusters)
    readings = floor.reference_readings(
        estimate, clusters=clusters, covariates=len(covariates(fit)), arms=[a for a, _ in published]
    )
    rows = [
        floor.arm_row(
            readings[arm],
            cell=cell_name(fit, law, clusters, arm),
            role=role,
            replicate=replicate,
            n=10 * clusters,
            requested=requested(budget),
            truth=truth,
            clusters=clusters,
        )
        for arm, role in published
    ]
    rows[0][BOUND_COLUMN] = bound_reached(result)
    if replicate < floor.CANDIDATE_DRAWS:
        wanted = requested(budget, candidate=True)
        for candidate in candidates(fit, clusters):
            rows.append(
                floor.candidate_copy(
                    rows[0],
                    arm=floor.reference_for(candidate),
                    role=floor.CONTROL,
                    requested=wanted,
                )
            )
    return rows


def jackknife_rows(
    fit: str,
    law: str,
    clusters: int,
    replicate: int,
    frame: pd.DataFrame,
    *,
    budget: int | None = None,
    stream: str = "property",
) -> list[dict[str, Any]]:
    """(C)'s row of one draw, the key ``<fit>__jackknife``: its failures are its own misses.

    The package refuses the stacked jackknife until this study decides it
    (:data:`~cleverly.inference.cluster_jackknife.JACKKNIFE_FAMILY_REFUSALS`), so the study lifts
    that refusal to measure the construction. Each refit re-resolves ``"auto"`` from its own rows,
    as the shipped option would (plan 3.1).
    """
    frame = frame_of(fit, frame, draw_seed(law, clusters, replicate, stream))
    overrides = {"g_bounds": bounds(fit)}
    try:
        estimate = source.fit_estimate(SOURCES[fit], frame, clusters, **overrides)
    except Exception:  # (A)'s loss, recorded by (A)'s key, never (C)'s
        estimate = None
    with mock.patch.object(cluster_jackknife, "JACKKNIFE_FAMILY_REFUSALS", {}):
        reading = source.candidate_c(SOURCES[fit], frame, clusters, estimate, **overrides)
    return [
        floor.arm_row(
            reading,
            cell=cell_name(fit, law, clusters, floor.JACKKNIFE_ARM),
            role=floor.POSITIVE,
            replicate=replicate,
            n=10 * clusters,
            requested=requested(budget, candidate=True),
            truth=source.law_truth(law),
            clusters=clusters,
        )
        | {"property": floor.CANDIDATE_FAMILY}
    ]


# ------------------------------------------------------------------------------------ primary


def primary_truth() -> dict[str, float]:
    """The primary law's exact ATE."""
    return {TARGET: source.law_truth(PRIMARY_LAW)}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw: ``n // 10`` clusters of ``unequal_informative``, with the noise columns."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame = draw(PRIMARY_LAW, n // 10, seed)
    return frame_of(PRIMARY_FIT, frame, seed), primary_truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: the shipped stacked TMLE at ``"auto"``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_result(PRIMARY_FIT, frame, PRIMARY_CLUSTERS)
    return rows_from_result(STUDY, result, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )


__all__: Sequence[str] = (
    "MECHANISM_BOUNDS",
    "SOURCES",
    "STUDY",
    "arms",
    "draw",
    "draw_and_fit",
    "fit_estimate",
    "fit_rows",
    "grid",
    "jackknife_rows",
)

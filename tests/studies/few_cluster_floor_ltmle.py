r"""Registered evidence for the clustered interval floor of cross-fitted LTMLE (X54 (b)).

A cross-fitted clustered ``LTMLE`` reports its interval on :math:`t_{J-2}` and, below 20
clusters, takes ``"few_cluster_plugin"``. This study measures every count of
:data:`LADDER` (4 to 20, 30 and 39; 20 is the anchor of ``few-cluster-cross-fitted-ltmle``) and
sets the family's floor by the rule of plan 5.8
(:func:`tests.studies.few_cluster_floor.family_floor`). On the same draws it compares the
shipped reference (A) with Wang et al.'s inflated variance (B) and the delete-one-cluster
jackknife (C) (part (c), :mod:`tests.studies.few_cluster_floor`).

**Law.** The end-of-study law of :mod:`tests.studies.clustered_longitudinal_laws` with the
scaled latent, at ``equal40`` (40 rows per cluster) and ``unequal40`` (sizes uniform on 10 to
70), as ``few-cluster-cross-fitted-ltmle``.

**Fit.** The ``canonical-ltmle-crossfit`` subject (``clustered_crossfit_ltmle.subject``):
:math:`V = \min(5, J)` whole-cluster folds, the law's own treatment and censoring mechanisms,
the study quasibinomial GLM for every node regression. The learners are the subject's,
declared parametric (:class:`~tests.studies.few_cluster_floor.ParametricQuasiBinomialGLM`,
:class:`~tests.studies.few_cluster_floor.ParametricKnownMechanism`) so the jackknife admits
them; the declaration changes no fit. The GLM runs with ``numerical_separation=True``, the
``boundary-node-ltmle`` setting: on a target separated once rounded it keeps its last iterate
or continues IRLS, and a fit that still fails raises the recordable
``QuasiBinomialConvergenceError``, a lost key. Without it the first pilot stopped at
``equal40``, 4 clusters, pilot replicate 19. Each positive row also carries the same draw refitted
with :class:`cleverly.learners.GLM` node regressions (``package_glm_*`` columns, reported, M2).

**Arms** (``few_cluster_reference``). ``t_reference`` (positive, (A)), ``iid_t_control``
(control), ``normal_reference``, ``t_j_minus_1_reference`` and ``wang_inflated`` (B, every draw,
reported).

**Part (c) verdicts** (``few_cluster_candidate``, review M1). On the first 2,000 draws, (B) and,
at :math:`J \ge 10` where every fold holds 2 clusters, ``cluster_jackknife`` (C), each paired
with a copy of (A)'s row on the same draw. Below 10 R5 refuses (C) before any refit (M4). (C)
is fitted in its own key (``ltmle_cross_fitted__jackknife``), so its failure never costs (A)'s
row (review B1). The package-GLM refit never raises out of the key either: any exception is
recorded with its class and message.

**Primary.** 10 clusters of ``equal40``, 1,000 replications, no comparator. The policy is
``reporting``, declared before any run.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import pandas as pd

from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_longitudinal_laws as law
from tests.studies import few_cluster_floor as floor
from tests.studies.canonical_cvtmle import fitted_rows
from tests.studies.canonical_ltmle import regimen_initials, regimen_rows
from tests.studies.canonical_ltmle_crossfit import (
    CONTRAST_NAMES,
    ESTIMANDS,
    G_BOUNDS,
    REFERENCE,
    REGIMENS,
)
from tests.studies.clustered_crossfit_ltmle import (
    CLUSTER_SIZE,
    ID,
    LEARNER_FOLDS,
    NODES,
    RANDOM_STATE,
)
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate, stream_seed

LATENT: Final[str] = "scaled"
PRIMARY_REPLICATES: Final[int] = 1_000
PRIMARY_CLUSTERS: Final[int] = 10
PRIMARY_N: Final[int] = PRIMARY_CLUSTERS * CLUSTER_SIZE
SEED: Final[int] = 20265421
RESAMPLING_SEED: Final[int] = 20265422
SCENARIO: Final[str] = "floor_equal40_j10"
FAMILY: Final[str] = floor.FAMILY
TARGET: Final[str] = CONTRAST_NAMES[0]
SIZE_LAWS: Final[tuple[str, ...]] = ("equal40", "unequal40")
LADDER: Final[tuple[int, ...]] = (*range(4, 21), 30, 39)
ANCHOR: Final[int] = 20
FIT: Final[str] = "ltmle_cross_fitted"
#: The fit's baseline covariates, for (B)'s :math:`p = \max(2, d)`.
BASELINE_COVARIATES: Final[int] = len(NODES["baseline"])
MAX_FOLDS: Final[int] = 5
#: Replications per cell (plan 5.2): 4,000, raised by the family-wise rule (M5) off the
#: committed pilot; the pilot meets the 0.95 target at 4,000 (plug-in 1.000, predictive 0.814,
#: expected floor 19), so R stays 4,000.
PROPERTY_REPLICATES: Final[int] = 4_000
#: The pilot's predicted reds, grouped by the reading the declared route gives each
#: (``pilot-sizing.csv``, review M4): a positive whose plug-in G1 and G2 figure at the declared
#: R is below 0.90, and a control whose power under both clauses is below 0.90, which includes
#: a control that turns red jointly with its positive (clause 2). Every one is held by
#: :data:`~tests.studies.few_cluster_floor.RED_OWNER` as the property row
#: ``RedKey(slug, "property", "few_cluster_reference/<cell>")``; a
#: :data:`~tests.studies.few_cluster_floor.CHANCE_READING` red sits at the anchor and is held
#: only once a no-defect diagnosis is recorded. The design test checks both directions.
_PILOT_REDS: Final[dict[str, tuple[str, ...]]] = {
    floor.FINITE_SAMPLE_READING: (
        "ltmle_cross_fitted__unequal40__j7__t_reference",
        "ltmle_cross_fitted__unequal40__j8__t_reference",
    ),
}
PILOT_PREDICTED_REDS: Final[dict[str, str]] = {
    f"{FAMILY}/{cell}": reading for reading, cells in _PILOT_REDS.items() for cell in cells
}
#: Families whose plug-in pass probability stays under the target at ``R_CAP`` (review M4): none;
#: the pilot reads 1.000 at R = 4,000 from the expected floor 19.
PILOT_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: Families whose (C) arm the pilot measures as short of admissibility (review M4): none
#: (discordance 1.5% over 26 cells, clause (i) probability 1.000).
JACKKNIFE_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: The fit name of a (C) key (review B1).
JACKKNIFE_FIT: Final[str] = f"{FIT}__jackknife"


def folds(clusters: int) -> int:
    """The outer folds at ``clusters``: five, capped at :math:`J`."""
    return min(MAX_FOLDS, clusters)


def jackknife_counts() -> tuple[int, ...]:
    """The counts where (C) runs: every fold holds at least 2 clusters (M4)."""
    return tuple(c for c in LADDER if c >= 2 * folds(c))


def arms(clusters: int) -> tuple[tuple[str, str], ...]:
    """The ``few_cluster_reference`` arms of the fit at ``clusters``, each with its role."""
    del clusters
    return (
        (floor.REFERENCE_ARM, floor.POSITIVE),
        ("iid_t_control", floor.CONTROL),
        ("normal_reference", floor.DIAGNOSTIC),
        ("t_j_minus_1_reference", floor.DIAGNOSTIC),
        (floor.WANG_ARM, floor.DIAGNOSTIC),
    )


def candidates(clusters: int) -> tuple[str, ...]:
    """The candidates with a ``few_cluster_candidate`` pair at ``clusters``: (B), then (C)."""
    return floor.VERDICT_CANDIDATES if clusters in jackknife_counts() else (floor.WANG_ARM,)


def candidate_arms(clusters: int) -> tuple[tuple[str, str], ...]:
    """The ``few_cluster_candidate`` arms at ``clusters``: each candidate, then (A)'s copy."""
    return tuple(
        arm
        for candidate in candidates(clusters)
        for arm in ((candidate, floor.POSITIVE), (floor.reference_for(candidate), floor.CONTROL))
    )


def cell_name(sizes: str, clusters: int, arm: str) -> str:
    """The published name of one property cell."""
    return f"{FIT}__{sizes}__j{clusters}__{arm}"


def grid() -> tuple[tuple[str, int], ...]:
    """Every ``(law, J)`` of the property grid."""
    return tuple((sizes, clusters) for sizes in SIZE_LAWS for clusters in LADDER)


def requested(budget: int | None = None, *, candidate: bool = False) -> int:
    """A cell's requested replications, capped at ``budget``; a part (c) cell reads only the
    first :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS`."""
    replicates = PROPERTY_REPLICATES if budget is None else min(PROPERTY_REPLICATES, budget)
    return min(replicates, floor.CANDIDATE_DRAWS) if candidate else replicates


PUBLISHED_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(sizes, clusters, arm) for sizes, clusters in grid() for arm, _ in arms(clusters)
)
CANDIDATE_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(sizes, clusters, arm)
    for sizes, clusters in grid()
    for arm, _ in candidate_arms(clusters)
)
POOLED_CELLS: Final[tuple[str, ...]] = tuple(
    floor.pooled_name(FIT, candidate) for candidate in floor.VERDICT_CANDIDATES
)

STUDY = StudyRecord(
    name="the clustered interval floor of cross-fitted LTMLE",
    slug="few-cluster-floor-cross-fitted-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_floor_ltmle",
    document="docs/technical-reference/method-evidence/few-cluster-floor-cross-fitted-ltmle.md",
    anchor="the-clustered-interval-floor-of-cross-fitted-ltmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    failure_bound={"cleverly": floor.PRIMARY_FAILURE_BOUND},
    modules=(
        "tests/studies/few_cluster_floor_ltmle.py",
        "tests/studies/few_cluster_floor_ltmle_properties.py",
        "tests/studies/few_cluster_floor.py",
        "tests/studies/few_cluster_floor_pilot.py",
        "tests/studies/clustered_longitudinal_laws.py",
        "tests/studies/clustered_crossfit_ltmle.py",
        "tests/studies/clustered_few_cluster_tmle.py",
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
    runner_module="tests.studies.few_cluster_floor_ltmle",
    properties_module="tests.studies.few_cluster_floor_ltmle_properties",
    property_cells={
        FAMILY: PUBLISHED_CELLS,
        floor.CANDIDATE_FAMILY: CANDIDATE_CELLS,
        floor.POOLED_FAMILY: POOLED_CELLS,
    },
    publication_policy="reporting",
)

CONFIGURATION = {
    "law": "make_longitudinal end-of-study law, cluster component on the final outcome, "
    f"delta = {law.DELTA}, scaled latent",
    "size_laws": list(SIZE_LAWS),
    "ladder": list(LADDER),
    "cross_fit": True,
    "outer_folds": "min(5, J) whole-cluster folds",
    "learners": (
        "study quasibinomial GLM node regressions (numerical_separation=True); "
        "the law's own mechanisms"
    ),
    "package_glm": "the same draw with cleverly.learners.GLM node regressions (reported)",
    "regimens": list(REGIMENS),
    "target": TARGET,
    "candidate_draws": floor.CANDIDATE_DRAWS,
    "references": {
        "A": "cluster IC variance on t(J-2)",
        "B": "J/(J-2) times the IC variance on t(J-2), p = 2",
        "C": "cluster_variance='jackknife' on t(J-2), J >= 10",
    },
    "primary": f"{SCENARIO}: 10 clusters of equal40",
}


def draw(sizes: str, clusters: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of ``clusters`` clusters of ``sizes`` with the scaled latent."""
    return law.draw_end_of_study(clusters, sizes, seed, latent=LATENT)


def subject(clusters: int, *, package_glm: bool = False, **overrides: Any) -> LTMLE:
    """The ``canonical-ltmle-crossfit`` subject at ``clusters``, its learners declared parametric."""
    outcome: Any = (
        GLM() if package_glm else floor.ParametricQuasiBinomialGLM(numerical_separation=True)
    )
    pseudo: Any = (
        GLM() if package_glm else floor.ParametricQuasiBinomialGLM(numerical_separation=True)
    )
    settings: dict[str, Any] = {
        "reference": REFERENCE,
        "outcome_learner": outcome,
        "pseudo_learner": pseudo,
        "treatment_learner": floor.ParametricKnownMechanism("treatment"),
        "censoring_learner": floor.ParametricKnownMechanism("censoring"),
        "n_folds": folds(clusters),
        "learner_folds": LEARNER_FOLDS,
        "g_bounds": G_BOUNDS,
        "simultaneous": False,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": RANDOM_STATE,
        **overrides,
    }
    return LTMLE(dict(REGIMENS), **settings)


def fit_estimate(frame: pd.DataFrame, clusters: int, **overrides: Any) -> Any:
    """The target estimate of one draw, its ``reference_df`` checked when it supplies an
    interval (a withheld estimate reports none, review B1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = subject(clusters, **overrides).fit(frame, outcome="Y", id=ID, **NODES)
    estimate = result[TARGET]
    if estimate.supplies_inference and estimate.reference_df != clusters - 2:
        raise AssertionError(
            f"J = {clusters}: the fit reports reference_df {estimate.reference_df}, not J - 2"
        )
    return estimate


def _package_glm_reading(frame: pd.DataFrame, truth: float, clusters: int) -> dict[str, Any]:
    """The reported package-GLM refit of a draw (M2): raised, shown and covered.

    Any exception is caught and recorded with its class and message, so the refit never
    costs the main fit's row (review B1).
    """
    try:
        estimate = fit_estimate(frame, clusters, package_glm=True)
    except Exception as error:  # recorded, never raised: a reported column
        return {
            "package_glm_raised": 1,
            "package_glm_shown": 0,
            "package_glm_covered": 0,
            "package_glm_error": floor.error_text(error),
        }
    reading = floor.reference_readings(
        estimate, clusters=clusters, covariates=BASELINE_COVARIATES, arms=(floor.REFERENCE_ARM,)
    )[floor.REFERENCE_ARM]
    half = reading.critical * reading.std_error
    covered = reading.shown and reading.psi - half <= truth <= reading.psi + half
    return {
        "package_glm_raised": 0,
        "package_glm_shown": int(reading.shown),
        "package_glm_covered": int(covered),
        "package_glm_error": "",
    }


def package_glm_columns(frame: pd.DataFrame, truth: float, clusters: int) -> dict[str, Any]:
    """:func:`_package_glm_reading`, with its CPU seconds (read by the pilot only)."""
    started = time.process_time()
    out = _package_glm_reading(frame, truth, clusters)
    return {**out, "package_glm_cpu_seconds": time.process_time() - started}


def draw_seed(sizes: str, clusters: int, replicate: int, stream: str = "property") -> int:
    """The seed of one draw of a ``(law, J)`` cell on a named stream (``property`` or ``pilot``)."""
    return int(stream_seed(STUDY, stream, f"{sizes}/J{clusters}", replicate))


def candidate_c(frame: pd.DataFrame, clusters: int, estimate: Any | None) -> floor.Reading:
    """(C) on one draw, never raising: a refusal, a failed refit or an error is a withheld miss.

    A pre-fit refusal reads ``jackknife_refused``, any other exception ``jackknife_error`` with
    its class and message, and an estimate that withholds before any refit the main fit's
    trigger. The jackknife fit's point must equal (A)'s when the main fit ran.
    """
    from cleverly.exceptions import CapabilityError, DataError

    psi = float("nan") if estimate is None else float(estimate.psi)
    try:
        refit = fit_estimate(frame, clusters, cluster_variance="jackknife")
    except (CapabilityError, DataError) as error:
        if estimate is None:  # the main fit's own failure, not an R5 refusal (review m-c1)
            return floor.failed_jackknife(psi, clusters, error)
        return floor.refused_jackknife(psi, clusters)
    except Exception as error:  # (C)'s failure is (C)'s miss, never (A)'s
        return floor.failed_jackknife(psi, clusters, error)
    if estimate is not None and not math.isclose(
        float(refit.psi), float(estimate.psi), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise AssertionError(
            f"J = {clusters}: the jackknife fit's point {refit.psi} is not (A)'s {estimate.psi}"
        )
    return floor.jackknife_reading(refit, clusters=clusters)


def fit_rows(
    sizes: str,
    clusters: int,
    replicate: int,
    drawn: tuple[pd.DataFrame, Mapping[str, float]],
    *,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """Every row of one draw but (C)'s; ``budget`` caps the requested replications.

    The ``few_cluster_reference`` arms (the positive row with the package-GLM columns), then
    on the first :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS` draws (B) and each
    pair's copy of (A). Only the main fit raising loses the key.
    """
    frame, truths = drawn
    truth = float(truths[TARGET])
    estimate = fit_estimate(frame, clusters)
    published = arms(clusters)
    readings = floor.reference_readings(
        estimate,
        clusters=clusters,
        covariates=BASELINE_COVARIATES,
        arms=[arm for arm, _ in published],
    )
    rows = [
        floor.arm_row(
            readings[arm],
            cell=cell_name(sizes, clusters, arm),
            role=role,
            replicate=replicate,
            n=CLUSTER_SIZE * clusters,
            requested=requested(budget),
            truth=truth,
            clusters=clusters,
        )
        for arm, role in published
    ]
    rows[0].update(package_glm_columns(frame, truth, clusters))
    if replicate < floor.CANDIDATE_DRAWS:
        wanted = requested(budget, candidate=True)
        wang = rows[-1]
        for candidate in candidates(clusters):
            if candidate == floor.WANG_ARM:
                rows.append(
                    floor.candidate_copy(wang, arm=candidate, role=floor.POSITIVE, requested=wanted)
                )
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
    sizes: str,
    clusters: int,
    replicate: int,
    drawn: tuple[pd.DataFrame, Mapping[str, float]],
    *,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """(C)'s row of one draw, the key ``ltmle_cross_fitted__jackknife`` (review B1).

    The main fit runs again only to check that (C)'s point is (A)'s. Its raising never loses
    this key: (A)'s own key records that loss, and (C) then reads its own fit
    (:func:`candidate_c`), whose every failure is a withheld row.
    """
    frame, truths = drawn
    try:
        estimate = fit_estimate(frame, clusters)
    except Exception:  # (A)'s loss, recorded by (A)'s key, never (C)'s
        estimate = None
    reading = candidate_c(frame, clusters, estimate)
    return [
        floor.arm_row(
            reading,
            cell=cell_name(sizes, clusters, floor.JACKKNIFE_ARM),
            role=floor.POSITIVE,
            replicate=replicate,
            n=CLUSTER_SIZE * clusters,
            requested=requested(budget, candidate=True),
            truth=float(truths[TARGET]),
            clusters=clusters,
        )
        | {"property": floor.CANDIDATE_FAMILY}
    ]


# ------------------------------------------------------------------------------------ primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw of ``n // 40`` clusters of 40 rows."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, truth = draw(f"equal{CLUSTER_SIZE}", n // CLUSTER_SIZE, seed)
    return frame, {name: float(truth[name]) for name in ESTIMANDS}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = subject(PRIMARY_CLUSTERS).fit(frame, outcome="Y", id=ID, **NODES)
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


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )


__all__: Sequence[str] = (
    "STUDY",
    "arms",
    "draw",
    "draw_and_fit",
    "fit_estimate",
    "fit_rows",
    "grid",
    "jackknife_counts",
    "jackknife_rows",
)

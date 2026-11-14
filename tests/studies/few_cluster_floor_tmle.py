r"""Registered evidence for the clustered interval floor of point TMLE and in-sample LTMLE (X54 (a)).

Below 40 clusters a clustered estimate reports its interval on :math:`t_{J-2}`, and below a
family floor (10 clusters for point fits, 20 for ``LTMLE``) it takes ``"few_cluster_plugin"``:
the interval is shown with the experimental label. This study measures every count of a ladder
below those floors, per fit family, and sets each family's floor by the rule of plan 5.8
(:func:`tests.studies.few_cluster_floor.family_floor`). On the same draws it compares the
shipped reference (A) with Wang et al.'s inflated variance (B) and the delete-one-cluster
jackknife (C) (part (c), :mod:`tests.studies.few_cluster_floor`).

**Laws.** The two size laws of ``clustered-few-cluster-tmle``: ``equal10`` (clusters of 10 rows,
the arm-by-latent coefficient 8) and ``unequal_informative`` (sizes uniform on 2 to 18,
informative). Truths exact (``informative_truth``). The ``tmle_stacked_d5_*`` configurations add
three independent N(0, 1) baseline columns (:func:`with_noise`), drawn from a child stream after
the base draw, so their base columns equal those of the other fits.

**Ladder.** Point fits at :data:`POINT_LADDER` (4 to 10, plus 15, 20, 30 and 39, for the part
(c) decision below 40); in-sample LTMLE at :data:`LTMLE_LADDER` (4 to 20, 30, 39). 10 and 20 are
the anchors.

**Fits per draw** (:data:`FITS`; every fit of a ``(law, J)`` reads one draw):

==============================  ==========================================================
fit                             construction
==============================  ==========================================================
``tmle_in_sample_known``        ``TMLE(cross_fit=False)``, the exact propensity declared
                                known (``treatment_probabilities=``)
``tmle_in_sample_estimated``    the same, the propensity fitted by a binomial main-terms
                                ``GLM``
``tmle_stacked_*``              ``cross_fit=True``, :math:`V = \min(5, J)` whole-cluster folds
``tmle_fold_evaluated_*``       ``cv_evaluation=True``, :math:`V = \min(5, \lfloor J/2\rfloor)`,
                                equal sizes only (Fix-4)
``drtmle_known``                cross-fitted ``DRTMLE``, foldwise reduced regressions,
                                :math:`V = \min(5, J)`, the exact propensity declared known
``tmle_stacked_d5_*``           ``tmle_stacked`` with three noise columns, ``unequal_informative``;
                                configurations of ``tmle_stacked`` in part (c) only (M2)
``ltmle_in_sample``             in-sample ``LTMLE`` on one node, the study quasibinomial GLM and
                                the exact propensity (no projection slack)
==============================  ==========================================================

Every TMLE outcome regression is the unpenalized logistic regression in the fit's covariates.
The ``ltmle_in_sample`` rows also carry the same draw refitted with :class:`cleverly.learners.GLM`
for every node regression (``package_glm_*`` columns, reported, M2): a raise there (at
separation, ``DataError``) is counted with its class and message and never costs the row, and
the study learner's shown-only coverage is published split by it.

**Floor families** (:data:`ANCHORS`): every family of :data:`FITS` but the two
``tmle_stacked_d5_*`` configurations, whose verdicts and floor reading are reported evidence
(:data:`D5_FITS`, review M2): the package has no five-covariate family, and a stacked fit with
five covariates reads ``tmle_stacked``.

**Arms** (:func:`arms`, family ``few_cluster_reference``): ``t_reference`` (positive, (A)),
``iid_t_control`` (control), ``normal_reference`` and ``t_j_minus_1_reference`` (reported),
``t_j_minus_2_reference`` for the fold-evaluated fit, and for the part (c) families
(``tmle_in_sample``, ``tmle_stacked`` with its d5 configurations, ``ltmle_in_sample``)
``wang_inflated`` (B) on every draw (reported).

**Part (c) verdicts** (family ``few_cluster_candidate``, review M1): on the first 2,000 draws,
(B) and, at counts where R5 admits it (:func:`jackknife_counts`), (C) ``cluster_jackknife``,
each paired with a copy of (A)'s row on the same draw (:func:`candidates`). (C) is fitted in its
own key (fit name ``<fit>__jackknife``), so a failure of (C) costs (C)'s cell one draw and
never (A)'s row (review B1).

**Primary.** The in-sample LTMLE at 10 clusters on ``unequal_informative``, 1,000 replications,
no comparator. The policy is ``reporting``, declared before any run.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.learners import GLM
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_unequal_laws as laws
from tests.studies import few_cluster_floor as floor
from tests.studies.canonical_cvtmle import fitted_rows
from tests.studies.canonical_drtmle import ColumnLogistic
from tests.studies.canonical_ltmle import regimen_initials, regimen_rows
from tests.studies.clustered_few_cluster_properties import size_law_parameters
from tests.studies.clustered_few_cluster_tmle import fit_cleverly as fit_ltmle
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate, stream_seed

PRIMARY_REPLICATES: Final[int] = 1_000
PRIMARY_CLUSTERS: Final[int] = 10
#: The nominal rows of the primary: 10 clusters of mean size 10.
PRIMARY_N: Final[int] = 10 * PRIMARY_CLUSTERS
SEED: Final[int] = 20265411
RESAMPLING_SEED: Final[int] = 20265412
SCENARIO: Final[str] = "floor_unequal_informative_j10"
FAMILY: Final[str] = floor.FAMILY

SIZE_LAWS: Final[tuple[str, ...]] = ("equal10", "unequal_informative")
POINT_LADDER: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 39)
LTMLE_LADDER: Final[tuple[int, ...]] = (*range(4, 21), 30, 39)
#: The current floors the ladders' anchors repeat: one per floor family.
ANCHORS: Final[dict[str, int]] = {
    "tmle_in_sample": 10,
    "tmle_stacked": 10,
    "tmle_fold_evaluated": 10,
    "drtmle_known": 10,
    "ltmle_in_sample": 20,
}
#: Each fit and its family.
FITS: Final[dict[str, str]] = {
    "tmle_in_sample_known": "tmle_in_sample",
    "tmle_in_sample_estimated": "tmle_in_sample",
    "tmle_stacked_known": "tmle_stacked",
    "tmle_stacked_estimated": "tmle_stacked",
    "tmle_fold_evaluated_known": "tmle_fold_evaluated",
    "tmle_fold_evaluated_estimated": "tmle_fold_evaluated",
    "drtmle_known": "drtmle_known",
    "tmle_stacked_d5_known": "tmle_stacked",
    "tmle_stacked_d5_estimated": "tmle_stacked",
    "ltmle_in_sample": "ltmle_in_sample",
}
#: The five-covariate stacked configurations: part (c) configurations of ``tmle_stacked``
#: (plan 5.7, "to give (B) a cell where :math:`p > 2`"), outside the floor rule; their own
#: floor reading is reported (:data:`D5_REPORTED`, review M2).
D5_FITS: Final[frozenset[str]] = frozenset({"tmle_stacked_d5_known", "tmle_stacked_d5_estimated"})
#: The name of the reported d5 floor reading.
D5_REPORTED: Final[str] = "tmle_stacked_d5"
#: The fit families of the part (c) comparison (plan 5.7).
CANDIDATE_FAMILIES: Final[frozenset[str]] = frozenset(
    {"tmle_in_sample", "tmle_stacked", "ltmle_in_sample"}
)
#: The suffix of a (C) key's fit name: (C) is its own key (review B1).
JACKKNIFE_SUFFIX: Final[str] = "__jackknife"
#: The noise columns of ``tmle_stacked_d5``.
NOISE_COLUMNS: Final[tuple[str, ...]] = ("N1", "N2", "N3")
BASE_COVARIATES: Final[tuple[str, ...]] = ("W1", "W2")
#: Every fit's estimand name.
TARGET: Final[str] = "ate"
LTMLE_TARGET: Final[str] = "ate_regimen[always vs never]"
#: The propensity bounds of every TMLE fit (``clustered-unequal-cvtmle``) and of the fitted
#: DR-TMLE reduced treatment regression (the canonical DR-TMLE study's).
G_BOUNDS: Final[tuple[float, float]] = (1e-9, 1.0 - 1e-9)
DRTMLE_G_BOUNDS: Final[tuple[float, float]] = (0.01, 0.99)
#: The largest outer fold count of a cross-fitted fit.
MAX_FOLDS: Final[int] = 5
#: Replications per cell (plan 5.1): 4,000, raised per family by the family-wise rule (M5)
#: once the pilot is read; ``tests/unit/test_few_cluster_floor_design.py`` recomputes it.
PROPERTY_REPLICATES: Final[dict[str, int]] = dict.fromkeys(ANCHORS, 4000)
#: The pilot's predicted reds, ``cell -> reading`` (review M4): empty until the pilot is read;
#: the design test checks this against the committed ``pilot-sizing.csv``.
PILOT_PREDICTED_REDS: Final[dict[str, str]] = {}
#: Families whose plug-in pass probability stays under the target at ``R_CAP`` (review M4).
PILOT_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: Families whose (C) arm the pilot measures as short of admissibility (review M4).
JACKKNIFE_SHORTFALLS: Final[frozenset[str]] = frozenset()
#: The primary's estimands.
ESTIMANDS: Final[tuple[str, ...]] = (
    "ey_regimen[never]",
    "ey_regimen[always]",
    LTMLE_TARGET,
)
TRUTH_KEYS: Final[dict[str, str]] = {
    "ey_regimen[never]": "ey0",
    "ey_regimen[always]": "ey1",
    LTMLE_TARGET: "ate",
}
REGIMENS: Final[dict[str, int]] = {"never": 0, "always": 1}


def floor_fits(family: str) -> tuple[str, ...]:
    """The fits whose verdicts set ``family``'s floor; :data:`D5_REPORTED` reads the d5 fits."""
    if family == D5_REPORTED:
        return tuple(sorted(D5_FITS))
    return tuple(fit for fit, owner in FITS.items() if owner == family and fit not in D5_FITS)


def ladder(fit: str) -> tuple[int, ...]:
    """The cluster counts a fit runs at."""
    return LTMLE_LADDER if FITS[fit] == "ltmle_in_sample" else POINT_LADDER


def laws_of(fit: str) -> tuple[str, ...]:
    """The size laws a fit runs on: fold-evaluated at equal sizes, ``d5`` at informative ones."""
    if FITS[fit] == "tmle_fold_evaluated":
        return ("equal10",)
    if fit in D5_FITS:
        return ("unequal_informative",)
    return SIZE_LAWS


def folds(fit: str, clusters: int) -> int:
    """The outer folds of a fit at ``clusters``; 1 for an in-sample fit."""
    family = FITS[fit]
    if family in {"tmle_in_sample", "ltmle_in_sample"}:
        return 1
    if family == "tmle_fold_evaluated":
        return min(MAX_FOLDS, clusters // 2)
    return min(MAX_FOLDS, clusters)


def covariates(fit: str) -> tuple[str, ...]:
    """The fit's baseline covariates."""
    return (*BASE_COVARIATES, *NOISE_COLUMNS) if fit in D5_FITS else BASE_COVARIATES


def expected_reference_df(fit: str, clusters: int) -> int:
    """The df a fit must report: :math:`J - 2`, or :math:`J - V` when fold-evaluated."""
    if FITS[fit] == "tmle_fold_evaluated":
        return clusters - folds(fit, clusters)
    return clusters - 2


def jackknife_counts(fit: str) -> tuple[int, ...]:
    """The counts where (C) runs: a part (c) family, every fold holding at least 2 clusters (M4).

    An in-sample fit runs (C) at every count; a cross-fitted one only where each of its
    :math:`\\min(5, J)` folds holds 2 clusters, which is :math:`J \\ge 10`. Below that R5
    refuses (C) before any refit, and the decision reads it as withheld on every draw.
    """
    if FITS[fit] not in CANDIDATE_FAMILIES:
        return ()
    return tuple(c for c in ladder(fit) if folds(fit, c) == 1 or c >= 2 * folds(fit, c))


def arms(fit: str, clusters: int) -> tuple[tuple[str, str], ...]:
    """The ``few_cluster_reference`` arms one fit publishes at ``clusters``, with their roles."""
    del clusters
    out = [
        (floor.REFERENCE_ARM, floor.POSITIVE),
        ("iid_t_control", floor.CONTROL),
        ("normal_reference", floor.DIAGNOSTIC),
        ("t_j_minus_1_reference", floor.DIAGNOSTIC),
    ]
    if FITS[fit] == "tmle_fold_evaluated":
        out.append(("t_j_minus_2_reference", floor.DIAGNOSTIC))
    if FITS[fit] in CANDIDATE_FAMILIES:
        out.append((floor.WANG_ARM, floor.DIAGNOSTIC))
    return tuple(out)


def candidates(fit: str, clusters: int) -> tuple[str, ...]:
    """The candidates with a ``few_cluster_candidate`` pair at ``clusters``: (B), then (C)."""
    if FITS[fit] not in CANDIDATE_FAMILIES:
        return ()
    if clusters in jackknife_counts(fit):
        return floor.VERDICT_CANDIDATES
    return (floor.WANG_ARM,)


def candidate_arms(fit: str, clusters: int) -> tuple[tuple[str, str], ...]:
    """The ``few_cluster_candidate`` arms at ``clusters``: each candidate, then (A)'s copy."""
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
        (fit, law, clusters) for fit in FITS for law in laws_of(fit) for clusters in ladder(fit)
    )


def requested(fit: str, budget: int | None = None, *, candidate: bool = False) -> int:
    """A cell's requested replications, capped at ``budget``; a part (c) cell reads only the
    first :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS`."""
    replicates = PROPERTY_REPLICATES[FITS[fit]]
    if budget is not None:
        replicates = min(replicates, budget)
    return min(replicates, floor.CANDIDATE_DRAWS) if candidate else replicates


def configurations() -> tuple[str, ...]:
    """The fits of the part (c) families, each a configuration of its family (plan 5.7)."""
    return tuple(fit for fit in FITS if FITS[fit] in CANDIDATE_FAMILIES)


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
    floor.pooled_name(fit, candidate)
    for fit in configurations()
    for candidate in floor.VERDICT_CANDIDATES
)

STUDY = StudyRecord(
    name="the clustered interval floor of point TMLE and in-sample LTMLE",
    slug="few-cluster-floor-tmle",
    artifacts=ROOT / "tests" / "canonical" / "few_cluster_floor_tmle",
    document="docs/technical-reference/method-evidence/few-cluster-floor-tmle.md",
    anchor="the-clustered-interval-floor-of-point-tmle-and-in-sample-ltmle",
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
        "tests/studies/few_cluster_floor_tmle.py",
        "tests/studies/few_cluster_floor_tmle_properties.py",
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
    runner_module="tests.studies.few_cluster_floor_tmle",
    properties_module="tests.studies.few_cluster_floor_tmle_properties",
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
    "point_ladder": list(POINT_LADDER),
    "ltmle_ladder": list(LTMLE_LADDER),
    "fits": dict(FITS),
    "outcome_learner": "unpenalized logistic regression (C=1e6) in the fit's covariates",
    "estimated_treatment": "binomial main-terms GLM",
    "known_treatment": "treatment_probabilities= the law's exact propensity",
    "ltmle_learners": "study quasibinomial GLM; exact propensity",
    "package_glm": "the same LTMLE draw with cleverly.learners.GLM node regressions (reported)",
    "noise_columns": list(NOISE_COLUMNS),
    "g_bounds": list(G_BOUNDS),
    "candidate_draws": floor.CANDIDATE_DRAWS,
    "references": {
        "A": "cluster IC variance on t(J-2) (t(J-V) fold-evaluated)",
        "B": "J/(J-p) times the IC variance on t(J-p), p = max(2, d)",
        "C": "cluster_variance='jackknife' on t(J-2)",
    },
    "primary": f"{SCENARIO}: in-sample LTMLE, 10 clusters, unequal_informative",
}


# --------------------------------------------------------------------------------------- laws


def law_truth(law: str) -> float:
    """The exact ATE of ``law``."""
    return float(laws.informative_truth(**size_law_parameters(law))["ate"])


def draw(law: str, clusters: int, seed: int) -> pd.DataFrame:
    """One draw of ``clusters`` clusters of ``law``, with the exact propensity columns."""
    fixed = np.full(clusters, 10) if law == "equal10" else None
    frame = laws.draw_informative(
        clusters, np.random.default_rng(seed), sizes=fixed, **size_law_parameters(law)
    )
    p = laws.informative_propensity(frame["W1"].to_numpy(), frame["W2"].to_numpy())
    return frame.assign(g1=p, g0=1.0 - p)


def with_noise(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The frame with :data:`NOISE_COLUMNS`, independent N(0, 1), from a child stream of ``seed``."""
    rng = np.random.default_rng([seed, 5])
    noise = rng.normal(size=(len(frame), len(NOISE_COLUMNS)))
    return frame.assign(**{name: noise[:, i] for i, name in enumerate(NOISE_COLUMNS)})


# --------------------------------------------------------------------------------------- fits


def outcome_learner() -> LogisticRegression:
    """The unpenalized main-terms logistic outcome regression."""
    return LogisticRegression(C=1e6, max_iter=1000)


def _known(fit: str) -> bool:
    return fit.endswith("_known")


def fit_point(fit: str, frame: pd.DataFrame, clusters: int, **overrides: Any) -> Any:
    """The ATE estimate of one point fit; ``overrides`` reach the estimator."""
    family = FITS[fit]
    roles: dict[str, Any] = {
        "outcome": "Y",
        "treatment": "A",
        "covariates": list(covariates(fit)),
        "id": "cluster",
    }
    if _known(fit):
        roles["treatment_probabilities"] = {1.0: "g1", 0.0: "g0"}
    settings: dict[str, Any] = {
        "outcome_learner": outcome_learner(),
        "treatment_learner": GLM(family="binomial"),
        "targeting_scheme": "pooled",
        "stratify_folds": "none",
        "estimands": (TARGET,),
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    v = folds(fit, clusters)
    if family == "tmle_in_sample":
        settings["cross_fit"] = False
    elif family == "tmle_stacked":
        settings.update(cross_fit=True, n_folds=v)
    elif family == "tmle_fold_evaluated":
        settings.update(cross_fit=True, n_folds=v, cv_evaluation=True)
    elif family == "drtmle_known":
        settings.update(
            cross_fit=True,
            n_folds=v,
            reduced_outcome_learner=LinearRegression(),
            reduced_mechanism_residual_learner=LinearRegression(),
            reduced_treatment_learner=ColumnLogistic(),
            g_bounds=DRTMLE_G_BOUNDS,
            reduced_crossfit="foldwise",
        )
        settings.pop("treatment_learner")
        settings.update(overrides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return DRTMLE(**settings).fit(frame, **roles).single()[TARGET]
    else:  # pragma: no cover - a declared fit without a construction is a design defect
        raise KeyError(fit)
    settings.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(**settings).fit(frame, **roles).single()[TARGET]


def ltmle_settings(*, package_glm: bool = False) -> dict[str, Any]:
    """The in-sample LTMLE learners: the study GLM (declared parametric) or the package GLM."""
    if package_glm:
        return {"outcome_learner": GLM(), "pseudo_learner": GLM()}
    return {
        "outcome_learner": floor.ParametricQuasiBinomialGLM(),
        "pseudo_learner": floor.ParametricQuasiBinomialGLM(),
        "treatment_learner": floor.ParametricExactPropensity(),
    }


def fit_ltmle_estimate(frame: pd.DataFrame, **overrides: Any) -> Any:
    """The in-sample LTMLE contrast of one draw."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_ltmle(frame, **{**ltmle_settings(), **overrides})[LTMLE_TARGET]


def fit_estimate(fit: str, frame: pd.DataFrame, clusters: int, **overrides: Any) -> Any:
    """The reported estimate of one fit of one draw, its ``reference_df`` checked when it
    supplies an interval (a withheld estimate reports none, review B1)."""
    if FITS[fit] == "ltmle_in_sample":
        estimate = fit_ltmle_estimate(frame, **overrides)
    else:
        estimate = fit_point(fit, frame, clusters, **overrides)
    expected = expected_reference_df(fit, clusters)
    if estimate.supplies_inference and estimate.reference_df != expected:
        raise AssertionError(
            f"{fit} at J = {clusters} reports reference_df {estimate.reference_df}, not {expected}"
        )
    return estimate


def jackknife_estimate(fit: str, frame: pd.DataFrame, clusters: int) -> Any:
    """The same fit with ``cluster_variance="jackknife"``."""
    return fit_estimate(fit, frame, clusters, cluster_variance="jackknife")


def _package_glm_reading(frame: pd.DataFrame, truth: float, clusters: int) -> dict[str, Any]:
    """The reported package-GLM refit of an LTMLE draw (M2): raised, shown and covered.

    Any exception is caught and recorded with its class and message, so the refit never
    costs the main fit's row (review B1).
    """
    try:
        estimate = fit_ltmle_estimate(frame, **ltmle_settings(package_glm=True))
    except Exception as error:  # recorded, never raised: a reported column
        return {
            "package_glm_raised": 1,
            "package_glm_shown": 0,
            "package_glm_covered": 0,
            "package_glm_error": floor.error_text(error),
        }
    reading = floor.reference_readings(
        estimate, clusters=clusters, covariates=2, arms=(floor.REFERENCE_ARM,)
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


def draw_seed(law: str, clusters: int, replicate: int, stream: str = "property") -> int:
    """The seed of one draw of a ``(law, J)`` cell on a named stream (``property`` or ``pilot``)."""
    return int(stream_seed(STUDY, stream, f"{law}/J{clusters}", replicate))


def _frame(
    fit: str, law: str, clusters: int, replicate: int, frame: pd.DataFrame, stream: str
) -> pd.DataFrame:
    """The frame a fit reads: the draw, with the noise columns for a d5 configuration."""
    if fit in D5_FITS:
        return with_noise(frame, draw_seed(law, clusters, replicate, stream))
    return frame


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

    The ``few_cluster_reference`` arms, then on the first
    :data:`~tests.studies.few_cluster_floor.CANDIDATE_DRAWS` draws the ``few_cluster_candidate``
    rows this key owns: (B) and (A)'s copy for each candidate. (C) is :func:`jackknife_rows`,
    its own key. Only the main fit raising loses the key.
    """
    truth = law_truth(law)
    frame = _frame(fit, law, clusters, replicate, frame, stream)
    estimate = fit_estimate(fit, frame, clusters)
    published = arms(fit, clusters)
    readings = floor.reference_readings(
        estimate,
        clusters=clusters,
        covariates=len(covariates(fit)),
        arms=[arm for arm, _ in published],
    )
    rows = [
        floor.arm_row(
            readings[arm],
            cell=cell_name(fit, law, clusters, arm),
            role=role,
            replicate=replicate,
            n=10 * clusters,
            requested=requested(fit, budget),
            truth=truth,
            clusters=clusters,
        )
        for arm, role in published
    ]
    if FITS[fit] == "ltmle_in_sample":
        rows[0].update(package_glm_columns(frame, truth, clusters))
    if replicate < floor.CANDIDATE_DRAWS:
        rows.extend(candidate_rows(fit, clusters, rows, budget))
    return rows


def candidate_rows(
    fit: str, clusters: int, rows: Sequence[Mapping[str, Any]], budget: int | None
) -> list[dict[str, Any]]:
    """The ``few_cluster_candidate`` rows a main key writes: (B) and each pair's (A) copy."""
    by_arm = {str(row["cell"]).rsplit("__", 1)[1]: row for row in rows}
    wanted = requested(fit, budget, candidate=True)
    out = []
    for candidate in candidates(fit, clusters):
        if candidate == floor.WANG_ARM:
            out.append(
                floor.candidate_copy(
                    by_arm[floor.WANG_ARM], arm=candidate, role=floor.POSITIVE, requested=wanted
                )
            )
        out.append(
            floor.candidate_copy(
                by_arm[floor.REFERENCE_ARM],
                arm=floor.reference_for(candidate),
                role=floor.CONTROL,
                requested=wanted,
            )
        )
    return out


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
    """(C)'s row of one draw, the key ``<fit>__jackknife`` (review B1).

    The main fit runs again (deterministic given the draw) only to check that (C)'s point is
    (A)'s. Its raising never loses this key: (A)'s own key records that loss, and (C) then reads
    its own fit (:func:`candidate_c`), whose every failure is a withheld row.
    """
    frame = _frame(fit, law, clusters, replicate, frame, stream)
    try:
        estimate = fit_estimate(fit, frame, clusters)
    except Exception:  # (A)'s loss, recorded by (A)'s key, never (C)'s
        estimate = None
    reading = candidate_c(fit, frame, clusters, estimate)
    return [
        floor.arm_row(
            reading,
            cell=cell_name(fit, law, clusters, floor.JACKKNIFE_ARM),
            role=floor.POSITIVE,
            replicate=replicate,
            n=10 * clusters,
            requested=requested(fit, budget, candidate=True),
            truth=law_truth(law),
            clusters=clusters,
        )
        | {"property": floor.CANDIDATE_FAMILY}
    ]


def candidate_c(
    fit: str, frame: pd.DataFrame, clusters: int, estimate: Any | None
) -> floor.Reading:
    """(C) on one draw, never raising: a refusal, a failed refit or an error is a withheld miss.

    A pre-fit refusal (``CapabilityError``, ``DataError``) reads ``jackknife_refused``; any
    other exception reads ``jackknife_error`` with its class and message; an estimate that
    withholds before any refit reads the main fit's trigger (:func:`~tests.studies.
    few_cluster_floor.jackknife_reading`). When the main fit ran, the jackknife fit's point must equal (A)'s: the
    jackknife changes the variance only.
    """
    from cleverly.exceptions import CapabilityError, DataError

    psi = float("nan") if estimate is None else float(estimate.psi)
    try:
        refit = jackknife_estimate(fit, frame, clusters)
    except (CapabilityError, DataError):
        return floor.refused_jackknife(psi, clusters)
    except Exception as error:  # (C)'s failure is (C)'s miss, never (A)'s
        return floor.failed_jackknife(psi, clusters, error)
    if estimate is not None and not math.isclose(
        float(refit.psi), float(estimate.psi), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise AssertionError(
            f"{fit} at J = {clusters}: the jackknife fit's point {refit.psi} is not (A)'s "
            f"{estimate.psi}"
        )
    return floor.jackknife_reading(refit, clusters=clusters)


# ------------------------------------------------------------------------------------ primary


def primary_truth() -> dict[str, float]:
    """The primary law's truth of every reported name."""
    truth = laws.informative_truth(**size_law_parameters("unequal_informative"))
    return {name: float(truth[key]) for name, key in TRUTH_KEYS.items()}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw: ``n // 10`` clusters of ``unequal_informative``."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return draw("unequal_informative", n // 10, seed), primary_truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of the primary from this study's seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: the shipped in-sample LTMLE."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_ltmle(frame, **ltmle_settings())
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


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    return fitted_rows(
        STUDY, draw_scenario, cleverly_rows, replicates=replicates, n=n, n_jobs=n_jobs
    )


__all__: Sequence[str] = (
    "FITS",
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

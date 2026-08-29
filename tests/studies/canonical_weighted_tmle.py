"""Registered evidence study for weighted ordinary point-treatment TMLE."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cleverly._typing import EstimandName
from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.conftest import OracleOutcome, OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows
from tests.studies.weighted_point_common import (
    OBSERVATION_WEIGHTS,
    FinitePointLaw,
    G,
    Q,
    population_truth,
    sample_selected,
)

TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20261001
RESAMPLING_SEED = 20261102
SCENARIO = "binary_biased_sample"
ESTIMANDS: tuple[EstimandName, ...] = ("ey0", "ey1", "ate", "rr", "or")
G_BOUNDS = (0.01, 0.99)

STUDY = StudyRecord(
    name="ordinary weighted point-treatment TMLE",
    slug="weighted-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_weighted",
    document="docs/technical-reference/method-evidence/weighted-point-treatment-tmle.md",
    anchor="weighted-point-treatment-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-weighted-tmle",
    reference="tmle-r-weighted",
    modules=(
        "tests/studies/canonical_weighted_tmle.py",
        "tests/studies/weighted_point_common.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/weighted_tmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_weighted_tmle",
    properties_module="tests.studies.weighted_tmle_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "treatment_correct",
            "both_wrong",
        ),
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ate__correctly_specified",
            "ate__shrunken_se_control",
            "ate__noise_control",
        ),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
        "targeting_necessity": ("ate__targeted", "ate__untargeted"),
        "weight_necessity": ("ate__weighted", "ate__omitted_control"),
    },
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "ordinary binary point-treatment TMLE with fixed probability weights",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "sampling": "exact-size draws from the selected finite law",
    "weight": "inverse known inclusion probability",
    "nuisance_predictions": "exact selected-law values supplied to both implementations",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one exact-size biased sample from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_selected(Q, n, seed), population_truth(Q)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's published seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    estimands: Sequence[EstimandName] = ESTIMANDS,
    outcome_learner: Any = None,
    treatment_learner: Any = None,
    use_weights: bool = True,
) -> Any:
    """Fit this study's estimator configuration on one sample.

    The primary replications and every property cell fit the same estimator, so the
    configuration is stated once here.  A property cell varies only what it declares: the
    estimands it reports, the nuisance pair it deliberately misspecifies, and whether the
    fixed weights reach the fit at all.  Everything a verdict depends on -- the targeting
    controls, the propensity bounds, the seed, and the column roles -- is shared, so the
    property cells and the published replications cannot come to describe two estimators.

    Parameters
    ----------
    frame : pandas.DataFrame
        One drawn sample, carrying ``Y``, ``A``, ``W`` and ``obs_weight``.
    estimands : Sequence[cleverly._typing.EstimandName], optional
        The parameters to report.  Defaults to the five the study publishes.
    outcome_learner : Any, optional
        The outcome learner.  Defaults to the exact selected-law oracle.
    treatment_learner : Any, optional
        The treatment learner.  Defaults to the exact selected-law oracle.
    use_weights : bool, optional
        Whether to pass ``obs_weight`` to the fit.  ``False`` is the omitted-weight
        control, which targets the selected population instead.

    Returns
    -------
    Any
        The single fitted result.
    """
    law = FinitePointLaw()
    return (
        TMLE(
            estimands=tuple(estimands),
            outcome_learner=OracleOutcome(law) if outcome_learner is None else outcome_learner,
            treatment_learner=(
                OracleTreatment(law) if treatment_learner is None else treatment_learner
            ),
            cross_fit=False,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            weights="obs_weight" if use_weights else None,
        )
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Fit and convert one primary replication to the shared schema."""
    return primary_rows(
        result=fit_cleverly(frame),
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    levels = np.rint(frame["W"].to_numpy(dtype=float)).astype(int)
    sample = frame.copy()
    sample["qn0"] = Q[levels, 0]
    sample["qn1"] = Q[levels, 1]
    sample["gn1"] = G[levels]
    np.testing.assert_allclose(sample["obs_weight"], OBSERVATION_WEIGHTS[levels])
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, fit, and retain the exact nuisances and weights used by the R comparator."""
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

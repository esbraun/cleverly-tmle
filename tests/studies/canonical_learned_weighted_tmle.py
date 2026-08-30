"""Registered ordinary TMLE study with learned weighted nuisances."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly._typing import EstimandName
from cleverly.estimators import TMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.learned_weighted_point_common import sample_selected, truths
from tests.studies.point_study_helpers import initial_estimates, primary_rows

TMLE_VERSION = "2.1.1"
TMLE_SOURCE_SHA256 = "5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20261201
RESAMPLING_SEED = 20261202
SCENARIO = "continuous_selected_weighted_nuisances"
IMPLEMENTATION = "cleverly-learned-weighted-tmle"
REFERENCE = "tmle-r-learned-weighted"
ESTIMANDS: tuple[EstimandName, ...] = ("ey0", "ey1", "ate")
G_BOUNDS = (0.01, 0.99)

STUDY = StudyRecord(
    name="ordinary learned weighted point-treatment TMLE",
    slug="learned-weighted-tmle",
    artifacts=ROOT / "tests" / "canonical" / "tmle_learned_weighted",
    document="docs/technical-reference/method-evidence/learned-weighted-point-treatment-tmle.md",
    anchor="learned-weighted-point-treatment-tmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    modules=(
        "tests/studies/canonical_learned_weighted_tmle.py",
        "tests/studies/learned_weighted_point_common.py",
        "tests/studies/learned_weighted_tmle_properties.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_learned_weighted_tmle",
    properties_module="tests.studies.learned_weighted_tmle_properties",
    property_cells={
        "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": (
            "ate__treatment_correct",
            "ate__shrunken_se_control",
            "ate__noise_control",
        ),
        "type_i_error": ("target_null",),
        "power": ("alternative",),
        "learner_weight_necessity": (
            "ate__weighted_targeted",
            "ate__unweighted_targeted",
            "ate__weighted_plugin",
            "ate__unweighted_plugin_control",
        ),
    },
    calibration_efficiency_ratio=False,
)

REFERENCE_METADATA = {
    "tmle_version": TMLE_VERSION,
    "tmle_source_sha256": TMLE_SOURCE_SHA256,
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "construction": "ordinary weighted point-treatment TMLE with learned nuisances",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "fluctuation": "linear",
    "g_bounds": list(G_BOUNDS),
    "outcome_learner": "weighted main-effects linear regression",
    "treatment_learner": "weighted unpenalized logistic regression",
    "sampling": "exact-size inverse-CDF draws from the selected continuous law",
    "weight": "exact target-to-selected density ratio",
}


class UnweightedLinearRegression(LinearRegression):
    """Linear regression control that deliberately discards supplied learner weights."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> UnweightedLinearRegression:
        del sample_weight
        return super().fit(X, y)


class UnweightedLogisticRegression(LogisticRegression):
    """Logistic regression control that deliberately discards supplied learner weights."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> UnweightedLogisticRegression:
        del sample_weight
        return super().fit(X, y)


def outcome_learner(*, weighted: bool = True) -> Any:
    """Return the declared main-effects outcome learner or its learner-only control."""
    return LinearRegression() if weighted else UnweightedLinearRegression()


def treatment_learner(*, weighted: bool = True) -> Any:
    """Return the declared unpenalized treatment learner or its learner-only control."""
    cls = LogisticRegression if weighted else UnweightedLogisticRegression
    return cls(C=np.inf, max_iter=2_000, solver="lbfgs")


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one selected-law sample from an explicit seed."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_selected(n, seed), truths()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from the study's published seed stream."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_cleverly(
    frame: pd.DataFrame,
    *,
    estimands: Sequence[EstimandName] = ESTIMANDS,
    learner_weights: bool = True,
) -> Any:
    """Fit the declared estimator, optionally omitting weights from nuisance learning alone."""
    return (
        TMLE(
            estimands=tuple(estimands),
            outcome_learner=outcome_learner(weighted=learner_weights),
            treatment_learner=treatment_learner(weighted=learner_weights),
            cross_fit=False,
            simultaneous=False,
            fluctuation="linear",
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            weights="obs_weight",
        )
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    """Fit one primary replication and convert it to the shared schema."""
    result = fit_cleverly(frame)
    return primary_rows(
        result=result,
        truth=truth,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result, ESTIMANDS),
    )


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    sample = frame.copy()
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
    """Draw and fit primary rows while retaining the exact R comparator inputs."""
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]

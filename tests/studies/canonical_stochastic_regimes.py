"""Registered evidence study for known stochastic point-treatment regimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.interventions import Static, Stochastic
from cleverly.utils.parallel import map_parallel
from tests import discrete_law as law
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import replicate_seed
from tests.studies.intervention_study_helpers import (
    initial_regime_estimates,
    primary_rows,
    sample_discrete,
    truths,
)

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_000
SEED = 20260902
SCENARIO = "binary_known_stochastic"
ESTIMANDS = (
    "ey_regime[never]",
    "ey_regime[tilt]",
    "ate_regime[tilt vs never]",
)
G_BOUNDS = (0.01, 0.99)


def _levels(frame: Any) -> np.ndarray:
    return np.rint(np.asarray(frame["W"], dtype=float)).astype(int)


def interventions(*, uniform: bool = False) -> tuple[Any, ...]:
    density = np.full_like(law.REGIMES["tilt"], 0.5) if uniform else law.REGIMES["tilt"]
    return (
        Static(0.0, name="never"),
        Stochastic(lambda frame: density[_levels(frame)], name="tilt"),
    )


STUDY = StudyRecord(
    name="ordinary known stochastic point-treatment regimes",
    slug="stochastic-regimes",
    artifacts=ROOT / "tests" / "canonical" / "stochastic_regimes",
    document="docs/technical-reference/method-evidence/stochastic-point-treatment-regimes.md",
    anchor="stochastic-point-treatment-regimes",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260912,
    margins=Margins(),
    implementation="cleverly",
    reference=None,
    modules=(
        "tests/studies/canonical_stochastic_regimes.py",
        "tests/studies/stochastic_regime_properties.py",
        "tests/studies/intervention_study_helpers.py",
        "tests/studies/regime_property_helpers.py",
        "tests/discrete_law.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_stochastic_regimes",
    properties_module="tests.studies.stochastic_regime_properties",
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
            "tilt__correctly_specified",
            "tilt__shrunken_se_control",
            "tilt__noise_control",
        ),
        "type_i_error": ("sharp_null",),
        "power": ("alternative",),
        "targeting_necessity": ("tilt__targeted", "tilt__untargeted"),
        "density_necessity": ("tilt__declared", "tilt__uniform_control"),
    },
)

CONFIGURATION = {
    "construction": "ordinary",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(G_BOUNDS),
    "regimes": ["never", "tilt"],
    "external_comparator": None,
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample_discrete(law.PROBS, n, seed), truths(law.PROBS, ESTIMANDS)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_from_seed(scenario, n, replicate_seed(STUDY, scenario, replicate))


def fit_cleverly(frame: pd.DataFrame, *, uniform: bool = False) -> Any:
    return (
        TMLE(
            interventions=interventions(uniform=uniform),
            outcome_learner=LogisticRegression(C=1e6, max_iter=2_000),
            treatment_learner=OracleTreatment(law.DiscreteLaw()),
            cross_fit=False,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W"])
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    reference: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame)
    return primary_rows(
        result=result,
        reference=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        initials=initial_regime_estimates(result),
        estimands=ESTIMANDS,
    )


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    return cleverly_rows(frame, reference, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(
        _replicate,
        [((SCENARIO, replicate, n),) for replicate in range(replicates)],
        n_jobs=n_jobs,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, list(REPLICATE_COLUMNS)]

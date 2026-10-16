r"""Registered evidence for categorical incremental policies (``categorical-incremental``, X43).

The subject is the categorical odds tilt of ``X43-plan.md`` section 1: point
:class:`~cleverly.estimators.TMLE` with ``Incremental({level: factor})`` on a three-level
treatment (family P), and :class:`~cleverly.longitudinal.IPSI` with two three-level tilt
nodes (family L).  No maintained package ships a categorical tilt with its influence curve:
``npcausal::ipsi`` is binary only and ``imtp`` 0.1.0 omits the derivative term
(``docs/development/method-benchmarking.md``, the survey).  So the study has no comparator,
and it publishes Python property evidence (architecture invariants, "Theory and
comparators").

======================================  ===========================================================
scenario                                law and estimands
======================================  ===========================================================
``point_in_sample``                     ``P``, nuisances fitted in sample; the three tilted means
                                        and the two contrasts against the natural course
``point_crossfit``                      ``P``, five folds; the same estimands
``longitudinal_dropout_folds_5``        ``L`` (dropout after each node), five folds; the three plan
                                        means at the end of study and the two contrasts
``longitudinal_folds_5``                ``L0`` (no dropout), five folds; the same estimands
======================================  ===========================================================

The plans tilt by ``T_a = {b: 2, c: 0.5}`` and ``T_b = {b: 0.5, c: 2}``, at every node in
family L, against the natural course (``{a: 1}``).  Every nuisance is saturated
(:mod:`tests.studies.categorical_incremental_common`).

**Size** (declared before any run).  The record holds one primary size for every scenario.
The plan's rung rule (``X43-plan.md`` 5.3) takes the smallest rung at which every primary
contrast has a standardized bias of at most 0.25 in a pilot of 400 draws: 1,000 for family P
(:data:`POINT_PILOT_N`) and 2,000 for family L (:data:`LONGITUDINAL_PILOT_N`).  At those sizes
the 1,000-replicate bias gate of the twenty primary rows passes with a predictive chance of
about 0.35, so :data:`PRIMARY_N` is the smallest rung at which the gate's joint predictive
chance is at least 0.95, with each row's bias decaying as the root of ``n`` from its pilot
fit (``tests/unit/test_categorical_incremental_design.py``).

**Properties.** :mod:`tests.studies.categorical_incremental_properties` declares every family.

**Red-cell route** (declared before any run).  A red primary truth row blocks the merge and
gets a diagnosis.  A red calibration row is read by the rule of ``AGENT-BRIEF.md``, "Reading a
red calibration row": it is a finite-sample red only when its SE-ratio interval contains 1,
its coverage is consistent with 0.95 and the diagnosis finds no defect; it then moves to
``reporting`` with an owner row before any re-run.  Any other red row is a defect suspect: it
is diagnosed, a defect is fixed and the study re-runs under a fresh declaration.  No budget,
margin, law, learner, fold count or seed moves after a verdict is seen.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tests import discrete_law_categorical_ipsi_longitudinal as long_law
from tests import discrete_law_categorical_ipsi_point as point
from tests.parallel import STUDY_JOBS
from tests.studies import categorical_incremental_common as common
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
#: Family P's pilot rung (``tests/unit/test_categorical_incremental_design.py``).
POINT_PILOT_N = 1_000
#: Family L's pilot rung.
LONGITUDINAL_PILOT_N = 2_000
#: The primary size of both families: the bias gate's joint predictive chance >= 0.95.
PRIMARY_N = 8_000
#: The size of the point calibration and necessity cells.  Every rung of family P's pilot has
#: calibrated intervals; this one carries the 8,400-draw calibration pilot of the sizing rule.
POINT_CALIBRATION_N = 4_000
#: The two sizes of the ``single_robustness`` pair; the diagnostic cell reads the smaller.
#: The two pilot rungs at which the constant-outcome fit's bias is furthest inside the margin.
REQUIREMENT_SIZES = (8_000, 16_000)
SEED = 20261843
RESAMPLING_SEED = 2026184301

REFERENCE = point.REFERENCE
assert long_law.REFERENCE == REFERENCE

POINT_ESTIMANDS: tuple[str, ...] = point.MEANS + point.CONTRASTS
LONGITUDINAL_ESTIMANDS: tuple[str, ...] = (
    *(f"ey_regimen[{label}]" for label in long_law.PLANS),
    *(f"ate_regimen[{label} vs {REFERENCE}]" for label in long_law.PLANS if label != REFERENCE),
)
#: ``scenario -> (family, law, folds)``.
SCENARIOS: dict[str, tuple[str, Any, int]] = {
    "point_in_sample": ("P", point.LAW_P, 1),
    "point_crossfit": ("P", point.LAW_P, common.N_FOLDS),
    "longitudinal_dropout_folds_5": ("L", long_law.LAW_L, common.N_FOLDS),
    "longitudinal_folds_5": ("L", long_law.LAW_L0, common.N_FOLDS),
}
ESTIMANDS: dict[str, tuple[str, ...]] = {
    scenario: POINT_ESTIMANDS if family == "P" else LONGITUDINAL_ESTIMANDS
    for scenario, (family, _, _) in SCENARIOS.items()
}


def longitudinal_truth(target: Any, name: str) -> float:
    """The exact value of a reported ``ey_regimen`` or ``ate_regimen`` name on ``target``."""
    stem, _, rest = name.partition("[")
    parts = rest[:-1].split(" vs ")
    value = long_law.truth(target, long_law.PLANS[parts[0]])
    if stem == "ate_regimen":
        value -= long_law.truth(target, long_law.PLANS[parts[1]])
    return float(value)


def longitudinal_eif_sd(target: Any, name: str) -> float:
    """The efficient influence-function SD of a reported longitudinal name on ``target``."""
    stem, _, rest = name.partition("[")
    parts = rest[:-1].split(" vs ")
    other = long_law.PLANS[parts[1]] if stem == "ate_regimen" else None
    return long_law.eif_sd(target, long_law.PLANS[parts[0]], other=other)


def scenario_truth(scenario: str) -> dict[str, float]:
    family, target, _ = SCENARIOS[scenario]
    if family == "P":
        return {name: point.truth(target, name) for name in ESTIMANDS[scenario]}
    return {name: longitudinal_truth(target, name) for name in ESTIMANDS[scenario]}


TRUTH = {scenario: scenario_truth(scenario) for scenario in SCENARIOS}
#: The efficiency bound of each estimand.  The two point scenarios share their law, and the
#: longitudinal names are read on law ``L`` (the dropout law; ``L0`` has the same values and a
#: smaller bound, which the properties module reads where a cell is on ``L0``).
EFFICIENCY_SD: dict[str, float] = {
    **{name: point.eif_sd(point.LAW_P, name) for name in POINT_ESTIMANDS},
    **{name: longitudinal_eif_sd(long_law.LAW_L, name) for name in LONGITUDINAL_ESTIMANDS},
}

#: Every published property cell, by family (see the properties module).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in (
            "point_contrast_t_a",
            "point_contrast_t_b",
            "point_known_t_a",
            "point_stratified_t_a_v1",
            "longitudinal_dropout_contrast_t_a",
            "longitudinal_contrast_t_a",
        )
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "type_i_error": ("point_contrast_t_a__sharp_null",),
    "power": ("point_contrast_t_a__alternative",),
    "mechanism_term_necessity": (
        "point_contrast_t_a__full_eif",
        "point_contrast_t_a__mechanism_term_dropped",
        "point_contrast_t_a__binary_blip",
    ),
    "single_robustness": (
        *(f"point_contrast_t_a__mechanism_correct__n_{size}" for size in REQUIREMENT_SIZES),
        "point_contrast_t_a__mechanism_constant",
    ),
    "in_sample_fit": ("longitudinal_dropout_contrast_t_a__one_fold",),
    "censoring_complete_case": ("longitudinal_mean_t_a__complete_case",),
}

STUDY = StudyRecord(
    name="categorical incremental policies",
    slug="categorical-incremental",
    artifacts=ROOT / "tests" / "canonical" / "categorical_incremental",
    document="docs/technical-reference/method-evidence/categorical-incremental.md",
    anchor="categorical-incremental-policies",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-categorical-incremental",
    reference=None,
    failure_bound={"cleverly-categorical-incremental": FAILURE_BOUND},
    efficiency_bounds=EFFICIENCY_SD,
    modules=(
        "tests/studies/canonical_categorical_incremental.py",
        "tests/studies/categorical_incremental_properties.py",
        "tests/studies/categorical_incremental_common.py",
        "tests/discrete_law_categorical_ipsi_point.py",
        "tests/discrete_law_categorical_ipsi_longitudinal.py",
        "tests/discrete_law_longitudinal_incremental.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law.py",
        "tests/studies/intervention_study_helpers.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_categorical_incremental",
    properties_module="tests.studies.categorical_incremental_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": {
        "P": "point TMLE, softmax mechanism fluctuation (X43-plan.md 3.1)",
        "L": "IPSI one-step with categorical tilt nodes and dropout weights (X43-plan.md 3.2)",
    },
    "n_folds": {scenario: folds for scenario, (_, _, folds) in SCENARIOS.items()},
    "learners": "CellFrequencies for each mechanism, CellMeans for each regression",
    "q_bounds": list(common.Q_BOUNDS),
    "tilts": {name: dict(factors) for name, factors in point.TILTS.items()},
    "reference": REFERENCE,
}


# ---------------------------------------------------------------------------- fits


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The one fit of each primary scenario."""
    family, target, folds = SCENARIOS[scenario]
    if family == "P":
        return common.fit_point(frame, cross_fit=folds > 1)
    return common.fit_longitudinal(frame, target, n_folds=folds)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    family, target, _ = SCENARIOS[scenario]
    frame = point.sample(target, n, seed) if family == "P" else long_law.sample(target, n, seed)
    return frame, dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per estimand.  ``initial_estimate`` is the untargeted plug-in."""
    rows = []
    family = SCENARIOS[scenario][0]
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": int(result.n if family == "L" else result.data.n),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": common.initial_estimate(result, name, family=family),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    return result_rows(fit(frame, scenario), truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every primary row.  No comparator reads the samples, so none is kept."""
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for scenario in SCENARIOS for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])

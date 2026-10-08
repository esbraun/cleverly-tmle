"""Registered evidence for cross-fitted point-treatment survival (``point-treatment-survival-crossfit``).

The subject is ``LTMLE`` on the held design with five outer folds: each fold runs an
untargeted backward recursion on its training rows, and one pooled fluctuation per node
targets the stitched out-of-fold predictions (Díaz, Williams, Hoffman and Schenck 2023,
Section 5.2).  No maintained package ships this construction for a held baseline treatment,
so the study has no comparator and publishes an empty equivalence table.

The scenarios are ``survival`` (the five-visit law of ``point-treatment-survival``; the risks
of both arms and their difference at visits 1, 3 and 5) and ``three_arm`` (three arms with a
binary ``L_t`` on the wide layout; the risks of arms 0 and 2 and the differences of arms 1 and
2 against arm 0 at visits 1 and 3), each at n = 2,000 with 1,600 replications.  The learners are
the saturated cell means, which are correct on these finite laws.

**A separated fit keeps its last iterate.**  ``QuasiBinomialGLM`` keeps its last iterate,
with ``QuasiBinomialSeparationWarning``, when its coefficients diverge while its deviance
settles by R's ``glm.control`` rule.  R's ``glm`` returns such a fit with a warning, and the
comparator fits ``glm``.  A fit whose deviance has not settled still raises and fails the run.
A failure-only scan of every declared primary replication, which read no estimate, found no
such fit and no failed fit.

Publication policy is ``reporting``.  The red-cell route was declared before any run.
Every red cell is diagnosed before it is routed.  A genuine defect (an algorithm
defect, an inconsistent estimator, or a test or design bug that makes the cell measure the
wrong thing) is fixed and re-run under a fresh declaration commit that records the change.
Nothing changes to make a cell easier to pass: no margin, budget, law or cell moves.
A finite-sample red with no defect stays published red: a band cell under
``band-finite-sample``, and any other cell under ``X13-finite-sample``.  A replication that
raises is never redrawn: a failed replication fails the run.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import point_survival_common as common
from tests.studies.canonical_ltmle import regimen_rows
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261061
RESAMPLING_SEED = 2026106101
N_FOLDS = 5

SCENARIOS = {"survival": common.SURVIVAL, "three_arm": common.THREE_ARM}
REGIMENS = {
    "survival": {"arm0": 0, "arm1": 1},
    "three_arm": {"arm0": 0, "arm1": 1, "arm2": 2},
}
REPORTED = {"survival": (1, 3, 5), "three_arm": (1, 3)}
LEVELS_REPORTED = {"survival": ("arm0", "arm1"), "three_arm": ("arm0", "arm2")}


def _names(scenario: str) -> tuple[str, ...]:
    levels = tuple(
        common.risk_name(label, horizon)
        for label in LEVELS_REPORTED[scenario]
        for horizon in REPORTED[scenario]
    )
    contrasts = tuple(
        common.contrast_name(label, "arm0", horizon)
        for label in REGIMENS[scenario]
        if label != "arm0"
        for horizon in REPORTED[scenario]
    )
    return levels + contrasts


ESTIMANDS = {scenario: _names(scenario) for scenario in SCENARIOS}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": tuple(
        f"survival_t5__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in ("survival_t5", "three_arm_t3")
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "simultaneous_coverage": (
        "all_reported__simultaneous_band",
        "all_reported__pointwise_joint_control",
    ),
}

STUDY = StudyRecord(
    name="cross-fitted point-treatment survival",
    slug="point-treatment-survival-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "point_survival_crossfit",
    document="docs/technical-reference/method-evidence/cross-fitted-point-treatment-survival.md",
    anchor="cross-fitted-point-treatment-survival",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-point-survival",
    reference=None,
    modules=(
        "tests/studies/canonical_point_survival_crossfit.py",
        "tests/studies/point_survival_crossfit_properties.py",
        "tests/studies/point_survival_properties.py",
        "tests/studies/canonical_point_survival.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/datasets/survival_point.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_point_survival_crossfit",
    properties_module="tests.studies.point_survival_crossfit_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "held point-treatment survival, pooled cross-fitted targeting",
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "learners": "saturated cell means for every nuisance",
    "reported_nodes": {scenario: list(nodes) for scenario, nodes in REPORTED.items()},
}


def _truth(scenario: str) -> dict[str, float]:
    law = SCENARIOS[scenario]
    plans = {label: common.static(arm, law.levels) for label, arm in REGIMENS[scenario].items()}
    values = common.truths(law, plans, "arm0")
    return {name: values[name] for name in ESTIMANDS[scenario]}


TRUTH = {scenario: _truth(scenario) for scenario in SCENARIOS}


def fit(scenario: str, frame: pd.DataFrame) -> Any:
    """The subject's primary fit of one scenario's sample."""
    return LTMLE(
        REGIMENS[scenario],
        reference="arm0",
        n_folds=N_FOLDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        outcome_learner=common.CellMeans(),
        pseudo_learner=common.CellMeans(),
        treatment_learner=common.CellProbabilities(),
        censoring_learner=common.CellMeans(),
    ).fit(SCENARIOS[scenario].container(frame))


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return SCENARIOS[scenario].draw(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _initials(result: Any, names: tuple[str, ...]) -> dict[str, float]:
    from tests.studies.canonical_point_survival import initial_estimates

    return initial_estimates(result, names)


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit(scenario, frame)
    return regimen_rows(
        STUDY,
        result,
        truth,
        _initials(result, ESTIMANDS[scenario]),
        ESTIMANDS[scenario],
        scenario,
        replicate,
        n=len(frame),
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Every replication's rows; the study has no comparator, so no sample is kept."""
    payloads = [
        ((scenario, replicate, n),) for scenario in SCENARIOS for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, list(REPLICATE_COLUMNS)]


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit"]

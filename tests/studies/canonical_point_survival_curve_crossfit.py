"""Registered evidence for cross-fitted hazard targeting (``point-survival-curve-crossfit``).

The subject is ``LTMLE`` with ``survival_model="hazard"`` and five outer folds on the held design
of :mod:`tests.studies.canonical_point_survival_curve`.  Each fold fits the hazards, the treatment
and the censoring on its training rows, and one pooled targeting step reads the stitched
out-of-fold predictions.  The one-step walk targets every horizon at once; the per-horizon
iteration targets each horizon on its own.  No maintained package ships hazard targeting with
cross-fitted nuisances for a held baseline treatment, so the study has no comparator and
publishes an empty equivalence table.

The primary scenario is ``survival`` (five visits; the risks of both arms and their difference at
visits 1, 3 and 5), one step, at n = 2,000 with 1,600 replications.  The learners are
``learners("both_correct")`` of :mod:`tests.studies.point_survival_hazard_properties`: an additive
logistic hazard and saturated cell fits for the treatment and the censoring, each correct on this
finite law.  The property cells are in
:mod:`tests.studies.point_survival_curve_crossfit_properties`.

Publication policy is ``reporting``.  The red-cell route was declared before any run.  A band
cell goes to ``band-finite-sample`` only if the shortfall-reading diagnostic of
``tests/unit/test_band_shortfall_reading.py`` reproduces for it, and otherwise to
``X14-finite-sample``.  Any other red cell goes to ``X14-finite-sample``, which closes when a
re-declared cell passes.  No re-run uses more replicates, wider margins or a new law.  A
replication that raises is never redrawn.  A recordable fit failure is recorded and costs its
replicate under a zero failure bound, so its cells read ``bound exceeded``.  Such a cell goes
to ``X14-fit-failures``, with its failure count stated as a page limit, which closes when a
re-declared run records no failure.  A fatal error or a systematic failure stops the run.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import point_survival_common as common
from tests.studies.canonical_ltmle import regimen_rows
from tests.studies.canonical_point_survival import REFERENCE, REGIMENS, REPORTED, SCENARIOS
from tests.studies.canonical_point_survival_curve import STEP_SIZE, TOL
from tests.studies.canonical_point_survival_hazard import initial_estimates
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.point_survival_hazard_properties import learners

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261072
RESAMPLING_SEED = 2026107201
N_FOLDS = 5
ALGORITHM = "one_step"

SCENARIO = "survival"
LAW = SCENARIOS[SCENARIO]
LEVELS = tuple(
    common.risk_name(label, horizon) for label in REGIMENS for horizon in REPORTED[SCENARIO]
)
CONTRASTS = tuple(common.contrast_name("arm1", REFERENCE, t) for t in REPORTED[SCENARIO])
ESTIMANDS = {SCENARIO: LEVELS + CONTRASTS}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        "survival_t5__correctly_specified",
        "survival_t5__shrunken_se_control",
        "survival_t5__noise_control",
        "survival_t5_per_horizon__correctly_specified",
        "competing_t4__correctly_specified",
    ),
    "double_robustness": tuple(
        f"survival_t5__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "simultaneous_coverage": tuple(
        f"{label}__{kind}"
        for label in ("curve_arm1", "curve_arm0", "all_reported")
        for kind in ("simultaneous_band", "pointwise_joint_control")
    ),
}

STUDY = StudyRecord(
    name="cross-fitted point-treatment survival, hazard targeting",
    slug="point-survival-curve-crossfit",
    artifacts=ROOT / "tests" / "canonical" / "point_survival_curve_crossfit",
    document="docs/technical-reference/method-evidence/point-survival-curve-crossfit.md",
    anchor="point-survival-curve-crossfit",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-hazard-survival",
    reference=None,
    failure_bound={"cleverly-cross-fitted-hazard-survival": 0.01},
    modules=(
        "tests/studies/canonical_point_survival_curve_crossfit.py",
        "tests/studies/point_survival_curve_crossfit_properties.py",
        "tests/studies/canonical_point_survival_curve.py",
        "tests/studies/point_survival_curve_properties.py",
        "tests/studies/canonical_point_survival_hazard.py",
        "tests/studies/point_survival_hazard_properties.py",
        "tests/studies/canonical_point_survival.py",
        "tests/studies/point_survival_properties.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/ltmle_survival_properties.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/discrete_law_survival.py",
        "src/cleverly/datasets/survival_point.py",
        "src/cleverly/longitudinal/hazard.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_point_survival_curve_crossfit",
    properties_module="tests.studies.point_survival_curve_crossfit_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "held point-treatment survival, cross-fitted hazard targeting",
    "survival_model": "hazard",
    "targeting": ALGORITHM,
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "learners": "an additive logistic hazard; saturated cell fits for treatment and censoring",
    "reported_nodes": list(REPORTED[SCENARIO]),
}

TRUTH = {
    SCENARIO: {
        name: value
        for name, value in common.truths(
            LAW,
            {label: common.static(arm, LAW.levels) for label, arm in REGIMENS.items()},
            REFERENCE,
        ).items()
        if name in ESTIMANDS[SCENARIO]
    }
}


def fit(frame: pd.DataFrame) -> Any:
    """The subject's primary fit of one sample."""
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        n_folds=N_FOLDS,
        simultaneous=False,
        survival_model="hazard",
        targeting=ALGORITHM,
        step_size=STEP_SIZE,
        tol=TOL,
        random_state=0,
        **learners("both_correct"),
    ).fit(LAW.container(frame))


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return LAW.draw(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit(frame)
    names = ESTIMANDS[scenario]
    return regimen_rows(
        STUDY,
        result,
        truth,
        initial_estimates(result, names),
        names,
        scenario,
        replicate,
        n=len(frame),
    )


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every replication's rows; the study has no comparator, so no sample is kept."""
    outcomes = replication.fit_primary(
        [(SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit"]

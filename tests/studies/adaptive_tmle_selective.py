r"""Registered evidence for adaptive TMLE under selective enrollment (X90 T3, study S6).

The sibling of :mod:`tests.studies.adaptive_tmle_trial` for the plan's cell S6 and control K6
(Rev 2, from Karim, arXiv:2607.02787v2). The law is the (b) frame with selective enrollment,
``S ~ Bernoulli(expit(-0.9 + 0.8 W1 + 0.5 W2))`` (1.4% of the covariate mass has
:math:`P(S = 1 | W)` outside [0.05, 0.95]), and a large wiggly bias of four residual standard
deviations. It is a record of its own because its nuisances are the flexible Super Learner
(spline GLM, shallow regularized gradient boosting, lasso; ``nnloglik`` for probabilities), whose
cross-fitted cost is two orders above the spline learners of study A, so its replications go to
the local queue and its sample size is its own.

``s6_cv`` and ``s6_pooled``
    The Psi coverage of both forms with the Super Learner nuisances, gated by the primary
    performance tests (bias leg, coverage floor, SE sanity). The study stays
    ``gated``. The second run read the bias leg of ``s6_cv`` at 0.20 SD with no defect, and the
    owner accepted the study by the named exception ``red_cells.ACCEPTED_BY_EXCEPTION`` on
    2026-11-10, with no repeat run. On this
    law the projection and :math:`\Psi` coincide only if the dictionary approximates the wiggly bias, which the
    reported rows measure (``projection_bias_sd``).
K6
    The same fit and form with main-terms GLM nuisances, on the same draws, a reporting row.
    Karim reports coverage 0.81 falling to 0.47 as n grows from 500 to 3,200, but a 40-replicate
    pilot of the pooled form here read coverage 0.93 (bias +0.20, SE 0.26), so the contrast has
    no power on this law at n = 3,200 and cannot gate.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies import _atmle_laws as laws
from tests.studies import adaptive_tmle_trial as base
from tests.studies.adaptive_tmle_trial import Cell
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

SEED = 20290321
RESAMPLING_SEED = 20290322
PRIMARY_N = 3_200
#: Replications of each scenario: ``repeated_sizing.primary_replicates`` at the pilot that
#: ``test_adaptive_tmle_trial_design`` pins: 1,200 replicates, which the rule prices at its
#: floor of 2,400.
PRIMARY_REPLICATES = 2_400

IMPLEMENTATION = "cleverly-adaptive-tmle-selective"

CELLS: dict[str, Cell] = {
    "s6_cv": Cell("s6", "psi", learners="flexible", controls=("k6",)),
    "s6_pooled": Cell("s6", "psi", form="pooled", learners="flexible", controls=("k6",)),
}
SCENARIOS = {name: (cell.estimand,) for name, cell in CELLS.items()}

CONTROLS_ARTIFACT = base.CONTROLS_ARTIFACT
REPORTED_ARTIFACT = base.REPORTED_ARTIFACT

STUDY = StudyRecord(
    name="Adaptive TMLE under selective enrollment",
    slug="adaptive-tmle-selective",
    artifacts=ROOT / "tests" / "canonical" / "adaptive_tmle_selective",
    document="docs/technical-reference/method-evidence/adaptive-tmle-selective.md",
    anchor="adaptive-tmle-under-selective-enrollment",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: 0.01},
    modules=(
        "tests/studies/adaptive_tmle_selective.py",
        "tests/studies/adaptive_tmle_selective_properties.py",
        "tests/studies/adaptive_tmle_trial.py",
        "tests/studies/_atmle_laws.py",
        "src/cleverly/estimators/atmle.py",
        "src/cleverly/estimators/atmle_result.py",
        "src/cleverly/estimators/_working_basis.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.adaptive_tmle_selective",
    properties_module="tests.studies.adaptive_tmle_selective_properties",
    property_cells={},
    calibration_efficiency_ratio=False,
    publication_policy="gated",
    extra_artifacts=(REPORTED_ARTIFACT, CONTROLS_ARTIFACT),
    truth_varies_by_replicate=True,
)

CONFIGURATION = {
    "fit": str(base.CONFIGURATION["fit"]).replace("spline learners", "Super Learner learners")
    + "; n = 3200",
    "learners": (
        "SuperLearner(spline GLM: SplineTransformer(5 knots, degree 3) + logistic or linear; "
        "HistGradientBoosting(max_depth=3, learning_rate=0.05, l2_regularization=1.0, "
        "early_stopping=False); lasso of default_library), every candidate seeded by the "
        "ensemble's random_state, meta_learner='nnloglik' with clip=(0.01, 0.99) for "
        "probabilities, cc_ls for the outcome; K6: LinearRegression and LogisticRegression "
        "main terms"
    ),
    "cells": {name: vars(cell) for name, cell in CELLS.items()},
    "truth": base.CONFIGURATION["truth"],
    "fold_seed": "stream_seed(record, 'fold_partition', scenario, replicate)",
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in CELLS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    law = laws.LAWS[CELLS[scenario].law]
    targets = base.exact(CELLS[scenario].law)
    return laws.draw(law, n, seed), {"psi": targets["psi"], "psi_2": targets["psi_2"]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS, record: StudyRecord = STUDY
) -> replication.PhaseResult:
    """Draw and fit every primary replication of ``record``."""
    return base.draw_and_fit(replicates=replicates, n=n, n_jobs=n_jobs, record=record, cells=CELLS)


def controls_table(rows: pd.DataFrame, record: StudyRecord = STUDY) -> pd.DataFrame:
    """K6: the main-terms fit's Psi coverage on each form, a reporting row.

    The pilot (40 pooled-form replicates) read bias +0.20 against SE 0.26 and coverage 0.93, so
    the contrast has no power at this law and cannot gate (plan K6 expected a failure).
    """
    out = [
        base.coverage_row(
            group,
            record,
            f"k6_main_terms_{name.removeprefix('s6_')}",
            "reporting",
            target="control",
        )
        for name, group in rows.groupby("scenario")
    ]
    return pd.DataFrame.from_records(out)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        REPORTED_ARTIFACT: base.reported_table(rows, STUDY, CELLS),
        CONTROLS_ARTIFACT: controls_table(rows),
    }


def scientific_failures(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    controls = frames[CONTROLS_ARTIFACT]
    return {"control verdicts": controls.loc[~controls["passed"].astype(bool)]}

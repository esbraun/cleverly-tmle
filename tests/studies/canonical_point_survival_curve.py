"""Registered evidence for one-step hazard targeting of a whole curve (``point-survival-curve``).

The subject is ``LTMLE`` with ``survival_model="hazard"`` and ``targeting="one_step"`` on a
held design: one baseline treatment held over every node (``TimeToEvent``).  One walk moves
one set of discrete hazards toward the score equation of every horizon of every regimen at
once (Cai and van der Laan 2020; :mod:`cleverly.longitudinal.hazard`), so each risk curve is
non-decreasing in floating point.  The law is X13's ``survival`` scenario (S1) of
:mod:`tests.studies.point_survival_common`, so every truth is an exact finite sum and every
efficiency bound is exact.

**The comparator** is ``tmle3``'s ``tmle_survival`` at ``ed72f8a`` (``sl3`` at ``0e8f236``), on
the same realized samples, through ``tests/canonical/tmle3_survival/run_study.R`` on its own
image, which adds ``glmnet`` 5.0 to the ``tmle3`` pins:

* the treatment model is the logistic regression on ``W1 + W2`` on both sides;
* the event and censoring hazards are one logistic regression on ``A, W1, W2`` per visit on both
  sides (``Lrnr_stratified`` on ``t`` over ``Lrnr_glm``);
* ``tmle3`` fits one specification per arm, with ``target_times = 1:5``, and the contrast is
  the difference of the two arms' per-unit influence curves.  The risk is one minus the
  survival ``tmle3`` reports.

``tmle3`` runs its survival updater with ``fit_method = "l2"``, ``cvtmle = FALSE``,
``convergence_type = "sample_size"``, ``maxit = 100`` and ``delta_epsilon = 0.01``.  The runner
subclasses the updater to repair two R-level defects of its ``fit_submodel`` and nothing else:
the vector test ``fit_lambda == 1`` becomes ``length(fit_lambda) == 1``, and the retry assigns
``lambda.min.ratio``, so a retry refits a different path.  The runner records, per replicate
and arm, the steps with a nonzero step, the final :math:`\\max_t |P_n D_t|` against
:math:`1/n` and whether ``maxit`` bound.  A replicate whose arm took no nonzero step and left
the score unsolved fails the run.  A replicate with a nonzero step and an unsolved score stays
in the paired rows; the run log counts it and the evidence page reports it.

Declared unmatched settings: ``tmle3`` bounds ``g`` at 0.025 and the hazards at 0.025 and
0.005, and this package uses its shipped ``g_bounds``; ``tmle3`` fits the censoring hazard on
rows that include the failure row; ``tmle3`` takes ridge-penalized steps of at most
``delta_epsilon`` in norm and stops at :math:`1/n`, and this package walks the universal least
favorable submodel at ``step_size = 1e-3`` and stops at a relative score of ``1e-10``; and the
two repairs above.
The censoring convention makes this package's SE grow against ``tmle3``'s with the visit, by
about 2 percent at visit 5 in the A2 smoke.  A paired row at a late visit can therefore read
``superior`` on coverage rather than ``equivalent``; both pass.

**The scenario.**  ``survival`` (five integer visits; both arms' risks and their difference at
every visit), at n = 2,000 with 1,600 replications.  The property cells are in
:mod:`tests.studies.point_survival_curve_properties`.

Publication policy is ``reporting``.  The red-cell route was declared before any run.  A band
cell goes to ``band-finite-sample`` only if the shortfall-reading diagnostic of
``tests/unit/test_band_shortfall_reading.py`` reproduces for it, and otherwise to
``X14-finite-sample``.  A clustered cell goes to ``F28``.  A paired row whose reading names a
comparator convention above goes to ``X14-comparator-conventions``, which closes when a
re-declared paired cell passes.  Any other red cell goes to ``X14-finite-sample``, which
closes when a re-declared cell passes.  No re-run uses more replicates, wider margins or a new
law.  A replication that raises is never redrawn: a failed replication fails the run.  A walk
that stops short is data: the manifest counts it and the evidence page reports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import point_survival_common as common
from tests.studies.canonical_ltmle import regimen_rows
from tests.studies.canonical_point_survival import REFERENCE, REGIMENS, SCENARIOS
from tests.studies.canonical_point_survival_hazard import initial_estimates, primary_learners
from tests.studies.canonical_tmle import R_BASE_IMAGE, SL3_COMMIT, TMLE3_COMMIT
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
SEED = 20261071
RESAMPLING_SEED = 2026107101

#: The targeting settings of every one-step fit of this study.
ALGORITHM = "one_step"
STEP_SIZE = 1e-3
TOL = 1e-10
#: The settings of the part (a) cells, which target each horizon on its own.
PER_HORIZON_MAX_ITER = 100

#: ``tmle3``'s updater settings, which the runner transcribes.
TMLE3_MAXIT = 100
TMLE3_DELTA_EPSILON = 0.01
GLMNET_VERSION = "5.0"
GLMNET_TARBALL_SHA256 = "f0b23da6383a45d502cd138b64da6300422d9b8d7b2c4f14627d412cde79d2a0"

SCENARIO = "survival"
LAW = SCENARIOS[SCENARIO]
HORIZONS = tuple(range(1, LAW.n_times + 1))
LEVELS = tuple(common.risk_name(label, t) for t in HORIZONS for label in REGIMENS)
CONTRASTS = tuple(common.contrast_name("arm1", REFERENCE, t) for t in HORIZONS)
ESTIMANDS = {SCENARIO: LEVELS + CONTRASTS}
#: The exact truth of every reported level and contrast.
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

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        "survival_t5__correctly_specified",
        "survival_t5__shrunken_se_control",
        "survival_t5__noise_control",
        "competing_t4__correctly_specified",
        "rule_t3__correctly_specified",
        "policy_t3__correctly_specified",
        "weighted_t5__correctly_specified",
        "weighted_t5_per_horizon__correctly_specified",
        "isotonic_t2__correctly_specified",
    ),
    "isotonic_projection": ("isotonic_t2__moved",),
    "clustered_inference": (
        "clustered_t5__cluster_robust",
        "clustered_t5__iid_control",
        "clustered_t5_per_horizon__cluster_robust",
        "clustered_t5_per_horizon__iid_control",
    ),
    "double_robustness": tuple(
        f"survival_t5__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "targeting_necessity": ("survival_t5__targeted", "survival_t5__untargeted"),
    "root_n_and_efficiency": tuple(f"survival_t5__n_{size}" for size in (1_000, 2_000, 8_000)),
    "root_n_rate": ("survival_t5__empirical_sd", "survival_t5__reported_se"),
    "power": ("survival_t5__alternative",),
    "simultaneous_coverage": tuple(
        f"{label}__{kind}"
        for label in ("curve_arm1", "curve_arm0", "all_reported")
        for kind in ("simultaneous_band", "pointwise_joint_control")
    ),
}

STUDY = StudyRecord(
    name="point-treatment survival, one-step hazard targeting of the curve",
    slug="point-survival-curve",
    artifacts=ROOT / "tests" / "canonical" / "point_survival_curve",
    document="docs/technical-reference/method-evidence/point-survival-curve.md",
    anchor="point-survival-curve",
    scenarios=ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    reference="tmle3",
    modules=(
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
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/tmle3_survival/Dockerfile",
        "tests/canonical/tmle3_survival/run_study.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_point_survival_curve",
    properties_module="tests.studies.point_survival_curve_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

REFERENCE_METADATA = {
    "tmle3_commit": TMLE3_COMMIT,
    "sl3_commit": SL3_COMMIT,
    "glmnet_version": GLMNET_VERSION,
    "glmnet_tarball_sha256": GLMNET_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "tmle_survival per arm with target_times=1:5; tmle3_Update_survival with "
        "fit_method='l2', cvtmle=FALSE, convergence_type='sample_size', maxit=100, "
        "delta_epsilon=0.01, subclassed to repair the fit_lambda length test and the "
        "lambda.min.ratio retry; Lrnr_glm for A; Lrnr_stratified on t over Lrnr_glm for N "
        "and A_c; the contrast from the per-unit influence-curve difference"
    ),
    "pairing": (
        "read under the default margins; tmle3 takes ridge-penalized steps of at most "
        "delta_epsilon and stops at max |P_n D| <= 1/n"
    ),
}

CONFIGURATION = {
    "construction": "held point-treatment survival, one-step hazard targeting of the curve",
    "survival_model": "hazard",
    "targeting": ALGORITHM,
    "step_size": STEP_SIZE,
    "cross_fit": False,
    "outcome_learner": "unpenalized logistic regression on W1, W2, A per node",
    "treatment_learner": "unpenalized logistic regression on W1, W2",
    "censoring_learner": "unpenalized logistic regression on W1, W2, A per node",
    "regimens": list(REGIMENS),
    "reported_nodes": list(HORIZONS),
}


def fit(frame: pd.DataFrame) -> Any:
    """The subject's primary fit of one sample."""
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        n_folds=1,
        simultaneous=False,
        survival_model="hazard",
        targeting=ALGORITHM,
        step_size=STEP_SIZE,
        tol=TOL,
        random_state=0,
        **primary_learners(),
    ).fit(LAW.container(frame))


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return LAW.draw(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _replicate(
    payload: tuple[int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    replicate, n = payload
    frame, truth = draw_scenario(SCENARIO, n, replicate)
    result = fit(frame)
    sample = frame.loc[:, ["W1", "W2", "A", "time", "event"]].copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", SCENARIO)
    names = ESTIMANDS[SCENARIO]
    truth_rows = [
        {"scenario": SCENARIO, "replicate": replicate, "estimand": name, "truth": truth[name]}
        for name in names
    ]
    rows = regimen_rows(
        STUDY,
        result,
        truth,
        initial_estimates(result, names),
        names,
        SCENARIO,
        replicate,
        n=len(frame),
    )
    return sample, truth_rows, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every replication's sample, its truth rows and the subject's rows, in replicate order."""
    outcomes = map_parallel(
        _replicate, [((replicate, n),) for replicate in range(replicates)], n_jobs=n_jobs
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]


def truth_table() -> Mapping[str, Mapping[str, float]]:
    """The declared truth of every primary estimand."""
    return TRUTH


__all__ = ["STUDY", "draw_and_fit", "fit"]

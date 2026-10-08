"""Registered evidence for the per-arm outcome-adaptive C-TMLE, ``ctmle-oat-per-arm``.

The subject is ``CTMLE(strategy="oat")`` at its default ``oat_design="per_arm"``: for each arm
``a`` the treatment mechanism regresses ``1{A = a}`` on the one column ``Qbar_n(a, W)``.
Benkeser, Cai and van der Laan (2020), Theorem 1, give the influence curve of that
construction for one treatment-specific mean.  An indicator reduction per arm and a
fixed-dimension stack give the joint curve, and the contrasts follow by linearity and the
delta method.

**The laws** are those of :mod:`tests.studies.oat_per_arm_laws`: an arm-specific outcome index
and an instrument, so the per-arm design, the shared design and ``g_0`` have three different
limits.  Every cell of the study reads them.  The canonical OAT laws are not used, because
their outcome and propensity indices are nearly collinear and additive in the arm, so both
designs share one limit there.

**The comparator** is R ``drtmle`` 1.1.2 at ``538a3a2`` by a binary recode, through
``tests/canonical/drtmle_oat_per_arm/run_drtmle_oat_per_arm.R``.  For each arm ``a`` it fits
``drtmle(A = as.numeric(A == a), a_0 = c(1, 0), adapt_g = TRUE, Qn = list(qa, qa))`` with
``glm_g`` the same cubic in ``logit q`` as :func:`~tests.studies.oat_per_arm_laws.cubic_logit_learner`.
Level 1 is then exactly ``P(A = a | Qbar(a, W))``.  ``qa`` is this package's initial outcome
regression, so the outcome regression is pinned.  ``drtmle`` with ``a_0`` equal to every level
fits a sequential chain over the levels, which is not this design, so the recode is the
pairing.  ``drtmle`` returns no TMLE curve, so the runner rebuilds each arm's targeted
regression on its own one-dimensional submodel at the reported estimate, refuses the fit unless
that curve reproduces the reported variance to ``1e-8``, and forms each contrast from the
rebuilt curves.

**The primary sample size** is 1,500 on both laws.  The plan named 1,000 for the binary law;
the harness holds one ``n`` per study, and the three-arm law needs 1,500.

**What the Python rows also carry.**  ``shared_estimate`` is the ``oat_design="shared"`` fit on
the same sample, for the reported ``design_gap``.  ``rows_at_bound`` counts the rows whose
per-arm mechanism sits at either declared bound.  The laws keep the projection inside the
bounds, so the expected count is zero, and a nonzero count is reported, not dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import CTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import oat_per_arm_laws as laws
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.point_study_helpers import primary_rows

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 1_500
SEED = 20261017
RESAMPLING_SEED = 2026101701
IMPLEMENTATION = "cleverly-ctmle-oat-per-arm"
REFERENCE = "drtmle-r-oat-per-arm"
STRATIFY_FOLDS = "none"

BINARY_ESTIMANDS = ("ey0", "ey1", "ate", "rr", "or", "par", "paf")
THREE_ARM_ESTIMANDS = (
    "ey[0]",
    "ey[1]",
    "ey[2]",
    "ate[1 vs 0]",
    "ate[2 vs 0]",
    "rr[1 vs 0]",
    "rr[2 vs 0]",
    "or[1 vs 0]",
    "or[2 vs 0]",
)
#: Scenario -> the law it samples.
SCENARIO_LAWS: Mapping[str, laws.OatLaw] = {
    "binary_active": laws.BINARY_ACTIVE,
    "three_arm_active": laws.THREE_ARM_ACTIVE,
}

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": (
        "binary_active__correctly_specified",
        "binary_active__shrunken_se_control",
        "binary_active__noise_control",
        "three_arm_active__correctly_specified",
        "three_arm_active__shrunken_se_control",
        "three_arm_active__noise_control",
        "binary_weighted__correctly_specified",
        "binary_repeats__correctly_specified",
    ),
    "generated_design": (
        "binary_active__oracle_design",
        "binary_active__estimated",
        "three_arm_active__oracle_design",
        "three_arm_active__estimated",
    ),
    "type_i_error": ("sharp_null",),
    "power": ("alternative",),
    "root_n_and_efficiency": ("n_500", "n_2000", "n_8000"),
    "root_n_rate": ("empirical_sd", "reported_se"),
    "robustness_contract": ("outcome_correct", "outcome_wrong"),
    "crossfit_overfitting": ("cross_fitted_oat", "in_sample_control"),
    "simultaneous_coverage": (
        "three_arm_active__simultaneous_band",
        "three_arm_active__pointwise_joint_control",
    ),
}

#: The revert flag's cells, declared before the run.  If one of them is red and no defect is
#: found, the post-run commit sets ``OAT_PER_ARM_INFERENTIAL`` to ``False`` and the study
#: publishes under ``reporting``.  The primary coverage of every estimand of both scenarios
#: belongs to the list as well; :data:`REVERT_PRIMARY` names it.
REVERT_CELLS: tuple[tuple[str, str], ...] = (
    ("interval_calibration", "binary_active__correctly_specified"),
    ("interval_calibration", "three_arm_active__correctly_specified"),
    ("interval_calibration", "binary_weighted__correctly_specified"),
    ("interval_calibration", "binary_repeats__correctly_specified"),
    ("generated_design", "binary_active__oracle_design"),
    ("generated_design", "binary_active__estimated"),
    ("generated_design", "three_arm_active__oracle_design"),
    ("generated_design", "three_arm_active__estimated"),
    ("simultaneous_coverage", "three_arm_active__simultaneous_band"),
)
#: The primary coverage rows of the subject that belong to the revert list.
REVERT_PRIMARY = "coverage of every estimand of both scenarios, implementation " + IMPLEMENTATION

#: The owner of a red cell that reveals no defect, declared before the run.  The orchestrator
#: assigns its roadmap ID.
RED_CELL_OWNER = (
    "finite-sample limits of the per-arm outcome-adaptive C-TMLE: the listed red cells of "
    "ctmle-oat-per-arm"
)

STUDY = StudyRecord(
    name="outcome-adaptive per-arm C-TMLE",
    slug="ctmle-oat-per-arm",
    artifacts=ROOT / "tests" / "canonical" / "drtmle_oat_per_arm",
    document="docs/technical-reference/method-evidence/outcome-adaptive-per-arm-c-tmle.md",
    anchor="outcome-adaptive-per-arm-c-tmle",
    scenarios={"binary_active": BINARY_ESTIMANDS, "three_arm_active": THREE_ARM_ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    extra_artifacts=("design-gap.csv", "fit-diagnostics.csv"),
    modules=(
        "tests/studies/canonical_ctmle_oat_per_arm.py",
        "tests/studies/ctmle_oat_per_arm_properties.py",
        "tests/studies/oat_per_arm_laws.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/cvtmle_properties.py",
        "tests/studies/bounded_cv_laws.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/fractional_glm.py",
        "tests/conftest.py",
        "tests/studies/default_band_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/drtmle/Dockerfile",
        "tests/canonical/drtmle_oat_per_arm/run_drtmle_oat_per_arm.R",
        "tests/canonical/study_harness.R",
    ),
    runner_module="tests.studies.canonical_ctmle_oat_per_arm",
    properties_module="tests.studies.ctmle_oat_per_arm_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "per arm a: drtmle(W, A = as.numeric(A == a), Y, a_0 = c(1, 0), adapt_g = TRUE, "
        "Qn = list(qa, qa), glm_g = cubic in qlogis(pmin(pmax(Q1W, 1e-6), 1 - 1e-6)), "
        "cvFolds = 1, guard = NULL, targeted_se = TRUE, tolg = 0.01, maxIter = 100, "
        "tolIC = 1e-10); qa is this package's initial outcome regression; tmle$est[1] and "
        "tmle$cov[1, 1] give the arm mean and its variance; the contrasts are formed from the "
        "per-arm TMLE curves, rebuilt on the logistic submodel at the reported estimate and "
        "checked against tmle$cov[1, 1] to 1e-8"
    ),
    "rejected_candidates": (
        "drtmle with a_0 equal to every level: a sequential conditional-binary chain over "
        "the levels (R/estimate.R 337-371), not P(A = a | Qbar(a, W)); a single level a_0 = a "
        "enters the multi-level branch, whose glm_g sub-branch names the design columns "
        "wrongly (line 351)"
    ),
    "red_cell_owner": RED_CELL_OWNER,
    "revert_flag": "cleverly.estimators.ctmle.OAT_PER_ARM_INFERENTIAL",
    "revert_cells": [f"{family}/{cell}" for family, cell in REVERT_CELLS] + [REVERT_PRIMARY],
}

CONFIGURATION = {
    "strategy": "oat",
    "oat_design": "per_arm",
    "cross_fit": False,
    "simultaneous_intervals": False,
    "g_bounds": list(laws.G_BOUNDS),
    "stratify_folds": STRATIFY_FOLDS,
    "outcome_learner": "unpenalized logistic regression on [A indicators, W, A x W]",
    "treatment_learner": (
        "per arm, unpenalized logistic regression of 1{A = a} on (l, l^2, l^3), "
        "l = logit(clip(Qbar(a, W), 1e-6, 1 - 1e-6))"
    ),
    "laws": {name: law.name for name, law in SCENARIO_LAWS.items()},
    "failure_probe": (
        "before the declaration, 2,000 draws of every property cell, of the joint cell and "
        "of both primary scenarios raised no failure in the Python phase, and the R runner "
        "fitted 100 primary replicates of each law with no failure (points within 3.5e-7 of "
        "this package's, standard errors within 5.7e-9). A replicate that raises is not "
        "redrawn: the summary refuses a cell that lost one, and the R harness refuses a "
        "phase in which a replicate raised, as every R runner in the repository does"
    ),
    "measured_budget": (
        "single-process timing of a 20-replicate smoke: about 2.2 CPU-hours for the property "
        "study (0.94 for the four generated-design cells, 0.75 for the four calibration cells "
        "at R = 2,000), 0.06 for the primary Python "
        "phase (two fits per replicate), and under one CPU-hour for the R phase; the "
        "Python phase runs first, then the R phase"
    ),
    "primary_n": (
        "1,500 on both laws; the plan named 1,000 for the binary law, and the harness holds "
        "one n per study"
    ),
}


def fit_cleverly(
    frame: pd.DataFrame,
    scenario: str,
    *,
    design: str = "per_arm",
    simultaneous: bool = False,
) -> Any:
    """Fit this study's subject on one sample of ``scenario``'s law."""
    law = SCENARIO_LAWS[scenario]
    return (
        CTMLE(
            strategy="oat",
            oat_design=design,
            outcome_learner=laws.interaction_outcome(law.k),
            treatment_learner=(
                laws.cubic_logit_learner()
                if design == "per_arm"
                else laws.shared_cubic_logit_learner()
            ),
            cross_fit=False,
            estimands=(
                ("ey0", "ey1", "ate", "rr", "or", "par", "paf")
                if law.k == 2
                else ("ey", "ate", "rr", "or")
            ),
            reference="0" if law.k > 2 else None,
            simultaneous=simultaneous,
            g_bounds=laws.G_BOUNDS,
            stratify_folds=STRATIFY_FOLDS,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIO_LAWS:
        raise KeyError(scenario)
    frame, truth = SCENARIO_LAWS[scenario].sample(n, seed)
    return frame, {name: truth[name] for name in STUDY.scenarios[scenario]}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def initials(result: Any, scenario: str) -> dict[str, float]:
    """The untargeted plug-in of each estimand, as the R runner forms it from ``Qn``.

    The arm codes sort as the labels do on both laws, so position ``j`` is arm ``j``.
    """
    data = result.data
    means = [float(np.mean(result.nuisance.outcome.arms[code])) for code in data.arm_codes]
    observed = float(np.mean(data.outcome))

    def odds(p: float) -> float:
        return p / (1.0 - p)

    if scenario == "binary_active":
        return {
            "ey0": means[0],
            "ey1": means[1],
            "ate": means[1] - means[0],
            "rr": means[1] / means[0],
            "or": odds(means[1]) / odds(means[0]),
            "par": observed - means[0],
            "paf": 1.0 - means[0] / observed,
        }
    out = {f"ey[{j}]": value for j, value in enumerate(means)}
    for j in (1, 2):
        pair = f"{j} vs 0"
        out[f"ate[{pair}]"] = means[j] - means[0]
        out[f"rr[{pair}]"] = means[j] / means[0]
        out[f"or[{pair}]"] = odds(means[j]) / odds(means[0])
    return out


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    shared = fit_cleverly(frame, scenario, design="shared")
    data = result.data
    raw = np.asarray(result.nuisance.propensity.values, dtype=float)
    lower, upper = laws.G_BOUNDS
    at_bound = int(np.sum(np.any((raw <= lower) | (raw >= upper), axis=1)))
    sample = frame.loc[:, ["W1", "W2", "W3", "Y"]].copy()
    sample["A"] = [list(data.arm_codes).index(code) for code in data.treatment]
    for j in range(3):
        if j < len(data.arm_codes):
            code = data.arm_codes[j]
            sample[f"q{j}"] = np.asarray(result.nuisance.outcome.arms[code], dtype=float)
            sample[f"g{j}"] = raw[:, j]
        else:
            sample[f"q{j}"] = np.nan
            sample[f"g{j}"] = np.nan
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    rows = primary_rows(
        result=result,
        truth=truth,
        implementation=IMPLEMENTATION,
        scenario=scenario,
        replicate=replicate,
        estimands=STUDY.scenarios[scenario],
        initials=initials(result, scenario),
    )
    for row in rows:
        row["shared_estimate"] = float(shared[row["estimand"]].psi)
        row["rows_at_bound"] = at_bound
    return sample, truth_rows, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate,
        [
            ((scenario, replicate, n),)
            for scenario in SCENARIO_LAWS
            for replicate in range(replicates)
        ],
        n_jobs=n_jobs,
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]]


#: The columns the Python rows carry beside the published schema, read by
#: :func:`extra_artifacts`.
EXTRA_COLUMNS = ("shared_estimate", "rows_at_bound")
#: The column the R rows carry beside the published schema.
REFERENCE_COLUMNS = ("gn_max_abs_diff",)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The design gap of each estimand, and the per-replicate fit diagnostics.

    ``design-gap.csv`` publishes, per scenario and estimand, the mean of
    ``|psi_per_arm - psi_shared| / se`` over the primary replications: how far the shared design
    moves a point estimate, in this fit's own standard errors.  Its population counterpart is
    the law-level ``L2`` gap between the two limits.

    ``fit-diagnostics.csv`` publishes, per scenario and replication, the rows whose per-arm
    mechanism sits at a declared bound, and the largest gap between R ``drtmle``'s level-1
    mechanism and this package's, over every arm.
    """
    subject = rows.loc[rows["implementation"] == IMPLEMENTATION].copy()
    subject["design_gap"] = (subject["estimate"] - subject["shared_estimate"]).abs() / subject[
        "std_error"
    ]
    gap = (
        subject.groupby(["scenario", "estimand"], sort=True)["design_gap"]
        .agg(mean_design_gap="mean", max_design_gap="max")
        .reset_index()
    )
    reference = rows.loc[rows["implementation"] == REFERENCE]
    bound = subject.groupby(["scenario", "replicate"])["rows_at_bound"].first()
    mechanism = reference.groupby(["scenario", "replicate"])["gn_max_abs_diff"].max()
    diagnostics = pd.concat([bound, mechanism], axis=1).reset_index()
    return {"design-gap.csv": gap, "fit-diagnostics.csv": diagnostics}

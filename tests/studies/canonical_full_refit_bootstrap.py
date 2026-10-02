"""Registered evidence for the full-refit bootstrap and the derived post-fit contrasts.

No pinned comparator ships these outputs, so the study has no reference and every claim is a
repeated-sampling property against a known truth.  It measures four things.

* The log-scale interval of a ratio of two regimen levels, ``LongitudinalResult.ratio``, on
  the end-of-study law and on a four-node survival law.
* The interval of the RMST and of an RMST contrast, ``LongitudinalResult.rmst``, in sample and
  cross-fitted.
* The percentile interval of the ``LTMLE`` full-refit bootstrap, in sample, cross-fitted, on a
  survival curve, and with cluster resampling.
* The percentile interval of the point-treatment ``TMLE`` bootstrap, which no registered study
  measured before.

The primary scenario is the end-of-study ratio cell, because the framework publishes one
primary table per study.  Every other cell is an ``interval_calibration`` property cell in
:mod:`tests.studies.full_refit_bootstrap_properties`.

The red-cell policy, the failed-replicate cap and every number of the design are declared
here, before any run, and ``tests/unit/test_full_refit_bootstrap_cell_design.py`` pins them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as end_law
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import G_BOUNDS, declared_regimens
from tests.studies.evidence.properties import finite_support_sample
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

SEED = 20261002
PRIMARY_SCENARIO = "end_of_study_ratio"
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
RATIO_NAMES = ("rr_regimen[always vs never]", "or_regimen[always vs never]")

#: The bootstrap replicates inside each study replicate.
BOOTSTRAP_REPLICATES = 400

#: The pre-declared failed-replicate cap.  A bootstrap cell whose mean share of failed
#: bootstrap replicates exceeds it publishes its verdict as conditional on the surviving
#: replicates and claims no coverage, because dropping narrows the spread when failures
#: correlate with the estimate.
FAILED_REPLICATE_CAP = 0.01

#: The factor that shrinks a control's standard error, and a bootstrap control's percentile
#: interval about its median.  The shared ``calibration_controls`` value.
SHRUNKEN_SE_FACTOR = 0.70

#: One row per measured cell: label, the fit group it shares a sample and a fit with, law,
#: folds, n, replicates, bootstrap replicates, and whether a shrunken control is derived.
CELLS: tuple[dict[str, Any], ...] = (
    # Ratio calibration, end of study: the log risk and odds ratio of always versus never.
    {"label": "rr_end_of_study", "group": "ratio_end_of_study", "law": "end_of_study", "folds": 1, "n": 2_000, "replicates": 2_400, "bootstrap": 0, "control": True},
    {"label": "or_end_of_study", "group": "ratio_end_of_study", "law": "end_of_study", "folds": 1, "n": 2_000, "replicates": 2_400, "bootstrap": 0, "control": True},
    # Ratio calibration on the four-node survival law, at t = 4.
    {"label": "rr_survival", "group": "derived_survival", "law": "survival", "folds": 1, "n": 2_000, "replicates": 2_400, "bootstrap": 0, "control": True},
    {"label": "survival_rr_survival", "group": "derived_survival", "law": "survival", "folds": 1, "n": 2_000, "replicates": 2_400, "bootstrap": 0, "control": False},
    # RMST up to t = 5, and its contrast, in sample and cross-fitted.
    {"label": "rmst_survival", "group": "derived_survival", "law": "survival", "folds": 1, "n": 2_000, "replicates": 2_400, "bootstrap": 0, "control": False},
    {"label": "rmst_contrast_survival", "group": "derived_survival", "law": "survival", "folds": 1, "n": 2_000, "replicates": 2_400, "bootstrap": 0, "control": False},
    {"label": "rmst_crossfit", "group": "derived_crossfit", "law": "survival", "folds": 5, "n": 2_000, "replicates": 1_200, "bootstrap": 0, "control": False},
    {"label": "rmst_contrast_crossfit", "group": "derived_crossfit", "law": "survival", "folds": 5, "n": 2_000, "replicates": 1_200, "bootstrap": 0, "control": False},
    # The full-refit bootstrap percentile interval.
    {"label": "boot_ey_end_of_study", "group": "boot_end_of_study", "law": "end_of_study", "folds": 1, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": True},
    {"label": "boot_ate_end_of_study", "group": "boot_end_of_study", "law": "end_of_study", "folds": 1, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": True},
    {"label": "boot_ey_crossfit", "group": "boot_crossfit", "law": "end_of_study", "folds": 5, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
    {"label": "boot_ate_crossfit", "group": "boot_crossfit", "law": "end_of_study", "folds": 5, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
    {"label": "boot_risk2_survival", "group": "boot_survival", "law": "survival", "folds": 1, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
    {"label": "boot_risk4_survival", "group": "boot_survival", "law": "survival", "folds": 1, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
    {"label": "boot_rmst_survival", "group": "boot_survival", "law": "survival", "folds": 1, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
    {"label": "boot_ey_clustered", "group": "boot_clustered", "law": "clustered", "folds": 1, "n": 1_500, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
    {"label": "boot_ate_clustered", "group": "boot_clustered", "law": "clustered", "folds": 1, "n": 1_500, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
    {"label": "boot_ate_point_tmle", "group": "boot_point_tmle", "law": "point", "folds": 1, "n": 1_000, "replicates": 1_000, "bootstrap": BOOTSTRAP_REPLICATES, "control": False},
)  # fmt: skip

#: The clustered draw: 60 clusters of 25 rows sharing a latent effect modifier.
CLUSTERS = 60
CLUSTER_SIZE = 25
#: The shift a cluster's latent sign puts on the outcome probability, times
#: ``A1 + A2 - 1``.  Mean zero over the latent, so the truth is the end-of-study law's.
CLUSTER_SHIFT = 0.10

#: The red-cell policy, declared before the run.  The evidence page states the same rules.
RED_CELL_POLICY = (
    "1. The study policy is reporting. No budget, margin, law or learner changes after a "
    "verdict is seen.",
    "2. A red ratio or RMST cell is first checked against the exact-law tests of "
    "tests/unit/test_contrast_conveniences.py. A defect found there is fixed, and the study is "
    "regenerated once and declared in tests/canonical/provenance-revisions.md. A red cell the "
    "exact tests cannot explain is published red with owner X20-derived, and the ratio and "
    "rmst docstrings and the evidence page name the shortfall.",
    "3. A red longitudinal bootstrap cell sets LONGITUDINAL_BOOTSTRAP_INFERENTIAL to False, so "
    "the percentile interval ships as a diagnostic named through bootstrap_column, with owner "
    "X20-bootstrap.",
    "4. A red point-TMLE bootstrap cell is published under reporting with owner "
    "X20-point-bootstrap, and rule 3 applies to TMLEResult.",
    "5. A control that does not fail means the instrument cannot see the defect; the study "
    "then reports the control as underpowered by design and claims no positive verdict for "
    "its cell.",
)


def property_cells() -> dict[str, tuple[str, ...]]:
    """The cells the committed property summary must contain."""
    cells: list[str] = []
    for spec in CELLS:
        cells.append(f"{spec['label']}__correctly_specified")
        if spec["control"]:
            cells.append(f"{spec['label']}__shrunken_se_control")
    return {"interval_calibration": tuple(sorted(cells))}


STUDY = StudyRecord(
    name="full-refit bootstrap and derived contrasts",
    slug="full-refit-bootstrap-and-derived-contrasts",
    artifacts=ROOT / "tests" / "canonical" / "full_refit_bootstrap",
    document=(
        "docs/technical-reference/method-evidence/full-refit-bootstrap-and-derived-contrasts.md"
    ),
    anchor="full-refit-bootstrap-and-derived-contrasts",
    scenarios={PRIMARY_SCENARIO: RATIO_NAMES},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    #: Outcome, pseudo-outcome, treatment and censoring regressions.
    nuisance_count=4,
    margins=Margins(),
    implementation="cleverly-ltmle-derived",
    reference=None,
    modules=(
        "tests/studies/canonical_full_refit_bootstrap.py",
        "tests/studies/full_refit_bootstrap_properties.py",
        "tests/studies/survival_grid_law.py",
        "tests/studies/canonical_ltmle.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_full_refit_bootstrap",
    properties_module="tests.studies.full_refit_bootstrap_properties",
    property_cells=property_cells(),
    publication_policy="reporting",
)

CONFIGURATION = {
    "regimens": "never (reference) and always; the end-of-study law also fits treat_if_l2",
    "g_bounds": list(G_BOUNDS),
    "learners": (
        "saturated cell means on the end-of-study and point laws; on the four-node survival "
        "law, cell means over the columns each true conditional reads"
    ),
    "simultaneous_intervals": False,
    "fit_jobs": 1,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_interval": "two-sided percentile",
    "failed_replicate_cap": FAILED_REPLICATE_CAP,
    "shrunken_se_factor": SHRUNKEN_SE_FACTOR,
    "clustered_draw": {
        "clusters": CLUSTERS,
        "cluster_size": CLUSTER_SIZE,
        "latent_shift": CLUSTER_SHIFT,
    },
    "red_cell_policy": list(RED_CELL_POLICY),
}


def end_regimens() -> dict[str, Any]:
    """The end-of-study plans every fit on that law declares."""
    return declared_regimens(
        {key: end_law.REGIMEN_SPEC[key] for key in ("never", "always", "treat_if_l2")}
    )


def sample_end_of_study(n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows of the end-of-study law."""
    return finite_support_sample(
        end_law.PROBS,
        end_law.SUPPORT,
        n,
        seed,
        columns=("W", "A1", "C1", "L2", "A2", "C2", "Y"),
    )


def ratio_truths() -> dict[str, float]:
    """The risk and odds ratio of always versus never on the end-of-study law."""
    always = float(end_law.TRUTH["ey_regimen[always]"])
    never = float(end_law.TRUTH["ey_regimen[never]"])
    odds = (always / (1.0 - always)) / (never / (1.0 - never))
    return {RATIO_NAMES[0]: always / never, RATIO_NAMES[1]: odds}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary sample from an explicit seed, for the published-seed audit."""
    if scenario != PRIMARY_SCENARIO:
        raise KeyError(scenario)
    return sample_end_of_study(n, seed), ratio_truths()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def fit_end_of_study(frame: pd.DataFrame, *, n_folds: int = 1, **settings: Any) -> Any:
    """The end-of-study fit every cell on that law runs."""
    return LTMLE(
        end_regimens(),
        reference="never",
        outcome_learner=end_law.CellMeans(),
        pseudo_learner=end_law.CellMeans(),
        treatment_learner=end_law.CellMeans(),
        censoring_learner=end_law.CellMeans(),
        n_folds=n_folds,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        n_jobs=1,
        **settings,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
        **({"id": "cluster"} if "cluster" in frame else {}),
    )


def ratio_row(estimate: Any, truth: float, replicate: int, n: int) -> dict[str, Any]:
    """One primary row for a ratio, with inference on the log scale."""
    low, high = estimate.plugin_interval
    return {
        "implementation": STUDY.implementation,
        "scenario": PRIMARY_SCENARIO,
        "replicate": replicate,
        "n": n,
        "estimand": estimate.name,
        "truth": truth,
        "estimate": float(estimate.psi),
        "inference_estimate": float(estimate.log_psi),
        "std_error": float(estimate.plugin_std_error),
        "ci_lower": float(low),
        "ci_upper": float(high),
        "inference_scale": "log",
        "covered": int(low <= truth <= high),
        "initial_estimate": float("nan"),
    }


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Fit one primary sample and report both ratios."""
    if scenario != PRIMARY_SCENARIO:
        raise KeyError(scenario)
    result = fit_end_of_study(frame, random_state=0)
    a, b = "ey_regimen[always]", "ey_regimen[never]"
    estimates = (result.ratio(a, b), result.ratio(a, b, kind="or"))
    return [
        ratio_row(estimate, float(truth[estimate.name]), replicate, len(frame))
        for estimate in estimates
    ]


def _replicate(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, n = payload
    frame, truth = draw_scenario(PRIMARY_SCENARIO, n, replicate)
    return cleverly_rows(frame, truth, PRIMARY_SCENARIO, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Draw and fit every primary replication."""
    payloads = [((replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for rows in outcomes for row in rows])
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def groups() -> dict[str, tuple[dict[str, Any], ...]]:
    """The fit groups: the cells that share one sample and one fit per replicate."""
    out: dict[str, list[dict[str, Any]]] = {}
    for spec in CELLS:
        out.setdefault(spec["group"], []).append(spec)
    for name, specs in out.items():
        shared = {
            (spec["law"], spec["folds"], spec["n"], spec["replicates"], spec["bootstrap"])
            for spec in specs
        }
        if len(shared) != 1:
            raise ValueError(f"the cells of group {name!r} do not share one design")
    return {name: tuple(specs) for name, specs in out.items()}


def budget_core_seconds(timings: Mapping[str, float]) -> dict[str, float]:
    """Extrapolate per-replicate timings to the declared replicate counts.

    Parameters
    ----------
    timings : Mapping of str to float
        Seconds per study replicate of each fit group, measured by a smoke pass.

    Returns
    -------
    dict of str to float
        Core-seconds per group at the declared replicate count.
    """
    return {name: timings[name] * specs[0]["replicates"] for name, specs in groups().items()}


def _assert_cluster_witness() -> None:
    """The latent shift is nonzero and averages to zero, so it clusters without biasing."""
    if not CLUSTER_SHIFT > 0:
        raise AssertionError("the clustered draw needs a nonzero latent shift")
    if np.max(end_law.Q) + CLUSTER_SHIFT >= 1 or np.min(end_law.Q) - CLUSTER_SHIFT <= 0:
        raise AssertionError("the latent shift leaves the unit interval")


_assert_cluster_witness()

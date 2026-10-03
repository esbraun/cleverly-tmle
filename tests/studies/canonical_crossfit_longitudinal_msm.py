"""Registered evidence for the cross-fitted longitudinal MSM projection.

The subject is the ``longitudinal-msm`` subject at five outer folds: quasi-binomial node
regressions, the law's own treatment and censoring mechanism, the four regimens ``never``,
``always``, ``early`` and the dynamic rule, the ``(intercept)`` and ``duration`` terms, and
the declared projection weights.  Each fold runs the untargeted recursion of every regimen on
its training rows, and one stacked fluctuation per node targets every follower of every
regimen.  The law is ``make_longitudinal(censoring=True)``, and the truth is the fixed
projection of its quadrature regimen means, exactly as in ``longitudinal-msm``.

**Primary.** 800 replications, paired with R ``lmtp`` 1.5.4 through ``lmtp_tmle_with_folds``
with the realized fold column and the exact per-node density ratios.  ``lmtp`` has no MSM, so
the reference is the fixed projection of four ``lmtp`` regimen fits and their joint per-unit
influence curves.  The design has no effect modifier, so that projection is the same estimand.
The constructions differ twice: ``lmtp`` fluctuates on each training fold, and it targets each
regimen with its own scalar fluctuation where this package solves one stacked fluctuation.
Upstream commit ``9996b04`` classifies the 1.5.4 training-fold update as a bug, so the paired
verdicts validate neither fold-local update.

**Properties.** ``tests.studies.crossfit_longitudinal_msm_properties`` declares every family.

Publication policy is ``gated``.  The red-cell route was declared before any run: a red cell is
diagnosed for a defect first.  A defect found is fixed and the study regenerated once, with the
revision declared in ``tests/canonical/provenance-revisions.md``.  With none found, there is no
re-run: the run is published under ``publication_policy="reporting"`` with the owner row
"finite-sample limits of the cross-fitted longitudinal MSM projection: the listed red cells of
``cross-fitted-longitudinal-msm``", whose roadmap ID the orchestrator assigns then.  A
replication that raises is never redrawn: the run stops, the failing cell is recorded and
dropped with its gap stated as a page limit, and the run repeats without it with no other
change.  No margin, budget, law, learner, fold count or seed changes after a result is seen.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_longitudinal_msm import (
    COLUMNS,
    DURATION,
    ESTIMANDS,
    G_BOUNDS,
    PROJECTION_WEIGHT,
    REGIMENS,
    TERMS,
    declared_msm,
    initial_beta,
    project_means,
)
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism, QuasiBinomialGLM
from tests.studies.canonical_ltmle_crossfit import REFERENCE_METADATA as LMTP_METADATA
from tests.studies.evidence.constructions import (
    POOLED_LONGITUDINAL_CROSS_FIT,
    TRAINING_FOLD_FLUCTUATION,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 800
PRIMARY_N = 2_500
SEED = 20262701
RESAMPLING_SEED = 20262702
SCENARIO = "cross_fitted_regimen_projection"
N_FOLDS = 5
LEARNER_FOLDS = 2
RANDOM_STATE = 0
REFERENCE = "lmtp projected regimen fits"

#: The joint label of the band cell.
BAND_LABEL = "cross_fitted_msm"

#: Every published property cell, by family.  ``crossfit_longitudinal_msm_properties``
#: declares the same set with the law, the size and the stream of each cell.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": tuple(
        f"{term}__{configuration}"
        for term in TERMS
        for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    "root_n_and_efficiency": tuple(
        f"{term}__n_{size}" for term in TERMS for size in (500, 2000, 8000)
    ),
    "root_n_rate": tuple(
        f"{term}__{statistic}" for term in TERMS for statistic in ("empirical_sd", "reported_se")
    ),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in (*TERMS, "duration_logit")
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "type_i_error": ("duration__sharp_null",),
    "power": ("duration__alternative",),
    "targeting_necessity": ("duration__targeted", "duration__untargeted"),
    "projection_necessity": ("duration__declared_weights", "duration__uniform_weights"),
    "crossfit_overfitting": ("cross_fitted_msm", "in_sample_control"),
    "in_sample_agreement": tuple(f"{term}__in_sample_agreement" for term in TERMS),
    "simultaneous_coverage": (
        f"{BAND_LABEL}__simultaneous_band",
        f"{BAND_LABEL}__pointwise_joint_control",
    ),
}

STUDY = StudyRecord(
    name="cross-fitted longitudinal MSM projection",
    slug="cross-fitted-longitudinal-msm",
    artifacts=ROOT / "tests" / "canonical" / "lmtp_ltmle_msm",
    document="docs/technical-reference/method-evidence/cross-fitted-longitudinal-msm-projection.md",
    anchor="cross-fitted-longitudinal-msm-projection",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-cross-fitted-ltmle-msm",
    reference=REFERENCE,
    modules=(
        "tests/studies/canonical_crossfit_longitudinal_msm.py",
        "tests/studies/crossfit_longitudinal_msm_properties.py",
        "tests/studies/canonical_longitudinal_msm.py",
        "tests/studies/longitudinal_msm_properties.py",
        "tests/studies/ltmle_crossfit_properties.py",
        "tests/studies/ltmle_properties.py",
        "tests/studies/canonical_ltmle_crossfit.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/discrete_law_longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_ltmle_msm/run_study.R",
    ),
    runner_module="tests.studies.canonical_crossfit_longitudinal_msm",
    properties_module="tests.studies.crossfit_longitudinal_msm_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": POOLED_LONGITUDINAL_CROSS_FIT,
    "reference_construction": TRAINING_FOLD_FLUCTUATION,
    "reference_targeting": "one scalar fluctuation per regimen, projected afterwards",
    "outcome_kind": "end_of_study",
    "link": "identity",
    "cross_fit": True,
    "outer_folds": N_FOLDS,
    "folds": "random_partition from n and random_state=0; R lmtp receives the realized assignment",
    "learner_folds": LEARNER_FOLDS,
    "simultaneous_intervals": False,
    "variance_method": "ic",
    "g_bounds": list(G_BOUNDS),
    "regimens": list(REGIMENS),
    "terms": list(TERMS),
    "duration": DURATION,
    "projection_weights": PROJECTION_WEIGHT,
    "outcome_designs": [["W1", "W2"], ["W1", "W2", "L2"]],
    "mechanism": "supplied_from_the_law_to_both",
    "reference_density_ratios": "exact_per_node",
}

#: Provenance of the comparator, for the manifest's ``generated_with.reference`` block.
REFERENCE_METADATA = {
    **LMTP_METADATA,
    "reference_parameter": "fixed projection of correlated lmtp regimen estimates and EIFs",
}


def subject(*, n_folds: int = N_FOLDS, simultaneous: bool = False) -> LTMLE:
    """The ``longitudinal-msm`` subject at ``n_folds`` outer folds."""
    return LTMLE(
        REGIMENS,
        msm=declared_msm(),
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=QuasiBinomialGLM(),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=n_folds,
        learner_folds=LEARNER_FOLDS,
        g_bounds=G_BOUNDS,
        simultaneous=simultaneous,
        max_iter=100,
        tol=1e-10,
        random_state=RANDOM_STATE,
    )


def fit_cleverly(frame: pd.DataFrame, *, simultaneous: bool = False) -> Any:
    """The cross-fitted projection of one panel."""
    return subject(simultaneous=simultaneous).fit(frame, **COLUMNS)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One censored panel and the projection of its quadrature regimen means."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    frame, source_truth = make_longitudinal(n=n, seed=seed, censoring=True, backend="pandas")
    means = {label: float(source_truth[f"ey_regimen[{label}]"]) for label in REGIMENS}
    return frame, dict(zip(ESTIMANDS, project_means(means), strict=True))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _rows_from_result(
    result: Any, reference: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    initials = initial_beta(result)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(ESTIMANDS):
        estimate = result[name]
        low, high = estimate.ci
        target = float(reference[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": target,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= target <= high),
                "initial_estimate": float(initials[index]),
            }
        )
    return rows


def cleverly_rows(
    frame: pd.DataFrame, reference: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return _rows_from_result(fit_cleverly(frame), reference, scenario, replicate, len(frame))


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame)
    sample = frame.copy()
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference.items()
    ]
    return sample, truths, _rows_from_result(result, reference, scenario, replicate, n)


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw, fit, and keep each sample with its realized folds for R ``lmtp``."""
    payloads = [((SCENARIO, replicate, n),) for replicate in range(replicates)]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

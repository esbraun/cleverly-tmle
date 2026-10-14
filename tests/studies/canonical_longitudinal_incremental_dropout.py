r"""Registered evidence for IPSI under dropout and on survival outcomes (S2).

The subject is :class:`~cleverly.longitudinal.IPSI` with a censoring mechanism, the one-step
estimator of Kim et al. (2021, Algorithm 1 and Theorem 5.1).  No maintained package ships
that construction with dropout, so the study has no comparator (``X19-plan.md`` 1.5).  The
laws are ``C`` and ``D`` of :mod:`tests.discrete_law_longitudinal_incremental`: law ``A``
with dropout after each arm, and a three-node survival law with the same dropout.  The plans
tilt every node by :math:`\delta \in \{0.5, 1, 2\}`.  Two primary scenarios, both with five
folds:

======================  =====================================================================
scenario                law and estimands
======================  =====================================================================
``dropout_folds_5``     ``C``; the three end-of-study means and their two contrasts
``survival_folds_5``    ``D``; the risk of every plan by horizons 1, 2 and 3, and the two
                        contrasts at each horizon
======================  =====================================================================

**One fold** (a deviation from ``X19-plan.md`` 3.5, declared before any run).  The plan put
one-fold scenarios in the primary rows.  A one-fold fit with saturated cell means is the
plug-in g-formula on the empirical cell frequencies, and its tilt is nonlinear in
:math:`\hat\pi`.  So it carries a bias of order (cells)/n.  A probe of 400 draws of law ``C``
at n = 8,000 measured a standardized bias of +0.24 and -0.21 on the two contrasts (Monte Carlo
SE 0.05).  The one-fold fits are therefore the reported family ``in_sample_fit`` of the
properties module, and they state no verdict.

**Size.**  :data:`PRIMARY_N` is 24,000, not the plan's 2,000.  At 2,000 the fallback-read
bound of :mod:`tests.studies.longitudinal_incremental_common` exceeds 0.1 standard errors on
law ``D`` at five folds (``X19-plan.md`` 6.2 raises a cell's size until the bound holds).  The
five-fold contrasts also carry a second-order bias of order 1/n.  Probes of 400 draws of law
``C`` measured a standardized bias of 0.44 at 2,000, 0.22 at 8,000 and 0.05 at 32,000 (Monte
Carlo SE 0.05), against the primary rows' gate of 0.25.  The size matches
:mod:`tests.studies.canonical_longitudinal_incremental`.

**Properties.** :mod:`tests.studies.longitudinal_incremental_dropout_properties` declares every
family.

**Red-cell route** (declared before any run).  A red primary truth row blocks the merge and
gets a diagnosis.  A red property cell that reads as a finite-sample limit is re-registered
``reporting`` with an owner row before any re-run.  A replication that raises is never
redrawn: the failing cell drops to its owner with its failure count stated as a page limit,
and the run repeats without it.  No budget, margin, law, learner, fold count or seed moves
after a verdict is seen.  Controls carry the verdict their family states.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_incremental as law
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_incremental_common as common
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 24_000
SEED = 20261242
RESAMPLING_SEED = 2026124201

REFERENCE = law.REFERENCE
PLANS = common.constant_plans(law.DELTAS)
HORIZONS = (1, 2, 3)
#: ``scenario -> (law, horizons, fold count)``.
SCENARIOS: dict[str, tuple[Any, tuple[int, ...] | None, int]] = {
    "dropout_folds_5": (law.LAW_C, None, 5),
    "survival_folds_5": (law.LAW_D, HORIZONS, 5),
}
ESTIMANDS = {
    scenario: common.names(PLANS, REFERENCE, horizons=horizons)
    for scenario, (_, horizons, _) in SCENARIOS.items()
}
TRUTH = {
    scenario: common.truths(target, PLANS, ESTIMANDS[scenario])
    for scenario, (target, _, _) in SCENARIOS.items()
}
EFFICIENCY_SD = {
    name: common.eif_sd(target, PLANS, name)
    for scenario, (target, _, _) in SCENARIOS.items()
    for name in ESTIMANDS[scenario]
}

#: Every published property cell, by family (see the properties module).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in (
            "dropout_contrast_x2",
            "survival_contrast_x2_h3",
            "competing_contrast_x2_h3",
            "msm_log_delta",
            "weighted_contrast_x0_5",
        )
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "simultaneous_coverage": (
        "survival_grid__simultaneous_band",
        "survival_grid__pointwise_joint_control",
    ),
    "clustered_inference": ("mean_x2__cluster_robust", "mean_x2__iid_control"),
    "censoring_complete_case": ("mean_x2__complete_case",),
    "in_sample_fit": ("dropout_contrast_x2__one_fold", "survival_contrast_x2_h3__one_fold"),
}

STUDY = StudyRecord(
    name="longitudinal incremental propensity-score interventions under dropout",
    slug="longitudinal-incremental-dropout",
    artifacts=ROOT / "tests" / "canonical" / "longitudinal_incremental_dropout",
    document="docs/technical-reference/method-evidence/longitudinal-incremental-interventions.md",
    anchor="longitudinal-incremental-propensity-score-interventions-under-dropout",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-ipsi-dropout",
    reference=None,
    efficiency_bounds=EFFICIENCY_SD,
    modules=(
        "tests/studies/canonical_longitudinal_incremental_dropout.py",
        "tests/studies/longitudinal_incremental_dropout_properties.py",
        "tests/studies/longitudinal_incremental_common.py",
        "tests/discrete_law_longitudinal_incremental.py",
        "tests/discrete_law_longitudinal.py",
        "tests/longitudinal_ipsi.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_longitudinal_incremental_dropout",
    properties_module="tests.studies.longitudinal_incremental_dropout_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "construction": "IPSI one-step with dropout weights, training-fold recursion (Kim 2021)",
    "n_folds": {scenario: folds for scenario, (_, _, folds) in SCENARIOS.items()},
    "horizons": list(HORIZONS),
    "learners": "CellMeans at every slot",
    "g_bounds": list(common.G_BOUNDS),
    "q_bounds": list(common.Q_BOUNDS),
    "deltas": dict(law.DELTAS),
    "reference": REFERENCE,
}


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The one fit of each primary scenario."""
    target, horizons, folds = SCENARIOS[scenario]
    return common.fit(frame, target, PLANS, n_folds=folds, reference=REFERENCE, horizons=horizons)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    target, _, _ = SCENARIOS[scenario]
    return law.sample(target, n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per estimand.  ``initial_estimate`` is the uncorrected first-node plug-in."""
    rows = []
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": common.plug_in(result, name),
            }
        )
    return rows


def _replicate(scenario: str, replicate: int, n: int) -> list[dict[str, Any]]:
    frame, truth = draw_scenario(scenario, n, replicate)
    return result_rows(fit(frame, scenario), truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Every primary row.  No comparator reads the samples, so none is kept."""
    payloads = [
        (scenario, replicate, n) for scenario in SCENARIOS for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, list(REPLICATE_COLUMNS)]

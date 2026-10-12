r"""Registered evidence for the longitudinal IPSI estimator (``longitudinal-incremental``, S1).

The subject is :class:`~cleverly.longitudinal.IPSI`, the one-step estimator of Kennedy (2019,
Algorithm 1), paired with R ``npcausal`` 0.1.0's ``ipsi`` at commit ``56a5ac11``.  The law is
law ``A`` of :mod:`tests.discrete_law_longitudinal_incremental`: three nodes, binary
:math:`L_2, L_3`, and a continuous end-of-study outcome.  The plans tilt every node by
:math:`\delta \in \{0.5, 1, 2\}`.  Two primary scenarios:

======================  =====================================================================
scenario                subject and pairing
======================  =====================================================================
``ipsi_folds_2``        two folds; ``npcausal`` reads the same fold assignment
``ipsi_folds_5``        five folds; the same
======================  =====================================================================

**Exact pairing** (``X19-plan.md`` 3.4, 6.1 and 6.8).  Both sides fit saturated cell means:
:class:`~tests.discrete_law_longitudinal.CellMeans` here, ``SL.cellmeans`` in
``tests/canonical/npcausal_ipsi_adapter.R``.  ``npcausal`` fits one treatment model over every
``(id, time)`` row, so its treatment design is the time block and the history cell, and its
fallback reads the training mean of the same time block.  Its fold
draw is replaced by the Python assignment, and nothing else in the pinned body moves.  With
equal cell partitions and equal fallbacks the two recursions are the same function of the
sample.  The pairing gates G1 and G2 (``tests/canonical/npcausal_longitudinal_incremental/
gates.py``) assert agreement below :data:`PAIRING_GATE` on the estimate, every per-unit
influence value, and the standard error.  Both sides report the influence-curve standard
deviation over :math:`\sqrt n`, so no factor separates the two errors (``X19-plan.md`` 6.8
assumed one).  The paired rows then read the default ``Margins()``.

**Size.**  :data:`PRIMARY_N` is 5,000, not the plan's 2,000: at two folds the fallback-read
bound of :mod:`tests.studies.longitudinal_incremental_common` is 1.48 standard errors at
2,000 and 0.053 at 5,000 (``X19-plan.md`` 6.2 raises a cell's size until the bound holds).

**Properties.** :mod:`tests.studies.longitudinal_incremental_properties` declares every family.

**Red-cell route** (declared before any run).  A red primary truth row or a red paired row
blocks the merge and gets a diagnosis.  A red property cell that reads as a finite-sample limit
is re-registered ``reporting`` with an owner row before any re-run.  A replication that raises
is never redrawn: the failing cell drops to its owner with its failure count stated as a page
limit, and the run repeats without it.  No budget, margin, law, learner, fold count or seed
moves after a verdict is seen.  Controls carry the verdict their family states.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_incremental as law
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_incremental_common as common
from tests.studies.canonical_incremental_interventions import NPCAUSAL_COMMIT, R_BASE_IMAGE
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 5_000
SEED = 20261241
RESAMPLING_SEED = 2026124101
#: The pre-declaration gate on every paired quantity (G1 and G2).
PAIRING_GATE = 1e-10

LAW = law.LAW_A
REFERENCE = law.REFERENCE
PLANS = common.constant_plans(law.DELTAS)
ESTIMANDS = common.names(PLANS, REFERENCE)
FOLDS = {"ipsi_folds_2": 2, "ipsi_folds_5": 5}
TRUTH = common.truths(LAW, PLANS, ESTIMANDS)
EFFICIENCY_SD = {name: common.eif_sd(LAW, PLANS, name) for name in ESTIMANDS}

#: Every published property cell, by family (see the properties module).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in (
            "mean_x2",
            "contrast_x2",
            "contrast_x2_in_sample",
            "known_x2",
            "no_positivity",
            "per_node",
            "tilt_then_static",
        )
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "root_n_and_efficiency": tuple(f"contrast_x2__n_{size}" for size in (1_000, 4_000, 16_000)),
    "root_n_rate": ("contrast_x2__empirical_sd", "contrast_x2__reported_se"),
    "type_i_error": ("contrast_x2__sharp_null",),
    "power": ("contrast_x1_5__alternative",),
    "simultaneous_coverage": (
        "nine_tilts__simultaneous_band",
        "nine_tilts__pointwise_joint_control",
    ),
    "mechanism_term_necessity": ("contrast_x2__full_eif", "contrast_x2__mechanism_term_dropped"),
    "mechanism_requirement": (
        "contrast_x2__mechanism_correct__n_2000",
        "contrast_x2__mechanism_correct__n_8000",
        "contrast_x2__mechanism_constant",
    ),
    "targeting_necessity": ("contrast_x2__one_step", "contrast_x2__plug_in"),
}

STUDY = StudyRecord(
    name="longitudinal incremental propensity-score interventions",
    slug="longitudinal-incremental",
    artifacts=ROOT / "tests" / "canonical" / "npcausal_longitudinal_incremental",
    document="docs/technical-reference/method-evidence/longitudinal-incremental-interventions.md",
    anchor="longitudinal-incremental-interventions",
    scenarios=dict.fromkeys(FOLDS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-ipsi",
    reference="npcausal",
    efficiency_bounds=EFFICIENCY_SD,
    modules=(
        "tests/studies/canonical_longitudinal_incremental.py",
        "tests/studies/longitudinal_incremental_properties.py",
        "tests/studies/longitudinal_incremental_common.py",
        "tests/studies/canonical_incremental_interventions.py",
        "tests/discrete_law_longitudinal_incremental.py",
        "tests/discrete_law_longitudinal.py",
        "tests/longitudinal_ipsi.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
        "tests/canonical/npcausal_incremental/Dockerfile",
        "tests/canonical/npcausal_ipsi_adapter.R",
        "tests/canonical/npcausal_longitudinal_incremental/run_study.R",
    ),
    runner_module="tests.studies.canonical_longitudinal_incremental",
    properties_module="tests.studies.longitudinal_incremental_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

REFERENCE_METADATA = {
    "npcausal_commit": NPCAUSAL_COMMIT,
    "r_base_image": R_BASE_IMAGE,
    "reference_parameter": "longitudinal incremental odds curve, end-of-study outcome",
    "reference_construction": (
        "npcausal ipsi with fit = 'sl', one call over the full delta grid per fold mode, its "
        "fold draw replaced by the Python assignment; SL.cellmeans outcome fits and "
        "time-blocked treatment fits; returned ifvals un-recycled"
    ),
    "reference_adapter": "tests/canonical/npcausal_ipsi_adapter.R",
}

CONFIGURATION = {
    "construction": "IPSI one-step, training-fold recursion, pooled mean (Kennedy 2019)",
    "reference_construction": "npcausal ipsi with injected folds",
    "n_folds": dict(FOLDS),
    "learners": "CellMeans at every slot",
    "g_bounds": list(common.G_BOUNDS),
    "q_bounds": list(common.Q_BOUNDS),
    "deltas": dict(law.DELTAS),
    "reference": REFERENCE,
    "pairing_gate": PAIRING_GATE,
}

#: The wide columns the comparator reads, in node order.
SAMPLE_COLUMNS = ("W", "A1", "L2", "A2", "L3", "A3", "Y")


def fit(frame: pd.DataFrame, scenario: str, *, n_folds: int | None = None) -> Any:
    """The one fit of each primary scenario."""
    folds = FOLDS[scenario] if n_folds is None else n_folds
    return common.fit(frame, LAW, PLANS, n_folds=folds, reference=REFERENCE)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in FOLDS:
        raise KeyError(scenario)
    return law.sample(LAW, n, seed), dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per estimand.  ``initial_estimate`` is the uncorrected first-node plug-in."""
    rows = []
    for name in ESTIMANDS:
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


def paired_sample(frame: pd.DataFrame, result: Any, scenario: str, replicate: int) -> pd.DataFrame:
    """The replicate as the comparator reads it: the wide nodes, the row and its fold."""
    sample = frame.loc[:, list(SAMPLE_COLUMNS)].copy()
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", np.asarray(result.folds.assignment, dtype=int))
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    return sample


def _replicate(
    scenario: str, replicate: int, n: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit(frame, scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return (
        paired_sample(frame, result, scenario, replicate),
        truths,
        result_rows(result, truth, scenario, replicate),
    )


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Replicate-major, so both scenarios of one replicate are contiguous in the sample archive
    # and the comparator's runner reads them as one group.
    payloads = [(scenario, replicate, n) for replicate in range(replicates) for scenario in FOLDS]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truths, estimates.loc[:, list(REPLICATE_COLUMNS)]

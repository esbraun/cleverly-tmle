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

The plan's unpaired one-fold row is not a primary scenario here.  ``npcausal`` 0.1.0 cannot
fit one fold: with ``nsplits = 1`` its training rows are ``slong != 1``, which is empty.  A
primary scenario is paired, so the one-fold fit is the gated calibration cell
``contrast_x2_in_sample`` of the properties module instead.

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

**Size** (a deviation from ``X19-plan.md`` 3.5, declared before any run).  The cross-fitted
contrasts carry a second-order bias of order 1/n, so their standardized bias falls as
:math:`1/\sqrt n`.  The bias rule passes a row only when the 99% interval of its mean error
lies inside 0.25 empirical standard deviations, which at 1,000 replications needs a
standardized bias below about 0.17.  The sizing probe (``tests/diagnostics/x19_sizing/``)
measured 0.27 at n = 8,000 on the five-fold contrasts and 0.35 on the two-fold ones.
:data:`PRIMARY_N` gives every row of both scenarios a pass probability of at least 0.95 at
the projected bias, with one Monte Carlo SE added
(``tests/unit/test_longitudinal_incremental_design.py``).  The two-fold rows set it: they
need about 90,000.  The fallback-read bound holds far below this size.

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

from tests import discrete_law_longitudinal_incremental as law
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_incremental_common as common
from tests.studies.canonical_incremental_interventions import NPCAUSAL_COMMIT, R_BASE_IMAGE
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 96_000
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
    "root_n_and_efficiency": tuple(f"contrast_x2__n_{size}" for size in (24_000, 64_000, 96_000)),
    "root_n_rate": ("contrast_x2__empirical_sd", "contrast_x2__reported_se"),
    "type_i_error": ("contrast_x2__sharp_null",),
    "power": ("contrast_x1_5__alternative",),
    "simultaneous_coverage": (
        "nine_tilts__simultaneous_band",
        "nine_tilts__pointwise_joint_control",
    ),
    "mechanism_term_necessity": ("contrast_x2__full_eif", "contrast_x2__mechanism_term_dropped"),
    "single_robustness": (
        "contrast_x2__mechanism_correct__n_48000",
        "contrast_x2__mechanism_correct__n_96000",
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
    pending_reference="npcausal",
    pending_reason=(
        "the npcausal R phase runs on the local study queue (TRACKER item 9a) and registers "
        "in a later change; the owner's rule merges the method on its Python evidence"
    ),
    failure_bound={"cleverly-ipsi": 0.01, "npcausal": 0.01},
    comparison="matching implementation",
    efficiency_bounds=EFFICIENCY_SD,
    modules=(
        "tests/studies/canonical_longitudinal_incremental.py",
        "tests/studies/longitudinal_incremental_properties.py",
        "tests/studies/longitudinal_incremental_common.py",
        "tests/studies/canonical_incremental_interventions.py",
        "tests/discrete_law_longitudinal_incremental.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_competing.py",
        "tests/discrete_law_longitudinal_ipsi.py",
        "tests/discrete_law_survival.py",
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
    # The driver holds every replicate's sample until it writes the archive: 2,000 samples of
    # 96,000 rows.  Compact types keep that near 5 GB, and the concatenation peak near 9 GB.  Every column but ``Y``
    # is a binary node, and the CSV text R reads is the same number either way.
    sample = frame.loc[:, list(SAMPLE_COLUMNS)].astype(dict.fromkeys(SAMPLE_COLUMNS[:-1], np.int8))
    sample.insert(0, "row", np.arange(len(sample), dtype=np.int32))
    sample.insert(0, "fold", np.asarray(result.folds.assignment, dtype=np.int8))
    sample.insert(0, "replicate", np.full(len(sample), replicate, dtype=np.int32))
    sample.insert(0, "scenario", pd.Categorical([scenario] * len(sample), categories=list(FOLDS)))
    return sample


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
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


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    # Replicate-major, so both scenarios of one replicate are contiguous in the sample archive
    # and the comparator's runner reads them as one group.  Dependent: the sample carries the
    # fold assignment of the cleverly fit.
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in FOLDS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )

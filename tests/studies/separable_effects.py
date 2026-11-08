r"""Registered evidence for separable effects on hazard ``LTMLE`` (X91 part D).

The subject is ``Separable(arms=..., first="death")`` beside static arms on
``LTMLE(survival_model="hazard", targeting="one_step")``: the incidence of relapse and of death
when the part of the treatment acting on relapse and the part acting on death take different
levels, Stensrud et al. (2022) identification (8) with censoring (10).  The fit targets the
competing hazard and the event hazard among the competing-free apart, with one coefficient per
factor and target.  The law and every truth are those of
:mod:`tests.studies.separable_effects_law`.

=====================  ====================================================================
scenario               law, engine and estimands
=====================  ====================================================================
``in_sample``          n = 16,000, one fold; ``a0``, ``a1``, ``sep10`` and ``sep01``; the
                       incidence of both causes at ``t = 1, 2, 3`` and the contrasts
                       against ``a0``
``cross_fitted``       the same law, size and estimands with two outer folds, so the
                       targeting starts away from the truth and the walk hands over to
                       Newton steps
=====================  ====================================================================

Every nuisance is a saturated cell mean (``CellMeans``, ``CellProbabilities``), the
nonparametric model on this support.  The static arms of a separable fit go through the
factor-wise engine, a different TMLE from the shipped static one, so their rows are measured
here too.

**Scope** (declared before any run).  The record measures hazard ``LTMLE`` with saturated
learners at one and two folds.  Robustness to a wrong hazard factor is measured by the
properties module on the engine with the exact mechanism.

**Size** (declared before any run).  16,000, as the held-baseline scenario of
``eliminated-competing-events`` on the same law, where n = 4,000 left Kish sizes of 29 to 60 at
``t = 3``.  :data:`PRIMARY_REPLICATES` is the shared rule
:func:`~tests.studies.repeated_sizing.primary_replicates` on the pilot in
``tests/studies/separable_effects_pilot.py``, read by the design test.  The cross-fitted
scenario's lowest pilot coverage, 0.924, sets it.

**Unconverged fits** (declared after the pilot, before any declared draw).  A fit whose
targeting stops short of ``tol`` is kept: the shipped fit warns and reports the state with the
smallest relative score, and the study measures what it returns.  The pilot found none in sample
and 2.6% of cross-fitted fits, at relative scores of about 1e-3, with coverage 0.953 among them.
They are the draws where a fold's out-of-fold start leaves a score along nearly redundant
target directions (12 targets per factor and level against 6 saturated cells), so no nearby
state solves every score to ``tol``; the residual is a product of two fold differences.  A
dropped horizon would raise ``KeyError`` and stop the run; on this law at n = 16,000 every
horizon keeps units at risk.

**Comparator.** None.  No separable-effects TMLE implementation was located; the evidence is
the exact-law truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from cleverly.longitudinal import LTMLE
from tests.discrete_law_point_survival import CellMeans, CellProbabilities
from tests.parallel import STUDY_JOBS
from tests.studies import separable_effects_law as law
from tests.studies.canonical_point_survival_hazard import initial_estimates
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: :func:`~tests.studies.repeated_sizing.primary_replicates` on the pilot (design test).
PRIMARY_REPLICATES = 8_400
PRIMARY_N = 16_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 20261194
RESAMPLING_SEED = 2026119401
#: Cumulative bounds that never bind on this law.
G_BOUNDS = (1e-3, 1.0)

IN_SAMPLE = "in_sample"
CROSS_FITTED = "cross_fitted"

#: ``scenario -> outer folds``.
SCENARIOS: dict[str, int] = {IN_SAMPLE: 1, CROSS_FITTED: 2}
ESTIMANDS = dict.fromkeys(SCENARIOS, law.ESTIMANDS)
TRUTH: dict[str, dict[str, float]] = {scenario: dict(law.TRUTH) for scenario in SCENARIOS}
EFFICIENCY_SD: dict[str, float] = {name: law.efficiency_sd(name) for name in law.ESTIMANDS}

#: Every published property cell, by family (see the properties module).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": (
        "sep10_relapse_t3__event_hazard_wrong",
        "sep10_relapse_t3__competing_hazard_wrong",
        "sep10_relapse_t3__both_hazards_wrong",
        "sep10_relapse_t3__mechanism_wrong",
    ),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in ("in_sample_sde_relapse_t3", "cross_fitted_sep01_death_t3")
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
}

STUDY = StudyRecord(
    name="separable effects hazard longitudinal TMLE",
    slug="separable-effects",
    artifacts=ROOT / "tests" / "canonical" / "separable_effects",
    document="docs/technical-reference/method-evidence/separable-effects.md",
    anchor="separable-effects-hazard-longitudinal-tmle",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-separable-ltmle",
    reference=None,
    failure_bound={"cleverly-separable-ltmle": FAILURE_BOUND},
    efficiency_bounds=EFFICIENCY_SD,
    modules=(
        "tests/studies/separable_effects.py",
        "tests/studies/separable_effects_properties.py",
        "tests/studies/separable_effects_law.py",
        "tests/discrete_law_point_survival.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/canonical_point_survival_hazard.py",
        "tests/studies/canonical_point_survival.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_survival_common.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.separable_effects",
    properties_module="tests.studies.separable_effects_properties",
    property_cells=PROPERTY_CELLS,
)

CONFIGURATION = {
    "construction": "separable effects (Stensrud et al. 2022, identification (8), censoring (10))",
    "first_cause": law.FIRST,
    "within_node_order": ["censoring", "death", "relapse"],
    "regimens": {
        "a0": "A = 0 (reference)",
        "a1": "A = 1",
        "sep10": "relapse component 1, death component 0",
        "sep01": "relapse component 0, death component 1",
    },
    "engines": {
        IN_SAMPLE: "LTMLE survival_model='hazard', targeting='one_step', one fold",
        CROSS_FITTED: "LTMLE survival_model='hazard', targeting='one_step', two outer folds",
    },
    "sizes": {scenario: STUDY.n_of(scenario) for scenario in SCENARIOS},
    "learners": "CellMeans (hazards, censoring), CellProbabilities (treatment)",
    "initial_estimate": "the initial hazards' plug-in, from the targeting record",
    "g_bounds": list(G_BOUNDS),
    "simultaneous_intervals": False,
    "variance_method": "ic",
}


def fit(frame: pd.DataFrame, scenario: str, **learners: Any) -> Any:
    """The one fit of each primary scenario.  ``learners`` replace the declared ones."""
    settings: dict[str, Any] = {
        "survival_model": "hazard",
        "targeting": "one_step",
        "n_folds": SCENARIOS[scenario],
        "g_bounds": G_BOUNDS,
        "simultaneous": False,
        "random_state": 0,
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
        **learners,
    }
    return LTMLE(law.REGIMENS, reference=law.REFERENCE, **settings).fit(frame, **law.fit_columns())


def convergence(result: Any) -> tuple[bool, str, float]:
    """``(converged, failure, largest final relative score)`` over a fit's targeting records.

    ``result`` is an ``LTMLE`` result, or anything with a ``record`` (a robustness engine fit).
    """
    fits = getattr(result, "fits", None)
    records = (
        [result.record] if fits is None else [fit_record.hazard for fit_record in fits.values()]
    )
    failures = [str(record.failure) for record in records if not record.converged]
    score = max(float(record.trace[-1]) for record in records)
    return not failures, ",".join(failures), score


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return law.sample(n, seed), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    plug_in: Mapping[str, float],
) -> list[dict[str, Any]]:
    """One row per estimand on the identity scale."""
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
                "initial_estimate": float(plug_in[name]),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit(frame, scenario)
    plug_in = initial_estimates(result, ESTIMANDS[scenario])
    return result_rows(result, truth, scenario, replicate, plug_in)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every primary row, each scenario at its declared size.  No sample is kept."""
    sizes = STUDY.sizes(n)
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in SCENARIOS
            for replicate in range(replicates)
        ],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])

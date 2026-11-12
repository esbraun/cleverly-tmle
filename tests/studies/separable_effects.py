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
                       the iterative TMLE
=====================  ====================================================================

Every nuisance is a saturated cell mean (``CellMeans``, ``CellProbabilities``), the
nonparametric model on this support.  The static arms of a separable fit go through the
factor-wise engine, a different TMLE from the shipped static one, so their rows are measured
here too.

**Scope** (declared before the first run).  The record measures hazard ``LTMLE`` with saturated
learners at one and two folds.  Robustness to a wrong hazard factor is measured by the
properties module on the engine with the exact mechanism.

**Size** (declared before the first run, re-read before the rerun).  16,000, as the
held-baseline scenario of ``eliminated-competing-events`` on the same law, where n = 4,000 left
Kish sizes of 29 to 60 at ``t = 3``.  :data:`PRIMARY_REPLICATES` is the shared rule
:func:`~tests.studies.repeated_sizing.primary_replicates` on the pilot in
``tests/studies/separable_effects_pilot.py``, read by the design test.  The cross-fitted
scenario's lowest pilot coverage, 0.923, sets it.  The three cross-fitted pilot labels were
rerun under the stall fix of the targeting (plan ``X91-D-stall-plan.md`` rev 2.1), which moves
every cross-fitted fit; the sizes read that rerun.  The rule sits on an edge there: a covered
count of 924 in that cell gives 8,400, the 923 observed gives 9,200, and 922 would give 10,800.

**Unconverged fits** (declared after the rerun pilot, before the rerun).  A fit whose targeting
stops short of ``tol`` is kept: the shipped fit warns and reports the state with the smallest
relative score, and the study measures what it returns.  Every in-sample pilot fit converged in
the walk, which the fix leaves unchanged up to rounding, so those labels were not rerun.  In the
rerun, 0.6% of the cross-fitted primary and calibration fits (50 of 9,000) stopped short, every
one ``failure="cell_separated"``: a fold-cell whose own rows hold no event of a cause, so its
score is solved only at a hazard of zero, beyond the bound set by ``alpha``.  Their relative
scores are at most 2.3e-5.  The cross-fitted ``mechanism_wrong`` robustness label stopped short
in 7 of 300 fits, also all ``cell_separated``.  ``convergence.csv`` records every primary fit's
convergence, failure and targeting phase, and the property summary reports the unconverged fits
of each cell; neither decides a verdict.  :func:`primary_suspects` reads the declared defect
suspects (``regenerate.py``).  ``FAILURE_BOUND`` counts fits that raise, not fits that stop
short.  A dropped horizon would raise ``KeyError`` and stop the run; on this law at n = 16,000
every horizon keeps units at risk.

**Comparator.** None.  No separable-effects TMLE implementation was located; the evidence is
the exact-law truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
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
PRIMARY_REPLICATES = 9_200
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
    extra_artifacts=("convergence.csv",),
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


#: The per-fit convergence columns of the primary rows and of ``convergence.csv``.
CONVERGENCE_COLUMNS = ("converged", "failure", "relative_score", "phase")


def convergence(result: Any) -> tuple[bool, str, float, str]:
    """``(converged, failure, largest final relative score, phase)`` over a fit's targeting
    records; ``phase`` names the targeting phases that reached the records' reported states.

    ``result`` is an ``LTMLE`` result, or anything with a ``record`` (a robustness engine fit).
    """
    fits = getattr(result, "fits", None)
    records = (
        [result.record] if fits is None else [fit_record.hazard for fit_record in fits.values()]
    )
    failures = [str(record.failure) for record in records if not record.converged]
    score = max(float(record.trace[-1]) for record in records)
    phases = [str(record.phase) for record in records]
    # A joint walk shares one record between every parameter, so each label is listed once.
    return not failures, ",".join(dict.fromkeys(failures)), score, ",".join(dict.fromkeys(phases))


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
    """One row per estimand on the identity scale, with the fit's :data:`CONVERGENCE_COLUMNS`."""
    converged, failure, score, phase = convergence(result)
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
                "converged": int(converged),
                "failure": failure,
                "relative_score": score,
                "phase": phase,
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
    # The convergence columns ride along for :func:`extra_artifacts`; the driver drops them.
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *CONVERGENCE_COLUMNS]])


#: The declared readings of fits that stop short (``regenerate.py``): the largest share of a
#: cross-fitted cell's fits that may read ``cell_separated``, and the largest relative score an
#: unconverged fit may keep.
SEPARATED_SHARE = 0.015
UNCONVERGED_SCORE = 1e-4

#: The suspects the declared run flagged, read after it and investigated (2026-11-12): each
#: fit's ``(cell, replicate)`` and the relative score it stopped at, with
#: ``failure="newton_stalled"``, 1.2 to 2.4 times ``tol = 1e-10``.  The finding, which
#: ``python -m tests.studies.separable_effects_stalls`` reproduces, is :data:`STALL_FINDING`.
#: Eric chose to keep the run and record them here rather than change the solver and rerun.
INVESTIGATED_STALLS: Mapping[tuple[str, int], float] = MappingProxyType(
    {
        ("cross_fitted", 2716): 1.3930364321250592e-10,
        ("cross_fitted", 4377): 1.2182806251151364e-10,
        ("cross_fitted_sep01_death_t3__correctly_specified", 3264): 2.356290182224947e-10,
    }
)
#: How closely a listed fit must stop at its recorded relative score: refits of one draw on two
#: machines differ by about 4e-6 of it.
STALL_RTOL = 1e-3
STALL_FINDING = (
    "the remaining score lies in two directions in which the Jacobian of the factor scores is "
    "singular to rounding, which the Newton phase discards; steps forced along them reach a "
    "root within tol that moves no estimate by more than 0.0015 standard errors"
)


def convergence_suspects(
    fits: pd.DataFrame,
    cross_fitted: Mapping[str, bool],
    investigated: Mapping[tuple[str, int], float] = INVESTIGATED_STALLS,
) -> list[str]:
    """The declared defect suspects among fits, one row per fit, grouped by ``cell``.

    ``cross_fitted`` maps each cell to whether its fits are cross-fitted.  An in-sample fit is a
    suspect unless it converged in the walk.  In a cross-fitted cell, a fit that stops short for
    any reason but ``cell_separated``, or above :data:`UNCONVERGED_SCORE`, is a suspect, and so
    is a ``cell_separated`` share above :data:`SEPARATED_SHARE`.  A fit listed in
    ``investigated`` (by ``cell`` and ``replicate``) is passed only while it stops short with
    ``newton_stalled`` at its recorded relative score; a listed fit that does anything else is a
    suspect, so the list cannot outlive the run it describes.  Empty failures read back from a
    file as missing, so they are filled first.
    """
    found: list[str] = []
    for cell, group in fits.groupby("cell", sort=True):
        failure = group["failure"].fillna("").astype(str)
        short = group["converged"].astype(int) == 0
        listed = pd.Series(False, index=group.index)
        if "replicate" in group:
            recorded = group["replicate"].map(
                lambda r, cell=str(cell): investigated.get((cell, int(r)), np.nan)
            )
            listed = recorded.notna()
            same = short & failure.eq("newton_stalled")
            same &= np.isclose(group["relative_score"], recorded, rtol=STALL_RTOL, atol=0.0)
            if (listed & ~same).any():
                found.append(
                    f"{cell}: {int((listed & ~same).sum())} investigated fits no longer stall "
                    "as recorded"
                )
        if not cross_fitted[str(cell)]:
            off = short | group["phase"].fillna("").astype(str).ne("walk")
            if off.any():
                found.append(
                    f"{cell}: {int(off.sum())} in-sample fits left the walk or stopped short"
                )
            continue
        other = short & failure.ne("cell_separated") & ~listed
        if other.any():
            labels = sorted(set(failure[other]))
            found.append(f"{cell}: {int(other.sum())} fits stopped short with {labels}")
        high = short & (group["relative_score"] > UNCONVERGED_SCORE)
        if high.any():
            found.append(f"{cell}: {int(high.sum())} unconverged fits above {UNCONVERGED_SCORE}")
        share = float((short & failure.eq("cell_separated")).mean())
        if share > SEPARATED_SHARE:
            found.append(f"{cell}: cell_separated in {share:.2%} of fits")
    return found


def primary_suspects(convergence: pd.DataFrame) -> list[str]:
    """:func:`convergence_suspects` on ``convergence.csv``, one cell per scenario."""
    fits = convergence.rename(columns={"scenario": "cell"})
    return convergence_suspects(fits, {name: folds > 1 for name, folds in SCENARIOS.items()})


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``convergence.csv``: whether each primary fit's targeting reached ``tol``, one row per fit.

    The file decides no verdict.  It lets a reader split a cell's coverage by convergence, which
    the red-cell reading of ``cross_fitted`` needs.
    """
    columns = ["scenario", "replicate", *CONVERGENCE_COLUMNS]
    fits = rows.loc[rows["implementation"].eq(STUDY.implementation), columns]
    return {
        "convergence.csv": fits.drop_duplicates(["scenario", "replicate"]).sort_values(
            ["scenario", "replicate"], ignore_index=True
        )
    }

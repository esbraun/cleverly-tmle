r"""Registered evidence for the switched variance of the continuous-time TMLE (X6 part (a)).

The subject is :class:`~cleverly.survival.ContinuousSurvivalTMLE` with its default
``variance="switched"``: the influence-curve (IC) variance, raised to the predictable variation
(PV) of the influence curve's martingale part where the curve is sparse (its ``variance_rel_sd``
above 0.125), on every risk and risk difference (X6 plan rev 5, sections 2.1.4 and 5.1).  Each fit
records both variances, so every candidate interval is read on the same fits.

**Cells** (:data:`CELLS`), each the default libraries with 10 outer and 5 inner folds, on a law
of :mod:`tests.studies.continuous_survival_variance_law` that no registered study measures:

* ``v1_rare_cause2``: the cause-2 intercept at -2.6, target times at that law's event-time
  quartiles (0.52, 1.29, 2.69), :math:`n = 1{,}000`; about 9.5 events of the treated arm's
  cause 2 by the first quartile;
* ``v2_heavy_weights``: propensity :math:`\omega = (2, -1, 1.5)`, :math:`n = 1{,}000`; heavier
  control weights (the excess-spread mechanism M2);
* ``v3_half_n``: the base law at :math:`n = 500`, which decides whether ``MEASURED_N`` falls;
* ``v4_in_sample``: the base law, ``cross_fit=False``, where the IC standard error runs lowest;
* ``v7_quadratic``: :math:`0.3 (X_3^2 - 1)` in the cause-1 log hazard, outside every default
  Cox member, where the PV's omitted drift could show.

**Gates.** The twelve targets at the median and upper quartile of each cell
(:func:`~tests.studies.continuous_survival_variance_law.gated`) take the primary rule of
:data:`MARGINS` on the default's reported interval: the coverage interval above 0.90 and below
0.99, the SE ratio's interval inside (0.8, 1.2) and the bias equivalent to zero within 0.2 SD.
They are the merge gate.  The six first-quartile targets of each fit ride on every gated row of
that fit (:data:`EARLY_COLUMNS`) and are read by
:mod:`tests.studies.continuous_survival_variance_readings` under the calibration rule: they decide
the ship rule's benefit clause and the share floor, and are no merge gate.

**Readings** (``candidate-readings.csv``, ``ship-rule.csv``, ``floors.csv``; computed from the
committed rows by code, :mod:`tests.studies.continuous_survival_variance_readings`).  The ship rule
of plan 5.1 with its controls, the width-matched readings for (a2), and the three support floors.

**Size and seeds.** :data:`PRIMARY_REPLICATES` is the plan's 3,600 per cell, where the ship rule's
power is stated (plan 5.1; the at-target branch of ``guide/studies.md`` for the first-quartile
calibration rows).  The pilot (:mod:`tests.studies.continuous_survival_variance_pilot`, 500 draws
per cell on its own stream) reads V7's coefficient rule and witness and each first-quartile row's
plug-in pass probability, which can extend the pilot and raise a cell's count up to
:data:`REPLICATE_CAP`.  :data:`SEED` and :data:`RESAMPLING_SEED` are used by no other study;
replicate ``r`` of a cell draws from ``replicate_seed(STUDY, cell, r)``.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from cleverly.survival import ContinuousSurvivalTMLE, _targeting
from cleverly.survival import engine as engine_module
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_law as base
from tests.studies import continuous_survival_variance_law as laws
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: The plan's count per cell (5.1, "if S"): the ship rule's power is stated at 3,600.
PRIMARY_REPLICATES = 3_600
#: The cap per cell when the pilot's plug-in reading asks for more (plan 5.1).
REPLICATE_CAP = 7_200
#: The pilot's draws per cell (plan 5.1, budget).
PILOT_REPLICATES = 500
PRIMARY_N = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026611461
RESAMPLING_SEED = 2026611462
N_FOLDS = registered.N_FOLDS
LEARNER_FOLDS = registered.LEARNER_FOLDS

SUBJECT = "cleverly-continuous-survival-switched"

#: ``cell -> (law, n, cross-fitted)``.
CELLS: dict[str, tuple[str, int, bool]] = {
    "v1_rare_cause2": ("rare_cause2", PRIMARY_N, True),
    "v2_heavy_weights": ("heavy_weights", PRIMARY_N, True),
    "v3_half_n": ("base", 500, True),
    "v4_in_sample": ("base", PRIMARY_N, False),
    "v7_quadratic": ("quadratic", PRIMARY_N, True),
}
ESTIMANDS: dict[str, tuple[str, ...]] = {
    cell: laws.gated(law) for cell, (law, _, _) in CELLS.items()
}
MARGINS = Margins(standardized_bias=0.2)

READINGS_FILE = "targeting-readings.csv"
CANDIDATES_FILE = "candidate-readings.csv"
SHIP_FILE = "ship-rule.csv"
FLOORS_FILE = "floors.csv"

#: The per-estimate variance readings every target carries: the IC and PV standard errors,
#: the PV with the multinomial square dropped (T-a2's mutation, reported), the relative SD the
#: switch reads, and the oracle standard error (the law's efficient SD over root n).
VARIANCE_FIELDS = (
    "ic_std_error",
    "pv_std_error",
    "pv_nosq_std_error",
    "variance_rel_sd",
    "oracle_std_error",
)
#: The per-fit columns that carry the first-quartile targets on every gated row of the fit.
EARLY_FIELDS = ("estimate", "std_error", *VARIANCE_FIELDS)
EARLY_COLUMNS = tuple(f"early{index}_{field}" for index in range(6) for field in EARLY_FIELDS)

STUDY = StudyRecord(
    name="continuous-time survival switched variance",
    slug="continuous-survival-variance",
    artifacts=ROOT / "tests" / "canonical" / "continuous_survival_variance",
    document="docs/technical-reference/method-evidence/continuous-survival-variance.md",
    anchor="continuous-time-survival-switched-variance",
    scenarios=dict(ESTIMANDS),
    scenario_n={cell: n for cell, (_, n, _) in CELLS.items() if n != PRIMARY_N},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation=SUBJECT,
    reference=None,
    failure_bound={SUBJECT: FAILURE_BOUND},
    nuisance_count=3,
    extra_artifacts=(READINGS_FILE, CANDIDATES_FILE, SHIP_FILE, FLOORS_FILE),
    modules=(
        "tests/studies/continuous_survival_variance.py",
        "tests/studies/continuous_survival_variance_law.py",
        "tests/studies/continuous_survival_variance_properties.py",
        "tests/studies/continuous_survival_variance_readings.py",
        "tests/studies/continuous_survival.py",
        "tests/studies/continuous_survival_law.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        # Imported through the registered study's module; none of them is called by these fits.
        "tests/studies/stratified_incremental_msm_properties.py",
        "tests/studies/canonical_stratified_incremental_msm.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law.py",
    ),
    runner_module="tests.studies.continuous_survival_variance",
    properties_module="tests.studies.continuous_survival_variance_properties",
)

CONFIGURATION = {
    "construction": "Rytgaard and van der Laan (2024) one-step TMLE, both arms tilted; "
    "variance='switched' (IC raised to PV where variance_rel_sd > 0.125)",
    "laws": {cell: law for cell, (law, _, _) in CELLS.items()},
    "times": {cell: list(laws.LAWS[law].times) for cell, (law, _, _) in CELLS.items()},
    "quadratic": laws.QUADRATIC,
    "n_folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "learners": "cox.default_library() for each hazard; logistic + lasso C in (1, 0.1, 0.03)",
    "sizes": {cell: n for cell, (_, n, _) in CELLS.items()},
    "variance_method": "switched (influence curve raised to predictable variation)",
}


# -------------------------------------------------------------------------- the draws


def law_of(scenario: str) -> laws.Law:
    """The law a cell draws from."""
    return laws.LAWS[CELLS[scenario][0]]


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    name = CELLS[scenario][0]
    return laws.LAWS[name].sample(n, seed), dict(laws.truth(name))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# --------------------------------------------------------------------------- the fits


def engine(scenario: str) -> ContinuousSurvivalTMLE:
    """The default fit of a cell: its law's times, cross-fitted unless the cell is in sample."""
    _, _, cross_fit = CELLS[scenario]
    return ContinuousSurvivalTMLE(
        law_of(scenario).times,
        arms=base.ARMS,
        reference=base.REFERENCE,
        causes=base.CAUSES,
        cross_fit=cross_fit,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        simultaneous=False,
        random_state=0,
    )


def _no_square(clever: np.ndarray, dlam: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """T-a2's mutation of the multinomial moment: the squared term dropped."""
    P = clever.shape[1]
    out = np.zeros((P, P))
    for l in range(clever.shape[0]):
        flat = (clever[l] * np.sqrt(rho * dlam[l])[None]).reshape(P, -1)
        out += flat @ flat.T
    return out


@contextmanager
def recording_no_square(holder: list[np.ndarray]) -> Iterator[None]:
    """While open, every fit also appends its PV matrix without the multinomial square.

    The fit itself reads the shipped matrix; the mutated one is a reported reading only.
    """
    original = engine_module.predictable_covariance

    def wrapped(problem: Any, evaluation: Any, dlam: Any, reach: Any = None) -> np.ndarray:
        out = original(problem, evaluation, dlam, reach)
        shipped = _targeting._multinomial_moment
        _targeting._multinomial_moment = _no_square
        try:
            holder.append(original(problem, evaluation, dlam, reach))
        finally:
            _targeting._multinomial_moment = shipped
        return out

    engine_module.predictable_covariance = wrapped
    try:
        yield
    finally:
        engine_module.predictable_covariance = original


def fit(frame: pd.DataFrame, scenario: str) -> tuple[Any, np.ndarray]:
    """The subject's fit of one replicate of a cell, with the no-square PV matrix."""
    holder: list[np.ndarray] = []
    with recording_no_square(holder):
        result = registered.fit_with(frame, engine(scenario))
    return result, holder[0]


def variance_readings(
    result: Any, name: str, no_square: np.ndarray, scenario: str
) -> dict[str, float]:
    """:data:`VARIANCE_FIELDS` of one estimate of one fit."""
    estimate = result[name]
    record = estimate.switched
    gradient = result.predictable.gradients[name]
    # The fit holds no restricted rows, so the levels' matrix is the walk's over n - 1.
    nosq = float(gradient @ no_square @ gradient) / max(result.n - 1, 1)
    efficient = laws.efficiency(CELLS[scenario][0])[name]
    return {
        "ic_std_error": float(np.sqrt(record.influence_variance)),
        "pv_std_error": float(np.sqrt(max(record.predictable_variance, 0.0))),
        "pv_nosq_std_error": float(np.sqrt(max(nosq, 0.0))),
        "variance_rel_sd": float(record.relative_sd),
        "oracle_std_error": float(efficient / np.sqrt(result.n)),
    }


def result_rows(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    no_square: np.ndarray,
) -> list[dict[str, Any]]:
    """One row per gated target, on its reported scale, with its variance readings."""
    rows = []
    initial = result.initial_estimates
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": SUBJECT,
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
                "interval_scale": estimate.interval_scale,
                "covered": int(low <= reference <= high),
                "initial_estimate": float(initial[name]),
                **variance_readings(result, name, no_square, scenario),
            }
        )
    return rows


def early_readings(result: Any, scenario: str, no_square: np.ndarray) -> dict[str, float]:
    """The first-quartile targets of one fit, as :data:`EARLY_COLUMNS`."""
    out: dict[str, float] = {}
    for index, name in enumerate(laws.first_quartile(CELLS[scenario][0])):
        estimate = result[name]
        out[f"early{index}_estimate"] = float(estimate.psi)
        out[f"early{index}_std_error"] = float(estimate.std_error)
        for field, value in variance_readings(result, name, no_square, scenario).items():
            out[f"early{index}_{field}"] = value
    return out


#: The columns a primary row carries beside the published schema.
EXTRA_COLUMNS = (
    "interval_scale",
    *VARIANCE_FIELDS,
    "seconds",
    "steps",
    "halvings",
    "converged",
    "bounded_share",
    "last_time_floor_share",
    "frozen_share",
    "off_grid_share",
    *EARLY_COLUMNS,
)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result, no_square = fit(frame, scenario)
        seconds = time.perf_counter() - start
    extra = {
        "seconds": seconds,
        **registered.readings(result),
        **early_readings(result, scenario, no_square),
    }
    return [{**row, **extra} for row in result_rows(result, truth, scenario, replicate, no_square)]


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    scenarios: tuple[str, ...] | None = None,
) -> replication.PhaseResult:
    """Every cell of ``scenarios`` (all by default), each at its declared size."""
    sizes = STUDY.sizes(n)
    chosen = tuple(STUDY.scenarios) if scenarios is None else scenarios
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in chosen
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
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]])


# --------------------------------------------------------------- reported artifacts


def account_failures(
    failures: pd.DataFrame | None, artifacts: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """No scientific gate reads the extra tables, so a lost fit counts only against the bound."""
    return evidence_failures.audits({}, missing={})


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The targeting record, the candidates' readings, the ship rule and the floors."""
    from tests.studies import continuous_survival_variance_readings as readings

    candidates = readings.candidate_readings(rows)
    return {
        READINGS_FILE: registered.targeting_readings(rows),
        CANDIDATES_FILE: candidates,
        SHIP_FILE: readings.ship_rule(candidates),
        FLOORS_FILE: readings.floors(rows, candidates),
    }

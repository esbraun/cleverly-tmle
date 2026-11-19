r"""Registered evidence for the continuous-time TMLE with one cause and with three (X6 part (b)).

The subject is :class:`~cleverly.survival.ContinuousSurvivalTMLE` in its shipped default: the
default libraries, cross-fitted with 10 outer folds and selection inside 5 inner folds, both arms
tilted, and ``variance="switched"`` (the influence-curve variance raised to the predictable
variation where the curve is sparse, X6 part (a)).  The registered ``continuous-time-survival``
study measures it with two causes; this study measures it with one cause and with three, on the
laws of :mod:`tests.studies.continuous_survival_causes_law` (X6 plan rev 5, section 5.3).

**Cells** (:data:`CELLS`), each at :math:`n = 1{,}000` with target times at its law's event-time
quartiles:

* ``l1_cf_1k``: law L1, one cause (the base law's :math:`h_1`, :math:`h_2 \equiv 0`), times
  (0.58, 1.48, 3.24);
* ``l3_cf_1k``: law L3, three causes (the base law's :math:`h_1, h_2` and
  :math:`h_3 = \exp(-2.5 + 0.5 X_1 + 0.5 X_3 + 0.5 Z)`), times (0.37, 0.9, 1.86).

**Gates.** Every risk and risk difference at the times part (a) licensed takes the primary rule of
:data:`MARGINS`: the coverage interval above 0.90 and below 0.99, the SE ratio's interval inside
(0.8, 1.2) and the bias equivalent to zero within 0.2 SD.  Part (a) kept the share floor at 0.40
(``continuous-survival-variance``'s ``floors.csv``), so the first quartile is not licensed and
the gated times are the median and the upper quartile: 6 rows for L1 and 18 for L3
(:func:`~tests.studies.continuous_survival_causes_law.gated`).  The first quartile's targets ride
on every gated row of their fit (:data:`EARLY_COLUMNS`) and publish as ``early-time.csv`` with no
verdict.  The calibration cells, the double-robustness cells and their controls are in
:mod:`tests.studies.continuous_survival_causes_properties`.

**Size and seeds.**  The pilot (:mod:`tests.studies.continuous_survival_causes_pilot`,
:data:`PILOT_REPLICATES` draws per cell on its own stream) sets :data:`PRIMARY_REPLICATES` by the
shared primary rule (:func:`~tests.studies.repeated_sizing.primary_replicates`, on the 24 gated
rows' covered counts and one bias group per cell, read on each cell's first
:data:`PILOT_REPLICATES` draws), at least :data:`MINIMUM_REPLICATES`: 7,200, set by L3's
control cause-1 risk at the median (pilot coverage 0.926 on 800 draws).  The design test
recomputes the count from the committed pilot, and ``regenerate.py`` refuses a declared run
that disagrees with it.  A count above :data:`REPLICATE_CAP` stops the declaration (no
truncation).  :data:`SEED` and
:data:`RESAMPLING_SEED` are used by no other study; replicate ``r`` of a cell draws from
``replicate_seed(STUDY, cell, r)``.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_causes_law as laws
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

#: The count per cell: the shared primary rule on the committed pilot (``pilot.primary_count``),
#: checked by the design test.
PRIMARY_REPLICATES = 7_200
#: ``guide/studies.md``: primaries run at 1,000 replications or more.
MINIMUM_REPLICATES = 1_000
#: The plan's cap per cell (5.3, the registered study's 7,600).  A pilot whose rule asks for more
#: stops the declaration: the run is not truncated to the cap.
REPLICATE_CAP = 7_600
#: The plan's pilot per law (5.3).
PILOT_REPLICATES = 800
PRIMARY_N = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026611561
RESAMPLING_SEED = 2026611562
N_FOLDS = registered.N_FOLDS
LEARNER_FOLDS = registered.LEARNER_FOLDS

SUBJECT = "cleverly-continuous-survival-causes"

#: ``cell -> law``: every cell is the default cross-fitted fit at n = 1,000.
CELLS: dict[str, str] = {"l1_cf_1k": "one_cause", "l3_cf_1k": "three_causes"}
ESTIMANDS: dict[str, tuple[str, ...]] = {cell: laws.gated(law) for cell, law in CELLS.items()}
MARGINS = Margins(standardized_bias=0.2)

READINGS_FILE = "targeting-readings.csv"
EARLY_FILE = "early-time.csv"
#: The most first-quartile targets of one cell (L3: three causes, two arms and a difference).
EARLY_COUNT = max(len(laws.first_quartile(law)) for law in CELLS.values())
EARLY_FIELDS = ("estimate", "std_error", "covered")
#: The per-fit columns that carry the first quartile's targets on every gated row of the fit;
#: a cell with fewer targets leaves the rest empty.
EARLY_COLUMNS = tuple(
    f"early{index}_{field}" for index in range(EARLY_COUNT) for field in EARLY_FIELDS
)

STUDY = StudyRecord(
    name="continuous-time survival with one and three causes",
    slug="continuous-survival-causes",
    artifacts=ROOT / "tests" / "canonical" / "continuous_survival_causes",
    document="docs/technical-reference/method-evidence/continuous-survival-causes.md",
    anchor="continuous-time-survival-with-one-and-three-causes",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation=SUBJECT,
    reference=None,
    failure_bound={SUBJECT: FAILURE_BOUND},
    nuisance_count=3,
    extra_artifacts=(READINGS_FILE, EARLY_FILE),
    modules=(
        "tests/studies/continuous_survival_causes.py",
        "tests/studies/continuous_survival_causes_law.py",
        "tests/studies/continuous_survival_causes_properties.py",
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
    runner_module="tests.studies.continuous_survival_causes",
    properties_module="tests.studies.continuous_survival_causes_properties",
    property_cells={
        "double_robustness": (
            "l1_c1__mechanism_correct",
            "l1_c1__plug_in_control",
            "l1_c1__factual_tilt_control",
            "l3_c1__mechanism_correct",
            "l3_c1__plug_in_control",
            "l3_c1__factual_tilt_control",
            "l3_c3__mechanism_correct",
            "l3_c3__plug_in_control",
            "l3_c3__factual_tilt_control",
        ),
        "interval_calibration": tuple(
            f"{label}__{cell}"
            for label in ("l1_rd", "l3_rd")
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
    },
    # The calibration cells read the coverage and SE-ratio bands only; the efficient spread
    # scales the noise control and is not claimed as an efficiency ratio.
    calibration_efficiency_ratio=False,
)

CONFIGURATION = {
    "construction": "Rytgaard and van der Laan (2024) one-step TMLE, both arms tilted; "
    "variance='switched' (IC raised to PV where variance_rel_sd > 0.125)",
    "laws": dict(CELLS),
    "causes": {cell: laws.LAWS[law].causes for cell, law in CELLS.items()},
    "times": {cell: list(laws.LAWS[law].times) for cell, law in CELLS.items()},
    "cause3": list(laws.CAUSE3),
    "n_folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "learners": {
        "default": "cox.default_library() for each hazard; logistic + lasso C in (1, 0.1, 0.03)",
        "hazard_wrong": "CoxHazard('treatment')",
    },
    "sizes": dict.fromkeys(CELLS, PRIMARY_N),
    "variance_method": "switched (influence curve raised to predictable variation)",
}


# -------------------------------------------------------------------------- the draws


def law_of(scenario: str) -> laws.Law:
    """The law a cell draws from."""
    return laws.LAWS[CELLS[scenario]]


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return law_of(scenario).sample(n, seed), dict(laws.truth(CELLS[scenario]))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# --------------------------------------------------------------------------- the fits


def engine(
    law: laws.Law, *, hazard_wrong: bool = False, tilt: str = "all"
) -> ContinuousSurvivalTMLE:
    """The default fit of a law: its times and causes, cross-fitted.

    ``hazard_wrong`` replaces each cause's hazard library by the Cox model on the treatment
    alone, which ignores every confounder; ``tilt="factual"`` is the ``concrete`` 1.0.5 update
    the engine keeps for the mutation control.
    """
    settings: dict[str, Any] = {}
    if hazard_wrong:
        settings["hazard_library"] = (CoxHazard("treatment"),)
    return ContinuousSurvivalTMLE(
        law.times,
        arms=laws.ARMS,
        reference=laws.REFERENCE,
        causes=law.labels,
        cross_fit=True,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        simultaneous=False,
        random_state=0,
        tilt=tilt,  # type: ignore[arg-type]
        **settings,
    )


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The subject's fit of one replicate of a cell."""
    return registered.fit_with(frame, engine(law_of(scenario)))


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per gated target, on its reported scale."""
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
            }
        )
    return rows


def early_readings(result: Any, scenario: str) -> dict[str, float]:
    """The first quartile's targets of one fit, as :data:`EARLY_COLUMNS`."""
    truth = laws.truth(CELLS[scenario])
    out: dict[str, float] = dict.fromkeys(EARLY_COLUMNS, np.nan)
    for index, name in enumerate(laws.first_quartile(CELLS[scenario])):
        estimate = result[name]
        low, high = estimate.ci
        out[f"early{index}_estimate"] = float(estimate.psi)
        out[f"early{index}_std_error"] = float(estimate.std_error)
        out[f"early{index}_covered"] = float(low <= truth[name] <= high)
    return out


#: The columns a primary row carries beside the published schema.
EXTRA_COLUMNS = (
    "interval_scale",
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
        result = fit(frame, scenario)
        seconds = time.perf_counter() - start
    extra = {
        "seconds": seconds,
        **registered.readings(result),
        **early_readings(result, scenario),
    }
    return [{**row, **extra} for row in result_rows(result, truth, scenario, replicate)]


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


def early_time(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell and first-quartile target: bias, spread, SE ratio and coverage, with no verdict."""
    fits = rows.drop_duplicates(["scenario", "replicate"])
    records = []
    for scenario, group in fits.groupby("scenario", sort=True):
        truth = laws.truth(CELLS[str(scenario)])
        for index, name in enumerate(laws.first_quartile(CELLS[str(scenario)])):
            estimate = group[f"early{index}_estimate"].to_numpy(dtype=float)
            std_error = group[f"early{index}_std_error"].to_numpy(dtype=float)
            covered = group[f"early{index}_covered"].to_numpy(dtype=float)
            spread = float(np.std(estimate, ddof=1))
            records.append(
                {
                    "scenario": scenario,
                    "estimand": name,
                    "fits": int(estimate.size),
                    "truth": float(truth[name]),
                    "bias": float(np.mean(estimate) - truth[name]),
                    "standardized_bias": float((np.mean(estimate) - truth[name]) / spread),
                    "empirical_se": spread,
                    "mean_std_error": float(np.mean(std_error)),
                    "se_ratio": float(np.mean(std_error) / spread),
                    "coverage": float(np.mean(covered)),
                }
            )
    return pd.DataFrame.from_records(records)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The targeting record and the first quartile's readings; neither is a verdict."""
    return {READINGS_FILE: registered.targeting_readings(rows), EARLY_FILE: early_time(rows)}

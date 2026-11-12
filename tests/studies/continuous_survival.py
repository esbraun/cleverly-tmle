r"""Registered evidence for the continuous-time survival TMLE (X91 part G).

The subject is :class:`~cleverly.survival.ContinuousSurvivalTMLE`: the one-step TMLE of
Rytgaard and van der Laan (2024) for the cause-specific absolute risks of two static arms, with
both arms tilted for every unit (``X91-G-design.md`` section 4.2).  The law is Hage et al.
(2025) scenario 1 with the two stated departures of :mod:`tests.studies.continuous_survival_law`:
two causes, constant cause-specific hazards, and an arm-specific term :math:`\beta Z X_4` that
only the Cox ``"interaction"`` model holds.

``both_<mode>_<n>`` fits the default libraries on the base law at :math:`n = 1{,}000` or
:math:`4{,}000`.  ``<mode>`` is ``fs`` (every nuisance fitted on all rows, ``cross_fit=False``)
or ``cf`` (the default: 10 outer folds, selection inside 5 inner folds).  The record gates
``both_cf_1k`` alone (:data:`GATED_CELL`); ``both_fs_1k``, ``both_fs_4k`` and ``both_cf_4k`` are
reported cells of :mod:`tests.studies.continuous_survival_properties`, published as measured.
At 4,000 a fit takes 60 to 120 seconds against 5 at 1,000, and the shared rules asked for about
800 core-hours there (Eric's decision, 2026-11-12).  The default hazard
library is the unstratified Cox models on the treatment alone, on main terms and with treatment
interactions (:func:`cleverly.survival.cox.default_library`), selected by cross-validated
partial likelihood; the default treatment library is a main-terms logistic and three lasso
penalties, selected by cross-validated log loss.  The wrong-model cells, the calibration cell
and the reported cells are in :mod:`tests.studies.continuous_survival_properties`.

**Targets.** Each arm's risk of each cause, and each cause's risk difference, at the quartiles
of the observed event times, :data:`~tests.studies.continuous_survival_law.TIMES`.

**Gates.** Every row of the record is gated: standardized bias under :data:`MARGINS`,
coverage and the SE ratio, for each arm's risk of each cause and each risk difference at the
two later times (:data:`~tests.studies.continuous_survival_law.GATED`).  The first time's six
targets come from the same fits and publish as ``early-time.csv`` with no verdict
(:data:`~tests.studies.continuous_survival_law.EARLY`): the pilot read their standard error
below the sampling spread, a finite-sample finding the record states rather than gates
(Eric's decision, 2026-11-12).

**Size and seeds.** :data:`PRIMARY_REPLICATES` follows from the pilot
(``tests/studies/continuous_survival_pilot.py``) by the shared rule.  :data:`SEED` and
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
from sklearn.linear_model import LogisticRegression

from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.stratified_incremental_msm_properties import Columns

#: The pilot's draws per cell; the primary count follows from it.  The first 300 ran at
#: 4b232921; draws 300 to 999 extend the same streams (Eric's decision, 2026-11-12).
PILOT_REPLICATES = 1_000
#: The declared primary replicate count, set from the committed pilot by the shared rule
#: before any declared draw.  Provisional until the pilot is read.
PRIMARY_REPLICATES = 1_000
PRIMARY_N = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026911961
RESAMPLING_SEED = 2026911962
N_FOLDS = 10
LEARNER_FOLDS = 5

SUBJECT = "cleverly-continuous-survival-tmle"

#: ``cell -> (cross-fitted, n)`` of every default-library cell the pilot sized.
PILOT_CELLS: dict[str, tuple[bool, int]] = {
    f"both_{mode}_{n // 1000}k": (mode == "cf", n) for mode in ("fs", "cf") for n in (1_000, 4_000)
}
#: The record gates the default fit, cross-fitted, at n = 1,000 (Eric's decision, 2026-11-12).
#: The other three publish as reported cells of the properties module.
GATED_CELL = "both_cf_1k"
SCENARIOS: dict[str, tuple[bool, int]] = {GATED_CELL: PILOT_CELLS[GATED_CELL]}
ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(SCENARIOS, law.GATED)
MARGINS = Margins(standardized_bias=0.2)

READINGS_FILE = "targeting-readings.csv"
EARLY_FILE = "early-time.csv"
#: The per-fit columns that carry the first time's targets on every gated row of the fit.
EARLY_FIELDS = ("estimate", "std_error", "covered")
EARLY_COLUMNS = tuple(
    f"early{index}_{field}" for index in range(len(law.EARLY)) for field in EARLY_FIELDS
)

STUDY = StudyRecord(
    name="continuous-time survival TMLE",
    slug="continuous-time-survival",
    artifacts=ROOT / "tests" / "canonical" / "continuous_survival",
    document="docs/technical-reference/method-evidence/continuous-time-survival.md",
    anchor="continuous-time-survival-tmle",
    scenarios=dict(ESTIMANDS),
    scenario_n={name: n for name, (_, n) in SCENARIOS.items() if n != PRIMARY_N},
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
        "tests/studies/continuous_survival.py",
        "tests/studies/continuous_survival_law.py",
        "tests/studies/continuous_survival_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/stratified_incremental_msm_properties.py",
        # Imported through ``Columns``'s module; none of them is called by this study's fits.
        "tests/studies/canonical_stratified_incremental_msm.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law.py",
    ),
    runner_module="tests.studies.continuous_survival",
    properties_module="tests.studies.continuous_survival_properties",
    property_cells={
        "double_robustness": (
            "treated_c1__mechanism_correct",
            "treated_c1__outcome_correct",
            "treated_c1__both_wrong",
        ),
        "interval_calibration": tuple(
            f"both_rd__{cell}"
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
    },
    property_artifacts=("reported-cells.csv", "reported-replicates.csv"),
)

CONFIGURATION = {
    "construction": "Rytgaard and van der Laan (2024) one-step TMLE, both arms tilted",
    "law": "Hage et al. (2025) scenario 1 with departures (tests/studies/continuous_survival_law.py)",
    "times": list(law.TIMES),
    "beta": law.BETA,
    "n_folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "learners": {
        "default": "cox.default_library() for each hazard; logistic + lasso C in (1, 0.1, 0.03)",
        "hazard_wrong": "CoxHazard('treatment')",
        "propensity_wrong": "unpenalized logistic on X1, X2",
        "censoring_wrong": "CoxHazard('treatment') on the dependent-censoring law",
    },
    "sizes": {name: n for name, (_, n) in SCENARIOS.items()},
    "variance_method": "influence curve",
}


# -------------------------------------------------------------------------- the draws


def sample(n: int, seed: int, *, law_name: str = "base") -> pd.DataFrame:
    """One draw of the base law, the dependent-censoring law or the positivity law."""
    return law.sample(
        n,
        seed,
        positivity=law_name == "positivity",
        dependent_censoring=law_name == "dependent_censoring",
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return sample(n, seed), dict(law.TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# --------------------------------------------------------------------------- the fits


def engine(
    *,
    cross_fit: bool = True,
    hazard_wrong: bool = False,
    propensity_wrong: bool = False,
    censoring_wrong: bool = False,
) -> ContinuousSurvivalTMLE:
    """The subject with the default libraries, or with the named nuisances wrong.

    A wrong event hazard is the Cox model on the treatment alone, which ignores every
    confounder; a wrong treatment model is the unpenalized logistic
    on :math:`X_1, X_2`; a wrong censoring hazard is the Cox model on the treatment alone, which
    misses the dependence on :math:`X_3`.
    """
    settings: dict[str, Any] = {}
    if hazard_wrong:
        settings["hazard_library"] = (CoxHazard("treatment"),)
    if propensity_wrong:
        settings["treatment_learner"] = Columns(
            LogisticRegression(C=np.inf, max_iter=2000), keep=(0, 1)
        )
    if censoring_wrong:
        settings["censoring_library"] = (CoxHazard("treatment"),)
    return ContinuousSurvivalTMLE(
        law.TIMES,
        arms=law.ARMS,
        reference=law.REFERENCE,
        causes=law.CAUSES,
        cross_fit=cross_fit,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        simultaneous=False,
        random_state=0,
        **settings,
    )


def fit_with(frame: pd.DataFrame, subject: ContinuousSurvivalTMLE) -> Any:
    """One fit of a law's frame."""
    return subject.fit(
        frame, time="time", event="event", treatment="Z", baseline=list(law.BASELINE)
    )


def fit(frame: pd.DataFrame, scenario: str) -> Any:
    """The subject's fit of one replicate of one primary cell."""
    cross_fit, _ = SCENARIOS[scenario]
    return fit_with(frame, engine(cross_fit=cross_fit))


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per estimand on the identity scale."""
    rows = []
    initial = result.initial_estimates
    for name in law.GATED:
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
                "covered": int(low <= reference <= high),
                "initial_estimate": float(initial[name]),
            }
        )
    return rows


#: The columns a primary row carries beside the published schema.
EXTRA_COLUMNS = (
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


def early_readings(result: Any) -> dict[str, float]:
    """The first time's targets of one fit, as :data:`EARLY_COLUMNS`."""
    out: dict[str, float] = {}
    for index, name in enumerate(law.EARLY):
        estimate = result[name]
        low, high = estimate.ci
        truth = float(law.TRUTH[name])
        out[f"early{index}_estimate"] = float(estimate.psi)
        out[f"early{index}_std_error"] = float(estimate.std_error)
        out[f"early{index}_covered"] = float(low <= truth <= high)
    return out


def readings(result: Any) -> dict[str, float]:
    """The targeting record of one fit."""
    record = result.record
    return {
        "steps": float(record.steps),
        "halvings": float(record.halvings),
        "converged": float(record.converged),
        "bounded_share": float(record.bounded_share),
        "last_time_floor_share": float(record.last_time_floor_share),
        "frozen_share": float(record.frozen_share),
        "off_grid_share": float(record.off_grid_share),
    }


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
    extra = {"seconds": seconds, **readings(result), **early_readings(result)}
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


def targeting_readings(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell: steps, convergence, the positivity readings and run times."""
    fits = rows.drop_duplicates(["scenario", "replicate"])
    return (
        fits.groupby("scenario", sort=True)
        .agg(
            fits=("replicate", "size"),
            mean_steps=("steps", "mean"),
            max_steps=("steps", "max"),
            converged_share=("converged", "mean"),
            mean_bounded_share=("bounded_share", "mean"),
            mean_last_time_floor_share=("last_time_floor_share", "mean"),
            mean_frozen_share=("frozen_share", "mean"),
            mean_off_grid_share=("off_grid_share", "mean"),
            mean_seconds=("seconds", "mean"),
        )
        .reset_index()
    )


def account_failures(
    failures: pd.DataFrame | None, artifacts: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """No scientific gate reads the extra table, so a lost fit counts only against the bound."""
    return evidence_failures.audits({}, missing={})


def early_time(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell and first-time target: bias, spread, SE ratio and coverage, with no verdict."""
    fits = rows.drop_duplicates(["scenario", "replicate"])
    records = []
    for scenario, group in fits.groupby("scenario", sort=True):
        for index, name in enumerate(law.EARLY):
            estimate = group[f"early{index}_estimate"].to_numpy(dtype=float)
            std_error = group[f"early{index}_std_error"].to_numpy(dtype=float)
            covered = group[f"early{index}_covered"].to_numpy(dtype=float)
            truth = float(law.TRUTH[name])
            spread = float(np.std(estimate, ddof=1))
            records.append(
                {
                    "scenario": scenario,
                    "estimand": name,
                    "fits": int(estimate.size),
                    "truth": truth,
                    "bias": float(np.mean(estimate) - truth),
                    "standardized_bias": float((np.mean(estimate) - truth) / spread),
                    "empirical_se": spread,
                    "mean_std_error": float(np.mean(std_error)),
                    "se_ratio": float(np.mean(std_error) / spread),
                    "coverage": float(np.mean(covered)),
                }
            )
    return pd.DataFrame.from_records(records)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The reported tables; neither is a verdict."""
    return {READINGS_FILE: targeting_readings(rows), EARLY_FILE: early_time(rows)}

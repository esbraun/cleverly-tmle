r"""Registered evidence for the drift-corrected survival TMLE (X91 part F).

The subject is :class:`~cleverly.longitudinal.SurvivalDRTMLE`, Díaz (2019), Section 5: hazard
targeting of one binary baseline treatment with the drift covariates of eq. (9), as corrected
in :mod:`cleverly.longitudinal.drift`, and the influence curve of Theorem 2.  The law is L-S of
:mod:`tests.studies.survival_dr_tmle_law` (the audit's witness law, ``X91-F-design.md``
section 10), five integer times, horizon 5.

=================  ==================  ================  =================  ==================
cell               law                 hazard model      treatment          censoring
=================  ==================  ================  =================  ==================
``both_<n>``       observational       correct           correct            correct
``known_<n>``      trial               on A and W1 only  declared (0.5)     declared
=================  ==================  ================  =================  ==================

"Correct" is a large-``C`` logistic regression fitted at each node on ``[W1, W2, A]`` for the
hazard and the retention, and on ``W1, W2`` for the treatment; each is the true model's family.
A pooled fit over the nodes would not be: node 1 has no censoring, so a retention model linear
in the node time is misspecified (a first pilot read a bias of 0.7 SD at 16,000 in the
mechanism-correct robustness cell from it, for this method and hazard ``LTMLE`` alike).  ``known_<n>`` is row 2
of the design's case table: :math:`g` is declared, so it is never fluctuated, and the
remainder is exactly zero for any hazard model, so the interval is valid with the hazard model
wrong.  Every fit is cross-fitted with 10 outer folds (the default path, Theorem 3).

**Sizes.** :math:`n = 1{,}000`, :math:`4{,}000` and :math:`16{,}000` for both cells.

**Gates** (the shared margins, with the standardized-bias margin 0.2 of the design).  Every
row of the record is gated: the 99% upper bound of :math:`|\text{bias}|/\text{SD}` at most 0.2,
the coverage floor 0.90 at the lower end of the exact interval, and the SE-ratio screen
:math:`[0.8, 1.2]`, for the risk of each arm and the risk difference; and the fit-failure share
under :data:`FAILURE_BOUND`.

**Reported, not gated.** Hazard ``LTMLE`` (``survival_model="hazard"``, the same learners and
folds) is fitted on every replicate and carried on the subject's rows; ``paired-difference.csv``
holds the mean paired difference, in pooled standard deviations, and both RMSEs per cell.  It
is this library, not a comparator.  The loop's iterations, its convergence and the floored
denominators are in ``drift-readings.csv``.

**Size and seeds.** :data:`PRIMARY_REPLICATES` follows from the pilot
(``tests/studies/survival_dr_tmle_pilot.py``) by the shared rule.  :data:`SEED` and
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

from cleverly.longitudinal import LTMLE, LongitudinalData, SurvivalDRTMLE
from cleverly.validation._failures import recordable
from tests.parallel import STUDY_JOBS
from tests.studies import survival_dr_tmle_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.stratified_incremental_msm_properties import Columns

#: The pilot's draws per label; the primary count follows from it.
PILOT_REPLICATES = 1_000
#: The declared primary replicate count, set from the committed pilot by the shared rule
#: before any declared draw.
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 4_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026911901
RESAMPLING_SEED = 2026911902
N_FOLDS = 10
G_BOUNDS = (0.01, 1.0)
SIZES = (1_000, 4_000, 16_000)

SUBJECT = "cleverly-survival-dr-tmle"
HAZARD = "cleverly-hazard-ltmle"

#: ``cell -> (configuration, n)``.
SCENARIOS: dict[str, tuple[str, int]] = {
    f"{configuration}_{n // 1000}k": (configuration, n)
    for configuration in ("both", "known")
    for n in SIZES
}
ESTIMANDS = dict.fromkeys(SCENARIOS, law.ESTIMANDS)
MARGINS = Margins(standardized_bias=0.2)

PAIRED_FILE = "paired-difference.csv"
READINGS_FILE = "drift-readings.csv"

STUDY = StudyRecord(
    name="drift-corrected survival TMLE",
    slug="survival-dr-tmle",
    artifacts=ROOT / "tests" / "canonical" / "survival_dr_tmle",
    document="docs/technical-reference/method-evidence/survival-dr-tmle.md",
    anchor="drift-corrected-survival-tmle",
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
    extra_artifacts=(PAIRED_FILE, READINGS_FILE),
    modules=(
        "tests/studies/survival_dr_tmle.py",
        "tests/studies/survival_dr_tmle_law.py",
        "tests/studies/survival_dr_tmle_properties.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/stratified_incremental_msm_properties.py",
        # Imported through ``Columns``'s module; none of them is called by this study's fits.
        "tests/studies/canonical_stratified_incremental_msm.py",
        "tests/studies/stratified_alternating_law.py",
        "tests/studies/stratified_law.py",
        "tests/studies/point_study_helpers.py",
        "tests/discrete_law.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.survival_dr_tmle",
    properties_module="tests.studies.survival_dr_tmle_properties",
    property_cells={
        "double_robustness": (
            "a1_t5__mechanism_correct",
            "a1_t5__outcome_correct",
            "a1_t5__both_wrong",
        ),
        "interval_calibration": tuple(
            f"both_rd__{cell}"
            for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
        ),
    },
    property_artifacts=("reported-cells.csv", "reported-replicates.csv"),
)

CONFIGURATION = {
    "construction": "Díaz (2019), Section 5, with the corrected eq. (9) of cleverly.longitudinal.drift",
    "law": "L-S (tests/studies/survival_dr_tmle_law.py)",
    "horizon": law.HORIZON,
    "grid": list(law.GRID),
    "n_folds": N_FOLDS,
    "g_bounds": list(G_BOUNDS),
    "learners": {
        "correct": "LogisticRegression(C=1e6) per node on every design column",
        "hazard_wrong": "the same on W1 and A",
        "declared": "treatment_probabilities p1, p0 and censoring_probabilities r1..r4",
    },
    "engines": {
        SUBJECT: "SurvivalDRTMLE, max_iter=200",
        HAZARD: "LTMLE(survival_model='hazard'), reported beside the subject",
    },
    "sizes": {name: n for name, (_, n) in SCENARIOS.items()},
    "variance_method": "ic (Theorem 2 curve)",
}


# -------------------------------------------------------------------------- the draws


def container(
    frame: pd.DataFrame, *, treatment: bool = False, censoring: bool = False
) -> LongitudinalData:
    """The time-to-event container, declaring the treatment or the censoring when asked."""
    declared: dict[str, Any] = {}
    if treatment:
        declared["treatment_probabilities"] = {1: "p1", 0: "p0"}
    if censoring:
        declared["censoring_probabilities"] = {float(g): f"r{g}" for g in range(1, law.K)}
    return LongitudinalData.from_time_to_event(
        frame,
        time="time",
        event="event",
        treatment="A",
        baseline=["W1", "W2"],
        grid=list(law.GRID),
        **declared,
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    configuration, _ = SCENARIOS[scenario]
    return law.sample(n, seed, trial=configuration == "known"), dict(law.TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# --------------------------------------------------------------------------- the fits


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=2000)


def settings(configuration: str) -> dict[str, Any]:
    """The learners and settings both engines share in one configuration."""
    outcome: Any = _logistic()
    if configuration == "known":
        # The per-node hazard design is [W1, W2, A]: drop W2.
        outcome = Columns(_logistic(), keep=(0, 2))
    return {
        "reference": law.REFERENCE,
        "horizons": [float(law.HORIZON)],
        "outcome_learner": outcome,
        "treatment_learner": _logistic(),
        "censoring_learner": _logistic(),
        "n_folds": N_FOLDS,
        "g_bounds": G_BOUNDS,
        "random_state": 0,
    }


def fit(frame: pd.DataFrame, scenario: str, implementation: str = SUBJECT) -> Any:
    """One engine's fit of one replicate."""
    configuration, _ = SCENARIOS[scenario]
    known = configuration == "known"
    data = container(frame, treatment=known, censoring=known)
    common = settings(configuration)
    if implementation == SUBJECT:
        return SurvivalDRTMLE(law.ARMS, **common).fit(data)
    return LTMLE(law.ARMS, survival_model="hazard", simultaneous=False, **common).fit(data)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per estimand on the identity scale."""
    rows = []
    for name in law.ESTIMANDS:
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
                "initial_estimate": float(estimate.psi),
            }
        )
    return rows


#: The columns a primary row carries beside the published schema.
EXTRA_COLUMNS = (
    "seconds",
    f"seconds[{HAZARD}]",
    f"estimate[{HAZARD}]",
    f"failure[{HAZARD}]",
    "n_iter_a1",
    "n_iter_a0",
    "converged",
    "floored",
)


def _readings(result: Any) -> dict[str, float]:
    records = {arm: result.drift[(arm, law.HORIZON)] for arm in law.ARMS}
    floored = sum(
        sum(count for _, count in record.denominator_hits) + record.g_floor_hits
        for record in records.values()
    )
    return {
        "n_iter_a1": float(records["a1"].n_iter),
        "n_iter_a0": float(records["a0"].n_iter),
        "converged": float(all(record.converged for record in records.values())),
        "floored": float(floored),
    }


def _hazard(frame: pd.DataFrame, scenario: str) -> tuple[Any | None, float, str]:
    """Hazard ``LTMLE`` on the replicate; only a recordable failure is a reading."""
    start = time.perf_counter()
    try:
        result = fit(frame, scenario, HAZARD)
    except Exception as error:
        if not recordable(error):
            raise
        return None, time.perf_counter() - start, type(error).__name__
    return result, time.perf_counter() - start, ""


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
        other, other_seconds, failure = _hazard(frame, scenario)
    extra = {
        "seconds": seconds,
        f"seconds[{HAZARD}]": other_seconds,
        f"failure[{HAZARD}]": failure,
        **_readings(result),
    }
    rows = []
    for row in result_rows(result, truth, scenario, replicate):
        value = np.nan if other is None else float(other[row["estimand"]].psi)
        rows.append({**row, **extra, f"estimate[{HAZARD}]": value})
    return rows


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


def paired_difference(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell and estimand: both RMSEs and the mean paired difference in pooled SDs."""
    records = []
    for (scenario, estimand), group in rows.groupby(["scenario", "estimand"], sort=True):
        kept = group.loc[np.isfinite(group[f"estimate[{HAZARD}]"].astype(float))]
        subject = kept["estimate"].to_numpy(dtype=float)
        other = kept[f"estimate[{HAZARD}]"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        pooled = float(np.sqrt((subject.var(ddof=1) + other.var(ddof=1)) / 2.0))
        records.append(
            {
                "scenario": scenario,
                "estimand": estimand,
                "replicates": len(kept),
                "hazard_failures": len(group) - len(kept),
                "rmse_subject": float(np.sqrt(np.mean((subject - truth) ** 2))),
                "rmse_hazard": float(np.sqrt(np.mean((other - truth) ** 2))),
                "paired_difference_sd": float(np.mean(subject - other) / pooled),
            }
        )
    return pd.DataFrame(records)


def drift_readings(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell: the loop's mean iterations, its convergence share, floors and run times."""
    fits = rows.drop_duplicates(["scenario", "replicate"])
    return (
        fits.groupby("scenario", sort=True)
        .agg(
            fits=("replicate", "size"),
            mean_n_iter_a1=("n_iter_a1", "mean"),
            mean_n_iter_a0=("n_iter_a0", "mean"),
            converged_share=("converged", "mean"),
            fits_floored=("floored", lambda values: int((values > 0).sum())),
            mean_seconds=("seconds", "mean"),
            mean_seconds_hazard=(f"seconds[{HAZARD}]", "mean"),
        )
        .reset_index()
    )


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The two reported tables; neither is a verdict."""
    return {PAIRED_FILE: paired_difference(rows), READINGS_FILE: drift_readings(rows)}

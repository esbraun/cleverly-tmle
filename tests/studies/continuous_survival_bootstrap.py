r"""Registered evidence for the full-refit bootstrap of the continuous-time TMLE (X6 part (d)).

The subject is :class:`~cleverly.survival.ContinuousSurvivalTMLE` with ``n_bootstrap=500``
(:data:`N_BOOTSTRAP`), its default libraries and settings otherwise, on the registered study's base
law (:mod:`tests.studies.continuous_survival_law`) at :math:`n = 1{,}000`.  Each replicate of the
study is one bootstrap fit of a fresh draw: the fit, then 500 refits on resamples of its units,
every fold drawn on the resample's distinct units (X6 plan rev 5, sections 2.4 and 5.5).

**Cells** (:data:`CELLS`):

* ``b1_in_sample``: ``cross_fit=False``, gated.  The bootstrap delta method would cover it, given
  conditions not yet read first-hand, so its label stays experimental whatever this study reads
  (plan 8.9);
* ``b2_cross_fitted``: the default cross-fitted fit, reported only
  (``StudyRecord.reported_scenarios``).  No theorem covers it and Cai and van der Laan (2020) warn
  against it, so no study can move its label.

**Rows.** Every risk and risk difference at the three target times (0.46, 1.12, 2.32), 18 per cell
(:data:`ESTIMANDS`).  A primary row is the percentile interval of the bootstrap draws with the
bootstrap standard error (their SD), read by the primary rule of :data:`MARGINS` in ``b1``.  Each
row also carries the BCa interval on the same draws and the shipped Wald interval of the same fit
(:data:`EXTRA_COLUMNS`).

**Readings** (``calibration.csv``, :func:`calibration`).  Every row of ``b1`` takes the
calibration rule: the 99% interval of its SE ratio inside (0.93, 1.07) and its coverage interval
inside (0.92, 0.98).  The shrunk-SE control multiplies each bootstrap SE by
:data:`SHRUNKEN_SE_FACTOR` (0.85) and must leave the SE-ratio band below.  ``b1``'s calibration
rows and controls are a merge gate (:func:`scientific_failures`); the BCa and Wald readings and every
``b2`` reading are reported.

**Size and seeds.**  :data:`PRIMARY_REPLICATES` is the family-wise rule of plan 5.5: the smallest
multiple of 400 at which all 18 calibration rows of ``b1`` pass together with probability at least
0.95 at SE ratio 1 and coverage 0.95 (``pilot.at_target_count``: 3,200).  The pilot
(:mod:`tests.studies.continuous_survival_bootstrap_pilot`) reads each row's plug-in pass at that
count, the failures and the seconds per fit.  :data:`SEED` and :data:`RESAMPLING_SEED` are used by
no other study; replicate ``r`` of a cell draws from ``replicate_seed(STUDY, cell, r)`` and seeds
its bootstrap with the same value.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from cleverly.exceptions import DataError
from cleverly.survival import ContinuousSurvivalTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.performance import se_ratio_interval
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The family-wise rule's count (``pilot.at_target_count``), checked by the design test.
PRIMARY_REPLICATES = 3_200
#: The plan's cap per cell (the registered study's 7,600).  A count above it stops the declaration.
REPLICATE_CAP = 7_600
#: The plan's pilot of ``b1`` (5.5).
PILOT_REPLICATES = 400
#: The pilot of ``b2``, which reads its failures and seconds only.
PILOT_CROSS_FITTED = 50
#: The bootstrap replicates of every declared fit: the value the docs recommend (plan 5.5).
N_BOOTSTRAP = 500
#: The bootstrap replicates of a pilot fit.  The pilot reads the SE ratio and coverage of the
#: percentile interval, whose Monte Carlo error at 200 draws adds about 5% relative noise to each
#: fit's bootstrap SE and moves the mean ratio by under 0.2%; 500 would cost 2.5 times as much.
PILOT_BOOTSTRAP = 200
PRIMARY_N = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.  A fit whose
#: bootstrap is withheld (more than 1% of its refits failed) is one.
FAILURE_BOUND = 0.01
SEED = 2026611661
RESAMPLING_SEED = 2026611662
N_FOLDS = registered.N_FOLDS
LEARNER_FOLDS = registered.LEARNER_FOLDS
SHRUNKEN_SE_FACTOR = 0.85

SUBJECT = "cleverly-continuous-survival-bootstrap"

#: ``cell -> cross-fitted``.
CELLS: dict[str, bool] = {"b1_in_sample": False, "b2_cross_fitted": True}
GATED_CELL = "b1_in_sample"
REPORTED_CELLS = frozenset({"b2_cross_fitted"})
#: Every risk and risk difference at the three target times, in the engine's names.
ROWS: tuple[str, ...] = tuple(law.TRUTH)
ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(CELLS, ROWS)
MARGINS = Margins(standardized_bias=0.2)

CALIBRATION_FILE = "calibration.csv"
READINGS_FILE = "targeting-readings.csv"
#: The audit that gates ``b1``'s calibration rows and controls.
CALIBRATION_AUDIT = "bootstrap calibration"

STUDY = StudyRecord(
    name="continuous-time survival full-refit bootstrap",
    slug="continuous-survival-bootstrap",
    artifacts=ROOT / "tests" / "canonical" / "continuous_survival_bootstrap",
    document="docs/technical-reference/method-evidence/continuous-survival-bootstrap.md",
    anchor="continuous-time-survival-full-refit-bootstrap",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation=SUBJECT,
    reference=None,
    reported_scenarios=REPORTED_CELLS,
    failure_bound={SUBJECT: FAILURE_BOUND},
    nuisance_count=3,
    extra_artifacts=(CALIBRATION_FILE, READINGS_FILE),
    modules=(
        "tests/studies/continuous_survival_bootstrap.py",
        "tests/studies/continuous_survival_bootstrap_properties.py",
        "tests/studies/continuous_survival.py",
        "tests/studies/continuous_survival_law.py",
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
    runner_module="tests.studies.continuous_survival_bootstrap",
    properties_module="tests.studies.continuous_survival_bootstrap_properties",
)

CONFIGURATION = {
    "construction": "full-refit bootstrap of the Rytgaard and van der Laan (2024) one-step "
    "TMLE; every fold of a replicate drawn on its distinct units; variance='switched'",
    "cells": {cell: "cross-fitted" if cf else "in-sample" for cell, cf in CELLS.items()},
    "n_bootstrap": N_BOOTSTRAP,
    "interval": "percentile (gated), BCa and Wald on the same fits (reported)",
    "times": list(law.TIMES),
    "n_folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "learners": "cox.default_library() for each hazard; logistic + lasso C in (1, 0.1, 0.03)",
    "sizes": dict.fromkeys(CELLS, PRIMARY_N),
    "shrunken_se_factor": SHRUNKEN_SE_FACTOR,
}


# -------------------------------------------------------------------------- the draws


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return registered.sample(n, seed), dict(law.TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# --------------------------------------------------------------------------- the fits


def engine(*, cross_fit: bool, n_bootstrap: int, seed: int) -> ContinuousSurvivalTMLE:
    """The default fit of the base law with a bootstrap of ``n_bootstrap`` refits.

    ``seed`` seeds the folds and the bootstrap draws; the replicates run on one process each,
    since the study's workers are processes already.
    """
    return ContinuousSurvivalTMLE(
        law.TIMES,
        arms=law.ARMS,
        reference=law.REFERENCE,
        causes=law.CAUSES,
        cross_fit=cross_fit,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        simultaneous=False,
        random_state=int(seed),
        n_bootstrap=n_bootstrap,
        n_jobs=1,
    )


def bootstrap_fit(frame: pd.DataFrame, *, cross_fit: bool, n_bootstrap: int, seed: int) -> Any:
    """One bootstrap fit; a withheld bootstrap is a recordable failure of the fit."""
    result = registered.fit_with(
        frame, engine(cross_fit=cross_fit, n_bootstrap=n_bootstrap, seed=seed)
    )
    if result.bootstrap_status != "ran":
        raise DataError(
            f"the bootstrap was withheld: {len(result.bootstrap_failures)} of {n_bootstrap} "
            "refits failed"
        )
    return result


def interval_readings(result: Any, truth: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """Per row: the Wald, percentile and BCa intervals of one bootstrap fit, with coverage.

    The percentile and BCa limits are read from the fit's draws by the shared
    :meth:`~cleverly.inference.BootstrapResult.interval`, the BCa acceleration from the
    estimate's own influence curve, as the engine attaches them.
    """
    from cleverly.inference.bootstrap import bca_acceleration

    out = {}
    for name in ROWS:
        estimate = result[name]
        reference = float(truth[name])
        low, high = estimate.ci
        percentile = estimate.bootstrap
        bca = result.bootstrap.interval(
            name,
            estimate.alpha,
            interval="bca",
            estimate=estimate.psi,
            acceleration=bca_acceleration(estimate.influence_curve),
        )
        out[name] = {
            "estimate": float(estimate.psi),
            "wald_std_error": float(estimate.std_error),
            "wald_lower": float(low),
            "wald_upper": float(high),
            "wald_covered": float(low <= reference <= high),
            "bootstrap_std_error": float(percentile.std_error),
            "percentile_lower": float(percentile.ci[0]),
            "percentile_upper": float(percentile.ci[1]),
            "percentile_covered": float(percentile.ci[0] <= reference <= percentile.ci[1]),
            "bca_lower": float(bca.ci[0]),
            "bca_upper": float(bca.ci[1]),
            "bca_covered": float(bca.ci[0] <= reference <= bca.ci[1]),
        }
    return out


#: The per-row readings beside the published schema.
INTERVAL_COLUMNS = (
    "wald_std_error",
    "wald_lower",
    "wald_upper",
    "wald_covered",
    "bca_lower",
    "bca_upper",
    "bca_covered",
)
EXTRA_COLUMNS = (
    *INTERVAL_COLUMNS,
    "seconds",
    "failed_refits",
    "steps",
    "halvings",
    "converged",
    "bounded_share",
    "last_time_floor_share",
    "frozen_share",
    "off_grid_share",
)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One row per target: the percentile interval and the bootstrap SE, with the other two."""
    rows = []
    initial = result.initial_estimates
    for name, reading in interval_readings(result, truth).items():
        rows.append(
            {
                "implementation": SUBJECT,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": float(truth[name]),
                "estimate": reading["estimate"],
                "inference_estimate": reading["estimate"],
                "std_error": reading["bootstrap_std_error"],
                "ci_lower": reading["percentile_lower"],
                "ci_upper": reading["percentile_upper"],
                "inference_scale": "identity",
                "covered": int(reading["percentile_covered"]),
                "initial_estimate": float(initial[name]),
                **{column: reading[column] for column in INTERVAL_COLUMNS},
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result = bootstrap_fit(
            frame,
            cross_fit=CELLS[scenario],
            n_bootstrap=N_BOOTSTRAP,
            seed=replicate_seed(STUDY, scenario, replicate),
        )
        seconds = time.perf_counter() - start
    extra = {
        "seconds": seconds,
        "failed_refits": float(result.bootstrap.n_failed),
        **registered.readings(result),
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


# --------------------------------------------------------------- the calibration readings

#: ``variant -> (estimate's SE column, lower, upper, role)`` of the calibration table.  The
#: shrunk control scales the percentile interval about the estimate by the same factor as the SE.
VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "percentile": ("std_error", "ci_lower", "ci_upper", "positive"),
    "shrunken_se_control": ("std_error", "ci_lower", "ci_upper", "control"),
    "bca": ("std_error", "bca_lower", "bca_upper", "reported"),
    "wald": ("wald_std_error", "wald_lower", "wald_upper", "reported"),
}


def calibration(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell, row and interval: SE ratio and coverage with their 99% intervals and verdict.

    A ``positive`` row of ``b1`` passes when both intervals lie inside the calibration bands; a
    ``control`` row of ``b1`` passes when its SE ratio's interval lies wholly below the band; a
    ``reported`` row, and every row of ``b2``, carries no verdict (``gated`` false, ``passed``
    true by declaration).
    """
    margins = STUDY.margins
    low, high = margins.calibration_se_ratio
    records = []
    for (scenario, estimand), group in rows.groupby(["scenario", "estimand"], sort=True):
        estimate = group["estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        for variant, (se_column, lower, upper, role) in VARIANTS.items():
            factor = SHRUNKEN_SE_FACTOR if variant == "shrunken_se_control" else 1.0
            std_error = factor * group[se_column].to_numpy(dtype=float)
            bottom = estimate - factor * (estimate - group[lower].to_numpy(dtype=float))
            top = estimate + factor * (group[upper].to_numpy(dtype=float) - estimate)
            covered = int(np.sum((bottom <= truth) & (truth <= top)))
            ratio = se_ratio_interval(
                estimate,
                std_error,
                record=STUDY,
                seed=stream_seed(STUDY, "calibration", str(scenario), str(estimand), variant),
            )
            coverage = clopper_pearson(
                covered, len(group), confidence_level=margins.confidence_level
            )
            gated = scenario == GATED_CELL and role != "reported"
            if role == "positive":
                verdict = ratio.within(low, high) and coverage.within(*margins.calibration_coverage)
            else:
                verdict = ratio.high < low
            records.append(
                {
                    "scenario": scenario,
                    "estimand": estimand,
                    "variant": variant,
                    "role": role,
                    "fits": len(group),
                    "se_ratio": float(np.mean(std_error) / np.std(estimate, ddof=1)),
                    "se_ratio_ci_lower": ratio.low,
                    "se_ratio_ci_upper": ratio.high,
                    "coverage": covered / len(group),
                    "coverage_ci_lower": coverage.low,
                    "coverage_ci_upper": coverage.high,
                    "gated": gated,
                    "passed": bool(verdict) if gated else True,
                }
            )
    return pd.DataFrame.from_records(records)


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The calibration readings and the targeting record."""
    return {
        CALIBRATION_FILE: calibration(rows),
        READINGS_FILE: registered.targeting_readings(rows),
    }


def scientific_failures(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The gated calibration rows and controls of ``b1`` that did not pass."""
    table = artifacts[CALIBRATION_FILE]
    return {CALIBRATION_AUDIT: table.loc[table["gated"].astype(bool) & ~table["passed"]]}


def account_failures(
    failures: pd.DataFrame | None, artifacts: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """The calibration audit on the surviving fits; a lost fit counts against the bound only."""
    return evidence_failures.audits(scientific_failures(artifacts), missing={})

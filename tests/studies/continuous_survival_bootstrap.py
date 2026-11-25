r"""Registered evidence for the full-refit bootstrap of the continuous-time TMLE (X6 part (d)).

The subject is :class:`~cleverly.survival.ContinuousSurvivalTMLE` with ``n_bootstrap=500``
(:data:`N_BOOTSTRAP`), its default libraries and settings otherwise, on the registered study's base
law (:mod:`tests.studies.continuous_survival_law`) at :math:`n = 1{,}000`.  Each replicate of the
study is one bootstrap fit of a fresh draw: the fit, then 500 refits on resamples of its units,
every fold drawn on the resample's distinct units (X6 plan rev 5, sections 2.4 and 5.5).

**Cell** (:data:`CELLS`): ``b1_in_sample``, ``cross_fit=False``, gated by the primary rule.  The
bootstrap delta method would cover it, given conditions not yet read first-hand, so its label stays
experimental whatever this study reads (plan 8.9).  The default cross-fitted fit is a record of its
own, reported only (:mod:`tests.studies.continuous_survival_bootstrap_cross_fitted`): no theorem
covers it and Cai and van der Laan (2020) warn against it, so no study can move its label.

**Rows.** Every risk and risk difference at the median and upper quartile of the event times
(1.12, 2.32), 12 rows (:data:`ROWS`, the registered study's
:data:`~tests.studies.continuous_survival_law.GATED`).  A primary row is the percentile interval of
the bootstrap draws with the bootstrap standard error (their SD), read by the primary rule of
:data:`MARGINS`: the lower 99% limit of its coverage at least 0.90, the 99% interval of its SE
ratio inside (0.80, 1.20) and its bias within 0.2 empirical SDs.  That rule is the record's only
gate.  Each row also carries the BCa interval on the same draws and the shipped Wald
interval of the same fit (:data:`EXTRA_COLUMNS`).  The six targets at the first quartile (0.46)
come from the same fits and publish as ``early-time.csv`` with no verdict (:data:`EARLY`,
:func:`early_time`), as in the registered study: the first quartile's Wald interval
under-covers, and gating it would inform no label.

**Readings** (``calibration.csv``, :func:`calibration`).  Every row is read by the calibration
rule: the 99% interval of its SE ratio inside (0.93, 1.07) and its coverage interval inside
(0.92, 0.98).  The shrunk-SE control multiplies each bootstrap SE by :data:`SHRUNKEN_SE_FACTOR`
(0.85) and should leave the SE-ratio band below.  The table publishes each reading
(``meets_rule``) and no verdict: the calibration rule and the control are reported and cannot stop
the run, and the record has no ``scientific_failures`` hook.

**Paired readings** (``paired-wald.csv``, :func:`paired_wald`).  Per row, the percentile interval
against the Wald interval of the same fits: the mean bootstrap SE over the mean Wald SE with a 99%
delta-method interval, and the percentile coverage minus the Wald coverage with its discordant
counts and Newcombe's (1998) hybrid score interval for paired proportions.  Each carries a
reading against a declared margin (:data:`PAIRED_SE_MARGIN`, :data:`PAIRED_COVERAGE_MARGIN`).
They carry no verdict: the regeneration script's red-cell readings name them as the declared
diagnosis of a red primary row or a calibration row that misses the rule.

**Size and seeds.**  :data:`PRIMARY_REPLICATES` is kept at the calibration rule's at-target count
of plan 5.5, the smallest multiple of 400 at which all 12 calibration rows would pass together
with probability at least 0.95 at SE ratio 1 and coverage 0.95 (``pilot.at_target_count``: 3,200),
so that the reported calibration reading keeps the resolution it was sized for.  The primary rule
needs far fewer fits.  The pilot (:mod:`tests.studies.continuous_survival_bootstrap_pilot`) reads
each row's plug-in pass at that count, the failures and the seconds per fit.  :data:`SEED` and
:data:`RESAMPLING_SEED` are used by no other study; replicate ``r`` of a cell draws from
``replicate_seed(STUDY, cell, r)`` and seeds its bootstrap with the same value.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.exceptions import DataError
from cleverly.survival import ContinuousSurvivalTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.performance import se_ratio_interval
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

#: The calibration rule's at-target count (``pilot.at_target_count``), kept for the reported
#: calibration reading and checked by the design test.
PRIMARY_REPLICATES = 3_200
#: The plan's cap per cell (the registered study's 7,600).  A count above it stops the declaration.
REPLICATE_CAP = 7_600
#: The plan's pilot of ``b1`` (5.5).
PILOT_REPLICATES = 400
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
CELLS: dict[str, bool] = {"b1_in_sample": False}
GATED_CELL = "b1_in_sample"
#: Every risk and risk difference at the median and upper quartile, in the engine's names.
ROWS: tuple[str, ...] = law.GATED
#: The first quartile's six targets, read from the same fits and reported in ``early-time.csv``.
EARLY: tuple[str, ...] = law.EARLY
#: Every risk and risk difference at the three target times.
ALL_ROWS: tuple[str, ...] = tuple(law.TRUTH)
ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(CELLS, ROWS)
MARGINS = Margins(standardized_bias=0.2)

CALIBRATION_FILE = "calibration.csv"
READINGS_FILE = "targeting-readings.csv"
EARLY_FILE = "early-time.csv"
PAIRED_FILE = "paired-wald.csv"
#: The equivalence margin of the paired SE ratio (mean bootstrap SE over mean Wald SE on the same
#: fits): its 99% interval inside it reads the bootstrap SE as the Wald SE's.
PAIRED_SE_MARGIN = (0.95, 1.05)
#: The non-inferiority margin of the paired coverage difference (percentile minus Wald on the
#: same fits): its lower 99% limit above it reads the percentile coverage as no worse than the
#: Wald coverage.
PAIRED_COVERAGE_MARGIN = -0.025

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
    failure_bound={SUBJECT: FAILURE_BOUND},
    nuisance_count=3,
    extra_artifacts=(CALIBRATION_FILE, READINGS_FILE, EARLY_FILE, PAIRED_FILE),
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
    "rows": "risks and differences at the median and upper quartile (gated); first quartile "
    "reported in early-time.csv",
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


def interval_readings(
    result: Any, truth: Mapping[str, float], rows: tuple[str, ...] = ROWS
) -> dict[str, dict[str, float]]:
    """Per row: the Wald, percentile and BCa intervals of one bootstrap fit, with coverage.

    The percentile and BCa limits are read from the fit's draws by the shared
    :meth:`~cleverly.inference.BootstrapResult.interval`, the BCa acceleration from the
    estimate's own influence curve, as the engine attaches them.
    """
    from cleverly.inference.bootstrap import bca_acceleration

    out = {}
    for name in rows:
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
#: The fit's own readings beside each row.
FIT_COLUMNS = (
    "seconds",
    "failed_refits",
    "jump_bounded_refits",
    "steps",
    "halvings",
    "converged",
    "bounded_share",
    "last_time_floor_share",
    "jump_bound_share",
    "jump_bound_at_risk_share",
    "frozen_share",
    "off_grid_share",
)
#: Per first-quartile target, the readings ``early-time.csv`` reads.
EARLY_FIELDS = (
    "estimate",
    "bootstrap_std_error",
    "percentile_covered",
    "bca_covered",
    "wald_std_error",
    "wald_covered",
)
EARLY_COLUMNS = tuple(
    f"early{index}_{field}" for index in range(len(EARLY)) for field in EARLY_FIELDS
)
EXTRA_COLUMNS = (*INTERVAL_COLUMNS, *FIT_COLUMNS, *EARLY_COLUMNS)


def early_readings(result: Any, truth: Mapping[str, float]) -> dict[str, float]:
    """The first quartile's targets of one fit, as :data:`EARLY_COLUMNS`."""
    readings = interval_readings(result, truth, EARLY)
    return {
        f"early{index}_{field}": readings[name][field]
        for index, name in enumerate(EARLY)
        for field in EARLY_FIELDS
    }


def result_rows(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    *,
    rows: tuple[str, ...] = ROWS,
    implementation: str = SUBJECT,
) -> list[dict[str, Any]]:
    """One row per target: the percentile interval and the bootstrap SE, with the other two."""
    out = []
    initial = result.initial_estimates
    for name, reading in interval_readings(result, truth, rows).items():
        out.append(
            {
                "implementation": implementation,
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
    return out


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def fit_replicate(
    record: StudyRecord,
    cross_fit: bool,
    rows: tuple[str, ...],
    early: bool,
    key: tuple[str, int, int],
    drawn: tuple[pd.DataFrame, dict[str, float]],
) -> list[dict[str, Any]]:
    """One bootstrap fit of ``record``'s replicate ``key``: its rows, with the first quartile's
    readings beside each when ``early``."""
    scenario, replicate, _ = key
    frame, truth = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        result = bootstrap_fit(
            frame,
            cross_fit=cross_fit,
            n_bootstrap=N_BOOTSTRAP,
            seed=replicate_seed(record, scenario, replicate),
        )
        seconds = time.perf_counter() - start
    extra = {
        "seconds": seconds,
        "failed_refits": float(result.bootstrap.n_failed),
        "jump_bounded_refits": float(result.bootstrap_jump_bounded),
        **registered.readings(result),
        **(early_readings(result, truth) if early else {}),
    }
    found = result_rows(
        result, truth, scenario, replicate, rows=rows, implementation=record.implementation
    )
    return [{**row, **extra} for row in found]


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    return fit_replicate(STUDY, CELLS[key[0]], ROWS, True, key, drawn)


def fit_cells(
    record: StudyRecord,
    replicate: Any,
    draw: Any,
    columns: tuple[str, ...],
    *,
    replicates: int,
    n: int,
    n_jobs: int,
    scenarios: tuple[str, ...] | None,
) -> replication.PhaseResult:
    """Every cell of ``record`` named in ``scenarios`` (all by default), at its declared size."""
    sizes = record.sizes(n)
    chosen = tuple(record.scenarios) if scenarios is None else scenarios
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in chosen
            for replicate in range(replicates)
        ],
        replicate,
        draw=draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=record.implementation,
        seed_of=lambda key: replicate_seed(record, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *columns]])


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    scenarios: tuple[str, ...] | None = None,
) -> replication.PhaseResult:
    """Every cell of ``scenarios`` (all by default), each at its declared size."""
    return fit_cells(
        STUDY,
        _replicate,
        _draw,
        EXTRA_COLUMNS,
        replicates=replicates,
        n=n,
        n_jobs=n_jobs,
        scenarios=scenarios,
    )


# --------------------------------------------------------------- the calibration readings

#: ``variant -> (estimate's SE column, lower, upper, role)`` of the calibration table.  The
#: shrunk control scales the percentile interval about the estimate by the same factor as the SE.
VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "percentile": ("std_error", "ci_lower", "ci_upper", "positive"),
    "shrunken_se_control": ("std_error", "ci_lower", "ci_upper", "control"),
    "bca": ("std_error", "bca_lower", "bca_upper", "reported"),
    "wald": ("wald_std_error", "wald_lower", "wald_upper", "reported"),
}


def calibration(rows: pd.DataFrame, record: StudyRecord = STUDY) -> pd.DataFrame:
    """Per cell, row and interval: SE ratio and coverage with their 99% intervals, read by the
    calibration rule.

    ``meets_rule`` is true for a ``positive`` or ``reported`` row when both intervals lie inside
    the calibration bands, and for the ``control`` row when its SE ratio's interval lies wholly
    below the band.  It is a reading, not a verdict: no row of either record gates.
    """
    margins = record.margins
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
                record=record,
                seed=stream_seed(record, "calibration", str(scenario), str(estimand), variant),
            )
            coverage = clopper_pearson(
                covered, len(group), confidence_level=margins.confidence_level
            )
            if role == "control":
                reading = ratio.high < low
            else:
                reading = ratio.within(low, high) and coverage.within(*margins.calibration_coverage)
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
                    "meets_rule": bool(reading),
                }
            )
    return pd.DataFrame.from_records(records)


def _wilson(successes: int, trials: int, z: float) -> tuple[float, float]:
    """The Wilson score interval of a proportion at normal quantile ``z``."""
    share = successes / trials
    centre = (share + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * np.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials))
    return float(centre - half / (1 + z * z / trials)), float(centre + half / (1 + z * z / trials))


def paired_difference_interval(
    both: int, first_only: int, second_only: int, neither: int, *, confidence_level: float
) -> tuple[float, float]:
    """Newcombe's (1998) hybrid score interval for a difference of paired proportions.

    Method 10 of Newcombe (1998), *Statistics in Medicine* 17(22), 2635-2650: the Wilson
    intervals of the two marginal proportions, combined with the phi correlation of the 2 x 2
    table of pairs (0 when a margin is empty).  The difference is first minus second.
    """
    trials = both + first_only + second_only + neither
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    first = (both + first_only) / trials
    second = (both + second_only) / trials
    low_first, high_first = _wilson(both + first_only, trials, z)
    low_second, high_second = _wilson(both + second_only, trials, z)
    margins = float(
        (both + first_only)
        * (second_only + neither)
        * (both + second_only)
        * (first_only + neither)
    )
    phi = (both * neither - first_only * second_only) / np.sqrt(margins) if margins > 0 else 0.0
    down = np.sqrt(
        max(
            (first - low_first) ** 2
            - 2 * phi * (first - low_first) * (high_second - second)
            + (high_second - second) ** 2,
            0.0,
        )
    )
    up = np.sqrt(
        max(
            (high_first - first) ** 2
            - 2 * phi * (high_first - first) * (second - low_second)
            + (second - low_second) ** 2,
            0.0,
        )
    )
    return float(first - second - down), float(first - second + up)


def paired_wald(rows: pd.DataFrame, record: StudyRecord = STUDY) -> pd.DataFrame:
    r"""Per cell and row: the percentile interval against the Wald interval of the same fits.

    Each fit gives one pair.  ``paired_se_ratio`` is the mean bootstrap SE over the mean Wald SE,
    with a delta-method interval at the record's confidence level: the SE of a ratio of means
    :math:`\bar b / \bar w` is the SD of :math:`(b_i - r w_i) / \bar w` over :math:`\sqrt m`,
    which keeps the covariance of a fit's two SEs.  ``coverage_difference`` is the percentile
    coverage minus the Wald coverage, ``percentile_only`` and ``wald_only`` count the discordant
    fits, and its interval is :func:`paired_difference_interval`.  Both intervals are read against
    the declared margins (:data:`PAIRED_SE_MARGIN`, :data:`PAIRED_COVERAGE_MARGIN`); no row
    carries a verdict.
    """
    level = record.margins.confidence_level
    z = float(norm.ppf(0.5 + level / 2.0))
    records = []
    for (scenario, estimand), group in rows.groupby(["scenario", "estimand"], sort=True):
        truth = float(group["truth"].iloc[0])
        boot = group["std_error"].to_numpy(dtype=float)
        wald = group["wald_std_error"].to_numpy(dtype=float)
        fits = len(group)
        ratio = float(np.mean(boot) / np.mean(wald))
        spread = float(np.std((boot - ratio * wald) / np.mean(wald), ddof=1) / np.sqrt(fits))
        percentile = (group["ci_lower"].to_numpy(dtype=float) <= truth) & (
            truth <= group["ci_upper"].to_numpy(dtype=float)
        )
        shipped = (group["wald_lower"].to_numpy(dtype=float) <= truth) & (
            truth <= group["wald_upper"].to_numpy(dtype=float)
        )
        both = int(np.sum(percentile & shipped))
        percentile_only = int(np.sum(percentile & ~shipped))
        wald_only = int(np.sum(~percentile & shipped))
        neither = fits - both - percentile_only - wald_only
        low, high = paired_difference_interval(
            both, percentile_only, wald_only, neither, confidence_level=level
        )
        records.append(
            {
                "scenario": scenario,
                "estimand": estimand,
                "fits": fits,
                "mean_bootstrap_se": float(np.mean(boot)),
                "mean_wald_se": float(np.mean(wald)),
                "paired_se_ratio": ratio,
                "paired_se_ratio_ci_lower": ratio - z * spread,
                "paired_se_ratio_ci_upper": ratio + z * spread,
                "percentile_coverage": float(np.mean(percentile)),
                "wald_coverage": float(np.mean(shipped)),
                "coverage_difference": (percentile_only - wald_only) / fits,
                "percentile_only": percentile_only,
                "wald_only": wald_only,
                "coverage_difference_ci_lower": low,
                "coverage_difference_ci_upper": high,
                "se_ratio_within_margin": bool(
                    PAIRED_SE_MARGIN[0] < ratio - z * spread
                    and ratio + z * spread < PAIRED_SE_MARGIN[1]
                ),
                "coverage_difference_within_margin": bool(low > PAIRED_COVERAGE_MARGIN),
            }
        )
    return pd.DataFrame.from_records(records)


def early_time(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell and first-quartile target: bias, both SE ratios and the three coverages.

    No row carries a verdict.
    """
    fits = rows.drop_duplicates(["scenario", "replicate"])
    records = []
    for scenario, group in fits.groupby("scenario", sort=True):
        for index, name in enumerate(EARLY):
            column = f"early{index}_{{}}".format
            estimate = group[column("estimate")].to_numpy(dtype=float)
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
                    "bootstrap_se_ratio": float(
                        np.mean(group[column("bootstrap_std_error")]) / spread
                    ),
                    "wald_se_ratio": float(np.mean(group[column("wald_std_error")]) / spread),
                    "percentile_coverage": float(np.mean(group[column("percentile_covered")])),
                    "bca_coverage": float(np.mean(group[column("bca_covered")])),
                    "wald_coverage": float(np.mean(group[column("wald_covered")])),
                }
            )
    return pd.DataFrame.from_records(records)


def targeting_readings(rows: pd.DataFrame) -> pd.DataFrame:
    """The registered study's targeting record per cell, with how often the jump bound acted in
    the refits: the mean count per fit and the fits with any."""
    fits = rows.drop_duplicates(["scenario", "replicate"])
    refits = (
        fits.groupby("scenario", sort=True)["jump_bounded_refits"]
        .agg(
            mean_jump_bounded_refits="mean",
            fits_with_jump_bounded_refits=lambda counts: int((counts > 0).sum()),
        )
        .reset_index()
    )
    return registered.targeting_readings(rows).merge(refits, on="scenario", how="left")


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The calibration readings, the targeting record, the first quartile's readings and the
    paired readings against the Wald interval."""
    return {
        CALIBRATION_FILE: calibration(rows),
        READINGS_FILE: targeting_readings(rows),
        EARLY_FILE: early_time(rows),
        PAIRED_FILE: paired_wald(rows),
    }

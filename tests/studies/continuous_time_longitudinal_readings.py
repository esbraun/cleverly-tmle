r"""The tables and code-read gates of ``continuous-time-longitudinal`` (X6 plan rev 7, 5.7).

Every table is computed from the primary rows of
:mod:`tests.studies.continuous_time_longitudinal`, whose ``day_cf`` rows carry the secondary fits
(``<cell>_estimate`` and the rest), so the declared run writes them from one phase:

* ``longitudinal-rows.csv.gz``: every primary row with its support readings
  (``variance_rel_sd``, the event share at the row's time, the regime's followers at the last
  target time and its share past the visit cap), from which part (e)'s support constants are read;
* ``secondary-rows.csv.gz``: every secondary fit's rows (:data:`SECONDARY_COLUMNS`);
* ``calibration.csv`` (:func:`calibration`): the calibration rule at the two calibration rows, the
  shrunk-SE control on each, and ``sparse_calibration`` (reported);
* ``mechanism-correct.csv`` (:func:`mechanism`): the primary rule at ``mechanism_correct``'s 12
  rows and the plug-in control at the four day-30 risks of always and never;
* ``misspecified.csv`` (:func:`misspecified`): bias, coverage and the SE ratio of
  ``outcome_correct`` and ``both_wrong``, with no verdict;
* ``ltmle-pairing.csv`` (:func:`pairing`): X70's estimate against the continuous engine's on the
  same draws, the paired difference and the SE ratio, with no verdict.  X70 withholds the
  interval of a row whose recursion read a constant node (``constant_node_plugin``, in one of
  the smoke's two draws, at day 30 of always); its point estimate still pairs, its SE and
  coverage are left out, and ``x70_interval_refused`` counts such draws;
* ``targeting-readings.csv`` (:func:`targeting`): steps, convergence, bounds and seconds per cell.

:func:`gates` reads four gates, each failing on any row it returns: the primary rule of
``mechanism_correct`` (and its failure bound), the plug-in control, the calibration rule and
the shrunk-SE control.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import continuous_time_longitudinal as study
from tests.studies.evidence.inference import clopper_pearson, standardized_bias_verdict
from tests.studies.evidence.performance import independent_performance_tests, se_ratio_interval
from tests.studies.evidence.seeds import stream_seed

MECHANISM_GATE = "mechanism_correct primary rule"
PLUG_IN_GATE = "plug-in control"
CALIBRATION_GATE = "calibration rule"
SHRUNK_GATE = "shrunk-SE control"

#: The per-row readings ``longitudinal-rows.csv.gz`` keeps.
ROW_COLUMNS = (
    "scenario",
    "replicate",
    "estimand",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "initial_estimate",
    "variance_rel_sd",
    "event_share",
    "followers_at_last",
    "past_cap_share",
)
#: The columns of ``secondary-rows.csv.gz``.
SECONDARY_COLUMNS = (
    "cell",
    "replicate",
    "estimand",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "initial",
    "failed",
    "interval_refused",
    "seconds",
)
#: The reported misspecified cells.
MISSPECIFIED = ("outcome_correct", "both_wrong")
#: A mean paired difference above this share of the continuous engine's mean SE starts a
#: diagnosis of both engines on the same draws (plan 5.7's red-cell readings).
PAIRED_DIAGNOSIS = 0.5


def row_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Every primary row with its support readings."""
    return rows.loc[:, list(ROW_COLUMNS)].sort_values(
        ["scenario", "replicate", "estimand"], ignore_index=True
    )


def secondary_table(rows: pd.DataFrame) -> pd.DataFrame:
    """The secondary fits, one row per cell, replicate and target, from ``day_cf``'s rows."""
    day = rows.loc[rows["scenario"] == "day_cf"]
    parts = []
    for cell, (_, count, names) in study.SECONDARY.items():
        chosen = day.loc[day["estimand"].isin(names) & (day["replicate"] < count)]
        parts.append(
            pd.DataFrame(
                {
                    "cell": cell,
                    "replicate": chosen["replicate"].to_numpy(),
                    "estimand": chosen["estimand"].to_numpy(),
                    "truth": chosen["truth"].to_numpy(),
                    **{
                        field: chosen[f"{cell}_{field}"].to_numpy(dtype=float)
                        for field in study.SECONDARY_FIELDS
                    },
                }
            )
        )
    return pd.concat(parts, ignore_index=True).loc[:, list(SECONDARY_COLUMNS)]


def _kept(secondary: pd.DataFrame, cell: str) -> pd.DataFrame:
    chosen = secondary.loc[secondary["cell"] == cell]
    return chosen.loc[chosen["failed"].fillna(1.0) == 0.0]


def _failed_fits(secondary: pd.DataFrame, cell: str) -> int:
    chosen = secondary.loc[secondary["cell"] == cell]
    return int(chosen.loc[chosen["failed"].fillna(1.0) != 0.0, "replicate"].nunique())


def _ratio(values: np.ndarray, errors: np.ndarray, label: str) -> tuple[float, float, float]:
    interval = se_ratio_interval(
        values, errors, record=study.STUDY, seed=stream_seed(study.STUDY, "calibration", label)
    )
    return float(np.mean(errors) / np.std(values, ddof=1)), interval.low, interval.high


def calibration(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration rule at the two calibration rows, their shrunk-SE controls, and the
    sparse row (reported)."""
    margins = study.MARGINS
    low_band, high_band = margins.calibration_se_ratio
    cover_low, cover_high = margins.calibration_coverage
    cells = [
        *(
            (cell, study.CALIBRATION_ROW, "calibration", "positive")
            for cell in study.CALIBRATION_CELLS
        ),
        *(
            (cell, study.CALIBRATION_ROW, "shrunk_se", "control")
            for cell in study.CALIBRATION_CELLS
        ),
        ("day_cf", study.SPARSE_ROW, "sparse_calibration", "reported"),
    ]
    records = []
    for scenario, estimand, kind, role in cells:
        group = rows.loc[(rows["scenario"] == scenario) & (rows["estimand"] == estimand)]
        values = group["estimate"].to_numpy(dtype=float)
        errors = group["std_error"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        if kind == "shrunk_se":
            errors = errors * study.SHRUNKEN_SE_FACTOR
            half = 1.959963984540054 * errors
            covered = (np.abs(values - truth) <= half).astype(int)
        else:
            covered = group["covered"].to_numpy(dtype=int)
        ratio, ratio_low, ratio_high = _ratio(values, errors, f"{scenario}/{kind}/{estimand}")
        coverage = clopper_pearson(
            int(covered.sum()), int(covered.size), confidence_level=margins.confidence_level
        )
        meets = bool(
            low_band <= ratio_low
            and ratio_high <= high_band
            and cover_low <= coverage.low
            and coverage.high <= cover_high
        )
        passed = (not meets and ratio_high < low_band) if kind == "shrunk_se" else meets
        records.append(
            {
                "cell": kind,
                "scenario": scenario,
                "estimand": estimand,
                "role": role,
                "fits": int(values.size),
                "coverage": float(covered.mean()),
                "coverage_ci_lower": coverage.low,
                "coverage_ci_upper": coverage.high,
                "se_ratio": ratio,
                "se_ratio_ci_lower": ratio_low,
                "se_ratio_ci_upper": ratio_high,
                "meets_rule": meets,
                "gated": role != "reported",
                "passed": bool(passed) if role != "reported" else True,
            }
        )
    return pd.DataFrame.from_records(records)


def mechanism(secondary: pd.DataFrame) -> pd.DataFrame:
    """The primary rule at ``mechanism_correct``'s rows and the plug-in control of its fits."""
    kept = _kept(secondary, "mechanism_correct")
    failed = _failed_fits(secondary, "mechanism_correct")
    frame = pd.DataFrame(
        {
            "implementation": study.SUBJECT,
            "scenario": "mechanism_correct",
            "estimand": kept["estimand"].to_numpy(),
            "truth": kept["truth"].to_numpy(),
            "inference_estimate": kept["estimate"].to_numpy(dtype=float),
            "std_error": kept["std_error"].to_numpy(dtype=float),
            "inference_scale": "identity",
            "covered": kept["covered"].to_numpy(dtype=int),
        }
    )
    performance = independent_performance_tests(frame, record=study.STUDY, n_jobs=1)
    allowance = int(np.floor(study.FAILURE_BOUND * study.MECHANISM_REPLICATES))
    records: list[dict[str, Any]] = []
    for _, row in performance.iterrows():
        records.append(
            {
                "cell": "mechanism_correct",
                "estimand": row["estimand"],
                "role": "positive",
                "replicates": int(row["replicates"]),
                "failed_fits": failed,
                "standardized_bias": float(row["standardized_bias"]),
                "bias_ci_lower": float(row["bias_ci_lower"]),
                "bias_ci_upper": float(row["bias_ci_upper"]),
                "bias_margin": float(row["bias_margin"]),
                "bias_equivalent": bool(row["bias_equivalent"]),
                "coverage": float(row["coverage"]),
                "coverage_ci_lower": float(row["coverage_ci_lower"]),
                "coverage_ci_upper": float(row["coverage_ci_upper"]),
                "se_ratio": float(row["se_ratio"]),
                "se_ratio_ci_lower": float(row["se_ratio_ci_lower"]),
                "se_ratio_ci_upper": float(row["se_ratio_ci_upper"]),
                "passed": bool(row["passed"]) and failed <= allowance,
            }
        )
    for name in study.PLUG_IN_ROWS:
        group = kept.loc[kept["estimand"] == name]
        errors = group["initial"].to_numpy(dtype=float) - group["truth"].to_numpy(dtype=float)
        verdict = standardized_bias_verdict(
            errors,
            margin=study.MARGINS.standardized_bias,
            confidence_level=study.MARGINS.confidence_level,
        )
        records.append(
            {
                "cell": "plug_in_control",
                "estimand": name,
                "role": "control",
                "replicates": int(errors.size),
                "failed_fits": failed,
                "standardized_bias": verdict.standardized,
                "bias_ci_lower": verdict.interval.low,
                "bias_ci_upper": verdict.interval.high,
                "bias_margin": verdict.margin,
                "bias_equivalent": verdict.equivalent,
                "passed": bool(verdict.discriminated),
            }
        )
    return pd.DataFrame.from_records(records)


def _performance(chosen: pd.DataFrame, column: str = "estimate") -> dict[str, float]:
    error = chosen[column].to_numpy(dtype=float) - chosen["truth"].to_numpy(dtype=float)
    spread = float(np.std(chosen[column].to_numpy(dtype=float), ddof=1))
    return {
        "fits": int(error.size),
        "truth": float(chosen["truth"].iloc[0]),
        "bias": float(np.mean(error)),
        "standardized_bias": float(np.mean(error) / spread) if spread > 0 else np.nan,
        "empirical_se": spread,
        "mean_std_error": float(chosen["std_error"].mean()),
        "se_ratio": float(chosen["std_error"].mean() / spread) if spread > 0 else np.nan,
        "coverage": float(chosen["covered"].mean()),
    }


def misspecified(secondary: pd.DataFrame) -> pd.DataFrame:
    """``outcome_correct`` and ``both_wrong`` at their rows, with no verdict (plan 5.7, B3)."""
    records = []
    for cell in MISSPECIFIED:
        kept = _kept(secondary, cell)
        for name, group in kept.groupby("estimand", sort=True):
            records.append(
                {
                    "cell": cell,
                    "estimand": name,
                    "failed_fits": _failed_fits(secondary, cell),
                    **_performance(group),
                    "plug_in_standardized_bias": np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def pairing(rows: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """X70 against the continuous engine on the same draws: paired difference and SE ratio."""
    x70 = _kept(secondary, "ltmle_daily")
    day = rows.loc[
        rows["scenario"] == "day_cf", ["replicate", "estimand", "estimate", "std_error", "covered"]
    ]
    joined = x70.merge(day, on=["replicate", "estimand"], suffixes=("_x70", "_continuous"))
    records = []
    for name, group in joined.groupby("estimand", sort=True):
        difference = group["estimate_x70"] - group["estimate_continuous"]
        continuous_se = float(group["std_error_continuous"].mean())
        mean = float(difference.mean())
        records.append(
            {
                "estimand": name,
                "x70_estimand": study.x70_name(str(name)),
                "pairs": len(group),
                "failed_fits": _failed_fits(secondary, "ltmle_daily"),
                "truth": float(group["truth"].iloc[0]),
                "mean_difference": mean,
                "difference_se": float(difference.std(ddof=1) / np.sqrt(len(group))),
                "difference_over_se": mean / continuous_se,
                "se_ratio": float(group["std_error_x70"].mean()) / continuous_se,
                "x70_bias": float(group["estimate_x70"].mean() - group["truth"].iloc[0]),
                "continuous_bias": float(
                    group["estimate_continuous"].mean() - group["truth"].iloc[0]
                ),
                "x70_interval_refused": int(group["interval_refused"].sum()),
                "x70_coverage": float(group["covered_x70"].mean()),
                "continuous_coverage": float(group["covered_continuous"].mean()),
                "diagnose": bool(abs(mean) > PAIRED_DIAGNOSIS * continuous_se),
            }
        )
    return pd.DataFrame.from_records(records)


def targeting(rows: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Per cell: steps, convergence, the bounds, the support readings and run times."""
    fits = rows.drop_duplicates(["scenario", "replicate"])
    table = (
        fits.groupby("scenario", sort=True)
        .agg(
            fits=("replicate", "size"),
            mean_steps=("steps", "mean"),
            converged_share=("converged", "mean"),
            mean_bounded_share=("bounded_share", "mean"),
            mean_jump_bound_share=("jump_bound_share", "mean"),
            mean_visit_bound_share=("visit_bound_share", "mean"),
            largest_cap=("largest_cap", "max"),
            median_seconds=("seconds", "median"),
        )
        .reset_index()
    )
    support = (
        rows.groupby("scenario", sort=True)
        .agg(
            fewest_followers=("followers_at_last", "min"),
            largest_past_cap_share=("past_cap_share", "max"),
            largest_variance_rel_sd=("variance_rel_sd", "max"),
        )
        .reset_index()
    )
    table = table.merge(support, on="scenario")
    seconds = []
    for cell in study.SECONDARY:
        chosen = secondary.loc[(secondary["cell"] == cell) & (secondary["failed"] == 0.0)]
        per_fit = chosen.drop_duplicates("replicate")["seconds"]
        seconds.append(
            {
                "scenario": cell,
                "fits": int(per_fit.size),
                "median_seconds": float(per_fit.median()) if per_fit.size else np.nan,
            }
        )
    return pd.concat([table, pd.DataFrame.from_records(seconds)], ignore_index=True)


def artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every extra artifact of the record, from the primary rows."""
    secondary = secondary_table(rows)
    return {
        study.ROWS_FILE: row_table(rows),
        study.SECONDARY_FILE: secondary,
        study.CALIBRATION_FILE: calibration(rows),
        study.MECHANISM_FILE: mechanism(secondary),
        study.MISSPECIFIED_FILE: misspecified(secondary),
        study.PAIRING_FILE: pairing(rows, secondary),
        study.READINGS_FILE: targeting(rows, secondary),
    }


def gates(artifacts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The four code-read gates, each the rows that fail it."""
    table = artifacts[study.MECHANISM_FILE]
    positive = table.loc[table["role"] == "positive"]
    control = table.loc[table["role"] == "control"]
    calibrated = artifacts[study.CALIBRATION_FILE]
    return {
        MECHANISM_GATE: positive.loc[~positive["passed"].astype(bool)],
        # One day-30 risk of always or never must establish a bias beyond 0.2 SD.
        PLUG_IN_GATE: control.iloc[0:0] if control["passed"].astype(bool).any() else control,
        CALIBRATION_GATE: calibrated.loc[
            (calibrated["role"] == "positive") & ~calibrated["passed"].astype(bool)
        ],
        SHRUNK_GATE: calibrated.loc[
            (calibrated["role"] == "control") & ~calibrated["passed"].astype(bool)
        ],
    }

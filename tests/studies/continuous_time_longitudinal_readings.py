r"""The tables and code-read gates of ``continuous-time-longitudinal`` (X6 plan rev 7, 5.7).

Every table is computed from the primary rows of
:mod:`tests.studies.continuous_time_longitudinal`, whose ``day_cf`` rows carry the secondary fits
(``<cell>_estimate`` and the rest), so the declared run writes them from one phase:

* ``longitudinal-rows.csv.gz``: every primary row with its support readings
  (``variance_rel_sd``, the event share at the row's time, the regime's followers at the last
  target time and its share past the visit cap, the fit's bounded shares, the regime's own cap
  and the capped truth);
* ``secondary-rows.csv.gz``: every secondary fit's rows (:data:`SECONDARY_COLUMNS`);
* ``calibration.csv`` (:func:`calibration`): the calibration rule at the two calibration rows, the
  shrunk-SE control on each, and ``sparse_calibration`` (reported).  A shrunk control leaves the
  band below 0.93, or, when its row reads conservative (the row's SE-ratio interval lies above
  1), it leaves the row's own interval (``reading``, review B1);
* ``mechanism-correct.csv`` (:func:`mechanism`): the primary rule at ``mechanism_correct``'s 12
  rows and the plug-in control at the four day-30 risks of always and never;
* ``misspecified.csv`` (:func:`misspecified`): bias, coverage and the SE ratio of
  ``outcome_correct`` and ``both_wrong`` and the standardized bias of their plug-in, with no
  verdict;
* ``ltmle-pairing.csv`` (:func:`pairing`): X70's estimate against the continuous engine's on the
  same draws, each engine's bias against its own truth (the panel's risk for X70, :math:`\psi_r`
  for the continuous engine), the paired difference against its expected value and the SE ratio,
  with no verdict.  X70 withholds the interval of a row whose recursion read a constant node
  (``constant_node_plugin``, in one of the smoke's two draws, at day 30 of always); its point
  estimate still pairs, its SE and coverage are left out, and the refused share is reported;
* ``targeting-readings.csv`` (:func:`targeting`): steps, convergence, bounds and seconds per cell;
* ``visit-cap.csv`` (:func:`visit_cap`): per cell and row, the caps the fits chose and the bias
  against both the full and the capped truth (review S1);
* ``floors.csv`` (:func:`floors`): per gated primary row, its verdict and the readings from which
  :func:`floor_values` reads part (e)'s support constants after the run (plan 3.5.9, review S7).

:func:`gates` reads four gates, each failing on any row it returns: the primary rule of
``mechanism_correct`` (and its failure bound), the plug-in control, the calibration rule and
the shrunk-SE control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tests.studies import continuous_time_longitudinal as study
from tests.studies import continuous_time_longitudinal_law as laws
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
    "bounded_share",
    "jump_bound_share",
    "cap",
    "reference_cap",
    "capped_truth",
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
    "cap",
    "reference_cap",
    "capped_truth",
    "error",
)
#: The cap readings each fit's rows carry (review S1).
_CAP_COLUMNS = ("cap", "reference_cap", "capped_truth")
#: The columns the primary rule reads.
_PERFORMANCE_COLUMNS = (
    "implementation",
    "scenario",
    "estimand",
    "truth",
    "inference_estimate",
    "std_error",
    "inference_scale",
    "covered",
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
                    **{
                        field: chosen[f"{cell}_{field}"].fillna("").astype(str).to_numpy()
                        for field in study.SECONDARY_TEXT
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
    rows_interval: dict[str, tuple[float, float]] = {}
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
        reading = "rule"
        if kind == "calibration":
            rows_interval[scenario] = (ratio_low, ratio_high)
            passed = meets
        elif kind == "shrunk_se":
            row_low = rows_interval[scenario][0]
            if row_low > 1.0:
                # The row reads conservative (its SE-ratio interval lies above 1, the direction
                # plan 5.7 expects): the control must leave the row's own interval (review B1).
                reading = "below the row's interval"
                passed = ratio_high < row_low
            else:
                reading = "below the band"
                passed = not meets and ratio_high < low_band
        else:
            passed = meets
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
                "reading": reading,
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
                    "plug_in_standardized_bias": _performance(group, "initial")[
                        "standardized_bias"
                    ],
                }
            )
    return pd.DataFrame.from_records(records)


def pairing(rows: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    r"""X70 against the continuous engine on the same draws, each against its own truth.

    The panel coarsens the law, so X70 targets the panel's risk
    (``law.PANEL_TRUTHS``), not :math:`\psi_r`.  Each engine's bias reads its own truth, the
    expected paired difference is the panel's risk minus :math:`\psi_r`, and ``diagnose`` reads
    the departure of the mean paired difference from it (review S5).  X70's SE ratio and
    coverage read only the draws whose interval it gave, and the continuous engine's SE in the
    ratio reads the same draws (review m1).
    """
    x70 = _kept(secondary, "ltmle_daily")
    day = rows.loc[
        rows["scenario"] == "day_cf", ["replicate", "estimand", "estimate", "std_error", "covered"]
    ]
    joined = x70.merge(day, on=["replicate", "estimand"], suffixes=("_x70", "_continuous"))
    records = []
    for name, group in joined.groupby("estimand", sort=True):
        truth = float(group["truth"].iloc[0])
        panel = float(laws.PANEL_TRUTHS[str(name)])
        expected = panel - truth
        difference = group["estimate_x70"] - group["estimate_continuous"]
        continuous_se = float(group["std_error_continuous"].mean())
        mean = float(difference.mean())
        given = group.loc[group["interval_refused"] == 0.0]
        records.append(
            {
                "estimand": name,
                "x70_estimand": study.x70_name(str(name)),
                "pairs": len(group),
                "failed_fits": _failed_fits(secondary, "ltmle_daily"),
                "truth": truth,
                "panel_truth": panel,
                "expected_difference": expected,
                "mean_difference": mean,
                "difference_se": float(difference.std(ddof=1) / np.sqrt(len(group))),
                "departure": mean - expected,
                "departure_over_se": (mean - expected) / continuous_se,
                "x70_bias": float(group["estimate_x70"].mean()) - panel,
                "continuous_bias": float(group["estimate_continuous"].mean()) - truth,
                "x70_interval_refused_share": 1.0 - len(given) / len(group),
                "se_ratio": float(
                    given["std_error_x70"].mean() / given["std_error_continuous"].mean()
                )
                if len(given)
                else np.nan,
                "x70_coverage": float(given["covered_x70"].mean()) if len(given) else np.nan,
                "continuous_coverage": float(group["covered_continuous"].mean()),
                "diagnose": bool(abs(mean - expected) > PAIRED_DIAGNOSIS * continuous_se),
            }
        )
    return pd.DataFrame.from_records(records)


def _counts(values: pd.Series) -> str:
    """``K:count`` pairs of a cap column, ``;``-separated (empty when every value is missing)."""
    finite = values.dropna().astype(int)
    return ";".join(f"{k}:{c}" for k, c in finite.value_counts().sort_index().items())


def visit_cap(rows: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    r"""Per cell and row: the caps the fits chose and the bias against both truths (review S1).

    The gate reads the full risk :math:`\psi_r`; the capped truth :math:`\psi^{(K^*_r)}` at each
    fit's own caps (``law.capped_truth``) separates the cap's gap from the estimator's own bias.
    ``mean_gap`` is the mean of :math:`\psi^{(K^*_r)} - \psi_r` over the fits.
    """
    primary = rows.loc[rows["scenario"].isin(study.CELLS)].rename(columns={"scenario": "cell"})
    parts = [primary.loc[:, ["cell", "estimand", "estimate", "truth", *_CAP_COLUMNS]]]
    for cell in ("mechanism_correct", *MISSPECIFIED):
        kept = _kept(secondary, cell)
        parts.append(kept.loc[:, ["cell", "estimand", "estimate", "truth", *_CAP_COLUMNS]])
    frame = pd.concat(parts, ignore_index=True)
    records = []
    for (cell, name), group in frame.groupby(["cell", "estimand"], sort=True):
        estimate = group["estimate"].to_numpy(dtype=float)
        spread = float(np.std(estimate, ddof=1)) if estimate.size > 1 else np.nan
        capped = group.loc[np.isfinite(group["capped_truth"].to_numpy(dtype=float))]
        bias = float(np.mean(estimate - group["truth"].to_numpy(dtype=float)))
        capped_bias = (
            float(np.mean(capped["estimate"] - capped["capped_truth"])) if len(capped) else np.nan
        )
        records.append(
            {
                "cell": cell,
                "estimand": name,
                "fits": len(group),
                "truth": float(group["truth"].iloc[0]),
                "mean_capped_truth": float(capped["capped_truth"].mean())
                if len(capped)
                else np.nan,
                "mean_gap": float((capped["capped_truth"] - capped["truth"]).mean())
                if len(capped)
                else np.nan,
                "bias": bias,
                "capped_bias": capped_bias,
                "empirical_se": spread,
                "standardized_bias": bias / spread,
                "capped_standardized_bias": capped_bias / spread,
                "fits_without_capped_truth": int(len(group) - len(capped)),
                "caps": _counts(group["cap"]),
                "reference_caps": _counts(group["reference_cap"]),
            }
        )
    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------------- floors


def rel_sd_floor(rungs: Sequence[float], passed: Sequence[bool]) -> float:
    """The largest rung such that every row at or below it passes; ``0.0`` if the lowest fails.

    As ``continuous_survival_variance_readings.rel_sd_floor``: a red row caps the floor below
    its rung even when a higher rung passes.
    """
    order = np.argsort(np.asarray(rungs, dtype=float), kind="stable")
    floor = 0.0
    for index in order:
        if not passed[index]:
            break
        floor = float(rungs[index])
    return floor


def share_window(shares: Sequence[float], passed: Sequence[bool]) -> tuple[float, float]:
    """The event-share window: the passing rows' range, cut short of any red row.

    A red row at or below the median share of all rows moves the window's lower end above its
    share, and one at or above it moves the upper end below; the window then spans the passing
    rows left inside.  ``(nan, nan)`` when none is left.
    """
    share = np.asarray(shares, dtype=float)
    ok = np.asarray(passed, dtype=bool)
    middle = float(np.median(share))
    below, above = share[~ok & (share <= middle)], share[~ok & (share >= middle)]
    low_cut = float(below.max()) if below.size else -np.inf
    high_cut = float(above.min()) if above.size else np.inf
    inside = ok & (share > low_cut) & (share < high_cut)
    if not inside.any():
        return np.nan, np.nan
    return float(share[inside].min()), float(share[inside].max())


def floors(rows: pd.DataFrame) -> pd.DataFrame:
    """Per gated primary row: its verdict and the support readings part (e)'s floors read.

    Plan 3.5.9: ``variance_rel_sd`` as its median over the fits (the rung), the median event
    share at the row's time, the fewest followers at risk at the last target time, the largest
    and the 99th-percentile share past the cap, and the largest weight-floor and jump-bound
    shares.  ``rule_passed`` is the primary rule on the row's fits.  :func:`floor_values` reads
    the constants from this table.
    """
    gated = rows.loc[rows["scenario"].isin(study.CELLS)]
    performance = independent_performance_tests(
        gated.loc[:, list(_PERFORMANCE_COLUMNS)], record=study.STUDY, n_jobs=1
    )
    verdicts = {
        (str(row["scenario"]), str(row["estimand"])): bool(row["passed"])
        for _, row in performance.iterrows()
    }
    records = []
    for (scenario, name), group in gated.groupby(["scenario", "estimand"], sort=True):
        regime, _, _ = laws.parse(str(name))
        past = group["past_cap_share"].to_numpy(dtype=float)
        records.append(
            {
                "scenario": scenario,
                "estimand": name,
                "regime_kind": "rule" if regime == "rule" else "static",
                "fits": len(group),
                "median_rel_sd": float(group["variance_rel_sd"].median()),
                "median_event_share": float(group["event_share"].median()),
                "fewest_followers": float(group["followers_at_last"].min()),
                "largest_past_cap_share": float(past.max()),
                "q99_past_cap_share": float(np.quantile(past, 0.99)),
                "largest_bounded_share": float(group["bounded_share"].max()),
                "largest_jump_bound_share": float(group["jump_bound_share"].max()),
                "rule_passed": verdicts[(str(scenario), str(name))],
            }
        )
    return pd.DataFrame.from_records(records)


def floor_values(table: pd.DataFrame) -> dict[str, Any]:
    """Part (e)'s support constants from ``floors.csv`` (plan 3.5.9), read after the run.

    * ``longitudinal_rel_sd``: :func:`rel_sd_floor` over the static rows;
    * ``rules_supported``: every rule row passes (plan 8.10); then ``rules_rel_sd`` is the floor
      over every row;
    * the event-share window (:func:`share_window`) over the static rows;
    * ``fewest_followers``, ``largest_past_cap_share``, ``largest_bounded_share`` and
      ``jump_bound_acted`` over the static rows that pass; ``q99_past_cap_share`` is reported
      beside the largest share and sets nothing.
    """
    static = table.loc[table["regime_kind"] == "static"]
    passed = static["rule_passed"].astype(bool)
    rules = table.loc[table["regime_kind"] == "rule"]
    rules_supported = bool(len(rules)) and bool(rules["rule_passed"].astype(bool).all())
    low, high = share_window(static["median_event_share"].tolist(), passed.tolist())
    kept = static.loc[passed]
    return {
        "longitudinal_rel_sd": rel_sd_floor(static["median_rel_sd"].tolist(), passed.tolist()),
        "rules_supported": rules_supported,
        "rules_rel_sd": rel_sd_floor(
            table["median_rel_sd"].tolist(), table["rule_passed"].astype(bool).tolist()
        )
        if rules_supported
        else None,
        "event_share_low": low,
        "event_share_high": high,
        "fewest_followers": float(kept["fewest_followers"].min()) if len(kept) else np.nan,
        "largest_past_cap_share": float(kept["largest_past_cap_share"].max())
        if len(kept)
        else np.nan,
        "q99_past_cap_share": float(kept["q99_past_cap_share"].max()) if len(kept) else np.nan,
        "largest_bounded_share": float(kept["largest_bounded_share"].max())
        if len(kept)
        else np.nan,
        "jump_bound_acted": bool((kept["largest_jump_bound_share"] > 0).any()),
    }


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
        study.VISIT_CAP_FILE: visit_cap(rows, secondary),
        study.FLOORS_FILE: floors(rows),
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

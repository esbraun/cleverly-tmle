"""Shared machinery for the canonical point-treatment TMLE evidence study.

This module is deliberately test-side infrastructure.  It keeps the data-generating
law, estimand map, artifact schema, and acceptance rules in one place without turning
the documentation study into a public runtime API.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, t
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import DGP, binary_outcome_dgp
from cleverly.estimators import TMLE
from cleverly.utils.bounds import expit

TMLE3_COMMIT = "ed72f8a20e64c914ab25ffe015d865f7a9963d27"
SL3_COMMIT = "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.99
BOOTSTRAP_REPLICATES = 10_000
STANDARDIZED_BIAS_MARGIN = 0.25
COVERAGE_FLOOR = 0.88
SE_RATIO_LIMITS = (0.80, 1.20)
PAIRED_DIFFERENCE_MARGIN = 0.15
RMSE_NONINFERIORITY_MARGIN = 1.10
COVERAGE_NONINFERIORITY_MARGIN = -0.025
CALIBRATION_NONINFERIORITY_MARGIN = 0.05
PRIMARY_REPLICATES = 400
PRIMARY_N = 1000
SEED = 20240819

COMMON_ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc", "ey_obs", "par")
BINARY_ESTIMANDS = (*COMMON_ESTIMANDS, "paf", "rr", "or")
RATIO_ESTIMANDS = frozenset({"rr", "or"})

ARTIFACT_COLUMNS = (
    "implementation",
    "scenario",
    "replicate",
    "n",
    "estimand",
    "truth",
    "estimate",
    "inference_estimate",
    "std_error",
    "ci_lower",
    "ci_upper",
    "inference_scale",
    "covered",
    "initial_estimate",
)


def continuous_dgp() -> DGP:
    """A bounded continuous outcome with effect modification and comfortable overlap."""

    def propensity(w: np.ndarray) -> np.ndarray:
        return expit(0.6 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del z
        return expit(-0.5 + 0.7 * w[:, 0] - 0.4 * w[:, 1] + a * (0.8 + 0.3 * w[:, 0]))

    return DGP(
        name="canonical_tmle_continuous",
        n_latent=2,
        covariate_names=("W1", "W2"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        family="continuous",
        noise_scale=0.08,
    )


def sample_continuous(dgp: DGP, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw a bounded continuous outcome without clipping the conditional mean."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n, dgp.n_latent))
    g = np.asarray(dgp.propensity(w), dtype=float)
    a = rng.binomial(1, g).astype(float)
    q1 = np.asarray(dgp.outcome_mean(w, 1.0, None), dtype=float)
    q0 = np.asarray(dgp.outcome_mean(w, 0.0, None), dtype=float)
    mean = np.where(a == 1.0, q1, q0)
    concentration = 24.0
    y = rng.beta(mean * concentration, (1.0 - mean) * concentration)
    frame = pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1]})
    return frame, truth_for(dgp)


def truth_for(dgp: DGP) -> dict[str, float]:
    """Every shared estimand, including natural-course quantities absent from DGP.truth."""
    w = dgp.quadrature()
    g = np.asarray(dgp.propensity(w), dtype=float)
    q1 = np.asarray(dgp.outcome_mean(w, 1.0, None), dtype=float)
    q0 = np.asarray(dgp.outcome_mean(w, 0.0, None), dtype=float)
    contrast = q1 - q0
    ey1 = float(np.mean(q1))
    ey0 = float(np.mean(q0))
    ey_obs = float(np.mean(g * q1 + (1.0 - g) * q0))
    out = {
        "ey1": ey1,
        "ey0": ey0,
        "ate": float(np.mean(contrast)),
        "att": float(np.sum(g * contrast) / np.sum(g)),
        "atc": float(np.sum((1.0 - g) * contrast) / np.sum(1.0 - g)),
        "ey_obs": ey_obs,
        "par": ey_obs - ey0,
    }
    if dgp.family == "binomial":
        out.update(
            {
                "paf": 1.0 - ey0 / ey_obs,
                "rr": ey1 / ey0,
                "or": (ey1 / (1.0 - ey1)) / (ey0 / (1.0 - ey0)),
            }
        )
    return out


def scenario_dgp(scenario: str) -> DGP:
    if scenario == "continuous":
        return continuous_dgp()
    if scenario == "binary":
        return binary_outcome_dgp()
    raise KeyError(scenario)


def draw_scenario(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    dgp = scenario_dgp(scenario)
    if scenario == "continuous":
        return sample_continuous(dgp, n, seed)
    frame, _ = dgp.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(dgp)


def fit_cleverly(frame: pd.DataFrame, scenario: str) -> Any:
    """The explicitly non-cross-fitted configuration matched to ordinary R tmle3."""
    binary = scenario == "binary"
    outcome = (
        LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs") if binary else LinearRegression()
    )
    treatment = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    estimands = BINARY_ESTIMANDS if binary else COMMON_ESTIMANDS
    covariates = [column for column in frame.columns if column.startswith("W")]
    return (
        TMLE(
            outcome_learner=outcome,
            treatment_learner=treatment,
            cross_fit=False,
            estimands=estimands,
            simultaneous=False,
            g_bounds=(0.01, 0.99),
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=covariates)
        .single()
    )


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario)
    rows: list[dict[str, Any]] = []
    for name, estimate in result.estimates.items():
        reference = float(truth[name])
        low, high = estimate.ci
        inference_estimate = (
            float(estimate.log_psi)
            if estimate.scale == "ratio" and estimate.log_psi is not None
            else float(estimate.psi)
        )
        rows.append(
            {
                "implementation": "cleverly",
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": inference_estimate,
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "log" if estimate.scale == "ratio" else "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
            }
        )
    return rows


def truth_on_inference_scale(estimand: str, truth: float, scale: str) -> float:
    """Map truth to the native scale on which an implementation reports its SE."""
    if scale == "log":
        return math.log(truth)
    if scale == "negative_log_complement":
        return -math.log1p(-truth)
    if scale == "identity":
        return truth
    raise ValueError(f"unknown inference scale {scale!r} for {estimand}")


def _bootstrap_se_ratio(
    inference: np.ndarray,
    std_errors: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float]:
    """A deterministic percentile-bootstrap 99% interval for mean-SE / empirical-SD."""
    rng = np.random.default_rng(seed)
    n = len(inference)
    statistics = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch_size = 1_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, n, size=(stop - start, n))
        sampled_inference = inference[indices]
        sampled_se = std_errors[indices]
        statistics[start:stop] = sampled_se.mean(axis=1) / sampled_inference.std(axis=1, ddof=1)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(statistics, [tail, 1.0 - tail])
    return float(low), float(high)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute the published performance table from per-replication output."""
    records: list[dict[str, Any]] = []
    keys = ["implementation", "scenario", "estimand"]
    for key, group in rows.groupby(keys, sort=True):
        implementation, scenario, estimand = key
        estimates = group["estimate"].to_numpy(dtype=float)
        inference = group["inference_estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        truth_inference = truth_on_inference_scale(
            str(estimand), truth, str(group["inference_scale"].iloc[0])
        )
        bias = float(np.mean(estimates) - truth)
        empirical_se = float(np.std(inference, ddof=1))
        mean_se = float(group["std_error"].mean())
        coverage = float(group["covered"].mean())
        reps = len(group)
        records.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "estimand": estimand,
                "n": int(group["n"].iloc[0]),
                "replicates": reps,
                "truth": truth,
                "mean_estimate": float(np.mean(estimates)),
                "bias": bias,
                "bias_se": float(np.std(estimates, ddof=1) / math.sqrt(reps)),
                "root_n_bias": float(math.sqrt(group["n"].iloc[0]) * bias),
                "rmse": float(np.sqrt(np.mean((estimates - truth) ** 2))),
                "empirical_se": empirical_se,
                "mean_std_error": mean_se,
                "se_ratio": mean_se / empirical_se,
                "coverage": coverage,
                "coverage_se": math.sqrt(coverage * (1.0 - coverage) / reps),
                "inference_bias": float(np.mean(inference) - truth_inference),
            }
        )
    return pd.DataFrame.from_records(records)


def independent_performance_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """Test each implementation against known truth with 99% confidence procedures."""
    records: list[dict[str, Any]] = []
    keys = ["implementation", "scenario", "estimand"]
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    for group_index, (key, group) in enumerate(rows.groupby(keys, sort=True)):
        implementation, scenario, estimand = key
        inference = group["inference_estimate"].to_numpy(dtype=float)
        std_errors = group["std_error"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        scale = str(group["inference_scale"].iloc[0])
        truth_inference = truth_on_inference_scale(str(estimand), truth, scale)
        errors = inference - truth_inference
        replicates = len(group)
        empirical_se = float(np.std(inference, ddof=1))
        bias = float(np.mean(errors))
        bias_half_width = float(
            t.ppf(1.0 - tail, replicates - 1) * empirical_se / math.sqrt(replicates)
        )
        bias_low = bias - bias_half_width
        bias_high = bias + bias_half_width
        bias_margin = STANDARDIZED_BIAS_MARGIN * empirical_se
        bias_equivalent = bias_low >= -bias_margin and bias_high <= bias_margin

        covered = int(group["covered"].sum())
        coverage = covered / replicates
        coverage_low = float(beta.ppf(tail, covered, replicates - covered + 1)) if covered else 0.0
        coverage_high = (
            float(beta.ppf(1.0 - tail, covered + 1, replicates - covered))
            if covered < replicates
            else 1.0
        )
        coverage_calibrated = (
            coverage_low >= COVERAGE_FLOOR and coverage_low <= 1.0 - ALPHA <= coverage_high
        )

        se_ratio = float(np.mean(std_errors) / empirical_se)
        se_ratio_low, se_ratio_high = _bootstrap_se_ratio(
            inference,
            std_errors,
            seed=SEED + 10_000 + group_index,
        )
        se_calibrated = se_ratio_low >= SE_RATIO_LIMITS[0] and se_ratio_high <= SE_RATIO_LIMITS[1]
        records.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "estimand": estimand,
                "replicates": replicates,
                "confidence_level": CONFIDENCE_LEVEL,
                "inference_scale": scale,
                "bias": bias,
                "bias_ci_lower": bias_low,
                "bias_ci_upper": bias_high,
                "bias_margin": bias_margin,
                "bias_equivalent": bias_equivalent,
                "coverage": coverage,
                "coverage_ci_lower": coverage_low,
                "coverage_ci_upper": coverage_high,
                "coverage_floor": COVERAGE_FLOOR,
                "coverage_calibrated": coverage_calibrated,
                "se_ratio": se_ratio,
                "se_ratio_ci_lower": se_ratio_low,
                "se_ratio_ci_upper": se_ratio_high,
                "se_ratio_margin_lower": SE_RATIO_LIMITS[0],
                "se_ratio_margin_upper": SE_RATIO_LIMITS[1],
                "se_calibrated": se_calibrated,
                "passed": bias_equivalent and coverage_calibrated and se_calibrated,
            }
        )
    return pd.DataFrame.from_records(records)


def _bootstrap_noninferiority(
    group: pd.DataFrame,
    *,
    estimand: str,
    seed: int,
) -> tuple[float, float, float]:
    """One-sided 99% bounds for cleverly-vs-R loss, coverage, and calibration."""

    def paired(column: str) -> tuple[np.ndarray, np.ndarray]:
        wide = group.pivot(index="replicate", columns="implementation", values=column).dropna()
        return wide["cleverly"].to_numpy(dtype=float), wide["tmle3"].to_numpy(dtype=float)

    clever_estimate, r_estimate = paired("estimate")
    clever_covered, r_covered = paired("covered")
    clever_inference, r_inference = paired("inference_estimate")
    clever_se, r_se = paired("std_error")
    truth = float(group["truth"].iloc[0])
    n = len(clever_estimate)
    rng = np.random.default_rng(seed)
    rmse_ratios = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    coverage_differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    calibration_excesses = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch_size = 1_000
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, n, size=(stop - start, n))
        clever_error = clever_estimate[indices] - truth
        r_error = r_estimate[indices] - truth
        rmse_ratios[start:stop] = np.sqrt(
            np.mean(clever_error**2, axis=1) / np.mean(r_error**2, axis=1)
        )
        coverage_differences[start:stop] = np.mean(
            clever_covered[indices] - r_covered[indices], axis=1
        )
        clever_ratio = clever_se[indices].mean(axis=1) / clever_inference[indices].std(
            axis=1, ddof=1
        )
        r_ratio = r_se[indices].mean(axis=1) / r_inference[indices].std(axis=1, ddof=1)
        calibration_excesses[start:stop] = abs(clever_ratio - 1.0) - abs(r_ratio - 1.0)

    rmse_upper = float(np.quantile(rmse_ratios, CONFIDENCE_LEVEL))
    coverage_lower = float(np.quantile(coverage_differences, 1.0 - CONFIDENCE_LEVEL))
    calibration_upper = (
        math.nan
        if estimand == "paf"
        else float(np.quantile(calibration_excesses, CONFIDENCE_LEVEL))
    )
    return rmse_upper, coverage_lower, calibration_upper


def equivalence(rows: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    """Truth-anchored equivalence and asymmetric cleverly performance checks."""
    records: list[dict[str, Any]] = []
    for group_index, ((scenario, estimand), group) in enumerate(
        rows.groupby(["scenario", "estimand"], sort=True)
    ):
        wide = group.pivot(index="replicate", columns="implementation", values="estimate").dropna()
        if set(wide.columns) != {"cleverly", "tmle3"}:
            continue
        difference = wide["cleverly"] - wide["tmle3"]
        paired_se = float(difference.std(ddof=1) / math.sqrt(len(difference)))
        pooled_sd = float(np.sqrt(0.5 * (wide["cleverly"].var(ddof=1) + wide["tmle3"].var(ddof=1))))
        mean_difference = float(difference.mean())
        paired_half_width = float(
            t.ppf(0.5 + CONFIDENCE_LEVEL / 2.0, len(difference) - 1) * paired_se
        )
        paired_ci_lower = mean_difference - paired_half_width
        paired_ci_upper = mean_difference + paired_half_width
        mean_margin = PAIRED_DIFFERENCE_MARGIN * pooled_sd
        subset = summaries.query("scenario == @scenario and estimand == @estimand").set_index(
            "implementation"
        )
        rmse_ratio = float(subset.loc["cleverly", "rmse"] / subset.loc["tmle3", "rmse"])
        coverage_difference = float(
            abs(subset.loc["cleverly", "coverage"] - subset.loc["tmle3", "coverage"])
        )
        cleverly_coverage_shortfall = float(
            max(0.0, subset.loc["tmle3", "coverage"] - subset.loc["cleverly", "coverage"])
        )
        se_ratio_difference = float(
            abs(subset.loc["cleverly", "se_ratio"] - subset.loc["tmle3", "se_ratio"])
        )
        cleverly_calibration_excess = float(
            max(
                0.0,
                abs(subset.loc["cleverly", "se_ratio"] - 1.0)
                - abs(subset.loc["tmle3", "se_ratio"] - 1.0),
            )
        )
        # PAF uses a symmetric identity-scale CI in cleverly and a transformed log-risk
        # interval in tmle3.  Coverage is comparable; raw SE ratios are not on one scale.
        se_comparable = estimand != "paf"
        bias_z = float((subset["bias"].abs() / subset["bias_se"]).max())
        truth_compatible = bias_z <= 3.5
        calibrated = bool(
            (subset["coverage"] >= 0.90).all() and subset["se_ratio"].between(0.85, 1.15).all()
        )
        rmse_ratio_upper_99, coverage_difference_lower_99, calibration_excess_upper_99 = (
            _bootstrap_noninferiority(
                group,
                estimand=str(estimand),
                seed=SEED + 20_000 + group_index,
            )
        )
        cleverly_not_inferior_99 = bool(
            rmse_ratio_upper_99 <= RMSE_NONINFERIORITY_MARGIN
            and coverage_difference_lower_99 >= COVERAGE_NONINFERIORITY_MARGIN
            and (
                not se_comparable
                or calibration_excess_upper_99 <= CALIBRATION_NONINFERIORITY_MARGIN
            )
        )
        paired_mean_within_margin = abs(mean_difference) <= mean_margin
        paired_similarity_99 = paired_ci_lower >= -mean_margin and paired_ci_upper <= mean_margin
        passed = (
            truth_compatible
            and calibrated
            and cleverly_not_inferior_99
            and paired_similarity_99
            and 0.8 <= rmse_ratio <= 1.25
            and coverage_difference <= 0.05
            and (not se_comparable or se_ratio_difference <= 0.15)
        )
        records.append(
            {
                "scenario": scenario,
                "estimand": estimand,
                "paired_replicates": len(difference),
                "confidence_level": CONFIDENCE_LEVEL,
                "mean_difference": mean_difference,
                "paired_se": paired_se,
                "paired_ci_lower": paired_ci_lower,
                "paired_ci_upper": paired_ci_upper,
                "mean_margin": mean_margin,
                "paired_mean_within_margin": paired_mean_within_margin,
                "paired_similarity_99": paired_similarity_99,
                "max_absolute_bias_z": bias_z,
                "truth_compatible": truth_compatible,
                "calibrated": calibrated,
                "cleverly_not_inferior_99": cleverly_not_inferior_99,
                "rmse_ratio": rmse_ratio,
                "rmse_ratio_upper_99": rmse_ratio_upper_99,
                "rmse_noninferiority_margin": RMSE_NONINFERIORITY_MARGIN,
                "coverage_difference": coverage_difference,
                "cleverly_coverage_shortfall": cleverly_coverage_shortfall,
                "coverage_difference_lower_99": coverage_difference_lower_99,
                "coverage_noninferiority_margin": COVERAGE_NONINFERIORITY_MARGIN,
                "se_ratio_difference": se_ratio_difference if se_comparable else math.nan,
                "cleverly_calibration_excess": cleverly_calibration_excess,
                "calibration_excess_upper_99": calibration_excess_upper_99,
                "calibration_noninferiority_margin": (
                    CALIBRATION_NONINFERIORITY_MARGIN if se_comparable else math.nan
                ),
                "passed": passed,
            }
        )
    return pd.DataFrame.from_records(records)


def hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def write_manifest(
    path: Path,
    artifacts: Iterable[Path],
    *,
    replicates: int,
    n: int,
    reference_files: Iterable[Path] = (),
) -> None:
    payload = {
        "schema_version": 1,
        "study": "canonical point-treatment TMLE",
        "generated_with": {
            "cleverly": "working tree",
            "tmle3_commit": TMLE3_COMMIT,
            "sl3_commit": SL3_COMMIT,
            "r_base_image": R_BASE_IMAGE,
        },
        "configuration": {
            "replicates": replicates,
            "n": n,
            "seed": SEED,
            "alpha": ALPHA,
            "cross_fit": False,
            "simultaneous_intervals": False,
            "g_bounds": [0.01, 0.99],
            "statistical_testing": {
                "confidence_level": CONFIDENCE_LEVEL,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "standardized_bias_margin": STANDARDIZED_BIAS_MARGIN,
                "coverage_floor": COVERAGE_FLOOR,
                "se_ratio_limits": list(SE_RATIO_LIMITS),
                "paired_difference_margin_sd": PAIRED_DIFFERENCE_MARGIN,
                "rmse_ratio_noninferiority_margin": RMSE_NONINFERIORITY_MARGIN,
                "coverage_difference_noninferiority_margin": (COVERAGE_NONINFERIORITY_MARGIN),
                "calibration_excess_noninferiority_margin": (CALIBRATION_NONINFERIORITY_MARGIN),
            },
        },
        "reference_sha256": hashes(reference_files),
        "sha256": hashes(artifacts),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

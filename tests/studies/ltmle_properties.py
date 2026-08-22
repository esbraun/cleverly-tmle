"""Independent repeated-sampling properties for end-of-study longitudinal TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as law
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import G_BOUNDS, STUDY
from tests.studies.canonical_properties import (
    EXCLUDED_SLOPE,
    MINIMUM_POWER,
    ROOT_N_SLOPE,
    ROOT_N_SLOPE_MARGIN,
    finish,
)
from tests.studies.evidence.inference import Interval, percentile_interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    rate,
    se_ratio_interval,
    summarize_cells,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
NULL_REPLICATES = 400
NULL_N = 4_000
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70

REGIMENS = {key: law.REGIMEN_SPEC[key] for key in ("never", "always", "treat_if_l2")}
REFERENCE = "never"
CONTRASTS = {
    "static": "ate_regimen[always vs never]",
    "dynamic": "ate_regimen[treat_if_l2 vs never]",
}
EFFICIENCY_SD = {
    label: float(np.sqrt(np.sum(law.PROBS * law.eif(name) ** 2)))
    for label, name in CONTRASTS.items()
}


def _null_probabilities() -> np.ndarray:
    """The exact observed-data law after imposing a confounded sharp null."""
    probabilities = np.array(law.PROBS, copy=True)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, point in enumerate(law.SUPPORT):
        if point[-1] is not None:
            groups.setdefault(point[:-1], []).append(index)
    for history, indices in groups.items():
        if len(indices) != 2:
            raise AssertionError(f"outcome history {history} has {len(indices)} support points")
        mass = float(np.sum(law.PROBS[indices]))
        probability = 0.25 + 0.5 * float(history[0])
        for index in indices:
            probabilities[index] = mass * (
                probability if law.SUPPORT[index][-1] == 1 else 1.0 - probability
            )
    if not np.isclose(probabilities.sum(), 1.0):
        raise AssertionError("the null law does not sum to one")
    return probabilities


NULL_PROBS = _null_probabilities()
NULL_TRUTH = float(law.functional(NULL_PROBS, CONTRASTS["static"]))


def sample(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(law.SUPPORT), size=n, p=probs)
    return pd.DataFrame(
        {
            name: np.array(
                [
                    np.nan if point[position] is None else float(point[position])
                    for point in law.SUPPORT
                ]
            )[cells]
            for position, name in enumerate(("W", "A1", "C1", "L2", "A2", "C2", "Y"))
        }
    )


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return (
        law.CellMeans() if q_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if q_correct else DummyRegressor(strategy="mean"),
        law.CellMeans() if g_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if g_correct else DummyClassifier(strategy="prior"),
    )


def fit(frame: pd.DataFrame, configuration: str = "both_correct") -> Any:
    outcome, pseudo, treatment, censoring = _learners(configuration)
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _row(
    *,
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: Any,
) -> dict[str, Any]:
    low, high = estimate.ci
    return {
        "property": property_name,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
        "covered": int(low <= truth <= high),
        "rejected": int(estimate.pvalue < STUDY.margins.alpha),
    }


def _fit_replication(
    payload: tuple[str, str, int, int, int, int, str],
) -> list[dict[str, Any]]:
    property_name, cell_suffix, replicate, n, requested, seed, configuration = payload
    probs = NULL_PROBS if property_name == "type_i_error" else law.PROBS
    result = fit(sample(probs, n, seed), configuration)
    labels = ("static",) if property_name in {"type_i_error", "power"} else tuple(CONTRASTS)
    rows: list[dict[str, Any]] = []
    for label in labels:
        name = CONTRASTS[label]
        truth = NULL_TRUTH if property_name == "type_i_error" else float(law.TRUTH[name])
        role = (
            "control"
            if cell_suffix == "both_wrong"
            or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
            else "positive"
        )
        rows.append(
            _row(
                property_name=property_name,
                cell=f"{label}__{cell_suffix}",
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=result[name],
            )
        )
    return rows


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[tuple[str, str, int, int, str]] = []
    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
        specs.append(
            (
                "double_robustness",
                configuration,
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                configuration,
            )
        )
    for size in RATE_SIZES:
        specs.append(("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct"))
    specs.extend(
        [
            (
                "interval_calibration",
                "correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ("type_i_error", "sharp_null", NULL_N, NULL_REPLICATES, "both_correct"),
            ("power", "alternative", NULL_N, NULL_REPLICATES, "both_correct"),
        ]
    )
    payloads: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            payloads.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return payloads


def _calibration_controls(rows: pd.DataFrame) -> pd.DataFrame:
    source = rows.loc[
        (rows["property"] == "interval_calibration")
        & rows["cell"].str.endswith("__correctly_specified")
    ]
    controls: list[pd.DataFrame] = []
    critical = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
    for label in CONTRASTS:
        base = source.loc[source["cell"] == f"{label}__correctly_specified"].copy()
        shrunken = base.copy()
        shrunken["cell"] = f"{label}__shrunken_se_control"
        shrunken["role"] = "control"
        shrunken["std_error"] *= SHRUNKEN_SE_FACTOR
        shrunken["covered"] = (
            (shrunken["estimate"] - critical * shrunken["std_error"] <= shrunken["truth"])
            & (shrunken["truth"] <= shrunken["estimate"] + critical * shrunken["std_error"])
        ).astype(int)
        shrunken["rejected"] = (
            np.abs(shrunken["estimate"] / shrunken["std_error"]) > critical
        ).astype(int)
        controls.append(shrunken)

        noisy = base.copy()
        rng = np.random.default_rng(stream_seed(STUDY, "efficiency_noise", label))
        noisy["cell"] = f"{label}__noise_control"
        noisy["role"] = "control"
        noisy["estimate"] += rng.normal(
            scale=EFFICIENCY_SD[label] / np.sqrt(CALIBRATION_N), size=len(noisy)
        )
        noisy["covered"] = (
            (noisy["estimate"] - critical * noisy["std_error"] <= noisy["truth"])
            & (noisy["truth"] <= noisy["estimate"] + critical * noisy["std_error"])
        ).astype(int)
        noisy["rejected"] = (np.abs(noisy["estimate"] / noisy["std_error"]) > critical).astype(int)
        controls.append(noisy)
    return pd.concat(controls, ignore_index=True)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat([rows, _calibration_controls(rows)], ignore_index=True)
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def _known_ratio_interval(
    group: pd.DataFrame, *, label: str, statistic: str, seed: int
) -> Interval:
    values = group[["estimate", "std_error"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(STUDY.margins.bootstrap_replicates, len(values)))
    draws = values[picks]
    numerator = (
        draws[:, :, 0].std(axis=1, ddof=1)
        if statistic == "empirical"
        else draws[:, :, 1].mean(axis=1)
    )
    ratios = numerator * np.sqrt(int(group["n"].iloc[0])) / EFFICIENCY_SD[label]
    return percentile_interval(ratios, confidence_level=STUDY.margins.confidence_level)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    margins = STUDY.margins
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
    )
    additions = (
        "rate_sizes",
        "slope",
        "slope_ci_lower",
        "slope_ci_upper",
        "se_ratio_ci_lower",
        "se_ratio_ci_upper",
        "efficiency_empirical_ratio",
        "efficiency_empirical_ci_lower",
        "efficiency_empirical_ci_upper",
        "efficiency_reported_ratio",
        "efficiency_reported_ci_lower",
        "efficiency_reported_ci_upper",
    )
    for column in additions:
        summary[column] = np.nan
    summary["passed"] = False
    summary["property_passed"] = pd.Series([None] * len(summary), dtype=object, index=summary.index)

    robustness = summary["property"] == "double_robustness"
    positive = robustness & (summary["role"] == "positive")
    control = robustness & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    root_n = summary["property"] == "root_n_and_efficiency"
    root_positive = root_n & (summary["role"] == "positive")
    summary.loc[root_positive, "passed"] = (
        summary.loc[root_positive, "bias_equivalent"]
        & (summary.loc[root_positive, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[root_positive, "se_ratio"].between(*margins.se_ratio_sanity)
    )
    # The smallest size is retained as a negative control because the committed run
    # resolves its undercoverage rather than merely producing a noisy point below 0.95.
    # It still enters the three-size rate fit: root-n contraction and calibrated inference
    # are distinct claims, and the n=2,000 calibration cell is the latter's positive gate.
    root_control = root_n & (summary["role"] == "control")
    summary.loc[root_control, "passed"] = (
        summary.loc[root_control, "coverage_ci_upper"] < 1.0 - margins.alpha
    )

    calibration = summary["property"] == "interval_calibration"
    for index in summary.index[calibration]:
        cell = str(summary.loc[index, "cell"])
        label, kind = cell.split("__", 1)
        group = rows.loc[(rows["property"] == "interval_calibration") & (rows["cell"] == cell)]
        ratio = se_ratio_interval(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "interval_calibration", cell),
        )
        empirical = _known_ratio_interval(
            group,
            label=label,
            statistic="empirical",
            seed=stream_seed(STUDY, "efficiency_empirical", cell),
        )
        reported = _known_ratio_interval(
            group,
            label=label,
            statistic="reported",
            seed=stream_seed(STUDY, "efficiency_reported", cell),
        )
        summary.loc[index, ["se_ratio_ci_lower", "se_ratio_ci_upper"]] = [ratio.low, ratio.high]
        summary.loc[index, ["efficiency_empirical_ci_lower", "efficiency_empirical_ci_upper"]] = [
            empirical.low,
            empirical.high,
        ]
        summary.loc[index, ["efficiency_reported_ci_lower", "efficiency_reported_ci_upper"]] = [
            reported.low,
            reported.high,
        ]
        n = int(group["n"].iloc[0])
        summary.loc[index, "efficiency_empirical_ratio"] = float(
            group["estimate"].std(ddof=1) * np.sqrt(n) / EFFICIENCY_SD[label]
        )
        summary.loc[index, "efficiency_reported_ratio"] = float(
            group["std_error"].mean() * np.sqrt(n) / EFFICIENCY_SD[label]
        )
        coverage = Interval(
            float(summary.loc[index, "coverage_ci_lower"]),
            float(summary.loc[index, "coverage_ci_upper"]),
        )
        if kind == "correctly_specified":
            passed = (
                ratio.within(*margins.calibration_se_ratio)
                and coverage.within(*margins.calibration_coverage)
                and empirical.within(*EFFICIENCY_RATIO_BAND)
                and reported.within(*EFFICIENCY_RATIO_BAND)
            )
        elif kind == "shrunken_se_control":
            passed = ratio.high < margins.calibration_se_ratio[0]
        else:
            passed = empirical.low > EFFICIENCY_RATIO_BAND[1]
        summary.loc[index, "passed"] = bool(passed)

    null = summary["property"] == "type_i_error"
    summary.loc[null, "passed"] = (
        summary.loc[null, "rejection_ci_upper"] <= margins.alpha + margins.type_i_margin
    ) & (summary.loc[null, "coverage_ci_lower"] >= margins.coverage_floor)
    power = summary["property"] == "power"
    summary.loc[power, "passed"] = summary.loc[power, "rejection_ci_lower"] >= MINIMUM_POWER

    rates: list[dict[str, Any]] = []
    for label in CONTRASTS:
        selected = rows.loc[
            (rows["property"] == "root_n_and_efficiency")
            & rows["cell"].str.startswith(f"{label}__")
        ]
        for statistic, suffix in (("spread", "empirical_sd"), ("reported", "reported_se")):
            fitted = rate(
                selected,
                property_name="root_n_and_efficiency",
                statistic=statistic,
                bootstrap_replicates=margins.bootstrap_replicates,
                confidence_level=margins.confidence_level,
                seed=stream_seed(STUDY, "root_n_rate", label, suffix),
            )
            row: dict[str, Any] = dict.fromkeys(summary.columns, np.nan)
            row.update(
                {
                    "property": "root_n_rate",
                    "cell": f"{label}__{suffix}",
                    "role": "positive",
                    "n": max(RATE_SIZES),
                    "replicates": RATE_REPLICATES * len(RATE_SIZES),
                    "rate_sizes": ";".join(f"{size:,}" for size in RATE_SIZES),
                    "failed_replicates": 0,
                    "slope": fitted.slope,
                    "slope_ci_lower": fitted.interval.low,
                    "slope_ci_upper": fitted.interval.high,
                    "passed": bool(
                        fitted.equivalent_to(ROOT_N_SLOPE, ROOT_N_SLOPE_MARGIN)
                        and fitted.excludes(EXCLUDED_SLOPE)
                    ),
                }
            )
            rates.append(row)
    return finish(summary, rates)

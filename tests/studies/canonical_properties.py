"""Independent repeated-sampling properties for ordinary canonical TMLE."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import DGP, linear_dgp, nonlinear_dgp
from cleverly.estimators import TMLE
from cleverly.utils.bounds import expit
from cleverly.validation import CoverageStudy
from tests.conftest import OracleOutcomeContinuous, OracleTreatment


def _null_dgp() -> DGP:
    def propensity(w: np.ndarray) -> np.ndarray:
        return expit(0.5 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: np.ndarray, a: float, z: float | None) -> np.ndarray:
        del a, z
        return 1.0 + 0.9 * w[:, 0] + 0.6 * w[:, 1] - 0.4 * w[:, 2]

    return DGP(
        name="canonical_tmle_null",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def _run(
    dgp: DGP,
    outcome: Any,
    treatment: Any,
    *,
    n: int,
    replicates: int,
    seed: int,
) -> Any:
    return CoverageStudy(
        dgp=dgp,
        estimator=lambda: TMLE(
            outcome_learner=outcome,
            treatment_learner=treatment,
            cross_fit=False,
            estimands=("ate",),
            simultaneous=False,
            g_bounds=(0.01, 0.99),
            max_iter=100,
            tol=1e-10,
            random_state=0,
        ),
        n=n,
        n_replicates=replicates,
        estimands=("ate",),
        seed=seed,
        n_jobs=2,
    ).run()["ate"]


def _replicate_rows(property_name: str, cell: str, summary: Any) -> list[dict[str, Any]]:
    rows = []
    for replicate, (estimate, std_error, covered, rejected) in enumerate(
        zip(
            summary.estimates,
            summary.std_errors,
            summary.covered,
            summary.rejected,
            strict=True,
        )
    ):
        rows.append(
            {
                "property": property_name,
                "cell": cell,
                "replicate": replicate,
                "n": summary.n,
                "truth": summary.truth,
                "estimate": float(estimate),
                "std_error": float(std_error),
                "covered": int(covered),
                "rejected": int(rejected),
            }
        )
    return rows


def generate_property_rows(
    *,
    double_robust_replicates: int = 100,
    rate_replicates: int = 200,
    null_replicates: int = 400,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nonlinear = nonlinear_dgp()
    cells: tuple[tuple[str, Callable[[], Any], Callable[[], Any]], ...] = (
        (
            "both_correct",
            lambda: OracleOutcomeContinuous(nonlinear),
            lambda: OracleTreatment(nonlinear),
        ),
        (
            "outcome_correct",
            lambda: OracleOutcomeContinuous(nonlinear),
            lambda: LogisticRegression(max_iter=1000),
        ),
        (
            "treatment_correct",
            lambda: LinearRegression(),
            lambda: OracleTreatment(nonlinear),
        ),
        (
            "both_wrong",
            lambda: LinearRegression(),
            lambda: LogisticRegression(max_iter=1000),
        ),
    )
    for index, (cell, q_factory, g_factory) in enumerate(cells):
        summary = _run(
            nonlinear,
            q_factory(),
            g_factory(),
            n=700,
            replicates=double_robust_replicates,
            seed=7100 + index,
        )
        rows.extend(_replicate_rows("double_robustness", cell, summary))

    linear = linear_dgp()
    for n in (500, 2000):
        summary = _run(
            linear,
            LinearRegression(),
            LogisticRegression(max_iter=1000),
            n=n,
            replicates=rate_replicates,
            seed=8100,
        )
        rows.extend(_replicate_rows("root_n_and_efficiency", f"n_{n}", summary))

    null = _run(
        _null_dgp(),
        LinearRegression(),
        LogisticRegression(max_iter=1000),
        n=1000,
        replicates=null_replicates,
        seed=9100,
    )
    rows.extend(_replicate_rows("type_i_error", "sharp_null", null))
    return pd.DataFrame.from_records(rows)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (property_name, cell), group in rows.groupby(["property", "cell"], sort=True):
        estimates = group["estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        replicates = len(group)
        empirical_se = float(np.std(estimates, ddof=1))
        bias = float(np.mean(estimates) - truth)
        coverage = float(group["covered"].mean())
        records.append(
            {
                "property": property_name,
                "cell": cell,
                "n": int(group["n"].iloc[0]),
                "replicates": replicates,
                "truth": truth,
                "mean_estimate": float(np.mean(estimates)),
                "bias": bias,
                "bias_se": empirical_se / math.sqrt(replicates),
                "root_n_bias": math.sqrt(group["n"].iloc[0]) * bias,
                "empirical_se": empirical_se,
                "mean_std_error": float(group["std_error"].mean()),
                "se_ratio": float(group["std_error"].mean()) / empirical_se,
                "coverage": coverage,
                "coverage_se": math.sqrt(coverage * (1.0 - coverage) / replicates),
                "rejection_rate": float(group["rejected"].mean()),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary["passed"] = True

    dr = summary["property"] == "double_robustness"
    good = dr & (summary["cell"] != "both_wrong")
    summary.loc[good, "passed"] = summary.loc[good].apply(
        lambda row: abs(row.bias) <= max(3.5 * row.bias_se, 0.03), axis=1
    )
    wrong = dr & (summary["cell"] == "both_wrong")
    summary.loc[wrong, "passed"] = summary.loc[wrong].apply(
        lambda row: abs(row.bias) > 4.0 * row.bias_se and abs(row.bias) > 0.1, axis=1
    )

    rate = summary.query("property == 'root_n_and_efficiency'").set_index("cell")
    if set(rate.index) == {"n_500", "n_2000"}:
        se_rate = rate.loc["n_2000", "mean_std_error"] / rate.loc["n_500", "mean_std_error"]
        root_n_ok = abs(rate.loc["n_2000", "root_n_bias"]) <= 2.5 * max(
            abs(rate.loc["n_500", "root_n_bias"]), 0.1
        )
        efficient = all(0.85 <= value <= 1.15 for value in rate["se_ratio"])
        calibrated = all(value >= 0.90 for value in rate["coverage"])
        summary.loc[summary["property"] == "root_n_and_efficiency", "passed"] = (
            0.4 <= se_rate <= 0.6 and root_n_ok and efficient and calibrated
        )

    null = summary["property"] == "type_i_error"
    summary.loc[null, "passed"] = summary.loc[null].apply(
        lambda row: (
            abs(row.rejection_rate - 0.05) <= 3.0 * math.sqrt(0.05 * 0.95 / row.replicates)
            and row.coverage >= 0.90
        ),
        axis=1,
    )
    return summary

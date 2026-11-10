"""The reported cells of ``survival-ctmle`` (X91 part B): no property family, two artifacts.

The record's rows hold its gated cells only (:data:`tests.studies.survival_ctmle.GATED_SCENARIOS`),
because every row of a record is judged against its truth.  The other cells are reported and
never gated: the misspecified outcome model, :math:`n = 500`, the tertile-binarized covariates
and ``g_bounds`` lower bound :math:`10^{-8}`.  This module fits them in the property phase,
:data:`~tests.studies.survival_ctmle.REPORTED_REPLICATES` replicates each, all three
implementations on each draw, from the stream ``stream_seed(STUDY, "reported", scenario, r)``,
which no gated replicate and no pilot draw uses.

It declares no property cell, so it writes no property row and no property verdict.  Its fits
publish as two property artifacts: ``reported-replicates.csv``, one row per replicate,
implementation and estimand, and ``reported-cells.csv``, their summary per cell.  A recordable
fitting failure is a reading here, recorded with its error class in the replicate rows and
counted in the summary.  Any other raise is a defect and stops the run.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from cleverly.validation._failures import recordable
from tests.parallel import STUDY_JOBS
from tests.studies import survival_ctmle as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS as PROPERTY_COLUMNS
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed

#: No property fit runs, so no property failure can be recorded against this bound.
FAILURE_BOUND = 0.01
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS
#: The two published files, in ``STUDY.property_artifacts``.
CELLS_FILE = study.REPORTED_CELLS_FILE
REPLICATES_FILE = study.REPORTED_REPLICATES_FILE


def reported_seed(scenario: str, replicate: int) -> int:
    """The seed of replicate ``replicate`` of the reported cell ``scenario``."""
    return stream_seed(study.STUDY, "reported", scenario, replicate)


def _fit(scenario: str, replicate: int) -> list[dict[str, Any]]:
    """Every implementation's rows on one draw of a reported cell."""
    n = study.SCENARIOS[scenario][3]
    frame, truth = study.draw_from_seed(scenario, n, reported_seed(scenario, replicate))
    rows: list[dict[str, Any]] = []
    for implementation in study.IMPLEMENTATIONS:
        start = time.perf_counter()
        base = {"implementation": implementation, "scenario": scenario, "replicate": replicate}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = study.fit(frame, scenario, implementation)
        except Exception as error:
            if not recordable(error):
                raise
            seconds = time.perf_counter() - start
            rows.extend(
                {
                    **base,
                    "n": n,
                    "estimand": name,
                    "truth": float(truth[name]),
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "covered": np.nan,
                    "failure": type(error).__name__,
                    "seconds": seconds,
                }
                for name in study.ESTIMAND_NAMES
            )
            continue
        seconds = time.perf_counter() - start
        for name in study.ESTIMAND_NAMES:
            estimate = result[name]
            std_error, low, high = reported_inference(estimate)
            rows.append(
                {
                    **base,
                    "n": n,
                    "estimand": name,
                    "truth": float(truth[name]),
                    "estimate": float(estimate.psi),
                    "std_error": std_error,
                    "covered": float(low <= truth[name] <= high),
                    "failure": "",
                    "seconds": seconds,
                }
            )
    return rows


def reported_rows(
    *, replicates: int = study.REPORTED_REPLICATES, n_jobs: int = STUDY_JOBS
) -> pd.DataFrame:
    """Every reported cell's replicate rows, in a fixed order."""
    jobs = [
        (scenario, replicate)
        for scenario in study.REPORTED_SCENARIOS
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_fit, jobs, n_jobs=n_jobs)
    rows = pd.DataFrame([row for rows in outcomes for row in rows])
    return rows.sort_values(
        ["scenario", "replicate", "implementation", "estimand"], ignore_index=True
    )


def summarize_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell, implementation and estimand: fits, failures, bias, spread, RMSE, SE, coverage."""
    records = []
    keys = ["scenario", "implementation", "estimand"]
    for (scenario, implementation, estimand), group in rows.groupby(keys, sort=True):
        kept = group.loc[group["failure"].fillna("").eq("")]
        error = kept["estimate"] - kept["truth"]
        spread = float(error.std(ddof=1))
        records.append(
            {
                "scenario": scenario,
                "implementation": implementation,
                "estimand": estimand,
                "n": int(group["n"].iloc[0]),
                "fits": len(kept),
                "failures": len(group) - len(kept),
                "truth": float(group["truth"].iloc[0]),
                "bias": float(error.mean()),
                "empirical_sd": spread,
                "standardized_bias": float(error.mean() / spread),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mean_std_error": float(kept["std_error"].mean()),
                "se_ratio": float(kept["std_error"].mean() / spread),
                "coverage": float(kept["covered"].mean()),
                "mean_seconds": float(group["seconds"].mean()),
            }
        )
    return pd.DataFrame(records)


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS, replicates: int = study.REPORTED_REPLICATES
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """No property rows, and the reported cells' two artifacts."""
    rows = reported_rows(replicates=replicates, n_jobs=n_jobs)
    return (
        pd.DataFrame(columns=list(PROPERTY_COLUMNS)),
        {CELLS_FILE: summarize_cells(rows), REPLICATES_FILE: rows},
    )


def generate_property_rows(*, n_jobs: int = 1) -> pd.DataFrame:
    """No property rows; the record's property artifacts come from the hook above."""
    del n_jobs
    return pd.DataFrame(columns=list(PROPERTY_COLUMNS))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """No property verdicts."""
    del rows
    return pd.DataFrame(columns=["property", "cell", "passed"])

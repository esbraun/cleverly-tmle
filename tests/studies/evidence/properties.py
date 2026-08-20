"""Repeated-sampling property studies: the machinery, not the claims.

A property study asks whether a complete estimator behaves the way its source theory says
it does when applied to samples from a known law.  The cells, laws and learners are the
study's; the sampling loop, the replication accounting, the per-cell verdicts and the rate
estimator are shared, because every method that gets an evidence row needs the same four.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cleverly.validation import CoverageStudy, ReplicationRecord, summarize_replications
from tests.studies.evidence.inference import (
    Interval,
    clopper_pearson,
    percentile_interval,
    standardized_bias_verdict,
)

REPLICATE_COLUMNS = (
    "property",
    "cell",
    "replicate",
    "n",
    "requested_replicates",
    "failed_replicates",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "rejected",
)


@dataclass(frozen=True)
class PropertyCell:
    """One repeated-sampling cell: a law, a nuisance configuration, a size, a seed."""

    property: str
    cell: str
    dgp: Any
    outcome_learner: Callable[[], Any]
    treatment_learner: Callable[[], Any]
    n: int
    replicates: int
    seed: int
    estimand: str = "ate"
    fit_kwargs: dict[str, Any] = field(default_factory=lambda: {"outcome": "Y", "treatment": "A"})


def run_cells(
    cells: Sequence[PropertyCell],
    estimator: Callable[[PropertyCell], Callable[[], Any]],
    *,
    n_jobs: int,
) -> pd.DataFrame:
    """Run every cell and return the per-replication rows.

    Cells run one at a time with the whole core budget handed to the replication loop inside
    each, rather than several cells at once each with a slice.  Two levels of process pool
    over the same cores oversubscribe every one of them; a single level with a few hundred
    independent fits already keeps every worker fed.

    ``requested_replicates`` and ``failed_replicates`` travel on every row.  A replication
    whose fit raises is dropped by :class:`~cleverly.validation.CoverageStudy` -- correctly,
    so one bad draw cannot kill a study -- and a dropped replication silently widens the
    Monte Carlo error of every cell it touches.  Recording the count is what lets the
    verdicts refuse to be computed on a study that quietly shrank.
    """
    frames: list[pd.DataFrame] = []
    for cell in cells:
        result = CoverageStudy(
            dgp=cell.dgp,
            estimator=estimator(cell),
            n=cell.n,
            n_replicates=cell.replicates,
            estimands=(cell.estimand,),
            fit_kwargs=dict(cell.fit_kwargs),
            seed=cell.seed,
            n_jobs=n_jobs,
        ).run()
        records = tuple(
            record for record in result.replications if record.estimand == cell.estimand
        )
        frames.append(
            pd.DataFrame(
                {
                    "property": cell.property,
                    "cell": cell.cell,
                    "replicate": [record.replicate for record in records],
                    "n": cell.n,
                    "requested_replicates": cell.replicates,
                    "failed_replicates": result.n_failed,
                    "truth": [record.truth for record in records],
                    "estimate": [record.estimate for record in records],
                    "std_error": [record.std_error for record in records],
                    "covered": [int(record.covered) for record in records],
                    "rejected": [int(record.rejected) for record in records],
                }
            )
        )
    return pd.concat(frames, ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]


def require_complete(rows: pd.DataFrame) -> None:
    """Refuse a property table that lost replications.

    Every verdict below standardizes by a spread or a rate estimated from these rows, so a
    cell that lost replications is not merely noisier -- for a bias claim it is *easier*,
    because the interval it has to sit inside is estimated from the same shrunken sample.
    """
    for (property_name, cell), group in rows.groupby(["property", "cell"], sort=True):
        requested = int(group["requested_replicates"].iloc[0])
        failed = int(group["failed_replicates"].iloc[0])
        if failed or len(group) != requested:
            raise ValueError(
                f"{property_name}/{cell} has {len(group)} of {requested} replications "
                f"({failed} fits failed); a study that lost replications cannot be summarised "
                f"as though it had not"
            )


def summarize_cells(
    rows: pd.DataFrame,
    *,
    margin: float,
    confidence_level: float,
    alpha: float,
) -> pd.DataFrame:
    """Descriptive summary plus the per-cell interval verdicts, one row per cell."""
    require_complete(rows)
    records: list[dict[str, Any]] = []
    for (property_name, cell), group in rows.groupby(["property", "cell"], sort=True):
        estimates = group["estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        replicates = len(group)
        canonical = summarize_replications(
            tuple(
                ReplicationRecord(
                    replicate=int(row.replicate),
                    seed=-1,
                    estimand="cell",
                    truth=float(row.truth),
                    estimate=float(row.estimate),
                    std_error=float(row.std_error),
                    covered=bool(row.covered),
                    rejected=bool(row.rejected),
                    inference_estimate=float(row.estimate),
                    alpha=alpha,
                )
                for row in group.itertuples(index=False)
            ),
            estimand="cell",
            n=int(group["n"].iloc[0]),
        )
        bias = standardized_bias_verdict(
            estimates - truth, margin=margin, confidence_level=confidence_level
        )
        coverage = clopper_pearson(
            int(group["covered"].sum()), replicates, confidence_level=confidence_level
        )
        rejection = clopper_pearson(
            int(group["rejected"].sum()), replicates, confidence_level=confidence_level
        )
        empirical_se = bias.scale
        records.append(
            {
                "property": property_name,
                "cell": cell,
                "n": int(group["n"].iloc[0]),
                "replicates": replicates,
                "failed_replicates": int(group["failed_replicates"].iloc[0]),
                "truth": truth,
                "mean_estimate": canonical.mean_estimate,
                "bias": canonical.bias,
                "bias_se": canonical.bias_se,
                "bias_ci_lower": bias.interval.low,
                "bias_ci_upper": bias.interval.high,
                "bias_margin": bias.margin,
                "standardized_bias": bias.standardized,
                "bias_equivalent": bias.equivalent,
                "bias_discriminated": bias.discriminated,
                "root_n_bias": canonical.root_n_bias,
                "empirical_se": empirical_se,
                "mean_std_error": canonical.mean_std_error,
                "se_ratio": canonical.se_ratio,
                "coverage": canonical.coverage,
                "coverage_ci_lower": coverage.low,
                "coverage_ci_upper": coverage.high,
                "rejection_rate": canonical.rejection_rate,
                "rejection_ci_lower": rejection.low,
                "rejection_ci_upper": rejection.high,
                "nominal_size": alpha,
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class Rate:
    """A fitted ``log(quantity) ~ log(n)`` slope with a bootstrap interval."""

    slope: float
    interval: Interval

    def consistent_with(self, expected: float) -> bool:
        return self.interval.contains(expected)

    def excludes(self, value: float) -> bool:
        return not self.interval.contains(value)


def _slope(sizes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Least-squares slope of ``values`` on ``log(sizes)``, vectorised over leading axes."""
    x = np.log(sizes)
    centred = x - x.mean()
    return (values * centred).sum(axis=-1) / (centred**2).sum()


def rate(
    rows: pd.DataFrame,
    *,
    property_name: str,
    statistic: str = "spread",
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> Rate:
    """How fast the sampling distribution contracts as ``n`` grows.

    ``statistic="spread"`` regresses the log *empirical* standard deviation of the estimates
    on log ``n``; root-n asymptotics predict a slope of :math:`-1/2`.  This is a property of
    the estimator's sampling distribution and can come out wrong.

    ``statistic="reported"`` does the same for the mean reported standard error.  That one is
    close to arithmetic -- an influence-curve standard error is
    :math:`\\hat\\sigma/\\sqrt{n}`, so any estimator that divides by the right power of ``n``
    produces :math:`-1/2` whether or not it is consistent.  It is kept because it does catch
    a standard error carrying the wrong power of ``n``, and labelled so it is not mistaken
    for evidence of root-n consistency.
    """
    subset = rows.loc[rows["property"] == property_name]
    grouped = [(int(group["n"].iloc[0]), group) for _, group in subset.groupby("cell", sort=True)]
    grouped.sort(key=lambda item: item[0])
    if len(grouped) < 3:
        raise ValueError(
            f"{property_name} has {len(grouped)} sizes; a rate needs at least three so the "
            f"slope is estimated rather than read off one ratio"
        )
    sizes = np.array([size for size, _ in grouped], dtype=float)
    column = "estimate" if statistic == "spread" else "std_error"
    samples = [group[column].to_numpy(dtype=float) for _, group in grouped]

    def observed(values: np.ndarray) -> float:
        return float(
            np.log(np.std(values, ddof=1)) if statistic == "spread" else np.log(values.mean())
        )

    point = _slope(sizes, np.array([observed(values) for values in samples]))

    draws = np.empty((bootstrap_replicates, len(sizes)), dtype=float)
    for index, values in enumerate(samples):
        # A separate stream per size: the sizes are independent runs, and resampling them
        # with shared indices would pretend they were paired.
        rng = np.random.default_rng(seed + index)
        picks = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
        resampled = values[picks]
        draws[:, index] = (
            np.log(resampled.std(axis=1, ddof=1))
            if statistic == "spread"
            else np.log(resampled.mean(axis=1))
        )
    return Rate(
        slope=float(point),
        interval=percentile_interval(_slope(sizes, draws), confidence_level=confidence_level),
    )

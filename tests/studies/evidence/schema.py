"""The per-replication artefact contract, and the integrity checks every study gets.

The replication file is the only primary artefact: every summary, verdict and quoted number
is recomputed from it.  That makes its integrity the one thing nothing downstream can catch,
so it is checked here rather than in any single study.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence.registry import StudyRecord

REPLICATE_COLUMNS = (
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

#: The scales an implementation may report its standard error on.  ``paf`` is why there are
#: three: ``tmle3`` reports it as a log-risk-ratio quantity mapped through ``1 - exp(-x)``.
INFERENCE_SCALES = frozenset({"identity", "log", "negative_log_complement"})

#: A per-cell coverage below this is not a low-coverage finding, it is a broken join.
PLAUSIBLE_COVERAGE = 0.5

#: Relative slack on truth constancy, for the round trip through the reference container.
TRUTH_TOLERANCE = 1e-9


def reported_inference(estimate: Any) -> tuple[float, float, float]:
    """``(std_error, low, high)`` for a replicate row, under any inference status.

    Read through ``plugin_std_error`` and ``plugin_interval`` with no branch.  Those two
    accessors call the private bodies ``std_error`` and ``ci`` call, so an estimate whose
    inference the package supplies gives the numbers it always gave, bit for bit, and a
    selector-path collaborative estimate gives its retained working-mechanism
    diagnostic, which is the same arithmetic on the same curve.  The row records which
    of the two it measured only through the study that wrote it: the evidence pages say
    so for the two selector rows.

    One reader for every study is what keeps one rule here.
    ``multi_arm_common.rows_from_result`` and ``point_study_helpers.primary_rows`` are
    shared with studies that keep inferential estimates, and they read through this.

    The row schema does not move.  ``REPLICATE_COLUMNS`` is the shared contract of every
    registered study, and ``std_error``/``ci_lower``/``ci_upper``/``covered`` stay: the
    evidence pages say what those columns measure for the two selector rows.

    Parameters
    ----------
    estimate : Any
        A :class:`~cleverly.inference.ParameterEstimate` from a fitted result.

    Returns
    -------
    tuple of float
        The standard error and the interval endpoints the row records.
    """
    low, high = estimate.plugin_interval
    return float(estimate.plugin_std_error), float(low), float(high)


def truth_on_inference_scale(estimand: str, truth: float, scale: str) -> float:
    """Map truth to the scale an implementation reports its standard error on."""
    if scale == "identity":
        return truth
    if scale == "log":
        return math.log(truth)
    if scale == "negative_log_complement":
        return -math.log1p(-truth)
    raise ValueError(f"unknown inference scale {scale!r} for {estimand}")


def validate_replicates(rows: pd.DataFrame, *, record: StudyRecord) -> None:
    """Refuse a replication file that cannot be what it claims to be.

    The failure this exists for is a mis-joined truth column: every downstream summary stays
    arithmetically self-consistent while measuring the wrong thing, and a coverage column
    that came out all-zero or all-one reads as a catastrophic estimator rather than as a bad
    merge.  Truth constancy and the coverage-from-endpoints identity catch it directly, which
    a range check on the coverage column alone does not.
    """
    if tuple(rows.columns) != REPLICATE_COLUMNS:
        raise ValueError(
            f"{record.slug}: replication columns are {tuple(rows.columns)}, "
            f"expected {REPLICATE_COLUMNS}"
        )
    unknown = set(rows["inference_scale"]) - INFERENCE_SCALES
    if unknown:
        raise ValueError(f"{record.slug}: unknown inference scales {sorted(unknown)}")
    if set(rows["implementation"]) != set(record.implementations):
        raise ValueError(
            f"{record.slug}: implementations {sorted(set(rows['implementation']))} "
            f"do not match the declared {record.implementations}"
        )
    if set(rows["scenario"]) != set(record.scenarios):
        raise ValueError(
            f"{record.slug}: scenarios {sorted(set(rows['scenario']))} do not match "
            f"the declared {sorted(record.scenarios)}"
        )
    if not np.isin(rows["covered"], (0, 1)).all():
        raise ValueError(f"{record.slug}: the coverage column is not an indicator")
    if not np.isfinite(rows[["estimate", "inference_estimate", "std_error"]].to_numpy()).all():
        raise ValueError(f"{record.slug}: non-finite estimates or standard errors")
    if not (rows["std_error"] > 0.0).all():
        raise ValueError(f"{record.slug}: non-positive standard errors")
    if not (rows["ci_lower"] <= rows["ci_upper"]).all():
        raise ValueError(f"{record.slug}: inverted confidence intervals")

    recomputed = ((rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])).astype(
        int
    )
    disagreeing = int((recomputed != rows["covered"]).sum())
    if disagreeing:
        raise ValueError(
            f"{record.slug}: {disagreeing} rows whose coverage indicator disagrees with "
            f"their own interval and truth"
        )

    for (scenario, estimand), group in rows.groupby(["scenario", "estimand"], sort=True):
        if estimand not in record.scenarios[str(scenario)]:
            raise ValueError(f"{record.slug}: {scenario} reports undeclared estimand {estimand}")
        spread = float(group["truth"].max() - group["truth"].min())
        # Not exact equality: the truth travels to the reference implementation through a
        # CSV and back, so the two sides can differ in the last bit or two.  A mis-joined
        # truth differs by orders of magnitude, not by an ulp.
        if spread > TRUTH_TOLERANCE * max(1.0, abs(float(group["truth"].iloc[0]))):
            raise ValueError(
                f"{record.slug}: truth for {scenario}/{estimand} varies by {spread:g} across "
                f"rows; the truth column is joined wrong"
            )
    for scenario, estimands in record.scenarios.items():
        subset = rows.loc[rows["scenario"] == scenario]
        if set(subset["estimand"]) != set(estimands):
            raise ValueError(
                f"{record.slug}: {scenario} has estimands {sorted(set(subset['estimand']))}, "
                f"declared {sorted(estimands)}"
            )
    counts = rows.groupby(["implementation", "scenario", "estimand"], sort=True).size()
    wrong = counts[counts != record.replicates]
    if not wrong.empty:
        raise ValueError(
            f"{record.slug}: cells with a replication count other than {record.replicates}:\n"
            f"{wrong.to_string()}"
        )
    if not (rows["n"] == record.n).all():
        raise ValueError(f"{record.slug}: rows with a sample size other than {record.n}")
    coverage = rows.groupby(["implementation", "scenario", "estimand"], sort=True)["covered"].mean()
    implausible = coverage[coverage < PLAUSIBLE_COVERAGE]
    if not implausible.empty:
        raise ValueError(
            f"{record.slug}: cells whose coverage is below {PLAUSIBLE_COVERAGE:.0%}, which is a "
            f"broken join rather than a low-coverage finding:\n{implausible.to_string()}"
        )

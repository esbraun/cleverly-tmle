"""Shared verdict policy for repeated-sampling property studies.

Concrete study modules own their laws, learners, cells, and study-specific thresholds.  This
module owns the cross-study rate, power, calibration, and table-finalization rules so no method
family depends on another method family's study module to publish the same claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    rate,
    ratio_intervals,
    summarize_cells,
    summary_interval,
)
from tests.studies.evidence.registry import Margins, StudyRecord
from tests.studies.evidence.seeds import stream_seed

#: The contraction rate predicted by root-n asymptotics, the slower alternative the study must
#: exclude, and the equivalence margin halfway between them.
ROOT_N_SLOPE = -0.5
EXCLUDED_SLOPE = -0.25
ROOT_N_SLOPE_MARGIN = 0.125

#: A power control must reject often enough that an inert test cannot pass the type-I cell.
MINIMUM_POWER = 0.80


#: Columns every study family's property summary carries, whatever else it adds.
SHARED_COLUMNS = (
    "rate_sizes",
    "slope",
    "slope_ci_lower",
    "slope_ci_upper",
    "se_ratio_ci_lower",
    "se_ratio_ci_upper",
)

#: What a calibration cell publishes in addition when the study can supply an exact efficiency
#: bound.  Added to the summary only for such a study, so a report that claims nothing about
#: the bound does not carry six empty columns saying it might have.
EFFICIENCY_COLUMNS = (
    "efficiency_empirical_ratio",
    "efficiency_empirical_ci_lower",
    "efficiency_empirical_ci_upper",
    "efficiency_reported_ratio",
    "efficiency_reported_ci_lower",
    "efficiency_reported_ci_upper",
)


def apply_shared_verdicts(
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    extra_columns: Sequence[str] = (),
    rate_labels: Sequence[str] = ("",),
    efficiency_bounds: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Every cell the study families share, plus the two rate rows they both publish.

    Each verdict is an interval statement against a margin declared before the run.  For the
    double-robustness cells that means the positive cells must establish that the bias is
    *inside* the margin and the both-wrong control must establish that it is *outside* it --
    the same instrument in both directions, so neither can be passed by a study too small to
    say anything.

    Shared rather than copied per study: several method families publish these rules, so a
    verdict written twice is a verdict that can be *changed* once.  The caller supplies whatever
    further columns its own cells publish, because the rate rows are built from the summary's
    columns and have to be built after they all exist.

    ``rate_labels`` is for a study that reports the same size ladder for more than one
    parameter -- the longitudinal report runs one for a static contrast and one for a dynamic
    rule.  Each label selects the cells named ``f"{label}__..."`` and publishes its own pair
    of rate rows under the same prefix.  The default single empty label is the one-parameter
    case and reproduces the unprefixed cell names and seeds exactly.

    ``efficiency_bounds`` maps each of those labels to an independently computed
    :math:`\\sqrt{E_P[D^*(O)^2]}` -- available on a finite-support law, where it comes off a
    Gateaux derivative rather than off anything the estimator reports.  Given one, the six
    :data:`EFFICIENCY_COLUMNS` are filled from the *same* draws as the SE ratio, so the three
    published intervals agree with the arithmetic relating them.  The verdict stays the
    caller's: the band a ratio has to sit inside belongs to the study that can compute a bound
    at all.
    """
    margins = record.margins
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
    )
    efficiency = () if efficiency_bounds is None else EFFICIENCY_COLUMNS
    for column in (*SHARED_COLUMNS, *efficiency, *extra_columns):
        summary[column] = np.nan
    summary["passed"] = False
    # Object dtype, not the NaN float the numeric columns above get: a property with a
    # cross-row claim writes booleans into this, and ``.loc`` will not put one into a
    # float64 column.  ``None`` means "no joint claim beyond this row's own", which
    # :func:`finish` resolves.
    summary["property_passed"] = pd.Series([None] * len(summary), dtype=object, index=summary.index)

    robustness = summary["property"] == "double_robustness"
    positive = robustness & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    control = robustness & (summary["role"] == "control")
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    efficiency = summary["property"] == "root_n_and_efficiency"
    sizes = efficiency & (summary["role"] == "positive")
    summary.loc[sizes, "passed"] = (
        (summary.loc[sizes, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[sizes, "se_ratio"].between(*margins.se_ratio_sanity)
        & summary.loc[sizes, "bias_equivalent"]
    )
    # A size retained as a control is one whose inference the study does not claim -- the
    # smallest rung of a ladder that still has to be fitted for the rate.  What it must do is
    # *resolve*: land its exact coverage interval clear of nominal on one side or the other.
    # Below, and the study has established a small-sample limitation and published it rather
    # than widening a margin until it disappeared.  Above the floor, and the estimator turned
    # out to be adequate there after all, which is a result and not a failure -- a control
    # written as "must undercover" fails the day the code improves, which is the one direction
    # a gate must never point.  An interval straddling nominal is what fails, because it is
    # the case that says nothing in either direction.
    small = efficiency & (summary["role"] == "control")
    summary.loc[small, "passed"] = (
        summary.loc[small, "coverage_ci_upper"] < 1.0 - margins.alpha
    ) | (summary.loc[small, "coverage_ci_lower"] >= margins.coverage_floor)

    calibration = summary["property"] == "interval_calibration"
    for cell in sorted(summary.loc[calibration, "cell"]):
        group = rows.loc[(rows["property"] == "interval_calibration") & (rows["cell"] == cell)]
        bound = None if efficiency_bounds is None else efficiency_bounds[cell.split("__", 1)[0]]
        intervals = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(record, "interval_calibration", cell),
            bound=bound,
        )
        ratio = intervals["se_ratio"]
        mask = calibration & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = ratio.low
        summary.loc[mask, "se_ratio_ci_upper"] = ratio.high
        if bound is not None:
            scale = float(np.sqrt(int(group["n"].iloc[0]))) / bound
            for kind, point in (
                ("empirical", float(group["estimate"].std(ddof=1) * scale)),
                ("reported", float(group["std_error"].mean() * scale)),
            ):
                interval = intervals[f"efficiency_{kind}"]
                summary.loc[mask, f"efficiency_{kind}_ratio"] = point
                summary.loc[mask, f"efficiency_{kind}_ci_lower"] = interval.low
                summary.loc[mask, f"efficiency_{kind}_ci_upper"] = interval.high
        row = summary.loc[mask].iloc[0]
        coverage = Interval(float(row["coverage_ci_lower"]), float(row["coverage_ci_upper"]))
        # Two-sided, and both halves needed.  A reported standard error inflated by a
        # constant keeps coverage inside its band while failing the ratio; a curve that is
        # right on average but wrong replication by replication does the reverse.
        summary.loc[mask, "passed"] = bool(
            ratio.within(*margins.calibration_se_ratio)
            and coverage.within(*margins.calibration_coverage)
        )

    null = summary["property"] == "type_i_error"
    # One-sided: a test that over-rejects is invalid, and one that under-rejects is
    # conservative.  The power cell below is what stops "never rejects" from passing here.
    summary.loc[null, "passed"] = (
        summary.loc[null, "rejection_ci_upper"] <= margins.alpha + margins.type_i_margin
    ) & (summary.loc[null, "coverage_ci_lower"] >= margins.coverage_floor)

    power = summary["property"] == "power"
    summary.loc[power, "passed"] = summary.loc[power, "rejection_ci_lower"] >= MINIMUM_POWER

    rates: list[dict[str, Any]] = []
    ladder = rows.loc[rows["property"] == "root_n_and_efficiency"]
    for label in rate_labels:
        prefix = f"{label}__" if label else ""
        selected = ladder.loc[ladder["cell"].str.startswith(prefix)]
        for statistic, suffix in (("spread", "empirical_sd"), ("reported", "reported_se")):
            _rate_row(
                rates,
                selected,
                record,
                summary.columns,
                label=label,
                cell=f"{prefix}{suffix}",
                statistic=statistic,
                suffix=suffix,
            )
    return summary, rates


def _rate_row(
    rates: list[dict[str, Any]],
    rows: pd.DataFrame,
    record: StudyRecord,
    columns: Any,
    *,
    label: str,
    cell: str,
    statistic: str,
    suffix: str,
) -> None:
    """One fitted contraction rate, appended as a published row."""
    margins = record.margins
    fitted = rate(
        rows,
        property_name="root_n_and_efficiency",
        statistic=statistic,
        bootstrap_replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        # The label joins the stream only when there is one, so a single-parameter study
        # keeps the seed -- and therefore the published slope -- it already had.
        seed=stream_seed(record, "root_n_rate", *([label] if label else []), suffix),
    )
    row: dict[str, Any] = dict.fromkeys(columns, np.nan)
    sizes = sorted({int(value) for value in rows["n"]})
    row.update(
        {
            "property": "root_n_rate",
            "cell": cell,
            "role": "positive",
            "n": max(sizes),
            "replicates": len(rows),
            # The slope is fitted across all three sizes.  ``n`` and ``replicates`` above
            # are the largest and the sum, which read as one big cell; this is what the
            # published table shows instead.
            "rate_sizes": ";".join(f"{size:,}" for size in sizes),
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


def calibration_controls(
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    labels: Sequence[str],
    efficiency_bounds: Mapping[str, float],
    calibration_n: int,
    shrunken_se_factor: float,
    critical: float,
) -> pd.DataFrame:
    """The calibration cell's two deliberately invalid arms, derived from its own rows.

    Neither control needs a fit.  ``shrunken_se_control`` multiplies the reported standard
    error by a declared factor below one, so it must fail the SE-ratio band while the estimates
    stay exactly where they were.  ``noise_control`` adds one efficiency-bound unit of
    independent noise to each estimate, so the *empirical* ratio must rise above the band while
    the reported standard errors stay where they were.  Between them the two arms move each
    half of the calibration claim on its own, which is what stops a cell that is right on
    average and wrong replication by replication from passing.

    A pure transformation of ``rows``: it draws no sample and fits nothing, so a study gains
    these arms without spending replications on them.  Shared rather than copied because the
    arithmetic is the same wherever the claim is, and only the declared constants differ.
    """
    source = rows.loc[
        (rows["property"] == "interval_calibration")
        & rows["cell"].str.endswith("__correctly_specified")
    ]
    controls: list[pd.DataFrame] = []
    for label in labels:
        base = source.loc[source["cell"] == f"{label}__correctly_specified"].copy()
        shrunken = base.copy()
        shrunken["cell"] = f"{label}__shrunken_se_control"
        shrunken["role"] = "control"
        shrunken["std_error"] *= shrunken_se_factor
        controls.append(_recompute_interval_columns(shrunken, critical))

        noisy = base.copy()
        rng = np.random.default_rng(stream_seed(record, "efficiency_noise", label))
        noisy["cell"] = f"{label}__noise_control"
        noisy["role"] = "control"
        noisy["estimate"] += rng.normal(
            scale=efficiency_bounds[label] / np.sqrt(calibration_n), size=len(noisy)
        )
        controls.append(_recompute_interval_columns(noisy, critical))
    return pd.concat(controls, ignore_index=True)


def _recompute_interval_columns(rows: pd.DataFrame, critical: float) -> pd.DataFrame:
    """Rebuild ``covered`` and ``rejected`` after a mutation moved an estimate or an SE."""
    half = critical * rows["std_error"]
    rows["covered"] = (
        (rows["estimate"] - half <= rows["truth"]) & (rows["truth"] <= rows["estimate"] + half)
    ).astype(int)
    rows["rejected"] = (np.abs(rows["estimate"] / rows["std_error"]) > critical).astype(int)
    return rows


def calibration_verdicts(
    summary: pd.DataFrame, *, margins: Margins, efficiency_band: tuple[float, float]
) -> None:
    """Read each calibration cell's verdict against the rule its *kind* answers to.

    :func:`apply_shared_verdicts` writes the one-kind rule, which is right for a study whose
    calibration cell has no controls.  A study that declares the two arms above has three kinds
    held to three different rules, and each row must publish the verdict of its own kind rather
    than the positive arm's.  Written here rather than per study for the reason the arms are:
    the rules belong to the instrument, and only the band belongs to the study that can compute
    an exact bound at all.
    """
    calibration = summary["property"] == "interval_calibration"
    for index in summary.index[calibration]:
        kind = str(summary.loc[index, "cell"]).split("__", 1)[1]
        ratio = summary_interval(summary, index, "se_ratio")
        empirical = summary_interval(summary, index, "efficiency_empirical")
        reported = summary_interval(summary, index, "efficiency_reported")
        coverage = summary_interval(summary, index, "coverage")
        if kind == "correctly_specified":
            passed = (
                ratio.within(*margins.calibration_se_ratio)
                and coverage.within(*margins.calibration_coverage)
                and empirical.within(*efficiency_band)
                and reported.within(*efficiency_band)
            )
        elif kind == "shrunken_se_control":
            passed = ratio.high < margins.calibration_se_ratio[0]
        elif kind == "noise_control":
            passed = empirical.low > efficiency_band[1]
        else:
            raise ValueError(f"unknown calibration cell kind {kind!r}")
        summary.loc[index, "passed"] = bool(passed)


def finish(summary: pd.DataFrame, rates: list[dict[str, Any]]) -> pd.DataFrame:
    """Append the rate rows and put the table in its published order.

    ``property_passed`` defaults to the row's own verdict and is overwritten only by a
    property whose claim needs more than one cell to establish.  Publishing
    both columns is what lets a row state its own rule without losing the joint claim: the
    alternative, broadcasting one scalar across the property, made a deliberately in-sample
    control report the cross-fit arm's verdict as though it were its own.
    """
    summary = pd.concat([summary, pd.DataFrame(rates)], ignore_index=True)
    summary["passed"] = summary["passed"].astype(bool)
    # ``where`` rather than a whole-column default: the rate rows are built from
    # ``dict.fromkeys(summary.columns, nan)``, and a property with a cross-row claim wrote
    # the column only on its own rows.  Everything else is left unset, and ``astype(bool)``
    # would silently publish both ``NaN`` and ``None`` as ``True``.
    joint = summary["property_passed"]
    summary["property_passed"] = joint.where(joint.notna(), summary["passed"]).astype(bool)
    return summary.sort_values(["property", "cell"], ignore_index=True)

"""Shared verdict policy for repeated-sampling property studies.

Concrete study modules own their laws, learners, cells, and study-specific thresholds.  This
module owns the cross-study rate, power, calibration, and table-finalization rules so no method
family depends on another method family's study module to publish the same claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tests.studies.evidence.inference import Interval, standardized_bias_verdict
from tests.studies.evidence.properties import (
    Rate,
    coverage_gain_interval,
    paired_displacement,
    rate,
    ratio_intervals,
    se_ratio_interval,
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

#: How far a ``double_robustness`` cell's reported standard error may sit from the sampling
#: spread it estimates, before the cell stops describing the union model it names.
#:
#: An order of magnitude, and deliberately *not* :attr:`~Margins.se_ratio_sanity`.  Every cell in
#: this family fits at least one nuisance wrong on purpose, so its influence curve is not the
#: efficient one and no theory predicts that the standard error it reports is calibrated.  The
#: register shows that departure is real and points both ways: the end-of-study longitudinal
#: row's ``static__outcome_correct`` reports 0.61 beside a coverage of 0.7708, and the
#: stochastic-regime row's ``outcome_correct`` reports 2.31 beside a coverage of 1.0000.  Both
#: are the union model behaving as the theory allows, and
#: :attr:`~Margins.over_coverage_ceiling` already records that conservative inference is
#: reported rather than failed.  Holding this family to the calibrated screen would publish a
#: defect that neither estimator has.
#:
#: What the band excludes is a fit that collapsed.  A univariate guard regression handed a
#: constant single regressor reported a standard error 87.6 times its own empirical spread, and
#: the ``both_wrong`` control passed anyway, because the verdict read the bias and nothing read
#: the reported error at all.  A misspecified nuisance is still a regression whose influence
#: curve has the scale of the data.  Two orders of magnitude is a numerical failure wearing a
#: misspecification's name, and the bias verdict beside it is then evidence about the failure
#: rather than about the union model.
#:
#: The screen therefore binds on no registered cell, which is what a screen should do.  It is a
#: rule nothing can fail until something breaks, so
#: ``test_a_collapsed_reported_error_fails_the_union_model_cells`` mutates a committed cell and
#: requires the verdict to move.
UNION_MODEL_SE_BAND = (0.1, 10.0)

#: The three margins the ``crossfit_overfitting`` family answers to.  Shared rather than owned
#: by the study that first declared them: four families now make the same three statements
#: about a paired cross-fit and in-sample arm, and a margin written four times is a margin
#: that can be moved in one place and left stale in three.
#:
#: ``OVERFIT_SE_FLOOR`` is the cross-fit arm's floor, and the sanity band's *upper* limit is
#: its ceiling, so the arm is held to the same screen as any other estimator rather than to a
#: bound this family invented.  ``OVERFIT_SE_CONTROL_CEILING`` is what the in-sample arm must
#: fall below for the control to be the failure it claims to be.
OVERFIT_SE_FLOOR = 0.85
OVERFIT_SE_CONTROL_CEILING = 0.75
OVERFIT_COVERAGE_GAIN = 0.15

#: The two margins the ``clustered_inference`` family answers to.  Beside the overfitting
#: block for the reason that block gives: both families make the same three statements about
#: a positive cell, a deliberately wrong control, and the coverage the pair buys, so
#: :func:`_paired_cell_verdicts` states the rule once and the margins it reads live in one
#: module rather than inside a single study.
#:
#: The positive arm answers to :attr:`~Margins.calibration_se_ratio` and
#: :attr:`~Margins.calibration_coverage`, the band every calibrated cell answers to, so this
#: family declares no floor of its own.  ``CLUSTER_ROBUST_CONTROL_SE_CEILING`` is what the
#: IID control must fall below for the control to be the failure it claims to be, and
#: ``CLUSTERED_COVERAGE_GAIN`` is the coverage the cluster-robust variance has to buy over
#: that control on the same draws.
CLUSTER_ROBUST_CONTROL_SE_CEILING = 0.80
CLUSTERED_COVERAGE_GAIN = 0.03

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
    say anything.  Every cell of that family answers to :data:`UNION_MODEL_SE_BAND` as well,
    because a bias endpoint says nothing about the union model when the fit that produced it
    reported an error two orders of magnitude off its own spread.

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
    # Both roles answer to the same screen, and each keeps its own bias endpoint.  A positive
    # cell claims the bias is inside the margin and a control claims it is outside, but neither
    # is a claim about the union model unless the fit reported an error on the scale of its own
    # spread.  The control is where this was found: it is the arm whose bias endpoint a
    # collapsed nuisance cannot fail, so bias alone published a pass for a cell that reported
    # 87.6 times its empirical spread.
    scaled = summary["se_ratio"].between(*UNION_MODEL_SE_BAND)
    positive = robustness & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"] & scaled
    control = robustness & (summary["role"] == "control")
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"] & scaled

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


def fitted_rate_row(
    rows: pd.DataFrame,
    record: StudyRecord,
    columns: Any,
    *,
    ladder_property: str,
    property_name: str,
    cell: str,
    role: str,
    statistic: str,
    seed_labels: Sequence[str],
    verdict: Callable[[Rate], bool],
) -> dict[str, Any]:
    """One fitted slope over a size ladder, shaped as a published summary row.

    The fit and the row shape are here; the *rule* the slope answers to is the caller's,
    because a rate row means different things to different families.  Root-n contraction of a
    sampling spread has a predicted exponent and a slower alternative to exclude, while a
    bias-contraction claim only has to establish a direction.  Trying to state both as one
    threshold would produce a rule neither family wanted.

    Shared because the alternative is what this function replaced: a second family fitting a
    slope copied the seed derivation, the bootstrap arguments, the ``rate_sizes`` formatting
    and all nine row keys, so the published schema for a rate row existed twice and could be
    changed in one of them.

    Parameters
    ----------
    rows : pandas.DataFrame
        The ladder's replication rows, already restricted to this cell's arm.
    record : StudyRecord
        Supplies the resampling budget, the confidence level and the seed stream.
    columns : Any
        The summary's columns, so the row carries every one the table publishes.
    ladder_property : str
        The property the ladder's replication rows are filed under.  Not always the label the
        row publishes: the root-n ladder is filed under ``root_n_and_efficiency`` and its
        fitted slope publishes as ``root_n_rate``, so a single name would have to be wrong in
        one place or the other.
    property_name, cell, role : str
        How the row is named and which direction its verdict points.
    statistic : str
        Which quantity contracts -- see :func:`~tests.studies.evidence.properties.rate`.
    seed_labels : Sequence[str]
        Labels naming this row's resampling stream, after the property name.
    verdict : Callable[[Rate], bool]
        The family's rule, applied to the fitted rate.

    Returns
    -------
    dict[str, Any]
        The published row.
    """
    margins = record.margins
    fitted = rate(
        rows,
        property_name=ladder_property,
        statistic=statistic,
        bootstrap_replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(record, *seed_labels),
    )
    sizes = sorted({int(value) for value in rows["n"]})
    row: dict[str, Any] = dict.fromkeys(columns, np.nan)
    row.update(
        {
            "property": property_name,
            "cell": cell,
            "role": role,
            "n": max(sizes),
            "replicates": len(rows),
            # The slope is fitted across all the sizes.  ``n`` and ``replicates`` above
            # are the largest and the sum, which read as one big cell; this is what the
            # published table shows instead.
            "rate_sizes": ";".join(f"{size:,}" for size in sizes),
            "failed_replicates": 0,
            "slope": fitted.slope,
            "slope_ci_lower": fitted.interval.low,
            "slope_ci_upper": fitted.interval.high,
            "passed": bool(verdict(fitted)),
        }
    )
    return row


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
    """One fitted root-n contraction rate, appended as a published row."""
    rates.append(
        fitted_rate_row(
            rows,
            record,
            columns,
            ladder_property="root_n_and_efficiency",
            property_name="root_n_rate",
            cell=cell,
            role="positive",
            statistic=statistic,
            # The label joins the stream only when there is one, so a single-parameter study
            # keeps the seed -- and therefore the published slope -- it already had.
            seed_labels=("root_n_rate", *([label] if label else []), suffix),
            verdict=lambda fitted: (
                fitted.equivalent_to(ROOT_N_SLOPE, ROOT_N_SLOPE_MARGIN)
                and fitted.excludes(EXCLUDED_SLOPE)
            ),
        )
    )


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


def necessity_verdicts(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    family: str,
    labels: Sequence[str],
    arms: tuple[str, str],
    column: str,
    threshold: float,
) -> None:
    """Was the step load bearing, and did it carry the estimate the right way?

    A *necessity* family pairs a positive arm against a control that removes exactly one step
    -- the fluctuation, the backward recursion, the declared projection measure -- and leaves
    everything else, including the draw, alone.  Two claims come out of that, and the family
    needs both.  Each row's own endpoint says the positive arm's bias is inside the
    equivalence margin and the control's is outside it.  But a step that did nothing at all
    would satisfy *neither* arm's rule in a way that distinguishes it, because the control
    would then simply be the estimate.  So the pair needs a statement of its own, and it is
    the displacement: how far the control's mean sits from the positive arm's, in that arm's
    empirical spread, against a threshold the study declared before the run.

    Where a family reports more than one label, the published figure is the *least* displaced
    pair rather than the average.  The claim is that the step matters for every parameter the
    family reports, and an average lets a well-displaced label carry an inert one.

    Written here rather than per study for the reason
    :func:`crossfit_overfitting_verdicts` is.  Seven property modules published this rule --
    ``targeting_necessity`` in six of them, plus a survival recursion, a competing-risk
    recursion and a projection measure -- and a rule written seven times is a rule that can be
    changed in one of them.  The study supplies only its labels, its two arm suffixes, the
    column it publishes the displacement in, and the threshold.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary, which this writes ``passed``, ``column`` and ``property_passed`` on.
    rows : pandas.DataFrame
        The replication rows both arms were emitted from, paired on ``replicate``.
    family : str
        The property the cells are filed under.
    labels : Sequence[str]
        The parameters the family reports.  Cell names are ``f"{label}__{arm}"``.
    arms : tuple[str, str]
        The positive and control suffixes, in that order.
    column : str
        The summary column the joint displacement is published in.
    threshold : float
        The declared minimum displacement.
    """
    mask = summary["property"] == family
    if not mask.any():
        return
    positive_arm, control_arm = arms
    positive = mask & (summary["role"] == "positive")
    control = mask & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    displacement = min(
        paired_displacement(rows, family, f"{label}__{positive_arm}", f"{label}__{control_arm}")
        for label in labels
    )
    summary.loc[mask, column] = displacement
    summary.loc[mask, "property_passed"] = bool(
        summary.loc[mask, "passed"].all() and displacement >= threshold
    )


def alternative_target_necessity_verdicts(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    family: str,
    labels: Sequence[str],
    arms: tuple[str, str],
    alternative_truths: Mapping[str, float],
    column: str,
    threshold: float,
) -> None:
    """Require a declared analysis choice to select between two exact targets.

    Some necessity controls estimate a different valid parameter. An omitted observation
    weight, for example, targets the selected population instead of producing an arbitrary
    wrong number. A population-target bias check proves the omission matters. It does not
    prove the control converges to the alternative target that explains the failure.

    This helper adds that second direction to :func:`necessity_verdicts`. The positive arm
    must recover the study truth. The control must miss that truth and recover its declared
    alternative truth. The paired displacement must also clear the study's threshold.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary, which this function updates with both target verdicts.
    rows : pandas.DataFrame
        Paired replication rows for the positive and control arms.
    record : StudyRecord
        Supplies the standardized-bias margin and confidence level.
    family : str
        The property family that owns the paired cells.
    labels : Sequence[str]
        Parameter labels used as cell-name prefixes.
    arms : tuple[str, str]
        Positive and control cell suffixes, in that order.
    alternative_truths : Mapping[str, float]
        Exact control target for each label.
    column : str
        Summary column that publishes the minimum paired displacement.
    threshold : float
        Declared minimum displacement in positive-arm empirical standard deviations.
    """
    mask = summary["property"] == family
    if not mask.any():
        return
    missing = sorted(set(labels) - set(alternative_truths))
    if missing:
        raise ValueError(f"{family} has no alternative truth for {missing}")

    numeric = (
        "alternative_truth",
        "alternative_bias_ci_lower",
        "alternative_bias_ci_upper",
        "alternative_bias_margin",
    )
    for name in numeric:
        if name not in summary:
            summary[name] = np.nan
    if "alternative_bias_equivalent" not in summary:
        summary["alternative_bias_equivalent"] = pd.Series(
            [None] * len(summary), dtype=object, index=summary.index
        )
    else:
        summary["alternative_bias_equivalent"] = summary["alternative_bias_equivalent"].astype(
            object
        )
    if column not in summary:
        summary[column] = np.nan

    positive_arm, control_arm = arms
    positive = mask & (summary["role"] == "positive")
    control = mask & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"]
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"]

    alternative_passed: list[bool] = []
    displacements: list[float] = []
    for label in labels:
        control_cell = f"{label}__{control_arm}"
        group = rows.loc[(rows["property"] == family) & (rows["cell"] == control_cell)]
        if group.empty:
            raise ValueError(f"{family} has no rows for {control_cell}")
        truth = float(alternative_truths[label])
        verdict = standardized_bias_verdict(
            group["estimate"].to_numpy(dtype=float) - truth,
            margin=record.margins.standardized_bias,
            confidence_level=record.margins.confidence_level,
        )
        cell_mask = mask & (summary["cell"] == control_cell)
        if int(cell_mask.sum()) != 1:
            raise ValueError(f"{family} has {int(cell_mask.sum())} summary rows for {control_cell}")
        summary.loc[cell_mask, "alternative_truth"] = truth
        summary.loc[cell_mask, "alternative_bias_ci_lower"] = verdict.interval.low
        summary.loc[cell_mask, "alternative_bias_ci_upper"] = verdict.interval.high
        summary.loc[cell_mask, "alternative_bias_margin"] = verdict.margin
        summary.loc[cell_mask, "alternative_bias_equivalent"] = verdict.equivalent
        summary.loc[cell_mask, "passed"] = bool(
            summary.loc[cell_mask, "passed"].iloc[0] and verdict.equivalent
        )
        alternative_passed.append(bool(verdict.equivalent))
        displacements.append(
            paired_displacement(
                rows,
                family,
                f"{label}__{positive_arm}",
                control_cell,
            )
        )

    displacement = min(displacements)
    summary.loc[mask, column] = displacement
    summary.loc[mask, "property_passed"] = bool(
        summary.loc[mask, "passed"].all() and all(alternative_passed) and displacement >= threshold
    )


def _paired_cell_verdicts(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    family: str,
    positive_cell: str,
    control_cell: str,
    positive_rule: Callable[[Interval, Interval, Margins], bool],
    control_ceiling: float,
    gain_floor: float,
) -> None:
    """One positive cell, one deliberately wrong control, and the gain the pair buys.

    Three statements, and they do not all belong to the same row.  The positive arm claims a
    standard error its own family calls honest.  The control claims the opposite, that a
    deliberately wrong variance understates it by a wide margin.  The third is about the
    *pair*: the positive arm has to buy coverage the control does not have, on the same
    draws, which is why the two arms share a seed.

    Each row therefore publishes its own verdict in ``passed`` and the paired clause in
    ``property_passed``.  One scalar broadcast across the property published the positive
    arm's rule beside a control whose SE ratio was 0.58 and whose coverage was 0.65, so a
    reader could not tell which statement the "Pass" belonged to.

    Only ``positive_rule`` differs between the families that call this.  Everything else --
    the two resampled SE ratios, the paired coverage interval, the control's ceiling and the
    six published columns -- is one statement written once, for the reason
    :func:`~tests.studies.evidence.properties.coverage_gain_interval` is: a statistic
    written twice is a statistic that can be changed once.

    Parameters
    ----------
    summary : pandas.DataFrame
        The cell summary, which this writes the two intervals, ``passed`` and
        ``property_passed`` on.
    rows : pandas.DataFrame
        The replication rows both arms were emitted from, paired on ``replicate``.
    record : StudyRecord
        Supplies the margins and the resampling streams.
    family : str
        The property the two cells are filed under.
    positive_cell : str
        The cell the positive arm publishes under.
    control_cell : str
        The cell the control arm publishes under.
    positive_rule : Callable[[Interval, Interval, Margins], bool]
        The positive arm's own rule, read off its resampled SE ratio, its exact coverage
        interval and the study's margins.
    control_ceiling : float
        What the control's SE ratio must fall below.
    gain_floor : float
        The coverage the positive arm must buy over the control.
    """
    margins = record.margins
    selected = rows.loc[rows["property"] == family]
    positive = selected.loc[selected["cell"] == positive_cell]
    control = selected.loc[selected["cell"] == control_cell]
    positive_se = se_ratio_interval(
        positive,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(record, family, positive_cell),
    )
    control_se = se_ratio_interval(
        control,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(record, family, control_cell),
    )
    gain = coverage_gain_interval(
        positive,
        control,
        replicates=margins.bootstrap_replicates,
        confidence_level=margins.confidence_level,
        seed=stream_seed(record, family, "coverage_gain"),
    )
    positive_mask = (summary["property"] == family) & (summary["cell"] == positive_cell)
    coverage = summary_interval(summary, summary.index[positive_mask.to_numpy()][0], "coverage")
    verdicts = {
        positive_cell: bool(positive_rule(positive_se, coverage, margins)),
        control_cell: bool(control_se.high <= control_ceiling),
    }
    joint = bool(all(verdicts.values()) and gain[0] >= gain_floor)
    for cell, interval in ((positive_cell, positive_se), (control_cell, control_se)):
        mask = (summary["property"] == family) & (summary["cell"] == cell)
        summary.loc[mask, "se_ratio_ci_lower"] = interval.low
        summary.loc[mask, "se_ratio_ci_upper"] = interval.high
        summary.loc[mask, "coverage_gain_ci_lower"] = gain[0]
        summary.loc[mask, "coverage_gain_ci_upper"] = gain[1]
        summary.loc[mask, "passed"] = verdicts[cell]
        summary.loc[mask, "property_passed"] = joint


def _overfit_positive_rule(se_ratio: Interval, coverage: Interval, margins: Margins) -> bool:
    """Neither understated nor outside the study's own sanity screen.

    The floor is this family's, and the ceiling is the sanity band's *upper* limit, so the
    cross-fit arm is held to the screen any other estimator answers to.  Coverage is not read:
    the arm's claim is about the standard error it reports, and the pair's coverage clause is
    published separately by :func:`_paired_cell_verdicts`.
    """
    return bool(se_ratio.low >= OVERFIT_SE_FLOOR and se_ratio.high <= margins.se_ratio_sanity[1])


def _clustered_positive_rule(se_ratio: Interval, coverage: Interval, margins: Margins) -> bool:
    """Two-sided on the SE ratio, and calibrated coverage beside it.

    The same pair of statements :func:`apply_shared_verdicts` puts on an
    ``interval_calibration`` cell, and needed for the same reason: a cluster-robust variance
    inflated by a constant keeps coverage inside its band while failing the ratio, and one
    that is right on average but wrong replication by replication does the reverse.
    """
    return bool(
        se_ratio.within(*margins.calibration_se_ratio)
        and coverage.within(*margins.calibration_coverage)
    )


def crossfit_overfitting_verdicts(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    positive_cell: str,
    control_cell: str = "in_sample_control",
) -> None:
    """Require honest cross-fitted inference and a deliberately optimistic control.

    The cross-fit arm claims a standard error that is neither understated nor outside the
    study's own sanity screen.  The control claims the opposite, that fitting the same
    flexible learner in sample understates it by a wide margin.  Cross-fitting then has to
    buy coverage the in-sample fit does not have, on the same draws.

    The rule itself lives in :func:`_paired_cell_verdicts`, which the clustered family also
    states, for the reason
    :func:`~tests.studies.evidence.properties.coverage_gain_interval` is written once: a
    statistic written twice is a statistic that can be changed once.  The study supplies only
    the name of its positive arm.

    See Also
    --------
    _paired_cell_verdicts : The shared rule this states the cross-fit arm's half of.
    """
    _paired_cell_verdicts(
        summary,
        rows,
        record,
        family="crossfit_overfitting",
        positive_cell=positive_cell,
        control_cell=control_cell,
        positive_rule=_overfit_positive_rule,
        control_ceiling=OVERFIT_SE_CONTROL_CEILING,
        gain_floor=OVERFIT_COVERAGE_GAIN,
    )


def clustered_inference_verdicts(
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    record: StudyRecord,
    *,
    positive_cell: str = "cluster_robust",
    control_cell: str = "iid_control",
) -> None:
    """Require calibrated cluster-robust inference and a deliberately IID control.

    The cluster-robust arm claims a calibrated standard error and calibrated coverage.  The
    control claims the opposite, that treating correlated rows as independent understates the
    standard error by a wide margin.  The cluster-robust variance then has to buy coverage the
    IID variance does not have, on the same draws and from the same point estimate.

    See Also
    --------
    _paired_cell_verdicts : The shared rule this states the cluster-robust arm's half of.
    """
    _paired_cell_verdicts(
        summary,
        rows,
        record,
        family="clustered_inference",
        positive_cell=positive_cell,
        control_cell=control_cell,
        positive_rule=_clustered_positive_rule,
        control_ceiling=CLUSTER_ROBUST_CONTROL_SE_CEILING,
        gain_floor=CLUSTERED_COVERAGE_GAIN,
    )


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

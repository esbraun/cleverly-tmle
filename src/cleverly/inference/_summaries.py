"""The printed summaries of the inference objects a fit returns (X89, thread T1).

A :class:`~cleverly.inference.ParameterEstimate` (also what ``ratio()`` and ``contrast()``
return), a :class:`~cleverly.inference.WaldTest`, the bootstrap records, the simultaneous
bands and the spread across cross-fitting splits each print the shared summary of
:mod:`cleverly._summary`. Every label and note text comes from its vocabulary, and the
estimates table goes through :func:`~cleverly._summary._estimates.estimates_table`, so a
spread that is not a standard error never prints under that name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .._inference_status import MINIMUM_INTERVAL_CLUSTERS
from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    count_text,
    decimals_for,
    note,
    number_text,
    pvalue_text,
    short_repr,
    vocabulary,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .bootstrap import BootstrapResult
    from .influence import BootstrapSummary, ParameterEstimate, WaldTest
    from .multiplier import SimultaneousBands
    from .results import RepeatSpread


def level_text(alpha: float) -> str:
    """The confidence level of a significance level, such as ``"95%"``.

    Parameters
    ----------
    alpha : float
        The significance level.

    Returns
    -------
    str
        The level.
    """
    return f"{(1.0 - float(alpha)) * 100:g}%"


def _listed(words: Sequence[str]) -> str:
    items = list(words)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def reference_text(reference_df: int | None) -> str:
    """The reference distribution of an interval: normal, or t with its degrees of freedom.

    Parameters
    ----------
    reference_df : int or None
        The degrees of freedom of a t reference, or ``None`` for the normal one.

    Returns
    -------
    str
        The words.
    """
    return "normal" if reference_df is None else f"t, {reference_df} df"


def _scale_text(estimate: ParameterEstimate) -> str:
    if estimate.transform is not None:
        return f"transformed ({estimate.transform.name}); interval on that scale"
    return vocabulary.SCALE_WORDS.get(estimate.scale, estimate.scale)


# ----------------------------------------------------------------- one estimate


def estimate_summary(estimate: ParameterEstimate, detail: Detail) -> Summary:
    """The summary of one estimate.

    Parameters
    ----------
    estimate : ParameterEstimate
        The estimate.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    from .._summary._estimates import estimates_table, plain_support_reason
    from .influence import bootstrap_is_diagnostic

    table, notes = estimates_table({estimate.name: estimate}, level=1.0 - estimate.alpha)
    header: list[tuple[str, str]] = [
        (vocabulary.ESTIMAND, estimate.name),
        (vocabulary.SCALE, _scale_text(estimate)),
        (vocabulary.REFERENCE, reference_text(estimate.reference_df)),
        (vocabulary.ROWS, count_text(estimate.n)),
    ]
    if estimate.n_clusters != estimate.n:
        header.append((vocabulary.CLUSTERS, count_text(estimate.n_clusters)))
    sections: list[SummarySection] = []
    if detail == "full":
        level = level_text(estimate.alpha)
        null = 1 if estimate.scale == "ratio" else 0
        lines = [f"The interval is at the {level} level; the p-value tests the null value {null}."]
        summary = estimate.bootstrap
        if summary is not None:
            decimals = decimals_for([summary.std_error])
            low, high = summary.ci
            kind = vocabulary.BOOTSTRAP_INTERVAL_WORDS[summary.interval]
            limits = f"[{number_text(low, decimals)}, {number_text(high, decimals)}]"
            spread = number_text(summary.std_error, decimals)
            if bootstrap_is_diagnostic(estimate):
                lines.append(
                    f"Bootstrap SD (diagnostic) {spread}; {kind} range (diagnostic) {limits}, "
                    f"from {count_text(summary.n_replicates)} replicates."
                )
            else:
                lines.append(
                    f"Bootstrap SE {spread}; {kind} {level} CI {limits}, from "
                    f"{count_text(summary.n_replicates)} replicates."
                )
        if estimate.support.label == vocabulary.EXPERIMENTAL and estimate.support.reason:
            lines.append(
                f"Why {estimate.name} is experimental: "
                f"{plain_support_reason(estimate.support.reason)}"
            )
        sections.append(SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines)))
    return Summary.build(
        title=vocabulary.ESTIMATE_ONE_TITLE.format(name=estimate.name),
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def estimate_repr(estimate: ParameterEstimate) -> str:
    """The one-line repr of an estimate.

    Parameters
    ----------
    estimate : ParameterEstimate
        The estimate.

    Returns
    -------
    str
        For example ``ParameterEstimate(ate=1.513, 95% CI [1.422, 1.603], n=2000)``.
    """
    facts: list[tuple[str, Any]] = []
    if estimate.supplies_inference:
        decimals = decimals_for([estimate.std_error])
        low, high = estimate.ci
        facts.append((estimate.name, number_text(estimate.psi, decimals)))
        facts.append(
            (
                "",
                f"{level_text(estimate.alpha)} CI "
                f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
            )
        )
    else:
        decimals = decimals_for([estimate.plugin_std_error])
        facts.append((estimate.name, f"{number_text(estimate.psi, decimals)} (no interval)"))
    facts.append(("n", estimate.n))
    if estimate.support.label == vocabulary.EXPERIMENTAL:
        facts.append(("", vocabulary.EXPERIMENTAL))
    return short_repr("ParameterEstimate", facts)


# ----------------------------------------------------------------- a Wald test


def wald_summary(test: WaldTest, detail: Detail) -> Summary:
    """The summary of a Wald test.

    Parameters
    ----------
    test : WaldTest
        The test.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    on_scale = _test_scale(test)
    header = [
        (vocabulary.ESTIMAND, test.name),
        (vocabulary.NULL_VALUE, f"{test.null:g}"),
        (
            vocabulary.SCALE,
            f"transformed ({test.transform}); test on that scale"
            if test.transform is not None
            else vocabulary.SCALE_WORDS.get(test.scale, test.scale),
        ),
        (vocabulary.REFERENCE, reference_text(test.reference_df)),
    ]
    table = SummaryTable(
        ("", vocabulary.STATISTIC, vocabulary.P_VALUE),
        (
            (
                Cell(test.name, test.name),
                Cell(float(test.statistic), number_text(test.statistic, 3)),
                Cell(float(test.pvalue), pvalue_text(test.pvalue)),
            ),
        ),
    )
    notes: list[Note] = []
    if on_scale is not None:
        notes.append(note("null_scale", vocabulary.NOTE_TEXTS["null_scale"].format(scale=on_scale)))
    sections: list[SummarySection] = []
    if detail == "full":
        reference = (
            "the normal distribution"
            if test.reference_df is None
            else f"a t distribution with {test.reference_df} degrees of freedom"
        )
        lines = [
            "The statistic is the estimate minus the null value, divided by the std. error, "
            f"on the scale the interval is built on; the p-value is two-sided, from {reference}.",
        ]
        if on_scale is not None:
            lines.append(f"The null value on that scale: {test.null_inference_value:.4g}.")
        sections.append(SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines)))
    return Summary.build(
        title=vocabulary.WALD_TITLE.format(name=test.name),
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def _test_scale(test: WaldTest) -> str | None:
    """The scale the test is computed on, when it is not the scale of the estimate."""
    if test.transform is not None:
        return f"transformed ({test.transform})"
    if test.scale == "ratio":
        return "log"
    return None


def wald_repr(test: WaldTest) -> str:
    """The one-line repr of a Wald test.

    Parameters
    ----------
    test : WaldTest
        The test.

    Returns
    -------
    str
        For example ``WaldTest(ate, null=0, statistic=32.8, p-value=<0.0001)``.
    """
    return short_repr(
        "WaldTest",
        [
            ("", test.name),
            ("null", f"{test.null:g}"),
            ("statistic", f"{test.statistic:.3g}"),
            ("p-value", pvalue_text(test.pvalue)),
        ],
    )


# ----------------------------------------------------------------- bootstrap records


def _replicate_table(rows: Sequence[tuple[str, Any]]) -> SummaryTable:
    """One row per name: usable replicates, their mean, SD, smallest and largest."""
    finite = {name: [float(v) for v in draws if math.isfinite(float(v))] for name, draws in rows}
    sds = [_sd(values) for values in finite.values()]
    decimals = decimals_for(sds)
    cells = []
    for (name, _), values, sd in zip(rows, finite.values(), sds, strict=True):
        mean = sum(values) / len(values) if values else float("nan")
        smallest = min(values) if values else float("nan")
        largest = max(values) if values else float("nan")
        cells.append(
            (
                Cell(name, name),
                Cell(len(values), count_text(len(values))),
                Cell(mean, number_text(mean, decimals)),
                Cell(sd, number_text(sd, decimals)),
                Cell(smallest, number_text(smallest, decimals)),
                Cell(largest, number_text(largest, decimals)),
            )
        )
    words = vocabulary.REPLICATE_COLUMNS
    return SummaryTable(
        ("", words["usable"], words["mean"], words["sd"], words["smallest"], words["largest"]),
        tuple(cells),
    )


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def replicate_rows(bootstrap: BootstrapResult) -> dict[str, Any]:
    """The columns of :meth:`BootstrapResult.to_frame`.

    Parameters
    ----------
    bootstrap : BootstrapResult
        The replicates.

    Returns
    -------
    dict of str to list
        ``estimand``, ``n_replicates``, ``mean``, ``sd``, ``min`` and ``max``.
    """
    table = _replicate_table(list(bootstrap.draws.items()))
    columns = ("estimand", "n_replicates", "mean", "sd", "min", "max")
    return {name: [row[index].value for row in table.rows] for index, name in enumerate(columns)}


def _unit(bootstrap: BootstrapResult) -> str:
    return vocabulary.RESAMPLING_WORDS.get(bootstrap.resampling, bootstrap.resampling)


def _failed_note(failed: int, requested: int) -> Note:
    return note(
        "bootstrap_failed",
        vocabulary.NOTE_TEXTS["bootstrap_failed"].format(
            failed=count_text(failed), requested=count_text(requested)
        ),
    )


def bootstrap_result_summary(bootstrap: BootstrapResult, detail: Detail) -> Summary:
    """The summary of the full-refit bootstrap replicates.

    Parameters
    ----------
    bootstrap : BootstrapResult
        The replicates.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary. It prints no interval: the replicates alone do not say whether the
        fit supplies inference.
    """
    header = [
        (vocabulary.RESAMPLING, _unit(bootstrap)),
        (vocabulary.REPLICATES, count_text(bootstrap.n_requested)),
        (vocabulary.FAILED, count_text(bootstrap.n_failed)),
    ]
    notes = [note("bootstrap_describes", vocabulary.NOTE_TEXTS["bootstrap_describes"])]
    if bootstrap.n_failed:
        notes.append(_failed_note(bootstrap.n_failed, bootstrap.n_requested))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                (
                    "Each replicate refits the whole estimator on a resample of the "
                    f"{_unit(bootstrap)}, drawn with replacement.",
                ),
            )
        )
    return Summary.build(
        title=vocabulary.BOOTSTRAP_TITLE,
        header=header,
        tables=[_replicate_table(list(bootstrap.draws.items()))],
        notes=notes,
        sections=sections,
    )


def bootstrap_summary_summary(summary: BootstrapSummary, detail: Detail) -> Summary:
    """The summary of one estimate's bootstrap interval.

    Parameters
    ----------
    summary : BootstrapSummary
        The interval record.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary. A range the fit declares a diagnostic prints under diagnostic
        labels; otherwise the fitted result's summary says whether it is a confidence
        interval.
    """
    kind = vocabulary.BOOTSTRAP_INTERVAL_WORDS[summary.interval]
    decimals = decimals_for([summary.std_error])
    low, high = summary.ci
    diagnostic = summary.diagnostic or summary.below_floor
    limits = f"{kind} range" if diagnostic else kind
    suffix = " (diagnostic)" if diagnostic else ""
    table = SummaryTable(
        (
            "",
            vocabulary.REPLICATE_COLUMNS["sd"],
            f"{limits} lower{suffix}",
            f"{limits} upper{suffix}",
        ),
        (
            (
                Cell(vocabulary.REPLICATES_ROW, vocabulary.REPLICATES_ROW),
                Cell(float(summary.std_error), number_text(summary.std_error, decimals)),
                Cell(float(low), number_text(low, decimals)),
                Cell(float(high), number_text(high, decimals)),
            ),
        ),
    )
    requested = summary.n_replicates + summary.n_failed
    header = [
        (vocabulary.INTERVAL, kind),
        (vocabulary.REPLICATES, count_text(summary.n_replicates)),
        (vocabulary.FAILED, count_text(summary.n_failed)),
    ]
    if diagnostic:
        reason = "below_floor" if summary.below_floor else "declared"
        notes = [
            note(
                "bootstrap_diagnostic",
                vocabulary.BOOTSTRAP_DIAGNOSTIC[reason].format(
                    names="this row", clusters=MINIMUM_INTERVAL_CLUSTERS
                ),
            )
        ]
    else:
        notes = [note("bootstrap_describes", vocabulary.NOTE_TEXTS["bootstrap_describes_one"])]
    if summary.n_failed:
        notes.append(_failed_note(summary.n_failed, requested))
    if summary.support.label == vocabulary.EXPERIMENTAL:
        notes.append(
            note(
                "bootstrap_experimental",
                vocabulary.NOTE_TEXTS["bootstrap_experimental"].format(names="this row"),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        lines = [
            f"One-sided limits: lower {number_text(summary.ci_one_sided_lower, decimals)}, "
            f"upper {number_text(summary.ci_one_sided_upper, decimals)}."
        ]
        if (
            summary.interval == "bca"
            and summary.z0 is not None
            and summary.acceleration is not None
        ):
            lines.append(
                f"Bias correction {summary.z0:.3g} and acceleration {summary.acceleration:.3g}."
            )
        sections.append(SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines)))
    return Summary.build(
        title=vocabulary.BOOTSTRAP_INTERVAL_TITLE,
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def bootstrap_summary_repr(summary: BootstrapSummary) -> str:
    """The one-line repr of one estimate's bootstrap interval.

    Parameters
    ----------
    summary : BootstrapSummary
        The interval record.

    Returns
    -------
    str
        For example ``BootstrapSummary(sd=0.046, percentile [1.42, 1.60], replicates=200)``.
    """
    decimals = decimals_for([summary.std_error])
    low, high = summary.ci
    facts: list[tuple[str, Any]] = [
        ("sd", number_text(summary.std_error, decimals)),
        (
            "",
            f"{vocabulary.BOOTSTRAP_INTERVAL_WORDS[summary.interval]} "
            f"[{number_text(low, decimals)}, {number_text(high, decimals)}]",
        ),
        ("replicates", summary.n_replicates),
    ]
    if summary.diagnostic or summary.below_floor:
        facts.append(("", "diagnostic"))
    if summary.support.label == vocabulary.EXPERIMENTAL:
        facts.append(("", vocabulary.EXPERIMENTAL))
    return short_repr("BootstrapSummary", facts)


# ----------------------------------------------------------------- simultaneous bands


def bands_summary(bands: SimultaneousBands, detail: Detail) -> Summary:
    """The summary of joint confidence bands.

    Parameters
    ----------
    bands : SimultaneousBands
        The bands.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    level = level_text(bands.alpha)
    widths = [
        (high - low) / (2.0 * bands.critical_value)
        for low, high in bands.bands.values()
        if bands.critical_value > 0
    ]
    decimals = decimals_for(widths)
    rows = tuple(
        (
            Cell(name, name),
            Cell(float(low), number_text(low, decimals)),
            Cell(float(high), number_text(high, decimals)),
        )
        for name, (low, high) in bands.bands.items()
    )
    table = SummaryTable(("", vocabulary.LOWER, vocabulary.UPPER), rows)
    header = [
        (vocabulary.CRITICAL_VALUE, f"{bands.critical_value:.3f}"),
        (vocabulary.ROWS, count_text(len(bands.bands))),
        (vocabulary.MULTIPLIER_DRAWS, count_text(bands.n_replicates)),
        (vocabulary.MULTIPLIER_KIND, bands.kind),
    ]
    sentences = [
        vocabulary.NOTE_TEXTS["simultaneous_bands"].format(count=len(bands.bands), level=level)
    ]
    if bands.support.label == vocabulary.EXPERIMENTAL:
        sentences.append(vocabulary.NOTE_TEXTS["bands_experimental"])
    notes = [note("simultaneous_bands", " ".join(sentences))]
    sections: list[SummarySection] = []
    if detail == "full":
        lines = [
            f"Each band is built as the row's interval is, with {bands.critical_value:.3f} "
            f"standard errors in place of the {bands.pointwise_critical_value:.3f} one "
            "interval alone uses.",
        ]
        if bands.draw_critical_values:
            lines.append(
                "The critical value is the median over "
                f"{count_text(len(bands.draw_critical_values))} splits into folds."
            )
        sections.append(SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines)))
    return Summary.build(
        title=vocabulary.BANDS_TITLE.format(level=level),
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def bands_repr(bands: SimultaneousBands) -> str:
    """The one-line repr of joint confidence bands.

    Parameters
    ----------
    bands : SimultaneousBands
        The bands.

    Returns
    -------
    str
        For example ``SimultaneousBands(rows=5, level=95%, critical value=2.31)``.
    """
    facts: list[tuple[str, Any]] = [
        ("rows", len(bands.bands)),
        ("level", level_text(bands.alpha)),
        ("critical value", f"{bands.critical_value:.3f}"),
    ]
    if bands.support.label == vocabulary.EXPERIMENTAL:
        facts.append(("", vocabulary.EXPERIMENTAL))
    return short_repr("SimultaneousBands", facts)


# ----------------------------------------------------------------- spread across splits


def spread_summary(spread: RepeatSpread, detail: Detail) -> Summary:
    """The summary of the spread of each estimate across cross-fitting splits.

    Parameters
    ----------
    spread : RepeatSpread
        The spreads.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    names = list(spread)
    every = all(spread.inferential[name] for name in names)
    none = not any(spread.inferential[name] for name in names)
    se_label = (
        vocabulary.SPREAD_LABELS["plugin_std_err"] if none else vocabulary.SPREAD_LABELS["std_err"]
    )
    ratio_label = vocabulary.SPREAD_RATIO_PLUGIN if none else vocabulary.SPREAD_RATIO
    compared = {
        name: spread.std_errors[name] if (none or spread.inferential[name]) else float("nan")
        for name in names
    }
    decimals = decimals_for([*spread.values(), *compared.values()])
    rows = []
    for name in names:
        value, against = spread[name], compared[name]
        ratio = value / against if math.isfinite(against) and against > 0 else float("nan")
        rows.append(
            (
                Cell(name, name),
                Cell(float(value), number_text(value, decimals)),
                Cell(float(against), number_text(against, decimals)),
                Cell(ratio, number_text(ratio, 2)),
                Cell(spread.scales[name], spread.scales[name]),
            )
        )
    table = SummaryTable(
        ("", vocabulary.SPREAD_DIAGNOSTIC, se_label, ratio_label, vocabulary.SCALE.lower()),
        tuple(rows),
    )
    notes = [
        note(
            "split_spread",
            vocabulary.NOTE_TEXTS["split_spread"].format(
                repeats=count_text(spread.n_repeats),
                tail=vocabulary.SPLIT_SPREAD_TAILS["withheld" if none else "inference"],
            ),
        )
    ]
    missing = [name for name in names if not math.isfinite(spread[name])]
    if missing:
        notes.append(
            note(
                "spread_missing",
                vocabulary.NOTE_TEXTS["spread_missing"].format(
                    names=_listed(missing), verb="has" if len(missing) == 1 else "have"
                ),
            )
        )
    if not every and not none:
        withheld = [name for name in names if not spread.inferential[name]]
        notes.append(
            note(
                "no_inference",
                vocabulary.NOTE_TEXTS["spread_withheld"].format(names=_listed(withheld)),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                (
                    "A ratio's spread is taken on the log scale, the scale its std. error is "
                    "on, so the two columns compare like with like.",
                ),
            )
        )
    return Summary.build(
        title=vocabulary.SPREAD_TITLE,
        header=[(vocabulary.SPLITS, count_text(spread.n_repeats))],
        tables=[table],
        notes=notes,
        sections=sections,
    )


def spread_repr(spread: Mapping[str, float], n_repeats: int) -> str:
    """The one-line repr of the spread across splits.

    Parameters
    ----------
    spread : mapping of str to float
        The spreads.
    n_repeats : int
        The number of splits.

    Returns
    -------
    str
        For example ``RepeatSpread(ate=0.0065, ey1=0.0041, +3 more, splits=3)``.
    """
    items = list(spread.items())
    facts: list[tuple[str, Any]] = [(name, f"{value:.3g}") for name, value in items[:2]]
    if len(items) > 2:
        facts.append(("", f"+{len(items) - 2} more"))
    facts.append(("splits", n_repeats))
    return short_repr("RepeatSpread", facts)

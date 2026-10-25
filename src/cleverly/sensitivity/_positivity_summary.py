"""The printed summary of the overlap check (X89 T3).

The header names the cap and how many units it moved; the tables show the treatment
probabilities, the weights each arm keeps, the rows near 0 or 1, the weights in each
estimating equation and the other probabilities the weights divide by. The reading grades
the cap only, from :meth:`PositivityReport.grade`; every concentration measure is a note
and carries no threshold. The numbers are read from the report; nothing here decides a
tier.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    note,
    short_repr,
    vocabulary,
)
from .._summary._estimates import without_roadmap
from .._summary._render import count_text, share_text
from ..data.weighting import REPORTED_DRAW, SCORE_LOAD_VERDICT_RATIO_FORMAT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .positivity import OverlapGrade, PositivityReport

__all__ = ["mechanism_label", "mechanism_words", "positivity_repr", "positivity_summary"]

#: The two top shares every weight table prints, as fractions of the weights.
_TOP_SHARES = (0.01, 0.05)


def _design_parts(name: str) -> tuple[str, str] | None:
    from .positivity import _DESIGN_ROWS

    for (base, condition), design in _DESIGN_ROWS.items():
        if design == name:
            return base, condition
    return None


def mechanism_label(name: str) -> str:
    """A probability's short table label, completing "probability of".

    Parameters
    ----------
    name : str
        The report name, such as ``"P(Delta=1|A,W)"``.

    Returns
    -------
    str
        Such as ``"outcome recorded"``. A per-arm design's row names its condition; an
        unknown name prints as written.
    """
    parts = _design_parts(name)
    base = name if parts is None else parts[0]
    words = vocabulary.MECHANISM_LABELS.get(base, base)
    if parts is None:
        return words
    template = (
        vocabulary.MECHANISM_DESIGN_STRATA
        if parts[1].endswith(",S")
        else vocabulary.MECHANISM_DESIGN
    )
    return template.format(name=words)


def mechanism_words(name: str) -> str:
    """A probability's report name in plain words.

    Parameters
    ----------
    name : str
        The report name, such as ``"P(Delta=1|A,W)"``.

    Returns
    -------
    str
        Such as ``"probability that the outcome is recorded"``. A per-arm design's row
        names its condition; an unknown name prints as written.
    """
    parts = _design_parts(name)
    base = name if parts is None else parts[0]
    words = vocabulary.MECHANISM_NAMES.get(base, base)
    if parts is None:
        return words
    template = (
        vocabulary.MECHANISM_DESIGN_STRATA
        if parts[1].endswith(",S")
        else vocabulary.MECHANISM_DESIGN
    )
    return template.format(name=words)


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _limits(lower: float, upper: float) -> str:
    return f"[{lower:.3f}, {upper:.3f}]"


def _number(value: Any, decimals: int) -> Cell:
    if value is None or not math.isfinite(float(value)):
        return Cell(None, vocabulary.MISSING)
    return Cell(float(value), f"{float(value):.{decimals}f}")


def _share(value: Any) -> Cell:
    if value is None or not math.isfinite(float(value)):
        return Cell(None, vocabulary.MISSING)
    return Cell(float(value), share_text(value))


def _quantile_table(report: PositivityReport) -> SummaryTable:
    quantiles = sorted(next(iter(report.propensity_quantiles.values())))
    rows = tuple(
        (Cell(group, group), *(_share(values[quantile]) for quantile in quantiles))
        for group, values in report.propensity_quantiles.items()
    )
    columns = ("group", *(f"{quantile:.0%}" for quantile in quantiles))
    return SummaryTable(columns, rows, title=vocabulary.QUANTILE_TABLE, kind="other")


def _arm_table(report: PositivityReport) -> SummaryTable:
    rows = tuple(
        (
            Cell(arm, str(arm)),
            Cell(ess["n"], count_text(ess["n"])),
            _number(ess["effective"], 1),
            _share(ess["ratio"]),
            _share(report.weight_share[arm]["top_1pct"]),
            _share(report.weight_share[arm]["top_5pct"]),
        )
        for arm, ess in report.effective_sample_size.items()
    )
    columns = ("arm", "rows", *vocabulary.WEIGHT_SHARE_COLUMNS)
    return SummaryTable(columns, rows, title=vocabulary.ARM_TABLE)


def _tail_table(report: PositivityReport) -> SummaryTable:
    rows = tuple(
        (
            Cell(threshold, f"{threshold:.3g}"),
            _share(mass["below"]),
            _share(mass["above"]),
        )
        for threshold, mass in sorted(report.tail_mass.items())
    )
    return SummaryTable(vocabulary.TAIL_COLUMNS, rows, title=vocabulary.TAIL_TABLE)


def _capped(count: Any, fraction: Any) -> Cell:
    if not math.isfinite(float(fraction)):
        return Cell(None, vocabulary.MISSING)
    return Cell(float(fraction), f"{count_text(count)} ({_percent(float(fraction))})")


def _equation_table(report: PositivityReport) -> SummaryTable | None:
    if not report.group_score_load:
        return None
    rows = tuple(
        (
            Cell(group, group),
            Cell(load["n_targeted"], count_text(load["n_targeted"])),
            _number(load["effective"], 1),
            _share(load["targeted_ratio"]),
            _share(load["top_1pct"]),
            _share(load["top_5pct"]),
            _number(load["max_load"], 2),
            _capped(load["clipped_count"], load["clipped_fraction"]),
        )
        for group, load in report.group_score_load.items()
    )
    columns = ("group", "rows", *vocabulary.WEIGHT_SHARE_COLUMNS, "largest", "capped")
    return SummaryTable(columns, rows, title=vocabulary.EQUATION_TABLE)


def _mechanism_table(report: PositivityReport) -> SummaryTable | None:
    if not report.mechanisms:
        return None
    rows = tuple(
        (
            Cell(name, mechanism_label(name)),
            _share(stats["min"]),
            _share(stats["q05"]),
            _share(stats["median"]),
            _share(stats["ess_ratio"]),
            _share(stats["top_1pct"]),
            _share(stats["top_5pct"]),
            _capped(stats["clipped"], stats["clipped_fraction"]),
        )
        for name, stats in report.mechanisms.items()
    )
    columns = ("probability of", "min", "5%", "median", "eff. share", "top 1%", "top 5%")
    return SummaryTable((*columns, "capped"), rows, title=vocabulary.MECHANISM_TABLE)


def _tables(report: PositivityReport) -> list[SummaryTable]:
    found = [
        _quantile_table(report),
        _arm_table(report),
        _tail_table(report),
        _equation_table(report),
        _mechanism_table(report),
    ]
    return [table for table in found if table is not None]


def _composed_note(report: PositivityReport) -> Note | None:
    from .positivity import _COMPOSED_MECHANISMS

    if not report.composed_excluded:
        return None
    groups = "; ".join(
        vocabulary.COMPOSED_GROUP.format(
            group=group,
            what=vocabulary.COMPOSED_EXCLUSION_WORDS.get(
                group, vocabulary.COMPOSED_EXCLUSION_OTHER
            ),
        )
        for group in report.composed_excluded
    )
    composed = sorted(_COMPOSED_MECHANISMS & report.mechanisms.keys())
    lead = (
        vocabulary.COMPOSED_LEAD.format(name=mechanism_words(composed[0]))
        if composed
        else vocabulary.COMPOSED_LEAD_NONE
    )
    pointer = vocabulary.COMPOSED_POINTER if report.group_score_load else ""
    return note(
        "composed_excluded",
        vocabulary.NOTE_TEXTS["composed_excluded"].format(
            lead=lead, groups=groups, pointer=pointer
        ),
    )


def _notes(report: PositivityReport) -> list[Note]:
    texts = vocabulary.NOTE_TEXTS
    notes = []
    ratios = [
        float(ess["ratio"])
        for ess in report.effective_sample_size.values()
        if math.isfinite(ess["ratio"])
    ]
    if ratios:
        small, large = _TOP_SHARES
        notes.append(
            note(
                "effective_share",
                texts["effective_share"].format(
                    share=f"{min(ratios):.0%}", top_small=f"{small:.0%}", top_large=f"{large:.0%}"
                ),
            )
        )
    group_rows = [
        (group, load)
        for group, load in report.group_score_load.items()
        if math.isfinite(load["targeted_ratio"])
    ]
    if group_rows:
        group, load = min(group_rows, key=lambda item: item[1]["targeted_ratio"])
        notes.append(
            note(
                "equation_concentration",
                texts["equation_concentration"].format(
                    group=group,
                    effective=f"{load['effective']:.1f}",
                    rows=count_text(load["n_targeted"]),
                    share=f"{load['targeted_ratio']:{SCORE_LOAD_VERDICT_RATIO_FORMAT}}",
                ),
            )
        )
    if len(report.effective_sample_size) > 2 and report.simplex_deviation > 0:
        notes.append(
            note("simplex", texts["simplex"].format(deviation=f"{report.simplex_deviation:.3g}"))
        )
    composed = _composed_note(report)
    if composed is not None:
        notes.append(composed)
    for group, reason in report.group_score_load_omissions.items():
        reason_text = without_roadmap(reason).rstrip(".")
        notes.append(
            note("group_omitted", texts["group_omitted"].format(group=group, reason=reason_text))
        )
    if report.n_repeats > 1:
        notes.append(
            note(
                "first_split",
                texts["first_split"].format(split=REPORTED_DRAW, total=report.n_repeats),
            )
        )
    return notes


def _shares() -> dict[str, str]:
    from .positivity import SERIOUS_SHARE, STRAIN_SHARE

    return {"strain": f"{STRAIN_SHARE:.0%}", "serious": f"{SERIOUS_SHARE:.0%}"}


def _treatment_result(tier: str) -> str:
    return vocabulary.OVERLAP_RESULTS[tier].format(**_shares())


def result_words(grade: OverlapGrade) -> str:
    """The ``Result`` fact of the overlap check, worded from the branch that fired.

    Parameters
    ----------
    grade : OverlapGrade
        The report's grade.

    Returns
    -------
    str
        Such as ``"at most 1% of units capped"``, or ``"another probability capped"`` on
        a branch another probability fired.
    """
    if grade.mechanism is None:
        return _treatment_result(grade.branch)
    return vocabulary.OVERLAP_RESULTS[grade.branch]


def _grade_limits(grade: OverlapGrade) -> str:
    limits = _limits(*grade.limits)
    if grade.group is None:
        return limits
    return vocabulary.GROUP_LIMITS.format(limits=limits, group=grade.group)


def _reading(report: PositivityReport, grade: OverlapGrade) -> str:
    from .positivity import _COMPOSED_MECHANISMS, MECHANISM_ESS_SHARE

    readings = vocabulary.OVERLAP_READINGS
    if grade.mechanism is None:
        return readings[grade.branch].format(
            share=_percent(grade.fraction),
            limits=_grade_limits(grade),
            curve=vocabulary.CURVE_TREATMENT,
        )
    name = grade.mechanism
    stats = report.mechanisms[name]
    curve = vocabulary.CURVE_BOTH if name in _COMPOSED_MECHANISMS else vocabulary.CURVE_MECHANISM
    if grade.branch == "mechanism_capped":
        sentence = readings["mechanism_capped"].format(
            share=_percent(stats["clipped_fraction"]),
            name=mechanism_words(name),
            limits=_mechanism_limits(report, name),
            curve=curve,
        )
    else:
        sentence = readings["mechanism_concentrated"].format(
            name=mechanism_words(name),
            low=f"{stats['min']:.3g}",
            effective=f"{stats['ess_ratio']:.0%}",
            threshold=f"{MECHANISM_ESS_SHARE:.0%}",
            curve=curve,
        )
    return sentence + readings["mechanism_treatment"].format(
        share=_percent(grade.fraction), limits=_grade_limits(grade)
    )


def _mechanism_limits(report: PositivityReport, name: str) -> str:
    from .positivity import _COMPOSED_MECHANISMS, _COMPOSED_NUISANCE_FACTORS

    floor = _limits(report.nuisance_bound, 1.0)
    if name not in _COMPOSED_MECHANISMS:
        return floor
    factors = " x ".join([_limits(*report.bounds)] + [floor] * _COMPOSED_NUISANCE_FACTORS[name])
    return vocabulary.OVERLAP_DETAILS["factor_by_factor"].format(limits=factors)


def _details(report: PositivityReport) -> list[SummarySection]:
    from .positivity import _COMPOSITE_ROWS

    lines = vocabulary.OVERLAP_DETAILS
    found = [lines["extreme"].format(value=f"{report.truncated['most_extreme']:.4g}")]
    for group, value in report.clever_covariate_max.items():
        found.append(lines["largest"].format(group=group, value=f"{value:.4g}"))
    for group, load in report.group_score_load.items():
        limits = (
            _limits(load["lower_bound"], load["upper_bound"])
            if math.isfinite(load["lower_bound"])
            else vocabulary.NOT_APPLICABLE
        )
        found.append(
            lines["equation"].format(
                group=group,
                equation=load["equation"],
                zero=count_text(load["zero_load"]),
                limits=limits,
            )
        )
    for name in report.mechanisms:
        found.append(
            lines["mechanism_limits"].format(
                name=mechanism_words(name), limits=_mechanism_limits(report, name)
            )
        )
    for name in sorted(_COMPOSITE_ROWS & report.mechanisms.keys()):
        stats = report.mechanisms[name]
        found.append(
            lines["composite"].format(
                name=mechanism_words(name),
                low=f"{stats['min']:.4g}",
                share=_percent(stats["below_g_lower_fraction"]),
                floor=f"{report.bounds[0]:.4g}",
                lower=f"{stats['composite_lower']:.4g}",
                upper=f"{report.bounds[1]:.4g}",
            )
        )
    return [SummarySection(vocabulary.SECTION_DETAILS, tuple(found))]


def _header(report: PositivityReport, grade: OverlapGrade) -> list[tuple[str, str]]:
    header = [
        (vocabulary.ROWS, count_text(report.n)),
        (vocabulary.PROBABILITY_LIMITS, _limits(*report.bounds)),
    ]
    if report.treatment_mechanism == "known":
        header.append((vocabulary.TREATMENT_MODEL, vocabulary.KNOWN_TREATMENT))
    if report.n_repeats > 1:
        header.append(
            (
                vocabulary.SPLIT_SHOWN,
                vocabulary.SPLIT_OF.format(split=REPORTED_DRAW, total=report.n_repeats),
            )
        )
    header.append(
        (
            vocabulary.UNITS_CAPPED,
            f"{count_text(report.truncated['count'])} "
            f"({_percent(float(report.truncated['fraction']))})",
        )
    )
    header.append((vocabulary.RESULT, result_words(grade)))
    return header


def positivity_summary(report: PositivityReport, detail: Detail) -> Summary:
    """The printed summary of the overlap check.

    Parameters
    ----------
    report : PositivityReport
        The report.
    detail : {"short", "full"}
        ``"full"`` adds the equation names, the largest correction-step weights and the
        limits of each other probability.

    Returns
    -------
    Summary
        The summary.
    """
    grade = report.grade()
    return Summary.build(
        title=vocabulary.OVERLAP_TITLE,
        header=_header(report, grade),
        tables=_tables(report),
        notes=_notes(report),
        reading=_reading(report, grade),
        sections=_details(report) if detail == "full" else (),
    )


def positivity_repr(report: PositivityReport) -> str:
    """A one-line repr of the overlap check.

    Parameters
    ----------
    report : PositivityReport
        The report.

    Returns
    -------
    str
        Such as ``PositivityReport(600 rows, 0 capped (0.0%), at most 1% of units
        capped)``.
    """
    return short_repr(
        "PositivityReport",
        [
            ("", f"{count_text(report.n)} rows"),
            (
                "",
                f"{count_text(report.truncated['count'])} capped "
                f"({_percent(float(report.truncated['fraction']))})",
            ),
            ("", result_words(report.grade())),
        ],
    )

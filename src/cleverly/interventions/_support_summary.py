"""The readable summaries of the overlap reports of regimes, policies and tilts (X89 T2).

Each report states how much the data support an intervention: the smallest probability the
weights divide by, the largest weight, and the effective sample size those weights leave. The
summaries print those numbers with their meaning; the full detail adds how to read them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .._summary import (
    Cell,
    Detail,
    Note,
    Summary,
    SummarySection,
    SummaryTable,
    count_text,
    note,
    short_repr,
    vocabulary,
)

if TYPE_CHECKING:
    from .incremental import IncrementalSupport
    from .policy import PolicySupport
    from .support import RegimeSupport, SupportReport


def _number(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return vocabulary.MISSING
    return f"{value:.{digits}g}"


def _load(load: Mapping[str, Any] | None) -> Cell:
    """The score load: the rows that effectively carry it, of the rows its equation reads.

    The cell holds both numbers, ``(effective, rows)``, so a reader can check each one.
    """
    if load is None:
        return Cell(None, vocabulary.MISSING)
    effective = float(load["effective"])
    targeted = int(load["n_targeted"])
    return Cell((effective, targeted), f"{effective:,.1f} of {count_text(targeted)}")


def _load_title(loads: Sequence[Mapping[str, Any] | None]) -> str:
    stored = [load for load in loads if load]
    repeats = max((int(load.get("n_repeats", 1)) for load in stored), default=1)
    if repeats <= 1:
        return ""
    split = int(stored[0].get("reported_repeat", 1))
    return vocabulary.SUPPORT_SPLIT.format(split=split, repeats=repeats)


def _how_to_read(lines: list[str]) -> SummarySection:
    return SummarySection(vocabulary.SECTION_HOW_TO_READ, tuple(lines))


# ----------------------------------------------------------------- regimes


def _regime_row(name: str, item: RegimeSupport) -> tuple[Cell, ...]:
    return (
        Cell(name, name),
        Cell(float(item.min_support_propensity), _number(item.min_support_propensity)),
        Cell(float(item.max_ratio), _number(item.max_ratio)),
        Cell(float(item.effective_sample_size), f"{item.effective_sample_size:,.1f}"),
        _load(item.score_load),
        Cell(int(item.unsupported), count_text(int(item.unsupported))),
    )


def _regime_columns() -> tuple[str, ...]:
    return (
        vocabulary.SUPPORT_REGIME,
        vocabulary.SUPPORT_MIN_G,
        vocabulary.SUPPORT_MAX_RATIO,
        vocabulary.SUPPORT_RATIO_ESS,
        vocabulary.SUPPORT_LOAD_EFFECTIVE,
        vocabulary.SUPPORT_UNSUPPORTED,
    )


def support_report_summary(report: SupportReport, detail: Detail) -> Summary:
    """Build the summary of the overlap of each regime.

    Parameters
    ----------
    report : SupportReport
        The report.
    detail : {"short", "full"}
        ``"full"`` adds how to read each column, the ratio quantiles and the tail mass.

    Returns
    -------
    Summary
        The summary.
    """
    worst = report.worst
    header: list[tuple[str, str]] = [
        (vocabulary.ROWS, count_text(report.n)),
        (vocabulary.SUPPORT_REGIMES, str(len(report.regimes))),
    ]
    if worst is not None:
        header.append((vocabulary.SUPPORT_WORST, worst.name))
    loads = [item.score_load for item in report.regimes.values()]
    table = SummaryTable(
        _regime_columns(),
        tuple(_regime_row(name, item) for name, item in report.regimes.items()),
        title=vocabulary.SUPPORT_REGIME_TABLE + _load_title(loads),
        kind="other",
    )
    notes: list[Note] = []
    if any(loads):
        notes.append(note("score_load", vocabulary.NOTE_TEXTS["score_load"]))
    unsupported = [
        f"{name} {count_text(item.unsupported)}"
        for name, item in report.regimes.items()
        if item.unsupported > 0
    ]
    if unsupported:
        notes.append(
            note(
                "unsupported_rows",
                vocabulary.NOTE_TEXTS["unsupported_rows"].format(counts=", ".join(unsupported)),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        lines = [vocabulary.SUPPORT_TWO_MECHANISMS]
        for name, item in report.regimes.items():
            quantiles = ", ".join(
                f"{q:.0%} {_number(value)}" for q, value in sorted(item.ratio_quantiles.items())
            )
            tails = ", ".join(
                f"{share:.1%} below {threshold:g}"
                for threshold, share in sorted(item.tail_mass.items())
            )
            lines.append(
                vocabulary.SUPPORT_REGIME_DETAIL.format(name=name, quantiles=quantiles, tails=tails)
            )
        sections.append(_how_to_read(lines))
    return Summary.build(
        reading=None if report.regimes else vocabulary.SUPPORT_NO_REGIMES,
        title=vocabulary.REGIME_SUPPORT_TITLE,
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def support_report_repr(report: SupportReport) -> str:
    """The one-line repr of the overlap of each regime.

    Parameters
    ----------
    report : SupportReport
        The report.

    Returns
    -------
    str
        For example ``SupportReport(regimes=2, least supported=never, min probability=0.031)``.
    """
    worst = report.worst
    facts: list[tuple[str, Any]] = [("regimes", len(report.regimes))]
    if worst is not None:
        facts.append(("least supported", worst.name))
        facts.append(("min probability", _number(worst.min_support_propensity)))
    return short_repr(type(report).__name__, facts)


def regime_support_summary(item: RegimeSupport, detail: Detail) -> Summary:
    """Build the one-table summary of one regime's overlap.

    Parameters
    ----------
    item : RegimeSupport
        The row.
    detail : {"short", "full"}
        ``"full"`` adds how to read each column.

    Returns
    -------
    Summary
        The summary.
    """
    table = SummaryTable(
        _regime_columns(),
        (_regime_row(item.name, item),),
        title=vocabulary.SUPPORT_REGIME_ONE_TABLE + _load_title([item.score_load]),
        kind="other",
    )
    notes = (
        [] if item.score_load is None else [note("score_load", vocabulary.NOTE_TEXTS["score_load"])]
    )
    return Summary.build(
        title=vocabulary.REGIME_ROW_TITLE.format(name=item.name),
        tables=[table],
        notes=notes,
        sections=[_how_to_read([vocabulary.SUPPORT_TWO_MECHANISMS])] if detail == "full" else [],
    )


def regime_support_repr(item: RegimeSupport) -> str:
    """The one-line repr of one regime's overlap.

    Parameters
    ----------
    item : RegimeSupport
        The row.

    Returns
    -------
    str
        For example ``RegimeSupport(never, min probability=0.031, effective n=345)``.
    """
    return short_repr(
        type(item).__name__,
        [
            ("", item.name),
            ("min probability", _number(item.min_support_propensity)),
            ("effective n", f"{item.effective_sample_size:,.0f}"),
        ],
    )


# ----------------------------------------------------------------- policies


def policy_support_summary(row: PolicySupport, detail: Detail) -> Summary:
    """Build the summary of one policy's overlap.

    Parameters
    ----------
    row : PolicySupport
        The row.
    detail : {"short", "full"}
        ``"full"`` adds how to read the mean ratio, the atom masses and the score load.

    Returns
    -------
    Summary
        The summary.
    """
    weight = row.min_mechanism is not None
    header: list[tuple[str, str]] = [
        (vocabulary.SUPPORT_MOVED, f"{row.moved_fraction:.1%}"),
    ]
    if row.capped_fraction is not None:
        header.append((vocabulary.SUPPORT_CAPPED, f"{row.capped_fraction:.1%}"))
    header.append(
        (
            vocabulary.SUPPORT_MIN_DENSITY,
            vocabulary.SUPPORT_NO_DENSITY if row.min_density is None else _number(row.min_density),
        )
    )
    header.append(
        (
            vocabulary.SUPPORT_UNSUPPORTED_ROWS,
            vocabulary.SUPPORT_NOT_MEASURED
            if row.unsupported is None
            else count_text(row.unsupported),
        )
    )
    if row.outside_declared_support is not None:
        header.append((vocabulary.SUPPORT_OUTSIDE, count_text(row.outside_declared_support)))
    if weight:
        header.append((vocabulary.SUPPORT_MIN_MECHANISM, _number(row.min_mechanism)))
    label = vocabulary.SUPPORT_WEIGHT if weight else vocabulary.SUPPORT_RATIO
    quantile_rows = [
        (Cell(f"{q:.0%}", vocabulary.SUPPORT_QUANTILE.format(q=f"{q:.0%}")), Cell(v, _number(v)))
        for q, v in sorted(row.ratio_quantiles.items())
    ]
    rows = (
        *quantile_rows,
        (Cell("max", vocabulary.SUPPORT_LARGEST), Cell(row.max_ratio, _number(row.max_ratio))),
        (
            Cell("ess", vocabulary.SUPPORT_ESS_ROW),
            Cell(
                row.effective_sample_size,
                f"{row.effective_sample_size:,.0f} ({row.ess_ratio:.1%} of the rows)",
            ),
        ),
        (
            Cell("mean", vocabulary.SUPPORT_MEAN_RATIO),
            Cell(row.mean_ratio, _number(row.mean_ratio)),
        ),
        *(
            (
                Cell(f"fold {index}", vocabulary.SUPPORT_FOLD_MEAN.format(fold=index)),
                Cell(value, _number(value)),
            )
            for index, value in enumerate(row.fold_mean_ratio, 1)
            if len(row.fold_mean_ratio) > 1
        ),
        (Cell("load", vocabulary.SUPPORT_LOAD_EFFECTIVE), _load(row.score_load)),
    )
    tables = [
        SummaryTable(
            ("", label),
            tuple(rows),
            title=vocabulary.SUPPORT_POLICY_TABLE + _load_title([row.score_load]),
            kind="other",
        )
    ]
    notes: list[Note] = []
    if row.score_load is not None:
        notes.append(note("score_load_rows", vocabulary.NOTE_TEXTS["score_load_rows"]))
    if row.unsupported:
        notes.append(
            note(
                "unsupported_rows",
                vocabulary.NOTE_TEXTS["unsupported_density"].format(
                    name=row.name, count=count_text(row.unsupported)
                ),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        lines = [vocabulary.SUPPORT_MEAN_RATIO_READING]
        if weight:
            lines.append(vocabulary.SUPPORT_MECHANISM_NOTE)
        if row.atom_mass:
            masses = ", ".join(f"{atom:g}: {mass:.3g}" for atom, mass in row.atom_mass.items())
            lines.append(vocabulary.SUPPORT_ATOMS.format(masses=masses))
        if row.unsupported is None:
            lines.append(vocabulary.SUPPORT_NO_ZERO_CHECK)
        sections.append(_how_to_read(lines))
    return Summary.build(
        title=vocabulary.POLICY_SUPPORT_TITLE.format(name=row.name),
        header=header,
        header_split=len(header),
        tables=tables,
        notes=notes,
        sections=sections,
    )


def policy_support_repr(row: PolicySupport) -> str:
    """The one-line repr of one policy's overlap.

    Parameters
    ----------
    row : PolicySupport
        The row.

    Returns
    -------
    str
        For example ``PolicySupport(shift, max ratio=4.2, ESS=210)``.
    """
    return short_repr(
        type(row).__name__,
        [
            ("", row.name),
            ("max ratio", _number(row.max_ratio)),
            ("ESS", f"{row.effective_sample_size:,.0f}"),
        ],
    )


# ----------------------------------------------------------------- tilts


def tilt_support_summary(row: IncrementalSupport, detail: Detail) -> Summary:
    """Build the summary of one tilt's overlap.

    Parameters
    ----------
    row : IncrementalSupport
        The row.
    detail : {"short", "full"}
        ``"full"`` adds how to read the bound.

    Returns
    -------
    Summary
        The summary.
    """
    low, high = row.guaranteed
    moved = [f"level {level} x{factor:g}" for level, factor in row.delta if factor != 1]
    header = [
        (vocabulary.SUPPORT_FACTORS, ", ".join(moved) or vocabulary.SUPPORT_NO_FACTOR),
        (vocabulary.SUPPORT_BOUND, f"[{_number(low)}, {_number(high)}]"),
    ]
    table = SummaryTable(
        (
            vocabulary.SUPPORT_LARGEST,
            vocabulary.SUPPORT_ESS_ROW,
            vocabulary.SUPPORT_SHARE_OF_ROWS,
            vocabulary.SUPPORT_LOAD_EFFECTIVE,
            vocabulary.SUPPORT_MIN_G,
        ),
        (
            (
                Cell(row.max_ratio, _number(row.max_ratio)),
                Cell(row.effective_sample_size, f"{row.effective_sample_size:,.0f}"),
                Cell(row.ess_ratio, f"{row.ess_ratio:.1%}"),
                _load(row.score_load),
                Cell(row.min_propensity, _number(row.min_propensity)),
            ),
        ),
        title=vocabulary.SUPPORT_TILT_TABLE + _load_title([row.score_load]),
        kind="other",
    )
    notes: list[Note] = []
    if row.score_load is not None:
        notes.append(note("score_load_rows", vocabulary.NOTE_TEXTS["score_load_rows"]))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(_how_to_read([vocabulary.SUPPORT_TILT_READING]))
    return Summary.build(
        title=vocabulary.TILT_SUPPORT_TITLE.format(name=row.name),
        header=header,
        header_split=len(header),
        tables=[table],
        notes=notes,
        sections=sections,
    )


def tilt_support_repr(row: IncrementalSupport) -> str:
    """The one-line repr of one tilt's overlap.

    Parameters
    ----------
    row : IncrementalSupport
        The row.

    Returns
    -------
    str
        For example ``IncrementalSupport(odds x2, max=3.0, ESS=219)``.
    """
    return short_repr(
        type(row).__name__,
        [
            ("", row.name),
            ("max", _number(row.max_ratio)),
            ("ESS", f"{row.effective_sample_size:,.0f}"),
        ],
    )

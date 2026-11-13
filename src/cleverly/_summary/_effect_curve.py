"""The printed report of an :class:`~cleverly.effect_curves.EffectCurveResult`."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import Cell, Summary, SummarySection, SummaryTable, check_detail, note, vocabulary
from ._estimates import reason_kind
from ._render import count_text, decimals_for, number_text, share_text

__all__ = ["effect_curve_summary", "score_assessment_summary"]

_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _modifier_text(modifiers: Any) -> str:
    parts = []
    for column in modifiers.columns:
        domain = modifiers.domain(column)
        levels = modifiers.levels_of(column)
        if domain is not None:
            parts.append(f"{column} on [{domain[0]:g}, {domain[1]:g}]")
        elif levels is not None:
            parts.append(f"{column} in {{{', '.join(str(level) for level in levels)}}}")
        else:
            parts.append(column)
    return "; ".join(parts)


def effect_curve_summary(result: Any, *, detail: str = "short") -> Summary:
    """Build the summary of an effect curve.

    Parameters
    ----------
    result : EffectCurveResult
        The fitted curve.
    detail : {"short", "full"}
        ``"full"`` adds the band over the declared grid, when there is one.

    Returns
    -------
    Summary
        The report.
    """
    detail = check_detail(detail)
    if result.fitted_method == "ep":
        return _ep_summary(result, detail=detail)
    curve = result.model
    scale = "difference (CATE)" if result.scale == "difference" else "log ratio (log CRR)"
    header = [
        (vocabulary.QUESTION, f"how the effect changes with {', '.join(result.modifiers.columns)}"),
        (vocabulary.MODIFIERS, _modifier_text(result.modifiers)),
        (vocabulary.CURVE_SCALE, scale),
        (vocabulary.METHOD, "projection TMLE"),
        (vocabulary.BASIS, f"{count_text(len(curve.terms))} terms, fixed before the fit"),
        (vocabulary.ROWS, count_text(int(result.in_sample.size))),
        (vocabulary.CURVE_STATUS, result.status),
    ]
    spread = np.quantile(result.in_sample, _QUANTILES)
    places = decimals_for(np.abs(spread))
    quantiles = SummaryTable(
        ("quantile", "prediction"),
        tuple(
            (Cell(q, f"{q:.2f}"), Cell(float(value), number_text(float(value), places)))
            for q, value in zip(_QUANTILES, spread, strict=True)
        ),
        title="In-sample predictions",
    )
    se = np.sqrt(np.diag(curve.covariance))
    coefficients = SummaryTable(
        ("term", "coefficient", "std err"),
        tuple(
            (
                Cell(term, term),
                Cell(float(b), number_text(float(b), places)),
                Cell(float(e), number_text(float(e), places)),
            )
            for term, b, e in zip(curve.terms, curve.beta, se, strict=True)
        ),
        title="Coefficients",
    )
    notes = [
        note(
            "projection_target",
            vocabulary.NOTE_TEXTS["projection_target"].format(target=result.target_sentence),
        )
    ]
    if result.status == "experimental":
        notes.append(
            note(
                "curve_experimental",
                vocabulary.NOTE_TEXTS["curve_experimental"].format(
                    reason=vocabulary.EXPERIMENTAL_REASONS[
                        reason_kind(result.status_reason)
                    ].format(names="the curve")
                ),
            )
        )
    if result.outside_share > 0:
        notes.append(
            note(
                "curve_outside_range",
                vocabulary.NOTE_TEXTS["curve_outside_range"].format(
                    share=share_text(result.outside_share),
                    width=f"{result.upper_bound - result.lower_bound:g}",
                ),
            )
        )
    sections: list[SummarySection] = []
    if result.band is not None:
        band = result.band
        notes.append(
            note(
                "curve_band",
                vocabulary.NOTE_TEXTS["curve_band"].format(
                    level=share_text(1.0 - band.alpha),
                    points=count_text(int(band.estimate.size)),
                    critical=f"{band.critical_value:.3f}",
                ),
            )
        )
        if detail == "full":
            names = list(band.grid)
            sections.append(
                SummarySection(
                    "Band over the declared grid",
                    tuple(
                        ", ".join(f"{name}={band.grid[name][i]}" for name in names)
                        + f": {band.estimate[i]:.4g} [{band.lower[i]:.4g}, {band.upper[i]:.4g}]"
                        for i in range(band.estimate.size)
                    ),
                )
            )
    return Summary.build(
        title=vocabulary.EFFECT_CURVE_TITLE,
        header=header,
        tables=[coefficients, quantiles],
        notes=notes,
        sections=sections,
    )


_SIEVE_MODES = {
    "fixed": "fixed size",
    "schedule": "declared schedule",
    "cv": "chosen by cross-validation",
}


def _ep_reason_kind(result: Any) -> str:
    """The kind of an EP curve's gap: ``"theory"`` outside the source's theorems."""
    fit = result.fit
    declared_learner = fit is not None and fit.estimator.final_learner is not None
    if result.sieve.mode == "cv" or declared_learner:
        return "theory"
    return reason_kind(result.status_reason)


def _ep_summary(result: Any, *, detail: str) -> Summary:
    """The report of an EP curve: no coefficients with errors, and no interval."""
    sieve = result.sieve
    scale = "difference (CATE)" if result.scale == "difference" else "log ratio (log CRR)"
    header = [
        (vocabulary.QUESTION, f"how the effect changes with {', '.join(result.modifiers.columns)}"),
        (vocabulary.MODIFIERS, _modifier_text(result.modifiers)),
        (vocabulary.CURVE_SCALE, scale),
        (vocabulary.METHOD, "EP learner"),
        (
            vocabulary.SIEVE,
            f"size {sieve.chosen} per component, {_SIEVE_MODES[sieve.mode]}",
        ),
        (vocabulary.ROWS, count_text(int(result.in_sample.size))),
        (vocabulary.CURVE_STATUS, result.status),
    ]
    spread = np.quantile(result.in_sample, _QUANTILES)
    places = decimals_for(np.abs(spread))
    tables = [
        SummaryTable(
            ("quantile", "prediction"),
            tuple(
                (Cell(q, f"{q:.2f}"), Cell(float(value), number_text(float(value), places)))
                for q, value in zip(_QUANTILES, spread, strict=True)
            ),
            title="In-sample predictions",
        )
    ]
    if sieve.cv_risks and detail == "full":
        risk_places = decimals_for(np.abs(np.asarray(sieve.cv_risks)))
        tables.append(
            SummaryTable(
                ("size", "cross-validated risk"),
                tuple(
                    (
                        Cell(size, str(size)),
                        Cell(float(risk), number_text(float(risk), risk_places)),
                    )
                    for size, risk in zip(sieve.sizes, sieve.cv_risks, strict=True)
                ),
                title="Sieve sizes (lower risk is better)",
            )
        )
    notes = [
        note("ep_target", vocabulary.NOTE_TEXTS["ep_target"].format(target=result.target_sentence)),
        note("ep_no_interval", vocabulary.NOTE_TEXTS["ep_no_interval"]),
    ]
    if result.status == "experimental":
        notes.append(
            note(
                "ep_experimental",
                vocabulary.NOTE_TEXTS["ep_experimental"].format(
                    reason=vocabulary.EXPERIMENTAL_REASONS[_ep_reason_kind(result)].format(
                        names="the curve"
                    )
                ),
            )
        )
    if result.clipped_share > 0:
        notes.append(
            note(
                "ep_clipped",
                vocabulary.NOTE_TEXTS["ep_clipped"].format(
                    share=share_text(result.clipped_share),
                    width=f"{result.upper_bound - result.lower_bound:g}",
                ),
            )
        )
    if result.outside_share > 0:
        notes.append(
            note(
                "curve_outside_range",
                vocabulary.NOTE_TEXTS["curve_outside_range"].format(
                    share=share_text(result.outside_share),
                    width=f"{result.upper_bound - result.lower_bound:g}",
                ),
            )
        )
    from ..estimators._point_summary import _warning_notes

    notes.extend(_warning_notes(result.fit_warnings))
    return Summary.build(
        title=vocabulary.EFFECT_CURVE_TITLE, header=header, tables=tables, notes=notes
    )


def score_assessment_summary(result: Any, *, detail: str = "short") -> Summary:
    """Build the summary of the check of an external score, or of the comparison of two.

    Parameters
    ----------
    result : ScoreAssessment
        The check.
    detail : {"short", "full"}
        ``"full"`` adds the p-value of each estimate and the fluctuation's span record.

    Returns
    -------
    Summary
        The report.
    """
    detail = check_detail(detail)
    scale = "difference (CATE)" if result.scale == "difference" else "log ratio (log CRR)"
    rows_of = result.rows()
    estimates = [estimate for _, _, estimate in rows_of]
    level = share_text(1.0 - estimates[0].alpha)
    question = f"how well {result.column} predicts the conditional effect"
    if result.compare is not None:
        question += f", compared with {result.compare}"
    header = [
        (vocabulary.QUESTION, question),
        (vocabulary.CURVE_SCALE, scale),
        (vocabulary.METHOD, vocabulary.SCORE_ASSESSMENT_METHOD),
        (vocabulary.CURVE_STATUS, result.status),
    ]
    shown = [e for e in estimates if e.supplies_inference]
    values = [abs(float(e.psi)) for e in estimates] + [
        abs(float(v)) for e in shown for v in (e.std_error, *e.ci)
    ]
    places = decimals_for(np.asarray(values))
    rows = []
    for score, name, estimate in rows_of:
        # The score column names the pair, so the paired row prints its parameter bare.
        parameter = "risk_difference" if name == result.paired_name else name
        cells = [
            Cell(score, score),
            Cell(parameter, parameter),
            Cell(float(estimate.psi), number_text(float(estimate.psi), places)),
        ]
        if estimate.supplies_inference:
            low, high = (float(v) for v in estimate.ci)
            error = float(estimate.std_error)
            cells += [
                Cell(error, number_text(error, places)),
                Cell(low, number_text(low, places)),
                Cell(high, number_text(high, places)),
            ]
        else:
            cells += [Cell(None, vocabulary.NO_INFERENCE_MARK)] * 3
        label = estimate.support.label
        cells.append(Cell(label, label))
        rows.append(tuple(cells))
    columns = f"each of {result.column} and {result.compare}" if result.compare else result.column
    notes = [
        note(
            "score_external",
            vocabulary.NOTE_TEXTS["score_external"].format(column=columns),
        )
    ]
    if result.compare is not None:
        notes.append(
            note(
                "score_paired",
                vocabulary.NOTE_TEXTS["score_paired"].format(
                    column=result.column, compare=result.compare
                ),
            )
        )
    if not all(e.supplies_inference for e in estimates):
        notes.append(
            note(
                "score_withheld",
                vocabulary.NOTE_TEXTS["score_withheld"].format(mark=vocabulary.NO_INFERENCE_MARK),
            )
        )
    if result.status == "experimental":
        reason = " ".join(
            part[:1].upper() + part[1:] + "." for part in result.status_reason.split("; ") if part
        )
        notes.append(
            note(
                "score_experimental",
                vocabulary.NOTE_TEXTS["score_experimental"].format(reason=reason),
            )
        )
    sections = []
    if detail == "full":
        sections.append(
            SummarySection(
                "P-values against 0",
                tuple(
                    f"{score}, {'risk_difference' if name == result.paired_name else name}: "
                    + (f"{float(e.pvalue):.3g}" if e.supplies_inference else vocabulary.MISSING)
                    for score, name, e in rows_of
                ),
            )
        )
        span = result.span
        sections.append(
            SummarySection(
                vocabulary.SCORE_ASSESSMENT_SPAN_TITLE,
                (
                    vocabulary.SCORE_ASSESSMENT_SPAN_LINE.format(
                        rank=count_text(int(span["rank"])),
                        columns=count_text(int(span["columns"])),
                        norm=f"{float(span['score_norm']):.2g}",
                    ),
                ),
            )
        )
    return Summary.build(
        title=vocabulary.SCORE_ASSESSMENT_TITLE,
        header=header,
        tables=[
            SummaryTable(
                vocabulary.SCORE_ASSESSMENT_COLUMNS,
                tuple(rows),
                title=f"Estimates with {level} intervals",
            )
        ],
        notes=notes,
        sections=sections,
    )

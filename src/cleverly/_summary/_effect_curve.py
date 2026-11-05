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


def score_assessment_summary(result: Any, *, detail: str = "short") -> Summary:
    """Build the summary of the check of an external score.

    Parameters
    ----------
    result : ScoreAssessment
        The check.
    detail : {"short", "full"}
        ``"full"`` adds the p-value of each estimate.

    Returns
    -------
    Summary
        The report.
    """
    detail = check_detail(detail)
    scale = "difference (CATE)" if result.scale == "difference" else "log ratio (log CRR)"
    estimates = list(result.estimates.items())
    level = share_text(1.0 - estimates[0][1].alpha)
    header = [
        (vocabulary.QUESTION, f"how well {result.column} predicts the conditional effect"),
        (vocabulary.CURVE_SCALE, scale),
        (vocabulary.METHOD, "one-step estimates from an ATE fit's models"),
    ]
    values = [abs(float(v)) for _, e in estimates for v in (e.psi, e.std_error, *e.ci)]
    places = decimals_for(np.asarray(values))
    columns = ("parameter", "estimate", "std err", "lower", "upper")
    rows = []
    for name, estimate in estimates:
        low, high = (float(v) for v in estimate.ci)
        row = [
            Cell(name, name),
            Cell(float(estimate.psi), number_text(float(estimate.psi), places)),
            Cell(float(estimate.std_error), number_text(float(estimate.std_error), places)),
            Cell(low, number_text(low, places)),
            Cell(high, number_text(high, places)),
        ]
        rows.append(tuple(row))
    notes = [
        note(
            "score_external",
            vocabulary.NOTE_TEXTS["score_external"].format(column=result.column),
        )
    ]
    return Summary.build(
        title=vocabulary.SCORE_ASSESSMENT_TITLE,
        header=header,
        tables=[SummaryTable(columns, tuple(rows), title=f"Estimates with {level} intervals")],
        notes=notes,
        sections=(
            [
                SummarySection(
                    "P-values against 0",
                    tuple(f"{name}: {float(e.pvalue):.3g}" for name, e in estimates),
                )
            ]
            if detail == "full"
            else []
        ),
    )

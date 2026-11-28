"""The printed report of a :class:`~cleverly.survival.ContinuousLongitudinalResult`."""

from __future__ import annotations

from typing import Any

from . import Summary, SummarySection, check_detail, note, vocabulary
from ._continuous_survival import _experimental_lines, _model, _times
from ._estimates import estimates_table
from ._render import count_text, share_text

__all__ = ["continuous_longitudinal_summary"]


def continuous_longitudinal_summary(result: Any, *, detail: str = "short") -> Summary:
    """Build the summary of a continuous-time longitudinal fit.

    Parameters
    ----------
    result : ContinuousLongitudinalResult
        The fit.
    detail : {"short", "full"}
        ``"full"`` adds the assumptions and how the node walks ended.

    Returns
    -------
    Summary
        The report.
    """
    detail = check_detail(detail)
    record = result.record
    reference, *others = result.arms
    causes = f"each cause ({', '.join(result.causes)})" if len(result.causes) > 1 else "the event"
    method = vocabulary.CONTINUOUS_LONGITUDINAL_METHOD
    if record.cross_fitted:
        method += vocabulary.CONTINUOUS_FOLDS.format(k=result.n_folds)
    treatment = (
        vocabulary.USER_TREATMENT_MODEL
        if record.propensity is None
        else record.propensity.names[record.propensity.chosen]
    )
    caps = dict(zip(record.regimes, record.caps, strict=True))
    steps = sum(int(v) for row in record.steps for v in row)
    header = [
        (
            vocabulary.QUESTION,
            vocabulary.CONTINUOUS_LONGITUDINAL_QUESTION.format(causes=causes),
        ),
        (vocabulary.CONTINUOUS_TIMES, _times(result.horizons)),
        (
            vocabulary.CONTINUOUS_REGIMES,
            vocabulary.CONTINUOUS_REGIMES_TEXT.format(
                others=", ".join(others), reference=reference
            ),
        ),
        (vocabulary.UNITS, count_text(result.n)),
        (vocabulary.METHOD, method),
        (vocabulary.EVENT_MODELS, "; ".join(_model(h) for h in record.hazards)),
        *([(vocabulary.VISIT_MODEL, _model(record.visit))] if record.visit is not None else []),
        (vocabulary.CENSORING_MODEL, _model(record.censoring)),
        (vocabulary.TREATMENT_MODEL, treatment),
        (
            vocabulary.VISIT_CAP,
            "; ".join(
                vocabulary.VISIT_CAP_TEXT.format(regime=arm, cap=caps[arm]) for arm in result.arms
            ),
        ),
        (
            vocabulary.TARGETING_STEPS,
            vocabulary.TARGETING_READING[record.converged].format(steps=steps),
        ),
    ]
    level = 1.0 - next(iter(result.estimates.values())).alpha
    table, notes = estimates_table(result.estimates, level=level)
    worst = max(record.standardized_score)
    if not record.converged:
        notes.append(
            note(
                "continuous_longitudinal_not_converged",
                vocabulary.NOTE_TEXTS["continuous_longitudinal_not_converged"].format(
                    worst=f"{worst / record.cutoff:.3g}"
                ),
            )
        )
    sections = []
    if detail == "full":
        lines = vocabulary.CONTINUOUS_LONGITUDINAL_TARGETING_LINES
        sections = [
            SummarySection(
                vocabulary.SECTION_ASSUMPTIONS,
                (
                    vocabulary.ASSUMPTIONS_LEAD_GENERIC,
                    *vocabulary.CONTINUOUS_LONGITUDINAL_ASSUMPTIONS,
                ),
            ),
            SummarySection(
                vocabulary.SECTION_TARGETING,
                (
                    lines[0].format(grid=count_text(record.grid_size)),
                    lines[1].format(worst=f"{worst:.3g}", cutoff=f"{record.cutoff:.3g}"),
                    lines[2].format(bounded=share_text(record.bounded_share)),
                    lines[3].format(
                        death=share_text(record.jump_bound_share),
                        visit=share_text(record.visit_bound_share),
                    ),
                    lines[4].format(window=record.history_restriction),
                    *([vocabulary.CONTINUOUS_FIRST_FOLD] if record.cross_fitted else []),
                    *_experimental_lines(result.estimates),
                ),
            ),
        ]
    return Summary.build(
        title=vocabulary.CONTINUOUS_LONGITUDINAL_TITLE,
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )

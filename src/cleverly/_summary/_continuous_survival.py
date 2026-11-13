"""The printed report of a :class:`~cleverly.survival.ContinuousSurvivalResult`."""

from __future__ import annotations

from typing import Any

from ..survival.engine import OFF_GRID_TIMES
from . import Summary, SummarySection, check_detail, note, vocabulary
from ._estimates import estimates_table, plain_support_reason
from ._render import count_text, share_text

__all__ = ["continuous_survival_summary"]


def _times(horizons: tuple[float, ...]) -> str:
    return ", ".join(f"{t:g}" for t in horizons)


def _model(selection: Any) -> str:
    """The selected hazard model in words."""
    return str(getattr(selection.model, "label", selection.chosen_name))


def _experimental_lines(estimates: Any) -> list[str]:
    reasons: dict[str, list[str]] = {}
    for name, estimate in estimates.items():
        if estimate.support.label == vocabulary.EXPERIMENTAL and estimate.support.reason:
            reasons.setdefault(plain_support_reason(estimate.support.reason), []).append(name)
    lines = []
    for reason, names in reasons.items():
        every = len(names) == len(estimates)
        who = vocabulary.EVERY_ROW if every else ", ".join(names)
        verb = "is" if every or len(names) == 1 else "are"
        lines.append(f"Why {who} {verb} experimental: {reason}")
    return lines


def continuous_survival_summary(result: Any, *, detail: str = "short") -> Summary:
    """Build the summary of a continuous-time survival fit.

    Parameters
    ----------
    result : ContinuousSurvivalResult
        The fit.
    detail : {"short", "full"}
        ``"full"`` adds the assumptions and how the update ended.

    Returns
    -------
    Summary
        The report.
    """
    detail = check_detail(detail)
    record = result.record
    reference, treated = result.arms
    causes = f"each cause ({', '.join(result.causes)})" if len(result.causes) > 1 else "the event"
    method = vocabulary.CONTINUOUS_METHOD
    if record.cross_fitted:
        method += vocabulary.CONTINUOUS_FOLDS.format(k=result.n_folds)
    treatment = (
        vocabulary.USER_TREATMENT_MODEL
        if record.propensity is None
        else record.propensity.names[record.propensity.chosen]
    )
    header = [
        (
            vocabulary.QUESTION,
            vocabulary.CONTINUOUS_QUESTION.format(causes=causes),
        ),
        (vocabulary.CONTINUOUS_TIMES, _times(result.horizons)),
        (
            vocabulary.CONTINUOUS_ARMS,
            vocabulary.CONTINUOUS_ARMS_TEXT.format(treated=treated, reference=reference),
        ),
        (vocabulary.UNITS, count_text(result.n)),
        (vocabulary.METHOD, method),
        (vocabulary.EVENT_MODELS, "; ".join(_model(h) for h in record.hazards)),
        (vocabulary.CENSORING_MODEL, _model(record.censoring)),
        (vocabulary.TREATMENT_MODEL, treatment),
        (
            vocabulary.TARGETING_STEPS,
            vocabulary.TARGETING_READING[record.converged].format(steps=record.steps),
        ),
    ]
    level = 1.0 - next(iter(result.estimates.values())).alpha
    table, notes = estimates_table(result.estimates, level=level)
    worst = max(record.standardized_score)
    if not record.converged:
        notes.append(
            note(
                "continuous_not_converged",
                vocabulary.NOTE_TEXTS["continuous_not_converged"].format(
                    steps=record.steps, worst=f"{worst / record.cutoff:.3g}"
                ),
            )
        )
    sections = []
    if detail == "full":
        lines = vocabulary.CONTINUOUS_TARGETING_LINES
        sections = [
            SummarySection(
                vocabulary.SECTION_ASSUMPTIONS,
                (vocabulary.ASSUMPTIONS_LEAD_GENERIC, *vocabulary.CONTINUOUS_ASSUMPTIONS),
            ),
            SummarySection(
                vocabulary.SECTION_TARGETING,
                (
                    lines[0].format(grid=count_text(record.grid_size)),
                    lines[1].format(worst=f"{worst:.3g}", cutoff=f"{record.cutoff:.3g}"),
                    lines[2].format(halvings=count_text(record.halvings)),
                    lines[3].format(bounded=share_text(record.bounded_share)),
                    lines[4].format(last=share_text(record.last_time_floor_share)),
                    lines[5].format(frozen=share_text(record.frozen_share)),
                    lines[6].format(
                        draws=OFF_GRID_TIMES, off_grid=share_text(record.off_grid_share)
                    ),
                    *([vocabulary.CONTINUOUS_FIRST_FOLD] if record.cross_fitted else []),
                    *_experimental_lines(result.estimates),
                ),
            ),
        ]
    return Summary.build(
        title=vocabulary.CONTINUOUS_SURVIVAL_TITLE,
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )

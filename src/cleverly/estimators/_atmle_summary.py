"""The readable summary of an adaptive TMLE fit of a trial with external data (X90 T2).

``AdaptiveTMLEResult.summary()`` builds here, on the shared :mod:`cleverly._summary` layer.
Every fact is read from the result: the counts from its data and the engine's diagnostics,
the numbers from its estimates, and the working-model sizes from its selections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._summary import (
    Detail,
    Note,
    Summary,
    SummarySection,
    count_text,
    note,
    share_text,
    vocabulary,
)
from .._summary._estimates import estimates_table, plain_support_reason
from .._summary._learners import learner_name
from ._point_summary import _warning_notes

if TYPE_CHECKING:
    from .atmle_result import AdaptiveTMLEResult


def _header(result: AdaptiveTMLEResult) -> tuple[list[tuple[str, str]], int]:
    data = result.data
    support = result.support_report()
    outcome, treatment = str(data.outcome_name), str(data.treatment_name)
    left: list[tuple[str, str]] = [
        (vocabulary.QUESTION, vocabulary.ATMLE_QUESTION.format(A=treatment, Y=outcome)),
        (vocabulary.ATMLE_POPULATION, vocabulary.ATMLE_POPULATIONS[result.fit.population]),
        (vocabulary.ADJUSTED_FOR, ", ".join(data.covariate_names)),
        (vocabulary.METHOD, vocabulary.ATMLE_FORMS[result.fit.form]),
    ]
    arms = (
        vocabulary.EXTERNAL_CONTROLS_ONLY
        if support.controls_only
        else vocabulary.EXTERNAL_BOTH_ARMS.format(k=count_text(support.n_external_treated))
    )
    folds = result.fit.folds.n_folds
    right: list[tuple[str, str]] = [
        (vocabulary.ROWS, count_text(result.n)),
        (vocabulary.TRIAL_ROWS, count_text(support.n_trial)),
        (vocabulary.EXTERNAL_ROWS, count_text(support.n_external)),
        (vocabulary.EXTERNAL_ARMS, arms),
        (
            vocabulary.OUT_OF_FOLD,
            "no" if folds <= 1 else vocabulary.ATMLE_FOLDS_WORD.format(k=folds),
        ),
        (vocabulary.OUTCOME_MODEL, learner_name(result.learners.get("outcome"))),
        (vocabulary.TREATMENT_MODEL, learner_name(result.learners.get("treatment"))),
        (vocabulary.SOURCE_MODEL, learner_name(result.learners.get("trial membership"))),
    ]
    return [*left, *right], len(left)


def _notes(result: AdaptiveTMLEResult) -> list[Note]:
    support = result.support_report()
    _, _, median = support.external_source_quantiles
    notes = [
        note(
            "atmle_overlap",
            vocabulary.NOTE_TEXTS["atmle_overlap"].format(
                low=f"{support.external_source_min:.3g}",
                median=f"{median:.3g}",
                share=share_text(support.external_source_below_floor),
                floor=f"{support.floor:g}",
            ),
        )
    ]
    models = result.working_models
    folds = "the fold" if len(models) == 1 else f"{len(models)} folds"
    notes.append(
        note(
            "atmle_working_model",
            vocabulary.NOTE_TEXTS["atmle_working_model"].format(
                bias=max(len(model.bias_columns) for model in models),
                effect=max(len(model.effect_columns) for model in models),
                folds=folds,
            ),
        )
    )
    notes.extend(_warning_notes(result.fit_warnings))
    return notes


def _settings(result: AdaptiveTMLEResult) -> SummarySection:
    low, high = result.g_bounds
    corrections = result.corrections_report()
    lines = [
        vocabulary.ATMLE_SETTINGS_BOUNDS.format(low=f"{low:.3g}", high=f"{high:.3g}"),
        vocabulary.ATMLE_SETTINGS_PASSES.format(
            passes=corrections.passes, met="met" if corrections.converged else "did not meet"
        ),
    ]
    reasons = dict.fromkeys(
        estimate.support.reason
        for estimate in result.estimates.values()
        if estimate.support.label == vocabulary.EXPERIMENTAL and estimate.support.reason
    )
    for reason in reasons:
        lines.append(f"Why the rows are experimental: {plain_support_reason(reason)}")
    return SummarySection(vocabulary.SECTION_SETTINGS, tuple(lines))


def _assumptions() -> SummarySection:
    lines = [vocabulary.ASSUMPTIONS_LEAD_GENERIC]
    lines.extend(f"- {item}" for item in vocabulary.ATMLE_ASSUMPTIONS)
    return SummarySection(vocabulary.SECTION_ASSUMPTIONS, tuple(lines))


def atmle_summary(result: AdaptiveTMLEResult, detail: Detail) -> Summary:
    """Build the summary of an adaptive TMLE fit.

    Parameters
    ----------
    result : AdaptiveTMLEResult
        The fitted result.
    detail : {"short", "full"}
        ``"full"`` adds the Assumptions and Settings sections.

    Returns
    -------
    Summary
        The summary.
    """
    header, split = _header(result)
    table, notes = estimates_table(result.estimates, level=1.0 - result.method.inference.alpha)
    notes.extend(_notes(result))
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(_assumptions())
        sections.append(_settings(result))
    return Summary.build(
        title=vocabulary.ATMLE_TITLE,
        header=header,
        header_split=split,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def atmle_repr(result: AdaptiveTMLEResult) -> str:
    """The one-line repr of an adaptive TMLE fit.

    Parameters
    ----------
    result : AdaptiveTMLEResult
        The fitted result.

    Returns
    -------
    str
        The repr.
    """
    from ..longitudinal._summary import estimates_repr

    return estimates_repr(
        type(result).__name__,
        result.estimates,
        n=result.n,
        method="adaptive TMLE",
        alpha=result.method.inference.alpha,
    )

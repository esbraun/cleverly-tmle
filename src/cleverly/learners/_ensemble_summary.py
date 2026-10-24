"""The printed summary of a super learner's diagnostics (X89, thread T1)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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
    short_repr,
    vocabulary,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .super_learner import SuperLearnerDiagnostics


def _loss(diagnostics: SuperLearnerDiagnostics) -> str:
    return vocabulary.LOSS_WORDS.get(diagnostics.loss, diagnostics.loss.replace("_", " "))


def diagnostics_summary(diagnostics: SuperLearnerDiagnostics, detail: Detail) -> Summary:
    """The summary of one super learner fit.

    Parameters
    ----------
    diagnostics : SuperLearnerDiagnostics
        The diagnostics.
    detail : {"short", "full"}
        ``"full"`` adds a Settings section.

    Returns
    -------
    Summary
        The summary.
    """
    risks = [float(value) for value in diagnostics.cv_risk]
    finite = [value for value in risks if math.isfinite(value)]
    spread = [abs(a - b) for a in finite for b in finite if a != b] or finite or [1.0]
    decimals = max(3, decimals_for([min(spread)]))
    rows = tuple(
        (
            Cell(name, name),
            Cell(risk, number_text(risk, decimals)),
            Cell(float(weight), number_text(float(weight), 3)),
        )
        for name, risk, weight in zip(diagnostics.names, risks, diagnostics.weights, strict=True)
    )
    table = SummaryTable(("candidate", vocabulary.CV_RISK, vocabulary.WEIGHT), rows, kind="other")
    header = [
        (vocabulary.LOSS, _loss(diagnostics)),
        (vocabulary.CANDIDATES, count_text(len(diagnostics.names))),
        (vocabulary.BEST, diagnostics.best),
        (
            vocabulary.ENSEMBLE_RISK,
            f"{number_text(diagnostics.ensemble_cv_risk, decimals)} (on the folds that chose "
            "the weights)",
        ),
    ]
    notes: list[Note] = []
    if diagnostics.failed:
        notes.append(
            note(
                "learners_failed",
                vocabulary.NOTE_TEXTS["learners_failed"].format(
                    names=", ".join(diagnostics.failed)
                ),
            )
        )
    sections: list[SummarySection] = []
    if detail == "full":
        sections.append(
            SummarySection(
                vocabulary.SECTION_SETTINGS,
                (
                    "Each candidate's risk is scored on held-out folds. The ensemble's "
                    "weights are fitted to those same held-out predictions, so its risk is "
                    "optimistic beside the candidates' risks.",
                ),
            )
        )
    return Summary.build(
        title=vocabulary.ENSEMBLE_TITLE,
        header=header,
        tables=[table],
        notes=notes,
        sections=sections,
    )


def diagnostics_repr(diagnostics: SuperLearnerDiagnostics) -> str:
    """The one-line repr of one super learner fit.

    Parameters
    ----------
    diagnostics : SuperLearnerDiagnostics
        The diagnostics.

    Returns
    -------
    str
        For example ``SuperLearnerDiagnostics(best=LinearRegression, candidates=3)``.
    """
    return short_repr(
        "SuperLearnerDiagnostics",
        [("best", diagnostics.best), ("candidates", len(diagnostics.names))],
    )

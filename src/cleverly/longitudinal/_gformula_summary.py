"""The readable summary of a parametric g-formula simulation report (X89 T2)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from .._summary import (
    Cell,
    Note,
    Summary,
    SummaryTable,
    count_text,
    note,
    short_repr,
    vocabulary,
)

if TYPE_CHECKING:
    from .gformula import GFormulaReport


def _model_name(key: str, label: str) -> str:
    """The modelled column and its time point, from a model's key and message label."""
    quoted = re.search(r"'([^']*)'", label)
    name = quoted[1] if quoted else ""
    parts = key.split(":")
    if parts[0] == "covariate" and len(parts) == 3:
        return f"{parts[2]} at time {parts[1]}"
    if parts[0] in ("hazard", "treatment") and len(parts) == 2:
        what = "event" if parts[0] == "hazard" else (name or "treatment")
        return f"{what} at time {parts[1]}"
    return name or key


def _model_rows(report: GFormulaReport) -> tuple[tuple[Cell, ...], ...]:
    rows = []
    for model in report.models:
        name = _model_name(model.key, model.label)
        role = vocabulary.GFORMULA_ROLES.get(model.role, model.role)
        response = vocabulary.GFORMULA_RESPONSES.get(model.response, model.response)
        learner = (
            vocabulary.GFORMULA_CONSTANT
            if model.learner == "constant"
            else vocabulary.GFORMULA_LEARNERS.get(model.learner, model.learner)
        )
        read = vocabulary.YES if model.stacked else vocabulary.NO
        rows.append(
            (
                Cell(name, name),
                Cell(role, role),
                Cell(response, response),
                Cell(learner, learner),
                Cell(model.n_rows, count_text(model.n_rows)),
                Cell(model.stacked, read),
            )
        )
    return tuple(rows)


def _share_rows(report: GFormulaReport) -> tuple[tuple[Cell, ...], ...]:
    return tuple(
        (Cell(name, name), Cell(float(share), f"{share:.1%}"))
        for name, share in report.monte_carlo_share.items()
        if np.isfinite(share)
    )


def _support_rows(report: GFormulaReport) -> tuple[tuple[Cell, ...], ...]:
    rows = []
    for row in report.support:
        dose = vocabulary.GFORMULA_RULE_DOSE if row.dose is None else f"{row.dose:g}"
        rows.append(
            (
                Cell(row.plan, row.plan),
                Cell(row.name, row.name),
                Cell(row.dose, dose),
                Cell(float(row.off_support), f"{row.off_support:.1%}"),
                Cell(float(row.window), f"{row.window:.1%}"),
            )
        )
    return tuple(rows)


def gformula_report_summary(report: GFormulaReport) -> Summary:
    """Build the summary of a parametric g-formula simulation report.

    Parameters
    ----------
    report : GFormulaReport
        The fit's simulation record.

    Returns
    -------
    Summary
        The models, the simulation's share of the variance and the support of set doses.
    """
    from .estimator import MONTE_CARLO_WARN_SHARE
    from .gformula import SUPPORT_Z

    header = [
        (vocabulary.GFORMULA_DRAWS, count_text(report.n_draws)),
        (vocabulary.GFORMULA_DERIVATIVE_DRAWS, count_text(report.n_derivative_draws)),
        (vocabulary.GFORMULA_MODELS, str(len(report.models))),
        (vocabulary.GFORMULA_PARAMETRIC, vocabulary.YES if report.parametric else vocabulary.NO),
    ]
    tables = [
        SummaryTable(
            (
                vocabulary.GFORMULA_MODEL,
                vocabulary.GFORMULA_ROLE,
                vocabulary.GFORMULA_RESPONSE,
                vocabulary.GFORMULA_LEARNER,
                vocabulary.ROWS.lower(),
                vocabulary.GFORMULA_READ,
            ),
            _model_rows(report),
            title=vocabulary.GFORMULA_MODELS_TITLE,
            kind="other",
        )
    ]
    shares = _share_rows(report)
    if shares:
        tables.append(
            SummaryTable(
                ("", vocabulary.GFORMULA_SHARE),
                shares,
                title=vocabulary.GFORMULA_SHARE_TITLE,
                kind="other",
            )
        )
    support = _support_rows(report)
    if support:
        tables.append(
            SummaryTable(
                (
                    vocabulary.PLAN,
                    vocabulary.GFORMULA_COLUMN,
                    vocabulary.GFORMULA_DOSE,
                    vocabulary.GFORMULA_OFF.format(z=f"{SUPPORT_Z:g}"),
                    vocabulary.GFORMULA_WINDOW,
                ),
                support,
                title=vocabulary.GFORMULA_SUPPORT_TITLE,
                kind="other",
            )
        )
    notes: list[Note] = []
    if any(
        sum(1 for node, _, _ in report.covariate_types if node == time) >= 2
        for time in {node for node, _, _ in report.covariate_types}
    ):
        notes.append(note("draw_order", vocabulary.NOTE_TEXTS["draw_order"]))
    share = report.max_share
    if np.isfinite(share) and share > MONTE_CARLO_WARN_SHARE:
        worst = max(report.monte_carlo_share, key=lambda name: report.monte_carlo_share[name])
        notes.append(
            note(
                "simulation_share",
                vocabulary.NOTE_TEXTS["simulation_share"].format(share=f"{share:.1%}", name=worst),
            )
        )
    if report.model_reliance is not None:
        notes.append(note("model_reliance", report.model_reliance))
    return Summary.build(
        title=vocabulary.SIMULATION_TITLE,
        header=header,
        header_split=2,
        tables=tables,
        notes=notes,
    )


def gformula_report_repr(report: GFormulaReport) -> str:
    """The one-line repr of a simulation report.

    Parameters
    ----------
    report : GFormulaReport
        The report.

    Returns
    -------
    str
        For example ``GFormulaReport(draws=200, models=5, max simulation share=2.1%)``.
    """
    facts: list[tuple[str, Any]] = [("draws", report.n_draws), ("models", len(report.models))]
    if np.isfinite(report.max_share):
        facts.append(("max simulation share", f"{report.max_share:.1%}"))
    return short_repr(type(report).__name__, facts)

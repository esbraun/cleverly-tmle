"""Small summaries of two records a longitudinal fit keeps (X89 T2).

``result.msm_fits`` holds the working-model coefficients of an MSM fit, and an SNMM result's
``diagnostics`` holds what the fit measured about positivity. Each prints one table, and the
result's own summary is where their estimates are read with intervals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._summary import (
    Cell,
    Detail,
    Summary,
    SummarySection,
    SummaryTable,
    count_text,
    short_repr,
    vocabulary,
)

if TYPE_CHECKING:
    from .msm import MSMRegimenFit
    from .structural import StructuralDiagnostics


def _number(value: float) -> str:
    return vocabulary.MISSING if not np.isfinite(value) else f"{value:.4g}"


def msm_fit_summary(fit: MSMRegimenFit, detail: Detail) -> Summary:
    """Build the summary of one cause's working-model coefficients.

    Parameters
    ----------
    fit : MSMRegimenFit
        The fit.
    detail : {"short", "full"}
        ``"full"`` adds what the coefficients summarize.

    Returns
    -------
    Summary
        The summary.
    """
    model = fit.model
    header = [
        (vocabulary.MSM_LINK, model.link),
        (vocabulary.MSM_CELLS, count_text(len(model.cells))),
        (vocabulary.MSM_CONVERGED, vocabulary.YES if fit.converged else vocabulary.NO),
    ]
    if fit.cause is not None:
        header.insert(0, (vocabulary.MSM_CAUSE_LABEL, str(fit.cause)))
    beta = np.asarray(fit.beta, dtype=float)
    table = SummaryTable(
        (vocabulary.MSM_TERM, vocabulary.MSM_COEFFICIENT),
        tuple(
            (Cell(term, term), Cell(float(value), _number(float(value))))
            for term, value in zip(model.terms, beta, strict=True)
        ),
        title=vocabulary.MSM_TABLE,
        kind="other",
    )
    scale = vocabulary.MSM_LINK_WORDS.get(model.link, f"the {model.link} link")
    sections = (
        [
            SummarySection(
                vocabulary.SECTION_HOW_TO_READ, (vocabulary.MSM_HOW_TO_READ.format(scale=scale),)
            )
        ]
        if detail == "full"
        else []
    )
    return Summary.build(
        title=vocabulary.MSM_FIT_TITLE,
        header=header,
        header_split=len(header),
        tables=[table],
        reading=vocabulary.MSM_FIT_READING,
        sections=sections,
    )


def msm_fit_repr(fit: MSMRegimenFit) -> str:
    """The one-line repr of one cause's working-model coefficients.

    Parameters
    ----------
    fit : MSMRegimenFit
        The fit.

    Returns
    -------
    str
        For example ``MSMRegimenFit(terms=2, link=identity, converged=yes)``.
    """
    facts: list[tuple[str, object]] = []
    if fit.cause is not None:
        facts.append(("cause", fit.cause))
    facts += [
        ("terms", len(fit.model.terms)),
        ("link", fit.model.link),
        ("converged", vocabulary.YES if fit.converged else vocabulary.NO),
    ]
    return short_repr(type(fit).__name__, facts)


def structural_diagnostics_summary(diagnostics: StructuralDiagnostics, detail: Detail) -> Summary:
    """Build the summary of what an SNMM fit measured about positivity.

    Parameters
    ----------
    diagnostics : StructuralDiagnostics
        The record.
    detail : {"short", "full"}
        ``"full"`` adds what each column measures.

    Returns
    -------
    Summary
        The summary.
    """
    eigenvalues = diagnostics.smallest_eigenvalue
    shares = diagnostics.extreme_share
    count = max(len(eigenvalues), len(shares))

    def at(values: tuple[float, ...], index: int) -> float:
        return values[index] if index < len(values) else float("nan")

    rows = []
    for index in range(count):
        eigenvalue, share = at(eigenvalues, index), at(shares, index)
        rows.append(
            (
                Cell(index + 1, vocabulary.DIAGNOSTICS_TIME.format(time=index + 1)),
                Cell(eigenvalue, _number(eigenvalue)),
                Cell(share, vocabulary.MISSING if np.isnan(share) else f"{share:.1%}"),
            )
        )
    table = SummaryTable(
        (
            vocabulary.DIAGNOSTICS_NODE,
            vocabulary.DIAGNOSTICS_EIGENVALUE,
            vocabulary.DIAGNOSTICS_EXTREME,
        ),
        tuple(rows),
        title=vocabulary.DIAGNOSTICS_TABLE,
        kind="other",
    )
    header = [
        (vocabulary.DIAGNOSTICS_CONSTANT, count_text(diagnostics.n_constant_fits)),
    ]
    return Summary.build(
        title=vocabulary.DIAGNOSTICS_TITLE,
        header=header,
        header_split=len(header),
        tables=[table],
        sections=(
            [SummarySection(vocabulary.SECTION_HOW_TO_READ, (vocabulary.DIAGNOSTICS_READING,))]
            if detail == "full"
            else []
        ),
    )


def structural_diagnostics_repr(diagnostics: StructuralDiagnostics) -> str:
    """The one-line repr of what an SNMM fit measured about positivity.

    Parameters
    ----------
    diagnostics : StructuralDiagnostics
        The record.

    Returns
    -------
    str
        For example ``StructuralDiagnostics(smallest eigenvalue=0.21, constant fits=0)``.
    """
    eigenvalues = diagnostics.smallest_eigenvalue
    smallest = _number(min(eigenvalues)) if eigenvalues else vocabulary.MISSING
    return short_repr(
        type(diagnostics).__name__,
        [("smallest eigenvalue", smallest), ("constant fits", diagnostics.n_constant_fits)],
    )

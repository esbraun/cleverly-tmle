"""The printed summary every result object shares.

A summary has the shape of a statsmodels ``Summary``: a title, a header block of
``label: value`` facts in two columns, one or more tables, and numbered notes that apply
to this result only. ``summary(detail="full")`` adds sections after the notes. The
:class:`Summary` object is a ``str`` whose value is its text, so ``print`` shows the
table and a test can still search it, and it keeps every part as data: ``header``,
``tables``, ``notes`` and ``sections``. A notebook renders :meth:`Summary.as_html`.

Users meet these objects through ``result.summary()`` only. The module is private: only
``summary()`` is public.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import vocabulary
from ._render import (
    count_text,
    decimals_for,
    number_text,
    pvalue_text,
    render_html,
    render_text,
    share_text,
)

__all__ = [
    "Cell",
    "Detail",
    "Note",
    "Summary",
    "SummaryFacts",
    "SummarySection",
    "SummaryTable",
    "check_detail",
    "count_text",
    "decimals_for",
    "note",
    "number_text",
    "pvalue_text",
    "share_text",
    "short_repr",
    "vocabulary",
]

#: How much a summary prints: the analysis (``"short"``, the default), or the analysis plus
#: its assumptions, settings, method details and study description (``"full"``).
Detail = Literal["short", "full"]


def check_detail(detail: str) -> Detail:
    """Return ``detail`` when it names a detail level, and raise otherwise.

    Parameters
    ----------
    detail : str
        The ``detail=`` argument a ``summary()`` received.

    Returns
    -------
    {"short", "full"}
        The same value.

    Raises
    ------
    ValueError
        When ``detail`` is neither ``"short"`` nor ``"full"``.
    """
    if detail not in ("short", "full"):
        raise ValueError(f"detail must be 'short' or 'full'; got {detail!r}")
    return detail  # type: ignore[return-value]


@dataclass(frozen=True)
class Cell:
    """One table cell: the raw value and the text a reader sees.

    Parameters
    ----------
    value : Any
        The raw number, string or ``None``. :meth:`SummaryTable.to_frame` returns it.
    text : str
        The display string.
    """

    value: Any
    text: str

    @classmethod
    def of(cls, value: Any) -> Cell:
        """A cell whose text is ``str(value)``, or the missing mark for ``None``.

        Parameters
        ----------
        value : Any
            The raw value.

        Returns
        -------
        Cell
            The cell.
        """
        return cls(value, vocabulary.MISSING if value is None else str(value))


@dataclass(frozen=True)
class SummaryTable:
    """One table of a summary.

    Parameters
    ----------
    columns : tuple of str
        The column labels, row-label column first.
    rows : tuple of tuple of Cell
        One tuple of cells per row, one cell per column.
    title : str or None
        A heading printed above the column labels.
    kind : {"estimates", "other"}
        ``"estimates"`` for a table of estimates with their spread, which the contract
        test checks against the result's own frame. ``"other"`` for any other table.
    """

    columns: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]
    title: str | None = None
    kind: Literal["estimates", "other"] = "other"

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(
                    f"a row has {len(row)} cells and the table has {len(self.columns)} columns"
                )

    def column(self, label: str) -> tuple[Cell, ...]:
        """The cells of one column, top to bottom.

        Parameters
        ----------
        label : str
            A column label.

        Returns
        -------
        tuple of Cell
            The column's cells.
        """
        index = self.columns.index(label)
        return tuple(row[index] for row in self.rows)

    def row(self, label: str) -> dict[str, Cell]:
        """The cells of the row whose first cell has value ``label``, by column label.

        Parameters
        ----------
        label : str
            The raw value of the row-label cell.

        Returns
        -------
        dict of str to Cell
            The row's cells.

        Raises
        ------
        KeyError
            When no row has that label.
        """
        for row in self.rows:
            if row[0].value == label:
                return dict(zip(self.columns, row, strict=True))
        raise KeyError(label)

    def to_frame(self) -> Any:
        """The raw values, as a pandas frame when pandas is installed.

        Returns
        -------
        dataframe or dict
            One column per label, the row-label column first. A dict of lists without
            pandas.
        """
        columns = [label or "row" for label in self.columns]
        payload = {
            name: [row[index].value for row in self.rows] for index, name in enumerate(columns)
        }
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - pandas is a test dependency
            return payload
        return pd.DataFrame(payload, columns=columns)


@dataclass(frozen=True)
class Note:
    """One numbered note of a summary.

    Parameters
    ----------
    code : str
        A key of :data:`~cleverly._summary.vocabulary.NOTE_CODES`. Tests and programs
        match on it rather than on the wording.
    text : str
        The plain sentences.
    kind : {"warning", "note"}
        ``"warning"`` for a condition that changes how to read a number.
    """

    code: str
    text: str
    kind: Literal["warning", "note"] = "note"


def note(code: str, text: str) -> Note:
    """A note whose kind is the vocabulary's kind for ``code``.

    Parameters
    ----------
    code : str
        A key of :data:`~cleverly._summary.vocabulary.NOTE_CODES`.
    text : str
        The note's sentences.

    Returns
    -------
    Note
        The note.

    Raises
    ------
    KeyError
        When ``code`` is not in the vocabulary.
    """
    return Note(code, text, vocabulary.NOTE_CODES[code])


@dataclass(frozen=True)
class SummarySection:
    """One block that ``summary(detail="full")`` adds after the notes.

    Parameters
    ----------
    title : str
        The heading.
    lines : tuple of str
        One fact or sentence per line.
    """

    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class SummaryFacts:
    """What one method adds to a point-treatment summary.

    Returned by an estimator's ``_summary_facts(result)`` hook, which each method
    overrides to name itself and add the facts only it has.

    Parameters
    ----------
    title : str
        The summary title.
    method : str
        The Method header value.
    outcome_model : str or None
        The Outcome model header value, or ``None`` when the method fits no outcome model.
    treatment_model : str or None
        The Treatment model header value, or ``None`` when the method fits no treatment
        model.
    header : tuple of tuple of str
        More ``(label, value)`` facts, appended to the header.
    notes : tuple of Note
        More notes.
    sections : tuple of SummarySection
        More ``detail="full"`` sections.
    question : str or None
        A Question header value that replaces the one built from the estimands, for a
        method whose target the data chose.
    """

    title: str
    method: str
    outcome_model: str | None
    treatment_model: str | None
    header: tuple[tuple[str, str], ...] = ()
    notes: tuple[Note, ...] = ()
    sections: tuple[SummarySection, ...] = ()
    question: str | None = None


@dataclass(frozen=True)
class _Parts:
    title: str
    header: tuple[tuple[str, str], ...] = ()
    split: int | None = None
    tables: tuple[SummaryTable, ...] = ()
    notes: tuple[Note, ...] = ()
    reading: str | None = None
    sections: tuple[SummarySection, ...] = field(default=())


class Summary(str):
    """A printed summary of a result.

    Its ``str`` value is :meth:`as_text` at the default width, so ``print(summary)``
    shows it and a test can search it. Its parts stay readable as data, and a notebook
    renders :meth:`as_html`. Build one with :meth:`build`.

    The class is private; documentation calls it "the summary".
    """

    __slots__ = ("_parts",)
    _parts: _Parts

    #: The width the text is rendered at, statsmodels' default.
    WIDTH = 78

    def __new__(cls, parts: _Parts) -> Summary:
        made = super().__new__(cls, render_text(parts, cls.WIDTH))
        Summary._parts.__set__(made, parts)  # type: ignore[attr-defined]
        return made

    @classmethod
    def build(
        cls,
        *,
        title: str,
        header: Iterable[tuple[str, str]] = (),
        header_split: int | None = None,
        tables: Iterable[SummaryTable] = (),
        notes: Iterable[Note] = (),
        reading: str | None = None,
        sections: Iterable[SummarySection] = (),
    ) -> Summary:
        """Build a summary from its parts.

        Parameters
        ----------
        title : str
            The title.
        header : iterable of tuple of str
            ``(label, value)`` facts, rendered in two columns.
        header_split : int or None
            How many facts go in the left column. ``None`` puts the first half, rounded
            up, on the left.
        tables : iterable of SummaryTable
            The tables, in order.
        notes : iterable of Note
            The notes. Warnings print first, under "Warnings:", then the rest under
            "Notes:", numbered across both.
        reading : str or None
            One verdict sentence, for a check.
        sections : iterable of SummarySection
            The ``detail="full"`` sections.

        Returns
        -------
        Summary
            The summary.

        Raises
        ------
        ValueError
            When two notes share a code.
        """
        notes = tuple(notes)
        codes = [item.code for item in notes]
        if len(set(codes)) != len(codes):
            raise ValueError(f"two notes share a code: {codes}")
        return cls(
            _Parts(
                title=title,
                header=tuple((str(label), str(value)) for label, value in header),
                split=header_split,
                tables=tuple(tables),
                notes=tuple(sorted(notes, key=lambda item: item.kind != "warning")),
                reading=reading,
                sections=tuple(sections),
            )
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a summary is read-only")

    def __reduce__(self) -> tuple[Any, ...]:
        return (Summary, (self._parts,))

    def __repr__(self) -> str:
        return str.__str__(self)

    def __str__(self) -> str:
        return str.__str__(self)

    @property
    def heading(self) -> str:
        """The title. Named so that it leaves ``str.title`` alone."""
        return self._parts.title

    @property
    def header(self) -> tuple[tuple[str, str], ...]:
        """The ``(label, value)`` facts of the header block."""
        return self._parts.header

    @property
    def facts(self) -> dict[str, str]:
        """The header facts as a mapping from label to value."""
        return dict(self._parts.header)

    @property
    def tables(self) -> tuple[SummaryTable, ...]:
        """The tables, in order."""
        return self._parts.tables

    @property
    def notes(self) -> tuple[Note, ...]:
        """The notes, warnings first."""
        return self._parts.notes

    @property
    def note_codes(self) -> tuple[str, ...]:
        """The code of each note, in printed order."""
        return tuple(item.code for item in self._parts.notes)

    @property
    def reading(self) -> str | None:
        """The verdict sentence of a check, or ``None``."""
        return self._parts.reading

    @property
    def sections(self) -> tuple[SummarySection, ...]:
        """The ``detail="full"`` sections."""
        return self._parts.sections

    def note(self, code: str) -> Note:
        """The note with this code.

        Parameters
        ----------
        code : str
            A note code.

        Returns
        -------
        Note
            The note.

        Raises
        ------
        KeyError
            When the summary has no such note.
        """
        for item in self._parts.notes:
            if item.code == code:
                return item
        raise KeyError(code)

    def section(self, title: str) -> SummarySection:
        """The section with this title.

        Parameters
        ----------
        title : str
            A section title.

        Returns
        -------
        SummarySection
            The section.

        Raises
        ------
        KeyError
            When the summary has no such section.
        """
        for item in self._parts.sections:
            if item.title == title:
                return item
        raise KeyError(title)

    def table(self, kind: str = "estimates") -> SummaryTable:
        """The first table of a kind, or the first table titled ``kind``.

        Parameters
        ----------
        kind : str
            ``"estimates"``, ``"other"`` or a table title.

        Returns
        -------
        SummaryTable
            The table.

        Raises
        ------
        KeyError
            When no table matches.
        """
        for item in self._parts.tables:
            if item.kind == kind or item.title == kind:
                return item
        raise KeyError(kind)

    def as_text(self, width: int = 78) -> str:
        """The summary as plain text.

        Parameters
        ----------
        width : int
            The line width the header and notes wrap at. Tables never wrap.

        Returns
        -------
        str
            The text.
        """
        return render_text(self._parts, width)

    def as_html(self) -> str:
        """The summary as an HTML fragment, every string escaped.

        Returns
        -------
        str
            One ``<div class="cleverly-summary">``.
        """
        return render_html(self._parts)

    def _repr_html_(self) -> str:
        return self.as_html()


def short_repr(name: str, facts: Sequence[tuple[str, Any]] | Mapping[str, Any]) -> str:
    """A one-line repr ``Name(key=value, ...)`` of at most 100 characters.

    A fact whose key is empty prints its value alone, so ``("", "95% CI [1, 2]")`` reads
    as an unnamed part.

    Parameters
    ----------
    name : str
        The class name.
    facts : sequence of tuple or mapping
        The facts, in order.

    Returns
    -------
    str
        The repr, cut with ``...`` when it would pass 100 characters.
    """
    items = facts.items() if isinstance(facts, Mapping) else facts
    parts = [str(value) if not key else f"{key}={value}" for key, value in items]
    text = f"{name}({', '.join(parts)})"
    if len(text) <= 100:
        return text
    return text[:96].rstrip(", ") + "...)"


def finite(value: Any) -> bool:
    """Whether ``value`` is a finite real number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

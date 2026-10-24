"""The contract every printed summary keeps (X89 plan 8.2, items 1 to 7).

The fixtures come from every module of ``tests/unit/summary_fixtures``, so a seam thread that
migrates a summary adds its fixtures there and this file checks them. Each item is one
parametrized test. The deliberate-mutation controls patch the renderer and show that the
fidelity and internal-term checks fail when a summary misprints.
"""

from __future__ import annotations

import functools
import html.parser
import math
import pickle
import re
import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cleverly._inference_status import STATUS_RECORDS
from cleverly._summary import Cell, Summary, SummaryTable, _render, vocabulary
from cleverly._summary._estimates import (
    _ID,
    estimates_table,
    plain_support_reason,
    without_roadmap,
)
from tests.unit.summary_fixtures import Case, collect

FIXTURES = collect()
KEYS = sorted(FIXTURES)
WIDTH = Summary.WIDTH


@functools.cache
def _fitted(key: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FIXTURES[key]()


def _summary(key: str, detail: str = "short") -> Summary:
    return _fitted(key).summary(detail=detail)


# ----------------------------------------------------------------- reading the text


def _estimates_block(text: str) -> tuple[list[str], list[list[str]]]:
    """The column labels and the row cells of the first table in the printed text."""
    lines = text.splitlines()
    rule = next(index for index, line in enumerate(lines) if line and set(line) == {"-"})
    labels = re.split(r"\s{2,}", lines[rule - 1].strip())
    rows = []
    for line in lines[rule + 1 :]:
        if set(line) == {"="}:
            break
        cells = re.split(r"\s{2,}", line.strip())
        if rows and len(cells) == 1:
            # The rest of a long row name the renderer wrapped, at a space or after "[".
            name = rows[-1][0]
            rows[-1][0] = name + ("" if name.endswith("[") else " ") + cells[0]
            continue
        rows.append(cells)
    return labels, rows


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def _close(shown: str, value: float) -> bool:
    """Whether a printed number agrees with ``value`` to half a unit of its last place."""
    return abs(float(shown) - value) <= 0.5 * 10 ** -_decimals(shown) + 1e-12


def _column_of(label: str) -> str | tuple[str, str] | None:
    if label == vocabulary.ESTIMATE:
        return "psi"
    if label.endswith(" CI"):
        return ("ci_lower", "ci_upper")
    if label == vocabulary.DF:
        return "reference_df"
    if label == vocabulary.SUPPORT:
        return "support"
    for name, spread in vocabulary.SPREAD_LABELS.items():
        if spread == label and "_" in name:
            return name
    return None


def _check_fidelity(result: Any, text: str) -> None:
    """Every printed estimates cell equals the value ``to_frame()`` reports for that row."""
    frame = result.to_frame().set_index("estimand")
    labels, rows = _estimates_block(text)
    assert rows, "the summary printed no estimates rows"
    for cells in rows:
        name = cells[0].rstrip(vocabulary.NO_INFERENCE_MARK)
        record = frame.loc[name]
        for label, shown in zip(labels, cells[1:], strict=False):
            column = _column_of(label)
            assert column is not None, f"unknown column label {label!r}"
            if shown == vocabulary.MISSING:
                continue
            if isinstance(column, tuple):
                low, high = (part.strip() for part in shown.strip("[]").split(","))
                assert _close(low, float(record[column[0]])), (name, label, shown)
                assert _close(high, float(record[column[1]])), (name, label, shown)
            elif column == "p_value" and shown.startswith("<"):
                assert float(record[column]) < float(shown[1:]), (name, shown)
            elif column == "reference_df":
                value = record.get(column)
                if shown == "normal":
                    assert value is None or math.isnan(float(value)), (name, shown)
                else:
                    assert int(shown) == int(value), (name, shown)
            elif column == "support":
                assert shown == record[column], (name, shown)
            else:
                assert _close(shown, float(record[column])), (name, label, shown, record[column])


def _user_names(result: Any, key: str | None = None) -> set[str]:
    """The user-supplied names a summary prints: read off a result's data, or, for an
    object without data such as a check, the names its fixture declares."""
    data = getattr(result, "data", None)
    if data is None:
        return set(FIXTURES[key].names) if key is not None else set()
    names = {
        getattr(data, "outcome_name", None),
        getattr(data, "treatment_name", None),
        *getattr(data, "covariate_names", ()),
        *getattr(data, "indicator_sources", {}).values(),
        *(getattr(data, "strata_names", None) or ()),
        *(str(level) for level in getattr(data, "treatment_levels", None) or ()),
        # A longitudinal fit names its columns per time point.
        *getattr(data, "treatment_names", ()),
        *getattr(data, "baseline_names", ()),
        *(name for names in getattr(data, "time_varying_names", ()) for name in names),
        *getattr(data, "censoring_names", ()),
        *getattr(data, "event_names", ()),
        *getattr(data, "cause_labels", ()),
    }
    if data.is_weighted:
        names.add(data.weight_report().name)
    return {str(name) for name in names if name}


def _authored(result: Any, summary: Summary, key: str | None = None) -> str:
    """Every string of the summary, with user-supplied names and quoted warnings taken out.

    A fit-time warning is quoted as the user saw it, and its wording belongs to the warning
    site, so the check reads the note's own sentence around it.
    """
    pieces = [summary.heading]
    pieces.extend(f"{label}: {value}" for label, value in summary.header)
    for table in summary.tables:
        pieces.extend([table.title or "", *table.columns])
        pieces.extend(cell.text for row in table.rows for cell in row)
    messages = [record.message for record in getattr(result, "fit_warnings", ())]
    if key is not None:
        messages.extend(FIXTURES[key].quotes(result))
    for item in summary.notes:
        text = item.text
        for message in messages:
            text = text.replace(message, " ")
        pieces.append(text)
    for section in summary.sections:
        pieces.extend([section.title, *section.lines])
    text = "\n".join(pieces)
    for name in sorted(_user_names(result, key), key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", "NAME", text)
    return text


_ROADMAP_ID = re.compile(r"(?<![\w])(?:X|F|RM|G|T|S|B|L)\d+[a-z]?(?![\w])|(?<![\w])BP\d*(?![\w])")
_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{12,}(?![0-9a-fA-F])")
_VERSION = re.compile(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def _check_internals(result: Any, summary: Summary, key: str | None = None) -> None:
    text = _authored(result, summary, key)
    assert " -- " not in text
    assert "—" not in text
    assert not _ROADMAP_ID.search(text), _ROADMAP_ID.search(text)
    assert not _HEX.search(text), _HEX.search(text)
    assert not _VERSION.search(text), _VERSION.search(text)
    assert not _TIMESTAMP.search(text), _TIMESTAMP.search(text)
    lowered = text.lower()
    for term in vocabulary.INTERNAL_TERMS:
        assert term.lower() not in lowered, term


# ----------------------------------------------------------------- items 1 to 7


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("detail", ["short", "full"])
def test_a_summary_has_a_heading_a_table_and_fits_the_width(key: str, detail: str) -> None:
    summary = _summary(key, detail)
    assert isinstance(summary, Summary)
    assert summary.heading.strip()
    assert summary.tables
    long = [line for line in str(summary).splitlines() if len(line) > WIDTH]
    assert not long, long


def _check_numbers(case: Case, result: Any) -> None:
    """Read every printed number back: an estimates frame through ``to_frame()``, any other
    object through its fixture's reader. A fixture with neither fails rather than passing
    unread."""
    text = str(result.summary())
    if case.fidelity is not None:
        case.fidelity(result, text)
        return
    frame = result.to_frame() if hasattr(result, "to_frame") else None
    assert frame is not None and "estimand" in frame.columns, (
        "the object has no estimates frame and its fixture gives no fidelity reader"
    )
    _check_fidelity(result, text)


@pytest.mark.parametrize("key", KEYS)
def test_every_printed_number_equals_the_object(key: str) -> None:
    _check_numbers(FIXTURES[key], _fitted(key))


def test_a_fixture_without_a_reader_fails_the_number_check() -> None:
    """Control: an object that has no estimates frame cannot pass without a reader."""
    summary = Summary.build(
        title="A check",
        tables=[SummaryTable(("row", "value"), ((Cell("a", "a"), Cell(1.0, "1.000")),))],
    )
    bare = SimpleNamespace(summary=lambda: summary)
    with pytest.raises(AssertionError, match="no fidelity reader"):
        _check_numbers(Case(lambda: bare), bare)
    read: list[str] = []
    _check_numbers(Case(lambda: bare, fidelity=lambda obj, text: read.append(text)), bare)
    assert read == [str(summary)]


@pytest.mark.parametrize("key", KEYS)
def test_the_html_parses_and_escapes_every_name(key: str) -> None:
    result = _fitted(key)
    page = result.summary().as_html()

    class Reader(html.parser.HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.tags: set[str] = set()
            self.text: list[str] = []

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            self.tags.add(tag)

        def handle_data(self, data: str) -> None:
            self.text.append(data)

    reader = Reader()
    reader.feed(page)
    reader.close()
    allowed = {"div", "h4", "table", "caption", "thead", "tbody", "tr", "th", "td"}
    assert reader.tags <= allowed | {"p", "ol", "li", "ul", "b"}
    shown = " ".join(reader.text)
    for name in _user_names(result, key):
        if "<" in name or "&" in name:
            assert name in shown, name
            assert name in str(result.summary()), name
            assert name not in page, name


def test_the_markup_fixture_reaches_the_escaping_check() -> None:
    """Witness: the escaping test above has a name with markup to check."""
    names = _user_names(_fitted("point_tmle:markup_names"))
    assert any("<script>" in name for name in names)
    assert any("&" in name for name in names)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("detail", ["short", "full"])
def test_the_text_holds_no_library_internals(key: str, detail: str) -> None:
    result = _fitted(key)
    _check_internals(result, result.summary(detail=detail), key)


def test_a_covariate_named_like_an_id_prints_as_written() -> None:
    result = _fitted("point_tmle:id_like_covariate")
    text = str(result.summary())
    assert "X1, F18" in text
    _check_internals(result, result.summary())


@pytest.mark.parametrize("key", KEYS)
def test_the_repr_is_one_short_line(key: str) -> None:
    text = repr(_fitted(key))
    assert "\n" not in text
    assert len(text) <= 100
    assert text.startswith(type(_fitted(key)).__name__ + "(")


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("detail", ["short", "full"])
def test_notes_are_unique_and_from_the_vocabulary(key: str, detail: str) -> None:
    summary = _summary(key, detail)
    codes = summary.note_codes
    assert len(codes) == len(set(codes))
    assert set(codes) <= set(vocabulary.NOTE_CODES)
    texts = [item.text for item in summary.notes]
    assert len(texts) == len(set(texts))
    warnings_first = [item.kind for item in summary.notes]
    assert warnings_first == sorted(warnings_first, key=lambda kind: kind != "warning")


@pytest.mark.parametrize("key", KEYS)
def test_the_full_summary_holds_every_short_fact(key: str) -> None:
    short, full = _summary(key), _summary(key, "full")
    assert set(short.header) <= set(full.header)
    assert short.note_codes == full.note_codes
    assert [table.rows for table in short.tables] == [table.rows for table in full.tables]
    assert full.sections and not short.sections


# ----------------------------------------------------------------- coverage


#: The note codes the ``TMLEResult`` pilot prints, each fired by a fixture or by a literal
#: check below. The other codes of the vocabulary belong to the summaries the seam threads
#: migrate, and each of those threads adds the fixtures that fire them.
PILOT_CODES = {
    "no_inference",
    "experimental",
    "in_sample_flexible",
    "weights_capped",
    "weighted_population",
    "t_reference",
    "spread_scale",
    "repeated_splits",
    "simultaneous_bands",
    "bootstrap_diagnostic",
    "bootstrap_experimental",
    "score_unsolved",
    "stopping_rule",
    "folds_reduced",
    "fit_warning_other",
}
LITERAL_CODES = {"score_unsolved"}


def test_every_pilot_note_fires() -> None:
    fired = {code for key in KEYS for code in _summary(key).note_codes}
    assert fired | LITERAL_CODES >= PILOT_CODES, PILOT_CODES - fired - LITERAL_CODES
    assert set(vocabulary.NOTE_CODES) >= PILOT_CODES


def test_an_unsolved_score_equation_prints_its_warning() -> None:
    """A literal check: the compact fixtures all solve their score equations."""
    result = FIXTURES["point_tmle:single_ate"]()
    assert result.score_verdict.passed
    assert "score_unsolved" not in result.summary().note_codes
    result.__dict__["score_verdict"] = SimpleNamespace(
        rows=(),
        declared=False,
        passed=False,
        failures=(SimpleNamespace(name="ate"),),
        rule_failures=(),
        alpha_sig=0.05,
    )
    summary = result.summary()
    item = summary.note("score_unsolved")
    assert item.kind == "warning"
    assert item.text == vocabulary.NOTE_TEXTS["score_unsolved"].format(
        names="ate", spread="interval"
    )


def test_a_withheld_row_in_an_inferential_table_is_starred() -> None:
    """A mixed table: one row supplies no inference, the other does."""
    estimates = dict(_fitted("point_tmle:repeated").estimates)
    estimates["ey0"] = replace(estimates["ey0"], inference="no_cluster_variance_plugin")
    table, notes = estimates_table(estimates, level=0.95)
    rows = {row[0].value: row for row in table.rows}
    assert rows["ey0"][0].text == "ey0" + vocabulary.NO_INFERENCE_MARK
    assert all(cell.text == vocabulary.MISSING for cell in rows["ey0"][2:])
    assert rows["ate"][0].text == "ate"
    assert rows["ate"][2].text != vocabulary.MISSING
    (withheld,) = [item for item in notes if item.code == "no_inference"]
    assert STATUS_RECORDS["no_cluster_variance_plugin"].plain_reason in withheld.text


# ----------------------------------------------------------------- mutation controls


def _rebuilt(table: SummaryTable, rows: tuple[tuple[Cell, ...], ...]) -> SummaryTable:
    return replace(table, rows=rows)


def test_a_renderer_that_swaps_two_rows_fails_fidelity(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fitted("point_tmle:canonical")
    original = _render._table_lines

    def swapped(table: SummaryTable, width: int) -> Any:
        if table.kind == "estimates":
            first, second, *rest = table.rows
            table = _rebuilt(table, ((first[0], *second[1:]), (second[0], *first[1:]), *rest))
        return original(table, width)

    _check_fidelity(result, str(result.summary()))
    monkeypatch.setattr(_render, "_table_lines", swapped)
    with pytest.raises(AssertionError):
        _check_fidelity(result, str(result.summary()))


def test_a_renderer_that_swaps_estimate_and_spread_fails_fidelity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fitted("point_tmle:canonical")
    original = _render._table_lines

    def swapped(table: SummaryTable, width: int) -> Any:
        if table.kind == "estimates":
            table = _rebuilt(table, tuple((row[0], row[2], row[1], *row[3:]) for row in table.rows))
        return original(table, width)

    monkeypatch.setattr(_render, "_table_lines", swapped)
    with pytest.raises(AssertionError):
        _check_fidelity(result, str(result.summary()))


def test_a_data_hash_in_the_header_fails_the_internal_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from cleverly.estimators import _point_summary

    result = _fitted("point_tmle:canonical")
    digest = hashlib.sha256(result.to_frame().to_csv().encode()).hexdigest()[:16]
    original = _point_summary._header

    def leaky(*args: Any) -> Any:
        header, split = original(*args)
        return [*header, ("Data", digest)], split

    _check_internals(result, result.summary())
    monkeypatch.setattr(_point_summary, "_header", leaky)
    assert digest in str(result.summary())
    with pytest.raises(AssertionError):
        _check_internals(result, result.summary())


# ----------------------------------------------------------------- the objects


@pytest.mark.parametrize("key", KEYS)
def test_the_notebook_display_raises_no_warning(key: str) -> None:
    result = _fitted(key)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        page = result._repr_html_()
    assert page.startswith('<div class="cleverly-summary">')


@pytest.mark.parametrize("key", ["point_tmle:canonical", "point_tmle:collaborative"])
def test_a_summary_pickles_with_its_parts(key: str) -> None:
    summary = _summary(key, "full")
    copy = pickle.loads(pickle.dumps(summary))
    assert isinstance(copy, Summary)
    assert copy == summary
    assert copy.note_codes == summary.note_codes
    assert copy.tables == summary.tables
    assert copy.as_html() == summary.as_html()


def test_every_spread_a_result_prints_has_a_plain_label() -> None:
    """Each spread name an estimates table can print maps to a label, and a diagnostic
    spread says so in its label."""
    from cleverly.inference.influence import spread_name

    for status, record in STATUS_RECORDS.items():
        name = spread_name("std_err", status)
        label = vocabulary.SPREAD_LABELS[name]
        if record.interval == "withheld":
            assert "diagnostic" in label, (status, label)
        else:
            assert label == vocabulary.SPREAD_LABELS["std_err"], (status, label)


def test_note_numbers_come_from_constants() -> None:
    """No note template writes a number of its own: each comes through a placeholder."""
    for code, template in vocabulary.NOTE_TEXTS.items():
        literal = re.sub(r"\{[^}]*\}", "", template)
        assert not re.search(r"\d", literal), (code, literal)


def test_every_status_has_a_plain_reason_without_ids() -> None:
    for status, record in STATUS_RECORDS.items():
        assert record.plain_reason.strip(), status
        assert not _ID.search(record.plain_reason), (status, record.plain_reason)
        assert " -- " not in record.plain_reason


def test_a_support_reason_loses_its_ids_and_names_its_axes() -> None:
    reason = (
        "No registered study covers this configuration. Nearest row: T01.continuous "
        "(canonical-tmle), differing on dependence, qualifiers."
    )
    plain = plain_support_reason(reason)
    assert plain == (
        "No validation study covers this kind of fit. The nearest validated kind of fit "
        "differs in clustering or the number of clusters, strata or a known treatment "
        "mechanism."
    )


# ----------------------------------------------------------------- header numbers and assumptions


def _header_value(summary: Summary, label: str) -> str | None:
    values = [value for name, value in summary.header if name == label]
    return values[0] if values else None


def _check_header_counts(result: Any, summary: Summary) -> None:
    data = result.data
    assert _header_value(summary, vocabulary.ROWS) == f"{data.n:,}"
    share = _header_value(summary, vocabulary.TREATED_SHARE)
    if share is not None and not data.is_weighted:
        assert _close(share, float(np.mean(data.treatment)))


@pytest.mark.parametrize("key", KEYS)
def test_the_header_counts_equal_the_data(key: str) -> None:
    """The row count and the treated share print the numbers of the fitted data.

    An object without data, such as a check, prints counts its fidelity reader checks.
    """
    result = _fitted(key)
    data = getattr(result, "data", None)
    if data is None or getattr(data, "treatment", None) is None:
        assert FIXTURES[key].fidelity is not None, key
        return
    _check_header_counts(result, _summary(key))


def test_every_fitted_result_reaches_the_header_count_check() -> None:
    """Witness: the data guard above never skips a fitted result."""
    results = [key for key in KEYS if key.startswith("point_tmle:")]
    assert results
    assert all(getattr(_fitted(key).data, "treatment", None) is not None for key in results)


def test_a_header_that_misreports_the_rows_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: a header off by one row fails the check above."""
    from cleverly.estimators import _point_summary

    original = _point_summary._header

    def shifted(*args: Any, **kwargs: Any) -> Any:
        header, split = original(*args, **kwargs)
        moved = [
            (name, str(int(value.replace(",", "")) + 1) if name == vocabulary.ROWS else value)
            for name, value in header
        ]
        return moved, split

    monkeypatch.setattr(_point_summary, "_header", shifted)
    result = _fitted("point_tmle:single_ate")
    with pytest.raises(AssertionError):
        _check_header_counts(result, result.summary())


def _assumptions(key: str) -> list[str]:
    return list(_summary(key, "full").section("Assumptions").lines)


def test_missing_outcomes_add_their_assumptions() -> None:
    lines = _assumptions("point_tmle:missing_outcome")
    for text in vocabulary.MISSING_OUTCOME_ASSUMPTIONS:
        assert f"- {text.format(Y='Y', A='A')}" in lines
    for text in vocabulary.MISSING_OUTCOME_ASSUMPTIONS:
        assert f"- {text.format(Y='Y', A='A')}" not in _assumptions("point_tmle:single_ate")


def test_a_held_intermediate_adds_its_assumptions() -> None:
    lines = _assumptions("point_tmle:direct_effect")
    for text in vocabulary.INTERMEDIATE_ASSUMPTIONS:
        assert f"- {text.format(Y='Y', A='A', Z='Z', z='0')}" in lines


def test_a_declared_rule_without_an_interval_states_the_rule_plainly() -> None:
    """With no interval there is no coverage bound to state, only the rule."""
    summary = _summary("point_tmle:declared_rule_withheld")
    assert summary.note("stopping_rule").text == vocabulary.NOTE_TEXTS["stopping_rule_plain"]
    bounded = _summary("point_tmle:declared_rule").note("stopping_rule").text
    assert bounded != vocabulary.NOTE_TEXTS["stopping_rule_plain"]


def test_roadmap_pointers_leave_and_a_column_named_like_an_id_stays() -> None:
    text = "The point estimate stands. X58 in docs/roadmap.md describes the stack."
    assert without_roadmap(text) == "The point estimate stands."
    assert without_roadmap("No stacked interval (X58): it needs W") == (
        "No stacked interval: it needs W"
    )
    assert without_roadmap("column X1 is constant") == "column X1 is constant"


def test_a_long_column_name_in_the_header_is_never_broken() -> None:
    """A value with a word too long for its half of the header takes a full-width row."""
    names = "married, lung_heart_kidney_disease, previous_preterm_or_small and 12 more"
    header = (
        ("Question", "average effect of A on Y (1 compared with 0)"),
        ("Adjusted for", names),
        ("Rows", "51,874"),
        ("Treated share", "0.163"),
    )
    lines = _render._header_lines(header, WIDTH, 2)
    printed = " ".join(" ".join(lines).split())
    for name in ("lung_heart_kidney_disease,", "previous_preterm_or_small"):
        assert name in printed.split()
    assert all(len(line) <= WIDTH for line in lines)
    assert lines[-2].startswith("Adjusted for:")

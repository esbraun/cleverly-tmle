"""Witnesses for the point-estimate summaries (X89, thread T1).

The contract test reads every printed number back through the readers of
``summary_fixtures/_point_readers``. This file checks the rest: that every note code T1 adds
fires on a fixture, that the notes which name values name the record's own values, and that
the readers fail on a summary that misprints (deliberate-mutation controls).
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Any

import pytest

from cleverly._summary import Cell, Summary, SummaryTable, note, vocabulary
from cleverly.estimators import _method_summary
from cleverly.inference.results import RepeatSpread
from tests.unit.summary_fixtures import _point_readers as read
from tests.unit.summary_fixtures import point_methods as fixtures


@functools.cache
def _built(name: str) -> Any:
    builders = {**fixtures.FIXTURES, **{f"fit:{k}": v for k, v in fixtures.RESULTS.items()}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return builders[name]()


def _names() -> list[str]:
    return [*fixtures.FIXTURES, *(f"fit:{k}" for k in fixtures.RESULTS)]


def test_every_point_methods_note_fires() -> None:
    """Each note code T1 adds prints on at least one fixture."""
    fired: set[str] = set()
    for name in _names():
        for detail in ("short", "full"):
            fired.update(_built(name).summary(detail=detail).note_codes)
    assert set(vocabulary.T1_NOTE_CODES) <= fired, set(vocabulary.T1_NOTE_CODES) - fired


@pytest.mark.parametrize("name", ["fit:greedy", "fit:learners_unordered", "fit:stacked"])
def test_a_fit_summary_carries_its_method_record(name: str) -> None:
    """A fitted result prints its method's sentence and the record's notes."""
    summary = _built(name).summary(detail="full")
    assert summary.section(vocabulary.SECTION_METHOD)


def _left_out_witness(selection: Any) -> None:
    summary = selection.summary()
    dropped = selection.dropped
    assert dropped, "the fixture must leave a covariate out"
    assert "left_out" in summary.note_codes
    text = summary.note("left_out").text
    for covariate in dropped:
        assert covariate in text, covariate


def test_the_left_out_note_names_the_dropped_covariates() -> None:
    _left_out_witness(_built("greedy_selection"))


def test_a_builder_that_drops_the_left_out_note_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: the witness above fails when the note is not built."""
    monkeypatch.setattr(_method_summary, "_left_out_text", lambda selection: None)
    with pytest.raises(AssertionError):
        _left_out_witness(_built("greedy_selection"))


def test_the_stopping_note_names_the_record_candidate() -> None:
    selection = _built("ordered_patience")
    assert selection.stopped_at is not None
    text = selection.summary().note("search_stopped").text
    assert f"after candidate {selection.stopped_at + 1}:" in text


def test_the_rounds_note_names_the_rounds_run() -> None:
    fit = _built("iptw_capped")
    assert fit.stop != "tolerance"
    text = fit.summary().note("rounds_capped").text
    assert f"({fit.rounds}" in text


def test_the_guard_note_follows_the_record() -> None:
    assert _built("reduced").guard
    assert "guard_scope" in _built("reduced").summary().note_codes
    assert not _built("reduced_unguarded").guard
    assert _built("reduced_unguarded").summary().note_codes == ("ordinary_tmle",)


def test_a_spread_that_swaps_its_columns_fails_the_reader() -> None:
    """Deliberate mutation: the spread printed in the standard-error column fails."""
    spread = _built("spread")
    read.read_spread(spread, str(spread.summary()))
    swapped = RepeatSpread(
        spread.std_errors,
        spread.n_repeats,
        dict(spread),
        spread.inferential,
        spread.scales,
    )
    with pytest.raises(AssertionError):
        read.read_spread(spread, str(swapped.summary()))


def test_a_selection_that_swaps_its_risks_fails_the_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate mutation: in-sample and cross-validated risks swapped fail the reader."""
    selection = _built("greedy_selection")
    original = type(selection).to_frame

    def swapped(self: Any) -> Any:
        frame = dict(original(self))
        frame["train_risk"], frame["cv_risk"] = frame["cv_risk"], frame["train_risk"]
        return frame

    text = str(selection.summary())
    monkeypatch.setattr(type(selection), "to_frame", swapped)
    with pytest.raises(AssertionError):
        read.read_selection(selection, text)


def test_a_reader_rejects_a_missing_row() -> None:
    """Control: a table that prints one row fewer than the object holds fails."""
    bands = _built("bands")
    text = str(bands.summary())
    first = next(iter(bands.bands))
    kept = "\n".join(line for line in text.splitlines() if not line.startswith(first + " "))
    with pytest.raises(AssertionError):
        read.read_bands(bands, kept)


@pytest.mark.parametrize("word", ["plug-in", "out-of-fold"])
def test_a_note_never_breaks_a_word_at_its_hyphen(word: str) -> None:
    """A hyphenated word that reaches the wrap point moves whole to the next line."""
    table = SummaryTable(("", "value"), ((Cell("a", "a"), Cell(1.0, "1.0")),))
    for pad in range(60, 80):
        text = "x" * pad + f" {word} " + "y" * 10
        printed = str(Summary.build(title="T", tables=[table], notes=[note("split_spread", text)]))
        lines = printed.splitlines()
        assert any(word in line for line in lines), (pad, printed)
        assert not any(line.rstrip().endswith(word.split("-")[0] + "-") for line in lines)


def test_a_header_count_that_disagrees_with_the_object_fails() -> None:
    """Control: the reader compares the header counts with the object, not the summary."""
    bootstrap = _built("bootstrap")
    text = str(bootstrap.summary())
    read.read_bootstrap(bootstrap, text)
    other = replace(bootstrap, n_failed=bootstrap.n_failed + 1)
    with pytest.raises(AssertionError):
        read.check_header(other, text, {"Failed": other.n_failed})

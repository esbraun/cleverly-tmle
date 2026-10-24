"""The study-setup summaries (X89 T5): what each prints, and controls that fail on a misprint.

``tests/unit/test_summary_contract.py`` runs the shared contract over the fixtures in
``tests/unit/summary_fixtures/setup.py``. This file adds what the contract cannot see: that
each note fires where it should, that every identification statement reaches the condition
it names, that the fidelity readers fail on a summary that misprints, and the one-line
reprs of the objects that have no summary of their own.
"""

from __future__ import annotations

import pickle
import re
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from cleverly import ATE, CausalStudy, MethodAvailabilities, PointTreatment
from cleverly._summary import Summary, _setup, vocabulary
from cleverly.datasets import make_linear_ate
from cleverly.learners.crossfit import SplitPlan
from cleverly.methods import MethodAvailability
from tests.unit.summary_fixtures import _setup_readers as readers
from tests.unit.summary_fixtures import setup

# ----------------------------------------------------------------- the notes


#: The note codes the setup summaries print, each fired by a fixture or a literal check.
SETUP_CODES = {
    "further_conditions",
    "checkable",
    "model_restriction",
    "several_readings",
    "unbound_effect",
    "no_method",
    "other_methods",
    "failed_replications",
    "measures_diagnostic",
    "mixed_statuses",
    "spread_scale",
    "experimental",
    "worker_time",
}
LITERAL_CODES = {"no_method"}


def _codes(name: str) -> tuple[str, ...]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return setup.FIXTURES[name]().summary().note_codes


def test_every_setup_note_fires() -> None:
    fired = {code for name in setup.FIXTURES for code in _codes(name)}
    assert fired | LITERAL_CODES >= SETUP_CODES, SETUP_CODES - fired - LITERAL_CODES
    assert set(vocabulary.NOTE_CODES) >= SETUP_CODES


def _unavailable(records: Any) -> MethodAvailabilities:
    refusal = next(record.refusal for record in records if not record.available)
    return MethodAvailabilities(
        (replace(record, available=False, refusal=refusal) for record in records),
        question=records.question,
    )


def test_a_question_no_method_fits_warns_and_lists_every_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A literal check: every shipped question has a method that fits it."""
    effect = setup.ate_effect()
    records = _unavailable(effect.available_methods())
    monkeypatch.setattr(type(effect), "available_methods", lambda self: records)
    summary = effect.summary()
    assert summary.note("no_method").kind == "warning"
    assert "other_methods" not in summary.note_codes
    rows = summary.tables[-1].rows
    assert [row[0].value for row in rows] == [record.name for record in records]
    methods = records.summary()
    assert methods.note("no_method").text == vocabulary.NOTE_TEXTS["no_method"]


def test_the_question_lists_the_methods_that_fit_and_counts_the_rest() -> None:
    effect = setup.att_effect()
    records = effect.available_methods()
    fitting = [record.name for record in records if record.available]
    summary = effect.summary()
    assert [row[0].value for row in summary.tables[-1].rows] == fitting
    unfit = len(records) - len(fitting)
    assert f"{unfit} other methods cannot" in summary.note("other_methods").text
    full = effect.summary(detail="full").section(vocabulary.SECTION_UNAVAILABLE)
    assert len(full.lines) >= unfit


def test_one_further_condition_reads_in_the_singular() -> None:
    text = setup.att_effect().summary().note("further_conditions").text
    assert "on 1 further condition on" in text
    assert text.endswith("states it.")
    plural = setup.mtp_effect().summary().note("further_conditions").text
    assert "further conditions on" in plural


def test_an_unbound_question_warns_and_prints_no_rows() -> None:
    summary = setup.unbound_effect().summary()
    assert summary.note("unbound_effect").kind == "warning"
    assert vocabulary.ROWS not in dict(summary.header)
    bound = setup.ate_effect().summary()
    assert "unbound_effect" not in bound.note_codes


def test_a_dropped_column_is_quoted_and_left_out_of_the_adjustment() -> None:
    study = setup.study_dropped_column()
    summary = study.summary()
    assert dict(summary.header)[vocabulary.ADJUSTED_FOR] == "W1, W2"
    (item,) = [item for item in summary.notes if item.code == "fit_warning_other"]
    assert "'K'" in item.text
    effect = study.identify(ATE())
    assert dict(effect.summary().header)[vocabulary.ADJUSTED_FOR] == "W1, W2"


# ----------------------------------------------------------------- the conditions


#: Each question fixture's conditions table, in order: the plain name of each statement.
CONDITIONS = {
    "ate_effect": [
        "well-defined treatment",
        "no interference",
        "no hidden confounding",
        "overlap",
    ],
    "direct_effect": [
        "well-defined treatment",
        "no interference",
        "no hidden confounding",
        "no hidden confounding of Z",
        "overlap",
        "overlap for Z",
        "outcome missing at random",
        "overlap for recorded outcomes",
    ],
    "regime_effect": [
        "well-defined treatment",
        "no interference",
        "no hidden confounding at each time",
        "overlap at each time",
    ],
}


@pytest.mark.parametrize("name", sorted(CONDITIONS))
def test_every_statement_reaches_the_condition_it_names(name: str) -> None:
    summary = setup.FIXTURES[name]().summary()
    table = summary.tables[0]
    assert table.title == vocabulary.CONDITIONS_TITLE
    assert [row[0].text for row in table.rows] == CONDITIONS[name]


def test_a_mapping_that_misnames_a_condition_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: swapping the names of two conditions fails the pin above."""
    original = _setup.condition_words

    def swapped(statement: str) -> Any:
        found = original(statement)
        if found is not None and found[0] == "overlap":
            return ("no interference", *found[1:])
        return found

    monkeypatch.setattr(_setup, "condition_words", swapped)
    summary = setup.ate_effect.__wrapped__().summary()
    assert [row[0].text for row in summary.tables[0].rows] != CONDITIONS["ate_effect"]


# ----------------------------------------------------------------- the readers fail on a misprint


def _shifted_header(study: Any, label: str, value: str) -> str:
    """The study's summary with one header value replaced."""
    summary = study.summary()
    header, split = _setup.study_header(study)
    return str(
        Summary.build(
            title=summary.heading,
            header=[(name, value if name == label else shown) for name, shown in header],
            header_split=split,
            tables=summary.tables,
            notes=summary.notes,
        )
    )


def test_a_study_header_off_by_one_row_fails_its_reader() -> None:
    study = setup.point_study()
    text = str(study.summary())
    readers.read_study(study, text)
    wrong = _shifted_header(study, vocabulary.ROWS, f"{study.data.n + 1:,}")
    with pytest.raises(AssertionError):
        readers.read_study(study, wrong)


def test_a_misprinted_treated_share_fails_its_reader() -> None:
    study = setup.point_study()
    share = float(np.mean(study.data.treatment))
    wrong = _shifted_header(study, vocabulary.TREATED_SHARE, f"{1 - share:.3f}")
    with pytest.raises(AssertionError):
        readers.read_study(study, wrong)


def test_a_flipped_method_fails_its_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    methods = setup.ate_methods()
    readers.read_methods(methods, str(methods.summary()))
    original = _setup.methods_table

    def flipped(records: Any) -> Any:
        table = original(records)
        first, *rest = table.rows
        cell = first[1]
        return replace(
            table,
            rows=((first[0], replace(cell, text="no" if cell.value else "yes"), first[2]), *rest),
        )

    monkeypatch.setattr(_setup, "methods_table", flipped)
    with pytest.raises(AssertionError):
        readers.read_methods(methods, str(methods.summary()))


def test_swapped_simulation_columns_fail_their_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    result = setup.simulation_passing()
    readers.read_simulation(result, str(result.summary()))
    columns = list(_setup.SIMULATION_COLUMNS)
    columns[2], columns[3] = (columns[2][0], *columns[3][1:]), (columns[3][0], *columns[2][1:])
    monkeypatch.setattr(_setup, "SIMULATION_COLUMNS", tuple(columns))
    with pytest.raises(AssertionError):
        readers.read_simulation(result, str(result.summary()))


def test_a_stray_number_fails_the_reader() -> None:
    """A number the object does not hold fails, wherever the summary prints it."""
    study = setup.point_study()
    text = str(study.summary()) + "\nThe study found 17 problems."
    with pytest.raises(AssertionError):
        readers.read_study(study, text)


def test_misprinted_phase_times_fail_their_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = setup.phase_profile()
    readers.read_phases(profile, str(profile.summary()))
    monkeypatch.setattr(_setup, "_seconds", lambda value: _setup.number_text(value * 2, 4))
    with pytest.raises(AssertionError):
        readers.read_phases(profile, str(profile.summary()))


# ----------------------------------------------------------------- what the summaries print


def test_the_simulation_failures_name_no_seed() -> None:
    """The failed-replication section gives the replicate and the error, not its seed."""
    result = setup.simulation_undercovering()
    (failure,) = result.failures
    section = result.summary(detail="full").section(vocabulary.SECTION_FAILURES)
    assert section.lines == (f"replicate {failure.replicate}: LinAlgError: singular matrix",)
    assert str(failure.seed) not in str(result.summary(detail="full"))


def test_a_section_that_printed_the_seed_would_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation of the check above."""
    result = setup.simulation_undercovering()
    (failure,) = result.failures
    original = _setup._failures_section

    def leaky(study: Any) -> Any:
        section = original(study)
        return replace(section, lines=(*section.lines, f"seed {failure.seed}"))

    monkeypatch.setattr(_setup, "_failures_section", leaky)
    assert str(failure.seed) in str(result.summary(detail="full"))


def test_the_undercoverage_reading_names_the_factor() -> None:
    result = setup.simulation_undercovering()
    ratio = result.summaries["ate"].se_ratio
    assert f"is {1 / ratio:.2f}x too small" in result.verdict()
    assert result.verdict() in " ".join(str(result.summary()).split()) or (
        " ".join(result.verdict().split()) in " ".join(str(result.summary()).split())
    )


def test_a_ratio_estimand_says_its_spread_is_on_the_log_scale() -> None:
    summary = setup.simulation_passing().summary()
    assert "rr" in summary.note("spread_scale").text
    assert "spread_scale" not in setup.simulation_undercovering().summary().note_codes


def test_the_protocol_prints_its_fields_in_plain_lines() -> None:
    protocol = setup.protocol()
    lines = _setup.protocol_lines(protocol)
    assert "Eligibility: Discharged alive; Lives in the service area" in lines
    assert not any("[" in line for line in lines)
    full = setup.study_with_protocol().summary(detail="full")
    assert full.section(vocabulary.SECTION_STUDY).lines == lines
    assert dict(full.header)[vocabulary.PROTOCOL] == "recorded"
    assert dict(setup.point_study().summary().header)[vocabulary.PROTOCOL] == "none"


def test_no_protocol_line_names_the_fingerprint() -> None:
    protocol = setup.protocol()
    for line in protocol.summary_lines():
        assert protocol.fingerprint not in line
        assert "schema" not in line
    assert protocol._cache_token() == protocol.fingerprint


def test_a_longitudinal_study_counts_the_rows_followed_to_the_end() -> None:
    study = setup.longitudinal_study()
    header = dict(study.summary().header)
    share = float(np.mean(study.data.uncensored_through(study.data.n_times)))
    assert header[vocabulary.FOLLOWED] == f"{share:.3f}"
    assert header[vocabulary.CENSORING] == "C1, C2"
    roles = {row[0].text: row[1].text for row in study.summary().tables[0].rows}
    assert roles["measured before A2"] == "blood_pressure"


def test_a_randomized_design_says_it_adjusts_for_nothing() -> None:
    frame = make_linear_ate(n=100, seed=4)[0]
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", randomized=True))
    expected = vocabulary.RANDOMIZED_NO_ADJUSTMENT
    assert dict(study.summary().header)[vocabulary.ADJUSTED_FOR] == expected
    effect = study.identify(ATE())
    assert dict(effect.summary().header)[vocabulary.ADJUSTED_FOR] == expected


# ----------------------------------------------------------------- the objects


def test_methods_pickle_and_compare_as_a_tuple() -> None:
    methods = setup.ate_methods()
    copy = pickle.loads(pickle.dumps(methods))
    assert isinstance(copy, MethodAvailabilities)
    assert copy == methods
    assert copy.question == methods.question
    assert tuple(methods) == methods
    assert all(isinstance(record, MethodAvailability) for record in methods)
    frame = methods.to_frame()
    assert list(frame.columns) == ["method", "available", "reason_code", "reason"]
    assert list(frame["method"]) == [record.name for record in methods]


@pytest.mark.parametrize(
    "build",
    [
        lambda: setup.ate_methods()[0],
        lambda: next(record for record in setup.ate_methods() if not record.available),
        lambda: setup.simulation_passing().summaries["ate"],
        lambda: SplitPlan([[0, 1, 0, 1], [1, 0, 1, 0]]),
    ],
)
def test_an_object_without_a_summary_has_a_one_line_repr(build: Any) -> None:
    obj = build()
    text = repr(obj)
    assert "\n" not in text
    assert len(text) <= 100
    assert text.startswith(type(obj).__name__ + "(")


def test_a_split_plan_repr_is_short_for_a_large_plan() -> None:
    labels = np.random.default_rng(0).integers(0, 10, size=(3, 50_000))
    text = repr(SplitPlan(labels))
    assert len(text) <= 100
    assert "n=50000" in text
    assert re.search(r"repeats=3", text)


def test_a_method_that_does_not_fit_names_its_reason_code() -> None:
    record = next(record for record in setup.ate_methods() if not record.available)
    code = record.refusal.reason.value
    assert repr(record) == f"MethodAvailability({record.name}, does not fit: {code})"

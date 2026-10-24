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
from tests.unit.summary_fixtures._setup_questions import QUESTIONS

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
    assert text == "The full summary states 1 further condition the answer rests on."
    plural = setup.mtp_effect().summary().note("further_conditions").text
    assert "2 further conditions the answer" in plural


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


#: Each question's short conditions table, in order: the plain name, the checkable mark,
#: and the closing words of the sentence, which tell the entries apart; then how many
#: statements print only in the full summary.
PINS: dict[str, tuple[tuple[tuple[str, str, str], ...], int]] = {
    "ate": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of each compared level of A."),
        ),
        0,
    ),
    "att": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "had some chance of the reference level."),
        ),
        1,
    ),
    "atc": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "had some chance of each compared level."),
        ),
        1,
    ),
    "mean": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of the evaluated level of A."),
        ),
        0,
    ),
    "natural_course": (
        (),
        2,
    ),
    "incremental": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
        ),
        2,
    ),
    "risk_ratio": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of each compared level of A."),
        ),
        1,
    ),
    "odds_ratio": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of each compared level of A."),
        ),
        1,
    ),
    "par": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of the reference level of A."),
        ),
        1,
    ),
    "paf": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of the reference level of A."),
        ),
        2,
    ),
    "multi_arm_ate": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of each compared level of A."),
        ),
        0,
    ),
    "multi_arm_att": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "had some chance of the reference level."),
        ),
        1,
    ),
    "multi_arm_atc": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "had some chance of each compared level."),
        ),
        1,
    ),
    "dose_policy": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "had some chance of its observed dose."),
        ),
        2,
    ),
    "missing_ate": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("overlap", "partly", "chance of each compared level of A."),
            ("outcome missing at random", "no", "was recorded is unrelated to its value."),
            ("overlap for recorded outcomes", "partly", "had some chance of a recorded Y."),
        ),
        0,
    ),
    "missing_natural_course": (
        (
            ("outcome missing at random", "no", "was recorded is unrelated to its value."),
            ("overlap for recorded outcomes", "partly", "had some chance of a recorded Y."),
        ),
        1,
    ),
    "direct_effect": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "none of them is affected by A."),
            ("no hidden confounding of Z", "no", "to the outcomes they would have had."),
            ("overlap", "partly", "chance of each compared level of A."),
            ("overlap for Z", "partly", "of the value Z is held at."),
            ("outcome missing at random", "no", "was recorded is unrelated to its value."),
            ("overlap for recorded outcomes", "partly", "had some chance of a recorded Y."),
        ),
        1,
    ),
    "randomized_ate": (
        (
            ("well-defined treatment", "no", "the same level has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding", "no", "this, and the data cannot show it."),
            ("overlap", "partly", "of A, as a randomized assignment gives."),
        ),
        0,
    ),
    "uncensored_regime": (
        (
            ("well-defined treatment", "no", "the same treatment has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding at each time", "no", "to the outcomes it would have had."),
            ("overlap at each time", "partly", "chance of the treatment a regimen assigns."),
        ),
        0,
    ),
    "regime": (
        (
            ("well-defined treatment", "no", "the same treatment has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding at each time", "no", "to the outcomes it would have had."),
            ("overlap at each time", "partly", "regimen assigns and of staying in follow-up."),
        ),
        0,
    ),
    "regime_incremental": (
        (
            ("well-defined treatment", "no", "the same treatment has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            ("no hidden confounding at each time", "no", "to the outcomes it would have had."),
            ("follow-up loss at random", "no", "at risk had some chance of staying."),
        ),
        1,
    ),
    "regime_policy": (
        (
            ("well-defined treatment", "no", "the same treatment has the same effect."),
            ("no interference", "no", "treatment does not change another unit's outcome."),
            (
                "no hidden confounding at each time",
                "no",
                "standard form that each causal reading names.",
            ),
            ("overlap at each time", "partly", "had a chance of their observed dose."),
        ),
        1,
    ),
}


def _condition_rows(effect: Any) -> tuple[tuple[str, str, str], ...]:
    summary = effect.summary()
    tables = [table for table in summary.tables if table.title == vocabulary.CONDITIONS_TITLE]
    if not tables:
        return ()
    return tuple(
        (row[0].text, row[1].text, " ".join(row[2].text.split()[-7:])) for row in tables[0].rows
    )


def _further(effect: Any) -> int:
    summary = effect.summary()
    if "further_conditions" not in summary.note_codes:
        return 0
    return int(re.search(r"states (\d+) further", summary.note("further_conditions").text)[1])


def test_the_pins_cover_every_question() -> None:
    assert set(PINS) == set(QUESTIONS)


@pytest.mark.parametrize("name", sorted(PINS))
def test_every_statement_reaches_the_condition_it_names(name: str) -> None:
    effect = QUESTIONS[name]()
    rows, further = PINS[name]
    assert _condition_rows(effect) == rows
    assert _further(effect) == further
    statements = readers.question_statements(effect)
    assert len(rows) + further == len(statements)


def test_every_entry_of_the_condition_words_is_reached() -> None:
    """A witness for the pins: each entry and each variant names a statement some pinned
    question makes, except the entries no shipped question reaches yet."""
    statements = [
        item for build in QUESTIONS.values() for item in readers.question_statements(build())
    ]
    entries = [start for start, _ in vocabulary.CONDITION_WORDS]
    entries += [
        start for variants in vocabulary.CONDITION_VARIANTS.values() for start, _ in variants
    ]
    unreached = {
        start for start in entries if not any(item.startswith(start) for item in statements)
    }
    assert unreached <= UNREACHED, unreached - UNREACHED


#: Entries for statements none of the pinned questions makes: a vector treatment, a tilt,
#: a learned rule and a product of response and treatment overlap.
UNREACHED = {
    "positivity for the evaluated doses:",
    "positivity for the evaluated treatment values:",
    "positivity *for the regime*",
    "positivity for the treated arm wherever the tilt keeps it",
    "product positivity at the reference arm",
}


def test_swapped_condition_sentences_fail_the_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: two entries trade sentences, and the pins see it."""
    words = list(vocabulary.CONDITION_WORDS)
    index = {start: i for i, (start, _) in enumerate(words)}
    first, second = index["positivity: "], index["no unmeasured confounding:"]
    (s1, (n1, t1, c1)), (s2, (n2, t2, c2)) = words[first], words[second]
    words[first], words[second] = (s1, (n1, t2, c1)), (s2, (n2, t1, c2))
    monkeypatch.setattr(vocabulary, "CONDITION_WORDS", tuple(words))
    effect = QUESTIONS["ate"].__wrapped__()
    assert _condition_rows(effect) != PINS["ate"][0]


def test_a_mapping_that_misnames_a_condition_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: swapping the names of two conditions fails the pin above."""
    original = _setup.condition_words

    def swapped(statement: str, features: frozenset[str] = frozenset()) -> Any:
        found = original(statement, features)
        if found is not None and found[0] == "overlap":
            return ("no interference", *found[1:])
        return found

    monkeypatch.setattr(_setup, "condition_words", swapped)
    assert _condition_rows(QUESTIONS["ate"].__wrapped__()) != PINS["ate"][0]


def test_a_dropped_condition_fails_the_question_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate mutation: a builder that loses a statement fails the count the reader
    takes from the identification record."""
    effect = setup.ate_effect()
    readers.read_question(effect, str(effect.summary()))
    original = _setup.conditions

    def dropped(question: Any) -> Any:
        rows, others = original(question)
        return rows[:-1], others

    monkeypatch.setattr(_setup, "conditions", dropped)
    with pytest.raises(AssertionError):
        readers.read_question(effect, str(effect.summary()))


def test_the_full_summary_states_every_statement_verbatim() -> None:
    for name in ("ate", "direct_effect", "regime_policy", "dose_policy"):
        effect = QUESTIONS[name]()
        identification = effect.identification
        lines = effect.summary(detail="full").section(vocabulary.SECTION_ASSUMPTIONS).lines
        expected = [f"- {item}" for item in readers.question_statements(effect)]
        expected += [
            f"- {item.label}: {item.meaning} (needs {'; '.join(item.assumptions)})"
            for item in identification.interpretations
        ]
        expected += [f"- model restriction: {item}" for item in identification.model_restrictions]
        assert list(lines[1:]) == expected, name
        assert effect.functional.expression in lines[0]


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


# ----------------------------------------------------------------- review follow-ups


def test_a_small_detectable_deviation_is_named_not_passed() -> None:
    """A witness: a shortfall and a bias beyond Monte Carlo error but under the floors."""
    result = setup.simulation_small_deviation()
    summary = result.summaries["ate"]
    assert 0.95 - summary.coverage > 3 * summary.coverage_se
    assert abs(summary.bias) > 3 * summary.bias_se
    reading = result.verdict()
    assert reading != vocabulary.SIMULATION_PASS.format(level="95%")
    assert "though by no more than 2 percentage points" in reading
    assert "though no more than 2% of the larger of 1 and the absolute truth" in reading


def test_a_constant_wrong_estimate_is_named_as_biased() -> None:
    """A witness: every replicate returns 1.0 against a truth of 0.5, so the Monte Carlo SD
    is zero and a ratio to it cannot decide whether the bias is detectable."""
    result = setup.simulation_constant()
    summary = result.summaries["ate"]
    assert summary.bias_se == 0.0
    assert summary.bias == pytest.approx(0.5)
    reading = result.verdict()
    assert "ate: every replicate returned the same estimate, so its bias +0.5000" in reading
    assert "within Monte Carlo error of zero" not in reading


def test_a_held_intermediate_names_it_in_the_missingness_condition() -> None:
    """A witness for the held variant: its sentence names the intermediate, which the generic
    sentence does not, and the pins compare only sentence tails."""
    text = str(QUESTIONS["direct_effect"]().summary())
    flat = " ".join(text.split())
    assert "Among units with the same treatment, Z and adjusted-for values, whether Y" in flat
    generic = str(QUESTIONS["missing_ate"]().summary())
    assert "same treatment, Z and" not in " ".join(generic.split())


def test_a_passing_study_says_so() -> None:
    assert setup.simulation_passing().verdict() == vocabulary.SIMULATION_PASS.format(level="95%")


def test_each_mixed_estimand_reads_its_own_statuses() -> None:
    """``ate`` mixes two statuses that both report an interval; ``ey1`` mixes in one that
    reports none. The note gives each its own reading."""
    study = setup._study_result(
        {
            "ate": setup.simulation_record(
                inference="few_cluster_plugin",
                status_counts=(("few_cluster_plugin", 2), ("influence_curve", 2)),
            ),
            "ey1": setup.simulation_record(
                estimand="ey1",
                inference="no_cluster_variance_plugin",
                status_counts=(("no_cluster_variance_plugin", 3), ("influence_curve", 1)),
            ),
        }
    )
    text = " ".join(study.summary().note("mixed_statuses").text.split())
    assert "ate took few_cluster_plugin in 2, influence_curve in 2, and each of its columns " in (
        text
    )
    assert (
        "ate took few_cluster_plugin in 2, influence_curve in 2, and each of its columns reads the interval each replicate reported"
        in text
    )
    assert (
        "ey1 took no_cluster_variance_plugin in 3, influence_curve in 1, and each of its columns reads the no-cluster-variance plug-in diagnostic"
        in text
    )


def test_an_event_time_counts_only_censored_units_as_lost() -> None:
    """A unit after its event is fully observed, so the censored share is below the share
    that is no longer at risk."""
    study = setup.survival_study()
    data = study.data
    times, codes = data.to_time_to_event()
    censored = float(np.mean((codes == 0) & (times < data.time_grid[-1])))
    header = dict(study.summary().header)
    assert header[vocabulary.CENSORED_BEFORE_END] == f"{censored:.3f}"
    assert vocabulary.FOLLOWED not in header
    left = 1 - float(np.mean(data.uncensored_through(data.n_times)))
    assert censored < left - 0.1


def test_a_reshaped_design_names_the_columns_the_caller_declared() -> None:
    roles = {row[0].text: row[1].text for row in setup.long_panel_study().summary().tables[0].rows}
    assert roles["treatment"] == "offer"
    assert roles["measured over time"] == "engaged"
    assert not any("@" in value for value in roles.values())
    survival = {row[0].text: row[1].text for row in setup.survival_study().summary().tables[0].rows}
    assert survival == {"time": "time", "event": "event", "treatment": "A", "baseline": "W1, W2"}


def test_a_column_in_the_wrong_role_fails_the_study_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = setup.point_study()
    readers.read_study(study, str(study.summary()))
    original = _setup.study_roles

    def moved(question: Any) -> Any:
        roles = dict(original(question))
        roles["adjusted for"] = roles["adjusted for"][1:]
        roles["treatment"] = (*roles["treatment"], original(question)[2][1][0])
        return list(roles.items())

    monkeypatch.setattr(_setup, "study_roles", moved)
    with pytest.raises(AssertionError):
        readers.read_study(study, str(study.summary()))


def test_swapped_protocol_versions_fail_the_protocol_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = setup.protocol()
    readers.read_protocol(protocol, str(protocol.summary()))
    swapped = replace(protocol, treatment_versions=tuple(reversed(protocol.treatment_versions)))
    with pytest.raises(AssertionError):
        readers.read_protocol(protocol, str(swapped.summary()))


def test_a_randomized_question_is_answered_by_the_randomization() -> None:
    effect = QUESTIONS["randomized_ate"]()
    header = dict(effect.summary().header)
    assert header[vocabulary.ANSWERED_BY] == vocabulary.ANSWERED_BY_RANDOMIZED
    assert header[vocabulary.ADJUSTED_FOR] == vocabulary.RANDOMIZED_NO_ADJUSTMENT
    readers.read_question(effect, str(effect.summary()))


def test_an_adaptive_iptw_fit_leaves_the_restriction_to_its_own_lines() -> None:
    effect = setup.restricted_effect()
    (restriction,) = effect.identification.model_restrictions
    lines = effect._assumption_lines()
    assert lines[-1] == f"- model restriction: {restriction}"
    without = effect._assumption_lines("adaptive_iptw")
    assert without == lines[:-1]


def test_the_longitudinal_result_lines_keep_their_order_until_that_summary_moves() -> None:
    """``LongitudinalResult.summary`` still reads these lines: the restriction follows the
    assumptions, each reading has its own line, and an unknown detail is refused."""
    effect = setup.markov_effect()
    lines = effect._result_summary_lines()
    kinds = [line.split(":")[0] for line in lines]
    assert kinds.index("identification assumptions") + 1 == kinds.index("model restrictions")
    readings = QUESTIONS["regime_policy"]()._result_summary_lines()
    assert sum(line.startswith("interpretation ") for line in readings) == 2
    with pytest.raises(ValueError, match="detail must be 'short' or 'full'"):
        effect._result_summary_lines(detail="fingerprint")


@pytest.mark.parametrize(
    "build",
    [
        lambda: (
            __import__("tests.unit.summary_fixtures.point_tmle", fromlist=["single_ate"])
            .single_ate()
            .provenance
        ),
        lambda: setup.point_study().data,
        lambda: setup.longitudinal_study().data,
        setup.ate_effect,
    ],
    ids=["provenance", "causal_data", "longitudinal_data", "identified_effect"],
)
def test_a_record_repr_is_one_quiet_line(build: Any) -> None:
    obj = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = repr(obj)
    assert "\n" not in text
    assert len(text) <= 100
    assert text.startswith(type(obj).__name__ + "(")

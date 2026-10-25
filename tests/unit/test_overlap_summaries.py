"""The overlap check prints like every other check (X89 T3).

The contract test (``test_summary_contract.py``) checks width, notes, the repr and every
printed number of each fixture in ``summary_fixtures/overlap.py``. This module checks what is
particular to the overlap check: that the Result and the reading follow the branch the grade
took, with the numbers the grade read; that each note fires where its rule fires and only
there; that the disclaimers stay; that the thresholds come from the named constants; and
that the number reader fails a renderer that moves numbers between rows or columns.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any

import pytest

from cleverly._summary import SummaryTable, _render, vocabulary
from cleverly.sensitivity import positivity
from cleverly.sensitivity._positivity_summary import mechanism_label, mechanism_words

from .summary_fixtures import overlap
from .summary_fixtures._check_fidelity import read_overlap

REPORTS = {name: case.build for name, case in overlap.FIXTURES.items()}


@functools.cache
def _report(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return REPORTS[name]()


def test_every_note_of_the_check_fires() -> None:
    fired = {code for name in REPORTS for code in _report(name).summary().note_codes}
    wanted = set(vocabulary.T3_OVERLAP_NOTE_CODES)
    assert wanted <= fired, wanted - fired


def test_every_branch_of_the_grade_is_reached() -> None:
    branches = {_report(name).grade().branch for name in REPORTS}
    assert branches == set(vocabulary.OVERLAP_RESULTS), branches


def test_the_reading_is_the_verdict() -> None:
    for name in REPORTS:
        report = _report(name)
        assert report.verdict() == report.summary().reading
        assert "VERDICT" not in str(report.summary())


@pytest.mark.parametrize(
    ("name", "tier", "result"),
    [
        ("overlap_plain", "adequate", "at most 1% of units capped"),
        ("overlap_strained", "strain", "1% to 5% of units capped"),
        ("overlap_weak", "serious", "over 5% of units capped"),
        ("overlap_recording_capped", "serious", "another probability capped"),
        ("overlap_recording_concentrated", "adequate", "weights concentrated"),
    ],
)
def test_the_result_names_the_branch_and_the_tier_is_unchanged(
    name: str, tier: str, result: str
) -> None:
    report = _report(name)
    assert report.severity == tier
    assert dict(report.summary().header)[vocabulary.RESULT] == result


def test_the_readings_carry_the_graded_share_and_its_limits() -> None:
    weak = _report("overlap_weak")
    share = f"{weak.truncated['fraction']:.1%} of units were capped at [0.100, 0.900]"
    assert weak.verdict().startswith(share)
    assert "result.diagnostics.truncation_curve()" in weak.verdict()
    strained = _report("overlap_strained").verdict()
    assert strained.startswith("3.0% of units were capped")
    assert "Report result.diagnostics.truncation_curve() beside the estimate" in strained
    plain = _report("overlap_plain").verdict()
    assert plain.startswith("0.0% of units were capped")
    assert "This check grades the cap only" in plain


def test_a_group_cap_above_the_fits_names_the_group_limits() -> None:
    report = _report("overlap_group_limits")
    assert report.grade().group == "att"
    assert report.severity == "serious"
    assert report.verdict().startswith(
        "7.5% of units were capped at [0.025, 0.975], the limits of att"
    )
    # Witness: the same report without the group's cap reads the fit's own limits.
    assert _report("overlap_plain").grade().group is None


def test_a_capped_recording_probability_reads_before_the_treatment() -> None:
    reading = _report("overlap_recording_capped").verdict()
    assert reading.startswith(
        "The probability that the outcome is recorded was capped on 5.0% of the rows at "
        "[0.010, 1.000]."
    )
    assert "result.diagnostics.truncation_curve(mechanism=True)" in reading
    assert reading.endswith("7.5% of units were capped at the treatment limits [0.042, 0.958].")


def test_a_concentrated_joint_probability_is_reported_not_graded() -> None:
    report = _report("overlap_recording_concentrated")
    assert report.severity == "adequate"
    reading = report.verdict()
    assert "leaves an effective 40% of the rows it weights, below the 60% share" in reading
    # Either cap moves a joint row, so the reading names both curves.
    assert vocabulary.CURVE_BOTH in reading


def test_a_joint_rows_cap_is_the_treatments_and_keeps_the_treatment_branch() -> None:
    """A derived row's capped cells are the union over its factors, so they can be all ``g``.

    The control: the same cap on the recording probability's own row takes the
    mechanism branch.
    """
    report = _report("overlap_recording")

    def with_cap(name: str) -> Any:
        return dataclasses.replace(
            report,
            truncated={**report.truncated, "fraction": 0.09},
            mechanisms={
                **report.mechanisms,
                name: {**report.mechanisms[name], "clipped_fraction": 0.2},
            },
        )

    joint = with_cap("P(A=a,Delta=1|W)").grade()
    assert (joint.branch, joint.mechanism) == ("serious", None)
    factor = with_cap("P(Delta=1|A,W)").grade()
    assert (factor.branch, factor.mechanism) == ("mechanism_capped", "P(Delta=1|A,W)")


def test_the_thresholds_are_the_named_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moving a constant moves the tier and the printed words with it."""
    strained = _report("overlap_strained")
    assert strained.severity == "strain"
    monkeypatch.setattr(positivity, "SERIOUS_SHARE", 0.02)
    assert strained.severity == "serious"
    assert dict(strained.summary().header)[vocabulary.RESULT] == "over 2% of units capped"
    monkeypatch.setattr(positivity, "MECHANISM_ESS_SHARE", 0.3)
    assert _report("overlap_recording_concentrated").grade().mechanism is None


def test_the_disclaimers_stay() -> None:
    summary = _report("overlap_recording").summary()
    assert summary.note("effective_share").text.startswith(
        "The smallest arm keeps an effective 90% of its rows after weighting."
    )
    assert "it is not a test, and no threshold applies" in summary.note("effective_share").text
    concentration = summary.note("equation_concentration").text
    assert concentration.startswith(
        "In the estimating equation the weights concentrate most for mean: 108.5 effective "
        "rows of 309 (35%)."
    )
    assert "it is not the estimator's sample size, its precision or a test of overlap" in (
        concentration
    )


def test_the_excluded_estimands_say_what_they_divide_by() -> None:
    summary = _report("overlap_recording").summary()
    text = summary.note("composed_excluded").text
    assert text.startswith(
        "The joint probability of the treatment and a recorded outcome row does not describe "
        "att, which divides by the share treated rather than each row's treatment probability"
    )
    assert text.endswith("Their weights in the estimating equation are in the equation table.")
    # Silent where there is nothing to explain: no joint row at all on a complete-outcome fit.
    assert "composed_excluded" not in _report("overlap_plain").summary().note_codes


def test_the_split_and_the_known_law_are_stated() -> None:
    repeated = _report("overlap_repeated").summary()
    assert dict(repeated.header)[vocabulary.SPLIT_SHOWN] == "split 1 of 3"
    assert repeated.note("first_split").text == (
        "Overlap is shown for split 1 of 3; each split fits its own treatment model."
    )
    known = _report("overlap_known").summary()
    assert dict(known.header)[vocabulary.TREATMENT_MODEL] == "known (declared)"
    assert known.note("group_omitted").text.startswith("No equation row for mean: ")
    plain = _report("overlap_plain").summary()
    assert vocabulary.TREATMENT_MODEL not in dict(plain.header)
    assert {"first_split", "group_omitted", "simplex"}.isdisjoint(plain.note_codes)


def test_the_simplex_note_needs_more_than_two_arms() -> None:
    three = _report("overlap_three_arms")
    assert three.simplex_deviation > 0
    assert "simplex" in three.summary().note_codes
    # Witness: a two-arm report with a planted deviation stays silent, because the
    # complement form keeps two arms on the simplex.
    two = dataclasses.replace(_report("overlap_plain"), simplex_deviation=0.2)
    assert "simplex" not in two.summary().note_codes


def test_the_full_detail_states_the_correction_step_and_the_limits() -> None:
    report = _report("overlap_recording")
    (section,) = report.summary("full").sections
    assert section.title == vocabulary.SECTION_DETAILS
    lines = section.lines
    assert (
        f"Largest weight in the correction step for att: {report.clever_covariate_max['att']:.4g}."
        in lines
    )
    assert "The probability that the outcome is recorded is capped at [0.010, 1.000]." in lines
    assert any(line.startswith("mean: equation h0, ") for line in lines)
    assert report.summary().sections == ()


@pytest.mark.parametrize(
    ("name", "label", "words"),
    [
        ("P(Delta=1|A,W)", "outcome recorded", "probability that the outcome is recorded"),
        (
            "P(A=a,Delta=1|Qbar(a,W))",
            "arm and outcome recorded, given the fitted outcome means",
            "joint probability of the treatment and a recorded outcome, given the fitted "
            "outcome means",
        ),
        (
            "P(A=a,Delta=1|Qbar(a,W),S)",
            "arm and outcome recorded, given the fitted outcome means and the strata",
            "joint probability of the treatment and a recorded outcome, given the fitted "
            "outcome means and the strata",
        ),
        ("P(new)", "P(new)", "P(new)"),
    ],
)
def test_probability_names_read_in_plain_words(name: str, label: str, words: str) -> None:
    assert mechanism_label(name) == label
    assert mechanism_words(name) == words


def test_every_known_probability_has_a_name_and_a_label() -> None:
    names = set(positivity._COMPOSED_ROWS.values()) | {
        "P(Delta=1|A,W)",
        "P(Z=z|A,W)",
        positivity.TREATMENT_OBSERVATION,
    }
    assert names <= set(vocabulary.MECHANISM_NAMES)
    assert set(vocabulary.MECHANISM_NAMES) == set(vocabulary.MECHANISM_LABELS)


def test_the_repr_is_one_line() -> None:
    assert repr(_report("overlap_weak")) == (
        "PositivityReport(400 rows, 204 capped (51.0%), over 5% of units capped)"
    )


# ----------------------------------------------------------------- mutation controls

_ORIGINAL = _render._table_lines


def _swap_rows(table: SummaryTable, width: int) -> Any:
    """The renderer with the numbers of the first two rows of every table swapped."""
    if len(table.rows) >= 2:
        first, second, *rest = table.rows
        rows = ((first[0], *second[1:]), (second[0], *first[1:]), *rest)
        table = dataclasses.replace(table, rows=tuple(rows))
    return _ORIGINAL(table, width)


def _differ(table: SummaryTable) -> bool:
    if len(table.rows) < 2:
        return False
    first, second = table.rows[:2]
    return [cell.value for cell in first[1:]] != [cell.value for cell in second[1:]]


SWAPPABLE = sorted(
    name for name in REPORTS if any(_differ(table) for table in _report(name).summary().tables)
)


@pytest.mark.parametrize("name", SWAPPABLE)
def test_a_renderer_that_swaps_two_rows_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(name)
    read_overlap(report, str(report.summary()))
    monkeypatch.setattr(_render, "_table_lines", _swap_rows)
    with pytest.raises(AssertionError):
        read_overlap(report, str(report.summary()))


def test_the_row_swap_reaches_every_fixture() -> None:
    assert sorted(REPORTS) == SWAPPABLE, set(REPORTS) - set(SWAPPABLE)


@pytest.mark.parametrize("name", sorted(REPORTS))
def test_a_renderer_that_swaps_two_columns_fails_the_reader(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(name)

    def swapped(table: SummaryTable, width: int) -> Any:
        if "top 1%" in table.columns and "top 5%" in table.columns:
            i, j = table.columns.index("top 1%"), table.columns.index("top 5%")
            rows = []
            for row in table.rows:
                cells = list(row)
                cells[i], cells[j] = cells[j], cells[i]
                rows.append(tuple(cells))
            table = dataclasses.replace(table, rows=tuple(rows))
        return _ORIGINAL(table, width)

    monkeypatch.setattr(_render, "_table_lines", swapped)
    with pytest.raises(AssertionError):
        read_overlap(report, str(report.summary()))

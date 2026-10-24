"""The summaries of the SNMM, its model check, the g-formula report and the overlap reports.

The contract test runs every shared check over the fixtures in
``tests/unit/summary_fixtures/longitudinal_records.py``. These tests cover what it cannot:
each conditional note fires on a witness and stays off its control, and each fidelity
reader fails on a misprinted number.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from cleverly._summary import Summary, vocabulary
from tests.unit.summary_fixtures import _record_readers as read
from tests.unit.summary_fixtures import longitudinal_records as fixtures


def _swap_first_number(text: str, old: str, new: str) -> str:
    """Replace the first ``old`` that a table row prints, below a ``-`` rule."""
    lines = text.splitlines()
    rule = next(i for i, line in enumerate(lines) if line and set(line) == {"-"})
    for index in range(rule + 1, len(lines)):
        if old in lines[index]:
            lines[index] = lines[index].replace(old, new, 1)
            return "\n".join(lines)
    raise AssertionError(f"{old!r} is not printed in a table row")


# ----------------------------------------------------------------- readers


@pytest.mark.parametrize(
    ("fixture", "reader", "value"),
    [
        ("regime_report", read.read_support_report, lambda r: f"{r.regimes['all'].max_ratio:.3g}"),
        ("policy_row", read.read_policy_support, lambda r: f"{r.max_ratio:.3g}"),
        ("tilt_row", read.read_tilt_support, lambda r: f"{r.min_propensity:.3g}"),
        (
            "structural_diagnostics",
            read.read_structural_diagnostics,
            lambda r: f"{r.smallest_eigenvalue[0]:.4g}",
        ),
        ("msm_fit", read.read_msm_fit, lambda r: f"{float(r.beta[0]):.4g}"),
        ("check", read.read_check, lambda r: f"{r.statistic:.3f}"),
    ],
)
def test_a_reader_fails_on_a_misprinted_number(fixture: str, reader, value) -> None:  # type: ignore[no-untyped-def]
    """Deliberate mutation: one printed number changed, and the reader must fail."""
    obj = getattr(fixtures, fixture)()
    text = str(obj.summary())
    reader(obj, text)
    shown = value(obj)
    wrong = "9" + shown if not shown.startswith("9") else "1" + shown
    with pytest.raises(AssertionError):
        reader(obj, _swap_first_number(text, f" {shown}", f" {wrong}"))


def test_the_gformula_reader_fails_on_a_misprinted_share() -> None:
    report = fixtures.simulation_set_dose()
    text = str(report.summary())
    read.read_gformula_report(report, text)
    row = report.support[0]
    shown = f"{row.off_support:.1%}"
    with pytest.raises(AssertionError):
        read.read_gformula_report(report, _swap_first_number(text, f" {shown}", " 99.9%"))


# ----------------------------------------------------------------- support


def test_a_regime_with_unsupported_rows_warns_and_a_supported_one_does_not() -> None:
    report = fixtures.regime_report()
    assert "unsupported_rows" not in report.summary().note_codes
    regimes = dict(report.regimes)
    regimes["screen"] = dataclasses.replace(regimes["screen"], unsupported=3)
    witness = dataclasses.replace(report, regimes=regimes).summary()
    assert witness.note("unsupported_rows").text.startswith(
        "Rows given an arm whose estimated probability is exactly zero: screen 3."
    )
    # A structural zero hits every regime that assigns the arm: one note names them all.
    regimes["none"] = dataclasses.replace(regimes["none"], unsupported=4)
    both = dataclasses.replace(report, regimes=regimes).summary()
    assert both.note_codes.count("unsupported_rows") == 1
    assert "zero: none 4, screen 3." in both.note("unsupported_rows").text


def test_a_row_without_a_score_load_drops_the_score_note() -> None:
    row = fixtures.regime_row()
    assert "score_load" in row.summary().note_codes
    bare = dataclasses.replace(row, score_load=None).summary()
    assert "score_load" not in bare.note_codes
    (table,) = bare.tables
    assert table.rows[0][4].text == vocabulary.MISSING


def test_the_natural_course_names_no_factor() -> None:
    row = fixtures.tilt_row()
    assert row.summary().facts["Odds factors"] == "level 1 x0.5"
    natural = dataclasses.replace(row, delta=tuple((level, 1.0) for level, _ in row.delta))
    assert natural.summary().facts["Odds factors"] == "none (the natural course)"


def test_a_policy_prints_a_fold_mean_only_over_several_folds() -> None:
    row = fixtures.policy_row()
    labels = [line[0].value for line in row.summary().tables[0].rows]
    assert not any("fold" in str(label) for label in labels)
    folded = dataclasses.replace(row, fold_mean_ratio=(0.9, 1.1))
    labels = [line[0].text for line in folded.summary().tables[0].rows]
    assert labels[-3:-1] == ["mean ratio, fold 1", "mean ratio, fold 2"]


def test_an_unchecked_density_says_so_in_full_detail() -> None:
    row = fixtures.policy_row()
    assert (
        vocabulary.SUPPORT_NO_ZERO_CHECK
        not in row.summary(detail="full").section("How to read").lines
    )
    unchecked = dataclasses.replace(row, unsupported=None)
    lines = unchecked.summary(detail="full").section("How to read").lines
    assert vocabulary.SUPPORT_NO_ZERO_CHECK in lines


# ----------------------------------------------------------------- g-formula


def test_two_covariates_at_one_time_add_the_draw_order_note() -> None:
    report = fixtures.simulation()
    assert "draw_order" not in report.summary().note_codes
    (node, name, kind), *_ = report.covariate_types
    two = dataclasses.replace(
        report, covariate_types=(*report.covariate_types, (node, name + "b", kind))
    )
    assert "draw_order" in two.summary().note_codes


def test_a_large_simulation_share_adds_its_note() -> None:
    from cleverly.longitudinal.estimator import MONTE_CARLO_WARN_SHARE

    report = fixtures.simulation()
    assert report.max_share <= MONTE_CARLO_WARN_SHARE
    assert "simulation_share" not in report.summary().note_codes
    name = next(iter(report.monte_carlo_share))
    shares = {**report.monte_carlo_share, name: 2 * MONTE_CARLO_WARN_SHARE}
    text = dataclasses.replace(report, monte_carlo_share=shares).summary()
    assert f"({name})" in text.note("simulation_share").text


def test_a_set_dose_prints_its_support_and_its_model_reliance() -> None:
    plain, dosed = fixtures.simulation().summary(), fixtures.simulation_set_dose().summary()
    assert len(plain.tables) == 2 and "model_reliance" not in plain.note_codes
    assert dosed.tables[-1].title == vocabulary.GFORMULA_SUPPORT_TITLE
    assert "model_reliance" in dosed.note_codes


# ----------------------------------------------------------------- structural


def test_the_check_reads_its_test_and_an_untested_check_says_why() -> None:
    check = fixtures.check()
    summary = check.summary()
    assert summary.reading == vocabulary.CHECK_READING
    assert [table.title for table in summary.tables] == [
        vocabulary.CHECK_TABLE_TITLE,
        vocabulary.CHECK_JOINT_TITLE,
    ]
    untested = dataclasses.replace(check, reason="the contrasts are collinear")
    summary = untested.summary()
    assert summary.reading == "No test: the contrasts are collinear"
    assert len(summary.tables) == 1


def test_an_in_sample_snmm_says_every_row_is_experimental() -> None:
    supported = fixtures.snmm().summary()
    assert "experimental" not in supported.note_codes
    text = fixtures.snmm_in_sample().summary().note("experimental").text
    assert "marks every row experimental" in text


def test_every_record_summary_is_a_summary_with_a_short_repr() -> None:
    for build in (
        fixtures.check,
        fixtures.structural_diagnostics,
        fixtures.msm_fit,
        fixtures.simulation,
        fixtures.regime_report,
        fixtures.regime_row,
        fixtures.policy_row,
        fixtures.tilt_row,
    ):
        obj = build()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = obj.summary()
        assert isinstance(summary, Summary)
        assert "\n" not in repr(obj) and len(repr(obj)) <= 100
        with pytest.raises(ValueError):
            obj.summary(detail="long")


def test_the_positivity_record_marks_a_dose_without_an_extreme_share() -> None:
    record = fixtures.structural_diagnostics()
    dose = dataclasses.replace(record, extreme_share=(float("nan"), *record.extreme_share[1:]))
    (table,) = dose.summary().tables
    assert table.rows[0][2].text == vocabulary.MISSING
    assert np.isnan(table.rows[0][2].value)


def test_every_row_replaces_the_list_only_when_it_names_every_row_of_a_long_table() -> None:
    from cleverly._summary._estimates import estimates_table

    experimental = fixtures.snmm_in_sample().estimates
    supported = fixtures.snmm().estimates
    names = list(experimental)
    assert len(names) > 3

    def text(rows: dict) -> str:  # type: ignore[type-arg]
        _, notes = estimates_table(rows, level=0.95)
        return next(item.text for item in notes if item.code == "experimental")

    assert "marks every row experimental" in text(dict(experimental))
    short = {name: experimental[name] for name in names[:3]}
    assert f"marks {', '.join(names[:3])} experimental" in text(short)
    other = next(name for name in supported if name not in names[:4])
    mixed = {**{name: experimental[name] for name in names[:4]}, other: supported[other]}
    assert "every row" not in text(mixed)
    assert f"marks {', '.join(names[:4])} experimental" in text(mixed)


def test_the_split_title_reads_the_draw_the_load_records() -> None:
    row = fixtures.regime_row()
    assert row.score_load is not None
    load = {**row.score_load, "n_repeats": 3, "reported_repeat": 2}
    (table,) = dataclasses.replace(row, score_load=load).summary().tables
    assert table.title.endswith("(score load from split 2 of 3)")


def test_the_split_note_needs_several_splits_and_the_label_needs_an_experimental_check() -> None:
    check = fixtures.check()
    assert "check_splits" in check.summary().note_codes
    assert "experimental" in check.summary().note_codes
    one = dataclasses.replace(check, n_splits=1).summary()
    assert "check_splits" not in one.note_codes
    supported = next(iter(fixtures.snmm().estimates.values())).support
    assert supported.label != vocabulary.EXPERIMENTAL
    assert "experimental" not in dataclasses.replace(check, support=supported).summary().note_codes


def test_a_policy_states_its_zero_density_rows_and_its_declared_support() -> None:
    row = fixtures.policy_row()
    assert row.unsupported == 0 and row.outside_declared_support is None
    facts = row.summary().facts
    assert facts["Rows sent to zero density"] == "0"
    assert "Rows sent outside the declared support" not in facts
    assert "unsupported_rows" not in row.summary().note_codes
    unchecked = dataclasses.replace(row, unsupported=None).summary().facts
    assert unchecked["Rows sent to zero density"] == "not measured"
    declared = dataclasses.replace(row, outside_declared_support=7).summary().facts
    assert declared["Rows sent outside the declared support"] == "7"
    sent = dataclasses.replace(row, unsupported=5).summary()
    assert sent.note("unsupported_rows").text.startswith("up moves 5 rows to a dose")

"""The longitudinal summary reads the fit it describes (X89 T2).

The contract test checks the shape every summary keeps. These tests check what only a
longitudinal summary prints: each row's meaning, the plan table's counts, and the conditional
notes, each with a positive witness that fires it and a fit that does not.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pytest

from cleverly._summary import Summary, SummaryTable, vocabulary
from cleverly._summary._estimates import without_roadmap
from cleverly._summary._render import _table_lines, _wrap_name
from cleverly.longitudinal import SDR, SequentialGComputation
from cleverly.longitudinal._summary import max_truncated, meaning
from tests.unit.summary_fixtures import longitudinal as fixtures


@functools.cache
def _fit(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fixtures.FIXTURES[name]()


def _table(summary: Summary, title: str) -> SummaryTable:
    (table,) = [table for table in summary.tables if (table.title or "").startswith(title)]
    return table


def _column(table: SummaryTable, label: str) -> list[Any]:
    index = table.columns.index(label)
    return [row[index].value for row in table.rows]


# ----------------------------------------------------------------- the object


def test_the_summary_is_a_summary_with_a_one_line_repr() -> None:
    result = _fit("means")
    summary = result.summary()
    assert isinstance(summary, Summary)
    assert summary.heading.strip() == "Longitudinal estimate (LTMLE)"
    assert result._repr_html_() == summary.as_html()
    assert repr(result).startswith("LongitudinalResult(ey_regimen[always]=")
    with pytest.raises(ValueError):
        result.summary(detail="long")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "short"),
    [
        ("means", "LTMLE"),
        ("sdr", "SDR"),
        ("ipsi", "IPSI"),
        ("gcomp", "g-computation"),
        ("ipw", "IPW"),
        ("gformula", "parametric g-formula"),
        ("working_model", "LTMLE, working model"),
    ],
)
def test_the_title_names_the_estimator(name: str, short: str) -> None:
    assert _fit(name).summary().heading.strip() == f"Longitudinal estimate ({short})"


# ----------------------------------------------------------------- what each row estimates


def test_each_row_meaning_names_its_plans_outcome_and_time() -> None:
    result = _fit("means")
    assert meaning(result, "ey_regimen[always]") == "mean Y if every unit followed plan always"
    assert meaning(result, "ate_regimen[always vs never]") == (
        "mean Y if every unit followed plan always, minus that mean under plan never"
    )
    survival = _fit("survival")
    assert meaning(survival, "risk_regimen[never @ t=2]") == (
        "probability of the event by time 2 if every unit followed plan never"
    )
    competing = _fit("competing")
    assert meaning(competing, "cif_regimen[always, death @ t=1]") == (
        "probability of death by time 1 if every unit followed plan always"
    )
    repeated = _fit("repeated_outcome")
    name = next(name for name in repeated.estimates if name.endswith("@ t=3]"))
    assert meaning(repeated, name).startswith("mean Y3 if every unit followed plan")
    model = _fit("working_model")
    assert meaning(model, "msm_regimen[duration]") == (
        "coefficient of duration in the working model of the plan means"
    )


def test_the_key_table_holds_every_row_and_a_single_row_has_none() -> None:
    result = _fit("means")
    key = _table(result.summary(), vocabulary.KEY_TITLE)
    assert [row[0].value for row in key.rows] == list(result.estimates)
    single = _fit("contrast").summary()
    assert not [table for table in single.tables if table.title == vocabulary.KEY_TITLE]
    assert dict(single.header)[vocabulary.QUESTION] == (
        "average effect on Y of plan always instead of never"
    )


def test_censoring_and_competing_causes_are_stated_once_in_the_header() -> None:
    header = dict(_fit("competing").summary().header)
    assert header[vocabulary.CENSORING] == "C1, C2; each plan keeps every unit observed"
    assert header[vocabulary.OUTCOME].endswith("(competing causes)")
    for row in _table(_fit("competing").summary(), vocabulary.KEY_TITLE).rows:
        assert "observation" not in row[1].text
        assert "competing" not in row[1].text


# ----------------------------------------------------------------- the plan table


def test_the_plan_table_counts_the_units_that_followed_each_plan() -> None:
    result = _fit("means")
    table = _table(result.summary(), vocabulary.PLANS_TITLE)
    deepest = max(fit.horizon for fit in result.fits.values())
    fits = {fit.regimen.label: fit for fit in result.fits.values() if fit.horizon == deepest}
    assert _column(table, vocabulary.PLAN) == list(fits)
    followed = _column(table, vocabulary.IN_PLAN)
    assert followed == [int(fit.scored_nodes[-1].trained_on.sum()) for fit in fits.values()]
    assert _column(table, vocabulary.LARGEST_WEIGHT) == [fit.max_weight for fit in fits.values()]
    assert _column(table, vocabulary.ASSIGNS) == [
        "1 at every time point",
        "0 at every time point",
    ]


def test_a_rule_plan_and_a_tilt_plan_print_in_words() -> None:
    rule = _table(_fit("rule").summary(), vocabulary.PLANS_TITLE)
    assert "1 at time 1; a declared rule at time 2" in _column(rule, vocabulary.ASSIGNS)
    tilt = _table(_fit("ipsi").summary(), vocabulary.PLANS_TITLE)
    assert vocabulary.OBSERVED_TO_END in tilt.columns
    assert "odds of treatment times 2 at every time point" in _column(tilt, vocabulary.ASSIGNS)


@pytest.mark.parametrize("name", ["gcomp", "gformula"])
def test_an_estimate_that_reads_no_weight_prints_none(name: str) -> None:
    """G-computation and the g-formula fit treatment models for the overlap check only."""
    summary = _fit(name).summary()
    table = _table(summary, vocabulary.PLANS_TITLE)
    assert vocabulary.LARGEST_WEIGHT not in table.columns
    assert vocabulary.EFFECTIVE_N not in table.columns
    assert vocabulary.PROBABILITY_LIMITS not in dict(summary.header)


def test_a_repeated_fit_says_which_split_the_plan_table_reads() -> None:
    table = _table(_fit("repeats").summary(), vocabulary.PLANS_TITLE)
    assert table.title == vocabulary.PLANS_SPLIT_TITLE.format(repeats=3)


# ----------------------------------------------------------------- conditional notes


def test_the_capped_note_fires_with_the_share_the_limit_moved() -> None:
    capped = _fit("capped")
    item = capped.summary().note("weights_capped")
    shares = max_truncated(capped)
    assert shares
    for plan, share, time in shares:
        assert f"plan {plan} through time {time} to [0.2, 1]" in item.text
        assert f"{share:.1%}" in item.text
    # The default limits on the same data move nothing the note would report.
    assert "weights_capped" not in _fit("contrast").summary().note_codes


def test_the_capped_note_reads_the_path_share_on_sdr() -> None:
    frame = fixtures._frame()
    result = SDR(fixtures.PLANS, reference="never", g_bounds=(0.2, 1.0), **fixtures.learners()).fit(
        frame, **fixtures.COLUMNS
    )
    item = result.summary().note("weights_capped")
    assert "followed paths of plan" in item.text
    binding = max(
        step.bound_binding
        for fit in result.fits.values()
        for step in fit.steps
        if step.bound_binding is not None
    )
    assert f"{binding:.1%}" in item.text


def test_g_computation_never_prints_the_capped_note() -> None:
    """Witness for M1: a high limit moves only the overlap check of g-computation."""
    result = SequentialGComputation(
        fixtures.PLANS, reference="never", g_bounds=(0.2, 1.0), outcome_learner=_glm()
    ).fit(fixtures._frame(), **fixtures.COLUMNS)
    assert "weights_capped" not in result.summary().note_codes
    assert "model_based_interval" in result.summary().note_codes


def _glm() -> Any:
    from cleverly.learners import GLM

    return GLM()


def test_an_unsolved_equation_says_the_estimate_may_be_off() -> None:
    item = _fit("unsolved").summary().note("score_unsolved")
    assert item.kind == "warning"
    assert "the estimate and its interval may be off" in item.text
    assert "score_unsolved" not in _fit("means").summary().note_codes


def test_the_in_sample_note_needs_a_flexible_learner() -> None:
    assert "in_sample_flexible" in _fit("flexible_in_sample").summary().note_codes
    assert "in_sample_flexible" not in _fit("contrast").summary().note_codes
    assert "in_sample_flexible" not in _fit("means").summary().note_codes


def test_the_working_model_note_names_its_terms() -> None:
    item = _fit("working_model").summary().note("working_model")
    assert "(intercept), duration" in item.text


# ----------------------------------------------------------------- full detail


def test_full_detail_adds_assumptions_settings_method_and_study() -> None:
    full = _fit("means").summary(detail="full")
    assert [section.title for section in full.sections] == [
        vocabulary.SECTION_ASSUMPTIONS,
        vocabulary.SECTION_SETTINGS,
        vocabulary.SECTION_METHOD,
        vocabulary.SECTION_STUDY,
    ]
    settings = full.section(vocabulary.SECTION_SETTINGS).lines
    assert settings[0].startswith("Probability limits [0.01, 1] (the default)")
    assert "Folds: 2." in settings


def test_a_rule_plan_says_its_counts_describe_this_sample() -> None:
    lines = _fit("rule").summary(detail="full").section(vocabulary.SECTION_SETTINGS).lines
    assert vocabulary.RULE_FOLLOWERS in lines


# ----------------------------------------------------------------- shared layer


def test_a_long_row_name_wraps_at_a_space_or_after_a_bracket() -> None:
    assert _wrap_name("ate_regimen[always vs never @ t=1]", 29) == [
        "ate_regimen[always vs never",
        "@ t=1]",
    ]
    assert _wrap_name("ey_regimen[always]", 12) == ["ey_regimen[", "always]"]
    table = _fit("competing").summary().table("estimates")
    head, body = _table_lines(table, Summary.WIDTH)
    assert max(len(line) for line in [*head, *body]) <= Summary.WIDTH
    assert len(body) > len(table.rows)


def test_a_roadmap_pointer_in_a_warning_leaves_the_summary() -> None:
    assert without_roadmap("a diagnostic only (roadmap X20). Read") == "a diagnostic only. Read"
    assert without_roadmap("x (roadmap X57 part d) y") == "x y"
    assert without_roadmap("column (X1) stays") == "column stays"
    text = str(_fit("means").summary())
    assert "roadmap" not in text


def test_the_header_counts_the_units_and_the_time_points() -> None:
    result = _fit("survival")
    header = dict(result.summary().header)
    assert header[vocabulary.ROWS] == f"{result.n:,}"
    assert header[vocabulary.TIME_POINTS] == str(result.data.n_times)
    assert np.isclose(float(header[vocabulary.PROBABILITY_LIMITS].strip("[]").split(",")[0]), 0.01)

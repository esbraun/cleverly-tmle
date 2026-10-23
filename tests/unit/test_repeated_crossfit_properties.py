"""The ``repeat_stability`` family of ``repeated-crossfit-tmle``, checked without the study.

Each trial of the family fits the three-draw median and draws 0 and 1 alone on one fixed sample.
``three_repeats`` and its base arm ``one_repeat`` are positive rows that publish the ratio
``SD(three_repeats) / SD(one_repeat)`` and pass when its 99% upper endpoint is below 1.
``second_draw_control`` is the control: it publishes ``SD(second_draw_control) / SD(one_repeat)``
and passes when its upper endpoint is at or above 1.  The tests below hold both rules to nonzero
witnesses, because a control that reads the positive's interval cannot fail on its own.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import CrossFitPlan
from tests.studies import repeated_crossfit as study_module
from tests.studies import repeated_crossfit_properties as properties
from tests.studies.evidence import property_failures
from tests.studies.evidence.seeds import stream_seed

STUDY = study_module.STUDY
ARMS = [cell for cell, _ in properties.STABILITY_ARMS]
SUMMARY_COLUMNS = pd.Index(
    [
        "property",
        "cell",
        "role",
        "n",
        "replicates",
        "failed_replicates",
        "bias",
        "spread_ratio",
        "spread_ratio_ci_lower",
        "spread_ratio_ci_upper",
        "spread_ratio_boundary",
        "passed",
        "property_passed",
    ]
)


def _rows(
    three: np.ndarray, one: np.ndarray, second: np.ndarray, *, lost: tuple[int, ...] = ()
) -> pd.DataFrame:
    """The three arms' replicate rows over ``len(one)`` requested trials, ``lost`` ones missing."""
    records = []
    requested = len(one)
    for (cell, role), estimates in zip(
        properties.STABILITY_ARMS, (three, one, second), strict=True
    ):
        for trial, value in enumerate(estimates):
            if trial in lost:
                continue
            records.append(
                {
                    "property": "repeat_stability",
                    "cell": cell,
                    "role": role,
                    "replicate": trial,
                    "n": 1_000,
                    "requested_replicates": requested,
                    "failed_replicates": len(lost),
                    "truth": 0.0,
                    "estimate": float(value),
                    "std_error": 0.01,
                    "covered": 1,
                    "rejected": 0,
                }
            )
    return pd.DataFrame(records)


def _draws(trials: int = 80) -> np.ndarray:
    return np.random.default_rng(3).normal(0.0, 1e-2, trials)


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    return properties._summarize_repeat_stability(rows, SUMMARY_COLUMNS).set_index("cell")


def test_the_family_declares_three_cells_with_the_declared_roles() -> None:
    cells = {
        cell.cell: cell.role
        for cell in properties.declared_cells()
        if cell.property == "repeat_stability"
    }
    assert cells == {
        "three_repeats": "positive",
        "one_repeat": "positive",
        "second_draw_control": "control",
    }
    assert study_module.PROPERTY_CELLS["repeat_stability"] == tuple(cells)
    assert tuple(f"repeat_stability/{cell}" for cell in cells) == properties.STABILITY_CELLS
    assert set(study_module.CONFIGURATION["repeat_stability"]["seed_streams"]) >= {
        "bootstrap",
        "bootstrap/second_draw",
    }


def test_a_second_draw_with_the_medians_spread_turns_the_control_red() -> None:
    """The witness that the control can fail: draw 1 reads the median's spread (ratio 0.5)."""
    one = _draws()
    rng = np.random.default_rng(4)
    summary = _summary(_rows(0.5 * one, one, rng.normal(0.0, 0.5e-2, len(one))))
    assert summary.loc["three_repeats", "spread_ratio_ci_upper"] < 1.0
    assert summary.loc["second_draw_control", "spread_ratio_ci_upper"] < 1.0
    assert summary.loc["three_repeats", "passed"] and summary.loc["one_repeat", "passed"]
    assert not summary.loc["second_draw_control", "passed"]
    assert not summary["property_passed"].any()


def test_a_second_draw_with_the_first_draws_spread_keeps_every_row_green() -> None:
    one = _draws()
    summary = _summary(_rows(0.5 * one, one, one[::-1].copy()))
    control = summary.loc["second_draw_control"]
    assert control["spread_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert control["spread_ratio_ci_lower"] < 1.0 <= control["spread_ratio_ci_upper"]
    assert summary["passed"].all() and summary["property_passed"].all()


def test_a_median_that_equals_draw_0_turns_the_positive_red() -> None:
    """The deliberate mutation: no repeat layer means the median is draw 0 and the ratio is 1."""
    one = _draws()
    summary = _summary(_rows(one.copy(), one, one[::-1].copy()))
    assert summary.loc["three_repeats", "spread_ratio_ci_upper"] == 1.0
    assert not summary.loc["three_repeats", "passed"]
    assert not summary.loc["one_repeat", "passed"]
    assert summary.loc["second_draw_control", "passed"]
    assert not summary["property_passed"].any()


def test_each_arm_publishes_its_own_ratio_and_the_base_arm_the_pair_s() -> None:
    one = _draws()
    summary = _summary(_rows(0.5 * one, one, one[::-1].copy()))
    for column in ("spread_ratio", "spread_ratio_ci_lower", "spread_ratio_ci_upper"):
        assert summary.loc["one_repeat", column] == summary.loc["three_repeats", column]
    assert summary.loc["three_repeats", "spread_ratio"] == pytest.approx(0.5)
    assert summary.loc["second_draw_control", "spread_ratio"] == pytest.approx(1.0)
    assert set(summary["spread_ratio_boundary"]) == {study_module.MAX_REPEAT_SPREAD_RATIO}
    assert summary.loc["three_repeats", "bias"] != summary.loc["three_repeats", "bias"]


def test_the_control_resamples_with_its_own_stream() -> None:
    positive = stream_seed(STUDY, "repeat_stability", "bootstrap")
    control = stream_seed(STUDY, "repeat_stability", "bootstrap", *properties.CONTROL_STREAM)
    assert positive != control
    assert positive == properties._bootstrap_seed()
    assert control == properties._bootstrap_seed(*properties.CONTROL_STREAM)


def _context(rows: pd.DataFrame, summary: pd.DataFrame) -> property_failures.Context:
    return property_failures.Context(STUDY, rows, summary.reset_index(), {}, {})


def test_the_declared_rule_reads_the_published_verdicts() -> None:
    """``rule_verdicts`` recomputes ``passed`` and the family's joint verdict from the columns."""
    one = _draws()
    rng = np.random.default_rng(4)
    rule = property_failures.rule_for("repeat_stability", properties)
    assert rule is not None and rule.group == "family"
    for second in (rng.normal(0.0, 0.5e-2, len(one)), one[::-1].copy()):
        rows = _rows(0.5 * one, one, second)
        summary = _summary(rows)
        read = property_failures.rule_verdicts(rule, summary.reset_index(), _context(rows, summary))
        read = read.set_index("cell")
        assert (read["passed"].astype(bool) == summary["passed"].astype(bool)).all()
        assert (
            read["property_passed"].astype(bool) == summary["property_passed"].astype(bool)
        ).all()


def test_the_lost_trials_of_the_control_are_filled_to_lower_its_ratio() -> None:
    """The control clause lowers the ratio; the positive clause raises it."""
    one = _draws(200)
    rows = _rows(0.5 * one, one, one[::-1].copy(), lost=(3, 4, 5, 6))
    summary = _summary(rows)
    context = _context(rows, summary)
    rule = property_failures.rule_for("repeat_stability", properties)
    assert rule is not None
    control_clause = rule.clause("control spread ratio reaches its boundary")
    positive_clause = rule.clause("repeat spread ratio under its boundary")
    frame = summary.reset_index()
    control_row = frame.loc[frame["cell"] == "second_draw_control"].iloc[0]
    positive_row = frame.loc[frame["cell"] == "three_repeats"].iloc[0]
    complete = control_clause.read(context, control_row)
    candidates = control_clause.candidates(context, control_row)
    assert len(candidates) == 1 + 4
    assert all(judged.slack <= complete.slack + 1e-12 for _, judged in candidates)
    assert min(judged.slack for _, judged in candidates) < complete.slack
    complete = positive_clause.read(context, positive_row)
    raised = positive_clause.candidates(context, positive_row)
    assert all(judged.slack <= complete.slack + 1e-12 for _, judged in raised)
    assert min(judged.slack for _, judged in raised) < complete.slack
    applying = {
        str(row["cell"]): {c.name for c in rule.clauses if c.applies(context, row)}
        for _, row in frame.iterrows()
    }
    assert applying["three_repeats"] == {positive_clause.name}
    assert applying["one_repeat"] == {positive_clause.name}
    assert applying["second_draw_control"] == {control_clause.name}


def test_with_nothing_lost_the_candidates_are_the_complete_case() -> None:
    one = _draws(200)
    rows = _rows(0.5 * one, one, one[::-1].copy())
    summary = _summary(rows)
    context = _context(rows, summary)
    rule = property_failures.rule_for("repeat_stability", properties)
    assert rule is not None
    for _, row in summary.reset_index().iterrows():
        for clause in rule.clauses:
            if not clause.applies(context, row):
                continue
            complete = clause.read(context, row)
            for _, judged in clause.candidates(context, row):
                assert judged.passed == complete.passed
                assert judged.slack == pytest.approx(complete.slack, abs=1e-9)


# ------------------------------------------------------------------- the trial itself


def _frame() -> pd.DataFrame:
    frame, _ = study_module.draw_from_seed("binary", 400, 99)
    return frame


@pytest.fixture(scope="module")
def trial() -> tuple[pd.DataFrame, tuple[dict[str, object], ...], int]:
    frame = _frame()
    base = 12345
    return frame, properties._stability_trial(frame, 0.0, 0, 400, base), base


def test_a_trial_publishes_the_three_arms(
    trial: tuple[pd.DataFrame, tuple[dict[str, object], ...], int],
) -> None:
    _, rows, _ = trial
    assert [row["cell"] for row in rows] == ARMS
    assert [row["role"] for row in rows] == ["positive", "positive", "control"]
    # The median of three draws is one of them, so only the two single draws must differ.
    assert rows[1]["estimate"] != rows[2]["estimate"]


def test_the_second_draw_row_is_a_one_repeat_refit_at_the_second_seed(
    trial: tuple[pd.DataFrame, tuple[dict[str, object], ...], int],
) -> None:
    """Nonzero witness: the control row holds draw 1, not draw 0 and not the median."""
    frame, rows, base = trial
    seeds = CrossFitPlan(
        n_folds=study_module.N_FOLDS, random_state=base, repeats=study_module.REPEATS
    ).seeds()
    refit = properties._stability_fit(frame, repeats=1, random_state=seeds[1])
    repeated = properties._stability_fit(frame, repeats=study_module.REPEATS, random_state=base)
    by_cell = {row["cell"]: row for row in rows}
    assert by_cell["second_draw_control"]["estimate"] == refit["ate"].psi
    assert by_cell["second_draw_control"]["estimate"] == repeated.repeats[1].psi["ate"]
    assert by_cell["second_draw_control"]["estimate"] != repeated.repeats[0].psi["ate"]
    assert by_cell["one_repeat"]["estimate"] == repeated.repeats[0].psi["ate"]
    assert by_cell["three_repeats"]["estimate"] == repeated["ate"].psi
    assert properties._repeat_seeds(base) == (seeds[0], seeds[1])


def test_a_trial_whose_draws_share_a_split_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collapse refusal: draw 1 with draw 0's fold assignment would make the control draw 0."""
    same = np.arange(400) % 5
    draw = SimpleNamespace(nuisance=SimpleNamespace(folds=SimpleNamespace(assignment=same)))
    fake = SimpleNamespace(repeats=[draw, draw, draw])
    monkeypatch.setattr(properties, "_stability_fit", lambda *args, **kwargs: fake)
    with pytest.raises(RuntimeError, match="share one fold assignment"):
        properties._stability_trial(_frame(), 0.0, 0, 400, 12345)


def test_a_one_repeat_refit_that_is_not_the_draw_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-trial equality checks refuse a refit at a seed that is not the draw's."""
    original = properties._stability_fit

    def shifted(frame: pd.DataFrame, *, repeats: int, random_state: int) -> object:
        return original(frame, repeats=repeats, random_state=random_state + (repeats == 1))

    monkeypatch.setattr(properties, "_stability_fit", shifted)
    with pytest.raises(RuntimeError, match="is not the repeated fit's draw"):
        properties._stability_trial(_frame(), 0.0, 0, 400, 12345)

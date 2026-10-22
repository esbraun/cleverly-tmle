"""The ``longitudinal-repeated-crossfit`` drivers, checked without running the study.

One primary replication of every scenario at n = 2,000 checks the row contract the shared
driver publishes.  Two property fits under a checkpoint root check that a part run in chunks
resumes: a second call reuses every replicate the first one finished and returns the same rows.
"""

from __future__ import annotations

import gzip
import io
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleverly.learners.crossfit import CrossFitPlan
from tests.canonical.longitudinal_repeated_crossfit import inherit, part
from tests.canonical.regenerate import digest, run_key, runs_root
from tests.studies import longitudinal_repeated_crossfit as study
from tests.studies import longitudinal_repeated_crossfit_properties as properties
from tests.studies.evidence import property_failures, property_verdicts, replication
from tests.studies.evidence.manifest import write_rows
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import stream_seed

pytestmark = pytest.mark.xdist_group("longitudinal_repeated_crossfit_study")


def test_the_primary_driver_returns_the_replication_contract() -> None:
    result = study.draw_and_fit(replicates=1, n=2_000, n_jobs=1)
    assert isinstance(result, replication.PhaseResult)
    rows = result.rows
    assert tuple(rows.columns) == REPLICATE_COLUMNS
    assert set(rows["implementation"]) == {study.STUDY.implementation}
    assert set(rows["replicate"]) == {0} and set(rows["n"]) == {2_000}
    for scenario, names in study.SCENARIOS.items():
        subset = rows.loc[rows["scenario"] == scenario]
        assert list(subset["estimand"]) == list(names)
    assert np.isfinite(rows[["estimate", "std_error"]].to_numpy()).all()
    recomputed = (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
    assert (recomputed.astype(int) == rows["covered"]).all()
    ratio = rows.loc[rows["estimand"] == study.D_RATIO]
    assert set(ratio["inference_scale"]) == {"log"}


def test_a_part_resumes_from_its_checkpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A chunk of one replicate, then a chunk of two: the second fits only the new one."""
    with replication.checkpoints(tmp_path):
        first = properties.fit_part("rate_2000", n_jobs=1, first=1)
    capsys.readouterr()
    with replication.checkpoints(tmp_path):
        second = properties.fit_part("rate_2000", n_jobs=1, first=2)
    assert "reusing 1 checkpointed replicates" in capsys.readouterr().out
    head = pd.DataFrame(second)
    head = head.loc[head["replicate"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(head, pd.DataFrame(first))
    assert set(pd.DataFrame(second)["requested_replicates"]) == {properties.RATE_REPLICATES}


def test_an_inherited_part_is_reused_without_a_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committed rows, written as checkpoints, are the rows a declared run reuses unfitted."""
    law, scenarios, n, _ = properties.FIT_SETS["rate_2000"]
    monkeypatch.setitem(properties.FIT_SETS, "rate_2000", (law, scenarios, n, 2))
    with replication.checkpoints(tmp_path / "fitted"):
        fitted = pd.DataFrame(properties.fit_part("rate_2000", n_jobs=1))
    capsys.readouterr()
    committed = pd.read_csv(io.StringIO(fitted.to_csv(index=False)), float_precision="round_trip")
    assert inherit.inherit_part("rate_2000", committed, tmp_path / "inherited") == 2
    with replication.checkpoints(tmp_path / "inherited"):
        reused = pd.DataFrame(properties.fit_part("rate_2000", n_jobs=1))
    assert "reusing 2 checkpointed replicates" in capsys.readouterr().out
    pd.testing.assert_frame_equal(reused, fitted)
    with pytest.raises(RuntimeError, match="does not hold replicates"):
        inherit.inherit_part("rate_2000", committed.iloc[:-1], tmp_path / "short")


def test_the_inherited_and_fitted_parts_split_the_plan() -> None:
    """The X79 rerun fits the two stability laws, and inherits everything else."""
    assert inherit.FITTED == ("stability_d", "stability_i")
    assert set(inherit.FITTED) | set(inherit.INHERITED) == set(properties.plan())
    assert not set(inherit.FITTED) & set(inherit.INHERITED)


def test_every_part_is_a_declared_fit_set() -> None:
    assert ("primary", *properties.FIT_SETS, "stability_d", "stability_i") == part.PARTS
    with pytest.raises(ValueError, match="at least 1"):
        properties.fit_part("rate_2000", n_jobs=1, first=0)


def test_a_part_writes_into_the_declared_run_scratch() -> None:
    """The driver keys a declared run by the record at the declared count and size."""
    record = study.STUDY
    expected = runs_root() / record.slug / digest(run_key(record))[:12]
    assert part.scratch()[0] == expected


# ------------------------------------------------------- failure accounting (RM39b2)


def _row(cell: str, replicate: int, requested: int, estimate: float = 0.1) -> dict[str, object]:
    return {
        "property": "ipsi_primary",
        "cell": cell,
        "role": "positive",
        "replicate": replicate,
        "n": 100,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": estimate,
        "std_error": 0.1,
        "covered": int(abs(estimate) <= 0.196),
        "rejected": int(abs(estimate) > 0.196),
    }


#: The replicates the stubbed fit loses: one raises, one returns a non-finite estimate.
_RAISES, _NON_FINITE = 1, 2


def _flaky(key: tuple[str, int, int], frame: object) -> list[dict[str, object]]:
    """Every ``ipsi_primary`` cell of one key; two replicates fail in the two recorded ways."""
    del frame
    _, replicate, requested = key
    if replicate == _RAISES:
        raise np.linalg.LinAlgError("singular matrix")
    estimate = 0.1 * float(np.sin(replicate + 1.0))
    if replicate == _NON_FINITE:
        estimate = float("nan")
    return [
        _row(f"{label}__repeated", replicate, requested, estimate)
        for label in study.IPSI_PRIMARY_LABELS
    ]


def _stub(monkeypatch: pytest.MonkeyPatch, worker: object = _flaky) -> None:
    monkeypatch.setattr(properties, "fit_rows", worker)
    monkeypatch.setattr(properties, "draw", lambda key: None)


def test_a_recorded_failure_costs_its_cells_and_replays_on_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A part records a failure, keeps going, and a resume replays the checkpointed record."""
    _stub(monkeypatch)
    for attempt in range(2):
        with replication.checkpoints(tmp_path):
            rows = pd.DataFrame(properties.fit_part("ipsi_primary", n_jobs=1, first=6))
        out = capsys.readouterr().out
        assert sorted(set(rows["replicate"])) == [0, 3, 4, 5]
        # The driver is the one writer of the stamp; the loop leaves it at zero.
        assert set(rows["failed_replicates"]) == {0}
        assert "2 recorded property failures" in out
        assert ("reusing 6 checkpointed replicates" in out) == (attempt == 1)
    checkpoint = tmp_path / study.STUDY.implementation / f"properties/ipsi_primary_r{1_600}"
    assert checkpoint.is_dir()


def _accounted(
    monkeypatch: pytest.MonkeyPatch, lost: set[int], requested: int = 200
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The driver's property steps on ``ipsi_primary`` rows whose fits ``lost`` raised."""

    def worker(key: tuple[str, int, int], frame: object) -> list[dict[str, object]]:
        _, replicate, requested = key
        if replicate in lost:
            raise np.linalg.LinAlgError("singular matrix")
        estimate = 0.1 * float(np.sin(replicate + 1.0))
        return [
            _row(f"{label}__repeated", replicate, requested, estimate)
            for label in study.IPSI_PRIMARY_LABELS
        ]

    _stub(monkeypatch, worker)
    # The table holds one family, so the summary fits no size ladder.
    monkeypatch.setattr(study, "RATE_LABELS", ())
    keys = [("ipsi_primary", replicate, requested) for replicate in range(requested)]
    with replication.recording(study.STUDY.implementation) as recorded:
        rows = pd.DataFrame(properties._fit("ipsi_primary", keys, 1))
    records = property_failures.frame(recorded)
    rows = property_failures.stamp(rows, records)
    property_failures.reconcile(rows, records, derived=properties.DERIVED_CELLS)
    summary = properties.summarize_properties(rows)
    accounting = property_failures.property_accounting(
        rows, records, summary, record=study.STUDY, module=properties, n_jobs=1
    )
    corrected = property_failures.correct_properties(summary, accounting, module=properties)
    return corrected.set_index("cell"), accounting


def test_the_driver_reads_a_lost_replicate_under_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At ``floor(0.01 R)`` losses each clause is read at its worst case; one more is red."""
    assert property_failures.allowed(properties.FAILURE_BOUND, 200) == 2
    within, accounting = _accounted(monkeypatch, {7, 8})
    assert set(within["failed_replicates"]) == {2}
    assert set(within["replicates"]) == {198}
    legs = set(accounting["leg"])
    assert {
        property_failures.BOUND_LEG,
        "bias equivalent",
        "coverage floor",
        "se ratio in sanity band",
    } <= legs
    assert (accounting["allowed"] == 2).all()
    assert not (accounting["reason"] == property_failures.BOUND_EXCEEDED).any()
    exceeded, accounting = _accounted(monkeypatch, {7, 8, 9})
    assert set(exceeded["failed_replicates"]) == {3}
    assert (accounting["reason"] == property_failures.BOUND_EXCEEDED).any()
    assert not exceeded["passed"].any() and not exceeded["property_passed"].any()


def test_the_rules_are_the_ones_the_summary_applies() -> None:
    """Every gated family resolves to a rule; the reported ones are diagnostic rows."""
    families = {cell.property for cell in properties.declared_cells()}
    for family in families - {"single_draw_reading", "split_displacement"}:
        assert property_failures.rule_for(family, properties) is not None, family
    roles = {
        (cell.property, cell.role)
        for cell in properties.declared_cells()
        if cell.property in {"single_draw_reading", "split_displacement"}
    }
    assert {role for _, role in roles} == {property_verdicts.DIAGNOSTIC_ROLE}


# ---------------------------------------------------- repeat stability (X79)


def _stability_frame(
    arms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The replicate rows and a bare summary of ``label -> (three, one, second)`` estimates."""
    rows, cells = [], []
    for label, values in arms.items():
        for (cell, _), estimates in zip(properties.STABILITY_ARMS, values, strict=True):
            cells.append(f"{label}__{cell}")
            for trial, value in enumerate(estimates):
                row = _row(f"{label}__{cell}", trial, len(estimates), estimate=float(value))
                row.update(property="repeat_stability")
                rows.append(row)
    summary = pd.DataFrame({"property": "repeat_stability", "cell": cells})
    return pd.DataFrame(rows), summary


def _stability_arms(
    *,
    three: float,
    second: float,
    trials: int = 60,
    rounding: bool = True,
    reverse: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Every label with one-draw spread 1e-2, the median at ``three`` times draw 0.

    Draw 1 is an independent draw of spread ``second`` times draw 0's, or with ``reverse`` draw
    0's own values in reverse order times ``second``, whose spread ratio is exactly ``second``.
    Law I's label sits at rounding level (1e-9) when ``rounding`` holds.
    """
    rng = np.random.default_rng(0)
    out = {}
    for label in properties.STABILITY:
        scale = 1e-9 if rounding and label == "i_contrast_x2" else 1e-2
        one = rng.normal(0.0, scale, trials)
        independent = rng.normal(0.0, second * scale, trials)
        out[label] = (three * one, one, second * one[::-1] if reverse else independent)
    return out


def _verdicts(arms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows, summary = _stability_frame(arms)
    properties.stability_verdicts(summary, rows)
    summary["role"] = [
        dict(properties.STABILITY_ARMS)[cell.rsplit("__", 1)[1]] for cell in summary["cell"]
    ]
    return summary.set_index("cell")


def _context(summary: pd.DataFrame, rows: pd.DataFrame) -> property_failures.Context:
    return property_failures.Context(
        study.STUDY, rows, summary.reset_index(), {}, properties.PROPERTY_GROUPS
    )


def test_a_stability_label_at_rounding_spread_is_red() -> None:
    """mi2: a one-draw spread below the floor turns the label red, whatever its ratio reads.

    Draw 1 has an independent spread equal to draw 0's, so the control reads 1 on every label,
    and the median has half of draw 0's spread, so the positive reads 0.5.
    """
    rows, summary = _stability_frame(_stability_arms(three=0.5, second=1.0))
    properties.stability_verdicts(summary, rows)
    verdicts = summary.set_index("cell")["passed"]
    for arm, _ in properties.STABILITY_ARMS:
        assert not verdicts[f"i_contrast_x2__{arm}"]
    assert verdicts["d_ltmle_static__three_repeats"]
    assert verdicts["d_ltmle_static__one_repeat"]
    assert verdicts["d_ltmle_static__second_draw_control"]
    shares = summary.set_index("cell")["one_draw_spread_share"]
    for arm, _ in properties.STABILITY_ARMS:
        assert shares[f"i_contrast_x2__{arm}"] <= properties.STABILITY_SPREAD_FLOOR
    # The rounding label is red on every row, and the whole label's joint verdict with it.
    joint = summary.set_index("cell")["property_passed"]
    assert not joint.filter(like="i_contrast_x2__").any()
    assert joint.filter(like="d_ltmle_static__").all()
    # The declared rules read the same verdicts off the published columns.
    summary["role"] = [
        dict(properties.STABILITY_ARMS)[cell.rsplit("__", 1)[1]] for cell in summary["cell"]
    ]
    rule = property_failures.rule_for("repeat_stability", properties)
    assert rule is not None
    read = property_failures.rule_verdicts(rule, summary, _context(summary.set_index("cell"), rows))
    expected = summary.set_index("cell").loc[:, ["passed", "property_passed"]]
    published = read.set_index("cell").loc[expected.index, ["passed", "property_passed"]]
    assert published.astype(bool).equals(expected.astype(bool))


def test_a_second_draw_with_the_medians_spread_turns_the_control_red() -> None:
    """The witness that the control can fail: draw 1 has the median's spread (ratio 0.5).

    The median reads 0.5 against draw 0, so the positive stays green.  Draw 1 reads the same
    0.5, a reduction no independent single draw can show, so the control is red and the joint
    verdict of the label with it.
    """
    summary = _verdicts(_stability_arms(three=0.5, second=0.5))
    cell = "d_ltmle_static"
    assert summary.loc[f"{cell}__three_repeats", "spread_ratio_ci_upper"] < 1.0
    assert summary.loc[f"{cell}__second_draw_control", "spread_ratio_ci_upper"] < 1.0
    assert summary.loc[f"{cell}__three_repeats", "passed"]
    assert summary.loc[f"{cell}__one_repeat", "passed"]
    assert not summary.loc[f"{cell}__second_draw_control", "passed"]
    for arm, _ in properties.STABILITY_ARMS:
        assert not summary.loc[f"{cell}__{arm}", "property_passed"]
    # The control publishes its own ratio, never the positive's (X79: the old control read the
    # pair's interval and could not fail on its own).
    assert summary.loc[f"{cell}__second_draw_control", "spread_ratio"] == pytest.approx(
        0.5, abs=0.15
    )
    assert (
        summary.loc[f"{cell}__second_draw_control", "spread_ratio_ci_upper"]
        != summary.loc[f"{cell}__three_repeats", "spread_ratio_ci_upper"]
    )


def test_a_second_draw_with_the_first_draws_spread_keeps_both_green() -> None:
    """Draw 1 is a permutation of draw 0's values: the same spread, an independent pairing."""
    arms = _stability_arms(three=0.5, second=1.0)
    arms = {label: (three, one, one[::-1].copy()) for label, (three, one, _) in arms.items()}
    summary = _verdicts(arms)
    cell = "d_ltmle_static"
    control = summary.loc[f"{cell}__second_draw_control"]
    assert control["spread_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert control["spread_ratio_ci_lower"] < 1.0 <= control["spread_ratio_ci_upper"]
    for arm, _ in properties.STABILITY_ARMS:
        assert summary.loc[f"{cell}__{arm}", "passed"]
        assert summary.loc[f"{cell}__{arm}", "property_passed"]


def test_a_median_that_equals_draw_0_turns_the_positive_red() -> None:
    """The deliberate mutation: with no repeat layer the median is draw 0 and the ratio is 1.

    A ratio of exactly 1 has the upper endpoint 1, which is not below 1, so the positive and
    its base arm fail while the control (independent, equal spread) is green.
    """
    arms = _stability_arms(three=1.0, second=1.0)
    arms = {label: (one, one, second) for label, (_, one, second) in arms.items()}
    summary = _verdicts(arms)
    cell = "d_ltmle_static"
    assert summary.loc[f"{cell}__three_repeats", "spread_ratio_ci_upper"] == 1.0
    assert not summary.loc[f"{cell}__three_repeats", "passed"]
    assert not summary.loc[f"{cell}__one_repeat", "passed"]
    for arm, _ in properties.STABILITY_ARMS:
        assert not summary.loc[f"{cell}__{arm}", "property_passed"]


def test_the_stability_arms_carry_the_declared_roles() -> None:
    """``one_repeat`` is a positive row that shares the pair's verdict; only the second draw is
    a control."""
    roles = {
        cell.cell: cell.role
        for cell in properties.declared_cells()
        if cell.property == "repeat_stability"
    }
    assert len(roles) == 3 * len(properties.STABILITY)
    for label in properties.STABILITY:
        assert roles[f"{label}__three_repeats"] == "positive"
        assert roles[f"{label}__one_repeat"] == "positive"
        assert roles[f"{label}__second_draw_control"] == "control"
    assert set(study.PROPERTY_CELLS["repeat_stability"]) == {
        f"{label}__{cell}"
        for label in properties.STABILITY
        for cell, _ in properties.STABILITY_ARMS
    }
    assert all("one_repeat_control" not in cell for cell in roles)


def test_a_label_names_its_three_cells_as_one_group() -> None:
    groups = properties.PROPERTY_GROUPS["repeat_stability"]
    assert len(groups) == len(properties.STABILITY)
    for label, members in zip(properties.STABILITY, groups, strict=True):
        assert members == (
            f"{label}__three_repeats",
            f"{label}__one_repeat",
            f"{label}__second_draw_control",
        )
    rule = property_failures.rule_for("repeat_stability", properties)
    assert rule is not None and rule.group == "label"
    assert [clause.name for clause in rule.clauses] == [
        "repeat spread ratio under its boundary",
        "control spread ratio reaches its boundary",
        "one-draw split variance above the spread floor",
    ]


def test_the_control_bootstrap_has_its_own_stream() -> None:
    seeds = {
        properties._stability_seed(study.STUDY, "d_ltmle_static"),
        properties._stability_seed(study.STUDY, "d_ltmle_static", control=True),
        properties._stability_seed(study.STUDY, "d_sdr_static", control=True),
    }
    assert len(seeds) == 3


def _stability_accounted(
    monkeypatch: pytest.MonkeyPatch,
    lost: set[int],
    *,
    second: float = 1.0,
    trials: int = 200,
    reverse: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The driver's property steps on stability rows whose fits ``lost`` raised.

    Returns the stamped rows, the complete-case summary and the failure accounting.
    """
    arms = _stability_arms(three=0.5, second=second, trials=trials, rounding=False, reverse=reverse)

    def worker(key: tuple[str, int, int], frame: object) -> list[dict[str, object]]:
        law, replicate, requested = key
        if replicate in lost:
            raise np.linalg.LinAlgError("singular matrix")
        out = []
        for label, (owner, _, _) in properties.STABILITY.items():
            if owner != law:
                continue
            for (cell, role), estimates in zip(properties.STABILITY_ARMS, arms[label], strict=True):
                row = _row(f"{label}__{cell}", replicate, requested, float(estimates[replicate]))
                row.update(property="repeat_stability", role=role, std_error=0.01)
                out.append(row)
        return out

    _stub(monkeypatch, worker)
    monkeypatch.setattr(study, "RATE_LABELS", ())
    with replication.recording(study.STUDY.implementation) as recorded:
        rows = pd.DataFrame(
            [
                row
                for law in ("d", "i")
                for row in properties._fit(
                    f"stability_{law}", [(law, r, trials) for r in range(trials)], 1
                )
            ]
        )
    records = property_failures.frame(recorded)
    rows = property_failures.stamp(rows, records)
    property_failures.reconcile(rows, records, derived=properties.DERIVED_CELLS)
    summary = properties.summarize_properties(rows)
    accounting = property_failures.property_accounting(
        rows, records, summary, record=study.STUDY, module=properties, n_jobs=1
    )
    return rows, summary, accounting


def test_a_lost_trial_is_read_at_the_worst_case_of_each_stability_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two lost trials of the 200 are inside the bound; every row reads its applying clauses.

    The control row reads the control ratio clause and the share clause, and its worst case
    fills the second draw at its mean and draw 0 at each vertex split, which can only lower its
    ratio.  The positive rows read the positive ratio clause, which raises the ratio.
    """
    rows, summary, accounting = _stability_accounted(monkeypatch, {7, 8})
    label = "d_ltmle_static"
    legs = {
        arm: accounting.loc[
            (accounting["cell"] == f"{label}__{arm}")
            & ~accounting["leg"].isin([property_failures.BOUND_LEG, property_failures.WITNESS_LEG])
        ].set_index("leg")
        for arm, _ in properties.STABILITY_ARMS
    }
    ratio, control_ratio, share = (
        "repeat spread ratio under its boundary",
        "control spread ratio reaches its boundary",
        "one-draw split variance above the spread floor",
    )
    assert set(legs["three_repeats"].index) == {ratio, share}
    assert set(legs["one_repeat"].index) == {ratio, share}
    assert set(legs["second_draw_control"].index) == {control_ratio, share}
    complete = summary.set_index("cell")
    control = complete.loc[f"{label}__second_draw_control"]
    leg = legs["second_draw_control"].loc[control_ratio]
    assert leg["complete_case"] == pytest.approx(control["spread_ratio_ci_upper"] - 1.0)
    assert leg["worst_case"] <= leg["complete_case"] + 1e-12
    positive = legs["three_repeats"].loc[ratio]
    assert positive["worst_case"] <= positive["complete_case"] + 1e-12
    assert property_failures.allowed(properties.FAILURE_BOUND, 200) == 2
    assert (accounting["failed"] == 2).all()
    assert not (accounting["reason"] == property_failures.BOUND_EXCEEDED).any()
    # The joint verdict of a label reads every member: a corrected summary is red on all three
    # when any one leg fails, and untouched labels stay as they were.
    corrected = property_failures.correct_properties(summary, accounting, module=properties)
    members = [f"{label}__{arm}" for arm, _ in properties.STABILITY_ARMS]
    red = ~accounting.loc[
        accounting["cell"].isin(members)
        & ~accounting["leg"].isin([property_failures.BOUND_LEG, property_failures.WITNESS_LEG])
    ]["leg_passed"].astype(bool)
    joint = corrected.set_index("cell").loc[members, "property_passed"].astype(bool)
    assert joint.nunique() == 1
    assert bool(joint.iloc[0]) is (not red.any())
    # The label is a declared group of the module, so the accounting resolved it.
    assert rows["requested_replicates"].eq(200).all()


#: The second-draw spread, as a share of draw 0's, whose reversed pairing gives a control whose
#: complete-case upper endpoint sits just above 1 (read on the declared margins and seeds).
NARROW_SECOND = 0.865


def test_the_control_clause_turns_a_narrow_pass_red_inside_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A control read at 0.865 of draw 0's spread is green at its complete case.

    Draw 1 is draw 0's values in reverse order, so the spread ratio is exactly 0.865 and the
    99% upper endpoint of its paired bootstrap sits just above 1.  Two lost trials, filled
    adversely (draw 1 at its mean, draw 0 at each vertex split), take the endpoint below 1.
    """
    cell = "d_ltmle_static__second_draw_control"
    _, summary, accounting = _stability_accounted(
        monkeypatch, {7, 8}, second=NARROW_SECOND, reverse=True
    )
    reading = summary.set_index("cell").loc[cell]
    assert reading["passed"] and 1.0 <= reading["spread_ratio_ci_upper"] < 1.05
    rows = accounting.loc[accounting["cell"] == cell]
    leg = rows.loc[rows["leg"] == "control spread ratio reaches its boundary"].iloc[0]
    assert leg["complete_case"] >= 0.0
    assert leg["worst_case"] < 0.0
    assert not bool(leg["leg_passed"])
    assert not rows["gate_passed"].astype(bool).any()


@pytest.mark.filterwarnings("ignore:Cumulative mechanism truncation")
@pytest.mark.filterwarnings("ignore:.*read their influence curve from few rows")
def test_the_stability_trial_reads_draw_1_without_a_refit() -> None:
    """A trial publishes three cells per label; the control is draw 1 of the three-draw fit.

    At a small sample the trial runs in about a second.  Draw 1 equals a ``repeats=1`` fit at
    the plan's second seed (``LongitudinalRepeat`` documents it), so the refit below is the
    independent check that the row holds draw 1 and not draw 0 or the median.
    """
    law = "d"
    frame = study.sample(law, 600, 12345)
    rows = properties.stability_rows((law, 0, 400), frame)
    cells = [row["cell"] for row in rows]
    expected = [
        f"{label}__{cell}"
        for label, (owner, _, _) in properties.STABILITY.items()
        if owner == law
        for cell, _ in properties.STABILITY_ARMS
    ]
    assert cells == expected
    assert {row["role"] for row in rows if row["cell"].endswith("second_draw_control")} == {
        "control"
    }
    base = stream_seed(properties.STUDY, "repeat_stability", "fold_seed", law, 0)
    second_seed = CrossFitPlan(
        n_folds=study.N_FOLDS, random_state=base, repeats=study.REPEATS
    ).seeds()[1]
    scenario = "law_d_ltmle"
    refit = study.fit_scenario(scenario, frame.copy(), repeats=1, random_state=second_seed)
    name = study.D_CONTRASTS["static"]
    by_cell = {row["cell"]: row for row in rows}
    control = by_cell["d_ltmle_static__second_draw_control"]
    assert control["estimate"] == float(refit[name].psi)
    assert control["estimate"] != by_cell["d_ltmle_static__one_repeat"]["estimate"]
    assert control["std_error"] == pytest.approx(float(refit[name].plugin_std_error))


@pytest.mark.filterwarnings("ignore:Cumulative mechanism truncation")
@pytest.mark.filterwarnings("ignore:.*read their influence curve from few rows")
def test_a_trial_whose_draws_share_a_split_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collapse refusal: draw 1 with draw 0's assignment would make the control draw 0."""
    frame = study.sample("d", 600, 12345)
    original = study.fit_scenario

    def collapsed(scenario: str, data: pd.DataFrame, **options: object) -> object:
        result = original(scenario, data, **options)
        if options.get("repeats", study.REPEATS) > 1:
            same = result.repeats[0].folds.assignment
            object.__setattr__(result.repeats[1].folds, "assignment", same.copy())
        return result

    monkeypatch.setattr(study, "fit_scenario", collapsed)
    with pytest.raises(RuntimeError, match="share one fold assignment"):
        properties.stability_rows(("d", 0, 400), frame)


# ------------------------------------------------- the inherit check across the rename (X79)


def _inherit_tables(*, renamed: bool) -> dict[str, pd.DataFrame]:
    """Tiny primary, replicate and summary tables of a baseline (old arm names) or a rerun."""
    arm = "one_repeat" if renamed else "one_repeat_control"
    arms = ["three_repeats", arm, *(["second_draw_control"] if renamed else [])]
    roles = {"three_repeats": "positive", arm: "positive" if renamed else "control"}
    roles["second_draw_control"] = "control"
    primary = pd.DataFrame(
        {
            "implementation": "cleverly",
            "scenario": "law_d",
            "replicate": [0, 1],
            "estimand": "ate",
            "estimate": [0.25, 0.5],
        }
    )
    replicate_rows = [
        {
            "property": "interval_calibration",
            "cell": "d__correctly_specified",
            "replicate": trial,
            "role": "positive",
            "estimate": 0.1 * (trial + 1),
            "requested_replicates": 2,
            "failed_replicates": 0,
        }
        for trial in range(2)
    ]
    summary = [
        {
            "property": "interval_calibration",
            "cell": "d__correctly_specified",
            "role": "positive",
            "coverage": 0.95,
            "passed": True,
        }
    ]
    for index, cell in enumerate(arms):
        replicate_rows.extend(
            {
                "property": "repeat_stability",
                "cell": f"d__{cell}",
                "replicate": trial,
                "role": roles[cell],
                "estimate": 0.01 * (index + 1) * (trial + 1),
                "requested_replicates": 2,
                "failed_replicates": 0,
            }
            for trial in range(2)
        )
        summary.append(
            {
                "property": "repeat_stability",
                "cell": f"d__{cell}",
                "role": roles[cell],
                "n": 100,
                "replicates": 2,
                "failed_replicates": 0,
                "spread_ratio": 0.5 + 0.1 * index,
                "spread_ratio_ci_lower": 0.4,
                "spread_ratio_ci_upper": 0.9 + 0.2 * index,
                "spread_ratio_boundary": 1.0,
                "one_draw_spread_share": 0.3,
                "passed": index < 2,
                "property_passed": index < 2,
            }
        )
    return {
        inherit.PRIMARY_ROWS: primary,
        inherit.ROWS: pd.DataFrame(replicate_rows),
        inherit.SUMMARY: pd.DataFrame(summary),
    }


def _inherit_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit: Callable[[dict[str, pd.DataFrame]], None] | None = None,
) -> list[str]:
    old, new = _inherit_tables(renamed=False), _inherit_tables(renamed=True)
    if edit is not None:
        edit(new)
    directory = tmp_path / "new"
    directory.mkdir(exist_ok=True)
    for name, frame in new.items():
        if name.endswith(".parquet"):
            write_rows(frame, directory / name)
        else:
            frame.to_csv(directory / name, index=False)
    # The baseline commit predates the Parquet row files: it holds the gzipped CSV names.
    blobs = {name: frame.to_csv(index=False).encode() for name, frame in old.items()}
    held = {
        inherit.CSV_NAMES.get(name, name): gzip.compress(blob)
        if name in inherit.CSV_NAMES
        else blob
        for name, blob in blobs.items()
    }

    def baseline(commit: str, name: str) -> bytes:
        if name not in held:
            raise subprocess.CalledProcessError(128, ["git", "show"])
        return held[name]

    monkeypatch.setattr(inherit, "HERE", directory)
    monkeypatch.setattr(inherit, "baseline", baseline)
    return inherit.check("baseline")


def test_the_inherit_check_accepts_the_rename_and_the_new_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _inherit_setup(tmp_path, monkeypatch) == []


def test_the_inherit_check_tolerates_last_bit_noise_in_a_kept_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def nudge(new: dict[str, pd.DataFrame]) -> None:
        rows = new[inherit.ROWS]
        kept = rows["property"].eq("repeat_stability") & rows["cell"].str.endswith("__one_repeat")
        rows.loc[kept, "estimate"] *= 1.0 + 1e-14
        new[inherit.PRIMARY_ROWS]["estimate"] *= 1.0 + 1e-14

    assert _inherit_setup(tmp_path, monkeypatch, nudge) == []


def test_the_inherit_check_refuses_a_kept_arm_whose_estimate_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def move(new: dict[str, pd.DataFrame]) -> None:
        rows = new[inherit.ROWS]
        rows.loc[rows["cell"] == "d__three_repeats", "estimate"] *= 1.0 + 1e-6

    assert _inherit_setup(tmp_path, monkeypatch, move)


def test_the_inherit_check_refuses_a_kept_arm_whose_verdict_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def move(new: dict[str, pd.DataFrame]) -> None:
        summary = new[inherit.SUMMARY]
        summary.loc[summary["cell"] == "d__one_repeat", "passed"] = False

    differences = _inherit_setup(tmp_path, monkeypatch, move)
    assert any("stability arms differ" in line for line in differences)


def test_the_inherit_check_does_not_read_the_new_arm_but_reads_every_other_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reshape_new_arm(new: dict[str, pd.DataFrame]) -> None:
        rows = new[inherit.ROWS]
        rows.loc[rows["cell"].str.endswith("__second_draw_control"), "estimate"] = 99.0

    assert _inherit_setup(tmp_path, monkeypatch, reshape_new_arm) == []

    def move_other(new: dict[str, pd.DataFrame]) -> None:
        rows = new[inherit.ROWS]
        rows.loc[rows["property"] == "interval_calibration", "estimate"] *= 1.0 + 1e-15

    assert _inherit_setup(tmp_path, monkeypatch, move_other)

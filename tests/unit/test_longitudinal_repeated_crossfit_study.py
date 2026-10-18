"""The ``longitudinal-repeated-crossfit`` drivers, checked without running the study.

One primary replication of every scenario at n = 2,000 checks the row contract the shared
driver publishes.  Two property fits under a checkpoint root check that a part run in chunks
resumes: a second call reuses every replicate the first one finished and returns the same rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.canonical.longitudinal_repeated_crossfit import part
from tests.canonical.regenerate import digest, run_key, runs_root
from tests.studies import longitudinal_repeated_crossfit as study
from tests.studies import longitudinal_repeated_crossfit_properties as properties
from tests.studies.evidence import property_failures, property_verdicts, replication
from tests.studies.evidence.schema import REPLICATE_COLUMNS

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


def test_a_stability_label_at_rounding_spread_is_red() -> None:
    """mi2: a one-draw spread below the floor turns the label red, whatever its ratio reads."""
    rng = np.random.default_rng(0)
    rows, cells = [], []
    for label in properties.STABILITY:
        scale = 1e-9 if label == "i_contrast_x2" else 1e-2
        one = rng.normal(0.0, scale, 60)
        for cell, values in (
            ("three_repeats", 0.5 * one),
            ("one_repeat_control", one),
        ):
            cells.append(f"{label}__{cell}")
            for trial, value in enumerate(values):
                row = _row(f"{label}__{cell}", trial, 60, estimate=float(value))
                row.update(property="repeat_stability")
                rows.append(row)
    summary = pd.DataFrame({"property": "repeat_stability", "cell": cells})
    properties.stability_verdicts(summary, pd.DataFrame(rows))
    verdicts = summary.set_index("cell")["passed"]
    assert not verdicts["i_contrast_x2__three_repeats"]
    assert verdicts["d_ltmle_static__three_repeats"]
    shares = summary.set_index("cell")["one_draw_spread_share"]
    assert shares["i_contrast_x2__three_repeats"] <= properties.STABILITY_SPREAD_FLOOR
    # The declared rule reads the same verdict off the published columns.
    summary["role"] = "positive"
    rule = property_failures.rule_for("repeat_stability", properties)
    assert rule is not None
    clause = rule.clauses[0]
    for _, row in summary.iterrows():
        assert clause.read(None, row).passed == verdicts[row["cell"]]  # type: ignore[arg-type]

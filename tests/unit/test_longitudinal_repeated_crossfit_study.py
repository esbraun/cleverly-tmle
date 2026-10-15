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
from tests.studies.evidence import replication
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


# ------------------------------------------------------------------ failure accounting


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
        "covered": 1,
        "rejected": 0,
    }


def _flaky(key: tuple[str, int, int], drawn: object = None) -> list[dict[str, object]]:
    """Replicate 1 raises a recordable error and replicate 2 returns a non-finite estimate."""
    del drawn
    _, replicate, requested = key
    if replicate == 1:
        raise np.linalg.LinAlgError("singular matrix")
    estimate = float("nan") if replicate == 2 else 0.1 * replicate
    return [_row("mean_x1__repeated", replicate, requested, estimate)]


def test_a_recorded_failure_costs_one_replicate_and_replays_on_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """MA2: a failure is recorded and counted, not raised, and a resume counts it again."""
    keys = [("fake", replicate, 6) for replicate in range(6)]
    for _ in range(2):
        with replication.checkpoints(tmp_path):
            rows = properties._fit("fake", _flaky, keys, n_jobs=1)
        assert [row["replicate"] for row in rows] == [0, 3, 4, 5]
        assert {row["failed_replicates"] for row in rows} == {2}
        assert "2 of 6 replicates recorded a fitting failure" in capsys.readouterr().out


def _table(failed: dict[str, int], requested: int = 200) -> pd.DataFrame:
    """Two paired cells per fit set; each loses its fit set's first ``failed`` replicates."""
    rows = []
    for cell, lost in failed.items():
        family = "root_n_and_efficiency" if "__n_" in cell else "ipsi_primary"
        for replicate in range(lost, requested):
            row = _row(cell, replicate, requested, estimate=0.01 * replicate)
            row.update(property=family, failed_replicates=lost)
            rows.append(row)
    return pd.DataFrame(rows)


def test_the_summary_reads_survivors_and_applies_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At most ``floor(0.01 R)`` failures keep the survivors' verdict; one more is red."""
    seen: list[pd.DataFrame] = []

    def summarize(rows: pd.DataFrame) -> pd.DataFrame:
        seen.append(rows)
        properties.require_complete(rows)
        cells = rows.groupby(["property", "cell"], as_index=False).size()
        summary = cells.rename(columns={"size": "replicates"}).assign(
            failed_replicates=0, passed=True, property_passed=True
        )
        rate = {"property": "root_n_rate", "cell": "ltmle_static__empirical_sd"}
        return pd.concat([summary, pd.DataFrame([rate])], ignore_index=True).assign(
            passed=True, property_passed=True
        )

    monkeypatch.setattr(properties, "_summarize", summarize)
    table = _table(
        {"mean_x1__repeated": 2, "mean_x2__repeated": 3, "ltmle_static__n_2000": 3},
    )
    summary = properties.summarize_properties(table).set_index("cell")
    assert properties.allowed_failures(200) == 2
    assert summary.loc["mean_x1__repeated", "passed"]
    assert summary.loc["mean_x1__repeated", "replicates"] == 198
    assert summary.loc["mean_x1__repeated", "failed_replicates"] == 2
    assert not summary.loc["mean_x2__repeated", "passed"]
    assert not summary.loc["mean_x2__repeated", "property_passed"]
    assert not summary.loc["ltmle_static__empirical_sd", "passed"]
    assert summary.loc["ltmle_static__empirical_sd", "failed_replicates"] == 3
    (survived,) = seen
    assert set(survived["failed_replicates"]) == {0}
    for _, group in survived.groupby("cell"):
        assert list(group["replicate"]) == list(range(len(group)))


def test_survivors_leave_a_complete_table_unchanged() -> None:
    table = _table({"mean_x1__repeated": 0})
    pd.testing.assert_frame_equal(properties.survivors(table), table)


def test_an_unaccounted_loss_is_refused() -> None:
    table = _table({"mean_x1__repeated": 2})
    table["failed_replicates"] = 1
    with pytest.raises(ValueError, match="not accounted for"):
        properties.failure_counts(table)


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

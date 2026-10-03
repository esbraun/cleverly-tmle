"""The scale-probe publication hook of the missing-outcome attributable-effect study.

The continuous scenario makes two R population-mean fits per replication, the natural
course and the reference arm, and each plants the outcome scale on two excluded rows.
``scale-probe.csv`` checks that workaround on both paths of every continuous replication, and
``scientific_failures`` refuses publication when a row fails or a path is not covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.studies import canonical_mar_attributable as study

PROBE = "scale-probe.csv"
FAILURE = "scale-workaround probe"
COVERAGE = "scale-workaround probe coverage"


def _passing(replicates: int = 3) -> pd.DataFrame:
    rows = [
        {"scenario": study.CONTINUOUS, "replicate": replicate, "path": path, "passed": True}
        for replicate in range(replicates)
        for path in study.PROBE_PATHS
    ]
    return pd.DataFrame(rows)


def _failures(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return study.scientific_failures({PROBE: frame})


def test_a_passing_probe_publishes() -> None:
    failures = _failures(_passing())
    assert list(failures) == [FAILURE]
    assert failures[FAILURE].empty


def test_a_failing_row_on_either_path_refuses_publication() -> None:
    for path in study.PROBE_PATHS:
        frame = _passing()
        frame.loc[(frame["path"] == path) & (frame["replicate"] == 1), "passed"] = False
        assert len(_failures(frame)[FAILURE]) == 1


def test_a_missing_verdict_refuses_publication() -> None:
    for missing in (np.nan, pd.NA):
        frame = _passing().astype({"passed": "object"})
        frame.loc[2, "passed"] = missing
        assert len(_failures(frame)[FAILURE]) == 1


def test_a_probe_that_misses_one_r_path_refuses_publication() -> None:
    """The per-path coverage check: one path's rows alone are not a covered replication."""
    for path in study.PROBE_PATHS:
        frame = _passing()
        one_path = frame.loc[frame["path"] != path]
        assert not _failures(one_path)[COVERAGE].empty
        partial = frame.drop(frame.index[(frame["path"] == path) & (frame["replicate"] == 2)])
        assert not _failures(partial)[COVERAGE].empty


def test_a_short_duplicated_or_foreign_probe_refuses_publication() -> None:
    frame = _passing()
    for short in (
        frame.iloc[:0],
        frame.loc[frame["replicate"] > 0],
        frame.assign(replicate=frame["replicate"].replace({2: 1})),
        frame.assign(scenario="binary_mar_attributable"),
    ):
        failures = _failures(short)
        assert COVERAGE in failures
        assert not failures[COVERAGE].empty  # the driver ignores empty failure frames


def test_every_committed_continuous_fit_passed_the_probe_on_both_paths() -> None:
    """The committed artifact: both paths of every continuous replication pass."""
    probe = pd.read_csv(study.STUDY.artifact(PROBE))
    assert set(probe["scenario"]) == {study.CONTINUOUS}
    for path in study.PROBE_PATHS:
        rows = probe.loc[probe["path"] == path]
        assert sorted(rows["replicate"]) == list(range(study.STUDY.replicates))
    assert probe["passed"].astype(bool).all()
    assert probe["scale_exact"].astype(bool).all()
    assert probe["unplanted_scale_differs"].astype(bool).all()
    assert (probe[["planted_scale_lower", "planted_scale_upper"]] == [0.0, 1.0]).all().all()
    assert (probe["moved_rows_difference"] == 0).all()
    assert (probe["rebuild_difference"] <= probe["rebuild_tolerance"]).all()
    assert (probe["rebuild_tolerance"] == 1e-12).all()
    assert (probe["unplanted_point_difference"] > 0).all()
    assert _failures(probe)[FAILURE].empty

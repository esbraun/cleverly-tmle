"""Spilled samples: the primary phase streams each sample to disk and assembles the archive.

``fit_primary(..., spill_samples_at=...)`` writes each fitted sample to disk, and
``assembled_samples`` joins the chunks in key order.  The archive must equal what
``write_csv(pd.concat(frames, ignore_index=True), path, compression=...)`` writes, including the
dtypes ``pd.concat`` gives a column that some frames lack.  A rerun must reuse the chunks of the
checkpoints it reuses, and the driver must accept the path.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_mtp_pricing as pricing
from tests.studies.evidence import failures, replication
from tests.studies.evidence.manifest import write_csv
from tests.unit import test_python_phase_failures as toy

ARCHIVE = {"method": "gzip", "mtime": 0}
DRAW = np.random.default_rng


def _written(frames: list[pd.DataFrame], path: Path) -> bytes:
    write_csv(pd.concat(frames, ignore_index=True), path, compression=ARCHIVE)
    return path.read_bytes()


def _mixed(replicate: int) -> pd.DataFrame:
    """Frames whose columns differ: an int, a bool and an all-missing column come and go."""
    base = {"scenario": f"s{replicate % 3}", "replicate": replicate, "a": np.arange(4)}
    kind = replicate % 3
    if kind == 0:
        extra = {"b": [True, False, True, True], "c": [np.nan] * 4, "w": [0.1, 0.25, 1e-9, 3.0]}
    elif kind == 1:
        extra = {"d": ["x", "y", "z", "x"], "k": np.arange(4, dtype=np.int32)}
    else:
        extra = {"b": [False] * 4, "c": [1.5, np.nan, 2.5, 0.5], "d": list("pqrs"), "w": 0.5}
    return pd.DataFrame({**base, **extra})


def _mixed_fit(key: tuple[str, int], drawn: None) -> tuple[pd.DataFrame, None, list]:
    return _mixed(key[1]), None, []


def test_the_assembled_archive_has_the_bytes_of_the_concatenated_frames(tmp_path: Path) -> None:
    keys = [("law", replicate) for replicate in range(7)]
    outcomes = replication.fit_primary(
        keys,
        _mixed_fit,
        draw=lambda key: None,
        n_jobs=2,
        implementation="cleverly",
        spill_samples_at=0,
    )
    assert outcomes.spill_directory is not None
    assert all(isinstance(value[0], replication.SpilledSample) for value in outcomes.fitted)
    path = replication.assembled_samples(outcomes)
    expected = pd.concat([_mixed(replicate) for _, replicate in keys], ignore_index=True)
    # The test is only as sharp as the dtype upcasts it covers.
    assert expected["a"].dtype == np.int64
    assert expected["b"].dtype == object and expected["k"].dtype == np.float64
    assert path.read_bytes() == _written(
        [_mixed(replicate) for _, replicate in keys], tmp_path / "samples.csv.gz"
    )
    assert replication.load_samples(path).shape == expected.shape


def test_an_unspilled_outcome_cannot_be_assembled() -> None:
    outcomes = replication.fit_primary(
        [("law", 0), ("law", 1)],
        _mixed_fit,
        draw=lambda key: None,
        n_jobs=2,
        implementation="cleverly",
    )
    with pytest.raises(ValueError, match="spilled no samples"):
        replication.assembled_samples(outcomes)


def test_the_pricing_study_streams_all_six_scenarios(tmp_path: Path) -> None:
    n, replicates = 100, 2
    result = pricing.draw_and_fit(replicates=replicates, n=n, n_jobs=2)
    assert isinstance(result.samples, Path)
    keys = [
        (scenario, replicate, n)
        for replicate in range(replicates)
        for scenario in pricing.SCENARIOS
    ]
    assert len(pricing.SCENARIOS) == 6
    frames = [pricing._replicate(key, pricing._draw(key))[0] for key in keys]
    assert len({tuple(frame.columns) for frame in frames}) > 1  # tau 2, 3 and 5 differ
    expected = _written(frames, tmp_path / "samples.csv.gz")
    assert gzip.decompress(result.samples.read_bytes()) == gzip.decompress(expected)
    assert result.samples.read_bytes() == expected


@dataclass(frozen=True)
class _Fit:
    """The toy fit; it logs each call and stops the run at ``stop``."""

    log: Path
    stop: int | None = None

    def __call__(self, key: tuple[str, int], y: np.ndarray) -> Any:
        (self.log / f"{key[1]}").touch()
        if key[1] == self.stop:
            raise RuntimeError("the run stopped here")
        return toy._fit(key, y)


def _study(here: Path, fit: _Fit) -> Any:
    study = toy._study(here, from_fit=True)

    def draw_and_fit(*, replicates: int, n: int, n_jobs: int) -> replication.PhaseResult:
        outcomes = replication.fit_primary(
            [("law", replicate) for replicate in range(replicates)],
            fit,
            draw=toy._draw,
            n_jobs=n_jobs,
            implementation=toy.SUBJECT,
            spill_samples_at=0,
        )
        rows = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
        truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
        return replication.result(
            outcomes, rows, samples=replication.assembled_samples(outcomes), truths=truths
        )

    study.draw_and_fit = draw_and_fit
    return study


def _expected(tmp_path: Path) -> bytes:
    frames = [
        toy._sample(("law", replicate), toy._draw(("law", replicate)))
        for replicate in range(toy.REPLICATES)
        if replicate != toy.FAILED
    ]
    return _written(frames, tmp_path / "samples.csv.gz")


def test_the_driver_takes_a_path_and_writes_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "log").mkdir()
    out = toy._run(tmp_path, monkeypatch, _study(tmp_path, _Fit(tmp_path / "log")))
    archive = tmp_path / "cache" / "samples.csv.gz"
    assert archive.read_bytes() == _expected(tmp_path)
    recorded = failures.read(out / failures.FAILURES)
    assert recorded is not None and toy.FAILED not in set(pd.read_csv(archive)["replicate"])
    assert not (
        tmp_path / "cache" / "checkpoints" / toy.SUBJECT / "primary-samples" / "samples.csv.gz"
    ).exists()


def test_a_killed_run_resumes_from_its_checkpoints_and_its_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    with pytest.raises(RuntimeError, match="the run stopped here"):
        toy._run(tmp_path, monkeypatch, _study(tmp_path, _Fit(first, stop=toy.REPLICATES - 2)))
    assert not (tmp_path / "cache" / "samples.csv.gz").exists()
    toy._run(tmp_path, monkeypatch, _study(tmp_path, _Fit(second)))
    refitted = len(list(second.iterdir()))
    assert 0 < refitted < toy.REPLICATES  # the keys that finished were not fitted again
    assert (tmp_path / "cache" / "samples.csv.gz").read_bytes() == _expected(tmp_path)

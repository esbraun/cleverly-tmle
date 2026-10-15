"""The driver records a failed ``cleverly`` fit as one lost replicate (RM39b).

A toy study runs through :func:`tests.canonical.regenerate.main` end to end, with one subject fit
that raises a recordable error.  An independent study still hands the replicate's sample to the
reference.  A study whose reference input comes from the subject's fit loses the sample, and
the driver records the reference's loss as induced by the subject.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.canonical import regenerate
from tests.studies.evidence import failures, replication
from tests.studies.evidence.registry import Margins, StudyRecord

SUBJECT, REFERENCE = "cleverly", "toy ref"
REPLICATES, FAILED = 30, 3
Z = 2.576


def _record(here: Path) -> StudyRecord:
    return StudyRecord(
        name="toy failure study",
        slug="toy-failure-study",
        artifacts=here,
        document="toy.md",
        anchor="toy",
        scenarios={"law": ("ey",)},
        replicates=REPLICATES,
        n=50,
        seed=5,
        reference=REFERENCE,
        comparison="matching implementation",
        failure_bound={SUBJECT: 0.05, REFERENCE: 0.05},
        margins=Margins(bootstrap_replicates=200),
        publication_policy="reporting",
    )


def _row(implementation: str, replicate: int, values: np.ndarray) -> dict[str, Any]:
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    low, high = estimate - Z * se, estimate + Z * se
    return {
        "implementation": implementation,
        "scenario": "law",
        "replicate": replicate,
        "n": len(values),
        "estimand": "ey",
        "truth": 0.0,
        "estimate": estimate,
        "inference_estimate": estimate,
        "std_error": se,
        "ci_lower": low,
        "ci_upper": high,
        "inference_scale": "identity",
        "covered": int(low <= 0.0 <= high),
        "initial_estimate": estimate,
    }


def _draw(key: tuple[str, int]) -> np.ndarray:
    return np.random.default_rng([5, key[1]]).normal(size=50)


def _sample(key: tuple[str, int], y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"scenario": key[0], "replicate": key[1], "y": y})


def _truth(key: tuple[str, int]) -> list[dict[str, Any]]:
    return [{"scenario": key[0], "replicate": key[1], "estimand": "ey", "truth": 0.0}]


def _fit(key: tuple[str, int], y: np.ndarray) -> tuple[pd.DataFrame, list, list]:
    if key[1] == FAILED:
        np.linalg.inv(np.zeros((2, 2)))
    return _sample(key, y), _truth(key), [_row(SUBJECT, key[1], y)]


def _reference_input(key: tuple[str, int], y: np.ndarray) -> tuple[pd.DataFrame, list, list]:
    return _sample(key, y), _truth(key), []


def _study(here: Path, *, from_fit: bool) -> SimpleNamespace:
    record = _record(here)

    def draw_and_fit(*, replicates: int, n: int, n_jobs: int) -> replication.PhaseResult:
        outcomes = replication.fit_replicates(
            [("law", replicate) for replicate in range(replicates)],
            _fit,
            draw=_draw,
            on_failure=None if from_fit else _reference_input,
            validate=lambda value: replication.invalid_rows(value[2]),
            n_jobs=n_jobs,
            label="primary",
            implementation=SUBJECT,
            cell_of=lambda key: key[0],
            replicate_of=lambda key: key[1],
        )
        kept = outcomes.values if from_fit else outcomes.fitted
        return replication.PhaseResult(
            rows=pd.DataFrame([row for _, _, rows in kept for row in rows]),
            samples=pd.concat([sample for sample, _, _ in kept], ignore_index=True),
            truths=pd.DataFrame([row for _, rows, _ in kept for row in rows]),
            failures=replication.records(outcomes),
            reference_input_from_fit=from_fit,
        )

    return SimpleNamespace(
        STUDY=record,
        PRIMARY_REPLICATES=REPLICATES,
        PRIMARY_N=50,
        CONFIGURATION={},
        draw_and_fit=draw_and_fit,
    )


def _reference() -> SimpleNamespace:
    def run(here: Path, samples: Path, truths: Path, output: Path, *, cores: int) -> None:
        frame = pd.read_csv(samples)
        rows = [
            _row(REFERENCE, int(replicate), group["y"].to_numpy(float) + 1e-4)
            for (_, replicate), group in frame.groupby(["scenario", "replicate"])
        ]
        pd.DataFrame(rows).to_csv(output, index=False)

    return SimpleNamespace(run=run, interpreter="python", files=lambda here: [])


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study: Any, **flags: Any) -> Path:
    out, cache = tmp_path / "out", tmp_path / "cache"
    options: dict[str, Any] = {
        "replicates": REPLICATES,
        "n": 50,
        "skip_properties": True,
        "primary_only": True,
        "allow_failures": False,
        "output": out,
        "jobs": 1,
        "reference_jobs": 1,
        "skip_reference": False,
        "cache": cache,
        "fresh": False,
        "keep_cache": False,
        "refresh_python": False,
    }
    arguments = argparse.Namespace(**{**options, **flags})
    monkeypatch.setattr(regenerate, "_arguments", lambda *args: arguments)
    monkeypatch.setattr(regenerate, "_reference_key", lambda *args: None)
    regenerate.main(study, SimpleNamespace(), here=tmp_path, reference=_reference())  # type: ignore[arg-type]
    return out


def test_an_independent_studys_reference_fits_the_replicate_the_subject_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run(tmp_path, monkeypatch, _study(tmp_path, from_fit=False))
    recorded = failures.read(out / failures.FAILURES)
    assert recorded is not None
    assert recorded[["implementation", "group_id", "class"]].values.tolist() == [
        [SUBJECT, f"primary/law/r{FAILED:09d}", "fitting_failure"]
    ]
    rows = pd.read_csv(out / "replicates.csv.gz")
    assert FAILED in set(rows.loc[rows["implementation"] == REFERENCE, "replicate"])
    assert FAILED not in set(rows.loc[rows["implementation"] == SUBJECT, "replicate"])
    paired = pd.read_csv(out / "equivalence.csv")
    assert paired["dropped_replications"].tolist() == [1]
    accounting = pd.read_csv(out / failures.ACCOUNTING)
    subject_truth = accounting.loc[
        (accounting["table"] == "performance") & (accounting["implementation"] == SUBJECT)
    ]
    assert set(subject_truth["leg"]) == {"failure bound", "bias", "coverage", "se ratio"}
    assert set(accounting.loc[accounting["table"] == "equivalence", "subject_failed"]) == {1}


def test_a_dependent_studys_reference_loses_the_replicate_by_an_induced_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _run(tmp_path, monkeypatch, _study(tmp_path, from_fit=True))
    recorded = failures.read(out / failures.FAILURES)
    assert recorded is not None
    assert recorded[["implementation", "class", "replicates"]].values.tolist() == [
        [SUBJECT, "fitting_failure", str(FAILED)],
        [REFERENCE, failures.PAIRED_INPUT_MISSING, str(FAILED)],
    ]
    rows = pd.read_csv(out / "replicates.csv.gz")
    assert FAILED not in set(rows["replicate"])
    samples = pd.read_csv(tmp_path / "cache" / "samples.csv.gz")
    assert FAILED not in set(samples["replicate"])
    accounting = pd.read_csv(out / failures.ACCOUNTING)
    paired = accounting.loc[accounting["table"] == "equivalence"].iloc[0]
    # The reference's own bound pays nothing for the induced loss.
    assert (paired["subject_failed"], paired["reference_failed"]) == (1, 0)


def test_a_rerun_reuses_the_phase_and_its_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, from_fit=False)
    out = _run(tmp_path, monkeypatch, study)
    first = (out / failures.FAILURES).read_bytes()
    checkpoints = tmp_path / "cache" / regenerate.CHECKPOINTS / SUBJECT / "primary"
    assert len(list(checkpoints.glob("*.pkl.gz"))) == REPLICATES
    study.draw_and_fit = lambda **kwargs: pytest.fail("the cached phase was refitted")
    _run(tmp_path, monkeypatch, study)
    assert any("reused the Python phase" in note for note in regenerate.RUN_NOTES)
    assert (out / failures.FAILURES).read_bytes() == first


def test_refresh_python_and_fresh_discard_the_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(tmp_path, from_fit=False)
    _run(tmp_path, monkeypatch, study)
    directory = tmp_path / "cache" / regenerate.CHECKPOINTS / SUBJECT / "primary"
    for flag in ("refresh_python", "fresh"):
        stale = directory / "stale.pkl.gz"
        stale.write_bytes(b"")
        _run(tmp_path, monkeypatch, study, **{flag: True})
        assert not stale.exists(), flag
        assert len(list(directory.glob("*.pkl.gz"))) == REPLICATES

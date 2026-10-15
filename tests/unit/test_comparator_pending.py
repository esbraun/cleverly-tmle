"""A comparator whose runs are pending: the registry state, the driver and the manifest check.

The owner's rule: a method merges once its Python studies pass, and its comparator study is
registered by a later change.  The pending state is asserted wherever it is read, so these
tests mutate each part and require the check to fail.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tests.canonical import regenerate
from tests.studies import canonical_tmle
from tests.studies.evidence.manifest import comparator_findings
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS

COMPARED = canonical_tmle.STUDY
REFERENCE = str(COMPARED.reference)
REASON = "the R runs wait in the local queue; owner X99"


def _uncompared() -> StudyRecord:
    """``COMPARED`` with no comparator, so with no bound for one."""
    bounds = {COMPARED.implementation: COMPARED.failure_bound[COMPARED.implementation]}
    return dataclasses.replace(COMPARED, reference=None, failure_bound=bounds)


def _pending(**changes: Any) -> Any:
    fields = {"reference": None, "pending_reference": REFERENCE, "pending_reason": REASON}
    return dataclasses.replace(COMPARED, **{**fields, **changes})


def _manifest(record: Any) -> dict[str, Any]:
    return {
        "generated_with": {"subject": {}, "reference": None},
        "reference_sha256": {},
        "comparator": {
            "status": "pending",
            "implementation": record.pending_reference,
            "reason": record.pending_reason,
            "metadata": {},
            "declared_sources_sha256": {"tests/canonical/run.R": "0" * 64},
        },
    }


class TestTheRecord:
    def test_the_three_states(self) -> None:
        assert COMPARED.comparator_state == "compared"
        assert _pending().comparator_state == "pending"
        assert _uncompared().comparator_state == "none"

    def test_a_pending_record_pairs_no_rows(self) -> None:
        assert _pending().implementations == (COMPARED.implementation,)

    def test_a_pending_comparator_may_carry_a_declared_failure_bound(self) -> None:
        assert _pending(failure_bound={REFERENCE: 0.01}).failure_bound == {REFERENCE: 0.01}

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"reference": REFERENCE}, "either a compared reference or a pending one"),
            ({"pending_reason": " "}, "needs a pending_reason"),
            ({"accepted_reference_failure": "x"}, "a pending comparator has not run"),
        ],
    )
    def test_an_incoherent_pending_record_is_refused(
        self, changes: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _pending(**changes)

    def test_a_reason_without_a_pending_comparator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="without a pending_reference"):
            dataclasses.replace(COMPARED, pending_reason=REASON)

    def test_a_truth_that_varies_cannot_wait_for_a_comparator(self) -> None:
        with pytest.raises(ValueError, match="cannot declare a pending reference"):
            dataclasses.replace(
                COMPARED,
                reference=None,
                pending_reference=REFERENCE,
                pending_reason=REASON,
                truth_varies_by_replicate=True,
            )


class TestTheManifestCheck:
    def test_a_coherent_pending_manifest_passes(self) -> None:
        record = _pending()
        assert comparator_findings(record, _manifest(record), ["cleverly"], 0) == []

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda m: m.pop("comparator"), "does not say so"),
            (lambda m: m["comparator"].update(status="compared"), "does not say so"),
            (lambda m: m["comparator"].update(implementation="other"), "pending comparator is"),
            (lambda m: m["comparator"].update(reason="other"), "pending reason"),
            (lambda m: m["comparator"].update(declared_sources_sha256={}), "no declared"),
            (lambda m: m["generated_with"].update(reference={"x": 1}), "reference provenance"),
            (lambda m: m.update(reference_sha256={"r.R": "1" * 64}), "as run"),
        ],
    )
    def test_each_manifest_mutation_is_caught(self, mutate: Any, message: str) -> None:
        record = _pending()
        manifest = _manifest(record)
        mutate(manifest)
        assert any(
            message in found for found in comparator_findings(record, manifest, ["cleverly"], 0)
        )

    def test_published_reference_rows_are_caught(self) -> None:
        record = _pending()
        found = comparator_findings(record, _manifest(record), ["cleverly", REFERENCE], 0)
        assert any("rows of the pending comparator" in item for item in found)

    def test_a_paired_row_never_reads_as_passed_while_pending(self) -> None:
        """An empty paired table reads as every verdict passed, so a non-empty one is refused."""
        record = _pending()
        found = comparator_findings(record, _manifest(record), ["cleverly"], 3)
        assert any("3 paired row(s)" in item for item in found)

    def test_a_compared_record_with_a_pending_manifest_is_caught(self) -> None:
        manifest = _manifest(_pending())
        manifest["generated_with"]["reference"] = {"implementation": REFERENCE}
        found = comparator_findings(COMPARED, manifest, ["cleverly", REFERENCE], 1)
        assert any("records one as pending" in item for item in found)

    def test_a_record_without_a_comparator_publishing_paired_rows_is_caught(self) -> None:
        record = _uncompared()
        manifest = {"generated_with": {"reference": None}, "reference_sha256": {}}
        assert any(
            "paired row(s)" in item
            for item in comparator_findings(record, manifest, ["cleverly"], 2)
        )


def _rows(implementation: str) -> pd.DataFrame:
    row: dict[str, Any] = dict.fromkeys(REPLICATE_COLUMNS, 0.0)
    row.update(
        {
            "implementation": implementation,
            "scenario": "scenario",
            "estimand": "ate",
            "inference_scale": "identity",
        }
    )
    return pd.DataFrame([row])


@pytest.fixture(autouse=True)
def _scratch_in_tmp(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    runs = tmp_path_factory.mktemp("runs")
    monkeypatch.setattr(regenerate, "runs_root", lambda: runs)


def _run_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    independent_passed: bool,
    skip_properties: bool = False,
) -> dict[str, Any]:
    """A complete run of a pending record; returns what ``write_manifest`` was handed."""
    record = dataclasses.replace(_pending(), artifacts=tmp_path)
    study = SimpleNamespace(
        STUDY=record, PRIMARY_REPLICATES=record.replicates, PRIMARY_N=record.n, CONFIGURATION={}
    )
    arguments = argparse.Namespace(
        replicates=2,
        n=50,
        skip_properties=skip_properties,
        primary_only=False,
        allow_failures=False,
        output=tmp_path,
        jobs=1,
        cache=None,
    )
    seen: dict[str, Any] = {}

    def parsed(study: Any, here: Path, reference: Any) -> argparse.Namespace:
        seen["parser_reference"] = reference
        return arguments

    monkeypatch.setattr(regenerate, "_arguments", parsed)
    monkeypatch.setattr(
        regenerate, "_python_phase", lambda *args: regenerate._Phase(rows=_rows("cleverly"))
    )
    monkeypatch.setattr(
        regenerate, "_reference_rows", lambda *args, **kwargs: pytest.fail("a reference ran")
    )
    monkeypatch.setattr(
        regenerate.Reference, "run", lambda *args, **kwargs: pytest.fail("a container ran")
    )
    monkeypatch.setattr(
        regenerate, "equivalence", lambda *args, **kwargs: pytest.fail("rows were paired")
    )
    monkeypatch.setattr(regenerate, "validate_replicates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        regenerate, "summarize", lambda rows, truth_varies: pd.DataFrame({"summary": [1.0]})
    )
    monkeypatch.setattr(
        regenerate,
        "independent_performance_tests",
        lambda *args, **kwargs: pd.DataFrame(
            {"implementation": ["cleverly"], "passed": [independent_passed]}
        ),
    )
    if not skip_properties:
        monkeypatch.setattr(
            regenerate,
            "_property_artifacts",
            lambda *args, **kwargs: (
                pd.DataFrame({"passed": [True], "property_passed": [True]}),
                (),
            ),
        )

    def manifest(path: Path, record: Any, artifacts: Any, **options: Any) -> None:
        seen["record"] = record
        seen.update(options)

    monkeypatch.setattr(regenerate, "write_manifest", manifest)
    reference = regenerate.Reference("image", "run_study.R")
    seen["declared"] = reference.files(tmp_path)
    properties = SimpleNamespace(
        FAILURE_BOUND=0.01,
        generate_property_rows=lambda **kwargs: pytest.fail("the property study ran"),
    )
    regenerate.main(study, properties, here=tmp_path, reference=reference)
    seen["equivalence"] = pd.read_csv(tmp_path / "equivalence.csv")
    return seen


def test_a_pending_run_fits_no_reference_and_records_its_declared_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _run_pending(monkeypatch, tmp_path, independent_passed=True)
    assert seen["parser_reference"] is None, "a pending run must not offer the reference flags"
    assert seen["record"].comparator_state == "pending"
    assert seen["reference_files"] == seen["declared"]
    assert seen["equivalence"].empty


def test_a_pending_run_reuses_the_committed_property_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A property study run elsewhere and committed first survives the pending primary run."""
    committed = {
        "property-replicates.csv.gz": b"committed property rows",
        "properties.csv": b"property,cell,passed\nscore,a,True\n",
    }
    for name, payload in committed.items():
        (tmp_path / name).write_bytes(payload)
    seen = _run_pending(monkeypatch, tmp_path, independent_passed=True, skip_properties=True)
    assert seen["record"].comparator_state == "pending"
    assert seen["equivalence"].empty
    for name, payload in committed.items():
        assert (tmp_path / name).read_bytes() == payload


def test_a_pending_run_still_gates_the_subject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pending relaxes nothing about the Python evidence: a red independent gate refuses."""
    with pytest.raises(RuntimeError, match="independent performance"):
        _run_pending(monkeypatch, tmp_path, independent_passed=False)


def test_a_pending_study_must_name_its_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = _pending()
    study = SimpleNamespace(STUDY=record)
    with pytest.raises(RuntimeError, match="passes no Reference"):
        regenerate.main(study, SimpleNamespace(), here=tmp_path, reference=None)


def _committed(tmp_path: Path, *, replicates: int = 2, n: int = 50) -> Path:
    rows = pd.concat([_rows("cleverly"), _rows(REFERENCE)], ignore_index=True)
    rows.loc[0, "estimate"] = 0.1 + 0.2
    path = tmp_path / "replicates.csv.gz"
    regenerate.write_csv(rows, path, compression={"method": "gzip", "mtime": 0})
    digest = regenerate.hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        regenerate.json.dumps(
            {
                "configuration": {"replicates": replicates, "n": n},
                "sha256": {"replicates.csv.gz": digest},
            }
        ),
        encoding="utf-8",
    )
    return path


def _reuse(tmp_path: Path) -> regenerate._Phase:
    study = SimpleNamespace(STUDY=COMPARED)
    arguments = argparse.Namespace(replicates=2, n=50)
    return regenerate._committed_subject(study, arguments, tmp_path)  # type: ignore[arg-type]


def test_reuse_subject_reads_the_committed_subject_rows_bit_for_bit(tmp_path: Path) -> None:
    _committed(tmp_path)
    phase = _reuse(tmp_path)
    assert set(phase.rows["implementation"]) == {"cleverly"}
    assert phase.rows.loc[0, "estimate"] == 0.1 + 0.2
    assert phase.subject_failures is None


def test_reuse_subject_refuses_a_row_file_its_manifest_does_not_describe(tmp_path: Path) -> None:
    path = _committed(tmp_path)
    regenerate.write_csv(_rows("cleverly"), path, compression={"method": "gzip", "mtime": 0})
    with pytest.raises(RuntimeError, match="no longer matches its manifest hash"):
        _reuse(tmp_path)


def test_reuse_subject_refuses_another_budget(tmp_path: Path) -> None:
    _committed(tmp_path, replicates=3)
    with pytest.raises(RuntimeError, match="not 2 x n=50"):
        _reuse(tmp_path)


def test_reuse_subject_is_offered_only_when_no_reference_process_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = SimpleNamespace(STUDY=COMPARED, PRIMARY_REPLICATES=2, PRIMARY_N=50)
    monkeypatch.setattr("sys.argv", ["regenerate", "--reuse-subject"])
    assert regenerate._arguments(study, tmp_path, None).reuse_subject  # type: ignore[arg-type]
    with pytest.raises(SystemExit):
        regenerate._arguments(study, tmp_path, regenerate.Reference("i", "r"))  # type: ignore[arg-type]

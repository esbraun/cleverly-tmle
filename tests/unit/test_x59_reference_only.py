"""X59 section 3: the driver's reference-only mode, on a small synthetic study.

The mode reuses the committed subject rows, re-draws only the declared paired prefix from its
seeds, runs only the reference, and writes a manifest that carries the committed subject
provenance forward.  Docker is replaced by :func:`fake_reference_run`, which computes what an R
runner would from the archive it is handed.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tests.canonical import regenerate
from tests.studies.evidence import parity
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import ROOT
from tests.unit import _x59_reference_only_support as synthetic

PROPERTY_FILES = {
    "property-replicates.parquet": b"committed property rows",
    "properties.csv": b"property,cell,passed\nscore,a,True\n",
}
PROPERTIES = SimpleNamespace(FAILURE_BOUND=0.01)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    runs = tmp_path_factory.mktemp("runs")
    monkeypatch.setattr(regenerate, "runs_root", lambda: runs)
    monkeypatch.setattr(regenerate, "image_content", lambda image: "content:fake")
    monkeypatch.setattr(regenerate.Reference, "run", synthetic.fake_reference_run)
    monkeypatch.setattr(synthetic, "DRAWS_REFUSED", False)
    monkeypatch.setattr(synthetic, "RUNS", [])
    monkeypatch.delenv(synthetic.MUTATION_ENV, raising=False)
    monkeypatch.delenv(regenerate.DOCKER_MEMORY_ENV, raising=False)


def _reference() -> regenerate.Reference:
    # Real repository files, because the manifest hashes them by their repository path.
    canonical = ROOT / "tests" / "canonical"
    return regenerate.Reference(
        "image",
        "ltmle/run_study.R",
        mount_runner=True,
        build_context=canonical / "ltmle",
        runner_root=canonical,
    )


def _main(monkeypatch: pytest.MonkeyPatch, here: Path, *flags: str) -> None:
    monkeypatch.setattr(sys, "argv", ["regenerate", "--jobs", "1", *flags])
    regenerate.main(synthetic, PROPERTIES, here=here, reference=_reference())  # type: ignore[arg-type]


def _committed(monkeypatch: pytest.MonkeyPatch, here: Path) -> None:
    """The pending study's committed run: subject rows, property files and a manifest."""
    here.mkdir(parents=True, exist_ok=True)
    for name, payload in PROPERTY_FILES.items():
        (here / name).write_bytes(payload)
    _main(monkeypatch, here, "--skip-properties")


def _subject(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[rows["implementation"] == synthetic.SUBJECT].reset_index(drop=True)


def _reference_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[rows["implementation"] == synthetic.REFERENCE].reset_index(drop=True)


def test_reference_only_pairs_the_declared_prefix_and_keeps_the_subject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    committed_rows = read_rows(here / "replicates.parquet")
    committed_manifest = json.loads((here / "manifest.json").read_text())
    assert committed_manifest["comparator"]["status"] == "pending"

    _main(monkeypatch, here, "--reference-only")
    rows = read_rows(here / "replicates.parquet")
    pd.testing.assert_frame_equal(_subject(rows), _subject(committed_rows))
    reference = _reference_rows(rows)
    declared = synthetic.COMPARATOR_DECLARATION["reference_replicates"]
    assert {
        scenario: sorted(group["replicate"]) for scenario, group in reference.groupby("scenario")
    } == {scenario: list(range(count)) for scenario, count in declared.items()}
    for name, payload in PROPERTY_FILES.items():
        assert (here / name).read_bytes() == payload, f"{name} lost its committed bytes"

    manifest = json.loads((here / "manifest.json").read_text())
    assert manifest["generated_with"]["subject"] == committed_manifest["generated_with"]["subject"]
    assert manifest["study_module_sha256"] == committed_manifest["study_module_sha256"]
    assert manifest["generated_with"]["reference"]["implementation"] == synthetic.REFERENCE
    assert "comparator" not in manifest
    for name in regenerate.PARITY_ARTIFACTS:
        assert name in manifest["sha256"], f"the manifest does not hash {name}"

    paired = pd.read_csv(here / "equivalence.csv")
    assert dict(zip(paired["scenario"], paired["paired_replicates"], strict=True)) == declared
    table = pd.read_csv(here / "parity.csv")
    assert table["passed"].all(), table
    assert set(table["gate"]) == {"G-P", "G-C", "G1", "witness"}


def test_reference_only_gives_the_rows_of_a_full_run_at_the_same_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    _main(monkeypatch, here, "--reference-only")
    reference_only = _reference_rows(read_rows(here / "replicates.parquet"))

    full = tmp_path / "full"
    full.mkdir()
    for name, payload in PROPERTY_FILES.items():
        (full / name).write_bytes(payload)
    monkeypatch.setattr(synthetic, "STUDY", synthetic.compared())
    monkeypatch.setattr(synthetic, "COMPARATOR_DECLARATION", {})
    _main(monkeypatch, full, "--skip-properties")
    pd.testing.assert_frame_equal(
        _reference_rows(read_rows(full / "replicates.parquet")), reference_only
    )


def test_a_prefix_override_is_a_smoke_run_refused_into_the_committed_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    with pytest.raises(RuntimeError, match="smoke run"):
        _main(monkeypatch, here, "--reference-only", "--reference-prefix", "7")
    smoke = tmp_path / "smoke"
    _main(monkeypatch, here, "--reference-only", "--reference-prefix", "7", "--output", str(smoke))
    reference = _reference_rows(read_rows(smoke / "replicates.parquet"))
    assert set(reference["replicate"]) == set(range(7))
    assert len(reference) == 7 * len(synthetic.STUDY.scenarios)


def test_a_run_stopped_in_the_reference_resumes_without_drawing_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    seen: list[str] = []

    def stopped(*args: Any, **kwargs: Any) -> None:
        seen.append(regenerate.content_sha(args[2]))
        raise RuntimeError("the reference stopped: a rerun resumes")

    monkeypatch.setattr(regenerate.Reference, "run", stopped)
    with pytest.raises(RuntimeError, match="a rerun resumes"):
        _main(monkeypatch, here, "--reference-only")

    def resumed(self: Any, here_: Path, samples: Path, *args: Any, **kwargs: Any) -> None:
        # The same archive, so the R leaf key and its checkpoint directory are the same.
        seen.append(regenerate.content_sha(samples))
        synthetic.fake_reference_run(self, here_, samples, *args, **kwargs)

    monkeypatch.setattr(regenerate.Reference, "run", resumed)
    monkeypatch.setattr(synthetic, "DRAWS_REFUSED", True)
    _main(monkeypatch, here, "--reference-only")
    # Per scenario: the stopped run of the first, then its resumed run on the same archive.
    assert len(seen) == 1 + len(synthetic.STUDY.scenarios) and seen[0] == seen[1]
    assert not _reference_rows(read_rows(here / "replicates.parquet")).empty


def test_each_scenario_has_its_own_archive_in_replicate_order(tmp_path: Path) -> None:
    record = dataclasses.replace(synthetic.compared(), reference_replicates={"a": 3, "b": 2})
    inputs = regenerate.write_reference_inputs(synthetic, record, 60, tmp_path, jobs=1)
    assert list(inputs.scenarios) == ["a", "b"]
    for name, count in (("a", 3), ("b", 2)):
        held = inputs.scenarios[name]
        samples = pd.read_csv(held.samples)
        assert set(samples["scenario"]) == {name}
        assert samples["replicate"].drop_duplicates().tolist() == list(range(count))
        assert held.lines == 60
        assert pd.read_csv(held.truth)["replicate"].tolist() == list(range(count))
    assert len(pd.read_csv(inputs.paths["truth.csv"])) == 5
    frame, _ = synthetic.draw_scenario("b", 60, 1)
    drawn = pd.read_csv(inputs.scenarios["b"].samples)
    drawn = drawn.loc[drawn["replicate"] == 1, "x"]
    assert drawn.tolist() == frame["x"].tolist(), "a re-drawn replicate is not its seeded sample"


def test_each_scenario_runs_alone_with_a_chunk_of_one_replicate_per_worker_and_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    _main(monkeypatch, here, "--reference-only", "--reference-jobs", "3")
    chunks = [run[regenerate.CHUNK_LINES_ENV] for run in synthetic.RUNS]
    assert chunks == [str((3 + 1) * synthetic.PRIMARY_N)] * len(synthetic.STUDY.scenarios)
    assert all(run[synthetic.MUTATION_ENV] == "none" for run in synthetic.RUNS)


def test_a_second_mutation_draws_its_own_archive_and_reference_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    archives: list[str] = []
    original = synthetic.fake_reference_run

    def recorded(self: Any, here_: Path, samples: Path, *args: Any, **kwargs: Any) -> None:
        archives.append(regenerate.content_sha(samples))
        original(self, here_, samples, *args, **kwargs)

    monkeypatch.setattr(regenerate.Reference, "run", recorded)
    unmutated = tmp_path / "none"
    _main(
        monkeypatch, here, "--reference-only", "--reference-prefix", "7", "--output", str(unmutated)
    )
    monkeypatch.setenv(synthetic.MUTATION_ENV, "shift")
    mutated = tmp_path / "shift"
    _main(
        monkeypatch, here, "--reference-only", "--reference-prefix", "7", "--output", str(mutated)
    )
    scenarios = len(synthetic.STUDY.scenarios)
    assert len(archives) == 2 * scenarios, "the mutated run reused the unmutated reference rows"
    assert set(archives[:scenarios]).isdisjoint(archives[scenarios:])
    assert [run[synthetic.MUTATION_ENV] for run in synthetic.RUNS] == ["none"] * scenarios + [
        "shift"
    ] * scenarios
    before = _reference_rows(read_rows(unmutated / "replicates.parquet"))
    after = _reference_rows(read_rows(mutated / "replicates.parquet"))
    assert (after["estimate"] - before["estimate"]).round(9).eq(synthetic.SHIFT).all()
    manifest = json.loads((mutated / "manifest.json").read_text())
    reference = manifest["generated_with"]["reference"]
    assert reference["environment"] == {synthetic.MUTATION_ENV: "shift"}
    assert "tests/unit/_x59_reference_only_support.py" in reference["input_module_sha256"]


def test_a_declared_run_without_a_prefix_is_refused_before_any_draw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    monkeypatch.setattr(synthetic, "COMPARATOR_DECLARATION", {"paired_role": "descriptive"})
    monkeypatch.setattr(synthetic, "DRAWS_REFUSED", True)
    with pytest.raises(RuntimeError, match="declares no reference_replicates for"):
        _main(monkeypatch, here, "--reference-only")
    assert synthetic.RUNS == []


def test_an_undeclared_tolerance_is_refused_before_any_draw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    here = tmp_path / "study"
    _committed(monkeypatch, here)
    monkeypatch.setattr(synthetic, "PARITY", parity.ParityDeclaration(cells=synthetic.PARITY.cells))
    monkeypatch.setattr(synthetic, "DRAWS_REFUSED", True)
    with pytest.raises(ValueError, match="not declared yet"):
        _main(monkeypatch, here, "--reference-only")
    assert synthetic.RUNS == []


def test_the_memory_limit_reaches_the_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = tmp_path / "samples.csv.gz"
    command = regenerate._docker_command(
        _reference(), tmp_path, samples, tmp_path / "t.csv", tmp_path / "o.csv", "r.R", 2, None
    )
    assert "--memory" not in command
    monkeypatch.setenv(regenerate.DOCKER_MEMORY_ENV, "6000m")
    command = regenerate._docker_command(
        _reference(), tmp_path, samples, tmp_path / "t.csv", tmp_path / "o.csv", "r.R", 2, None
    )
    assert command[command.index("--memory") + 1] == "6000m"


def test_the_prefix_declaration_is_checked() -> None:
    record = synthetic.compared()
    with pytest.raises(ValueError, match="unknown scenarios"):
        dataclasses.replace(record, reference_replicates={"c": 3})
    with pytest.raises(ValueError, match="integer in"):
        dataclasses.replace(record, reference_replicates={"a": record.replicates + 1})
    assert record.replicates_of(synthetic.REFERENCE, "a") == 10
    assert record.replicates_of(synthetic.SUBJECT, "a") == record.replicates
    assert parity.ParityDeclaration(cells=()).tolerance["G-P"] is None


def test_every_draw_of_the_inputs_is_at_the_scenario_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The column probe once drew replicate 0 at n = 200, whose plug-in left the Q bound that
    # the declared size keeps (X59 smoke row 50).
    sizes: list[int] = []
    original = synthetic.reference_sample

    def recorded(scenario: str, n: int, replicate: int) -> Any:
        sizes.append(n)
        return original(scenario, n, replicate)

    monkeypatch.setattr(synthetic, "reference_sample", recorded)
    record = dataclasses.replace(synthetic.compared(), reference_replicates={"a": 2, "b": 2})
    regenerate.write_reference_inputs(synthetic, record, 60, tmp_path, jobs=1)
    assert sizes and set(sizes) == {60}


def test_a_smoke_pairs_a_subject_that_is_not_committed_yet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The parity records commit no subject before their declared run, so their smoke fits the
    # subject into its own directory and pairs the reference with it (X59 smoke row 50).
    here = tmp_path / "study"
    here.mkdir()
    for name, payload in PROPERTY_FILES.items():
        (here / name).write_bytes(payload)
    subject = tmp_path / "subject"
    _main(monkeypatch, here, "--skip-properties", "--output", str(subject))
    assert not (here / "manifest.json").exists()
    with pytest.raises(SystemExit):
        _main(monkeypatch, here, "--reference-only", "--subject", str(subject))
    smoke = tmp_path / "smoke"
    _main(
        monkeypatch, here, "--reference-only", "--reference-prefix", "7",
        "--subject", str(subject), "--output", str(smoke),
    )  # fmt: skip
    rows = read_rows(smoke / "replicates.parquet")
    assert _subject(rows).equals(_subject(read_rows(subject / "replicates.parquet")))
    assert set(_reference_rows(rows)["replicate"]) == set(range(7))

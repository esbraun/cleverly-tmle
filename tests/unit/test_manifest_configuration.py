"""A study's ``CONFIGURATION`` never names a key that the record writes to the manifest.

:func:`tests.studies.evidence.manifest.write_manifest` spreads the caller's ``configuration``
after the keys it writes from the record, so a colliding key would replace the record's value
unseen. The writer refuses such a key, and every registered study passes its own check.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.studies.evidence.manifest import RECORD_CONFIGURATION_KEYS, write_manifest
from tests.studies.evidence.registry import StudyRecord, registered

STUDIES = registered()


def _study_module(record: StudyRecord) -> ModuleType:
    """The module of ``record.modules`` whose ``STUDY`` is the record."""
    for path in record.modules:
        module = importlib.import_module(path.removesuffix(".py").replace("/", "."))
        study = getattr(module, "STUDY", None)
        if isinstance(study, StudyRecord) and study.slug == record.slug:
            return module
    raise LookupError(f"{record.slug}: no module of the record defines its STUDY")


def _record(tmp_path: Path) -> StudyRecord:
    return dataclasses.replace(STUDIES[0], artifacts=tmp_path)


@pytest.mark.parametrize("key", sorted(RECORD_CONFIGURATION_KEYS))
def test_a_configuration_key_that_the_record_owns_is_refused(tmp_path: Path, key: str) -> None:
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="which the record owns"):
        write_manifest(path, _record(tmp_path), [], configuration={key: "anything"})
    assert not path.exists()


def test_a_key_the_record_does_not_own_is_written_as_given(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    record = _record(tmp_path)
    write_manifest(path, record, [], configuration={"replicates_by_stream": {"cal": 2}})
    configuration = json.loads(path.read_text(encoding="utf-8"))["configuration"]
    assert configuration["replicates_by_stream"] == {"cal": 2}
    assert configuration["replicates"] == record.replicates


def test_every_record_owned_key_is_one_the_writer_can_write(tmp_path: Path) -> None:
    """Mutation control: the owned set names the writer's keys, not an unrelated list."""
    written: set[str] = set()
    for record in STUDIES:
        path = tmp_path / f"{record.slug}.json"
        write_manifest(path, dataclasses.replace(record, artifacts=tmp_path), [])
        written |= set(json.loads(path.read_text(encoding="utf-8"))["configuration"])
    assert written <= RECORD_CONFIGURATION_KEYS
    assert {"replicates", "n", "seed", "scenarios", "margins", "failure_bound"} <= written


@pytest.mark.parametrize("record", STUDIES, ids=[record.slug for record in STUDIES])
def test_no_registered_study_names_a_record_owned_key(record: StudyRecord) -> None:
    configuration = getattr(_study_module(record), "CONFIGURATION", None) or {}
    assert not RECORD_CONFIGURATION_KEYS & set(configuration), record.slug

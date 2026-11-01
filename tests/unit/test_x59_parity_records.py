"""X59 section 5: the two parity-only records, on their own seed streams.

Neither record is registered, so the registry's own checks never see them.  This module holds
what a registered study gets for free: the record validates once promoted, its stream is its
own, its inputs carry what the runner reads, and its empty property phase runs through the
driver.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest

from tests.canonical import regenerate
from tests.studies import canonical_long_horizon_lmtp_parity as lmtp_parity
from tests.studies import canonical_long_horizon_ltmle_parity as ltmle_parity
from tests.studies import long_horizon_parity_properties as properties
from tests.studies.evidence.registry import StudyRecord, registered
from tests.studies.evidence.seeds import replicate_seed

MODULES = (ltmle_parity, lmtp_parity)
IDS = [module.STUDY.slug for module in MODULES]


def _promoted(module) -> StudyRecord:  # type: ignore[no-untyped-def]
    record = module.STUDY
    return dataclasses.replace(
        record,
        reference=record.pending_reference,
        pending_reference=None,
        pending_reason="",
        **module.COMPARATOR_DECLARATION,
    )


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_record_is_parity_only_and_validates_once_promoted(module) -> None:  # type: ignore[no-untyped-def]
    assert module.STUDY.slug not in {study.slug for study in registered()}
    record = _promoted(module)
    assert record.paired_role == "descriptive"
    assert record.replicates == 500
    for cell in module.PARITY.cells:
        assert cell.estimand in record.scenarios[cell.scenario], cell
    assert all(not name.endswith("__bca") for names in record.scenarios.values() for name in names)


@pytest.mark.parametrize("module", MODULES, ids=IDS)
def test_the_seed_stream_is_its_own(module) -> None:  # type: ignore[no-untyped-def]
    """No replicate seed of the record is one a registered study, or the other record, draws."""
    record = module.STUDY

    def seeds(study: StudyRecord) -> set[int]:
        return {
            replicate_seed(study, scenario, replicate)
            for scenario in study.scenarios
            for replicate in range(20)
        }

    mine = seeds(record)
    others = [*registered()] + [other.STUDY for other in MODULES if other is not module]
    for study in others:
        assert record.seed != study.seed, study.slug
        assert not mine & seeds(study), f"{record.slug} shares a replicate seed with {study.slug}"


@pytest.mark.parametrize(
    ("module", "comparator"),
    [(ltmle_parity, "ltmle"), (lmtp_parity, "lmtp")],
    ids=IDS,
)
def test_the_reference_sample_names_its_comparator(module, comparator: str) -> None:  # type: ignore[no-untyped-def]
    for scenario in module.SCENARIOS:
        sample = module.reference_sample(scenario.name, scenario.n, 0)
        frame = sample.frame
        assert {"fold", "design__K", "design__sdr"} <= set(frame.columns)
        sdr = int(frame["design__sdr"].iloc[0])
        if comparator == "lmtp":
            assert sdr == int(module.COMPARATORS[scenario.name] == "lmtp_sdr")
            assert frame["fold"].nunique() == scenario.n_folds
            assert "ratio__mix__5" in frame.columns and "shift__mix__5__4" in frame.columns
        else:
            assert sdr == 0 and "abar__switch__5" in frame.columns
            assert set(sample.plugins) == set(module.STUDY.scenarios[scenario.name])


@pytest.mark.filterwarnings("ignore::tests.studies.fractional_glm.QuasiBinomialSeparationWarning")
def test_the_subject_phase_is_wald_only() -> None:
    phase = ltmle_parity.draw_and_fit(replicates=1, n=ltmle_parity.PRIMARY_N, n_jobs=1)
    assert set(phase.rows["estimand"]) == set(ltmle_parity.STUDY.scenarios["parity_binary_k5"])
    assert phase.samples is None, "the comparator re-draws its inputs from the seeds"


def test_the_empty_property_phase_runs_through_the_driver(tmp_path: Path) -> None:
    names = ("property-replicates.parquet", "properties.csv")
    paths = {name: tmp_path / name for name in names}
    arguments = argparse.Namespace(skip_properties=False, jobs=1)
    summary, failures = regenerate._property_artifacts(
        properties, arguments, tmp_path, paths, ltmle_parity.STUDY
    )
    assert failures == ()
    assert summary is not None and summary.empty
    assert all(path.exists() for path in paths.values())

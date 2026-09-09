"""Regenerate ordinary weighted end-of-study LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_weighted_ltmle, weighted_ltmle_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-ltmle-reference:1.3-0",
    runner="weighted_ltmle/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R", "ltmle_regimen_adapter.R"),
    build_context=ROOT / "tests" / "canonical" / "ltmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_weighted_ltmle,
        weighted_ltmle_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

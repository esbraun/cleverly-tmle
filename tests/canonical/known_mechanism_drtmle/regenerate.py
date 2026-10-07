"""Regenerate the known-treatment-mechanism DR-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_known_mechanism_drtmle, known_mechanism_drtmle_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-drtmle-reference:538a3a2",
    runner="known_mechanism_drtmle/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "drtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_known_mechanism_drtmle,
        known_mechanism_drtmle_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

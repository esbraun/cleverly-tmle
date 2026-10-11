"""Regenerate the per-arm outcome-adaptive C-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ctmle_oat_per_arm, ctmle_oat_per_arm_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-drtmle-reference:538a3a2",
    runner="drtmle_oat_per_arm/run_drtmle_oat_per_arm.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "drtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_ctmle_oat_per_arm,
        ctmle_oat_per_arm_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

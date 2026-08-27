"""Regenerate multi-arm DR-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_multi_arm_drtmle, multi_arm_drtmle_properties
from tests.studies.evidence.registry import ROOT

if __name__ == "__main__":
    main(
        canonical_multi_arm_drtmle,
        multi_arm_drtmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-drtmle-reference:538a3a2",
            runner="multi_arm_drtmle/run_multi_arm_drtmle.R",
            mount_runner=True,
            extra_files=("study_harness.R", "multi_arm_helpers.R"),
            build_context=ROOT / "tests" / "canonical" / "drtmle",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

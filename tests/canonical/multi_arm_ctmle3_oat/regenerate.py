"""Regenerate outcome-adaptive multi-arm C-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_multi_arm_ctmle_oat, multi_arm_ctmle_oat_properties
from tests.studies.evidence.registry import ROOT

if __name__ == "__main__":
    main(
        canonical_multi_arm_ctmle_oat,
        multi_arm_ctmle_oat_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-ctmle3-oat-reference:a4ea77b",
            runner="multi_arm_ctmle3_oat/run_multi_arm_ctmle3_oat.R",
            mount_runner=True,
            extra_files=("study_harness.R", "multi_arm_helpers.R"),
            build_context=ROOT / "tests" / "canonical" / "ctmle3_oat",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

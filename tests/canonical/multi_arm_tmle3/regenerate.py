"""Regenerate ordinary multi-arm TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_multi_arm_tmle, multi_arm_tmle_properties
from tests.studies.evidence.registry import ROOT

if __name__ == "__main__":
    main(
        canonical_multi_arm_tmle,
        multi_arm_tmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-tmle3-reference:ed72f8a",
            runner="multi_arm_tmle3/run_multi_arm_tmle3.R",
            mount_runner=True,
            extra_files=("study_harness.R", "multi_arm_helpers.R"),
            build_context=ROOT / "tests" / "canonical" / "tmle3",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

"""Regenerate the stacked arm-indexed MAR CV-TMLE evidence artifacts."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_mar_arm_indexed_cvtmle as study
from tests.studies import mar_arm_indexed_cvtmle_properties as properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-tmle-mar-arm-indexed-cvtmle:2.1.1",
    runner="tmle_mar_arm_indexed_cvtmle/run_study.R",
    mount_runner=True,
    extra_files=(
        "study_harness.R",
        "tmle_point_adapter.R",
        "tmle_continuous_point_adapter.R",
        "multi_arm_helpers.R",
        "tmle_mar_arm_indexed_cvtmle/probe_scale_workaround.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "tmle_mar",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        study,
        properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

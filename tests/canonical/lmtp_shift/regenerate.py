"""Regenerate continuous modified-treatment-policy evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_shift_policies, shift_policy_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="lmtp_shift/run_study.R",
    mount_runner=True,
    extra_files=("lmtp_point_adapter.R", "study_harness.R"),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_shift_policies,
        shift_policy_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

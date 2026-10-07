"""Regenerate policies at a held baseline treatment (``point-treatment-survival-policies``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_point_survival_policies, point_survival_policies_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="point_survival_policies_runner.R",
    mount_runner=True,
    extra_files=(
        "lmtp_held_survival_adapter.R",
        "lmtp_policy_adapter.R",
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_point_survival_policies,
        point_survival_policies_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

"""Regenerate longitudinal modified-treatment-policy evidence (``longitudinal-mtp``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_longitudinal_mtp, longitudinal_mtp_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="longitudinal_mtp_runner.R",
    mount_runner=True,
    extra_files=(
        "lmtp_mtp_adapter.R",
        "lmtp_policy_adapter.R",
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_longitudinal_mtp,
        longitudinal_mtp_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

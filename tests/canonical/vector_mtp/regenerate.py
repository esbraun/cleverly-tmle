"""Regenerate vector modified-treatment-policy evidence (``vector-mtp``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_vector_mtp, vector_mtp_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="vector_mtp_runner.R",
    mount_runner=True,
    extra_files=(
        "lmtp_vector_adapter.R",
        "lmtp_policy_adapter.R",
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_vector_mtp,
        vector_mtp_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

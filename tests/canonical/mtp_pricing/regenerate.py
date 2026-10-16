"""Regenerate boundary-binding price-plan evidence (``mtp-pricing``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_mtp_pricing, mtp_pricing_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="mtp_pricing_runner.R",
    mount_runner=True,
    extra_files=(
        "lmtp_mtp_adapter.R",
        "lmtp_policy_adapter.R",
        "lmtp_sdr_adapter.R",
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_mtp_pricing,
        mtp_pricing_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

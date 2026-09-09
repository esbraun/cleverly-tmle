"""Regenerate cross-fitted weighted end-of-study LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_weighted_ltmle_crossfit, weighted_ltmle_crossfit_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="weighted_lmtp_ltmle/run_study.R",
    mount_runner=True,
    extra_files=(
        "lmtp_crossfit_adapter.R",
        "lmtp_weighted_glm_adapter.R",
        "lmtp_crossfit/smoke_weighted.R",
        "study_harness.R",
        "ltmle_regimen_adapter.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_weighted_ltmle_crossfit,
        weighted_ltmle_crossfit_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

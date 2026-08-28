"""Regenerate ordinary categorical longitudinal TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_categorical_ltmle, categorical_ltmle_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="categorical_ltmle_runner.R",
    mount_runner=True,
    extra_files=("lmtp_crossfit_adapter.R", "study_harness.R"),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_categorical_ltmle,
        categorical_ltmle_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

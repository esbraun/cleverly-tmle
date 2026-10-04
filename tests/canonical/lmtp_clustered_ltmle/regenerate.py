"""Regenerate clustered cross-fitted end-of-study LTMLE evidence, paired with R lmtp 1.5.4."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import clustered_crossfit_ltmle, clustered_crossfit_ltmle_properties
from tests.studies.evidence.registry import ROOT

#: The digest-pinned ``lmtp`` image of the two cross-fitted studies, with ``ife`` 0.2.3.
REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="lmtp_clustered_ltmle/run_study.R",
    mount_runner=True,
    extra_files=(
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
        "ltmle_regimen_adapter.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        clustered_crossfit_ltmle,
        clustered_crossfit_ltmle_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

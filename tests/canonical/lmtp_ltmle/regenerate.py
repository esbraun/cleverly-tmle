"""Regenerate cross-fitted end-of-study LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ltmle_crossfit, ltmle_crossfit_properties
from tests.studies.evidence.registry import ROOT

#: One digest-pinned ``lmtp`` image serves both cross-fitted studies, so the Docker context
#: and the sourced adapter sit above either study's directory.  ``runner_root`` is what lets
#: the runner be named relative to that shared root.
REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="lmtp_ltmle/run_study.R",
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
        canonical_ltmle_crossfit,
        ltmle_crossfit_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

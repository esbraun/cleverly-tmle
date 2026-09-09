"""Regenerate clustered point-treatment CV-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_clustered_tmle, clustered_tmle_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-lmtp-clustered:1.5.4-ife0.2.3",
    runner="lmtp_clustered_tmle/run_study.R",
    mount_runner=True,
    extra_files=("lmtp_crossfit_adapter.R", "study_harness.R"),
    build_context=ROOT / "tests" / "canonical" / "lmtp_clustered_tmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_clustered_tmle,
        clustered_tmle_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

"""Regenerate ordinary missing-outcome TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_mar_tmle, mar_tmle_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-tmle-mar:2.1.1",
    runner="tmle_mar/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "tmle_mar",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_mar_tmle,
        mar_tmle_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

"""Regenerate MAR natural-course TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_mar_natural_course, mar_natural_course_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-tmle-mar-natural-course:2.1.1",
    runner="tmle_mar_natural_course/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R", "tmle_mar_natural_course/probe_scale_workaround.R"),
    build_context=ROOT / "tests" / "canonical" / "tmle_mar",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_mar_natural_course,
        mar_natural_course_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

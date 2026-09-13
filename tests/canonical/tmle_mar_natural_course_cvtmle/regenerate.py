"""Regenerate the stacked MAR natural-course CV-TMLE evidence artifacts."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_mar_natural_course_cvtmle as study
from tests.studies import mar_natural_course_cvtmle_properties as properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-tmle-mar-natural-course-cvtmle:2.1.1",
    runner="tmle_mar_natural_course_cvtmle/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "tmle_mar",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        study,
        properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

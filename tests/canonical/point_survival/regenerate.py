"""Regenerate point-treatment survival evidence (``point-treatment-survival``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_point_survival, point_survival_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-survtmle:1.1.1",
    runner="point_survival_runner.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "survtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_point_survival,
        point_survival_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

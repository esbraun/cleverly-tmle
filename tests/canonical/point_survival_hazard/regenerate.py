"""Regenerate per-horizon hazard targeting evidence (``point-survival-hazard``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_point_survival_hazard, point_survival_hazard_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-survtmle:1.1.1",
    runner="point_survival_hazard_runner.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "survtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_point_survival_hazard,
        point_survival_hazard_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

"""Regenerate one-step hazard targeting evidence (``point-survival-curve``)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_point_survival_curve, point_survival_curve_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-tmle3-survival:ed72f8a-glmnet5.0",
    runner="tmle3_survival/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "tmle3_survival",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_point_survival_curve,
        point_survival_curve_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

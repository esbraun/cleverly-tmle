"""Regenerate point-treatment MSM projection evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_point_msm, point_msm_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_point_msm,
        point_msm_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-tmle3-reference:ed72f8a",
            runner="tmle3_msm/run_study.R",
            mount_runner=True,
            extra_files=("study_harness.R",),
            build_context=ROOT / "tests" / "canonical" / "tmle3",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

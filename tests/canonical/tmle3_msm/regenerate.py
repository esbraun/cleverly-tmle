"""Regenerate point-treatment MSM projection evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_point_msm, point_msm_properties

if __name__ == "__main__":
    main(
        canonical_point_msm,
        point_msm_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(image="cleverly-tmle3-msm-reference:ed72f8a", runner="run_study.R"),
    )

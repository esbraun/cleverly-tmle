"""Regenerate ordinary longitudinal MSM projection evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_longitudinal_msm, longitudinal_msm_properties

if __name__ == "__main__":
    main(
        canonical_longitudinal_msm,
        longitudinal_msm_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-ltmle-msm-reference:1.3-0",
            runner="run_study.R",
            mount_runner=True,
        ),
    )

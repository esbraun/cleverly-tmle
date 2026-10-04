"""Regenerate cross-fitted longitudinal MSM evidence, paired with projected R lmtp 1.5.4 fits."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_crossfit_longitudinal_msm, crossfit_longitudinal_msm_properties
from tests.studies.evidence.registry import ROOT

#: The digest-pinned ``lmtp`` image of the cross-fitted longitudinal studies.
REFERENCE = Reference(
    image="cleverly-lmtp-crossfit:1.5.4",
    runner="lmtp_ltmle_msm/run_study.R",
    mount_runner=True,
    extra_files=(
        "lmtp_crossfit_adapter.R",
        "study_harness.R",
        "ltmle_regimen_adapter.R",
    ),
    build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    main(
        canonical_crossfit_longitudinal_msm,
        crossfit_longitudinal_msm_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

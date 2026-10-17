"""Regenerate the long-horizon SDR evidence; the R lmtp comparator is pending (X59)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_long_horizon_sdr, long_horizon_sdr_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_long_horizon_sdr,
        long_horizon_sdr_properties,
        here=Path(__file__).resolve().parent,
        # Declared for the manifest only: a pending comparator runs no reference process.
        reference=Reference(
            image="cleverly-lmtp-crossfit:1.5.4",
            runner="long_horizon_sdr/run_study.R",
            mount_runner=True,
            extra_files=("study_harness.R",),
            build_context=ROOT / "tests" / "canonical" / "lmtp_crossfit",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

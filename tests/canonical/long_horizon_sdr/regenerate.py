"""Regenerate the long-horizon SDR evidence; the R lmtp comparator runs reference-only (X59)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_long_horizon_sdr, long_horizon_sdr_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_long_horizon_sdr,
        long_horizon_sdr_properties,
        here=Path(__file__).resolve().parent,
        # The record is pending: a full run fits no reference, and --reference-only runs it
        # on the paired prefix of COMPARATOR_DECLARATION (X59).
        reference=Reference(
            image="cleverly-lmtp:9996b04",
            runner="long_horizon_sdr/run_study.R",
            mount_runner=True,
            extra_files=(
                "study_harness.R",
                "lmtp_sdr_adapter.R",
                "lmtp_crossfit_adapter.R",
                "lmtp_policy_adapter.R",
                "lmtp_9996b04_crossfit_adapter.R",
                "lmtp_9996b04_policy_adapter.R",
                "lmtp_horizon_learners.R",
                "long_horizon_lmtp_adapter.R",
            ),
            build_context=ROOT / "tests" / "canonical" / "lmtp_9996b04",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

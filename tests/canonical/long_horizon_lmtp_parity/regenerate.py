"""Regenerate the X59 parity record ``long-horizon-lmtp-parity``: the subject, then R lmtp 9996b04 reference-only."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_long_horizon_lmtp_parity, long_horizon_parity_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_long_horizon_lmtp_parity,
        long_horizon_parity_properties,
        here=Path(__file__).resolve().parent,
        # The record is pending: the first run fits the subject, and --reference-only then
        # fits the comparator on the same seeds (X59 section 5).
        reference=Reference(
            image="cleverly-lmtp:9996b04",
            runner="long_horizon_lmtp_parity/run_study.R",
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

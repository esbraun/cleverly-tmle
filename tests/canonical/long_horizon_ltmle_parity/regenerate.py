"""Regenerate the X59 parity record ``long-horizon-ltmle-parity``: the subject, then R ltmle 1.3-0 reference-only."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_long_horizon_ltmle_parity, long_horizon_parity_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_long_horizon_ltmle_parity,
        long_horizon_parity_properties,
        here=Path(__file__).resolve().parent,
        # The record is pending: the first run fits the subject, and --reference-only then
        # fits the comparator on the same seeds (X59 section 5).
        reference=Reference(
            image="cleverly-ltmle-reference:1.3-0",
            runner="long_horizon_ltmle/run_study.R",
            mount_runner=True,
            extra_files=("study_harness.R", "ltmle_horizon_adapter.R"),
            build_context=ROOT / "tests" / "canonical" / "ltmle",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

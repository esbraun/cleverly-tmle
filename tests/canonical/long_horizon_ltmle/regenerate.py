"""Regenerate the long-horizon ordinary LTMLE evidence; the R ltmle 1.3-0 comparator runs reference-only (X59)."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_long_horizon_ltmle, long_horizon_ltmle_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_long_horizon_ltmle,
        long_horizon_ltmle_properties,
        here=Path(__file__).resolve().parent,
        # --reference-only refits the reference alone on the record's paired prefix
        # (reference_replicates), against the committed subject rows (X59).
        reference=Reference(
            image="cleverly-ltmle-reference:1.3-0",
            runner="long_horizon_ltmle/run_study.R",
            mount_runner=True,
            extra_files=("study_harness.R", "ltmle_horizon_adapter.R"),
            build_context=ROOT / "tests" / "canonical" / "ltmle",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

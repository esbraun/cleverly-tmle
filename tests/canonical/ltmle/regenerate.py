"""Regenerate ordinary end-of-study LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ltmle, ltmle_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_ltmle,
        ltmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-ltmle-reference:1.3-0",
            runner="ltmle/run_study.R",
            mount_runner=True,
            extra_files=(
                "ltmle/generate_reference.R",
                "study_harness.R",
                "ltmle_regimen_adapter.R",
            ),
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

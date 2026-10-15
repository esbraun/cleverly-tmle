"""Regenerate repeated-outcome LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ltmle_repeated, ltmle_repeated_properties

ROOT = Path(__file__).parents[3]

if __name__ == "__main__":
    main(
        canonical_ltmle_repeated,
        ltmle_repeated_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-ltmle-reference:1.3-0",
            runner="ltmle_repeated/run_study.R",
            mount_runner=True,
            extra_files=("study_harness.R", "ltmle_repeated_adapter.R"),
            build_context=ROOT / "tests" / "canonical" / "ltmle",
            runner_root=ROOT / "tests" / "canonical",
        ),
    )

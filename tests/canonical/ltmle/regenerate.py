"""Regenerate ordinary end-of-study LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ltmle, ltmle_properties

if __name__ == "__main__":
    main(
        canonical_ltmle,
        ltmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-ltmle-reference:1.3-0",
            runner="run_study.R",
            mount_runner=True,
            extra_files=("generate_reference.R",),
        ),
    )

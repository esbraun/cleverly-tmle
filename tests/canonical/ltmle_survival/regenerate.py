"""Regenerate ordinary survival-curve LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ltmle_survival, ltmle_survival_properties

if __name__ == "__main__":
    main(
        canonical_ltmle_survival,
        ltmle_survival_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-ltmle-survival-reference:1.3-0",
            runner="run_study.R",
            mount_runner=True,
        ),
    )

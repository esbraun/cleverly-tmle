"""Regenerate the canonical DR-TMLE evidence study."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_drtmle, drtmle_properties

if __name__ == "__main__":
    main(
        canonical_drtmle,
        drtmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-drtmle-reference:538a3a2",
            runner="run_drtmle.R",
        ),
    )

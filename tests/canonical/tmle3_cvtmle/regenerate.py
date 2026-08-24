"""Regenerate stacked point-treatment CV-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_cvtmle, stacked_cvtmle_properties

if __name__ == "__main__":
    main(
        canonical_cvtmle,
        stacked_cvtmle_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image="cleverly-tmle3-cvtmle-reference:ed72f8a", runner="run_tmle3_cvtmle.R"
        ),
    )

"""Regenerate selector-based C-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_ctmle_selector, ctmle_selector_properties

if __name__ == "__main__":
    main(
        canonical_ctmle_selector,
        ctmle_selector_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(image="cleverly-ctmle-reference:18de559", runner="run_ctmle.R"),
    )

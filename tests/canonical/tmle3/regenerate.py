"""Regenerate canonical point-treatment TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_properties, canonical_tmle

if __name__ == "__main__":
    main(
        canonical_tmle,
        canonical_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(image="cleverly-tmle3-reference:ed72f8a", runner="run_tmle3.R"),
    )

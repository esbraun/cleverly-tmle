"""Regenerate cross-fitted end-of-study LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_ltmle_crossfit, ltmle_crossfit_properties

if __name__ == "__main__":
    main(
        canonical_ltmle_crossfit,
        ltmle_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

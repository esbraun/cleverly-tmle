"""Regenerate clustered point-treatment CV-TMLE evidence at unequal cluster sizes."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import clustered_unequal_cvtmle, clustered_unequal_properties

if __name__ == "__main__":
    main(
        clustered_unequal_cvtmle,
        clustered_unequal_properties,
        here=Path(__file__).resolve().parent,
    )

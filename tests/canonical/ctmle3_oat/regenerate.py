"""Regenerate outcome-adaptive C-TMLE evidence."""

from pathlib import Path

from tests.canonical.ctmle_regenerate import main
from tests.studies import canonical_ctmle_oat, ctmle_oat_properties

if __name__ == "__main__":
    main(
        canonical_ctmle_oat,
        ctmle_oat_properties,
        here=Path(__file__).resolve().parent,
        image="cleverly-ctmle3-oat-reference:a4ea77b",
        runner="run_ctmle3_oat.R",
    )

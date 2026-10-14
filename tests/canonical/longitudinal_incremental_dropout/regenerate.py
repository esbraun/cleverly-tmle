"""Regenerate the longitudinal IPSI dropout evidence (``longitudinal-incremental-dropout``)."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import (
    canonical_longitudinal_incremental_dropout,
    longitudinal_incremental_dropout_properties,
)

if __name__ == "__main__":
    main(
        canonical_longitudinal_incremental_dropout,
        longitudinal_incremental_dropout_properties,
        here=Path(__file__).resolve().parent,
    )

"""Regenerate repeated-outcome longitudinal IPSI evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_ipsi_repeated, ipsi_repeated_properties

if __name__ == "__main__":
    main(
        canonical_ipsi_repeated,
        ipsi_repeated_properties,
        here=Path(__file__).resolve().parent,
    )

"""Regenerate tuned-cap evidence on law Q (``mtp-tuning``)."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_mtp_tuning, mtp_tuning_properties

if __name__ == "__main__":
    main(
        canonical_mtp_tuning,
        mtp_tuning_properties,
        here=Path(__file__).resolve().parent,
    )

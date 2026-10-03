"""Regenerate the default simultaneous-band evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import default_band_properties, default_bands

if __name__ == "__main__":
    main(default_bands, default_band_properties, here=Path(__file__).resolve().parent)

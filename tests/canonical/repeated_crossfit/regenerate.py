"""Regenerate repeated point-treatment CV-TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import repeated_crossfit, repeated_crossfit_properties

if __name__ == "__main__":
    main(
        repeated_crossfit,
        repeated_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

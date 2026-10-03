"""Regenerate the full-refit bootstrap and derived-contrast evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_full_refit_bootstrap, full_refit_bootstrap_properties

if __name__ == "__main__":
    main(
        canonical_full_refit_bootstrap,
        full_refit_bootstrap_properties,
        here=Path(__file__).resolve().parent,
    )

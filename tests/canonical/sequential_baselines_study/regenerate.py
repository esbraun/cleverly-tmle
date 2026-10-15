"""Regenerate the sequential g-computation and IPW evidence (``sequential-baselines``)."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_sequential_baselines, sequential_baselines_properties

if __name__ == "__main__":
    main(
        canonical_sequential_baselines,
        sequential_baselines_properties,
        here=Path(__file__).resolve().parent,
        reference=None,
    )

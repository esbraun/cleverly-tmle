"""Regenerate known-stochastic-regime evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_stochastic_regimes, stochastic_regime_properties

if __name__ == "__main__":
    main(
        canonical_stochastic_regimes,
        stochastic_regime_properties,
        here=Path(__file__).resolve().parent,
    )

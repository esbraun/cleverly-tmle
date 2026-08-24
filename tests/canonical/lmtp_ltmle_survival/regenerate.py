"""Regenerate cross-fitted survival-curve LTMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_ltmle_survival_crossfit, ltmle_survival_crossfit_properties

if __name__ == "__main__":
    main(
        canonical_ltmle_survival_crossfit,
        ltmle_survival_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

"""Regenerate cross-fitted point-treatment survival evidence (``point-treatment-survival-crossfit``)."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_point_survival_crossfit, point_survival_crossfit_properties

if __name__ == "__main__":
    main(
        canonical_point_survival_crossfit,
        point_survival_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

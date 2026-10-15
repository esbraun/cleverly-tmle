"""Regenerate cross-fitted hazard targeting evidence (``point-survival-curve-crossfit``)."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import (
    canonical_point_survival_curve_crossfit,
    point_survival_curve_crossfit_properties,
)

if __name__ == "__main__":
    main(
        canonical_point_survival_curve_crossfit,
        point_survival_curve_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

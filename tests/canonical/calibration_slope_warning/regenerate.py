"""Regenerate the calibration-slope warning evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import calibration_slope_warning, calibration_slope_warning_properties

if __name__ == "__main__":
    main(
        calibration_slope_warning,
        calibration_slope_warning_properties,
        here=Path(__file__).resolve().parent,
    )

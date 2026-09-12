"""Regenerate the stacked MAR natural-course CV-TMLE evidence artifacts."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_mar_natural_course_cvtmle as study
from tests.studies import mar_natural_course_cvtmle_properties as properties

if __name__ == "__main__":
    main(study, properties, here=Path(__file__).resolve().parent)

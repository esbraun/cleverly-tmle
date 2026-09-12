"""Regenerate MAR natural-course TMLE evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_mar_natural_course, mar_natural_course_properties

if __name__ == "__main__":
    main(
        canonical_mar_natural_course,
        mar_natural_course_properties,
        here=Path(__file__).resolve().parent,
    )

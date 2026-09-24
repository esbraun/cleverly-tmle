"""Regenerate the omitted-variable bound standard-error evidence."""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import omitted_variable_bound, omitted_variable_bound_properties

if __name__ == "__main__":
    main(
        omitted_variable_bound,
        omitted_variable_bound_properties,
        here=Path(__file__).resolve().parent,
    )

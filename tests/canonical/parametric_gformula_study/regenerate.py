"""Regenerate the parametric g-formula evidence (``parametric-gformula``).

The record declares no comparator, so this driver runs no reference process.  The
``pygformula`` comparison is a fixture check that ``compare_pygformula.py`` runs in its own
image (``README.md``, "Comparator").
"""

from pathlib import Path

from tests.canonical.regenerate import main
from tests.studies import canonical_parametric_gformula, parametric_gformula_properties

if __name__ == "__main__":
    main(
        canonical_parametric_gformula,
        parametric_gformula_properties,
        here=Path(__file__).resolve().parent,
    )

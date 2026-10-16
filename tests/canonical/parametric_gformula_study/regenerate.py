"""Regenerate the parametric g-formula evidence (``parametric-gformula``).

The record still declares ``pending_reference``, so this driver runs no reference process and
the manifest records the bytes of the declared comparator sources.  The comparator itself is
implemented: ``compare_pygformula.py`` runs it in its own image on the fixture panels
(``README.md``, "Comparator").  The ``Reference`` below is the one that module runs.
"""

from pathlib import Path

from tests.canonical.parametric_gformula_study.compare_pygformula import REFERENCE
from tests.canonical.regenerate import main
from tests.studies import canonical_parametric_gformula, parametric_gformula_properties

if __name__ == "__main__":
    main(
        canonical_parametric_gformula,
        parametric_gformula_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

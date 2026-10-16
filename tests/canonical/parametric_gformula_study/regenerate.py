"""Regenerate the parametric g-formula evidence (``parametric-gformula``).

The comparator ``pygformula`` is pending: the record declares ``pending_reference``, so the
driver runs no reference process.  The ``Reference`` below declares the sources whose bytes the
manifest records, and the later comparator change moves the record to ``reference``.
"""

from pathlib import Path

from tests.canonical.regenerate import Reference, main
from tests.studies import canonical_parametric_gformula, parametric_gformula_properties

if __name__ == "__main__":
    main(
        canonical_parametric_gformula,
        parametric_gformula_properties,
        here=Path(__file__).resolve().parent,
        reference=Reference(
            image=f"cleverly-pygformula-reference:{canonical_parametric_gformula.PYGFORMULA}",
            runner="run_pygformula.py",
            mount_runner=True,
            extra_files=("requirements.txt",),
            interpreter="python",
        ),
    )

"""Regenerate repeated point-treatment CV-TMLE evidence under the declared run form.

The study is Python only and has no comparator.  A declared run passes ``--output`` (an empty
scratch directory outside the repository) and optionally ``--jobs`` and ``--fresh``, and
nothing else.  :mod:`tests.canonical.declared_run` describes the guard, the run log and the
copy into this directory.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import repeated_crossfit, repeated_crossfit_properties

if __name__ == "__main__":
    run(
        repeated_crossfit,
        repeated_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

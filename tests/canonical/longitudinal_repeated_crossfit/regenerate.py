"""Regenerate repeated cross-fitted longitudinal evidence under the declared run form.

The study is Python only and has no comparator.  A declared run passes ``--output`` (an empty
scratch directory outside the repository) and optionally ``--jobs`` and ``--fresh``, and
nothing else.  :mod:`tests.canonical.declared_run` describes the guard, the run log and the
copy into this directory.  A stopped run resumes when it is run again from the same commit
with a new empty ``--output``: the driver keeps every finished primary and property replicate
as a checkpoint in the run's scratch.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import longitudinal_repeated_crossfit, longitudinal_repeated_crossfit_properties

if __name__ == "__main__":
    run(
        longitudinal_repeated_crossfit,
        longitudinal_repeated_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

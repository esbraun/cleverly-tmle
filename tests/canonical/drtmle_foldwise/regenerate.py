"""Regenerate the fold-wise DR-TMLE evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory.  The study has no R
comparator, so the run starts no container.  A killed run resumes when the same command runs
again at the same commit with a new empty ``--output``.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import drtmle_foldwise, drtmle_foldwise_properties

if __name__ == "__main__":
    run(
        drtmle_foldwise,
        drtmle_foldwise_properties,
        here=Path(__file__).resolve().parent,
    )

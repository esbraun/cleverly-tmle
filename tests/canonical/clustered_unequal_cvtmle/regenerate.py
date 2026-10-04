"""Regenerate clustered CV-TMLE evidence at unequal cluster sizes under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs``, and nothing else.  :mod:`tests.canonical.declared_run` describes the
guard, the run log and the copy into this directory.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import clustered_unequal_cvtmle, clustered_unequal_properties

if __name__ == "__main__":
    run(
        clustered_unequal_cvtmle,
        clustered_unequal_properties,
        here=Path(__file__).resolve().parent,
    )

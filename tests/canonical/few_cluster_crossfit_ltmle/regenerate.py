"""Regenerate the cross-fitted clustered LTMLE evidence at few clusters under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs``, and nothing else.  :mod:`tests.canonical.declared_run` describes the
guard, the run log and the copy into this directory.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import few_cluster_crossfit_ltmle, few_cluster_crossfit_properties

if __name__ == "__main__":
    run(
        few_cluster_crossfit_ltmle,
        few_cluster_crossfit_properties,
        here=Path(__file__).resolve().parent,
    )

"""Regenerate the evidence of the fold-wise DR-TMLE default on the RM18 binary draws (Fix-6 unit B).

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory.  The study has no R
comparator, so the run starts no container.  The study module refuses a declared run while its
budgets are planning values.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import drtmle_foldwise_n1500

if __name__ == "__main__":
    run(drtmle_foldwise_n1500, drtmle_foldwise_n1500, here=Path(__file__).resolve().parent)

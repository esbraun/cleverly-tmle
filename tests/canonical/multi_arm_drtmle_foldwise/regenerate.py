"""Regenerate the evidence of multi-arm DR-TMLE with fold-wise reduced regressions (Fix-6 unit M).

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory.  The study has no R
comparator, so the run starts no container.  The study module refuses a declared run while its
budgets are planning values.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import multi_arm_drtmle_foldwise

if __name__ == "__main__":
    run(multi_arm_drtmle_foldwise, multi_arm_drtmle_foldwise, here=Path(__file__).resolve().parent)

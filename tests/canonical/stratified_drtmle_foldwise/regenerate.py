"""Regenerate the evidence of stratified fold-wise DR-TMLE (Fix-6 unit S).

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory.  The study has no R
comparator, so the run starts no container.  The primary phase and the contraction ladder
both refuse a declared run while the record's budgets are planning values.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import stratified_drtmle_foldwise, stratified_drtmle_foldwise_ladder

if __name__ == "__main__":
    run(
        stratified_drtmle_foldwise,
        stratified_drtmle_foldwise_ladder,
        here=Path(__file__).resolve().parent,
    )

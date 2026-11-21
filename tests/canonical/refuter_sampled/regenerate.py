"""Regenerate the sampled refutation evidence (X106 part H) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container. The cells, trial counts, law, learners, defect
levels, gates and seeds are those of :mod:`tests.studies.refuter_sampled`, fixed before the run.

Red-cell reading (declared before any run, comparison plan section 6).

* A red absolute size or power cell of a full cell (I2-F) is a finding on the X82 rule, routed
  to the refutation owner, not to X106 H.
* A red relative or absolute cell of a sampled cell whose full partner is green is a defect
  suspect in the sampled path. It is diagnosed first, by W1's identity on the failing trials'
  seeds (a sampled call against ``refute()`` of the estimator fitted on the sampled rows).
* A mutation cell that is not red means the study cannot see the full-data reference it is
  built to detect; the sampled claim is withheld until the study is repaired under a fresh
  declaration.
* A verdict-scope break (an I2 trial without a printed verdict, an I1 or clustered trial with
  one) stops the run.
* No trial count, law, threshold, defect size or seed moves after a verdict is seen. No rerun
  buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import refuter_sampled, refuter_sampled_properties

if __name__ == "__main__":
    run(refuter_sampled, refuter_sampled_properties, here=Path(__file__).resolve().parent)

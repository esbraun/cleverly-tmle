"""Regenerate the adaptive TMLE evidence (``adaptive-tmle-trial``) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard (a clean tree, ``HEAD`` equal to its upstream, every thread variable 1), the
run log and the copy into this directory. A smoke run passes ``--replicates`` other than the
declared budget, and writes to a scratch ``--output`` outside the repository.

The R ``atmle`` comparator is not registered on this record (``COMPARATOR`` in the study
module says why), so the driver starts no reference process.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import adaptive_tmle_trial, adaptive_tmle_trial_properties

if __name__ == "__main__":
    run(
        adaptive_tmle_trial,
        adaptive_tmle_trial_properties,
        here=Path(__file__).resolve().parent,
    )

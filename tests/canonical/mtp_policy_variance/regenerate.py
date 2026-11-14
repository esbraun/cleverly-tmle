"""Regenerate the ``mtp-policy-variance`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator (R ``lmtp`` has no policy-weighted variance), so the run starts no container.
The twins, the fit sets, the learners, the controls, the replication counts and the declared
readings are those of :mod:`tests.studies.canonical_mtp_policy_variance` and
:mod:`tests.studies.mtp_policy_variance_properties`, fixed before the run; the red-cell reading
is the driver's module docstring.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_mtp_policy_variance, mtp_policy_variance_properties

if __name__ == "__main__":
    run(
        canonical_mtp_policy_variance,
        mtp_policy_variance_properties,
        here=Path(__file__).resolve().parent,
    )

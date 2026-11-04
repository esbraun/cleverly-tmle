"""Regenerate the AdaptiveIPTW pairing with R ``drtmle`` (``adaptive-iptw-pairing``).

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.
:mod:`tests.canonical.declared_run` describes the guard, the run log and the copy into this
directory.

The Python phase fits every scenario and writes, per replicate, the raw columns, the fold
permutation, the raw mechanism factors and each key's initial mechanism and ``Qr``.  The R
phase (``run_study.R`` in the pinned ``drtmle`` image) composes each key's mechanism from the
factors, checks it against the package's, and calls ``drtmle::adaptive_iptw`` on it.

**Reading** (declared before any run, ``X21-pairing-plan.md`` rev 2 in the plans repository).

* A fatal runner check (fold, composition, curve or stub) stops the run; it is a defect of
  the runner or the package, never a red row.
* A nuisance-gate, range or convergence miss is a recorded reference failure of the
  replicate.
* ``agreement`` (``canonical_adaptive_iptw_pairing.agreement``) is the pairing verdict: a key
  whose 99th-percentile gap passes its bound agrees.  A key that fails it is a defect suspect.
* A key whose mean targeting movement is under ``WITNESS_MULTIPLE`` times its bound is labelled
  "targeting not witnessed" on the page; no verdict reads the label.
* No bound, size, replicate count, law, learner, fold count or seed moves after a verdict is
  seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.canonical.regenerate import Reference
from tests.studies import adaptive_iptw_pairing_properties, canonical_adaptive_iptw_pairing
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-drtmle-reference:538a3a2",
    runner="adaptive_iptw_pairing/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "drtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    run(
        canonical_adaptive_iptw_pairing,
        adaptive_iptw_pairing_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

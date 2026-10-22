"""Regenerate the AdaptiveIPTW gate evidence (``adaptive-iptw``) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard (a clean tree, ``HEAD`` equal to its upstream, every thread variable 1), the
run log and the copy into this directory.

**Resume.**  A run keeps its checkpoints in ``~/.cache/cleverly/runs/adaptive-iptw/<key>``.  The
key holds the declaration hash, the git state and the package versions.  A run that stops (a
reclaimed container that keeps ``~/.cache``, a kill or a crash) resumes when the same command
runs again at the same commit with a new empty ``--output``.  ``run.log`` then records what the
run reused.  ``--fresh`` discards the checkpoints first.  A container that loses ``~/.cache``
restarts the run: the full Python run costs about 4 CPU-hours.

The R ``drtmle`` comparator is pending (``STUDY.pending_reference``).  The driver starts no
reference process.  :data:`REFERENCE` names the pinned image and its runner, which the manifest
records as declared sources.  They are a placeholder until ``adaptive-iptw-pairing`` lands and
clears ``pending_reference`` and ``comparison``.

**Red-row reading** (declared before any run, ``X21-plan.md`` 3.5 rev 3.9 and ``AGENT-BRIEF.md``,
"Reading a red calibration row").  :data:`tests.studies.canonical_adaptive_iptw.RED_READING`
holds the text.

* A red primary row blocks the merge and gets a diagnosis.
* A red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage interval contains 0.95 or lies inside [0.92, 0.98], and the diagnosis
  finds no defect.  It then moves to ``reporting`` with an owner row.
* A red positive bias, ``root_n_points`` or targeted necessity row is a finite-sample red only
  when the 99% interval of its standardized bias contains the design-predicted ``b``, the other
  rows of its scene agree with their own ``b``, and the diagnosis finds no defect.
* A red displacement row, or a control that does not fail, is a defect suspect and never a
  finite-sample red.
* About 3.4 red rows are expected by chance in a correct run (2.0 calibration and 1.5 bias,
  ladder and necessity rows, ``EXPECTED_CHANCE_REDS``).  That count is no defect.
* No margin, sample size, replicate count, law, learner, fold count or seed moves after a verdict
  is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.canonical.regenerate import Reference
from tests.studies import adaptive_iptw_properties, canonical_adaptive_iptw
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image="cleverly-drtmle-reference:538a3a2",
    runner="drtmle/run_drtmle.R",
    mount_runner=True,
    build_context=ROOT / "tests" / "canonical" / "drtmle",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    run(
        canonical_adaptive_iptw,
        adaptive_iptw_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

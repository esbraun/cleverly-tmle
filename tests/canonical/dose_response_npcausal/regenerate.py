"""Regenerate the dose-response comparison with npcausal ``ctseff`` (X93 S2), declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The law, learners, bandwidths, size and margins are those of
:mod:`tests.studies.dose_response_npcausal`, fixed before any run from the committed pilot
(:mod:`tests.studies.dose_response_npcausal_pilot`).  The reference runs in the image the
incremental studies build from ``tests/canonical/npcausal_incremental`` (npcausal 0.1.0 at
``56a5ac1`` on R 4.5.2), unchanged, with ``run_study.R`` mounted at ``/fixture``.

Smoke before the declared run.  A smoke passes ``--replicates`` other than the declared count,
so the driver skips the (empty) property phase and refuses an ``--output`` in the repository::

    python -m tests.canonical.dose_response_npcausal.regenerate --replicates 20 \
        --output <scratch> --allow-failures --jobs <workers>

The smoke must confirm, before the declared run, what the declaration took from the Python
pilot because R could not run there: that ``ctseff`` returns finite rows at every pair, that its
estimates sit near the Python transcription's on the same draw (the transcription interpolates
no binned fit, so expect agreement to a few hundredths of a standard error, not to rounding),
and that a call costs seconds.  Read them off the smoke's ``replicates.parquet`` and
``run.log``.  The smoke cannot judge a coverage or a paired verdict at 20 replicates.

Projected paired verdicts (declared before any run, from the pilot with the transcription
standing in for ``ctseff``).  At dose 3.5 the subject's reported SE runs above its spread
(pilot SE ratio 1.07 at both bandwidths, against 1.00 and 1.01 for the transcription), the
conservatism the ``dose-response`` study already reports at interior pairs.  The calibration
leg there reads a calibration excess near 0.06 against the margin 0.05, so those two paired cells
are projected red (``inconclusive``) whatever the count.  The other four project green on every
leg, and the RMSE leg's projected bounds stay at or below 1.09.

Red-cell readings (declared before any run).  The record is ``gated``.  No cell is re-sized,
re-drawn or downgraded after a verdict is seen; a red that the diagnosis finds is not a defect
moves the record to ``reporting`` with an owner row and the reading below, and the run is
repeated under the new commit on the same seeds.

* A red subject truth row is a defect suspect of ``DoseResponseTMLE``: the truths are exact,
  the outcome learner holds the law's regression exactly and the doses are interior.  Diagnose
  it against ``tests/unit/test_dose_response.py`` and the ``dose-response`` study.
* A red ``npcausal`` truth row is a finding about the reference at this law and n, published
  under ``reporting`` with ``accepted_reference_failure`` naming it; it does not turn the
  subject's row red (``tests.studies.evidence.comparison`` keeps the two questions apart).  The
  Python transcription read ``ctseff``'s coverage at each pair from the pilot draws
  (``tests/unit/test_dose_response_npcausal_design.py``).
* A red similarity leg is a defect suspect of whichever side's truth row is also off.  With
  both truth rows green it reads as the two estimators' finite-sample difference on one target
  (different kernels' shapes, density estimators and fold use) and takes the route above.
* A red RMSE leg with both truth rows green reads as the two estimators' variance difference at
  matched roughness: the route above, with the measured ratio stated.
* A red calibration leg at dose 3.5 whose subject SE ratio lies above 1 (conservative) while
  the subject's coverage and truth row are green is the projected reading above: the subject's
  conservative influence-curve SE at that dose, not a defect.  It takes the route above, with
  the measured ratios stated.  A red calibration leg with the subject's SE ratio below 1, or
  outside the sanity band, is a variance defect suspect.
* A replication that raises is never redrawn, and the failure record governs: a recordable
  subject error or an R error inside ``study_stage`` costs one replicate of its side, read
  within the declared 1% bound.  An error outside a stage stops the run.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.canonical.regenerate import Reference
from tests.studies import dose_response_npcausal, dose_response_npcausal_properties
from tests.studies.evidence.registry import ROOT

REFERENCE = Reference(
    image=dose_response_npcausal.NPCAUSAL_IMAGE,
    runner="dose_response_npcausal/run_study.R",
    mount_runner=True,
    extra_files=("study_harness.R",),
    build_context=ROOT / "tests" / "canonical" / "npcausal_incremental",
    runner_root=ROOT / "tests" / "canonical",
)

if __name__ == "__main__":
    run(
        dose_response_npcausal,
        dose_response_npcausal_properties,
        here=Path(__file__).resolve().parent,
        reference=REFERENCE,
    )

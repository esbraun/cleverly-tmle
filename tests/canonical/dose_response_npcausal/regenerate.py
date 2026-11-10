"""Regenerate the dose-response comparison with npcausal ``ctseff`` (X93 S2), declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The law, learners, doses and margins are those of :mod:`tests.studies.dose_response_npcausal`.
The bandwidths 2.0 and 2.5 and the 4,800 replications are
:func:`tests.studies.dose_response_npcausal_pilot.declaration_from_pilot`'s reading of the
committed local R pilot, which ``tests/unit/test_dose_response_npcausal_design.py`` checks.  The
reference runs in the image the incremental studies build from
``tests/canonical/npcausal_incremental`` (npcausal 0.1.0 at ``56a5ac1`` on R 4.5.2), unchanged,
with ``run_study.R`` mounted at ``/fixture``.

Order of work, all local:

1. Done: ``python -m tests.canonical.dose_response_npcausal.pilot probe`` (review smoke steps
   S1, S2, S4, S6 and S7) and ``python -m tests.canonical.dose_response_npcausal.pilot
   reference``, which committed the R pilot: 4,000 ``ctseff`` calls at a measured 18.65 CPU-s
   each, about 21 CPU-hours.
2. The smoke (S8), which draws declared replicates 0 to 19 and so informs no declared value::

    python -m tests.canonical.dose_response_npcausal.regenerate --replicates 20 \
        --output <scratch> --allow-failures --jobs <workers>

3. The declared run, from a clean tree whose ``HEAD`` is pushed, with every thread variable
   set to 1: about 50 R CPU-hours at 4,800 replications (9,600 calls), beside about 3 for the
   ``cleverly`` fits::

    python -m tests.canonical.dose_response_npcausal.regenerate --output <empty scratch> \
        --jobs <workers>

4. Register the record with the run's artifacts.

Run history.  The declared run (commit ``c1aac5fe``) exited 0.  All 12 truth rows pass.
``ctseff`` failed 6 of 4,800 replications inside ``study_stage`` and ``cleverly`` none, so
every paired row reads 4,794: five ``locpoly`` gridsize errors at bandwidth 2, which the
transcription fails or counts as knife-edge, and at replication 2540 a type the pilot never
met, ``density.default``'s "'x' contains missing values".  There ``ctseff``'s least-squares
dose-variance fit is negative at one row, so the standardized dose is ``NaN``; the
transcription rebuilds the input and refuses at the same point.  The pair at h = 2, a = 1.5 is
superior and the pairs at h = 2.5, a = 1.5 and 2.5 equivalent.  Three are red.  h = 2, a = 3.5
is inconclusive on its complete-case calibration leg (upper bound 0.0558), the pre-registered
reading.  h = 2, a = 2.5 and h = 2.5, a = 3.5 conclude equivalent on their complete cases and
are unaccounted: the failure accounting's worst case, the six lost ``ctseff`` SEs and estimates
at the surviving minimum, lowers ``ctseff``'s SE ratio by about 0.01 and the calibration slack
by 0.011 to 0.013, past the complete-case slack of 0.0111 and 0.0107.  The paired-leg sizing
read complete cases and never priced that imputation in, a miss in the declaration and not a
defect of ``DoseResponseTMLE``.  The dose-2.5 red lies outside the pre-registered set; the
design test admits the two rows under a signature labelled post-run, and nothing was re-sized,
redrawn or re-margined.  The evidence page's "Results of the declared run" holds the numbers.

Pilot readings (the design test recomputes every number here).  ``ctseff`` failed 77, 9, 1 and
0 of 1,000 calls at bandwidths 1, 1.5, 2 and 2.5, all inside ``study_stage`` with
``locpoly``'s gridsize error.  Those are exactly the calls the Python transcription fails (74,
7, 1 and 0) or counts as failed because they hinge on the last bit of the highest dose's bin
(:func:`~tests.studies.dose_response_npcausal_pilot.knife_edge`), pilot draw 11 at bandwidth 1
among them.  The rule selects 2 and 2.5, and the count is 4,800, set by the paired legs (the
shared rule reads 2,400 both ways).  At dose 3.5 the subject's mean reported SE is 1.10 and
1.11 times the exact efficient SD.  At bandwidth 2.5 its spread rises to 1.11 times efficient
as well, so the RMSE ratio against ``ctseff`` there is 1.10 and the RMSE leg projects red at any
count while the calibration leg projects green; at bandwidth 2 the RMSE ratio is 1.05.  The
four cells at doses 1.5 and 2.5 pass every leg with projected probability at least 0.96 at that
count, with RMSE ratios at or below 1.00 and calibration excesses at or below 0.015.

Red-cell readings, declared before any run.  The record is ``reporting``: a red row publishes,
and ``TestThePreregisteredRedSet`` in the design test admits only the signatures below.  No cell
is re-sized, re-drawn or downgraded after a verdict is seen, and owner rows are filed after the
run against the rows the test admits.

* A red calibration leg at dose 3.5, with every other leg of the row green, the subject's SE
  ratio above 1, its coverage at least 0.95 and its truth row green, is the pre-registered
  reading: the subject's influence-curve SE over-reports at the upper end of the dose axis.
  The reflected pilot fits locate the over-report there whichever end of the law the dose
  sits at, so its source is the pooled-hazard density, whose bin hazards compound upward
  through the survival product, not the law and not ``ctseff``.  The dose-response study sees
  the same effect at its upper boundary dose 4.75; this law's wider dose spread brings it to
  an interior dose.
* Where the declaration's paired-leg count hit its cap (``paired`` is ``None``), a red
  calibration leg at dose 1.5 or 2.5, with every other leg green, the subject's SE ratio in
  [1, 1.07], the reference's below 1 and both truth rows green, reads as the reference's
  anti-conservative SE plus straddle noise: when the two ratios sit on opposite sides of 1 the
  leg adds their errors instead of cancelling them.
* Any other red fails the design test and is diagnosed before anything is published.  A red
  subject truth row is a defect suspect of ``DoseResponseTMLE``: the truths are exact, the
  outcome learner holds the law's regression and the doses are interior.  A red ``npcausal``
  truth row is a finding about the reference at this law and n; it does not turn the subject's
  row red (``tests.studies.evidence.comparison`` keeps the two questions apart).
* A replication that raises is never redrawn, and the failure record governs: a recordable
  subject error or an R error inside ``study_stage`` costs one replicate of its side, read
  within the declared 1% bound.  The selected bandwidths keep the R pilot's failure rate inside
  that bound with probability 0.99.  An error outside a stage stops the run.
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

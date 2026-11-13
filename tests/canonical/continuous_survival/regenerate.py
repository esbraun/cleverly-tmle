"""Regenerate the continuous-time-survival evidence (X91 part G) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The law, learners, sizes,
margins and replicate counts are those of :mod:`tests.studies.continuous_survival` and its
properties module, fixed before any run from the committed pilot
(``tests/studies/continuous_survival_pilot.csv.gz``, at aa19c6d6) and checked by
``tests/unit/test_continuous_survival_design.py``: 7,600 primary replicates of ``both_cf_1k``,
3,200 calibration replicates and 4,800 per robustness cell, all at n = 1,000.

Scope (Eric's decisions, 2026-11-12).  The record gates the default fit, cross-fitted, at
n = 1,000, on the twelve targets at the two later times.  At n = 4,000 the shared rules asked for
about 800 core-hours, so the full-sample fit and both n = 4,000 fits are reported cells.  The
first time's six targets come from the same primary fits and publish as ``early-time.csv`` with
no verdict.  At that time the influence-curve standard error of the treated arm's risk of cause 2
runs below the sampling spread (SE ratio 0.90 at n = 1,000 cross-fitted, 0.95 at 4,000), while the
estimate is unbiased and its spread matches the efficient one; the roadmap carries the finding.

Pilot readings the declaration rests on:

* ``both_cf_1k`` (995 kept of 1,000 draws): on the gated targets every standardized bias was
  within 0.051, coverage 0.925 to 0.959 and the SE ratio 0.957 to 1.037.  On the early targets
  coverage was 0.878 to 0.956 and the SE ratio 0.904 to 1.037.  99.3% of fits converged.
* Reported cells (1,000 draws each in the pilot): ``both_fs_1k`` gated-target coverage 0.891 to
  0.963 (SE ratio down to 0.876), ``both_fs_4k`` 0.924 to 0.954, ``both_cf_4k`` 0.944 to 0.961.
  ``positivity_4k`` (100 draws) covered 0.37 to 0.94, with SE ratios down to 0.26 on the control
  arm.
* Calibration (8,000 draws, 7,985 kept): SE ratio 1.000, coverage 0.944, both efficiency ratios
  1.017.
* Robustness (300 draws each): standardized bias 0.084 for ``mechanism_correct`` and 0.006 for
  ``outcome_correct`` on the treated arm's risk of cause 1 at the middle time; the
  ``both_wrong`` control read 0.87.  The untargeted plug-in of ``mechanism_correct`` sat at a
  median of more than 2 SE from the truth (the design's gate on the wrong hazard).

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner
row X91 and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red bias row of ``both_cf_1k`` is a defect suspect of the walk or the curves: every
  nuisance library holds the truth.  Diagnose it against ``tests/unit/test_continuous_survival.py``
  (the exact-law influence curve and its mutation controls).
* A red coverage or SE-ratio row of ``both_cf_1k`` at the middle time on cause 2 of the treated
  arm, with its SE ratio's 99% interval below 1, its bias equivalent and its coverage interval
  reaching 0.90, reads as the early-time standard-error finding reaching the middle time at
  n = 1,000, and takes the route above with the measured ratio stated.  Any other red
  coverage row is a variance defect suspect.
* A red positive ``double_robustness`` cell is a defect suspect, except that
  ``mechanism_correct`` (pilot bias 0.084 SD) red on equivalence alone, with its measured
  standardized bias inside the 0.2 margin and ``outcome_correct`` green, reads as the
  finite-sample bias of the branch whose event hazard is wrong at n = 1,000, and takes the
  route above with the measured bias stated.  A control that does not clear the margin is an
  instrument finding, and the cells' claim to witness the wrong mechanisms is withheld.
* A calibration cell red on its SE ratio while its coverage is green and its efficiency ratios
  sit inside the band reads as a finite-sample variance finding at that n and takes the route
  above with the measured ratio stated.  Red on coverage, or with an efficiency ratio outside the
  band, is a variance defect suspect.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The reported cells and ``early-time.csv`` carry no verdict and no prediction; their readings
  are published as measured.
* A replication that raises is never redrawn.  A ``DataError`` (an initial hazard fit whose
  survival reaches zero before the last target, or whose combined jump reaches one) is a
  recordable fit failure, bounded by the failure bound: the pilot lost 0.5% of ``both_cf_1k``
  fits and 0.19% of calibration fits to it.

Departures from the design (X91-G-design.md section 15): the primary and property cells run at
n = 1,000 rather than 4,000, the first target time is reported rather than gated, and the
n = 4,000 and full-sample fits are reported cells.  The first 300 draws of each primary pilot
cell ran before the walk fix of 38a742bf, and the later 700 after it.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import continuous_survival, continuous_survival_properties

if __name__ == "__main__":
    run(
        continuous_survival,
        continuous_survival_properties,
        here=Path(__file__).resolve().parent,
    )

"""Regenerate the survival-dr-tmle evidence (X91 part F) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The law, learners, sizes,
margins and replicate counts are those of :mod:`tests.studies.survival_dr_tmle` and its
properties module, fixed before any run from the committed pilot
(``tests/studies/survival_dr_tmle_pilot.csv.gz``, at 61f7bf55) and checked by
``tests/unit/test_survival_dr_tmle_design.py``: 6,000 primary replicates, 3,600 calibration
replicates and 5,200 per robustness cell.

Pilot readings the declaration rests on (1,000 draws per primary cell, 300 per robustness cell,
8,000 for calibration): every primary row had a standardized bias within 0.06, coverage 0.933
to 0.962 and an SE ratio of 0.944 to 1.039.  The calibration cell read an SE ratio of 1.006.
The positive robustness cells read standardized biases of -0.001 (mechanism correct) and -0.086
(outcome correct) for the risk of arm 1; the control read 4.7.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner
row X91 and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row of ``both_*`` is a defect suspect of the drift loop or its curve: every
  nuisance is correctly specified and the pilot read no departure.  Diagnose it against
  ``tests/unit/test_survival_drtmle.py`` (the exact-law drift identity and the independent
  reference), fix a defect, and re-run under a fresh declaration.  A red coverage row of
  ``both_1k`` reads as the finite-sample variance limit of the cross-fitted fit at
  n = 1,000 only when its SE ratio's 99% interval lies below 1, its bias is equivalent and its
  coverage interval reaches 0.90; it then takes the route above.
* A red primary row of ``known_*`` is a defect suspect of the held-factor rule (design 4.5a, row
  2): with both mechanisms declared the remainder is exactly zero for any hazard model while the
  ``g_bounds`` floor does not bind.  The arm rows witness the rule: the hazard model is wrong, so
  an untargeted plug-in sits about 3.6 SD off each arm's risk at n = 200,000.  The risk
  difference row is gated but is no witness, since that error cancels between the randomized
  arms (+0.03 SD).
* A red positive ``double_robustness`` cell is a defect suspect, except that ``outcome_correct``
  (pilot bias -0.086 SD) red on equivalence alone, with its measured standardized bias negative
  and inside the 0.2 margin and with ``mechanism_correct`` green, reads as the finite-sample bias
  of the branch that rests on the hazard model alone at 16,000, and takes the route above with
  the measured bias stated.  The control is read against
  its pilot gap; a control that does not clear the margin is an instrument finding, and the
  cells' claim to witness the wrong mechanisms is withheld.
* A calibration cell red on its SE ratio while its coverage is green and its efficiency ratios
  sit inside the band reads as a finite-sample variance finding at that n, and takes the route
  above with the measured ratio stated.  Red on coverage, or with an efficiency ratio outside the
  band, is a variance defect suspect.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The reported cells carry no verdict and no prediction.  They are descriptive, not controls,
  and no predicted drift term was computed for them.  At c = 4 the slow cells' bias is well
  under two standard deviations (the pilot read -0.60 for the slow censoring cell and 0.26 for
  the slow hazard cell at 16,000), so a small reading shows neither validity nor failure under
  a slowly converging nuisance.  ``insample_4k`` is a failure mode, not a test (a 12-replicate
  smoke read a bias near -5 SD and coverage near 0.1).  Their readings are published as
  measured.

Departure from the design (X91-F-design.md section 8): the law-P cells are dropped (the law is a
reconstruction with open items and its l1 learner was never built), and the slow and in-sample
must-fail controls are reported cells.  The first-order drift term under a misspecified index
rests on the exact-law check 5(b) of ``tests/unit/test_survival_drtmle.py`` alone, and Díaz's
Section 6 simulation is not reproduced: the evidence covers GLM learners on the two-covariate
law L-S only.
* A replication that raises is never redrawn.  A ``LongitudinalError`` from a submodel with no
  finite maximum is a recordable fit failure, bounded by the failure bound; the pilot lost no fit.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import survival_dr_tmle, survival_dr_tmle_properties

if __name__ == "__main__":
    run(
        survival_dr_tmle,
        survival_dr_tmle_properties,
        here=Path(__file__).resolve().parent,
    )

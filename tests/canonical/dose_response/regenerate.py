"""Regenerate the dose-response evidence (X93 S1) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator (npcausal ``ctseff`` comes with X93 part S2), so the run starts no
container.  Laws, learners, sizes, margins and replicate
counts are those of :mod:`tests.studies.dose_response` and its properties module, fixed before
any run from the committed pilot (:mod:`tests.studies.dose_response_pilot`).

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner
row X93 and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row of ``binary``, ``strata`` or ``gaussian`` is a defect suspect of the
  quadrature, the per-pair fluctuation or the influence curve: the truths are exact integrals
  and the outcome sieve holds the covariate part of the law exactly.  Diagnose it against
  ``tests/unit/test_dose_response.py`` and the doubling diagnostic
  (``result.quadrature_shift``), fix a defect, and re-run under a fresh declaration.  A red row
  at a boundary pair (dose 0.25 or 4.75) whose bias is equivalent (``bias_equivalent``) and
  whose SE-ratio interval (``se_ratio_ci_upper``) lies below 1 reads as the finite-sample
  variance limit of a boundary weight at n = 4,000.  With no defect found it takes the
  route above, and the boundary pairs' Wald claim stays experimental with the measured coverage.
  Any other red row is a defect suspect.
* A red ``tail`` row is a finding about the support diagnostic, not a defect suspect and not a
  reason to raise the floor.  The pilot read coverage 0.938 below the floor and 0.934 above it,
  coverage fell as the fitted Kish size rose, and the stratified pair at dose 0.25 in ``V = 1``
  at n = 4,000 under-covered (0.908) at a mean Kish size of 69 with 2.9% of its fits labelled.
  A floor raised on the Kish size would label more of the fits that cover and few of those that
  do not.  The route is the one above, with the measured coverage split by the label
  (``pair-support.csv``) stated in the owner row, and roadmap X95 takes the relabelling: a
  diagnostic that tracks the interval's error, not the floor's value.  The gate never moves.
* A red ``simultaneous_band`` cell whose 99% joint-coverage interval lies below 0.92 while every
  ``binary`` primary row passes reads as the finite-sample limit of the multiplier band over 15
  near-collinear curves: the route above, with the measured joint coverage stated in the owner
  row.  A red band cell above 0.98 is a critical-value defect suspect.  The band is measured
  conditional on one multiplier draw (``random_state=0``), whose critical value sits within 0.02
  of the oracle's.  A ``pointwise_joint_control`` that does not fall below 0.95 is an instrument
  defect, and the run is void: its exact-law ``p0`` and power are in
  ``tests/unit/test_dose_response_design.py``.
* A calibration cell red on the upper end of its SE ratio alone (``se_ratio_ci_upper`` above
  1.07 with ``se_ratio_ci_lower`` at or above 0.93), while its coverage is green and both
  efficiency ratios sit inside the band, reads as a conservative reported SE at n = 4,000: the
  route above, with the measured ratio stated.  The pilot's reasons for expecting it (the
  reported SE runs 2% to 6% above the efficient SD at interior pairs) support the upper side
  only.  A red on the lower end (``se_ratio_ci_lower`` below 0.93), a red coverage interval, or
  an efficiency ratio outside the band is an anti-conservative variance defect suspect.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* A red positive ``double_robustness`` cell says the outcome regression no longer carries the
  fit when the density is wrong: a defect suspect of the outcome-side fit or the influence curve,
  read against its population limit 0.017 efficient SEs off the truth
  (``test_the_robustness_limits``).  It is not a targeting witness: with the outcome right the
  fluctuation is near zero.  A red joint clause on the targeting gain (``targeting_gain`` or
  ``targeting_gain_past_truth`` below 1.5 efficient SEs) is a defect suspect of the targeting
  step, the clever covariate or its inverse density: diagnose it against the mutation tests of
  ``tests/unit/test_dose_response.py``.  A control that does not clear the margin is read
  against its population limit, 3.03 efficient SEs off; with no defect found it takes the
  route above, and the cell's claim to witness the density in the clever covariate is
  withheld, leaving the exact remainder tests to carry it.
* A replication that raises is never redrawn, and the failure record governs.  A recordable
  error (:func:`cleverly.validation._failures.recordable`) costs its cell one replicate: a
  ``DataError`` naming a zero fitted density inside a kernel's support, a dose interval outside
  the observed range, or a fluctuation that did not solve its score.  The failure accounting
  then reads the cell within the declared 1% bound, and a cell that loses more, or whose
  verdict a lost replicate could flip, is red; such a red is a finding about the declaration at
  that n and takes the route above.  The pilot lost none of about 13,000 fits.  An exception
  that is not recordable stops the run and is a defect suspect.

Run history.  The first declared run (``dfd56920``, artifacts at ``4edc3ab6``) exited red on
the two targeting ``diagnostic`` rows alone.  The properties module wrote their union-model
endpoint into ``passed``, which the driver gates, although the declaration said a diagnostic
row is never red.  That is an instrument defect, not a finding: every gated row and the
targeting-gain clause (3.34 and 2.65 efficient SEs against a floor of 1.5) were green.  The
module now marks diagnostic rows as never red, and the run is repeated in full under the new
commit on the same seeds, as the route above prescribes.

The claim-boundary reading (the pointwise interval read against the unsmoothed curve at dose
2.5) is reported from the ``binary`` rows and gates nothing; the properties module states its
expected values.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import dose_response, dose_response_properties

if __name__ == "__main__":
    run(dose_response, dose_response_properties, here=Path(__file__).resolve().parent)

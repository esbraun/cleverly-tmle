"""Regenerate the dose-response evidence (X93 S1) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The comparator, npcausal ``ctseff``, is pending (X93 part S2) and has no runner yet, so the run
passes no ``Reference`` and starts no container.  Laws, learners, sizes, margins and replicate
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
* A red ``tail`` row is the declared measurement of the support floor, not a defect suspect.
  The route is a scope change written into a fresh declaration commit: raise the default
  ``DoseResponse.support_floor``, or the smallest default bandwidth fraction of
  ``DoseResponse.spanning``, so the defaults no longer admit the measured pair without the
  ``low_dose_support_plugin`` label, and re-run under that declaration.  The gate never moves.
* A red ``simultaneous_band`` cell whose 99% joint-coverage interval lies below 0.92 while every
  ``binary`` primary row passes reads as the finite-sample limit of the multiplier band over 15
  near-collinear curves: the route above, with the measured joint coverage stated in the owner
  row.  A red band cell above 0.98 is a critical-value defect suspect.  A
  ``pointwise_joint_control`` that does not fall below 0.95 is an instrument defect, and the run
  is void: its exact-law ``p0`` and power are in ``tests/unit/test_dose_response_design.py``.
* A calibration cell red on its SE ratio while its coverage is green and its efficiency ratios
  sit inside the band reads as a finite-sample variance finding at n = 4,000: the route above,
  with the measured ratio stated.  Red on coverage, or with an efficiency ratio outside the
  band, is a variance defect suspect.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* A red positive ``double_robustness`` cell is a defect suspect of the targeting step: its
  population limit sits 0.017 efficient SEs off the truth
  (``test_the_robustness_limits``).  A control that does not clear the margin is read against its
  population limit, 3.03 efficient SEs off; with no defect found it takes the route above, and the
  cell's claim to witness the density in the clever covariate is withheld, leaving the exact
  remainder tests to carry it.
* A replication that raises is never redrawn.  The pilot lost no fit.  A ``DataError`` naming a
  zero fitted density inside a kernel's support, or a dose interval outside the observed range,
  is a defect of the declaration at that n and voids the run.  Any other exception in the
  dose-response path is a defect suspect.

The claim-boundary reading (the pointwise interval read against the unsmoothed curve at dose
2.5) is reported from the ``binary`` rows and gates nothing; the properties module states its
expected values.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import dose_response, dose_response_properties

if __name__ == "__main__":
    run(dose_response, dose_response_properties, here=Path(__file__).resolve().parent)

"""Regenerate the targeted missingness tilt evidence (A2, CDR-001) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.missingness_tilt_inference` and its
properties module, fixed before any run from the committed pilot
(:mod:`tests.studies.missingness_tilt_pilot`, the full pilot of about 21,000 fits).

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the A2
owner row and the reading below, the run is repeated under the new commit on the same seeds,
and the rule the cell licenses reads descriptive.  No cell is re-sized, re-drawn or downgraded
after a verdict is seen.

* A red primary row of ``in_sample``, ``in_sample_4k``, ``cross_fit``, ``cv_evaluation`` or
  ``weighted`` is a defect suspect of the tilt fluctuation, the tilted influence curve, or the
  way a composition (fold evaluation, weights, the log scale of ``rr``) enters them: the truths
  are exact quadratures and every nuisance model is correctly specified.  Diagnose it against
  ``tests/unit/test_missingness_tilt_targeted.py`` and its exact-law and mutation tests.  A red
  row at n = 1,000 whose ``in_sample_4k`` twin is green, whose bias is equivalent and whose
  SE-ratio interval lies above 0.93, reads as the finite-sample limit of the Wald interval at
  n = 1,000: the route above, with the measured coverage stated, and the rule stays licensed
  only at the ladder's size.
* A red ``natural_course`` row is a defect suspect of the observed-arm tilt (``ey_obs``) or of
  the joint influence curve of ``par``; diagnose it the same way.
* A red ``ate_gamma2__correctly_specified`` calibration cell on the lower end of its SE ratio,
  a red coverage interval, or an efficiency ratio outside [0.90, 1.10] is an
  anti-conservative variance defect suspect of the tilted influence curve.  A red on the upper
  end of the SE ratio alone, with coverage green, reads as a conservative reported SE: the route
  above, with the measured ratio stated.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.  ``mar_variance_control`` (the A1-era interval)
  that does not fall below the band means the declared law no longer separates the MAR
  variance from the tilted one at gamma 2 (the law's exact ratio is 0.81): an instrument
  defect, and the run is void; the law changes under a fresh declaration.
* A red ``ate_gamma2__targeted`` necessity arm (bias not equivalent) is a defect suspect of the
  tilt fluctuation, read against the second-order remainder of the main-terms regression: its
  large-n limit, from four fits at n = 3,000,000, is -0.019 (SE 0.005) targeted-arm SDs at
  n = 1,000, and the untargeted arm's is +0.379 (SE 0.005).  An ``untargeted`` arm whose bias is not discriminated, or
  a displacement below 0.25, says the declared misspecification no longer moves the plug-in:
  an instrument defect, and the run is void.
* A replication that raises is never redrawn, and the failure record governs.  A recordable
  error costs its cell one replicate, within the declared 1% bound; the pilot lost none of
  about 21,000 fits.  An exception that is not recordable stops the run and is a defect
  suspect.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import missingness_tilt_inference, missingness_tilt_properties

if __name__ == "__main__":
    run(
        missingness_tilt_inference,
        missingness_tilt_properties,
        here=Path(__file__).resolve().parent,
    )

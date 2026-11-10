"""Regenerate the ep-assessment evidence (P1 part (b)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The laws, readings, thresholds, controls and replicate counts are those of
:mod:`tests.studies.ep_assessment`, whose pass probabilities the committed pilot fixed.

Red-cell reading (declared before any run; P1 plan sections 5.0 and 5.4, and the PR-b design
review's B2, S2 and S5).

* A red calibration row with its stretched control green is a defect suspect of the
  out-of-fold refits or of the curve: diagnose, fix, re-run under a fresh declaration.  A red
  control (the stretched slope's upper bound not below 0.85) with a green calibration row
  cannot happen while the slope identity holds; it is a defect of the reading.  A red
  calibration row with its control red too means a slope above 1.275 (the stretched slope is
  exactly the slope over 1.5): the curve is flatter than the held-out contrasts, so it
  understates how the effect varies.  It is a defect suspect of the curve's final stage or of
  the reading, diagnosed as above, and the calibration claim is withheld.
* Stability is reported, not gated (plan amendment rev 2.4): it measures sensitivity to the
  cross-fitting split on the same data, not sampling noise, and states no verdict.
* A red strong-law sign share is a defect suspect of the DR risk reading.  The claim is near
  certain at the strong law (the pilot's reported difference sits about 22 SDs from 0), so it
  detects only a gross defect, such as a sign error in the risk's cross term, and says nothing
  about the reading's accuracy.
* A row red only because a replicate failed ("failures not accountable") is read from the
  recorded exception: a learner or numerical refusal is a finding about the learner, reported
  with the exception, and the claim it blocks stays withheld; an exception in the assessment
  code is a defect suspect.  The pilot lost no fit, so this is an accepted risk.
* The weak-law sign share, the risk differences, ``relative_sd`` and the weak-law slope are
  diagnostic and state no verdict.
* No replicate count, law, threshold or seed moves after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import ep_assessment, ep_assessment_properties

if __name__ == "__main__":
    run(ep_assessment, ep_assessment_properties, here=Path(__file__).resolve().parent)

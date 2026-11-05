"""Regenerate the ep-assessment evidence (P1 part (b)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The laws, readings, thresholds, controls and replicate counts are those of
:mod:`tests.studies.ep_assessment`, whose pass probabilities the committed pilot fixed.

Red-cell reading (declared before any run; P1 plan sections 5.0 and 5.4).

* A red calibration row with its stretched control green is a defect suspect of the
  out-of-fold refits or of the curve: diagnose, fix, re-run under a fresh declaration.  A red
  control (the stretched slope's upper bound not below 0.85) with a green calibration row
  cannot happen while the slope identity holds; it is a defect of the reading.
* A red strong-law stability share is a defect suspect of the refits; a red weak-law share
  says the split-seed spread does not see a curve of noise at this ``n``, and the stability
  warning's claim is withheld (the reading stays a point reading with no validity claim).
* A red strong-law sign share is a defect suspect of the DR risk reading.
* The weak-law sign share, the risk differences, ``relative_sd`` and the weak-law slope are
  diagnostic and state no verdict.
* No replicate count, law, threshold or seed moves after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import ep_assessment, ep_assessment_properties

if __name__ == "__main__":
    if ep_assessment.BLOCKED:
        raise SystemExit(f"refused: {ep_assessment.BLOCKED}")
    run(ep_assessment, ep_assessment_properties, here=Path(__file__).resolve().parent)

"""Regenerate the separable-effects evidence (X91 part D) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.separable_effects` and its properties
module, fixed before any declared draw from the committed pilot.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner
row X91 and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is a defect suspect of the factor-wise targeting or its influence curve: the
  truths are exact identification-(8) formulas and the fits are saturated.  Diagnose it against
  ``tests/unit/test_separable_effects_law.py``.  A red coverage row of ``cross_fitted`` reads
  as the stopped-short fits only when the coverage among the converged fits of that cell is
  green; the pilot's 2.6% of unconverged cross-fitted fits covered at 0.953.  With no defect
  found it takes the route above with the count stated.
* A red positive ``double_robustness`` cell is a defect suspect: its exact-law limit is the
  truth (``test_a_wrong_factor_with_the_exact_mechanism_has_the_truth_as_limit``,
  ``test_the_wrong_mechanism_cell_lands_near_the_truth``).
* The control ``both_hazards_wrong`` that does not clear the margin is read against its
  exact-law gap (``test_the_control_is_discriminated``).  With no defect found it takes the
  route above, and the cells' claim that both factors carry the robustness is withheld; the
  unit robustness tests carry it alone.
* A calibration cell red on its SE ratio while its coverage is green and its efficiency ratios
  sit inside the band reads as a finite-sample variance finding at that n: the route above,
  with the measured ratio stated.  Red on coverage, or with an efficiency ratio outside the
  band, is a variance defect suspect.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* A replication that raises is never redrawn.  The pilot lost no fit at the declared size, so
  any recorded failure is a defect suspect.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import separable_effects, separable_effects_properties

if __name__ == "__main__":
    run(separable_effects, separable_effects_properties, here=Path(__file__).resolve().parent)

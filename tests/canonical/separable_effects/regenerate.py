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
  as the stopped-short fits only when the coverage among that cell's converged fits, joined
  from ``convergence.csv``, reaches the coverage band.  With no defect found it takes the
  route above with the count stated.
* A red positive ``double_robustness`` cell is a defect suspect: its exact-law limit is the
  truth (``test_a_wrong_factor_with_the_exact_mechanism_has_the_truth_as_limit``,
  ``test_the_wrong_mechanism_cell_lands_near_the_truth``).  The cross-fitted
  ``mechanism_wrong`` cell stopped short in 2.3% of its rerun pilot fits, all
  ``cell_separated``; its ``nonconverged_fits``
  is reported and decides nothing, and a red verdict there is read against the converged rows
  of ``property-replicates.parquet`` as for ``cross_fitted`` above.
* The control ``both_hazards_wrong`` that does not clear the margin is read against its
  exact-law gap (``test_the_control_is_discriminated``).  With no defect found it takes the
  route above, and the cells' claim that both factors carry the robustness is withheld; the
  unit robustness tests carry it alone.
* A calibration cell red on its SE ratio while its coverage is green and its efficiency ratios
  sit inside the band reads as a finite-sample variance finding at that n: the route above,
  with the measured ratio stated.  Red on coverage, or with an efficiency ratio outside the
  band, is a variance defect suspect.  ``cross_fitted_sep01_death_t3`` stopped short in 0.55% of
  its pilot fits and is read against its converged rows in the same way.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* Unconverged fits (declared after the rerun pilot under the stall fix, plan rev 2.1).  The
  expected share is about 0.6% of cross-fitted fits, every one ``failure="cell_separated"`` at a
  relative score of at most about 2e-5, read from ``convergence.csv`` and the ``failure`` and
  ``phase`` columns of ``property-replicates.parquet``.  Each of these is a defect suspect of the
  targeting, diagnosed before registration: a ``newton_stalled`` fit with a relative score
  above 1e-6; ``cell_separated`` in more than 1.5% of the primary cross-fitted fits or 5% of a
  property cell's fits; and any in-sample fit whose ``phase`` is not ``walk``.  After the run
  every ``cell_separated`` primary fit is refitted from its seed, and one whose logits at the
  clip sit on rows that hold an event of that cause (lower clip) or a row without one (upper
  clip), which separation does not explain, is a defect suspect too.  ``FAILURE_BOUND`` counts fits that raise, not fits that stop short.
* A replication that raises is never redrawn.  The pilot lost no fit at the declared size, so
  any recorded failure is a defect suspect.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import separable_effects, separable_effects_properties

if __name__ == "__main__":
    run(separable_effects, separable_effects_properties, here=Path(__file__).resolve().parent)

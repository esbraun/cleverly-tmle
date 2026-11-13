"""Regenerate the separable-effects evidence (X91 part D) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.separable_effects` and its properties
module, fixed from the committed pilot before the first declared draw.  The rerun under the stall
fix of the targeting (plan ``X91-D-stall-plan.md`` rev 2.1) changes no law, learner, margin, seed
or cell; ``PRIMARY_REPLICATES`` is re-read by the shared rule from the rerun pilot.

Red-cell readings (declared before the first run, revised before the rerun).  The record is
``gated``, so any red verdict fails the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner
row X91 and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is a defect suspect of the factor-wise targeting or its influence curve: the
  truths are exact identification-(8) formulas and the fits are saturated.  Diagnose it against
  ``tests/unit/test_separable_effects_law.py``.  A red coverage row of ``cross_fitted`` reads
  as the stopped-short fits only when the coverage among that cell's converged fits, joined
  from ``convergence.csv``, reaches the coverage band.  At about 0.6% of fits stopped short,
  that reading can explain a near miss of the band only.  With no defect found it takes the
  route above with the count stated.
* A red positive ``double_robustness`` cell is a defect suspect: its exact-law limit is the
  truth (``test_a_wrong_factor_with_the_exact_mechanism_has_the_truth_as_limit``,
  ``test_the_wrong_mechanism_cell_lands_near_the_truth``).  The cross-fitted
  ``mechanism_wrong`` cell stopped short in 7 of its 300 rerun pilot fits, all
  ``cell_separated``; refitted with the right treatment model the same draws still separate, so
  that rate is a fluctuation of the pilot's draws; its ``nonconverged_fits``
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
* Unconverged fits (declared after the rerun pilot under the stall fix, plan rev 2.1, before
  the rerun).  The expected share is about 0.6% of each cross-fitted cell's fits, every one
  ``failure="cell_separated"``.  :func:`tests.studies.separable_effects.primary_suspects` and
  :func:`tests.studies.separable_effects_properties.convergence_suspects` read the defect
  suspects from ``convergence.csv`` and ``property-replicates.parquet``, and
  ``test_the_committed_run_holds_no_suspect`` requires none: an in-sample fit that stops short or
  leaves the walk; a cross-fitted fit that stops short for any other reason (``newton_stalled``,
  ``max_iter_reached``, ``line_search_exhausted``) or above a relative score of 1e-4;
  ``cell_separated`` in more than 1.5% of a fitted cross-fitted cell's fits (the derived
  calibration controls reuse their base cell's fits and are not counted).  After the run,
  ``python -m tests.studies.separable_effects_separation run DIR`` refits every
  ``cell_separated`` fit, and one with a clipped logit on a row that pulls it inward (an event
  at the lower clip, a row without one at the upper clip), which separation does not explain,
  is a defect suspect too.  The in-sample primary rows of replicates 0 to 8,399 must equal the
  previous run's within 1e-12, since the fix leaves the walk unchanged.  ``FAILURE_BOUND``
  counts fits that raise, not fits that stop short.

  Read after the run (2026-11-12): the separation check refitted 102 ``cell_separated`` fits and
  found no inward-pulling clipped row, the in-sample rows of replicates 0 to 8,399 equal the
  previous run's within 5e-16, and no cell's separated share passed 1.5%.  Three cross-fitted
  fits (two primary, one ``cross_fitted_sep01_death_t3`` calibration fit) stopped with
  ``newton_stalled`` at 1.2 to 2.4 times ``tol``.  They were investigated
  (``python -m tests.studies.separable_effects_stalls``): the remaining score lies in two
  directions in which the Jacobian of the factor scores is singular to rounding, which the Newton
  phase discards, and steps forced along them reach a root within ``tol`` that moves no estimate
  by more than 0.0015 standard errors.  Eric chose to keep the run and record
  them in :data:`tests.studies.separable_effects.INVESTIGATED_STALLS`, which the suspect check
  passes only while each stalls at its recorded score.
* A replication that raises is never redrawn.  The pilot lost no fit at the declared size, so
  any recorded failure is a defect suspect.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import separable_effects, separable_effects_properties

if __name__ == "__main__":
    run(separable_effects, separable_effects_properties, here=Path(__file__).resolve().parent)

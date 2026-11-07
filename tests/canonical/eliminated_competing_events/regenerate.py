"""Regenerate the eliminated-competing-events evidence (X91 part A) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.eliminated_competing_events` and its
properties module, fixed before any run from the committed pilot.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner
row X91 and the reading below, the run is repeated under the new commit, which refits the same
deterministic seeds and so reproduces the same rows, and the claim the cell carries stays
experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is a defect suspect of the eliminated recursion, its two-factor retention or
  its influence curve: the truths are exact g-formulas and the fits are saturated.  Diagnose it
  against ``tests/unit/test_eliminated_competing_law.py``, fix a defect, and re-run under a
  fresh declaration.  A red coverage row of ``hazard_held_baseline`` at ``t = 3`` reads as the
  finite-sample variance limit of a sparse cell only when all of these hold in
  ``performance-tests.csv``: the 99% interval of its SE ratio (``se_ratio_ci_upper``) lies
  below 1, its bias is equivalent (``bias_equivalent``), and its coverage interval's upper end
  (``coverage_ci_upper``) reaches 0.90.  With no defect found it takes the route above, and
  the hazard path's Wald claim at ``t = 3`` stays experimental with the measured coverage.  Any
  other red primary row of that scenario is a defect suspect.
* A red positive ``double_robustness`` cell is a defect suspect: its exact-law limit is the
  truth (``test_every_positive_limit_is_the_truth``).
* A ``double_robustness`` control that does not clear the margin is read against its exact-law
  gap (``test_each_control_is_discriminated``).  With no defect found it takes the route above,
  and the cells' claim to witness the death factor is withheld; the unit mutation controls carry
  that claim alone.
* A calibration cell red on its SE ratio while its coverage is green and its efficiency ratios
  sit inside the band reads as a finite-sample variance finding at that n: the route above,
  with the measured ratio stated.  Red on coverage, or with an efficiency ratio outside the
  band, is a variance defect suspect.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* A replication that raises is never redrawn.  A ``CapabilityError`` for a node regression with a
  constant target is the known limit of a sparse saturated cell, not a defect: the pilot lost no
  fit at the declared sizes, and a cell red only because such failures exceeded its bound takes
  the route above with the count stated.  Any other exception in the eliminated path is a defect
  suspect.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import eliminated_competing_events, eliminated_competing_events_properties

if __name__ == "__main__":
    run(
        eliminated_competing_events,
        eliminated_competing_events_properties,
        here=Path(__file__).resolve().parent,
    )

"""Regenerate the ``isotonic-binding`` evidence (X76 part (g)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The law, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_isotonic_binding`,
:mod:`tests.studies.isotonic_binding_properties` and :mod:`tests.studies._isotonic_binding_law`,
fixed before the run and recomputed by ``tests/unit/test_isotonic_binding_design.py`` from the
committed pre-read (``tests/studies/isotonic_binding_pilot.csv``) and tolerance table
(``tests/studies/isotonic_binding_tolerance.csv``).

What the study can claim.  On law ``LG2`` (two nodes, continuous ``W``, a strictly increasing
risk) the projection moves the never regimen's horizon-1 level, often by a tenth of an SE or
more, because the two horizons' node-1 regressions use different learner classes (Fix-8 plan
2.7).  The gated cells read the interval at the projected level, which keeps the fit's SE, and
the share of draws on which the level moves.  At these move sizes the projected and unprojected
intervals disagree on about half a percent of draws, so the calibration cell shows that the
projection changed coverage by at most that share; it cannot show a gain, and it says nothing of
a flat truth, larger moves or other learner pairings.

Red-cell reading (declared before any run; Fix-8 plan sections 5.0 and 5.8).  The design test
applies it mechanically to the committed tables (``gating_findings`` over the 5 gated cells).

* A red ``interval_calibration/projected_t1__correctly_specified`` takes the standard reading:
  it is a finite-sample red only when the 99% interval of its SE ratio contains 1, its coverage
  is consistent with 0.95 and the diagnosis finds no defect, and it then gets an owner row in the
  roadmap's Red-cell owners table.  Otherwise it is a defect suspect; unfixed, it becomes a
  shortfall of ``G31.isotonic`` with the measured number, so the label reads experimental.
* A derived control (``*_shrunken_se_control``, ``*_noise_control``) that fails to fail is a
  design bug or a defect suspect.
* A red ``binding_witness`` or ``material_move`` becomes a shortfall of ``G31.isotonic``, and
  the limitation states the measured share.
* ``reported_levels`` and ``reported_large_move`` turn no cell red.  Horizon-2 coverage is read
  on the draws that report an SE, a selection, and is never gated.
* No replicate count, law, learner, threshold, margin or seed moves after a verdict is seen, and
  no re-run on a new law, more replicates or another threshold buys a pass.

Tolerance.  The logistic learner is solved to ``tol=1e-13``.  Before the declaration the design
test reads 300 pilot draws refitted at ``tol=1e-10``: every binding and material-move indicator
agrees, every move agrees to ``1e-4`` relative, no move falls inside ``(1e-9, 1e-5)`` SE and
every logistic fit converged.  Had any check failed, the study would not have been declared.

Sizes.  Every cell runs ``REPLICATES`` replications on stream ``lg2``, and the reported
larger-move law the same on stream ``lg2_gs2``: the arm-pooled predictive rule on the pooled
pre-read of ``projected_t1``, never below plan 5.0's single-cell floor of 2,400.  The witnesses
pass with probability above 0.9999 at the pre-read shares.  The primary phase runs 1,600.

Budget.  One fit per replication and stream at about 0.07 s on the 4-vCPU cloud machine, so
about 0.2 CPU-hours with the harness; plan 5.9 allows 0.5 plus the extra calibration draws.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_isotonic_binding, isotonic_binding_properties

if __name__ == "__main__":
    run(
        canonical_isotonic_binding,
        isotonic_binding_properties,
        here=Path(__file__).resolve().parent,
    )

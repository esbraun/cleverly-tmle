"""Regenerate the ``option-patience`` evidence (X76 part (c)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The laws, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_option_patience`,
:mod:`tests.studies.option_patience_properties` and :mod:`tests.studies._option_patience_laws`,
fixed before the run and recomputed by ``tests/unit/test_option_patience_design.py`` from the
committed pilot (``tests/studies/option_patience_pilot.csv``).

Red-cell reading (declared before any run; Fix-8 plan section 5.0).  The study has no
calibration cell: selector paths withhold the interval (F18), and no row of it is a
finite-sample red.  The design test applies the reading mechanically to the committed tables
(``gating_findings`` over the six gated cells).

* A red ``patience_bias/lc1_greedy`` (bias outside 0.25 SD) is a defect suspect in the greedy
  path with patience: diagnose it, fix the defect, and re-run under a fresh declaration.  If the
  diagnosis finds no defect, the bias is a property of the early-stopped greedy estimator on
  this law, ``C12.greedy`` is not added, and the greedy path with ``patience=`` stays
  experimental with the measured number.
* A red ``patience_witness/lc1_greedy`` or ``late_confounder_stop/lc3_ordered`` (a share whose
  lower limit is below 0.3) is a design bug: the cell it guards no longer measures an early
  stop.  The claim of that cell is withheld until the design is repaired under a fresh
  declaration.
* An ``ordered_late_confounder/lc3_ordered`` control that fails to fail is read with its stop
  witness: a low stop share is the design bug above; a stop share above its floor with a small
  bias means the conditional bias moved, which ``late_confounder_stop`` publishes, and is
  diagnosed as a defect suspect of the counter or of the ordered path.
* A red ``ordered_whole_path/lc3_ordered`` is a defect suspect in the ordered path or a design
  bug of the law; its limit bias is at most 0.024 SD (the truncation limit bound at the
  declared ``g_bounds``), and the W1 exclusions at n = 2,000 bring the sized bias to at most
  0.124 SD, so it is never narrated as a finite-sample effect.
* A red ``greedy_late_confounder/lc3_greedy`` (the mean paired difference outside 0.25 SD of
  the greedy whole path) is a defect suspect in the greedy path with patience, read as the
  first bullet.
* Every ``reported_*`` row is published and turns no cell red.  The greedy whole-path bias on
  LC3 (``reported_estimate/lc3_greedy_whole``) is quoted in the limit of ``C01.greedy`` and
  ``C12.greedy`` at registration, whatever its value.
* The primary (``lc1``, the ``patience_bias`` estimator on its own draws) is declared
  point-only: its bias row gates as a positive, and its interval columns are the plug-in
  diagnostic of a selector path, published with no coverage or SE-ratio verdict (F18).
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No re-run
  buys a pass.

Pilot.  ``tests/studies/option_patience_pilot.py``, 2,000 draws of each gated stream from draw 0
of its own seed stream, at the declared n, learners and ``g_bounds``; the design test reads the
committed summary.  ``LC1``: the greedy stop share is 0.853 at patience 1 and 0.515 at
patience 2, so ``p*`` is 1 (plan 5.5: the smallest p in {1, 2} with a share of at least 0.5);
the bias at ``p*`` is -0.007 SD.  ``LC3``: the ordered fit with patience 2 leaves ``W1`` out on
69.6% of draws, with a bias of 0.114 (3.55 SDs of the whole ordered path) on those draws and a
standardized bias of 1.36 for the cell; the whole ordered path selects ``W1`` on 98.4% of draws
with standardized bias 0.023; the greedy fit with patience 2 returns the greedy whole path's
estimate on 98.25% of draws (mean paired difference 0.024 whole-path SDs), and the greedy whole
path's standardized bias is 0.42.

Sizes.  Every cell runs 1,600 replications and the primary 1,600 (plan 5.5).  The bias
positives pass with probability above 0.9999 at their pilot biases.  The control is sized by
the mixture formula at the pilot's 99% lower limits (stop share 0.669, conditional bias 3.50
whole-path SDs): limit 1.21 SD, cap 1.42, at least 0.61 SD down to a conditional bias of one SD,
and discrimination power 1.0 to six places at 1,600.  The two shares' lower limits (0.831 and
0.669) clear the 0.3 floor with probability 1.0 to six places.  The LC3 whole path's limit bias
is at most 0.024 SD at the declared ``g_bounds`` of (0.01, 0.99): the bound
``E|1 - g0/g_t| + E|1 - (1 - g0)/(1 - g_t)|`` over the 99% lower limit of the pilot SD.  The
sample-size default bounds (0.0147 at n = 2,000) would leave a bound of 0.056 SD, which is why
LC3 declares its own (plan 5.0, the Part C note).  That is a limit bound: at n = 2,000 the whole
path drops W1 on 1.65% of pilot draws, and that share times its conditional bias at their 99%
upper limits (0.0253 and 3.96 SD) adds 0.100 SD.  At 0.124 SD the cell passes with probability
0.993 at R 1,600, and the pilot reading of 0.023 SD already includes the exclusions.

Budget.  A smoke of the property phase fitted one draw of every stream in about 6 seconds on one
core of the 4-vCPU cloud machine, so 1,600 draws and the primary are about 3 CPU-hours of fits
and about 6 with the harness; plan 5.9 allows 10.  About 25 minutes at ``--jobs 16``.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_option_patience, option_patience_properties

if __name__ == "__main__":
    run(
        canonical_option_patience,
        option_patience_properties,
        here=Path(__file__).resolve().parent,
    )

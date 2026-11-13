"""Regenerate the asymmetric-wald evidence (Fix-3 PR 3) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. No external
comparator implements either construction for a TMLE curve, so the run starts no container and
records an empty equivalence artifact. Its independent evidence is the exact or closed-form
truths, the unit witnesses (``tests/unit/test_asymmetric_intervals.py``), the controls, the
falsifiers and the two mutation witnesses. The groups, laws, sizes, fits, controls, clauses and
design numbers are those of :mod:`tests.studies.asymmetric_wald`,
:mod:`tests.studies.asymmetric_wald_properties` and :mod:`tests.studies.asymmetric_wald_laws`,
fixed before the run (Fix-3 plan rev 6, sections 9.5 to 9.7; ``rows/Fix-3/pr3-notes.md``).

The shipped default does not change before this run (plan section 7, items 7 and 8), so the
study computes PR 3's rule from the public functions the flip will read: ``mover_ci`` with
``implied_correlation`` on the arms' range-logit intervals for a bounded difference, the
shipped range-logit interval for a bounded mean, and ``skew_widened_ci`` at
``max(small_sample_df, 5)`` with the estimate's ``acceleration`` for a continuous target. The
primary publishes MOVER's interval of ``strata-boundary-mean``'s ``ate[V=2]`` at R = 1,000 (a
withheld interval as the point, a miss); the shipped interval of the same fits is in
``primary-shipped.csv``, which no verdict reads.

Declaration. Every number is the design test's (``tests/unit/test_asymmetric_wald_design.py``):
12 gated cells, each R = 4,000; the coverage legs read the first 2,000 replicates by index, and
the tail clause (threshold 0.040), the C3 width clause (threshold 1.08), the witnesses and the
paired gains read all 4,000. At perfect calibration the 12 cells and the three paired gains pass
together with probability at least 0.9720; with the owner cells at their exploratory readings
the designed chance of no red row is at least 0.9032, so the designed chance of a red row is at
most 0.097, almost all of it ``ho_hurdle_ey1``. Witness powers 0.99621 and 1.00000. The
held-out laws D1, D2, C1, C2 and C3 had no estimator fitted on them before this declaration.

The width clause's bootstrap (review 5, nit R6-1), fixed before the run: the statistic is the
ratio of mean widths of the skew-widened interval over the capped t across the cell's shown
fits; fits are resampled with replacement with their two widths kept together, B = 10,000, seed
``stream_seed(STUDY, "width_ratio", cell)`` (its entropy ``RESAMPLING_SEED = 2026111302``),
percentile interval, upper bound the 0.995 quantile.

Red-cell reading (declared before any run; plan 9.6 and ``guide/studies.md``).

* The falsifiers read lost fits at the worst case, as the gated rows' failure accounting
  does: the lower leg counts a lost fit among the first 2,000 as a miss, and the tail clause adds
  every lost fit of the cell to each side's one-sided count and to the shown total, one side at
  a time. A lost held-out fit can therefore only make a falsifier fire, never keep one quiet.
* A falsifier that fires (``falsifiers.csv``) stops PR 3 for the class it names: that class
  does not flip, no control is adopted, the owner's red rows stay open with the reading, and the
  plan returns to review. The ``means`` falsifier fires on the shipped PR 2 rule: it is read as
  a defect suspect, and the plan returns to review.
* A red gated owner row (``ho_sparse_treat_ate``, ``ho_hurdle_ey1``, ``lognormal06_ey1``) with
  no falsifier fired keeps its owner open (``F3-difference-near-bound``,
  ``F3-skewed-continuous``) with the reading. Its decomposition (SE ratio, oracle-SD coverage,
  both tails, the candidates on the same fits) is reported; a red that the SE leg explains (the
  SE-ratio 99% interval entirely below 1 with oracle-SD coverage consistent with 0.95) goes to
  the owner of that SE leg, as study 1 read H2.
* An upper-leg red on a gated cell is a defect suspect, an over-covering rule, and never goes to
  an owner. The C3 rows' over-coverage is reported; their harm reading is the width clause.
* H2 replicates whose sample leaves a stratum-2 design cell empty are recorded failed fits
  (``empty_design_cell``): misses in the lower leg when among the first 2,000, absent from the
  shown-only leg, and counted as one-sided misses in the tail clause's worst case. Their count
  is read against the exact expectation, published as ``expected_failed``.
* A witness that does not fail its tail clause, or a control that passes where it should not,
  means the instrument cannot discriminate; it is reported and nothing is re-declared. No
  replicate is added after a verdict, no margin moves, no cell is dropped, and no re-run buys a
  pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import asymmetric_wald, asymmetric_wald_properties

if __name__ == "__main__":
    run(asymmetric_wald, asymmetric_wald_properties, here=Path(__file__).resolve().parent)

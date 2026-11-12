"""Regenerate the small-sample-wald evidence (Fix-3) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. No external
comparator implements the rule for a TMLE curve, so the run starts no container and records an
empty equivalence artifact. The groups, laws, sizes, fits, controls and design numbers are those
of :mod:`tests.studies.small_sample_wald` and :mod:`tests.studies.small_sample_laws`, fixed
before the run (Fix-3 plan rev 4, sections 5.1 to 5.4).

PR 1 leaves the shipped default unchanged, so the study computes the rule from the public
functions the default will read (``rule_interval``), and the primary rows record the shipped
interval of ``strata-boundary-mean``'s ``ey[1][V=2]`` and ``ate[V=2]``, the baseline the rule is
read against. A red primary ``ey[1][V=2]`` row is that owner's registered shortfall, not a
verdict on the rule; ``primary-rule.csv`` holds the rule's reading of the same fits.

Red-cell reading (declared before any run; plan 5.4 and ``guide/studies.md``).

* A lower-leg red on a positive cell: read the falsifiers of plan 5.3 first
  (``falsifiers.csv``). Otherwise a defect suspect in the correction, diagnosed with the
  decomposition (SE ratio, bias/SD, oracle-SD coverage), the candidates of plan 2.8 on the same
  fits, and each withheld rate against its exact probability. If the oracle-SD coverage is
  consistent with 0.95 and the SE-ratio interval lies below 0.95, the red is the variance
  estimate's and goes to the owner of that SE leg.
* An upper-leg red on a positive cell (shown-only above 0.98): on a held-out cell it is read by
  the falsifier declared there. Elsewhere it is a defect suspect, an over-covering rule, and it
  never goes to an owner.
* The reported no-harm cells (``composite_high``, ``mar_multi_arm_ate``,
  ``identity_msm_v2_a``) are compared with their owner's law and owner, never with the
  correction. The reported leverage cells are rows: if either arm's SE-ratio 99% interval
  excludes 1, or its coverage interval leaves (0.92, 0.98), it is a red row, diagnosed with the
  decomposition and proposed through the coordinator for an owner of flexible-learner SE ratios.
* An H2 replicate whose sample leaves a stratum-2 design cell empty is a recorded failed fit
  (``empty_design_cell``): a miss in the lower leg, absent from the shown-only upper leg. Its
  count is read against the exact expectation, 3.1 fits at R = 2,000.
* A red control or witness means the instrument cannot discriminate; it is reported and not
  re-declared. No replicate is added after a verdict, no margin moves, no cell is dropped, and
  no re-run buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import small_sample_wald, small_sample_wald_properties

if __name__ == "__main__":
    run(small_sample_wald, small_sample_wald_properties, here=Path(__file__).resolve().parent)

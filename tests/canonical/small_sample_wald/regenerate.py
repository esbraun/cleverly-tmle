"""Regenerate the small-sample-wald evidence (Fix-3) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory. No external
comparator implements the rule for a TMLE curve, so the run starts no container and records an
empty equivalence artifact. The groups, laws, sizes, fits, controls and design numbers are those
of :mod:`tests.studies.small_sample_wald` and :mod:`tests.studies.small_sample_laws`, fixed
before the run (Fix-3 plan rev 4, sections 5.1 to 5.4).

PR 1 leaves the shipped default unchanged, so the study computes the rule from the public
functions the default will read (``rule_interval``). The primary rows publish the rule's
interval of ``strata-boundary-mean``'s ``ey[1][V=2]`` and ``ate[V=2]``, so the primary measures
the correction; a withheld interval is published as the point, a miss. The shipped interval of
the same fits, the baseline the rule replaces, is in ``primary-shipped.csv``, which no verdict
reads.

Declaration (plan rev 4, section 7 item 5, after review ``impl-review-pr1.md`` and its
remediation). Every number is the design test's (``tests/unit/test_small_sample_wald_design.py``):
the 14 gated two-sided cells, both paired gains and the non-H1 pair pass together with
probability 0.952 at perfect calibration, withholding and H2's empty design cells included; at
the exploratory readings the designed chance of at least one red gated row is 0.094, most of it
``lognormal06_ey1`` (0.057). The primary publishes the rule's interval of
``strata-boundary-mean``'s law at R = 1,000; its ``ey[1][V=2]`` row is the rule's
``boundary_mean`` reading at half the replicates, read by the same declared reading. The
held-out laws H1 to H3 had no estimator fitted on them before this declaration.

Red-cell reading (declared before any run; plan 5.4 and ``guide/studies.md``).

* A lower-leg red on a positive cell: read the falsifiers of plan 5.3 first
  (``falsifiers.csv``). Otherwise a defect suspect in the correction, diagnosed with the
  decomposition (SE ratio, bias/SD, oracle-SD coverage), the candidates of plan 2.8 on the same
  fits, and each withheld rate against its exact probability. If the oracle-SD coverage is
  consistent with 0.95 and the SE-ratio 99% interval lies entirely below 1, the red is the
  variance estimate's and goes to the owner of that SE leg.
* The guide's finite-sample route applies to a coverage red as written there: when the SE-ratio
  99% interval contains 1, the coverage is consistent with 0.95 and the diagnosis finds no
  defect, the red goes to ``reporting`` with an owner in the Red-cell owners table. At R = 2,000
  a lower-leg red rarely meets it, and an upper-leg red cannot (below).
* An upper-leg red on a positive cell (shown-only above 0.98): on a held-out cell it is read by
  the falsifier declared there. Elsewhere it is a defect suspect, an over-covering rule, and it
  never goes to an owner.
* The reported no-harm cells (``composite_high``, ``mar_multi_arm_ate``,
  ``identity_msm_v2_a``) are compared with their owner's law and owner, never with the
  correction. The reported leverage cells are rows: if either arm's SE-ratio 99% interval
  excludes 1, or its declared coverage legs leave (0.92, 0.98), its ``passed`` is false and it
  is a red row, diagnosed with the decomposition (and the paired difference of the two arms'
  SE ratios) and proposed through the coordinator for an owner of flexible-learner SE ratios.
  Its role is diagnostic, so the red-cell ledger will not list it: the row thread files it
  through the coordinator itself.
* An H2 replicate whose sample leaves a stratum-2 design cell empty is a recorded failed fit
  (``empty_design_cell``): a miss in the lower leg, absent from the shown-only upper leg. Its
  count is read against the exact expectation, 3.1 fits at R = 2,000, which the summary
  publishes as ``expected_failed`` beside ``failed_replicates``.
* A red control or witness means the instrument cannot discriminate; it is reported and not
  re-declared. No replicate is added after a verdict, no margin moves, no cell is dropped, and
  no re-run buys a pass.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import small_sample_wald, small_sample_wald_properties

if __name__ == "__main__":
    run(small_sample_wald, small_sample_wald_properties, here=Path(__file__).resolve().parent)

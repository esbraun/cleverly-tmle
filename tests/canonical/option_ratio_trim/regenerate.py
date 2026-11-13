"""Regenerate the ``option-ratio-trim`` evidence (X76 part (e)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else.  :mod:`tests.canonical.declared_run`
describes the guard, the run log, the checkpoints and the copy into this directory.  The study
has no comparator, so the run starts no container.  The law, learners, cells, sizes and design
numbers are those of :mod:`tests.studies.canonical_option_ratio_trim`,
:mod:`tests.studies.option_ratio_trim_properties` and :mod:`tests.studies._option_trim_law`,
fixed before the run and recomputed by ``tests/unit/test_option_ratio_trim_design.py`` from the
committed pilot (``tests/studies/option_ratio_trim_pilot.csv``).

What the study can claim.  A trimmed fit withholds its interval (``trimmed_ratio_plugin``), and
its remainder is first order in the outcome-regression error on the trimmed share.  The study
claims the point estimate only, with the declared linear outcome regressions, which contain
every iterated regression of the law.  The plug-in SE ratio of each trimmed cell is published
as a diagnostic and turns no cell red.

Red-cell reading (declared before any run; Fix-8 plan section 5.0).  The design test applies it
mechanically to the committed tables (``gating_findings`` over the 8 gated cells).

* A red ``trim_point`` positive (bias not inside the margin) is a defect suspect: its exact-law
  limit is the truth, and the pilot read at most 0.006 SD.  Diagnose it with the exact-law
  limit, fix the defect, and re-run under a fresh declaration.  A positive that fails only by
  a small finite-sample bias, with the defect ruled out, gets an owner row in the roadmap's
  Red-cell owners table, and its ``G56`` row is not added.
* A ``trim_necessity`` control (``*_trimmed``) that fails to fail is a design bug or a defect
  suspect: its exact-law bias is 0.068 (LTMLE) and 0.083 (SDR), 1.95 and 2.50 pilot SDs, so a
  control inside the margin means the trim was not applied.
* A red ``trim_necessity`` positive (``*_untrimmed``) is a defect suspect in the untrimmed
  path, whose exact limit is the truth by double robustness.
* Every ``trim_reported`` row is published beside its exact limit and turns no cell red.
* No replicate count, law, learner, margin or seed moves after a verdict is seen.  No re-run
  buys a pass.

Trim.  Plan 5.6 declares ``ratio_trim=0.95`` for the gated cells.  At 0.99 the W-blind
controls read 0.41 SD (LTMLE) and 0.52 SD (SDR) on the 4,000-draw pilot, against the 0.45 SD
floor, so 0.99 is reported only.  The trim binds by construction: 15.8% of the positive ratios
lie above the 0.95 quantile.

Sizes.  Every cell runs 1,600 replications on one draw stream, so every fit of a replicate reads
the same sample: plan 5.6's count, which the rule in the design test (the six bias positives
pass together with probability at least 0.95 at their pilot biases, and each control is
discriminated with probability above 0.9999) does not raise.  The primary phase runs 1,600.

Budget.  About 0.2 CPU-hours of fits on the 4-vCPU cloud machine (the pilot fitted 4,000 draws
of the 14 fits in 26 CPU-minutes, and the run fits 1,600 draws of them plus the primary), under
an hour with the harness; plan 5.9 allows 2.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_option_ratio_trim, option_ratio_trim_properties

if __name__ == "__main__":
    run(
        canonical_option_ratio_trim,
        option_ratio_trim_properties,
        here=Path(__file__).resolve().parent,
    )

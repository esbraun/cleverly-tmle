"""Regenerate the ``mtp-density-tails`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_mtp_density_tails` and its
properties module, fixed before any run and checked by
``tests/unit/test_mtp_density_tails_study_design.py``.

Order of the work: the pilots of :mod:`tests.studies.mtp_density_tails_pilot`, one label per
run and committed after each (``--collect`` folds the checkpoints into ``pilot.csv.gz``); then
the design test reads them, extends any calibration arm whose plug-in pass probability is below
0.90 by 400 draws, and rebudgets every cell from the pilot's per-fit seconds; then a declaration
commit in which the design test skips nothing; then the declared run.  A cell too large for one
cloud part runs in replicate ranges first (``python -m tests.studies.mtp_density_tails_properties
--parts DIR --streams <stream> --replicates A:B``); each part checkpoints draw by draw under
``DIR``, and the declared run reuses those checkpoints once ``DIR``'s contents sit in its
scratch ``checkpoints`` directory.  A split changes no draw: each replicate has its own seed.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is reported, not gated: the primaries publish no verdict beyond the shared
  primary checks.  A primary row outside them is diagnosed as a defect suspect of the density
  route, since every nuisance is exact or correctly specified.
* A red ``ic_gap`` row on G, T, C, B or U is a defect suspect: diagnose it with paired fits and
  the exact-mass harness of ``tests/unit/test_density_reconstruction.py`` before any rerun.  The
  S and ``Gscale`` rows are reported and cannot be red.
* A red calibration or density DR row is a finite-sample red only when three things hold: its
  SE-ratio 99% interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds
  no defect.  It then goes to ``reporting`` with an owner row in the roadmap's Red-cell owners
  table.  Any other red row is a defect suspect, and an unfixed defect never ends as an owner
  row.  This holds for the density arms (G, T, C, and G and T in DR), the classifier arms (G, T,
  S) and the longitudinal arms (``LTMLE`` and ``SDR`` on ``ate_regimen[up vs natural]``).  The
  six other longitudinal estimands are reported; one outside the band is diagnosed and reported,
  not a red row.
* The study's designed chance of at least one red gated row under a correct method is about
  0.18: the calibration family passes with 0.975, the ``ic_gap`` clauses with 0.900 predictive
  (0.979 plug-in) and the two ``cluster_warning`` clauses with 0.977 x 0.961.  That is why each
  red route above is declared now.
* A red ``cluster_warning`` row is read against the probe's rates (C warned in 20 of 20 draws at
  20,000, G and T in none): it is diagnosed on the doses of the ``ic_gap`` draws it read.  A
  helper that misses C's far component is a defect suspect; a G or T draw that warns is reported
  with its cluster.
* A control that passes the clause it must fail is a design bug.  Before the run it blocks the
  declaration; in the run it voids its family's verdict and blocks the pull request.  That holds
  for the shrunken-SE and noise controls, the DR both-wrong and pre-fix controls, the ``ic_gap``
  controls (pre-fix, support ignored, no row-wise break) and the two cluster controls.
* A replication that raises is never redrawn.  A recordable failure costs its replicate under
  the declared bound of 0.01; a cell past its bound takes the route above with the count stated.

Reported cells (``pilot.csv.gz`` labels), each with its declared reading.  None publishes a
verdict, so none can be red; each reading below is a statement the evidence page checks.

* **Boosting** (the default library's booster, cross-fitted) at 2,000 (20 draws per law) and
  20,000 (10 draws on G, T, C and S), the shipped and the pre-fix reading on the same draws.  On
  G and C the median ratio error falls by at least a third from 2,000 to 20,000.  **Tail
  reading**: at 2,000 at most 2 of 20 draws exceed a ratio error of 1.0 and none exceeds 3.0;
  at 20,000 at most 1 of 10 exceeds 0.5.  **T and B are declared at risk**: their pass
  probabilities for "at most 2 of 20 above 1.0" are 0.597 and 0.555 (the design test's at-risk table, recomputed by the
  design test from the pilot).  Every breaching draw, on any law, is diagnosed on its fit with
  the **three-mark misfit signature** (:class:`tests.studies.mtp_density_tails_fits.Signature`):
  (1) the exact bin probabilities on the same grid read a ratio error of at most 0.15; (2) the
  five worst rows carry at least half the squared error; (3) at the worst row the fitted log
  probability of the bin of ``A`` is at least 2 below the exact one, or that of the bin of
  ``A - 1`` at least 2 above it.  A breach with all three is reported as a learner limitation,
  with the docs line on the default booster.  A breach without them is a defect suspect,
  diagnosed before any rerun.  Two guards: if the number of 2,000 draws above 1.0 exceeds the
  0.95 predictive quantile of that table's posterior for that law (the probe's ceilings out of 20:
  G 3, T 6, C 2, S 2, B 8, U 4, recomputed from the pilot), the grid choice is reopened for that
  law whatever the signature says; and the signature is declared only because a deliberately
  broken reading (occupancy from the wrong sample) fails at least one mark on every draw.
* **Default stacked density** at 2,000, 5 draws on G, T, C and S, both readings: the boosting
  tail reading at 2,000 applies, and a breach is routed the same way.
* **Logistic learners** (pooled and per-bin slope) at 2,000 and 20,000: reported without a rule.
* **B and U with learned hazards** (boosting and pooled logistic, 20 draws), read with the
  declared support and with it ignored: with the support the median ratio error is below the
  pre-fix reading's and no more than 10% above the support-ignored reading's, and the boosting
  tail reading applies with the signature routing (B is at risk); without it the error is
  reported.
* **The classifier route** on G, T, C and S at 2,000 (20 draws) and 20,000 (10 draws), and on
  Gcap and Bp2 at 2,000, with and without the policy steps: on T with the steps the 20,000
  median ratio error is at most 0.8 times its 2,000 median, and without them at least 0.9 times;
  on Gcap and Bp2 the steps cut the 2,000 median by at least a quarter; on G, C and S at 2,000
  the steps change the median by at most 10%.  At 20,000 the steps run on T and G only.
* **The classifier route's DR cells** (``<law>_classifier_dr__reported``, 400 draws on G and T,
  the wrong regression): on G the mean remainder proxy lies within 0.2 SE of zero; on T it is
  negative, about 0.3 to 0.5 SE, with coverage near 0.93.  A T coverage below 0.90 is reported
  with its remainder; it is not a red cell and not rerun.  The both-wrong contrast (the
  classifier without ``W1``, on G) shows what a ratio that misses a confounder does.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import canonical_mtp_density_tails, mtp_density_tails_properties

if __name__ == "__main__":
    run(
        canonical_mtp_density_tails,
        mtp_density_tails_properties,
        here=Path(__file__).resolve().parent,
    )

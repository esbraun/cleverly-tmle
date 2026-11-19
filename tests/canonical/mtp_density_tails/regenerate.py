"""Regenerate the ``mtp-density-tails`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Laws, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.canonical_mtp_density_tails` and its
properties module, fixed before any run and checked by
``tests/unit/test_mtp_density_tails_study_design.py``.

Order of the work: the pilots of :mod:`tests.studies.mtp_density_tails_pilot`, committed after
each label (``--collect`` folds the checkpoints into ``pilot.csv.gz``); then the design test reads
them, applies the extension rule below to each gated calibration arm, and rebudgets every cloud
part from the pilot's per-fit seconds; then a declaration commit in which the design test skips
nothing; then the declared run.

The declared run's heavy streams run first as eleven cloud parts
(``python -m tests.studies.mtp_density_tails_properties --parts DIR --part P<k>``), each at most
55 CPU-h at the smoke's per-fit times: P1 to P3 the classifier calibration streams on G, T and
S; P4 the classifier DR streams with ``control_density_G`` and ``control_classifier_G``; P5 and
P6 ``longitudinal_LTMLE`` in two ranges; P7 ``control_LTMLE``; P8 to P11 ``longitudinal_SDR`` in
four ranges (``RUNNER_PARTS``).  Each part checkpoints draw by draw under ``DIR``, and the
assembly run, this script, reuses those checkpoints once every part's ``DIR`` contents sit in
its scratch ``checkpoints`` directory; it fits the density, DR, ``ic_gap`` and
``cluster_warning`` cells itself, about 3 CPU-h.  A split changes no draw: each replicate has
its own seed.

**The extension rule** (declared before any pilot reading).  A gated calibration arm's plug-in
pass probability at R = 3,200 is read at its pooled pilot's SE ratio and coverage.  While it is
below 0.90, the pilot is extended by 400 draws on its own stream and pooled.  It stops at the
first of three outcomes: clear, when the plug-in reaches 0.90; a confirmed departure, when the
pooled 99% bootstrap interval of the SE ratio excludes 1 or the 99% coverage interval excludes
0.95; or the cap, three extensions (1,600 draws in all).  A confirmed departure is a defect
suspect, diagnosed before the declaration, and so is an arm at the cap.  If no defect is found
the arm runs at 3,200 and its plug-in pass probability enters the family and whole-study
figures.  For a calibrated arm the rule extends 0.91 times on average and ends clear with
probability 0.897, at a departure with 0.026 and at the cap with 0.076.  The reserve is the cap
on the classifier G and T, ``LTMLE`` and ``SDR`` pilots, 161.4 CPU-h.

**The interval the coverage leg reads.**  Every positive calibration and density DR arm reads
coverage at the interval the fit ships, because the study validates what ships: the
skew-widened t interval of a point estimate of a continuous outcome, and the MOVER interval of
the longitudinal contrast ``ate_regimen[up vs natural]``.  The shrunken-SE and noise controls
and the pre-fix row form a symmetric interval at the normal critical value themselves.  The
pilot rows predate the shipped interval, so the design test reads pilot coverage at it: the
density and DR pilots are refitted for their shipped limits (``pilot-limits.csv.gz``), and on
the classifier and longitudinal pilots a draw within 1% of a symmetric limit is counted as able
to change side, each arm's stop holding at the worst case.

**Pass probabilities and floors.**  Nine calibration-type arms are gated, each on one estimand:
density calibration on G, T and C, density DR on G and T, classifier calibration on G and T, and
the longitudinal default route on ``LTMLE`` and ``SDR``.  At R = 3,200 each SE-ratio clause
passes with 0.9975 at target, so the family passes with 0.978.  The ``ic_gap`` 20,000 clause
reads the largest of 30 gaps; its pass probability is the Jeffreys beta-binomial chance of no
exceedance in 30 draws from the exceedances among 3,000 tail-pilot draws per gated law (0.995
at none).  The cluster clauses read 3,000-draw dose pilots (0.99999 and 1.0 at the probe rates).
The whole-study designed pass, the product of every gated clause, is **0.954** predictive at no
tail exceedance (0.978 plug-in): 0.978 (calibration) x 0.975 (``ic_gap``) x 1.0 (falls and
2,000 bounds) x 1.0 (clusters).  The design test computes it from the pilot.  The declared
floors are 0.90 per gated ``ic_gap`` law and per calibration arm, 0.85 for the ``ic_gap``
family, and 0.90 for the whole study.  A pilot that reads below a floor is diagnosed before the
declaration: sizing may change, a threshold may not.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* A red primary row is reported, not gated: the primaries publish no verdict beyond the shared
  primary checks.  A primary row outside them is diagnosed as a defect suspect of the density
  route, since every nuisance is exact or correctly specified.
* A red ``ic_gap`` row on G, T, C, B or U, and a red ``cluster_warning`` row (``C__n_20000`` or
  ``GT__pooled``), is a defect suspect.  It is diagnosed before any rerun: the design test's
  chance of that red is stated, the red draws are re-read with the exact-mass harness of
  ``tests/unit/test_density_reconstruction.py``, and paired fits on fresh seeds test the
  suspected cause.  A red ``GT__pooled`` row is diagnosed on each warning draw's doses.  A
  genuine defect is fixed, and the study reruns under a fresh declaration that records what
  changed.  When the diagnosis finds no defect, the row stays red in the committed artifacts and
  is not rerun.  Such a row reaches ``reporting`` only if a recorded project decision extends
  the finite-sample route to non-calibration rows; until then that law's grid row does not
  claim the clause.  No margin, law, budget or cell changes to make a
  row pass.  The S and ``Gscale`` ``ic_gap`` rows are reported and cannot be red.
* A red calibration or density DR row is a finite-sample red only when three things hold: its
  SE-ratio 99% interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds
  no defect.  It then goes to ``reporting`` with an owner row in the roadmap's Red-cell owners
  table.  Any other red row is a defect suspect, and an unfixed defect never ends as an owner
  row.  This holds for the density arms (G, T, C, and G and T in DR), the classifier arms (G and
  T) and the longitudinal arms (``LTMLE`` and ``SDR`` on ``ate_regimen[up vs natural]``).  The
  classifier arm on S and the six other longitudinal estimands are reported; one outside the
  band is diagnosed and reported, not a red row.  S's influence curve has an infinite fourth
  moment, so its SE ratio spreads beyond the normal sizing at any feasible R.
* A control that passes the clause it must fail is a design bug.  Before the run it blocks the
  declaration; in the run it voids its family's verdict and blocks the pull request.  That holds
  for the shrunken-SE and noise controls, each of which must fail the SE-ratio clause on its own
  stream of 1,000 draws (``control_density_G``, ``control_classifier_G``, ``control_LTMLE``);
  the DR both-wrong controls (as built: mean z +1.96 on G and +1.82 on T, coverage 0.467 and
  0.550) and the pre-fix control (mean z -1.60 on G, coverage 0.620), each of which must fail
  coverage; the ``ic_gap`` controls (pre-fix, support ignored, no row-wise break); and the two
  cluster controls.
* A replication that raises is never redrawn.  A recordable failure costs its replicate under
  the declared bound of 0.01; a cell past its bound takes the route above with the count stated.

Reported cells (``pilot.csv.gz`` labels), each with its declared reading.  None publishes a
verdict, so none can be red; :func:`tests.studies.mtp_density_tails_pilot.reported_readings`
computes each reading below, and a fast test compares it with the committed ``readings.json``.

* **Boosting** (the default library's booster, cross-fitted) at 2,000 (20 draws per law) and
  20,000 (10 draws on G, T, C and S), the shipped and the pre-fix reading on the same draws.  On
  G and C the median ratio error falls by at least a third from 2,000 to 20,000.  **Tail
  reading**: at 2,000 at most 2 of 20 draws exceed a ratio error of 1.0 and none exceeds 3.0;
  at 20,000 at most 1 of 10 exceeds 0.5.  **T and B are declared at risk**: their pass
  probabilities for "at most 2 of 20 above 1.0" are 0.597 and 0.555 (the design test's at-risk
  table).  Every breaching draw, on any law, is routed by the **three-mark misfit signature**
  (:class:`tests.studies.mtp_density_tails_fits.Signature`), whose marks the pilot writes on
  each draw: (1) the exact bin probabilities on the same grid read a ratio error of at most
  0.15; (2) the five worst rows carry at least half the squared error; (3) at the worst row the
  fitted log probability of the bin of ``A`` is at least 2 below the exact one, or that of the
  bin of ``A - 1`` at least 2 above it.  A breach with all three is reported as a learner
  limitation, with the docs line on the default booster.  A breach without them is a defect
  suspect, diagnosed before any rerun.  Two guards: the frozen ceilings
  (``BOOSTER_CEILINGS``: G 3, T 6, C 2, S 2, B 8 and U 4 of 20 draws above 1.0 at 2,000, the
  0.95 predictive quantiles of the probe's counts, committed before any booster pilot draw)
  apply to ``boosting_<law>_n2000`` and the boosters of ``learned_B`` and ``learned_U``; a count
  above its ceiling reopens the grid choice for that law whatever the signature says.  And the
  signature is declared only because a deliberately broken reading (occupancy from the wrong
  sample) makes the booster breach on each control draw and fails at least one mark there.
* **Default stacked density** at 2,000, 5 draws on G, T, C and S, both readings: the boosting
  tail reading at 2,000 applies with no ceiling, and a breach is routed the same way.  The cells
  are budgeted at 1.8, 1.7, 7 and 14 CPU-h on G, T, C and S.
* **Logistic learners** (pooled and per-bin slope) at 2,000 and 20,000: reported without a rule.
* **B and U with learned hazards** (boosting and pooled logistic, 20 draws), read with the
  declared support and with it ignored: with the support the median ratio error is below the
  pre-fix reading's and no more than 10% above the support-ignored reading's, and the boosting
  tail reading applies with the signature routing (B is at risk); without it the error is
  reported.  On B and U the policy is the row-capped shift of 1.
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
* **The reported calibration rows**: the density route on S, B and U, the classifier route on
  S, and the six reported longitudinal estimands, each read on the SE ratio and coverage with
  no rule.
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

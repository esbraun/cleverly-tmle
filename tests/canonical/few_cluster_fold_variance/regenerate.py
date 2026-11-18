"""Regenerate the ``few-cluster-fold-variance`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container. Laws, learners, cells, sizes,
margins, seeds and replicate counts are those of :mod:`tests.studies.few_cluster_fold_variance`
and :mod:`tests.studies.few_cluster_fold_variance_properties`, fixed before any run and
recomputed by ``tests/unit/test_few_cluster_fold_variance_design.py``. The run was committed at
``de658a44`` and the record is registered.

Order of the work (plan X54 rev 3.5, sections 2.4, 3.3, 5.3, 5.8 and 9): the cells and arms; the
pilot of :mod:`tests.studies.few_cluster_fold_variance_pilot` on its own seed stream, 400 draws
of each of the twelve cells and 1,600 of the eight whose plug-in pass probability was below 0.90
at 400 under rev 3.1's rule; the rev 3.2 redesign from that pilot; the refit of the pilot's 43
lost draws under the constant-target rule (``--refit-lost``); the pre-run design review and its
fixes (rev 3.4: the deflation sized from the pilot, the red-cell owner, the declared route in
code, the known mechanism declared through ``treatment_probabilities=``); the merge of
``origin/main``; the confirmation review's reading of a chance red (rev 3.5); this declaration;
the declared run.

Budget: measured at 0.03 to 0.07 s per key (two fits) on one core, so the 96,000 property keys
take about 1.5 CPU-h, and the run with its 1,000-draw primary, summaries and failure accounting
about 1.5 to 2 CPU-h. The ceiling stays the plan's 20 CPU-h.

**Design changes from the pilot, made before this declaration.** Each follows from the law or
from the mechanism the pilot showed, not from a verdict:

* The covariate law's cells measure ``ey1``, not the ATE: the arm mean carries the plug-in part
  of the curve, which holds the cluster effect on that law, and the ATE nearly cancels it, so
  the first pilot (discarded) read the IID control at 0.97 to 1.06 on the ATE.
* Cells are read by training clusters, ``J (V - 1) / V`` = 2, 4, 6, 8, 16, 24, with the clusters
  per validation fold (2, 4, 6) as a second axis. At 2 clusters per validation fold the V = 5
  cells read variance ratios of 1.15 and 1.04 and the V = 2 cells 7.6 and 17: the limit is the
  complement the nuisances are fitted on, not the validation fold.
* ``known_propensity`` (the same draw refitted with the law's propensity declared known through
  ``treatment_probabilities=``) is a second gated configuration with its own floor, and its
  deflated control ``known_deflated_control`` is added. The pilot supplied the same propensity as
  an oracle learner; the design test shows the two forms give the same point, SE and status.
* The positive rule gates the variance ratio's lower end only. With an estimated propensity
  fitted on 2 to 6 training clusters, a few draws report an SE of 10 to 107 SDs of the estimate;
  they carry the mean of the squared SE (ratio 7.6 with coverage 0.952), so the upper side
  measures that tail, not calibration. It is reported with the 5%-trimmed ratio and the largest
  SE; the coverage band's upper leg still gates a variance too large on typical draws.
* The IID control reads the median paired IID/cluster SE ratio: Fix-4's ratio of means has no
  power at 2 training clusters, where the cluster SE's heavy lower tail drives it to 0.69 and 0.89.
* A deflated control must fail the positive rule, not only its variance leg. Where the estimated
  propensity's tail raises the positive variance ratio, a deflated variance can still read above
  0.8649 while its coverage falls; a control of the lower end alone would have no power there
  (pilot: 6 of 12 cells). The deflation is 0.80 squared, an SE 20% too small, sized from the
  committed pilot columns: at 0.85 squared the estimated ``equal10__v2_j12`` control failed the
  rule with probability 0.55, because the positive there is conservative, and more draws would
  lower that probability, not raise it.
* A training complement that holds one outcome value takes the constant-target rule (a plain
  ``TMLE``'s pooled outcome regression predicts that value, ``tests/unit/test_fold_policy_rules.py``)
  instead of refusing. The pilot's 43 lost draws, all missing an outcome class, now fit; a
  complement without an arm is still refused, and such a draw is lost and a miss in G1.

**What is gated** (:mod:`tests.studies.few_cluster_fold_variance_properties`). On each positive
row, ``t_reference`` and ``known_propensity``, all three of: the exact 99% interval of shown-fit
coverage inside [0.92, 0.98]; G1, the exact 99% lower end of coverage over every requested draw
(withheld and lost fits as misses), at least 0.90; the 99% percentile-bootstrap lower end of the
variance ratio, mean squared SE over the variance of the estimates over the shown fits, at least
0.8649. On each ``deflated_control`` and ``known_deflated_control`` row (the positive's variance
times 0.64): the positive rule fails. On each ``iid_t_control`` row: the distribution-free exact
99% upper end of the median paired IID/cluster SE ratio below 0.95. ``pooled_centered`` is not
gated (:data:`~tests.studies.few_cluster_fold_variance.POOLED_CENTERED_GATED`). Reported: the
variance ratio's interval, its trimmed form, the largest SE and the median SE in SDs, the SE ratio
with ``c_k``, the ratio-of-means IID reading, the withheld counts and every diagnostic arm.

**The declared route** of each configuration is code,
:func:`~tests.studies.few_cluster_fold_variance_properties.declared_route`, tested branch by
branch in the design test. Its training floor is the smallest training count from which every
positive cell of the configuration, on both laws, passes
(:func:`~tests.studies.few_cluster_fold_variance_properties.training_floor`).

Pilot readings (``pilot.csv.gz``; per configuration the variance ratio, its 5%-trimmed form, the
largest SE in SDs, shown coverage, all-draw coverage, and the plug-in pass probability at 8,000):

=====  =============================  ======================================  =====================================
train  cell                           estimated                               known
=====  =============================  ======================================  =====================================
2      ``equal10__v2_j4``             7.63, 2.04, 25.7, 0.952, 0.901: 0.013   1.16, 0.99, 2.7, 0.937, 0.887: 0.000
2      ``cluster_covariate__v2_j4``   17.1, 0.89, 106, 0.917, 0.917: 0.000    0.96, 0.86, 2.0, 0.921, 0.921: 0.008
4      ``equal10__v2_j8``             1.66, 1.18, 9.5, 0.939, 0.938: 1.000    1.06, 0.99, 2.0, 0.926, 0.925: 0.231
4      ``cluster_covariate__v2_j8``   1.09, 0.98, 9.5, 0.949, 0.949: 1.000    1.02, 0.97, 1.6, 0.944, 0.944: 1.000
6      ``equal10__v2_j12``            1.31, 1.14, 6.2, 0.955, 0.955: 1.000    1.12, 1.06, 1.7, 0.954, 0.954: 1.000
6      ``cluster_covariate__v2_j12``  1.00, 0.97, 1.3, 0.950, 0.950: 1.000    0.99, 0.96, 1.3, 0.950, 0.950: 1.000
8      ``equal10__v5_j10``            1.15, 1.04, 4.1, 0.944, 0.944: 1.000    1.03, 0.95, 2.0, 0.938, 0.938: 1.000
8      ``cluster_covariate__v5_j10``  1.04, 0.98, 1.7, 0.945, 0.945: 1.000    1.05, 0.98, 1.7, 0.945, 0.945: 1.000
16     ``equal10__v5_j20``            1.03, 0.99, 1.7, 0.941, 0.941: 1.000    0.99, 0.95, 1.5, 0.936, 0.936: 0.997
16     ``cluster_covariate__v5_j20``  1.10, 1.07, 1.4, 0.961, 0.961: 1.000    1.10, 1.07, 1.4, 0.960, 0.960: 1.000
24     ``equal10__v5_j30``            1.00, 0.98, 1.3, 0.948, 0.948: 1.000    0.98, 0.95, 1.3, 0.950, 0.950: 1.000
24     ``cluster_covariate__v5_j30``  1.02, 1.00, 1.3, 0.958, 0.958: 1.000    1.02, 1.00, 1.3, 0.958, 0.958: 1.000
=====  =============================  ======================================  =====================================

**Sizing and predictions** (``tests/unit/test_few_cluster_fold_variance_design.py``). Every cell
takes 8,000 draws. Predicted reds at the pilot's readings
(:data:`~tests.studies.few_cluster_fold_variance.PREDICTED_RED`), each held by the red-cell owner
``X54-training-floor`` (:data:`~tests.studies.few_cluster_fold_variance.PREDICTED_RED_OWNERS`):
estimated, both cells at 2 training clusters (the covariate cell's coverage band, ``equal10``'s
G1 at 5.4% withheld); known, the same two and ``equal10__v2_j8`` (coverage 0.926). Predicted
floors: 4 estimated and 6 known, the plan's. Above them the plug-in family figure, with every
cell's deflated and IID controls included, is 1.000 (estimated) and 0.997 (known), against 0.970
and 0.847 at 4,000. Every deflated control fails the rule with plug-in power at least 0.999, and
the median IID control has power 1.000 in every cell (medians 0.35 to 0.47 on the covariate law,
0.72 to 0.90 on ``equal10``). The Jeffreys-predictive family figures, with each cell's shown
coverage integrated over its pilot posterior, are 0.718 (estimated) and 0.646 (known)
(:data:`~tests.studies.few_cluster_fold_variance.PREDICTIVE_FAMILY`). The weakest cells are known
``equal10__v5_j20`` (0.87) and both configurations' ``cluster_covariate__v5_j10`` (0.91), so a
chance red at 16 or 24 training clusters is read against these figures.

**Red-cell readings** (plan 5.8, declared before any run, and coded in
:func:`~tests.studies.few_cluster_fold_variance_properties.declared_route`). The record is
``reporting``, so a red gated verdict is published and takes the route named here. No cell is
re-sized, re-drawn, dropped or relabelled after a verdict is seen.

* ``unlicensed``: in some cell the configuration's deflated control passes the positive rule, or
  the IID control sees no cluster effect. The instrument could not see the failure it targets,
  so nothing ships from the configuration, and the study is redesigned under a fresh
  declaration (a design-test failure, not a red cell).
* ``calibrated`` (``T* = 2``): no training floor for that mechanism; cv-tmle.md and inference.md
  state that the centred fold variance is calibrated from 2 training clusters at 2 to 6 clusters
  per validation fold (V = 2 and 5), equal sizes, with that propensity; the X54 (e) row is
  deleted when both configurations have shipped.
* ``training floor`` (``T* > 2``): ``MINIMUM_FOLD_TRAINING_CLUSTERS[mechanism] = T*`` (plan 3.3,
  keyed ``"estimated"`` and ``"known"``), so a fold-evaluated estimate whose smallest training
  complement holds fewer positive-mass clusters takes ``"few_cluster_plugin"``; validation folds
  keep only the preflight's 2. Each red positive cell below ``T*`` whose variance-ratio lower end
  passes is held by ``X54-training-floor`` (its coverage band and G1 legs) under one of two
  readings. Below the configuration's predicted floor (4 estimated, 6 known) the reading is
  ``training-floor limit``, and inference.md and cv-tmle.md state the floor per mechanism and
  that the limit is the training complement. At or above the predicted floor, or on the
  non-monotone signature (a red count above a count of the same configuration at which every
  cell passes, which the training-complement mechanism, monotone in the count, cannot give), the
  reading is ``finite-sample (plug-in sizing)``: the run is sized on the plug-in figures (1.000
  and 0.997), and the predictive figures 0.718 and 0.646 give such a red that chance. The floor
  still rises past the red cell, and the docs state the measured floor without naming the
  training complement as its cause.
* ``no floor (finite-sample)``: a red at 24 training clusters, so no training floor, with every
  red's variance-ratio lower end passing and a no-defect diagnosis recorded for that
  configuration in :data:`~tests.studies.few_cluster_fold_variance.NO_DEFECT_DIAGNOSES` (empty
  at this declaration). The configuration ships no ``MINIMUM_FOLD_TRAINING_CLUSTERS`` entry; its
  estimates keep only the cluster-count floor (``few_cluster_plugin`` below 10 positive-mass
  clusters today, X54 part (a)'s ``CLUSTER_INTERVAL_FLOORS`` once shipped). The evidence page
  and F28 record "no certified training count" for the configuration with each red cell's
  reading, and ``X54-training-floor`` holds the red at 24 under the finite-sample reading.
* ``defect suspect``: no training count from which every cell passes without that recorded
  diagnosis, or a red variance-ratio lower end (an underestimated variance, which no owner
  holds). Diagnose (the within-fold weight renormalisation, the centring) and fix before
  anything ships.
* The variance ratio's upper side, the trimmed ratio or the largest SE shows outlier variance
  estimates: a reported limit, not a red cell; cv-tmle.md states it with the trimmed ratio and
  the largest SE at each training count.
* The primary (``equal10__v5_j20``, 1,000 draws, ``ey0``, ``ey1`` and ``ate`` of the shipped
  report) is read by the harness's G1, G2 and SE-ratio band of 0.8 to 1.2. A red primary is a
  defect suspect.
* A replication that raises is never redrawn. A recordable failure costs its key in every arm of
  its cell under the declared bound of 0.05 and is a miss in G1. After the constant-target rule
  the pilot lost no draw. A cell past its bound is red with the count stated.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import few_cluster_fold_variance, few_cluster_fold_variance_properties

if __name__ == "__main__":
    run(
        few_cluster_fold_variance,
        few_cluster_fold_variance_properties,
        here=Path(__file__).resolve().parent,
    )

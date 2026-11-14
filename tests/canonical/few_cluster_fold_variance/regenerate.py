"""Regenerate the ``few-cluster-fold-variance`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container. Laws, learners, cells, sizes,
margins, seeds and replicate counts are those of :mod:`tests.studies.few_cluster_fold_variance`
and :mod:`tests.studies.few_cluster_fold_variance_properties`, fixed before any run and
recomputed by ``tests/unit/test_few_cluster_fold_variance_design.py``. The record is not in
``registered()`` until its run is committed; until then its property module is an uncommitted
subject of the property differential.

Order of the work (plan X54 rev 3.2, sections 2.4, 3.3, 5.3 and 5.8): the cells and arms; the
pilot of :mod:`tests.studies.few_cluster_fold_variance_pilot` on its own seed stream, 400 draws
of each of the twelve cells and 1,600 of the eight whose plug-in pass probability was below 0.90
at 400 under rev 3.1's rule; the rev 3.2 redesign from that pilot; the refit of the pilot's 43
lost draws under the constant-target rule (``--refit-lost``); this declaration; the declared run.

Budget: 0.03 to 0.07 s per key (two fits) on one core, so the 96,000 property keys take about
1.3 CPU-h, and the 1,000-draw primary, the summaries and the failure accounting bring the run to
about 2 CPU-h. Declared 20 CPU-h, the plan's figure (it assumed 0.25 s per fit).

**Design changes from the pilot, made before this declaration.** Each follows from the law or
from the mechanism the pilot showed, not from a verdict:

* The covariate law's cells measure ``ey1``, not the ATE: the arm mean carries the plug-in part
  of the curve, which holds the cluster effect on that law, and the ATE nearly cancels it, so
  the first pilot (discarded) read the IID control at 0.97 to 1.06 on the ATE.
* Cells are read by training clusters, ``J (V - 1) / V`` = 2, 4, 6, 8, 16, 24, with the clusters
  per validation fold (2, 4, 6) as a second axis. At 2 clusters per validation fold the V = 5
  cells read variance ratios of 1.15 and 1.04 and the V = 2 cells 7.6 and 17: the limit is the
  complement the nuisances are fitted on, not the validation fold.
* ``known_propensity`` (the same draw refitted with the law's propensity declared known) is a
  second gated configuration with its own floor, and its deflated control
  ``known_deflated_control`` is added.
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
  (pilot: 6 of 12 cells).
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
times 0.7225): the positive rule fails. On each ``iid_t_control`` row: the distribution-free exact
99% upper end of the median paired IID/cluster SE ratio below 0.95. ``pooled_centered`` is not
gated (:data:`~tests.studies.few_cluster_fold_variance.POOLED_CENTERED_GATED`). Reported: the
variance ratio's interval, its trimmed form, the largest SE and the median SE in SDs, the SE ratio
with ``c_k``, the ratio-of-means IID reading, the withheld counts and every diagnostic arm.

**The training floor** of each configuration
(:func:`~tests.studies.few_cluster_fold_variance_properties.training_floor`) is the smallest
training count from which every cell of the configuration, on both laws, passes and has its
deflated control fail the rule.

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
(:data:`~tests.studies.few_cluster_fold_variance.PREDICTED_RED`): estimated, both cells at 2
training clusters (the covariate cell's coverage band, ``equal10``'s G1 at 5.4% withheld);
known, the same two and ``equal10__v2_j8`` (coverage 0.926). Predicted floors: 4 estimated and 6
known, the plan's. The plug-in pass probability of every cell above the predicted floor is 1.000
(estimated) and 0.997 (known). The deflated controls fail the rule with power at least 0.999 in
every cell but one: ``deflated_control`` of ``equal10__v2_j12``, power 0.55, where the estimated
propensity's tail leaves the deflated variance ratio at 0.94 and its coverage at 0.927
(:data:`~tests.studies.few_cluster_fold_variance.DEFLATED_PREDICTED_PASS`). If that control
passes the rule, the estimated floor reads 8 instead of 4. The median IID control has power
1.000 in every cell (medians 0.35 to 0.47 on the covariate law, 0.72 to 0.90 on ``equal10``).

**Red-cell readings** (plan 5.8, declared before any run). The record is ``reporting``, so a red
gated verdict is published and takes the route named here. No cell is re-sized, re-drawn,
dropped or relabelled after a verdict is seen.

* Every cell of a configuration passes (``T* = 2``): no training floor for it; cv-tmle.md and
  inference.md state that the centred fold variance is calibrated from 2 training clusters at 2
  to 6 clusters per validation fold (V = 2 and 5), equal sizes, with that propensity; the X54 (e)
  row is deleted.
* A configuration passes from ``T* > 2``: ``MINIMUM_FOLD_TRAINING_CLUSTERS[mechanism] = T*``
  (plan 3.3, keyed ``"estimated"`` and ``"known"``), so a fold-evaluated estimate whose smallest
  training complement holds fewer positive-mass clusters takes ``"few_cluster_plugin"``;
  validation folds keep only the preflight's 2. inference.md and cv-tmle.md state the floor per
  mechanism and that the limit is the training complement; F28 gains the measured limit.
* The variance ratio's upper side, the trimmed ratio or the largest SE shows outlier variance
  estimates: a reported limit, not a red cell; cv-tmle.md states it with the trimmed ratio and
  the largest SE at each training count.
* No cell of a configuration passes even at 24 training clusters, or a variance-ratio lower end
  fails with its interval excluding 1 at 16 and 24: a defect suspect, diagnosed (the within-fold
  weight renormalisation, the centring) and fixed before anything ships.
* A coverage interval reaching below 0.92 in a cell whose variance-ratio lower end passes: the
  variance is too small on typical draws behind a heavy upper tail; the cell is red and counts
  against the floor, and the reading is recorded with the median SE ratio.
* A deflated control passes the rule in a cell: the instrument cannot see a deflated variance
  there, so that cell certifies nothing for its configuration and the floor is read above it
  (:func:`~tests.studies.few_cluster_fold_variance_properties.training_floor`). Predicted for
  ``equal10__v2_j12`` (estimated) with probability 0.45. If a deflated control passes in a cell
  at 16 or 24 training clusters, or in more than two cells, the study cannot license a
  calibration verdict for that configuration (plan 5.8): nothing ships from it, and it is
  redesigned under a fresh declaration.
* An ``iid_t_control`` red: the law has no cluster effect the instrument can see in that cell, a
  design-test failure, redesigned under a fresh declaration.
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

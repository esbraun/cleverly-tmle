"""Regenerate the ``few-cluster-fold-variance`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container. Laws, learners, cells, sizes,
margins, seeds and replicate counts are those of :mod:`tests.studies.few_cluster_fold_variance`
and :mod:`tests.studies.few_cluster_fold_variance_properties`, fixed before any run and
recomputed by ``tests/unit/test_few_cluster_fold_variance_design.py``. The record is not in
``registered()`` until its run is committed; until then its property module is an uncommitted
subject of the property differential.

Order of the work (plan X54 rev 3.1, sections 2.4, 5.3 and 5.8): the cells and arms; the pilot
of :mod:`tests.studies.few_cluster_fold_variance_pilot` on its own seed stream, 400 draws of each
of the twelve cells and 1,600 of the eight whose plug-in pass probability was below 0.90 at 400,
committed as ``pilot.csv.gz``; this declaration; the declared run. Budget: about 0.05 s per fit
and two fits per key, so 12 cells by 8,000 draws plus the 1,000-draw primary is about 3 CPU-h on
one core; declared 12 CPU-h, the plan's figure.

Two design changes were made from the pilot before this declaration, and the first pilot was
discarded with them. Each follows from the law, not from a verdict:

* The covariate law's cells measure ``ey1``, not the ATE. The plan chose that law because the
  cluster-shared covariate puts the cluster effect in the plug-in part of the curve. The arm mean
  carries that part, as in the covariate pair of ``clustered-unequal-cvtmle``; the ATE nearly
  cancels it, and the first pilot read the IID control's paired SE ratio at 0.97 to 1.06 there,
  so no IID control could fail. ``equal10`` keeps the ATE.
* A reported ``known_propensity`` arm refits each draw with the law's true propensity. It is the
  paired diagnosis the plans guide asks for before a confirmed departure is declared, and it is
  never gated.

What is gated (plan 5.3): on every ``t_reference`` row, the 99% percentile-bootstrap interval of
the variance ratio, mean squared SE over the variance of the estimates over the shown fits,
inside [0.8649, 1.1449], and the exact 99% interval of shown-fit coverage inside [0.92, 0.98];
on every ``iid_t_control`` row, the 99% upper end of the paired mean IID SE over mean cluster SE
below 0.80; on every ``deflated_control`` row (the shipped variance times 0.7225), the 99% upper
end of its variance ratio below 0.8649. ``pooled_centered`` is not gated anywhere
(:data:`~tests.studies.few_cluster_fold_variance.POOLED_CENTERED_GATED` states the derivation
and the pilot readings). Published beside them: the SE ratio and its interval, the reference
degrees of freedom ``k = J - V`` with ``c_k``, the withheld count, the over-coverage reading, and
the variance ratio and coverage of every diagnostic arm, ``known_propensity`` included.

Pilot readings (``pilot.csv.gz``; variance ratio of ``t_reference``, its median-SE form, shown
coverage, and the ``known_propensity`` variance ratio):

=============================  ======  ======  =====  =====
cell                           VR      median  cover  known
=============================  ======  ======  =====  =====
``equal10__v2_j4``             7.63    1.17    0.952  1.17
``equal10__v2_j8``             1.66    1.13    0.939  1.06
``equal10__v2_j12``            1.31    1.13    0.955  1.12
``equal10__v5_j10``            1.15    1.01    0.944  1.03
``equal10__v5_j20``            1.03    1.01    0.941  0.99
``equal10__v5_j30``            1.00    0.99    0.948  0.98
``cluster_covariate__v2_j4``   17.4    0.79    0.920  0.97
``cluster_covariate__v2_j8``   1.09    1.01    0.949  1.02
``cluster_covariate__v2_j12``  1.00    1.00    0.950  0.99
``cluster_covariate__v5_j10``  1.04    0.98    0.945  1.05
``cluster_covariate__v5_j20``  1.10    1.10    0.961  1.10
``cluster_covariate__v5_j30``  1.02    1.02    0.958  1.02
=============================  ======  ======  =====  =====

The sizing (plan 5.3, at exact calibration and each cell's pilot spread) puts every cell at 8,000.
The two cells with 2 training clusters per fold cannot reach 0.90 at any affordable R (their
relative variance of the squared SE is 25 and 306), and the product over the other ten is 0.933,
a recorded shortfall of the family rule. Seven positive cells are predicted red at the pilot's
own readings (:data:`~tests.studies.few_cluster_fold_variance.PREDICTED_RED`), and the IID control
of ``equal10__v2_j4`` has no power there.

Diagnosis before the declaration (guide: "diagnose a confirmed departure before the
declaration"). Every confirmed departure is above the band, so the shipped interval is
conservative there and coverage reads 0.92 to 0.96. Most of the excess goes with the fitted
propensity on small training folds: the ``known_propensity`` arm on the same draws reads 7.63 to
1.17, 17.4 to 0.97, 1.66 to 1.06 and 1.15 to 1.03. The median-SE form shows a tail: a training
fold of 2 to 6 clusters occasionally fits extreme propensities, which give curve variances that
the bounded targeted point does not follow. The variance construction is the package's own, and
the renormalisation and centring are pinned by the arm tests of the design test. Two residual
excesses are open: ``equal10`` at V = 2 keeps 1.06 to 1.17 with the true propensity, and
``cluster_covariate__v5_j20`` reads 1.10 in both arms while its neighbours at 10 and 30 clusters
read 1.04 and 1.02.

Red-cell readings (declared before any run). The record is ``reporting``, so a red gated verdict
is published and takes the route of plan section 5.8 named here. No cell is re-sized, re-drawn,
dropped or relabelled after a verdict is seen.

* Every gated row passes: no estimator change; cv-tmle.md and inference.md state that the centred
  fold variance is calibrated at 2, 4 and 6 clusters per fold, V = 2 and 5, at equal sizes; the
  X54 (e) row is deleted.
* Positive cells pass from ``m*`` clusters per fold and fail below it on both laws and both fold
  counts: ``MINIMUM_FOLD_CLUSTERS = m*`` (plan 3.3), and a fold-evaluated estimate whose smallest
  validation fold has fewer than ``m*`` positive-mass clusters takes ``"few_cluster_plugin"``.
* Positive cells fail at V = 2 and pass at V = 5 for the same clusters per fold: the limit is the
  training fold, ``J (V - 1) / V`` clusters, not the validation fold. The 5.8 per-fold floor does
  not describe it. The failing designs are recorded under F28 as a measured limit with their
  training-cluster counts and their ``known_propensity`` readings, and the floor that covers them
  is left to a fresh plan decision; no floor is set from this run alone.
* A red positive cell whose variance-ratio interval lies above the band, whose coverage interval
  reaches no lower than 0.92, and whose ``known_propensity`` arm reads inside the band is a
  conservative finite-sample limit of the fitted propensity on that training fold, owned by F28.
  The same red with a ``known_propensity`` reading above the band is a defect suspect for the
  construction and is diagnosed before anything else (plan 5.8: the within-fold weight
  renormalisation, the centring).
* A variance ratio below the band, or a coverage interval reaching below 0.92, in any cell is a
  defect suspect (an underestimated variance), diagnosed before anything else.
* A ``deflated_control`` reads 0.7225 times its positive's variance ratio on the same draws. Its
  red in a cell whose positive interval lies above the band is implied by that positive and is
  read with it. Its red in a cell whose positive passes is the instrument failure of plan 5.8: the
  study cannot license a calibration verdict, and it is redesigned under a fresh declaration.
* An ``iid_t_control`` red in a cell whose positive is predicted red by its tail
  (``equal10__v2_j4``, declared here) is read with that positive. Elsewhere it means the law has
  no cluster effect the instrument can see in that cell, a design-test failure redesigned under a
  fresh declaration.
* The primary (``equal10__v5_j20``, 1,000 draws, ``ey0``, ``ey1`` and ``ate`` of the shipped
  report) is read by the harness's G1, G2 and SE-ratio band of 0.8 to 1.2. A red primary is a
  defect suspect.
* A replication that raises is never redrawn. A recordable failure costs its key in all seven
  arms of its cell under the declared bound of 0.05; the pilot lost 2.1% of the draws of
  ``equal10__v2_j4`` and 0.6% of ``cluster_covariate__v2_j4`` (a training fold with one outcome
  class or one arm in fewer than two units) and none elsewhere. A cell past its bound is red with
  the count stated.
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

"""Regenerate the ``few-cluster-adaptive-learners`` evidence under the declared run form.

**State: declared.** The 400-draw pilot of :mod:`tests.studies.few_cluster_adaptive_pilot`
(design review B1) is committed as ``pilot.csv.gz``: 38 fits of 400 draws, 25,600 rows, no
fit raised, 26.9 CPU-h. The predicted pass probabilities, the predicted red cells, the primary
count and the budget below are derived from it, and the design test recomputes each one.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
Run it from the repository root as a module,
``python -m tests.canonical.few_cluster_adaptive_learners.regenerate``, so that ``tests``
imports, with every BLAS and OpenMP thread variable at 1 (``README.md``). The record has no
comparator, so the run starts no container. Laws, learners, fits, arms, cells, margins, seeds
and replicate counts are those of :mod:`tests.studies.few_cluster_adaptive_learners` and
:mod:`tests.studies.few_cluster_adaptive_properties`, fixed before any run and recomputed by
``tests/unit/test_few_cluster_adaptive_design.py``. The record is not in ``registered()`` until
its run is committed; until then its property module is an uncommitted subject of the property
differential, on synthetic rows.

Order of the work (plan X54 rev 3.2, section 5.4, and the S-h design review): the laws (L2 in
:mod:`tests.studies.clustered_nonlinear_law`, its truth by quadrature checked at two
resolutions and against a large draw); the 400-draw pilot on its own seed stream, committed as
``pilot.csv.gz``; a diagnosis of any departure it confirms, read against the paired GLM arms
(below); this declaration; the declared run.

What the pilot reads (400 draws per cell; ``few_cluster_adaptive_pilot.pilot_readings``):

* Stacked default library: the variance ratio is 0.93 to 1.13, each 99% interval containing 1,
  and the paired all-draw coverage difference to the stacked GLM arm on the same draws is
  -0.008 to 0.013, each exact 99% interval containing 0. The two departures the 20-draw pilot
  hinted at (L1 at 20 clusters with a variance ratio of 0.52, L2 at 10 with 0.54) are not
  confirmed: they read 1.00 and 1.01. No departure of the stacked library from the parametric
  baseline is confirmed, so no diagnosis was needed before this declaration.
* At 10 clusters both laws read coverage 0.922 and 0.920 for the stacked default library and
  0.920 and 0.928 for the stacked GLM arm on the same draws (paired differences 0.003 and
  -0.008). The committed exact-propensity GLM of ``clustered-few-cluster-tmle`` reads 0.935 at
  the same law and count over 4,000 draws. The deficit belongs to the few-cluster interval and
  the parametric baseline shares it; the shown-coverage floor of 0.92 is what sees it.
* L2 at 40 clusters reads coverage 0.930 and a variance ratio of 0.93 for the stacked default
  library, and 0.918 and 0.90 for the stacked GLM arm on the same draws (paired difference
  0.013). L2 at 20 reads 0.958 and 1.13. Neither interval excludes nominal coverage or a ratio of
  1, so the reading may be one draw set; it is not a departure of the library from the baseline.
* In sample: variance ratios 0.68 to 0.88 and coverage 0.888 to 0.918, 1.5 to 3.3 points under
  the in-sample GLM arm on the same draws at every count, which is the overfitting the theory
  predicts (design review M2) and gates nothing.
* The stacked control covers 0.81 to 0.85 with a variance ratio of 0.45 to 0.53 at every count,
  and its power to fail at 4,000 draws is at least 0.989 at every count (``PREDICTED_PASS``), so
  the stacked caveat can be removed at every count where the positive passes. The cross-fitted
  ``LTMLE`` cell reads the L1 control at 20 clusters (power 1.000).
* Every IID control passes with predictive probability 1.000.

Sizing. At target (variance ratio 1, nominal coverage, paired difference 0) every clause of
every positive row passes with probability at least 0.9993 at 4,000 draws on the pilot's
spreads (``few_cluster_adaptive_pilot.at_target_pass``), so no cell is raised to 4,800. The
in-sample cells, reported only, keep 4,000 draws: they share their draws with the stacked and
GLM fits, and their bias over SD and paired GLM difference are read against the stacked rows'
at equal precision, which is the in-sample against cross-fitted comparison of Remark 3. The
primary takes 1,200 draws, the smallest multiple of 400 of at least 1,000 whose predictive
family-wise pass reaches 0.95 (0.956: coverage floor 0.959, SE-ratio band 0.999, bias 0.997;
0.933 at 1,000). No interim stop (design review m3), for three reasons. The two cells it would
stop, the stacked rows at 10 clusters, still pass with probability 0.12 to 0.17, and a stop at
1,000 draws would forgo that chance. Their declared reading rests on the paired GLM difference,
which 4,000 draws resolve to a 99% half-width of about 0.007. The harness has no interim rule,
and one would need its own validation and failure accounting. Those cells cost about 53 CPU-h,
so an interim would save at most about 40.

Budget. Pilot CPU seconds per fit, on the shared 4-core cloud machine, by count 10, 20 and 40:
stacked default library 24.0, 26.8 and 34.4 on L1 and 23.8, 25.4 and 32.8 on L2; in sample 4.9,
5.5 and 7.4 on L1 and 4.8, 5.3 and 7.0 on L2; cross-fitted ``LTMLE`` 36.2; the stacked control
0.5 to 0.7; every other fit under 0.1. At 4,000 draws per cell: stacked 186 CPU-h, in sample 39,
L3 40, the controls and GLM fits 5, the 1,200-draw primary 9, and the summary's bootstraps
under 1, so about 279 CPU-h (``few_cluster_adaptive_pilot.projected_cpu_hours``), against the
plan's declared 270. That is about 17.5 hours on the local 16-core queue. The heavy keys run
first (``few_cluster_adaptive_properties.keys``). Runner rule (plan 5.4, design review m2): the smoke
re-measures each per-fit time on the queue's machine, and a time more than 1.5 times the
pilot's figure above goes back to the row thread before the run.

What is gated, and why (plan 5.4, review findings B3 and N3, design review M1 and M2). The
gated fits are stacked ``TMLE`` and cross-fitted ``LTMLE``. Each positive row (``t_reference``
of ``tmle_stacked`` at each point law and count, and of ``ltmle_crossfit``, 7 rows) passes when
all of these hold:

* ``coverage floor`` (G1): the exact 99% lower end of all-draw coverage at least 0.90, a
  withheld fit a miss and a lost fit a miss in the worst case of the failure accounting;
* ``bias equivalent`` (G2): the 99% bias interval inside 0.25 empirical SD;
* ``variance ratio lower end``: the 99% percentile-bootstrap lower end of mean(SE^2) over the
  variance of the estimates, on shown fits, at least 0.8649. The variance ratio and not the SE
  ratio, because the SE ratio of a calibrated variance sits at the Jensen factor;
* ``shown coverage lower end``: the exact 99% lower end of shown-only coverage at least 0.92. It
  reads coverage directly, so a few outlier SEs cannot carry it as they can the variance ratio;
* ``paired non-inferiority``, on stacked L1 rows: the exact 99% lower end of the paired
  all-draw coverage difference to the ``glm_reference`` arm of the same draws above -0.02.

Each gated ``iid_t_control`` passes the two-clause rule of the registered few-cluster studies,
and ``stacked_control`` passes when the exact 99% upper end of its all-draw coverage is below
0.90. A configuration's caveat goes at a law and count only where its positive row, its IID
control and its must-fail control all pass (``caveat_removed``). The stacked control's outcome
is an unpruned random forest fitted on the full sample, standing in for every out-of-fold
prediction, so its residuals shrink but do not vanish, and the out-of-fold assembly, the
stacked targeting and the stacked variance all act on it (design review M1). If the pilot reads
its power below 0.90 at a count, the stacked caveat cannot be removed at that count (plan 5.4).
No control plants an overfit ``LTMLE`` outcome fit, so the cross-fitted ``LTMLE`` cell reads the
stacked control of L1 at 20 clusters, a point-``TMLE`` control; the evidence page states that
limit.

The in-sample default-library fit is reported, not gated (design review M2, option (b)). The
Super Learner cross-validates only its weights and predicts from each candidate's full-sample
fit (``cleverly/learners/super_learner.py``), so an in-sample fit's residuals shrink at any
cluster count and no Donsker argument covers it. It therefore takes the ``adaptive_nuisance``
shortfall on the ``t`` tier by theory (``cleverly._support.ADAPTIVE_IN_SAMPLE_REASON``), and at
40 or more clusters no clustered in-sample row covers it. Its cells, its IID control and the
interpolating control are published with every reading, among them the paired coverage
difference to the GLM arm on L1 and L2.

The ``glm_reference`` arm fits a main-terms logistic outcome and an estimated main-terms logistic
propensity. The committed parametric baseline uses the exact propensity. On L1 the main-terms
propensity is correctly specified, and an estimated correct propensity makes the influence-curve
variance conservative under a misspecified outcome, so the paired comparator is at least as
strict as the exact-propensity one and the arms differ only in the learner library.

Red-cell readings (declared before any run). The record is ``reporting``, so a red gated verdict
is published, and it takes the route of plan section 5.8 named here. No cell is re-sized,
re-drawn, dropped or relabelled after a verdict is seen, and no margin differs from the harness
defaults except the declared thresholds above.

* A stacked positive row passes G1 and G2 but fails the variance-ratio, shown-coverage or
  non-inferiority clause at a count below 40, or its control fails or lacks power there: the
  stacked configuration takes the ``adaptive_nuisance`` shortfall of its ``t``-tier row
  (``T11.stacked``) at that count and below
  (``cleverly._support_rows.ADAPTIVE_NUISANCE_SHORTFALLS``, experimental label, interval
  shown). The floor reading is unchanged, and F28 records the limit.
* A stacked positive row fails G1 or G2 at 10 or 20 clusters: the same shortfall, recorded with
  its coverage, withheld share and bias.
* A stacked positive row is red at 40 clusters (design review M3). It is first diagnosed as a
  defect suspect: the stacked targeting and variance code, read against the paired GLM arm and
  the stacked control on the same draws. A code defect is fixed and the study re-run under a
  fresh declaration. With no code defect, the paired all-draw coverage difference to the
  stacked GLM arm on the same draws decides the reading. If its exact 99% lower end is at or
  below -0.02 (the non-inferiority margin), the red is the default library's limit at 400 rows:
  the clustered row ``T10.unweighted`` takes an entry with ``below=None``, so every clustered
  stacked fit with an adaptive nuisance learner is experimental at every cluster count, with the
  sentence of ``AdaptiveNuisanceShortfall.sentence`` ("measured with parametric nuisance
  learners only: ... fails at 40 clusters for stacked TMLE, and no study measures a larger
  count"). The qualifier is set on the ``clustered`` tier for this, and
  ``tests/unit/test_adaptive_nuisance_support.py`` holds its witness. The ``t``-tier row takes
  the shortfall at every count below 40 too, because a limit measured at 40 holds below it. If
  the lower end is above -0.02, the parametric baseline shares the deficit on the same draws:
  a label naming the learner would misattribute it, so no shortfall is added, the stacked
  caveat stays at that law, and the evidence page states the law, the count, both arms'
  coverage and the paired difference. Plan 5.8 names this case ("defect suspect for the library
  or the law").
* Every clause of the stacked configuration passes at every count: no shortfall for it, and the
  docs sentences of plan 5.8 row 1 change for stacked fits; the in-sample sentence stays.
* A red row of the cross-fitted ``LTMLE`` cell takes the shortfall for cross-fitted ``LTMLE``
  at 20 to 39 clusters (``G07``).
* A must-fail control passes (covers): the study cannot see an overfit stacked outcome fit at
  that law and count, so the stacked caveat stays there. That is not a shortfall of the positive
  row.
* The primary (L1 at 20 clusters, stacked, 1,200 draws) reads the configuration of the
  ``tmle_stacked__equal10__j20`` cell. A red primary leg takes that cell's reading; it is not a
  separate defect.
* A replication that raises is never redrawn. A lost fit costs its key in its fit's cells only,
  under the declared bound of 0.01 per side, and every declared rule reads it at the worst
  fill. A cell past its bound is red with the count stated.

Predicted reds (``few_cluster_adaptive_learners.PREDICTED_REDS``, a predictive pass
probability below 0.50, the plug-in rule naming the same rows), each owned by
``X54-adaptive-learners`` (``PREDICTED_RED_OWNERS``; the roadmap's Red-cell owners table):

* ``tmle_stacked__equal10__j10__t_reference`` (0.171): the shown-coverage clause (0.26) and G1
  (0.76). Expected reading: the deficit is shared with the parametric baseline (paired
  difference 0.003, non-inferiority 0.965). Route: the ``T11.stacked`` shortfall below 20.
* ``tmle_stacked__nonlinear10__j10__t_reference`` (0.123): the same clauses (0.21 and 0.70) and
  the same expected reading (paired difference -0.008). Route: the same shortfall.
* ``tmle_stacked__nonlinear10__j40__t_reference`` (0.239): the shown-coverage clause (0.46) and
  the variance-ratio clause (0.58). Expected reading: shared with the parametric baseline on
  the same draws (paired difference 0.013), so no shortfall after the defect-suspect diagnosis,
  and the stacked caveat stays at L2. A ``default-library limit`` reading would take the
  ``T10.unweighted`` route above.

The other positives are not predicted red but can go red: L1 at 20 (0.756), L1 at 40 (0.870),
L2 at 20 (0.974) and the cross-fitted ``LTMLE`` cell (0.924). A red there takes the route above
for its count, and its owner is ``X54-adaptive-learners`` under the same two readings.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import few_cluster_adaptive_learners, few_cluster_adaptive_properties

if __name__ == "__main__":
    run(
        few_cluster_adaptive_learners,
        few_cluster_adaptive_properties,
        here=Path(__file__).resolve().parent,
    )

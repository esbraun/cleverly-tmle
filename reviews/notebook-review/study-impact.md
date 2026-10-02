# Study impact of the generator fixes

Base: `agent/notebook-review` = origin/main `5f33902b`. Local `.venv` is Python 3.11.13, SciPy 1.17.1,
NumPy 2.4.6, scikit-learn 1.9.0, pandas 3.0.5. The four affected studies used exactly this runtime.
Docker 29.8.0 runs, and every reference image is built locally (`cleverly-tmle3-cvtmle-reference:ed72f8a`,
`cleverly-zepid-reference:16a0f96`, `cleverly-ctmle3-oat-reference:a4ea77b`, ...).

Scratch evidence is in `.tmp/notebook-review/study-impact/`:

| file | what it holds |
| --- | --- |
| `probe_cells.py` | enumerates every registered study's property cells, then matches each cell's propensity against `nonlinear_dgp().propensity` on 500 fixed covariate rows |
| `patch_a.py`, `patch_b.py` | pytest plugins that monkeypatch fix A (`g = 0.05 + 0.90 expit(logit)`) and fix B (softmax mixed with a uniform floor, lambda = 0.06) |
| `baseline.txt`, `patched_a.txt`, `patched_a_evidence.txt` | fix A: affected tests, unpatched and patched; evidence-recompute tests, patched |
| `baseline_b.txt`, `patched_b.txt` | fix B: affected tests, unpatched and patched |
| `probe_msm*.py`, `probe_msm*.txt` | MS-02 slope-coverage sweep, raw and floored law |
| `probe_sn.py`, `probe_sn_variants.py`, `probe_sn_100.txt` | SN-01 / SN-04 longitudinal encoding probe |

## Candidate A: `nonlinear_dgp` propensity bounded to [0.05, 0.95]

The fix changes the treatment draw `A ~ Bern(g)`. It therefore changes every sample drawn by
`nonlinear_dgp`, `nonlinear_bounded_dgp`, `make_nonlinear_ate`, `make_nonlinear_bounded` and
`navigation_data`, with any seed. `ey1`, `ey0` and `ate` stay unchanged. `att` and `atc` change.

### A.1 Registered studies that sample the law

`probe_cells.py` found eight cells in four studies. All eight sample `nonlinear_bounded(concentration=12.0)`
through `bounded_cv_laws.nonlinear_dgp()`. No primary scenario uses the law: the primary runners draw
`canonical_tmle.continuous_dgp` or `binary_outcome_dgp`.

| study slug | artifacts | affected cells (n x reps) | publication | reference | evidence page |
| --- | --- | --- | --- | --- | --- |
| `canonical-cvtmle` | `tests/canonical/tmle3_cvtmle/` | `crossfit_overfitting/stacked_cvtmle`, `crossfit_overfitting/in_sample_control` (500 x 400 each) | gated | R `tmle3` (Docker) | `docs/technical-reference/method-evidence/stacked-point-treatment-cv-tmle.md` |
| `fold-evaluated-cvtmle` | `tests/canonical/cvtmle_fold/` | `crossfit_overfitting/fold_evaluated_cvtmle`, `.../in_sample_control` (500 x 400) | gated | none | `fold-evaluated-point-treatment-cv-tmle.md` |
| `fold-targeted-cvtmle` | `tests/canonical/zepid_cvtmle/` | `crossfit_overfitting/fold_targeted_cvtmle`, `.../in_sample_control` (500 x 400) | gated | Python `zEpid` (Docker) | `fold-targeted-point-treatment-cv-tmle.md` |
| `canonical-ctmle-oat` | `tests/canonical/ctmle3_oat/` | `robustness_contract/outcome_correct`, `robustness_contract/outcome_wrong` (700 x 1200); `crossfit_overfitting/cross_fitted_oat`, `.../in_sample_control` (500 x 400) | reporting | R `ctmle3` (Docker) | `outcome-adaptive-point-treatment-c-tmle.md` |

The ledger's PT-N1 row names `canonical_properties.py` as affected. That is wrong. `double_robustness_dgp()` (`canonical_properties.py:105-137`)
calls `replace(nonlinear_dgp(), propensity=<own tanh-squashed copy of the logit>)`. The shipped propensity
never runs in that law. `canonical-tmle`, `repeated-crossfit-tmle`, `canonical-ctmle-selector` and
`canonical-multi-arm-drtmle` import the modules, but they sample no affected cell. The double-robustness
cells of the four studies above are also unaffected (exact ATE 0.2368749 / 1.75).

**Trap.** `tests/unit/test_study_drivers.py::test_the_bounded_mechanism_is_the_shipped_linear_predictor_squashed`
fails under fix A, because it derives the expected value from `nonlinear_dgp().propensity`. Fix the test.
Compare against the shipped linear predictor, for example by exposing the logit as a module function in
`synthetic.py`. Do **not** re-derive `double_robustness_dgp` from the new propensity. That would move the
double-robustness cells of `canonical-tmle`, `canonical-cvtmle`, `fold-evaluated-cvtmle`,
`fold-targeted-cvtmle`, `repeated-crossfit-tmle` and the bounded twins.

### A.2 Regeneration commands and wall time (16 cores)

| study | command | measured wall time | source |
| --- | --- | --- | --- |
| `canonical-cvtmle` | `python -m tests.canonical.tmle3_cvtmle.regenerate --jobs 16 --output <empty dir>` | 86.3 min complete (Python primary + R `tmle3` reference 82 min, properties about 4 min) | `~/.codex/review-runs/20260929-01a0efbd/state.json`, run.log `wall seconds: 5178.9` |
| `fold-evaluated-cvtmle` | `python -m tests.canonical.cvtmle_fold.regenerate --jobs 16 --output <dir>` | 4.2 min | same, `wall seconds: 251.0` |
| `fold-targeted-cvtmle` | `python -m tests.canonical.zepid_cvtmle.regenerate --jobs 16 --output <dir>` | 7.1 min (includes zEpid container) | same, `wall seconds: 426.6` |
| `canonical-ctmle-oat` | `python -m tests.canonical.ctmle3_oat.regenerate --jobs 16 --output <dir>` | 3 min 20 s with the R comparator not re-run | commit `a01177b6` message |

Total: about 1 h 45 min with every reference re-run. About 20 to 40 min if the R reference rows are
reused through `--skip-reference`. The canonical-cvtmle Python-only primary time is not recorded
separately. Run the studies one at a time. Do not run the fast suite beside them.

### A.3 Reference (R / Docker) runs

No reference input moves. The comparators run only on the primary scenarios, and fix A does not touch
those laws. `tests/canonical/regenerate.py` offers `--skip-reference` ("reuse the committed reference
rows"). Precedent: `a01177b6` regenerated `canonical-ctmle-oat` without re-running R. The Python
primary phase still refits, so `replicates.csv.gz`, `summary.csv`, `equivalence.csv` and
`performance-tests.csv` should come back byte-identical on the 3.11.13 / 1.17.1 venv. That is a
free witness that only the property cells moved. Report it in the PR. If the runtime differs, expect
roundoff drift (roadmap, "Targeted execution amendment").

A targeted rerun is not eligible. `testing-strategy.md#targeted-reruns` requires a "fixed design:
preserve the laws". Use complete reruns. Three of the four manifests are now `targeted_seed_repair`
composites. A complete rerun replaces them with a plain manifest. `test_targeted_composites_attribute_carried_and_replaced_evidence`
accepts that, because the new `generated_with` differs from the plan's `baseline_generated_with`.
`tests/canonical/seed-repair-plan.json` stays as history.

### A.4 Obligations before the run

| obligation | where | source |
| --- | --- | --- |
| declare the affected studies, the commands, and that laws other than the propensity, learners, n, budgets, seeds, margins and reading rules are unchanged | a `docs/roadmap.md` section, in a commit pushed before any run (precedent `bb4772a9`, "Declare the affected simulation study reruns") | `method-benchmarking.md` "Independent statistical evidence"; memory "regenerate once, last" |
| declare the red-cell route before the run. Three studies are `gated`. If an overfitting cell goes red, the only admissible route is `publication_policy="reporting"`, declared before the run that writes artifacts. A gated failure refuses publication | the same roadmap section and `registry.py`, if chosen | `method-benchmarking.md` "Registration and acceptance"; memory "red cells go to reporting policy" |
| no margin, budget, seed, or learner change after seeing a result. No dropped replications | — | `method-benchmarking.md` |
| disposable smoke first. `--replicates 10` skips properties, so the smoke checks only the plumbing | output outside the repository | `method-benchmarking.md`; `regenerate.py:_arguments` |
| record each study, command, moved artifacts, and byte-identical artifacts in the PR. Run `python -m tests.studies.evidence.document`. Check the manifest hashes. Write LF | PR body | `pull-requests.md` "Evidence pull requests" |

### A.5 Reader-facing text that quotes these studies or this law

| place | what moves |
| --- | --- |
| `stacked-point-treatment-cv-tmle.md:138-139, 197-201, 243` | overfitting rows and measured values (generated blocks plus the measured-values table) |
| `fold-evaluated-point-treatment-cv-tmle.md:82-83, 132-136, 171` | same. Line 171 describes the tree "fitted on the nonlinear law" |
| `fold-targeted-point-treatment-cv-tmle.md:77-78, 122-126` | same |
| `outcome-adaptive-point-treatment-c-tmle.md:86-93, 127-128, 141-146, 202` | `robustness_contract` and overfitting rows and values |
| `method-evidence/red-cells.md`, `validation-grid.md` | regenerate with `python -m tests.studies.evidence.red_cells` and `document`, if any verdict moves |
| `docs/technical-reference/cv-tmle.md:235` | hand-quoted probe on `make_nonlinear_ate(n=400, seed=11)`: scale (-3.37, 10.39) to (-9.11, 73.49), 0.450 to 0.145. No test checks it. Re-measure |
| `docs/technical-reference/dr-tmle/supported-estimands.md:46-67` | code example only, no printed numbers |
| `src/cleverly/validation/drtmle.py:160-166` | comment: residual `2e-19` / `1.3e-04` "measured on nonlinear_dgp at n = 400" |
| `docs/examples/point-treatment-tmle.ipynb`, `cross-fitting.ipynb`, `dr-tmle.ipynb`, `interventions.ipynb` and `docs/examples/index.md` | every stored output that uses `navigation_data` or `make_nonlinear_bounded`. Re-execute with `python scripts/execute_notebook.py <path>` (needs network). Then re-narrate, and drop the PT-N1 "E[1/g] infinite" narration |
| `tests/unit/tutorial_semantics/{point_treatment_tmle,cross_fitting,dr_tmle,interventions}.py` | callbacks pin stored numbers |
| `src/cleverly/datasets/synthetic.py` docstrings of `nonlinear_dgp`, `nonlinear_bounded_dgp` | state the bound and why |
| `tests/prose-report.md` | refresh with `python -m tests.prose --update` |

### A.6 Fast tests the law change breaks

Measured: patched 10 failed, 1268 passed. Unpatched baseline: 1278 passed. The test set was every file
that names the law, plus `test_documentation_api.py` (all package doctests) and `test_documentation_runtime.py`.

| failing test | why |
| --- | --- |
| `tests/unit/test_datasets.py::test_an_existing_law_draws_what_it_drew_before_the_beta_family[nonlinear_ate]` | golden fingerprint table (`test_datasets.py:850`) pins `A`, `Y`, and `atc` |
| `tests/unit/test_study_drivers.py::test_the_bounded_mechanism_is_the_shipped_linear_predictor_squashed` | see the trap in A.1 |
| `tests/unit/test_drtmle_fit.py::TestTheReportedCurveIsCentredWhereTheBoundBinds::test_the_identity_holds_on_the_draw_that_clips_as_well_as_the_one_that_does_not` | the fixture needs a draw whose mechanism clips. Bounded g no longer clips. Pick a new witness law or seed, and keep a nonzero witness (CLAUDE.md) |
| `...::test_the_targeted_mechanism_no_longer_leaves_the_bounds_at_all` | same cause |
| `tests/unit/test_drtmle_fit.py::TestBothUpdateOrdersReachTheTheoremsExit::test_the_two_routes_agree_on_the_estimate` | draw-specific agreement |
| `tests/e2e/test_ipsi.py::TestTheMechanismIsTheHalfThatMustBeRight::test_a_misspecified_mechanism_biases_the_contrast` | the bias-detection control needs the W2^2 tail. Its control power falls under the bound. Re-check the law, n, or the threshold (a control may re-select its discrimination quantity) |
| `tests/unit/test_documentation_runtime.py::test_tutorial_semantics_at_documented_size[...]` for the four notebooks above | callbacks re-run against the new law |

Every package doctest still passes (`synthetic.py`, `navigation.py`, `simulation.py`, `estimators/tmle.py`, `cleverly/__init__.py`).

### A.7 Fast-suite tests that recompute verdicts from committed artifacts

`test_method_evidence.py` recomputes verdicts, checks the manifest hashes, binds each committed truth
to `cell.dgp.truth()[estimand]`, and checks composites. `test_red_cell_ledger.py`, `test_seed_repair.py`,
`test_study_provenance.py` and `test_bounded_cv_laws.py` also read the artifacts. Measured under fix A:
1012 passed, 0 failed, for the four studies plus the seed-repair, red-cell, bounded-law and provenance
selections. **No fast test notices that the committed artifacts are stale.** The affected cells all
estimate `ate`, whose truth does not move. Staleness is caught only by regeneration. The PR must name
the four studies explicitly.

## Candidate B: `multi_arm_dgp` arm probabilities bounded away from 0

### B.1 Registered studies

None. `multi_arm_dgp` is imported in `tests/studies/` only by `multi_arm_common.law()`. That function
calls `replace(multi_arm_dgp(family="binomial"), arm_logits=<own logits 0.25/0.15>, outcome_mean=<own>)`.
Constraint: implement the floor **inside `multi_arm_dgp`'s `arm_logits` closure**, for example
`log((1 - lam) softmax + lam / K)`. Do not put it in `MultiArmDGP.probabilities()` or in a new
dataclass field. `replace()` would carry a method or field change into the five multi-arm studies:
`canonical-multi-arm-tmle`, `-ctmle-oat`, `-ctmle-selector`, `-drtmle` (20 h), and those using `multi_arm_properties`.
`canonical_point_msm` uses its own law. Consider flooring the gaussian family only. Comment at
`multi_arm_common.py:61` ("generic fixture deliberately has much stronger tails") needs a reword.

No regeneration, no R / Docker run, and no pre-registration are needed.

### B.2 Is the fix warranted? MS-02 re-measured

Same fit as the notebook (`LinearRegression`, `LogisticRegression`, in sample, n = 3000). The fits
reproduce the verifier's per-seed slope and SE exactly (diff below 1e-16).

| seeds | raw law coverage (SE/SD) | floored, lambda 0.06 (SE/SD) |
| --- | --- | --- |
| 2001-2300 (verifier's) | 0.920 (0.930) | 0.937 (1.048) |
| 10000-10999 | 0.952 (1.004) | 0.961 (1.079) |
| 20000-22999 | 0.943 (0.987) | 0.951 (1.024) |
| pooled 4,300 | **0.944** (MC se 0.0035) | **0.952** |

The 0.920-0.925 in the ledger is partly a seed-range excursion. The raw law undercovers by about 0.6
points below 0.95 at n = 3000. The floor removes it. Fix B is defensible but small. A wording fix
("about 94% over 4,300 draws; the heavy tail comes from true arm probabilities near 1e-4") also
suffices. MS-02's quoted 0.925 / 0.920 must not be printed as the law's coverage either way.

### B.3 What else moves

Patched: 3 failed, 1725 passed. Baseline: 1728 passed.

| item | action |
| --- | --- |
| `test_datasets.py::test_an_existing_law_draws_what_it_drew_before_the_beta_family[multi_arm]` and `[multi_arm_binary]` | update golden `A` strings (binary only if floored) |
| `test_documentation_runtime.py::test_tutorial_semantics_at_documented_size[docs/examples/msm-projections.ipynb]` | re-execute `msm-projections.ipynb`. Update `tests/unit/tutorial_semantics/msm_projections.py` pins (slope 0.268753, miss -0.2084, 1.20%, ...) |
| `ey[...]` truths and the MSM projection | unchanged. Only conditional `att` / `atc` truths change |

## Candidate C: other ledger rows

| row | generator | studies it would move | verdict |
| --- | --- | --- | --- |
| CF-04 team law (`make_clustered(..., family="binomial")`, arm x team logit 8.0) | `clustered_dgp` | `canonical-clustered-tmle` (`tests/studies/canonical_clustered_tmle.py`) | wording fix suffices. A strong team effect is a legitimate law. No defect in identification or inference |
| IV-05 negative intensity (`make_shift_dose`, P(A < 0) = 0.056) | `shift_dgp` | `shift-policies` (`canonical_shift_policies.py`, `shift_policy_properties.py`) | wording fix suffices ("intensity index"). A real-valued exposure is valid for an MTP |
| IV-04 "standardized" intensity outcome | none (prose) | none | wording |
| LS-02 / LS-NF2 death-as-censoring functional | optional new truth key in `datasets/longitudinal.py` | adding a key moves no study row. `test_datasets.py` golden table may need the key | optional; wording fix suffices |
| LT-01..LT-08, SN-01..SN-09 laws | `make_longitudinal`, `make_missing_outcome*` | longitudinal studies use `make_longitudinal*`. No study uses `missing_outcome_dgp` | wording only |
| PT-N1 sibling laws (CT `make_instrument`, MS, every logistic-linear law) | — | — | E[1/g] finite. No fix |

## SN-01 / SN-04: does the shipped library already handle MAR among survivors?

Yes, through `LongitudinalTreatment` with two nodes, one of which is a copy. Encoding: `treatment=("A1", "A2")`
with `A2 = A1`; `time_varying=((), ("L", "D"))` puts the post-assignment auxiliary `L` and death `D`
before the response node; `censoring=("C1", "C2")` with `C1 = 1` everywhere and `C2 = R` (dead units
coded observed with the worst score); `RegimeContrast({"always": 1, "never": 0})`.

| estimator (n = 4000, 100 seeds, composite truth ATE 1.4810) | mean bias (MC se) | 95% CI coverage |
| --- | --- | --- |
| longitudinal TMLE, encoding above | +0.0027 (0.0064) | 0.94 |
| point-treatment TMLE, `missingness="R"`, adjustment W only | +0.0389 (0.0075) | 0.88 |

Two frictions, both library-facing:

1. `censoring=` must name one column per treatment node. `censoring=("C2",)` raises `DataError`.
2. An all-ones censoring node raises scikit-learn's raw `ValueError: ... only one class` from
   `LogisticRegression`. The probe needed a wrapper learner that returns a constant on one class.
   A constant observation node, or a node copied from the previous one, should either be accepted
   (g = 1, no fit) or refused with a named reason. That is a candidate library row.

So SN-01 / SN-04 can be fixed in the notebook. Show the sequential-regression route rather than call
the gap a scientific requirement. Alternatively, open a library row for first-class
point-treatment "post-assignment auxiliaries / death before response" support.

## Recommended order of operations

1. **Library**: fix A in `synthetic.py`. Expose the logit, and bound g in [0.05, 0.95]. Update the
   docstrings. Fix B inside `multi_arm_dgp.arm_logits` only, if chosen. No other generator changes.
2. **Tests** (fast, before any run): golden fingerprints in `test_datasets.py`; rewrite the
   `test_study_drivers` mechanism test against the logit, keeping `double_robustness_dgp` byte-for-byte;
   re-witness the three `test_drtmle_fit` tests and the `test_ipsi` control with a nonzero witness;
   the `drtmle.py:160-166` comment; and the `multi_arm_common.py` comment. Run the full fast suite
   (`pytest -q -n auto --dist loadgroup`).
3. **Declare** in `docs/roadmap.md`, pushed before any run: the four studies, the four commands, the
   unchanged design, the `--skip-reference` choice and its reason, the runtime (3.11.13 / 1.17.1 venv),
   and the red-cell route for the three gated studies.
4. **Regenerate once**, sequentially on 16 cores, with nothing else running:
   `fold-evaluated-cvtmle` (4 min), then `fold-targeted-cvtmle` (7 min), then `canonical-ctmle-oat`
   (3-4 min), then `canonical-cvtmle` (86 min with R, much less with `--skip-reference`). Copy the outputs
   in. Verify the manifest hashes and LF. Confirm the primary artifacts are byte-identical.
5. **Docs**: run `python -m tests.studies.evidence.document` and `red_cells`. Re-measure `cv-tmle.md:235`.
   Re-execute the four point-treatment notebooks (and `msm-projections` for fix B). Update the
   callbacks. Re-narrate (PT-N1 narration removed). Run `python -m tests.prose --update` and `nox -s docs`.
   Then run the fast suite again.

Estimated regeneration wall time: about 1 h 45 min if every reference re-runs. About 20-40 min if the
reference rows are reused. Fix B and the candidate C rows add none.

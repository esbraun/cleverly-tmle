# Review gate 2: the implementation plan

Input: `plan.md` at the untracked state of 2026-10-01, read against `ledger.md`, `sibling-sweep.md`,
`refusal-inventory.md`, `learner-experiment.md`, `study-impact.md`, `gates/gate1-findings.md`,
`CLAUDE.md`, `docs/development/method-benchmarking.md`, `docs/development/example-notebooks.md`,
and `docs/development/pull-requests.md`. Light checks ran against the tree: `tests/canonical/regenerate.py`
(`--skip-reference` registration), the four study records, `tests/unit/test_red_cell_ledger.py`,
`tests/studies/evidence/red_cells.py`, `tests/studies/evidence/property_verdicts.py`, the committed
`properties.csv` of the four studies, `tests/studies/canonical_properties.py`, and `synthetic.py`.

Verdict: the plan is sound in its decisions and order. It is not yet executable as written in two
places (S3 command for `fold-evaluated-cvtmle`, the red-cell route), and its pre-registration
section is short of what `method-benchmarking.md` and the fast suite will demand. Seven required
changes follow.

## CONFIRMED

| item | finding | evidence |
| --- | --- | --- |
| coverage of material rows | every `wrong` or `misleading` ledger row maps to a change: PT-01/02/08/N1 (S1, S2, N1), CF-01 (N2), CT-01/03/04/05/N1 (S2, N5, N6, N9), DR-01/02/04/06/07 (S2, N3), IV-01/02/03/04/05/N1 (N4), SN-01/03/04/N1 (N6), LT-01/02/03/04/N1 (S2, N7), MS-02 (N9), TW-01/02/03/04 (S4, N10) | ledger tables against the order table |
| Gate 1 corrections | the three section-6 corrections are in the decision table: "flexible parametric" (P1 on IV), the dropped "halves the interval" flag (premises cite the 153 and 200-seed runs), the squeeze justified by the omitted-variable bound | `plan.md` rows PT-N1 and P1 |
| order | S1 (generator) before S3 (studies) before N1 to N4 (pages that cite them); S4 and S5 before every notebook; each notebook once; regeneration once | `plan.md` order table; memory rule |
| fix A shape | inside `nonlinear_dgp`, so `navigation_data`, the bounded twin, and the four studies share one law. `canonical_properties.double_robustness_dgp` replaces the propensity with its own, so it does not move | `canonical_properties.py:126-136` [read] |
| ATE truth unchanged, ATT and ATC move | 0.1628580 both laws; 0.1791 to 0.1773 and 0.1493 to 0.1505. The PT callback pins hold with 0.0145 margin | `learner-experiment.md`, Gate 1 |
| learner (b) | SE/SD 1.09, coverage 0.960, calibration slope 0.96, default `robustness_value()` succeeds on 100% of 200 seeds on the new law. The bare booster stays wrong (SE/SD 1.84, refusal 92.5%). Choosing (b) over (c) is right: (c) covers 0.99 with slope 0.72, so it still overfits g | `learner-experiment.md` strong-law table |
| MS-02 | no generator change. 0.944 over 4,300 seeds (MC SE 0.0035) is 1.7 SE below nominal; the 0.920 was a seed-range excursion. A wording fix is proportionate, and fix B would have moved golden fingerprints for no evidence gain | `study-impact.md` B.2 |
| SN-01 / SN-04 | the two-part statement (MAR among survivors; no deaths in this law, so the two coincide) is the correct identification statement. Contact attempts are a limit of the point-treatment fit, not a rule | Gate 1 section 4 derivation; `study-impact.md` SN probe |
| DR E-value | the continuous path standardises by the marginal observed SD and converts by `exp(0.91 d)`; "approximate" is the right label | `evalue.py:546-557` per Gate 1 |
| C-TMLE workflow | point and selection path from the collaborative fit; the plain TMLE interval as the inferential result; `simulated_confounding` on the collaborative fit read as movement, not a bound. No `CollaborativeTMLEMethod` strategy reports an interval, so CT-10 belongs in S2 | `refusal-inventory.md` collaborative section [executed] |
| longitudinal covariate-drop benchmark | leave-one-covariate-out refits, reported in SE units and called a benchmark. Sound as a calibration of "a cause as strong as X" | `refusal-inventory.md` LT and LS sections [executed] |
| regeneration eligibility | a targeted rerun is not eligible (the law changes), so complete reruns are right. The three `targeted_seed_repair` manifests become plain manifests; `test_targeted_composites_attribute_carried_and_replaced_evidence` accepts that | `study-impact.md` A.3 |
| `--skip-reference` reuse | no reference input moves (the comparators run only on the primary laws), and the committed reference rows pair by `replicate` and `n`. The Python primary artifacts should return byte-identical on the 3.11.13 / 1.17.1 venv, a free witness that only the property cells moved | `regenerate.py:250-261`; A.3 |
| staleness is invisible to the fast suite | the affected cells estimate `ate`, whose truth does not move, so no test notices stale artifacts. The PR must name the four studies. The plan does | A.7 |
| flat-curve trap recognised | N1 names the flat truncation curve as a new lesson rather than a sentence | `plan.md` N1 |
| prose ledger and docs build | F runs `python -m tests.prose --update`, the fast suite, and `nox -s docs` | `plan.md` F |

## REQUIRED CHANGES

### R-1. The S3 command is wrong for `fold-evaluated-cvtmle`

`--skip-reference` is registered only when the study declares a reference
(`regenerate.py:_arguments`, `if reference is not None`). `fold_evaluated_cvtmle.py:65` has
`reference=None`, so `python -m tests.canonical.cvtmle_fold.regenerate --skip-reference` exits
with an argparse error. Write the four commands in full, one row each:

| study | command |
| --- | --- |
| `fold-evaluated-cvtmle` | `python -m tests.canonical.cvtmle_fold.regenerate --jobs 16 --output <empty dir outside the repo>` |
| `fold-targeted-cvtmle` | `python -m tests.canonical.zepid_cvtmle.regenerate --jobs 16 --skip-reference --output <dir>` |
| `canonical-ctmle-oat` | `python -m tests.canonical.ctmle3_oat.regenerate --jobs 16 --skip-reference --output <dir>` |
| `canonical-cvtmle` | `python -m tests.canonical.tmle3_cvtmle.regenerate --jobs 16 --skip-reference --output <dir>` |

Run them in this order (short first, 86-minute study last), serially, with no fast suite beside
them. Copy the outputs into `tests/canonical/<dir>/` and verify LF and the manifest hashes.

### R-2. The red-cell route cannot be "reporting policy plus a sentence on the evidence page"

`tests/unit/test_red_cell_ledger.py::test_every_red_row_has_an_owner` requires every red
verdict to be owned in `red_cells.CLAIMS`, and `test_every_owner_is_an_id_the_roadmap_lists`
requires each owner to be an `id` in the roadmap's RM18 "What this row asks for" table
(`red_cells.py:533`). A reporting study with an unowned red row fails the fast suite. The
plan's route therefore collides with the "no roadmap additions" ruling, and the plan must say so
before the run rather than discover it after. Declare:

1. The margin at stake. The gated overfitting block requires the cross-fit arm's coverage gain
   over the in-sample control to clear `OVERFIT_COVERAGE_GAIN = 0.15` and the control's SE ratio
   to sit under `OVERFIT_SE_CONTROL_CEILING = 0.75` (`property_verdicts.py:124-125`). Today the
   gain CI lower bound is 0.38 on all three cross-fit studies and the control coverage is 0.4875.
   Bounding g away from 0 can narrow that gap (Gate 1). The `robustness_contract` cells of
   `canonical-ctmle-oat` are on a `reporting` record already.
2. The route if a gated cell goes red: the driver refuses to publish. Set
   `publication_policy="reporting"` on that study's record, add the red key to
   `red_cells.CLAIMS` under an owner, and re-run the identical declared command (seeds are fixed,
   so the artifacts are the same bytes). The owner must be a roadmap `id`. State explicitly that
   this one roadmap table row is reporting bookkeeping that the fast suite enforces, not a
   deferred fix, and that the orchestrator records it in `progress.md` with the user's ruling
   named. If the orchestrator prefers to avoid the roadmap entirely, the only alternative is to
   treat a red cell as a defect of fix A and stop; the plan must pick one route now.
3. The route if `canonical-ctmle-oat` goes red: it publishes immediately under `reporting`, and
   the same owner step applies before the commit.
4. Margins, budgets, seeds, laws other than the propensity, learners, and `n` are never changed
   after a result is seen. The plan says this; keep it.

### R-3. The pre-registration section is short of the checklist

Declaring in a pushed `plan.md` rather than `docs/roadmap.md` is adequate: `method-benchmarking.md`
requires the declaration before the run and names no venue; the roadmap was precedent only
(`bb4772a9`). It is adequate only if the declaration is immutable and findable. Add to the section:

| item | what to write |
| --- | --- |
| immutability | the commit hash of the push, recorded in `progress.md` and in the PR body. After S3 starts, the section is not edited; a "what the run found" subsection is appended |
| runtime | Python 3.11.13, SciPy 1.17.1, NumPy 2.4.6, scikit-learn 1.9.0, pandas 3.0.5 (`study-impact.md`). The four studies ran on this runtime, so roundoff drift is not expected |
| smoke run | a disposable `--replicates 10` run of one study to a directory outside the repo, before the declared run. Note that `--replicates 10` skips properties, so it checks plumbing only |
| byte-identical witness | `replicates.csv.gz`, `summary.csv`, `equivalence.csv`, and `performance-tests.csv` are expected unchanged for every study. Report any difference in the PR as a runtime finding |
| manifests | the three `targeted_seed_repair` composites are replaced by plain manifests; `tests/canonical/seed-repair-plan.json` stays as history |
| what is unchanged | the current bullet, plus the explicit list: `nonlinear_bounded(concentration=12.0)` outcome law, `n=500 x 400` and `n=700 x 1200`, the learner libraries, `OVERFIT_COVERAGE_GAIN`, `OVERFIT_SE_CONTROL_CEILING`, the bias margins, `publication_policy` (subject to R-2), and the shared property seeds |
| evidence pages | `python -m tests.studies.evidence.document` and `python -m tests.studies.evidence.red_cells` regenerate the generated blocks. The hand-written measured-values prose at `stacked-point-treatment-cv-tmle.md:243`, `fold-evaluated-point-treatment-cv-tmle.md:171`, `fold-targeted-point-treatment-cv-tmle.md:122-126`, and `outcome-adaptive-point-treatment-c-tmle.md:202` is rewritten by hand against the new `properties.csv` in the S3 commit, and no number is typed that `claims.py` can supply |
| red-cell route | R-2 |

### R-4. The PT truncation lesson needs a nonzero witness

`CLAUDE.md` forbids an exact-law check that is blind to a term that vanishes. On the new law the
default bound 0.0114 never bites learner (b) (truncation on 0.5% of 200 fits), so the stored
curve is flat and a callback that pins the curve pins nothing. N1 offers two options; choose the
second and specify it: Step 9 runs `truncation_curve` over a grid that includes a deliberately
tight bound (for example `g_bounds=(0.10, 0.90)`, which bites because the true g reaches 0.05),
prints the truncated share at each bound, and reads the default as "never bites on this law by
construction". The callback asserts the truncated share is zero at the default and positive at the
tight bound, and that the estimate moves there. DR-09 gets the same treatment in N3: the support
row reads "the fitted model's support; 0% of rows below the bound, which agrees with a law whose
true g lies in [0.05, 0.95]", and the reading keeps P16's sentence that the report describes the
fitted model only.

### R-5. The CF failure-mode narration needs a sweep on the new law

CF-01's replacement numbers ("about 0.7", in-sample coverage 0.78 to 0.82) were measured on the
old law. N2 says "rewritten from the new output", which is one draw. R3 and R4 bind every
coverage or ratio sentence to a sweep. Add to N2's verification: a committed probe under
`reviews/notebook-review/probes/` that fits the in-sample and cross-fitted learner-(b)
configurations on `navigation_data` under the new law for at least 200 seeds and reports
SE/SD and coverage for each. The failure-mode and trust cells quote that probe, and the trust
row leads with the regenerated study's coverage gap.

### R-6. IV-02 needs a decision, and the dose axis is not on the new law

The shift fits cover 0.60 to 0.65 at `+1.0` and 0.57 to 0.60 for the gap. Under R1 a shown fit
with that coverage is not a working example unless the page says so. N4 must decide: attempt the
ledger's repair (a quadratic parametric Q and more density bins), sweep at least 100 seeds, and
adopt it if coverage reaches about 0.90; otherwise keep the fit, state the measured
under-coverage with the committed sweep number, and present the gap as a point contrast. Also
correct the decision-table row "P1 on IV": `make_shift_dose` does not use `nonlinear_dgp`, so the
dose-axis sweep is on an unchanged law. Only the regime and incremental axes re-sweep on the
new law. The IPSI truth carries g, so N4 must re-read `incremental_truth` from the new output
and not from the stored cell.

### R-7. R8's probe directory does not exist, and three S1 sites are missing

`reviews/notebook-review/probes/` is absent; every sweep the plan quotes sits under `.tmp/`.
Order 0 must copy each probe script that a reading will cite (CT, MS, SN, LT, LS, TW sweeps,
and the learner experiment's `run2.py`) into `probes/` with its seed ranges and output tables,
and N1 to N4 re-run the PT, CF, DR, and IV probes on the new law after S1. Add to S1's file list
the `src/cleverly/validation/drtmle.py:160-166` comment (residuals "measured on nonlinear_dgp at
n = 400" move with the law), the `tests/unit/test_study_drivers.py` fix in the form
`study-impact.md` A.1 requires (compare against the exposed logit; never re-derive
`double_robustness_dgp`), and a pinned test that `nonlinear_bounded_dgp().truth()["ate"]` is
0.1628580 to 1e-7, which witnesses that only the propensity moved.

## Optional suggestions

- DR E-value wording: write "it reads the estimate and its interval, so the doubted g enters
  only through the estimate, which DR-TMLE protects when the outcome model is consistent". "Does
  not test the doubted assignment model" alone overstates the independence.
- Longitudinal benchmark wording (Gate 1 section 3): say each move is signed and specific to the
  dropped covariate, and that a refit without a confounder is a biased estimator of the same
  estimand. Dropping `engagement_day7` removes a node-2 history variable from both mechanisms,
  which is the right analogue of omission.
- SN trust row: `study-impact.md` shows the sequential route exists in the library through a
  two-node `LongitudinalTreatment` (bias +0.0027, coverage 0.94 at n = 4,000). Name it as the
  route for a page that records deaths or post-assignment contacts, instead of "estimator limit"
  alone.
- The IV learner is chosen by reading sweep coverage. That is fine for a tutorial, but the page
  should say the learner was checked on a seed sweep, so the choice is disclosed rather than
  presented as prior knowledge.
- `cv-tmle.md:235`: the current text says the probes ran at commit `1cf6628`. Recompute and name
  the new commit in the same sentence.
- S1's doctest at `tmle.py:64` prints only sorted keys, so it passes unchanged; keep it in the
  list only to confirm it ran.
- Between S1 and N4 four tutorial callbacks fail. State in `progress.md` that the per-commit
  gate accepts exactly those four failures and no others, so a new failure is not masked.

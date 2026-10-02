# Notebook review: implementation plan

Branch `agent/notebook-review` from `origin/main` at 5f33902b. This plan turns the verified
findings in `ledger.md` (114 rows), the patterns in `sibling-sweep.md` (P1-P18), the refusal
replacements in `refusal-inventory.md`, the learner experiment in `learner-experiment.md`, and the
study impact in `study-impact.md` into ordered changes. Gate 1 (`gates/gate1-findings.md`) confirmed
every material finding and corrected three items, which this plan adopts.

## Rulings that bind every change

| ruling | source | effect |
| --- | --- | --- |
| R1. Notebooks are end-to-end examples that work. They do not showcase refusals | user, 2026-10-01 | every refusal cell is replaced by a working analysis or deleted (`refusal-inventory.md`, corrected below). Statistical failure-mode cells stay |
| R2. A fix that would move a registered study is implemented, and the study is regenerated. Nothing is deferred to the roadmap | user, 2026-10-01 | fix A below, and study regeneration S3 |
| R3. A sentence that holds on fewer than 90% of swept draws is tagged "on this draw" or quotes printed numbers without a multiplier | P12 | every reading |
| R4. A miss is "sampling variation" only when a seed sweep of the shown configuration covers near nominal | P8 | every reading of an interval that misses |
| R5. A trust row names the study's law (same or other), nuisance construction (supplied, oracle, fitted), outcome type, fold layer, and the cell's role | P5 | every `trust` cell |
| R6. No Monte Carlo "true nu^2" of the old law is quoted (7.77, 8.22, 7.75) | PT-N1 | PT, CF, DR, callback `dr_tmle.py:144` |
| R7. Each changed notebook re-executes once, with `python scripts/execute_notebook.py <path> > log 2>&1; echo $?`, under 60 s, with `n_jobs=1` and explicit cheap learners | goal | every notebook change |
| R8. Every quoted repeated-draw number comes from a committed probe script under `reviews/notebook-review/probes/`, or from a registered study, and the callback lists it in `UNPRINTED_DECIMALS` with its source | P12, narration test | readings that quote sweep numbers |

## Decisions on contested items

| item | decision | evidence |
| --- | --- | --- |
| PT-N1, the infinite E[1/g] of `nonlinear_dgp` | fix A: `g = 0.05 + 0.90 expit(logit)` inside `nonlinear_dgp`, so every `navigation_data` page and the four studies share a law with E[1/g] <= 20 and nu^2 = 4.83 | closed form (orchestrator, Gate 1); ATE truth unchanged to 1e-7; ATT 0.1791 to 0.1773, ATC 0.1493 to 0.1505 (`learner-experiment.md`) |
| P1, the treatment learner on PT and CF | `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=<seed>)`, described as a shallow, regularized booster | new law, 200 seeds: coverage 0.960, SE/SD 1.09, calibration slope 0.96, default `robustness_value()` succeeds on 100% (`learner-experiment.md`). The bare booster stays wrong on the new law (SE/SD 1.84, refusal on 92.5%) |
| P1 on IV | the editor sweeps (b) and the degree-2 logistic g on the new law for the regime and incremental axes (>= 100 seeds each; the dose axis uses `make_shift_dose`, which this plan does not change) and picks the one with coverage nearest 0.95; the page calls it "flexible parametric" or "regularized", never "correctly specified" | Gate 1 correction 1; IV-N1 (calibration wrapper fails) |
| MS-02, near-zero cadence probabilities | no generator change. Coverage over 4,300 seeds is 0.944; the 0.920 was a seed-range excursion. Fix the reason text only | `study-impact.md` fix B |
| SN-01 / SN-04 | wording: the page states that missing at random on the composite holds only when no patient dies; the composite needs missing at random among survivors; this law has no deaths, so the two coincide here. Contact attempts are an estimator limit of this page's point-treatment fit, not a scientific rule. No roadmap item | `study-impact.md`; SN verifier |
| DR sensitivity | approximate E-value, labelled approximate (marginal observed SD, exp(0.91 d) conversion), and stated as not testing the doubted assignment model | Gate 1, decision 3 |
| C-TMLE interval | the page reports the C-TMLE point estimate and selection path, and the plain TMLE interval as the inferential result; `simulated_confounding` on the collaborative fit | `refusal-inventory.md`; CT-10 |
| longitudinal sensitivity | refits that drop one recorded covariate at a time, reported as benchmark moves in standard errors, not as a bound | `refusal-inventory.md` |
| MSM label refusal | replaced by a second known design coded by cadence step | `refusal-inventory.md` |
| survey PAF refusal | Step 10 deleted | `refusal-inventory.md` |

## Declared study reruns (pre-registration)

This section is pushed before any regeneration runs. The push commit hash is recorded in
`progress.md` and in the pull request body. After S3 starts, this section is not edited. A
"What the run found" subsection is appended below it instead.

| order | study | command |
| --- | --- | --- |
| 1 | `fold-evaluated-cvtmle` | `python -m tests.canonical.cvtmle_fold.regenerate --jobs 16 --output <empty dir outside the repo>` (it declares no reference, so it takes no `--skip-reference`) |
| 2 | `fold-targeted-cvtmle` | `python -m tests.canonical.zepid_cvtmle.regenerate --jobs 16 --skip-reference --output <dir>` |
| 3 | `canonical-ctmle-oat` | `python -m tests.canonical.ctmle3_oat.regenerate --jobs 16 --skip-reference --output <dir>` |
| 4 | `canonical-cvtmle` | `python -m tests.canonical.tmle3_cvtmle.regenerate --jobs 16 --skip-reference --output <dir>` |

The studies run serially in this order, with no fast suite or notebook execution beside them. The
outputs are copied into their `tests/canonical/<dir>/`, with LF endings and matching manifest
hashes.

| item | declaration |
| --- | --- |
| what changes | only the propensity of `nonlinear_dgp`: `g = 0.05 + 0.90 expit(logit)` with the same logit |
| what is unchanged | the `nonlinear_bounded(concentration=12.0)` outcome law; `n = 500 x 400` and `n = 700 x 1200`; the learner libraries; `OVERFIT_COVERAGE_GAIN = 0.15`; `OVERFIT_SE_CONTROL_CEILING = 0.75`; the bias margins; each study's `publication_policy`; the shared property seeds; budgets and reading rules |
| reference rows | the R comparator inputs do not move, so the committed reference rows are reused (`--skip-reference`, precedent `a01177b6`) |
| runtime | Python 3.11.13, SciPy 1.17.1, NumPy 2.4.6, scikit-learn 1.9.0, pandas 3.0.5, the runtime of the committed runs |
| smoke run | one disposable `--replicates 10` run of `fold-evaluated-cvtmle` to a directory outside the repo before the declared runs. It checks plumbing only, because it skips properties |
| byte-identical witness | `replicates.csv.gz`, `summary.csv`, `equivalence.csv`, and `performance-tests.csv` are expected unchanged for every study, because no primary scenario uses `nonlinear_dgp`. Any difference is reported in the pull request as a finding |
| manifests | the three `targeted_seed_repair` composite manifests become plain manifests. `tests/canonical/seed-repair-plan.json` stays as history |
| evidence pages | `python -m tests.studies.evidence.document` and `python -m tests.studies.evidence.red_cells` regenerate the generated blocks. The hand-written measured values at `stacked-point-treatment-cv-tmle.md:243`, `fold-evaluated-point-treatment-cv-tmle.md:171`, `fold-targeted-point-treatment-cv-tmle.md:122-126`, and `outcome-adaptive-point-treatment-c-tmle.md:202` are rewritten against the new `properties.csv` in the S3 commit |
| red-cell route | if a gated cell is red after the run, the orchestrator stops and reports to the user before any change. The fast suite requires a roadmap owner for every red row (`test_red_cell_ledger`), and the user has ruled out roadmap additions, so that conflict needs the user's decision. Margins, budgets, seeds, laws, learners, and `n` are never changed after a result is seen. `canonical-ctmle-oat` already publishes under `reporting`; a new red row there takes the same stop |

## Order of changes

Each change is one subagent and one commit, run serially. After each commit a Fable gate reviews it,
and confirmed problems are fixed before the next change starts.

| order | change | findings | files | verification |
| --- | --- | --- | --- | --- |
| 0 | plan commit and push | all | `reviews/notebook-review/**` (with `probes/`, the frozen review probe scripts), `pyproject.toml` (ruff excludes `reviews/`) | push succeeds before S3; hash recorded |
| S1 | fix A: strong-positivity propensity | PT-N1, PT-08, P6, DR-N1 | `src/cleverly/datasets/synthetic.py` (docstrings say the propensity lies in [0.05, 0.95] and why), `navigation.py` docs; tests pinned to the old law (golden fingerprint, `test_drtmle_fit` witnesses, `test_ipsi` control, `test_study_drivers` bounded-mechanism test fixed against the logit), `tmle.py:64` doctest, `cv-tmle.md:235` probe values recomputed | failing-first unit test that the propensity lies in [0.05, 0.95] and E[1/g] is finite by quadrature; fast suite of touched modules; tutorial callbacks are allowed to fail until their notebook change. Also: the `validation/drtmle.py:160-166` residual comment; `test_study_drivers` compared against the exposed logit (never re-derive `double_robustness_dgp`); a pinned test that `nonlinear_bounded_dgp().truth()["ate"]` equals 0.1628580 to 1e-7 |
| S2 | library strings and docstrings | PT-01 and P4 ("max bias" to "bias scale", RV string, cf_d wording), LT-04 (`estimator.py:2555`, `cv-tmle.md:287`), DR-06 (`validation/drtmle.py:600` contract for cross-fitted fits), SN-11 (`population_intervention.py:115`), IV-06 (`builtin.py:227` cap text, warning when a cap exceeds the observed dose range), IV-11 (`ShiftDGP` and `shifted()` docstrings), CT-10 (`methods.py:798`), CT-03 (`ctmle.py:222-224`, `collaborative-tmle.md:154`), CT-N2 (rerun the e2e class, align MAE text), MS-N1 (`test_msm_projection_weights.py:8`), `validation-methods.md:734,833,856` | `src/cleverly/...`, tests, technical reference | failing-first test for each changed printed string and for the cap warning; `ruff`, `mypy`, targeted pytest |
| S3 | study regeneration, run once | R2 | the four studies' artifacts and manifests, evidence pages via `python -m tests.studies.evidence.document`, the four method-evidence pages | manifest hashes match LF artifacts; the fast suite's artifact-recompute tests pass; red cells follow the declared route |
| S4 | narration harness | P10, TW-04 | `tests/unit/tutorial_semantics/__init__.py` and its tests | failing-first test on a narrated `1.23e-04` that no output prints |
| S5 | house rules and shared design | R1, R3-R5, P1, P3, P7, P8, P13, P17, LS-NF2 | `docs/development/example-notebooks.md` (outline row 11, refusal rule and the not-derived table replaced by a table of the working sensitivity analysis for each kind of fit, learner rule, trust checklist, sweep rules, protocol check, no raw sub-1e-8 residuals), `docs/examples/index.md` (covariates are independent standard-normal draws; composite rows; LS failure-mode wording) | prose report, links test |
| N1 | point-treatment-tmle | PT-01..09, PT-N1, P1, P3, refusal rows | nb, cb | new law and learner (b); Step 9 runs `truncation_curve` over a grid that includes a deliberately tight bound such as `g_bounds=(0.10, 0.90)`, prints the truncated share at each bound, and reads the default as not binding on this law; the callback asserts zero truncation at the default, positive truncation at the tight bound, and that the estimate moves there (nonzero witness); sensitivity runs the default call with no `try`; benchmarks read with the clip disclosed; seed claims per R3 |
| N2 | cross-fitting | CF-01..06, CF-N1, P1, P8 | nb, cb | learner (b); a committed probe under `probes/` fits the in-sample and cross-fitted learner-(b) configurations on the new law for >= 200 seeds and reports SE/SD and coverage; the failure-mode and trust cells quote that probe; trust row leads with the registered coverage gap (regenerated numbers) |
| N3 | dr-tmle | DR-01..10, DR-N1/N2 | nb, cb | delete DR-01 sentence; DR-02 reworded with a committed sweep on the new law; theorem scope (DR-06); trust per DR-04/05; E-value replaces the refusal; DR-09: the support row reads as the fitted model's support, with 0% of rows below the bound, which agrees with a true g in [0.05, 0.95], and keeps P16's sentence that the report describes the fitted model only |
| N4 | interventions | IV-01..13, IV-N1..N3 | nb, cb | learner chosen by sweep on the new law; `incremental_truth` re-read from the new output; IV-02: try the repair (quadratic parametric Q and more density bins), sweep >= 100 seeds, adopt it if coverage reaches about 0.90, else keep the fit, state the measured under-coverage from the committed sweep, and present the gap as a point contrast; IPSI readings per R4 with committed probes; trust rows say "mechanism supplied exactly"; intensity called an index; treatment-only stress surface |
| N5 | collaborative-tmle | CT-01..09, CT-N1, CT-N4 | nb, cb | outcome text per P2; selection read with rounded shares; plain TMLE interval; plug-in SE direction stated; no refusal cells |
| N6 | survey-nonresponse | SN-01..10, SN-N1/N2 | nb, cb | composite assumption stated; P2 wording; Step 10 deleted; tipping read on the sample scale with the `use_ci=True` value printed; tilt curve called a plug-in away from 0 |
| N7 | longitudinal-tmle | LT-01..08, LT-N1 | nb, cb | learner-form sentence fixed; calibration reading fixed; ordinary in-sample study cited as closest; no clustered study claimed; death coding stated; within-stratum engagement shares; covariate-drop benchmark replaces the capability list |
| N8 | longitudinal-survival | LS-01..06, LS-NF1/NF2 | nb, cb | competing section censors at plan exit (`censoring=True`); death-as-censoring functional printed with its label; trust rows scoped; covariate-drop benchmark replaces the refusals |
| N9 | msm-projections | MS-01..07 | nb, cb | "by 0.0008 on this draw"; reason text per decision table; cadence-step design replaces the label refusal; share-weight shift read against one SE |
| N10 | twins-causal-inference | TW-01..12, TW-N1..N3 | nb, cb, `docs/references.md` | network check first (no code edits if unreachable); hand TMLE uses the package bound and two arm covariates, compared against the SE; sensitivity robust on >= 28 of 30 pair samples; explicit number formats; trust row matches the binary clustered study; published citations |
| F | final gates | all | `tests/prose-report.md` | `python -m tests.prose --update` with judged reasons; `pytest -q -n auto --dist loadgroup`; `nox -s docs`; final Fable gate on the whole diff; PR; CI green |

## Per-change rules for editors

- Read `docs/development/example-notebooks.md` (as changed by S5), `CLAUDE.md`, the ledger rows,
  and the relevant gate notes before editing.
- Edit the notebook and its callback together. Keep every cell id that a callback reads.
- Run `ruff format` and `ruff check` on the notebook, then execute it once. Check the exit status,
  contiguous execution counts, and the absence of error outputs. Then run `--check`.
- Run the targeted tests in `example-notebooks.md` step 10, and `python -m tests.prose --path`.
- Commit with a subject in the house style and a body that answers the four questions of
  `docs/development/pull-requests.md`. Report the files, commands, and exit codes.

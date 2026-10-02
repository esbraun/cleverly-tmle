# Verification: `docs/examples/longitudinal-survival.ipynb`

| item | value |
| --- | --- |
| findings verified | `reviews/notebook-review/findings/longitudinal-survival.md` (LS-01..LS-06) |
| commit | `5f33902b`, `cleverly.__file__` asserted under this worktree's `src` |
| own scripts | `.tmp/notebook-review/verify-longitudinal-survival/functional.py` (60-point Gauss-Hermite quadrature of the death-as-censoring functional, both within-interval orderings, and the total effect), `sweep_v.py` (80 draws, seeds 2000-2079, disjoint from the reviewer's 1000-1200; competing fit, death-as-censoring fit, and a competing fit with `censoring=True`), `elim_cens.py` (30 draws, death-as-censoring combined with enrollment censoring) |
| seed check | `sweep_v.py` at seed 53 reproduces the notebook's -0.0296 and -0.0075 exactly |

Verdicts: CONFIRMED 3 (LS-02, LS-04, LS-06), PARTIAL 3 (LS-01, LS-03, LS-05), REFUTED 0.

## LS-01

- verdict: PARTIAL (the defect is real; severity is lower than claimed)
- severity: weak (reviewer: misleading)
- evidence:
  - [read] cell `competing-reading` says "This section runs without censoring, so the causes are the only way to leave the risk set." Cell `competing-events` calls `make_longitudinal_competing(n=4_000, seed=53, censoring=False)`. `event_protocol` inherits the eligibility "Enrolled in the health plan at discharge" and has no disenrollment entry. The quoted text exists.
  - [read] Both generators use the same all-cause hazards (`src/cleverly/datasets/longitudinal.py:830,843` call `_hazard_one`/`_hazard_two`, as does the survival generator at `:634,649`). The printed truths therefore add up: Step 2's never-plan 60-day plan-exit risk is 0.4552, and Step 9's never-plan readmission (0.2443) plus death (0.2109) is also 0.4552. The page relabels one event process. Step 2 calls it "plan exit", and Step 8 calls it "readmission or death" with no disenrollment at all. An expert who adds the two printed truths sees this. That strengthens the reviewer's point.
  - [executed] `censoring=True` works. Over 80 draws, the competing fit with `censoring=("C1","C2")` gives a 60-day death difference bias of +0.0015, coverage 0.9625, and SE/SD 0.98. The never-plan death t=2 bias is -0.0028. Period-1 censoring ranges from 428 to 524 of 4,000. The reviewer reported 451 at seed 53, and I reproduce that.
  - [executed] The Step 10 recoding still runs with enrollment censoring when the censoring indicator becomes enrolled-and-alive per node (`elim_cens.py`). The 30-draw mean difference is -0.0363, SD 0.0175, against the functional -0.0403 (Monte Carlo SE 0.0032).
- Why the severity is lower: the page states the omission in plain words. The estimand does not change. Without censoring the target is the CIF itself, and with disenrollment censoring it is the same CIF under the hypothetical of continued observation. The page misleads only about practice, because it teaches that this analysis on plan data needs no censoring. It does not make a numerical claim false.
- fix: the reviewer's fix is right and preferable. Use `censoring=True`, rename `C1`/`C2` to `enrolled_p1`/`enrolled_p2`, and add the protocol entry for disenrollment. Disenrollment is censoring, not a competing event, because readmission can still happen after a patient leaves the plan. The alternative, which says the section uses a different cohort with complete follow-up (for example linked all-payer discharge records), is cheaper. It leaves the "same plan" narrative inconsistent, and it drops the only demonstration of censoring combined with competing risks. Cost: Step 10's recoding must combine the two indicators (`K_k = C_k * (1 - D_k)`, keeping the carried readmission 1). Steps 8-11 decimals change, and so do the callback's miss assertions (`tests/unit/tutorial_semantics/longitudinal_survival.py:193-214`) and the epsilon claim. No registered study moves, because the competing study already covers monotone censoring (its accuracy rows are all "with monotone censoring").

## LS-02

- verdict: CONFIRMED
- severity: weak
- evidence:
  - [recomputed] My own quadrature (`functional.py`) gives a death-first functional of never 0.3049, always 0.2646, and difference -0.0403. The readmission-first ordering gives -0.0093. The total effect is never 0.2443, always 0.2467, difference +0.0024. All three match the reviewer.
  - [read] The recoding matches the death-first ordering. `alive_p1 = 1 - death_p1` is a censoring node, and the design places it before `readmission_p1`. Readmission is set to NaN where death occurred in the same period.
  - [executed] Over 80 draws, the death-as-censoring fit's t=2 difference has mean -0.0393 (bias +0.0010 against -0.0403), SD 0.0195, and mean SE 0.0200. Its CI covers the functional in 0.9625 of draws and covers the total effect +0.0024 in 0.4125. The level means are 0.3029 against 0.3049 and 0.2636 against 0.2646. The reviewer reported bias +0.0003, coverage 0.94 and 0.405 over 200 draws. The fit converges to the functional and not to the total effect.
  - [executed] The paired gap (censored minus total) has mean -0.0416 and a maximum of -0.0246 over 80 draws. The notebook's gap is -0.0221, which is smaller than every swept draw in both sweeps.
  - [read] Seed 53's interval (-0.0738, 0.0146) contains +0.0024.
- Is "Do not read -0.0296 as an estimate of the controlled direct effect" wrong? No. It is cautious, and its stated reason is weak. The generator has no structural world without death, because it draws one all-cause event and then a cause label. "Eliminate death" is therefore not an intervention in this law, and no CDE exists to estimate. The functional the fit targets depends on a within-interval ordering that the analyst chooses (-0.0403 against -0.0093). The sentence protects the reader from that. Its given reason is that "the learners specify neither the death mechanism nor the readmission regression correctly", and that misattributes the problem. Empirically the misspecification costs nothing here (bias +0.001, coverage 0.96). There is also internal tension: the Step 10 table labels the row's estimand "the controlled direct effect" and prints -0.0296 in its difference column.
- fix: endorse printing the identified functional, labeled as the reviewer proposes. Add that it depends on the death-before-readmission ordering the recoding imposes. Keep the warning, but change its reason to the real one: no world without death exists in this law, and the number is a CDE only under exchangeability for death as an intervened node. Printing SEs in the Step 10 table is good. Exposing the functional from the generator is optional and needs a dataset test. Computing it in the notebook is simpler, but it puts private hazards into a tutorial. No study moves.

## LS-03

- verdict: PARTIAL
- severity: weak (wording)
- evidence:
  - [read] Cell `competing-estimate-reading` says "The page's learners model the treatment mechanism correctly. TMLE is then consistent although the outcome regressions are misspecified. The competing-risk study measures that property and the coverage of this contrast over repeated draws on its own law." The text exists.
  - [read] `tests/studies/ltmle_competing_properties.py:147-159`: `mechanism_correct` uses `canonical.KnownCompetingMechanism`, which supplies the true probabilities. The study page's limitations row says "The mechanisms are supplied". The law is a finite-support law, not the page's law. The trust cell says "Both studies fix or supply the mechanisms, and this page learns them."
  - [read] The page's treatment mechanism is in fact correctly specified by a logistic model on the declared history (`longitudinal.py:824,840`).
- Assessment: the study does measure the stated property, consistency with a correct mechanism and misspecified outcome regressions. The sentence already scopes the coverage to "its own law". The gap is only supplied against fitted-correct mechanisms, which mainly affects the variance (an influence curve that ignores g-estimation tends to be conservative). That is a nuance, and the trust cell states it. Calling the two cells contradictory overstates it.
- fix: the reviewer's wording is correct and safe. Notebook only.

## LS-04

- verdict: CONFIRMED
- severity: weak
- evidence:
  - [read] Cell `trust` says "The survival study also measures a reported standard error slightly below the sampling spread at the second horizon. Read each 60-day interval as slightly too narrow."
  - [read] In `ordinary-survival-curve-longitudinal-tmle.md:90`, `interval_calibration/static_t2__correctly_specified` has an SE ratio of 0.9411 to 0.9774 and coverage of 0.9276 to 0.9407. Both nuisances are correctly specified on an exact binary law. The competing study's matching cell (`death_static_t2__correctly_specified`) has an SE ratio of 0.9700 to 1.0066, inside its band. "Each 60-day interval" therefore includes competing-risk intervals that the cited finding does not cover.
  - [read] The reviewer's 200-draw summary (`.tmp/notebook-review/longitudinal-survival/summary.log`) gives retention t=2 SE/SD of 0.99 to 1.08 and coverage of 0.955 to 0.965. With 200 draws the SE/SD ratio has about 5% Monte Carlo error. That supports "no evidence of narrow intervals on the page's law". It cannot rule out a 3% shortfall.
- fix: scope the sentence to the study, as the reviewer proposes. Do not replace it with "the page's intervals are not too narrow", because the sweep cannot resolve that. Notebook only.

## LS-05

- verdict: PARTIAL
- severity: none to weak
- evidence:
  - [read] The cell text exists. `docs/development/example-notebooks.md:161` requires exactly this treatment: "Explain an interval that misses the true value as sampling variation, and give the distance in standard errors." The page does both and gives -2.9 SE.
  - [read] The reviewer's own figure undercuts the finding. Some interval among the 12 misses by at least 2.9 SE in 5.5% of draws, which is roughly one draw in 20. The size is unusual for one prespecified interval and ordinary for the largest of 12. The one-in-20 sentence therefore covers it adequately.
- fix: an optional sentence is fine, for example "the largest of 12 distances reaches 2.9 SE on about one draw in 20". Do not change the seed to avoid the miss. That rewrites Steps 8-11 and the callback to make a draw look better, which runs against the spirit of house rule 160. If LS-01 is fixed, the seed's draws change anyway.

## LS-06

- verdict: CONFIRMED
- severity: weak
- evidence:
  - [read] The cell `title` row and the `data-reading` "tracked" definition exist. The protocol says "Count death as plan exit (composite strategy)", so the page discloses the composite but never says that the composite cannot answer the retention question.
  - [read] Because the two generators share the all-cause hazards (see LS-01), the competing law is a cause decomposition of the plan-exit process itself. Navigation cuts 60-day death from 0.2109 to 0.0705, while readmission is flat (+0.0024). The 60-day plan-exit reduction of -0.1380 therefore comes entirely from death in the page's own laws. That makes the reviewer's point concrete, not hypothetical.
- fix: endorse. State that the composite does not separate death from disenrollment. Name a concrete source of lost tracking, such as gaps in the eligibility feed. Notebook only.

## New findings

- NF-1 (weak, related to LS-01 and LS-06). The page uses one all-cause event process under two names. Step 2's "plan exit" truth (never 0.4552) equals Step 9's readmission plus death truth (0.2443 + 0.2109). Readmission does not end plan enrollment, so the second half's events are not "plan exit". A reader who adds the printed truths finds that the retention question in Steps 6-7 is really "readmission or death". Fixing LS-01 with enrollment censoring and LS-06's composite caveat addresses most of this. One further sentence should say that the two sections use separate synthetic laws.
- NF-2 (weak, part of LS-02). Step 10's table assigns the estimand "the controlled direct effect" to the -0.0296 row, and the next paragraph says not to read -0.0296 as a CDE estimate. The table should label the row "death-as-censoring g-formula functional (a CDE only under exchangeability for death)".
- Not re-checked: the Young et al. (2020) section numbers. I relied on the reviewer's [read] of PMC7811594.

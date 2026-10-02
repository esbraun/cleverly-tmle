# Review: `docs/examples/longitudinal-survival.ipynb`

| item | value |
| --- | --- |
| notebook | `docs/examples/longitudinal-survival.ipynb` (callback `tests/unit/tutorial_semantics/longitudinal_survival.py`) |
| commit | `5f33902b` (branch `agent/notebook-review`), `cleverly.__file__` under this worktree's `src`, Python 3.11.13 |
| `--check` | exit 0, wall time 9 s. "every cell reproduced its stored non-image output" (`.tmp/notebook-review/longitudinal-survival/check.log`) |
| independent truths | `.tmp/notebook-review/longitudinal-survival/truth.py`: re-implemented structural equations, 4,000,000-unit direct counterfactual simulation, plus a 60-point Gauss-Hermite quadrature of the death-as-censoring functional. The library truth helpers were not called |
| seed sweep | `.tmp/notebook-review/longitudinal-survival/sweep.py` (all three fits exactly as the notebook codes them, n = 4,000, survival seeds 1000-1199, competing seeds 1001-1200, 200 draws), summary `summarize.py` / `summary.log`. Seed 52/53 reproduces every stored number (`seed52.csv`) |
| extra probe | `censored_competing.py`: competing fit with `censoring=True` |

Counts: wrong 0, misleading 1, weak 5.

## Findings

### LS-01: the competing-risk analysis assumes nobody leaves the plan, on a page whose first half shows 26-46% plan exit
- Cell `competing-reading`: "This section runs without censoring, so the causes are the only way to leave the risk set." Cell `competing-events`: `make_longitudinal_competing(n=4_000, seed=53, censoring=False)`; `event_protocol` inherits eligibility "Enrolled in the health plan at discharge" and says nothing about plan exit.
- Problem: a health plan observes readmission only while the patient is enrolled. Steps 2-7 of the same page report that 26% (never) to 15% (always) of patients leave within 30 days and 46%/32% within 60 days. The readmission/death analysis therefore needs plan exit (other than death) as censoring, and then the target is the hypothetical "had every patient stayed enrolled". That needs sequential exchangeability and positivity for remaining enrolled. The page neither declares that censoring nor names the hypothetical, and the protocol for Step 8 has no intercurrent-event entry for disenrollment. A hostile reviewer reads the competing estimates as complete-follow-up numbers that this plan's data cannot produce. The task question "is plan exit treated as censoring, then which hypothetical is targeted?" has no answer on the page.
- Severity: misleading.
- Evidence: [read] `docs/examples/longitudinal-survival.ipynb` cells `competing-events`, `competing-reading`; `src/cleverly/datasets/longitudinal.py:779-898` supports `censoring=True`. [executed] `censored_competing.py`: with `censoring=True`, seed 53, 451 censored at period 1 and 233 at period 2; the same design plus `censoring=("C1","C2")` fits and all eight CIFs land near truth (for example never/death t=2 0.2098 vs 0.2109).
- Proposed fix: generate with `censoring=True`, rename `C1`/`C2` to `enrolled_p1`/`enrolled_p2`, declare them with `censoring=`, and add a protocol entry: "Treat disenrollment before readmission or death as censoring; the target is the incidence had every patient stayed enrolled (hypothetical strategy)". Add enrollment to the assumption rationale. Files: the notebook and its callback (decimals and the "without censoring" sentence). Not a library bug. No registered study result moves; the competing study already covers monotone censoring (`ordinary-competing-risk-longitudinal-tmle.md`, "monotone censoring").

### LS-02: the failure mode is shown without its population value, although the law determines one and the fit recovers it
- Cell `failure-reading`: "In this recoded fit the learners specify neither the death mechanism nor the readmission regression correctly. No registered study covers the construction. Do not read -0.0296 as an estimate of the controlled direct effect." and "The generator returns no population value for this estimand, because it defines no world without death."
- Problem: the generator indeed defines no world without death, so no structural CDE truth exists. It does define the identified death-as-censoring functional under the Young et al. ordering (death precedes readmission in an interval): hazard `h s_R / (1 - h s_D)` per node. The fit converges to that functional. Without it, the page demonstrates the failure mode on one draw with no standard error. On the shown draw the censored contrast's CI (-0.0738, 0.0146) contains the total-effect value +0.0022, so the printed numbers alone do not show that the two questions differ. The shown draw is also atypical: its paired gap is smaller than in every swept draw.
- Severity: weak.
- Evidence: [recomputed] death-as-censoring functional: never 0.3049, always 0.2646, difference -0.0403. Total effect: never 0.2444, always 0.2466, difference +0.0022. Alternative ordering (readmission first): difference -0.0093. [executed] 200 draws: censored fit vs the functional: bias +0.0003 for the t=2 difference, SE/SD 0.99, coverage 0.94; never t=2 coverage 0.915, always t=2 0.955. Against the total-effect truth: difference coverage 0.405, never t=2 level coverage 0.075. Paired gap (censored minus total, same data) mean -0.043, SD 0.008, largest -0.0249 over 200 draws; the notebook's gap is -0.0221. The claims "larger reduction" (200/200) and "raises never more than always" (200/200) hold. [read] Young et al. Section 2 states the ordering `(D_k, Y_k, L_k)`, which the notebook's recoding matches.
- Proposed fix: print the identified functional with a label such as "death-as-censoring g-formula value; it is the CDE only if death is exchangeable given the recorded history" and print SEs in the Step 10 table. Optionally expose the functional from `make_longitudinal_competing` under a new truth key. Soften "Do not read -0.0296 as an estimate" to "no registered study covers it". Files: notebook, callback, optionally `src/cleverly/datasets/longitudinal.py` plus a dataset test. Not a bug. No registered study moves.

### LS-03: Step 9 says the competing-risk study "measures" learned-mechanism consistency, while it supplies the true mechanism
- Cell `competing-estimate-reading`: "The page's learners model the treatment mechanism correctly. TMLE is then consistent although the outcome regressions are misspecified. The competing-risk study measures that property and the coverage of this contrast over repeated draws on its own law."
- Problem: the study's `double_robustness/*__mechanism_correct` cell uses `canonical.KnownCompetingMechanism`, the true probabilities, not a fitted logistic model. The page's own trust cell says "Both studies fix or supply the mechanisms, and this page learns them", and the study's limitations say "The mechanisms are supplied". The two cells contradict each other.
- Severity: weak.
- Evidence: [read] `tests/studies/ltmle_competing_properties.py:147-159`; `tests/canonical/lmtp_ltmle_competing/properties.csv` row `death_static_t2__mechanism_correct` (bias -0.0002, coverage 0.9475, n = 4000). [executed] On this page's law the learned-mechanism fit is in fact unbiased: 200 draws, |bias| at most 0.0010 for every CIF and contrast.
- Proposed fix: "The study measures that property with the true mechanism supplied. This page fits a correctly specified mechanism, which no registered study covers." Notebook only.

### LS-04: "Read each 60-day interval as slightly too narrow" moves a supplied-mechanism, n = 2,000 finding onto every 60-day interval on the page
- Cell `trust`: "The survival study also measures a reported standard error slightly below the sampling spread at the second horizon. Read each 60-day interval as slightly too narrow."
- Problem: the survival study's horizon-two shortfall (SE-ratio upper bound 0.9774, coverage upper bound 0.9407) is for supplied mechanisms at n = 2,000 on a finite-support law. "Each 60-day interval" also covers the competing-risk intervals, which come from a different study with no such finding. With learned mechanisms, the influence-curve variance tends to be conservative. On this page's own laws the 60-day intervals are not too narrow.
- Severity: weak.
- Evidence: [read] `ordinary-survival-curve-longitudinal-tmle.md` measured values. [executed] 200 draws, retention fit: t=2 SE/SD always 0.99, never 1.08, difference 1.07; coverage 0.960, 0.955, 0.965. Competing t=2 coverage ranges from 0.92 to 0.975. Monte Carlo SE is about 0.015.
- Proposed fix: scope the sentence to the study: "With supplied mechanisms at n = 2,000, the survival study measures horizon-two SEs a few percent below the sampling spread. This page learns the mechanisms, and no study measures that case." Notebook only.

### LS-05: the competing-risk draw shown is a tail draw, and the explanation accounts for a miss but not for its size
- Cell `competing-estimate-reading`: "Read both misses as sampling variation. A 95% interval misses on about one draw in 20, and this page prints 12 intervals from one fit."
- Problem: the reading itself is right. But a -2.9 SE miss happened for never/death t=2 in 1 of 200 draws (0.5%), and some interval among the 12 misses by at least 2.9 SE in 5.5% of draws. The one-in-20 argument explains a miss, not this size. A reader may suspect bias that the sweep rules out.
- Severity: weak.
- Evidence: [executed] never/death t=2 over 200 draws: bias +0.0007, SE/SD 0.93, coverage 0.930, SD of z 1.08. Mean misses per draw 0.645 of 12; P(at least 2 misses) 0.185.
- Proposed fix: add one sentence on the rarity of the size, or pick a typical seed (that rewrites the Step 8-10 decimals and the callback's miss assertions). Notebook and callback.

### LS-06: practicality. The composite outcome cannot answer "keeps patients enrolled", and "loss of tracking" of enrollment is implausible for a plan
- Cell `title`: "does navigation in each period keep more patients enrolled at 30 and 60 days? | plan exit for any reason, including death". Cell `data-reading`: "`tracked_p1` ... 1 if the plan can observe exit in that period".
- Problem: a reduction in the composite outcome can come entirely from fewer deaths. The page's own competing law has navigation cut 60-day death from 21% to 7%. That is not retention in the business sense. A health plan also observes enrollment administratively, so the page needs a concrete reason why exit becomes unobservable (for example gaps in the eligibility feed). Otherwise the censoring role looks contrived.
- Severity: weak.
- Evidence: [read] cells `title`, `protocol`, `data-reading`; [recomputed] death CIF at t=2: never 0.2107, always 0.0704.
- Proposed fix: state that the composite does not separate death from disenrollment, and point to Steps 8-9 for the cause split (with LS-01 that becomes disenrollment-vs-death). Name a concrete source of loss of tracking. Notebook only.

## Checked and sound

| claim | tag | evidence |
| --- | --- | --- |
| current API; every effective seed set | [read]/[executed] | generator seeds 52/53, learners `random_state=41`, `Runtime(random_state=41)`; `--check` exit 0 |
| truths 0.2579/0.4552/0.1497/0.3172, ATE -0.1082/-0.1380 | [recomputed] | 4M-unit simulation: 0.2577, 0.4552, 0.1496, 0.3170 (MC SE at most 0.00025); quadrature 0.4552/0.3172 |
| competing truths: death t=2 0.2109/0.0705, relapse 0.1187/0.2467/0.1452/0.2443, contrasts -0.0265/+0.0024/-0.0817/-0.1404 | [recomputed] | simulation 0.2107/0.0704, 0.1187/0.2466/0.1452/0.2444, -0.0265/+0.0022/-0.0817/-0.1404 |
| offered patients are older and less ready; needs raise the day-31 offer and the hazard | [read] | `longitudinal.py:628,645,493,498`, `_L2={"w1":0.6,"a1":0.9}` |
| naive comparison understates the benefit "on this draw" | [executed] | true at t=1 in 198/200 draws and at t=2 in 163/200 |
| retention fit: unbiased, calibrated; "60-day reduction is larger" | [executed] | 200 draws: bias at most 0.0007, coverage 0.945-0.970; claim holds in 191/200; both diff CIs cover in 188/200 |
| simultaneous bands "built to hold all four parameters at once with 95% probability" | [executed] | joint coverage 0.96 over 200 draws (no registered study; the page says so) |
| survival view mirrors levels, swaps bounds | [read] | callback asserts exact equality |
| competing fit consistent with a correct mechanism and misspecified outcome regressions | [executed] | 200 draws: |bias| at most 0.0010, coverage 0.92-0.975 |
| readmission difference "negative at 30 days, shrinks by 60" (on this draw); death difference larger at 60 | [executed] | 168/200 and 200/200 |
| `incidence_total` excess 0.0 | [executed] | 200/200 |
| largest epsilon on never/death t=2 "on this draw" | [executed] | draw-specific (82/200), and the table labels it so |
| total effect vs CDE terms; Young et al. (2020) | [read] | Stat Med 39(8):1199-1236, doi:10.1002/sim.8471 (published version, PMC7811594). Section 4.1 is the controlled direct effect, Section 4.2 the total effect. Ordering `(D_k, Y_k, L_k)` matches the recoding |
| Step 10 "what it needs": exchangeability and positivity for death, a well-defined intervention | [read] | matches Young et al. Section 4.1 |
| death-as-censoring gives a larger reduction; never rises more than always | [recomputed]/[executed] | functional: +0.0605 never, +0.0180 always; 200/200 draws |
| assumptions named, untestable ones marked "no" | [read] | cell `identify-reading`; the summary prints sequential positivity "for treatment and remaining under observation" |
| refusal of competing-event elimination and of longitudinal sensitivity | [read] | `longitudinal-tmle.md:305,313`, `estimator.py:235`; anchor `#cross-fitting-the-recursion` resolves |
| trust cell: studies, R references, limits (clustering, eliminated events, pointwise only, survival page silent on clustering) | [read] | both study pages' limitation tables; `longitudinal-tmle.md:395` |
| support numbers: ESS ratio 77.8%, 473/608, no truncation | [executed] | seed-52 reproduction; over 200 draws minimum ESS ratio 0.63-0.87 and truncation always 0 |
| protocol field counts (7 changed and 3 kept; 4; 3) | [read] | callback `changed_fields` assertions |

## Patterns for siblings

- A "How far to trust this" cell that cites a supplied-mechanism study for a learned-mechanism fit. Check that the wording says which one the study covers (LS-03, LS-04).
- A study-specific calibration caveat (horizon-two SE shortfall) copied as a blanket instruction to read every interval as narrow. Re-measure it on the notebook's own law before keeping it.
- A failure-mode step demonstrated on one draw without a population value or SE. Where the law determines the alternative estimand's identified functional, print it. Check the shown draw's gap against a sweep.
- Shown seeds with tail misses (|z| near 3) that the prose excuses with the one-in-20 argument.
- Sections that switch generator settings (`censoring=False`, no clusters) and silently drop a design feature the earlier sections taught as essential.

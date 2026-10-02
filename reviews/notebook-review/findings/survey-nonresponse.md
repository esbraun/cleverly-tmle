# Notebook review: survey-nonresponse

| item | value |
| --- | --- |
| notebook | `docs/examples/survey-nonresponse.ipynb` (callback `tests/unit/tutorial_semantics/survey_nonresponse.py`) |
| commit | `5f33902b` (branch `agent/notebook-review`), Python 3.11.13, `cleverly.__file__` under this worktree's `src` |
| `--check` | exit 0, wall 9 s, "every cell reproduced its stored non-image output" (`.tmp/notebook-review/survey-nonresponse/check.log`) |
| independent truths | `.tmp/notebook-review/survey-nonresponse/truth.py` (4e6-draw MC from the structural equations in `src/cleverly/datasets/synthetic.py:966-1111`, no library truth helper) -> `truth.log` |
| seed sweep | `sweep.py` over seeds 1000-1499 (500 seeds, n = 4000, notebook method, notebook assess call, binary law at seed+1) -> `sweep_1000.csv`, `sweep_1200.csv`, `sweep200.log`, `sweep300.log`; seed 71 reproduces every stored number (`sweep71.log`) |
| probes | `sens.py` (tilt bias, tipping gamma vs n, composite death) -> `sens.log`; `shift71.py` -> `shift71.log`; `boxcal.py` 600 seeds -> `boxcal.log`; `useci.py` |

Independent truths ([recomputed], `truth.log`): strength-2 ATE 1.2000 (MC 1.2004); P(Delta=1) 0.7518;
E[W1 | Delta=1] = -0.2538; respondent-standardized effect E[1.2 - 0.9 W1 | Delta=1] = **1.4284**;
strength-1 ATE 1.2000. Binary law: ey0 0.3764, ey1 0.5626, ate 0.1862, rr 1.4946, or 2.1309. All
printed population values match.

## Findings

### SN-01 (wrong): the composite death rule breaks the declared missingness-at-random assumption

- Cells: `title` ("The protocol scores death before day 30 as the worst transition score (composite
  strategy). That score counts as observed, so only living patients who do not respond are
  missing."), `protocol` (intercurrent-event handling plus the MAR rationale), `identify` /
  `identify-reading` (MAR "transition_score is independent of responded given (transition_navigation, W)").
  Also `docs/examples/index.md:42-43`.
- Problem: if a death is scored worst and counted as observed (Delta = 1), then
  P(Delta = 1 | Y = worst, A, W) = 1 while P(Delta = 1 | Y != worst, A, W) = P(respond | alive, A, W) < 1.
  Y is then dependent on Delta given (A, W) whenever any patient dies before day 30 and any living
  patient fails to respond. The declared MAR assumption is false by construction, and the shipped
  functional E_W[E(Y | A=a, Delta=1, W)] over-weights deaths among "observed" rows. The plausible
  assumption is MAR among survivors (Y independent of R given A, W, D=0). That identifies
  E[Y^a] = E_W[P(D=1|a,W) worst + P(D=0|a,W) E(Y | a, W, D=0, R=1)], which the single-indicator
  estimator does not compute. The synthetic law has no deaths, so the notebook's numbers are
  unaffected; the applied workflow it teaches is not.
- Evidence [recomputed] (`sens.log` part 3): strength-2 law plus death before day 30 with
  P(D | a, W) = expit(-3.5 + 0.8 W1 - 0.5 a) (2.4-3.9 % mortality) and worst score -7. True ATE
  1.3324; the declared functional evaluates to 1.5866; bias +0.254, about 3.5 of the notebook's
  standard errors. Arm means are biased by -0.47 (control) and -0.21 (navigation).
- Proposed fix: state in the applied question and in `assumption_rationale` that the shipped
  estimator needs MAR for the composite outcome, that deaths always being observed makes that
  fail, and that the page therefore assumes no deaths before day 30 (or negligible mortality), or
  change the intercurrent-event strategy for this page. Add a roadmap item for a two-part
  (death always observed, response among survivors) estimator. Same issue for every sibling that
  combines the shared composite strategy with `missingness=`. Files: the notebook,
  `docs/examples/index.md`, possibly `navigation_protocol()` wording. Library bug: no (the library
  cannot see death). Registered study result move: no.

### SN-02 (misleading): tipping gamma 1.306 is read as a property of the survey, but it is seed-specific and scale-dependent

- Cells: `sensitivity`, `sensitivity-reading` ("The tipping gamma is 1.306 ... The review must
  decide whether a shift of that size is plausible for this survey.").
- Problem: (a) the tilt acts on the logit of the outcome after min-max scaling to the **observed
  sample range** (`missingness.py:445-478`, scaler from the fit). That range is a sample extreme of
  an unbounded score, so the same departure in score units maps to a different gamma at each n
  and each draw. (b) No uncertainty for gamma is shown, and the shown seed sits in the upper tail.
- Evidence [executed]: over 500 seeds at n = 4000, tipping gamma ranges 0.674-1.684 (5 %-95 %:
  0.942-1.453, median 1.170); 21.6 % of seeds reach >= 1.306. Scale range 17.6-36.0 (seed 71:
  20.05). Same law, 10 seeds each (`sens.log` part 2): mean tipping gamma 1.301 (n = 1000), 1.231
  (4000), 1.076 (16000), 0.933 (64000), while the mean range grows 17.98 -> 29.32. The quantity
  drifts with n, so it is not a population parameter.
- Proposed fix: report the departure in score units first (and in outcome-SD units, see SN-08);
  state that gamma is defined on the sample min-max scale and is not comparable across samples or
  studies; drop the reading that asks the reviewer to judge "a shift of that size" in gamma. Better:
  make the tilt scale a declared `q_bounds`-like support, or tilt on the outcome scale directly
  (library design change; would need its own test). Files: notebook,
  `docs/technical-reference/validation-methods.md#missingness-tilt-and-tipping-gamma` (no warning
  about scale dependence there either). Library bug: design limitation, not a defect. Study move: no.

### SN-03 (misleading): "a standardized score has no finite support" is the wrong reason for the in-sample fit

- Cells: `protocol-reading` ("A standardized score takes its scale from the data, so the analyst
  can declare no finite support for it"), `estimate-heading` table, `refusal-reading` table
  ("the Step 2 score has no known finite support").
- Problem: a patient-reported transition score is a bounded instrument (for example the Care
  Transitions Measure). Standardizing it "to the baseline distribution" with fixed reference
  constants maps a known finite range to a known finite range, so `q_bounds` is declarable and the
  cross-fitted contract would be open. The true reason the page cannot declare support is that the
  synthetic outcome is Gaussian (`synthetic.py:335`, unbounded). The page teaches a false rule
  about standardized scores.
- Proposed fix: say the synthetic score is unbounded, so no support exists to declare; add that a
  real bounded instrument should declare its support as `q_bounds` and may then cross-fit.
  Files: notebook only. Library bug: no. Study move: no.

### SN-04 (misleading): "contact attempts after assignment cannot enter" the response model is stated as a general rule

- Cells: `title` ("Contact attempts after assignment cannot enter it, because navigation can change
  them"), `protocol` (`assumption_rationale` entry), `plan`.
- Problem: a post-assignment auxiliary L must stay out of the *adjustment set* (it is affected by
  treatment). It need not stay out of the *missingness assumption*. MAR given (A, W, L) is often the
  more plausible assumption (contact attempts and readmission predict both response and score), and
  it is identified by the sequential regression E_W E[E(Y | A, W, L, Delta=1) | A, W]. What is true
  is that this point-treatment estimator cannot use such an L. The page states an estimator
  limitation as a scientific requirement, and it leaves readers believing MAR given baseline only
  is the best available assumption.
- Evidence [read]: `src/cleverly/estimators/tmle.py` single response mechanism on (A, W);
  identification text in `docs/technical-reference/point-treatment-tmle.md:97-106`.
- Proposed fix: rewrite as "This design can only condition response on the arm and baseline
  variables. Contact attempts are post-assignment, so they cannot enter the adjustment set, and
  using them in the response assumption needs a sequential (longitudinal) estimator." Name that as
  a limitation in "How far to trust this". Files: notebook (and protocol strings). Library bug: no.

### SN-05 (weak): the complete-case target is a covariate-standardized contrast, not an effect in a fixed population

- Cells: `failure-reading` ("it targets the respondents' average effect. That effect is 1.409 on
  this draw ... Even correct nuisance models would target 1.409 ... answers a question about a
  different population").
- Problem: response depends on the arm, so "respondents" is a post-treatment group, not a baseline
  population. The complete-case target is E[CATE(W) | Delta=1], the CATE standardized to the pooled
  respondents' covariate law, which also depends on the assignment mechanism g. The index wording
  ("effect standardized to the respondents' covariate distribution") is the correct one. Also,
  1.409 is the sample value; correct nuisances target the population value 1.428.
- Evidence [recomputed]: population respondent-standardized effect 1.4284 (`truth.log`); 500-seed
  complete-case mean 1.4385 (mean minus 1.4284 = +0.0106, MC SE 0.0028); complete-case interval
  covers 1.4284 in 94.0 % of seeds and covers 1.2 in 3.6 %. The index claim holds.
- Proposed fix: use the index wording in Step 7; say "targets 1.428 in the population (1.409 for
  this draw's respondents)"; add one sentence that the respondent covariate mix depends on who was
  offered navigation. Files: notebook. Library bug: no.

### SN-06 (weak): Step 7 invites a nuisance-error explanation the probe does not support

- Cells: `failure-reading` ("The nuisance models of the complete-case fit are also wrong among
  respondents" table; "The estimate lies about one standard error above 1.409. One draw cannot
  separate that nuisance error from sampling variation.").
- Problem: `docs/development/example-notebooks.md:161` asks the page to name another cause only
  when a probe supports it. The probe gives the answer the page says one draw cannot: the
  nuisance-error contribution is +0.011 (0.17 SE). The one-SE distance is sampling variation.
- Evidence [executed]: 500 seeds; (cc_psi - respondent_effect)/se has mean 0.16 and lies in
  (0.5, 1.5) on 28.8 % of seeds; "shift is most of the gap" holds on 100 %. The callback pins
  `0.5 < distance < 1.5` (`survey_nonresponse.py:150-151`), which is a seed-71 relation.
- Proposed fix: replace with "1.0 standard error above 1.409, which is sampling variation; across
  repeated draws the misspecified nuisances move the complete-case estimate by about 0.01". Keep the
  table as the reason the two-nuisance argument does not apply, but quantify it. Study move: no.

### SN-07 (weak): this page's own fits under-cover slightly in repeated samples; the trust table does not say so

- Cells: `estimate-reading`, `top-box-reading`, `trust`.
- Problem: the page correctly says one covered interval is not evidence. But the trust table lists
  "calibration" from the stacked study, and a reader will transfer it. On this page's laws and
  learners, coverage is below nominal.
- Evidence [executed]: Step 6 in-sample fit, 500 seeds: coverage 0.920 (binomial 95 % CI about
  0.894-0.944), bias -0.0045 (MC SE 0.0035), empirical SD 0.0793 vs mean SE 0.0741, median SE 0.0703;
  4.4 % of seeds have SE > 0.10 (max 0.194, extreme weights). Step 9 binary fit (correct logistic
  models), 600 further seeds (`boxcal.log`): 5-fold coverage 0.928, SE ratio 0.960; in-sample
  0.925, ratio 0.955. Sweep: ate/rr/or coverage 0.922 each over 500 seeds.
- Proposed fix: add one row to the trust table: "this page's laws, 500 seeds: coverage about 0.92".
  Library bug: unknown; the anti-conservative SE (ratio 0.89-0.96) at n = 4000 with correct
  parametric models deserves a probe before any claim. Could a registered result move: no (no study
  uses these laws), but it argues for a continuous-law row.

### SN-08 (weak): the score-unit conversion overstates and is unanchored; the CI-based tipping point is not shown

- Cells: `sensitivity`, `sensitivity-reading` ("at most 0.315 of the score range. That is 6.32
  score units").
- Problem: 6.32 is a supremum over hypothetical fitted means. "Score units" have no reference SD.
  The interval-based tipping point (when the conclusion stops being significant) comes earlier.
- Evidence [executed] (`shift71.log`, `useci.py`): at gamma = 1.306 the actual shift averaged over
  the navigation-arm non-response probability is 5.86 units (mean 5.52, max 6.32). The observed
  score SD is 2.13 (navigation arm 1.66), so the shift is about 2.8 SD. `tipping_gamma(...,
  use_ci=True)` returns 1.130 (lower limit -0.00001 there). The prior `use_ci` defect is fixed
  (`missingness.py:580-588` follows the signed limit); the notebook does not call it.
- Proposed fix: print the weighted average shift and the SD ratio; print the `use_ci=True` value
  beside the point tipping gamma. Notebook only.

### SN-09 (weak): the tilt curve away from gamma = 0 is an untargeted plug-in with a misspecified Q

- Cells: `sensitivity-heading` table, `sensitivity-reading`.
- Problem: the curve mixes the MAR-targeted Q* (a wrong linear model here) with its logit-tilted
  version. Targeting solved the gamma = 0 score only, so double robustness does not carry to
  gamma != 0. The table says the intervals ignore uncertainty about gamma, but not this.
- Evidence [recomputed] (`sens.py` part 1, 40 seeds): estimated minus population tilted ATE on the
  same scale is -0.020 (MC SE 0.013) at gamma = 1 and -0.049 (MC SE 0.015) at gamma = 2. Population
  tipping gamma on the seed-71 scale is 1.303 vs reported 1.306, so the headline is barely moved.
- Proposed fix: one sentence: "Away from gamma = 0 the curve is a plug-in. Its accuracy depends on
  the outcome model, which is wrong here." Notebook and technical reference.

### SN-10 (weak): the omitted-variable reason contradicts the ledger's own reason

- Cell: `assessment-reading` ("no derivation here covers a fit that also models response").
- Problem: the ledger detail says the opposite: "Theorem 2 of Chernozhukov, Cinelli, Newey, Sharma
  and Syrgkanis (2026) covers it, and the bound is well posed here. Three pieces are missing"
  (implementation pieces). [executed] `assess().to_frame()` row `omitted_confounding`.
- Proposed fix: "The bound is well posed for this fit, but the package has not implemented its
  response-aware representer and variance." Notebook only.

### SN-11 (weak): the attributable-fraction refusal is framed as a scientific boundary rather than a package policy

- Cell: `refusal-reading` ("The 2026-09-12 source audit found no published derivation of that
  construction").
- Problem: under MAR given (A, W), E[Y] = E[m(A, W)] and E[Y^0] = E[m(0, W)]. Both are shipped,
  with known influence curves (`point-treatment-tmle.md:99-106` gives the first), and PAF is a
  smooth function of the two. A hostile expert reads the refusal as the package's evidence policy,
  not an identification gap. The roadmap (RM8/F20) states it as a policy.
- Proposed fix: say "The package refuses it until a published result covers the joint construction.
  The pieces exist; the joint targeting and inference are not validated here." Notebook only.

## Checked and sound

| claim | tag | evidence |
| --- | --- | --- |
| Every effective seed is set; current API | [read] | generator seeds 71/72, `Runtime(random_state=...)`, `random_state` on both logistic learners; `LinearRegression` has none |
| Stored outputs reproduce | [executed] | `--check` exit 0, 9 s; `sweep.py 71` reproduces 1.189/0.068, 1.467, 1.409, 1.223, 0.1918/1.5139/2.1817, tipping 1.306, 6.32 |
| Population ATE 1.200 for both laws, binary truths 0.1862/1.4947/2.1312, shares 0.376/0.563 | [recomputed] | `truth.log` MC from structural equations |
| Respondents lower-risk; non-respondents higher | [executed] | W1 respondent mean < 0 < non-respondent mean in 500/500 seeds (means -0.255, 0.766) |
| Naive respondent difference overshoots ATE | [executed] | unadjusted > 1.2 in 500/500 seeds (mean 1.75) |
| Marginal response gap small though law has an arm term | [recomputed] | population 0.747 vs 0.757; navigation higher in 75 % of seeds, which the page does not overclaim |
| Learner table: g and response model correctly specified, Q wrong | [read] | `synthetic.py:992-1007` |
| Step 6 fit centered on truth (second DR route) | [executed] | 500-seed mean 1.1987, bias -0.0045 (MC SE 0.0035) |
| Complete-case fit excludes the ATE and is centered on the respondent-standardized effect | [executed] | covers 1.2 in 3.6 %, covers 1.4284 in 94.0 %; cc > 1.2 in 100 % |
| "That shift is most of the gap" | [executed] | 100 % of 500 seeds |
| Complete-case treatment model misspecified among respondents | [recomputed] | logit P(A=1 | W, Delta=1) = logit g + log(pi1/pi0), nonlinear in W1, W3 |
| Mild law: constant effect, linear Q, complete-case consistent | [executed] | coverage 94.8 % over 500 seeds; mean 1.2005 |
| OR > RR > 1, log-scale intervals asymmetric | [executed] | 100 % of 500 seeds |
| PAF refused before learners; NaturalCourseMean contracts | [read] | callback lines 202-241 run them |
| E-value unavailability reason | [executed] | ledger detail matches the page |
| Tilt definitions, arm_gamma direction, gamma = 0 reproduces MAR, SE reused | [read]+[executed] | `missingness.py:428-485`; curve rows match |
| Largest-shift formula expit(gamma/2), 2m-1 | [recomputed] | derivative of m - expit(logit m - gamma) vanishes at s = 1 - m; 0.315, 0.658 |
| `use_ci=True` defect from the prior review | [executed] | fixed; returns 1.130 on seed 71 |
| Index claim "complete-case fit estimates the effect standardized to the respondents' covariate distribution" | [recomputed] | target 1.4284; complete-case mean 1.4385 (nuisance excess +0.011) |
| Trust cell, ordinary missing-outcome study | [read] | `tests/canonical/tmle_mar/properties.csv` 19/19 pass; binary law, finite-support oracle nuisances, `ate`/`ey0`/`ey1`, complete-case control (bias -0.123 to -0.117); page agrees |
| Trust cell, stacked arm-indexed study | [read] | page: 10 folds, depth-5 trees, L1-L4 binary and bounded continuous, 2 and 3 arms, rr/or rows, overfitting cells |
| Literature | [read] | the notebook cites no paper. The technical reference attributes the logit mean shift to Scharfstein, Rotnitzky and Robins (1999); see Patterns |
| Practicality | [read] | the question (regional plan, survey non-response, top-box scorecard, RR vs OR) is realistic; SN-01 and SN-04 are the practical gaps |

## Patterns for siblings

1. Any sibling that combines the shared composite death strategy (`docs/examples/index.md:42-43`)
   with `missingness=` inherits SN-01. Deaths are always observed, so MAR for the composite fails.
2. Data-dependent min-max scaling makes any logit-scale sensitivity parameter (tilt gamma, and
   anything else read on the [0, 1] scale) sample-specific. Check every sibling that reports a
   gamma or a scaled quantity as a substantive magnitude.
3. "No finite support" is claimed for synthetic Gaussian outcomes that the prose calls bounded
   instruments. Check sibling rationales for the in-sample fit.
4. Callback relations such as `0.5 < distance < 1.5` pin seed-specific facts; the prose around
   them should say "on this draw" and give the repeated-sample picture.
5. The technical reference (`validation-methods.md:1624-1636`, `missingness.py` docstring) calls the
   logit shift of the mean "the Scharfstein, Rotnitzky and Robins (1999) tilt". SRR (JASA 94:1096-1120)
   is a selection-model exponential tilt of the outcome density. It reduces to a logit shift of
   the mean only for a binary outcome. For a continuous outcome on a min-max scale this is a
   pattern-mixture mean model. The notebook's own "pattern-mixture" label is correct.
6. Coverage of fits on the notebooks' own laws at n = 4000 ran 0.92-0.93 here; other siblings with
   strong weights may show the same.

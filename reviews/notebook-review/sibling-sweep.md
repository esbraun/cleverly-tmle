# Sibling sweep: cross-notebook patterns

Commit `5f33902b`. The sweep reads every cell of the ten notebooks through a dump of sources and
stored outputs (`.tmp/notebook-review/sweep/dump.py` -> `cells.txt`, one `stem|cell id|kind|line`
row per line). `.tmp/notebook-review/sweep/narration_gaps.py` -> `narration_gaps.log` checks P9 and
P10 by script. Finding IDs refer to `ledger.md`. Notebook stems are abbreviated as in the ledger:
PT, CF, CT, DR, IV, SN, LT, LS, MS, TW.

## Summary

| pattern | locations | fix shape | recommended single fix |
| --- | --- | --- | --- |
| P1 default HGB propensity | PT, CF, IV (regime, incremental, dose density); TW partly | per notebook, with one shared rule | correctly specified (degree-2 logistic) or seed-checked g; 200-seed coverage check first |
| P2 "standardized, so no finite support" | CT, SN, IV, MS; PT and IV phrasing nuance | per notebook, one wording | "this synthetic score is unbounded; a bounded instrument declares `q_bounds`" |
| P3 "standardized (mean 0, SD 1)" covariates | all 8 synthetic pages plus `index.md:57` | shared (index) plus one line per page | "independent standard-normal draws" |
| P4 sensitivity vocabulary | PT, CF, TW; library docstrings and summary strings; `validation-methods.md` | shared library change plus notebook rows | one vocabulary table, library strings updated |
| P5 trust cells overstate study coverage | all ten | per notebook, one shared checklist | require four columns: law, nuisance construction, outcome type, fold layer |
| P6 positivity tail of `navigation_data` | PT, CF, DR, IV (regime) | shared narration plus per-page row | narrate PT-N1 once in `index.md` and on each page; defer generator |
| P7 composite death rule | every point-treatment protocol, LT, SN, LS, `index.md` | shared (protocol helper plus index) | state the frame coding and the MAR scope in `index.md` and the helper |
| P8 misses read as "sampling variation" | IV (wrong), CF, LS, SN (inverse) | shared house rule | require a seed sweep before either explanation |
| P9 stale callback comments | CT, LT, MS; CF/PT "retired law" note | per callback | update numbers; optional comment check |
| P10 sci-notation blind spot | TW only | shared harness | extend narration check with mantissa-relative tolerance |
| P11 unsupported evidence words in library messages | LT message; `cv-tmle.md`; one more checked | library string | "reports a cluster-robust variance" |
| P12 single-draw claims as method properties | CT, IV, MS, TW, LT, DR, PT, CF; `ctmle.py`, tech ref | per notebook plus callbacks | "on this draw" plus a rounded sweep share, or drop the multiplier |
| P13 protocol text contradicts the synthetic law | IV, CT, SN, MS, DR, LS, CF | per notebook | check every protocol field against the generator |
| P14 library text contradicts library code | 7 sites | library docstrings and strings | one docstring pass |
| P15 "correct learners make the data-reuse condition plausible" | LT, MS | per notebook | "finite-dimensional learners make it plausible" |
| P16 diagnostics from a fitted model read as population facts | DR, IV, CF, PT | per notebook | one sentence per support row |
| P17 machine-noise numbers in stored outputs | TW (fails), DR, IV (latent) | per notebook | print pass/fail or a rounded ratio |
| P18 refusal-inventory proposals that conflict with PT-N1 | PT, CF | plan-level | do not publish DR-nu^2 limits on `navigation_data` without PT-N1 |

---

## P1. Default or unregularised `HistGradientBoostingClassifier` propensity

The bare booster has no early stopping below 10,000 rows. On `navigation_data` its out-of-fold
calibration slope is about 0.47 on every swept seed (PT, CF, IV, `refusal-inventory.md`).

| location | quote | consequence measured | ledger |
| --- | --- | --- | --- |
| PT `estimate` | `treatment_learner=HistGradientBoostingClassifier(random_state=21)` | SE/SD 1.56 to 2.1; DR nu^2 refused on 38/40 and 56/60 seeds | PT-02 |
| PT `failure-mode` | `HistGradientBoostingClassifier(random_state=21)` (flexible g arm) | not swept separately; same learner | PT-02 |
| CF `estimate` (`boosted`, reused by `failure-mode`, `reuse-split`, `constructions`, `repeated-folds`) | `treatment_learner=HistGradientBoostingClassifier(random_state=34)` | cross-fitted SE/SD 1.62 to 2.12; the "0.27" ratio becomes about 0.7 with a calibrated g | CF-01, CF-03 |
| IV `regime-fit` (method reused by `incremental-fit`) | `treatment_learner=HistGradientBoostingClassifier(random_state=31)` | incremental: coverage 0.45 to 0.48, bias +2 SE; regime: SE/SD 1.46 to 1.80 | IV-01, IV-09 |
| IV `dose-fit` | `treatment_learner=HistGradientBoostingClassifier(random_state=32)` (40-bin pooled-hazard density) | +1.0 shift bias -0.024 to -0.028; reviewer probe ties most of it to the density, not Q | IV-02 |
| TW `production-fit` (first mention is the `setup` import) | `HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=30, random_state=SEED)` inside a Super Learner | partly regularised, weight 0.156; propensity slope 0.8127 warns; DR nu^2 negative on seeds 5 and 11 from one extreme cross-fitted g (0.00129) | TW-02 |
| DR `estimate` | `outcome_learner=HistGradientBoostingRegressor(random_state=55)` only; g is the deliberately misspecified main-effects logistic | not affected by P1 | none |

- IV needs a correctly specified g, not calibration. The IV verifier measured sigmoid
  `CalibratedClassifierCV` at coverage 0.63 with 80% of the bias left (IV-N1). The IPSI functional
  carries g inside the estimand, so a slope repair does not fix the error that matters.
- PT and CF also work with a regularised booster or degree-2 logistic g (CF verifier: SE/SD 0.94,
  coverage 0.917 for both; PT verifier: 0.87, coverage 0.95).
- The evidence conflicts on the size of the gain. `refusal-inventory.md` (20 shared seeds) found
  calibrated-g SE/SD of 0.64 to 0.71 and coverage 16 to 17 of 20. The PT and CF sweeps (40 to 60
  seeds each) found 0.87 to 1.06. PT-N1 implies a heavy-tailed target, so 20 draws cannot settle it.
- **Fix shape:** per notebook, under one shared rule.
- **Recommended single fix:** use the degree-2 logistic g (`make_pipeline(StandardScaler(),
  PolynomialFeatures(2), LogisticRegression(...))`) on PT, CF, and the IV binary axes. It is the
  one learner that repairs all three estimands in the sweeps. Present it as the analyst's flexible
  parametric choice, not as truth-tuned hyperparameters (PT-02 verifier). Before adoption, run one
  200-seed coverage check on this law, as `refusal-inventory.md` asks. Keep the bare booster only
  as the Step 9 diagnostic lesson, if at all. Add one row to the code-cell rules in
  `docs/development/example-notebooks.md`: "Do not pass a bare `HistGradientBoostingClassifier` as
  a propensity learner below 10,000 rows". For IV `dose-fit`, the density learner fix is IV-02's
  (more bins plus a quadratic Q, then re-sweep).

## P2. "A standardized score has no finite support" as the reason for an in-sample fit

The shared protocol's outcome is "Patient-reported transition score, as a share of the maximum
score" (`src/cleverly/datasets/navigation.py:177`), which is bounded. Four pages replace it with a
"standardized" score and then argue from standardization to unbounded support. The real reason is
the synthetic Gaussian outcome. An affine standardization of a bounded instrument stays bounded.

| location | quote | ledger |
| --- | --- | --- |
| CT `protocol` / output | `outcome="Patient-reported transition score, standardized to the baseline distribution"` | CT-01 |
| CT `protocol-reading` | "A standardized score takes its scale from the data, so the analyst can declare no finite support for it." | CT-01 |
| CT `estimate-heading` | "This page's score is standardized, so it has no known finite support to declare" | CT-01 |
| CT `trust` | "The standardized score has no support to declare, so no fit here can hold rows out" | CT-01 |
| SN `protocol` / output | "Patient-reported transition score, standardized to the baseline distribution, from the survey sent on day 30 ..." | SN-03 |
| SN `protocol-reading` | "A standardized score takes its scale from the data, so the analyst can declare no finite support for it." | SN-03 |
| SN `refusal-reading` | "the Step 2 score has no known finite support" | SN-03 |
| IV `dose-identify-heading` | "This law records a standardized score, which takes its scale from the data, so no finite support can be declared for it." | IV-04 |
| IV `dose-fit-heading` | "a standardized score with no known finite support, so a cross-fitted fit could declare no `q_bounds`" | IV-04, IV-N3 |
| IV `regime-fit-heading` | "Axis 2 uses another law, whose outcome has no such support" | IV-04 (same fix) |
| MS `protocol` / output | `outcome="Patient-reported transition score, standardized to the baseline distribution"` | CT-N1 |
| MS `protocol-reading` | "the score is standardized to the baseline distribution, so it has no fixed maximum" | CT-N1 |
| MS `estimate-arms-heading` | "standardized score, which has no such support, so `cleverly` refuses a cross-fitted fit of it" | CT-N1 |
| PT `populations-heading` | "The second law has a Gaussian outcome with no known finite support, so a cross-fitted fit could declare no `q_bounds`." | reason correct; "could declare no" too strong (IV-N3 nuance) |

- **Fix shape:** per notebook, with one shared sentence. CT, IV, and MS protocol edits move the
  printed fingerprints, so those notebooks need re-execution.
- **Recommended single fix:** replace the `outcome` text with "Patient-reported transition score on
  a synthetic scale with no documented range". In each reading write: "This synthetic score has
  Gaussian noise, so it has no finite support. A real instrument has a published range. Declare
  that range as `q_bounds` on the instrument's own scale and cross-fit." Change "could declare no
  `q_bounds`" to "has no support to declare; any bound would be a convention". The refusal
  framing goes under the refusal-removal ruling (`refusal-inventory.md` class P rows).

## P3. "Standardized (mean 0, SD 1)" for independent N(0, 1) draws or an unstandardized outcome

Every synthetic generator draws its covariates as independent standard normals
(`synthetic.py:366-367` `_latent`; `longitudinal.py:375-376, 623-624, 821-822`). No sample
standardization happens, and the covariates are mutually independent.

| location | quote |
| --- | --- |
| `docs/examples/index.md:57` | "The synthetic laws standardize their baseline covariates to mean 0 and SD 1." |
| PT `data-reading` | "The four baseline covariates are standardized (mean 0, SD 1), so a negative ..." (PT-09) |
| CF | inherits through "The law is the point-treatment tutorial's law" (`data-heading`); no own sentence |
| CT `data-reading` | "The three baseline covariates are standardized (mean 0, SD 1)" |
| DR `data-reading` | "the four baseline covariates are standardized (mean 0, SD 1) \| a negative value is below the average" |
| IV `data-reading` | "The baseline covariates are standardized (mean 0, SD 1)." |
| SN `data-reading` | "Covariates are standardized (mean 0, SD 1), and scores are in synthetic units." |
| MS `data-reading` | "covariates are standardized (mean 0, SD 1), and the outcome is in synthetic units" |
| LT `data-reading` | "baseline covariates, standardized (mean 0, SD 1)" |
| LS `data-reading` | "standardized baseline covariates (mean 0, SD 1)" |
| IV `dose-identify`, CT/SN/MS `protocol` | the outcome is called "standardized" though unstandardized (IV-04 wrong; P2) |

- TW uses real data and does not make the claim.
- **Fix shape:** shared. Fix `index.md:57` once, then one line on each page.
- **Recommended single fix:** "The synthetic laws draw each baseline covariate independently from a
  standard normal distribution. A negative value is below the population average. Real covariates
  would be correlated." Each page can then say "independent standard-normal draws" and link the
  index.

## P4. Sensitivity vocabulary

| location | quote | problem | ledger |
| --- | --- | --- | --- |
| PT `sensitivity-heading` | "The largest bias is $\sqrt{\sigma^2 \nu^2}$, and every number below scales with it" | S is a scale; the factor can exceed 1 | PT-01 |
| CF `sensitivity-heading` | "Every bound below scales with it" | correct; no "largest" claim | none |
| TW `sensitivity` output (library) | "maximal bias sqrt(sigma^2 nu^2) = 0.37673" | library summary string `omitted_variable.py:1145` | PT-01 sibling |
| TW `diagnostics` output (library) | "max_bias=0.3767" | field name follows DoubleML; `assessment.py:3186` | PT-01 sibling |
| `omitted_variable.py:14` | `\underbrace{\sqrt{\sigma^2 \nu^2}}_{\text{max bias}}` | module docstring | PT-01 |
| `omitted_variable.py:520,544,1196,1256` | "the maximal bias", "the multiplier on the maximal bias" | docstrings | PT-01 |
| `validation-methods.md:833,856` | "the curve of the maximal bias"; table row "maximal bias" | reader-facing doc | PT-01 |
| `validation-methods.md:734` | "the largest bias an unmeasured confounder of declared strength can produce ... with declared partial-$R^2$ strength in each" | "of declared strength" makes it correct; "partial-R^2 in each" is the PLM reading | PT-05 sibling |
| PT, CF `sensitivity-heading` | "`cf_d` \| the share of the remaining treatment variation that a hidden confounder explains" | PLM reading | PT-05, CF-05 |
| TW `sensitivity-heading` | "Its strength cf_d is the share of the residual treatment variation." | PLM reading | TW-09 |
| TW `sensitivity` output (library) | "a confounder explaining 14.2% of the residual variation in BOTH the outcome and treatment" | library RV string `omitted_variable.py:1155`, PLM reading | TW-09 |
| `omitted_variable.py:909, 1239, 1414` | "Share of the residual treatment variation ..." (param docs; `:1414` is the benchmark's cf_d, which is a gain) | PLM reading; benchmark is gain-form | PT-04, PT-05 |
| `omitted_variable.py:21-22` | "`cf_d` -- the corresponding gain in the Riesz representer, i.e. how much the confounder would improve prediction of treatment" | the notebooks paraphrase its second half | CF-05 |
| PT `sensitivity-reading` | "The implied `cf_y` reaches 1.0000, which is the whole of the remaining outcome variation" | silent clip of 1.316 | PT-04 |
| TW `sensitivity` output | "implied cf_y = 0.0000" | no clip; stable reading | TW-10 |

- PT-04's share-form library proposal is refuted (gain form is the paper's convention). The
  library change is only a clip flag or the unclipped value.
- **Fix shape:** shared. One library change covers the strings that the notebooks print, and the
  notebooks then share one vocabulary table.
- **Recommended single fix:** add one vocabulary table to `validation-methods.md` (S is "the scale
  of the bound"; `cf_y` a share of residual outcome variance; `cf_d` "the share of the Riesz
  representer's second moment, here the average inverse-propensity weight, that only the hidden
  confounder would add"; benchmarks report gains that `cleverly` clips to [0, 1]). Then:
  - rename "max bias" and "maximal bias" to "bias scale" in the docstrings, the summary string
    `omitted_variable.py:1145`, and `validation-methods.md:833,856`. Keep the `max_bias` field
    name (DoubleML parity), or rename it in the same change with its callers (no compatibility
    shim, per `CLAUDE.md`);
  - change the RV string `:1155` to "a confounder with `cf_y` = `cf_d` = 14.2% ...";
  - point the PT, CF, and TW tables at that vocabulary.
  The summary-string change moves the TW stored output and needs a failing-first test on the
  string. No study moves.

## P5. "How far to trust this" overstates registered-study coverage

| page | trust-cell quote | gap | ledger |
| --- | --- | --- | --- |
| PT | "It adds a cross-fitted tree-learner control." | role reversed; same-law n = 500 cells uncited | PT-07 |
| CF | "without splitting, the boosted fit reported a standard error 0.27 times the cross-fitted one" | yardstick conservative; coverage gap 48.8% vs 93.3% not the lead | CF-01, CF-N1 |
| CT | "The registered study repeats this control over many draws." | bounded twin, cross-fitted, n = 1,500; scores bias, not selection | CT-07 |
| DR | "That study has a cell for this page's case" | parametric GLM outcome, GLM reductions, ten folds, binary law | DR-04 |
| DR | "At n = 1,500 the bias exceeded the equivalence margin." | point bias inside margin | DR-05 |
| DR | "The coverage of that interval rests on the rate conditions" | omits cross-fitting and condition (S) | DR-06 |
| IV | per-axis table lists "no flexible learners" | every study supplies g or the density exactly (`OracleShiftDensity`) | IV-03 |
| SN | stacked arm-indexed study row lists "calibration" | page's own law: coverage 0.92 over 500 seeds | SN-07 |
| LT | "[cross-fitted end-of-study study] is the closest" | the ordinary in-sample study is closer | LT-03 |
| LT | "No registered study covers clustered fits with estimated mechanisms." | no clustered longitudinal study of any kind | LT-N1 |
| LS | "The competing-risk study measures that property" (`competing-estimate-reading`) | mechanism supplied | LS-03 |
| LS | "Read each 60-day interval as slightly too narrow." | a supplied-mechanism, n = 2,000 survival-study cell | LS-04 |
| MS | study limits omit oracle nuisances; "Correctly specified parametric learners are what makes that condition plausible" | exact nuisances, fixed bounds; page coverage 0.92 at n = 3,000 | MS-02, MS-06 |
| TW | "clustered point-treatment CV-TMLE study, with a continuous outcome, a linear outcome regression" | study regenerated on a binary law | TW-03 |
| TW | semi-synthetic row: cross-fitted end-of-study study \| "any finding about the real prenatal records" | **new sibling instance:** the "does not establish" column omits that the study supplies the mechanisms and uses another law; TW fits both mechanisms with `LogisticRegression` | new (weak); same fix as LT-03 / LS-03 |

- **Fix shape:** per notebook, with one shared checklist.
- **Recommended single fix:** add one rule to `docs/development/example-notebooks.md` row 12
  (`trust`). Each cited study row names four things: the study's law (same or other), its nuisance
  construction (supplied, oracle, or fitted), its outcome type, and its fold layer (in sample or
  cross-fitted). It also names the cell's role (positive or control). Then apply the rule to the
  rows above. Where a page quotes a review sweep, it says "a review probe, not a registered study".

## P6. Positivity tail of the `navigation_data` law (PT-N1)

The propensity logit of `nonlinear_dgp` (`synthetic.py:608`) gives E[1/g] = infinity. E[1/(1-g)] is
finite, so the problem sits in the offer arm. This is the only quadratic propensity among the
generators (`grep` for `** 2` in `synthetic.py` and `longitudinal.py`). The MS softmax law and
every logistic-linear law have E[1/g] finite (exponential moments of a Gaussian).

| page | cells that use the law | claim at risk | ledger |
| --- | --- | --- | --- |
| PT | `data` (`navigation_data(n=3_000, seed=21)`) | `plan`: "positivity must hold, and a support report cannot verify it"; `sensitivity-reading`: "The doubly robust estimator of $\nu^2$ ... equals the true $\nu_0^2$ minus the squared error"; `truncation-reading` | PT-08, PT-N1 |
| CF | `data` (seed 34), `reuse-split` (seed 35) | `sensitivity-reading`: "That estimator equals the true $\nu_0^2$ minus the squared error of the fitted representer, so a negative value reports a fitted treatment mechanism far from the treatment law"; CF-03 reviewer value "true nu^2 = 7.77" | CF-03, PT-N1 |
| DR | `data` (seed 55) | `sensitivity-reading`: "The default estimator of $\nu^2$ equals the true $\nu_0^2$ minus the squared error ..."; cb `dr_tmle.py:138-139` "sample second moment ... is 7.75"; DR-07 "population nu^2 8.22" | DR-07, DR-09, DR-N1 |
| IV | `data` (seed 31): regime and incremental axes | offer-to-all and the screen both divide by g for offered patients; `regime-support-heading`: "A rule needs positivity only where it assigns" is true, but flagged patients also reach g near 0 through `W2`. The incremental axis needs no positivity (correct) | IV-09 (context) |
| `example-notebooks.md:172` | DR row of the omitted-variable table | "A wrong treatment model makes the estimate of $\nu^2$ too small" | DR-07 |
| studies | `cvtmle_properties.py`, `bounded_cv_laws.py`, `canonical_properties.py`, `ctmle_oat_properties.py` | share `nonlinear_dgp().propensity` | PT-N1 option (ii) |

- The CF team law, the CT instrument law, the SN missing-outcome law, the IV dose law, the MS law,
  and the longitudinal laws are not affected.
- **Fix shape:** shared narration, with a row on each affected page.
- **Recommended single fix:** add one sentence to the shared design in `docs/examples/index.md`:
  "In `navigation_data`, the true chance of an offer approaches 0 fast enough that E[1/g] is
  infinite. The fitted propensity bound makes the effect estimable, and each page's truncation or
  support step measures that choice." On PT, CF, DR, and IV, add a data-table row pointing there.
  Replace the "equals the true nu_0^2 minus ..." sentences with "estimates a truncation-dependent
  second moment". Never quote a Monte Carlo "true nu^2". Defer the generator change, which would
  move four studies.

## P7. Composite death rule versus missing at random and censoring

| location | quote | interaction | ledger |
| --- | --- | --- | --- |
| `docs/examples/index.md:42-43` | "The protocol scores death before day 30 as the worst transition score (composite strategy)" / "the composite score counts as observed. Only living patients who do not respond are missing" | pairs the rule with a single-indicator MAR analysis | SN-01 |
| `index.md:57-59` | "its protocol scores death before day 30 as not top box" | LT censoring coding unstated | LT-05 |
| `navigation.py:181-182`, `:263` | the helper strings that every protocol inherits | source of the rule | SN-01, LT-05 |
| SN `title`, `protocol`, `identify` | "That score counts as observed, so only living patients who do not respond are missing" + MAR | MAR false when anyone dies (bias +0.254 in the witness) | SN-01 (wrong) |
| LT `protocol-reading` | "loss to tracking ... is not an intercurrent event" | deaths must be coded tracked, top box 0 | LT-05 |
| LS `protocol` | "Count death as plan exit (composite strategy)" | composite cannot answer "keeps patients enrolled" | LS-06 |
| PT, CF, DR, IV (3 protocols), MS, CT `protocol` outputs | inherited "worst transition score" entry | harmless with no missingness. **New sibling of SN-N1:** CT, IV `dose-identify`, and MS also declare an unbounded "standardized" score, so "the worst transition score" has no value | SN-N1 (weak siblings) |

- Only SN combines the program rule with `missingness=`, and only LT combines it with
  `censoring=`. LS combines its own composite (death counts as plan exit) with
  `censoring=("tracked_p1", "tracked_p2")`. A grep of the design calls verifies all three.
- **Fix shape:** shared.
- **Recommended single fix:** add two rows to the shared design table in `index.md`. One row:
  "a death before day 30 enters the frame as an observed outcome (worst score, or not top box).
  Only living patients can be missing or censored." Another row: "Missing at random then applies
  to survivors only; the point-treatment estimator assumes it for the composite and so assumes no
  deaths". Mirror the coding sentence in the `navigation_protocol()` and
  `longitudinal_navigation_protocol()` rationale strings, which moves every protocol fingerprint.
  Fixing P2 (bounded instrument) removes the "worst score of an unbounded scale" conflict.

## P8. Interval misses read as "sampling variation"

House rule `docs/development/example-notebooks.md:161`: "Explain an interval that misses the true
value as sampling variation, and give the distance in standard errors. Name another cause only
when a probe or a study supports it."

| location | quote | sweep result | ledger |
| --- | --- | --- | --- |
| IV `incremental-fit-reading` | "A 95% interval misses on about one draw in 20, so one miss is consistent with sampling variation." | coverage 0.45 to 0.48; same-sign bias on 57/60 | IV-01 (wrong) |
| IV `dose-fit-reading` | "... fit bear on that miss" (+1.0 shift, attributed to in-sample fitting and strain) | coverage 0.60 to 0.65, systematic negative bias | IV-02 |
| CF `failure-mode-reading` | "A 95% interval misses on about one draw in 20, so one miss can be sampling variation." | in-sample coverage 0.73 to 0.83 with this learner, so the miss is the failure mode itself. **New sibling instance (weak):** the hedge undercuts the step's lesson | new |
| LS `competing-estimate-reading` | "Read both misses as sampling variation. A 95% interval misses on about one draw in 20" | supported (coverage 0.93; largest of 12 at 2.9 SE in 5.5% of draws) | LS-05 (none) |
| SN `failure-reading` | "One draw cannot separate that nuisance error from sampling variation." | inverse case: the probe shows sampling variation (mean distance 0.16 SE) | SN-06 |

- **Fix shape:** shared house rule, then per notebook.
- **Recommended single fix:** rewrite the rule: "Before you explain a miss, run a seed sweep of the
  shown configuration (at least 60 draws). Call it sampling variation only when the sweep's
  coverage is near nominal. Otherwise name the measured bias or under-coverage and its probed
  cause. Commit the sweep script or cite it as a review probe." CF's sentence becomes "The in-sample
  interval misses here, and in-sample intervals miss more often than one draw in 20 (Step 13's
  study: 48.8% coverage)."

## P9. Stale numbers in callback comments

`narration_gaps.py` compares every decimal in each callback comment with the notebook's stored
outputs.

| location | comment | stored | ledger |
| --- | --- | --- | --- |
| `collaborative_tmle.py:123` (ledger cites `:120`) | "AUC of 0.843" | 0.844 | CT-09 |
| `longitudinal_tmle.py:161` | "measured 33.3" | 35.239 | LT-08 |
| `msm_projections.py:214` | "1.23% of the units" | 1.20% | MS-07 |
| `cross_fitting.py:74-76`, `point_treatment_tmle.py:55` | "scaled by the ratio of the two ATEs (0.1629 against 1.750)" of "the retired Gaussian law" | a margin derived from a law no page uses; not a stale printed number | new (weak): restate the margin's basis |
| `dr_tmle.py:138-139` | "nu^2 is 4.316 ... is 7.75" | computed in the callback, not printed | not stale; the 7.75 "true" value is a PT-N1 divergent integral, so reword |
| `longitudinal_survival.py:132,189,224` | ratios 1.37, 1.55; "2.9 and 2.3" | recomputed from stored outputs: 0.1576/0.1147 = 1.374, 0.1090/0.0704 = 1.548; distances printed | correct |

- **Fix shape:** per callback.
- **Recommended single fix:** update the three numbers and the two margin notes. Optionally add a
  fast test that applies `narration_mismatches` to comments of the form `# "<quoted prose>"` in
  each callback (a shared change to `tests/unit/tutorial_semantics/__init__.py`).

## P10. Narration test blind to scientific notation

`_DECIMAL` (`tests/unit/tutorial_semantics/__init__.py:255`) refuses a mantissa followed by `e`.
`_NOT_NARRATION` (`:249`) also strips inline code, so a decimal inside backticks is never checked.

| location | literal | printed at that precision? |
| --- | --- | --- |
| TW `targeting-reading` | 5.0875e-04, 2.9606e-19, 4.2286e-03 | no (stored table prints 0.0005, 0.0000, 0.0042) |
| TW `targeting-reading` | 5.114e-04 | yes |
| TW `production-fit-reading` | -1.130e-06, 5.114e-04 | yes |
| all ten notebooks, inline-code decimals | every decimal inside backticks | yes (no mismatch found) |

- No other notebook narrates a scientific-notation literal. The inline-code gap has no current
  victim.
- **Fix shape:** shared (harness).
- **Recommended single fix:** extend `narrated_decimals` with a scientific-notation pattern whose
  tolerance is half a unit of the mantissa's last place times 10^exponent, with a failing-first
  test on a narrated `1.23e-04` that no output prints (TW-04). Decide in the same change whether
  inline-code decimals should be checked; today they are not.

## P11. Unsupported evidence words in library messages

| location | text | status | ledger |
| --- | --- | --- | --- |
| `src/cleverly/longitudinal/estimator.py:2555` | "Fit in sample (...), which is clustered and evidenced, or drop id= from fit." | unsupported: no registered clustered longitudinal study | LT-04 |
| `docs/technical-reference/cv-tmle.md:287` | quotes the same message | follows the message | LT-04 |
| LT `estimate-heading` | "it names the in-sample fit, which is clustered and evidenced" | echoes the message | LT-04 |
| `src/cleverly/longitudinal/estimator.py:2008` | "Pass n_folds=1 for the evidenced in-sample longitudinal MSM construction." | supported: `method-evidence/ordinary-longitudinal-msm-projection.md` exists | checked |
| `src/cleverly/targets/population_intervention.py:115` | "docs/roadmap.md F20 tracks this identification boundary." | an evidence-policy boundary, not an identification one | SN-11 |
| `src/cleverly/validation/drtmle.py:600` | "this fit is Theorem 1's estimator" (also for cross-fitted fits) | overclaims for cross-fitted fits | DR-06 |
| `src/cleverly/targets/builtin.py:227` | "... is exactly what the cap is declared to secure" | holds only for a cap inside the conditional support | IV-06 |

- **Fix shape:** library strings, one change.
- **Recommended single fix:** one pass over the four strings. "which reports a cluster-robust
  variance"; "F20 tracks this unvalidated composition"; "this fit uses Theorem 1's construction;
  the theorem does not cover cross-fitting" for `cross_fit=True`; "a cap inside the conditional
  support of the dose secures it". Update `cv-tmle.md:287`, re-execute LT, DR, and IV, and refresh
  the prose ledger. No test matches the LT phrase (LT-04 verifier); check pins for the others.

## P12. Single-draw claims stated as method properties

| location | claim | sweep | ledger |
| --- | --- | --- | --- |
| `docs/technical-reference/collaborative-tmle.md:154`, `src/cleverly/estimators/ctmle.py:222-224` | "includes the confounder in every seed and still leaves the instrument out" | three e2e seeds; test allows the instrument once | CT-03, CT-N3 |
| CT `control-reading` | "In this known law, the search keeps ... It drops the pure assignment predictor." | instrument left out on 92-93% | CT-03 |
| CT `selection-reading` et al. | the draw's selection read as the selector's behaviour | empty g on more than half of draws | CT-02 |
| CT `sensitivity-reading` | "On this draw, the plug-in value is smaller" (systematic) | inverse case: a systematic fact stated as a draw fact | CT-04 |
| IV `dose-fit-reading` | "about twice as wide" (cb `> 1.8`) | above 1.8 on 25-28% | IV-08 |
| MS `failure-reading` | "differs ... in the fourth decimal" (cb `< 0.005`) | holds on about half of draws | MS-01 |
| TW `sensitivity-reading` | "about four times the benchmark" (cb `3.5 < 0.05/cf_d < 4.5`) | 5/28 samples | TW-10 |
| TW `production-fit` | `assert abs(package_gap) < 0.1 * abs(targeting_shift)` | fails on 11/30 samples | TW-01 |
| LT `reports-reading` | lowest `auc` at node-2 censoring (cb asserts it) | node 1 on 19% | LT-07 |
| PT `sensitivity-reading` | `medication_burden` "mirror image" (cb `cf_y < 0.2 < 0.4 < cf_d`) | 7.5-13% of draws | PT-03 |
| CF `failure-mode-reading` | "less than a third" (cb `< cross_fitted.std_error / 3`) | about 0.7 with a calibrated g | CF-01 |
| SN `failure-reading` | cb `0.5 < distance < 1.5` | 28.8% | SN-06 |
| DR `assessment-reading` | "In most of the `gr1` fits, the data favor a nonlinear reduction" | holds on 81%; other families mostly linear | DR-10 |
| PT `failure-reading` | "about three times" | hedged "On this draw"; ratio above 2 on 58/60 | sound |

- **Fix shape:** per notebook and callback, plus two library/doc texts.
- **Recommended single fix:** apply one rule. A sentence that holds on fewer than 90% of swept draws
  says "On this draw" and gives a rounded share ("on about a quarter of draws"), or it quotes the
  printed numbers without a multiplier. A callback pin of a minority relation stays only when the
  prose is draw-tagged. Fix the `ctmle.py` docstring and the technical entry as CT-03 states.

## Other patterns

### P13. Protocol text or prose that contradicts the synthetic law

| location | contradiction | ledger |
| --- | --- | --- |
| IV `dose-identify` | outcome "standardized" (E[Y] = 3.395); "navigator contact units" with 5.1% negative | IV-04, IV-05 |
| CT, SN, MS `protocol` | "standardized to the baseline distribution" for a Gaussian score | P2 |
| DR `title` | "The recorded assignment rule has a squared term ..." with a main-effects model | DR-08 |
| CF `team-clusters-heading` | "the team law measures the same program on a coarser scale" | CF-04 |
| LS Steps 2 and 8 | one event process called "plan exit" and "readmission or death" | LS-NF1 |
| `docs/examples/index.md:20` | LS failure mode "coding death as censoring targets a controlled direct effect" | **new sibling of LS-02 / LS-NF2 (weak):** no CDE exists in this law; the fit targets a death-as-censoring functional that is a CDE only under exchangeability for death |

- **Recommended single fix:** add to the protocol-step rules in `example-notebooks.md`: "Check each
  protocol field and each unit name against the generator's structural equations." Then fix the
  rows above.

### P14. Library or test text that contradicts the code

| location | text | code | ledger |
| --- | --- | --- | --- |
| `methods.py:798-799` | "For an interval, use `strategy="oat"` or `TMLEMethod`" | `oat` reports no interval (`:764-767`) | CT-10 |
| `synthetic.py:1712` and `shifted()` docstring | "d(a, w) = min(a + delta, u)" | holds at own dose | IV-11, IV-N2 |
| `tests/unit/test_msm_projection_weights.py:8` | "the reported standard error is too small" | `msm.py:323` "direction is not universal" | MS-N1 |
| `ctmle.py:222-224` | "in every seed" | three seeds, `included <= 1` | CT-03 |
| `ctmle.py:224-225`, `collaborative-tmle.md:155-156` vs `tests/e2e/test_ctmle.py:734-735` | 0.017 / 0.696 vs 0.036 / 0.695 | not re-run | CT-N2 |
| `validation/drtmle.py:600` | "Theorem 1's estimator" for cross-fitted fits | P11 | DR-06 |
| DR `estimate` (notebook) | "The default reduced cross-fitting refuses fewer than three folds" | default `pooled` accepts two | DR-01 |

- **Recommended single fix:** one docstring-and-string pass with the P11 strings. Each docstring
  claim that cites a test names the test and its seed count.

### P15. "Correct parametric learners make the data-reuse condition plausible"

| location | quote | ledger |
| --- | --- | --- |
| LT `trust` | "Logistic and linear learners of the law's own form are what makes that condition plausible on this page." | LT-01 (wrong: outcome learners are not of the law's form) |
| MS `trust` | "Correctly specified parametric learners are what makes that condition plausible on this page." | **new sibling (weak):** true learners, wrong reason; finite dimension, not correctness, satisfies the Donsker-type condition |
| MS `estimate-arms-heading` | "correctly specified parametric learners need no cross-fitting" | same; MS-02 |

- **Recommended single fix:** "The learners are finite-dimensional regressions, which makes the
  data-reuse condition plausible. Whether they are correctly specified is a separate condition."

### P16. Diagnostics from a fitted model read as facts about the population

| location | issue | ledger |
| --- | --- | --- |
| DR `assessment` | support row from the doubted g reads as clean overlap | DR-09 |
| IV `dose-fit-reading` | "The capped policy raises no warning" next to warnings that are checks | IV-06 |
| CF `assessment-reading` | 22.9% ESS with no population reference | CF-03 |
| LT `reports-reading` | in-sample slope near 1 read as correct specification | LT-02 |
| CT, SN `assessment-reading` | say the in-sample report tests no slope | sound |

- **Recommended single fix:** each support or calibration reading names the fitted model it comes
  from and says what that model cannot see.

### P17. Machine-noise numbers in stored outputs

| location | printed | status |
| --- | --- | --- |
| TW `manual-estimators`, `ltmle-diagnostics` | 2.9606e-19 in prose; solver scores 3.66e-18 | `--check` fails (TW-04, TW-05) |
| TW `diagnostics` | score 3.791e-13 | passed this run; same exposure |
| DR `assessment` | `D*_g` 9.312e-11, -2.482e-12 | `--check` exit 0 here; latent BLAS exposure |
| IV `incremental-fit` | fluctuation 7.179e-13, 2.934e-11 | `--check` exit 0 here; latent exposure |

- **Recommended single fix:** print the score check as pass/fail and the rounded ratio to the
  threshold. That is the library's own `worst abs(score) / threshold` line. Do not print raw
  sub-1e-8 residuals.

### P18. Refusal-inventory replacements that conflict with PT-N1

`refusal-inventory.md` replaces the PT and CF sensitivity refusals with a refit on a calibrated
propensity, then reads rv, rva, the benchmarks, and the one-sided limits. On `navigation_data`
nu_0^2 is infinite (PT-N1). The DR nu^2 of a smooth g then estimates the second moment of the law
without the `W2*W3` tail. The PT verifier measured 4.6 to 5.6, against 4.79 for the law with
`medication_burden` dropped. Publishing DR limits there puts the page outside the regularity of
Lemma 3 and Theorem 4, which would be a new error (PT-02 verifier).

- **Recommended single fix:** before the plan adopts the refit, either (a) state PT-N1 in the
  sensitivity step and present rv and rva as conditional on the truncated law, without one-sided
  limits, or (b) move the PT and CF sensitivity step onto a law with strong positivity. Do not let
  the refit's passing DR nu^2 read as evidence that the bound is valid on this law.

# Review: `docs/examples/longitudinal-tmle.ipynb`

| item | value |
| --- | --- |
| notebook | `docs/examples/longitudinal-tmle.ipynb` (callback `tests/unit/tutorial_semantics/longitudinal_tmle.py`) |
| commit | `5f33902b` (branch `agent/notebook-review`), `cleverly.__file__` under this worktree's `src`, Python 3.11.13 |
| `--check` | exit 0, wall 9 s; "every cell reproduced its stored non-image output" (`.tmp/notebook-review/longitudinal-tmle/check.log`) |
| independent truth | `.tmp/notebook-review/longitudinal-tmle/truth_and_mechanism.py` (re-implements the structural equations; does not import `cleverly`) -> `truth_and_mechanism.log` |
| claim probes | `.tmp/notebook-review/longitudinal-tmle/probes.py` -> `probes.log` |
| seed sweep | `.tmp/notebook-review/longitudinal-tmle/sweep.py` (the notebook's exact code path, n=8000, cluster_size=20, seeds 1000-1399) -> `sweep.csv`; summary `summarize.py` -> `summary.log` |

Severity counts: wrong 1, misleading 3, weak 4.

## Findings

### LT-01  `trust`: "Logistic and linear learners of the law's own form are what makes that condition plausible on this page."

- **Problem.** The outcome-side learners are not of the law's form. `_outcome_probability` adds
  `0.5 * tanh(L2)` to the logit (`src/cleverly/datasets/longitudinal.py` `_Y["kink"]` and the
  docstring "so a `glm` nuisance learner here is misspecified rather than accidentally exact").
  The node-1 pseudo-outcome regression targets `E[expit(...) | W, A1]`, which is not linear in `W`,
  yet the page fits it with `LinearRegression`. Only the four treatment and censoring mechanisms
  (`expit(0.3W1-0.4W2)`, `expit(2.2+0.3W1-0.3A1)`, `expit(0.5L2+0.6A1-0.2W2)`, `expit(2.4+0.2L2)`)
  are of the fitted main-terms logistic form. The sentence also gives the wrong reason: the
  data-reuse (Donsker) condition holds because the learners are finite-dimensional parametric
  fits, whether or not they are correct. Correctness of form is a separate condition, and here it
  holds for the mechanisms only, so consistency rests on double robustness through `g`.
- **Severity.** wrong.
- **Evidence.** [read] `src/cleverly/datasets/longitudinal.py` (the `_Y` dict and
  `_outcome_probability` docstring). [read] `tests/studies/canonical_ltmle.py:160` uses the same
  misspecified main-term `Q` formulas for the registered in-sample study.
  [executed] sweep: bias of the always-vs-never estimate over 400 seeds is near zero (see the
  checked list), which is what correct `g` with misspecified `Q` predicts.
- **Proposed fix.** "The logistic and linear learners are finite-dimensional, which makes the
  data-reuse condition plausible. The treatment and censoring learners match the law's form. The
  outcome regressions do not, because the law has a `tanh` term in engagement. The estimate
  therefore relies on the correct mechanisms (double robustness)." Files: the notebook `trust`
  cell, and `tests/prose-report.md` refresh. Library bug: no. Registered study moves: no.

### LT-02  `reports-reading`: "Each model is measured on the rows it was fitted on, and a maximum-likelihood fit of the correct form reproduces those rows by construction."

- **Problem.** The in-sample calibration slope of a logistic maximum-likelihood fit with an
  intercept equals 1 for any linear predictor, correct or not. The score equations
  `sum (Y - p) = 0` and `sum (Y - p) X beta = 0` are exactly the calibration fit at intercept 0 and
  slope 1. "Of the correct form" is not a condition. Together with LT-01 the sentence tells the
  reader that the models here are correct, and the outcome model at node 2 (slope 1.003) is not.
- **Severity.** misleading.
- **Evidence.** [recomputed] `probes.log`: on the seed-41 frame a deliberately wrong outcome
  model (`Y ~ W1 + W2`, no treatment, no engagement) gives in-sample calibration slope 1.0009 with
  sklearn's default penalty and 0.9999 unpenalized. The notebook-form model gives 1.0009. The
  small departures from 1 in the stored table come from sklearn's default L2 penalty (`C=1`).
- **Proposed fix.** "A logistic fit with an intercept reproduces its own rows by construction, so
  an in-sample slope near 1 holds whether or not the model is correct." Files: notebook
  `reports-reading`. Library bug: no. Study moves: no.

### LT-03  `trust`: "The [cross-fitted end-of-study study] is the closest. It supplies the mechanisms rather than estimating them, and it has no clusters."

- **Problem.** The page fits in sample (`CrossFitting(enabled=False)`). The registered study of
  that construction is the
  [ordinary end-of-study study](../../../docs/technical-reference/method-evidence/ordinary-end-of-study-longitudinal-tmle.md).
  It draws `make_longitudinal` (the page's law, unclustered), fits the same three plans including
  the same `L2 > 0` rule, uses the same misspecified main-term `Q` formulas, and is non-cross-fitted
  against R `ltmle`. The cross-fitted study is a different estimator (pooled fluctuation over
  out-of-fold predictions) and is the farther of the two. The ordinary study also carries the
  property cell that matches this page best: `static__mechanism_correct` and
  `dynamic__mechanism_correct` (double robustness with only the mechanisms correct).
- **Severity.** misleading.
- **Evidence.** [read] `tests/studies/canonical_ltmle.py:20,160,217` (`make_longitudinal`,
  `q_formulas`), `ordinary-end-of-study-longitudinal-tmle.md` accuracy table
  (`ate_regimen[always vs never]` coverage 0.9413, SE ratio 0.9890, pass; rule-vs-never 0.9387,
  pass) and its limitations row "excludes ... clustering ... cross-fitting". [read]
  `cross-fitted-end-of-study-longitudinal-tmle.md` lines 1-40.
- **Proposed fix.** Cite the ordinary study as the closest: same law, same plans, in sample, no
  clusters, mechanisms supplied in the accuracy cells. Name the `mechanism_correct` property cells.
  Keep the cross-fitted study as a second link if wanted. Files: notebook `trust`. Library bug: no.
  Study moves: no.

### LT-04  `estimate-heading`: "`cleverly` refuses that composition, and it names the in-sample fit, which is clustered and evidenced."

- **Problem.** No registered study covers a clustered longitudinal fit, in sample or
  cross-fitted. The `trust` cell of the same page says so ("No registered study covers clustered
  fits with estimated mechanisms"), and the ordinary study lists clustering as excluded. The only
  evidence for the in-sample clustered interval is an e2e test that the clustered standard error
  is larger than the independent one (`tests/e2e/test_ltmle.py:654-670`), plus status tests. That
  is a consistency check, not coverage evidence. The word comes from the library's refusal
  message (`src/cleverly/longitudinal/estimator.py:2555`), and `docs/technical-reference/cv-tmle.md:287`
  quotes it.
- **Severity.** misleading.
- **Evidence.** [read] the files above. [executed] my sweep happens to support the in-sample
  clustered interval on this law (coverage 0.935, 374/400 seeds), but that is a review probe, not
  registered evidence.
- **Proposed fix.** Notebook: "it names the in-sample fit, which reports a cluster-robust
  variance. An e2e test checks that variance; no registered study covers its coverage." Library
  message: replace "which is clustered and evidenced" with "which reports a cluster-robust
  variance". Files: notebook, `src/cleverly/longitudinal/estimator.py:2555`,
  `docs/technical-reference/cv-tmle.md:287`, and any test that matches the message text. Library
  bug: wording only. Study moves: no.

### LT-05  `protocol-reading` / `data-reading`: the composite death rule versus "Patients lost from outcome tracking are censored."

- **Problem.** The protocol scores death before day 30 as not top box. The shared design adds that
  the composite score "counts as observed. Only living patients who do not respond are missing"
  (`docs/examples/index.md:42-43`). The notebook never tells the analyst that a patient who dies must
  enter the frame as `tracked_day30 = 1, transition_top_box = 0`, not as lost to tracking. If deaths
  were left as untracked, the `censoring=` role would treat them as missing at random and the fit
  would target a hypothetical "no death" estimand, not the protocol's composite. The synthetic law
  has no deaths, so the page cannot show the coding, but the reading of "loss to tracking ... is not
  an intercurrent event" holds only under that coding. The sibling `longitudinal-survival` page's
  whole failure mode is this confusion.
- **Severity.** weak.
- **Evidence.** [read] `src/cleverly/datasets/longitudinal.py` `make_longitudinal` (no death node);
  notebook `protocol` output and `protocol-reading` table row "loss to tracking".
- **Proposed fix.** One sentence in the `protocol-reading` table row: "A death before day 30 is
  recorded as tracked with `transition_top_box = 0`. Only living patients lost from follow-up are
  censored." Files: notebook. Library bug: no. Study moves: no.

### LT-06  `association-reading`: "Among engaged patients, a share of 0.705 received day-seven navigation. Among the others, the share is 0.427. Engagement therefore responds to the first decision and drives the second."

- **Problem.** The printed share is pooled over discharge navigation. Discharge navigation raises
  both engagement (+0.9) and day-seven navigation (+0.6 on the logit), so part of the 0.278 gap is
  discharge navigation, not engagement. The conclusion is true in the law, but this display does
  not isolate it.
- **Severity.** weak.
- **Evidence.** [recomputed] `probes.log`, seed 41: within `A1 = 0` the share is 0.381 vs 0.593;
  within `A1 = 1` it is 0.557 vs 0.770 (gap 0.21 in each stratum). The share with discharge
  navigation is 0.633 among engaged patients and 0.26 among the others.
- **Proposed fix.** Group by `["navigation_discharge", engaged]` and read the within-stratum gap.
  Files: notebook `association` and `association-reading`, callback `shares` assertion. Library
  bug: no. Study moves: no.

### LT-07  `reports-reading`: "The lowest value is 0.557, for the censoring model at node 2, so that model barely separates the patients who stay tracked."

- **Problem.** Which censoring model has the lowest `auc` is a property of the draw. Over the seed
  sweep the node-1 censoring model is lowest in 77 of 400 seeds (19%),
  and the reading explains only node 2. The oracle `auc` of the true node-2 tracking probability is
  0.573, and of the node-1 probability 0.588, so both are expected to sit near 0.5 to 0.6.
- **Severity.** weak (true for the seed shown; the explanation is correct in the law).
- **Evidence.** [executed] sweep `lowest_auc_role`; [recomputed] `probes.log` oracle AUCs (2e6 draws).
- **Proposed fix.** "Both censoring models have an `auc` near 0.55 to 0.6. In `make_longitudinal`
  each tracking probability depends on few variables through small coefficients, and the true
  probabilities give an `auc` of about 0.57 at node 2 and 0.59 at node 1." Files: notebook,
  callback assertion `lowest["role"], lowest["time"]`. Library bug: no. Study moves: no.

### LT-08  callback `tests/unit/tutorial_semantics/longitudinal_tmle.py`: `# ... the largest weight (measured 33.3) is well under the cap of 100` and `assert (support["max_weight"] < 60.0).all()`

- **Problem.** The stored largest weight is 35.239, not 33.3 (stale comment). Over seeds the
  largest weight reaches 97.5 (median 27.7), so "well under the cap" is a seed-41 relation; the
  page's own wording ("On this draw the bound replaces no row") is correctly hedged. The
  truncation share at the default bound was 0 in all 400 seeds.
- **Severity.** weak.
- **Evidence.** [executed] stored `assessment` output; sweep `max_weight` maximum.
- **Proposed fix.** Update the comment to 35.2. Files: callback. Library bug: no. Study moves: no.

## Checked and sound

- [recomputed] Truths. Independent Rao-Blackwellized Monte Carlo over 1e7 draws of the structural
  equations: `E[Y_always]` 0.78043 (MC se 5e-5), `E[Y_never]` 0.41888, early 0.64408, late 0.58115,
  rule 0.74002; always-never 0.36154, rule-never 0.32114, rule-always -0.04041. All printed truths
  (0.7804, 0.4189, 0.6441, 0.5811, 0.3616, 0.7400, 0.3212, -0.040) agree to 4 decimals. A direct
  Bernoulli counterfactual simulation agrees within MC error. The cluster construction keeps the
  `L2` noise marginal N(0,1), so the superpopulation truth is unchanged.
- [recomputed] Failure-mode mechanism (true conditional means, no learner error, 4e5 draws).
  Adjusting for engagement has limit 0.2376 (bias -0.124); with the `A1 -> L2` effect removed the
  bias is -0.002, so the bias is the blocked path. Restricting the same contrast to the whole
  population instead of the agreement subset gives 0.2448, so selection on agreement adds only
  about 0.007. Baseline-only has limit 0.4061 (bias +0.045); with the `L2 -> A2` arrow removed the
  bias is -0.0035, so the bias is confounding of day-seven navigation by engagement. The table's
  "structural problem" column is correct, and the hedge about selection is honest.
- [executed] Sweep over 400 seeds (n=8000, 400 teams; `summary.log`): adjusted-for-engagement
  estimate below truth in 400/400 (mean -0.125, range -0.189 to -0.050; CI excludes truth 399/400);
  baseline-only above truth in 395/400 (mean +0.044, range -0.007 to +0.088; CI excludes truth
  274/400); opposite directions 395/400. The crude difference exceeds the truth in 400/400 (mean
  +0.119). The page's "on this draw ... sizes not general" hedge is appropriate; the directions are
  general.
- [executed] Sequential always-vs-never: mean bias -0.0005 (MC se 0.0009); clustered-CI coverage
  0.935 (374/400, MC se about 0.012); empirical SD 0.0182 vs mean clustered SE 0.0176; an iid SE
  (0.0157) would cover 0.905. The cluster-robust variance is needed and roughly calibrated here.
  "Within one standard error" (seed 41) holds in 278/400 (69.5%), as expected; the page presents it
  as a draw relation.
- [executed] Rule-vs-never coverage 0.950 (380/400), bias -0.0006; rule-vs-always coverage 0.963,
  interval excludes zero 399/400, so "the rule gives up some top-box share" holds beyond the draw.
  Day-seven `share_assigned_1` under the rule ranges 0.759 to 0.853.
- [executed] `share_truncated` is 0 at the default bound in 400/400; scores pass 400/400; truncation
  curve moves the estimate under 0.3 SE in 394/400 (max 0.62 SE); min Kish ratio 0.653 to median
  0.803; in-sample calibration slopes 0.9935 to 1.0081.
- [executed] Engagement gap by discharge navigation exceeds 0.5 in 400/400, and engaged patients
  have a higher day-seven share in 400/400.
- [executed] `cross-fit + cluster` is refused with `LongitudinalError`; a cross-fitted fit without
  `cluster` reports `solver` score rows only and passes ("A cross-fitted fit also reports `solver`
  rows only").
- [read] Estimand matches the question (static always vs never at two nodes, binary top-box, 30-day
  horizon) and the protocol; the rule is declared as a third strategy before any fit.
- [read] Sequential exchangeability and sequential positivity are named; exchangeability,
  consistency and no interference are marked untestable; positivity "partly" testable via the
  support report. Correct.
- [read] Sensitivity cell: every longitudinal sensitivity operation is printed as `unavailable` with
  its reason, and the reading does not interpret any number. Six "derivation is registered" reasons
  counted correctly. Methods list matches.
- [read] `g_bounds (0.01, 1)` caps each weight at 100, and is R `ltmle`'s default `gbounds`.
- [read] Kish effective-n ratio 1380.454/1736 = 0.795 matches the "79.5%" summary; 11175 =
  3478+2362+3599+1736.
- [read] Literature: Hernán and Robins, *What If*, chapter 20 is "Treatment-confounder feedback",
  and sections 20.2 and 20.3 cover the bias of traditional methods, including the collider path
  through an unmeasured cause of `L`. The notebook defers Bang and Robins (2005) and van der Laan and
  Gruber (2012) to the technical reference (`longitudinal-tmle.md:70`), which names them.
- [executed] `--check` reproduces every stored output; all seeds are set (generator 41,
  `Runtime(random_state=41)`, `LogisticRegression(random_state=41)`; `LinearRegression` is
  deterministic).
- [read] Practicality: a plan that offers a second contact based on recorded engagement, with loss
  to follow-up and navigator teams, is a realistic regional-health-plan question; reporting a
  resource-saving rule beside the full plan, with a joint-IC contrast, is the right decision output.

## Patterns for siblings

- A reading that says a nuisance learner is "of the law's own form" or "correct" must be checked
  against the generator. Several generators add a deliberate misspecification term (`tanh` here).
- An in-sample calibration slope near 1 is guaranteed for any logistic MLE with an intercept; any
  reading that ties it to correct form is wrong.
- The "How far to trust this" cell should cite the study of the construction actually fitted
  (in-sample vs cross-fitted). Check the study's `tests/studies/*.py` for law, plans and formulas.
- The library's refusal message calls the in-sample clustered longitudinal fit "evidenced"; any
  page that echoes it overclaims.
- A protocol with a composite death rule needs one sentence on how death is coded in the frame,
  or censoring silently changes the estimand.
- Pooled association tables used to argue "X drives the second decision" should stratify on the
  earlier decision.

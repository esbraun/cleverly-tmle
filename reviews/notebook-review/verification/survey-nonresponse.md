# Verification: survey-nonresponse

Verifier run at `5f33902b`, worktree `.venv` (Python 3.11), `cleverly.__file__` asserted under this
worktree's `src`. Scratch: `.tmp/notebook-review/verify-survey-nonresponse/` (`nb.txt` cell dump,
`death.py`, `tg.py` + `tg_seeds.log` + `tg_n.log`, `useci.py`). All runs single-threaded.

Summary: 11 findings. CONFIRMED 8 (SN-01, SN-03, SN-04, SN-05, SN-06, SN-07, SN-09, SN-10),
PARTIAL 3 (SN-02, SN-08, SN-11), REFUTED 0. Two new findings (N1, N2).

---

### SN-01: composite death rule vs declared MAR

verdict: CONFIRMED. severity: wrong (the applied workflow; the stored numbers are unaffected).

- Notebook text [read]: `title` says "The protocol scores death before day 30 as the worst
  transition score (composite strategy). That score counts as observed, so only living patients
  who do not respond are missing." `protocol` adds "Given the arm and the baseline variables, survey
  response carries no further information about the unobserved score (missingness at random)". The
  identify summary prints "transition_score is independent of responded given
  (transition_navigation, W)". `docs/examples/index.md` "unreturned surveys" row repeats the
  composite rule. `navigation.py:181` carries the death clause into this protocol.
- Synthetic law has no deaths [read]: `missing_outcome_dgp` (`synthetic.py:966-1018`) has only
  propensity, a Gaussian outcome mean, and a response mechanism; `_make` (`synthetic.py:312-346`)
  draws Y Gaussian and Delta from `missingness` only. No death node exists. The numbers on the page
  are therefore not biased by this.
- Why the stated assumption is wrong, not only incomplete. Let D = death before day 30 (assumed
  ascertained for everyone), R = survey response among survivors, Delta = D + (1-D)R, Y = y_worst if
  D = 1. Then P(Delta=1 | Y=y_worst, A, W) = 1, and P(Delta=1 | survivor Y, A, W) = P(R=1 | A, W, D=0).
  Y independent of Delta given (A, W) therefore fails whenever 0 < P(D=1|A,W) and P(R=1|A,W,D=0) < 1.
  It is false as written, not just underspecified. The shipped functional is
  E_W[ (p_D y_worst + (1-p_D) pi_s mu_s) / (p_D + (1-p_D) pi_s) ], which over-weights deaths.
- Assumption the composite actually needs: MAR among survivors, Y independent of R given
  (A, W, D=0), with vital status observed for all and P(R=1 | A=a, W, D=0) > 0. Identification:
  E[Y^a] = E_W[ P(D=1|A=a,W) y_worst + P(D=0|A=a,W) E(Y | A=a, W, D=0, R=1) ]
  = E_W E[ E(Y | A=a, W, D, Delta=1) | A=a, W ]. This is a sequential regression on a
  post-assignment variable D, the same structure SN-04 needs for contact attempts. The shipped
  point-treatment estimator cannot compute it.
- Numerical witness [recomputed] (`death.py`, 4e6 covariate draws, closed-form arm means, the
  reviewer's death model expit(-3.5 + 0.8 W1 - 0.5 a), worst = -7): true ATE 1.3324, shipped
  functional 1.5866, bias +0.2542; arm biases -0.468 (control) and -0.214 (navigation); mortality
  3.9 % / 2.4 %. Identical to the reviewer's `sens.log` part 3.
- Fix: the reviewer's fix is right. Preferred wording: state that the shipped estimator assumes MAR
  for the composite score, that deaths are always observed so this fails when anyone dies before
  day 30, and that the page therefore assumes no deaths before day 30 (the synthetic law has none).
  Name MAR-among-survivors and its two-part formula as the assumption a real program needs, and
  list it under "How far to trust this". A roadmap item for a sequential (death, then response)
  estimator is appropriate. Also fix the `index.md` "unreturned surveys" row, which pairs the
  composite rule with an implicit single-indicator analysis. No other notebook combines
  `navigation_protocol()` with `missingness=` (grep). No registered study moves.

### SN-02: tipping gamma scale and seed dependence

verdict: PARTIAL. severity: weak on the page (misleading in the technical reference, which is
silent on it).

- Code [read]: the tilt acts on the scaled Q* (`missingness.py:428-485`, `_tilted`). The scaler is
  `OutcomeScaler.from_outcome` (`utils/bounds.py:199-235`): with `q_bounds=None` it is the observed
  outcome range padded by 10 % on each side, so it is sample-dependent. With `q_bounds` declared it
  is the declared support and fixed. The reviewer's mechanism is right.
- [executed] (`tg.py`, own script calling `tipping_gamma` directly): seed 71 reproduces 1.3059,
  range 20.05. 60 seeds (5000-5059), n = 4000: mean 1.182, median 1.167, 5-95 % 0.975-1.442, range
  0.915-1.458, 23.3 % at or above 1.306 (reviewer: median 1.170, 21.6 %). Correlation of tipping
  gamma with the scaler range -0.45. Mean over 8 seeds per n: 1.235 (n=1000), 1.221 (4000), 1.004
  (16000), 0.884 (64000), while mean range grows 19.6 -> 30.7. Drift with n is confirmed.
- Why only partial: the page does not present gamma alone. `sensitivity-reading` converts it at
  once: "moves by at most 0.315 of the score range. That is 6.32 score units". That max-shift in
  score units is far less n-dependent (about 5.9 units at n = 1000 to 6.7 at n = 64000 from the
  means above), so the closing "shift of that size" judgment is mostly anchored in score units. The
  page still never says that gamma is defined on a sample scale, and gives no uncertainty for it.
- Fix: add one sentence that gamma is read on this fit's sample-derived [0, 1] scale, so it is not
  comparable across samples or studies, and judge plausibility in score units. In the technical
  reference, state that `q_bounds` fixes the tilt scale. The reviewer's "make the tilt scale a
  declared q_bounds-like support" is already the library behaviour: declaring `q_bounds` does it
  (`tmle.py:1590`). No library change is needed.

### SN-03: "a standardized score has no finite support"

verdict: CONFIRMED. severity: misleading.

- Text [read]: `protocol-reading` "A standardized score takes its scale from the data, so the
  analyst can declare no finite support for it"; `estimate-heading` and `refusal-reading` repeat it.
- An affine standardization of a bounded instrument is bounded. The shared protocol's own outcome
  is "Patient-reported transition score, as a share of the maximum score" (`navigation.py:177`),
  a [0, 1] support; this page replaces that text to make the score unbounded. The real reason is the
  synthetic Gaussian outcome (`synthetic.py:335`). The cross-fit refusal asks for "q_bounds equal
  to the known outcome support" (`tmle.py:1873-1877`), which a bounded instrument supplies.
- One nuance for the fix: if the standardizing constants are estimated from the same sample, the
  transformed bounds are data-dependent. The clean advice is to fit on the instrument's own scale
  with `q_bounds` equal to its range.
- Same sentence in `collaborative-tmle.ipynb`, `interventions.ipynb`, `msm-projections.ipynb` (grep).
- Fix: reviewer's fix is right; add the instrument-scale nuance. Notebook only.

### SN-04: post-assignment contact attempts

verdict: CONFIRMED. severity: misleading.

- Text [read]: `title` "Contact attempts after assignment cannot enter it, because navigation can
  change them"; `assumption_rationale` "Contact attempts after assignment stay out of the response
  model, because navigation can change them".
- Being affected by A disqualifies L as a confounder of A, not as a conditioning variable for
  response. Under no unmeasured confounding given W and Y independent of Delta given (A, W, L),
  E[Y^a] = E_W E[ E(Y | A=a, W, L, Delta=1) | A=a, W ], the standard sequential (g-computation)
  formula. MAR given (A, W, L) is often more plausible than MAR given (A, W). The stated reason
  ("navigation can change them") is a non sequitur; the true constraint is that this single-time
  point estimator has one response model on (A, W) (`tmle.py`, `point-treatment-tmle.md:97-106`).
- Fix: reviewer's rewrite is right. Add that the same sequential structure is what SN-01's
  composite outcome needs (D in place of L), and list the limitation in "How far to trust this".

### SN-05: complete-case target wording

verdict: CONFIRMED. severity: weak. [read] The code defines 1.409 as the mean CATE over this draw's
respondents (`failure-mode` cell; callback lines 140-145), which is the sample analogue of
E[CATE(W) | Delta=1]. "Even correct nuisance models would target 1.409" mixes the sample value
with the population target 1.428 (reviewer `truth.log`; I did not rerun). Fix is right.

### SN-06: one-SE distance attributed to possible nuisance error

verdict: CONFIRMED. severity: weak. [recomputed from the reviewer's CSV, checked by my own formula]
`dist_se` mean 0.158 over 500 seeds; 28.8 % lie in (0.5, 1.5). The callback pins
`0.5 < distance < 1.5` (`survey_nonresponse.py:150-151`), a seed-71 relation. House rule
`example-notebooks.md:161` asks for a probe before naming another cause. Fix is right; keep "on
this draw".

### SN-07: coverage below nominal on this page's law

verdict: CONFIRMED. severity: weak. [recomputed] From the reviewer's 500-seed CSV, coverage
recomputed as |psi - 1.2| <= 1.96 se gives 0.920 (stored flag agrees); bias -0.0045, empirical SD
0.0793 vs mean SE 0.0741. Binomial 95 % interval about 0.894-0.943 excludes 0.95. Fix (one trust
row) is right; phrase it as "a 500-seed probe, not a registered study".

### SN-08: score-unit conversion and the CI-based tipping point

verdict: PARTIAL. severity: weak. [executed] `useci.py`: `tipping_gamma(..., use_ci=True)` = 1.1302
on seed 71; observed score SD 2.128, navigation arm 1.659. Matches the reviewer. "Overstates" is
not right: the page says "at most 0.315 of the score range", correctly framed as a maximum. The
missing SD anchor and the missing `use_ci=True` value are real. Fix: print the `use_ci=True` value
and the SD ratio; the weighted-average shift is optional.

### SN-09: tilt curve away from gamma = 0 is a plug-in

verdict: CONFIRMED. severity: weak. [read] `_tilted_psi` reuses the gamma = 0 targeted Q*
(`missingness.py:452`, `fluctuations[...].targeted`); no re-targeting per gamma. The bias size is
the reviewer's 40-seed number (not rerun). Fix is right.

### SN-10: omitted-variable reason contradicts the ledger

verdict: CONFIRMED. severity: weak. [read] `omitted_variable.py:253-262` (`_RESPONSE_BOUND_REFUSAL`):
"the omitted-variable bound is not implemented for a fit with a response mechanism ... Three pieces
are missing", i.e. an implementation gap, while the page says "no derivation here covers a fit that
also models response". House rule (`example-notebooks.md`, refusal row) asks for "not implemented".
Fix is right.

### SN-11: attributable-fraction refusal framing

verdict: PARTIAL. severity: weak. [read] The page sentence ("The 2026-09-12 source audit found no
published derivation of that construction") is true per `roadmap.md:229-241`. The identification
point is right: E[Y] and E[Y^0] are both identified under MAR given (A, W), and the page itself says
`NaturalCourseMean()` is supported. The larger defect is the library refusal string, which the page
prints: "docs/roadmap.md F20 tracks this identification boundary". It is an evidence-policy
boundary, not an identification one. Fix: reviewer's notebook wording, plus change "identification
boundary" in the refusal message (library string; check tests that pin it).

---

## New findings

- N1 (misleading): the composite rule and the "no finite support" claim contradict each other.
  "The worst transition score" presupposes a finite minimum, which the page's
  `protocol-reading` says the standardized score lacks. The base protocol's bounded outcome
  ("as a share of the maximum score", `navigation.py:177`) is replaced here, which creates the
  conflict. Fixing SN-03 (bounded instrument, `q_bounds`) also resolves this.
- N2 (weak): SN-01 and SN-04 have one root. Both need a sequential regression over a
  post-assignment variable (D, or contact attempts). One "How far to trust this" row can state the
  limitation for both, rather than two separate caveats.

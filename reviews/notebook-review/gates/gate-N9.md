# Gate N9: `docs/examples/msm-projections.ipynb` at 6927899e

Reviewed every cell and stored output, the callback
`tests/unit/tutorial_semantics/msm_projections.py`, the probe directory
`reviews/notebook-review/probes/msm-projections-final/` (`sweep.py`, `sweep.csv`, `sweep.log`,
`truth.py`, `truth.log`), ledger rows MS-01 to MS-07 and MS-N1 and the refuted-part row for
MS-02/MS-06, plan row N9, `docs/development/example-notebooks.md` (rules 69-73, 174, 176, 211,
the trust step), `docs/technical-reference/msm-projections.md`,
`docs/technical-reference/method-evidence/point-treatment-msm-projection.md`,
`tests/canonical/tmle3_msm/{summary,properties}.csv`, `tests/studies/canonical_point_msm.py`,
`src/cleverly/datasets/synthetic.py:1639-1700` (`multi_arm_dgp`),
`src/cleverly/datasets/navigation.py:180-190`, `src/cleverly/sensitivity/positivity.py`
(`_verdict_parts`), `src/cleverly/estimators/tmle.py:1976`, gates N5 and N6 (death wording).
Scratch: `.tmp/notebook-review/gate-N9/` (not tracked). Light probes only: one refit of the
shown draw for the influence-curve kurtosis, and recomputations from `sweep.csv`.

## Verdict

PASS with one wording fix. Every printed number reproduces (`--check` exit 0), every
repeated-draw number recomputes from `sweep.csv`, every truth in `truth.py` matches the structural
equations of `multi_arm_dgp` and the closed-form projections, the two new designs (Step 8 cadence
step, Step 10 fixed 1:10:1 weight) are correctly read, the protocol matches the collaborative and
survey pages, and no cell narrates a refusal. The one defect is the stated cause of the one-point
coverage shortfall: the direction is right, but the number it names (a true probability near
0.0001) never enters the influence curve, and the mechanism it names (heavy tails "at this sample
size") is not what the probe shows.

## REQUIRED FIXES

1. Trust cell (`trust`, cell 41), the cause paragraph: "Some patients have a true chance of `high`
   or `medium` near 0.0001, so the influence curve has heavy tails at this sample size." Two
   problems.
   - The fit truncates g at [0.0114, 0.9886] (Step 7 summary; the same bound applies in the
     probe's true-g run). A true probability of 0.0001 is clipped to 0.0114 before it reaches
     the clever covariate, and `truth.log` puts the share below 0.001 at 0.00006, about 0.2
     patients per draw of 3,000. The heavy tail of the influence curve comes from the roughly 5%
     of rows per arm with g below 0.05 (support quantiles 0.0478 and 0.0470), where the clever
     covariate reaches 6 / 0.02 = 268.5.
   - The shortfall is a standard-error scale, not a non-normal sampling distribution. From
     `sweep.csv`: the z-statistic (slope minus 0.265714) / SE has excess kurtosis 0.05 over 4,000
     draws, and a normal interval with the recorded SE / SD ratio 0.984 covers 0.946, inside the
     Wilson interval 0.933 to 0.947. Coverage is flat across quartiles of the smallest fitted g
     (0.937 to 0.945), and misses split evenly above and below the median smallest g (0.504).
     The influence curve itself is heavy-tailed on this law: excess kurtosis 29.2 on the shown
     draw (`.tmp/notebook-review/gate-N9/`), which is why the plug-in SE understates the spread
     at n = 3,000.
   Write, in substance: "The slope's influence curve has heavy tails on this law. About one
   patient in a hundred has a true probability of `high` or `medium` below 0.01 (`truth.py`),
   and the clever covariate divides the contact count by that probability. The reported standard
   error therefore understates the empirical spread, at 0.984 of it, and a normal interval with
   that ratio covers 0.946. The same ratio, 0.986, occurs with the true treatment mechanism, so
   the fitted model does not cause it." Then update `UNPRINTED_DECIMALS` in the callback: retire
   or re-source the `"0.0001"` entry, and add `"0.946"` with its derivation (2 Phi(1.96 x 0.984)
   - 1 from the `sweep.log` SE / SD ratio). `"0.01"` is already listed for the study bounds; if
   the text prints the `truth.log` shares below 0.01 (0.0055, 0.0057), list them too. Markdown
   and callback only; no stored output changes.

## Advisory, not gating

- Step 8 reading (cell 25): under the uniform weight the per-step slope with coding 0, 1, 2 is
  exactly (E[Y(high)] - E[Y(low)]) / 2. The fit shows it: 1.4681 / 2 = 0.73405, printed 0.7341.
  One sentence saying so would give the per-step row the same "fixed contrast" reading that
  Step 7 gives the per-contact row, and would make clear why the per-step interval (0.6902,
  0.7779) is the `high vs low` interval halved. Optional.
- `docs/examples/index.md:44`, "Only living patients can be missing or censored", is false for
  every page that uses the hypothetical strategy, now three. Shared with N5 and N6; not this
  page's cell.
- "at this sample size" in the trust cell rests on the MS-02 ledger recompute (0.947 at
  n = 12,000), which no committed probe records. The sentence is defensible without the clause.

## OK

- Projection estimand. The technical entry's beta = argmin E sum_a h(a,V) (Q(a,W) - phi'b)^2
  with h = 1 is what the page describes. The uniform-weight slope on contacts 1, 2, 6 is
  (-2, -1, 3) / 14 (deviations from the mean 3, sum of squares 14): 0.265714 in `truth.log`, and
  the Step 9 output prints 0.2657 for both the least-squares line and the contrast. The claim that
  stabilized IPW uses h(a) = P(A = a) is the Neugebauer and van der Laan (2007) projection weight;
  the page does not cite the paper, and the technical entry's citations are the published ones.
- Truths spot-checked against `synthetic.py:1675-1687`: step (0, 1, 2.4) x 0.6 gives 0, 0.6,
  1.44 with mean-zero covariate terms; arm logits (0, 0.8 W1 - 0.4 W2, -0.5 W1 + 0.8 W2) match
  `truth.py` `COEF`. 1:10:1 weight: weighted mean of contacts 2.25, Sxx 16.25, Sxy 3.9, slope
  0.2400. Per-step: slope (1.44 - 0) / 2 = 0.72, misses (-0.04, 0.08, -0.04). Closed-form
  E[1/g] by the log-normal moment formula (4.05, 7.27, 7.34) agrees with the 2e6-row Monte Carlo
  (4.052, 7.275, 7.357).
- Step 8 replaces the `MSM.linear` label refusal with a second known design. The reading names
  what each coding counts (one step versus four contacts for `medium` to `high`), states that both
  slopes are well defined and neither line fits, and says which the board asked for. The callback
  pins the coding, the 0.7200 target, coverage of each slope by its own target and not the other
  coding's, and the misfit of the step line. Both intervals contain their population slope on this
  draw, as stated.
- Step 10. Shift 0.0065 against SE 0.0086, called "less than one standard error"; probe share
  3,984 of 4,000 (`sweep.log` 0.9960). The 1:10:1 fit is declared `weights_kind="known"` and the
  text says why that is true (the weight is the same for every sample). "Each interval contains
  its own population slope and not the other" is qualified "on this draw"; the probe shows the
  exclusion holds on 0.80 and 0.72 of draws, so the qualifier is load-bearing and present. MS-03
  wording ("omits the weight's influence-curve term. It can be too small or too large") matches
  `msm.py` and the already-corrected docstring of `test_msm_projection_weights.py` (MS-N1).
- Step 9 (MS-01): "differs from the Step 7 miss by 0.0008 on this draw, because the arm fit and
  the MSM fit target separately"; stored output prints 0.0008; callback pins the formatted
  magnitude. Interval excludes zero (probe 4,000 of 4,000) and contains the population miss
  (probe 0.9493).
- MS-05: assessment reading says "one row for the fluctuation's two-coefficient score, reported
  as its norm, and one influence-curve row per coefficient", matching the three-row score table.
  MS-06: "That study uses exact nuisances and fixed bounds of [0.01, 0.99]" (`G_BOUNDS`), not
  tied to the shortfall. MS-07: callback comment reads 1.20%, matching the stored output.
- Protocol. Six changed fields, pinned by the callback; fingerprint `ec77f97f713c2306` on the
  cell, the arm fit, and the trend fit. Death wording matches N5/N6: hypothetical strategy,
  treated as censoring in a real analysis, needs exchangeability and positivity for death given
  the recorded history, the law draws no deaths. The program's composite entry
  (`navigation.py:185`) is correctly described as scoring death as the worst score. "No
  documented range" replaces "standardized"; the q_bounds reason matches `tmle.py:1976`.
- Step 11 and Step 13. Saturated coefficients equal `ey[low]`, `ate[medium vs low]`,
  `ate[high vs low]` with influence curves within 1e-12. The robustness values 0.204 (0.181) and
  0.420 (0.394) are read from the separate `ATE(reference="low")` fit, as rule 211 prescribes,
  and the text says they bound the arm contrasts and not the slope. The Riesz second-moment
  condition is stated correctly for a contrast (E[1/g] finite for each arm). No refusal is shown:
  zero matches for "refus"; the `Not run` row is read as a ledger status.
- Repeated-draw numbers against `sweep.log`, recomputed from `sweep.csv` (4,000 rows, seeds
  30000 to 33999, unique): 3762/0.9405, 3755/0.9387, 3782/0.9455, 3762 (1:10:1), SE/SD 0.984,
  0.986, 0.997, blocks 0.943, 0.936, 0.944, 0.939 (page: 936 to 944), shift below one SE 3,984.
  The sweep asserts the worktree `src` on every worker.
- Trust rows against the study artifacts: primary `msm[a]` coverage 0.94625 and SE ratio 0.99686
  (`summary.csv`), weight `1 + 0.5 * treatment + 5 * W` (`canonical_point_msm.py:118`);
  `a__correctly_specified` coverage 0.9357 to 0.9641, SE ratio 0.9537 to 1.0430;
  `W__declared_weights` bias -0.000906 to 0.00177, margin 0.00449; `W__uniform_weights` bias
  -0.1150 to -0.1129, margin 0.00346. Study limits (two arms, three terms, pointwise intervals)
  match the evidence page lines 151-152. Each row names law, construction, outcome type, fold
  layer, and role.
- Support reading: "warns when that share is above 1%" matches `_verdict_parts`
  (`clipped_fraction > 0.01`); ESS is reported and not graded, as the verdict text says. The
  268.5 clever covariate is 6 / min fitted g(high) among `high` patients, pinned by the callback.
  Truncation-curve range 0.2654 to 0.2700 and largest movement 0.0033 pinned.
- Stored outputs quoted correctly: 0.272/0.368/0.360, 1.324/0.808, 0.532/-0.387, -0.343/0.474,
  -0.0038/0.6399/1.4642, (0.5641, 0.7156), -0.1068, 0.26875, (0.25191, 0.2856), 2.107,
  0.2688/0.7341/(0.2519, 0.2856)/(0.6902, 0.7779)/0.2657/0.7200, 0.1486/-0.1857/0.0371,
  0.2092/-0.2084/(-0.2566, -0.1602)/0.0008, -0.1171, 0.2593/0.0065/0.0086/0.2400/0.2398/
  (0.2214, 0.2582), 0.6437/1.4681, 1.20%/36/0.477/268.5/0.6093/0.0033, 0.204/0.181/0.420/0.394.
  The `R²` glyph in cell 40 is U+00B2.
- Execution counts 1 to 13 contiguous; no error outputs. `scripts/execute_notebook.py
  docs/examples/msm-projections.ipynb --check` printed "every cell reproduced its stored
  non-image output", exit 0 (`.tmp/notebook-review/gate-N9/check.log`).

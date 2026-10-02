# Gate N5: `docs/examples/collaborative-tmle.ipynb` at 5f85da29

Reviewed the notebook (every cell and stored output), the callback
`tests/unit/tutorial_semantics/collaborative_tmle.py`, the probe directory
`reviews/notebook-review/probes/collaborative-tmle-final/` (`sweep.py`, `sweep.csv`, `sweep.log`,
`summary.log`, `crossfit_check.py/.log`, `truth.log`), ledger rows CT-01 to CT-10 and CT-N1 to
CT-N4, plan row N5, `refusal-inventory.md` (collaborative section), `gate2-plan.md` (decision table
row "C-TMLE workflow", R-5, R-6), `docs/development/example-notebooks.md`, `docs/examples/index.md`
(collaborative row, shared design, survival row), gates N1 to N4, `synthetic.py` (`instrument_dgp`,
`make_instrument`), `tmle.py` (`_outcome_scale_refusal`), `validation-methods.md` (stress surface),
`collaborative-tmle.md`, the two study scripts and their `properties.csv`. Scratch:
`.tmp/notebook-review/gate-N5/` (not tracked).

## Verdict

Every printed number reproduces, every repeated-draw number matches `summary.log`, the trust rows
match the two `properties.csv` files, and no cell is a refusal. The page does not pass on its
workflow question. Its reported inferential result is a plain TMLE whose assignment model holds the
queue lottery, an interval the page itself measures at 555 of 600 (SE/SD 0.92), and it tells the
reader to "report the plain interval with its measured coverage", which a real program cannot do.
The design knowledge that excludes the lottery is already written into the page's own protocol. One
structural fix with re-execution, two wording fixes.

## The workflow question

| question | finding | evidence |
| --- | --- | --- |
| does the program know the lottery is an instrument? | yes, by design. Step 2: "The program fixes the hash before assignment, prevents staff overrides, and verifies that the draw changes no other service." The protocol's `assumption_rationale` records "An encounter-ID hash fixes the queue draw before assignment, without staff overrides." Step 25 then refuses to act on that knowledge: "That fit relies on the exclusion restriction, which the data cannot check, so this page does not report it." Every identification assumption on the page is uncheckable by data; the hash is the one with the strongest design basis | cells 7, 12, 25 |
| is excluding a known instrument from the adjustment set standard practice? | yes. The disjunctive cause criterion (VanderWeele 2019, *Eur J Epidemiol* 34:211) adjusts for each pre-exposure cause of exposure or outcome and excludes a variable known to be an instrument; bias amplification (Pearl 2011; Myers et al. 2011) is the reason. `make_instrument`'s own docstring: "Adjusting for `W2` is not merely unnecessary, it is harmful" | `synthetic.py:970-979` |
| what does the design-based fit do? | covers at the nominal rate and is the most precise of the three. Over the committed 600 draws: bias +0.0020 (MC SE 0.0020), empirical SD 0.0479, SE/SD 0.973, coverage 572 of 600. The all-three plain fit: SD 0.0694, SE/SD 0.919, 555. C-TMLE: SD 0.0554, plug-in 532 | `summary.log` (`p_ni_*`), recomputed from `sweep.csv` |
| on the page's draw? | `adjustment=("W1", "W3")`, same learners, in sample: psi 1.012, SE 0.0466, CI (0.921, 1.104); no g below 0.1 or above 0.9; ESS 0.903 and 0.892. The C-TMLE estimate 0.955 lies inside that interval, and did so on 598 of 600 draws | `probe44.py` |
| does C-TMLE agree as a data-adaptive check? | yes. On the full approved set the selector left the lottery out on 591 of 600 draws under the linear Q, and on 565 under the constant Q. It is not a precision gain over the design-based fit: its SD is 1.16 times that fit's. Its gain (0.80) is over the fit that includes the instrument | `summary.log` |
| post-selection inference | using C-TMLE's selected set to define the reported TMLE is invalid: the selection varies by draw (empty on 347, `social_support` alone on 162), and an interval conditional on it has no result. Design-based exclusion is fixed before the data and is not selection. The page must say both | `summary.log` selection shares |
| what the instrument also does to the sensitivity analysis | the omitted-variable bound on the design-based fit gives rv 0.381 and rva 0.357 (nu2 4.49, sigma2 0.971) against the page's 0.227 and 0.204 (nu2 11.96). Extreme propensities inflate the representer's second moment, so the instrument weakens the robustness value as well | `probe44.py` |
| is the current page acceptable with its disclosure? | no. `example-notebooks.md` rule 174 permits a sub-nominal interval only with the measured cause named; the page names it. But the cause is a design error the protocol already rules out, and the recommendation that follows ("report the plain interval with its measured coverage") is not a workflow: a program has no truth to measure coverage against. A hostile reader asks why the analyst kept a variable the protocol calls a randomized hash |

Recommendation: the reported analysis is the plain TMLE on the design-based adjustment set
`("baseline_readiness", "social_support")`. C-TMLE on the full approved set is the data-adaptive
check that agrees. The all-three plain fit stays as the Step 8 failure mode, which is what happens
when predictive accuracy chooses g.

## REQUIRED FIXES

1. Reported analysis by design (Steps 2, 5, 8, 11, trust, learning table; callback;
   `docs/examples/index.md:15`). Reframe the applied question: the operations review certifies the
   queue mechanism, so the lottery leaves the adjustment set by design before any model runs; the
   clinical review cannot certify whether readiness and social support confound, and that is the
   selector's question. Add the second `PointTreatment` with
   `adjustment=("baseline_readiness", "social_support")`, fit the same `plain_method` on it, and
   report its interval as the inferential result (this draw: 1.012, SE 0.0466, (0.921, 1.104)).
   Read it against the sweep: 572 of 600, SE/SD 0.97, SD 0.048, with the sources added to
   `UNPRINTED_DECIMALS`. Keep the all-three plain fit in Step 8 as the failure mode, with its
   tails, 555 of 600 and SE/SD 0.92 as measured consequences of an instrument in g. Read C-TMLE on
   the full approved set as the check: it left the lottery out on 591 of 600 draws, and its estimate
   lay inside the reported interval on 598 of 600. State once that the C-TMLE spread (0.055) is
   larger than the reported fit's (0.048), so the selector is the safeguard when design knowledge
   is absent, not a precision gain over it. State explicitly that the reported set is fixed by
   design and that refitting a plain TMLE on the set C-TMLE selected (here `social_support`) would
   be post-selection inference with no result behind it. Move the omitted-variable bound in Step 11
   to the reported fit and read the two robustness values side by side (0.381 against 0.227 on this
   draw), naming nu2 as the mechanism. Delete "report the plain interval with its measured
   coverage" and the trust row "nominal coverage at this size, because the draw in g lowers it".
   Rewrite the index row so it does not say "C-TMLE leaves it out" as the design's answer; the
   design excludes it and the selector agrees on most draws. Re-execute; the protocol fingerprint
   need not move if the rationale text stays.
2. Hypothetical strategy for death (Step 4 reading, cell 13). The strategy is coherent as an ICH
   E9(R1) choice forced by a score with no worst value, but the page states none of its
   consequences while the survival row of `index.md:20` does for the same program. Add one
   sentence: under this strategy a real analysis treats death as censoring and needs
   exchangeability and positivity for death given the recorded history; this law draws no deaths,
   so the choice does not touch the estimate here. Markdown only.
3. Cell 25, last paragraph, and cell 35, first limit: delete the sentence that the exclusion
   restriction "cannot be checked, so this page does not report it" and the instruction to report
   a sub-nominal interval. Covered by fix 1; listed so neither sentence survives a partial edit.

## Advisory, not gating

- In-sample reason (cells 13, 17, 35). The library's refusal asks for the "known outcome support"
  (`tmle.py:1771-1781`), so "no documented range" is a true fact, not the N4 error. The reason the
  in-sample fit is valid is the linear learners (Donsker), and the five-fold probe with a
  convention range moved both estimates by less than 0.003. Lead with the learner condition and
  keep the range as the fact, as N4 fix 2 did.
- Assessment summary (cell 30). Six `unavailable` sensitivity rows print without a reason in the
  reading. Not a refusal showcase: nothing is narrated as a refusal and no cell raises. One phrase
  ("unavailable because this fit reports no influence-curve inference; Step 11 reads them on the
  reported fit") would close it.
- Trust row for `crossfit_overfitting`: "the bounded law of `navigation_data`" is the bounded twin
  of `nonlinear_dgp` at concentration 12, which `navigation_data` wraps; correct, and 400
  replicates could be named as the selector row names 800.

## OK

- `scripts/execute_notebook.py docs/examples/collaborative-tmle.ipynb --check`: "every cell
  reproduced its stored non-image output", exit 0 (`check.log`).
- Every quoted printed number matches a stored output: 1.977, 0.321/-0.288, 0.476/-0.514,
  `b81fd2923b8e38c8`, 0.955, `working_mechanism_plugin`, 0.0440, (0.869, 1.041), 6.63098, 6.65024,
  -0.00011 (6.63098 - 6.63109), 0.1040, 0.1140, 43.7542, 0.4719, 0.4499, 0.844, 0.9985, 0.9984,
  0.0626, (0.758, 1.004), 0.0923, 0.2343, 0.394, 26.2934 to 24.9909, 24.8761 to 24.9728, 0.9030,
  0.8915, 6.3471, 0.4434 to 0.5814, 0.516, 0.227, 0.204, 0.968, 11.963, 0.95497, 1.0232,
  +0.068278, 0.63882, -0.31615, 0.60972, -0.34525, -0.0480, -0.0252, 0.0545; "about a third" is
  0.33 and 0.36.
- Every repeated-draw number matches `summary.log`: more than half (347), about a quarter (162),
  about one in seven (84), 9 of 600, 0.069/0.055, 0.92/0.81, 555/532, four fifths (0.798), 595 of
  600, 572, about 0.1 (+0.1109), 513, about a third (0.330), overstated (1.377), 599, 565, 0.98
  (0.976). `sweep.log` names this worktree's `src`; `truth.log` confirms ATE 1 on 10^6 rows.
- Law readings: logit `0.8 W1 + 1.5 W2`, so "the draw moves assignment more strongly than
  readiness" holds; `W2` absent from the outcome mean; independent standard normals (`DGP.sample`).
- Trust rows against `ctmle_selector/properties.csv` and `ctmle3_oat/properties.csv`: 0.241,
  0.0037, 0.0030, 0.80, red (`passed` False); 0.0495 to 0.0511, 0.0022; 0.92, 0.99; 0.54, 0.47;
  n 1,500 and 500; 800 replicates; five outer folds; `DummyRegressor` Q, `LogisticRegression` g,
  greedy default; `DecisionTreeRegressor(min_samples_leaf=1)`; `empty_control` is
  `strategy="discrete", candidates=((),)`. The five items of the trust-step rule are present.
- `simulated_confounding`: read as a stress test, "not a bound"; the top-10% flip matches
  `U >= Phi^{-1}(1 - k_A)`; the chance association is pinned by redrawing the latent vector.
- CT-01 to CT-09, CT-N1, CT-N4 are delivered as the ledger specifies; CT-09 comment reads 0.844.
- Callback: nonzero witnesses present (ESS below 0.95 under the constant Q, clever covariate above
  twice the linear-Q value, risk rising on the path, HC0 above 1.1 times the plug-in); no refusal
  pins; `UNPRINTED_DECIMALS` names a source for each quoted sweep and study decimal.

## Re-check of 29bf0bc1

PASS. The three required fixes and the index row are delivered; the two advisory items are also
closed.

| item | finding | evidence |
| --- | --- | --- |
| reported inference | Step 5 builds `design_set = ("baseline_readiness", "social_support")`; Step 8 fits `plain_method` on it and reads 1.012, SE 0.0466, (0.921, 1.104), ESS 0.9030 and 0.8915, no tail rows. Cell 0 cites VanderWeele 2019 (doi 10.1007/s10654-019-00494-6) for excluding a known instrument | cells 0, 14, 15, 24, 25 |
| failure mode and check | the all-three fit keeps its tails (0.1040, 0.1140, 43.7542), 0.0626, (0.758, 1.004); the 600-draw table reads 0.048/0.069/0.055, 0.97/0.92/0.81, 572/555/532; "left the draw out of g on 591 of 600", "inside the reported interval on 598", C-TMLE spread "larger than the spread of the reported fit" (1.157) | cell 25 against `summary.log` |
| post-selection inference | named in cell 25 ("That is post-selection inference") and the trust section ("do not report an interval refitted on its selected set"); selection shares (more than half, about a quarter, one in seven, 9 of 600) match 347/162/84/9 | cells 22, 25, 35 |
| Step 4 death statement | cell 13: hypothetical strategy treats death as censoring, needs exchangeability and positivity for death given the recorded history, the law draws no deaths | cell 13 |
| the two sentences | zero matches for "measured coverage", "cannot be checked, so this page does not report", "because the draw in g lowers it" | grep over the `.ipynb` |
| `index.md:15` | "The design excludes the lottery, and the reported TMLE adjusts for the other two variables. C-TMLE on all three variables agrees on most draws of a review probe" | `docs/examples/index.md:15` |
| Step 11 bound on the reported fit | 0.381/0.357, sigma2 0.971, nu2 4.486 against 0.227/0.204, 11.963; "larger on the design-based fit on 596 of 600"; fitted nu2 range 4.164 to 4.702 | cell 33 output, cell 34, `summary.log` |
| population nu2 4.357 | `truth.py` integrates `g(W1) = E[expit(0.8 W1 + 1.5 W2) | W1]` by 80-node Gauss-Hermite quadrature and takes `E[1 / (g (1 - g))]`. The library's nu2 is the doubly robust `2 m_alpha - alpha^2`, whose population value is `E[alpha^2]` with `alpha = A/g - (1-A)/(1-g)`, so `E[alpha^2] = E[1/(g(1-g))]`. The design-based representer depends on W1 only, since W3 does not move assignment, so 4.357 is the nu2 of that representer. The 600-draw mean 4.367 sits beside it. The page states that the logistic g approximates the averaged curve | `truth.py`, `truth.log`, `omitted_variable.py:690-716` |
| `simulated_confounding` | unchanged: starts from 0.95497, chance association -0.0480 pinned by redrawing the latent vector in the callback, "not a bound", flips move the estimate down by about a third | cells 33, 34; callback |
| six `unavailable` rows | cell 31: five omitted-variable rows have no nu2 derivation for a collaborative fit (`_CTMLE_BOUND_REFUSAL`, `omitted_variable.py:191`), and `sensitivity.evalue` needs an interval (`evalue.py:88`, "an E-value is built from the reported estimate and its interval") | cell 31, source |
| in-sample reason | cells 13, 17, 35 lead with the linear learners and the Donsker condition; the range is kept as the fact | cells 13, 17, 35 |
| callback | nonzero witnesses: design-based ESS between the all-three value and 0.95, `reported.std_error < plain.std_error`, `reported_bounds["nu2"] < 0.5 * plain_bounds["nu2"]`, sigma2 gap below 0.05, `covers(reported, collaborative.psi)`, HC0 above 1.1 times the plug-in, chance association reproduced to 1e-9. `UNPRINTED_DECIMALS` adds 0.048, 0.97, 4.357, 4.164, 4.702 with sources | `tests/unit/tutorial_semantics/collaborative_tmle.py` |
| `--check` | `scripts/execute_notebook.py docs/examples/collaborative-tmle.ipynb --check`: "every cell reproduced its stored non-image output", exit 0 | `.tmp/notebook-review/gate-N5/recheck.log` |

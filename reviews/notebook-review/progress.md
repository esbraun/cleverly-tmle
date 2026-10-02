# Notebook review progress

Branch `agent/notebook-review`, from `origin/main` at 5f33902b. Resume from the first row that is
not `done`. Findings live in `findings/<stem>.md`, verification in `verification/`, and the plan in
`plan.md`.

| notebook | 1 review | 1 verify | 1 sweep | gate: findings | 2 plan | 3 implement | gate: commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| point-treatment-tmle | done (2 wrong, 3 misl, 4 weak) | done (4 conf, 5 partial, 0 refuted, 1 new) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| cross-fitting | done (0 wrong, 1 misl, 5 weak) | done (6 conf, 0 partial, 0 refuted, 1 new) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| collaborative-tmle | done (0 wrong, 5 misl, 5 weak) | done (7 conf, 3 partial, 0 refuted, 1 new) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| dr-tmle | done (1 wrong, 5 misl, 4 weak) | done (7 conf, 3 partial, 0 refuted, 1 new) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| interventions | done (2 wrong, 4 misl, 7 weak) | done (10 conf, 3 partial, 0 refuted) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| survey-nonresponse | done (1 wrong, 3 misl, 7 weak) | done (8 conf, 3 partial, 0 refuted, 2 new) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| longitudinal-tmle | done (1 wrong, 3 misl, 4 weak) | done (7 conf, 1 partial, 0 refuted, 1 new) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| longitudinal-survival | done (0 wrong, 1 misl, 5 weak) | done (3 conf, 3 partial, 0 refuted) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| msm-projections | done (1 wrong, 2 misl, 4 weak) | done (5 conf, 2 partial, 0 refuted) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |
| twins-causal-inference | done (1 wrong, 3 misl, 8 weak) | done (5 conf, 7 partial, 0 refuted) | done | done (confirmed; 3 corrections) | gate 2 running | | | | |

## Implementation (plan order)

| order | change | status | commit | gate |
| --- | --- | --- | --- | --- |
| 0 | plan commit and push | done, pushed | 0893e590 (pre-registration) | gate 2 done (7 required changes applied) |
| S1 | fix A: strong-positivity propensity | committed | 5bc92b5e | PASS (gate-S1.md) |
| S2 | library strings and docstrings | committed | d0fc2e45, fix 753c9992 | PASS after fix (gate-S2.md) |
| S3 | study regeneration (four studies) | committed (no red cell; primaries byte-identical) | 1f9cf969 | PASS (gate-S3.md) |
| S4 | narration harness | committed | d546049f | PASS (gate-S4.md) |
| S6 | shift ratio: trim reverted; docstring fixed; per-fold mean-ratio diagnostic (IV-N3) | committed (fast suite 13511 passed) | cd2a1452, revert 19e91585, 9ae18141, fix 87144895 | PASS after fix (gate-S6.md) |
| S5 | house rules and shared design | committed | a8661971, fixes 84d26a6b | PASS after fixes (gate-S5.md) |
| N1 | point-treatment-tmle | committed (15 s; --check 0) | a9684c17, fix 7352f691 | PASS after fix (gate-N1.md) |
| N2 | cross-fitting | committed (25 s; --check 0) | 5d5e08a0, fix bf2ce120 | PASS after fix (gate-N2.md) |
| N3 | dr-tmle | committed (15 s; --check 0) | 215545d2, fix 91b5f0ef | PASS after fix (gate-N3.md) |
| N4 | interventions | committed (10.5 s; --check 0) | f62ed489, fix 99cb68d0 | PASS after fix (gate-N4.md re-check) |
| N5 | collaborative-tmle | committed (6 s; --check 0) | 5f85da29, fix 29bf0bc1 | PASS after fix (gate-N5.md re-check) |
| N6 | survey-nonresponse | committed (--check 0) | 198c79a5, fix f28f0b6b | PASS after fixes (gate-N6.md); Step 6 and 9 near nominal on 3000 draws |
| N7 | longitudinal-tmle | committed (6 s; --check 0) | 6714262d, fix 4d4d5816 | PASS after fix (gate-N7.md) |
| N8 | longitudinal-survival | committed (5.8 s; --check 0) | 2692722c, fix aed3986b | PASS after fix (gate-N8.md) |
| N9 | msm-projections | committed (5.4 s; --check 0) | 6927899e, fix cafb67af | PASS after fix (gate-N9.md) |
| N10 | twins-causal-inference | committed (42 s, network; --check 0) | aaf4c484, fix 87fec724 | PASS after fixes (gate-N10.md) |
| F | prose ledger, fast suite, docs build, final gate, PR, CI | F1, F2 done; prose ledger current (0 undecided); fast suite running | 4fc4b6c8 (F1); 92b24e4f d029b7a8 010be6ea 62e6e31c 87fec724 (F2) | |

## Log

- 2026-10-01: branch created; ten reviewers launched in parallel.
- 2026-10-01: all ten reviews returned (9 wrong, 30 misleading, 50 weak in total). TWINS `--check`
  exits 1 on stored platform noise and the Python version line (TW-04, TW-05). Ten refutation
  verifiers launched in parallel.
- 2026-10-01: user guidance: the examples must not showcase refusals; they are end-to-end
  examples that work. The plan replaces every refusal showcase with a working analysis or drops it,
  and changes the house rules in `docs/development/example-notebooks.md` that ask for refusal
  cells. Refusal inventory agent launched (`refusal-inventory.md`).
- 2026-10-01: verification N1 (point-treatment) independently confirmed by the orchestrator in
  closed form: the `nonlinear_dgp` propensity logit gives E[1/g] proportional to
  E exp(0.525 W2^2), which diverges, so the ATE efficiency bound of the `navigation_data` law is
  infinite. Affects every `navigation_data` page. A generator change would move registered studies,
  so the plan narrates it and defers the generator change.
- 2026-10-01: all ten verifications returned, 0 refuted. Consolidation and sibling-sweep agent launched (`ledger.md`, `sibling-sweep.md`).
- 2026-10-01: refusal inventory returned (`refusal-inventory.md`); every replacement executed. Its
  20-seed probe disagrees with two verifiers on calibrated-learner coverage, so a 300-seed learner
  experiment runs before the plan fixes the shared treatment learner (`learner-experiment.md`).
- 2026-10-01: sweep done (`ledger.md`: 7 wrong, 27 misleading, 75 weak, 5 none; `sibling-sweep.md`
  P1-P18). No study calls `navigation_data`, so the plan candidate is a strong-positivity
  propensity for that law only (experiment arm added). Gate 1 (Fable) reviewing findings.
- 2026-10-01: user ruling: fixes that would move registered studies are implemented and the studies regenerated; nothing goes to the roadmap. N1 fix moves to `nonlinear_dgp` itself (four studies regenerate). Study-impact research launched (`study-impact.md`).
- 2026-10-01: Gate 1 passed (`gates/gate1-findings.md`). Corrections adopted: the degree-2 logistic
  g is "flexible parametric", not correctly specified (the logit has a step in W4); the inventory's
  20-seed "calibrated g halves the interval" flag is dropped (153 seeds: SE/SD 1.04-1.12); the
  squeeze is needed for the omitted-variable bound, not the ATE intervals. The plan must carry the
  ATT/ATC truth moves, the IPSI re-sweep on the new law, new PT truncation and DR support lessons,
  the reporting policy before regeneration, the pinned tests, and the four evidence pages.
- 2026-10-01: Gate 2 required seven plan changes (S3 commands, red-cell route = stop and ask the user, pre-registration checklist, PT truncation witness, CF probe, IV-02 decision, probes directory); all applied. Plan committed and pushed as the pre-registration.
- 2026-10-01: S1 committed 5bc92b5e (fast suite: 13494 passed, 4 expected tutorial failures). Smoke run (fold-evaluated, --replicates 10, outside the repo) wrote all seven artifacts; it exits 1 on underpowered performance gates, as expected at 10 replicates. Gate S1 running.
- 2026-10-01: Gate S1 PASS. S3 declared runs started (serial, outputs outside the repo). S2 waits until S3 ends because manifests hash module sources; S4 runs now (tests only).
- 2026-10-01: S3 committed 1f9cf969: four runs exit 0, primary artifacts byte-identical, only property cells moved, no red cell (declared stop not triggered). Cross-fitting notebook must update 48.8%/93.3% to the regenerated values. Gate S3 and S2 run concurrently (disjoint files).
- 2026-10-01: S2 committed d0fc2e45 (fast suite 13502 passed, 4 expected tutorial failures). Stored outputs that now print changed strings: twins (maximal bias, BOTH), dr-tmle (contract line), survey-nonresponse (F20, deleted step), interventions (identification text); longitudinal-tmle prose quotes 'clustered and evidenced'.
- 2026-10-01: N1 committed a9684c17 (200-seed sweep: coverage 189/200). Gate N1 probes the DR nu^2 estimate falling below 4.83 on 197/200 draws.
- 2026-10-01: N2 committed 5d5e08a0. Seed 34 kept; the in-sample interval now covers on this draw, so the failure mode rests on the 200-seed probe (153/200 vs 188/200). Final-stage follow-up: delete the unused helper assert_plugin_limits_refuse.
- 2026-10-01: N3 committed 215545d2 (200 draws: no DR-TMLE advantage on the new law; page says so). Final-stage follow-up: index.md:16 wording 'recorded assignment rule that is difficult to model'.
- 2026-10-02: N4 committed f62ed489: degree-2 logistic g chosen by a 120-seed sweep (IPSI coverage 0.967); dose-axis repair tried and rejected (+1.0 coverage 0.70 stated). Gate N4 also probes IV-N3 (SE inflation under wide q_bounds).
- 2026-10-02: Gate N4: regime and incremental axes pass; dose axis must adopt quadratic Q plus a regularized booster density; IV-N3 is a library defect (shift density ratio never bounded at targeting time). New change S6 fixes it test-first; study impact decides any pre-registered rerun.
- 2026-10-02: S6 committed cd2a1452 (0.999-quantile trim, lmtp convention; default on). It moves lmtp_shift primary rows and properties, and the rm18 comparator density fixture. Decision: comparator fits run shift_trim=1.0 to match the R adapter's .trim=1; the property study is pre-registered and regenerated under the default. The SE/SD 2.68 that remains is under root-cause investigation first.
- 2026-10-02: IV-N3 root cause (investigations/iv-n3.md): a poor held-out density ratio from the bare-booster density learner; the influence curve is correct, and cross-fitting with an exact density gives SE/SD 0.98. Decision reversed: revert the trim, fix the ShiftSet.ratio docstring, add a per-fold mean-ratio diagnostic, document the finding. No study moves.
- 2026-10-02: N5 committed 5f85da29. New finding: the reported plain TMLE interval under-covers (555/600) because the lottery instrument is in its assignment model. Gate N5 judges a design-based exclusion workflow.
- 2026-10-02: N6 committed 198c79a5. New finding SN-N3: the Step 9 cross-fitted stacked fit covers 0.922 over 500 draws with bias near 0 (SE too small); root-cause investigation running beside gate N6.
- 2026-10-02: Gate N6 passes with two fixes (trust lead names Steps 6 and 9; Step 9 cause after SN-N3). Final shared pass must reconcile index.md:44 with the hypothetical death strategy on the collaborative and survey pages. N7 runs beside the SN-N3 investigation (4 workers).
- 2026-10-02: SN-N3 resolved (investigations/sn-step9.md): no library bug. The seed block 1001-1500 is low for the oracle too (463/500); 3000 draws give 0.946. The N6 fix must drop the Step 9 under-coverage claim and re-measure Step 6 on a larger, disjoint seed range before keeping its 0.92.
- 2026-10-02: N8 committed 2692722c. Final shared pass adds index.md:20 ('a controlled direct effect only under exchangeability for death') and index.md:44 (death strategy). N7 markdown fix and gate N8 run in parallel.
- 2026-10-02: Gate N10 surfaced a sibling pattern the per-page gates missed: five notebooks (collaborative, dr-tmle, msm, survey, twins) print the assessment summary with 'unavailable' rows and narrate them, against house rule 'print only the rows that run'. longitudinal-tmle filters correctly. Sweep fix F2 queued after F1, together with TWINS section 11 (truth coverage 55/60).

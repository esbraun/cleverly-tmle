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
| N5 | collaborative-tmle | running | | |
| N6 | survey-nonresponse | pending | | |
| N7 | longitudinal-tmle | pending | | |
| N8 | longitudinal-survival | pending | | |
| N9 | msm-projections | pending | | |
| N10 | twins-causal-inference | pending | | |
| F | prose ledger, fast suite, docs build, final gate, PR, CI | pending | | |

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

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
| 0 | plan commit and push | pending | | gate 2 done (7 required changes applied) |
| S1 | fix A: strong-positivity propensity | pending | | |
| S2 | library strings and docstrings | pending | | |
| S3 | study regeneration (four studies) | pending | | |
| S4 | narration harness | pending | | |
| S5 | house rules and shared design | pending | | |
| N1 | point-treatment-tmle | pending | | |
| N2 | cross-fitting | pending | | |
| N3 | dr-tmle | pending | | |
| N4 | interventions | pending | | |
| N5 | collaborative-tmle | pending | | |
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

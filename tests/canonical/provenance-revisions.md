# Provenance revisions

Each `manifest.json` records a sha256 over every source that produced that study's committed
artifacts. `tests/unit/test_method_evidence.py` checks the reference sources against the working
tree, so an edit to a Dockerfile, an R runner, or a shared R harness fails the fast tier.

That check cannot tell two different edits apart. An edit to a container's package set, or to a
runner's model call, is **result-determining**: the committed artifacts no longer describe the
code, and the study must be regenerated. An edit that moves a script between the image and a
mount, or renames a local, is **result-neutral**: a regeneration would spend hours to write
identical bytes.

Rewriting the recorded hash in place clears the failure and makes the manifest say something
false. The manifest would then claim that bytes which did not exist at generation time produced
the result. Record the judgement here instead. The manifest keeps the hash of what ran, and the
row below carries the reason the difference does not change it.

Read the rule for the other two hash groups in
[method benchmarking strategy](../../docs/development/method-benchmarking.md#what-makes-a-study-stale).

**This table is a gate.** An unrecorded difference fails. A row whose `recorded` hash no longer
appears in any manifest also fails, on the terms `StudyRecord.accepted_reference_failure` already
sets: remove a declaration rather than carry a stale exception. A row whose `current` hash no
longer matches the file fails too, because the file moved again after the judgement was made.

Each `judgement` is a falsifiable claim, not a waiver. `pytest -m slow` re-executes each study,
so a result-neutral reason that is wrong shows up as a changed artifact.

| source | recorded | current | judgement |
| --- | --- | --- | --- |
| `tests/canonical/drtmle/Dockerfile` | `ddeb48b470fb77e50165f04489f4317cfc28cc23532a765bffaad6085c243bad` | `470471189bf2b407411d6a89a533e073fd4f2dffb971a9480702658d5da88fe2` | result-neutral: the image stopped baking `run_drtmle.R` and stopped naming it in `ENTRYPOINT`, so one shared context can now mount either study's runner. The `rocker/r-ver` digest, the `drtmle` commit, and the install steps are unchanged, so the fit ran in the same R environment either way |
| `tests/canonical/lmtp_crossfit_adapter.R` | `788af2d43a3489597231e271769eca5d2b0b15e786be8482233819fe0e023a00` | `00fda6c717324a618c041ced90114eb909b2531d21e70379997d144afe93f560` | result-neutral: every existing study passes no identifier. Three use five labels. The ordinary competing row uses one label, and it reaches its folds through `lmtp_competing_adapter.R`, which builds the single-fold case itself and never calls `fold_list`. Each one declares a binomial outcome, or a survival outcome above horizon one, and the continuous-initial branch tests `outcome_type == "continuous"`, which neither value is. The one-label, identifier, cluster-integrity, and continuous-initial branches do not run for those recorded studies |
| `tests/canonical/lmtp_crossfit_adapter.R` | `9ec0c76f9ed1ee41276b73071d51baaacae8537662e2f61994ac49cc4493b71f` | `00fda6c717324a618c041ced90114eb909b2531d21e70379997d144afe93f560` | result-neutral: the categorical studies pass no identifier and have a binary outcome. The identifier, cluster-integrity, and continuous-initial branches do not run for those recorded studies |

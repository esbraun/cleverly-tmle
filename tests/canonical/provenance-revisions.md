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

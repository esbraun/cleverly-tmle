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
| `tests/canonical/lmtp_crossfit_adapter.R` | `788af2d43a3489597231e271769eca5d2b0b15e786be8482233819fe0e023a00` | `7f5d2b663e4dca986ade973a8d263ea132a666ff8d59444ecaa407f607b0f8dc` | result-neutral: every existing study passes no identifier and no weights. Three use five labels. The ordinary competing row uses one label, and it reaches its folds through `lmtp_competing_adapter.R`, which builds the single-fold case itself and never calls `fold_list`. Each one declares a binomial outcome, or a survival outcome above horizon one, and the continuous-initial branch tests `outcome_type == "continuous"`, which neither value is. The identifier, weight, cluster-integrity, and continuous-initial branches do not run for those recorded studies |
| `tests/canonical/lmtp_crossfit_adapter.R` | `9ec0c76f9ed1ee41276b73071d51baaacae8537662e2f61994ac49cc4493b71f` | `7f5d2b663e4dca986ade973a8d263ea132a666ff8d59444ecaa407f607b0f8dc` | result-neutral: the categorical studies pass no identifier and no weights, and they have a binary outcome. The identifier, weight, cluster-integrity, and continuous-initial branches do not run for those recorded studies |
| `tests/canonical/lmtp_crossfit_adapter.R` | `00fda6c717324a618c041ced90114eb909b2531d21e70379997d144afe93f560` | `7f5d2b663e4dca986ade973a8d263ea132a666ff8d59444ecaa407f607b0f8dc` | result-neutral: the clustered study passes an identifier but no weights. Its identifier and cluster-integrity branches are unchanged, and the new optional weight branch stays inactive |
| `tests/canonical/lmtp_crossfit/Dockerfile` | `ab293b84b4336a5fb8abd3f414aa58835b153a76df252949720eb150c0a57a5d` | `a083e24a15a8a80bd28b491edfb425d791be3f39f10ffd6b1430c6f4ff1db9b1` | result-neutral: for the existing studies, the image now downloads and digest-pins `ife` 0.2.3 instead of asking `install2.r` for the current release. The recorded image also resolved `ife` 0.2.3, and the dependency additions `collapse` and `S7` are the packages that pinned tarball requires. The R base digest, `lmtp` commit and tarball, and every fitted package version stay unchanged |
| `tests/canonical/ltmle_regimen_adapter.R` | `03f1b3d2ae35ae8f63096b42c1e00875845e916a9c74b5ffb947ac2a1f28baab` | `51715276e3a0a4dec7d0a904d3c377b22e61775f46fe631d1297b4d75eef7cb0` | result-neutral: every recorded caller omits `observation.weights`. The new weighted-mean branch is inactive for those fits, and the former unweighted `mean(predictions)` expression remains the active branch |
| `tests/canonical/tmle_learned_weighted/run_study.R` | `2ed957cb726ba1b9ac48f1ebd7374dfa0e3600faeb46b785031f80a7c73adb24` | `9ed6d17d16c9fd56907edcad35b3aca037180e2923d60daf443c159290e081ac` | result-neutral: the runner gained the empty-sample guard `if (!nrow(samples)) stop("samples contain no observations")`, which `tests/canonical/tmle_weighted/run_study.R` already carried. The guard raises only on a sample file with zero rows. The recorded run fitted 800 replications of 2,000 rows, so the branch never executes for the committed artifacts. Nothing else in the runner moved, and the edit restores parity between the two weighted runners |
| `tests/canonical/tmle_cde/run_study.R` | `ace28808cdd303c84a19c896a5c270db2545b92a236a2d3f35a86808c1994258` | `d1e0aeb838fd85a4f422dd29acebaad6c30df003353b01675afb777f38cfb85a` | result-neutral: the runner now reads one sample block per replicate and applies both existing level recodes to that frame. The former file repeated the same frame under two scenario labels. Each `tmle` call receives the same `Y`, `A`, `W`, `Z`, `Delta`, `Q`, `Q.Z1`, `g1W`, `pZ1`, and `pDelta1` values as before. The edit also removes an `identical()` guard over columns built from the same vectors, so that guard could never fail. The fit calls and exported scenario keys are unchanged |
| `tests/canonical/tmle_cde/probe_native_result2.R` | `e4ba2f586cb11dcca1a086d449ea6cfd12123a5cba070db7bad7e95ada81c087` | `3c20d4d5355742d25997e4a692a6db0001a6a015633a911e84236f12722cefb2` | result-neutral: the probe selects replication zero from the deduplicated sample file instead of selecting its former level-one duplicate. Both former scenario blocks contained identical observed and nuisance rows. The native second-result call, supplied truth, and output labels are unchanged |

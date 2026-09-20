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

Each `judgement` is a falsifiable claim, not a waiver. The next result-determining regeneration
exposes a wrong result-neutral reason as a changed artifact.

| source | recorded | current | judgement |
| --- | --- | --- | --- |
| `tests/canonical/lmtp_crossfit_adapter.R` | `788af2d43a3489597231e271769eca5d2b0b15e786be8482233819fe0e023a00` | `7f5d2b663e4dca986ade973a8d263ea132a666ff8d59444ecaa407f607b0f8dc` | result-neutral: every existing study passes no identifier and no weights. Three use five labels. The ordinary competing row uses one label, and it reaches its folds through `lmtp_competing_adapter.R`, which builds the single-fold case itself and never calls `fold_list`. Each one declares a binomial outcome, or a survival outcome above horizon one, and the continuous-initial branch tests `outcome_type == "continuous"`, which neither value is. The identifier, weight, cluster-integrity, and continuous-initial branches do not run for those recorded studies |
| `tests/canonical/lmtp_crossfit_adapter.R` | `9ec0c76f9ed1ee41276b73071d51baaacae8537662e2f61994ac49cc4493b71f` | `7f5d2b663e4dca986ade973a8d263ea132a666ff8d59444ecaa407f607b0f8dc` | result-neutral: the categorical studies pass no identifier and no weights, and they have a binary outcome. The identifier, weight, cluster-integrity, and continuous-initial branches do not run for those recorded studies |
| `tests/canonical/lmtp_crossfit/Dockerfile` | `ab293b84b4336a5fb8abd3f414aa58835b153a76df252949720eb150c0a57a5d` | `a083e24a15a8a80bd28b491edfb425d791be3f39f10ffd6b1430c6f4ff1db9b1` | result-neutral: for the existing studies, the image now downloads and digest-pins `ife` 0.2.3 instead of asking `install2.r` for the current release. The recorded image also resolved `ife` 0.2.3, and the dependency additions `collapse` and `S7` are the packages that pinned tarball requires. The R base digest, `lmtp` commit and tarball, and every fitted package version stay unchanged |
| `tests/canonical/ltmle_regimen_adapter.R` | `03f1b3d2ae35ae8f63096b42c1e00875845e916a9c74b5ffb947ac2a1f28baab` | `d0edd452d315cb6b3c19e4c1ef4794a618031d6f2a9a3f13ec5a7b08b45362c6` | result-neutral: every older recorded caller omits `observation.weights`. The weighted-mean branch stays inactive for those fits. The adapter also returns the native summary standard error as a new list element. Those callers ignore the element, so their estimates and exported rows are unchanged |
| `tests/canonical/tmle_learned_weighted/run_study.R` | `2ed957cb726ba1b9ac48f1ebd7374dfa0e3600faeb46b785031f80a7c73adb24` | `9ed6d17d16c9fd56907edcad35b3aca037180e2923d60daf443c159290e081ac` | result-neutral: the runner gained the empty-sample guard `if (!nrow(samples)) stop("samples contain no observations")`, which `tests/canonical/tmle_weighted/run_study.R` already carried. The guard raises only on a sample file with zero rows. The recorded run fitted 800 replications of 2,000 rows, so the branch never executes for the committed artifacts. Nothing else in the runner moved, and the edit restores parity between the two weighted runners |
| `tests/canonical/tmle_cde/run_study.R` | `ace28808cdd303c84a19c896a5c270db2545b92a236a2d3f35a86808c1994258` | `d1e0aeb838fd85a4f422dd29acebaad6c30df003353b01675afb777f38cfb85a` | result-neutral: the runner now reads one sample block per replicate and applies both existing level recodes to that frame. The former file repeated the same frame under two scenario labels. Each `tmle` call receives the same `Y`, `A`, `W`, `Z`, `Delta`, `Q`, `Q.Z1`, `g1W`, `pZ1`, and `pDelta1` values as before. The edit also removes an `identical()` guard over columns built from the same vectors, so that guard could never fail. The fit calls and exported scenario keys are unchanged |
| `tests/canonical/tmle_cde/probe_native_result2.R` | `e4ba2f586cb11dcca1a086d449ea6cfd12123a5cba070db7bad7e95ada81c087` | `3c20d4d5355742d25997e4a692a6db0001a6a015633a911e84236f12722cefb2` | result-neutral: the probe selects replication zero from the deduplicated sample file instead of selecting its former level-one duplicate. Both former scenario blocks contained identical observed and nuisance rows. The native second-result call, supplied truth, and output labels are unchanged |
| `tests/canonical/tmle_mar_natural_course/probe_scale_workaround.R` | `a91e13a077e164482a84bc54980eb1fcbc071fc3c0361078d9ae14daaf19f488` | `39261cc47e160227d580b67b754767cdb066d4c205d6054c82e683285f293f30` | result-neutral: only the header comment on check 5 changed. The former comment said the comparison would see a wrong scale, and the evidence page shows it would not. R ignores comments, so every fit call, check, and exported column is unchanged |

## Study modules a manifest omits

`tests/unit/test_study_provenance.py` requires a manifest to record every study-specific Python
module that the study's runner and properties modules import. A result-neutral refactor can open
a gap. For example, a helper moves out of a study module into a new module. The manifest cannot
record a hash for the new module, because those bytes did not run.

Declare each such module in the table below. The `study` cell holds the registered study slug and
the `source` cell holds the repository-relative module path. The judgement starts with
`result-neutral:`. The gate fails on a row whose module the manifest records, or that the study
no longer imports. Remove the row when you regenerate the study.

| study | source | judgement |
| --- | --- | --- |
| `canonical-ltmle` | `tests/studies/fractional_glm.py` | result-neutral: `QuasiBinomialGLM` moved out of `tests/studies/canonical_ltmle.py` into `tests/studies/fractional_glm.py`, which re-exports it to that module. The class source is byte-identical, and so are the `math`, `numpy`, `scipy.special.expit` and `sklearn.base.BaseEstimator` names it resolves. The study fits the same coefficients from the same rows. The new module never ran, so the manifest can record no hash for it |
| `weighted-ltmle` | `tests/studies/fractional_glm.py` | result-neutral: `QuasiBinomialGLM` moved out of `tests/studies/canonical_ltmle.py` into `tests/studies/fractional_glm.py`, which re-exports it to that module. The class source is byte-identical, and so are the `math`, `numpy`, `scipy.special.expit` and `sklearn.base.BaseEstimator` names it resolves. The study fits the same coefficients from the same rows. The new module never ran, so the manifest can record no hash for it |
| `canonical-categorical-ltmle` | `tests/studies/fractional_glm.py` | result-neutral: `QuasiBinomialGLM` moved out of `tests/studies/canonical_ltmle.py` into `tests/studies/fractional_glm.py`, which re-exports it to that module. The class source is byte-identical, and so are the `math`, `numpy`, `scipy.special.expit` and `sklearn.base.BaseEstimator` names it resolves. The study fits the same coefficients from the same rows. The new module never ran, so the manifest can record no hash for it |
| `longitudinal-msm` | `tests/studies/fractional_glm.py` | result-neutral: `QuasiBinomialGLM` moved out of `tests/studies/canonical_ltmle.py` into `tests/studies/fractional_glm.py`, which re-exports it to that module. The class source is byte-identical, and so are the `math`, `numpy`, `scipy.special.expit` and `sklearn.base.BaseEstimator` names it resolves. The study fits the same coefficients from the same rows. The new module never ran, so the manifest can record no hash for it |
| `canonical-ltmle-survival` | `tests/studies/fractional_glm.py` | result-neutral: `QuasiBinomialGLM` moved out of `tests/studies/canonical_ltmle.py` into `tests/studies/fractional_glm.py`, which re-exports it to that module. The class source is byte-identical, and so are the `math`, `numpy`, `scipy.special.expit` and `sklearn.base.BaseEstimator` names it resolves. The study fits the same coefficients from the same rows. The new module never ran, so the manifest can record no hash for it |
| `canonical-ltmle-competing` | `tests/studies/fractional_glm.py` | result-neutral: `QuasiBinomialGLM` moved out of `tests/studies/canonical_ltmle.py` into `tests/studies/fractional_glm.py`, which re-exports it to that module. The class source is byte-identical, and so are the `math`, `numpy`, `scipy.special.expit` and `sklearn.base.BaseEstimator` names it resolves. The study fits the same coefficients from the same rows. The new module never ran, so the manifest can record no hash for it |

## Descriptive manifest text

A manifest's `configuration` block can also carry a descriptive string that no computation reads.
When the study module corrects such a string, the manifest keeps the text written at generation
time, for the same reason it keeps a recorded hash. Add a row to the table below for each such
correction. The row states the correction and why it leaves every artifact unchanged. Remove the
row when you regenerate the study. No correction is declared now, so the table has no rows. No
test parses this table. The gated table above keeps its own header.

| study | manifest field | correction | judgement |
| --- | --- | --- | --- |

## Completed manifest provenance

The stacked missing-outcome natural-course manifest initially omitted two Python helpers that
produced its rows. The manifest now records their SHA-256 values. No fitted row or verdict changed.

The manifest names commit `f899841` for the run. It also records `cleverly_worktree_clean=false`,
so that commit does not reconstruct the executed tree. The other recorded module hashes match
commit `df0b3cf`. The study runner hash matches `df0b3cf` and not `f899841`.

Both added files have the same bytes at `f899841` and at `df0b3cf`. The git blobs at both commits
hash to the recorded values. Each file therefore matches the run under either commit.

The study also loads `tests/studies/evidence/pairing.py` through the shared evidence framework.
The manifest does not record that file. `tests/unit/test_study_provenance.py` excludes the
framework from the required module list, so this is not a gap.

| study | added source | recorded SHA-256 | judgement |
| --- | --- | --- | --- |
| stacked missing-outcome natural-course CV-TMLE | `tests/studies/point_study_helpers.py` | `f7fa94d4116739718c78000f7d51d506a7907af9c8c0a02bf5354eeab4738d5a` | result-neutral: the helper serialized the existing initial estimate and primary rows. Its bytes are identical at `f899841` and `df0b3cf` |
| stacked missing-outcome natural-course CV-TMLE | `tests/conftest.py` | `6e991807afeae6a901f953040ab9080b86baed5e7a9a7ed043f7d6f4f6fb45ea` | result-neutral: the property study used its oracle outcome and response learners. Its bytes are identical at `f899841` and `df0b3cf` |

# Cross-fitted end-of-study LTMLE evidence

This directory holds the registered paired study for Cleverly's five-fold end-of-study recursion
against pinned R `lmtp` 1.5.4. Both implementations receive the identical panels, the identical
fold assignment stored with each panel, and the identical treatment and censoring mechanism.

R `ltmle`, the comparator the ordinary longitudinal rows use, has no cross-fitting, so it cannot
witness this construction. `lmtp` can, but it has no `gform` argument to be handed a known
mechanism through. The adapter in `tests/canonical/lmtp_crossfit_adapter.R` therefore substitutes
exact per-node density ratios, the same way it substitutes the fold assignment, and checks the
substitution against `lmtp`'s own estimate on every run.

That substitution is what makes this a comparison of the recursion rather than of two
mechanism-fitting pipelines. Letting each side estimate the mechanism its own way was tried first
and measured something else: `lmtp` fits its ratio with `SL.glm`, whose linear logit cannot
represent the exact classifier log-odds `-log g` for a deterministic regime, so the ratio came out
shrunken, the targeting under-moved, and its intervals covered 0.75 to 0.91 instead of 0.95.

Run a disposable probe before the declared study.

```console
python -m tests.canonical.lmtp_ltmle.regenerate --replicates 8 --n 400 --primary-only --output <temporary-directory> --cache <temporary-directory>
```

Then the complete registered study, which writes the artifacts and refuses failed replications or
failed evidence gates.

```console
python -m tests.canonical.lmtp_ltmle.regenerate
python -m tests.studies.evidence.document --slug canonical-ltmle-crossfit
```

`tests.canonical.lmtp_crossfit.audit` runs the comparator outside the registered gate and
summarizes it, for examining a candidate comparator without publishing a row.

## Known defect in the shared adapter, due at the next regeneration

`lmtp_crossfit_adapter.R` screens the supplied density ratios with
`abs(mean(supplied) - 1) > 0.5` on the *cumulative* product. That anchor is wrong in general. A
unit whose follow-up ends at the first node has no second-node arm, so its later columns are
structurally zero, and the cumulative ratio's expectation is the probability of reaching the
second node under the plan rather than one.

This study is unaffected. Its law draws few first-node events, so the cumulative mean stays inside
the band. The competing-risk rows are not: their law ends follow-up at the first node for most
units, and the same screen rejected a correct matrix at 0.3188 against an expectation of 0.3125.
`lmtp_competing_adapter.R` carries the corrected form, which screens the first column, whose
expectation is exactly one whatever the event process does downstream.

The shared adapter keeps the wrong anchor because its bytes are hashed into four manifests, and
correcting it invalidates this study's provenance. Take the corrected form across when this row is
next regenerated for a reason of its own.

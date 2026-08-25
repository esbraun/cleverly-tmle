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

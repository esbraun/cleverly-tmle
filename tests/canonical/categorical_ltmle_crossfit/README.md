# Cross-fitted categorical longitudinal TMLE fixture

This registered study fits five-fold longitudinal TMLE on the same two-node, three-level law as
the ordinary row. It compares the result with pinned R `lmtp` 1.5.4 on identical samples, exact
treatment probabilities, and the same rowwise fold assignment.

Run a disposable smoke study first:

```bash
python tests/canonical/categorical_ltmle_crossfit/regenerate.py --replicates 2 --n 500 --skip-properties --output .tmp/categorical-ltmle-crossfit-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/categorical_ltmle_crossfit/regenerate.py
python -m tests.studies.evidence.document --slug canonical-categorical-ltmle-crossfit
```

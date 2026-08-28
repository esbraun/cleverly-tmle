# Ordinary categorical longitudinal TMLE fixture

This registered study fits ordinary longitudinal TMLE on a two-node law with three labelled
treatment levels. It compares the result with pinned R `lmtp` 1.5.4 on identical samples. Both
implementations receive the exact treatment mechanism and one fold that contains every row.

Run a disposable smoke study first:

```bash
python tests/canonical/categorical_ltmle/regenerate.py --replicates 2 --n 500 --skip-properties --output .tmp/categorical-ltmle-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/categorical_ltmle/regenerate.py
python -m tests.studies.evidence.document --slug canonical-categorical-ltmle
```

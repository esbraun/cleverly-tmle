# Sequential g-computation and IPW evidence (`sequential-baselines`)

This study fits `SequentialGComputation` (ICE) and `SequentialIPW` (Hajek and Horvitz-Thompson)
with `GLM` nuisances on exact laws. It is Python only and has no reference implementation.
`tests/studies/canonical_sequential_baselines.py` states the primary scenarios and the reading
rule. `tests/studies/sequential_baselines_properties.py` states the property families.
`tests/studies/sequential_baselines_laws.py` holds the laws and their exact truths.
`tests/unit/test_sequential_baselines_design.py` pins the design.

The W9 fixture in `tests/canonical/sequential_baselines/` is a separate fast-tier check. The R
`ltmle` parity (plan cell P) is in `tests/canonical/sequential_baselines_ltmle/` and registers
later.

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/sequential_baselines_study/regenerate.py --primary-only --replicates 4 --n 2000 --output .tmp/sequential-baselines-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/sequential_baselines_study/regenerate.py
python -m tests.studies.evidence.document --slug sequential-baselines
```

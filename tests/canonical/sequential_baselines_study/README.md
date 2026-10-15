# Sequential g-computation and IPW evidence (`sequential-baselines`)

This study fits `SequentialGComputation` (ICE) and `SequentialIPW` (Hajek and Horvitz-Thompson)
with `GLM` nuisances on exact laws. It is Python only and has no reference implementation.
`tests/studies/canonical_sequential_baselines.py` states the primary scenarios and the reading
rule. `tests/studies/sequential_baselines_properties.py` states the property families.
`tests/studies/sequential_baselines_laws.py` holds the laws and their exact truths.
`tests/unit/test_sequential_baselines_design.py` pins the design.

The W9 fixture in `tests/canonical/sequential_baselines/` is a separate fast-tier check. The R
`ltmle` parity (plan cell P) is a separate study, and a later pull request registers it.

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/sequential_baselines_study/regenerate.py --primary-only --replicates 4 --n 2000 --output .tmp/sequential-baselines-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/sequential_baselines_study/regenerate.py
python -m tests.studies.evidence.document --slug sequential-baselines
```

## Declared routes and deviations

A property-phase stop has a declared route. The shared property summary refuses a cell that lost
a replicate, so the declared 0.01 failure bound is not yet operative there. A recordable
`DataError` in the property phase stops the phase with its records. The phase then reruns
unchanged once RM39b2 makes the 0.01 bound operative. The laws, `n`, `R` and seeds do not change.

The adaptive cell (A) deviates from plan 5.2. The plan fits LTMLE beside ICE and IPW and reads
bias against `sqrt(n)`. The study fits ICE and Hajek at `n = 1,000` with 100 replicates and no
LTMLE. One LTMLE fit with the default library costs about 920 CPU-seconds at `n = 1,000`, and
LTMLE has its own studies. The cell reports bias in SD units to about +/- 0.1, at one size.

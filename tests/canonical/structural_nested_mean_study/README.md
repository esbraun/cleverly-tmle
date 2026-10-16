# Structural nested mean evidence (`structural-nested-mean`)

This study fits `StructuralNestedMean` with `GLM` nuisances on exact finite laws. Law `N` uses an
early-stopped booster instead. It is Python only for now. The EconML 0.17.0 `DynamicDML` comparator is
declared and pending (`STUDY.pending_reference`). A later change registers the comparator runs.

| file | role |
| --- | --- |
| `tests/studies/canonical_structural_nested_mean.py` | The record, the fits, the primary phase, the labels and the reading rule. |
| `tests/studies/structural_nested_mean_properties.py` | The property families and their rules. |
| `tests/studies/structural_nested_mean_laws.py` | The setups, the laws, their exact truths and the samplers. |
| `tests/discrete_law_structural.py` | The independent g-formula that the truths come from. |
| `tests/unit/test_structural_nested_mean_design.py` | The design test: every declared number and the pinned pilots. |
| `tests/canonical/econml_dynamic_dml/` | The declared comparator sources: Dockerfile, lock and generator. |

## State of this directory

No run artifact exists yet. The design is complete and the study is not in `registered()`,
because a registered study needs its committed artifacts. The property module is listed in
`SYNTHETIC_MODULES` of `tests/canonical/property_differential.py` and in `UNCOMMITTED_MODULES`
of `tests/unit/test_method_evidence.py`, as the incremental studies were before their runs.
The change that commits the run artifacts also adds the registry line.

The design declares no red row. The two negative controls (`LU` and `VB`) read as a detection
rule: a cell passes when its bias interval lies beyond the margin on the side of the computed
gap. The size cells of the model check on the sparse-positivity law `SP` run at n = 2,000. The
check has about 8 degrees of freedom there, so it has low power on `SP`. The cell `SP` of the
family `positivity_reporting` reports the structural fit under weak positivity and gates nothing.

The booster of law `N` stops on a validation split, so the data choose its number of trees. Its
capacity therefore grows with n. The design test pins a probe at n = 128,000 and n = 512,000,
with 24 draws at each size. At both sizes, the bias of each `N` estimand is less than 0.46 of its
reported SE, and each 99% interval contains zero. The fixed booster of an earlier draft fails
this check at n = 512,000.

The calibration cells run 8,000 replicates each. A sizing rule gives this count. The rule takes
the smallest multiple of 400 at which all 61 cells pass with a chance of at least 0.95. The
rule takes the true SE ratio of each cell from its own pilot, shrunk toward the mean of its setup.
At 8,000 replicates, the family passes with a chance of 0.952.
`tests/unit/test_structural_nested_mean_design.py` checks the rule.

The `positivity_reporting` fits of `SP` use outcome order 3 and treatment order 2. At these
orders, each nuisance regression is exact. The model-check fits use order 2 for both roles.
A pilot of 1,000 draws at n = 2,000 shows no fit failure.

## Run

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/structural_nested_mean_study/regenerate.py --primary-only --replicates 2 --n 2000 --output .tmp/structural-nested-mean-smoke
```

Run a disposable smoke of every property cell at two replicates:

```bash
python -c "
from tests.studies import structural_nested_mean_properties as p
from tests.studies.evidence import replication
with replication.recording(p.STUDY.implementation, witness=p.PROPERTY_WITNESS):
    rows = p.generate_property_rows(n_jobs=4, budget=2)
print(p.summarize_properties(rows).shape)
"
```

Regenerate the declared study and refresh its document. Set `CLEVERLY_SUPPORT_RECORD` on the
run, so that each fit records its support key. The support rows of this study come from these
records. `REC` is the record root of the RM42b measurement.

```bash
CLEVERLY_SUPPORT_RECORD=$REC/studies/structural_nested_mean_study \
  python tests/canonical/structural_nested_mean_study/regenerate.py
python -m tests.studies.support_measurement --records $REC --out $REC/measurement.md
python -m tests.studies.evidence.document --slug structural-nested-mean
```

`SUPPORT_KEYS` in `tests/studies/canonical_structural_nested_mean.py` declares the key and the
learner class of each setup. The design test fits each setup with the record switch on and
compares the record with that declaration.

`tests/unit/test_structural_nested_mean_design.py` prints the budget in CPU-hours from the
timed pilot fits.

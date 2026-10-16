# Structural nested mean evidence (`structural-nested-mean`)

This study fits `StructuralNestedMean` with `GLM` nuisances (a shallow booster on law `N`) on
exact finite laws. It is Python only for now. The EconML 0.17.0 `DynamicDML` comparator is
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

The calibration cells run 21,200 replicates each. The arm-pooled predictive rule needs a chance
of 0.95 that all 61 cells pass. The adaptive arm `N` cannot reach it at any count, because its
pilot SE ratios spread between cells. Its chance tends to 0.67. The rule therefore applies to the
other arms, and it gives a minimum of 17,600. The declared count is above that minimum. The
adaptive arm passes with chance 0.45, and the whole family with chance 0.44. A red adaptive cell
is read by the declared red-cell rule.

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

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/structural_nested_mean_study/regenerate.py
python -m tests.studies.evidence.document --slug structural-nested-mean
```

`tests/unit/test_structural_nested_mean_design.py` prints the budget in CPU-hours from the
timed pilot fits.

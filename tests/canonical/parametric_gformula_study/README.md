# Parametric g-formula evidence (`parametric-gformula`)

This study fits `ParametricGFormula` (NICE) on five exact laws. The comparator `pygformula`
1.2.0 is pending, and a follow-up PR from X45 registers it. `tests/studies/canonical_parametric_gformula.py`
states the primary scenarios and the reading rule. `tests/studies/parametric_gformula_properties.py`
states the property families. `tests/studies/parametric_gformula_laws.py` holds the laws and
their exact truths. `tests/unit/test_parametric_gformula_design.py` pins the design.

Run a disposable smoke of the primary phase:

```bash
python tests/canonical/parametric_gformula_study/regenerate.py --primary-only --replicates 4 --n 2000 --output .tmp/parametric-gformula-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/parametric_gformula_study/regenerate.py
python -m tests.studies.evidence.document --slug parametric-gformula
```

The declared run is committed. The artifacts in this directory are its results.

## Red-cell reading

The module `tests/studies/canonical_parametric_gformula.py` declares the reading before any run.
Each gated calibration cell is sized for a pass chance of at least 0.95 at its shrunk SE ratio
and at least 0.925 with the posterior SD of that ratio counted. The expected count of
finite-sample reds is 1.26 over 114 gated cells, and the chance of at least one is 0.72. The count
includes the 42 derived-control cells. Two reds read as expected. A red
calibration row is a finite-sample red only when its 99% SE-ratio interval contains 1, its
coverage is consistent with 0.95, and the diagnosis finds no defect. Any other red row is a
defect suspect. The `mc_removed` cells report and carry no verdict.

### Result of the declared run

The run has 119 property cells, and 118 pass. One cell is red:
`interval_calibration/nice_g1_treat_all__correctly_specified` (n = 2,000, 2,400 replications).

| quantity | value |
| --- | --- |
| SE ratio | 0.9619, with a 99% interval of 0.9259 to 1.0000. The band is 0.93 to 1.07 |
| coverage | 0.9413, with a 99% interval of 0.9278 to 0.9529. The band is 0.92 to 0.98 |
| bias | 0.000158, with a standard error of 0.000393 |

The cell reads as finite-sample under the declared reading. The SE-ratio interval contains 1, the
coverage lies inside its band, and no defect shows. These witnesses support the last condition.

| witness | SE ratio |
| --- | --- |
| the same law, plan and size, independent draws in `shared_target/target_g1_treat_all__nice` (R = 1,192) | 1.0022 |
| the same law, plan and size, independent draws in `ice_reported/pair_g1_treat_all__nice` (R = 1,198) | 0.9701 |
| the other `G1` calibration cells | 0.972 to 0.993 |
| the `G1` contrast ladder at n = 2,000, 4,000 and 8,000 | 1.021, 0.986 and 1.006 |

The count agrees with the declaration. The run expected 1.26 reds over 114 gated cells and found
one. The roadmap owner `X45-finite-sample` holds the cell.

### Recorded fitting failures

The property phase recorded 10 fitting failures in `property-failures.csv`. Two of 1,200
replicates failed in `paired_g1`, and eight of 1,200 in `shared_g1`. Each is a GLM with a
singular information matrix. Both rates are under `FAILURE_BOUND` (0.01).

The failing fit is not the `ParametricGFormula` fit. Replicate `paired_g1/293` and replicate
`shared_g1/248` fit `ParametricGFormula` without error when run alone. The fit that fails is the
sequential outcome GLM of the comparator in the same fit set. It is `SequentialGComputation`
with `method="ice"` in `paired_g1`, and `LTMLE` in `shared_g1`. The cells of both fit sets carry
no verdict, so no gate reads the lost replicates.

## Pending comparator

The record declares `pending_reference="pygformula"`. The driver runs no reference process, and
the manifest records the bytes of `Dockerfile`, `requirements.txt` and `run_pygformula.py` as
declared sources. The files are a stub. The comparator change replaces `run_pygformula.py`,
moves the name to `reference`, and regenerates the study.

`pygformula` cannot express node-specific covariate models. It parses bare column names and
pools the residual SD across time. So the fixtures use laws where pooled and node-specific
models agree.

| fixture | law | what it tests |
| --- | --- | --- |
| covariate laws | `T = 2`, one simulated node | pooled and node-specific models are equal, so the estimate and `sigma` match |
| survival | `T >= 3`, time-interacted outcome statements | the hazard-product risk |
| one competing cause | the sequential form, generated and fitted in that form | a test-only path |

Each fixture compares coefficients first (`statsmodels` against `GLM`, `rtol=1e-6`). It then
compares the estimate, with the Monte Carlo SE from at least 20 seeds at `n_simul = n`. The
gate is the difference inside four combined Monte Carlo SE. The run uses its own environment
and never a package extra.

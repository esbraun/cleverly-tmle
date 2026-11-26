# Parametric g-formula evidence (`parametric-gformula`)

This study fits `ParametricGFormula` (NICE) on five exact laws. The record declares no
comparator. `pygformula` 1.2.0 runs on separate fixtures, and their results are committed (see
[Comparator](#comparator)). `tests/studies/canonical_parametric_gformula.py`
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
finite-sample reds is 1.31 over 120 gated cells, and the chance of at least one is 0.73. The count
includes the 46 derived-control cells and the rule cells of `G2` and `G5`. Two reds read as
expected. A red
calibration row is a finite-sample red only when its 99% SE-ratio interval contains 1, its
coverage is consistent with 0.95, and the diagnosis finds no defect. Any other red row is a
defect suspect. The `mc_removed` cells report and carry no verdict.

### Result of the declared run

The run at `b778e4dd` has 125 property cells, and 124 pass. It is a complete rerun that adds the
rule cells of `G2` and `G5` and writes every row under the shipped intervals: a continuous
estimate takes the skew-widened t (`docs/technical-reference/inference.md`). Against the previous
run, every estimate and standard error is unchanged, the intervals move, no verdict changes, and
the largest coverage change of a property cell is 0.0084. One cell is red, as before:
`interval_calibration/nice_g1_treat_all__correctly_specified` (n = 2,000, 2,400 replications).

| quantity | value |
| --- | --- |
| SE ratio | 0.9619, with a 99% interval of 0.9259 to 1.0000. The band is 0.93 to 1.07 |
| coverage | 0.9438, with a 99% interval of 0.9305 to 0.9552. The band is 0.92 to 0.98 |
| bias | 0.000158, with a standard error of 0.000393 |

The cell reads as finite-sample under the declared reading. The SE-ratio interval contains 1, the
coverage lies inside its band, and no defect shows. These witnesses support the last condition.

| witness | SE ratio |
| --- | --- |
| the same law, plan and size, independent draws in `shared_target/target_g1_treat_all__nice` (R = 1,192) | 1.0022 |
| the same law, plan and size, independent draws in `ice_reported/pair_g1_treat_all__nice` (R = 1,198) | 0.9701 |
| the other `G1` calibration cells | 0.972 to 0.993 |
| the `G1` contrast ladder at n = 2,000, 4,000 and 8,000 | 1.021, 0.986 and 1.006 |

The count agrees with the declaration. The run expected 1.31 reds over 120 gated cells and found
one. The roadmap owner `X45-finite-sample` holds the cell.

The six rule cells pass. Each correctly specified cell is inside both bands, and each control
falls below the SE-ratio band as declared.

| cell | coverage (99% interval) | SE ratio (99% interval) |
| --- | --- | --- |
| `nice_g2_rule__correctly_specified` | 0.9421 (0.9287 to 0.9537) | 0.9820 (0.9477 to 1.0192) |
| `nice_g5_rule__correctly_specified` | 0.9446 (0.9314 to 0.9559) | 0.9966 (0.9617 to 1.0353) |
| `nice_g2_rule__shrunken_se_control` | 0.8271 | 0.6874 (0.6633 to 0.7136) |
| `nice_g2_rule__noise_control` | 0.8246 | 0.7053 (0.6808 to 0.7333) |
| `nice_g5_rule__shrunken_se_control` | 0.8250 | 0.6976 (0.6728 to 0.7246) |
| `nice_g5_rule__noise_control` | 0.8292 | 0.7082 (0.6849 to 0.7351) |

### Recorded fitting failures

The property phase recorded 10 fitting failures in `property-failures.csv`. Two of 1,200
replicates failed in `paired_g1`, and eight of 1,200 in `shared_g1`. Each is a GLM with a
singular information matrix. Both rates are under `FAILURE_BOUND` (0.01).

The failing fit is not the `ParametricGFormula` fit. Replicate `paired_g1/293` and replicate
`shared_g1/248` fit `ParametricGFormula` without error when run alone. The fit that fails is the
sequential outcome GLM of the comparator in the same fit set. It is `SequentialGComputation`
with `method="ice"` in `paired_g1`, and `LTMLE` in `shared_g1`. The cells of both fit sets carry
no verdict, so no gate reads the lost replicates.

## Comparator

The record declares no comparator. `pygformula` runs on four fixtures in its own image, and
`compare_pygformula.py` writes the results beside this README. All 63 coefficient rows and all
21 estimate rows pass. `tests/unit/test_parametric_gformula_comparator.py` recomputes the
verdicts from the committed rows. It also checks the digests in `pygformula-run.json` against
`pygformula-results.csv`, `Dockerfile`, `requirements.txt` and `run_pygformula.py`.

`pygformula` cannot express node-specific covariate models. It parses bare column names and
pools the residual SD across time. So the fixtures use laws where pooled and node-specific
models agree. The laws are in `tests/canonical/parametric_gformula_study/fixtures.py`.

| fixture | law | what it tests |
| --- | --- | --- |
| `cov_gaussian`, `cov_binary` | `T = 2`, one simulated node, a normal and a binary covariate, a Gaussian or binary outcome | pooled and node-specific models are equal, so the estimate and `sigma` match |
| `survival` | `T = 3`, no time-varying covariate, outcome statement `0 + C(t) + C(t):W + C(t):A + C(t):lag1_A + C(t):lag2_A` | the hazard-product risk at three horizons |
| `competing` | `T = 2`, one competing cause at node 2, drawn before the event of interest | the sequential form, fitted through the test-only `SequentialHazard` learner |

Each fixture compares coefficients first (`statsmodels` against `GLM`, `rtol=1e-6`). It then
compares the estimate. Both sides run 24 seeds on the one panel, and the Monte Carlo SE of a side
is the standard deviation of its seeds over the square root of 24. The gate is a difference of
means inside four combined Monte Carlo SE. `pygformula` runs at `n_simul = n` and
`sim_trunc=False`. `cleverly` runs its default 10 histories per unit.

The survival and competing fixtures draw no covariate, so neither side has simulation noise
there. Those estimates must agree to `1e-10`. A fixture with a drawn covariate and `T >= 3` cannot
reach equal models, because the pooled covariate model would differ from the node-specific ones.

### Packaging of `pygformula` 1.2.0

| fact | handling |
| --- | --- |
| the wheel imports `pygformula.version` and does not ship it | `run_pygformula.py` registers a version module before the import |
| `pygformula.plot` calls `matplotlib.use("Qt5Agg")`, which fails without a display | the runner registers a plotting module that does nothing |
| `fit_covariate_model` and `comparison_calculate` call `groupby.apply(include_groups=False)` | `requirements.txt` pins `pandas==2.2.3` |

The run uses its own environment and never a package extra. `requirements.txt` pins `pygformula`
and every resolved dependency.

### Run

The run needs Docker. From the repository root:

```bash
python -m tests.canonical.parametric_gformula_study.compare_pygformula
```

The command builds the image, runs the container on the panels, fits `cleverly` on the same
panels and writes five files beside this README: `pygformula-results.csv` (the raw container
rows), `pygformula-coefficients.csv`, `pygformula-estimates.csv`, `pygformula-seeds.csv` and
`pygformula-run.json` (sources, image, settings). The command exits with code 1 when a row
fails. Outside Docker, the container run took 35 seconds and the `cleverly` fits took 9 seconds.
The time of the image build is not measured.

`--reuse-results` reads the committed `pygformula-results.csv` and refits only the `cleverly`
side, so it needs no Docker.

### Registration

The fixture comparison does not fit the paired-row contract of a `reference`. A `reference`
pairs rows by scenario and replicate in `replicates.parquet`, and `validate_replicates` requires a
positive standard error and an interval for each. `pygformula` reports neither without a
bootstrap, and each fixture is one panel with many seeds. So the record declares no comparator,
and the fixture files are the `pygformula` evidence.

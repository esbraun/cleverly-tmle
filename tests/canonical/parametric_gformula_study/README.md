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

Do not start the declared run before the design review of the pilot has closed.

## Red-cell reading

The module `tests/studies/canonical_parametric_gformula.py` declares the reading before any run.
Each gated cell is sized for a pass chance of at least 0.90 under perfect calibration. The
expected count of finite-sample reds is 1.07, and the chance of at least one is 0.66. A red
calibration row is a finite-sample red only when its 99% SE-ratio interval contains 1, its
coverage is consistent with 0.95, and the diagnosis finds no defect. Any other red row is a
defect suspect. The `mc_removed` cells report and carry no verdict.

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

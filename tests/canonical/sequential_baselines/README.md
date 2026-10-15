# R `ltmle` points for the sequential baselines (W9)

This directory freezes the g-computation and IPTW points of R `ltmle` 1.3-0 on one
nonsaturated panel. `tests/unit/test_sequential_baselines_ltmle_r.py` compares them with
`SequentialGComputation` and `SequentialIPW` to `1e-7`.

| file | content |
| --- | --- |
| `make_data.py` | draws `data.csv`: two nodes, two baseline covariates, a continuous `L2`, censoring, and a binary `Y` |
| `generate_reference.R` | fits `ltmle` twice per regime and writes `reference.csv` and `versions.txt` |
| `reference.csv` | one row per regime and estimator |

The two `ltmle` calls are these:

| point | call |
| --- | --- |
| g-computation | `gcomp=TRUE`, `stratify=TRUE` |
| IPTW | `gcomp=FALSE`, `stratify=FALSE`. `CalcIPTW` divides by the weight sum, which is the Hajek form |

Both calls use `SL.library="glm"` and `gbounds=c(0.01, 1)`. Each `gform` reads the full
history. Each `Qform` reads the covariates and no arm column, as `outcome_design` does at a
label or rule node. A rule's arm is not a linear function of the covariates, so an arm column
changes a main-terms fit. The regimes are always treat, never treat, and the rule
$A_1 = 1\{W_1 > 0\}$, $A_2 = 1\{L_2 > 0\}$.

The exact laws in `tests/unit/test_sequential_baselines.py` are the oracle for the target. This
fixture checks only that the `GLM` fits and the weights match the reference implementation.

## Regeneration

Run from the repository root, with Docker available. The image is the pinned image of
`tests/canonical/ltmle`.

```powershell
python -m tests.canonical.sequential_baselines.make_data
docker build -t cleverly-ltmle-reference:1.3-0 tests/canonical/ltmle
docker run --rm `
  -v "${PWD}/tests/canonical/sequential_baselines:/fixture" `
  cleverly-ltmle-reference:1.3-0 `
  /fixture/generate_reference.R /fixture
```

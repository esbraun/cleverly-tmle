# Canonical R `ltmle` witness

This directory freezes two end-of-study fits and one survival fit from R `ltmle` 1.3-0.
It answers two deliberately narrow questions that the independent exact laws cannot:

1. Is each raw treatment-and-censoring prefix multiplied before `gbounds` is applied, as
   in `ltmle::CalcCumG`?
2. When targeting moves, is the cumulative inverse probability placed in the loss weight
   of the intercept update, as in `ltmle::UpdateQ`, rather than in the submodel direction?

All fixtures use fixed numeric g predictions, intercept-only Q regressions, no
cross-fitting, `abar = c(1, 1)`, and `gbounds = c(0.2, 0.99)`. The lower bound binds only
in one baseline stratum at the second node, and the deepest epsilon has magnitude above
0.4. Agreement therefore cannot be caused by correct nuisances making targeting vanish.
The `censored` variant has two regimen followers censored at `C1` and two at `C2`, so it
also witnesses the rows entering each update. The upper bound never binds. The original
end-of-study fit's first-node epsilon is structurally zero; both nodes move in the survival
fit.

These references deliberately use `variance.method="ic"` so the stored influence curve is
the quantity cleverly reports. R warns during regeneration that IC-only variance may be
significantly anti-conservative in this bound-active fixture; its default robust method is
also unavailable with `stratify=TRUE`. The fixture is algorithm evidence, not evidence
that active-truncation intervals have correct coverage.

The independent parameter and EIF evidence remains in
`tests/discrete_law_longitudinal.py`, `tests/discrete_law_survival.py`, and their Gateaux
test modules. This reference is secondary implementation evidence, not an oracle for the
estimand.

## Sources and regeneration

- Lendle, Schwab, Petersen & van der Laan (2017), *ltmle: An R Package Implementing
  Targeted Minimum Loss-Based Estimation for Longitudinal Data*,
  <https://doi.org/10.18637/jss.v081.i01>.
- Canonical package source, especially `CalcCumG`, `FixedTimeTMLE`, and `UpdateQ`:
  <https://rdrr.io/github/joshuaschwab/ltmle/src/R/ltmle.R>.
- R package version 1.3-0:
  <https://cran.r-project.org/package=ltmle>.

From the repository root, with Docker available:

```powershell
docker build -t cleverly-ltmle-reference:1.3-0 tests/canonical/ltmle
docker run --rm `
  -v "${PWD}/tests/canonical/ltmle:/fixture" `
  cleverly-ltmle-reference:1.3-0 `
  /fixture/generate_reference.R /fixture
```

The image pins R 4.5.2 by digest and checks that the installed package reports version
1.3.0. The generated artifacts are `longitudinal.csv`, `censored.csv`, `survival.csv`,
and `reference.csv`; `tests/unit/test_ltmle_canonical_r.py` consumes them without requiring
R or Docker in CI.

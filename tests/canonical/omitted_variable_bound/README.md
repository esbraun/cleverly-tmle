# Omitted-variable bound standard error

This directory holds a repeated-sampling study of the one-sided limits of `cleverly`'s
omitted-variable bound. Each bound is read as an estimate of a known population value. The
study reads its standard error back off the reported limit, as `(lower - ci_lower) / z_0.95`
for a lower bound and `(ci_upper - upper) / z_0.95` for an upper bound.

The law is `make_linear_ate(n=1000)`. An in-sample TMLE fits it with a linear outcome regression
and an unpenalized main-effects logistic treatment model, and both are correctly specified. The
strength is `cf_y = 0.5`, `cf_d = 0.3` and `rho = 1`. Every truth is a closed form of the law.

The property study holds one `interval_calibration` family. Each end of the ATT, ATC and ATE
bounds has a positive cell. Each end of the ATT bound also has an `inflated_se_control`, which
reads the curve of `nu^2` without the conditioning-share term. That is the curve the package
reported before RM22. All cells read the same draws.

No canonical implementation is compared. DoubleML omits the share term, and `dml.sensemakr` is R
code with a cross-fitted share and a different point estimate. `equivalence.csv` is empty and
schema-valid.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.omitted_variable_bound.regenerate
```

The declarations are in `tests/studies/omitted_variable_bound.py` and
`tests/studies/omitted_variable_bound_properties.py`. The RM22 plan in `docs/roadmap.md` fixed
them before the run. The reader-facing scope, measurements and limitations are in
[`docs/technical-reference/method-evidence/omitted-variable-bound-standard-error.md`](../../../docs/technical-reference/method-evidence/omitted-variable-bound-standard-error.md).
